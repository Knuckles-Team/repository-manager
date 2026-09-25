"""Bounded, receipt-bound dependency observations for fleet graph reconciliation.

The caller supplies every source file explicitly. Missing source families and dynamic
targets remain unresolved evidence; this module never infers a complete fleet DAG.
"""

from __future__ import annotations

import ast
import hashlib
import json
import re
import tomllib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

SourceKind = Literal["python_project", "python_source", "compose", "frontend_package"]
EdgeClass = Literal[
    "production_runtime_package", "development_test", "deployment_composition"
]
Modality = Literal["manifest_declared", "statically_resolved", "dynamically_unresolved"]
Resolution = Literal["declared", "statically_resolved", "unresolved"]

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_REVISION = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True, slots=True)
class CensusSource:
    path: str
    kind: SourceKind


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    path: str
    kind: SourceKind
    sha256: str
    size: int


@dataclass(frozen=True, slots=True)
class EdgeObservation:
    source_id: str
    target_id: str | None
    relation: str
    edge_class: EdgeClass
    modality: Modality
    resolution: Resolution
    source_path: str
    selector: str
    guard: str | None = None
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class CensusReceipt:
    schema: str
    source_commit: str
    manifest_sha256: str
    sources: tuple[SourceIdentity, ...]
    observations: tuple[EdgeObservation, ...]
    missing_kinds: tuple[SourceKind, ...]
    unresolved_count: int
    complete: bool
    digest: str

    def payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source_commit": self.source_commit,
            "manifest_sha256": self.manifest_sha256,
            "sources": [asdict(source) for source in self.sources],
            "observations": [asdict(edge) for edge in self.observations],
            "missing_kinds": list(self.missing_kinds),
            "unresolved_count": self.unresolved_count,
            "complete": self.complete,
        }

    def to_json(self) -> bytes:
        return _canonical({**self.payload(), "digest": self.digest})


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()


def _read_source(root: Path, relative: str) -> bytes:
    path = root / relative
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"source escapes workspace: {relative}")
    if not path.is_file():
        raise ValueError(f"source is not a regular file: {relative}")
    return path.read_bytes()


def _observation(
    source: str,
    target: str | None,
    relation: str,
    classification: tuple[EdgeClass, Modality, Resolution],
    origin: tuple[str, str],
    *,
    guard: str | None = None,
    reason: str | None = None,
) -> EdgeObservation:
    edge_class, modality, resolution = classification
    path, selector = origin
    return EdgeObservation(
        source,
        target,
        relation,
        edge_class,
        modality,
        resolution,
        path,
        selector,
        guard,
        reason,
    )


def _requirements(document: dict[str, Any]) -> set[str]:
    project = document.get("project", {})
    groups = [project.get("dependencies", [])]
    groups.extend(project.get("optional-dependencies", {}).values())
    names: set[str] = set()
    for group in groups:
        for requirement in group:
            match = _REQUIREMENT_NAME.match(requirement)
            if match:
                names.add(match.group(1).lower().replace("_", "-"))
    return names


def _python_project(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    document = tomllib.loads(raw.decode())
    project = document.get("project", {})
    owner = f"python:{project.get('name', Path(path).parent.name)}"
    declared = _requirements(document)
    sources = document.get("tool", {}).get("uv", {}).get("sources", {})
    rows: list[EdgeObservation] = []
    for name, location in sorted(sources.items()):
        normalized = name.lower().replace("_", "-")
        matched = normalized in declared
        guard = json.dumps(location, sort_keys=True, separators=(",", ":"))
        rows.append(
            _observation(
                owner,
                f"python:{normalized}",
                "resolver_source",
                (
                    "production_runtime_package",
                    "manifest_declared",
                    "declared" if matched else "unresolved",
                ),
                (path, f"tool.uv.sources.{name}"),
                guard=guard,
                reason=None
                if matched
                else "resolver source has no dependency declaration",
            )
        )
    return tuple(rows)


def _python_source(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    tree = ast.parse(raw, filename=path)
    module_names = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "importlib"
    }
    function_names = {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "importlib"
        for alias in node.names
        if alias.name == "import_module"
    }
    rows: list[EdgeObservation] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        direct = isinstance(node.func, ast.Name) and node.func.id in {
            "__import__",
            *function_names,
        }
        qualified = (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "import_module"
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id in module_names
        )
        if not (direct or qualified):
            continue
        argument = node.args[0] if node.args else None
        literal = argument.value if isinstance(argument, ast.Constant) else None
        target = literal if isinstance(literal, str) and literal else None
        resolved = target is not None
        rows.append(
            _observation(
                f"module:{path}",
                f"module:{target}" if target else None,
                "imports",
                (
                    "production_runtime_package",
                    "statically_resolved" if resolved else "dynamically_unresolved",
                    "statically_resolved" if resolved else "unresolved",
                ),
                (path, f"line:{node.lineno}"),
                reason=None if resolved else "dynamic import target is not a literal",
            )
        )
    return tuple(rows)


def _compose(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    document = yaml.safe_load(raw)
    if not isinstance(document, dict) or not isinstance(document.get("services"), dict):
        raise ValueError(f"Compose source has no services mapping: {path}")
    rows: list[EdgeObservation] = []
    for service, body in sorted(document["services"].items()):
        if not isinstance(body, dict):
            raise ValueError(f"invalid Compose service: {service}")
        dependencies = body.get("depends_on", [])
        if not isinstance(dependencies, (list, dict)):
            raise ValueError(f"invalid Compose depends_on: {service}")
        for dependency in sorted(dependencies):
            rows.append(
                _observation(
                    f"service:{service}",
                    f"service:{dependency}",
                    "depends_on",
                    ("deployment_composition", "manifest_declared", "declared"),
                    (path, f"services.{service}.depends_on.{dependency}"),
                )
            )
    return tuple(rows)


def _frontend_package(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    document = json.loads(raw)
    owner = f"npm:{document.get('name', Path(path).parent.name)}"
    rows: list[EdgeObservation] = []
    sections: tuple[tuple[str, EdgeClass], ...] = (
        ("dependencies", "production_runtime_package"),
        ("optionalDependencies", "production_runtime_package"),
        ("devDependencies", "development_test"),
    )
    for section, edge_class in sections:
        for name, version in sorted(document.get(section, {}).items()):
            rows.append(
                _observation(
                    owner,
                    f"npm:{name}",
                    "depends_on",
                    (edge_class, "manifest_declared", "declared"),
                    (path, f"{section}.{name}"),
                    guard=str(version),
                )
            )
    return tuple(rows)


_EXTRACTORS: dict[SourceKind, Any] = {
    "python_project": _python_project,
    "python_source": _python_source,
    "compose": _compose,
    "frontend_package": _frontend_package,
}


def capture_census(
    root: Path,
    manifest: str,
    sources: tuple[CensusSource, ...],
    *,
    source_commit: str,
) -> CensusReceipt:
    """Capture only named artifacts, preserving unknown coverage in the receipt."""

    if not _REVISION.fullmatch(source_commit):
        raise ValueError("source_commit must be a full Git commit SHA-1")
    manifest_hash = hashlib.sha256(_read_source(root, manifest)).hexdigest()
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("census source paths must be unique")
    identities: list[SourceIdentity] = []
    observations: list[EdgeObservation] = []
    for source in sorted(sources, key=lambda item: item.path):
        raw = _read_source(root, source.path)
        identities.append(
            SourceIdentity(
                source.path, source.kind, hashlib.sha256(raw).hexdigest(), len(raw)
            )
        )
        observations.extend(_EXTRACTORS[source.kind](source.path, raw))
    observations.sort(
        key=lambda item: (item.source_path, item.selector, item.target_id or "")
    )
    missing: tuple[SourceKind, ...] = tuple(
        kind for kind in _EXTRACTORS if kind not in {s.kind for s in sources}
    )
    unresolved = sum(edge.resolution == "unresolved" for edge in observations)
    payload = {
        "schema": "fleet_dependency_census/v1",
        "source_commit": source_commit,
        "manifest_sha256": manifest_hash,
        "sources": [asdict(source) for source in identities],
        "observations": [asdict(edge) for edge in observations],
        "missing_kinds": missing,
        "unresolved_count": unresolved,
        "complete": False,
    }
    return CensusReceipt(
        "fleet_dependency_census/v1",
        source_commit,
        manifest_hash,
        tuple(identities),
        tuple(observations),
        missing,
        unresolved,
        False,
        hashlib.sha256(_canonical(payload)).hexdigest(),
    )


def verify_census(root: Path, manifest: str, receipt: CensusReceipt) -> bool:
    """Verify immutable source bytes and the receipt's canonical digest."""

    try:
        if (
            hashlib.sha256(_read_source(root, manifest)).hexdigest()
            != receipt.manifest_sha256
        ):
            return False
        if any(
            hashlib.sha256(_read_source(root, source.path)).hexdigest() != source.sha256
            for source in receipt.sources
        ):
            return False
    except (OSError, ValueError):
        return False
    return hashlib.sha256(_canonical(receipt.payload())).hexdigest() == receipt.digest
