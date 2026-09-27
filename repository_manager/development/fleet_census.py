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
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import yaml

if TYPE_CHECKING:
    from .fleet_manifest import FleetManifestClosure
    from .fleet_source_universe import SourceUniverseReceipt

SourceKind = Literal[
    "python_project",
    "python_source",
    "rust_manifest",
    "rust_source",
    "compose",
    "frontend_package",
    "frontend_source",
    "raw_manifest",
]
EdgeClass = Literal[
    "production_runtime_package",
    "development_test",
    "build_generation",
    "deployment_composition",
]
Modality = Literal["manifest_declared", "statically_resolved", "dynamically_unresolved"]
Resolution = Literal["declared", "statically_resolved", "unresolved"]

_REQUIREMENT_NAME = re.compile(r"^\s*([A-Za-z0-9][A-Za-z0-9_.-]*)")
_REVISION = re.compile(r"^[0-9a-f]{40}$")
MAX_CENSUS_SOURCES = 16384
MAX_CENSUS_SOURCE_BYTES = 2 * 1024 * 1024
MAX_CENSUS_TOTAL_BYTES = 256 * 1024 * 1024


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
    source_universe_digest: str | None = None

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
            "source_universe_digest": self.source_universe_digest,
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
    if path.stat().st_size > MAX_CENSUS_SOURCE_BYTES:
        raise ValueError(f"source exceeds bounded size: {relative}")
    raw = path.read_bytes()
    if len(raw) > MAX_CENSUS_SOURCE_BYTES:
        raise ValueError(f"source exceeds bounded size: {relative}")
    return raw


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


def _importlib_module_names(tree: ast.Module) -> set[str]:
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
        if alias.name == "importlib"
    }


def _import_module_function_names(tree: ast.Module) -> set[str]:
    return {
        alias.asname or alias.name
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module == "importlib"
        for alias in node.names
        if alias.name == "import_module"
    }


def _is_dynamic_python_import(
    call: ast.Call, module_names: set[str], function_names: set[str]
) -> bool:
    function = call.func
    if isinstance(function, ast.Name):
        return function.id == "__import__" or function.id in function_names
    return (
        isinstance(function, ast.Attribute)
        and function.attr == "import_module"
        and isinstance(function.value, ast.Name)
        and function.value.id in module_names
    )


def _python_import_observation(path: str, node: ast.Call) -> EdgeObservation:
    argument = node.args[0] if node.args else None
    literal = argument.value if isinstance(argument, ast.Constant) else None
    target = literal if isinstance(literal, str) and literal else None
    resolved = target is not None
    return _observation(
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


def _python_source(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    tree = ast.parse(raw, filename=path)
    module_names = _importlib_module_names(tree)
    function_names = _import_module_function_names(tree)
    rows: list[EdgeObservation] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_dynamic_python_import(
            node, module_names, function_names
        ):
            rows.append(_python_import_observation(path, node))
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


_CARGO_SECTIONS: tuple[tuple[str, EdgeClass], ...] = (
    ("dependencies", "production_runtime_package"),
    ("dev-dependencies", "development_test"),
    ("build-dependencies", "build_generation"),
)


def _cargo_dependency_target(path: str, alias: str, specification: object) -> str:
    if not isinstance(alias, str) or not alias:
        raise ValueError(f"invalid Cargo dependency name: {path}")
    if not isinstance(specification, (str, dict)):
        raise ValueError(f"invalid Cargo dependency specification: {path}")
    target = (
        specification.get("package", alias)
        if isinstance(specification, dict)
        else alias
    )
    if not isinstance(target, str) or not target:
        raise ValueError(f"invalid Cargo dependency target: {path}")
    return target


def _cargo_dependency_row(
    path: str,
    name: str,
    section: str,
    edge_class: EdgeClass,
    guard: str | None,
    alias: str,
    specification: object,
) -> EdgeObservation:
    target = _cargo_dependency_target(path, alias, specification)
    unresolved = (
        isinstance(specification, dict)
        and specification.get("workspace") is True
        and "package" not in specification
    )
    condition = json.dumps(specification, sort_keys=True, separators=(",", ":"))
    return _observation(
        f"rust:{name}",
        f"rust:{target}",
        "depends_on",
        (edge_class, "manifest_declared", "unresolved" if unresolved else "declared"),
        (path, f"{section}.{alias}"),
        guard=f"{guard or ''}:{condition}",
        reason="Cargo workspace dependency alias needs root resolution"
        if unresolved
        else None,
    )


def _cargo_table_rows(
    path: str,
    name: str,
    section: str,
    table: object,
    edge_class: EdgeClass,
    guard: str | None,
) -> list[EdgeObservation]:
    if not isinstance(table, dict):
        raise ValueError(f"invalid Cargo {section}: {path}")
    return [
        _cargo_dependency_row(
            path, name, section, edge_class, guard, alias, specification
        )
        for alias, specification in sorted(table.items())
    ]


def _cargo_target_rows(
    path: str, name: str, target_tables: object
) -> list[EdgeObservation]:
    if not isinstance(target_tables, dict):
        raise ValueError(f"invalid Cargo target metadata: {path}")
    rows: list[EdgeObservation] = []
    for condition, target in sorted(target_tables.items()):
        if not isinstance(target, dict):
            raise ValueError(f"invalid Cargo target condition: {path}")
        for section, edge_class in _CARGO_SECTIONS:
            rows.extend(
                _cargo_table_rows(
                    path, name, section, target.get(section, {}), edge_class, condition
                )
            )
    return rows


def _rust_manifest(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    document = tomllib.loads(raw.decode())
    package = document.get("package", {})
    if not isinstance(package, dict):
        raise ValueError(f"invalid Cargo package metadata: {path}")
    name = package.get("name")
    if name is None:
        if not isinstance(document.get("workspace"), dict):
            raise ValueError(f"Cargo manifest has no package or workspace: {path}")
        return ()  # Workspace dependency templates are resolved by member crates.
    if not isinstance(name, str) or not name:
        raise ValueError(f"invalid Cargo package name: {path}")
    rows: list[EdgeObservation] = []
    for section, edge_class in _CARGO_SECTIONS:
        rows.extend(
            _cargo_table_rows(
                path, name, section, document.get(section, {}), edge_class, None
            )
        )
    rows.extend(_cargo_target_rows(path, name, document.get("target", {})))
    return tuple(rows)


_RUST_IMPORT = re.compile(
    r"(?m)^\s*(?:pub(?:\([^)]*\))?\s+)?(?:use|extern\s+crate)\s+([A-Za-z_][A-Za-z_0-9]*)"
)
_FRONTEND_IMPORT = re.compile(
    r"\b(?:from\s*|import\s*\(|require\s*\()\s*['\"]([^'\"]+)['\"]"
)
_FRONTEND_SIDE_EFFECT_IMPORT = re.compile(r"\bimport\s+['\"]([^'\"]+)['\"]")
_FRONTEND_DYNAMIC = re.compile(r"\b(?:import|require)\s*\(\s*(?!['\"])")


def _rust_source(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    text = raw.decode("utf-8")
    rows: list[EdgeObservation] = []
    for match in _RUST_IMPORT.finditer(text):
        name = match.group(1)
        if name in {"crate", "self", "super", "std", "core", "alloc"}:
            continue
        rows.append(
            _observation(
                f"module:{path}",
                f"rust:{name.replace('_', '-')}",
                "imports",
                (
                    "production_runtime_package",
                    "statically_resolved",
                    "statically_resolved",
                ),
                (path, f"byte:{match.start()}"),
            )
        )
    if "include!(" in text or "libloading::" in text:
        rows.append(
            _observation(
                f"module:{path}",
                None,
                "dynamic_source_or_library",
                ("production_runtime_package", "dynamically_unresolved", "unresolved"),
                (path, "dynamic-source"),
                reason="Rust include or library target needs resolution",
            )
        )
    return tuple(rows)


def _frontend_source(path: str, raw: bytes) -> tuple[EdgeObservation, ...]:
    text = raw.decode("utf-8")
    rows: list[EdgeObservation] = []
    for match in (
        *_FRONTEND_IMPORT.finditer(text),
        *_FRONTEND_SIDE_EFFECT_IMPORT.finditer(text),
    ):
        target = match.group(1)
        if target.startswith((".", "/", "node:")):
            continue
        package = (
            "/".join(target.split("/")[:2])
            if target.startswith("@")
            else target.split("/")[0]
        )
        rows.append(
            _observation(
                f"module:{path}",
                f"npm:{package}",
                "imports",
                (
                    "production_runtime_package",
                    "statically_resolved",
                    "statically_resolved",
                ),
                (path, f"byte:{match.start()}"),
            )
        )
    for match in _FRONTEND_DYNAMIC.finditer(text):
        rows.append(
            _observation(
                f"module:{path}",
                None,
                "dynamic_import",
                ("production_runtime_package", "dynamically_unresolved", "unresolved"),
                (path, f"byte:{match.start()}"),
                reason="frontend import target is not a literal",
            )
        )
    return tuple(rows)


_EXTRACTORS: dict[SourceKind, Any] = {
    "python_project": _python_project,
    "python_source": _python_source,
    "rust_manifest": _rust_manifest,
    "rust_source": _rust_source,
    "compose": _compose,
    "frontend_package": _frontend_package,
    "frontend_source": _frontend_source,
    "raw_manifest": lambda path, raw: (),
}


def _capture_source_rows(
    root: Path, sources: tuple[CensusSource, ...]
) -> tuple[list[SourceIdentity], list[EdgeObservation]]:
    identities: list[SourceIdentity] = []
    observations: list[EdgeObservation] = []
    total_bytes = 0
    for source in sorted(sources, key=lambda item: item.path):
        raw = _read_source(root, source.path)
        total_bytes += len(raw)
        if total_bytes > MAX_CENSUS_TOTAL_BYTES:
            raise ValueError("census source bytes exceed bound")
        identities.append(
            SourceIdentity(
                source.path, source.kind, hashlib.sha256(raw).hexdigest(), len(raw)
            )
        )
        observations.extend(_EXTRACTORS[source.kind](source.path, raw))
    observations.sort(
        key=lambda item: (item.source_path, item.selector, item.target_id or "")
    )
    return identities, observations


def _missing_source_kinds(sources: tuple[CensusSource, ...]) -> tuple[SourceKind, ...]:
    present = {source.kind for source in sources}
    required = {"python_project", "python_source", "compose", "frontend_package"}
    return tuple(
        kind for kind in _EXTRACTORS if kind in required and kind not in present
    )


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
    if len(sources) > MAX_CENSUS_SOURCES:
        raise ValueError("census source count exceeds bound")
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("census source paths must be unique")
    identities, observations = _capture_source_rows(root, sources)
    missing = _missing_source_kinds(sources)
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
        "source_universe_digest": None,
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


def certify_census(
    root: Path,
    manifest: str,
    receipt: CensusReceipt,
    closure: FleetManifestClosure,
    universe: SourceUniverseReceipt,
) -> CensusReceipt:
    """Mark complete only after an independent Git-tree source inventory."""

    from .fleet_source_universe import verify_source_universe

    if receipt.complete or not verify_census(root, manifest, receipt):
        raise ValueError("declared census is not independently verified")
    if tuple(source.path for source in receipt.sources) != universe.source_paths:
        raise ValueError("census source set differs from Git tree universe")
    if not verify_source_universe(root, closure, universe):
        raise ValueError("fleet Git tree universe proof failed")
    certified = replace(receipt, complete=True, source_universe_digest=universe.digest)
    return replace(
        certified,
        digest=hashlib.sha256(_canonical(certified.payload())).hexdigest(),
    )


def verify_census(
    root: Path,
    manifest: str,
    receipt: CensusReceipt,
    *,
    closure: FleetManifestClosure | None = None,
    universe: SourceUniverseReceipt | None = None,
) -> bool:
    """Re-extract exact source bytes; a rehashed forged observation is refused."""

    try:
        sources = tuple(CensusSource(row.path, row.kind) for row in receipt.sources)
        expected = capture_census(
            root, manifest, sources, source_commit=receipt.source_commit
        )
        if receipt.complete:
            if closure is None or universe is None:
                return False
            expected = certify_census(root, manifest, expected, closure, universe)
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False
    return expected == receipt
