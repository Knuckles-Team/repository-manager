"""Strict repository identity and declared-source closure for fleet census.

This parses the target repository metadata shape without changing legacy workspace
selection. A legacy manifest is rejected as incomplete instead of being promoted to
verified topology.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from repository_manager.development.fleet_census import (
    CensusReceipt,
    CensusSource,
    SourceKind,
    capture_census,
)


@dataclass(frozen=True, slots=True)
class FleetRepository:
    repository_id: str
    path: str
    plane: str
    layer: str
    fleet_ordinal: str
    provides: tuple[str, ...]
    consumes: tuple[str, ...]
    dependency_classes: tuple[str, ...]
    manifests: tuple[str, ...]
    discovery_sources: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FleetManifestClosure:
    repositories: tuple[FleetRepository, ...]
    sources: tuple[CensusSource, ...]
    manifest_sha256: str
    declared_repository_count: int


_FIELDS = (
    "id",
    "path",
    "plane",
    "layer",
    "fleet_ordinal",
    "provides",
    "consumes",
    "dependency_classes",
    "manifests",
)
_SCALAR_FIELDS = ("plane", "layer", "fleet_ordinal")
_LIST_FIELDS = (
    "provides",
    "consumes",
    "dependency_classes",
    "manifests",
    "discovery_sources",
)
_ORDINALS = {
    "L0_contracts",
    "L1_eg_authority",
    "L2_au_application",
    "L3_adapters",
    "L4_products",
    "L5_composition",
}
_DEPENDENCY_CLASSES = {
    "production",
    "runtime",
    "package",
    "build",
    "generation",
    "development",
    "test_support",
    "deployment",
    "documentation",
}


def _repo_entries(node: object, parent: tuple[str, ...] = ()) -> list[tuple[str, dict]]:
    if not isinstance(node, dict):
        raise ValueError("workspace directory must be a mapping")
    rows: list[tuple[str, dict]] = []
    repositories = node.get("repositories", [])
    if not isinstance(repositories, list):
        raise ValueError("repositories must be a list")
    for repo in repositories:
        if not isinstance(repo, dict) or not isinstance(repo.get("url"), str):
            raise ValueError("repository must have a URL")
        name = repo["url"].strip().rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        if not name or name in {".", ".."}:
            raise ValueError("repository URL has no safe basename")
        rows.append(("/".join((*parent, name)), repo))
    subdirs = node.get("subdirectories", {})
    if not isinstance(subdirs, dict):
        raise ValueError("subdirectories must be a mapping")
    for name, child in sorted(subdirs.items()):
        if not isinstance(name, str) or not name or name in {".", ".."} or "/" in name:
            raise ValueError("subdirectory has unsafe name")
        rows.extend(_repo_entries(child, (*parent, name)))
    return rows


def _strings(repo: dict[str, Any], field: str) -> tuple[str, ...]:
    value = repo.get(field, [])
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise ValueError(f"repository {field} must be a string list")
    if len(value) != len(set(value)):
        raise ValueError(f"repository {field} has duplicates")
    return tuple(value)


def _source_kind(path: str) -> SourceKind:
    basename = Path(path).name
    if basename == "pyproject.toml":
        return "python_project"
    if basename == "package.json":
        return "frontend_package"
    if basename in {
        "compose.yml",
        "compose.yaml",
        "docker-compose.yml",
        "docker-compose.yaml",
    }:
        return "compose"
    if path.endswith(".py"):
        return "python_source"
    return "raw_manifest"


def _safe_source_path(repository_path: str, local: str) -> str:
    value = Path(local)
    if (
        value.is_absolute()
        or not value.parts
        or value.as_posix() != local
        or any(part in {".", ".."} for part in value.parts)
    ):
        raise ValueError(f"unsafe repository source path: {local}")
    return f"{repository_path}/{value.as_posix()}"


def _parse_repository(path: str, repo: dict[str, Any]) -> FleetRepository:
    missing = sorted(set(_FIELDS) - repo.keys())
    if missing:
        raise ValueError(f"repository {path} lacks fleet fields: {', '.join(missing)}")
    if repo["id"] != f"repo:{path}" or repo["path"] != path:
        raise ValueError(f"repository {path} has conflicting fleet identity")
    for field in _SCALAR_FIELDS:
        if not isinstance(repo[field], str) or not repo[field]:
            raise ValueError(f"repository {path} has invalid {field}")
    values = {field: _strings(repo, field) for field in _LIST_FIELDS}
    if not values["manifests"]:
        raise ValueError(f"repository {path} has no source manifests")
    if repo["fleet_ordinal"] not in _ORDINALS:
        raise ValueError(f"repository {path} has invalid fleet ordinal")
    if (
        not values["dependency_classes"]
        or set(values["dependency_classes"]) - _DEPENDENCY_CLASSES
    ):
        raise ValueError(f"repository {path} has invalid dependency classes")
    for local in (*values["manifests"], *values["discovery_sources"]):
        _safe_source_path(path, local)
    return FleetRepository(
        repo["id"],
        path,
        repo["plane"],
        repo["layer"],
        repo["fleet_ordinal"],
        values["provides"],
        values["consumes"],
        values["dependency_classes"],
        values["manifests"],
        values["discovery_sources"],
    )


def load_fleet_manifest(
    root: Path, manifest: str = "workspace.yml"
) -> FleetManifestClosure:
    """Require metadata and a source list for every repository in the workspace."""

    manifest_path = root / manifest
    if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("workspace manifest escapes root")
    raw = manifest_path.read_bytes()
    document = yaml.safe_load(raw)
    entries = _repo_entries(document)
    paths = [path for path, _ in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate nested repository path")
    urls = [repo["url"].strip().rstrip("/") for _, repo in entries]
    if len(urls) != len(set(urls)):
        raise ValueError("duplicate repository URL")
    repositories = tuple(
        sorted(
            (_parse_repository(path, repo) for path, repo in entries),
            key=lambda item: item.path,
        )
    )
    sources: list[CensusSource] = []
    for repo in repositories:
        for local in (*repo.manifests, *repo.discovery_sources):
            path = _safe_source_path(repo.path, local)
            sources.append(CensusSource(path, _source_kind(local)))
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("source listed by multiple manifest records")
    return FleetManifestClosure(
        repositories,
        tuple(sorted(sources, key=lambda item: item.path)),
        hashlib.sha256(raw).hexdigest(),
        len(entries),
    )


def capture_manifest_census(
    root: Path, *, manifest: str = "workspace.yml", source_commit: str
) -> tuple[FleetManifestClosure, CensusReceipt]:
    """Capture every declared repository artifact while preserving unknown runtime scope."""

    closure = load_fleet_manifest(root, manifest)
    receipt = capture_census(
        root, manifest, closure.sources, source_commit=source_commit
    )
    if receipt.manifest_sha256 != closure.manifest_sha256:
        raise ValueError("manifest changed during census capture")
    return closure, receipt
