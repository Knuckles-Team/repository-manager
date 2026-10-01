"""Strict repository identity and declared-source closure for fleet census.

This parses the target repository metadata shape without changing legacy workspace
selection. A legacy manifest is rejected as incomplete instead of being promoted to
verified topology.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from repository_manager.development.fleet_census import (
    CensusReceipt,
    CensusSource,
    EdgeObservation,
    SourceKind,
    capture_census,
    certify_census,
    verify_census,
)
from repository_manager.development.fleet_source_universe import (
    SourceUniverseReceipt,
    capture_source_universe,
)
from repository_manager.development.workspace_release import (
    DependencyGraph,
    FleetEdgeObservation,
    FleetEvidenceKind,
    FleetOrderReceipt,
    build_fleet_order_receipt,
)
from repository_manager.workspace_manifest import synchronize_workspace_manifest


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
    artifact_ids: tuple[str, ...]
    unresolved_fields: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FleetManifestClosure:
    repositories: tuple[FleetRepository, ...]
    sources: tuple[CensusSource, ...]
    manifest_sha256: str
    declared_repository_count: int
    unresolved_repository_ids: tuple[str, ...]
    third_party_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class FleetMirrorReceipt:
    """Exact read-only source/runtime/portable-seed identity."""

    source_digest: str
    runtime_digest: str
    seed_digest: str
    repository_ids: tuple[str, ...]
    digest: str


@dataclass(frozen=True, slots=True)
class _ReconcileContext:
    owners: dict[str, str]
    source_owners: dict[str, str]
    source_kinds: dict[str, SourceKind]
    family_by_kind: dict[str, FleetEvidenceKind]
    external: set[str]
    source_digests: dict[str, str]
    rust_aliases: dict[tuple[str, str], str]
    rank_by_owner: dict[str, int]


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
_ALLOWED_REPOSITORY_FIELDS = set(_FIELDS) | {
    "url",
    "description",
    "discovery_sources",
    "artifact_ids",
    "unresolved_fields",
    "metadata_state",
}
_SCALAR_FIELDS = ("plane", "layer", "fleet_ordinal")
_LIST_FIELDS = (
    "provides",
    "consumes",
    "dependency_classes",
    "manifests",
    "discovery_sources",
    "artifact_ids",
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
MAX_FLEET_MANIFEST_BYTES = 2 * 1024 * 1024
MAX_FLEET_REPOSITORIES = 2048
MAX_FLEET_SOURCES = 16384


def _repo_entries(node: object, parent: tuple[str, ...] = ()) -> list[tuple[str, dict]]:
    if len(parent) > 16:
        raise ValueError("workspace nesting exceeds fleet bound")
    if not isinstance(node, dict):
        raise ValueError("workspace directory must be a mapping")
    repositories = node.get("repositories", [])
    if not isinstance(repositories, list):
        raise ValueError("repositories must be a list")
    rows = _repository_rows(repositories, parent)
    subdirs = node.get("subdirectories", {})
    if not isinstance(subdirs, dict):
        raise ValueError("subdirectories must be a mapping")
    rows.extend(_subdirectory_rows(subdirs, parent))
    return rows


def _repository_rows(
    repositories: list, parent: tuple[str, ...]
) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
    for repo in repositories:
        if not isinstance(repo, dict) or not isinstance(repo.get("url"), str):
            raise ValueError("repository must have a URL")
        name = repo["url"].strip().rstrip("/").rsplit("/", 1)[-1].removesuffix(".git")
        if not name or name in {".", ".."}:
            raise ValueError("repository URL has no safe basename")
        rows.append(("/".join((*parent, name)), repo))
    return rows


def _subdirectory_rows(
    subdirs: dict, parent: tuple[str, ...]
) -> list[tuple[str, dict]]:
    rows: list[tuple[str, dict]] = []
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
    if basename == "Cargo.toml":
        return "rust_manifest"
    if basename in {
        "compose.yml",
        "compose.yaml",
        "docker-compose.yml",
        "docker-compose.yaml",
    }:
        return "compose"
    if path.endswith(".py"):
        return "python_source"
    if path.endswith(".rs"):
        return "rust_source"
    if Path(path).suffix in {".js", ".jsx", ".ts", ".tsx"}:
        return "frontend_source"
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
    _validate_repository_identity(path, repo)
    values = _validated_repository_lists(path, repo)
    unresolved = _strings(repo, "unresolved_fields")
    _validate_repository_state(path, repo, values, unresolved)
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
        values["artifact_ids"],
        unresolved,
    )


def _validate_repository_identity(path: str, repo: dict[str, Any]) -> None:
    unknown = set(repo) - _ALLOWED_REPOSITORY_FIELDS
    if unknown:
        raise ValueError(f"repository {path} has unsupported fleet fields")
    missing = sorted(set(_FIELDS) - repo.keys())
    if missing:
        raise ValueError(f"repository {path} lacks fleet fields: {', '.join(missing)}")
    if repo["id"] != f"repo:{path}" or repo["path"] != path:
        raise ValueError(f"repository {path} has conflicting fleet identity")
    for field in _SCALAR_FIELDS:
        if not isinstance(repo[field], str) or not repo[field]:
            raise ValueError(f"repository {path} has invalid {field}")


def _validated_repository_lists(
    path: str, repo: dict[str, Any]
) -> dict[str, tuple[str, ...]]:
    values = {field: _strings(repo, field) for field in _LIST_FIELDS}
    if any(
        ":" not in artifact or artifact.startswith(":")
        for artifact in values["artifact_ids"]
    ):
        raise ValueError(f"repository {path} has invalid artifact identity")
    return values


def _validate_repository_state(
    path: str,
    repo: dict[str, Any],
    values: dict[str, tuple[str, ...]],
    unresolved: tuple[str, ...],
) -> None:
    _validate_repository_metadata_state(path, repo, values, unresolved)
    _validate_repository_classification(path, repo, values, unresolved)


def _validate_repository_metadata_state(
    path: str,
    repo: dict[str, Any],
    values: dict[str, tuple[str, ...]],
    unresolved: tuple[str, ...],
) -> None:
    if set(unresolved) - set((*_FIELDS, "discovery_sources")):
        raise ValueError(f"repository {path} has invalid unresolved fields")
    if repo.get("metadata_state", "resolved") not in {"resolved", "unresolved"}:
        raise ValueError(f"repository {path} has invalid metadata state")
    if bool(unresolved) != (repo.get("metadata_state", "resolved") == "unresolved"):
        raise ValueError(f"repository {path} has inconsistent metadata state")
    if not values["manifests"] and "manifests" not in unresolved:
        raise ValueError(f"repository {path} has no source manifests")


def _validate_repository_classification(
    path: str,
    repo: dict[str, Any],
    values: dict[str, tuple[str, ...]],
    unresolved: tuple[str, ...],
) -> None:
    if repo["fleet_ordinal"] not in _ORDINALS and not (
        repo["fleet_ordinal"] == "unknown" and "fleet_ordinal" in unresolved
    ):
        raise ValueError(f"repository {path} has invalid fleet ordinal")
    if set(values["dependency_classes"]) - _DEPENDENCY_CLASSES or (
        not values["dependency_classes"] and "dependency_classes" not in unresolved
    ):
        raise ValueError(f"repository {path} has invalid dependency classes")
    if repo["layer"] == "unknown" and "layer" not in unresolved:
        raise ValueError(f"repository {path} has unresolved layer without marker")


def parse_fleet_manifest(raw: bytes) -> FleetManifestClosure:
    """Validate every nested repo and retain explicit unknown metadata."""

    if type(raw) is not bytes or len(raw) > MAX_FLEET_MANIFEST_BYTES:
        raise ValueError("fleet manifest exceeds byte bound")
    document = yaml.safe_load(raw)
    entries = _repo_entries(document)
    if len(entries) > MAX_FLEET_REPOSITORIES:
        raise ValueError("fleet repository count exceeds bound")
    _validate_entry_identities(entries)
    repositories = tuple(
        sorted(
            (_parse_repository(path, repo) for path, repo in entries),
            key=lambda item: item.path,
        )
    )
    third_party = _third_party_artifacts(document, repositories)
    sources = _declared_sources(repositories)
    return FleetManifestClosure(
        repositories,
        sources,
        hashlib.sha256(raw).hexdigest(),
        len(entries),
        tuple(repo.repository_id for repo in repositories if repo.unresolved_fields),
        third_party,
    )


def _validate_entry_identities(entries: list[tuple[str, dict]]) -> None:
    paths = [path for path, _ in entries]
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate nested repository path")
    urls = [repo["url"].strip().rstrip("/") for _, repo in entries]
    if len(urls) != len(set(urls)):
        raise ValueError("duplicate repository URL")


def _third_party_artifacts(
    document: dict, repositories: tuple[FleetRepository, ...]
) -> tuple[str, ...]:
    artifact_ids = [item for repo in repositories for item in repo.artifact_ids]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise ValueError("fleet artifact identity has multiple owners")
    third_party = _strings(document, "third_party_artifact_ids")
    if len(third_party) > MAX_FLEET_SOURCES or set(third_party) & set(artifact_ids):
        raise ValueError("fleet third-party artifacts overlap owners or exceed bound")
    if any(":" not in artifact or artifact.startswith(":") for artifact in third_party):
        raise ValueError("fleet third-party artifact identity is invalid")
    return third_party


def _declared_sources(
    repositories: tuple[FleetRepository, ...],
) -> tuple[CensusSource, ...]:
    sources: list[CensusSource] = []
    for repo in repositories:
        for local in (*repo.manifests, *repo.discovery_sources):
            path = _safe_source_path(repo.path, local)
            sources.append(CensusSource(path, _source_kind(local)))
            if len(sources) > MAX_FLEET_SOURCES:
                raise ValueError("fleet source count exceeds bound")
    if len({source.path for source in sources}) != len(sources):
        raise ValueError("source listed by multiple manifest records")
    return tuple(sorted(sources, key=lambda item: item.path))


def load_fleet_manifest(
    root: Path, manifest: str = "workspace.yml"
) -> FleetManifestClosure:
    """Read the canonical workspace source and validate its fleet metadata."""

    manifest_path = root / manifest
    if manifest_path.is_symlink() or not manifest_path.resolve().is_relative_to(
        root.resolve()
    ):
        raise ValueError("workspace manifest escapes root")
    if manifest_path.stat().st_size > MAX_FLEET_MANIFEST_BYTES:
        raise ValueError("fleet manifest exceeds byte bound")
    return parse_fleet_manifest(manifest_path.read_bytes())


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


def capture_complete_manifest_census(
    root: Path, *, manifest: str = "workspace.yml", source_commit: str
) -> tuple[FleetManifestClosure, CensusReceipt, SourceUniverseReceipt]:
    """Independently prove every supported source is declared and committed."""

    closure, declared = capture_manifest_census(
        root, manifest=manifest, source_commit=source_commit
    )
    if (
        closure.unresolved_repository_ids
        or declared.missing_kinds
        or declared.unresolved_count
    ):
        raise ValueError("fleet source or repository metadata remains unresolved")
    universe = capture_source_universe(root, closure)
    return (
        closure,
        certify_census(root, manifest, declared, closure, universe),
        universe,
    )


def validate_fleet_manifest_mirrors(
    source: Path, *, runtime: Path, seed: Path
) -> FleetMirrorReceipt:
    """Require complete target metadata and exact governed mirror projections.

    This is a read-only release preflight. The ordinary selector/mirror path
    remains available for current manifests that have not yet been upgraded.
    """

    report = synchronize_workspace_manifest(
        source, runtime_destination=runtime, seed_destination=seed, check=True
    )
    if not report.synchronized:
        raise ValueError("fleet manifest mirrors differ from canonical projections")
    source_bytes = source.read_bytes()
    runtime_bytes = runtime.read_bytes()
    seed_bytes = seed.read_bytes()
    if hashlib.sha256(source_bytes).hexdigest() != report.source_digest:
        raise ValueError("fleet manifest source changed during mirror check")
    if runtime_bytes != source_bytes:
        raise ValueError("fleet runtime mirror differs from canonical source")
    closure = parse_fleet_manifest(source_bytes)
    seed_closure = parse_fleet_manifest(seed_bytes)
    if closure.unresolved_repository_ids or seed_closure.unresolved_repository_ids:
        raise ValueError("fleet manifest metadata remains unresolved")
    if (
        closure.repositories != seed_closure.repositories
        or closure.sources != seed_closure.sources
        or closure.third_party_artifact_ids != seed_closure.third_party_artifact_ids
    ):
        raise ValueError("portable fleet projection changed topology")
    payload = (
        report.source_digest,
        hashlib.sha256(runtime_bytes).hexdigest(),
        hashlib.sha256(seed_bytes).hexdigest(),
        tuple(row.repository_id for row in closure.repositories),
    )
    digest = hashlib.sha256(
        json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return FleetMirrorReceipt(*payload, digest)


def validated_declared_source_receipts(
    root: Path,
    closure: FleetManifestClosure,
    receipt: CensusReceipt,
    *,
    universe: SourceUniverseReceipt | None = None,
) -> dict[FleetEvidenceKind, str]:
    """Derive four source-scope digests from a reverified declared census.

    This proves the declared source set, not that the manifest listed every
    runtime discovery point in the product tree. Fleet activation still needs
    independent source-universe and live-consumer gates.
    """

    _validate_declared_census(root, closure, receipt, universe)
    by_kind = {
        FleetEvidenceKind.DYNAMIC: ("python_source", "rust_source"),
        FleetEvidenceKind.RESOLVER: ("python_project", "rust_manifest"),
        FleetEvidenceKind.DEPLOYMENT: ("compose",),
        FleetEvidenceKind.FRONTEND: ("frontend_package", "frontend_source"),
    }
    result: dict[FleetEvidenceKind, str] = {}
    for family, source_kinds in by_kind.items():
        identities = tuple(
            (source.path, source.sha256, source.size)
            for source in receipt.sources
            if source.kind in source_kinds
        )
        if not identities:
            raise ValueError("fleet census is missing a source class")
        payload = (closure.manifest_sha256, receipt.digest, family.value, identities)
        result[family] = hashlib.sha256(
            json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode()
        ).hexdigest()
    return result


def _validate_declared_census(
    root: Path,
    closure: FleetManifestClosure,
    receipt: CensusReceipt,
    universe: SourceUniverseReceipt | None,
) -> None:
    if load_fleet_manifest(root) != closure:
        raise ValueError("fleet manifest closure differs from canonical source")
    if closure.unresolved_repository_ids:
        raise ValueError("fleet repository metadata remains unresolved")
    if receipt.manifest_sha256 != closure.manifest_sha256:
        raise ValueError("fleet census manifest differs from target schema")
    if tuple((s.path, s.kind) for s in receipt.sources) != tuple(
        (s.path, s.kind) for s in closure.sources
    ):
        raise ValueError("fleet census source set differs from manifest")
    if receipt.missing_kinds or receipt.unresolved_count:
        raise ValueError("fleet census source classes or edges remain unresolved")
    if not verify_census(
        root, "workspace.yml", receipt, closure=closure, universe=universe
    ):
        raise ValueError("fleet census source receipt failed re-extraction")


def reconcile_fleet_census(
    root: Path,
    closure: FleetManifestClosure,
    receipt: CensusReceipt,
    graph: DependencyGraph,
    *,
    universe: SourceUniverseReceipt | None = None,
) -> FleetOrderReceipt:
    """Map every declared observation to a canonical owner or explicit external.

    Unknown artifact identities never become inferred repository edges. Dev/test
    and explicit third-party observations are accounted for but excluded from
    the production/deployment order. The resulting order still represents the
    declared source universe only.
    """

    coverage = validated_declared_source_receipts(
        root, closure, receipt, universe=universe
    )
    context = _reconcile_context(closure, receipt, graph)
    rows = [
        edge
        for observation in receipt.observations
        if (edge := _product_observation_edge(observation, context)) is not None
    ]
    return build_fleet_order_receipt(
        graph,
        manifest_digest=closure.manifest_sha256,
        observations=tuple(sorted(set(rows))),
        coverage_receipts=coverage,
    )


def _reconcile_context(
    closure: FleetManifestClosure, receipt: CensusReceipt, graph: DependencyGraph
) -> _ReconcileContext:
    repository_ids = {repo.repository_id for repo in closure.repositories}
    if {project.project_id for project in graph.projects} != repository_ids:
        raise ValueError("fleet package graph membership differs from manifest")
    owners = {
        artifact: repo.repository_id
        for repo in closure.repositories
        for artifact in repo.artifact_ids
    }
    source_owners = {
        f"{repo.path}/{local}": repo.repository_id
        for repo in closure.repositories
        for local in (*repo.manifests, *repo.discovery_sources)
    }
    source_kinds = {source.path: source.kind for source in closure.sources}
    family_by_kind = {
        "python_source": FleetEvidenceKind.DYNAMIC,
        "rust_source": FleetEvidenceKind.DYNAMIC,
        "python_project": FleetEvidenceKind.RESOLVER,
        "rust_manifest": FleetEvidenceKind.RESOLVER,
        "compose": FleetEvidenceKind.DEPLOYMENT,
        "frontend_package": FleetEvidenceKind.FRONTEND,
        "frontend_source": FleetEvidenceKind.FRONTEND,
    }
    external = set(closure.third_party_artifact_ids)
    source_digests = {source.path: source.sha256 for source in receipt.sources}
    rust_aliases = _cargo_aliases(receipt, source_kinds, owners)
    rank_by_owner = _layer_ranks(closure, graph)
    return _ReconcileContext(
        owners,
        source_owners,
        source_kinds,
        family_by_kind,
        external,
        source_digests,
        rust_aliases,
        rank_by_owner,
    )


def _cargo_aliases(
    receipt: CensusReceipt,
    source_kinds: dict[str, SourceKind],
    owners: dict[str, str],
) -> dict[tuple[str, str], str]:
    rust_aliases: dict[tuple[str, str], str] = {}
    for observation in receipt.observations:
        if source_kinds.get(observation.source_path) != "rust_manifest":
            continue
        owner = owners.get(observation.source_id)
        if owner is None or observation.target_id is None:
            continue
        alias = observation.selector.rsplit(".", 1)[-1].replace("_", "-")
        key = (owner, f"rust:{alias}")
        old = rust_aliases.setdefault(key, observation.target_id)
        if old != observation.target_id:
            raise ValueError("Cargo alias resolves to conflicting artifact identities")
    return rust_aliases


def _layer_ranks(
    closure: FleetManifestClosure, graph: DependencyGraph
) -> dict[str, int]:
    layer_rank = {value: index for index, value in enumerate(sorted(_ORDINALS))}
    rank_by_owner = {
        repo.repository_id: layer_rank[repo.fleet_ordinal]
        for repo in closure.repositories
    }
    for edge_dependent, edge_dependency in graph.project_edges:
        if rank_by_owner[edge_dependency] > rank_by_owner[edge_dependent]:
            raise ValueError("fleet package edge points to a higher layer")
    return rank_by_owner


def _observation_source(
    observation: EdgeObservation, context: _ReconcileContext
) -> tuple[str, str, SourceKind]:
    path = observation.source_path
    kind = context.source_kinds.get(path)
    if kind is None or kind not in context.family_by_kind:
        raise ValueError("fleet observation source is not a supported evidence class")
    target_id = observation.target_id
    if target_id is None or observation.resolution == "unresolved":
        raise ValueError("fleet observation remains unresolved")
    dependent: str | None
    if observation.source_id.startswith("module:") and kind in {
        "python_source",
        "rust_source",
        "frontend_source",
    }:
        dependent = context.source_owners[path]
    else:
        dependent = context.owners.get(observation.source_id)
    if dependent is None:
        raise ValueError("fleet observation source has no declared artifact owner")
    if kind == "rust_source":
        target_id = context.rust_aliases.get((dependent, target_id), target_id)
    return dependent, target_id, kind


def _product_observation_edge(
    observation: EdgeObservation, context: _ReconcileContext
) -> FleetEdgeObservation | None:
    dependent, target_id, kind = _observation_source(observation, context)
    if target_id in context.external:
        return None
    dependency = context.owners.get(target_id)
    if dependency is None:
        raise ValueError("fleet observation target has no declared artifact owner")
    if observation.edge_class in {"development_test", "build_generation"}:
        return None
    if observation.edge_class not in {
        "production_runtime_package",
        "deployment_composition",
    }:
        raise ValueError("fleet observation edge class is unsupported")
    if context.rank_by_owner[dependency] > context.rank_by_owner[dependent]:
        raise ValueError("fleet product edge points to a higher layer")
    if dependent == dependency:
        return None
    return FleetEdgeObservation(
        context.family_by_kind[kind],
        dependent,
        dependency,
        context.source_digests[observation.source_path],
    )
