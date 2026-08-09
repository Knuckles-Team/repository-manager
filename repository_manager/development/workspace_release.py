"""Pure C-11 workspace release records and dependency graph validation.

This module is deliberately a planning boundary, not a release executor.  It
contains immutable records for repository/package identity, versions, dependency
edges, and version floors, together with a deterministic in-memory graph builder.
It does not read a checkout, run a command, contact a network, or write a file.
The bounded metadata adapters live in :mod:`workspace_metadata`.

Repository identity is always the canonical workspace-relative identifier.  A
repository's final path component is a display label only; it is never used as
an identity or as an owner lookup key.  This matters for workspaces containing,
for example, both ``services/foo`` and ``agent-packages/foo``.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeVar, cast

C11_CONTRACT_VERSION = 1

# These are model-level bounds.  File and parsed-document bounds belong to the
# metadata reader's ``MetadataLimits`` so callers can tune those independently.
MAX_PROJECTS = 2_048
MAX_PACKAGES = 8_192
MAX_EDGES = 32_768
MAX_PLAN_STAGES = 16_384
MAX_STRING_LENGTH = 4_096

_SHA = re.compile(r"^[0-9a-fA-F]{40,64}$")
_VERSION = re.compile(
    r"^(?P<major>0|[1-9][0-9]*)\."
    r"(?P<minor>0|[1-9][0-9]*)\."
    r"(?P<patch>0|[1-9][0-9]*)"
    r"(?:[-+](?P<suffix>[0-9A-Za-z][0-9A-Za-z.-]*))?$"
)
_PYTHON_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RUST_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]*$")
_NODE_NAME = re.compile(r"^(?:@[A-Za-z0-9._~-]+/)?[A-Za-z0-9._~-]+$")
_FLOOR = re.compile(r"^(?P<operator>\^|~=|~|>=|>|==)?\s*(?P<version>[^\s,|]+)$")
_T = TypeVar("_T")


class WorkspaceReleaseError(ValueError):
    """Base error for an invalid pure workspace release record or graph."""


class GraphDiagnosticCode(StrEnum):
    """Stable diagnostic codes emitted before a release plan can be applied."""

    DUPLICATE_PROJECT = "duplicate_project"
    DUPLICATE_PACKAGE = "duplicate_package"
    CONFLICTING_VERSION_SOURCE = "conflicting_version_source"
    MISSING_PROJECT = "missing_project"
    MISSING_PACKAGE = "missing_package"
    AMBIGUOUS_PACKAGE_OWNER = "ambiguous_package_owner"
    DUPLICATE_EDGE = "duplicate_edge"
    DUPLICATE_REWRITE = "duplicate_rewrite"
    CONFLICTING_FLOOR = "conflicting_floor"
    CONFLICTING_REWRITE = "conflicting_rewrite"
    CYCLE = "cycle"
    UNKNOWN_FIELD = "unknown_field"
    INVALID_METADATA = "invalid_metadata"


@dataclass(frozen=True, slots=True)
class Diagnostic:
    """One deterministic, privacy-safe planning diagnostic."""

    code: GraphDiagnosticCode
    subject: str
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, GraphDiagnosticCode):
            raise WorkspaceReleaseError("diagnostic code must be supported")
        _bounded_text(self.subject, "diagnostic subject", max_length=MAX_STRING_LENGTH)
        _bounded_text(self.message, "diagnostic message", max_length=MAX_STRING_LENGTH)
        details: list[tuple[str, str]] = []
        for item in _bounded_sequence(
            self.details, "diagnostic details", max_items=MAX_PLAN_STAGES
        ):
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise WorkspaceReleaseError(
                    "diagnostic details must contain key/value pairs"
                )
            key, value = item
            if not isinstance(key, str) or not isinstance(value, str):
                raise WorkspaceReleaseError("diagnostic details must contain strings")
            details.append((key, value))
        object.__setattr__(
            self,
            "details",
            tuple(
                sorted(
                    (
                        (
                            _bounded_text(key, "diagnostic detail key", max_length=128),
                            _bounded_text(
                                value,
                                "diagnostic detail value",
                                max_length=MAX_STRING_LENGTH,
                            ),
                        )
                        for key, value in details
                    )
                )
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return stable JSON-compatible data for reports and graph digests."""

        return {
            "code": self.code.value,
            "subject": self.subject,
            "message": self.message,
            "details": [[key, value] for key, value in self.details],
        }


class GraphValidationError(WorkspaceReleaseError):
    """A graph refused to build because one or more diagnostics were found."""

    def __init__(self, diagnostics: Iterable[Diagnostic]):
        ordered = tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.code.value,
                    item.subject,
                    item.message,
                    item.details,
                ),
            )
        )
        if not ordered:
            raise ValueError("GraphValidationError requires at least one diagnostic")
        self.diagnostics = ordered
        summary = "; ".join(
            f"{item.code.value}: {item.subject}: {item.message}"
            + (
                f" ({', '.join(value for _, value in item.details)})"
                if item.details
                else ""
            )
            for item in ordered
        )
        super().__init__(summary)


def _bounded_text(
    value: object, field_name: str, *, max_length: int = MAX_STRING_LENGTH
) -> str:
    if not isinstance(value, str):
        raise WorkspaceReleaseError(f"{field_name} must be a string")
    if not value or value.strip() != value:
        raise WorkspaceReleaseError(f"{field_name} must be non-blank and trimmed")
    if len(value) > max_length:
        raise WorkspaceReleaseError(f"{field_name} exceeds the bounded length")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise WorkspaceReleaseError(f"{field_name} contains a control character")
    return value


def _bounded_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise WorkspaceReleaseError(f"{field_name} must be a boolean")
    return value


def _bounded_sequence(
    value: object, field_name: str, *, max_items: int
) -> tuple[object, ...]:
    """Copy one collection while refusing strings, mappings, and overflows."""

    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Iterable
    ):
        raise WorkspaceReleaseError(f"{field_name} must be a sequence")
    result = tuple(value)
    if len(result) > max_items:
        raise WorkspaceReleaseError(f"{field_name} exceeds the bounded item count")
    return result


def _typed_sequence(
    value: object,
    field_name: str,
    item_type: type[_T],
    *,
    max_items: int,
) -> tuple[_T, ...]:
    values = _bounded_sequence(value, field_name, max_items=max_items)
    if any(not isinstance(item, item_type) for item in values):
        raise WorkspaceReleaseError(
            f"{field_name} entries must be {item_type.__name__} values"
        )
    return cast(tuple[_T, ...], values)


def canonical_repository_id(value: str) -> str:
    """Return the canonical ``repo:<workspace-relative-path>`` identity.

    Callers may pass the wire form or the workspace-relative path.  Absolute
    paths, URL-looking values, backslashes, and traversal components are
    rejected so a basename cannot accidentally become an identity.
    """

    text = _bounded_text(value, "repository identity", max_length=MAX_STRING_LENGTH)
    if text.startswith("repo:"):
        text = text[5:]
    if not text or "://" in text or "\\" in text or ":" in text:
        raise WorkspaceReleaseError(
            "repository identity must be a workspace-relative path, not a URL"
        )
    raw_parts = text.split("/")
    if any(part in {"", ".", ".."} for part in raw_parts):
        raise WorkspaceReleaseError(
            "repository identity must be a canonical workspace-relative path"
        )
    path = PurePosixPath(text)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise WorkspaceReleaseError(
            "repository identity must be a canonical workspace-relative path"
        )
    return f"repo:{path.as_posix()}"


class Ecosystem(StrEnum):
    """Metadata ecosystem supported by checkpoint 1."""

    PYTHON = "python"
    RUST = "rust"
    NODE = "node"


class EdgeConfidence(StrEnum):
    """How strongly a resolved edge owner was established."""

    EXPLICIT = "explicit"
    INFERRED = "inferred"


def _normalize_package_name(ecosystem: Ecosystem, value: str) -> str:
    text = _bounded_text(value, "package name", max_length=256)
    pattern = {
        Ecosystem.PYTHON: _PYTHON_NAME,
        Ecosystem.RUST: _RUST_NAME,
        Ecosystem.NODE: _NODE_NAME,
    }[ecosystem]
    if not pattern.fullmatch(text):
        raise WorkspaceReleaseError(
            f"invalid {ecosystem.value} package name: {value!r}"
        )
    if ecosystem == Ecosystem.PYTHON:
        return re.sub(r"[-_.]+", "-", text).lower()
    if ecosystem == Ecosystem.RUST:
        return text.replace("_", "-").lower()
    return text


@dataclass(frozen=True, order=True, slots=True)
class PackageKey:
    """Canonical package identity scoped by its owning repository."""

    repository_id: str
    ecosystem: Ecosystem
    name: str

    def __post_init__(self) -> None:
        if not isinstance(self.ecosystem, Ecosystem):
            raise WorkspaceReleaseError("package ecosystem must be supported")
        object.__setattr__(
            self, "repository_id", canonical_repository_id(self.repository_id)
        )
        object.__setattr__(
            self, "name", _normalize_package_name(self.ecosystem, self.name)
        )

    @property
    def value(self) -> str:
        """Stable wire identity; the repository path is intentionally included."""

        return f"{self.repository_id}::{self.ecosystem.value}:{self.name}"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True, slots=True)
class Version:
    """Strict three-component release version used by all three ecosystems."""

    value: str

    def __post_init__(self) -> None:
        value = _bounded_text(self.value, "version", max_length=128)
        match = _VERSION.fullmatch(value)
        if match is None:
            raise WorkspaceReleaseError(
                "version must use MAJOR.MINOR.PATCH with an optional suffix"
            )
        object.__setattr__(self, "value", value)

    @property
    def numeric(self) -> tuple[int, int, int]:
        """Return the numeric portion for deterministic floor comparisons."""

        match = _VERSION.fullmatch(self.value)
        assert match is not None
        return tuple(int(match.group(part)) for part in ("major", "minor", "patch"))  # type: ignore[return-value]

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True, order=True, slots=True)
class VersionFloor:
    """A supported lower-bound/equality constraint for one dependency edge."""

    operator: str
    version: Version

    def __post_init__(self) -> None:
        if not isinstance(self.version, Version):
            raise WorkspaceReleaseError("floor version must be a Version")
        operator = _bounded_text(self.operator, "floor operator", max_length=2)
        if operator not in {"^", "~", "~=", ">=", ">", "=="}:
            raise WorkspaceReleaseError(
                f"unsupported dependency floor operator: {operator}"
            )
        object.__setattr__(self, "operator", operator)

    @property
    def value(self) -> str:
        return f"{self.operator}{self.version.value}"

    @classmethod
    def parse(cls, value: str) -> VersionFloor:
        """Parse one bounded floor; compound/disjunctive specs are refused."""

        text = _bounded_text(value, "dependency floor", max_length=256)
        match = _FLOOR.fullmatch(text)
        if match is None:
            raise WorkspaceReleaseError(
                f"unsupported dependency floor format: {value!r}"
            )
        operator = match.group("operator") or "=="
        return cls(operator=operator, version=Version(match.group("version")))


@dataclass(frozen=True, order=True, slots=True)
class VersionSource:
    """One declared version and its non-executable metadata location."""

    location: str
    version: Version

    def __post_init__(self) -> None:
        if not isinstance(self.version, Version):
            raise WorkspaceReleaseError("version source must contain a Version")
        object.__setattr__(
            self, "location", _bounded_text(self.location, "version source location")
        )


def resolve_version_sources(
    package: PackageKey, sources: Iterable[VersionSource]
) -> tuple[Version, tuple[VersionSource, ...]]:
    """Resolve matching version declarations or raise a conflict diagnostic."""

    if not isinstance(package, PackageKey):
        raise WorkspaceReleaseError("version source package must be a PackageKey")
    ordered = tuple(
        sorted(sources, key=lambda source: (source.location, source.version.value))
    )
    if not ordered:
        raise WorkspaceReleaseError(f"package {package} has no version source")
    versions = {source.version.value for source in ordered}
    if len(versions) > 1:
        details = tuple((source.location, source.version.value) for source in ordered)
        diagnostic = Diagnostic(
            code=GraphDiagnosticCode.CONFLICTING_VERSION_SOURCE,
            subject=package.value,
            message="version sources disagree",
            details=details,
        )
        raise GraphValidationError((diagnostic,))
    return ordered[0].version, ordered


@dataclass(frozen=True, order=True, slots=True)
class PackageReference:
    """Dependency target before owner resolution; repository is optional."""

    ecosystem: Ecosystem
    name: str
    repository_id: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.ecosystem, Ecosystem):
            raise WorkspaceReleaseError("package reference ecosystem must be supported")
        object.__setattr__(
            self, "name", _normalize_package_name(self.ecosystem, self.name)
        )
        if self.repository_id is not None:
            object.__setattr__(
                self, "repository_id", canonical_repository_id(self.repository_id)
            )


@dataclass(frozen=True, order=True, slots=True)
class DependencySpec:
    """One package dependency as read from project metadata or an overlay."""

    target: PackageReference
    floor: VersionFloor | None = None
    source: str = "metadata"

    def __post_init__(self) -> None:
        if not isinstance(self.target, PackageReference):
            raise WorkspaceReleaseError("dependency target must be a PackageReference")
        if self.floor is not None and not isinstance(self.floor, VersionFloor):
            raise WorkspaceReleaseError("dependency floor must be a VersionFloor")
        object.__setattr__(
            self, "source", _bounded_text(self.source, "dependency source")
        )


@dataclass(frozen=True, order=True, slots=True)
class PackageRecord:
    """Immutable package metadata collected from one repository."""

    key: PackageKey
    version: Version
    version_sources: tuple[VersionSource, ...]
    dependencies: tuple[DependencySpec, ...] = ()
    metadata_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.key, PackageKey):
            raise WorkspaceReleaseError("package record key must be a PackageKey")
        if not isinstance(self.version, Version):
            raise WorkspaceReleaseError("package record version must be a Version")
        source_values = _typed_sequence(
            self.version_sources,
            "package version sources",
            VersionSource,
            max_items=MAX_STRING_LENGTH,
        )
        resolved, sources = resolve_version_sources(self.key, source_values)
        if resolved != self.version:
            raise GraphValidationError(
                (
                    Diagnostic(
                        code=GraphDiagnosticCode.CONFLICTING_VERSION_SOURCE,
                        subject=self.key.value,
                        message="package version disagrees with version sources",
                        details=(("record", self.version.value),),
                    ),
                )
            )
        object.__setattr__(self, "version_sources", sources)
        dependency_values = _typed_sequence(
            self.dependencies,
            "package dependencies",
            DependencySpec,
            max_items=MAX_EDGES,
        )
        object.__setattr__(
            self,
            "dependencies",
            tuple(
                sorted(
                    dependency_values,
                    key=lambda item: (
                        item.target.ecosystem.value,
                        item.target.name,
                        item.target.repository_id or "",
                        item.floor.value if item.floor else "",
                        item.source,
                    ),
                )
            ),
        )
        metadata_values = _bounded_sequence(
            self.metadata_files,
            "package metadata files",
            max_items=MAX_STRING_LENGTH,
        )
        if any(not isinstance(item, str) for item in metadata_values):
            raise WorkspaceReleaseError("package metadata files must be strings")
        object.__setattr__(
            self,
            "metadata_files",
            tuple(
                sorted(
                    set(
                        _bounded_text(item, "package metadata file")
                        for item in metadata_values
                    )
                )
            ),
        )


@dataclass(frozen=True, order=True, slots=True)
class ProjectRecord:
    """One canonical repository and its package metadata."""

    repository_id: str
    tree_sha: str = ""
    packages: tuple[PackageRecord, ...] = ()
    metadata_files: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "repository_id", canonical_repository_id(self.repository_id)
        )
        if self.tree_sha:
            value = _bounded_text(self.tree_sha, "tree SHA", max_length=64)
            if _SHA.fullmatch(value) is None:
                raise WorkspaceReleaseError(
                    "tree SHA must be a full hexadecimal Git SHA"
                )
            object.__setattr__(self, "tree_sha", value.lower())
        package_values = _typed_sequence(
            self.packages,
            "project packages",
            PackageRecord,
            max_items=MAX_PACKAGES,
        )
        packages = tuple(sorted(package_values, key=lambda item: item.key.value))
        for package in packages:
            if package.key.repository_id != self.repository_id:
                raise WorkspaceReleaseError(
                    "package identity must use its owning project's repository identity"
                )
        object.__setattr__(self, "packages", packages)
        metadata_values = _bounded_sequence(
            self.metadata_files,
            "project metadata files",
            max_items=MAX_STRING_LENGTH,
        )
        if any(not isinstance(item, str) for item in metadata_values):
            raise WorkspaceReleaseError("project metadata files must be strings")
        object.__setattr__(
            self,
            "metadata_files",
            tuple(
                sorted(
                    set(
                        _bounded_text(item, "project metadata file")
                        for item in metadata_values
                    )
                )
            ),
        )

    @property
    def project_id(self) -> str:
        return self.repository_id


@dataclass(frozen=True, order=True, slots=True)
class DependencyEdge:
    """A resolved package edge with its provenance and optional floor."""

    dependent: PackageKey
    dependency: PackageKey
    floor: VersionFloor | None = None
    source: str = "metadata"
    confidence: EdgeConfidence = EdgeConfidence.EXPLICIT

    def __post_init__(self) -> None:
        if not isinstance(self.dependent, PackageKey) or not isinstance(
            self.dependency, PackageKey
        ):
            raise WorkspaceReleaseError(
                "dependency edge endpoints must be PackageKey values"
            )
        if self.floor is not None and not isinstance(self.floor, VersionFloor):
            raise WorkspaceReleaseError("dependency edge floor must be a VersionFloor")
        if not isinstance(self.confidence, EdgeConfidence):
            raise WorkspaceReleaseError("dependency edge confidence must be supported")
        if self.dependent == self.dependency:
            raise WorkspaceReleaseError("dependency graph cannot contain self-edges")
        object.__setattr__(self, "source", _bounded_text(self.source, "edge source"))

    @property
    def dependent_project_id(self) -> str:
        return self.dependent.repository_id

    @property
    def dependency_project_id(self) -> str:
        return self.dependency.repository_id

    @property
    def value(self) -> str:
        floor = self.floor.value if self.floor else ""
        return f"{self.dependent.value}->{self.dependency.value}{floor}"


@dataclass(frozen=True, order=True, slots=True)
class FloorRewrite:
    """Pure old/new floor data; applying it is owned by a later checkpoint."""

    dependent: PackageKey
    dependency: PackageKey
    old_floor: VersionFloor
    new_floor: VersionFloor
    source: str = "metadata"

    def __post_init__(self) -> None:
        if not isinstance(self.dependent, PackageKey) or not isinstance(
            self.dependency, PackageKey
        ):
            raise WorkspaceReleaseError(
                "floor rewrite endpoints must be PackageKey values"
            )
        if not isinstance(self.old_floor, VersionFloor) or not isinstance(
            self.new_floor, VersionFloor
        ):
            raise WorkspaceReleaseError(
                "floor rewrite values must be VersionFloor values"
            )
        object.__setattr__(self, "source", _bounded_text(self.source, "floor source"))


class ReleaseStage(StrEnum):
    """C-11 stage labels; execution is intentionally out of scope here."""

    VALIDATE = "validate"
    BUMP = "bump"
    BUILD = "build"
    PACKAGE = "package"
    LAND = "land"
    PUSH = "push"


@dataclass(frozen=True, order=True, slots=True)
class PlanStage:
    """An immutable stage declaration without a WorkItem or executor."""

    stage_id: str
    stage: ReleaseStage
    project_id: str
    depends_on: tuple[str, ...] = ()
    requires_consent: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "stage_id", _bounded_text(self.stage_id, "stage ID"))
        if not isinstance(self.stage, ReleaseStage):
            raise WorkspaceReleaseError("stage must use a supported release stage")
        object.__setattr__(self, "project_id", canonical_repository_id(self.project_id))
        dependency_values = _bounded_sequence(
            self.depends_on, "stage dependencies", max_items=MAX_PLAN_STAGES
        )
        dependencies = tuple(
            _bounded_text(dependency, "stage dependency ID")
            for dependency in dependency_values
        )
        if len(dependencies) != len(set(dependencies)):
            raise WorkspaceReleaseError("stage dependencies must not be duplicated")
        object.__setattr__(self, "depends_on", tuple(sorted(dependencies)))
        object.__setattr__(
            self,
            "requires_consent",
            _bounded_bool(self.requires_consent, "requires_consent"),
        )
        if self.stage == ReleaseStage.PUSH and not self.requires_consent:
            raise WorkspaceReleaseError("push stages require explicit consent")


@dataclass(frozen=True, slots=True)
class WorkspaceReleasePlan:
    """Frozen C-11 plan payload, containing no executable command or mutation."""

    workspace_id: str
    source_sha: str
    selected_projects: tuple[str, ...]
    projects: tuple[ProjectRecord, ...]
    edges: tuple[DependencyEdge, ...] = ()
    floor_rewrites: tuple[FloorRewrite, ...] = ()
    stages: tuple[PlanStage, ...] = ()
    parallel_groups: tuple[tuple[str, ...], ...] = ()
    allow_push: bool = False
    plan_digest: str = ""
    contract_version: int = C11_CONTRACT_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "workspace_id", _bounded_text(self.workspace_id, "workspace ID")
        )
        object.__setattr__(
            self, "source_sha", _bounded_text(self.source_sha, "source SHA")
        )
        if _SHA.fullmatch(self.source_sha) is None:
            raise WorkspaceReleaseError("source SHA must be a full hexadecimal Git SHA")
        object.__setattr__(self, "source_sha", self.source_sha.lower())
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version != C11_CONTRACT_VERSION
        ):
            raise WorkspaceReleaseError("unsupported C-11 contract version")
        selected_values = _typed_sequence(
            self.selected_projects,
            "selected projects",
            str,
            max_items=MAX_PROJECTS,
        )
        project_values = _typed_sequence(
            self.projects,
            "release plan projects",
            ProjectRecord,
            max_items=MAX_PROJECTS,
        )
        if not selected_values or not project_values:
            raise WorkspaceReleaseError("release plan must select at least one project")
        object.__setattr__(
            self, "allow_push", _bounded_bool(self.allow_push, "allow_push")
        )
        project_map = {project.project_id: project for project in project_values}
        if len(project_map) != len(project_values):
            raise GraphValidationError(
                (
                    Diagnostic(
                        GraphDiagnosticCode.DUPLICATE_PROJECT,
                        "workspace-release-plan",
                        "projects must have unique canonical repository identities",
                    ),
                )
            )
        selected = tuple(
            sorted(canonical_repository_id(project) for project in selected_values)
        )
        if selected != tuple(sorted(project_map)):
            raise WorkspaceReleaseError(
                "selected_projects must exactly match the frozen project records"
            )
        object.__setattr__(self, "selected_projects", selected)
        object.__setattr__(
            self, "projects", tuple(project_map[key] for key in selected)
        )
        package_map: dict[str, PackageKey] = {}
        duplicate_packages: list[Diagnostic] = []
        for project in self.projects:
            for package in project.packages:
                package_id = package.key.value
                if package_id in package_map:
                    duplicate_packages.append(
                        Diagnostic(
                            GraphDiagnosticCode.DUPLICATE_PACKAGE,
                            package_id,
                            "frozen plan contains a duplicate package identity",
                        )
                    )
                else:
                    package_map[package_id] = package.key
        if duplicate_packages:
            raise GraphValidationError(duplicate_packages)
        edge_values = _typed_sequence(
            self.edges,
            "release plan edges",
            DependencyEdge,
            max_items=MAX_EDGES,
        )
        object.__setattr__(self, "edges", _normalize_plan_edges(edge_values))
        for edge in self.edges:
            if edge.dependent.value not in package_map:
                raise WorkspaceReleaseError(
                    "release plan edge dependent package is not frozen"
                )
            if edge.dependency.value not in package_map:
                raise WorkspaceReleaseError(
                    "release plan edge dependency package is not frozen"
                )
            if {
                edge.dependent_project_id,
                edge.dependency_project_id,
            } - set(selected):
                raise WorkspaceReleaseError(
                    "release plan edge names an unselected project"
                )
        rewrite_values = _typed_sequence(
            self.floor_rewrites,
            "release plan floor rewrites",
            FloorRewrite,
            max_items=MAX_EDGES,
        )
        object.__setattr__(
            self,
            "floor_rewrites",
            _normalize_plan_rewrites(rewrite_values),
        )
        for rewrite in self.floor_rewrites:
            if rewrite.dependent.value not in package_map:
                raise WorkspaceReleaseError(
                    "release plan floor rewrite dependent package is not frozen"
                )
            if rewrite.dependency.value not in package_map:
                raise WorkspaceReleaseError(
                    "release plan floor rewrite dependency package is not frozen"
                )
            if {
                rewrite.dependent.repository_id,
                rewrite.dependency.repository_id,
            } - set(selected):
                raise WorkspaceReleaseError(
                    "release plan floor rewrite names an unselected project"
                )
        stage_values = _typed_sequence(
            self.stages,
            "release plan stages",
            PlanStage,
            max_items=MAX_PLAN_STAGES,
        )
        object.__setattr__(
            self,
            "stages",
            tuple(sorted(stage_values, key=lambda stage: stage.stage_id)),
        )
        if len(self.stages) > MAX_PLAN_STAGES:
            raise WorkspaceReleaseError("release plan stage count exceeds the bound")
        stage_ids = [stage.stage_id for stage in self.stages]
        if len(stage_ids) != len(set(stage_ids)):
            raise WorkspaceReleaseError("release plan stage IDs must be unique")
        stage_id_set = set(stage_ids)
        for stage in self.stages:
            if stage.project_id not in selected:
                raise WorkspaceReleaseError(
                    "release plan stage names an unselected project"
                )
            if stage.stage == ReleaseStage.PUSH and not self.allow_push:
                raise WorkspaceReleaseError("push stage requires plan push consent")
            if stage.stage_id in stage.depends_on:
                raise WorkspaceReleaseError(
                    "release plan stage cannot depend on itself"
                )
            if set(stage.depends_on) - stage_id_set:
                raise WorkspaceReleaseError(
                    "release plan stage depends on an unknown stage"
                )
        _validate_stage_dag(self.stages)
        group_values = _bounded_sequence(
            self.parallel_groups,
            "release plan parallel groups",
            max_items=MAX_PROJECTS,
        )
        if not group_values:
            raise WorkspaceReleaseError(
                "release plan parallel groups must be complete and non-empty"
            )
        normalized_groups: list[tuple[str, ...]] = []
        for group in group_values:
            members = _typed_sequence(
                group,
                "release plan parallel group",
                str,
                max_items=MAX_PROJECTS,
            )
            normalized = tuple(canonical_repository_id(project) for project in members)
            if not normalized:
                raise WorkspaceReleaseError(
                    "release plan parallel groups must not contain empty groups"
                )
            if len(normalized) != len(set(normalized)):
                raise WorkspaceReleaseError(
                    "release plan parallel groups must not duplicate projects"
                )
            if normalized != tuple(sorted(normalized)):
                raise WorkspaceReleaseError(
                    "release plan parallel group members must be canonical and ordered"
                )
            normalized_groups.append(normalized)
        normalized_parallel_groups = tuple(normalized_groups)
        object.__setattr__(self, "parallel_groups", normalized_parallel_groups)
        grouped_projects = [
            project for group in self.parallel_groups for project in group
        ]
        if any(project not in selected for project in grouped_projects):
            raise WorkspaceReleaseError("parallel groups name an unselected project")
        if len(grouped_projects) != len(set(grouped_projects)):
            raise WorkspaceReleaseError("parallel groups must not repeat a project")
        project_edges = tuple(
            sorted(
                {
                    (edge.dependent_project_id, edge.dependency_project_id)
                    for edge in self.edges
                    if edge.dependent_project_id != edge.dependency_project_id
                }
            )
        )
        expected_groups, cycle_diagnostics = _topological_groups(
            selected, project_edges
        )
        if cycle_diagnostics:
            raise GraphValidationError(cycle_diagnostics)
        if normalized_parallel_groups != expected_groups:
            raise WorkspaceReleaseError(
                "release plan parallel groups must match deterministic dependency order"
            )
        if not isinstance(self.plan_digest, str):
            raise WorkspaceReleaseError("plan digest must be a string")
        if self.plan_digest:
            digest = _bounded_text(self.plan_digest, "plan digest", max_length=64)
            if re.fullmatch(r"[0-9a-fA-F]{64}", digest) is None:
                raise WorkspaceReleaseError("plan digest must be a SHA-256 digest")
            object.__setattr__(self, "plan_digest", digest.lower())
            expected = plan_digest(self)
            if self.plan_digest != expected:
                raise WorkspaceReleaseError(
                    "plan digest does not match frozen contents"
                )
        else:
            object.__setattr__(self, "plan_digest", plan_digest(self))

    def canonical_payload(self, *, include_digest: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "contract_version": self.contract_version,
            "workspace_id": self.workspace_id,
            "source_sha": self.source_sha.lower(),
            "selected_projects": self.selected_projects,
            "projects": [_project_payload(project) for project in self.projects],
            "edges": [_edge_payload(edge) for edge in self.edges],
            "floor_rewrites": [
                _floor_rewrite_payload(item) for item in self.floor_rewrites
            ],
            "stages": [_stage_payload(stage) for stage in self.stages],
            "parallel_groups": self.parallel_groups,
            "allow_push": self.allow_push,
        }
        if include_digest:
            payload["plan_digest"] = self.plan_digest
        return payload


@dataclass(frozen=True, slots=True)
class DependencyGraph:
    """Deterministic package and project topology produced by the pure builder."""

    projects: tuple[ProjectRecord, ...]
    packages: tuple[PackageRecord, ...]
    edges: tuple[DependencyEdge, ...]
    project_edges: tuple[tuple[str, str], ...]
    parallel_groups: tuple[tuple[str, ...], ...]
    digest: str

    def canonical_payload(self) -> dict[str, object]:
        return {
            "projects": [_project_payload(project) for project in self.projects],
            "packages": [_package_payload(package) for package in self.packages],
            "edges": [_edge_payload(edge) for edge in self.edges],
            "project_edges": self.project_edges,
            "parallel_groups": self.parallel_groups,
        }


@dataclass(frozen=True, slots=True)
class LegacyPhase:
    """Read-only compatibility projection of one existing phase declaration."""

    name: str
    phase: int
    project_references: tuple[str, ...] = ()
    bulk_bump: bool = False
    bulk_push: bool = False
    wait_minutes: int = 0


@dataclass(frozen=True, slots=True)
class LegacyPhaseManifest:
    """Immutable view of the current phase manifest during shadow operation."""

    phases: tuple[LegacyPhase, ...]


def _package_payload(package: PackageRecord) -> dict[str, object]:
    return {
        "key": package.key.value,
        "version": package.version.value,
        "version_sources": [
            {"location": source.location, "version": source.version.value}
            for source in package.version_sources
        ],
        "dependencies": [
            {
                "ecosystem": dependency.target.ecosystem.value,
                "name": dependency.target.name,
                "repository_id": dependency.target.repository_id,
                "floor": dependency.floor.value if dependency.floor else None,
                "source": dependency.source,
            }
            for dependency in package.dependencies
        ],
        "metadata_files": package.metadata_files,
    }


def _project_payload(project: ProjectRecord) -> dict[str, object]:
    return {
        "repository_id": project.repository_id,
        "tree_sha": project.tree_sha.lower(),
        "packages": [_package_payload(package) for package in project.packages],
        "metadata_files": project.metadata_files,
    }


def _edge_payload(edge: DependencyEdge) -> dict[str, object]:
    return {
        "dependent": edge.dependent.value,
        "dependency": edge.dependency.value,
        "floor": edge.floor.value if edge.floor else None,
        "source": edge.source,
        "confidence": edge.confidence.value,
    }


def _floor_rewrite_payload(item: FloorRewrite) -> dict[str, object]:
    return {
        "dependent": item.dependent.value,
        "dependency": item.dependency.value,
        "old_floor": item.old_floor.value,
        "new_floor": item.new_floor.value,
        "source": item.source,
    }


def _stage_payload(stage: PlanStage) -> dict[str, object]:
    return {
        "stage_id": stage.stage_id,
        "stage": stage.stage.value,
        "project_id": stage.project_id,
        "depends_on": stage.depends_on,
        "requires_consent": stage.requires_consent,
    }


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def plan_digest(plan: WorkspaceReleasePlan) -> str:
    """Hash a frozen plan without recursively hashing its digest field."""

    return hashlib.sha256(
        _canonical_json(plan.canonical_payload()).encode()
    ).hexdigest()


def _edge_sort_key(edge: DependencyEdge) -> tuple[str, str, str, str, str]:
    return (
        edge.dependent.value,
        edge.dependency.value,
        edge.floor.value if edge.floor else "",
        edge.source,
        edge.confidence.value,
    )


def _normalize_plan_edges(
    values: Iterable[DependencyEdge],
) -> tuple[DependencyEdge, ...]:
    """Sort plan edges canonically and refuse duplicate endpoint declarations."""

    ordered = tuple(sorted(values, key=_edge_sort_key))
    grouped: dict[tuple[str, str], list[DependencyEdge]] = {}
    for edge in ordered:
        grouped.setdefault((edge.dependent.value, edge.dependency.value), []).append(
            edge
        )
    diagnostics: list[Diagnostic] = []
    for endpoint in sorted(grouped):
        candidates = grouped[endpoint]
        if len(candidates) <= 1:
            continue
        details = tuple(
            (
                f"edge_{index}",
                f"{edge.floor.value if edge.floor else ''}|{edge.source}|{edge.confidence.value}",
            )
            for index, edge in enumerate(candidates)
        )
        code = (
            GraphDiagnosticCode.CONFLICTING_FLOOR
            if len({edge.floor.value if edge.floor else "" for edge in candidates}) > 1
            else GraphDiagnosticCode.DUPLICATE_EDGE
        )
        diagnostics.append(
            Diagnostic(
                code,
                f"{endpoint[0]}->{endpoint[1]}",
                "frozen plan declares duplicate dependency endpoints",
                details,
            )
        )
    if diagnostics:
        raise GraphValidationError(diagnostics)
    return ordered


def _normalize_plan_rewrites(
    values: Iterable[FloorRewrite],
) -> tuple[FloorRewrite, ...]:
    """Sort floor rewrites canonically and refuse endpoint collisions."""

    ordered = tuple(
        sorted(
            values,
            key=lambda item: (
                item.dependent.value,
                item.dependency.value,
                item.old_floor.value,
                item.new_floor.value,
                item.source,
            ),
        )
    )
    grouped: dict[tuple[str, str], list[FloorRewrite]] = {}
    for rewrite in ordered:
        grouped.setdefault(
            (rewrite.dependent.value, rewrite.dependency.value), []
        ).append(rewrite)
    diagnostics: list[Diagnostic] = []
    for endpoint in sorted(grouped):
        candidates = grouped[endpoint]
        if len(candidates) <= 1:
            continue
        details = tuple(
            (
                f"rewrite_{index}",
                f"{item.old_floor.value}->{item.new_floor.value}|{item.source}",
            )
            for index, item in enumerate(candidates)
        )
        old_new = {(item.old_floor.value, item.new_floor.value) for item in candidates}
        code = (
            GraphDiagnosticCode.DUPLICATE_REWRITE
            if len(old_new) == 1
            else GraphDiagnosticCode.CONFLICTING_REWRITE
        )
        diagnostics.append(
            Diagnostic(
                code,
                f"{endpoint[0]}->{endpoint[1]}",
                "frozen plan declares duplicate floor rewrite endpoints",
                details,
            )
        )
    if diagnostics:
        raise GraphValidationError(diagnostics)
    return ordered


def _validate_stage_dag(stages: tuple[PlanStage, ...]) -> None:
    """Refuse cycles in the immutable stage dependency declarations."""

    dependencies = {stage.stage_id: set(stage.depends_on) for stage in stages}
    remaining = set(dependencies)
    while remaining:
        ready = tuple(
            sorted(stage_id for stage_id in remaining if not dependencies[stage_id])
        )
        if not ready:
            cycle = _cycle_path(remaining, dependencies)
            raise GraphValidationError(
                (
                    Diagnostic(
                        GraphDiagnosticCode.CYCLE,
                        "release plan stages",
                        "release stage declarations contain a cycle",
                        (("path", " -> ".join(cycle)),),
                    ),
                )
            )
        for stage_id in ready:
            remaining.remove(stage_id)
            for dependents in dependencies.values():
                dependents.discard(stage_id)


def _sorted_diagnostics(diagnostics: Iterable[Diagnostic]) -> tuple[Diagnostic, ...]:
    return tuple(
        sorted(
            diagnostics,
            key=lambda item: (
                item.code.value,
                item.subject,
                item.message,
                item.details,
            ),
        )
    )


def build_dependency_graph(
    projects: Iterable[ProjectRecord],
    *,
    overlay_edges: Iterable[DependencyEdge] = (),
) -> DependencyGraph:
    """Resolve package owners and return a stable project/package DAG.

    All input collections are copied and sorted.  Diagnostics are accumulated
    and raised together, so callers can show every missing or ambiguous edge in
    one refusal before any later mutation path is considered.
    """

    project_values = tuple(projects)
    if any(not isinstance(project, ProjectRecord) for project in project_values):
        raise WorkspaceReleaseError("workspace projects must be ProjectRecord values")
    project_values = tuple(
        sorted(
            project_values,
            key=lambda project: (
                project.project_id,
                project.tree_sha,
                tuple(
                    (
                        package.key.value,
                        package.version.value,
                        tuple(
                            (source.location, source.version.value)
                            for source in package.version_sources
                        ),
                    )
                    for package in project.packages
                ),
            ),
        )
    )
    if len(project_values) > MAX_PROJECTS:
        raise WorkspaceReleaseError("workspace project count exceeds the bound")
    overlay_values = tuple(overlay_edges)
    if any(not isinstance(edge, DependencyEdge) for edge in overlay_values):
        raise WorkspaceReleaseError(
            "workspace overlay edges must be DependencyEdge values"
        )
    overlay_values = tuple(sorted(overlay_values, key=_edge_sort_key))
    if len(overlay_values) > MAX_EDGES:
        raise WorkspaceReleaseError("workspace overlay edge count exceeds the bound")
    diagnostics: list[Diagnostic] = []
    project_map: dict[str, ProjectRecord] = {}
    for project in project_values:
        if project.project_id in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.DUPLICATE_PROJECT,
                    project.project_id,
                    "canonical repository identity appears more than once",
                )
            )
        else:
            project_map[project.project_id] = project

    package_map: dict[str, PackageRecord] = {}
    owners: dict[tuple[Ecosystem, str], list[PackageKey]] = {}
    for project in sorted(project_map.values(), key=lambda item: item.project_id):
        for package in project.packages:
            key = package.key.value
            if key in package_map:
                diagnostics.append(
                    Diagnostic(
                        GraphDiagnosticCode.DUPLICATE_PACKAGE,
                        key,
                        "package identity appears more than once",
                    )
                )
                continue
            package_map[key] = package
            owners.setdefault((package.key.ecosystem, package.key.name), []).append(
                package.key
            )
    if len(package_map) > MAX_PACKAGES:
        raise WorkspaceReleaseError("workspace package count exceeds the bound")
    for owner_key in owners:
        owners[owner_key].sort(key=lambda item: item.value)

    edges: list[DependencyEdge] = []
    seen_edges: dict[tuple[str, str], DependencyEdge] = {}
    overlay_by_reference: dict[tuple[str, Ecosystem, str], list[DependencyEdge]] = {}
    for edge in overlay_values:
        overlay_by_reference.setdefault(
            (
                edge.dependent.value,
                edge.dependency.ecosystem,
                edge.dependency.name,
            ),
            [],
        ).append(edge)
    consumed_overlay_edges: set[DependencyEdge] = set()
    for package in sorted(package_map.values(), key=lambda item: item.key.value):
        for spec in package.dependencies:
            target = spec.target
            explicit_overlays = tuple(
                overlay_by_reference.get(
                    (package.key.value, target.ecosystem, target.name), []
                )
            )
            if explicit_overlays:
                if target.repository_id is None:
                    overlay_owners = tuple(
                        sorted({edge.dependency.value for edge in explicit_overlays})
                    )
                    if len(overlay_owners) > 1:
                        diagnostics.append(
                            Diagnostic(
                                GraphDiagnosticCode.AMBIGUOUS_PACKAGE_OWNER,
                                f"{package.key.value}->{target.ecosystem.value}:{target.name}",
                                "overlay dependency has more than one possible owner",
                                tuple(
                                    (f"owner_{index}", owner)
                                    for index, owner in enumerate(overlay_owners)
                                ),
                            )
                        )
                        consumed_overlay_edges.update(explicit_overlays)
                        continue
                if target.repository_id is not None:
                    mismatched = tuple(
                        edge
                        for edge in explicit_overlays
                        if edge.dependency.repository_id != target.repository_id
                    )
                    if mismatched:
                        diagnostics.append(
                            Diagnostic(
                                GraphDiagnosticCode.INVALID_METADATA,
                                f"{package.key.value}->{target.ecosystem.value}:{target.name}",
                                "overlay edge owner conflicts with explicit metadata owner",
                            )
                        )
                        continue
                for edge in explicit_overlays:
                    _append_edge(edge, edges, seen_edges, diagnostics)
                    consumed_overlay_edges.add(edge)
                continue
            candidates = owners.get((target.ecosystem, target.name), [])
            if target.repository_id is not None:
                candidate_id = PackageKey(
                    target.repository_id, target.ecosystem, target.name
                ).value
                candidate_record = package_map.get(candidate_id)
                candidates = [candidate_record.key] if candidate_record else []
                if target.repository_id not in project_map:
                    diagnostics.append(
                        Diagnostic(
                            GraphDiagnosticCode.MISSING_PROJECT,
                            target.repository_id,
                            "explicit dependency edge names an unknown project",
                            (("dependent", package.key.repository_id),),
                        )
                    )
            if not candidates:
                diagnostics.append(
                    Diagnostic(
                        GraphDiagnosticCode.MISSING_PACKAGE,
                        f"{package.key.value}->{target.ecosystem.value}:{target.name}",
                        "dependency package has no known owner",
                    )
                )
                continue
            if len(candidates) > 1:
                diagnostics.append(
                    Diagnostic(
                        GraphDiagnosticCode.AMBIGUOUS_PACKAGE_OWNER,
                        f"{package.key.value}->{target.ecosystem.value}:{target.name}",
                        "dependency package has more than one possible owner; use an explicit repository identity",
                        tuple(
                            (f"owner_{index}", item.value)
                            for index, item in enumerate(candidates)
                        ),
                    )
                )
                continue
            edge = DependencyEdge(
                dependent=package.key,
                dependency=candidates[0],
                floor=spec.floor,
                source=spec.source,
                confidence=(
                    EdgeConfidence.EXPLICIT
                    if target.repository_id is not None
                    else EdgeConfidence.INFERRED
                ),
            )
            _append_edge(edge, edges, seen_edges, diagnostics)

    for edge in overlay_values:
        if edge not in consumed_overlay_edges:
            _append_edge(edge, edges, seen_edges, diagnostics)
        if edge.dependent.repository_id not in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PROJECT,
                    edge.dependent.repository_id,
                    "overlay edge dependent project is not selected",
                )
            )
        if edge.dependency.repository_id not in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PROJECT,
                    edge.dependency.repository_id,
                    "overlay edge dependency project is not selected",
                )
            )
        if edge.dependent.value not in package_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PACKAGE,
                    edge.dependent.value,
                    "overlay edge dependent package is not declared",
                )
            )
        if edge.dependency.value not in package_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PACKAGE,
                    edge.dependency.value,
                    "overlay edge dependency package is not declared",
                )
            )

    if len(edges) > MAX_EDGES:
        raise WorkspaceReleaseError("workspace dependency edge count exceeds the bound")
    project_edges = tuple(
        sorted(
            {
                (edge.dependent_project_id, edge.dependency_project_id)
                for edge in edges
                if edge.dependent_project_id != edge.dependency_project_id
            }
        )
    )
    groups, cycle_diagnostics = _topological_groups(
        tuple(sorted(project_map)), project_edges
    )
    diagnostics.extend(cycle_diagnostics)
    if diagnostics:
        raise GraphValidationError(_sorted_diagnostics(diagnostics))

    ordered_projects = tuple(project_map[key] for key in sorted(project_map))
    ordered_packages = tuple(package_map[key] for key in sorted(package_map))
    ordered_edges = tuple(sorted(edges, key=_edge_sort_key))
    payload = {
        "projects": [_project_payload(project) for project in ordered_projects],
        "packages": [_package_payload(package) for package in ordered_packages],
        "edges": [_edge_payload(edge) for edge in ordered_edges],
        "project_edges": project_edges,
        "parallel_groups": groups,
    }
    digest = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
    return DependencyGraph(
        projects=ordered_projects,
        packages=ordered_packages,
        edges=ordered_edges,
        project_edges=project_edges,
        parallel_groups=groups,
        digest=digest,
    )


def _append_edge(
    edge: DependencyEdge,
    edges: list[DependencyEdge],
    seen: dict[tuple[str, str], DependencyEdge],
    diagnostics: list[Diagnostic],
) -> None:
    key = (edge.dependent.value, edge.dependency.value)
    previous = seen.get(key)
    if previous is None:
        seen[key] = edge
        edges.append(edge)
        return
    if previous.floor != edge.floor:
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.CONFLICTING_FLOOR,
                f"{edge.dependent.value}->{edge.dependency.value}",
                "metadata and overlay edges declare different floors",
                (
                    ("first", previous.floor.value if previous.floor else ""),
                    ("second", edge.floor.value if edge.floor else ""),
                ),
            )
        )
    else:
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.DUPLICATE_EDGE,
                f"{edge.dependent.value}->{edge.dependency.value}",
                "dependency edge was declared more than once",
            )
        )


def _topological_groups(
    projects: tuple[str, ...], project_edges: tuple[tuple[str, str], ...]
) -> tuple[tuple[tuple[str, ...], ...], tuple[Diagnostic, ...]]:
    dependencies: dict[str, set[str]] = {project: set() for project in projects}
    dependents: dict[str, set[str]] = {project: set() for project in projects}
    for dependent, dependency in project_edges:
        if dependent == dependency:
            continue
        if dependent not in dependencies or dependency not in dependencies:
            continue
        dependencies[dependent].add(dependency)
        dependents[dependency].add(dependent)
    remaining = set(projects)
    groups: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(project for project in remaining if not dependencies[project])
        )
        if not ready:
            cycle = _cycle_path(remaining, dependencies)
            return tuple(groups), (
                Diagnostic(
                    GraphDiagnosticCode.CYCLE,
                    "workspace dependency graph",
                    "dependency graph contains a cycle",
                    (("path", " -> ".join(cycle)),),
                ),
            )
        groups.append(ready)
        for project in ready:
            remaining.remove(project)
            for dependent in dependents[project]:
                dependencies[dependent].discard(project)
    return tuple(groups), ()


def _cycle_path(
    remaining: set[str], dependencies: dict[str, set[str]]
) -> tuple[str, ...]:
    """Return one stable cycle path for a deterministic diagnostic."""

    visiting: set[str] = set()
    stack: list[str] = []

    def visit(node: str) -> tuple[str, ...] | None:
        if node in visiting:
            index = stack.index(node)
            return tuple((*stack[index:], node))
        visiting.add(node)
        stack.append(node)
        for dependency in sorted(dependencies[node]):
            if dependency not in remaining:
                continue
            result = visit(dependency)
            if result:
                return result
        stack.pop()
        visiting.remove(node)
        return None

    for node in sorted(remaining):
        result = visit(node)
        if result:
            return result
    return tuple(sorted(remaining))


def phase_manifest_from_mapping(value: Mapping[str, object]) -> LegacyPhaseManifest:
    """Copy the current phase manifest into an immutable compatibility view.

    This function accepts the historical ``maintenance`` mapping only as
    read-only input.  It intentionally preserves bare project references for
    the later shadow comparator; those strings are never used as graph keys.
    """

    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise WorkspaceReleaseError("phase manifest keys must be strings")
    allowed = {"description", "phases"}
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise WorkspaceReleaseError(
            f"phase manifest has unsupported fields: {', '.join(unknown)}"
        )
    phases_value = value.get("phases", ())
    if not isinstance(phases_value, (list, tuple)):
        raise WorkspaceReleaseError("phase manifest phases must be a sequence")
    if len(phases_value) > MAX_PLAN_STAGES:
        raise WorkspaceReleaseError("phase manifest exceeds the stage bound")
    phases: list[LegacyPhase] = []
    total_project_references = 0
    phase_keys = {
        "name",
        "phase",
        "project",
        "projects",
        "bulk_bump",
        "bulk_push",
        "wait_minutes",
        "updates",
        "exclude",
    }
    for index, raw in enumerate(phases_value):
        if not isinstance(raw, Mapping):
            raise WorkspaceReleaseError(f"phase {index} must be a mapping")
        unsupported = sorted(set(raw) - phase_keys)
        if unsupported:
            raise WorkspaceReleaseError(
                f"phase {index} has unsupported fields: {', '.join(unsupported)}"
            )
        name = _bounded_text(raw.get("name", f"phase-{index + 1}"), "phase name")
        phase = raw.get("phase", index + 1)
        if isinstance(phase, bool) or not isinstance(phase, int) or phase < 1:
            raise WorkspaceReleaseError(f"phase {index} number must be positive")
        project = raw.get("project")
        projects = raw.get("projects", ())
        refs: list[str] = []
        if project is not None:
            refs.append(_bounded_text(project, f"phase {index} project"))
        if not isinstance(projects, (list, tuple)):
            raise WorkspaceReleaseError(f"phase {index} projects must be a sequence")
        refs.extend(_bounded_text(item, f"phase {index} project") for item in projects)
        if len(refs) > MAX_PROJECTS:
            raise WorkspaceReleaseError(
                f"phase {index} projects exceed the project bound"
            )
        total_project_references += len(refs)
        if total_project_references > MAX_EDGES:
            raise WorkspaceReleaseError(
                "phase manifest project references exceed the bounded total"
            )
        if len(refs) != len(set(refs)):
            raise WorkspaceReleaseError(
                f"phase {index} project references must be unique"
            )
        wait = raw.get("wait_minutes", 0)
        if isinstance(wait, bool) or not isinstance(wait, int) or wait < 0:
            raise WorkspaceReleaseError(
                f"phase {index} wait_minutes must be non-negative"
            )
        bulk_bump = raw.get("bulk_bump", False)
        bulk_push = raw.get("bulk_push", False)
        if not isinstance(bulk_bump, bool) or not isinstance(bulk_push, bool):
            raise WorkspaceReleaseError(f"phase {index} bulk flags must be booleans")
        phases.append(
            LegacyPhase(
                name=name,
                phase=phase,
                project_references=tuple(refs),
                bulk_bump=bulk_bump,
                bulk_push=bulk_push,
                wait_minutes=wait,
            )
        )
    return LegacyPhaseManifest(
        phases=tuple(sorted(phases, key=lambda item: (item.phase, item.name)))
    )


# Compact aliases make the pure seam discoverable without exposing an executor.
build_graph = build_dependency_graph
read_phase_manifest = phase_manifest_from_mapping


__all__ = [
    "C11_CONTRACT_VERSION",
    "DependencyEdge",
    "DependencyGraph",
    "DependencySpec",
    "Diagnostic",
    "EdgeConfidence",
    "Ecosystem",
    "FloorRewrite",
    "GraphDiagnosticCode",
    "GraphValidationError",
    "LegacyPhase",
    "LegacyPhaseManifest",
    "PackageKey",
    "PackageRecord",
    "PackageReference",
    "PlanStage",
    "ProjectRecord",
    "ReleaseStage",
    "Version",
    "VersionFloor",
    "VersionSource",
    "WorkspaceReleaseError",
    "WorkspaceReleasePlan",
    "build_dependency_graph",
    "build_graph",
    "canonical_repository_id",
    "phase_manifest_from_mapping",
    "plan_digest",
    "read_phase_manifest",
    "resolve_version_sources",
]
