"""Bounded, non-executing metadata readers for the C-11 workspace graph.

Only declarative files are inspected: ``pyproject.toml``, ``Cargo.toml``, and
``package.json``.  Readers use the standard-library TOML/JSON parsers, reject
symlinked metadata files, enforce byte/shape/item limits, and never invoke a
project tool, subprocess, network client, package manager, or build backend.

The explicit overlay is an in-memory, versioned input for dependency edges that
cannot be inferred safely from package metadata (for example a local path
dependency).  It is copied into immutable records and rejects unknown fields;
it is not a replacement workspace manifest and has no write path.
"""

from __future__ import annotations

import json
import math
import os
import re
import stat
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from .workspace_release import (
    DependencyEdge,
    DependencySpec,
    Diagnostic,
    Ecosystem,
    GraphDiagnosticCode,
    GraphValidationError,
    PackageKey,
    PackageRecord,
    PackageReference,
    ProjectRecord,
    Version,
    VersionFloor,
    VersionSource,
    WorkspaceReleaseError,
    build_dependency_graph,
    canonical_repository_id,
    resolve_version_sources,
)

DEFAULT_MAX_METADATA_BYTES = 1 << 20
DEFAULT_MAX_METADATA_DEPTH = 24
DEFAULT_MAX_METADATA_ITEMS = 4_096
DEFAULT_MAX_METADATA_STRING = 4_096
DEFAULT_MAX_DEPENDENCIES = 1_024
DEFAULT_MAX_PROJECT_PACKAGES = 256

_PACKAGE_ID = re.compile(
    r"^(?P<repository>repo:[^:]+(?:/[^:]+)*)::(?P<ecosystem>python|rust|node):(?P<name>.+)$"
)
_PARTIAL_FLOOR = re.compile(
    r"^(?P<operator>\^|~=|~|>=|>|==)?(?P<numbers>[0-9]+(?:\.[0-9]+){0,2})$"
)
_REQUIREMENT_NAME = re.compile(r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?P<rest>.*)$")
_NODE_NAME = re.compile(r"^(?:@[A-Za-z0-9._~-]+/)?[A-Za-z0-9._~-]+$")


class MetadataError(WorkspaceReleaseError):
    """Metadata was malformed, unsupported, oversized, or unsafe to inspect."""


@dataclass(frozen=True, slots=True)
class MetadataLimits:
    """Resource bounds applied before and after parsing one metadata document."""

    max_bytes: int = DEFAULT_MAX_METADATA_BYTES
    max_depth: int = DEFAULT_MAX_METADATA_DEPTH
    max_items: int = DEFAULT_MAX_METADATA_ITEMS
    max_string_length: int = DEFAULT_MAX_METADATA_STRING
    max_dependencies: int = DEFAULT_MAX_DEPENDENCIES
    max_packages: int = DEFAULT_MAX_PROJECT_PACKAGES

    def __post_init__(self) -> None:
        for field_name in (
            "max_bytes",
            "max_depth",
            "max_items",
            "max_string_length",
            "max_dependencies",
            "max_packages",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise MetadataError(f"{field_name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class OverlayInput:
    """Explicit edge/version input with a strict, bounded schema."""

    schema_version: int
    edges: tuple[DependencyEdge, ...] = ()
    versions: tuple[tuple[PackageKey, VersionSource], ...] = ()

    def __post_init__(self) -> None:
        if (
            isinstance(self.schema_version, bool)
            or not isinstance(self.schema_version, int)
            or self.schema_version != 1
        ):
            raise MetadataError("overlay schema_version must be 1")
        if isinstance(self.edges, (str, bytes, bytearray)) or not isinstance(
            self.edges, Sequence
        ):
            raise MetadataError("overlay edges must be a sequence")
        edges = tuple(self.edges)
        if len(edges) > DEFAULT_MAX_METADATA_ITEMS:
            raise MetadataError("overlay edge count exceeds the bound")
        if any(not isinstance(edge, DependencyEdge) for edge in edges):
            raise MetadataError("overlay edges must contain DependencyEdge values")
        if isinstance(self.versions, (str, bytes, bytearray)) or not isinstance(
            self.versions, Sequence
        ):
            raise MetadataError("overlay versions must be a sequence")
        versions = tuple(self.versions)
        if len(versions) > DEFAULT_MAX_METADATA_ITEMS:
            raise MetadataError("overlay version count exceeds the bound")
        normalized_versions: list[tuple[PackageKey, VersionSource]] = []
        for item in versions:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise MetadataError(
                    "overlay versions must contain package/source pairs"
                )
            package, source = item
            if not isinstance(package, PackageKey) or not isinstance(
                source, VersionSource
            ):
                raise MetadataError(
                    "overlay versions must contain PackageKey/VersionSource pairs"
                )
            normalized_versions.append((package, source))
        object.__setattr__(
            self,
            "edges",
            tuple(
                sorted(
                    edges,
                    key=lambda edge: (
                        edge.value,
                        edge.source,
                        edge.confidence.value,
                    ),
                )
            ),
        )
        object.__setattr__(
            self,
            "versions",
            tuple(
                sorted(
                    normalized_versions,
                    key=lambda item: (
                        item[0].value,
                        item[1].location,
                        item[1].version.value,
                    ),
                )
            ),
        )

    @classmethod
    def from_mapping(
        cls, value: Mapping[str, object], *, limits: MetadataLimits | None = None
    ) -> OverlayInput:
        """Validate and copy the explicit overlay without retaining the mapping."""

        active = limits or MetadataLimits()
        _bound_value(value, active)
        _strict_mapping(value, {"schema_version", "edges", "versions"}, "overlay")
        schema_version = value.get("schema_version")
        if isinstance(schema_version, bool) or schema_version != 1:
            raise MetadataError("overlay schema_version must be 1")
        edges_value = value.get("edges", ())
        versions_value = value.get("versions", ())
        edges = tuple(
            _overlay_edge(raw, index, active)
            for index, raw in enumerate(_sequence(edges_value, "overlay.edges"))
        )
        versions = tuple(
            _overlay_version(raw, index, active)
            for index, raw in enumerate(_sequence(versions_value, "overlay.versions"))
        )
        return cls(schema_version=1, edges=edges, versions=versions)


@dataclass(frozen=True, slots=True)
class MetadataInventory:
    """Read-only inventory consumed by the pure graph builder."""

    projects: tuple[ProjectRecord, ...]
    overlay: OverlayInput = OverlayInput(schema_version=1)

    def graph(self):
        """Build the deterministic graph, applying only explicit overlay edges."""

        projects = _apply_overlay_versions(self.projects, self.overlay)
        return build_dependency_graph(projects, overlay_edges=self.overlay.edges)


def _strict_mapping(
    value: object, allowed: set[str], label: str
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MetadataError(f"{label} must be a mapping")
    keys = tuple(value)
    if any(not isinstance(key, str) for key in keys):
        raise MetadataError(f"{label} keys must be strings")
    unsupported = sorted(set(keys) - allowed)
    if unsupported:
        raise MetadataError(f"{label} has unsupported fields: {', '.join(unsupported)}")
    return value


def _sequence(value: object, label: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise MetadataError(f"{label} must be a sequence")
    return tuple(value)


def _string(value: object, label: str, limits: MetadataLimits) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise MetadataError(f"{label} must be a non-blank trimmed string")
    if len(value) > limits.max_string_length:
        raise MetadataError(f"{label} exceeds the bounded length")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise MetadataError(f"{label} contains a control character")
    return value


def _bound_value(value: object, limits: MetadataLimits, *, depth: int = 0) -> None:
    """Reject deeply nested, oversized, or non-string-key metadata structures."""

    if depth > limits.max_depth:
        raise MetadataError("metadata nesting exceeds the bound")
    if isinstance(value, Mapping):
        if len(value) > limits.max_items:
            raise MetadataError("metadata mapping exceeds the item bound")
        for key, child in value.items():
            _string(key, "metadata field", limits)
            _bound_value(child, limits, depth=depth + 1)
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        if len(value) > limits.max_items:
            raise MetadataError("metadata sequence exceeds the item bound")
        for child in value:
            _bound_value(child, limits, depth=depth + 1)
        return
    if isinstance(value, str):
        _string(value, "metadata value", limits)
        return
    if isinstance(value, float) and not math.isfinite(value):
        raise MetadataError("metadata numeric value must be finite")


def _read_bounded_file(path: Path, limits: MetadataLimits) -> bytes:
    """Read one regular, non-symlink file at most once and within a byte bound."""

    candidate = Path(path)
    if candidate.is_symlink():
        raise MetadataError(f"metadata file must not be a symlink: {candidate.name}")
    try:
        descriptor = os.open(candidate, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except OSError as exc:
        raise MetadataError(
            f"metadata file could not be opened: {candidate.name}"
        ) from exc
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise MetadataError(
                f"metadata path is not a regular file: {candidate.name}"
            )
        if file_stat.st_size > limits.max_bytes:
            raise MetadataError(
                f"metadata file exceeds the byte bound: {candidate.name}"
            )
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = -1
            content = handle.read(limits.max_bytes + 1)
        if len(content) > limits.max_bytes:
            raise MetadataError(
                f"metadata file exceeds the byte bound: {candidate.name}"
            )
        return content
    except OSError as exc:
        raise MetadataError(
            f"metadata file could not be read: {candidate.name}"
        ) from exc
    finally:
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _project_file(project_root: Path, filename: str) -> Path:
    root = Path(project_root)
    if root.is_symlink():
        raise MetadataError("project root must not be a symlink")
    if not root.is_absolute():
        root = root.resolve(strict=False)
    candidate = root / filename
    try:
        candidate.resolve(strict=False).relative_to(root.resolve(strict=False))
    except ValueError as exc:
        raise MetadataError("metadata path escaped the project root") from exc
    return candidate


def _load_toml(path: Path, limits: MetadataLimits) -> Mapping[str, object]:
    content = _read_bounded_file(path, limits)
    try:
        document = tomllib.loads(content.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise MetadataError(f"invalid TOML metadata: {path.name}") from exc
    _bound_value(document, limits)
    if not isinstance(document, Mapping):
        raise MetadataError(f"TOML metadata root must be a mapping: {path.name}")
    return document


def _load_json(path: Path, limits: MetadataLimits) -> Mapping[str, object]:
    content = _read_bounded_file(path, limits)

    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, item in items:
            if key in result:
                raise MetadataError(f"duplicate JSON field: {key}")
            result[key] = item
        return result

    try:
        document = json.loads(content.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise MetadataError(f"invalid JSON metadata: {path.name}") from exc
    _bound_value(document, limits)
    if not isinstance(document, Mapping):
        raise MetadataError(f"JSON metadata root must be a mapping: {path.name}")
    return document


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise MetadataError(f"{label} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise MetadataError(f"{label} keys must be strings")
    return value


def _require_string(value: object, label: str, limits: MetadataLimits) -> str:
    return _string(value, label, limits)


def _canonical_project_id(value: str) -> str:
    try:
        return canonical_repository_id(value)
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _metadata_version(value: object, label: str, limits: MetadataLimits) -> Version:
    try:
        return Version(_require_string(value, label, limits))
    except MetadataError:
        raise
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _metadata_package_key(
    repository_id: str, ecosystem: Ecosystem, name: str
) -> PackageKey:
    try:
        return PackageKey(repository_id, ecosystem, name)
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _metadata_package_reference(ecosystem: Ecosystem, name: str) -> PackageReference:
    try:
        return PackageReference(ecosystem, name)
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _parse_package_id(value: object, label: str, limits: MetadataLimits) -> PackageKey:
    text = _require_string(value, label, limits)
    match = _PACKAGE_ID.fullmatch(text)
    if match is None:
        raise MetadataError(
            f"{label} must use repo:<path>::<ecosystem>:<name> identity"
        )
    try:
        return PackageKey(
            repository_id=match.group("repository"),
            ecosystem=Ecosystem(match.group("ecosystem")),
            name=match.group("name"),
        )
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _overlay_edge(value: object, index: int, limits: MetadataLimits) -> DependencyEdge:
    label = f"overlay.edges[{index}]"
    try:
        mapping = _strict_mapping(
            value, {"dependent", "dependency", "floor", "source"}, label
        )
        dependent = _parse_package_id(
            mapping.get("dependent"), f"{label}.dependent", limits
        )
        dependency = _parse_package_id(
            mapping.get("dependency"), f"{label}.dependency", limits
        )
        floor_value = mapping.get("floor")
        floor = (
            None
            if floor_value in (None, "")
            else VersionFloor.parse(
                _require_string(floor_value, f"{label}.floor", limits)
            )
        )
        source = _require_string(
            mapping.get("source", "overlay"), f"{label}.source", limits
        )
        return DependencyEdge(
            dependent=dependent, dependency=dependency, floor=floor, source=source
        )
    except MetadataError:
        raise
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _overlay_version(
    value: object, index: int, limits: MetadataLimits
) -> tuple[PackageKey, VersionSource]:
    label = f"overlay.versions[{index}]"
    try:
        mapping = _strict_mapping(value, {"package", "version", "source"}, label)
        package = _parse_package_id(mapping.get("package"), f"{label}.package", limits)
        version = Version(
            _require_string(mapping.get("version"), f"{label}.version", limits)
        )
        source = _require_string(
            mapping.get("source", "overlay"), f"{label}.source", limits
        )
        return package, VersionSource(location=source, version=version)
    except MetadataError:
        raise
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _parse_floor(
    value: object,
    label: str,
    limits: MetadataLimits,
    ecosystem: Ecosystem | None = None,
) -> VersionFloor | None:
    if value is None:
        return None
    text = _require_string(value, label, limits)
    if text in {"", "*"}:
        return None
    if ecosystem in {Ecosystem.RUST, Ecosystem.NODE}:
        partial = _PARTIAL_FLOOR.fullmatch(text)
        if partial is not None:
            numbers = partial.group("numbers").split(".")
            numbers.extend(["0"] * (3 - len(numbers)))
            operator = partial.group("operator")
            if operator is None:
                operator = "^" if ecosystem == Ecosystem.RUST else "=="
            text = f"{operator}{'.'.join(numbers)}"
    try:
        return VersionFloor.parse(text)
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _parse_requirement(
    value: object,
    ecosystem: Ecosystem,
    source: str,
    limits: MetadataLimits,
) -> DependencySpec:
    if ecosystem == Ecosystem.NODE:
        return _parse_node_requirement(value, source, limits)
    text = _require_string(value, source, limits)
    # Environment markers are not part of the C-11 floor.  Retain the package
    # edge while refusing to execute or evaluate the marker expression.
    requirement = text.split(";", 1)[0].strip()
    match = _REQUIREMENT_NAME.fullmatch(requirement)
    if match is None:
        raise MetadataError(f"unsupported dependency declaration: {text!r}")
    name = match.group("name")
    rest = match.group("rest").strip()
    if rest.startswith("["):
        closing = rest.find("]")
        if closing < 0:
            raise MetadataError(f"unsupported dependency declaration: {text!r}")
        rest = rest[closing + 1 :].strip()
    if rest.startswith("@"):
        raise MetadataError(f"direct dependency references are unsupported: {text!r}")
    floor = _parse_floor(rest, source, limits, ecosystem) if rest else None
    try:
        return DependencySpec(
            target=_metadata_package_reference(ecosystem, name),
            floor=floor,
            source=source,
        )
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _parse_node_requirement(
    value: object,
    source: str,
    limits: MetadataLimits,
    *,
    package_name: str | None = None,
) -> DependencySpec:
    """Parse one Node package name/range without merging name and version text."""

    text = _require_string(value, source, limits)
    requirement = text.split(";", 1)[0].strip()
    if package_name is None:
        if requirement.startswith("@"):
            slash = requirement.find("/")
            if slash <= 1:
                raise MetadataError(f"unsupported dependency declaration: {text!r}")
            separator = requirement.find("@", slash + 1)
        else:
            separator = requirement.find("@")
        if separator > 0:
            package_name = requirement[:separator]
            range_value = requirement[separator + 1 :]
            if not range_value:
                raise MetadataError(f"unsupported dependency declaration: {text!r}")
        else:
            package_name = requirement
            range_value = ""
    else:
        range_value = requirement
    if not _NODE_NAME.fullmatch(package_name):
        raise MetadataError(f"unsupported Node package name: {package_name!r}")
    if range_value.startswith(
        ("workspace:", "file:", "link:", "git:", "git+", "npm:", "http:", "https:")
    ):
        raise MetadataError(
            f"non-versioned Node dependency requires an overlay: {text!r}"
        )
    floor = (
        _parse_floor(range_value, source, limits, Ecosystem.NODE)
        if range_value
        else None
    )
    try:
        return DependencySpec(
            target=_metadata_package_reference(Ecosystem.NODE, package_name),
            floor=floor,
            source=source,
        )
    except WorkspaceReleaseError as exc:
        raise MetadataError(str(exc)) from exc


def _dependency_specs(
    value: object,
    ecosystem: Ecosystem,
    source: str,
    limits: MetadataLimits,
) -> tuple[DependencySpec, ...]:
    if value is None:
        return ()
    if isinstance(value, Mapping):
        items = tuple(value.items())
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = tuple((str(index), item) for index, item in enumerate(value))
    else:
        raise MetadataError(f"{source} dependencies must be a mapping or sequence")
    if len(items) > limits.max_dependencies:
        raise MetadataError(f"{source} dependency count exceeds the bound")
    specs: list[DependencySpec] = []
    for key, raw in items:
        if isinstance(value, Mapping):
            if not isinstance(key, str):
                raise MetadataError(f"{source} dependency name must be a string")
            if ecosystem == Ecosystem.RUST and isinstance(raw, Mapping):
                raw_map = _mapping(raw, f"{source}.{key}")
                dep_name = raw_map.get("package", key)
                dep_version = raw_map.get("version")
                if dep_version is None and any(
                    marker in raw_map for marker in ("path", "git", "workspace")
                ):
                    # Owner resolution remains possible by package name; the
                    # missing floor must be supplied by an explicit overlay if
                    # release policy requires one.
                    specs.append(
                        DependencySpec(
                            target=_metadata_package_reference(
                                ecosystem,
                                _require_string(
                                    dep_name, f"{source}.{key}.package", limits
                                ),
                            ),
                            source=f"{source}.{key}",
                        )
                    )
                    continue
                if dep_version is None:
                    raise MetadataError(f"{source}.{key} has no supported version")
                raw_requirement = _require_string(
                    dep_version, f"{source}.{key}.version", limits
                )
                if raw_requirement and raw_requirement[0] not in "^~<>=":
                    raw_requirement = f"^{raw_requirement}"
                specs.append(
                    DependencySpec(
                        target=_metadata_package_reference(
                            ecosystem,
                            _require_string(
                                dep_name, f"{source}.{key}.package", limits
                            ),
                        ),
                        floor=_parse_floor(
                            raw_requirement,
                            f"{source}.{key}.version",
                            limits,
                            ecosystem,
                        ),
                        source=f"{source}.{key}",
                    )
                )
                continue
            if ecosystem == Ecosystem.NODE:
                specs.append(
                    _parse_node_requirement(
                        raw,
                        f"{source}.{key}",
                        limits,
                        package_name=key,
                    )
                )
                continue
            if ecosystem == Ecosystem.PYTHON:
                if not isinstance(raw, str):
                    raise MetadataError(
                        f"{source}.{key} Python dependency must be a string"
                    )
                raw = f"{key}{raw}"
            else:
                raw = f"{key} {raw}" if isinstance(raw, str) else raw
            specs.append(_parse_requirement(raw, ecosystem, f"{source}.{key}", limits))
        else:
            specs.append(_parse_requirement(raw, ecosystem, f"{source}[{key}]", limits))
    return tuple(specs)


def read_python_metadata(
    repository_id: str, path: Path, *, limits: MetadataLimits | None = None
) -> ProjectRecord:
    """Read one Python ``pyproject.toml`` without invoking a build backend."""

    active = limits or MetadataLimits()
    repository_id = _canonical_project_id(repository_id)
    document = _load_toml(Path(path), active)
    project = _mapping(document.get("project"), f"{Path(path).name}.[project]")
    name = _require_string(project.get("name"), "Python project name", active)
    version_sources: list[VersionSource] = []
    if project.get("version") is not None:
        version_sources.append(
            VersionSource(
                location=f"{Path(path).name}:[project].version",
                version=_metadata_version(
                    project["version"], "Python project version", active
                ),
            )
        )
    dynamic = project.get("dynamic", ())
    if dynamic is not None and (
        isinstance(dynamic, (str, bytes, bytearray))
        or not isinstance(dynamic, Sequence)
    ):
        raise MetadataError("pyproject [project].dynamic must be a sequence")
    tool = _mapping(document.get("tool", {}), f"{Path(path).name}.[tool]")
    bumpversion = tool.get("bumpversion")
    if bumpversion is not None:
        bump_map = _mapping(bumpversion, f"{Path(path).name}.[tool.bumpversion]")
        current = bump_map.get("current_version")
        if current is not None:
            version_sources.append(
                VersionSource(
                    location=f"{Path(path).name}:[tool.bumpversion].current_version",
                    version=_metadata_version(
                        current, "bumpversion current_version", active
                    ),
                )
            )
    if not version_sources:
        raise MetadataError(f"{Path(path).name} has no static Python version source")
    package_key = _metadata_package_key(repository_id, Ecosystem.PYTHON, name)
    version, resolved_sources = resolve_version_sources(package_key, version_sources)
    dependencies = list(
        _dependency_specs(
            project.get("dependencies"),
            Ecosystem.PYTHON,
            "pyproject:[project].dependencies",
            active,
        )
    )
    optional = project.get("optional-dependencies", {})
    optional_map = _mapping(optional, "pyproject:[project].optional-dependencies")
    for extra_name in sorted(optional_map):
        values = optional_map[extra_name]
        dependencies.extend(
            _dependency_specs(
                values,
                Ecosystem.PYTHON,
                f"pyproject:[project].optional-dependencies.{extra_name}",
                active,
            )
        )
    package = PackageRecord(
        key=package_key,
        version=version,
        version_sources=resolved_sources,
        dependencies=tuple(dependencies),
        metadata_files=(Path(path).name,),
    )
    return ProjectRecord(
        repository_id=repository_id,
        packages=(package,),
        metadata_files=(Path(path).name,),
    )


def read_rust_metadata(
    repository_id: str, path: Path, *, limits: MetadataLimits | None = None
) -> ProjectRecord:
    """Read one Rust ``Cargo.toml`` package and declared dependencies."""

    active = limits or MetadataLimits()
    repository_id = _canonical_project_id(repository_id)
    document = _load_toml(Path(path), active)
    package = _mapping(document.get("package"), f"{Path(path).name}.[package]")
    name = _require_string(package.get("name"), "Cargo package name", active)
    raw_version = package.get("version")
    if not isinstance(raw_version, str):
        raise MetadataError("Cargo package version must be a static string")
    package_key = _metadata_package_key(repository_id, Ecosystem.RUST, name)
    version_source = VersionSource(
        location=f"{Path(path).name}:[package].version",
        version=_metadata_version(raw_version, "Cargo package version", active),
    )
    dependencies: list[DependencySpec] = []
    for section in ("dependencies", "dev-dependencies", "build-dependencies"):
        dependencies.extend(
            _dependency_specs(
                document.get(section),
                Ecosystem.RUST,
                f"{Path(path).name}:[{section}]",
                active,
            )
        )
    record = PackageRecord(
        key=package_key,
        version=version_source.version,
        version_sources=(version_source,),
        dependencies=tuple(dependencies),
        metadata_files=(Path(path).name,),
    )
    return ProjectRecord(
        repository_id=repository_id,
        packages=(record,),
        metadata_files=(Path(path).name,),
    )


def read_node_metadata(
    repository_id: str, path: Path, *, limits: MetadataLimits | None = None
) -> ProjectRecord:
    """Read one Node ``package.json`` without resolving or installing packages."""

    active = limits or MetadataLimits()
    repository_id = _canonical_project_id(repository_id)
    document = _load_json(Path(path), active)
    name = _require_string(document.get("name"), "Node package name", active)
    version = _metadata_version(document.get("version"), "Node package version", active)
    package_key = _metadata_package_key(repository_id, Ecosystem.NODE, name)
    dependencies: list[DependencySpec] = []
    for section in (
        "dependencies",
        "devDependencies",
        "peerDependencies",
        "optionalDependencies",
    ):
        dependencies.extend(
            _dependency_specs(
                document.get(section), Ecosystem.NODE, f"package.json.{section}", active
            )
        )
    source = VersionSource(location="package.json:version", version=version)
    record = PackageRecord(
        key=package_key,
        version=version,
        version_sources=(source,),
        dependencies=tuple(dependencies),
        metadata_files=(Path(path).name,),
    )
    return ProjectRecord(
        repository_id=repository_id,
        packages=(record,),
        metadata_files=(Path(path).name,),
    )


def _merge_project_records(
    records: Sequence[ProjectRecord], repository_id: str
) -> ProjectRecord:
    packages: list[PackageRecord] = []
    metadata_files: set[str] = set()
    for record in records:
        packages.extend(record.packages)
        metadata_files.update(record.metadata_files)
    return ProjectRecord(
        repository_id=repository_id,
        packages=tuple(packages),
        metadata_files=tuple(sorted(metadata_files)),
    )


def read_project_metadata(
    repository_id: str,
    project_root: Path,
    *,
    limits: MetadataLimits | None = None,
) -> ProjectRecord:
    """Read all supported declarative metadata at one project root."""

    active = limits or MetadataLimits()
    root = Path(project_root)
    records: list[ProjectRecord] = []
    pyproject = _project_file(root, "pyproject.toml")
    cargo = _project_file(root, "Cargo.toml")
    node = _project_file(root, "package.json")
    if pyproject.exists():
        records.append(read_python_metadata(repository_id, pyproject, limits=active))
    if cargo.exists():
        records.append(read_rust_metadata(repository_id, cargo, limits=active))
    if node.exists():
        records.append(read_node_metadata(repository_id, node, limits=active))
    if sum(len(record.packages) for record in records) > active.max_packages:
        raise MetadataError("project metadata package count exceeds the bound")
    return _merge_project_records(records, repository_id)


def read_workspace_metadata(
    projects: Mapping[str, Path],
    *,
    overlay: OverlayInput | Mapping[str, object] | None = None,
    limits: MetadataLimits | None = None,
) -> MetadataInventory:
    """Read selected projects in canonical key order and return immutable input."""

    active = limits or MetadataLimits()
    values = tuple(projects.items())
    if len(values) > active.max_packages:
        raise MetadataError("workspace project count exceeds the metadata bound")
    records: list[ProjectRecord] = []
    for repository_id, root in sorted(
        values, key=lambda item: canonical_repository_id(item[0])
    ):
        if not isinstance(root, Path):
            root = Path(root)
        records.append(read_project_metadata(repository_id, root, limits=active))
    explicit = (
        overlay
        if isinstance(overlay, OverlayInput)
        else OverlayInput.from_mapping(overlay, limits=active)
        if overlay is not None
        else OverlayInput(schema_version=1)
    )
    return MetadataInventory(
        projects=tuple(sorted(records, key=lambda item: item.project_id)),
        overlay=explicit,
    )


def _apply_overlay_versions(
    projects: Sequence[ProjectRecord], overlay: OverlayInput
) -> tuple[ProjectRecord, ...]:
    versions: dict[str, list[VersionSource]] = {}
    for package_key, source in overlay.versions:
        versions.setdefault(package_key.value, []).append(source)
    if not versions:
        return tuple(projects)
    updated: list[ProjectRecord] = []
    diagnostics: list[Diagnostic] = []
    for project in projects:
        packages: list[PackageRecord] = []
        for record in project.packages:
            sources = versions.get(record.key.value)
            if not sources:
                packages.append(record)
                continue
            try:
                version, resolved = resolve_version_sources(
                    record.key, (*record.version_sources, *sources)
                )
            except GraphValidationError as exc:
                diagnostics.extend(exc.diagnostics)
                continue
            packages.append(
                PackageRecord(
                    key=record.key,
                    version=version,
                    version_sources=resolved,
                    dependencies=record.dependencies,
                    metadata_files=record.metadata_files,
                )
            )
        updated.append(
            ProjectRecord(
                repository_id=project.repository_id,
                tree_sha=project.tree_sha,
                packages=tuple(packages),
                metadata_files=project.metadata_files,
            )
        )
    if diagnostics:
        raise GraphValidationError(diagnostics)
    missing = sorted(
        package_id
        for package_id in versions
        if not any(
            package_id == record.key.value
            for project in projects
            for record in project.packages
        )
    )
    if missing:
        raise GraphValidationError(
            tuple(_missing_overlay_package(package_id) for package_id in missing)
        )
    return tuple(updated)


def _missing_overlay_package(package_id: str) -> Diagnostic:
    return Diagnostic(
        code=GraphDiagnosticCode.MISSING_PACKAGE,
        subject=package_id,
        message="overlay version names an undeclared package",
    )


# Compatibility-friendly names for callers that prefer ecosystem-specific verbs.
read_python_project = read_python_metadata
read_rust_project = read_rust_metadata
read_node_project = read_node_metadata
parse_overlay = OverlayInput.from_mapping


__all__ = [
    "DEFAULT_MAX_METADATA_BYTES",
    "MetadataError",
    "MetadataInventory",
    "MetadataLimits",
    "OverlayInput",
    "parse_overlay",
    "read_node_metadata",
    "read_node_project",
    "read_project_metadata",
    "read_python_metadata",
    "read_python_project",
    "read_rust_metadata",
    "read_rust_project",
    "read_workspace_metadata",
]
