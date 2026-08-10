"""Pure semantic-version and dependency-floor planning.

Checkpoint 3 consumes the verified graph and selected closure from the earlier
RMDD-18 checkpoints.  It adds no checkout reader, writer, package-manager
integration, or execution path.  Callers provide immutable metadata-site
descriptors so every preview can point at an exact declarative source without
requiring the planner to open a project file.

The deliberately small site descriptors are the security boundary: paths are
relative and bounded, selectors are explicit, source values are compared with
the frozen graph, and the representation/policy pair is never inferred from a
range string.  A plan contains only deterministic, restart-safe data.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeVar, cast

from .workspace_release import (
    MAX_EDGES,
    MAX_PACKAGES,
    MAX_STRING_LENGTH,
    DependencyEdge,
    DependencyGraph,
    Ecosystem,
    PackageKey,
    PackageRecord,
    ProjectRecord,
    Version,
    VersionFloor,
    WorkspaceReleaseError,
    _canonical_json,
)
from .workspace_selection import SelectedChangeClosure

MAX_VERSION_SITES = MAX_PACKAGES
MAX_FLOOR_SITES = MAX_EDGES
MAX_WITNESSES = 16
MAX_VERSION_TEXT = 256

_SHA = re.compile(r"^[0-9a-fA-F]{40,64}$")
_STABLE_VERSION = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)$")
_PARTIAL_VERSION = re.compile(
    r"^(0|[1-9][0-9]*)(?:\.(0|[1-9][0-9]*))?(?:\.(0|[1-9][0-9]*))?$"
)
_T = TypeVar("_T")
_AUTO_PLAN_DIGEST = "__version_plan_digest_auto__"


class VersionPlanningCode(StrEnum):
    """Stable refusal categories for the pure version planner."""

    INVALID_INPUT = "invalid_input"
    UNBOUNDED_INPUT = "unbounded_input"
    PATH_TRAVERSAL = "path_traversal"
    SYMLINK = "symlink"
    SOURCE_SHA = "source_sha"
    GRAPH_DRIFT = "graph_drift"
    SELECTION_DRIFT = "selection_drift"
    CYCLE = "cycle"
    MISSING_PACKAGE = "missing_package"
    MISSING_SOURCE_SITE = "missing_source_site"
    DUPLICATE_SOURCE_SITE = "duplicate_source_site"
    CONFLICTING_VERSION_SOURCE = "conflicting_version_source"
    MISSING_FLOOR_SITE = "missing_floor_site"
    DUPLICATE_REWRITE_SITE = "duplicate_rewrite_site"
    CONFLICTING_REWRITE_SITE = "conflicting_rewrite_site"
    UNSUPPORTED_SPECIFIER = "unsupported_specifier"
    NON_SEMVER = "non_semver"
    PRERELEASE = "prerelease"
    LOCAL_VERSION = "local_version"
    DIGEST = "digest"


class VersionPlanningError(WorkspaceReleaseError):
    """A version/floor input or frozen result failed closed."""

    def __init__(
        self,
        code: VersionPlanningCode,
        message: str,
        *,
        details: Iterable[tuple[str, str]] = (),
    ) -> None:
        if not isinstance(code, VersionPlanningCode):
            raise ValueError("version planning code must be supported")
        self.code = code
        self.details = _bounded_pairs(details, "planning diagnostics")
        super().__init__(f"{code.value}: {message}")


class MetadataRepresentation(StrEnum):
    """Declarative metadata representation supported by this checkpoint."""

    PYTHON = "python"
    RUST = "rust"
    NODE = "node"

    # Descriptive aliases keep the wire vocabulary obvious without creating
    # additional accepted representations.
    PYTHON_TOML = "python"
    RUST_TOML = "rust"
    NODE_JSON = "node"


class VersionBump(StrEnum):
    """Explicit semantic-version transition policy."""

    MAJOR = "major"
    MINOR = "minor"
    PATCH = "patch"
    EXACT = "exact"


class FloorPolicy(StrEnum):
    """Explicit output policy for a dependency minimum floor."""

    RANGE = "range"  # >=
    COMPATIBLE = "compatible"  # Python ~=
    CARET = "caret"  # Rust/Node ^
    TILDE = "tilde"  # Rust/Node ~
    EXACT = "exact"  # == (or representation-specific exact syntax)


class VersionPreviewReason(StrEnum):
    """Stable explanation for a version preview."""

    BUMP = "version_bump"
    ALREADY_CURRENT = "already_current"


class FloorPreviewReason(StrEnum):
    """Stable explanation for a floor preview."""

    TRANSITIVE_MINIMUM = "transitive_minimum"
    ALREADY_SATISFIED = "already_satisfied"
    DEPENDENCY_UNCHANGED = "dependency_unchanged"


def _fail(
    code: VersionPlanningCode,
    message: str,
    *,
    details: Iterable[tuple[str, str]] = (),
) -> VersionPlanningError:
    return VersionPlanningError(code, message, details=details)


def _bounded_text(
    value: object,
    field_name: str,
    *,
    max_length: int = MAX_VERSION_TEXT,
    code: VersionPlanningCode = VersionPlanningCode.INVALID_INPUT,
) -> str:
    if not isinstance(value, str):
        raise _fail(code, f"{field_name} must be a string")
    if not value or value.strip() != value:
        raise _fail(code, f"{field_name} must be non-blank and trimmed")
    if len(value) > max_length:
        raise _fail(
            VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} exceeds the bound"
        )
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise _fail(code, f"{field_name} contains a control character")
    return value


def _bounded_pairs(
    value: object, field_name: str, *, max_items: int = MAX_WITNESSES
) -> tuple[tuple[str, str], ...]:
    if type(value) not in (tuple, list):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            f"{field_name} must use an exact builtin sequence",
        )
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Iterable
    ):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT, f"{field_name} must be a sequence"
        )
    try:
        iterator = iter(value)
    except Exception:
        raise _fail(
            VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} could not be read"
        ) from None
    result: list[tuple[str, str]] = []
    for _ in range(max_items):
        try:
            item = next(iterator)
        except StopIteration:
            return tuple(result)
        except Exception:
            raise _fail(
                VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} could not be read"
            ) from None
        # A list/tuple subclass can override ``__len__`` (or iteration).  Check
        # exact builtin containers before touching either operation so hostile
        # nested diagnostics fail closed without executing caller code.
        if type(item) not in (tuple, list) or len(item) != 2:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT, f"{field_name} must contain pairs"
            )
        key, item_value = item
        result.append(
            (
                _bounded_text(key, f"{field_name} key", max_length=128),
                _bounded_text(item_value, f"{field_name} value"),
            )
        )
    try:
        next(iterator)
    except StopIteration:
        return tuple(result)
    except Exception:
        raise _fail(
            VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} could not be read"
        ) from None
    raise _fail(VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} exceeds the bound")


def _bounded_sequence(
    value: object,
    field_name: str,
    *,
    max_items: int,
    exact_builtin: bool = False,
) -> tuple[object, ...]:
    if exact_builtin and type(value) not in (tuple, list):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            f"{field_name} must use an exact builtin sequence",
        )
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Iterable
    ):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT, f"{field_name} must be a sequence"
        )
    try:
        iterator = iter(value)
    except Exception:
        raise _fail(
            VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} could not be read"
        ) from None
    result: list[object] = []
    for _ in range(max_items):
        try:
            result.append(next(iterator))
        except StopIteration:
            return tuple(result)
        except Exception:
            raise _fail(
                VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} could not be read"
            ) from None
    try:
        next(iterator)
    except StopIteration:
        return tuple(result)
    except Exception:
        raise _fail(
            VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} could not be read"
        ) from None
    raise _fail(VersionPlanningCode.UNBOUNDED_INPUT, f"{field_name} exceeds the bound")


def _typed_sequence(
    value: object,
    field_name: str,
    item_type: type[_T],
    *,
    max_items: int,
) -> tuple[_T, ...]:
    values = _bounded_sequence(value, field_name, max_items=max_items)
    if any(type(item) is not item_type for item in values):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            f"{field_name} entries must be {item_type.__name__} values",
        )
    return cast(tuple[_T, ...], values)


def _safe_relative_path(value: object) -> str:
    text = _bounded_text(
        value,
        "metadata file path",
        max_length=MAX_VERSION_TEXT,
        code=VersionPlanningCode.PATH_TRAVERSAL,
    )
    if (
        text.startswith("/")
        or "\\" in text
        or ":" in text
        or "//" in text
        or "\x00" in text
    ):
        raise _fail(
            VersionPlanningCode.PATH_TRAVERSAL,
            "metadata file path must be a relative POSIX path",
        )
    parts = text.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise _fail(
            VersionPlanningCode.PATH_TRAVERSAL,
            "metadata file path contains traversal components",
        )
    path = PurePosixPath(text)
    if path.is_absolute() or path.as_posix() != text:
        raise _fail(
            VersionPlanningCode.PATH_TRAVERSAL,
            "metadata file path is not canonical",
        )
    return text


def _sha(value: object, field_name: str) -> str:
    text = _bounded_text(value, field_name, max_length=64)
    if _SHA.fullmatch(text) is None:
        raise _fail(VersionPlanningCode.SOURCE_SHA, f"{field_name} must be a full SHA")
    return text.lower()


def _stable_version(value: object, field_name: str) -> Version:
    if isinstance(value, Version):
        text = value.value
    elif isinstance(value, str):
        text = value
    else:
        raise _fail(
            VersionPlanningCode.NON_SEMVER, f"{field_name} must be a semantic version"
        )
    if "+" in text:
        raise _fail(
            VersionPlanningCode.LOCAL_VERSION,
            f"{field_name} cannot use a local version",
        )
    if "-" in text:
        raise _fail(
            VersionPlanningCode.PRERELEASE, f"{field_name} cannot use a prerelease"
        )
    if _STABLE_VERSION.fullmatch(text) is None:
        raise _fail(
            VersionPlanningCode.NON_SEMVER, f"{field_name} must use MAJOR.MINOR.PATCH"
        )
    try:
        return Version(text)
    except WorkspaceReleaseError:
        raise _fail(
            VersionPlanningCode.NON_SEMVER, f"{field_name} is not semantic-version data"
        ) from None


def _version_literal(
    value: str,
    field_name: str,
    representation: MetadataRepresentation | None = None,
) -> tuple[str, str]:
    text = _bounded_text(value, field_name)
    if len(text) >= 2 and text[0] in {'"', "'"} and text[-1] == text[0]:
        inner = text[1:-1]
        if not inner or "\\" in inner or text[0] in inner:
            raise _fail(
                VersionPlanningCode.UNSUPPORTED_SPECIFIER,
                f"{field_name} has an unsupported quoted value",
            )
        if representation is MetadataRepresentation.NODE and text[0] != '"':
            raise _fail(
                VersionPlanningCode.UNSUPPORTED_SPECIFIER,
                f"{field_name} must use JSON double quotes",
            )
        return inner, text[0]
    if text[:1] in {'"', "'"} or text[-1:] in {'"', "'"}:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            f"{field_name} has an unterminated quoted value",
        )
    if representation is MetadataRepresentation.NODE:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            f"{field_name} must be a JSON string literal",
        )
    return text, ""


def _render_literal(
    value: str,
    new_value: str,
    representation: MetadataRepresentation | None = None,
) -> str:
    _, quote = _version_literal(value, "metadata value", representation)
    return f"{quote}{new_value}{quote}" if quote else new_value


def _validate_package_representation(
    package: PackageKey,
    representation: MetadataRepresentation,
    field_name: str,
) -> None:
    """Bind a metadata representation to the package it describes."""

    if not isinstance(representation, MetadataRepresentation):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            f"{field_name} representation is unsupported",
        )
    if not isinstance(package.ecosystem, Ecosystem):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            f"{field_name} package ecosystem is unsupported",
        )
    if package.ecosystem.value != representation.value:
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            f"{field_name} representation does not match its package ecosystem",
        )


def _validate_representation_path(
    representation: MetadataRepresentation, file_path: str
) -> None:
    expected = {
        MetadataRepresentation.PYTHON: "pyproject.toml",
        MetadataRepresentation.RUST: "Cargo.toml",
        MetadataRepresentation.NODE: "package.json",
    }[representation]
    if PurePosixPath(file_path).name != expected:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            f"{representation.value} sites must use {expected}",
        )


def _validate_selector(file_path: str, selector: str) -> str:
    text = _bounded_text(selector, "metadata source selector")
    if not text.startswith(f"{file_path}:"):
        raise _fail(
            VersionPlanningCode.CONFLICTING_VERSION_SOURCE,
            "metadata selector is not bound to its file path",
        )
    suffix = text[len(file_path) + 1 :]
    if not suffix or ".." in suffix or "\\" in suffix or "\x00" in suffix:
        raise _fail(
            VersionPlanningCode.PATH_TRAVERSAL,
            "metadata selector is not a bounded declarative location",
        )
    return text


def _typed_witness(value: object) -> tuple[str, ...]:
    values = _bounded_sequence(value, "preview witnesses", max_items=MAX_WITNESSES)
    if any(type(item) is not str for item in values):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT, "preview witnesses must be strings"
        )
    return tuple(
        _bounded_text(item, "preview witness") for item in cast(tuple[str, ...], values)
    )


@dataclass(frozen=True, slots=True)
class VersionSourcePolicy:
    """Explicit version-source selector and semantic transition policy."""

    source_location: str
    representation: MetadataRepresentation
    bump: VersionBump
    exact_version: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_location",
            _bounded_text(self.source_location, "version source location"),
        )
        if not isinstance(self.representation, MetadataRepresentation):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version representation must be supported",
            )
        if not isinstance(self.bump, VersionBump):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT, "version bump must be explicit"
            )
        if self.exact_version is not None:
            object.__setattr__(
                self,
                "exact_version",
                _bounded_text(self.exact_version, "exact next version"),
            )
        if self.bump is VersionBump.EXACT and self.exact_version is None:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "exact version policy requires exact_version",
            )
        if self.bump is not VersionBump.EXACT and self.exact_version is not None:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "non-exact version policy cannot carry exact_version",
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "source_location": self.source_location,
            "representation": self.representation.value,
            "bump": self.bump.value,
            "exact_version": self.exact_version,
        }


@dataclass(frozen=True, slots=True)
class VersionSourceSite:
    """One exact version literal site supplied by a declarative reader."""

    package: PackageKey
    file_path: str
    old_text: str
    policy: VersionSourcePolicy
    symlink: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.package, PackageKey):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version site package must be a PackageKey",
            )
        file_path = _safe_relative_path(self.file_path)
        object.__setattr__(self, "file_path", file_path)
        if not isinstance(self.policy, VersionSourcePolicy):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version site policy must be explicit",
            )
        _validate_package_representation(
            self.package, self.policy.representation, "version site"
        )
        _validate_representation_path(self.policy.representation, file_path)
        _validate_selector(file_path, self.policy.source_location)
        object.__setattr__(
            self, "old_text", _bounded_text(self.old_text, "version site old text")
        )
        _version_literal(
            self.old_text,
            "version site old text",
            self.policy.representation,
        )
        if type(self.symlink) is not bool:
            raise _fail(
                VersionPlanningCode.SYMLINK, "version site symlink flag must be boolean"
            )
        if self.symlink:
            raise _fail(
                VersionPlanningCode.SYMLINK, "version metadata site cannot be a symlink"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "package": self.package.value,
            "file_path": self.file_path,
            "old_text": self.old_text,
            "policy": self.policy.canonical_payload(),
        }


@dataclass(frozen=True, slots=True)
class FloorRewriteSite:
    """One exact dependency literal site and its explicit floor policy."""

    dependent: PackageKey
    dependency: PackageKey
    file_path: str
    source_location: str
    representation: MetadataRepresentation
    old_text: str
    policy: FloorPolicy
    symlink: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.dependent, PackageKey) or not isinstance(
            self.dependency, PackageKey
        ):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor site endpoints must be PackageKey values",
            )
        file_path = _safe_relative_path(self.file_path)
        object.__setattr__(self, "file_path", file_path)
        if not isinstance(self.representation, MetadataRepresentation):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor representation must be explicit",
            )
        _validate_package_representation(
            self.dependent, self.representation, "floor site"
        )
        _validate_representation_path(self.representation, file_path)
        object.__setattr__(
            self, "source_location", _validate_selector(file_path, self.source_location)
        )
        object.__setattr__(
            self,
            "old_text",
            _bounded_text(
                self.old_text, "floor site old text", max_length=MAX_VERSION_TEXT
            ),
        )
        _floor_literal(self.old_text, self.representation)
        if not isinstance(self.policy, FloorPolicy):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT, "floor policy must be explicit"
            )
        if type(self.symlink) is not bool:
            raise _fail(
                VersionPlanningCode.SYMLINK, "floor site symlink flag must be boolean"
            )
        if self.symlink:
            raise _fail(
                VersionPlanningCode.SYMLINK,
                "dependency metadata site cannot be a symlink",
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "dependent": self.dependent.value,
            "dependency": self.dependency.value,
            "file_path": self.file_path,
            "source_location": self.source_location,
            "representation": self.representation.value,
            "old_text": self.old_text,
            "policy": self.policy.value,
        }


@dataclass(frozen=True, slots=True)
class VersionPlanningInput:
    """Closed immutable input bundle for one pure planning invocation."""

    graph: DependencyGraph
    selection: SelectedChangeClosure
    version_sites: tuple[VersionSourceSite, ...] = ()
    floor_sites: tuple[FloorRewriteSite, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.graph, DependencyGraph):
            raise _fail(
                VersionPlanningCode.GRAPH_DRIFT,
                "planning input graph must be a DependencyGraph",
            )
        if not isinstance(self.selection, SelectedChangeClosure):
            raise _fail(
                VersionPlanningCode.SELECTION_DRIFT,
                "planning input selection must be a SelectedChangeClosure",
            )
        object.__setattr__(
            self,
            "version_sites",
            _typed_sequence(
                self.version_sites,
                "planning input version sites",
                VersionSourceSite,
                max_items=MAX_VERSION_SITES,
            ),
        )
        object.__setattr__(
            self,
            "floor_sites",
            _typed_sequence(
                self.floor_sites,
                "planning input floor sites",
                FloorRewriteSite,
                max_items=MAX_FLOOR_SITES,
            ),
        )
        # Site collections are evidence sets, not ordered instructions.  Keep
        # duplicates/conflicts intact while making their payload independent of
        # caller iteration order.
        object.__setattr__(
            self,
            "version_sites",
            tuple(
                sorted(
                    self.version_sites,
                    key=lambda site: _canonical_json(site.canonical_payload()),
                )
            ),
        )
        object.__setattr__(
            self,
            "floor_sites",
            tuple(
                sorted(
                    self.floor_sites,
                    key=lambda site: _canonical_json(site.canonical_payload()),
                )
            ),
        )

    def canonical_payload(self) -> dict[str, object]:
        """Return deterministic input evidence without retaining live records."""

        return {
            "graph_digest": self.graph.digest,
            "selection_digest": self.selection.digest,
            "version_sites": [site.canonical_payload() for site in self.version_sites],
            "floor_sites": [site.canonical_payload() for site in self.floor_sites],
        }


def _stable_floor(value: VersionFloor | None, field_name: str) -> VersionFloor | None:
    if value is None:
        return None
    if not isinstance(value, VersionFloor):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT, f"{field_name} must be a VersionFloor"
        )
    _stable_version(value.version, f"{field_name} version")
    return value


def _partial_floor(text: str, representation: MetadataRepresentation) -> VersionFloor:
    if text in {"", "*"}:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            "an empty floor is not a floor value",
        )
    operator = ""
    numbers = text
    if text.startswith(">="):
        operator, numbers = ">=", text[2:]
    elif text.startswith("<=") or text.startswith(">") or text.startswith("<"):
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            "upper-bound or open range is unsupported",
        )
    elif text.startswith("~="):
        operator, numbers = "~=", text[2:]
    elif text.startswith("^"):
        operator, numbers = "^", text[1:]
    elif text.startswith("~"):
        operator, numbers = "~", text[1:]
    elif text.startswith("=="):
        operator, numbers = "==", text[2:]
    elif representation is MetadataRepresentation.RUST and text.startswith("="):
        operator, numbers = "==", text[1:]
    if any(char in numbers for char in (",", "|", " ")):
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            "compound dependency ranges are unsupported",
        )
    match = _PARTIAL_VERSION.fullmatch(numbers)
    if match is None:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            "dependency floor is not a supported semantic version",
        )
    components = [part or "0" for part in match.groups()]
    if not operator:
        operator = "^" if representation is MetadataRepresentation.RUST else "=="
    if operator == "~=" and representation is not MetadataRepresentation.PYTHON:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            "compatible floors are Python-only",
        )
    if operator in {"^", "~"} and representation is MetadataRepresentation.PYTHON:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER,
            "caret/tilde floors are not Python policy",
        )
    try:
        return VersionFloor(operator=operator, version=Version(".".join(components)))
    except WorkspaceReleaseError:
        raise _fail(
            VersionPlanningCode.UNSUPPORTED_SPECIFIER, "dependency floor is unsupported"
        ) from None


def _floor_literal(
    value: str, representation: MetadataRepresentation
) -> tuple[VersionFloor | None, str]:
    if value == "":
        return None, ""
    text = _bounded_text(value, "floor site old text", max_length=MAX_VERSION_TEXT)
    inner, quote = _version_literal(text, "floor site old text", representation)
    if inner == "*":
        return None, quote
    return _partial_floor(inner, representation), quote


def _floor_operator(policy: FloorPolicy, representation: MetadataRepresentation) -> str:
    if policy is FloorPolicy.RANGE:
        return ">="
    if policy is FloorPolicy.COMPATIBLE:
        if representation is not MetadataRepresentation.PYTHON:
            raise _fail(
                VersionPlanningCode.UNSUPPORTED_SPECIFIER,
                "compatible policy is supported only for Python",
            )
        return "~="
    if policy is FloorPolicy.CARET:
        if representation is MetadataRepresentation.PYTHON:
            raise _fail(
                VersionPlanningCode.UNSUPPORTED_SPECIFIER,
                "caret policy is unsupported for Python",
            )
        return "^"
    if policy is FloorPolicy.TILDE:
        if representation is MetadataRepresentation.PYTHON:
            raise _fail(
                VersionPlanningCode.UNSUPPORTED_SPECIFIER,
                "tilde policy is unsupported for Python",
            )
        return "~"
    return "=="


def _new_floor(
    version: Version, policy: FloorPolicy, representation: MetadataRepresentation
) -> VersionFloor:
    return VersionFloor(
        operator=_floor_operator(policy, representation), version=version
    )


def _render_floor(
    floor: VersionFloor,
    policy: FloorPolicy,
    representation: MetadataRepresentation,
    quote: str,
) -> str:
    if policy is FloorPolicy.EXACT and representation is MetadataRepresentation.NODE:
        text = floor.version.value
    elif policy is FloorPolicy.EXACT and representation is MetadataRepresentation.RUST:
        text = f"={floor.version.value}"
    else:
        text = floor.value
    return f"{quote}{text}{quote}" if quote else text


def _project_for_package(
    projects: Mapping[str, ProjectRecord], package: PackageKey
) -> ProjectRecord:
    project = projects.get(package.repository_id)
    if project is None:
        raise _fail(
            VersionPlanningCode.MISSING_PACKAGE,
            f"package owner is not frozen: {package.value}",
        )
    return project


def _metadata_file_known(
    project: ProjectRecord, package: PackageRecord, file_path: str
) -> bool:
    known = set(project.metadata_files) | set(package.metadata_files)
    return not known or file_path in known


def _current_version_literal(
    old_text: str,
    expected: Version,
    representation: MetadataRepresentation,
) -> None:
    literal, _ = _version_literal(old_text, "version site old text", representation)
    observed = _stable_version(literal, "version site old text")
    if observed != expected:
        raise _fail(
            VersionPlanningCode.CONFLICTING_VERSION_SOURCE,
            "version site old text disagrees with the frozen package version",
            details=(("frozen", expected.value), ("site", observed.value)),
        )


def _next_version(current: Version, policy: VersionSourcePolicy) -> Version:
    stable = _stable_version(current, "current package version")
    if policy.bump is VersionBump.EXACT:
        assert policy.exact_version is not None
        return _stable_version(policy.exact_version, "exact next version")
    major, minor, patch = stable.numeric
    if policy.bump is VersionBump.MAJOR:
        candidate = f"{major + 1}.0.0"
    elif policy.bump is VersionBump.MINOR:
        candidate = f"{major}.{minor + 1}.0"
    else:
        candidate = f"{major}.{minor}.{patch + 1}"
    # Reuse the bounded semantic-version adapter so an overflowing numeric
    # component cannot leak WorkspaceReleaseError from Version(...).
    return _stable_version(candidate, "next package version")


def _package_batches(
    package_ids: tuple[str, ...], edges: tuple[DependencyEdge, ...]
) -> tuple[tuple[str, ...], ...]:
    dependencies: dict[str, set[str]] = {
        package_id: set() for package_id in package_ids
    }
    dependents: dict[str, set[str]] = {package_id: set() for package_id in package_ids}
    for edge in edges:
        dependent, dependency = edge.dependent.value, edge.dependency.value
        if dependent == dependency:
            raise _fail(
                VersionPlanningCode.CYCLE,
                "package dependency graph contains a self-cycle",
            )
        if dependent not in dependencies or dependency not in dependencies:
            raise _fail(
                VersionPlanningCode.MISSING_PACKAGE,
                "package edge is outside the frozen selection",
            )
        dependencies[dependent].add(dependency)
        dependents[dependency].add(dependent)
    remaining = set(package_ids)
    groups: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(
                package_id for package_id in remaining if not dependencies[package_id]
            )
        )
        if not ready:
            raise _fail(
                VersionPlanningCode.CYCLE, "package dependency graph contains a cycle"
            )
        groups.append(ready)
        for package_id in ready:
            remaining.remove(package_id)
            for dependent in dependents[package_id]:
                dependencies[dependent].discard(package_id)
    return tuple(groups)


def _digest(value: object, field_name: str) -> str:
    text = _bounded_text(value, field_name, max_length=64)
    if len(text) != 64 or any(char not in "0123456789abcdefABCDEF" for char in text):
        raise _fail(
            VersionPlanningCode.DIGEST, f"{field_name} must be a SHA-256 digest"
        )
    return text.lower()


@dataclass(frozen=True, slots=True)
class VersionPreview:
    """Exact old/new version text bound to a frozen source and plan."""

    project_id: str
    package: PackageKey
    file_path: str
    source_location: str
    representation: MetadataRepresentation
    source_sha: str
    old_text: str
    new_text: str
    current_version: Version
    next_version: Version
    policy: VersionSourcePolicy
    reason: VersionPreviewReason
    witness: tuple[str, ...]
    graph_digest: str
    selection_digest: str
    is_noop: bool
    plan_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.package, PackageKey):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview package must be a PackageKey",
            )
        if not isinstance(self.representation, MetadataRepresentation):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview representation is invalid",
            )
        _validate_package_representation(
            self.package, self.representation, "version preview"
        )
        file_path = _safe_relative_path(self.file_path)
        object.__setattr__(self, "file_path", file_path)
        _validate_representation_path(self.representation, file_path)
        object.__setattr__(
            self, "source_location", _validate_selector(file_path, self.source_location)
        )
        object.__setattr__(
            self, "source_sha", _sha(self.source_sha, "version preview source SHA")
        )
        current = _stable_version(
            self.current_version, "version preview current version"
        )
        next_value = _stable_version(self.next_version, "version preview next version")
        object.__setattr__(self, "current_version", current)
        object.__setattr__(self, "next_version", next_value)
        object.__setattr__(
            self, "old_text", _bounded_text(self.old_text, "version preview old text")
        )
        object.__setattr__(
            self, "new_text", _bounded_text(self.new_text, "version preview new text")
        )
        _version_literal(
            self.old_text,
            "version preview old text",
            self.representation,
        )
        if self.representation is MetadataRepresentation.NODE:
            _version_literal(
                self.new_text,
                "version preview new text",
                self.representation,
            )
        project_id = _bounded_text(
            self.project_id,
            "version preview project ID",
            max_length=MAX_STRING_LENGTH,
        )
        if project_id != self.package.repository_id:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview project ID must own its package",
            )
        object.__setattr__(self, "project_id", project_id)
        if (
            not isinstance(self.policy, VersionSourcePolicy)
            or self.policy.source_location != self.source_location
        ):
            raise _fail(
                VersionPlanningCode.CONFLICTING_VERSION_SOURCE,
                "version preview policy is not bound to its source",
            )
        if self.policy.representation is not self.representation:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview policy representation is not bound to its site",
            )
        if not isinstance(self.reason, VersionPreviewReason):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT, "version preview reason is invalid"
            )
        object.__setattr__(self, "witness", _typed_witness(self.witness))
        if type(self.is_noop) is not bool:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview no-op flag must be boolean",
            )
        expected_reason = (
            VersionPreviewReason.ALREADY_CURRENT
            if self.is_noop
            else VersionPreviewReason.BUMP
        )
        if self.reason is not expected_reason:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview reason does not match no-op evidence",
            )
        if self.is_noop != (self.old_text == self.new_text and current == next_value):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version preview no-op evidence is inconsistent",
            )
        object.__setattr__(
            self,
            "graph_digest",
            _digest(self.graph_digest, "version preview graph digest"),
        )
        object.__setattr__(
            self,
            "selection_digest",
            _digest(self.selection_digest, "version preview selection digest"),
        )
        if self.plan_digest:
            object.__setattr__(
                self,
                "plan_digest",
                _digest(self.plan_digest, "version preview plan digest"),
            )

    def canonical_payload(
        self, *, include_plan_digest: bool = False
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "project_id": self.project_id,
            "package": self.package.value,
            "file_path": self.file_path,
            "source_location": self.source_location,
            "representation": self.representation.value,
            "source_sha": self.source_sha,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "current_version": self.current_version.value,
            "next_version": self.next_version.value,
            "policy": self.policy.canonical_payload(),
            "reason": self.reason.value,
            "witness": self.witness,
            "graph_digest": self.graph_digest,
            "selection_digest": self.selection_digest,
            "is_noop": self.is_noop,
        }
        if include_plan_digest:
            payload["plan_digest"] = self.plan_digest
        return payload


@dataclass(frozen=True, slots=True)
class FloorPreview:
    """Exact old/new dependency-floor text and normalized values."""

    project_id: str
    dependent: PackageKey
    dependency: PackageKey
    file_path: str
    source_location: str
    representation: MetadataRepresentation
    source_sha: str
    old_text: str
    new_text: str
    old_normalized: str
    new_normalized: str
    policy: FloorPolicy
    reason: FloorPreviewReason
    witness: tuple[str, ...]
    graph_digest: str
    selection_digest: str
    is_noop: bool
    plan_digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.dependent, PackageKey) or not isinstance(
            self.dependency, PackageKey
        ):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor preview endpoints must be PackageKey values",
            )
        _validate_package_representation(
            self.dependent, self.representation, "floor preview"
        )
        file_path = _safe_relative_path(self.file_path)
        object.__setattr__(self, "file_path", file_path)
        if not isinstance(self.representation, MetadataRepresentation):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor preview representation is invalid",
            )
        _validate_representation_path(self.representation, file_path)
        object.__setattr__(
            self, "source_location", _validate_selector(file_path, self.source_location)
        )
        object.__setattr__(
            self, "source_sha", _sha(self.source_sha, "floor preview source SHA")
        )
        object.__setattr__(
            self,
            "old_text",
            _bounded_text(
                self.old_text, "floor preview old text", max_length=MAX_VERSION_TEXT
            )
            if self.old_text
            else "",
        )
        object.__setattr__(
            self,
            "new_text",
            _bounded_text(
                self.new_text, "floor preview new text", max_length=MAX_VERSION_TEXT
            )
            if self.new_text
            else "",
        )
        _floor_literal(self.old_text, self.representation)
        if self.representation is MetadataRepresentation.NODE and self.new_text:
            _floor_literal(self.new_text, self.representation)
        project_id = _bounded_text(
            self.project_id,
            "floor preview project ID",
            max_length=MAX_STRING_LENGTH,
        )
        if project_id != self.dependent.repository_id:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor preview project ID must own its dependent package",
            )
        object.__setattr__(self, "project_id", project_id)
        for field_name in ("old_normalized", "new_normalized"):
            value = getattr(self, field_name)
            if value:
                _partial_floor(_bounded_text(value, field_name), self.representation)
            elif field_name == "new_normalized" and not self.is_noop:
                raise _fail(
                    VersionPlanningCode.INVALID_INPUT,
                    "changed floor preview must have a new normalized value",
                )
        if not isinstance(self.policy, FloorPolicy):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT, "floor preview policy is invalid"
            )
        _floor_operator(self.policy, self.representation)
        if not isinstance(self.reason, FloorPreviewReason):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT, "floor preview reason is invalid"
            )
        object.__setattr__(self, "witness", _typed_witness(self.witness))
        if type(self.is_noop) is not bool:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor preview no-op flag must be boolean",
            )
        if self.is_noop and self.reason not in {
            FloorPreviewReason.ALREADY_SATISFIED,
            FloorPreviewReason.DEPENDENCY_UNCHANGED,
        }:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "no-op floor preview reason is unsupported",
            )
        if (
            not self.is_noop
            and self.reason is not FloorPreviewReason.TRANSITIVE_MINIMUM
        ):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "changed floor preview reason does not match evidence",
            )
        if self.is_noop != (
            self.old_text == self.new_text
            and self.old_normalized == self.new_normalized
        ):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor preview no-op evidence is inconsistent",
            )
        object.__setattr__(
            self,
            "graph_digest",
            _digest(self.graph_digest, "floor preview graph digest"),
        )
        object.__setattr__(
            self,
            "selection_digest",
            _digest(self.selection_digest, "floor preview selection digest"),
        )
        if self.plan_digest:
            object.__setattr__(
                self,
                "plan_digest",
                _digest(self.plan_digest, "floor preview plan digest"),
            )

    def canonical_payload(
        self, *, include_plan_digest: bool = False
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "project_id": self.project_id,
            "dependent": self.dependent.value,
            "dependency": self.dependency.value,
            "file_path": self.file_path,
            "source_location": self.source_location,
            "representation": self.representation.value,
            "source_sha": self.source_sha,
            "old_text": self.old_text,
            "new_text": self.new_text,
            "old_normalized": self.old_normalized,
            "new_normalized": self.new_normalized,
            "policy": self.policy.value,
            "reason": self.reason.value,
            "witness": self.witness,
            "graph_digest": self.graph_digest,
            "selection_digest": self.selection_digest,
            "is_noop": self.is_noop,
        }
        if include_plan_digest:
            payload["plan_digest"] = self.plan_digest
        return payload


def _preview_digest_failure() -> VersionPlanningError:
    """Return one privacy-safe error for any forged preview evidence."""

    return _fail(
        VersionPlanningCode.DIGEST,
        "preview evidence does not match the enclosing plan digest",
    )


def _validate_version_preview_evidence(preview: VersionPreview) -> None:
    """Cross-bind version text, parsed values, policy, and site identity."""

    try:
        old_literal, old_quote = _version_literal(
            preview.old_text,
            "version preview old text",
            preview.representation,
        )
        new_literal, new_quote = _version_literal(
            preview.new_text,
            "version preview new text",
            preview.representation,
        )
        old_version = _stable_version(old_literal, "version preview old text")
        new_version = _stable_version(new_literal, "version preview new text")
        if old_version != preview.current_version:
            raise _preview_digest_failure()
        if new_version != preview.next_version or old_quote != new_quote:
            raise _preview_digest_failure()
        if (
            _next_version(preview.current_version, preview.policy)
            != preview.next_version
        ):
            raise _preview_digest_failure()
        if (
            _render_literal(
                preview.old_text,
                preview.next_version.value,
                preview.representation,
            )
            != preview.new_text
        ):
            raise _preview_digest_failure()
    except VersionPlanningError:
        raise _preview_digest_failure() from None
    except WorkspaceReleaseError:
        raise _preview_digest_failure() from None


def _validate_floor_preview_evidence(preview: FloorPreview) -> None:
    """Cross-bind floor text, normalized values, policy, and site identity."""

    try:
        old_floor, old_quote = _floor_literal(preview.old_text, preview.representation)
        new_floor, new_quote = _floor_literal(preview.new_text, preview.representation)
        old_normalized = old_floor.value if old_floor is not None else ""
        new_normalized = new_floor.value if new_floor is not None else ""
        if (
            old_normalized != preview.old_normalized
            or new_normalized != preview.new_normalized
            or old_quote != new_quote
        ):
            raise _preview_digest_failure()
        if preview.is_noop:
            if preview.new_text != preview.old_text:
                raise _preview_digest_failure()
            return
        if new_floor is None:
            raise _preview_digest_failure()
        if new_floor.operator != _floor_operator(
            preview.policy, preview.representation
        ):
            raise _preview_digest_failure()
        if (
            _render_floor(
                new_floor,
                preview.policy,
                preview.representation,
                old_quote,
            )
            != preview.new_text
        ):
            raise _preview_digest_failure()
    except VersionPlanningError:
        raise _preview_digest_failure() from None
    except WorkspaceReleaseError:
        raise _preview_digest_failure() from None


def _version_plan_payload(
    plan: VersionPlan, *, include_preview_plan_digest: bool = False
) -> dict[str, object]:
    return {
        "graph_digest": plan.graph_digest,
        "selection_digest": plan.selection_digest,
        "next_versions": [
            (package_id, version.value) for package_id, version in plan.next_versions
        ],
        "package_batches": plan.package_batches,
        "version_previews": [
            preview.canonical_payload(include_plan_digest=include_preview_plan_digest)
            for preview in plan.version_previews
        ],
        "floor_previews": [
            preview.canonical_payload(include_plan_digest=include_preview_plan_digest)
            for preview in plan.floor_previews
        ],
    }


def _version_plan_digest(plan: VersionPlan) -> str:
    return hashlib.sha256(
        _canonical_json(_version_plan_payload(plan)).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class VersionPlan:
    """Immutable, digest-bound version and floor preview result."""

    graph_digest: str
    selection_digest: str
    next_versions: tuple[tuple[str, Version], ...]
    package_batches: tuple[tuple[str, ...], ...]
    version_previews: tuple[VersionPreview, ...] = ()
    floor_previews: tuple[FloorPreview, ...] = ()
    plan_digest: str = ""

    def __post_init__(self) -> None:
        graph_digest = _digest(self.graph_digest, "plan graph digest")
        selection_digest = _digest(self.selection_digest, "plan selection digest")
        raw_versions = _bounded_sequence(
            self.next_versions,
            "plan next versions",
            max_items=MAX_PACKAGES,
            exact_builtin=True,
        )
        normalized_versions: list[tuple[str, Version]] = []
        for item in raw_versions:
            if type(item) not in (tuple, list):
                raise _fail(
                    VersionPlanningCode.INVALID_INPUT,
                    "plan next versions must contain package/version pairs",
                )
            pair = cast(tuple[object, object], item)
            if (
                len(pair) != 2
                or type(pair[0]) is not str
                or not isinstance(pair[1], Version)
            ):
                raise _fail(
                    VersionPlanningCode.INVALID_INPUT,
                    "plan next versions must contain package/version pairs",
                )
            package_id = _bounded_text(pair[0], "plan package identity")
            version = _stable_version(pair[1], "plan next version")
            normalized_versions.append((package_id, version))
        normalized_versions.sort(key=lambda item: item[0])
        if not normalized_versions or len(
            {item[0] for item in normalized_versions}
        ) != len(normalized_versions):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "plan next versions must be unique and non-empty",
            )
        version_map = dict(normalized_versions)

        raw_batches = _bounded_sequence(
            self.package_batches,
            "plan package batches",
            max_items=MAX_PACKAGES,
            exact_builtin=True,
        )
        batches: list[tuple[str, ...]] = []
        seen_packages: set[str] = set()
        for raw_batch in raw_batches:
            values = _bounded_sequence(
                raw_batch,
                "plan package batch",
                max_items=MAX_PACKAGES,
                exact_builtin=True,
            )
            if not values or any(type(item) is not str for item in values):
                raise _fail(
                    VersionPlanningCode.INVALID_INPUT,
                    "plan package batches must contain non-empty strings",
                )
            batch = tuple(
                _bounded_text(item, "plan package identity")
                for item in cast(tuple[str, ...], values)
            )
            if (
                batch != tuple(sorted(batch))
                or len(batch) != len(set(batch))
                or set(batch) & seen_packages
            ):
                raise _fail(
                    VersionPlanningCode.INVALID_INPUT,
                    "plan package batches must be canonical and disjoint",
                )
            batches.append(batch)
            seen_packages.update(batch)
        if seen_packages != set(version_map):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "plan package batches must cover every selected package",
            )
        if not batches:
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "plan package batches must not be empty",
            )
        batch_rank = {
            package_id: index
            for index, batch in enumerate(batches)
            for package_id in batch
        }

        versions = _typed_sequence(
            self.version_previews,
            "plan version previews",
            VersionPreview,
            max_items=MAX_VERSION_SITES,
        )
        floors = _typed_sequence(
            self.floor_previews,
            "plan floor previews",
            FloorPreview,
            max_items=MAX_FLOOR_SITES,
        )
        versions = tuple(
            sorted(
                versions,
                key=lambda item: (
                    batch_rank.get(item.package.value, MAX_PACKAGES + 1),
                    item.package.value,
                    item.file_path,
                    item.source_location,
                ),
            )
        )
        floors = tuple(
            sorted(
                floors,
                key=lambda item: (
                    batch_rank.get(item.dependency.value, MAX_PACKAGES + 1),
                    batch_rank.get(item.dependent.value, MAX_PACKAGES + 1),
                    item.dependency.value,
                    item.dependent.value,
                    item.file_path,
                    item.source_location,
                ),
            )
        )
        for version_preview in versions:
            _validate_version_preview_evidence(version_preview)
        for floor_preview in floors:
            _validate_floor_preview_evidence(floor_preview)
        for version_preview in versions:
            if (
                version_preview.graph_digest != graph_digest
                or version_preview.selection_digest != selection_digest
            ):
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "preview digest does not match the enclosing plan",
                )
        for floor_preview in floors:
            if (
                floor_preview.graph_digest != graph_digest
                or floor_preview.selection_digest != selection_digest
            ):
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "preview digest does not match the enclosing plan",
                )
        for version_preview in versions:
            if version_preview.package.value not in version_map:
                raise _fail(
                    VersionPlanningCode.MISSING_PACKAGE,
                    "version preview package is not in the plan",
                )
        for floor_preview in floors:
            if (
                floor_preview.dependent.value not in version_map
                or floor_preview.dependency.value not in version_map
            ):
                raise _fail(
                    VersionPlanningCode.MISSING_PACKAGE,
                    "floor preview package is not in the plan",
                )

        object.__setattr__(self, "graph_digest", graph_digest)
        object.__setattr__(self, "selection_digest", selection_digest)
        object.__setattr__(self, "next_versions", tuple(normalized_versions))
        object.__setattr__(self, "package_batches", tuple(batches))
        object.__setattr__(self, "version_previews", versions)
        object.__setattr__(self, "floor_previews", floors)
        expected = _version_plan_digest(self)
        if not isinstance(self.plan_digest, str):
            raise _fail(
                VersionPlanningCode.DIGEST, "plan digest must be a SHA-256 digest"
            )
        auto_digest = self.plan_digest == _AUTO_PLAN_DIGEST
        if not auto_digest and not self.plan_digest:
            raise _fail(
                VersionPlanningCode.DIGEST,
                "plan digest is required for a frozen plan",
            )
        if not auto_digest:
            supplied = _digest(self.plan_digest, "plan digest")
            if supplied != expected:
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "plan digest does not match frozen contents",
                )
        if (
            any(
                (not version_preview.plan_digest)
                or version_preview.plan_digest != expected
                for version_preview in versions
            )
            or any(
                (not floor_preview.plan_digest) or floor_preview.plan_digest != expected
                for floor_preview in floors
            )
        ) and not auto_digest:
            raise _fail(
                VersionPlanningCode.DIGEST,
                "preview plan digest does not match enclosing contents",
            )
        object.__setattr__(self, "plan_digest", expected)
        object.__setattr__(
            self,
            "version_previews",
            tuple(
                preview
                if preview.plan_digest == expected
                else replace(preview, plan_digest=expected)
                for preview in versions
            ),
        )
        object.__setattr__(
            self,
            "floor_previews",
            tuple(
                preview
                if preview.plan_digest == expected
                else replace(preview, plan_digest=expected)
                for preview in floors
            ),
        )

    def canonical_payload(self, *, include_digest: bool = False) -> dict[str, object]:
        payload = _version_plan_payload(
            self, include_preview_plan_digest=include_digest
        )
        if include_digest:
            payload["plan_digest"] = self.plan_digest
        return payload

    def validate_against(
        self, graph: DependencyGraph, selection: SelectedChangeClosure
    ) -> None:
        """Refuse reuse after graph/selection or preview-evidence drift."""

        try:
            if (
                not isinstance(graph, DependencyGraph)
                or graph.digest != self.graph_digest
            ):
                raise _fail(
                    VersionPlanningCode.GRAPH_DRIFT,
                    "current graph does not match the frozen plan",
                )
            if (
                not isinstance(selection, SelectedChangeClosure)
                or selection.digest != self.selection_digest
            ):
                raise _fail(
                    VersionPlanningCode.SELECTION_DRIFT,
                    "current selection does not match the frozen plan",
                )
            if not self.plan_digest:
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "plan digest is required for validation",
                )
            if any(not preview.plan_digest for preview in self.version_previews) or any(
                not preview.plan_digest for preview in self.floor_previews
            ):
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "preview plan digest is required for validation",
                )
            if _version_plan_digest(self) != self.plan_digest:
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "plan digest does not match frozen contents",
                )
            version_sites = tuple(
                VersionSourceSite(
                    package=preview.package,
                    file_path=preview.file_path,
                    old_text=preview.old_text,
                    policy=preview.policy,
                )
                for preview in self.version_previews
            )
            floor_sites = tuple(
                FloorRewriteSite(
                    dependent=preview.dependent,
                    dependency=preview.dependency,
                    file_path=preview.file_path,
                    source_location=preview.source_location,
                    representation=preview.representation,
                    old_text=preview.old_text,
                    policy=preview.policy,
                )
                for preview in self.floor_previews
            )
            expected = plan_version_floors(
                graph,
                selection,
                version_sites=version_sites,
                floor_sites=floor_sites,
            )
            if self.canonical_payload(
                include_digest=True
            ) != expected.canonical_payload(include_digest=True):
                raise _fail(
                    VersionPlanningCode.DIGEST,
                    "plan evidence does not match recomputed contents",
                )
        except VersionPlanningError:
            raise
        except Exception:
            # Forged construct/model-copy instances and lower-level bounded
            # record failures must not expose implementation or caller data.
            raise _fail(
                VersionPlanningCode.DIGEST,
                "plan evidence could not be validated",
            ) from None


def _site_project_and_package(
    projects: Mapping[str, ProjectRecord],
    packages: Mapping[str, PackageRecord],
    package: PackageKey,
) -> tuple[ProjectRecord, PackageRecord]:
    record = packages.get(package.value)
    if record is None:
        raise _fail(
            VersionPlanningCode.MISSING_PACKAGE,
            f"site package is not frozen: {package.value}",
        )
    project = _project_for_package(projects, package)
    if project.tree_sha == "":
        raise _fail(
            VersionPlanningCode.SOURCE_SHA,
            f"project has no immutable source SHA: {project.project_id}",
        )
    _sha(project.tree_sha, "project source SHA")
    return project, record


def plan_version_floors(
    graph: DependencyGraph,
    selection: SelectedChangeClosure,
    *,
    version_sites: Iterable[VersionSourceSite],
    floor_sites: Iterable[FloorRewriteSite],
) -> VersionPlan:
    """Build a deterministic current-to-next and transitive-floor preview.

    Every changed dependency edge needs one explicit floor site.  Edges whose
    dependency is unchanged may omit a site; supplied sites still produce
    explicit no-op evidence.  All package/version/floor values come from the
    frozen graph, while exact source text comes from the immutable site
    descriptors.
    """

    if not isinstance(graph, DependencyGraph):
        raise _fail(
            VersionPlanningCode.GRAPH_DRIFT,
            "version planning requires a DependencyGraph",
        )
    if not isinstance(selection, SelectedChangeClosure):
        raise _fail(
            VersionPlanningCode.SELECTION_DRIFT,
            "version planning requires a SelectedChangeClosure",
        )
    if (
        graph.digest != selection.source_graph.digest
        or graph.canonical_payload() != selection.source_graph.canonical_payload()
    ):
        raise _fail(
            VersionPlanningCode.GRAPH_DRIFT,
            "graph and selected closure do not share the frozen source graph",
        )

    projects = {project.project_id: project for project in selection.projects}
    packages = {
        package.key.value: package
        for project in selection.projects
        for package in project.packages
    }
    if not packages:
        raise _fail(
            VersionPlanningCode.MISSING_PACKAGE, "selected closure contains no packages"
        )
    package_ids = tuple(sorted(packages))
    batches = _package_batches(package_ids, selection.edges)
    batch_rank = {
        package_id: index for index, batch in enumerate(batches) for package_id in batch
    }

    raw_version_sites = _typed_sequence(
        version_sites, "version sites", VersionSourceSite, max_items=MAX_VERSION_SITES
    )
    version_by_package: dict[str, VersionSourceSite] = {}
    for version_site in raw_version_sites:
        package_id = version_site.package.value
        if package_id in version_by_package:
            prior_version_site = version_by_package[package_id]
            code = (
                VersionPlanningCode.DUPLICATE_SOURCE_SITE
                if prior_version_site.canonical_payload()
                == version_site.canonical_payload()
                else VersionPlanningCode.CONFLICTING_VERSION_SOURCE
            )
            raise _fail(
                code,
                "a package has duplicate or conflicting version source sites",
                details=(("package", package_id),),
            )
        project, package = _site_project_and_package(
            projects, packages, version_site.package
        )
        if not _metadata_file_known(project, package, version_site.file_path):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "version site file is not in frozen metadata files",
            )
        _current_version_literal(
            version_site.old_text,
            _stable_version(package.version, "package version"),
            version_site.policy.representation,
        )
        if not any(
            source.location == version_site.policy.source_location
            for source in package.version_sources
        ):
            raise _fail(
                VersionPlanningCode.CONFLICTING_VERSION_SOURCE,
                "version site source is not frozen for the package",
            )
        version_by_package[package_id] = version_site

    current_versions = {
        package_id: _stable_version(package.version, f"package {package_id} version")
        for package_id, package in packages.items()
    }
    next_versions = dict(current_versions)
    version_previews: list[VersionPreview] = []
    for package_id in sorted(
        version_by_package, key=lambda item: (batch_rank[item], item)
    ):
        version_site = version_by_package[package_id]
        package = packages[package_id]
        project = projects[package_id.split("::", 1)[0]]
        current = current_versions[package_id]
        next_value = _next_version(current, version_site.policy)
        next_versions[package_id] = next_value
        is_noop = current == next_value
        version_reason = (
            VersionPreviewReason.ALREADY_CURRENT
            if is_noop
            else VersionPreviewReason.BUMP
        )
        version_previews.append(
            VersionPreview(
                project_id=project.project_id,
                package=package.key,
                file_path=version_site.file_path,
                source_location=version_site.policy.source_location,
                representation=version_site.policy.representation,
                source_sha=project.tree_sha,
                old_text=version_site.old_text,
                new_text=_render_literal(
                    version_site.old_text,
                    next_value.value,
                    version_site.policy.representation,
                ),
                current_version=current,
                next_version=next_value,
                policy=version_site.policy,
                reason=version_reason,
                witness=(
                    f"package:{package_id}",
                    f"source:{version_site.policy.source_location}",
                ),
                graph_digest=graph.digest,
                selection_digest=selection.digest,
                is_noop=is_noop,
            )
        )

    edge_by_key = {
        (edge.dependent.value, edge.dependency.value): edge for edge in selection.edges
    }
    raw_floor_sites = _typed_sequence(
        floor_sites, "floor sites", FloorRewriteSite, max_items=MAX_FLOOR_SITES
    )
    floor_by_edge: dict[tuple[str, str], FloorRewriteSite] = {}
    for floor_site in raw_floor_sites:
        key = (floor_site.dependent.value, floor_site.dependency.value)
        if key in floor_by_edge:
            prior_floor_site = floor_by_edge[key]
            code = (
                VersionPlanningCode.DUPLICATE_REWRITE_SITE
                if prior_floor_site.canonical_payload()
                == floor_site.canonical_payload()
                else VersionPlanningCode.CONFLICTING_REWRITE_SITE
            )
            raise _fail(
                code,
                "a dependency edge has duplicate or conflicting rewrite sites",
                details=(("edge", f"{key[0]}->{key[1]}"),),
            )
        edge = edge_by_key.get(key)
        if edge is None:
            raise _fail(
                VersionPlanningCode.MISSING_PACKAGE,
                "floor site endpoint is not a frozen dependency edge",
            )
        if floor_site.source_location != edge.source:
            raise _fail(
                VersionPlanningCode.CONFLICTING_REWRITE_SITE,
                "floor site source is not the frozen dependency source",
                details=(
                    ("frozen", edge.source),
                    ("site", floor_site.source_location),
                ),
            )
        project, package = _site_project_and_package(
            projects, packages, floor_site.dependent
        )
        if not _metadata_file_known(project, package, floor_site.file_path):
            raise _fail(
                VersionPlanningCode.INVALID_INPUT,
                "floor site file is not in frozen metadata files",
            )
        observed, _ = _floor_literal(floor_site.old_text, floor_site.representation)
        expected = _stable_floor(edge.floor, "frozen dependency floor")
        if (observed is None) != (expected is None) or (
            observed is not None and observed != expected
        ):
            raise _fail(
                VersionPlanningCode.CONFLICTING_REWRITE_SITE,
                "floor site old text disagrees with the frozen dependency edge",
            )
        # Validate the output matrix now, even when the edge will be a no-op.
        _floor_operator(floor_site.policy, floor_site.representation)
        floor_by_edge[key] = floor_site

    changed_dependency_edges = {
        key
        for key, edge in edge_by_key.items()
        if next_versions[key[1]] != current_versions[key[1]]
    }
    missing_sites = sorted(changed_dependency_edges - set(floor_by_edge))
    if missing_sites:
        raise _fail(
            VersionPlanningCode.MISSING_FLOOR_SITE,
            "a changed dependency has no exact rewrite site",
            details=tuple(
                ("edge", f"{dependent}->{dependency}")
                for dependent, dependency in missing_sites
            ),
        )

    floor_previews: list[FloorPreview] = []
    for key, floor_site in sorted(
        floor_by_edge.items(),
        key=lambda item: (
            batch_rank[item[0][1]],
            batch_rank[item[0][0]],
            item[0][1],
            item[0][0],
            item[1].file_path,
            item[1].source_location,
        ),
    ):
        edge = edge_by_key[key]
        project = projects[floor_site.dependent.repository_id]
        old_normalized = edge.floor.value if edge.floor else ""
        dependency_changed = next_versions[key[1]] != current_versions[key[1]]
        floor_reason: FloorPreviewReason
        witness: tuple[str, ...]
        if not dependency_changed:
            new_normalized = old_normalized
            new_text = floor_site.old_text
            floor_reason = FloorPreviewReason.DEPENDENCY_UNCHANGED
            is_noop = True
            witness = (
                f"edge:{key[0]}->{key[1]}",
                f"dependency-version:{next_versions[key[1]].value}",
                f"frozen-source:{edge.source}",
            )
        else:
            required = next_versions[key[1]]
            old_floor = edge.floor
            already_satisfied = (
                old_floor is not None and old_floor.version.numeric >= required.numeric
            )
            if already_satisfied:
                new_normalized = old_normalized
                new_text = floor_site.old_text
                floor_reason = FloorPreviewReason.ALREADY_SATISFIED
                is_noop = True
            else:
                generated = _new_floor(
                    required, floor_site.policy, floor_site.representation
                )
                new_normalized = generated.value
                _, quote = (
                    _floor_literal(floor_site.old_text, floor_site.representation)
                    if floor_site.old_text
                    else (None, "")
                )
                new_text = _render_floor(
                    generated, floor_site.policy, floor_site.representation, quote
                )
                floor_reason = FloorPreviewReason.TRANSITIVE_MINIMUM
                is_noop = False
            witness = (
                f"edge:{key[0]}->{key[1]}",
                f"dependency-next:{key[1]}={required.value}",
                f"topological-batch:{batch_rank[key[1]]}",
                f"frozen-source:{edge.source}",
            )
        floor_previews.append(
            FloorPreview(
                project_id=project.project_id,
                dependent=floor_site.dependent,
                dependency=floor_site.dependency,
                file_path=floor_site.file_path,
                source_location=floor_site.source_location,
                representation=floor_site.representation,
                source_sha=project.tree_sha,
                old_text=floor_site.old_text,
                new_text=new_text,
                old_normalized=old_normalized,
                new_normalized=new_normalized,
                policy=floor_site.policy,
                reason=floor_reason,
                witness=witness,
                graph_digest=graph.digest,
                selection_digest=selection.digest,
                is_noop=is_noop,
            )
        )

    return VersionPlan(
        graph_digest=graph.digest,
        selection_digest=selection.digest,
        next_versions=tuple(sorted(next_versions.items())),
        package_batches=batches,
        version_previews=tuple(version_previews),
        floor_previews=tuple(floor_previews),
        plan_digest=_AUTO_PLAN_DIGEST,
    )


def plan_version_input(request: VersionPlanningInput) -> VersionPlan:
    """Plan one already-frozen input bundle without adding an effect path."""

    if not isinstance(request, VersionPlanningInput):
        raise _fail(
            VersionPlanningCode.INVALID_INPUT,
            "version planning requires a VersionPlanningInput",
        )
    return plan_version_floors(
        request.graph,
        request.selection,
        version_sites=request.version_sites,
        floor_sites=request.floor_sites,
    )


# A descriptive alias for callers that use the lane's user-facing wording.
plan_workspace_versions = plan_version_floors


__all__ = [
    "FloorPolicy",
    "FloorPreview",
    "FloorPreviewReason",
    "FloorRewriteSite",
    "MetadataRepresentation",
    "VersionBump",
    "VersionPlan",
    "VersionPlanningCode",
    "VersionPlanningError",
    "VersionPlanningInput",
    "VersionPreview",
    "VersionPreviewReason",
    "VersionSourcePolicy",
    "VersionSourceSite",
    "plan_version_floors",
    "plan_version_input",
    "plan_workspace_versions",
]
