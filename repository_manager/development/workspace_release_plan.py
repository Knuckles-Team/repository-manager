"""Pure RMDD-18 checkpoint-4 frozen release plans.

This module is the boundary between the graph/version/floor evidence produced by
checkpoints 1--3 and the later WorkItem/execution lanes.  It intentionally has no
filesystem, subprocess, Git, package-manager, network, or persistence imports.
The only inputs accepted by :func:`freeze_release_plan` are already materialized
checkpoint models and small, bounded builtin values.

The two useful guarantees here are deliberately separate:

* :class:`FrozenReleasePlan` binds the selected graph, repository tree/base and
  generation identities, the complete checkpoint-3 preview, profile digests,
  and a single exact SHA-256 plan digest.
* :class:`StagePreview` is a declarative DAG record.  It has a deterministic
  opaque ID and input digest, but no command, WorkItem, executor, or status.  A
  dependency failure therefore only means ``block_dependents`` in the preview;
  no failure is observed or acted upon here.

Push is represented independently from the ordinary validate/bump/land/build/
package stages.  A push preview can only be materialized when an immutable
:class:`PushConsentReference` is supplied.  A boolean such as ``allow_push`` is
never an authorization or a substitute for that reference.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum
from typing import TypeVar, cast

from .workspace_release import (
    MAX_EDGES,
    MAX_PACKAGES,
    MAX_PLAN_STAGES,
    MAX_PROJECTS,
    MAX_STRING_LENGTH,
    DependencyEdge,
    DependencyGraph,
    PackageKey,
    PackageRecord,
    ProjectRecord,
    Version,
    WorkspaceReleaseError,
    _canonical_json,
)
from .workspace_selection import SelectedChangeClosure
from .workspace_versions import (
    FloorPreview,
    VersionPlan,
    VersionPlanningError,
    VersionPreview,
)

C11_FROZEN_PLAN_VERSION = 1
MAX_PROFILE_BINDINGS = MAX_PROJECTS
MAX_CONSENT_LENGTH = 256
MAX_GENERATION_LENGTH = 256
MAX_STAGE_DEPENDENCIES = MAX_PLAN_STAGES
MAX_DIGEST_LENGTH = 64

_SHA = re.compile(r"^[0-9a-fA-F]{40,64}$")
_DIGEST = re.compile(r"^[0-9a-fA-F]{64}$")
_OPAQUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:-]*$")
_T = TypeVar("_T")


class ReleasePlanCode(StrEnum):
    """Stable, privacy-safe refusal categories for checkpoint 4."""

    INVALID_INPUT = "invalid_input"
    UNBOUNDED_INPUT = "unbounded_input"
    DIGEST = "digest"
    SOURCE_SHA = "source_sha"
    BASE_SHA = "base_sha"
    TREE_SHA = "tree_sha"
    GENERATION_ID = "generation_id"
    GRAPH_DRIFT = "graph_drift"
    SELECTION_DRIFT = "selection_drift"
    VERSION_PLAN_DRIFT = "version_plan_drift"
    IDENTITY = "identity"
    DUPLICATE = "duplicate"
    CONFLICT = "conflict"
    MISSING = "missing"
    AMBIGUOUS = "ambiguous"
    PROFILE = "profile"
    CONSENT = "consent"
    UNKNOWN_STAGE = "unknown_stage"
    STAGE_DEPENDENCY = "stage_dependency"
    CYCLE = "cycle"
    PUSH_CONSENT = "push_consent"


class ReleasePlanError(WorkspaceReleaseError):
    """A bounded CP4 input, frozen model, or declarative DAG was refused."""

    def __init__(
        self,
        code: ReleasePlanCode,
        message: str,
        *,
        details: Iterable[tuple[str, str]] = (),
    ) -> None:
        if not isinstance(code, ReleasePlanCode):
            raise ValueError("release-plan code must be supported")
        self.code = code
        self.details = _bounded_pairs(details, "planning diagnostics")
        # Deliberately do not include caller-controlled values in exception text.
        super().__init__(f"{code.value}: {message}")


# Compatibility names used by callers that describe the same boundary as a
# frozen-plan rather than a release-plan error.
FrozenPlanError = ReleasePlanError
WorkspaceReleasePlanError = ReleasePlanError
FrozenPlanCode = ReleasePlanCode


class StageKind(StrEnum):
    """Declarative stage kinds; these names do not select an executor."""

    VALIDATE = "validate"
    BUMP = "bump"
    LOCAL_LAND = "local-land"
    # ``LAND`` is an intentional wire alias for older phase vocabulary.
    LAND = "local-land"
    BUILD = "build"
    PACKAGE = "package"
    PUSH = "push"


ReleaseStageKind = StageKind
StageType = StageKind


class FailurePolicy(StrEnum):
    """What a later executor must do with a failed upstream stage."""

    BLOCK_DEPENDENTS = "block_dependents"


class ProfileKind(StrEnum):
    VALIDATION = "validation"
    BUILD = "build"


def _fail(code: ReleasePlanCode, message: str) -> ReleasePlanError:
    return ReleasePlanError(code, message)


def _strict_text(
    value: object,
    field_name: str,
    *,
    max_length: int = MAX_STRING_LENGTH,
    code: ReleasePlanCode = ReleasePlanCode.INVALID_INPUT,
) -> str:
    # Exact builtin scalars are required.  This prevents str subclasses with
    # hostile ``strip``/iteration implementations from crossing the boundary.
    if type(value) is not str:
        raise _fail(code, f"{field_name} must be a builtin string")
    if not value or value.strip() != value:
        raise _fail(code, f"{field_name} must be non-blank and trimmed")
    if len(value) > max_length:
        raise _fail(ReleasePlanCode.UNBOUNDED_INPUT, f"{field_name} exceeds the bound")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise _fail(code, f"{field_name} contains a control character")
    return value


def _strict_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:
        raise _fail(ReleasePlanCode.INVALID_INPUT, f"{field_name} must be boolean")
    return cast(bool, value)


def _strict_sha(
    value: object,
    field_name: str,
    *,
    code: ReleasePlanCode = ReleasePlanCode.DIGEST,
) -> str:
    text = _strict_text(value, field_name, max_length=MAX_DIGEST_LENGTH, code=code)
    if _SHA.fullmatch(text) is None:
        raise _fail(code, f"{field_name} must be a hexadecimal Git SHA")
    return text.lower()


def _strict_digest(
    value: object,
    field_name: str,
    *,
    code: ReleasePlanCode = ReleasePlanCode.DIGEST,
) -> str:
    text = _strict_text(value, field_name, max_length=MAX_DIGEST_LENGTH, code=code)
    if _DIGEST.fullmatch(text) is None:
        raise _fail(code, f"{field_name} must be a SHA-256 digest")
    return text.lower()


def _strict_opaque(value: object, field_name: str, *, max_length: int) -> str:
    text = _strict_text(
        value,
        field_name,
        max_length=max_length,
        code=ReleasePlanCode.GENERATION_ID,
    )
    if _OPAQUE.fullmatch(text) is None or "://" in text:
        raise _fail(ReleasePlanCode.GENERATION_ID, f"{field_name} is not an opaque ID")
    return text


def _bounded_tuple(
    value: object,
    field_name: str,
    *,
    max_items: int,
    item_type: type[_T] | None = None,
) -> tuple[_T, ...] | tuple[object, ...]:
    """Copy a builtin tuple/list while bounding iteration and normalizing errors."""

    if type(value) not in (tuple, list):
        raise _fail(
            ReleasePlanCode.INVALID_INPUT, f"{field_name} must be a tuple or list"
        )
    raw = cast(tuple[object, ...] | list[object], value)
    if len(raw) > max_items:
        raise _fail(ReleasePlanCode.UNBOUNDED_INPUT, f"{field_name} exceeds the bound")
    values = tuple(raw)
    if item_type is not None and any(type(item) is not item_type for item in values):
        raise _fail(ReleasePlanCode.INVALID_INPUT, f"{field_name} has invalid items")
    return cast(tuple[_T, ...], values)


def _bounded_pairs(
    value: object,
    field_name: str,
    *,
    max_items: int = 32,
) -> tuple[tuple[str, str], ...]:
    values = _bounded_tuple(value, field_name, max_items=max_items)
    pairs: list[tuple[str, str]] = []
    for item in values:
        if type(item) not in (tuple, list):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, f"{field_name} has invalid pairs"
            )
        pair = cast(tuple[object, ...] | list[object], item)
        if len(pair) != 2:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, f"{field_name} has invalid pairs"
            )
        key, val = pair
        pairs.append(
            (
                _strict_text(key, f"{field_name} key", max_length=128),
                _strict_text(val, f"{field_name} value"),
            )
        )
    return tuple(sorted(pairs))


def _strict_mapping(value: object, field_name: str) -> dict[object, object]:
    if type(value) is not dict:
        raise _fail(
            ReleasePlanCode.INVALID_INPUT, f"{field_name} must be a builtin mapping"
        )
    raw = cast(dict[object, object], value)
    if len(raw) > MAX_PROFILE_BINDINGS:
        raise _fail(ReleasePlanCode.UNBOUNDED_INPUT, f"{field_name} exceeds the bound")
    # Iterating a builtin dict is safe after the exact-type check; keys are
    # still checked before any caller value can enter an error message.
    return dict(raw)


def _canonical(value: object) -> str:
    """Canonical JSON for already-validated builtin/enum values."""

    try:
        return _canonical_json(value)
    except Exception:
        # A malformed/forged nested object must never expose its repr/details.
        raise _fail(
            ReleasePlanCode.DIGEST, "canonical payload could not be encoded"
        ) from None


def _digest_payload(value: object) -> str:
    try:
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(ReleasePlanCode.DIGEST, "digest could not be computed") from None


def _canonical_repository(value: object) -> str:
    if type(value) is not str:
        raise _fail(ReleasePlanCode.IDENTITY, "repository identity must be canonical")
    try:
        # Import lazily only as a pure function; this avoids duplicating C-11
        # identity rules and keeps the new module additive.
        from .workspace_release import canonical_repository_id

        return canonical_repository_id(value)
    except Exception:
        raise _fail(
            ReleasePlanCode.IDENTITY, "repository identity is invalid"
        ) from None


def _canonical_project_ids(value: object, field_name: str) -> tuple[str, ...]:
    raw = _bounded_tuple(value, field_name, max_items=MAX_PROJECTS)
    result = tuple(_canonical_repository(item) for item in raw)
    if not result or tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise _fail(
            ReleasePlanCode.IDENTITY, f"{field_name} must be unique and ordered"
        )
    return result


def _canonical_dependencies(value: object, field_name: str) -> tuple[str, ...]:
    raw = _bounded_tuple(value, field_name, max_items=MAX_STAGE_DEPENDENCIES)
    result = tuple(_strict_text(item, f"{field_name} entry") for item in raw)
    if tuple(sorted(result)) != result or len(set(result)) != len(result):
        raise _fail(
            ReleasePlanCode.STAGE_DEPENDENCY, f"{field_name} must be ordered and unique"
        )
    return result


def _project_payload(project: ProjectRecord) -> dict[str, object]:
    """Return selected project evidence without retaining caller containers."""

    packages: list[dict[str, object]] = []
    for package in project.packages:
        dependencies = []
        for dependency in package.dependencies:
            dependencies.append(
                {
                    "target": {
                        "ecosystem": dependency.target.ecosystem.value,
                        "name": dependency.target.name,
                        "repository_id": dependency.target.repository_id,
                    },
                    "floor": dependency.floor.value if dependency.floor else None,
                    "source": dependency.source,
                }
            )
        packages.append(
            {
                "key": package.key.value,
                "version": package.version.value,
                "version_sources": tuple(
                    (source.location, source.version.value)
                    for source in package.version_sources
                ),
                "dependencies": tuple(dependencies),
                "metadata_files": package.metadata_files,
            }
        )
    return {
        "repository_id": project.project_id,
        "tree_sha": project.tree_sha,
        "packages": tuple(packages),
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


def _preview_payload(preview: VersionPreview | FloorPreview) -> dict[str, object]:
    if isinstance(preview, VersionPreview):
        return preview.canonical_payload(include_plan_digest=True)
    return preview.canonical_payload(include_plan_digest=True)


def _preview_digest(preview: VersionPreview | FloorPreview) -> str:
    return _digest_payload(_preview_payload(preview))


@dataclass(frozen=True, slots=True)
class ProfileBinding:
    """One project-scoped immutable profile name and digest."""

    project_id: str
    name: str
    digest: str
    kind: ProfileKind

    def __post_init__(self) -> None:
        project = _canonical_repository(self.project_id)
        name = _strict_text(
            self.name, "profile name", max_length=256, code=ReleasePlanCode.PROFILE
        )
        if "/" in name or "\\" in name or "://" in name:
            raise _fail(
                ReleasePlanCode.PROFILE, "profile name must not be a path or URL"
            )
        if not isinstance(self.kind, ProfileKind):
            raise _fail(ReleasePlanCode.PROFILE, "profile kind is unsupported")
        digest = _strict_digest(
            self.digest, "profile digest", code=ReleasePlanCode.PROFILE
        )
        object.__setattr__(self, "project_id", project)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "digest", digest)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "name": self.name,
            "digest": self.digest,
            "kind": self.kind.value,
        }


ValidationProfileBinding = ProfileBinding
BuildProfileBinding = ProfileBinding


@dataclass(frozen=True, slots=True)
class ValidationProfile:
    """Convenience immutable profile descriptor accepted by the planner."""

    name: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _strict_text(
                self.name,
                "validation profile name",
                max_length=256,
                code=ReleasePlanCode.PROFILE,
            ),
        )
        object.__setattr__(
            self,
            "digest",
            _strict_digest(
                self.digest, "validation profile digest", code=ReleasePlanCode.PROFILE
            ),
        )


@dataclass(frozen=True, slots=True)
class BuildProfile:
    """Convenience immutable build profile descriptor accepted by the planner."""

    name: str
    digest: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "name",
            _strict_text(
                self.name,
                "build profile name",
                max_length=256,
                code=ReleasePlanCode.PROFILE,
            ),
        )
        object.__setattr__(
            self,
            "digest",
            _strict_digest(
                self.digest, "build profile digest", code=ReleasePlanCode.PROFILE
            ),
        )


@dataclass(frozen=True, slots=True)
class PushConsentReference:
    """Explicit immutable consent evidence required for push previews."""

    reference: str
    digest: str
    scope: str = "workspace-release-push"

    def __post_init__(self) -> None:
        reference = _strict_opaque(
            self.reference, "consent reference", max_length=MAX_CONSENT_LENGTH
        )
        digest = _strict_digest(
            self.digest, "consent digest", code=ReleasePlanCode.CONSENT
        )
        scope = _strict_opaque(self.scope, "consent scope", max_length=128)
        object.__setattr__(self, "reference", reference)
        object.__setattr__(self, "digest", digest)
        object.__setattr__(self, "scope", scope)

    @property
    def immutable_reference(self) -> str:
        """Stable reference token; no mutable/boolean authorization is exposed."""

        return self.reference

    def canonical_payload(self) -> dict[str, object]:
        return {
            "reference": self.reference,
            "digest": self.digest,
            "scope": self.scope,
        }


ConsentReference = PushConsentReference
PushConsent = PushConsentReference


def _profile_digest_from_object(value: object, kind: ProfileKind) -> tuple[str, str]:
    """Normalize only a bounded profile descriptor; never retain it."""

    if isinstance(value, (ValidationProfile, BuildProfile)):
        if kind is ProfileKind.VALIDATION and not isinstance(value, ValidationProfile):
            raise _fail(
                ReleasePlanCode.PROFILE, "profile kind does not match descriptor"
            )
        if kind is ProfileKind.BUILD and not isinstance(value, BuildProfile):
            raise _fail(
                ReleasePlanCode.PROFILE, "profile kind does not match descriptor"
            )
        return value.name, value.digest
    if type(value) is str:
        name = _strict_text(
            value, "profile name", max_length=256, code=ReleasePlanCode.PROFILE
        )
        if "/" in name or "\\" in name or "://" in name:
            raise _fail(
                ReleasePlanCode.PROFILE, "profile name must not be a path or URL"
            )
        if _DIGEST.fullmatch(name):
            return "declared", name.lower()
        return name, _digest_payload({"profile_kind": kind.value, "profile_name": name})
    if type(value) is dict:
        mapping = _strict_mapping(value, "profile descriptor")
        if set(mapping) - {"name", "digest"}:
            raise _fail(
                ReleasePlanCode.PROFILE, "profile descriptor has unsupported fields"
            )
        if "name" not in mapping or "digest" not in mapping:
            raise _fail(ReleasePlanCode.PROFILE, "profile descriptor is incomplete")
        name = _strict_text(
            mapping["name"],
            "profile name",
            max_length=256,
            code=ReleasePlanCode.PROFILE,
        )
        digest = _strict_digest(
            mapping["digest"], "profile digest", code=ReleasePlanCode.PROFILE
        )
        return name, digest
    # Existing profile types from RMDD-11/10 are accepted through their two
    # explicit public fields only.  Attribute access is guarded and any
    # exception becomes a fixed privacy-safe refusal.
    try:
        name_value = value.name  # type: ignore[attr-defined]
        try:
            digest_value = value.config_digest  # type: ignore[attr-defined]
        except AttributeError:
            digest_value = value.digest  # type: ignore[attr-defined]
        if digest_value is None:
            digest_value = value.digest  # type: ignore[attr-defined]
        name = _strict_text(
            name_value, "profile name", max_length=256, code=ReleasePlanCode.PROFILE
        )
        digest = _strict_digest(
            digest_value, "profile digest", code=ReleasePlanCode.PROFILE
        )
        return name, digest
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(
            ReleasePlanCode.PROFILE, "profile descriptor is unsupported"
        ) from None


def _normalize_profiles(
    value: object,
    selected: tuple[str, ...],
    kind: ProfileKind,
    *,
    global_value: object | None = None,
) -> tuple[ProfileBinding, ...]:
    """Expand global or per-project declarations into canonical bindings."""

    selected_set = set(selected)
    entries: dict[str, object] = {}
    if value is not None:
        if type(value) is dict:
            raw = _strict_mapping(value, f"{kind.value} profiles")
            for key, item in raw.items():
                project = _canonical_repository(key)
                if project in entries:
                    raise _fail(
                        ReleasePlanCode.DUPLICATE, "profile project appears twice"
                    )
                entries[project] = item
        elif type(value) in (tuple, list):
            pairs = _bounded_tuple(
                value, f"{kind.value} profile bindings", max_items=MAX_PROFILE_BINDINGS
            )
            for pair in pairs:
                if type(pair) is ProfileBinding:
                    if pair.kind is not kind:
                        raise _fail(
                            ReleasePlanCode.PROFILE,
                            "profile binding kind does not match declaration",
                        )
                    project = _canonical_repository(pair.project_id)
                    if project in entries:
                        raise _fail(
                            ReleasePlanCode.DUPLICATE, "profile project appears twice"
                        )
                    entries[project] = pair
                    continue
                if type(pair) not in (tuple, list):
                    raise _fail(
                        ReleasePlanCode.PROFILE, "profile bindings must contain pairs"
                    )
                pair_values = cast(tuple[object, ...] | list[object], pair)
                if len(pair_values) != 2:
                    raise _fail(
                        ReleasePlanCode.PROFILE, "profile bindings must contain pairs"
                    )
                project = _canonical_repository(pair_values[0])
                if project in entries:
                    raise _fail(
                        ReleasePlanCode.DUPLICATE, "profile project appears twice"
                    )
                entries[project] = pair_values[1]
        else:
            global_value = value
    if global_value is not None:
        if entries:
            raise _fail(
                ReleasePlanCode.CONFLICT, "global and per-project profiles conflict"
            )
        entries = {project: global_value for project in selected}
    if not entries:
        default_name = (
            "default-validation" if kind is ProfileKind.VALIDATION else "default-build"
        )
        default_digest = _digest_payload(
            {"profile_kind": kind.value, "profile_name": default_name}
        )
        entries = {
            project: ValidationProfile(default_name, default_digest)
            if kind is ProfileKind.VALIDATION
            else BuildProfile(default_name, default_digest)
            for project in selected
        }
    if set(entries) != selected_set:
        raise _fail(
            ReleasePlanCode.IDENTITY,
            "profiles must cover exactly the selected projects",
        )
    bindings: list[ProfileBinding] = []
    for project in selected:
        name, digest = _profile_digest_from_object(entries[project], kind)
        bindings.append(ProfileBinding(project, name, digest, kind))
    return tuple(bindings)


def _profile_map(
    bindings: tuple[ProfileBinding, ...], kind: ProfileKind
) -> dict[str, ProfileBinding]:
    result = {
        binding.project_id: binding for binding in bindings if binding.kind is kind
    }
    if len(result) != len(bindings):
        raise _fail(ReleasePlanCode.PROFILE, "profile binding kind is inconsistent")
    return result


def _project_map(projects: tuple[ProjectRecord, ...]) -> dict[str, ProjectRecord]:
    result: dict[str, ProjectRecord] = {}
    for project in projects:
        if type(project) is not ProjectRecord:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan projects must be C-11 records"
            )
        if project.project_id in result:
            raise _fail(
                ReleasePlanCode.DUPLICATE, "plan project identity is duplicated"
            )
        if not project.tree_sha:
            raise _fail(
                ReleasePlanCode.TREE_SHA, "selected project has no immutable tree SHA"
            )
        _strict_sha(project.tree_sha, "project tree SHA", code=ReleasePlanCode.TREE_SHA)
        result[project.project_id] = project
    return result


def _package_map(projects: tuple[ProjectRecord, ...]) -> dict[str, PackageKey]:
    result: dict[str, PackageKey] = {}
    for project in projects:
        for package in project.packages:
            if type(package) is not PackageRecord:
                raise _fail(
                    ReleasePlanCode.INVALID_INPUT, "plan packages must be C-11 records"
                )
            if package.key.value in result:
                raise _fail(
                    ReleasePlanCode.DUPLICATE, "plan package identity is duplicated"
                )
            result[package.key.value] = package.key
    return result


def _project_edges(edges: tuple[DependencyEdge, ...]) -> tuple[tuple[str, str], ...]:
    values: set[tuple[str, str]] = set()
    for edge in edges:
        pair = (edge.dependent_project_id, edge.dependency_project_id)
        if pair[0] != pair[1]:
            values.add(pair)
    return tuple(sorted(values))


def _topological_groups(
    projects: tuple[str, ...],
    edges: tuple[tuple[str, str], ...],
) -> tuple[tuple[tuple[str, ...], ...], tuple[str, ...]]:
    """Return dependency-first project groups or a stable cycle witness."""

    remaining = set(projects)
    dependencies: dict[str, set[str]] = {project: set() for project in projects}
    for dependent, dependency in edges:
        if dependent not in remaining or dependency not in remaining:
            raise _fail(
                ReleasePlanCode.MISSING, "project edge names an unknown project"
            )
        if dependent == dependency:
            raise _fail(
                ReleasePlanCode.CYCLE, "project dependency graph contains a cycle"
            )
        dependencies[dependent].add(dependency)
    groups: list[tuple[str, ...]] = []
    while remaining:
        ready = tuple(
            sorted(project for project in remaining if not dependencies[project])
        )
        if not ready:
            # Do not include arbitrary caller data in the refusal.  A fixed code
            # is sufficient for automation and keeps diagnostics privacy-safe.
            raise _fail(
                ReleasePlanCode.CYCLE, "project dependency graph contains a cycle"
            )
        groups.append(ready)
        for project in ready:
            remaining.remove(project)
        for project in remaining:
            dependencies[project].difference_update(ready)
    return tuple(groups), ()


def _stage_payload_without_digests(
    *,
    kind: StageKind,
    project_id: str,
    base_sha: str,
    tree_sha: str,
    generation_id: str = "",
    graph_digest: str,
    selection_digest: str,
    version_plan_digest: str,
    version_preview_digests: tuple[str, ...],
    floor_preview_digests: tuple[str, ...],
    validation_profile_digest: str,
    build_profile_digest: str,
    depends_on: tuple[str, ...],
    consent_reference: PushConsentReference | None,
) -> dict[str, object]:
    return {
        "kind": kind.value,
        "project_id": project_id,
        "base_sha": base_sha,
        "tree_sha": tree_sha,
        "generation_id": generation_id,
        "graph_digest": graph_digest,
        "selection_digest": selection_digest,
        "version_plan_digest": version_plan_digest,
        "version_preview_digests": version_preview_digests,
        "floor_preview_digests": floor_preview_digests,
        "validation_profile_digest": validation_profile_digest,
        "build_profile_digest": build_profile_digest,
        "depends_on": depends_on,
        "consent_reference": consent_reference.canonical_payload()
        if consent_reference
        else None,
        "failure_policy": FailurePolicy.BLOCK_DEPENDENTS.value,
    }


def _stage_identity(payload: dict[str, object]) -> tuple[str, str]:
    input_digest = _digest_payload(payload)
    stage_id = (
        "stage:"
        + str(payload["kind"])
        + ":"
        + _digest_payload({"input_digest": input_digest, "kind": payload["kind"]})
    )
    return stage_id, input_digest


@dataclass(frozen=True, slots=True)
class StagePreview:
    """One deterministic, dependency-linked stage declaration."""

    stage_id: str
    kind: StageKind
    project_id: str
    base_sha: str
    tree_sha: str
    generation_id: str
    graph_digest: str
    selection_digest: str
    version_plan_digest: str
    version_preview_digests: tuple[str, ...] = ()
    floor_preview_digests: tuple[str, ...] = ()
    validation_profile_digest: str = ""
    build_profile_digest: str = ""
    depends_on: tuple[str, ...] = ()
    consent_reference: PushConsentReference | None = None
    failure_policy: FailurePolicy = FailurePolicy.BLOCK_DEPENDENTS
    input_digest: str = ""

    def __post_init__(self) -> None:
        if type(self.stage_id) is not str or not self.stage_id:
            raise _fail(ReleasePlanCode.DIGEST, "stage ID is required")
        stage_id = _strict_text(
            self.stage_id, "stage ID", max_length=256, code=ReleasePlanCode.DIGEST
        )
        if not isinstance(self.kind, StageKind):
            raise _fail(ReleasePlanCode.UNKNOWN_STAGE, "stage kind is unsupported")
        project_id = _canonical_repository(self.project_id)
        base_sha = _strict_sha(
            self.base_sha, "stage base SHA", code=ReleasePlanCode.BASE_SHA
        )
        tree_sha = _strict_sha(
            self.tree_sha, "stage tree SHA", code=ReleasePlanCode.TREE_SHA
        )
        generation_id = _strict_opaque(
            self.generation_id, "stage generation ID", max_length=MAX_GENERATION_LENGTH
        )
        graph_digest = _strict_digest(self.graph_digest, "stage graph digest")
        selection_digest = _strict_digest(
            self.selection_digest, "stage selection digest"
        )
        version_plan_digest = _strict_digest(
            self.version_plan_digest, "stage version-plan digest"
        )
        raw_version_digests = _bounded_tuple(
            self.version_preview_digests,
            "stage version preview digests",
            max_items=MAX_PACKAGES,
        )
        raw_floor_digests = _bounded_tuple(
            self.floor_preview_digests,
            "stage floor preview digests",
            max_items=MAX_EDGES,
        )
        version_digests: tuple[str, ...] = tuple(
            _strict_digest(item, "stage version preview digest")
            for item in raw_version_digests
        )
        floor_digests: tuple[str, ...] = tuple(
            _strict_digest(item, "stage floor preview digest")
            for item in raw_floor_digests
        )
        if tuple(sorted(version_digests)) != version_digests or len(
            set(version_digests)
        ) != len(version_digests):
            raise _fail(
                ReleasePlanCode.DIGEST,
                "stage version preview digests must be ordered and unique",
            )
        if tuple(sorted(floor_digests)) != floor_digests or len(
            set(floor_digests)
        ) != len(floor_digests):
            raise _fail(
                ReleasePlanCode.DIGEST,
                "stage floor preview digests must be ordered and unique",
            )
        validation_digest = (
            ""
            if not self.validation_profile_digest
            else _strict_digest(
                self.validation_profile_digest,
                "stage validation profile digest",
                code=ReleasePlanCode.PROFILE,
            )
        )
        build_digest = (
            ""
            if not self.build_profile_digest
            else _strict_digest(
                self.build_profile_digest,
                "stage build profile digest",
                code=ReleasePlanCode.PROFILE,
            )
        )
        dependencies = _canonical_dependencies(self.depends_on, "stage dependencies")
        if (
            self.consent_reference is not None
            and type(self.consent_reference) is not PushConsentReference
        ):
            raise _fail(
                ReleasePlanCode.CONSENT, "stage consent must be an immutable reference"
            )
        if self.kind is StageKind.PUSH and self.consent_reference is None:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push stage requires immutable consent"
            )
        if not isinstance(self.failure_policy, FailurePolicy):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "stage failure policy is unsupported"
            )
        if self.failure_policy is not FailurePolicy.BLOCK_DEPENDENTS:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "stage failure policy is unsupported"
            )
        input_digest = _strict_digest(self.input_digest, "stage input digest")
        payload = _stage_payload_without_digests(
            kind=self.kind,
            project_id=project_id,
            base_sha=base_sha,
            tree_sha=tree_sha,
            generation_id=generation_id,
            graph_digest=graph_digest,
            selection_digest=selection_digest,
            version_plan_digest=version_plan_digest,
            version_preview_digests=version_digests,
            floor_preview_digests=floor_digests,
            validation_profile_digest=validation_digest,
            build_profile_digest=build_digest,
            depends_on=dependencies,
            consent_reference=self.consent_reference,
        )
        expected_id, expected_input = _stage_identity(payload)
        if stage_id != expected_id or input_digest != expected_input:
            raise _fail(
                ReleasePlanCode.DIGEST,
                "stage identity or input digest does not match frozen contents",
            )
        object.__setattr__(self, "stage_id", stage_id)
        object.__setattr__(self, "project_id", project_id)
        object.__setattr__(self, "base_sha", base_sha)
        object.__setattr__(self, "tree_sha", tree_sha)
        object.__setattr__(self, "generation_id", generation_id)
        object.__setattr__(self, "graph_digest", graph_digest)
        object.__setattr__(self, "selection_digest", selection_digest)
        object.__setattr__(self, "version_plan_digest", version_plan_digest)
        object.__setattr__(self, "version_preview_digests", version_digests)
        object.__setattr__(self, "floor_preview_digests", floor_digests)
        object.__setattr__(self, "validation_profile_digest", validation_digest)
        object.__setattr__(self, "build_profile_digest", build_digest)
        object.__setattr__(self, "depends_on", dependencies)
        object.__setattr__(self, "input_digest", input_digest)

    @property
    def stage(self) -> StageKind:
        """Compatibility alias for callers using ``stage`` vocabulary."""

        return self.kind

    @property
    def requires_consent(self) -> bool:
        """Whether this preview is consent-gated (derived, never authority)."""

        return self.kind is StageKind.PUSH

    @property
    def blocked_by(self) -> tuple[str, ...]:
        """Declarative upstream block set; no status or execution is consulted."""

        return self.depends_on

    @property
    def dependencies(self) -> tuple[str, ...]:
        return self.depends_on

    def canonical_payload(self, *, include_digests: bool = True) -> dict[str, object]:
        payload: dict[str, object] = _stage_payload_without_digests(
            kind=self.kind,
            project_id=self.project_id,
            base_sha=self.base_sha,
            tree_sha=self.tree_sha,
            generation_id=self.generation_id,
            graph_digest=self.graph_digest,
            selection_digest=self.selection_digest,
            version_plan_digest=self.version_plan_digest,
            version_preview_digests=self.version_preview_digests,
            floor_preview_digests=self.floor_preview_digests,
            validation_profile_digest=self.validation_profile_digest,
            build_profile_digest=self.build_profile_digest,
            depends_on=self.depends_on,
            consent_reference=self.consent_reference,
        )
        if include_digests:
            payload["stage_id"] = self.stage_id
            payload["input_digest"] = self.input_digest
        return payload


ReleaseStagePreview = StagePreview
StagePlan = StagePreview
ReleasePlanStage = StagePreview
ReleaseStage = StageKind


def _validate_stage_dag(stages: tuple[StagePreview, ...]) -> None:
    stage_map: dict[str, StagePreview] = {}
    for stage in stages:
        if stage.stage_id in stage_map:
            raise _fail(ReleasePlanCode.DUPLICATE, "stage identity is duplicated")
        stage_map[stage.stage_id] = stage
    for stage in stages:
        if stage.stage_id in stage.depends_on:
            raise _fail(
                ReleasePlanCode.STAGE_DEPENDENCY, "stage cannot depend on itself"
            )
        if any(dependency not in stage_map for dependency in stage.depends_on):
            raise _fail(
                ReleasePlanCode.STAGE_DEPENDENCY, "stage depends on an unknown stage"
            )
    remaining = set(stage_map)
    dependencies = {
        stage_id: set(stage_map[stage_id].depends_on) for stage_id in stage_map
    }
    while remaining:
        ready = tuple(
            sorted(stage_id for stage_id in remaining if not dependencies[stage_id])
        )
        if not ready:
            raise _fail(
                ReleasePlanCode.CYCLE, "stage dependency graph contains a cycle"
            )
        for stage_id in ready:
            remaining.remove(stage_id)
        for stage_id in remaining:
            dependencies[stage_id].difference_update(ready)


def _plan_payload(
    plan: FrozenReleasePlan, *, include_digest: bool = False
) -> dict[str, object]:
    payload: dict[str, object] = {
        "contract_version": plan.contract_version,
        "workspace_id": plan.workspace_id,
        "source_sha": plan.source_sha,
        "base_sha": plan.base_sha,
        "generation_id": plan.generation_id,
        "graph_digest": plan.graph_digest,
        "selection_digest": plan.selection_digest,
        "version_plan_digest": plan.version_plan_digest,
        "selected_projects": plan.selected_projects,
        "projects": tuple(_project_payload(project) for project in plan.projects),
        "edges": tuple(_edge_payload(edge) for edge in plan.edges),
        "parallel_groups": plan.parallel_groups,
        "version_preview_digests": plan.version_preview_digests,
        "floor_preview_digests": plan.floor_preview_digests,
        "version_plan": plan.version_plan.canonical_payload(include_digest=True),
        "validation_profiles": tuple(
            binding.canonical_payload() for binding in plan.validation_profiles
        ),
        "build_profiles": tuple(
            binding.canonical_payload() for binding in plan.build_profiles
        ),
        "stages": tuple(
            stage.canonical_payload(include_digests=True) for stage in plan.stages
        ),
        "push_consent": plan.push_consent.canonical_payload()
        if plan.push_consent
        else None,
    }
    if include_digest:
        payload["plan_digest"] = plan.plan_digest
    return payload


@dataclass(frozen=True, slots=True)
class FrozenReleasePlan:
    """Complete C-11 release preview, frozen and bound to one exact digest."""

    workspace_id: str
    source_sha: str
    base_sha: str
    generation_id: str
    graph_digest: str
    selection_digest: str
    version_plan_digest: str
    selected_projects: tuple[str, ...]
    projects: tuple[ProjectRecord, ...]
    edges: tuple[DependencyEdge, ...]
    parallel_groups: tuple[tuple[str, ...], ...]
    version_plan: VersionPlan
    validation_profiles: tuple[ProfileBinding, ...]
    build_profiles: tuple[ProfileBinding, ...]
    stages: tuple[StagePreview, ...]
    push_consent: PushConsentReference | None = None
    plan_digest: str = ""
    contract_version: int = C11_FROZEN_PLAN_VERSION

    def __post_init__(self) -> None:
        _validate_frozen_plan_fields(self, require_digest=True)

    @property
    def digest(self) -> str:
        return self.plan_digest

    @property
    def push_stages(self) -> tuple[StagePreview, ...]:
        return tuple(stage for stage in self.stages if stage.kind is StageKind.PUSH)

    @property
    def push_enabled(self) -> bool:
        return bool(self.push_stages)

    @property
    def version_preview_digests(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                _preview_digest(preview)
                for preview in self.version_plan.version_previews
            )
        )

    @property
    def floor_preview_digests(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                _preview_digest(preview) for preview in self.version_plan.floor_previews
            )
        )

    @property
    def version_previews(self) -> tuple[VersionPreview, ...]:
        """Checkpoint-3 version previews carried by the frozen plan."""

        return self.version_plan.version_previews

    @property
    def floor_previews(self) -> tuple[FloorPreview, ...]:
        """Checkpoint-3 floor previews carried by the frozen plan."""

        return self.version_plan.floor_previews

    @property
    def next_versions(self) -> tuple[tuple[str, Version], ...]:
        """Canonical package/current-to-next version evidence for consumers."""

        return self.version_plan.next_versions

    @property
    def stage_dag(self) -> tuple[StagePreview, ...]:
        """Compatibility alias for the immutable stage preview sequence."""

        return self.stages

    @property
    def stage_ids(self) -> tuple[str, ...]:
        return tuple(stage.stage_id for stage in self.stages)

    @property
    def stage_input_digests(self) -> tuple[str, ...]:
        return tuple(stage.input_digest for stage in self.stages)

    @property
    def repository_trees(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (project.project_id, project.tree_sha) for project in self.projects
        )

    @property
    def tree_shas(self) -> tuple[tuple[str, str], ...]:
        return self.repository_trees

    def canonical_payload(self, *, include_digest: bool = False) -> dict[str, object]:
        return _plan_payload(self, include_digest=include_digest)

    def validate(self) -> None:
        """Recompute all nested evidence and the exact plan digest.

        This method is intentionally useful on instances created with
        ``object.__new__`` or a Pydantic/dataclass copy bypass.  It does not trust
        the stored digest and it never mutates the object.
        """

        _validate_frozen_plan_fields(self, require_digest=True)

    def validate_against(
        self,
        graph: DependencyGraph,
        selection: SelectedChangeClosure,
        *,
        version_plan: VersionPlan | None = None,
    ) -> None:
        """Refuse reuse when graph, closure, tree, preview, or profile evidence drifts."""

        try:
            if type(graph) is not DependencyGraph or graph.digest != self.graph_digest:
                raise _fail(
                    ReleasePlanCode.GRAPH_DRIFT,
                    "current graph does not match frozen plan",
                )
            if (
                type(selection) is not SelectedChangeClosure
                or selection.digest != self.selection_digest
            ):
                raise _fail(
                    ReleasePlanCode.SELECTION_DRIFT,
                    "current selection does not match frozen plan",
                )
            if graph.canonical_payload() != selection.source_graph.canonical_payload():
                raise _fail(
                    ReleasePlanCode.GRAPH_DRIFT,
                    "current graph evidence does not match frozen selection",
                )
            current_version_plan = version_plan or self.version_plan
            if type(current_version_plan) is not VersionPlan:
                raise _fail(
                    ReleasePlanCode.VERSION_PLAN_DRIFT,
                    "current version plan is unsupported",
                )
            if current_version_plan.plan_digest != self.version_plan_digest:
                raise _fail(
                    ReleasePlanCode.VERSION_PLAN_DRIFT,
                    "current version plan does not match frozen plan",
                )
            current_version_plan.validate_against(graph, selection)
            expected = freeze_release_plan(
                graph,
                selection,
                current_version_plan,
                workspace_id=self.workspace_id,
                source_sha=self.source_sha,
                base_sha=self.base_sha,
                generation_id=self.generation_id,
                validation_profiles=self.validation_profiles,
                build_profiles=self.build_profiles,
                push_consent=self.push_consent,
            )
            if expected.canonical_payload(
                include_digest=True
            ) != self.canonical_payload(include_digest=True):
                raise _fail(
                    ReleasePlanCode.DIGEST,
                    "frozen release plan evidence does not match recomputed contents",
                )
        except ReleasePlanError:
            raise
        except VersionPlanningError:
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT,
                "version plan evidence could not be validated",
            ) from None
        except Exception:
            raise _fail(
                ReleasePlanCode.DIGEST, "frozen release plan could not be validated"
            ) from None

    @classmethod
    def rebuild(
        cls,
        frozen: FrozenReleasePlan,
        graph: DependencyGraph,
        selection: SelectedChangeClosure,
    ) -> FrozenReleasePlan:
        """Rebuild a plan from its frozen fields after validating current inputs."""

        if type(frozen) is not cls:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "frozen plan type is unsupported"
            )
        frozen.validate_against(graph, selection)
        return freeze_release_plan(
            graph,
            selection,
            frozen.version_plan,
            workspace_id=frozen.workspace_id,
            source_sha=frozen.source_sha,
            base_sha=frozen.base_sha,
            generation_id=frozen.generation_id,
            validation_profiles=frozen.validation_profiles,
            build_profiles=frozen.build_profiles,
            push_consent=frozen.push_consent,
        )


FrozenWorkspaceReleasePlan = FrozenReleasePlan
FrozenPlan = FrozenReleasePlan
ReleasePlan = FrozenReleasePlan


def _revalidate_project(project: ProjectRecord) -> ProjectRecord:
    """Re-materialize nested C-11 records so forged ``__new__`` values fail."""

    try:
        if type(project) is not ProjectRecord:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan project is not a C-11 record"
            )
        raw_packages = _bounded_tuple(
            project.packages, "project packages", max_items=MAX_PACKAGES
        )
        packages: list[PackageRecord] = []
        for raw_package in raw_packages:
            if type(raw_package) is not PackageRecord:
                raise _fail(
                    ReleasePlanCode.INVALID_INPUT, "plan package is not a C-11 record"
                )
            packages.append(
                PackageRecord(
                    key=raw_package.key,
                    version=raw_package.version,
                    version_sources=raw_package.version_sources,
                    dependencies=raw_package.dependencies,
                    metadata_files=raw_package.metadata_files,
                )
            )
        return ProjectRecord(
            repository_id=project.repository_id,
            tree_sha=project.tree_sha,
            packages=tuple(packages),
            metadata_files=project.metadata_files,
        )
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(
            ReleasePlanCode.DIGEST, "nested project evidence could not be validated"
        ) from None


def _revalidate_version_plan(version_plan: VersionPlan) -> VersionPlan:
    """Reconstruct CP3 evidence instead of trusting a forged dataclass shell."""

    try:
        if type(version_plan) is not VersionPlan:
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT, "version plan is not a C-11 record"
            )
        return VersionPlan(
            graph_digest=version_plan.graph_digest,
            selection_digest=version_plan.selection_digest,
            next_versions=version_plan.next_versions,
            package_batches=version_plan.package_batches,
            version_previews=version_plan.version_previews,
            floor_previews=version_plan.floor_previews,
            plan_digest=version_plan.plan_digest,
        )
    except ReleasePlanError:
        raise
    except VersionPlanningError:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version plan evidence is invalid"
        ) from None
    except Exception:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT,
            "version plan evidence could not be validated",
        ) from None


def _revalidate_stage(stage: StagePreview) -> StagePreview:
    """Reconstruct a stage to verify ID/input/dependency binding after forgery."""

    try:
        if type(stage) is not StagePreview:
            raise _fail(ReleasePlanCode.INVALID_INPUT, "stage is not a stage preview")
        return StagePreview(
            stage_id=stage.stage_id,
            kind=stage.kind,
            project_id=stage.project_id,
            base_sha=stage.base_sha,
            tree_sha=stage.tree_sha,
            generation_id=stage.generation_id,
            graph_digest=stage.graph_digest,
            selection_digest=stage.selection_digest,
            version_plan_digest=stage.version_plan_digest,
            version_preview_digests=stage.version_preview_digests,
            floor_preview_digests=stage.floor_preview_digests,
            validation_profile_digest=stage.validation_profile_digest,
            build_profile_digest=stage.build_profile_digest,
            depends_on=stage.depends_on,
            consent_reference=stage.consent_reference,
            failure_policy=stage.failure_policy,
            input_digest=stage.input_digest,
        )
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(
            ReleasePlanCode.DIGEST, "stage evidence could not be validated"
        ) from None


def _revalidate_profile(binding: ProfileBinding) -> ProfileBinding:
    try:
        if type(binding) is not ProfileBinding:
            raise _fail(
                ReleasePlanCode.PROFILE, "profile binding is not a frozen record"
            )
        return ProfileBinding(
            project_id=binding.project_id,
            name=binding.name,
            digest=binding.digest,
            kind=binding.kind,
        )
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(
            ReleasePlanCode.PROFILE, "profile binding could not be validated"
        ) from None


def _revalidate_consent(
    consent: PushConsentReference | None,
) -> PushConsentReference | None:
    if consent is None:
        return None
    try:
        if type(consent) is not PushConsentReference:
            raise _fail(
                ReleasePlanCode.CONSENT, "consent is not an immutable reference"
            )
        return PushConsentReference(
            reference=consent.reference,
            digest=consent.digest,
            scope=consent.scope,
        )
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(
            ReleasePlanCode.CONSENT, "consent reference could not be validated"
        ) from None


def _validate_frozen_plan_fields(
    plan: FrozenReleasePlan, *, require_digest: bool
) -> None:
    """Validate a plan without trusting or rewriting any field."""

    try:
        if type(plan) is not FrozenReleasePlan:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "frozen plan type is unsupported"
            )
        workspace_id = _strict_opaque(
            plan.workspace_id, "workspace ID", max_length=MAX_STRING_LENGTH
        )
        source_sha = _strict_sha(
            plan.source_sha, "source SHA", code=ReleasePlanCode.SOURCE_SHA
        )
        base_sha = _strict_sha(plan.base_sha, "base SHA", code=ReleasePlanCode.BASE_SHA)
        generation_id = _strict_opaque(
            plan.generation_id, "generation ID", max_length=MAX_GENERATION_LENGTH
        )
        graph_digest = _strict_digest(plan.graph_digest, "graph digest")
        selection_digest = _strict_digest(plan.selection_digest, "selection digest")
        version_plan_digest = _strict_digest(
            plan.version_plan_digest, "version plan digest"
        )
        if (
            type(plan.contract_version) is not int
            or plan.contract_version != C11_FROZEN_PLAN_VERSION
        ):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "unsupported frozen plan contract"
            )
        selected = _canonical_project_ids(plan.selected_projects, "selected projects")
        projects = _bounded_tuple(
            plan.projects, "plan projects", max_items=MAX_PROJECTS
        )
        if any(type(project) is not ProjectRecord for project in projects):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan projects must be C-11 records"
            )
        project_values = tuple(
            _revalidate_project(project)
            for project in cast(tuple[ProjectRecord, ...], projects)
        )
        project_map = _project_map(project_values)
        if tuple(sorted(project_map)) != selected:
            raise _fail(
                ReleasePlanCode.IDENTITY,
                "selected projects do not match frozen records",
            )
        if tuple(project_map[project_id] for project_id in selected) != project_values:
            raise _fail(ReleasePlanCode.IDENTITY, "frozen projects are not canonical")
        packages = _package_map(project_values)
        edge_values = _bounded_tuple(plan.edges, "plan edges", max_items=MAX_EDGES)
        if any(type(edge) is not DependencyEdge for edge in edge_values):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan edges must be C-11 records"
            )
        edge_tuple = tuple(
            DependencyEdge(
                dependent=edge.dependent,
                dependency=edge.dependency,
                floor=edge.floor,
                source=edge.source,
                confidence=edge.confidence,
            )
            for edge in cast(tuple[DependencyEdge, ...], edge_values)
        )
        if tuple(sorted(edge_tuple, key=lambda edge: edge.value)) != edge_tuple:
            raise _fail(ReleasePlanCode.DIGEST, "frozen edges are not canonical")
        if len({edge.value for edge in edge_tuple}) != len(edge_tuple):
            raise _fail(ReleasePlanCode.DUPLICATE, "frozen edge identity is duplicated")
        for edge in edge_tuple:
            if (
                edge.dependent.value not in packages
                or edge.dependency.value not in packages
            ):
                raise _fail(
                    ReleasePlanCode.MISSING, "frozen edge names an unknown package"
                )
            if {edge.dependent_project_id, edge.dependency_project_id} - set(selected):
                raise _fail(
                    ReleasePlanCode.IDENTITY, "frozen edge names an unselected project"
                )
        project_edges = _project_edges(edge_tuple)
        raw_groups = _bounded_tuple(
            plan.parallel_groups, "parallel groups", max_items=MAX_PROJECTS
        )
        groups: list[tuple[str, ...]] = []
        grouped: list[str] = []
        for raw_group in raw_groups:
            group = _canonical_project_ids(raw_group, "parallel group")
            groups.append(group)
            grouped.extend(group)
        expected_groups, _ = _topological_groups(selected, project_edges)
        if (
            tuple(groups) != expected_groups
            or set(grouped) != set(selected)
            or len(grouped) != len(set(grouped))
        ):
            raise _fail(
                ReleasePlanCode.DIGEST,
                "parallel groups do not match frozen dependency order",
            )
        if (
            type(plan.version_plan) is not VersionPlan
            or plan.version_plan.plan_digest != version_plan_digest
        ):
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT, "version plan digest is not bound"
            )
        version_plan = _revalidate_version_plan(plan.version_plan)
        if (
            version_plan.graph_digest != graph_digest
            or version_plan.selection_digest != selection_digest
        ):
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT,
                "version plan graph or selection is not bound",
            )
        version_digests = tuple(
            sorted(
                _preview_digest(preview) for preview in version_plan.version_previews
            )
        )
        floor_digests = tuple(
            sorted(_preview_digest(preview) for preview in version_plan.floor_previews)
        )
        if (
            plan.version_preview_digests != version_digests
            or plan.floor_preview_digests != floor_digests
        ):
            raise _fail(
                ReleasePlanCode.DIGEST,
                "preview digests do not match frozen version plan",
            )
        validation_values = _bounded_tuple(
            plan.validation_profiles,
            "validation profiles",
            max_items=MAX_PROFILE_BINDINGS,
        )
        build_values = _bounded_tuple(
            plan.build_profiles, "build profiles", max_items=MAX_PROFILE_BINDINGS
        )
        if any(
            type(item) is not ProfileBinding
            for item in validation_values + build_values
        ):
            raise _fail(ReleasePlanCode.PROFILE, "profile bindings are unsupported")
        validation = tuple(
            _revalidate_profile(item)
            for item in cast(tuple[ProfileBinding, ...], validation_values)
        )
        build = tuple(
            _revalidate_profile(item)
            for item in cast(tuple[ProfileBinding, ...], build_values)
        )
        if any(item.kind is not ProfileKind.VALIDATION for item in validation) or any(
            item.kind is not ProfileKind.BUILD for item in build
        ):
            raise _fail(ReleasePlanCode.PROFILE, "profile binding kind is inconsistent")
        if (
            tuple(item.project_id for item in validation) != selected
            or tuple(item.project_id for item in build) != selected
        ):
            raise _fail(
                ReleasePlanCode.PROFILE,
                "profiles must cover selected projects canonically",
            )
        consent = _revalidate_consent(plan.push_consent)
        stage_values = _bounded_tuple(
            plan.stages, "plan stages", max_items=MAX_PLAN_STAGES
        )
        if any(type(item) is not StagePreview for item in stage_values):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan stages must be stage previews"
            )
        stages = tuple(
            _revalidate_stage(stage)
            for stage in cast(tuple[StagePreview, ...], stage_values)
        )
        _validate_stage_dag(stages)
        if any(
            stage.graph_digest != graph_digest
            or stage.selection_digest != selection_digest
            or stage.version_plan_digest != version_plan_digest
            for stage in stages
        ):
            raise _fail(
                ReleasePlanCode.DIGEST, "stage evidence is not bound to the frozen plan"
            )
        if any(stage.project_id not in selected for stage in stages):
            raise _fail(ReleasePlanCode.IDENTITY, "stage names an unselected project")
        if any(
            stage.kind is StageKind.PUSH and stage.consent_reference != consent
            for stage in stages
        ):
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT,
                "push stage consent is not bound to the plan",
            )
        if consent is None and any(stage.kind is StageKind.PUSH for stage in stages):
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push stage requires immutable consent"
            )
        if not stages:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "frozen plan must contain stages"
            )
        expected_digest = _digest_payload(_plan_payload(plan, include_digest=False))
        if require_digest:
            supplied = _strict_digest(plan.plan_digest, "plan digest")
            if supplied != expected_digest:
                raise _fail(
                    ReleasePlanCode.DIGEST, "plan digest does not match frozen contents"
                )
        # The local variables above intentionally force every normalized scalar
        # through validation; no object.__setattr__ is used during verification.
        _ = (workspace_id, source_sha, base_sha, generation_id)
    except ReleasePlanError:
        raise
    except Exception:
        raise _fail(
            ReleasePlanCode.DIGEST, "frozen plan evidence could not be validated"
        ) from None


def _stage_preview(
    *,
    kind: StageKind,
    project: ProjectRecord,
    base_sha: str,
    generation_id: str,
    graph_digest: str,
    selection_digest: str,
    version_plan_digest: str,
    version_preview_digests: tuple[str, ...],
    floor_preview_digests: tuple[str, ...],
    validation_profile_digest: str,
    build_profile_digest: str,
    depends_on: tuple[str, ...],
    consent_reference: PushConsentReference | None,
) -> StagePreview:
    payload = _stage_payload_without_digests(
        kind=kind,
        project_id=project.project_id,
        base_sha=base_sha,
        tree_sha=project.tree_sha,
        generation_id=generation_id,
        graph_digest=graph_digest,
        selection_digest=selection_digest,
        version_plan_digest=version_plan_digest,
        version_preview_digests=version_preview_digests,
        floor_preview_digests=floor_preview_digests,
        validation_profile_digest=validation_profile_digest,
        build_profile_digest=build_profile_digest,
        depends_on=depends_on,
        consent_reference=consent_reference,
    )
    stage_id, input_digest = _stage_identity(payload)
    return StagePreview(
        stage_id=stage_id,
        kind=kind,
        project_id=project.project_id,
        base_sha=base_sha,
        tree_sha=project.tree_sha,
        generation_id=generation_id,
        graph_digest=graph_digest,
        selection_digest=selection_digest,
        version_plan_digest=version_plan_digest,
        version_preview_digests=version_preview_digests,
        floor_preview_digests=floor_preview_digests,
        validation_profile_digest=validation_profile_digest,
        build_profile_digest=build_profile_digest,
        depends_on=depends_on,
        consent_reference=consent_reference,
        input_digest=input_digest,
    )


@dataclass(frozen=True, slots=True)
class FrozenReleasePlanInput:
    """Closed, immutable request bundle for one freeze operation.

    This mirrors checkpoint-3's ``VersionPlanningInput`` so callers can freeze
    a request once and replay it without retaining mutable mapping/list inputs.
    The direct three-argument :func:`freeze_release_plan` form remains available
    for small callers and compatibility adapters.
    """

    graph: DependencyGraph
    selection: SelectedChangeClosure
    version_plan: VersionPlan
    workspace_id: str = "workspace"
    source_sha: str = ""
    base_sha: str = ""
    generation_id: str = ""
    validation_profiles: object | None = None
    build_profiles: object | None = None
    push_consent: PushConsentReference | None = None
    include_push: bool | None = None

    def __post_init__(self) -> None:
        if type(self.graph) is not DependencyGraph:
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT,
                "input graph must be a frozen C-11 model",
            )
        if type(self.selection) is not SelectedChangeClosure:
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "input selection must be a frozen C-11 model",
            )
        if type(self.version_plan) is not VersionPlan:
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT,
                "input version plan must be a frozen C-11 model",
            )
        _strict_opaque(self.workspace_id, "workspace ID", max_length=MAX_STRING_LENGTH)
        source = _strict_sha(
            self.source_sha or self.base_sha,
            "source SHA",
            code=ReleasePlanCode.SOURCE_SHA,
        )
        base = _strict_sha(
            self.base_sha or source,
            "base SHA",
            code=ReleasePlanCode.BASE_SHA,
        )
        object.__setattr__(self, "source_sha", source)
        object.__setattr__(self, "base_sha", base)
        _strict_opaque(
            self.generation_id,
            "generation ID",
            max_length=MAX_GENERATION_LENGTH,
        )
        if self.include_push is not None:
            _strict_bool(self.include_push, "push inclusion flag")
        if (
            self.push_consent is not None
            and type(self.push_consent) is not PushConsentReference
        ):
            raise _fail(
                ReleasePlanCode.CONSENT,
                "input push consent must be immutable evidence",
            )
        # Materialize the two profile collections now, rather than retaining a
        # caller-owned dict/list.  This also makes missing profiles explicit via
        # deterministic defaults.
        selected = tuple(self.selection.selected_project_ids)
        validation = _normalize_profiles(
            self.validation_profiles,
            selected,
            ProfileKind.VALIDATION,
        )
        build = _normalize_profiles(
            self.build_profiles,
            selected,
            ProfileKind.BUILD,
        )
        object.__setattr__(self, "validation_profiles", validation)
        object.__setattr__(self, "build_profiles", build)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "workspace_id": self.workspace_id,
            "source_sha": self.source_sha,
            "base_sha": self.base_sha,
            "generation_id": self.generation_id,
            "graph_digest": self.graph.digest,
            "selection_digest": self.selection.digest,
            "version_plan_digest": self.version_plan.plan_digest,
            "validation_profiles": tuple(
                binding.canonical_payload()
                for binding in cast(
                    tuple[ProfileBinding, ...], self.validation_profiles
                )
            ),
            "build_profiles": tuple(
                binding.canonical_payload()
                for binding in cast(tuple[ProfileBinding, ...], self.build_profiles)
            ),
            "push_consent": self.push_consent.canonical_payload()
            if self.push_consent
            else None,
            "include_push": self.include_push,
        }


ReleasePlanInput = FrozenReleasePlanInput
FrozenPlanInput = FrozenReleasePlanInput
PlanFreezeInput = FrozenReleasePlanInput


def freeze_release_plan(
    graph: DependencyGraph | FrozenReleasePlanInput,
    selection: SelectedChangeClosure | None = None,
    version_plan: VersionPlan | None = None,
    *,
    workspace_id: str = "workspace",
    source_sha: str | None = None,
    base_sha: str | None = None,
    generation_id: str = "",
    validation_profiles: object | None = None,
    build_profiles: object | None = None,
    validation_profile: object | None = None,
    build_profile: object | None = None,
    validation_profile_digest: object | None = None,
    build_profile_digest: object | None = None,
    push_consent: PushConsentReference | None = None,
    consent_reference: PushConsentReference | None = None,
    consent_ref: PushConsentReference | None = None,
    include_push: bool | None = None,
    allow_push: bool | None = None,
) -> FrozenReleasePlan:
    """Freeze a verified graph/version plan and translate its stage DAG.

    ``include_push`` and ``allow_push`` are intentionally not authorities.  If
    either asks for push, an explicit immutable consent reference is mandatory;
    a bare boolean can never create a push preview.
    """

    try:
        if type(graph) is FrozenReleasePlanInput:
            request = graph
            if selection is not None or version_plan is not None:
                raise _fail(
                    ReleasePlanCode.CONFLICT,
                    "input bundle conflicts with positional graph/selection",
                )
            return freeze_release_plan(
                request.graph,
                request.selection,
                request.version_plan,
                workspace_id=request.workspace_id,
                source_sha=request.source_sha,
                base_sha=request.base_sha or request.source_sha,
                generation_id=request.generation_id,
                validation_profiles=request.validation_profiles,
                build_profiles=request.build_profiles,
                push_consent=request.push_consent,
                include_push=request.include_push,
            )
        if selection is None or version_plan is None:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT,
                "graph, selection, and version plan are required",
            )
        if (
            type(graph) is not DependencyGraph
            or type(selection) is not SelectedChangeClosure
        ):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT,
                "graph and selection must be frozen C-11 models",
            )
        if type(version_plan) is not VersionPlan:
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT,
                "version plan must be a frozen C-11 model",
            )
        if (
            graph.digest != selection.source_graph.digest
            or graph.canonical_payload() != selection.source_graph.canonical_payload()
        ):
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT,
                "graph does not match frozen selection source",
            )
        version_plan.validate_against(graph, selection)
        workspace = _strict_opaque(
            workspace_id, "workspace ID", max_length=MAX_STRING_LENGTH
        )
        source = _strict_sha(
            source_sha
            if source_sha is not None
            else (base_sha if base_sha is not None else ""),
            "source SHA",
            code=ReleasePlanCode.SOURCE_SHA,
        )
        base = _strict_sha(
            base_sha if base_sha is not None else source,
            "base SHA",
            code=ReleasePlanCode.BASE_SHA,
        )
        generation = _strict_opaque(
            generation_id, "generation ID", max_length=MAX_GENERATION_LENGTH
        )
        selected = tuple(selection.selected_project_ids)
        projects = tuple(selection.projects)
        project_map = _project_map(projects)
        if (
            tuple(sorted(project_map)) != selected
            or tuple(project_map[item] for item in selected) != projects
        ):
            raise _fail(
                ReleasePlanCode.IDENTITY, "selection projects are not canonical"
            )
        edges = tuple(sorted(selection.edges, key=lambda edge: edge.value))
        packages = _package_map(projects)
        for edge in edges:
            if (
                edge.dependent.value not in packages
                or edge.dependency.value not in packages
            ):
                raise _fail(
                    ReleasePlanCode.MISSING, "selection edge names an unknown package"
                )
        project_edges = _project_edges(edges)
        groups, _ = _topological_groups(selected, project_edges)
        if groups != tuple(selection.parallel_groups):
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "selection groups do not match frozen dependency order",
            )
        if validation_profile is not None and validation_profile_digest is not None:
            raise _fail(
                ReleasePlanCode.CONFLICT,
                "validation profile declarations conflict",
            )
        if build_profile is not None and build_profile_digest is not None:
            raise _fail(
                ReleasePlanCode.CONFLICT,
                "build profile declarations conflict",
            )
        validation_profile = (
            validation_profile
            if validation_profile is not None
            else validation_profile_digest
        )
        build_profile = (
            build_profile if build_profile is not None else build_profile_digest
        )
        validation = _normalize_profiles(
            validation_profiles,
            selected,
            ProfileKind.VALIDATION,
            global_value=validation_profile,
        )
        build = _normalize_profiles(
            build_profiles, selected, ProfileKind.BUILD, global_value=build_profile
        )
        validation_map = _profile_map(validation, ProfileKind.VALIDATION)
        build_map = _profile_map(build, ProfileKind.BUILD)
        consent_candidates = tuple(
            item
            for item in (push_consent, consent_reference, consent_ref)
            if item is not None
        )
        if len(set(consent_candidates)) > 1:
            raise _fail(ReleasePlanCode.CONFLICT, "push consent references conflict")
        consent = consent_candidates[0] if consent_candidates else None
        if consent is not None and type(consent) is not PushConsentReference:
            raise _fail(
                ReleasePlanCode.CONSENT, "push consent must be immutable evidence"
            )
        if include_push is not None and type(include_push) is not bool:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push inclusion flag must be boolean"
            )
        if allow_push is not None and type(allow_push) is not bool:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push authorization flag must be boolean"
            )
        requested_push = (
            bool(include_push)
            if include_push is not None
            else bool(allow_push)
            if allow_push is not None
            else consent is not None
        )
        if requested_push and consent is None:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT,
                "push requires an immutable consent reference",
            )
        if allow_push is True and include_push is False:
            raise _fail(ReleasePlanCode.CONFLICT, "push flags conflict")
        version_digests = tuple(
            sorted(
                _preview_digest(preview) for preview in version_plan.version_previews
            )
        )
        floor_digests = tuple(
            sorted(_preview_digest(preview) for preview in version_plan.floor_previews)
        )
        # Keep stage ordering semantically useful: all dependency-independent
        # validation stages, then bump/land/build/package, followed by optional
        # dependency-linked pushes.  IDs themselves remain opaque content hashes.
        stage_by_key: dict[tuple[StageKind, str], StagePreview] = {}
        project_dependencies: dict[str, set[str]] = {
            project: set() for project in selected
        }
        for dependent, dependency in project_edges:
            project_dependencies[dependent].add(dependency)
        stage_order = (
            StageKind.VALIDATE,
            StageKind.BUMP,
            StageKind.LOCAL_LAND,
            StageKind.BUILD,
            StageKind.PACKAGE,
        )
        for kind in stage_order:
            for group in groups:
                for project_id in group:
                    project = project_map[project_id]
                    dependencies: list[str] = []
                    if kind is StageKind.BUMP:
                        dependencies.append(
                            stage_by_key[(StageKind.VALIDATE, project_id)].stage_id
                        )
                        dependencies.extend(
                            stage_by_key[(StageKind.BUMP, dependency)].stage_id
                            for dependency in sorted(project_dependencies[project_id])
                        )
                    elif kind is StageKind.LOCAL_LAND:
                        dependencies.append(
                            stage_by_key[(StageKind.BUMP, project_id)].stage_id
                        )
                    elif kind is StageKind.BUILD:
                        dependencies.append(
                            stage_by_key[(StageKind.LOCAL_LAND, project_id)].stage_id
                        )
                    elif kind is StageKind.PACKAGE:
                        dependencies.append(
                            stage_by_key[(StageKind.BUILD, project_id)].stage_id
                        )
                    preview = _stage_preview(
                        kind=kind,
                        project=project,
                        base_sha=base,
                        generation_id=generation,
                        graph_digest=graph.digest,
                        selection_digest=selection.digest,
                        version_plan_digest=version_plan.plan_digest,
                        version_preview_digests=version_digests,
                        floor_preview_digests=floor_digests,
                        validation_profile_digest=validation_map[project_id].digest,
                        build_profile_digest=build_map[project_id].digest,
                        depends_on=tuple(sorted(set(dependencies))),
                        consent_reference=None,
                    )
                    stage_by_key[(kind, project_id)] = preview
        if requested_push:
            assert consent is not None
            for group in groups:
                for project_id in group:
                    project = project_map[project_id]
                    dependencies = [
                        stage_by_key[(StageKind.PACKAGE, project_id)].stage_id
                    ]
                    dependencies.extend(
                        stage_by_key[(StageKind.PUSH, dependency)].stage_id
                        for dependency in sorted(project_dependencies[project_id])
                    )
                    stage_by_key[(StageKind.PUSH, project_id)] = _stage_preview(
                        kind=StageKind.PUSH,
                        project=project,
                        base_sha=base,
                        generation_id=generation,
                        graph_digest=graph.digest,
                        selection_digest=selection.digest,
                        version_plan_digest=version_plan.plan_digest,
                        version_preview_digests=version_digests,
                        floor_preview_digests=floor_digests,
                        validation_profile_digest=validation_map[project_id].digest,
                        build_profile_digest=build_map[project_id].digest,
                        depends_on=tuple(sorted(set(dependencies))),
                        consent_reference=consent,
                    )
        stages = tuple(
            stage_by_key[(kind, project)]
            for kind in (*stage_order, *((StageKind.PUSH,) if requested_push else ()))
            for group in groups
            for project in group
        )
        _validate_stage_dag(stages)
        # Constructing the plan itself requires a digest.  Compute from an
        # equivalent object with a temporary impossible digest is avoided by
        # calculating a self-contained preimage payload here.
        preimage = {
            "contract_version": C11_FROZEN_PLAN_VERSION,
            "workspace_id": workspace,
            "source_sha": source,
            "base_sha": base,
            "generation_id": generation,
            "graph_digest": graph.digest,
            "selection_digest": selection.digest,
            "version_plan_digest": version_plan.plan_digest,
            "selected_projects": selected,
            "projects": tuple(_project_payload(project) for project in projects),
            "edges": tuple(_edge_payload(edge) for edge in edges),
            "parallel_groups": groups,
            "version_preview_digests": version_digests,
            "floor_preview_digests": floor_digests,
            "version_plan": version_plan.canonical_payload(include_digest=True),
            "validation_profiles": tuple(
                binding.canonical_payload() for binding in validation
            ),
            "build_profiles": tuple(binding.canonical_payload() for binding in build),
            "stages": tuple(
                stage.canonical_payload(include_digests=True) for stage in stages
            ),
            "push_consent": consent.canonical_payload() if consent else None,
        }
        digest = _digest_payload(preimage)
        return FrozenReleasePlan(
            workspace_id=workspace,
            source_sha=source,
            base_sha=base,
            generation_id=generation,
            graph_digest=graph.digest,
            selection_digest=selection.digest,
            version_plan_digest=version_plan.plan_digest,
            selected_projects=selected,
            projects=projects,
            edges=edges,
            parallel_groups=groups,
            version_plan=version_plan,
            validation_profiles=validation,
            build_profiles=build,
            stages=stages,
            push_consent=consent if requested_push else None,
            plan_digest=digest,
        )
    except ReleasePlanError:
        raise
    except VersionPlanningError:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT,
            "version plan evidence could not be validated",
        ) from None
    except Exception:
        raise _fail(
            ReleasePlanCode.INVALID_INPUT,
            "release plan inputs could not be materialized",
        ) from None


def build_frozen_release_plan(*args: object, **kwargs: object) -> FrozenReleasePlan:
    """Compatibility factory alias for :func:`freeze_release_plan`."""

    return freeze_release_plan(*args, **kwargs)  # type: ignore[arg-type]


def freeze_workspace_release_plan(*args: object, **kwargs: object) -> FrozenReleasePlan:
    return freeze_release_plan(*args, **kwargs)  # type: ignore[arg-type]


def translate_stage_dag(*args: object, **kwargs: object) -> FrozenReleasePlan:
    """Build the frozen plan and return its declarative stage-DAG preview."""

    return freeze_release_plan(*args, **kwargs)  # type: ignore[arg-type]


def build_stage_dag_preview(
    *args: object, **kwargs: object
) -> tuple[StagePreview, ...]:
    return freeze_release_plan(*args, **kwargs).stages  # type: ignore[arg-type]


def validate_frozen_release_plan(
    plan: FrozenReleasePlan,
    graph: DependencyGraph | None = None,
    selection: SelectedChangeClosure | None = None,
) -> None:
    """Validate a frozen instance, optionally against current graph evidence."""

    if type(plan) is not FrozenReleasePlan:
        raise _fail(ReleasePlanCode.INVALID_INPUT, "frozen plan type is unsupported")
    if (graph is None) != (selection is None):
        raise _fail(
            ReleasePlanCode.INVALID_INPUT,
            "graph and selection must be supplied together",
        )
    if graph is None or selection is None:
        plan.validate()
    else:
        plan.validate_against(graph, selection)


def rebuild_frozen_release_plan(
    plan: FrozenReleasePlan,
    graph: DependencyGraph,
    selection: SelectedChangeClosure,
) -> FrozenReleasePlan:
    return FrozenReleasePlan.rebuild(plan, graph, selection)


reconstruct_frozen_release_plan = rebuild_frozen_release_plan
plan_workspace_release = freeze_release_plan
freeze_plan = freeze_release_plan
build_release_plan = freeze_release_plan


def plan_digest(plan: FrozenReleasePlan) -> str:
    """Return the exact digest of a validated frozen-plan preimage."""

    if type(plan) is not FrozenReleasePlan:
        raise _fail(ReleasePlanCode.INVALID_INPUT, "frozen plan type is unsupported")
    _validate_frozen_plan_fields(plan, require_digest=False)
    return _digest_payload(_plan_payload(plan, include_digest=False))


frozen_plan_digest = plan_digest


__all__ = [
    "BuildProfile",
    "BuildProfileBinding",
    "C11_FROZEN_PLAN_VERSION",
    "ConsentReference",
    "FailurePolicy",
    "FrozenPlan",
    "FrozenPlanCode",
    "FrozenPlanError",
    "FrozenPlanInput",
    "FrozenReleasePlan",
    "FrozenReleasePlanInput",
    "FrozenWorkspaceReleasePlan",
    "ProfileBinding",
    "ProfileKind",
    "PushConsent",
    "PushConsentReference",
    "ReleasePlan",
    "ReleasePlanCode",
    "ReleasePlanError",
    "ReleasePlanInput",
    "ReleasePlanStage",
    "ReleaseStage",
    "ReleaseStageKind",
    "ReleaseStagePreview",
    "StageKind",
    "StagePlan",
    "StagePreview",
    "StageType",
    "PlanFreezeInput",
    "ValidationProfile",
    "ValidationProfileBinding",
    "build_frozen_release_plan",
    "build_stage_dag_preview",
    "freeze_release_plan",
    "freeze_plan",
    "freeze_workspace_release_plan",
    "build_release_plan",
    "plan_workspace_release",
    "rebuild_frozen_release_plan",
    "reconstruct_frozen_release_plan",
    "plan_digest",
    "frozen_plan_digest",
    "translate_stage_dag",
    "validate_frozen_release_plan",
]
