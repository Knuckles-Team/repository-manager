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
from dataclasses import dataclass, field
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
    DependencySpec,
    Ecosystem,
    EdgeConfidence,
    PackageKey,
    PackageRecord,
    PackageReference,
    ProjectRecord,
    Version,
    VersionFloor,
    VersionSource,
    WorkspaceReleaseError,
    _canonical_json,
    build_dependency_graph,
)
from .workspace_selection import (
    InclusionMode,
    SelectedChangeClosure,
    SelectionExplanation,
    SelectionPolicy,
    SelectionReason,
)
from .workspace_versions import (
    FloorPolicy,
    FloorPreview,
    FloorPreviewReason,
    FloorRewriteSite,
    MetadataRepresentation,
    VersionBump,
    VersionPlan,
    VersionPlanningError,
    VersionPreview,
    VersionPreviewReason,
    VersionSourcePolicy,
    VersionSourceSite,
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

# Only these failures are treated as malformed/untrusted provider data.  In
# particular, RuntimeError is deliberately absent: an injected RuntimeError
# from a trusted model/helper is a programmer failure and must remain visible.
_UNTRUSTED_DATA_ERRORS = (
    AttributeError,
    IndexError,
    KeyError,
    OverflowError,
    RecursionError,
    TypeError,
    UnicodeError,
    ValueError,
)


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
    except _UNTRUSTED_DATA_ERRORS:
        # A malformed/forged nested object must never expose its repr/details.
        raise _fail(
            ReleasePlanCode.DIGEST, "canonical payload could not be encoded"
        ) from None


def _digest_payload(value: object) -> str:
    try:
        return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(ReleasePlanCode.DIGEST, "digest could not be computed") from None


def _canonical_repository(value: object) -> str:
    if type(value) is not str:
        raise _fail(ReleasePlanCode.IDENTITY, "repository identity must be canonical")
    try:
        # Import lazily only as a pure function; this avoids duplicating C-11
        # identity rules and keeps the new module additive.
        from .workspace_release import canonical_repository_id

        return canonical_repository_id(value)
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.IDENTITY, "repository identity is invalid"
        ) from None


def _canonical_repository_exact(value: object, field_name: str) -> str:
    """Require a CP2/CP3 identity to already use its canonical wire form."""

    normalized = _canonical_repository(value)
    if normalized != value:
        raise _fail(ReleasePlanCode.IDENTITY, f"{field_name} is not canonical")
    return normalized


def _canonical_project_ids(
    value: object, field_name: str, *, allow_empty: bool = False
) -> tuple[str, ...]:
    raw = _bounded_tuple(value, field_name, max_items=MAX_PROJECTS)
    result = tuple(
        _canonical_repository_exact(item, f"{field_name} entry") for item in raw
    )
    if (
        (not result and not allow_empty)
        or tuple(sorted(result)) != result
        or len(set(result)) != len(result)
    ):
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


def _strict_selection_policy(policy: SelectionPolicy) -> SelectionPolicy:
    """Rebuild CP2 policy only after exact scalar/container validation."""

    if type(policy) is not SelectionPolicy:
        raise _fail(ReleasePlanCode.SELECTION_DRIFT, "selection policy is unsupported")
    changed = _canonical_project_ids(
        policy.changed_projects, "changed projects", allow_empty=False
    )
    explicit = _canonical_project_ids(
        policy.selected_projects, "policy selected projects", allow_empty=True
    )
    if type(policy.upstream_mode) is not InclusionMode:
        raise _fail(
            ReleasePlanCode.SELECTION_DRIFT, "upstream selection mode is invalid"
        )
    if type(policy.downstream_mode) is not InclusionMode:
        raise _fail(
            ReleasePlanCode.SELECTION_DRIFT, "downstream selection mode is invalid"
        )
    return SelectionPolicy(
        changed_projects=changed,
        selected_projects=explicit,
        upstream_mode=policy.upstream_mode,
        downstream_mode=policy.downstream_mode,
    )


def _strict_selection_explanation(
    explanation: SelectionExplanation,
) -> SelectionExplanation:
    """Rebuild a CP2 explanation after exact bool/enum/provenance checks."""

    if type(explanation) is not SelectionExplanation:
        raise _fail(ReleasePlanCode.SELECTION_DRIFT, "selection explanation is invalid")
    project_id = _canonical_repository_exact(
        explanation.project_id, "selection explanation project"
    )
    if type(explanation.included) is not bool:
        raise _fail(
            ReleasePlanCode.SELECTION_DRIFT, "selection explanation flag is invalid"
        )
    reasons = _bounded_tuple(
        explanation.reasons, "selection explanation reasons", max_items=5
    )
    if any(type(reason) is not SelectionReason for reason in reasons):
        raise _fail(
            ReleasePlanCode.SELECTION_DRIFT, "selection explanation reason is invalid"
        )
    via_projects = _canonical_project_ids(
        explanation.via_projects,
        "selection explanation witnesses",
        allow_empty=True,
    )
    return SelectionExplanation(
        project_id=project_id,
        included=explanation.included,
        reasons=cast(tuple[SelectionReason, ...], reasons),
        via_projects=via_projects,
    )


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
        if type(self.kind) is not ProfileKind:
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


@dataclass(frozen=True, slots=True)
class ImmutableDigestReference:
    """Opaque, content-addressed decision evidence.

    These references deliberately contain names and digests only.  They are
    labels for later adapters, never commands, paths, credentials, or execution
    authority.
    """

    name: str
    digest: str

    def __post_init__(self) -> None:
        name = _strict_opaque(self.name, "reference name", max_length=256)
        digest = _strict_digest(
            self.digest, "reference digest", code=ReleasePlanCode.DIGEST
        )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "digest", digest)

    def canonical_payload(self) -> dict[str, str]:
        return {"name": self.name, "digest": self.digest}


DecisionReference = ImmutableDigestReference
OpaqueDigestReference = ImmutableDigestReference
ReleaseProfileReference = ImmutableDigestReference
CandidateReference = ImmutableDigestReference
CertificateReference = ImmutableDigestReference
ConfigReference = ImmutableDigestReference
ToolchainReference = ImmutableDigestReference
CommandReference = ImmutableDigestReference
ArtifactContractReference = ImmutableDigestReference
ResourceProfileReference = ImmutableDigestReference


class RetryPolicy(StrEnum):
    NONE = "none"
    FIXED = "fixed"


class TimeoutPolicy(StrEnum):
    NONE = "none"
    FAIL = "fail"


def _default_reference(name: str) -> ImmutableDigestReference:
    return ImmutableDigestReference(
        name=name,
        digest=hashlib.sha256(("rmdd18:" + name).encode("ascii")).hexdigest(),
    )


def _normalize_target_branch(value: object) -> str:
    text = _strict_text(
        value,
        "target branch",
        max_length=256,
        code=ReleasePlanCode.IDENTITY,
    )
    if text == "@":
        raise _fail(ReleasePlanCode.IDENTITY, "target branch is not canonical")
    if text.startswith("refs/heads/"):
        suffix = text[len("refs/heads/") :]
    else:
        suffix = text
    if text.startswith("refs/") and not text.startswith("refs/heads/"):
        raise _fail(ReleasePlanCode.IDENTITY, "target branch is not canonical")
    if (
        not suffix
        or suffix.startswith("/")
        or suffix.endswith("/")
        or suffix.startswith(".")
        or suffix.endswith(".")
        or "//" in suffix
        or ".." in suffix
        or "@{" in suffix
        or any(char in suffix for char in (":", "~", "^", "?", "*", "[", "]", "\\"))
        or "://" in suffix
        or suffix.startswith("/")
        or any(ord(char) <= 0x20 or ord(char) == 0x7F for char in suffix)
    ):
        raise _fail(ReleasePlanCode.IDENTITY, "target branch is not canonical")
    components = suffix.split("/")
    if any(
        not component
        or component in {".", ".."}
        or component.startswith(".")
        or component.endswith(".")
        or component.endswith(".lock")
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.+\-]*", component)
        for component in components
    ):
        raise _fail(ReleasePlanCode.IDENTITY, "target branch is not canonical")
    return "refs/heads/" + "/".join(components)


@dataclass(frozen=True, slots=True)
class ReleaseDecisionContext:
    """All immutable CP4 decisions which influence a release preview."""

    release_profile: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("release-profile:default")
    )
    target_branch: str = "refs/heads/main"
    candidate: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("candidate:default")
    )
    certificate: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("certificate:default")
    )
    config: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("config:default")
    )
    toolchain: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("toolchain:default")
    )
    command: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("command:preview")
    )
    artifact_contract: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("artifact-contract:default")
    )
    resource_profile: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("resource-profile:default")
    )
    retry_policy: RetryPolicy = RetryPolicy.NONE
    retry_count: int = 0
    timeout_policy: TimeoutPolicy = TimeoutPolicy.NONE
    timeout_seconds: int = 0

    def __post_init__(self) -> None:
        fields = (
            ("release profile", "release_profile", self.release_profile),
            ("candidate reference", "candidate", self.candidate),
            ("certificate reference", "certificate", self.certificate),
            ("config reference", "config", self.config),
            ("toolchain reference", "toolchain", self.toolchain),
            ("command reference", "command", self.command),
            (
                "artifact contract reference",
                "artifact_contract",
                self.artifact_contract,
            ),
            ("resource profile reference", "resource_profile", self.resource_profile),
        )
        for field_name, attribute, value in fields:
            if type(value) is not ImmutableDigestReference:
                raise _fail(
                    ReleasePlanCode.PROFILE,
                    f"{field_name} must be an immutable name/digest reference",
                )
            # Re-materialize even exact-type forged dataclass shells.
            normalized = ImmutableDigestReference(value.name, value.digest)
            object.__setattr__(self, attribute, normalized)
        object.__setattr__(
            self, "target_branch", _normalize_target_branch(self.target_branch)
        )
        if type(self.retry_policy) is not RetryPolicy:
            raise _fail(ReleasePlanCode.INVALID_INPUT, "retry policy is unsupported")
        if type(self.timeout_policy) is not TimeoutPolicy:
            raise _fail(ReleasePlanCode.INVALID_INPUT, "timeout policy is unsupported")
        if type(self.retry_count) is not int or isinstance(self.retry_count, bool):
            raise _fail(ReleasePlanCode.INVALID_INPUT, "retry count must be an integer")
        if not 0 <= self.retry_count <= 32:
            raise _fail(
                ReleasePlanCode.UNBOUNDED_INPUT, "retry count exceeds the bound"
            )
        if type(self.timeout_seconds) is not int or isinstance(
            self.timeout_seconds, bool
        ):
            raise _fail(ReleasePlanCode.INVALID_INPUT, "timeout must be an integer")
        if not 0 <= self.timeout_seconds <= 604800:
            raise _fail(ReleasePlanCode.UNBOUNDED_INPUT, "timeout exceeds the bound")
        if self.retry_policy is RetryPolicy.NONE and self.retry_count != 0:
            raise _fail(
                ReleasePlanCode.CONFLICT, "retry count conflicts with retry policy"
            )
        if self.retry_policy is RetryPolicy.FIXED and self.retry_count < 1:
            raise _fail(ReleasePlanCode.CONFLICT, "fixed retry policy requires retries")
        if self.timeout_policy is TimeoutPolicy.NONE and self.timeout_seconds != 0:
            raise _fail(
                ReleasePlanCode.CONFLICT, "timeout seconds conflict with timeout policy"
            )
        if self.timeout_policy is TimeoutPolicy.FAIL and self.timeout_seconds < 1:
            raise _fail(
                ReleasePlanCode.CONFLICT, "timeout policy requires timeout seconds"
            )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "release_profile": self.release_profile.canonical_payload(),
            "target_branch": self.target_branch,
            "candidate": self.candidate.canonical_payload(),
            "certificate": self.certificate.canonical_payload(),
            "config": self.config.canonical_payload(),
            "toolchain": self.toolchain.canonical_payload(),
            "command": self.command.canonical_payload(),
            "artifact_contract": self.artifact_contract.canonical_payload(),
            "resource_profile": self.resource_profile.canonical_payload(),
            "retry_policy": self.retry_policy.value,
            "retry_count": self.retry_count,
            "timeout_policy": self.timeout_policy.value,
            "timeout_seconds": self.timeout_seconds,
        }

    @property
    def digest(self) -> str:
        return _digest_payload(self.canonical_payload())


ReleaseProfile = ImmutableDigestReference
DecisionContext = ReleaseDecisionContext
ReleasePlanDecisions = ReleaseDecisionContext


def _normalize_decision_context(
    context: object | None,
    *,
    release_profile: object | None = None,
    target_branch: object | None = None,
    candidate: object | None = None,
    certificate: object | None = None,
    config: object | None = None,
    toolchain: object | None = None,
    command: object | None = None,
    artifact_contract: object | None = None,
    resource_profile: object | None = None,
    retry_policy: object | None = None,
    retry_count: object | None = None,
    timeout_policy: object | None = None,
    timeout_seconds: object | None = None,
) -> ReleaseDecisionContext:
    aliases = (
        release_profile,
        target_branch,
        candidate,
        certificate,
        config,
        toolchain,
        command,
        artifact_contract,
        resource_profile,
        retry_policy,
        retry_count,
        timeout_policy,
        timeout_seconds,
    )
    if context is not None and type(context) is not ReleaseDecisionContext:
        raise _fail(ReleasePlanCode.PROFILE, "decision context is unsupported")
    if context is not None and any(item is not None for item in aliases):
        raise _fail(ReleasePlanCode.CONFLICT, "decision context aliases conflict")
    if context is not None:
        return ReleaseDecisionContext(
            release_profile=context.release_profile,
            target_branch=context.target_branch,
            candidate=context.candidate,
            certificate=context.certificate,
            config=context.config,
            toolchain=context.toolchain,
            command=context.command,
            artifact_contract=context.artifact_contract,
            resource_profile=context.resource_profile,
            retry_policy=context.retry_policy,
            retry_count=context.retry_count,
            timeout_policy=context.timeout_policy,
            timeout_seconds=context.timeout_seconds,
        )
    defaults = ReleaseDecisionContext()
    values = {
        "release_profile": defaults.release_profile
        if release_profile is None
        else release_profile,
        "target_branch": defaults.target_branch
        if target_branch is None
        else target_branch,
        "candidate": defaults.candidate if candidate is None else candidate,
        "certificate": defaults.certificate if certificate is None else certificate,
        "config": defaults.config if config is None else config,
        "toolchain": defaults.toolchain if toolchain is None else toolchain,
        "command": defaults.command if command is None else command,
        "artifact_contract": defaults.artifact_contract
        if artifact_contract is None
        else artifact_contract,
        "resource_profile": defaults.resource_profile
        if resource_profile is None
        else resource_profile,
        "retry_policy": defaults.retry_policy if retry_policy is None else retry_policy,
        "retry_count": defaults.retry_count if retry_count is None else retry_count,
        "timeout_policy": defaults.timeout_policy
        if timeout_policy is None
        else timeout_policy,
        "timeout_seconds": defaults.timeout_seconds
        if timeout_seconds is None
        else timeout_seconds,
    }
    return ReleaseDecisionContext(**values)  # type: ignore[arg-type]


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
    except _UNTRUSTED_DATA_ERRORS:
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
    decision_digest: str,
    resource_profile: ImmutableDigestReference,
    retry_policy: RetryPolicy,
    retry_count: int,
    timeout_policy: TimeoutPolicy,
    timeout_seconds: int,
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
        "decision_digest": decision_digest,
        "resource_profile": resource_profile.canonical_payload(),
        "retry_policy": retry_policy.value,
        "retry_count": retry_count,
        "timeout_policy": timeout_policy.value,
        "timeout_seconds": timeout_seconds,
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
    decision_digest: str = ""
    resource_profile: ImmutableDigestReference = field(
        default_factory=lambda: _default_reference("resource-profile:default")
    )
    retry_policy: RetryPolicy = RetryPolicy.NONE
    retry_count: int = 0
    timeout_policy: TimeoutPolicy = TimeoutPolicy.NONE
    timeout_seconds: int = 0

    def __post_init__(self) -> None:
        if type(self.stage_id) is not str or not self.stage_id:
            raise _fail(ReleasePlanCode.DIGEST, "stage ID is required")
        stage_id = _strict_text(
            self.stage_id, "stage ID", max_length=256, code=ReleasePlanCode.DIGEST
        )
        if type(self.kind) is not StageKind:
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
        consent_reference = _revalidate_consent(self.consent_reference)
        if self.kind is StageKind.PUSH and consent_reference is None:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push stage requires immutable consent"
            )
        if type(self.failure_policy) is not FailurePolicy:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "stage failure policy is unsupported"
            )
        if self.failure_policy is not FailurePolicy.BLOCK_DEPENDENTS:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "stage failure policy is unsupported"
            )
        decision_digest = _strict_digest(self.decision_digest, "stage decision digest")
        if type(self.resource_profile) is not ImmutableDigestReference:
            raise _fail(
                ReleasePlanCode.PROFILE,
                "stage resource profile must be an immutable reference",
            )
        resource_profile = ImmutableDigestReference(
            self.resource_profile.name, self.resource_profile.digest
        )
        if type(self.retry_policy) is not RetryPolicy:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "stage retry policy is unsupported"
            )
        if type(self.timeout_policy) is not TimeoutPolicy:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "stage timeout policy is unsupported"
            )
        if type(self.retry_count) is not int or isinstance(self.retry_count, bool):
            raise _fail(ReleasePlanCode.INVALID_INPUT, "stage retry count is invalid")
        if not 0 <= self.retry_count <= 32:
            raise _fail(
                ReleasePlanCode.UNBOUNDED_INPUT, "stage retry count exceeds bound"
            )
        if type(self.timeout_seconds) is not int or isinstance(
            self.timeout_seconds, bool
        ):
            raise _fail(ReleasePlanCode.INVALID_INPUT, "stage timeout is invalid")
        if not 0 <= self.timeout_seconds <= 604800:
            raise _fail(ReleasePlanCode.UNBOUNDED_INPUT, "stage timeout exceeds bound")
        if self.retry_policy is RetryPolicy.NONE and self.retry_count != 0:
            raise _fail(
                ReleasePlanCode.CONFLICT, "stage retry count conflicts with policy"
            )
        if self.retry_policy is RetryPolicy.FIXED and self.retry_count < 1:
            raise _fail(ReleasePlanCode.CONFLICT, "stage retry policy requires retries")
        if self.timeout_policy is TimeoutPolicy.NONE and self.timeout_seconds != 0:
            raise _fail(ReleasePlanCode.CONFLICT, "stage timeout conflicts with policy")
        if self.timeout_policy is TimeoutPolicy.FAIL and self.timeout_seconds < 1:
            raise _fail(
                ReleasePlanCode.CONFLICT, "stage timeout policy requires seconds"
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
            consent_reference=consent_reference,
            decision_digest=decision_digest,
            resource_profile=resource_profile,
            retry_policy=self.retry_policy,
            retry_count=self.retry_count,
            timeout_policy=self.timeout_policy,
            timeout_seconds=self.timeout_seconds,
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
        object.__setattr__(self, "consent_reference", consent_reference)
        object.__setattr__(self, "input_digest", input_digest)
        object.__setattr__(self, "decision_digest", decision_digest)
        object.__setattr__(self, "resource_profile", resource_profile)

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
            decision_digest=self.decision_digest,
            resource_profile=self.resource_profile,
            retry_policy=self.retry_policy,
            retry_count=self.retry_count,
            timeout_policy=self.timeout_policy,
            timeout_seconds=self.timeout_seconds,
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


def _derive_stage_sequence(
    *,
    selected: tuple[str, ...],
    project_map: dict[str, ProjectRecord],
    project_edges: tuple[tuple[str, str], ...],
    groups: tuple[tuple[str, ...], ...],
    base_sha: str,
    generation_id: str,
    graph_digest: str,
    selection_digest: str,
    version_plan_digest: str,
    version_preview_digests: tuple[str, ...],
    floor_preview_digests: tuple[str, ...],
    validation: dict[str, ProfileBinding],
    build: dict[str, ProfileBinding],
    decision_context: ReleaseDecisionContext,
    consent: PushConsentReference | None,
) -> tuple[StagePreview, ...]:
    """Derive the only accepted stage composition from frozen source fields."""

    project_dependencies: dict[str, set[str]] = {project: set() for project in selected}
    for dependent, dependency in project_edges:
        if (
            dependent not in project_dependencies
            or dependency not in project_dependencies
        ):
            raise _fail(
                ReleasePlanCode.MISSING, "stage source edge names an unknown project"
            )
        project_dependencies[dependent].add(dependency)
    stage_by_key: dict[tuple[StageKind, str], StagePreview] = {}
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
                stage_by_key[(kind, project_id)] = _stage_preview(
                    kind=kind,
                    project=project,
                    base_sha=base_sha,
                    generation_id=generation_id,
                    graph_digest=graph_digest,
                    selection_digest=selection_digest,
                    version_plan_digest=version_plan_digest,
                    version_preview_digests=version_preview_digests,
                    floor_preview_digests=floor_preview_digests,
                    validation_profile_digest=validation[project_id].digest,
                    build_profile_digest=build[project_id].digest,
                    depends_on=tuple(sorted(set(dependencies))),
                    consent_reference=None,
                    decision_context=decision_context,
                )
    if consent is not None:
        for group in groups:
            for project_id in group:
                dependencies = [stage_by_key[(StageKind.PACKAGE, project_id)].stage_id]
                dependencies.extend(
                    stage_by_key[(StageKind.PUSH, dependency)].stage_id
                    for dependency in sorted(project_dependencies[project_id])
                )
                stage_by_key[(StageKind.PUSH, project_id)] = _stage_preview(
                    kind=StageKind.PUSH,
                    project=project_map[project_id],
                    base_sha=base_sha,
                    generation_id=generation_id,
                    graph_digest=graph_digest,
                    selection_digest=selection_digest,
                    version_plan_digest=version_plan_digest,
                    version_preview_digests=version_preview_digests,
                    floor_preview_digests=floor_preview_digests,
                    validation_profile_digest=validation[project_id].digest,
                    build_profile_digest=build[project_id].digest,
                    depends_on=tuple(sorted(set(dependencies))),
                    consent_reference=consent,
                    decision_context=decision_context,
                )
    stages = tuple(
        stage_by_key[(kind, project_id)]
        for kind in (*stage_order, *((StageKind.PUSH,) if consent is not None else ()))
        for group in groups
        for project_id in group
    )
    _validate_stage_dag(stages)
    return stages


def _plan_payload(
    plan: FrozenReleasePlan, *, include_digest: bool = False
) -> dict[str, object]:
    source_graph = _snapshot_graph(plan.graph)
    source_selection = _snapshot_selection(plan.selection)
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
        "source_graph": {
            **source_graph.canonical_payload(),
            "digest": source_graph.digest,
        },
        "selection": source_selection.canonical_payload(include_digest=True),
        "validation_profiles": tuple(
            binding.canonical_payload() for binding in plan.validation_profiles
        ),
        "build_profiles": tuple(
            binding.canonical_payload() for binding in plan.build_profiles
        ),
        "decision_context": plan.decision_context.canonical_payload(),
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
    graph: DependencyGraph
    selection: SelectedChangeClosure
    decision_context: ReleaseDecisionContext
    push_consent: PushConsentReference | None = None
    plan_digest: str = ""
    contract_version: int = C11_FROZEN_PLAN_VERSION

    def __post_init__(self) -> None:
        _validate_frozen_plan_fields(self, require_digest=True)

    @property
    def digest(self) -> str:
        return self.plan_digest

    @property
    def release_profile(self) -> ImmutableDigestReference:
        return self.decision_context.release_profile

    @property
    def target_branch(self) -> str:
        return self.decision_context.target_branch

    @property
    def candidate(self) -> ImmutableDigestReference:
        return self.decision_context.candidate

    @property
    def certificate(self) -> ImmutableDigestReference:
        return self.decision_context.certificate

    @property
    def config(self) -> ImmutableDigestReference:
        return self.decision_context.config

    @property
    def toolchain(self) -> ImmutableDigestReference:
        return self.decision_context.toolchain

    @property
    def command(self) -> ImmutableDigestReference:
        return self.decision_context.command

    @property
    def artifact_contract(self) -> ImmutableDigestReference:
        return self.decision_context.artifact_contract

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
            graph = _snapshot_graph(graph)
            selection = _snapshot_selection(selection)
            if graph.digest != self.graph_digest:
                raise _fail(
                    ReleasePlanCode.GRAPH_DRIFT,
                    "current graph does not match frozen plan",
                )
            if selection.digest != self.selection_digest:
                raise _fail(
                    ReleasePlanCode.SELECTION_DRIFT,
                    "current selection does not match frozen plan",
                )
            if graph.canonical_payload() != selection.source_graph.canonical_payload():
                raise _fail(
                    ReleasePlanCode.GRAPH_DRIFT,
                    "current graph evidence does not match frozen selection",
                )
            current_version_plan = (
                self.version_plan if version_plan is None else version_plan
            )
            if type(current_version_plan) is not VersionPlan:
                raise _fail(
                    ReleasePlanCode.VERSION_PLAN_DRIFT,
                    "current version plan is unsupported",
                )
            current_version_plan = _revalidate_version_plan(current_version_plan)
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
                decision_context=self.decision_context,
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
        except _UNTRUSTED_DATA_ERRORS:
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
            decision_context=frozen.decision_context,
        )


FrozenWorkspaceReleasePlan = FrozenReleasePlan
FrozenPlan = FrozenReleasePlan
ReleasePlan = FrozenReleasePlan


def _exact_nested_tuple(
    value: object, field_name: str, *, max_items: int
) -> tuple[object, ...]:
    """Reject lazy/subclass containers before any iteration or sorting."""

    return _bounded_tuple(value, field_name, max_items=max_items)


def _revalidate_version(value: object, field_name: str) -> Version:
    if type(value) is not Version:
        raise _fail(ReleasePlanCode.INVALID_INPUT, f"{field_name} is not a Version")
    return Version(_strict_text(value.value, field_name, max_length=128))


def _revalidate_package_key(value: object, field_name: str) -> PackageKey:
    if type(value) is not PackageKey:
        raise _fail(ReleasePlanCode.IDENTITY, f"{field_name} is not a package identity")
    if type(value.ecosystem) is not Ecosystem:
        raise _fail(ReleasePlanCode.IDENTITY, f"{field_name} ecosystem is invalid")
    repository_id = _canonical_repository_exact(value.repository_id, field_name)
    name = _strict_text(value.name, f"{field_name} name", max_length=256)
    candidate = PackageKey(repository_id, value.ecosystem, name)
    if candidate.repository_id != repository_id or candidate.name != name:
        raise _fail(ReleasePlanCode.IDENTITY, f"{field_name} is not canonical")
    return candidate


def _revalidate_floor(value: object, field_name: str) -> VersionFloor:
    if type(value) is not VersionFloor:
        raise _fail(
            ReleasePlanCode.INVALID_INPUT, f"{field_name} is not a version floor"
        )
    return VersionFloor(
        _strict_text(value.operator, f"{field_name} operator", max_length=2),
        _revalidate_version(value.version, f"{field_name} version"),
    )


def _revalidate_package(package: PackageRecord) -> PackageRecord:
    if type(package) is not PackageRecord:
        raise _fail(ReleasePlanCode.INVALID_INPUT, "plan package is not a C-11 record")
    key = _revalidate_package_key(package.key, "package key")
    version = _revalidate_version(package.version, "package version")
    raw_sources = _exact_nested_tuple(
        package.version_sources, "package version sources", max_items=MAX_STRING_LENGTH
    )
    sources: list[VersionSource] = []
    for source in raw_sources:
        if type(source) is not VersionSource:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "package version source is invalid"
            )
        sources.append(
            VersionSource(
                _strict_text(source.location, "version source location"),
                _revalidate_version(source.version, "version source version"),
            )
        )
    raw_dependencies = _exact_nested_tuple(
        package.dependencies, "package dependencies", max_items=MAX_EDGES
    )
    dependencies: list[DependencySpec] = []
    for dependency in raw_dependencies:
        if type(dependency) is not DependencySpec:
            raise _fail(ReleasePlanCode.INVALID_INPUT, "package dependency is invalid")
        target = dependency.target
        if (
            type(target) is not PackageReference
            or type(target.ecosystem) is not Ecosystem
        ):
            raise _fail(
                ReleasePlanCode.IDENTITY, "package dependency target is invalid"
            )
        repository_id = target.repository_id
        if repository_id is not None:
            repository_id = _canonical_repository_exact(
                repository_id, "dependency target repository"
            )
        floor = (
            None
            if dependency.floor is None
            else _revalidate_floor(dependency.floor, "dependency floor")
        )
        dependencies.append(
            DependencySpec(
                PackageReference(
                    target.ecosystem,
                    _strict_text(target.name, "dependency target name", max_length=256),
                    repository_id,
                ),
                floor,
                _strict_text(dependency.source, "dependency source"),
            )
        )
    raw_metadata = _exact_nested_tuple(
        package.metadata_files, "package metadata files", max_items=MAX_STRING_LENGTH
    )
    metadata = tuple(
        _strict_text(item, "package metadata file") for item in raw_metadata
    )
    return PackageRecord(key, version, tuple(sources), tuple(dependencies), metadata)


def _revalidate_project(project: ProjectRecord) -> ProjectRecord:
    """Re-materialize all nested C-11 records before any provider traversal."""

    try:
        if type(project) is not ProjectRecord:
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan project is not a C-11 record"
            )
        raw_packages = _exact_nested_tuple(
            project.packages, "project packages", max_items=MAX_PACKAGES
        )
        packages = tuple(
            _revalidate_package(cast(PackageRecord, item)) for item in raw_packages
        )
        raw_metadata = _exact_nested_tuple(
            project.metadata_files,
            "project metadata files",
            max_items=MAX_STRING_LENGTH,
        )
        metadata = tuple(
            _strict_text(item, "project metadata file") for item in raw_metadata
        )
        tree_value = project.tree_sha
        if type(tree_value) is not str:
            raise _fail(ReleasePlanCode.TREE_SHA, "project tree SHA is not a string")
        return ProjectRecord(
            repository_id=_canonical_repository_exact(
                project.repository_id, "project repository"
            ),
            tree_sha=(
                ""
                if len(tree_value) == 0
                else _strict_sha(
                    tree_value, "project tree SHA", code=ReleasePlanCode.TREE_SHA
                )
            ),
            packages=packages,
            metadata_files=metadata,
        )
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.DIGEST, "nested project evidence could not be validated"
        ) from None


def _revalidate_edge(edge: DependencyEdge) -> DependencyEdge:
    if type(edge) is not DependencyEdge:
        raise _fail(ReleasePlanCode.INVALID_INPUT, "dependency edge is unsupported")
    floor = None if edge.floor is None else _revalidate_floor(edge.floor, "edge floor")
    if type(edge.confidence) is not EdgeConfidence:
        raise _fail(ReleasePlanCode.INVALID_INPUT, "edge confidence is unsupported")
    return DependencyEdge(
        _revalidate_package_key(edge.dependent, "edge dependent"),
        _revalidate_package_key(edge.dependency, "edge dependency"),
        floor,
        _strict_text(edge.source, "edge source"),
        edge.confidence,
    )


def _revalidate_representation(
    value: object, field_name: str
) -> MetadataRepresentation:
    if type(value) is not MetadataRepresentation:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, f"{field_name} is invalid")
    return cast(MetadataRepresentation, value)


def _revalidate_version_policy(policy: VersionSourcePolicy) -> VersionSourcePolicy:
    if type(policy) is not VersionSourcePolicy:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "version policy is invalid")
    exact_version = policy.exact_version
    if exact_version is not None:
        exact_version = _strict_text(
            exact_version, "exact next version", max_length=256
        )
    if type(policy.representation) is not MetadataRepresentation:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT,
            "version policy representation is invalid",
        )
    if type(policy.bump) is not VersionBump:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version policy bump is invalid"
        )
    return VersionSourcePolicy(
        source_location=_strict_text(policy.source_location, "version source location"),
        representation=policy.representation,
        bump=policy.bump,
        exact_version=exact_version,
    )


def _revalidate_version_site(site: VersionSourceSite) -> VersionSourceSite:
    if type(site) is not VersionSourceSite:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version source site is invalid"
        )
    if type(site.symlink) is not bool:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version source site flag is invalid"
        )
    return VersionSourceSite(
        package=_revalidate_package_key(site.package, "version source site package"),
        file_path=_strict_text(site.file_path, "version source site path"),
        old_text=_strict_text(site.old_text, "version source site old text"),
        policy=_revalidate_version_policy(site.policy),
        symlink=site.symlink,
    )


def _revalidate_floor_site(site: FloorRewriteSite) -> FloorRewriteSite:
    if type(site) is not FloorRewriteSite:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "floor source site is invalid")
    if type(site.representation) is not MetadataRepresentation:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "floor site representation is invalid"
        )
    if type(site.policy) is not FloorPolicy:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "floor site policy is invalid")
    if type(site.symlink) is not bool:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "floor site flag is invalid")
    return FloorRewriteSite(
        dependent=_revalidate_package_key(site.dependent, "floor site dependent"),
        dependency=_revalidate_package_key(site.dependency, "floor site dependency"),
        file_path=_strict_text(site.file_path, "floor site path"),
        source_location=_strict_text(site.source_location, "floor site location"),
        representation=site.representation,
        old_text=_strict_text(site.old_text, "floor site old text", max_length=256),
        policy=site.policy,
        symlink=site.symlink,
    )


def _revalidate_witness(value: object, field_name: str) -> tuple[str, ...]:
    raw = _bounded_tuple(value, field_name, max_items=16)
    return tuple(_strict_text(item, f"{field_name} entry") for item in raw)


def _revalidate_optional_digest(value: object, field_name: str) -> str:
    if type(value) is not str:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, f"{field_name} is invalid")
    if value == "":
        return ""
    return _strict_digest(value, field_name)


def _revalidate_optional_text(
    value: object, field_name: str, *, max_length: int = MAX_STRING_LENGTH
) -> str:
    """Validate an optional CP3 text field before testing its omission value."""

    if type(value) is not str:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, f"{field_name} is invalid")
    if value == "":
        return ""
    return _strict_text(value, field_name, max_length=max_length)


def _revalidate_version_preview(preview: VersionPreview) -> VersionPreview:
    if type(preview) is not VersionPreview:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "version preview is invalid")
    if type(preview.reason) is not VersionPreviewReason:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version preview reason is invalid"
        )
    if type(preview.is_noop) is not bool:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version preview flag is invalid"
        )
    return VersionPreview(
        project_id=_canonical_repository_exact(
            preview.project_id, "version preview project"
        ),
        package=_revalidate_package_key(preview.package, "version preview package"),
        file_path=_strict_text(preview.file_path, "version preview path"),
        source_location=_strict_text(
            preview.source_location, "version preview location"
        ),
        representation=_revalidate_representation(
            preview.representation, "version preview representation"
        ),
        source_sha=_strict_sha(preview.source_sha, "version preview source SHA"),
        old_text=_strict_text(preview.old_text, "version preview old text"),
        new_text=_strict_text(preview.new_text, "version preview new text"),
        current_version=_revalidate_version(
            preview.current_version, "version preview current"
        ),
        next_version=_revalidate_version(preview.next_version, "version preview next"),
        policy=_revalidate_version_policy(preview.policy),
        reason=preview.reason,
        witness=_revalidate_witness(preview.witness, "version preview witness"),
        graph_digest=_strict_digest(
            preview.graph_digest, "version preview graph digest"
        ),
        selection_digest=_strict_digest(
            preview.selection_digest, "version preview selection digest"
        ),
        is_noop=preview.is_noop,
        plan_digest=_revalidate_optional_digest(
            preview.plan_digest, "version preview plan digest"
        ),
    )


def _revalidate_floor_preview(preview: FloorPreview) -> FloorPreview:
    if type(preview) is not FloorPreview:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "floor preview is invalid")
    if type(preview.representation) is not MetadataRepresentation:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT,
            "floor preview representation is invalid",
        )
    if type(preview.policy) is not FloorPolicy:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "floor preview policy is invalid"
        )
    if type(preview.reason) is not FloorPreviewReason:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "floor preview reason is invalid"
        )
    if type(preview.is_noop) is not bool:
        raise _fail(ReleasePlanCode.VERSION_PLAN_DRIFT, "floor preview flag is invalid")
    old_text = _revalidate_optional_text(
        preview.old_text, "floor preview old text", max_length=256
    )
    new_text = _revalidate_optional_text(
        preview.new_text, "floor preview new text", max_length=256
    )
    old_normalized = _revalidate_optional_text(
        preview.old_normalized, "floor preview old normalized", max_length=256
    )
    new_normalized = _revalidate_optional_text(
        preview.new_normalized, "floor preview new normalized", max_length=256
    )
    return FloorPreview(
        project_id=_canonical_repository_exact(
            preview.project_id, "floor preview project"
        ),
        dependent=_revalidate_package_key(preview.dependent, "floor preview dependent"),
        dependency=_revalidate_package_key(
            preview.dependency, "floor preview dependency"
        ),
        file_path=_strict_text(preview.file_path, "floor preview path"),
        source_location=_strict_text(preview.source_location, "floor preview location"),
        representation=preview.representation,
        source_sha=_strict_sha(preview.source_sha, "floor preview source SHA"),
        old_text=old_text,
        new_text=new_text,
        old_normalized=old_normalized,
        new_normalized=new_normalized,
        policy=preview.policy,
        reason=preview.reason,
        witness=_revalidate_witness(preview.witness, "floor preview witness"),
        graph_digest=_strict_digest(preview.graph_digest, "floor preview graph digest"),
        selection_digest=_strict_digest(
            preview.selection_digest, "floor preview selection digest"
        ),
        is_noop=preview.is_noop,
        plan_digest=_revalidate_optional_digest(
            preview.plan_digest, "floor preview plan digest"
        ),
    )


def _validate_edge_source_evidence(
    projects: tuple[ProjectRecord, ...], edges: tuple[DependencyEdge, ...]
) -> None:
    """Bind graph edges to package dependency declarations before derivation.

    An explicit overlay may resolve an ambiguous owner or supply a missing
    floor, so the C-11 edge itself remains the source evidence in those two
    bounded cases.  For an unambiguous metadata edge, however, changing its
    floor or provenance while recomputing the graph digest is not a valid
    graph: both values must still match the package declaration.
    """

    packages = tuple(package for project in projects for package in project.packages)
    package_map = {package.key.value: package for package in packages}
    owner_counts: dict[tuple[Ecosystem, str], int] = {}
    for package in packages:
        owner_key = (package.key.ecosystem, package.key.name)
        owner_counts[owner_key] = owner_counts.get(owner_key, 0) + 1
    for edge in edges:
        dependent = package_map.get(edge.dependent.value)
        if dependent is None or edge.dependency.value not in package_map:
            raise _fail(
                ReleasePlanCode.MISSING,
                "graph edge names an unknown package",
            )
        candidates = tuple(
            spec
            for spec in dependent.dependencies
            if spec.target.ecosystem is edge.dependency.ecosystem
            and spec.target.name == edge.dependency.name
            and (
                spec.target.repository_id is None
                or spec.target.repository_id == edge.dependency.repository_id
            )
        )
        if not candidates:
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT,
                "graph edge is not supported by package dependency evidence",
            )
        target = candidates[0].target
        owner_count = owner_counts.get(
            (target.ecosystem, target.name),
            0,
        )
        # A uniquely resolved target with a declared floor has a complete
        # source declaration. Overlay-only edges are permitted to resolve an
        # explicit owner or fill a missing floor.
        if candidates[0].floor is not None and (
            target.repository_id is not None or owner_count == 1
        ):
            spec = candidates[0]
            if spec.floor != edge.floor or spec.source != edge.source:
                raise _fail(
                    ReleasePlanCode.GRAPH_DRIFT,
                    "graph edge provenance does not match package evidence",
                )


def _snapshot_graph(graph: DependencyGraph) -> DependencyGraph:
    """Copy a graph only after proving every container is bounded and builtin."""

    try:
        if type(graph) is not DependencyGraph:
            raise _fail(ReleasePlanCode.GRAPH_DRIFT, "graph is not a C-11 record")
        raw_projects = _exact_nested_tuple(
            graph.projects, "graph projects", max_items=MAX_PROJECTS
        )
        raw_packages = _exact_nested_tuple(
            graph.packages, "graph packages", max_items=MAX_PACKAGES
        )
        raw_edges = _exact_nested_tuple(graph.edges, "graph edges", max_items=MAX_EDGES)
        raw_project_edges = _exact_nested_tuple(
            graph.project_edges, "graph project edges", max_items=MAX_EDGES
        )
        raw_groups = _exact_nested_tuple(
            graph.parallel_groups, "graph parallel groups", max_items=MAX_PROJECTS
        )
        projects = tuple(
            _revalidate_project(cast(ProjectRecord, item)) for item in raw_projects
        )
        packages = tuple(
            _revalidate_package(cast(PackageRecord, item)) for item in raw_packages
        )
        edges = tuple(
            _revalidate_edge(cast(DependencyEdge, item)) for item in raw_edges
        )
        _validate_edge_source_evidence(projects, edges)
        project_edges: list[tuple[str, str]] = []
        for pair in raw_project_edges:
            if (
                type(pair) not in (tuple, list)
                or len(cast(tuple[object, ...] | list[object], pair)) != 2
            ):
                raise _fail(
                    ReleasePlanCode.INVALID_INPUT, "graph project edge is invalid"
                )
            left, right = cast(tuple[object, ...] | list[object], pair)
            project_edges.append(
                (
                    _canonical_repository_exact(left, "graph project edge endpoint"),
                    _canonical_repository_exact(right, "graph project edge endpoint"),
                )
            )
        groups: list[tuple[str, ...]] = []
        for raw_group in raw_groups:
            group = _canonical_project_ids(raw_group, "graph parallel group")
            groups.append(group)
        digest = _strict_digest(graph.digest, "graph digest")
        derived = build_dependency_graph(projects, overlay_edges=edges)
        candidate = DependencyGraph(
            projects=projects,
            packages=packages,
            edges=edges,
            project_edges=tuple(project_edges),
            parallel_groups=tuple(groups),
            digest=digest,
        )
        if (
            tuple(project.project_id for project in projects)
            != tuple(project.project_id for project in derived.projects)
            or tuple(package.key.value for package in packages)
            != tuple(package.key.value for package in derived.packages)
            or tuple(edge.value for edge in edges)
            != tuple(edge.value for edge in derived.edges)
            or tuple(project_edges) != derived.project_edges
            or tuple(groups) != derived.parallel_groups
        ):
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT,
                "graph inventory does not match package evidence",
            )
        if derived.digest != digest:
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT, "graph digest does not match sources"
            )
        if candidate.projects != projects or candidate.packages != packages:
            raise _fail(ReleasePlanCode.GRAPH_DRIFT, "graph records are not canonical")
        if candidate.digest != _digest_payload(candidate.canonical_payload()):
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT, "graph digest does not match contents"
            )
        if (
            tuple(project_edges) != candidate.project_edges
            or tuple(groups) != candidate.parallel_groups
            or candidate.edges != edges
        ):
            raise _fail(ReleasePlanCode.GRAPH_DRIFT, "graph topology is not canonical")
        return candidate
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.GRAPH_DRIFT, "graph evidence could not be validated"
        ) from None


def _snapshot_selection(selection: SelectedChangeClosure) -> SelectedChangeClosure:
    """Copy a selected closure without walking lazy/hostile containers."""

    try:
        if type(selection) is not SelectedChangeClosure:
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT, "selection is not a C-11 record"
            )
        source = _snapshot_graph(selection.source_graph)
        policy = _strict_selection_policy(selection.policy)
        known = _canonical_project_ids(
            selection.known_project_ids, "known project IDs", allow_empty=False
        )
        selected = _canonical_project_ids(
            selection.selected_project_ids, "selected project IDs", allow_empty=False
        )
        projects = tuple(
            _revalidate_project(cast(ProjectRecord, item))
            for item in _exact_nested_tuple(
                selection.projects, "closure projects", max_items=MAX_PROJECTS
            )
        )
        edges = tuple(
            _revalidate_edge(cast(DependencyEdge, item))
            for item in _exact_nested_tuple(
                selection.edges, "closure edges", max_items=MAX_EDGES
            )
        )
        raw_project_edges = _exact_nested_tuple(
            selection.project_edges, "closure project edges", max_items=MAX_EDGES
        )
        project_edges: list[tuple[str, str]] = []
        for pair in raw_project_edges:
            if (
                type(pair) not in (tuple, list)
                or len(cast(tuple[object, ...] | list[object], pair)) != 2
            ):
                raise _fail(
                    ReleasePlanCode.SELECTION_DRIFT, "closure project edge is invalid"
                )
            project_edges.append(
                (
                    _canonical_repository_exact(
                        cast(tuple[object, ...] | list[object], pair)[0],
                        "closure project edge endpoint",
                    ),
                    _canonical_repository_exact(
                        cast(tuple[object, ...] | list[object], pair)[1],
                        "closure project edge endpoint",
                    ),
                )
            )
        raw_groups = _exact_nested_tuple(
            selection.parallel_groups, "closure parallel groups", max_items=MAX_PROJECTS
        )
        groups = tuple(
            _canonical_project_ids(group, "closure parallel group")
            for group in raw_groups
        )
        raw_explanations = _exact_nested_tuple(
            selection.explanations, "closure explanations", max_items=MAX_PROJECTS
        )
        explanations = [
            _strict_selection_explanation(cast(SelectionExplanation, item))
            for item in raw_explanations
        ]
        digest = _strict_digest(selection.digest, "selection digest")
        candidate = SelectedChangeClosure(
            policy=policy,
            known_project_ids=known,
            selected_project_ids=selected,
            projects=projects,
            edges=edges,
            project_edges=tuple(project_edges),
            parallel_groups=groups,
            explanations=tuple(explanations),
            source_graph=source,
            digest=digest,
        )
        if candidate.digest != digest:
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "selection digest does not match contents",
            )
        return candidate
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.SELECTION_DRIFT, "selection evidence could not be validated"
        ) from None


def _revalidate_version_plan(version_plan: VersionPlan) -> VersionPlan:
    """Reconstruct CP3 evidence instead of trusting a forged dataclass shell."""

    try:
        if type(version_plan) is not VersionPlan:
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT, "version plan is not a C-11 record"
            )
        graph_digest = _strict_digest(
            version_plan.graph_digest, "version plan graph digest"
        )
        selection_digest = _strict_digest(
            version_plan.selection_digest, "version plan selection digest"
        )
        plan_digest = _strict_digest(version_plan.plan_digest, "version plan digest")
        raw_next = _exact_nested_tuple(
            version_plan.next_versions,
            "version plan next versions",
            max_items=MAX_PACKAGES,
        )
        next_versions: list[tuple[str, Version]] = []
        for pair in raw_next:
            if (
                type(pair) not in (tuple, list)
                or len(cast(tuple[object, ...] | list[object], pair)) != 2
            ):
                raise _fail(
                    ReleasePlanCode.VERSION_PLAN_DRIFT,
                    "version plan next version pair is invalid",
                )
            pair_values = cast(tuple[object, ...] | list[object], pair)
            next_versions.append(
                (
                    _strict_text(pair_values[0], "version plan package identity"),
                    _revalidate_version(pair_values[1], "version plan next version"),
                )
            )
        raw_batches = _exact_nested_tuple(
            version_plan.package_batches,
            "version plan package batches",
            max_items=MAX_PACKAGES,
        )
        package_batches: list[tuple[str, ...]] = []
        for batch in raw_batches:
            raw_batch = _exact_nested_tuple(
                batch, "version plan package batch", max_items=MAX_PACKAGES
            )
            package_batches.append(
                tuple(
                    _strict_text(item, "version plan package identity")
                    for item in raw_batch
                )
            )
        raw_versions = _exact_nested_tuple(
            version_plan.version_previews,
            "version plan version previews",
            max_items=MAX_PACKAGES,
        )
        raw_floors = _exact_nested_tuple(
            version_plan.floor_previews,
            "version plan floor previews",
            max_items=MAX_EDGES,
        )
        versions = tuple(
            _revalidate_version_preview(cast(VersionPreview, item))
            for item in raw_versions
        )
        floors = tuple(
            _revalidate_floor_preview(cast(FloorPreview, item)) for item in raw_floors
        )
        candidate = VersionPlan(
            graph_digest=graph_digest,
            selection_digest=selection_digest,
            next_versions=tuple(next_versions),
            package_batches=tuple(package_batches),
            version_previews=versions,
            floor_previews=floors,
            plan_digest=plan_digest,
        )
        if (
            candidate.next_versions != tuple(next_versions)
            or candidate.package_batches != tuple(package_batches)
            or candidate.version_previews != versions
            or candidate.floor_previews != floors
            or candidate.plan_digest != plan_digest
        ):
            raise _fail(
                ReleasePlanCode.VERSION_PLAN_DRIFT,
                "version plan evidence is not canonical",
            )
        return candidate
    except ReleasePlanError:
        raise
    except VersionPlanningError:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT, "version plan evidence is invalid"
        ) from None
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT,
            "version plan evidence could not be validated",
        ) from None


def _revalidate_stage(stage: StagePreview) -> StagePreview:
    """Reconstruct a stage to verify ID/input/dependency binding after forgery."""

    try:
        if type(stage) is not StagePreview:
            raise _fail(ReleasePlanCode.INVALID_INPUT, "stage is not a stage preview")
        consent = _revalidate_consent(stage.consent_reference)
        return StagePreview(
            stage_id=stage.stage_id,
            kind=stage.kind,
            project_id=_canonical_repository_exact(stage.project_id, "stage project"),
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
            consent_reference=consent,
            failure_policy=stage.failure_policy,
            input_digest=stage.input_digest,
            decision_digest=stage.decision_digest,
            resource_profile=stage.resource_profile,
            retry_policy=stage.retry_policy,
            retry_count=stage.retry_count,
            timeout_policy=stage.timeout_policy,
            timeout_seconds=stage.timeout_seconds,
        )
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
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
            project_id=_canonical_repository_exact(
                binding.project_id, "profile project"
            ),
            name=binding.name,
            digest=binding.digest,
            kind=binding.kind,
        )
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
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
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.CONSENT, "consent reference could not be validated"
        ) from None


def _revalidate_decision_context(
    context: ReleaseDecisionContext,
) -> ReleaseDecisionContext:
    try:
        if type(context) is not ReleaseDecisionContext:
            raise _fail(ReleasePlanCode.PROFILE, "decision context is unsupported")
        return ReleaseDecisionContext(
            release_profile=context.release_profile,
            target_branch=context.target_branch,
            candidate=context.candidate,
            certificate=context.certificate,
            config=context.config,
            toolchain=context.toolchain,
            command=context.command,
            artifact_contract=context.artifact_contract,
            resource_profile=context.resource_profile,
            retry_policy=context.retry_policy,
            retry_count=context.retry_count,
            timeout_policy=context.timeout_policy,
            timeout_seconds=context.timeout_seconds,
        )
    except ReleasePlanError:
        raise
    except _UNTRUSTED_DATA_ERRORS:
        raise _fail(
            ReleasePlanCode.PROFILE, "decision context could not be validated"
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
        if type(plan.decision_context) is not ReleaseDecisionContext:
            raise _fail(ReleasePlanCode.PROFILE, "decision context is not frozen")
        decision_context = _revalidate_decision_context(plan.decision_context)
        source_graph = _snapshot_graph(plan.graph)
        source_selection = _snapshot_selection(plan.selection)
        if source_graph.digest != graph_digest:
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT, "frozen graph evidence is not bound"
            )
        if source_selection.digest != selection_digest:
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "frozen selection evidence is not bound",
            )
        if (
            source_selection.source_graph.canonical_payload()
            != source_graph.canonical_payload()
        ):
            raise _fail(ReleasePlanCode.GRAPH_DRIFT, "frozen selection source drifted")
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
        if (
            selected != source_selection.selected_project_ids
            or project_values != source_selection.projects
        ):
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "frozen projects do not match selection evidence",
            )
        packages = _package_map(project_values)
        edge_values = _bounded_tuple(plan.edges, "plan edges", max_items=MAX_EDGES)
        if any(type(edge) is not DependencyEdge for edge in edge_values):
            raise _fail(
                ReleasePlanCode.INVALID_INPUT, "plan edges must be C-11 records"
            )
        edge_tuple = tuple(
            _revalidate_edge(edge)
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
        if edge_tuple != source_selection.edges:
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "frozen edges do not match selection evidence",
            )
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
        if tuple(groups) != source_selection.parallel_groups:
            raise _fail(
                ReleasePlanCode.SELECTION_DRIFT,
                "frozen groups do not match selection evidence",
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
        version_plan.validate_against(source_graph, source_selection)
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
        if any(
            stage.decision_digest != decision_context.digest
            or stage.resource_profile != decision_context.resource_profile
            or stage.retry_policy is not decision_context.retry_policy
            or stage.retry_count != decision_context.retry_count
            or stage.timeout_policy is not decision_context.timeout_policy
            or stage.timeout_seconds != decision_context.timeout_seconds
            for stage in stages
        ):
            raise _fail(
                ReleasePlanCode.DIGEST,
                "stage decision evidence is not bound to the frozen plan",
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
        expected_stages = _derive_stage_sequence(
            selected=selected,
            project_map=project_map,
            project_edges=project_edges,
            groups=tuple(groups),
            base_sha=base_sha,
            generation_id=generation_id,
            graph_digest=graph_digest,
            selection_digest=selection_digest,
            version_plan_digest=version_plan_digest,
            version_preview_digests=version_digests,
            floor_preview_digests=floor_digests,
            validation=_profile_map(validation, ProfileKind.VALIDATION),
            build=_profile_map(build, ProfileKind.BUILD),
            decision_context=decision_context,
            consent=consent,
        )
        if stages != expected_stages:
            raise _fail(
                ReleasePlanCode.STAGE_DEPENDENCY,
                "stage composition does not match frozen source evidence",
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
    except _UNTRUSTED_DATA_ERRORS:
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
    decision_context: ReleaseDecisionContext,
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
        decision_digest=decision_context.digest,
        resource_profile=decision_context.resource_profile,
        retry_policy=decision_context.retry_policy,
        retry_count=decision_context.retry_count,
        timeout_policy=decision_context.timeout_policy,
        timeout_seconds=decision_context.timeout_seconds,
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
        decision_digest=decision_context.digest,
        resource_profile=decision_context.resource_profile,
        retry_policy=decision_context.retry_policy,
        retry_count=decision_context.retry_count,
        timeout_policy=decision_context.timeout_policy,
        timeout_seconds=decision_context.timeout_seconds,
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
    source_sha: str | None = None
    base_sha: str | None = None
    generation_id: str = ""
    validation_profiles: object | None = None
    build_profiles: object | None = None
    push_consent: PushConsentReference | None = None
    include_push: bool | None = None
    allow_push: bool | None = None
    decision_context: ReleaseDecisionContext | None = None

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
        graph = _snapshot_graph(self.graph)
        selection = _snapshot_selection(self.selection)
        frozen_version_plan = _revalidate_version_plan(self.version_plan)
        if (
            graph.digest != selection.source_graph.digest
            or graph.canonical_payload() != selection.source_graph.canonical_payload()
        ):
            raise _fail(
                ReleasePlanCode.GRAPH_DRIFT, "input graph does not match selection"
            )
        object.__setattr__(self, "graph", graph)
        object.__setattr__(self, "selection", selection)
        workspace = _strict_opaque(
            self.workspace_id, "workspace ID", max_length=MAX_STRING_LENGTH
        )
        object.__setattr__(self, "workspace_id", workspace)
        if self.source_sha is None:
            raise _fail(ReleasePlanCode.SOURCE_SHA, "source SHA is required")
        if self.base_sha is None:
            raise _fail(ReleasePlanCode.BASE_SHA, "base SHA is required")
        source = _strict_sha(
            self.source_sha, "source SHA", code=ReleasePlanCode.SOURCE_SHA
        )
        base = _strict_sha(self.base_sha, "base SHA", code=ReleasePlanCode.BASE_SHA)
        object.__setattr__(self, "source_sha", source)
        object.__setattr__(self, "base_sha", base)
        generation = _strict_opaque(
            self.generation_id,
            "generation ID",
            max_length=MAX_GENERATION_LENGTH,
        )
        object.__setattr__(self, "generation_id", generation)
        if self.include_push is not None:
            _strict_bool(self.include_push, "push inclusion flag")
        if self.allow_push is not None:
            _strict_bool(self.allow_push, "push authorization flag")
        consent = _revalidate_consent(self.push_consent)
        object.__setattr__(self, "push_consent", consent)
        decisions = (
            None
            if self.decision_context is None
            else _revalidate_decision_context(self.decision_context)
        )
        object.__setattr__(self, "decision_context", decisions)
        object.__setattr__(self, "version_plan", frozen_version_plan)
        # Materialize the two profile collections now, rather than retaining a
        # caller-owned dict/list.  This also makes missing profiles explicit via
        # deterministic defaults.
        selected = _canonical_project_ids(
            self.selection.selected_project_ids, "selected project IDs"
        )
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
            "allow_push": self.allow_push,
            "decision_context": (
                self.decision_context.canonical_payload()
                if self.decision_context is not None
                else None
            ),
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
    decision_context: ReleaseDecisionContext | None = None,
    release_profile: object | None = None,
    target_branch: object | None = None,
    candidate: object | None = None,
    certificate: object | None = None,
    config: object | None = None,
    toolchain: object | None = None,
    command: object | None = None,
    artifact_contract: object | None = None,
    resource_profile: object | None = None,
    retry_policy: object | None = None,
    retry_count: object | None = None,
    timeout_policy: object | None = None,
    timeout_seconds: object | None = None,
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
                base_sha=request.base_sha,
                generation_id=request.generation_id,
                validation_profiles=request.validation_profiles,
                build_profiles=request.build_profiles,
                push_consent=request.push_consent,
                include_push=request.include_push,
                allow_push=request.allow_push,
                decision_context=request.decision_context,
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
        graph = _snapshot_graph(graph)
        selection = _snapshot_selection(selection)
        version_plan = _revalidate_version_plan(version_plan)
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
        if source_sha is None:
            raise _fail(ReleasePlanCode.SOURCE_SHA, "source SHA is required")
        if base_sha is None:
            raise _fail(ReleasePlanCode.BASE_SHA, "base SHA is required")
        source = _strict_sha(source_sha, "source SHA", code=ReleasePlanCode.SOURCE_SHA)
        base = _strict_sha(base_sha, "base SHA", code=ReleasePlanCode.BASE_SHA)
        generation = _strict_opaque(
            generation_id, "generation ID", max_length=MAX_GENERATION_LENGTH
        )
        decisions = _normalize_decision_context(
            decision_context,
            release_profile=release_profile,
            target_branch=target_branch,
            candidate=candidate,
            certificate=certificate,
            config=config,
            toolchain=toolchain,
            command=command,
            artifact_contract=artifact_contract,
            resource_profile=resource_profile,
            retry_policy=retry_policy,
            retry_count=retry_count,
            timeout_policy=timeout_policy,
            timeout_seconds=timeout_seconds,
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
        consent_aliases = (push_consent, consent_reference, consent_ref)
        consent_candidates = tuple(item for item in consent_aliases if item is not None)
        if any(type(item) is not PushConsentReference for item in consent_candidates):
            raise _fail(
                ReleasePlanCode.CONSENT, "push consent must be immutable evidence"
            )
        # Rebuild every exact-type consent before comparing aliases.  A forged
        # dataclass shell may carry hostile scalar fields whose hash/equality
        # methods must never run during alias conflict detection.
        consent_candidates = tuple(
            cast(PushConsentReference, _revalidate_consent(item))
            for item in consent_candidates
        )
        if len(set(consent_candidates)) > 1:
            raise _fail(ReleasePlanCode.CONFLICT, "push consent references conflict")
        consent = consent_candidates[0] if consent_candidates else None
        if include_push is not None and type(include_push) is not bool:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push inclusion flag must be boolean"
            )
        if allow_push is not None and type(allow_push) is not bool:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT, "push authorization flag must be boolean"
            )
        if (
            include_push is not None
            and allow_push is not None
            and include_push != allow_push
        ):
            raise _fail(ReleasePlanCode.CONFLICT, "push flags conflict")
        if (include_push is False or allow_push is False) and consent is not None:
            raise _fail(
                ReleasePlanCode.CONFLICT,
                "explicit push exclusion conflicts with immutable consent",
            )
        if include_push is not None:
            requested_push = include_push
        elif allow_push is not None:
            requested_push = allow_push
        else:
            requested_push = consent is not None
        if requested_push and consent is None:
            raise _fail(
                ReleasePlanCode.PUSH_CONSENT,
                "push requires an immutable consent reference",
            )
        accepted_consent = consent if requested_push else None
        version_digests = tuple(
            sorted(
                _preview_digest(preview) for preview in version_plan.version_previews
            )
        )
        floor_digests = tuple(
            sorted(_preview_digest(preview) for preview in version_plan.floor_previews)
        )
        stages = _derive_stage_sequence(
            selected=selected,
            project_map=project_map,
            project_edges=project_edges,
            groups=groups,
            base_sha=base,
            generation_id=generation,
            graph_digest=graph.digest,
            selection_digest=selection.digest,
            version_plan_digest=version_plan.plan_digest,
            version_preview_digests=version_digests,
            floor_preview_digests=floor_digests,
            validation=validation_map,
            build=build_map,
            decision_context=decisions,
            consent=accepted_consent,
        )
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
            "source_graph": {
                **graph.canonical_payload(),
                "digest": graph.digest,
            },
            "selection": selection.canonical_payload(include_digest=True),
            "validation_profiles": tuple(
                binding.canonical_payload() for binding in validation
            ),
            "build_profiles": tuple(binding.canonical_payload() for binding in build),
            "decision_context": decisions.canonical_payload(),
            "stages": tuple(
                stage.canonical_payload(include_digests=True) for stage in stages
            ),
            "push_consent": accepted_consent.canonical_payload()
            if accepted_consent
            else None,
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
            graph=graph,
            selection=selection,
            decision_context=decisions,
            push_consent=accepted_consent,
            plan_digest=digest,
        )
    except ReleasePlanError:
        raise
    except VersionPlanningError:
        raise _fail(
            ReleasePlanCode.VERSION_PLAN_DRIFT,
            "version plan evidence could not be validated",
        ) from None
    except _UNTRUSTED_DATA_ERRORS:
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
    "ArtifactContractReference",
    "CandidateReference",
    "CertificateReference",
    "C11_FROZEN_PLAN_VERSION",
    "CommandReference",
    "ConfigReference",
    "ConsentReference",
    "DecisionContext",
    "DecisionReference",
    "FailurePolicy",
    "FrozenPlan",
    "FrozenPlanCode",
    "FrozenPlanError",
    "FrozenPlanInput",
    "FrozenReleasePlan",
    "FrozenReleasePlanInput",
    "FrozenWorkspaceReleasePlan",
    "ImmutableDigestReference",
    "OpaqueDigestReference",
    "ProfileBinding",
    "ProfileKind",
    "PushConsent",
    "PushConsentReference",
    "ReleaseDecisionContext",
    "ReleasePlanDecisions",
    "ReleaseProfile",
    "ReleaseProfileReference",
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
    "RetryPolicy",
    "ResourceProfileReference",
    "TimeoutPolicy",
    "ToolchainReference",
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
