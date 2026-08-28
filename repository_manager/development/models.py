"""Typed v1 contracts for Repository Manager distributed development.

This module is deliberately policy-free: it describes durable inputs,
correlations, evidence, and results, but it never opens a repository, resolves
an inventory host, starts a process, or writes a WorkItem.  Later lanes own
those effects and consume these models at their boundaries.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Annotated, ClassVar, Literal

from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .contracts.version import CONTRACT_VERSION, ContractVersion
from .enums import (
    BuildOutcome,
    CandidateState,
    EvidenceOutcome,
    ExecutionOutcome,
    FailureClass,
    GenerationState,
    JobState,
    LandingOutcome,
    LaneState,
    OperationKind,
    RefusalCode,
    ReleasePlanState,
    ReservationState,
    TargetKind,
    ValidationStage,
)
from .serialization import canonical_digest as _canonical_digest
from .serialization import canonical_json as _canonical_json

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_FORBIDDEN_RE = re.compile(r"[\x00-\x20~^:?*\\\[]")
_ENV_REF_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_.:/-]*$")
_UUID_PATTERN = (
    r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"
)
_JOB_ID_RE = re.compile(rf"^rmjob:{_UUID_PATTERN}$")
_WORK_ITEM_ID_RE = re.compile(rf"^workitem:repository_manager:{_UUID_PATTERN}$")
_MAX_OUTPUT_TAIL_BYTES = 64 * 1024
_MAX_ARTIFACT_BYTES = 1024 * 1024 * 1024


def _require_opaque_id(value: str) -> str:
    if not value or value.strip() != value:
        raise ValueError(
            "identifier must be non-blank and have no surrounding whitespace"
        )
    if any(ord(char) < 0x20 for char in value):
        raise ValueError("identifier must not contain control characters")
    return value


def _require_git_sha(value: str) -> str:
    if not _SHA_RE.fullmatch(value):
        raise ValueError("Git SHA must be exactly 40 lowercase hexadecimal characters")
    return value


def _require_digest(value: str) -> str:
    if not _DIGEST_RE.fullmatch(value):
        raise ValueError("digest must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_git_ref_shape(value: str) -> None:
    if not value or value.strip() != value:
        raise ValueError("Git ref must be non-blank and have no surrounding whitespace")
    if _SHA_RE.fullmatch(value):
        raise ValueError("a full Git SHA is not accepted where a named ref is required")


def _require_git_ref_syntax(value: str) -> None:
    if value.startswith("-") or value.endswith(".") or value.endswith(".lock"):
        raise ValueError(f"invalid Git ref: {value!r}")
    if ".." in value or "@{" in value or "//" in value:
        raise ValueError(f"invalid Git ref: {value!r}")
    if _REF_FORBIDDEN_RE.search(value):
        raise ValueError(f"invalid Git ref: {value!r}")


def _require_git_ref_not_reserved(value: str) -> None:
    if value in {".", "..", "HEAD"}:
        raise ValueError(f"invalid or moving Git ref: {value!r}")


def _require_git_ref(value: str) -> str:
    _require_git_ref_shape(value)
    _require_git_ref_syntax(value)
    _require_git_ref_not_reserved(value)
    return value


def _require_absolute_path(value: str) -> str:
    path = Path(value)
    if not path.is_absolute():
        raise ValueError("path must be canonical and absolute")
    if ".." in path.parts:
        raise ValueError("path must not contain a parent traversal component")
    resolved = path.resolve(strict=False)
    if resolved != path:
        raise ValueError(f"path is not canonical: {value!r}")
    return str(path)


def _require_relative_path(value: str) -> str:
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("relative path must not be absolute or escape its root")
    if not value or value.strip() != value:
        raise ValueError(
            "relative path must be non-blank and have no surrounding whitespace"
        )
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return value


def _as_tuple(value: object, *, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence, not a string")
    if not isinstance(value, Iterable):
        raise ValueError(f"{field_name} must be a sequence")
    return tuple(value)


def _validate_string_tuple(
    value: object, *, field_name: str, sort_values: bool = False
) -> tuple[str, ...]:
    values = _as_tuple(value, field_name=field_name)
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        result.append(_require_opaque_id(item))
    if sort_values:
        result.sort()
    return tuple(result)


def _path_within_roots(path: str, roots: tuple[str, ...]) -> bool:
    candidate = Path(path)
    return any(
        candidate == Path(root) or Path(root) in candidate.parents for root in roots
    )


OpaqueId = Annotated[StrictStr, AfterValidator(_require_opaque_id)]
GitSha = Annotated[StrictStr, AfterValidator(_require_git_sha)]
Digest = Annotated[StrictStr, AfterValidator(_require_digest)]
GitRef = Annotated[StrictStr, AfterValidator(_require_git_ref)]
AbsolutePath = Annotated[StrictStr, AfterValidator(_require_absolute_path)]
RelativePath = Annotated[StrictStr, AfterValidator(_require_relative_path)]
UtcDateTime = Annotated[datetime, AfterValidator(_require_utc)]
JobId = Annotated[
    StrictStr,
    AfterValidator(lambda value: _validate_prefixed_id(value, _JOB_ID_RE, "job")),
]
WorkItemId = Annotated[
    StrictStr,
    AfterValidator(
        lambda value: _validate_prefixed_id(value, _WORK_ITEM_ID_RE, "WorkItem")
    ),
]


def _validate_prefixed_id(value: str, pattern: re.Pattern[str], label: str) -> str:
    _require_opaque_id(value)
    if not pattern.fullmatch(value):
        raise ValueError(f"{label} ID must use its full namespaced UUID form")
    return value


class ContractModel(BaseModel):
    """Base model with additive versioning and deterministic serialization."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_assignment=True,
        use_enum_values=False,
    )

    contract_version: Literal["1"] = Field(default=CONTRACT_VERSION, frozen=True)
    CONTRACT_NAME: ClassVar[str] = "repository-development"
    CONTRACT_VERSION: ClassVar[ContractVersion] = CONTRACT_VERSION

    def canonical_payload(self) -> dict[str, object]:
        """Return the versioned JSON payload used for persistence and identity."""

        return self.model_dump(mode="json", exclude_none=False)

    def canonical_json(self) -> str:
        """Serialize this model with stable key ordering."""

        return _canonical_json(self.canonical_payload())

    def digest(self) -> str:
        """Return the full SHA-256 identity of this model's canonical payload."""

        return _canonical_digest(self.canonical_payload())

    @classmethod
    def from_canonical_json(cls, payload: str | bytes | bytearray) -> ContractModel:
        """Load a model while preserving the version/extra-field checks."""

        return cls.model_validate_json(payload)


class TargetPolicy(ContractModel):
    """A local target or an authorized tunnel-manager inventory alias."""

    kind: TargetKind = TargetKind.LOCAL
    alias: OpaqueId | None = None
    capability_labels: tuple[StrictStr, ...] = ()

    @field_validator("capability_labels", mode="before")
    @classmethod
    def normalize_labels(cls, value: object) -> tuple[str, ...]:
        values = _as_tuple(value, field_name="capability_labels")
        labels: list[str] = []
        for item in values:
            if not isinstance(item, str) or not item or item.strip() != item:
                raise ValueError("capability labels must be non-blank strings")
            labels.append(item)
        return tuple(sorted(labels))

    @model_validator(mode="after")
    def validate_target(self) -> TargetPolicy:
        if self.kind == TargetKind.LOCAL and self.alias is not None:
            raise ValueError("local target must not carry an inventory alias")
        if self.kind == TargetKind.INVENTORY_ALIAS:
            if not self.alias:
                raise ValueError("remote target requires an inventory alias")
            if any(token in self.alias for token in ("/", "\\", "@", ":")):
                raise ValueError(
                    "remote target accepts an inventory alias only, not a host, URL, "
                    "username, key, or proxy value"
                )
        return self


class RepositoryIdentity(ContractModel):
    """Repository identity independent of display basename or directory."""

    repository_id: OpaqueId
    canonical_path: AbsolutePath
    configured_roots: tuple[AbsolutePath, ...] = ()
    origin: StrictStr | None = None

    @field_validator("configured_roots", mode="before")
    @classmethod
    def normalize_roots(cls, value: object) -> tuple[object, ...]:
        return _as_tuple(value, field_name="configured_roots")

    @field_validator("origin")
    @classmethod
    def reject_origin_credentials(cls, value: str | None) -> str | None:
        if value is not None and ("@" in value or "password" in value.lower()):
            raise ValueError("repository origin must not embed credentials")
        return value

    @model_validator(mode="after")
    def validate_root(self) -> RepositoryIdentity:
        if self.configured_roots and not _path_within_roots(
            self.canonical_path, self.configured_roots
        ):
            raise ValueError("canonical repository path is outside configured roots")
        return self


class ValidationPolicy(ContractModel):
    """Stages and path policy requested for one development operation."""

    stages: tuple[ValidationStage, ...] = (ValidationStage.FEEDBACK,)
    differential: StrictBool = False
    changed_paths: tuple[RelativePath, ...] = ()
    blocking: StrictBool = True

    @field_validator("stages", mode="before")
    @classmethod
    def normalize_stages(cls, value: object) -> tuple[object, ...]:
        return _as_tuple(value, field_name="stages")

    @field_validator("changed_paths", mode="before")
    @classmethod
    def normalize_paths(cls, value: object) -> tuple[object, ...]:
        values = _as_tuple(value, field_name="changed_paths")
        paths: list[str] = []
        for item in values:
            if not isinstance(item, str):
                raise ValueError("changed_paths entries must be strings")
            paths.append(_require_relative_path(item))
        return tuple(sorted(paths))

    @model_validator(mode="after")
    def validate_stage_order(self) -> ValidationPolicy:
        if not self.stages:
            raise ValueError("validation policy must request at least one stage")
        order = {stage: index for index, stage in enumerate(ValidationStage)}
        if len(set(self.stages)) != len(self.stages):
            raise ValueError("validation stages must not be duplicated")
        if tuple(order[stage] for stage in self.stages) != tuple(
            sorted(order[stage] for stage in self.stages)
        ):
            raise ValueError("validation stages must be in lifecycle order")
        if self.differential and not self.changed_paths:
            raise ValueError("differential validation requires changed_paths")
        return self


class ConsentPolicy(ContractModel):
    """Explicit operator consent for effects beyond lane-local development."""

    allow_push: StrictBool = False
    allow_destructive_cleanup: StrictBool = False
    risk_acknowledged: StrictBool = False
    risk_marker: OpaqueId | None = None

    @model_validator(mode="after")
    def validate_risk(self) -> ConsentPolicy:
        if (self.allow_push or self.allow_destructive_cleanup) and not (
            self.risk_acknowledged and self.risk_marker
        ):
            raise ValueError(
                "push or destructive cleanup requires a risk acknowledgement and marker"
            )
        if self.risk_marker and not self.risk_acknowledged:
            raise ValueError("risk_marker cannot be supplied without acknowledgement")
        return self


class ResourceRequest(ContractModel):
    """Weighted admission request; estimates are not reservations."""

    resource_class: OpaqueId = "light-check"
    concurrency_key: OpaqueId = "light-check"
    cpu_weight: StrictInt = Field(default=1, ge=1, le=1000)
    memory_mib: StrictInt = Field(default=256, ge=1, le=1_048_576)
    disk_mib: StrictInt = Field(default=256, ge=1, le=10_485_760)
    process_slots: StrictInt = Field(default=1, ge=1, le=256)
    host_labels: tuple[StrictStr, ...] = ()
    preferred_target: TargetPolicy = Field(default_factory=TargetPolicy)
    required_target: TargetPolicy | None = None
    anti_affinity: tuple[OpaqueId, ...] = ()
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    fairness_group: OpaqueId = "default"
    queue_deadline: UtcDateTime | None = None
    disk_low_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    disk_high_watermark_mib: StrictInt | None = Field(default=None, ge=0)

    @field_validator("host_labels", mode="before")
    @classmethod
    def normalize_host_labels(cls, value: object) -> tuple[str, ...]:
        values = _as_tuple(value, field_name="host_labels")
        result: list[str] = []
        for item in values:
            if not isinstance(item, str) or not item or item.strip() != item:
                raise ValueError("host labels must be non-blank strings")
            result.append(item)
        return tuple(sorted(result))

    @field_validator("anti_affinity", mode="before")
    @classmethod
    def normalize_anti_affinity(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(
            value, field_name="anti_affinity", sort_values=True
        )

    @model_validator(mode="after")
    def validate_watermarks(self) -> ResourceRequest:
        low = self.disk_low_watermark_mib
        high = self.disk_high_watermark_mib
        if low is not None and high is not None and low > high:
            raise ValueError("disk low watermark must not exceed high watermark")
        if (
            self.required_target is not None
            and self.preferred_target.kind == TargetKind.LOCAL
            and self.required_target.kind == TargetKind.INVENTORY_ALIAS
            and self.preferred_target.alias is not None
        ):
            raise ValueError(
                "preferred and required targets cannot express conflicting forms"
            )
        return self


class CapacitySnapshot(ContractModel):
    """Admission snapshot captured with a reservation decision."""

    cpu_weight_total: StrictInt = Field(ge=0)
    cpu_weight_available: StrictInt = Field(ge=0)
    memory_mib_total: StrictInt = Field(ge=0)
    memory_mib_available: StrictInt = Field(ge=0)
    disk_mib_total: StrictInt = Field(ge=0)
    disk_mib_available: StrictInt = Field(ge=0)
    process_slots_total: StrictInt = Field(ge=0)
    process_slots_available: StrictInt = Field(ge=0)

    @model_validator(mode="after")
    def validate_available(self) -> CapacitySnapshot:
        pairs = (
            (self.cpu_weight_available, self.cpu_weight_total, "CPU"),
            (self.memory_mib_available, self.memory_mib_total, "memory"),
            (self.disk_mib_available, self.disk_mib_total, "disk"),
            (self.process_slots_available, self.process_slots_total, "process slots"),
        )
        for available, total, label in pairs:
            if available > total:
                raise ValueError(
                    f"available {label} capacity cannot exceed total capacity"
                )
        return self


class ResourceReservation(ContractModel):
    """Durable admission result, distinct from a resource estimate."""

    reservation_id: OpaqueId
    request: ResourceRequest
    selected_target: TargetPolicy
    fence: OpaqueId
    capacity: CapacitySnapshot
    state: ReservationState = ReservationState.RESERVED
    reason: StrictStr = ""
    reserved_at: UtcDateTime
    expires_at: UtcDateTime
    released_at: UtcDateTime | None = None

    @model_validator(mode="after")
    def validate_reservation(self) -> ResourceReservation:
        if self.expires_at <= self.reserved_at:
            raise ValueError("reservation expiry must be after reservation time")
        if self.state == ReservationState.REFUSED and not self.reason:
            raise ValueError("refused reservations require an actionable reason")
        if self.state == ReservationState.RELEASED and self.released_at is None:
            raise ValueError("released reservations require released_at")
        if self.state != ReservationState.RELEASED and self.released_at is not None:
            raise ValueError("released_at is only valid for released reservations")
        return self


class ArtifactReference(ContractModel):
    """Bounded content-addressed artifact metadata."""

    content_address: Digest
    relative_path: RelativePath
    size_bytes: StrictInt = Field(ge=0, le=_MAX_ARTIFACT_BYTES)
    media_type: OpaqueId
    producer_job_id: JobId | None = None


class LogReference(ContractModel):
    """Bounded log reference; full output stays outside the WorkItem record."""

    content_address: Digest
    relative_path: RelativePath
    size_bytes: StrictInt = Field(ge=0, le=_MAX_ARTIFACT_BYTES)
    tail_bytes: StrictInt = Field(ge=0, le=_MAX_OUTPUT_TAIL_BYTES)
    media_type: OpaqueId = "text/plain"


class ExecutionCommand(ContractModel):
    """A fixed argv command with bounded, canonical execution inputs."""

    argv: tuple[StrictStr, ...]
    workdir: AbsolutePath
    environment_refs: tuple[StrictStr, ...] = ()
    timeout_seconds: StrictInt = Field(default=3600, ge=1, le=86_400)
    max_stdout_bytes: StrictInt = Field(
        default=_MAX_OUTPUT_TAIL_BYTES, ge=0, le=_MAX_ARTIFACT_BYTES
    )
    max_stderr_bytes: StrictInt = Field(
        default=_MAX_OUTPUT_TAIL_BYTES, ge=0, le=_MAX_ARTIFACT_BYTES
    )
    max_artifact_bytes: StrictInt = Field(
        default=_MAX_ARTIFACT_BYTES, ge=0, le=_MAX_ARTIFACT_BYTES
    )
    heartbeat_interval_seconds: StrictInt = Field(default=30, ge=1, le=3600)
    cancellation_channel: OpaqueId | None = None

    @field_validator("argv", mode="before")
    @classmethod
    def normalize_argv(cls, value: object) -> tuple[str, ...]:
        values = _as_tuple(value, field_name="argv")
        if not values:
            raise ValueError("execution argv must not be empty")
        result: list[str] = []
        for item in values:
            if not isinstance(item, str) or not item:
                raise ValueError("execution argv entries must be non-empty strings")
            if "\x00" in item:
                raise ValueError("execution argv must not contain NUL bytes")
            result.append(item)
        return tuple(result)

    @field_validator("environment_refs", mode="before")
    @classmethod
    def normalize_environment_refs(cls, value: object) -> tuple[str, ...]:
        values = _as_tuple(value, field_name="environment_refs")
        result: list[str] = []
        for item in values:
            if not isinstance(item, str) or not _ENV_REF_RE.fullmatch(item):
                raise ValueError(
                    "environment_refs must contain approved reference names, not raw values"
                )
            result.append(item)
        return tuple(sorted(result))


class LeaseRecord(ContractModel):
    """WorkItem lease/fence information attached to a repository job."""

    owner: OpaqueId
    fence: OpaqueId
    attempt: StrictInt = Field(ge=1)
    heartbeat_at: UtcDateTime
    expires_at: UtcDateTime
    checkpoint: OpaqueId | None = None

    @model_validator(mode="after")
    def validate_lease(self) -> LeaseRecord:
        if self.expires_at <= self.heartbeat_at:
            raise ValueError("lease expiry must be after heartbeat")
        return self


class ExecutionResult(ContractModel):
    """Structured result for a fixed-argv command."""

    command_id: OpaqueId
    outcome: ExecutionOutcome
    exit_code: StrictInt | None = None
    signal: StrictInt | None = Field(default=None, ge=1)
    started_at: UtcDateTime
    finished_at: UtcDateTime
    duration_ms: StrictInt = Field(ge=0)
    worker_id: OpaqueId
    fence: OpaqueId
    stdout_tail: StrictStr = ""
    stderr_tail: StrictStr = ""
    log_refs: tuple[LogReference, ...] = ()
    artifact_refs: tuple[ArtifactReference, ...] = ()
    failure_class: FailureClass | None = None
    cleanup_ok: StrictBool = True

    @field_validator("stdout_tail", "stderr_tail")
    @classmethod
    def bound_output_tail(cls, value: str) -> str:
        if len(value.encode("utf-8")) > _MAX_OUTPUT_TAIL_BYTES:
            raise ValueError("stdout/stderr tails exceed the bounded output limit")
        return value

    @model_validator(mode="after")
    def validate_result(self) -> ExecutionResult:
        if self.finished_at < self.started_at:
            raise ValueError("execution finished_at cannot precede started_at")
        if self.exit_code is not None and self.signal is not None:
            raise ValueError("execution cannot report both an exit code and a signal")
        if self.outcome == ExecutionOutcome.SUCCEEDED:
            if self.exit_code != 0 or self.failure_class is not None:
                raise ValueError(
                    "successful execution requires exit_code=0 and no failure"
                )
        elif (
            self.outcome
            in {
                ExecutionOutcome.FAILED,
                ExecutionOutcome.REFUSED,
            }
            and self.failure_class is None
        ):
            raise ValueError("failed or refused execution requires a failure class")
        return self


class ValidationEvidence(ContractModel):
    """Evidence for one stage/gate against one immutable tree."""

    evidence_id: OpaqueId
    stage: ValidationStage
    generation_id: OpaqueId | None = None
    tree_sha: GitSha
    gate_config_digest: Digest
    command_digest: Digest
    target: TargetPolicy
    host_id: OpaqueId
    toolchain_digest: Digest
    started_at: UtcDateTime
    finished_at: UtcDateTime
    outcome: EvidenceOutcome
    baseline_tree_sha: GitSha | None = None
    differential: StrictBool = False
    failure_ids: tuple[OpaqueId, ...] = ()
    log_refs: tuple[LogReference, ...] = ()
    artifact_refs: tuple[ArtifactReference, ...] = ()

    @field_validator("failure_ids", mode="before")
    @classmethod
    def normalize_failure_ids(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(value, field_name="failure_ids", sort_values=True)

    @model_validator(mode="after")
    def validate_evidence(self) -> ValidationEvidence:
        if self.finished_at < self.started_at:
            raise ValueError("evidence finished_at cannot precede started_at")
        if self.differential and self.baseline_tree_sha is None:
            raise ValueError("differential evidence requires baseline_tree_sha")
        if (
            self.stage == ValidationStage.CERTIFICATION
            and self.outcome == EvidenceOutcome.PASSED
        ):
            if self.generation_id is None:
                raise ValueError(
                    "passed certification requires the exact generation_id"
                )
        if self.outcome == EvidenceOutcome.PASSED and self.failure_ids:
            raise ValueError("passed evidence cannot carry failure IDs")
        return self


class CandidateVersion(ContractModel):
    """Immutable candidate membership item in generation order."""

    candidate_id: OpaqueId
    version: StrictInt = Field(ge=1)
    candidate_sha: GitSha


class Candidate(ContractModel):
    """One immutable branch submission offered for generation formation."""

    candidate_id: OpaqueId
    version: StrictInt = Field(ge=1)
    repository: RepositoryIdentity
    branch: GitRef
    candidate_sha: GitSha
    base_sha: GitSha
    lane_id: OpaqueId
    owner_id: OpaqueId
    config_digest: Digest
    concept_claims: tuple[OpaqueId, ...] = ()
    enqueued_at: UtcDateTime
    state: CandidateState = CandidateState.QUEUED
    generation_id: OpaqueId | None = None
    reason: StrictStr = ""

    @field_validator("concept_claims", mode="before")
    @classmethod
    def normalize_concepts(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(
            value, field_name="concept_claims", sort_values=True
        )

    @model_validator(mode="after")
    def validate_candidate_state(self) -> Candidate:
        if (
            self.state in {CandidateState.LANDING, CandidateState.LANDED}
            and self.generation_id is None
        ):
            raise ValueError("landing candidate requires its generation_id")
        if (
            self.state
            in {
                CandidateState.REJECTED,
                CandidateState.WITHDRAWN,
                CandidateState.FAILED,
            }
            and not self.reason
        ):
            raise ValueError(
                "rejected, withdrawn, and failed candidates require a reason"
            )
        if (
            self.state in {CandidateState.QUEUED, CandidateState.VALIDATING}
            and self.generation_id
        ):
            raise ValueError(
                "queued or validating candidate cannot already belong to a generation"
            )
        return self


class Generation(ContractModel):
    """Sealed ordered candidate set with exact certification/landing evidence."""

    generation_id: OpaqueId
    repository: RepositoryIdentity
    target_branch: GitRef
    target: TargetPolicy = Field(default_factory=TargetPolicy)
    base_sha: GitSha
    expected_landing_base_sha: GitSha
    candidate_versions: tuple[CandidateVersion, ...]
    config_digest: Digest
    toolchain_digest: Digest
    state: GenerationState = GenerationState.OPEN
    sealed_at: UtcDateTime | None = None
    synthetic_commit_sha: GitSha | None = None
    tree_sha: GitSha | None = None
    validation_evidence_ids: tuple[OpaqueId, ...] = ()
    build_artifact_refs: tuple[ArtifactReference, ...] = ()
    bisection_lineage: tuple[OpaqueId, ...] = ()
    landing_fence: OpaqueId | None = None
    landing_result: LandingOutcome | None = None
    reason: StrictStr = ""

    @field_validator("candidate_versions", mode="before")
    @classmethod
    def normalize_candidates(cls, value: object) -> tuple[object, ...]:
        values = _as_tuple(value, field_name="candidate_versions")
        if not values:
            raise ValueError("a generation must contain at least one candidate")
        return values

    @field_validator("validation_evidence_ids", "bisection_lineage", mode="before")
    @classmethod
    def normalize_id_sequences(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(
            value, field_name="generation IDs", sort_values=True
        )

    @classmethod
    def derive_id(
        cls,
        *,
        repository_id: str,
        target_branch: str,
        base_sha: str,
        candidate_versions: tuple[CandidateVersion, ...] | list[CandidateVersion],
        config_digest: str,
        toolchain_digest: str,
    ) -> str:
        """Derive a stable generation ID from immutable membership inputs."""

        payload = {
            "repository_id": repository_id,
            "target_branch": target_branch,
            "base_sha": base_sha,
            "candidate_versions": candidate_versions,
            "config_digest": config_digest,
            "toolchain_digest": toolchain_digest,
        }
        return f"generation:{_canonical_digest(payload)}"

    @model_validator(mode="after")
    def validate_generation_state(self) -> Generation:
        _validate_generation_candidate_ids(self)
        _validate_generation_seal(self)
        _validate_generation_certification(self)
        _validate_generation_landing(self)
        return self


_GENERATION_CERTIFIED_STATES = {
    GenerationState.CERTIFIED,
    GenerationState.LANDING,
    GenerationState.LANDED,
}


def _validate_generation_candidate_ids(generation: Generation) -> None:
    ids = [item.candidate_id for item in generation.candidate_versions]
    if len(set(ids)) != len(ids):
        raise ValueError("generation candidate IDs must be unique")


def _validate_generation_seal(generation: Generation) -> None:
    if generation.state != GenerationState.OPEN and generation.sealed_at is None:
        raise ValueError("sealed or terminal generations require sealed_at")
    if generation.state == GenerationState.OPEN and generation.sealed_at is not None:
        raise ValueError("open generation cannot carry a seal time")


def _validate_generation_certification(generation: Generation) -> None:
    if generation.state in _GENERATION_CERTIFIED_STATES and generation.tree_sha is None:
        raise ValueError("certified or landing generations require tree_sha")
    if (
        generation.state in _GENERATION_CERTIFIED_STATES
        and not generation.validation_evidence_ids
    ):
        raise ValueError("certified or landing generations require validation evidence")


def _validate_generation_landing(generation: Generation) -> None:
    if generation.state in {GenerationState.LANDING, GenerationState.LANDED}:
        if generation.landing_fence is None:
            raise ValueError("landing generations require a landing fence")
    if (
        generation.state == GenerationState.LANDED
        and generation.landing_result != LandingOutcome.LANDED
    ):
        raise ValueError("landed generation requires a landed result")
    if generation.state == GenerationState.REJECTED and not generation.reason:
        raise ValueError("rejected generation requires a reason")


class LaneReference(ContractModel):
    """Durable identity and quota state for one development worktree."""

    lane_id: OpaqueId
    repository: RepositoryIdentity
    branch: GitRef
    base_sha: GitSha
    worktree_path: AbsolutePath
    owner_id: OpaqueId
    session_id: OpaqueId
    host_target: TargetPolicy = Field(default_factory=TargetPolicy)
    created_at: UtcDateTime
    heartbeat_at: UtcDateTime
    expires_at: UtcDateTime
    disk_budget_mib: StrictInt = Field(ge=1)
    observed_disk_mib: StrictInt = Field(default=0, ge=0)
    concept_ids: tuple[OpaqueId, ...] = ()
    active_job_ids: tuple[JobId, ...] = ()
    active_candidate_id: OpaqueId | None = None
    cleanup_anchors: tuple[OpaqueId, ...] = ()
    state: LaneState = LaneState.ALLOCATING

    @field_validator("concept_ids", "active_job_ids", "cleanup_anchors", mode="before")
    @classmethod
    def normalize_lane_ids(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(value, field_name="lane IDs", sort_values=True)

    @model_validator(mode="after")
    def validate_lane(self) -> LaneReference:
        if self.expires_at <= self.heartbeat_at:
            raise ValueError("lane expiry must be after heartbeat")
        if self.repository.configured_roots and not _path_within_roots(
            self.worktree_path, self.repository.configured_roots
        ):
            raise ValueError("lane worktree path is outside configured roots")
        if self.state == LaneState.LANDED and self.active_job_ids:
            raise ValueError("landed lane cannot retain active jobs")
        return self


class DevelopmentRequest(ContractModel):
    """C-01 request consumed by later application and WorkItem lanes."""

    request_id: OpaqueId
    idempotency_key: OpaqueId
    repository: RepositoryIdentity
    operation: OperationKind
    base_ref: GitRef
    base_sha: GitSha
    lane_id: OpaqueId | None = None
    candidate_id: OpaqueId | None = None
    generation_id: OpaqueId | None = None
    owner_id: OpaqueId
    session_id: OpaqueId
    tenant_id: OpaqueId
    fairness_group: OpaqueId
    dependencies: tuple[OpaqueId, ...] = ()
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    validation_policy: ValidationPolicy = Field(default_factory=ValidationPolicy)
    resources: ResourceRequest = Field(default_factory=ResourceRequest)
    target: TargetPolicy = Field(default_factory=TargetPolicy)
    consent: ConsentPolicy = Field(default_factory=ConsentPolicy)
    config_digest: Digest | None = None

    @field_validator("dependencies", mode="before")
    @classmethod
    def normalize_dependencies(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(
            value, field_name="dependencies", sort_values=True
        )

    @model_validator(mode="after")
    def validate_correlations(self) -> DevelopmentRequest:
        _validate_request_identifiers(self)
        _validate_request_operation_correlation(self)
        return self


def _validate_request_identifiers(request: DevelopmentRequest) -> None:
    if request.request_id == request.idempotency_key:
        raise ValueError(
            "request_id and idempotency_key must remain distinct identifiers"
        )


def _validate_request_lane_and_candidate(request: DevelopmentRequest) -> None:
    if (
        request.operation == OperationKind.LANE_ALLOCATE
        and request.generation_id is not None
    ):
        raise ValueError("lane allocation cannot target an existing generation")
    if request.operation == OperationKind.CANDIDATE_SUBMIT and not request.lane_id:
        raise ValueError("candidate submission requires a lane_id")


def _validate_request_generation_and_push(request: DevelopmentRequest) -> None:
    if (
        request.operation == OperationKind.GENERATION_CERTIFY
        and not request.generation_id
    ):
        raise ValueError("generation certification requires a generation_id")
    if request.operation == OperationKind.BRANCH_LAND and not request.generation_id:
        raise ValueError("branch landing requires a generation_id")
    if (
        request.operation in {OperationKind.RELEASE, OperationKind.WORKSPACE_PUSH}
        and not request.consent.allow_push
    ):
        raise ValueError("release or workspace push requires explicit push consent")


def _validate_request_operation_correlation(request: DevelopmentRequest) -> None:
    _validate_request_lane_and_candidate(request)
    _validate_request_generation_and_push(request)


class RepositoryJobResult(ContractModel):
    """C-02 repository extension projected onto one authoritative WorkItem."""

    job_id: JobId
    work_item_id: WorkItemId
    request_id: OpaqueId
    operation: OperationKind
    state: JobState
    repository: RepositoryIdentity
    target: TargetPolicy = Field(default_factory=TargetPolicy)
    lane_id: OpaqueId | None = None
    candidate_id: OpaqueId | None = None
    generation_id: OpaqueId | None = None
    input_digest: Digest
    config_digest: Digest
    lease: LeaseRecord | None = None
    attempt: StrictInt = Field(default=1, ge=1)
    retry_class: OpaqueId | None = None
    checkpoint: OpaqueId | None = None
    result: ExecutionResult | None = None
    failure_class: FailureClass | None = None
    refusal_code: RefusalCode | None = None
    log_refs: tuple[LogReference, ...] = ()
    artifact_refs: tuple[ArtifactReference, ...] = ()

    @model_validator(mode="after")
    def validate_job_result(self) -> RepositoryJobResult:
        _validate_job_result_codes(self)
        _validate_job_result_succeeded(self)
        _validate_job_result_lease(self)
        _validate_job_result_failed(self)
        _validate_job_result_nonterminal(self)
        _validate_job_result_cancelled(self)
        return self


def _validate_job_result_codes(job: RepositoryJobResult) -> None:
    if job.failure_class is not None and job.refusal_code is not None:
        raise ValueError("a job result cannot carry both failure and refusal codes")


def _validate_job_result_succeeded(job: RepositoryJobResult) -> None:
    if job.state == JobState.SUCCEEDED:
        if (
            job.result is None
            or job.result.outcome != ExecutionOutcome.SUCCEEDED
            or job.failure_class
            or job.refusal_code
        ):
            raise ValueError("succeeded job requires a result and no failure/refusal")


def _validate_job_result_lease(job: RepositoryJobResult) -> None:
    if job.state in {JobState.LEASED, JobState.RUNNING} and job.lease is None:
        raise ValueError("leased or running job requires lease/fence evidence")


def _validate_job_result_failed(job: RepositoryJobResult) -> None:
    if job.state in {JobState.FAILED, JobState.DEAD_LETTER}:
        if (
            job.result is None
            and job.failure_class is None
            and job.refusal_code is None
        ):
            raise ValueError(
                "failed job requires structured result or failure/refusal code"
            )


def _validate_job_result_nonterminal(job: RepositoryJobResult) -> None:
    if job.state in {
        JobState.SUBMITTED,
        JobState.READY,
        JobState.LEASED,
        JobState.RUNNING,
    } and (job.result or job.failure_class or job.refusal_code):
        raise ValueError("non-terminal job cannot carry terminal result fields")


def _validate_job_result_cancelled(job: RepositoryJobResult) -> None:
    if job.state == JobState.CANCELLED and job.refusal_code not in {
        None,
        RefusalCode.CANCELLED_DEADLINE,
    }:
        raise ValueError(
            "cancelled job must use the cancellation/deadline refusal code"
        )


class WorkspaceProject(ContractModel):
    """One selected workspace project and its immutable release inputs."""

    project_id: OpaqueId
    repository: RepositoryIdentity
    tree_sha: GitSha
    current_version: OpaqueId
    next_version: OpaqueId


class DependencyEdge(ContractModel):
    """One directed dependency edge in a frozen workspace release plan."""

    dependent_project_id: OpaqueId
    dependency_project_id: OpaqueId
    current_floor: OpaqueId
    next_floor: OpaqueId


class FloorRewrite(ContractModel):
    """One dependency floor rewrite planned for a release."""

    project_id: OpaqueId
    dependency_project_id: OpaqueId
    old_floor: OpaqueId
    new_floor: OpaqueId


class WorkspaceReleasePlan(ContractModel):
    """C-11 frozen workspace validation/version/build/push plan."""

    plan_id: OpaqueId
    workspace_id: OpaqueId
    selected_projects: tuple[OpaqueId, ...]
    projects: tuple[WorkspaceProject, ...]
    dependency_edges: tuple[DependencyEdge, ...] = ()
    floor_rewrites: tuple[FloorRewrite, ...] = ()
    validation_stages: tuple[ValidationStage, ...]
    build_job_ids: tuple[JobId, ...] = ()
    push_job_ids: tuple[JobId, ...] = ()
    parallel_groups: tuple[tuple[OpaqueId, ...], ...] = ()
    consent: ConsentPolicy = Field(default_factory=ConsentPolicy)
    plan_digest: Digest
    state: ReleasePlanState = ReleasePlanState.DRAFT
    created_at: UtcDateTime
    frozen_at: UtcDateTime | None = None

    @field_validator("selected_projects", mode="before")
    @classmethod
    def normalize_selected_projects(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(
            value, field_name="selected_projects", sort_values=True
        )

    @field_validator("projects", mode="before")
    @classmethod
    def normalize_projects(cls, value: object) -> tuple[object, ...]:
        values = _as_tuple(value, field_name="projects")
        if not values:
            raise ValueError("workspace release plan must select at least one project")
        return values

    @field_validator("dependency_edges", "floor_rewrites", mode="before")
    @classmethod
    def normalize_plan_records(cls, value: object) -> tuple[object, ...]:
        return _as_tuple(value, field_name="release plan records")

    @field_validator("dependency_edges", "floor_rewrites", mode="after")
    @classmethod
    def sort_plan_records(
        cls, value: tuple[ContractModel, ...]
    ) -> tuple[ContractModel, ...]:
        return tuple(sorted(value, key=lambda record: record.canonical_json()))

    @field_validator("validation_stages", mode="before")
    @classmethod
    def normalize_plan_stages(cls, value: object) -> tuple[object, ...]:
        values = _as_tuple(value, field_name="validation_stages")
        if not values:
            raise ValueError("workspace release plan requires validation stages")
        return values

    @field_validator("build_job_ids", "push_job_ids", mode="before")
    @classmethod
    def normalize_plan_jobs(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(
            value, field_name="release plan jobs", sort_values=True
        )

    @field_validator("parallel_groups", mode="before")
    @classmethod
    def normalize_parallel_groups(cls, value: object) -> tuple[tuple[str, ...], ...]:
        groups = _as_tuple(value, field_name="parallel_groups")
        normalized: list[tuple[str, ...]] = []
        for group in groups:
            members = _validate_string_tuple(group, field_name="parallel group")
            if not members:
                raise ValueError("parallel groups must not be empty")
            normalized.append(tuple(sorted(members)))
        return tuple(normalized)

    @classmethod
    def derive_digest(
        cls,
        *,
        workspace_id: str,
        selected_projects: tuple[str, ...] | list[str],
        projects: tuple[WorkspaceProject, ...] | list[WorkspaceProject],
        dependency_edges: tuple[DependencyEdge, ...] | list[DependencyEdge],
        floor_rewrites: tuple[FloorRewrite, ...] | list[FloorRewrite],
        validation_stages: tuple[ValidationStage, ...] | list[ValidationStage],
        build_job_ids: tuple[str, ...] | list[str],
        push_job_ids: tuple[str, ...] | list[str],
        parallel_groups: tuple[tuple[str, ...], ...] | list[list[str]],
        consent: ConsentPolicy,
    ) -> str:
        """Derive the immutable identity of a workspace release plan."""

        return _canonical_digest(
            {
                "workspace_id": workspace_id,
                "selected_projects": selected_projects,
                "projects": projects,
                "dependency_edges": dependency_edges,
                "floor_rewrites": floor_rewrites,
                "validation_stages": validation_stages,
                "build_job_ids": build_job_ids,
                "push_job_ids": push_job_ids,
                "parallel_groups": parallel_groups,
                "consent": consent,
            }
        )

    @model_validator(mode="after")
    def validate_plan(self) -> WorkspaceReleasePlan:
        selected = _validate_plan_projects(self)
        graph = _validate_plan_dependency_graph(self, selected)
        _validate_plan_floor_rewrites(self, selected)
        _validate_plan_parallel_groups(self, selected, graph)
        _validate_plan_consent_and_state(self)
        _validate_plan_digest(self)
        return self


def _validate_plan_projects(plan: WorkspaceReleasePlan) -> set[str]:
    selected = set(plan.selected_projects)
    project_ids = [project.project_id for project in plan.projects]
    if len(selected) != len(plan.selected_projects):
        raise ValueError("selected workspace projects must be unique")
    if len(project_ids) != len(set(project_ids)):
        raise ValueError("workspace release project IDs must be unique")
    if selected != set(project_ids):
        raise ValueError("selected_projects and projects must describe the same set")
    return selected


def _validate_plan_dependency_graph(
    plan: WorkspaceReleasePlan, selected: set[str]
) -> dict[str, list[str]]:
    graph: dict[str, list[str]] = {project_id: [] for project_id in selected}
    for edge in plan.dependency_edges:
        if edge.dependent_project_id == edge.dependency_project_id:
            raise ValueError("workspace dependency graph cannot contain self-edges")
        if {
            edge.dependent_project_id,
            edge.dependency_project_id,
        } - selected:
            raise ValueError("dependency edge references an unselected project")
        graph[edge.dependent_project_id].append(edge.dependency_project_id)
    _assert_acyclic(graph)
    return graph


def _validate_plan_floor_rewrites(
    plan: WorkspaceReleasePlan, selected: set[str]
) -> None:
    for rewrite in plan.floor_rewrites:
        if {
            rewrite.project_id,
            rewrite.dependency_project_id,
        } - selected:
            raise ValueError("floor rewrite references an unselected project")


def _validate_parallel_group(
    group: tuple[str, ...],
    selected: set[str],
    grouped: set[str],
    graph: dict[str, list[str]],
) -> None:
    if set(group) - selected:
        raise ValueError("parallel group references an unselected project")
    if grouped.intersection(group):
        raise ValueError("parallel groups must not repeat a project")
    for project_id in group:
        if any(dependency in group for dependency in graph[project_id]):
            raise ValueError("parallel group contains dependent projects")


def _validate_plan_parallel_groups(
    plan: WorkspaceReleasePlan,
    selected: set[str],
    graph: dict[str, list[str]],
) -> None:
    grouped: set[str] = set()
    for group in plan.parallel_groups:
        _validate_parallel_group(group, selected, grouped, graph)
        grouped.update(group)


def _validate_plan_consent_and_state(plan: WorkspaceReleasePlan) -> None:
    if plan.push_job_ids and not plan.consent.allow_push:
        raise ValueError("workspace push jobs require explicit push consent")
    if (
        plan.state in {ReleasePlanState.FROZEN, ReleasePlanState.APPLIED}
        and plan.frozen_at is None
    ):
        raise ValueError("frozen or applied plan requires frozen_at")
    if plan.state == ReleasePlanState.DRAFT and plan.frozen_at is not None:
        raise ValueError("draft plan cannot carry frozen_at")


def _validate_plan_digest(plan: WorkspaceReleasePlan) -> None:
    expected_digest = plan.derive_digest(
        workspace_id=plan.workspace_id,
        selected_projects=plan.selected_projects,
        projects=plan.projects,
        dependency_edges=plan.dependency_edges,
        floor_rewrites=plan.floor_rewrites,
        validation_stages=plan.validation_stages,
        build_job_ids=plan.build_job_ids,
        push_job_ids=plan.push_job_ids,
        parallel_groups=plan.parallel_groups,
        consent=plan.consent,
    )
    if plan.plan_digest != expected_digest:
        raise ValueError("plan_digest does not match the frozen plan contents")


def _assert_acyclic(graph: dict[str, list[str]]) -> None:
    """Reject dependency cycles before a release plan can mutate anything."""

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in visiting:
            raise ValueError("workspace dependency graph contains a cycle")
        if node in visited:
            return
        visiting.add(node)
        for dependency in graph[node]:
            visit(dependency)
        visiting.remove(node)
        visited.add(node)

    for node in graph:
        visit(node)


class BuildKey(ContractModel):
    """C-05 immutable build identity independent of a mutable worktree."""

    repository: RepositoryIdentity
    tree_sha: GitSha
    build_spec_digest: Digest
    feature_set: tuple[OpaqueId, ...] = ()
    toolchain_digest: Digest
    target_triple: OpaqueId
    artifact_contract_digest: Digest

    @field_validator("feature_set", mode="before")
    @classmethod
    def normalize_features(cls, value: object) -> tuple[str, ...]:
        return _validate_string_tuple(value, field_name="feature_set", sort_values=True)


class BuildResult(ContractModel):
    """C-05 result distinction for exact-key cache consumers."""

    key: BuildKey
    outcome: BuildOutcome
    producer_job_id: JobId | None = None
    artifact_refs: tuple[ArtifactReference, ...] = ()
    reason: StrictStr = ""

    @model_validator(mode="after")
    def validate_build_result(self) -> BuildResult:
        if (
            self.outcome
            in {
                BuildOutcome.HIT,
                BuildOutcome.WAITED_HIT,
                BuildOutcome.PRODUCED_MISS,
            }
            and not self.artifact_refs
        ):
            raise ValueError("successful/cache-hit build outcomes require artifacts")
        if (
            self.outcome
            in {
                BuildOutcome.DEGRADED_UNCACHEABLE,
                BuildOutcome.CORRUPTED_ENTRY,
                BuildOutcome.REFUSED,
                BuildOutcome.FAILED,
            }
            and not self.reason
        ):
            raise ValueError(
                "degraded, refused, and failed build outcomes require a reason"
            )
        return self


__all__ = [
    "AbsolutePath",
    "ArtifactReference",
    "BuildKey",
    "BuildResult",
    "Candidate",
    "CandidateVersion",
    "CapacitySnapshot",
    "ConsentPolicy",
    "ContractModel",
    "DependencyEdge",
    "DevelopmentRequest",
    "ExecutionCommand",
    "ExecutionResult",
    "FloorRewrite",
    "Generation",
    "GitRef",
    "GitSha",
    "LaneReference",
    "LeaseRecord",
    "LogReference",
    "OpaqueId",
    "RelativePath",
    "RepositoryIdentity",
    "RepositoryJobResult",
    "ResourceRequest",
    "ResourceReservation",
    "TargetPolicy",
    "UtcDateTime",
    "ValidationEvidence",
    "ValidationPolicy",
    "WorkItemId",
    "JobId",
    "WorkspaceProject",
    "WorkspaceReleasePlan",
]
