"""Durable repository job application service.

RMDD-06 keeps the Repository Manager domain independent from the graph-os
implementation.  :class:`RepositoryJobService` depends on the small
``RepositoryJobPort`` below; the production port delegates to the Agent
Utilities repository WorkItem adapter, while tests use ``FakeRepositoryJobPort``.

The service deliberately has no executor, thread, future, Git mutation, or
second job store.  A submitted job is an immutable WorkItem request.  Reads
are tenant/owner scoped, list pages are bounded by an explicit keyset cursor,
and reconciliation produces repair WorkItems rather than changing a checkout
or process in the request path.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from enum import StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictFloat,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

from .enums import JobState, OperationKind, RefusalCode
from .models import DevelopmentRequest, ResourceRequest
from .payloads import BuildExecutionDescriptor, operation_payload_from_mapping
from .serialization import canonical_digest, canonical_json

if TYPE_CHECKING:
    from repository_manager.resource_profiles import ResourceProfileRegistry

_CURSOR_PREFIX = "rmpage:v1:"
_CURSOR_VERSION = 1
_MAX_PAGE_SIZE = 1000
_MAX_SCAN_ROWS = 1000
_REPAIR_OPERATION = "repair"
_RESOLVED_PROFILE_AUTHORITY = "repository_manager:resource_profile_registry:v1"
_OPERATION_PAYLOAD_KIND = "repository.build-execution/v1"
_OPERATION_PAYLOAD_VERSION = "1"
_PAYLOAD_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_OPERATION_KIND: Mapping[str, str] = {
    "lane.allocate": "repository.lane.allocate",
    "lane.check": "repository.lane.check",
    "repository": "repository.operation",
    "validation": "repository.validation",
    "build": "repository.build",
    "merge": "repository.merge",
    "release": "repository.release",
    "candidate.submit": "repository.candidate.submit",
    "generation.certify": "repository.generation.certify",
    "branch.land": "repository.branch.land",
    "workspace.validate": "repository.workspace.validate",
    "workspace.bump": "repository.workspace.bump",
    "workspace.push": "repository.workspace.push",
    _REPAIR_OPERATION: "repository.repair",
}


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(ord(char) < 0x20 for char in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _as_tuple(value: object, field_name: str) -> tuple[object, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence")
    try:
        return tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence") from exc


def _string_tuple(value: object, field_name: str) -> tuple[str, ...]:
    return tuple(_nonblank(item, field_name) for item in _as_tuple(value, field_name))


class RepositoryJobServiceCode(StrEnum):
    """Stable service-level refinements of the C-10 refusal categories."""

    INVALID_REQUEST = RefusalCode.INVALID_REQUEST.value
    UNAUTHORIZED = RefusalCode.UNAUTHORIZED_TARGET.value
    CONFLICT = RefusalCode.CONFLICT_BASE_MOVED.value
    INVALID_STATE = RefusalCode.INVALID_STATE_COMBINATION.value
    DUPLICATE = RefusalCode.DUPLICATE_REQUEST.value
    INPUT_CONFLICT = "input_conflict"
    TYPED_EXECUTION_PAYLOAD_REQUIRED = "typed_execution_payload_required"
    TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE = (
        "typed_execution_payload_authority_unavailable"
    )
    RECONCILIATION_REQUIRED = RefusalCode.RECONCILIATION_REQUIRED.value
    INTERNAL = RefusalCode.INTERNAL_ERROR.value


class RepositoryJobServiceError(ValueError):
    """A structured, automation-safe service refusal."""

    def __init__(
        self,
        code: RepositoryJobServiceCode | str,
        message: str,
        *,
        job_id: str | None = None,
    ) -> None:
        self.code = str(code)
        self.job_id = job_id
        super().__init__(message)


_SAFE_ERROR_MESSAGES: Mapping[RepositoryJobServiceCode, str] = {
    RepositoryJobServiceCode.INVALID_REQUEST: "repository job request is invalid",
    RepositoryJobServiceCode.UNAUTHORIZED: "repository job is not authorized for this scope",
    RepositoryJobServiceCode.CONFLICT: "repository job conflicts with durable state",
    RepositoryJobServiceCode.INVALID_STATE: "repository job lifecycle state refuses this operation",
    RepositoryJobServiceCode.DUPLICATE: "repository job idempotency key conflicts with immutable input",
    RepositoryJobServiceCode.INPUT_CONFLICT: "repository execution input conflicts with durable state",
    RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED: "typed repository execution input is required",
    RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE: (
        "typed repository execution input authority is unavailable"
    ),
    RepositoryJobServiceCode.RECONCILIATION_REQUIRED: "repository job requires reconciliation",
    RepositoryJobServiceCode.INTERNAL: "repository job authority failed",
}


def _safe_port_error(
    error: Exception,
    *,
    job_id: str | None = None,
    value_error_code: RepositoryJobServiceCode = RepositoryJobServiceCode.INVALID_REQUEST,
) -> RepositoryJobServiceError:
    """Normalize adapter failures without exposing authority payloads."""

    if isinstance(error, RepositoryJobServiceError):
        try:
            code = RepositoryJobServiceCode(error.code)
        except ValueError:
            code = RepositoryJobServiceCode.INTERNAL
        scoped_job_id = error.job_id or job_id
    elif isinstance(error, (ValueError, TypeError)):
        code = value_error_code
        scoped_job_id = job_id
    else:
        code = RepositoryJobServiceCode.INTERNAL
        scoped_job_id = job_id
    return RepositoryJobServiceError(
        code,
        _SAFE_ERROR_MESSAGES[code],
        job_id=scoped_job_id,
    )


def _safe_exact_input_error(
    error: Exception, *, job_id: str
) -> RepositoryJobServiceError:
    """Normalize exact-input adapter failures without exposing private input."""

    if isinstance(error, RepositoryJobServiceError):
        try:
            code = RepositoryJobServiceCode(error.code)
        except ValueError:
            code = (
                RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
            )
        if code not in {
            RepositoryJobServiceCode.INPUT_CONFLICT,
            RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED,
            RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE,
        }:
            code = (
                RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
            )
    else:
        code = RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
    return RepositoryJobServiceError(
        code,
        _SAFE_ERROR_MESSAGES[code],
        job_id=job_id,
    )


class JobAuthorization(BaseModel):
    """Verified caller identity supplied by the MCP/REST/CLI boundary."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    tenant_id: StrictStr
    owner_id: StrictStr
    session_id: StrictStr | None = None

    @field_validator("tenant_id", "owner_id", "session_id")
    @classmethod
    def validate_identity(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)


class DurableJobView(BaseModel):
    """A normalized read-only WorkItem projection returned by a port.

    The shape intentionally follows the merged AU repository WorkItem view,
    while retaining optional timestamps used only for bounded pagination.  A
    port may construct this model from a native AU model or a mapping; no
    service state is authoritative.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)

    contract_version: StrictStr = "1"
    job_id: StrictStr
    work_item_id: StrictStr
    request_id: StrictStr
    operation: StrictStr
    kind: StrictStr = "repository.operation"
    state: JobState
    repository_id: StrictStr
    tenant_id: StrictStr
    owner_id: StrictStr
    session_id: StrictStr
    base_ref: StrictStr
    base_sha: StrictStr
    target_kind: StrictStr = "local"
    target_alias: StrictStr | None = None
    lane_id: StrictStr | None = None
    candidate_id: StrictStr | None = None
    generation_id: StrictStr | None = None
    dependencies: tuple[StrictStr, ...] = ()
    input_digest: StrictStr
    config_digest: StrictStr | None = None
    correlation_id: StrictStr | None = None
    resource_class: StrictStr = "light-check"
    concurrency_key: StrictStr = "light-check"
    fairness_group: StrictStr = "default"
    priority: StrictInt = Field(default=0, ge=0, le=10_000)
    cpu_weight: StrictInt = Field(default=1, ge=1)
    memory_mib: StrictInt = Field(default=256, ge=1)
    disk_mib: StrictInt = Field(default=256, ge=1)
    process_slots: StrictInt = Field(default=1, ge=1)
    host_labels: tuple[StrictStr, ...] = ()
    preferred_target: Mapping[str, Any] = Field(default_factory=dict)
    required_target: Mapping[str, Any] | None = None
    anti_affinity: tuple[StrictStr, ...] = ()
    queue_deadline: datetime | None = None
    disk_low_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    disk_high_watermark_mib: StrictInt | None = Field(default=None, ge=0)
    consent: Mapping[str, Any] = Field(default_factory=dict)
    attempt: StrictInt = Field(default=0, ge=0)
    max_attempts: StrictInt = Field(default=1, ge=1)
    checkpoint: StrictStr | None = None
    retry_class: StrictStr | None = None
    result_ref: StrictStr | None = None
    error_ref: StrictStr | None = None
    lease_owner: StrictStr | None = None
    lease_fence: StrictStr | None = None
    lease_expires_at: datetime | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    completed_at: datetime | None = None
    # Only the typed payload summary crosses ordinary projections.  The
    # canonical body is available through the exact-input method below.
    operation_payload_kind: StrictStr | None = None
    operation_payload_version: StrictStr | None = None
    operation_payload_digest: StrictStr | None = None

    @field_validator(
        "job_id",
        "work_item_id",
        "request_id",
        "operation",
        "kind",
        "repository_id",
        "tenant_id",
        "owner_id",
        "session_id",
        "base_ref",
        "base_sha",
        "input_digest",
        "target_kind",
        "resource_class",
        "concurrency_key",
        "fairness_group",
    )
    @classmethod
    def validate_strings(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "target_alias",
        "lane_id",
        "candidate_id",
        "generation_id",
        "config_digest",
        "correlation_id",
        "checkpoint",
        "retry_class",
        "result_ref",
        "error_ref",
        "lease_owner",
        "lease_fence",
        "operation_payload_kind",
        "operation_payload_version",
        "operation_payload_digest",
    )
    @classmethod
    def validate_optional_strings(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("operation_payload_kind")
    @classmethod
    def validate_payload_kind(cls, value: str | None) -> str | None:
        if value is not None and value != _OPERATION_PAYLOAD_KIND:
            raise ValueError("operation payload kind is unknown")
        return value

    @field_validator("operation_payload_version")
    @classmethod
    def validate_payload_version(cls, value: str | None) -> str | None:
        if value is not None and value != _OPERATION_PAYLOAD_VERSION:
            raise ValueError("operation payload version is unknown")
        return value

    @field_validator("operation_payload_digest")
    @classmethod
    def validate_payload_digest(cls, value: str | None) -> str | None:
        if value is not None and not _PAYLOAD_DIGEST_RE.fullmatch(value):
            raise ValueError("operation payload digest is invalid")
        return value

    @field_validator("dependencies", "host_labels", "anti_affinity", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object, info: Any) -> tuple[str, ...]:
        return _string_tuple(value, info.field_name)

    @field_validator(
        "queue_deadline", "lease_expires_at", "created_at", "updated_at", "completed_at"
    )
    @classmethod
    def validate_timestamps(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("timestamps must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_scope(self) -> DurableJobView:
        if self.state in {JobState.LEASED, JobState.RUNNING} and (
            self.lease_owner is None or self.lease_fence is None
        ):
            raise ValueError("leased/running jobs require lease owner and fence")
        summary = (
            self.operation_payload_kind,
            self.operation_payload_version,
            self.operation_payload_digest,
        )
        if any(value is not None for value in summary) and not all(
            value is not None for value in summary
        ):
            raise ValueError("operation payload summary must be complete")
        if self.operation_payload_kind is not None and self.operation != "build":
            raise ValueError("operation payload summary does not match operation")
        return self


class JobPage(BaseModel):
    """One bounded result page and an explicit continuation token."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    items: tuple[DurableJobView, ...] = ()
    next_cursor: StrictStr | None = None
    scanned: StrictInt = Field(default=0, ge=0, le=_MAX_SCAN_ROWS)
    exhausted: StrictBool = False

    @field_validator("next_cursor")
    @classmethod
    def validate_cursor(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, "next_cursor")


class JobSubmitResult(BaseModel):
    """Stable submit response, including cross-service deduplication."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job: DurableJobView
    deduplicated: StrictBool = False
    retry_of: StrictStr | None = None


class JobMutationResult(BaseModel):
    """Result of a durable cancel operation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job: DurableJobView
    changed: StrictBool = False


class JobFilters(BaseModel):
    """Tenant-scoped domain filters for a bounded list call."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    repository_id: StrictStr | None = None
    lane_id: StrictStr | None = None
    candidate_id: StrictStr | None = None
    generation_id: StrictStr | None = None
    correlation_id: StrictStr | None = None
    kind: StrictStr | None = None
    owner_id: StrictStr | None = None
    operation: OperationKind | StrictStr | None = None
    states: tuple[JobState, ...] = ()
    resource_class: StrictStr | None = None
    host_alias: StrictStr | None = None
    created_after: datetime | None = None
    created_before: datetime | None = None
    limit: StrictInt = Field(default=100, ge=1, le=_MAX_PAGE_SIZE)
    cursor: StrictStr | None = None

    @field_validator(
        "repository_id",
        "lane_id",
        "candidate_id",
        "generation_id",
        "correlation_id",
        "kind",
        "owner_id",
        "resource_class",
        "host_alias",
    )
    @classmethod
    def validate_optional_filter(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("created_after", "created_before")
    @classmethod
    def validate_filter_time(cls, value: datetime | None) -> datetime | None:
        if value is not None and (value.tzinfo is None or value.utcoffset() is None):
            raise ValueError("time filters must be timezone-aware")
        return value

    @model_validator(mode="after")
    def validate_time_order(self) -> JobFilters:
        if (
            self.created_after is not None
            and self.created_before is not None
            and self.created_after > self.created_before
        ):
            raise ValueError("created_after must not exceed created_before")
        return self


class ReconciliationObservation(BaseModel):
    """Read-only worker-side observation used by reconciliation.

    ``None`` means the observer did not inspect that dimension.  This avoids
    turning an incomplete probe into a false destructive finding.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_id: StrictStr
    repository_id: StrictStr | None = None
    worktree_expected: StrictBool = True
    worktree_present: StrictBool | None = None
    process_present: StrictBool | None = None
    process_last_heartbeat: datetime | None = None
    process_stale_after_seconds: StrictFloat = Field(default=300.0, gt=0)
    observed_artifact_refs: tuple[StrictStr, ...] = ()
    expected_artifact_refs: tuple[StrictStr, ...] = ()
    observed_fence: StrictStr | None = None
    expected_fence: StrictStr | None = None
    observed_target_sha: StrictStr | None = None
    expected_target_sha: StrictStr | None = None
    already_completed_effect: StrictBool = False
    observed_at: datetime

    @field_validator("job_id")
    @classmethod
    def validate_job(cls, value: str) -> str:
        return _nonblank(value, "job_id")

    @field_validator(
        "repository_id",
        "observed_fence",
        "expected_fence",
        "observed_target_sha",
        "expected_target_sha",
    )
    @classmethod
    def validate_optional_observation(cls, value: str | None, info: Any) -> str | None:
        return None if value is None else _nonblank(value, info.field_name)

    @field_validator("observed_artifact_refs", "expected_artifact_refs", mode="before")
    @classmethod
    def normalize_artifacts(cls, value: object, info: Any) -> tuple[str, ...]:
        return _string_tuple(value, info.field_name)

    @field_validator("process_last_heartbeat", "observed_at")
    @classmethod
    def validate_observation_time(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("observation timestamps must be timezone-aware")
        return value


class ReconciliationClass(StrEnum):
    CLEAN = "clean"
    MISSING_WORKTREE = "missing_worktree"
    STALE_PROCESS = "stale_process"
    ORPHAN_ARTIFACT = "orphan_artifact"
    STALE_FENCE = "stale_fence"
    TARGET_DRIFT = "target_drift"
    ALREADY_COMPLETED_EFFECT = "already_completed_effect"


class RepairProposal(BaseModel):
    """Previewable, deterministic repair intent; not a Git/process mutation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    repair_id: StrictStr
    source_job_id: StrictStr
    classifications: tuple[ReconciliationClass, ...]
    idempotency_key: StrictStr
    operation: StrictStr = _REPAIR_OPERATION
    preview_only: StrictBool = True
    enqueued_job_id: StrictStr | None = None


class ReconciliationFinding(BaseModel):
    """One durable job and its deterministic observation comparison."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job: DurableJobView
    classifications: tuple[ReconciliationClass, ...]
    details: Mapping[str, Any] = Field(default_factory=dict)
    repair: RepairProposal | None = None


class ReconciliationResult(BaseModel):
    """Bounded reconciliation response with explicit continuation."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    findings: tuple[ReconciliationFinding, ...] = ()
    next_cursor: StrictStr | None = None
    scanned: StrictInt = Field(default=0, ge=0, le=_MAX_SCAN_ROWS)
    exhausted: StrictBool = False


class ShadowMismatch(BaseModel):
    """Visible legacy-vs-durable disagreement; neither side is overwritten."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    job_id: StrictStr
    fields: tuple[StrictStr, ...]
    durable: Mapping[str, Any]
    legacy: Mapping[str, Any]


@runtime_checkable
class RepositoryJobPort(Protocol):
    """The only state seam used by :class:`RepositoryJobService`."""

    def execution_input_authority_available(self) -> bool: ...

    def submit(
        self,
        request: Mapping[str, Any],
        *,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> JobSubmitResult: ...

    def get(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> DurableJobView | None: ...

    def list_page(
        self,
        *,
        tenant_id: str,
        filters: JobFilters,
        cursor: tuple[float, str] | None,
    ) -> JobPage: ...

    def cancel(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> JobMutationResult: ...

    def retry(
        self,
        job: DurableJobView,
        *,
        tenant_id: str,
        owner_id: str,
        now: datetime | None = None,
    ) -> JobSubmitResult: ...

    def submit_repair(
        self,
        job: DurableJobView,
        proposal: RepairProposal,
        *,
        tenant_id: str,
        owner_id: str,
        session_id: str | None,
        now: datetime | None = None,
    ) -> JobSubmitResult: ...

    def get_exact_execution_input(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> BuildExecutionDescriptor | None: ...


def _cursor_tenant_digest(tenant_id: str) -> str:
    return hashlib.sha256(tenant_id.encode("utf-8")).hexdigest()


def encode_cursor(tenant_id: str, cursor: tuple[float, str]) -> str:
    """Encode a tenant-bound, opaque keyset continuation token."""

    created_at, work_item_id = cursor
    payload = {
        "v": _CURSOR_VERSION,
        "tenant": _cursor_tenant_digest(_nonblank(tenant_id, "tenant_id")),
        "created_at": float(created_at),
        "id": _nonblank(work_item_id, "work_item_id"),
    }
    encoded = base64.urlsafe_b64encode(canonical_json(payload).encode()).decode()
    return _CURSOR_PREFIX + encoded.rstrip("=")


def decode_cursor(tenant_id: str, token: str | None) -> tuple[float, str] | None:
    """Validate and decode one cursor, refusing cross-tenant reuse."""

    if token is None:
        return None
    token = _nonblank(token, "cursor")
    if not token.startswith(_CURSOR_PREFIX):
        raise RepositoryJobServiceError(
            RepositoryJobServiceCode.INVALID_REQUEST, "invalid repository job cursor"
        )
    encoded = token[len(_CURSOR_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        value = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RepositoryJobServiceError(
            RepositoryJobServiceCode.INVALID_REQUEST, "invalid repository job cursor"
        ) from exc
    if (
        not isinstance(value, dict)
        or value.get("v") != _CURSOR_VERSION
        or value.get("tenant") != _cursor_tenant_digest(tenant_id)
        or not isinstance(value.get("id"), str)
        or not isinstance(value.get("created_at"), (int, float))
    ):
        raise RepositoryJobServiceError(
            RepositoryJobServiceCode.UNAUTHORIZED,
            "repository job cursor is not valid for this tenant",
        )
    return (float(value["created_at"]), _nonblank(value["id"], "cursor id"))


def _timestamp(value: datetime | None) -> datetime:
    return (value or datetime.now(UTC)).astimezone(UTC)


def _datetime_from_epoch(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("boolean is not a valid epoch timestamp")
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise ValueError(
        "authority timestamp must be an aware datetime, epoch, or ISO string"
    )


def _as_mapping(value: object) -> dict[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", exclude_none=False)
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError("job request must be a mapping or Pydantic model")


def _translate_authority_error(
    authority: Any, error: Exception
) -> RepositoryJobServiceError:
    """Map AU adapter failures onto stable RMDD service codes."""

    # Keep only a short, non-public classification hint.  The exception text
    # may contain model input or adapter payloads and must never cross this API.
    lowered = str(error).lower()[:256]
    conflict_type = getattr(authority, "RepositoryWorkItemConflict", ())
    error_type = getattr(authority, "RepositoryWorkItemError", ())
    is_conflict = bool(conflict_type) and isinstance(error, conflict_type)
    is_authority_error = bool(error_type) and isinstance(error, error_type)
    if is_conflict or is_authority_error:
        if "typed_execution_payload_required" in lowered:
            code = RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED
        elif "typed_execution_payload_authority_unavailable" in lowered:
            code = (
                RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
            )
        elif "input_conflict" in lowered:
            code = RepositoryJobServiceCode.INPUT_CONFLICT
        elif is_conflict:
            if (
                "tenant" in lowered
                or "authenticated" in lowered
                or "outside" in lowered
            ):
                code = RepositoryJobServiceCode.UNAUTHORIZED
            elif "base moved" in lowered or "target moved" in lowered:
                code = RepositoryJobServiceCode.CONFLICT
            elif (
                "idempot" in lowered
                or "immutable" in lowered
                or "operation" in lowered
                or "identity" in lowered
            ):
                code = RepositoryJobServiceCode.DUPLICATE
            else:
                code = RepositoryJobServiceCode.CONFLICT
        elif "tenant" in lowered or "authenticated" in lowered or "outside" in lowered:
            code = RepositoryJobServiceCode.UNAUTHORIZED
        elif "base moved" in lowered or "target moved" in lowered:
            code = RepositoryJobServiceCode.CONFLICT
        elif "reconciliation" in lowered:
            code = RepositoryJobServiceCode.RECONCILIATION_REQUIRED
        elif "idempot" in lowered or "immutable" in lowered or "operation" in lowered:
            code = RepositoryJobServiceCode.DUPLICATE
        elif "state" in lowered or "dependency" in lowered:
            code = RepositoryJobServiceCode.INVALID_STATE
        else:
            code = RepositoryJobServiceCode.INVALID_REQUEST
    elif isinstance(error, (TypeError, ValueError)):
        code = RepositoryJobServiceCode.INVALID_REQUEST
    else:
        code = RepositoryJobServiceCode.INTERNAL
    return RepositoryJobServiceError(code, _SAFE_ERROR_MESSAGES[code])


class GraphRepositoryJobPort:
    """Production adapter over the graph-os/AU repository WorkItem authority.

    Imports are lazy so the service remains unit-testable without a graph
    client.  ``engine`` is the already-authenticated graph authority supplied
    by the host; this adapter never creates credentials or a local store.
    """

    def __init__(
        self, engine: Any, *, profiles: ResourceProfileRegistry | None = None
    ) -> None:
        self.engine = engine
        # Reads remain constructible for compatibility, but native WorkItem
        # submission below refuses to run without this trusted registry.
        self.profiles = profiles

    @staticmethod
    def _authority_module() -> Any:
        from agent_utilities.orchestration import repository_work_item as authority

        return authority

    @staticmethod
    def _view(value: Any) -> DurableJobView:
        if isinstance(value, DurableJobView):
            return value
        if hasattr(value, "model_dump"):
            value = value.model_dump(mode="json", exclude_none=False)
        if not isinstance(value, Mapping):
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INTERNAL,
                "graph authority returned an invalid repository job view",
            )
        # AU's public projection intentionally exposes ``kind`` and a nested
        # ``lease`` object.  RM's service view normalizes those fields without
        # weakening the authority or accepting arbitrary extras.
        raw = dict(value)
        lease = raw.pop("lease", None)
        if isinstance(lease, Mapping):
            raw["lease_owner"] = lease.get("owner")
            raw["lease_fence"] = (
                None
                if lease.get("fencing_token") is None
                else str(lease.get("fencing_token"))
            )
            raw["lease_expires_at"] = _datetime_from_epoch(lease.get("expires_at"))
        for field in ("created_at", "updated_at", "completed_at"):
            if field in raw:
                raw[field] = _datetime_from_epoch(raw[field])
        return DurableJobView.model_validate(raw)

    def _resolved_submission(self, request: Mapping[str, Any]) -> dict[str, Any]:
        """Attach the trusted, fully resolved admission extension before submit."""

        profiles = self.profiles
        if profiles is None:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INTERNAL,
                "native WorkItem submission requires a trusted resource profile registry",
            )
        raw = dict(request)
        raw_resources = raw.get("resources")
        if raw_resources is None:
            raw_resources = {}
        if not isinstance(raw_resources, Mapping):
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_REQUEST,
                "repository resource request is invalid",
            )
        resource_input = dict(raw_resources)
        if "branch" in resource_input and "branch" not in raw:
            # Branch is an RM/AU correlation outside the public ResourceRequest
            # model.  Preserve an explicitly supplied value; never invent one
            # from base_ref at this boundary.
            raw["branch"] = resource_input["branch"]
        profile_name = resource_input.get("resource_class") or raw.get("resource_class")
        if profile_name is None:
            profile_name = "light-check"
        if not isinstance(profile_name, str) or not profile_name.strip():
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_REQUEST,
                "repository resource profile is invalid",
            )
        try:
            profile = profiles.resolve(profile_name)
            public_fields = set(ResourceRequest.model_fields)
            public_input = {
                key: value
                for key, value in resource_input.items()
                if key in public_fields
            }
            public_input["resource_class"] = profile.name
            resolved = profile.merge_request(
                ResourceRequest.model_validate(public_input)
            )
        except Exception as exc:  # noqa: BLE001 - fail closed at authority boundary
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_REQUEST,
                "repository resource profile could not be resolved",
            ) from exc
        resources = resolved.model_dump(mode="json", exclude_none=False)
        resources.update(
            {
                "profile_version": str(profile.profile_version),
                "concurrency_limit": profile.concurrency_limit,
                "repository_exclusive": profile.repository_exclusive,
                "branch_exclusive": profile.branch_exclusive,
                "disk_policy_key": f"{profile.name}:v{profile.profile_version}",
                "fairness_cost": max(1, resolved.cpu_weight + resolved.process_slots),
                # AU must persist this marker in its immutable extension.  It
                # is intentionally separate from WorkItem input_digest and
                # from RM's later reservation input fingerprint.
                "resolved_profile_authority": _RESOLVED_PROFILE_AUTHORITY,
            }
        )
        raw["resources"] = resources
        # Keep a transport-visible copy while AU's generated request adapter
        # learns to persist the nested marker under resource_reservation.
        raw["resolved_profile_authority"] = _RESOLVED_PROFILE_AUTHORITY
        return raw

    def submit(
        self,
        request: Mapping[str, Any],
        *,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        authority = self._authority_module()
        request_mapping = self._resolved_submission(_as_mapping(request))
        try:
            handle = authority.submit_repository_work_item(
                self.engine,
                request_mapping,
                now=None if now is None else now.timestamp(),
                max_attempts=max_attempts,
                resolved_profile_projection=True,
            )
        except Exception as exc:  # noqa: BLE001 - translate authority contract errors
            raise _translate_authority_error(authority, exc) from exc

        raw = handle.model_dump(mode="json", exclude_none=False)
        job_id = str(raw["job_id"])
        # The AU handle intentionally does not carry tenant; the authenticated
        # request is the source for the tenant-scoped lookup.
        tenant_id = str(
            request_mapping.get("tenant_id") or request_mapping.get("tenant") or ""
        )
        owner_id = str(request_mapping.get("owner_id") or "")
        view = self.get(job_id, tenant_id=tenant_id, owner_id=owner_id)
        if view is None:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INTERNAL,
                "repository WorkItem disappeared after submission",
            )
        return JobSubmitResult(
            job=view,
            deduplicated=bool(raw.get("deduplicated", False)),
        )

    def execution_input_authority_available(self) -> bool:
        """Report the absent EG-native atomic exact-input port."""

        return False

    def get_exact_execution_input(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> BuildExecutionDescriptor | None:
        """Refuse until EG supplies one atomic authenticated exact-input operation.

        The old tenant/owner adapter accepted a caller-controlled owner string
        and therefore could impersonate the submitter.  Public views remain
        available, but no production Graph port may read executable bytes
        until the native atomic exact-input port exists.
        """

        del tenant_id, owner_id
        raise RepositoryJobServiceError(
            RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE,
            _SAFE_ERROR_MESSAGES[
                RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
            ],
            job_id=job_id,
        )

    # Compatibility spelling for callers that use the shorter contract name;
    # both paths retain the exact same authorization and validation boundary.
    def get_execution_input(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> BuildExecutionDescriptor | None:
        return self.get_exact_execution_input(
            job_id, tenant_id=tenant_id, owner_id=owner_id
        )

    def get(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> DurableJobView | None:
        authority = self._authority_module()
        try:
            view = authority.get_repository_work_item(
                self.engine,
                job_id,
                tenant=tenant_id,
                owner_id=owner_id,
            )
            return None if view is None else self._view(view)
        except RepositoryJobServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - translate authority contract errors
            raise _translate_authority_error(authority, exc) from exc

    def list_page(
        self,
        *,
        tenant_id: str,
        filters: JobFilters,
        cursor: tuple[float, str] | None,
    ) -> JobPage:
        """Read exactly one native bounded page.

        The AU adapter's public convenience listing predates explicit cursors;
        this narrow production seam uses its existing private page primitive,
        which still routes through the AU authority object and its native
        tenant/kind query.  It intentionally does not loop to find matches.
        """

        authority = self._authority_module()
        try:
            rows = authority._repository_rows(  # noqa: SLF001 - adapter boundary
                self.engine,
                tenant=tenant_id,
                limit=filters.limit,
                kinds=(
                    [authority.repository_work_item_kind(filters.operation).value]
                    if filters.operation is not None
                    else authority._REPOSITORY_KINDS  # noqa: SLF001
                ),
                statuses=tuple(
                    "dead_letter" if state.value == "dead-letter" else state.value
                    for state in filters.states
                ),
                cursor=cursor,
            )
        except Exception as exc:  # noqa: BLE001 - translate authority contract errors
            raise _translate_authority_error(authority, exc) from exc
        try:
            views: list[DurableJobView] = []
            for row in rows:
                authority_view = authority._view_from_row(row)  # noqa: SLF001
                raw_view = authority_view.model_dump(mode="json", exclude_none=False)
                raw_view["created_at"] = row.get("created_at")
                raw_view["updated_at"] = row.get("updated_at")
                raw_view["completed_at"] = row.get("completed_at")
                view = self._view(raw_view)
                if _matches_filters(view, filters):
                    views.append(view)
            next_cursor = (
                encode_cursor(tenant_id, authority._row_cursor(rows[-1]))  # noqa: SLF001
                if len(rows) == filters.limit
                else None
            )
        except RepositoryJobServiceError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize malformed authority rows
            raise _translate_authority_error(authority, exc) from exc
        return JobPage(
            items=tuple(views),
            next_cursor=next_cursor,
            scanned=len(rows),
            exhausted=len(rows) < filters.limit,
        )

    def cancel(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> JobMutationResult:
        authority = self._authority_module()
        current = self.get(job_id, tenant_id=tenant_id, owner_id=owner_id)
        if (
            current is None
            or current.tenant_id != tenant_id
            or current.owner_id != owner_id
        ):
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "repository job is not accessible to this tenant/owner",
                job_id=job_id,
            )
        try:
            changed = authority.cancel_repository_work_item(
                self.engine,
                job_id,
                tenant=tenant_id,
                reason=reason,
                now=None if now is None else now.timestamp(),
            )
        except Exception as exc:  # noqa: BLE001 - translate authority contract errors
            raise _translate_authority_error(authority, exc) from exc
        view = self.get(job_id, tenant_id=tenant_id, owner_id=owner_id)
        if view is None:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INTERNAL,
                "repository WorkItem disappeared after cancellation",
                job_id=job_id,
            )
        return JobMutationResult(job=view, changed=bool(changed))

    def retry(
        self,
        job: DurableJobView,
        *,
        tenant_id: str,
        owner_id: str,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        if job.tenant_id != tenant_id or job.owner_id != owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "retry identity does not match the durable job",
                job_id=job.job_id,
            )
        # AU currently represents retry as a new WorkItem attempt.  Keep the
        # old terminal row immutable and derive a deterministic idempotency
        # key from its full handle.  The correlation is durable in the new row.
        attempt = job.attempt + 1
        operation_payload = (
            self.get_exact_execution_input(
                job.job_id, tenant_id=tenant_id, owner_id=owner_id
            )
            if job.operation == "build"
            else None
        )
        request = _retry_request_mapping(
            job,
            owner_id=owner_id,
            attempt=attempt,
            operation_payload=operation_payload,
        )
        remaining_attempts = job.max_attempts - job.attempt
        if remaining_attempts < 1:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_STATE,
                "job has exhausted its retry budget",
                job_id=job.job_id,
            )
        return self.submit(request, max_attempts=remaining_attempts, now=now)

    def submit_repair(
        self,
        job: DurableJobView,
        proposal: RepairProposal,
        *,
        tenant_id: str,
        owner_id: str,
        session_id: str | None,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        if job.tenant_id != tenant_id or job.owner_id != owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "repair identity does not match the durable job",
                job_id=job.job_id,
            )
        operation_payload = (
            self.get_exact_execution_input(
                job.job_id, tenant_id=tenant_id, owner_id=owner_id
            )
            if job.operation == "build"
            else None
        )
        request = _repair_request_mapping(
            job,
            proposal,
            owner_id=owner_id,
            session_id=session_id,
            operation_payload=operation_payload,
        )
        return self.submit(request, max_attempts=1, now=now)


def _retry_request_mapping(
    job: DurableJobView,
    *,
    owner_id: str,
    attempt: int,
    operation_payload: BuildExecutionDescriptor | None = None,
) -> dict[str, Any]:
    request = {
        "contract_version": "1",
        "request_id": f"retry:{job.job_id}:{attempt}:request",
        "idempotency_key": f"retry:{job.job_id}:{attempt}",
        "operation": job.operation,
        "repository_id": job.repository_id,
        "base_ref": job.base_ref,
        "base_sha": job.base_sha,
        "owner_id": owner_id,
        "session_id": job.session_id,
        "tenant_id": job.tenant_id,
        "lane_id": job.lane_id,
        "candidate_id": job.candidate_id,
        "generation_id": job.generation_id,
        "correlation_id": job.job_id,
        "priority": job.priority,
        "resources": {
            "resource_class": job.resource_class,
            "concurrency_key": job.concurrency_key,
            "cpu_weight": job.cpu_weight,
            "memory_mib": job.memory_mib,
            "disk_mib": job.disk_mib,
            "process_slots": job.process_slots,
            "host_labels": list(job.host_labels),
            "preferred_target": dict(job.preferred_target),
            "required_target": (
                None if job.required_target is None else dict(job.required_target)
            ),
            "anti_affinity": list(job.anti_affinity),
            "queue_deadline": (
                None if job.queue_deadline is None else job.queue_deadline.isoformat()
            ),
            "disk_low_watermark_mib": job.disk_low_watermark_mib,
            "disk_high_watermark_mib": job.disk_high_watermark_mib,
            "fairness_group": job.fairness_group,
        },
        "target": {
            "kind": job.target_kind,
            "alias": job.target_alias,
        },
        "consent": dict(job.consent),
        "config_digest": job.config_digest,
        "input_digest": job.input_digest,
        "retry_class": job.retry_class or "manual",
    }
    if operation_payload is not None:
        request["operation_payload"] = operation_payload.model_dump(
            mode="json", exclude_none=False
        )
    return request


def _repair_request_mapping(
    job: DurableJobView,
    proposal: RepairProposal,
    *,
    owner_id: str,
    session_id: str | None,
    operation_payload: BuildExecutionDescriptor | None = None,
) -> dict[str, Any]:
    request = _retry_request_mapping(
        job,
        owner_id=owner_id,
        attempt=job.attempt + 1,
        operation_payload=operation_payload,
    )
    repair_intent = {
        "contract": "repository-repair:v1",
        "source_job_id": job.job_id,
        "classifications": tuple(proposal.classifications),
    }
    repair_intent_digest = canonical_digest(repair_intent)
    # A typed build repair is still an executable build input.  The repair
    # identity/intent remains deterministic in its request id, idempotency
    # key, correlation, retry class, and intent digest; only the operation
    # discriminator stays build so the closed payload union remains valid.
    repair_operation = (
        "build"
        if job.operation == "build" and operation_payload is not None
        else _REPAIR_OPERATION
    )
    request.update(
        {
            "request_id": f"repair:{proposal.repair_id}:request",
            "idempotency_key": proposal.idempotency_key,
            "operation": repair_operation,
            "session_id": session_id or job.session_id,
            "correlation_id": job.job_id,
            "retry_class": "reconciliation",
            # A repair is a new immutable WorkItem input. The worker must
            # re-observe the correlated source rather than trust a stale
            # snapshot embedded in the request.
            "input_digest": repair_intent_digest,
            "repair_intent_digest": repair_intent_digest,
        }
    )
    return request


def _matches_filters(view: DurableJobView, filters: JobFilters) -> bool:
    checks: tuple[bool, ...] = (
        filters.repository_id is None or view.repository_id == filters.repository_id,
        filters.lane_id is None or view.lane_id == filters.lane_id,
        filters.candidate_id is None or view.candidate_id == filters.candidate_id,
        filters.generation_id is None or view.generation_id == filters.generation_id,
        filters.correlation_id is None or view.correlation_id == filters.correlation_id,
        filters.kind is None or view.kind == filters.kind,
        filters.owner_id is None or view.owner_id == filters.owner_id,
        filters.resource_class is None or view.resource_class == filters.resource_class,
        filters.host_alias is None
        or view.target_alias == filters.host_alias
        or filters.host_alias in view.host_labels,
        filters.operation is None or view.operation == str(filters.operation),
        not filters.states or view.state in filters.states,
        filters.created_after is None
        or (view.created_at is not None and view.created_at >= filters.created_after),
        filters.created_before is None
        or (view.created_at is not None and view.created_at <= filters.created_before),
    )
    return all(checks)


def _cursor_from_view(view: DurableJobView) -> tuple[float, str]:
    created = view.created_at.timestamp() if view.created_at is not None else 0.0
    return (created, view.work_item_id)


class RepositoryJobService:
    """Domain-facing durable job operations over one injected port."""

    def __init__(self, port: RepositoryJobPort) -> None:
        # Runtime-checkable protocols only check attribute presence; validate
        # callability explicitly so a public owner/tenant-shaped object cannot
        # masquerade as an exact-input authority by setting a truthy marker.
        required = (
            "execution_input_authority_available",
            "submit",
            "get",
            "list_page",
            "cancel",
            "retry",
            "submit_repair",
            "get_exact_execution_input",
        )
        missing = [name for name in required if not callable(getattr(port, name, None))]
        if missing:
            raise TypeError(f"repository job port is missing: {', '.join(missing)}")
        self._port = port

    @staticmethod
    def _assert_scope(view: DurableJobView, auth: JobAuthorization) -> None:
        if view.tenant_id != auth.tenant_id or view.owner_id != auth.owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INTERNAL,
                "job authority returned a job outside the authenticated scope",
                job_id=view.job_id,
            )

    @staticmethod
    def _authorize_request(
        request: DevelopmentRequest | Mapping[str, Any], auth: JobAuthorization
    ) -> dict[str, Any]:
        raw = _as_mapping(request)
        tenant = raw.get("tenant_id")
        owner = raw.get("owner_id")
        if tenant != auth.tenant_id or owner != auth.owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "request identity does not match authenticated tenant/owner",
            )
        return raw

    def submit(
        self,
        request: DevelopmentRequest | Mapping[str, Any],
        *,
        auth: JobAuthorization,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        if not 1 <= max_attempts <= 100:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_REQUEST,
                "max_attempts must be between 1 and 100",
            )
        try:
            raw = self._authorize_request(request, auth)
            result = self._port.submit(raw, max_attempts=max_attempts, now=now)
        except Exception as exc:  # noqa: BLE001 - normalize port boundary errors
            raise _safe_port_error(exc) from exc
        self._assert_scope(result.job, auth)
        return result

    def _visible(self, job_id: str, auth: JobAuthorization) -> DurableJobView:
        try:
            job_id = _nonblank(job_id, "job_id")
            view = self._port.get(
                job_id, tenant_id=auth.tenant_id, owner_id=auth.owner_id
            )
        except Exception as exc:  # noqa: BLE001 - normalize port boundary errors
            raise _safe_port_error(exc, job_id=job_id) from exc
        if view is None:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "repository job is not accessible to this tenant/owner",
                job_id=job_id,
            )
        try:
            # Rebuild through a mapping: Pydantic's default model_validate on
            # an existing instance may trust ``model_copy(update=...)`` and
            # skip validators.  Authority projections are therefore always
            # revalidated before service logic or exact-input reads.
            raw_view = (
                view.model_dump(mode="python", exclude_none=False)
                if isinstance(view, BaseModel)
                else view
            )
            validated = DurableJobView.model_validate(raw_view)
        except (TypeError, ValueError) as exc:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INTERNAL,
                "repository job authority returned an invalid projection",
                job_id=job_id,
            ) from exc
        if validated.tenant_id != auth.tenant_id or validated.owner_id != auth.owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "repository job is not accessible to this tenant/owner",
                job_id=job_id,
            )
        return validated

    def get(self, job_id: str, *, auth: JobAuthorization) -> DurableJobView:
        """Return one authorized durable job, refusing hidden identities."""

        return self._visible(job_id, auth)

    def get_exact_execution_input(
        self,
        job_id: str,
        *,
        auth: JobAuthorization,
    ) -> BuildExecutionDescriptor | None:
        """Return the exact typed build body inside the authenticated scope.

        A production port must report a native atomic exact-input authority
        before any public-row lookup; otherwise this method fails closed.  The
        pre-native Graph port never reports availability.  The explicit test
        fake is the only owner-scoped implementation used to exercise pure
        service semantics before the EG native operation exists.
        """

        try:
            available = bool(self._port.execution_input_authority_available())
        except Exception as exc:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE,
                _SAFE_ERROR_MESSAGES[
                    RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
                ],
                job_id=job_id,
            ) from exc
        if not available:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE,
                _SAFE_ERROR_MESSAGES[
                    RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE
                ],
                job_id=job_id,
            )
        view = self._visible(job_id, auth)
        try:
            payload = self._port.get_exact_execution_input(
                job_id,
                tenant_id=auth.tenant_id,
                owner_id=auth.owner_id,
            )
        except Exception as exc:  # noqa: BLE001 - exact input is privacy-sensitive
            raise _safe_exact_input_error(exc, job_id=job_id) from exc
        if payload is None:
            if view.operation == "build" and view.operation_payload_digest is None:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED,
                    _SAFE_ERROR_MESSAGES[
                        RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED
                    ],
                    job_id=job_id,
                )
            if view.operation_payload_digest is not None:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INPUT_CONFLICT,
                    _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                    job_id=job_id,
                )
            return None
        try:
            validated = operation_payload_from_mapping(payload)
        except (TypeError, ValueError) as exc:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INPUT_CONFLICT,
                _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                job_id=job_id,
            ) from exc
        if (
            view.operation_payload_digest != validated.payload_digest
            or view.operation_payload_kind != validated.kind
            or view.operation_payload_version != validated.schema_version
            or view.operation != "build"
            or validated.repository_id != view.repository_id
            or validated.base_sha != view.base_sha
        ):
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INPUT_CONFLICT,
                _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                job_id=job_id,
            )
        return validated

    def get_execution_input(
        self,
        job_id: str,
        *,
        auth: JobAuthorization,
    ) -> BuildExecutionDescriptor | None:
        """Short spelling for :meth:`get_exact_execution_input`."""

        return self.get_exact_execution_input(job_id, auth=auth)

    def list(
        self,
        *,
        auth: JobAuthorization,
        filters: JobFilters | None = None,
    ) -> JobPage:
        supplied = filters or JobFilters()
        if supplied.owner_id is not None and supplied.owner_id != auth.owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "owner filter does not match authenticated owner",
            )
        scoped = supplied.model_copy(update={"owner_id": auth.owner_id})
        cursor = decode_cursor(auth.tenant_id, scoped.cursor)
        try:
            page = self._port.list_page(
                tenant_id=auth.tenant_id,
                filters=scoped,
                cursor=cursor,
            )
        except Exception as exc:  # noqa: BLE001 - normalize port boundary errors
            raise _safe_port_error(exc) from exc
        for view in page.items:
            if view.tenant_id != auth.tenant_id or view.owner_id != auth.owner_id:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INTERNAL,
                    "job authority returned a job outside the authenticated scope",
                )
        return page

    def cancel(
        self,
        job_id: str,
        *,
        auth: JobAuthorization,
        reason: str = "cancelled by owner",
        now: datetime | None = None,
    ) -> JobMutationResult:
        view = self._visible(job_id, auth)
        if view.state == JobState.CANCELLED:
            return JobMutationResult(job=view, changed=False)
        if view.state not in {
            JobState.SUBMITTED,
            JobState.READY,
            JobState.LEASED,
            JobState.RUNNING,
        }:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_STATE,
                f"job in state {view.state.value} cannot be cancelled",
                job_id=job_id,
            )
        try:
            result = self._port.cancel(
                job_id,
                tenant_id=auth.tenant_id,
                owner_id=auth.owner_id,
                reason=_nonblank(reason, "reason"),
                now=now,
            )
        except Exception as exc:  # noqa: BLE001 - normalize port boundary errors
            raise _safe_port_error(
                exc,
                job_id=job_id,
                value_error_code=RepositoryJobServiceCode.INTERNAL,
            ) from exc
        self._assert_scope(result.job, auth)
        return result

    def retry(
        self,
        job_id: str,
        *,
        auth: JobAuthorization,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        view = self._visible(job_id, auth)
        if view.state not in {JobState.FAILED, JobState.DEAD_LETTER}:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_STATE,
                f"job in state {view.state.value} cannot be retried",
                job_id=job_id,
            )
        if view.attempt >= view.max_attempts:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_STATE,
                "job has exhausted its retry budget",
                job_id=job_id,
            )
        try:
            result = self._port.retry(
                view,
                tenant_id=auth.tenant_id,
                owner_id=auth.owner_id,
                now=now,
            )
        except Exception as exc:  # noqa: BLE001 - normalize port boundary errors
            raise _safe_port_error(
                exc,
                job_id=job_id,
                value_error_code=RepositoryJobServiceCode.INTERNAL,
            ) from exc
        self._assert_scope(result.job, auth)
        return result.model_copy(update={"retry_of": job_id})

    def reconcile(
        self,
        observation: ReconciliationObservation,
        *,
        auth: JobAuthorization,
        enqueue_repairs: bool = False,
        now: datetime | None = None,
    ) -> ReconciliationResult:
        """Compare one authorized job to a read-only observation.

        ``enqueue_repairs`` only submits an idempotent ``repair`` WorkItem. It
        never invokes Git, signals a process, deletes artifacts, or changes a
        target branch.
        """

        view = self._visible(observation.job_id, auth)
        if (
            observation.repository_id is not None
            and observation.repository_id != view.repository_id
        ):
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "observation repository does not match durable job",
                job_id=view.job_id,
            )
        classifications, details = _classify(view, observation)
        proposal = None
        if classifications and classifications != (ReconciliationClass.CLEAN,):
            repair_identity = {
                "contract": "repository-repair:v1",
                "source_job_id": view.job_id,
                "classifications": tuple(classifications),
            }
            repair_digest = canonical_digest(repair_identity)
            proposal = RepairProposal(
                repair_id="repair:" + repair_digest,
                source_job_id=view.job_id,
                classifications=classifications,
                idempotency_key="repair:" + repair_digest,
            )
            if enqueue_repairs:
                try:
                    submitted = self._port.submit_repair(
                        view,
                        proposal,
                        tenant_id=auth.tenant_id,
                        owner_id=auth.owner_id,
                        session_id=auth.session_id,
                        now=now,
                    )
                except Exception as exc:  # noqa: BLE001 - normalize port errors
                    raise _safe_port_error(
                        exc,
                        job_id=view.job_id,
                        value_error_code=RepositoryJobServiceCode.INTERNAL,
                    ) from exc
                self._assert_scope(submitted.job, auth)
                proposal = proposal.model_copy(
                    update={
                        "preview_only": False,
                        "enqueued_job_id": submitted.job.job_id,
                    }
                )
        finding = ReconciliationFinding(
            job=view,
            classifications=classifications,
            details=details,
            repair=proposal,
        )
        return ReconciliationResult(findings=(finding,), scanned=1, exhausted=True)

    def shadow_compare(
        self,
        job_id: str,
        legacy_record: Mapping[str, Any],
        *,
        auth: JobAuthorization,
    ) -> ShadowMismatch | None:
        """Compare legacy MCP data without allowing it to write durable state."""

        durable = self._visible(job_id, auth)
        mismatch = LegacyShadowAdapter.compare(durable, legacy_record)
        return mismatch


def _classify(
    view: DurableJobView, observation: ReconciliationObservation
) -> tuple[tuple[ReconciliationClass, ...], dict[str, Any]]:
    classes: list[ReconciliationClass] = []
    details: dict[str, Any] = {}
    if observation.worktree_expected and observation.worktree_present is False:
        classes.append(ReconciliationClass.MISSING_WORKTREE)
        details["worktree"] = "missing"
    if observation.process_present is False and view.state in {
        JobState.LEASED,
        JobState.RUNNING,
    }:
        classes.append(ReconciliationClass.STALE_PROCESS)
        details["process"] = "missing"
    elif observation.process_present:
        stale_at = observation.process_last_heartbeat
        if (
            stale_at is None
            or (observation.observed_at - stale_at).total_seconds()
            > observation.process_stale_after_seconds
        ):
            classes.append(ReconciliationClass.STALE_PROCESS)
            details["process"] = "stale"
    expected_artifacts = set(observation.expected_artifact_refs)
    orphan_artifacts = sorted(
        set(observation.observed_artifact_refs) - expected_artifacts
    )
    if orphan_artifacts:
        classes.append(ReconciliationClass.ORPHAN_ARTIFACT)
        details["orphan_artifacts"] = orphan_artifacts
    expected_fence = observation.expected_fence
    if expected_fence is None:
        expected_fence = view.lease_fence
    if (
        observation.observed_fence is not None
        and expected_fence is not None
        and observation.observed_fence != expected_fence
    ):
        classes.append(ReconciliationClass.STALE_FENCE)
        details["expected_fence"] = expected_fence
        details["observed_fence"] = observation.observed_fence
    if (
        observation.expected_target_sha is not None
        and observation.observed_target_sha is not None
        and observation.expected_target_sha != observation.observed_target_sha
    ):
        classes.append(ReconciliationClass.TARGET_DRIFT)
        details["expected_target_sha"] = observation.expected_target_sha
        details["observed_target_sha"] = observation.observed_target_sha
    if observation.already_completed_effect:
        classes.append(ReconciliationClass.ALREADY_COMPLETED_EFFECT)
        details["completed_effect"] = True
    if not classes:
        classes.append(ReconciliationClass.CLEAN)
    return tuple(classes), details


class LegacyShadowAdapter:
    """Pure legacy comparison helper used during RMDD-20 strangler cutover."""

    _FIELD_MAP: Mapping[str, str] = {
        "status": "state",
        "repo_name": "repository_id",
        "owner": "owner_id",
        "result_ref": "result_ref",
        "error": "error_ref",
    }

    @classmethod
    def compare(
        cls, durable: DurableJobView, legacy_record: Mapping[str, Any]
    ) -> ShadowMismatch | None:
        durable_values = {
            "state": durable.state.value,
            "repository_id": durable.repository_id,
            "owner_id": durable.owner_id,
            "result_ref": durable.result_ref,
            "error_ref": durable.error_ref,
        }
        fields: list[str] = []
        legacy_values: dict[str, Any] = {}
        for legacy_name, durable_name in cls._FIELD_MAP.items():
            if legacy_name not in legacy_record:
                continue
            legacy_value = legacy_record[legacy_name]
            legacy_values[durable_name] = legacy_value
            if legacy_value != durable_values[durable_name]:
                fields.append(durable_name)
        if not fields:
            return None
        return ShadowMismatch(
            job_id=durable.job_id,
            fields=tuple(sorted(set(fields))),
            durable={field: durable_values[field] for field in sorted(set(fields))},
            legacy={field: legacy_values[field] for field in sorted(set(fields))},
        )


class FakeRepositoryJobPort:
    """Test-only durable fake with an explicitly trusted exact-input seam."""

    def __init__(self) -> None:
        self.rows: dict[str, DurableJobView] = {}
        # The fake mirrors the authority's separate typed extension store;
        # ordinary rows retain only the summary fields.
        self.execution_inputs: dict[str, BuildExecutionDescriptor] = {}
        self.submit_calls = 0
        self.cancel_calls = 0

    def execution_input_authority_available(self) -> bool:
        """Opt into the test-only exact-input implementation."""

        return True

    def snapshot(self) -> dict[str, Any]:
        """Return a restart-safe fake snapshot, including typed bodies."""

        return {
            "rows": {
                job_id: view.model_dump(mode="json", exclude_none=False)
                for job_id, view in self.rows.items()
            },
            "execution_inputs": {
                job_id: payload.model_dump(mode="json", exclude_none=False)
                for job_id, payload in self.execution_inputs.items()
            },
        }

    @classmethod
    def from_snapshot(cls, snapshot: Mapping[str, Any]) -> FakeRepositoryJobPort:
        """Restore a fake without dropping the separate exact-input store."""

        if not isinstance(snapshot, Mapping):
            raise TypeError("fake repository job snapshot must be a mapping")
        rows = snapshot.get("rows", {})
        inputs = snapshot.get("execution_inputs", {})
        if not isinstance(rows, Mapping) or not isinstance(inputs, Mapping):
            raise ValueError("fake repository job snapshot has invalid stores")
        restored = cls()
        try:
            restored.rows = {
                str(job_id): DurableJobView.model_validate(value)
                for job_id, value in rows.items()
            }
            restored.execution_inputs = {
                str(job_id): operation_payload_from_mapping(value)
                for job_id, value in inputs.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError("fake repository job snapshot is invalid") from exc
        return restored

    @staticmethod
    def _job_id(tenant_id: str, idempotency_key: str) -> str:
        digest = hashlib.sha256(f"{tenant_id}\0{idempotency_key}".encode()).hexdigest()
        return f"rmjob:{digest[:8]}-{digest[8:12]}-{digest[12:16]}-{digest[16:20]}-{digest[20:32]}"

    @staticmethod
    def _work_item_id(job_id: str) -> str:
        return "workitem:repository_manager:" + job_id.split(":", 1)[1]

    @staticmethod
    def _request_payload(
        request: Mapping[str, Any],
    ) -> BuildExecutionDescriptor | None:
        raw_payload = request.get("operation_payload")
        if raw_payload is None:
            # An untyped legacy build descriptor must never become executable
            # input merely because it arrived under an old mapping key.
            if request.get("build_descriptor") is not None:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INVALID_REQUEST,
                    "repository execution input must use the typed operation_payload extension",
                )
            return None
        try:
            return operation_payload_from_mapping(raw_payload)
        except (TypeError, ValueError) as exc:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_REQUEST,
                "repository operation payload is invalid",
            ) from exc

    @staticmethod
    def _request_view(
        request: Mapping[str, Any], *, now: datetime, max_attempts: int
    ) -> DurableJobView:
        raw = dict(request)
        payload = FakeRepositoryJobPort._request_payload(raw)
        repository = dict(raw.get("repository") or {})
        resources = dict(raw.get("resources") or {})
        target = dict(raw.get("target") or {})
        repository_id = str(
            raw.get("repository_id") or repository.get("repository_id") or ""
        )
        tenant_id = str(raw.get("tenant_id") or raw.get("tenant") or "")
        owner_id = str(raw.get("owner_id") or "")
        session_id = str(raw.get("session_id") or "")
        job_id = FakeRepositoryJobPort._job_id(tenant_id, str(raw["idempotency_key"]))
        operation = str(raw.get("operation"))
        if payload is not None:
            if operation != "build":
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INVALID_REQUEST,
                    "operation payload discriminator does not match the operation",
                )
            if payload.repository_id != repository_id:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INVALID_REQUEST,
                    "operation payload repository does not match the job",
                )
            if payload.base_sha != str(raw["base_sha"]):
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INVALID_REQUEST,
                    "operation payload base does not match the job",
                )
        state = JobState.READY if not raw.get("dependencies") else JobState.SUBMITTED
        return DurableJobView(
            job_id=job_id,
            work_item_id=FakeRepositoryJobPort._work_item_id(job_id),
            request_id=str(raw["request_id"]),
            operation=operation,
            kind=_OPERATION_KIND.get(operation, "repository.operation"),
            state=state,
            repository_id=repository_id,
            tenant_id=tenant_id,
            owner_id=owner_id,
            session_id=session_id,
            base_ref=str(raw["base_ref"]),
            base_sha=str(raw["base_sha"]),
            target_kind=str(target.get("kind") or "local"),
            target_alias=target.get("alias"),
            lane_id=raw.get("lane_id"),
            candidate_id=raw.get("candidate_id"),
            generation_id=raw.get("generation_id"),
            dependencies=tuple(str(dep) for dep in raw.get("dependencies") or ()),
            # AU exposes its complete normalized immutable-request digest in
            # this projection, not the caller's optional source-input digest.
            # Always bind the fake row to the whole request as well so a
            # caller cannot keep ``input_digest`` constant while changing a
            # different immutable field under the same idempotency key.
            input_digest=canonical_digest(raw),
            config_digest=raw.get("config_digest"),
            correlation_id=raw.get("correlation_id") or raw.get("request_id"),
            resource_class=str(resources.get("resource_class") or "light-check"),
            concurrency_key=str(resources.get("concurrency_key") or "light-check"),
            fairness_group=str(resources.get("fairness_group") or "default"),
            priority=int(raw.get("priority") or resources.get("priority") or 0),
            cpu_weight=int(resources.get("cpu_weight", 1)),
            memory_mib=int(resources.get("memory_mib", 256)),
            disk_mib=int(resources.get("disk_mib", 256)),
            process_slots=int(resources.get("process_slots", 1)),
            host_labels=tuple(
                str(value) for value in resources.get("host_labels") or ()
            ),
            preferred_target=dict(resources.get("preferred_target") or {}),
            required_target=(
                None
                if resources.get("required_target") is None
                else dict(resources["required_target"])
            ),
            anti_affinity=tuple(
                str(value) for value in resources.get("anti_affinity") or ()
            ),
            queue_deadline=_parse_datetime(resources.get("queue_deadline")),
            disk_low_watermark_mib=resources.get("disk_low_watermark_mib"),
            disk_high_watermark_mib=resources.get("disk_high_watermark_mib"),
            consent=dict(raw.get("consent") or {}),
            attempt=0,
            max_attempts=max_attempts,
            retry_class=raw.get("retry_class"),
            created_at=now,
            updated_at=now,
            operation_payload_kind=(None if payload is None else payload.kind),
            operation_payload_version=(
                None if payload is None else payload.schema_version
            ),
            operation_payload_digest=(
                None if payload is None else payload.payload_digest
            ),
        )

    def submit(
        self,
        request: Mapping[str, Any],
        *,
        max_attempts: int = 3,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        self.submit_calls += 1
        timestamp = _timestamp(now)
        raw = _as_mapping(request)
        payload = self._request_payload(raw)
        tenant_id = str(raw.get("tenant_id") or "")
        idem = str(raw.get("idempotency_key") or "")
        job_id = self._job_id(tenant_id, idem)
        existing = self.rows.get(job_id)
        if existing is not None:
            incoming = self._request_view(raw, now=timestamp, max_attempts=max_attempts)
            if (
                existing.operation_payload_digest != incoming.operation_payload_digest
                or existing.operation_payload_kind != incoming.operation_payload_kind
                or existing.operation_payload_version
                != incoming.operation_payload_version
            ):
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INPUT_CONFLICT,
                    _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                    job_id=job_id,
                )
            if existing.input_digest != incoming.input_digest:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.DUPLICATE,
                    "idempotency key was reused with changed immutable input",
                    job_id=job_id,
                )
            return JobSubmitResult(job=existing, deduplicated=True)
        view = self._request_view(raw, now=timestamp, max_attempts=max_attempts)
        self.rows[job_id] = view
        if payload is not None:
            self.execution_inputs[job_id] = payload
        return JobSubmitResult(job=view)

    def get(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> DurableJobView | None:
        view = self.rows.get(job_id)
        return (
            view
            if view is not None
            and view.tenant_id == tenant_id
            and view.owner_id == owner_id
            else None
        )

    def get_exact_execution_input(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> BuildExecutionDescriptor | None:
        view = self.get(job_id, tenant_id=tenant_id, owner_id=owner_id)
        if view is None:
            return None
        try:
            view = DurableJobView.model_validate(
                view.model_dump(mode="python", exclude_none=False)
            )
        except (TypeError, ValueError) as exc:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INPUT_CONFLICT,
                _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                job_id=job_id,
            ) from exc
        stored = self.execution_inputs.get(job_id)
        if stored is None:
            if view.operation_payload_digest is not None:
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.INPUT_CONFLICT,
                    _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                    job_id=job_id,
                )
            if view.operation == "build":
                raise RepositoryJobServiceError(
                    RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED,
                    _SAFE_ERROR_MESSAGES[
                        RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED
                    ],
                    job_id=job_id,
                )
            return None
        try:
            payload = operation_payload_from_mapping(stored)
        except (TypeError, ValueError) as exc:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INPUT_CONFLICT,
                _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                job_id=job_id,
            ) from exc
        if payload.payload_digest != view.operation_payload_digest:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INPUT_CONFLICT,
                _SAFE_ERROR_MESSAGES[RepositoryJobServiceCode.INPUT_CONFLICT],
                job_id=job_id,
            )
        return payload

    def get_execution_input(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
    ) -> BuildExecutionDescriptor | None:
        return self.get_exact_execution_input(
            job_id, tenant_id=tenant_id, owner_id=owner_id
        )

    def list_page(
        self,
        *,
        tenant_id: str,
        filters: JobFilters,
        cursor: tuple[float, str] | None,
    ) -> JobPage:
        values = sorted(
            (view for view in self.rows.values() if view.tenant_id == tenant_id),
            key=lambda view: _cursor_from_view(view),
        )
        if cursor is not None:
            values = [view for view in values if _cursor_from_view(view) > cursor]
        page_values = values[: filters.limit]
        matching = tuple(
            view for view in page_values if _matches_filters(view, filters)
        )
        next_cursor = (
            encode_cursor(tenant_id, _cursor_from_view(page_values[-1]))
            if len(page_values) == filters.limit
            else None
        )
        return JobPage(
            items=matching,
            next_cursor=next_cursor,
            scanned=len(page_values),
            exhausted=len(page_values) < filters.limit,
        )

    def cancel(
        self,
        job_id: str,
        *,
        tenant_id: str,
        owner_id: str,
        reason: str,
        now: datetime | None = None,
    ) -> JobMutationResult:
        del reason
        view = self.get(job_id, tenant_id=tenant_id, owner_id=owner_id)
        if view is None:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "repository job is not accessible",
                job_id=job_id,
            )
        self.cancel_calls += 1
        if view.state == JobState.CANCELLED:
            return JobMutationResult(job=view, changed=False)
        updated = view.model_copy(
            update={"state": JobState.CANCELLED, "updated_at": _timestamp(now)}
        )
        self.rows[job_id] = updated
        return JobMutationResult(job=updated, changed=True)

    def retry(
        self,
        job: DurableJobView,
        *,
        tenant_id: str,
        owner_id: str,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        if job.tenant_id != tenant_id or job.owner_id != owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "retry identity does not match the durable job",
                job_id=job.job_id,
            )
        operation_payload = (
            self.get_exact_execution_input(
                job.job_id, tenant_id=tenant_id, owner_id=owner_id
            )
            if job.operation == "build"
            else None
        )
        raw = _retry_request_mapping(
            job,
            owner_id=owner_id,
            attempt=job.attempt + 1,
            operation_payload=operation_payload,
        )
        remaining_attempts = job.max_attempts - job.attempt
        if remaining_attempts < 1:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.INVALID_STATE,
                "job has exhausted its retry budget",
                job_id=job.job_id,
            )
        result = self.submit(raw, max_attempts=remaining_attempts, now=now)
        return result.model_copy(update={"retry_of": job.job_id})

    def submit_repair(
        self,
        job: DurableJobView,
        proposal: RepairProposal,
        *,
        tenant_id: str,
        owner_id: str,
        session_id: str | None,
        now: datetime | None = None,
    ) -> JobSubmitResult:
        if job.tenant_id != tenant_id or job.owner_id != owner_id:
            raise RepositoryJobServiceError(
                RepositoryJobServiceCode.UNAUTHORIZED,
                "repair identity does not match the durable job",
                job_id=job.job_id,
            )
        operation_payload = (
            self.get_exact_execution_input(
                job.job_id, tenant_id=tenant_id, owner_id=owner_id
            )
            if job.operation == "build"
            else None
        )
        raw = _repair_request_mapping(
            job,
            proposal,
            owner_id=owner_id,
            session_id=session_id,
            operation_payload=operation_payload,
        )
        result = self.submit(raw, max_attempts=1, now=now)
        return result.model_copy(update={"retry_of": job.job_id})


def _parse_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
    raise ValueError("timestamp must be an aware datetime or ISO string")


__all__ = [
    "BuildExecutionDescriptor",
    "DurableJobView",
    "FakeRepositoryJobPort",
    "GraphRepositoryJobPort",
    "JobAuthorization",
    "JobFilters",
    "JobMutationResult",
    "JobPage",
    "JobSubmitResult",
    "LegacyShadowAdapter",
    "ReconciliationClass",
    "ReconciliationFinding",
    "ReconciliationObservation",
    "ReconciliationResult",
    "RepairProposal",
    "RepositoryJobPort",
    "RepositoryJobService",
    "RepositoryJobServiceCode",
    "RepositoryJobServiceError",
    "ShadowMismatch",
    "decode_cursor",
    "encode_cursor",
]
