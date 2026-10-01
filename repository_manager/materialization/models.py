"""Typed contracts for governed proposal-to-Git materialization (RM-MATERIALIZE-001).

``ChangeProposal/v1`` is the only admissible input: a reviewer-approved,
bounded patch against an exact base commit.  ``ApprovalProof/v1`` is the
caller-supplied, signed evidence that a proposal was actually approved.
``MaterializationReceipt/v1`` is the immutable, redacted record this package
returns for every outcome -- including a refusal.  All three are closed
(``extra="forbid"``) so an unexpected field is a construction error, not a
silently ignored one, and every identity-bearing field is validated before a
caller can reach :mod:`repository_manager.materialization.service`.

Every model exposes its own ``digest`` property (canonical SHA-256 over its
own fields, the same ``canonical_digest``/``canonicalize`` helpers the rest of
repository-manager's durable contracts use) rather than storing a self digest
field, so there is never a "digest of everything except my own digest"
special case to get wrong.
"""

from __future__ import annotations

import hashlib
import re
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    StrictStr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from repository_manager.development.serialization import canonical_digest

__all__ = [
    "MAX_PATCH_BYTES",
    "ApprovalProof",
    "ChangeProposal",
    "MaterializationErrorCode",
    "MaterializationReceipt",
    "MaterializationState",
]

#: A materialization patch is a bounded, reviewed, text-only diff -- not an
#: artifact channel.  2 MiB comfortably covers a legitimate reviewed change
#: while keeping "oversized patch" (RM-MATERIALIZE-02) a cheap, pre-worktree
#: rejection.
MAX_PATCH_BYTES = 2 * 1024 * 1024

_SHA_RE = re.compile(r"[0-9a-f]{40}")
_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _nonblank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-blank string")
    if _CONTROL_RE.search(value):
        raise ValueError(f"{field_name} contains control characters")
    return value


def _sha40(value: str, field_name: str) -> str:
    value = _nonblank(value, field_name)
    if not _SHA_RE.fullmatch(value):
        raise ValueError(f"{field_name} must be a 40-character lowercase git SHA")
    return value


def _hex_digest(value: str, field_name: str) -> str:
    value = _nonblank(value, field_name)
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"{field_name} must be a lowercase sha256 hex digest")
    return value


def _relative_path(value: str, field_name: str) -> str:
    value = _nonblank(value, field_name)
    if value.startswith("/") or "\\" in value or re.match(r"^[A-Za-z]:", value):
        raise ValueError(f"{field_name} must be repository-relative")
    parts = value.split("/")
    if any(part in {"..", ""} for part in parts):
        raise ValueError(f"{field_name} contains an unsafe path component")
    return value


class MaterializationState(StrEnum):
    """Stages of plan.md's sequence, plus the one terminal refusal state."""

    PENDING = "pending"
    APPLIED = "applied"
    VALIDATED = "validated"
    COMMITTED = "committed"
    QUEUED = "queued"
    MERGED = "merged"
    REFUSED = "refused"


class MaterializationErrorCode(StrEnum):
    """The stable error vocabulary plan.md commits callers to."""

    UNAUTHORIZED = "unauthorized"
    EXPIRED_APPROVAL = "expired_approval"
    STALE_BASE = "stale_base"
    PATH_REJECTED = "path_rejected"
    GATE_FAILED = "gate_failed"
    QUEUE_CONFLICT = "queue_conflict"
    UNCERTAIN_DELIVERY = "uncertain_delivery"


class ChangeProposal(BaseModel):
    """``ChangeProposal/v1`` -- the one admissible materialization request."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["repository.change-proposal/v1"] = "repository.change-proposal/v1"
    schema_version: Literal["1"] = "1"
    proposal_id: StrictStr
    repository_id: StrictStr
    tenant_id: StrictStr
    base_ref: StrictStr = "main"
    base_sha: StrictStr
    approved_paths: tuple[StrictStr, ...]
    patch_digest: StrictStr
    patch_text: StrictStr
    decision_identity: StrictStr
    validation_profile: StrictStr
    expires_at: datetime

    @field_validator(
        "proposal_id",
        "repository_id",
        "tenant_id",
        "base_ref",
        "decision_identity",
        "validation_profile",
    )
    @classmethod
    def _validate_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("base_sha")
    @classmethod
    def _validate_base_sha(cls, value: str) -> str:
        return _sha40(value, "base_sha")

    @field_validator("patch_digest")
    @classmethod
    def _validate_patch_digest(cls, value: str) -> str:
        return _hex_digest(value, "patch_digest")

    @field_validator("approved_paths")
    @classmethod
    def _validate_approved_paths(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("approved_paths must not be empty")
        paths = tuple(_relative_path(item, "approved_paths") for item in value)
        return tuple(sorted(set(paths)))

    @field_validator("expires_at")
    @classmethod
    def _validate_expires_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("expires_at must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_patch(self) -> ChangeProposal:
        encoded = self.patch_text.encode("utf-8")
        if not self.patch_text:
            raise ValueError("patch_text must not be empty")
        if len(encoded) > MAX_PATCH_BYTES:
            raise ValueError(
                f"patch_text exceeds the {MAX_PATCH_BYTES}-byte materialization bound"
            )
        if hashlib.sha256(encoded).hexdigest() != self.patch_digest:
            raise ValueError("patch_digest does not match patch_text")
        return self

    @property
    def digest(self) -> str:
        """Canonical digest an :class:`ApprovalProof` must be bound to."""

        return canonical_digest(self)


class ApprovalProof(BaseModel):
    """``ApprovalProof/v1`` -- caller-supplied evidence of an approval decision.

    Structural binding (same proposal, repository, tenant, actor, not expired)
    is checked locally by
    :func:`repository_manager.materialization.service.check_binding`; the
    ``signature`` field's authenticity is verified only by the pluggable
    :class:`repository_manager.materialization.ports.ApprovalAuthority` --
    this model never claims to have verified itself.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["repository.approval-proof/v1"] = "repository.approval-proof/v1"
    schema_version: Literal["1"] = "1"
    approval_id: StrictStr
    proposal_digest: StrictStr
    repository_id: StrictStr
    tenant_id: StrictStr
    actor_id: StrictStr
    issued_at: datetime
    expires_at: datetime
    signature: StrictStr

    @field_validator("approval_id", "repository_id", "tenant_id", "actor_id", "signature")
    @classmethod
    def _validate_identifiers(cls, value: str, info: ValidationInfo) -> str:
        return _nonblank(value, info.field_name)

    @field_validator("proposal_digest")
    @classmethod
    def _validate_proposal_digest(cls, value: str) -> str:
        return _hex_digest(value, "proposal_digest")

    @field_validator("issued_at", "expires_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime, info: ValidationInfo) -> datetime:
        if value.tzinfo is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value

    @property
    def digest(self) -> str:
        return canonical_digest(self)


class MaterializationReceipt(BaseModel):
    """``MaterializationReceipt/v1`` -- the immutable record of one outcome.

    Deliberately carries no patch bytes and no local filesystem path: only
    digests, repository/tenant/actor identity, Git identity, and the queue's
    own report.  Still built for a refusal -- ``commit_sha``/``tree_sha``/
    ``queue_*`` stay ``None`` and ``error_code``/``error_detail`` explain why.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: Literal["repository.materialization-receipt/v1"] = (
        "repository.materialization-receipt/v1"
    )
    schema_version: Literal["1"] = "1"
    idempotency_key: StrictStr
    proposal_id: StrictStr
    proposal_digest: StrictStr
    approval_id: StrictStr
    approval_digest: StrictStr
    repository_id: StrictStr
    tenant_id: StrictStr
    base_sha: StrictStr
    actor_id: StrictStr
    gate_profile: StrictStr
    state: MaterializationState
    created_at: datetime
    updated_at: datetime
    commit_sha: StrictStr | None = None
    tree_sha: StrictStr | None = None
    queue_branch: StrictStr | None = None
    queue_accepted: bool | None = None
    queue_detail: StrictStr = ""
    error_code: MaterializationErrorCode | None = None
    error_detail: StrictStr = ""

    @field_validator("created_at", "updated_at")
    @classmethod
    def _validate_timestamps(cls, value: datetime, info: ValidationInfo) -> datetime:
        if value.tzinfo is None:
            raise ValueError(f"{info.field_name} must be timezone-aware")
        return value

    @model_validator(mode="after")
    def _validate_terminal_shape(self) -> MaterializationReceipt:
        """Only two shapes are legal: a refusal, or an uncertain-delivery pend.

        ``REFUSED`` is a content-based, terminal refusal and must name
        whichever stable error it refused for.  ``PENDING`` carrying
        ``UNCERTAIN_DELIVERY`` is the ONE other shape allowed to hold an
        error_code -- RM-MATERIALIZE-06's "stays pending" outcome for an
        unavailable dependency, never a claim about the proposal's content.
        Every other state must carry no error_code at all.
        """

        if self.state is MaterializationState.REFUSED:
            if self.error_code is None:
                raise ValueError("a refused receipt must carry an error_code")
            return self
        if self.state is MaterializationState.PENDING:
            if self.error_code not in (None, MaterializationErrorCode.UNCERTAIN_DELIVERY):
                raise ValueError(
                    "a pending receipt may only carry UNCERTAIN_DELIVERY, if any"
                )
            return self
        if self.error_code is not None:
            raise ValueError("only a refused or pending receipt may carry an error_code")
        return self

    @property
    def digest(self) -> str:
        return canonical_digest(self)
