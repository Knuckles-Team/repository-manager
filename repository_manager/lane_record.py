"""Durable lane record contracts and collision-safe repository identities.

The development contracts package deliberately stops at the typed ``LaneReference``
boundary.  This module adds the small amount of lifecycle information needed by the
local registry: an immutable request digest, a fencing token, and explicit ownership
metadata.  The registry stores the records; this module has no persistence or Git
side effects.
"""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class LaneLifecycleState(StrEnum):
    """States owned by the lane registry.

    ``observed_legacy`` is intentionally outside the managed lifecycle.  A
    discovery scan may create that state, but ownership is not claimed until an
    operator explicitly adopts the record.
    """

    ALLOCATING = "allocating"
    ACTIVE = "active"
    SUBMITTED = "submitted"
    LANDED = "landed"
    ABORTED = "aborted"
    REJECTED = "rejected"
    EXPIRED = "expired"
    QUARANTINED = "quarantined"
    OBSERVED_LEGACY = "observed_legacy"


TERMINAL_STATES = frozenset(
    {
        LaneLifecycleState.LANDED,
        LaneLifecycleState.ABORTED,
        LaneLifecycleState.REJECTED,
        LaneLifecycleState.QUARANTINED,
    }
)
ACTIVE_STATES = frozenset(
    {
        LaneLifecycleState.ALLOCATING,
        LaneLifecycleState.ACTIVE,
        LaneLifecycleState.SUBMITTED,
        LaneLifecycleState.EXPIRED,
    }
)


def _nonblank(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValueError(f"{field_name} must be a non-blank string")
    if any(ord(char) < 0x20 for char in value):
        raise ValueError(f"{field_name} must not contain control characters")
    return value


def _canonical_path(value: str | Path) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return str(path.resolve(strict=False))


def repository_id_for(path: str | Path, *, origin: str | None = None) -> str:
    """Return a stable repository identity that includes its canonical root.

    A basename is a display label, not an identity.  Including the canonical
    root prevents two repositories named ``repository-manager`` in different
    workspace roots from sharing a lane or idempotency key.  An origin, when
    supplied, is only an additional discriminator; it is never used without
    the path because mirrors can legitimately expose the same remote.
    """

    canonical = _canonical_path(path)
    origin_value = "" if origin is None else _nonblank(origin, "origin")
    material = "repository-manager:repository:v1\0" + canonical + "\0" + origin_value
    return "repo:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def lane_id_for(repository_id: str, request_key: str) -> str:
    """Derive a globally stable lane id from immutable allocation identity."""

    repository = _nonblank(repository_id, "repository_id")
    request = _nonblank(request_key, "request_key")
    material = "repository-manager:lane:v1\0" + repository + "\0" + request
    return "lane:" + hashlib.sha256(material.encode("utf-8")).hexdigest()


def new_fence() -> str:
    """Create an opaque fencing token for one lane attempt."""

    return "fence:" + uuid.uuid4().hex


def _tuple_strings(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence")
    try:
        values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence") from exc
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} entries must be strings")
        result.append(_nonblank(item, field_name))
    return tuple(sorted(set(result)))


class LaneRecord(BaseModel):
    """Immutable projection of one durable lane registry row."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=False)

    contract_version: str = "1"
    lane_id: str
    request_key: str
    input_digest: str
    repository_id: str
    repository_path: str
    repository_origin: str | None = None
    branch: str
    base_ref: str = "main"
    base_sha: str = ""
    worktree_path: str
    owner_id: str | None = None
    session_id: str | None = None
    host_id: str | None = None
    fence: str = Field(default_factory=new_fence)
    attempt: int = Field(default=1, ge=1)
    created_at: datetime
    heartbeat_at: datetime
    expires_at: datetime
    ttl_seconds: int = Field(default=3600, ge=1)
    predicted_disk_bytes: int = Field(default=0, ge=0)
    observed_disk_bytes: int = Field(default=0, ge=0)
    disk_budget_bytes: int = Field(default=1, ge=1)
    concept_ids: tuple[str, ...] = ()
    active_job_ids: tuple[str, ...] = ()
    active_candidate_id: str | None = None
    cleanup_anchors: tuple[str, ...] = ()
    state: LaneLifecycleState = LaneLifecycleState.ALLOCATING
    version: int = Field(default=1, ge=1)
    last_transition: str = "allocate"
    last_error: str | None = None

    @field_validator(
        "lane_id",
        "request_key",
        "input_digest",
        "repository_id",
        "repository_path",
        "branch",
        "base_ref",
        "worktree_path",
        "fence",
        "last_transition",
    )
    @classmethod
    def validate_required_text(cls, value: str, info: Any) -> str:
        return _nonblank(value, info.field_name)

    @field_validator(
        "repository_origin",
        "owner_id",
        "session_id",
        "host_id",
        "active_candidate_id",
        "last_error",
    )
    @classmethod
    def validate_optional_text(cls, value: str | None, info: Any) -> str | None:
        return (
            None if value is None or value == "" else _nonblank(value, info.field_name)
        )

    @field_validator("repository_path", "worktree_path")
    @classmethod
    def validate_absolute_paths(cls, value: str, info: Any) -> str:
        canonical = _canonical_path(value)
        if canonical != value:
            raise ValueError(f"{info.field_name} must be canonical and absolute")
        return value

    @field_validator("concept_ids", "active_job_ids", "cleanup_anchors", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object, info: Any) -> tuple[str, ...]:
        return _tuple_strings(value, info.field_name)

    @field_validator("created_at", "heartbeat_at", "expires_at")
    @classmethod
    def validate_timestamps(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("lane timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_record(self) -> LaneRecord:
        if self.expires_at <= self.heartbeat_at:
            raise ValueError("lane expiry must be after heartbeat")
        if self.predicted_disk_bytes > self.disk_budget_bytes:
            raise ValueError("predicted disk usage exceeds the lane disk budget")
        if self.state == LaneLifecycleState.OBSERVED_LEGACY:
            if self.owner_id is not None or self.session_id is not None:
                raise ValueError("observed legacy lanes cannot claim an owner")
        if self.state in TERMINAL_STATES and self.active_job_ids:
            raise ValueError("terminal lanes cannot retain active jobs")
        return self

    @property
    def managed(self) -> bool:
        return self.state != LaneLifecycleState.OBSERVED_LEGACY

    @property
    def active(self) -> bool:
        return self.state in ACTIVE_STATES

    @property
    def repository_basename(self) -> str:
        """Display-only basename; never used for identity or uniqueness."""

        return Path(self.repository_path).name

    def immutable_payload(self) -> dict[str, Any]:
        """Return fields that cannot change under one request key."""

        return {
            "repository_id": self.repository_id,
            "repository_path": self.repository_path,
            "repository_origin": self.repository_origin,
            "branch": self.branch,
            "base_ref": self.base_ref,
            "base_sha": self.base_sha,
            "worktree_path": self.worktree_path,
            "owner_id": self.owner_id,
            "session_id": self.session_id,
            "host_id": self.host_id,
            "predicted_disk_bytes": self.predicted_disk_bytes,
            "disk_budget_bytes": self.disk_budget_bytes,
            "ttl_seconds": self.ttl_seconds,
            "concept_ids": self.concept_ids,
            "active_job_ids": self.active_job_ids,
            "active_candidate_id": self.active_candidate_id,
        }

    def canonical_json(self) -> str:
        """Serialize a deterministic record payload for durable storage."""

        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    @classmethod
    def allocation_digest(cls, payload: Mapping[str, Any]) -> str:
        """Digest an immutable allocation input, excluding observations."""

        encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    @classmethod
    def derive_id(cls, repository_id: str, request_key: str) -> str:
        return lane_id_for(repository_id, request_key)

    @classmethod
    def new_expiry(cls, heartbeat_at: datetime, ttl_seconds: int) -> datetime:
        return heartbeat_at.astimezone(UTC) + timedelta(seconds=ttl_seconds)


__all__ = [
    "ACTIVE_STATES",
    "LaneLifecycleState",
    "LaneRecord",
    "TERMINAL_STATES",
    "lane_id_for",
    "new_fence",
    "repository_id_for",
]
