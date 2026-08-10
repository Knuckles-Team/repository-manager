"""RMDD-13 checkpoint 2: a durable, controller-only landing reservation.

The landing policy in :mod:`repository_manager.landing_policy` is deliberately
pure.  This module is the small controller seam that may precede a future
target compare-and-swap (CP3): it obtains one durable reservation for an exact
repository/target pair, then re-reads the mutable Git/canonical observations
while the existing reconciliation and canonical leases are held.

There are three important boundaries here:

* ``LandingReservationAuthority`` is the durable authority.  A local lock,
  SQLite row, JSON file, or an authority-shaped public DTO is never enough to
  authorize a reservation.  The authority authenticates the controller and
  atomically applies request-id/repository/target uniqueness and fencing.
* ``LandingStateReader`` is read-only and injected.  Every returned value is
  checked as a closed, bounded value before it is used.  The controller does
  not run Git commands, move refs, submit jobs, build, clean, or push.
* ``ReconciliationLeasePort`` composes the already-existing
  ``reconciliation-merge`` and canonical checkout leases.  It is an
  arbitration seam, not a second queue or process-local reservation store.

The post-acquire snapshot contains only opaque identities, Git object IDs, and
bounded state.  It intentionally excludes canonical paths, worktree paths,
hostnames, process IDs, exception text, and private WIP details.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import re
import unicodedata
from collections.abc import Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from agent_utilities.governance.lanes import LeaseUnavailable, hold_lease

from repository_manager.canonical_guard import BlockedByLease, hold_canonical_lease
from repository_manager.development import RepositoryIdentity
from repository_manager.merge_queue import MERGE_LEASE

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_TEXT_RE = re.compile(r"^[^\x00-\x1f\x7f\x80-\x9f]+$")
_REF_FORBIDDEN_RE = re.compile(r"[\x00-\x20~^:?*\\\[]")
_REF_INJECTION_RE = re.compile(r"[;|&$`'\"()<>\[\]{}]")
_MAX_TEXT_BYTES = 256
_MAX_ID_BYTES = 192
_MAX_DETAIL_BYTES = 1024
_MAX_WORKTREE_COUNT = 1024
_MAX_LEASE_EPOCH = (1 << 63) - 1
_TARGET_PREFIX = "refs/heads/"
_UNSAFE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Zl", "Zp"})
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)


class LandingReservationRefusalCode(StrEnum):
    """Stable wire-level refusal codes for the reservation boundary."""

    REQUEST_INVALID = "request_invalid"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    AUTHORITY_INVALID = "authority_invalid"
    LEASE_UNAVAILABLE = "lease_unavailable"
    RESERVATION_CONFLICT = "reservation_conflict"
    RESERVATION_INVALID = "reservation_invalid"
    RESERVATION_LOST = "reservation_lost"
    OWNER_MISMATCH = "owner_mismatch"
    FENCE_MISMATCH = "fence_mismatch"
    EPOCH_MISMATCH = "epoch_mismatch"
    REPOSITORY_MISMATCH = "repository_mismatch"
    TARGET_MISMATCH = "target_mismatch"
    TARGET_MOVED = "target_moved"
    TARGET_TREE_MISMATCH = "target_tree_mismatch"
    CERTIFICATION_INVALID = "certification_invalid"
    CERTIFICATION_CHANGED = "certification_changed"
    CANONICAL_LEASE_REQUIRED = "canonical_lease_required"
    CANONICAL_STATE_INVALID = "canonical_state_invalid"
    CANONICAL_STATE_CHANGED = "canonical_state_changed"
    CANONICAL_DIRTY = "canonical_dirty"
    PRIVATE_WIP = "private_wip"
    TARGET_OCCUPIED = "target_occupied"
    TARGET_OCCUPANCY_UNKNOWN = "target_occupancy_unknown"
    SOURCE_UNAVAILABLE = "source_unavailable"
    SOURCE_INVALID = "source_invalid"


class LandingReservationError(ValueError):
    """Malformed public or provider data at the reservation boundary."""


class LandingReservationConflict(RuntimeError):
    """The durable authority reports an active immutable-key conflict."""


class LandingReservationUnavailable(RuntimeError):
    """The durable authority/lease is not available for this attempt."""


class LandingReservationStale(RuntimeError):
    """The durable authority reports a stale reservation or fence."""


class LandingReservationOwnerMismatch(RuntimeError):
    """The returned durable reservation belongs to another owner."""


class LandingReservationFenceMismatch(RuntimeError):
    """The requested replay fence is not the current durable fence."""


class TrustedReservationRuntimeError(RuntimeError):
    """A trusted programmer/system failure that must cross the controller."""


def _text(value: object, field_name: str, *, maximum: int = _MAX_TEXT_BYTES) -> str:
    """Require an exact builtin, bounded, printable string.

    Exact builtin checks are intentional.  Calling ``strip``/``encode`` on a
    hostile ``str`` subclass is an attacker-controlled method call, not input
    validation.
    """

    if type(value) is not str or not value:
        raise LandingReservationError(f"{field_name} is invalid")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise LandingReservationError(f"{field_name} is invalid") from exc
    if len(encoded) > maximum:
        raise LandingReservationError(f"{field_name} is invalid")
    if (
        value.strip() != value
        or _SAFE_TEXT_RE.fullmatch(value) is None
        or any(
            unicodedata.category(char) in _UNSAFE_UNICODE_CATEGORIES
            or char in _BIDI_CONTROLS
            for char in value
        )
    ):
        raise LandingReservationError(f"{field_name} is invalid")
    return value


def _optional_text(
    value: object, field_name: str, *, maximum: int = _MAX_TEXT_BYTES
) -> str | None:
    if value is None:
        return None
    return _text(value, field_name, maximum=maximum)


def _sha(value: object, field_name: str) -> str:
    if type(value) is not str or _SHA_RE.fullmatch(value) is None:
        raise LandingReservationError(f"{field_name} is invalid")
    return value


def _digest(value: object, field_name: str) -> str:
    if type(value) is not str or _DIGEST_RE.fullmatch(value) is None:
        raise LandingReservationError(f"{field_name} is invalid")
    return value


def _positive_int(value: object, field_name: str) -> int:
    if type(value) is not int or value < 1 or value > _MAX_LEASE_EPOCH:
        raise LandingReservationError(f"{field_name} is invalid")
    return value


def _bounded_count(value: object, field_name: str) -> int:
    if type(value) is not int or not 0 <= value <= _MAX_WORKTREE_COUNT:
        raise LandingReservationError(f"{field_name} is invalid")
    return value


def normalize_target_ref(value: object) -> str:
    """Return one canonical local target ref and reject ref injection.

    Public callers may use ``main`` or ``refs/heads/main``.  Internally the
    full ``refs/heads/`` form is used everywhere, so a same-basename or remote
    ref cannot collide with a local target.
    """

    raw = _text(value, "target_ref", maximum=_MAX_TEXT_BYTES)
    branch = raw[len(_TARGET_PREFIX) :] if raw.startswith(_TARGET_PREFIX) else raw
    if (
        not branch
        or branch.startswith("/")
        or branch.endswith("/")
        or branch.endswith(".")
        or branch.endswith(".lock")
        or branch.startswith("-")
        or branch in {".", "..", "HEAD"}
        or ".." in branch
        or "//" in branch
        or "@{" in branch
        or _REF_FORBIDDEN_RE.search(branch)
        or _REF_INJECTION_RE.search(branch)
        or (raw.startswith("refs/") and not raw.startswith(_TARGET_PREFIX))
        or any(
            component in {".", ".."}
            or component.startswith(".")
            or component.endswith(".")
            or component.endswith(".lock")
            for component in branch.split("/")
        )
    ):
        raise LandingReservationError("target_ref is invalid")
    return f"{_TARGET_PREFIX}{branch}"


def _repository_id(repository: RepositoryIdentity) -> str:
    """Validate a repository model without retaining its private path."""

    if type(repository) is not RepositoryIdentity:
        raise LandingReservationError("repository is invalid")
    try:
        state = object.__getattribute__(repository, "__dict__")
        fields_set = object.__getattribute__(repository, "__pydantic_fields_set__")
    except (AttributeError, TypeError) as exc:
        raise LandingReservationError("repository is invalid") from exc
    if type(state) is not dict or type(fields_set) is not set:
        raise LandingReservationError("repository is invalid")
    expected = set(type(repository).model_fields)
    if set(state) != expected or fields_set != expected:
        raise LandingReservationError("repository is invalid")
    # Strict revalidation closes model_copy/model_construct scalar swaps.  The
    # result is used only as an identity check; its absolute path never enters
    # any public result or error.
    try:
        raw = RepositoryIdentity.model_dump(
            repository, mode="python", exclude_none=False, warnings=False
        )
        rebuilt = RepositoryIdentity.model_validate(raw, strict=True)
    except RuntimeError:
        raise
    except Exception as exc:
        raise LandingReservationError("repository is invalid") from exc
    if rebuilt != repository:
        raise LandingReservationError("repository is invalid")
    return _text(rebuilt.repository_id, "repository_id", maximum=_MAX_ID_BYTES)


def _safe_detail(code: LandingReservationRefusalCode) -> str:
    """Return a fixed message with no source values or private paths."""

    messages = {
        LandingReservationRefusalCode.REQUEST_INVALID: "reservation request is invalid",
        LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE: "reservation authority is unavailable",
        LandingReservationRefusalCode.AUTHORITY_INVALID: "reservation authority returned invalid data",
        LandingReservationRefusalCode.LEASE_UNAVAILABLE: "landing arbitration lease is unavailable",
        LandingReservationRefusalCode.RESERVATION_CONFLICT: "landing reservation conflicts with active durable state",
        LandingReservationRefusalCode.RESERVATION_INVALID: "durable landing reservation is invalid",
        LandingReservationRefusalCode.RESERVATION_LOST: "landing reservation is no longer current",
        LandingReservationRefusalCode.OWNER_MISMATCH: "landing reservation owner is not current",
        LandingReservationRefusalCode.FENCE_MISMATCH: "landing reservation fence is not current",
        LandingReservationRefusalCode.EPOCH_MISMATCH: "landing reservation epoch is not current",
        LandingReservationRefusalCode.REPOSITORY_MISMATCH: "repository identity changed",
        LandingReservationRefusalCode.TARGET_MISMATCH: "target identity changed",
        LandingReservationRefusalCode.TARGET_MOVED: "target commit changed",
        LandingReservationRefusalCode.TARGET_TREE_MISMATCH: "target tree changed",
        LandingReservationRefusalCode.CERTIFICATION_INVALID: "certification authority is invalid or not current",
        LandingReservationRefusalCode.CERTIFICATION_CHANGED: "certification changed across the reservation boundary",
        LandingReservationRefusalCode.CANONICAL_LEASE_REQUIRED: "canonical checkout lease is unavailable",
        LandingReservationRefusalCode.CANONICAL_STATE_INVALID: "canonical checkout state is invalid",
        LandingReservationRefusalCode.CANONICAL_STATE_CHANGED: "canonical checkout identity changed",
        LandingReservationRefusalCode.CANONICAL_DIRTY: "canonical checkout is not clean",
        LandingReservationRefusalCode.PRIVATE_WIP: "canonical checkout contains private work in progress",
        LandingReservationRefusalCode.TARGET_OCCUPIED: "target branch is occupied",
        LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN: "target occupancy is unknown",
        LandingReservationRefusalCode.SOURCE_UNAVAILABLE: "authoritative state source is unavailable",
        LandingReservationRefusalCode.SOURCE_INVALID: "authoritative state source returned invalid data",
    }
    return messages[code]


def _refuse(code: LandingReservationRefusalCode) -> LandingReservationResult:
    return LandingReservationResult(
        accepted=False,
        refusal_code=code,
        detail=_safe_detail(code),
    )


class CanonicalState(StrEnum):
    """Canonical checkout cleanliness classification from a read-only port."""

    CLEAN = "clean"
    DIRTY = "dirty"
    PRIVATE_WIP = "private_wip"
    UNKNOWN = "unknown"


class OccupancyState(StrEnum):
    """Target branch occupancy classification from a read-only port."""

    FREE = "free"
    OCCUPIED = "occupied"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ControllerIdentity:
    """Identity returned by a trusted authority, never accepted in a request."""

    controller_id: str
    owner_id: str
    tenant_id: str
    authority_epoch: int

    def __post_init__(self) -> None:
        _text(self.controller_id, "controller_id", maximum=_MAX_ID_BYTES)
        _text(self.owner_id, "owner_id", maximum=_MAX_ID_BYTES)
        _text(self.tenant_id, "tenant_id", maximum=_MAX_ID_BYTES)
        _positive_int(self.authority_epoch, "authority_epoch")


@dataclass(frozen=True, slots=True)
class LandingReservationRequest:
    """Immutable public request; owner identity is intentionally absent."""

    repository: RepositoryIdentity
    target_ref: str
    expected_target_sha: str
    expected_base_sha: str
    generation_id: str
    certificate_digest: str
    synthetic_commit_sha: str
    generation_tree_sha: str
    landing_fence: str
    request_id: str
    invocation_id: str
    expected_lease_epoch: int | None = None
    expected_lease_fence: str | None = None

    def __post_init__(self) -> None:
        _repository_id(self.repository)
        normalized = normalize_target_ref(self.target_ref)
        object.__setattr__(self, "target_ref", normalized)
        _sha(self.expected_target_sha, "expected_target_sha")
        _sha(self.expected_base_sha, "expected_base_sha")
        _text(self.generation_id, "generation_id", maximum=_MAX_ID_BYTES)
        _digest(self.certificate_digest, "certificate_digest")
        _sha(self.synthetic_commit_sha, "synthetic_commit_sha")
        _sha(self.generation_tree_sha, "generation_tree_sha")
        _text(self.landing_fence, "landing_fence", maximum=_MAX_TEXT_BYTES)
        _text(self.request_id, "request_id", maximum=_MAX_ID_BYTES)
        _text(self.invocation_id, "invocation_id", maximum=_MAX_ID_BYTES)
        if self.expected_lease_epoch is None:
            if self.expected_lease_fence is not None:
                raise LandingReservationError("lease replay anchor is incomplete")
        else:
            _positive_int(self.expected_lease_epoch, "expected_lease_epoch")
            _text(self.expected_lease_fence, "expected_lease_fence")

    @property
    def repository_id(self) -> str:
        return _repository_id(self.repository)

    def immutable_payload(self) -> dict[str, str]:
        """Return the idempotency input, excluding mutable retry anchors."""

        return {
            "repository_id": self.repository_id,
            "target_ref": self.target_ref,
            "expected_target_sha": self.expected_target_sha,
            "expected_base_sha": self.expected_base_sha,
            "generation_id": self.generation_id,
            "certificate_digest": self.certificate_digest,
            "synthetic_commit_sha": self.synthetic_commit_sha,
            "generation_tree_sha": self.generation_tree_sha,
            "landing_fence": self.landing_fence,
            "request_id": self.request_id,
            "invocation_id": self.invocation_id,
        }

    def digest(self) -> str:
        payload = json.dumps(
            self.immutable_payload(), sort_keys=True, separators=(",", ":")
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """Read-only target ref commit/tree observation."""

    repository_id: str
    target_ref: str
    commit_sha: str
    tree_sha: str

    def __post_init__(self) -> None:
        _text(self.repository_id, "target repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(self, "target_ref", normalize_target_ref(self.target_ref))
        _sha(self.commit_sha, "target commit_sha")
        _sha(self.tree_sha, "target tree_sha")


@dataclass(frozen=True, slots=True)
class CanonicalObservation:
    """Read-only canonical identity and cleanliness observation."""

    repository_id: str
    common_dir_id: str
    worktree_id: str
    state: CanonicalState
    private_wip: bool

    def __post_init__(self) -> None:
        _text(self.repository_id, "canonical repository_id", maximum=_MAX_ID_BYTES)
        _text(self.common_dir_id, "canonical common_dir_id", maximum=_MAX_ID_BYTES)
        _text(self.worktree_id, "canonical worktree_id", maximum=_MAX_ID_BYTES)
        if type(self.state) is not CanonicalState:
            raise LandingReservationError("canonical state is invalid")
        if type(self.private_wip) is not bool:
            raise LandingReservationError("canonical private_wip is invalid")


@dataclass(frozen=True, slots=True)
class OccupancyObservation:
    """Read-only target worktree occupancy observation."""

    repository_id: str
    target_ref: str
    other_worktree_count: int
    state: OccupancyState

    def __post_init__(self) -> None:
        _text(self.repository_id, "occupancy repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(self, "target_ref", normalize_target_ref(self.target_ref))
        _bounded_count(self.other_worktree_count, "other_worktree_count")
        if type(self.state) is not OccupancyState:
            raise LandingReservationError("occupancy state is invalid")


@dataclass(frozen=True, slots=True)
class CertificationObservation:
    """Current trusted generation/certificate identity and landing fence."""

    repository_id: str
    target_ref: str
    generation_id: str
    certificate_digest: str
    base_sha: str
    expected_landing_base_sha: str
    synthetic_commit_sha: str
    generation_tree_sha: str
    landing_fence: str
    certified: bool

    def __post_init__(self) -> None:
        _text(self.repository_id, "certification repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(self, "target_ref", normalize_target_ref(self.target_ref))
        _text(self.generation_id, "certification generation_id", maximum=_MAX_ID_BYTES)
        _digest(self.certificate_digest, "certification certificate_digest")
        _sha(self.base_sha, "certification base_sha")
        _sha(self.expected_landing_base_sha, "certification expected_landing_base_sha")
        _sha(self.synthetic_commit_sha, "certification synthetic_commit_sha")
        _sha(self.generation_tree_sha, "certification generation_tree_sha")
        _text(self.landing_fence, "certification landing_fence")
        if type(self.certified) is not bool:
            raise LandingReservationError("certification certified is invalid")


@dataclass(frozen=True, slots=True)
class DurableLandingReservation:
    """Durable authority result; local projections cannot manufacture it."""

    reservation_id: str
    request_id: str
    invocation_id: str
    repository_id: str
    target_ref: str
    request_digest: str
    controller_id: str
    owner_id: str
    lease_epoch: int
    fence: str
    active: bool = True

    def __post_init__(self) -> None:
        _text(self.reservation_id, "reservation_id", maximum=_MAX_ID_BYTES)
        _text(self.request_id, "reservation request_id", maximum=_MAX_ID_BYTES)
        _text(self.invocation_id, "reservation invocation_id", maximum=_MAX_ID_BYTES)
        _text(self.repository_id, "reservation repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(self, "target_ref", normalize_target_ref(self.target_ref))
        _digest(self.request_digest, "reservation request_digest")
        _text(self.controller_id, "reservation controller_id", maximum=_MAX_ID_BYTES)
        _text(self.owner_id, "reservation owner_id", maximum=_MAX_ID_BYTES)
        _positive_int(self.lease_epoch, "lease_epoch")
        _text(self.fence, "reservation fence")
        if type(self.active) is not bool:
            raise LandingReservationError("reservation active is invalid")


@dataclass(frozen=True, slots=True)
class LandingReservationSnapshot:
    """Bounded immutable CP3 input captured after reservation and re-read."""

    reservation_id: str
    request_digest: str
    repository_id: str
    target_ref: str
    expected_target_sha: str
    expected_base_sha: str
    observed_target_sha: str
    observed_target_tree_sha: str
    common_dir_id: str
    worktree_id: str
    lease_epoch: int
    lease_fence: str
    generation_id: str
    certificate_digest: str
    synthetic_commit_sha: str
    generation_tree_sha: str
    landing_fence: str
    target_worktree_count: int
    digest: str

    def immutable_payload(self) -> dict[str, object]:
        """Return the exact bounded fields covered by ``digest``."""

        return {
            "reservation_id": self.reservation_id,
            "request_digest": self.request_digest,
            "repository_id": self.repository_id,
            "target_ref": self.target_ref,
            "expected_target_sha": self.expected_target_sha,
            "expected_base_sha": self.expected_base_sha,
            "observed_target_sha": self.observed_target_sha,
            "observed_target_tree_sha": self.observed_target_tree_sha,
            "common_dir_id": self.common_dir_id,
            "worktree_id": self.worktree_id,
            "lease_epoch": self.lease_epoch,
            "lease_fence": self.lease_fence,
            "generation_id": self.generation_id,
            "certificate_digest": self.certificate_digest,
            "synthetic_commit_sha": self.synthetic_commit_sha,
            "generation_tree_sha": self.generation_tree_sha,
            "landing_fence": self.landing_fence,
            "target_worktree_count": self.target_worktree_count,
        }

    def __post_init__(self) -> None:
        _text(self.reservation_id, "snapshot reservation_id", maximum=_MAX_ID_BYTES)
        _digest(self.request_digest, "snapshot request_digest")
        _text(self.repository_id, "snapshot repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(self, "target_ref", normalize_target_ref(self.target_ref))
        for value, name in (
            (self.expected_target_sha, "snapshot expected_target_sha"),
            (self.expected_base_sha, "snapshot expected_base_sha"),
            (self.observed_target_sha, "snapshot observed_target_sha"),
            (self.observed_target_tree_sha, "snapshot observed_target_tree_sha"),
            (self.synthetic_commit_sha, "snapshot synthetic_commit_sha"),
            (self.generation_tree_sha, "snapshot generation_tree_sha"),
        ):
            _sha(value, name)
        _text(self.common_dir_id, "snapshot common_dir_id", maximum=_MAX_ID_BYTES)
        _text(self.worktree_id, "snapshot worktree_id", maximum=_MAX_ID_BYTES)
        _positive_int(self.lease_epoch, "snapshot lease_epoch")
        _text(self.lease_fence, "snapshot lease_fence")
        _text(self.generation_id, "snapshot generation_id", maximum=_MAX_ID_BYTES)
        _digest(self.certificate_digest, "snapshot certificate_digest")
        _text(self.landing_fence, "snapshot landing_fence")
        _bounded_count(self.target_worktree_count, "snapshot target_worktree_count")
        _digest(self.digest, "snapshot digest")
        expected_digest = _snapshot_digest(
            {"schema": "rmdd-13-landing-reservation:v1", **self.immutable_payload()}
        )
        if self.digest != expected_digest:
            raise LandingReservationError("snapshot digest is invalid")


@dataclass(frozen=True, slots=True)
class LandingReservationResult:
    """Accepted/refused controller result with no ambiguous partial success."""

    accepted: bool
    refusal_code: LandingReservationRefusalCode | None = None
    detail: str = ""
    snapshot: LandingReservationSnapshot | None = None

    def __post_init__(self) -> None:
        if type(self.accepted) is not bool:
            raise LandingReservationError("accepted is invalid")
        if self.accepted:
            if (
                self.refusal_code is not None
                or type(self.snapshot) is not LandingReservationSnapshot
            ):
                raise LandingReservationError("accepted reservation result is invalid")
            cast(LandingReservationSnapshot, self.snapshot).__post_init__()
        elif (
            type(self.refusal_code) is not LandingReservationRefusalCode
            or self.snapshot is not None
        ):
            raise LandingReservationError("refused reservation result is invalid")
        if type(self.detail) is not str:
            raise LandingReservationError("reservation detail is invalid")
        detail = self.detail.encode("utf-8")[:_MAX_DETAIL_BYTES].decode(
            "utf-8", errors="ignore"
        )
        object.__setattr__(self, "detail", detail)

    @property
    def refused(self) -> bool:
        return not self.accepted

    @property
    def code(self) -> LandingReservationRefusalCode | None:
        return self.refusal_code


@dataclass(frozen=True, slots=True)
class _LandingAuthorityInput:
    """Internal immutable authority input; no public source can forge owner."""

    request_id: str
    invocation_id: str
    repository_id: str
    target_ref: str
    request_digest: str
    expected_target_sha: str
    expected_base_sha: str
    generation_id: str
    certificate_digest: str
    synthetic_commit_sha: str
    generation_tree_sha: str
    landing_fence: str
    expected_lease_epoch: int | None
    expected_lease_fence: str | None


class LandingReservationAuthority(Protocol):
    """Durable, authenticated reservation authority.

    Implementations are expected to bind these calls to Graph-OS/WorkItem or
    an equivalent cross-host authority.  A local projection may implement
    status reads, but must not be substituted for these methods.
    """

    def authenticate_controller(self, invocation_id: str) -> ControllerIdentity:
        """Resolve the invocation to a trusted controller/owner identity."""

    def reserve_landing(
        self, request: _LandingAuthorityInput, controller: ControllerIdentity
    ) -> DurableLandingReservation:
        """Atomically reserve one repository+target and support replay."""

    def current_landing_reservation(
        self, reservation_id: str, controller: ControllerIdentity
    ) -> DurableLandingReservation | None:
        """Re-read the current durable reservation under its owner/fence."""


class LandingStateReader(Protocol):
    """Bounded read-only authority ports used after reservation acquisition."""

    def read_target(self, repository_id: str, target_ref: str) -> TargetObservation:
        """Read target commit/tree without changing refs or worktrees."""

    def read_canonical(self, repository_id: str) -> CanonicalObservation:
        """Read canonical common-dir identity and cleanliness/private-WIP state."""

    def read_occupancy(
        self, repository_id: str, target_ref: str
    ) -> OccupancyObservation:
        """Read target branch worktree occupancy."""

    def read_certification(
        self,
        repository_id: str,
        target_ref: str,
        generation_id: str,
        certificate_digest: str,
    ) -> CertificationObservation:
        """Read current generation/certificate/fence identity."""


class ReconciliationLeasePort(Protocol):
    """Existing arbitration lease composition; no second queue is introduced."""

    def hold(
        self, canonical_path: str, *, operation: str
    ) -> AbstractContextManager[None]:
        """Hold the existing repo merge/canonical leases."""


class ExistingReconciliationLease:
    """Default adapter composing RMDD-26 and the RMDD-12 merge lease."""

    def hold(
        self, canonical_path: str, *, operation: str
    ) -> AbstractContextManager[None]:
        @contextlib.contextmanager
        def _held() -> Iterator[None]:
            try:
                with hold_lease(
                    MERGE_LEASE,
                    operation=operation,
                    path=canonical_path,
                ):
                    with hold_canonical_lease(canonical_path, note=operation):
                        yield
            except (BlockedByLease, LeaseUnavailable) as exc:
                raise LandingReservationUnavailable(
                    "canonical or reconciliation lease unavailable"
                ) from exc

        return _held()


def _observation_error(
    call: Any,
    *args: object,
    **kwargs: object,
) -> tuple[object | None, LandingReservationRefusalCode | None]:
    """Call an injected source and normalize provider failures privately."""

    if not callable(call):
        return None, LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    try:
        return call(*args, **kwargs), None
    except TrustedReservationRuntimeError:
        raise
    except Exception:
        # State readers are provider boundaries.  They never return exception
        # text; a generic provider RuntimeError is data-source failure here.
        return None, LandingReservationRefusalCode.SOURCE_UNAVAILABLE


def _safe_method(source: object, name: str) -> object | None:
    """Resolve an injected method without leaking hostile attribute errors."""

    try:
        return getattr(source, name, None)
    except TrustedReservationRuntimeError:
        raise
    except Exception:
        return None


def _validate_observation(value: object, expected: type[Any]) -> object:
    if type(value) is not expected:
        raise LandingReservationError("state source returned an invalid observation")
    # Re-run the dataclass boundary after an adversarial object.__setattr__.
    try:
        cast(Any, value).__post_init__()
    except LandingReservationError:
        raise
    except (AttributeError, KeyError, IndexError, TypeError, UnicodeError) as exc:
        raise LandingReservationError(
            "state source returned an invalid observation"
        ) from exc
    except TrustedReservationRuntimeError:
        raise
    except Exception as exc:
        raise LandingReservationError(
            "state source returned an invalid observation"
        ) from exc
    return value


def _validate_lease(
    value: object,
    request: LandingReservationRequest,
    controller: ControllerIdentity,
) -> DurableLandingReservation:
    if type(value) is not DurableLandingReservation:
        raise LandingReservationError("authority returned an invalid reservation")
    lease = cast(DurableLandingReservation, value)
    lease.__post_init__()
    if (
        lease.request_id != request.request_id
        or lease.invocation_id != request.invocation_id
        or lease.repository_id != request.repository_id
        or lease.target_ref != request.target_ref
        or lease.request_digest != request.digest()
    ):
        raise LandingReservationError("authority reservation identity does not match")
    if (
        lease.controller_id != controller.controller_id
        or lease.owner_id != controller.owner_id
    ):
        raise LandingReservationOwnerMismatch(
            "authority reservation owner does not match"
        )
    if not lease.active:
        raise LandingReservationError("authority reservation is not active")
    if request.expected_lease_epoch is not None:
        if lease.lease_epoch != request.expected_lease_epoch:
            raise LandingReservationStale("reservation epoch is stale")
        if lease.fence != request.expected_lease_fence:
            raise LandingReservationFenceMismatch("reservation fence is stale")
    return lease


def _authority_call(
    call: Any,
    *args: object,
    unavailable: LandingReservationRefusalCode = LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE,
) -> tuple[object | None, LandingReservationRefusalCode | None]:
    if not callable(call):
        return None, unavailable
    try:
        return call(*args), None
    except (
        LandingReservationConflict,
        LandingReservationStale,
        LandingReservationUnavailable,
    ):
        raise
    except OSError:
        return None, unavailable
    except (ValueError, TypeError, KeyError, IndexError):
        return None, LandingReservationRefusalCode.AUTHORITY_INVALID


def _snapshot_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LandingReservationController:
    """Controller for CP2 reservation and post-acquire authoritative re-read."""

    def __init__(
        self,
        authority: LandingReservationAuthority,
        state_reader: LandingStateReader,
        *,
        lease: ReconciliationLeasePort | None = None,
    ) -> None:
        self._authority = authority
        self._reader = state_reader
        self._lease = lease or ExistingReconciliationLease()

    def _read_state(
        self, request: LandingReservationRequest
    ) -> (
        tuple[
            TargetObservation,
            CanonicalObservation,
            OccupancyObservation,
            CertificationObservation,
        ]
        | LandingReservationRefusalCode
    ):
        target, code = _observation_error(
            _safe_method(self._reader, "read_target"),
            request.repository_id,
            request.target_ref,
        )
        if code is not None:
            return code
        canonical, code = _observation_error(
            _safe_method(self._reader, "read_canonical"), request.repository_id
        )
        if code is not None:
            return code
        occupancy, code = _observation_error(
            _safe_method(self._reader, "read_occupancy"),
            request.repository_id,
            request.target_ref,
        )
        if code is not None:
            return code
        certification, code = _observation_error(
            _safe_method(self._reader, "read_certification"),
            request.repository_id,
            request.target_ref,
            request.generation_id,
            request.certificate_digest,
        )
        if code is not None:
            return code
        try:
            return (
                cast(
                    TargetObservation, _validate_observation(target, TargetObservation)
                ),
                cast(
                    CanonicalObservation,
                    _validate_observation(canonical, CanonicalObservation),
                ),
                cast(
                    OccupancyObservation,
                    _validate_observation(occupancy, OccupancyObservation),
                ),
                cast(
                    CertificationObservation,
                    _validate_observation(certification, CertificationObservation),
                ),
            )
        except LandingReservationError:
            return LandingReservationRefusalCode.SOURCE_INVALID

    @staticmethod
    def _check_state(
        request: LandingReservationRequest,
        state: tuple[
            TargetObservation,
            CanonicalObservation,
            OccupancyObservation,
            CertificationObservation,
        ],
    ) -> LandingReservationRefusalCode | None:
        target, canonical, occupancy, certification = state
        if target.repository_id != request.repository_id:
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        if target.target_ref != request.target_ref:
            return LandingReservationRefusalCode.TARGET_MISMATCH
        if target.commit_sha != request.expected_target_sha:
            return LandingReservationRefusalCode.TARGET_MOVED
        if canonical.repository_id != request.repository_id:
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        if canonical.state is CanonicalState.UNKNOWN:
            return LandingReservationRefusalCode.CANONICAL_STATE_INVALID
        if canonical.state is CanonicalState.PRIVATE_WIP or canonical.private_wip:
            return LandingReservationRefusalCode.PRIVATE_WIP
        if canonical.state is not CanonicalState.CLEAN:
            return LandingReservationRefusalCode.CANONICAL_DIRTY
        if (
            occupancy.repository_id != request.repository_id
            or occupancy.target_ref != request.target_ref
        ):
            return LandingReservationRefusalCode.TARGET_MISMATCH
        if occupancy.state is OccupancyState.UNKNOWN:
            return LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN
        if occupancy.state is not OccupancyState.FREE or occupancy.other_worktree_count:
            return LandingReservationRefusalCode.TARGET_OCCUPIED
        if certification.repository_id != request.repository_id:
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        if certification.target_ref != request.target_ref:
            return LandingReservationRefusalCode.TARGET_MISMATCH
        if (
            certification.generation_id != request.generation_id
            or certification.certificate_digest != request.certificate_digest
            or certification.base_sha != request.expected_base_sha
            or certification.expected_landing_base_sha != request.expected_base_sha
            or certification.synthetic_commit_sha != request.synthetic_commit_sha
            or certification.generation_tree_sha != request.generation_tree_sha
            or certification.landing_fence != request.landing_fence
            or not certification.certified
        ):
            return LandingReservationRefusalCode.CERTIFICATION_INVALID
        return None

    @staticmethod
    def _changed_state(
        before: tuple[
            TargetObservation,
            CanonicalObservation,
            OccupancyObservation,
            CertificationObservation,
        ],
        after: tuple[
            TargetObservation,
            CanonicalObservation,
            OccupancyObservation,
            CertificationObservation,
        ],
    ) -> LandingReservationRefusalCode | None:
        before_target, before_canonical, before_occupancy, before_cert = before
        after_target, after_canonical, after_occupancy, after_cert = after
        if before_target.commit_sha != after_target.commit_sha:
            return LandingReservationRefusalCode.TARGET_MOVED
        if before_target.tree_sha != after_target.tree_sha:
            return LandingReservationRefusalCode.TARGET_TREE_MISMATCH
        if before_canonical != after_canonical:
            return LandingReservationRefusalCode.CANONICAL_STATE_CHANGED
        if before_occupancy != after_occupancy:
            return (
                LandingReservationRefusalCode.TARGET_OCCUPIED
                if after_occupancy.state is not OccupancyState.FREE
                or after_occupancy.other_worktree_count
                else LandingReservationRefusalCode.CANONICAL_STATE_CHANGED
            )
        if before_cert != after_cert:
            return LandingReservationRefusalCode.CERTIFICATION_CHANGED
        return None

    def reserve(self, request: LandingReservationRequest) -> LandingReservationResult:
        """Reserve and re-read; never perform a Git or job mutation."""

        if type(request) is not LandingReservationRequest:
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)
        try:
            request.__post_init__()
            request_digest = request.digest()
        except LandingReservationError:
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)

        identity, code = _authority_call(
            _safe_method(self._authority, "authenticate_controller"),
            request.invocation_id,
        )
        if code is not None:
            return _refuse(code)
        try:
            if type(identity) is not ControllerIdentity:
                raise LandingReservationError("controller identity is invalid")
            controller = cast(ControllerIdentity, identity)
            controller.__post_init__()
        except LandingReservationError:
            return _refuse(LandingReservationRefusalCode.AUTHORITY_INVALID)

        before = self._read_state(request)
        if isinstance(before, LandingReservationRefusalCode):
            return _refuse(before)
        before_check = self._check_state(request, before)
        if before_check is not None:
            return _refuse(before_check)

        authority_input = _LandingAuthorityInput(
            request_id=request.request_id,
            invocation_id=request.invocation_id,
            repository_id=request.repository_id,
            target_ref=request.target_ref,
            request_digest=request_digest,
            expected_target_sha=request.expected_target_sha,
            expected_base_sha=request.expected_base_sha,
            generation_id=request.generation_id,
            certificate_digest=request.certificate_digest,
            synthetic_commit_sha=request.synthetic_commit_sha,
            generation_tree_sha=request.generation_tree_sha,
            landing_fence=request.landing_fence,
            expected_lease_epoch=request.expected_lease_epoch,
            expected_lease_fence=request.expected_lease_fence,
        )
        try:
            lease_context = self._lease.hold(
                request.repository.canonical_path,
                operation="reserve certified landing",
            )
        except (LandingReservationUnavailable, BlockedByLease):
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)
        except (ValueError, TypeError, OSError):
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)

        try:
            with lease_context:
                try:
                    durable, authority_code = _authority_call(
                        _safe_method(self._authority, "reserve_landing"),
                        authority_input,
                        controller,
                    )
                except LandingReservationConflict:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_CONFLICT)
                except LandingReservationStale:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                except LandingReservationUnavailable:
                    return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
                if authority_code is not None:
                    return _refuse(authority_code)
                try:
                    lease = _validate_lease(durable, request, controller)
                except LandingReservationOwnerMismatch:
                    return _refuse(LandingReservationRefusalCode.OWNER_MISMATCH)
                except LandingReservationFenceMismatch:
                    return _refuse(LandingReservationRefusalCode.FENCE_MISMATCH)
                except LandingReservationStale:
                    return _refuse(LandingReservationRefusalCode.EPOCH_MISMATCH)
                except LandingReservationError:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_INVALID)

                after = self._read_state(request)
                if isinstance(after, LandingReservationRefusalCode):
                    return _refuse(after)
                changed = self._changed_state(before, after)
                if changed is not None:
                    return _refuse(changed)
                after_check = self._check_state(request, after)
                if after_check is not None:
                    return _refuse(after_check)

                try:
                    current, current_code = _authority_call(
                        _safe_method(self._authority, "current_landing_reservation"),
                        lease.reservation_id,
                        controller,
                        unavailable=LandingReservationRefusalCode.RESERVATION_LOST,
                    )
                except LandingReservationStale:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                except LandingReservationUnavailable:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                if current_code is not None:
                    return _refuse(current_code)
                try:
                    current_lease = _validate_lease(current, request, controller)
                except (
                    LandingReservationOwnerMismatch,
                    LandingReservationFenceMismatch,
                    LandingReservationStale,
                ):
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                except LandingReservationError:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                if (
                    current_lease.reservation_id != lease.reservation_id
                    or current_lease.lease_epoch != lease.lease_epoch
                    or current_lease.fence != lease.fence
                ):
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)

                target, canonical, occupancy, certification = after
                snapshot_payload: dict[str, Any] = {
                    "reservation_id": lease.reservation_id,
                    "request_digest": request_digest,
                    "repository_id": request.repository_id,
                    "target_ref": request.target_ref,
                    "expected_target_sha": request.expected_target_sha,
                    "expected_base_sha": request.expected_base_sha,
                    "observed_target_sha": target.commit_sha,
                    "observed_target_tree_sha": target.tree_sha,
                    "common_dir_id": canonical.common_dir_id,
                    "worktree_id": canonical.worktree_id,
                    "lease_epoch": lease.lease_epoch,
                    "lease_fence": lease.fence,
                    "generation_id": certification.generation_id,
                    "certificate_digest": certification.certificate_digest,
                    "synthetic_commit_sha": certification.synthetic_commit_sha,
                    "generation_tree_sha": certification.generation_tree_sha,
                    "landing_fence": certification.landing_fence,
                    "target_worktree_count": occupancy.other_worktree_count,
                }
                digest_payload = {
                    "schema": "rmdd-13-landing-reservation:v1",
                    **snapshot_payload,
                }
                snapshot = LandingReservationSnapshot(
                    **snapshot_payload,
                    digest=_snapshot_digest(digest_payload),
                )
                return LandingReservationResult(
                    accepted=True,
                    detail="landing reservation acquired and state re-read",
                    snapshot=snapshot,
                )
        except (LandingReservationUnavailable, BlockedByLease):
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)
        except (ValueError, TypeError, OSError):
            return _refuse(LandingReservationRefusalCode.SOURCE_UNAVAILABLE)


def reserve_landing(
    request: LandingReservationRequest,
    *,
    authority: LandingReservationAuthority,
    state_reader: LandingStateReader,
    lease: ReconciliationLeasePort | None = None,
) -> LandingReservationResult:
    """Functional adapter for the controller's reservation operation."""

    return LandingReservationController(
        authority,
        state_reader,
        lease=lease,
    ).reserve(request)


__all__ = [
    "CanonicalObservation",
    "CanonicalState",
    "CertificationObservation",
    "ControllerIdentity",
    "DurableLandingReservation",
    "ExistingReconciliationLease",
    "LandingReservationAuthority",
    "LandingReservationConflict",
    "LandingReservationController",
    "LandingReservationError",
    "LandingReservationFenceMismatch",
    "LandingReservationOwnerMismatch",
    "LandingReservationRefusalCode",
    "LandingReservationRequest",
    "LandingReservationResult",
    "LandingReservationSnapshot",
    "LandingReservationStale",
    "LandingReservationUnavailable",
    "LandingStateReader",
    "OccupancyObservation",
    "OccupancyState",
    "ReconciliationLeasePort",
    "TargetObservation",
    "TrustedReservationRuntimeError",
    "normalize_target_ref",
    "reserve_landing",
]
