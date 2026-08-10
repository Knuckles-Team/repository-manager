"""RMDD-13 checkpoint 2: a durable, controller-only landing reservation.

The landing policy in :mod:`repository_manager.landing_policy` is deliberately
pure.  This module is the small controller seam that may precede a future
target compare-and-swap (CP3): a trusted authority obtains one durable
reservation for an exact repository/target pair, then provides a revisioned
Git/canonical snapshot and one final reservation/lease/source barrier while its
opaque leases are held.

There are three important boundaries here:

* ``LandingReservationAuthority`` is the durable authority.  A local lock,
  SQLite row, JSON file, or an authority-shaped public DTO is never enough to
  authorize a reservation.  The authority authenticates the controller and
  atomically applies request-id/repository/target uniqueness and fencing.
* The authority owns the read-only source and final barrier.  Every returned
  value is checked as a closed, bounded value before it is used.  The
  controller does not run Git commands, move refs, submit jobs, build, clean,
  or push.
* The authority privately composes the already-existing
  ``reconciliation-merge`` and canonical checkout leases.  Their handles are
  opaque and are never accepted from a public request or controller call; this
  is an arbitration seam, not a second queue or process-local store.

The post-acquire snapshot contains only opaque identities, Git object IDs, and
bounded state.  It intentionally excludes canonical paths, worktree paths,
hostnames, process IDs, exception text, and private WIP details.
"""

from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import re
import secrets
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from threading import Lock
from typing import Any, cast

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
_MAX_PATH_BYTES = 4096
_MAX_DETAIL_BYTES = 1024
_MAX_WORKTREE_COUNT = 1024
_MAX_LEASE_EPOCH = (1 << 63) - 1
_MAX_RECOVERY_RECORDS = 64
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

_AUTHORITY_SEAL = object()
_ISSUED_AUTHORITY_HANDLES: dict[int, object] = {}


class LandingReservationRefusalCode(StrEnum):
    """Stable wire-level refusal codes for the reservation boundary."""

    REQUEST_INVALID = "request_invalid"
    AUTHORITY_UNAVAILABLE = "authority_unavailable"
    AUTHORITY_INVALID = "authority_invalid"
    ATTESTATION_INVALID = "attestation_invalid"
    LEASE_UNAVAILABLE = "lease_unavailable"
    RESERVATION_CONFLICT = "reservation_conflict"
    RESERVATION_INVALID = "reservation_invalid"
    RESERVATION_LOST = "reservation_lost"
    OWNER_MISMATCH = "owner_mismatch"
    TENANT_MISMATCH = "tenant_mismatch"
    PRINCIPAL_MISMATCH = "principal_mismatch"
    SESSION_MISMATCH = "session_mismatch"
    AUTHORITY_EPOCH_MISMATCH = "authority_epoch_mismatch"
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
    RECOVERY_REQUIRED = "recovery_required"


class LandingReservationError(ValueError):
    """Malformed public or provider data at the reservation boundary."""


class LandingReservationConflict(RuntimeError):
    """The durable authority reports an active immutable-key conflict."""


class LandingReservationUnavailable(RuntimeError):
    """The durable authority/lease is not available for this attempt."""


class LandingReservationRecoveryRequired(RuntimeError):
    """A lease/reservation cleanup failure requires durable reconciliation."""


class LandingReservationSourceUnavailable(RuntimeError):
    """An authority-owned state source could not provide a bounded snapshot."""


class LandingReservationStale(RuntimeError):
    """The durable authority reports a stale reservation or fence."""


class LandingReservationOwnerMismatch(RuntimeError):
    """The returned durable reservation belongs to another owner."""


class LandingReservationTenantMismatch(RuntimeError):
    """The returned durable reservation belongs to another tenant."""


class LandingReservationPrincipalMismatch(RuntimeError):
    """The returned durable reservation has another authenticated principal."""


class LandingReservationSessionMismatch(RuntimeError):
    """The returned durable reservation has another authenticated session."""


class LandingReservationAuthorityEpochMismatch(RuntimeError):
    """The returned durable reservation belongs to an older authority epoch."""


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


def _strict_dataclass_values(
    value: object,
    expected_type: type[Any],
    field_names: tuple[str, ...],
) -> dict[str, object]:
    """Read a complete exact dataclass without trusting its public shape.

    ``object.__new__`` and ``object.__delattr__`` can manufacture a frozen
    dataclass with missing fields.  Reading one field before this complete
    provenance check would leak ``AttributeError`` (and, in a controller
    boundary, turn malformed provider data into a crash).  Bypass custom
    ``__getattribute__`` and require every declared slot to be present before
    any field is consumed.
    """

    if type(value) is not expected_type:
        raise LandingReservationError("typed authority value is invalid")
    result: dict[str, object] = {}
    for field_name in field_names:
        try:
            result[field_name] = object.__getattribute__(value, field_name)
        except TrustedReservationRuntimeError:
            raise
        except Exception as exc:
            raise LandingReservationError(
                "typed authority value is incomplete"
            ) from exc
    return result


def _path(value: object, field_name: str) -> str:
    """Validate an internal trusted canonical path without exposing it."""

    result = _text(value, field_name, maximum=_MAX_PATH_BYTES)
    if not result.startswith("/") or "\x00" in result or "/../" in f"/{result}/":
        raise LandingReservationError(f"{field_name} is invalid")
    return result


def _revision(value: object, field_name: str) -> str:
    """Require a bounded opaque source/authority revision token."""

    return _text(value, field_name, maximum=_MAX_ID_BYTES)


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
        LandingReservationRefusalCode.ATTESTATION_INVALID: "reservation attestation is invalid",
        LandingReservationRefusalCode.LEASE_UNAVAILABLE: "landing arbitration lease is unavailable",
        LandingReservationRefusalCode.RESERVATION_CONFLICT: "landing reservation conflicts with active durable state",
        LandingReservationRefusalCode.RESERVATION_INVALID: "durable landing reservation is invalid",
        LandingReservationRefusalCode.RESERVATION_LOST: "landing reservation is no longer current",
        LandingReservationRefusalCode.OWNER_MISMATCH: "landing reservation owner is not current",
        LandingReservationRefusalCode.TENANT_MISMATCH: "landing reservation tenant is not current",
        LandingReservationRefusalCode.PRINCIPAL_MISMATCH: "landing reservation principal is not current",
        LandingReservationRefusalCode.SESSION_MISMATCH: "landing reservation session is not current",
        LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH: "landing authority epoch is not current",
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
        LandingReservationRefusalCode.RECOVERY_REQUIRED: "reservation cleanup requires recovery",
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
    principal_id: str | None = None
    session_id: str | None = None

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            ControllerIdentity,
            (
                "controller_id",
                "owner_id",
                "tenant_id",
                "authority_epoch",
                "principal_id",
                "session_id",
            ),
        )
        _text(values["controller_id"], "controller_id", maximum=_MAX_ID_BYTES)
        _text(values["owner_id"], "owner_id", maximum=_MAX_ID_BYTES)
        _text(values["tenant_id"], "tenant_id", maximum=_MAX_ID_BYTES)
        _positive_int(values["authority_epoch"], "authority_epoch")
        _optional_text(values["principal_id"], "principal_id", maximum=_MAX_ID_BYTES)
        _optional_text(values["session_id"], "session_id", maximum=_MAX_ID_BYTES)

    def immutable_payload(self) -> dict[str, object]:
        values = _strict_dataclass_values(
            self,
            ControllerIdentity,
            (
                "controller_id",
                "owner_id",
                "tenant_id",
                "authority_epoch",
                "principal_id",
                "session_id",
            ),
        )
        self.__post_init__()
        return values


@dataclass(frozen=True, slots=True)
class ResolvedRepositoryIdentity:
    """Authority-resolved checkout identity used for lease selection.

    ``LandingReservationRequest.repository.canonical_path`` is only a caller
    hint.  This record is returned by the trusted repository authority and is
    the sole source allowed to select the canonical/reconciliation leases.
    The path participates in the private digest but is never copied to a
    public result.
    """

    repository_id: str
    canonical_path: str
    common_dir_id: str
    worktree_id: str
    authority_revision: str

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            ResolvedRepositoryIdentity,
            (
                "repository_id",
                "canonical_path",
                "common_dir_id",
                "worktree_id",
                "authority_revision",
            ),
        )
        _text(values["repository_id"], "resolved repository_id", maximum=_MAX_ID_BYTES)
        _path(values["canonical_path"], "resolved canonical_path")
        _text(values["common_dir_id"], "resolved common_dir_id", maximum=_MAX_ID_BYTES)
        _text(values["worktree_id"], "resolved worktree_id", maximum=_MAX_ID_BYTES)
        _revision(values["authority_revision"], "authority_revision")

    def immutable_payload(self) -> dict[str, str]:
        values = _strict_dataclass_values(
            self,
            ResolvedRepositoryIdentity,
            (
                "repository_id",
                "canonical_path",
                "common_dir_id",
                "worktree_id",
                "authority_revision",
            ),
        )
        self.__post_init__()
        return cast(dict[str, str], values)

    def digest(self) -> str:
        payload = {
            "schema": "rmdd-13-resolved-repository:v1",
            **self.immutable_payload(),
        }
        return _snapshot_digest(payload)


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
        values = _strict_dataclass_values(
            self,
            LandingReservationRequest,
            (
                "repository",
                "target_ref",
                "expected_target_sha",
                "expected_base_sha",
                "generation_id",
                "certificate_digest",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "request_id",
                "invocation_id",
                "expected_lease_epoch",
                "expected_lease_fence",
            ),
        )
        _repository_id(cast(RepositoryIdentity, values["repository"]))
        normalized = normalize_target_ref(values["target_ref"])
        object.__setattr__(self, "target_ref", normalized)
        _sha(values["expected_target_sha"], "expected_target_sha")
        _sha(values["expected_base_sha"], "expected_base_sha")
        _text(values["generation_id"], "generation_id", maximum=_MAX_ID_BYTES)
        _digest(values["certificate_digest"], "certificate_digest")
        _sha(values["synthetic_commit_sha"], "synthetic_commit_sha")
        _sha(values["generation_tree_sha"], "generation_tree_sha")
        _text(values["landing_fence"], "landing_fence", maximum=_MAX_TEXT_BYTES)
        _text(values["request_id"], "request_id", maximum=_MAX_ID_BYTES)
        _text(values["invocation_id"], "invocation_id", maximum=_MAX_ID_BYTES)
        if values["expected_lease_epoch"] is None:
            if values["expected_lease_fence"] is not None:
                raise LandingReservationError("lease replay anchor is incomplete")
        else:
            _positive_int(values["expected_lease_epoch"], "expected_lease_epoch")
            _text(values["expected_lease_fence"], "expected_lease_fence")

    @property
    def repository_id(self) -> str:
        values = _strict_dataclass_values(
            self, LandingReservationRequest, ("repository",)
        )
        return _repository_id(cast(RepositoryIdentity, values["repository"]))

    def immutable_payload(self) -> dict[str, str]:
        """Return the idempotency input, excluding mutable retry anchors."""

        values = _strict_dataclass_values(
            self,
            LandingReservationRequest,
            (
                "repository",
                "target_ref",
                "expected_target_sha",
                "expected_base_sha",
                "generation_id",
                "certificate_digest",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "request_id",
                "invocation_id",
            ),
        )
        self.__post_init__()
        return {
            "repository_id": _repository_id(
                cast(RepositoryIdentity, values["repository"])
            ),
            "target_ref": normalize_target_ref(values["target_ref"]),
            "expected_target_sha": cast(str, values["expected_target_sha"]),
            "expected_base_sha": cast(str, values["expected_base_sha"]),
            "generation_id": cast(str, values["generation_id"]),
            "certificate_digest": cast(str, values["certificate_digest"]),
            "synthetic_commit_sha": cast(str, values["synthetic_commit_sha"]),
            "generation_tree_sha": cast(str, values["generation_tree_sha"]),
            "landing_fence": cast(str, values["landing_fence"]),
            "request_id": cast(str, values["request_id"]),
            "invocation_id": cast(str, values["invocation_id"]),
        }

    def digest(
        self,
        resolved: ResolvedRepositoryIdentity | None = None,
        controller: ControllerIdentity | None = None,
    ) -> str:
        """Digest request plus trusted resolved/authenticated identity.

        The no-argument form remains a bounded request-only helper for
        diagnostics.  Reservation authority input always supplies both the
        resolved repository and authenticated controller so path/common-dir,
        tenant, epoch, principal, and session cannot be omitted.
        """

        payload = json.dumps(
            {
                "schema": "rmdd-13-landing-request:v2",
                **self.immutable_payload(),
                **(
                    {"resolved_repository": resolved.immutable_payload()}
                    if resolved is not None
                    else {}
                ),
                **(
                    {"controller": controller.immutable_payload()}
                    if controller is not None
                    else {}
                ),
            },
            sort_keys=True,
            separators=(",", ":"),
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
        values = _strict_dataclass_values(
            self,
            TargetObservation,
            ("repository_id", "target_ref", "commit_sha", "tree_sha"),
        )
        _text(values["repository_id"], "target repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(
            self, "target_ref", normalize_target_ref(values["target_ref"])
        )
        _sha(values["commit_sha"], "target commit_sha")
        _sha(values["tree_sha"], "target tree_sha")


@dataclass(frozen=True, slots=True)
class CanonicalObservation:
    """Read-only canonical identity and cleanliness observation."""

    repository_id: str
    common_dir_id: str
    worktree_id: str
    state: CanonicalState
    private_wip: bool
    index_clean: bool = True

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            CanonicalObservation,
            (
                "repository_id",
                "common_dir_id",
                "worktree_id",
                "state",
                "private_wip",
                "index_clean",
            ),
        )
        _text(values["repository_id"], "canonical repository_id", maximum=_MAX_ID_BYTES)
        _text(values["common_dir_id"], "canonical common_dir_id", maximum=_MAX_ID_BYTES)
        _text(values["worktree_id"], "canonical worktree_id", maximum=_MAX_ID_BYTES)
        if type(values["state"]) is not CanonicalState:
            raise LandingReservationError("canonical state is invalid")
        if type(values["private_wip"]) is not bool:
            raise LandingReservationError("canonical private_wip is invalid")
        if type(values["index_clean"]) is not bool:
            raise LandingReservationError("canonical index_clean is invalid")


@dataclass(frozen=True, slots=True)
class OccupancyObservation:
    """Read-only target worktree occupancy observation."""

    repository_id: str
    target_ref: str
    other_worktree_count: int
    state: OccupancyState

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            OccupancyObservation,
            ("repository_id", "target_ref", "other_worktree_count", "state"),
        )
        _text(values["repository_id"], "occupancy repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(
            self, "target_ref", normalize_target_ref(values["target_ref"])
        )
        _bounded_count(values["other_worktree_count"], "other_worktree_count")
        if type(values["state"]) is not OccupancyState:
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
        values = _strict_dataclass_values(
            self,
            CertificationObservation,
            (
                "repository_id",
                "target_ref",
                "generation_id",
                "certificate_digest",
                "base_sha",
                "expected_landing_base_sha",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "certified",
            ),
        )
        _text(
            values["repository_id"],
            "certification repository_id",
            maximum=_MAX_ID_BYTES,
        )
        object.__setattr__(
            self, "target_ref", normalize_target_ref(values["target_ref"])
        )
        _text(
            values["generation_id"],
            "certification generation_id",
            maximum=_MAX_ID_BYTES,
        )
        _digest(values["certificate_digest"], "certification certificate_digest")
        _sha(values["base_sha"], "certification base_sha")
        _sha(
            values["expected_landing_base_sha"],
            "certification expected_landing_base_sha",
        )
        _sha(values["synthetic_commit_sha"], "certification synthetic_commit_sha")
        _sha(values["generation_tree_sha"], "certification generation_tree_sha")
        _text(values["landing_fence"], "certification landing_fence")
        if type(values["certified"]) is not bool:
            raise LandingReservationError("certification certified is invalid")


@dataclass(frozen=True, slots=True)
class LandingStateSnapshot:
    """One authority-owned, revisioned read of all landing sources."""

    resolved_repository_digest: str
    target: TargetObservation
    canonical: CanonicalObservation
    occupancy: OccupancyObservation
    certification: CertificationObservation
    target_revision: str
    canonical_revision: str
    occupancy_revision: str
    certification_revision: str
    snapshot_revision: str

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            LandingStateSnapshot,
            (
                "resolved_repository_digest",
                "target",
                "canonical",
                "occupancy",
                "certification",
                "target_revision",
                "canonical_revision",
                "occupancy_revision",
                "certification_revision",
                "snapshot_revision",
            ),
        )
        _digest(
            values["resolved_repository_digest"], "state resolved repository digest"
        )
        _validate_observation(values["target"], TargetObservation)
        _validate_observation(values["canonical"], CanonicalObservation)
        _validate_observation(values["occupancy"], OccupancyObservation)
        _validate_observation(values["certification"], CertificationObservation)
        _revision(values["target_revision"], "target revision")
        _revision(values["canonical_revision"], "canonical revision")
        _revision(values["occupancy_revision"], "occupancy revision")
        _revision(values["certification_revision"], "certification revision")
        _revision(values["snapshot_revision"], "snapshot revision")

    def immutable_payload(self) -> dict[str, object]:
        values = _strict_dataclass_values(
            self,
            LandingStateSnapshot,
            (
                "resolved_repository_digest",
                "target",
                "canonical",
                "occupancy",
                "certification",
                "target_revision",
                "canonical_revision",
                "occupancy_revision",
                "certification_revision",
                "snapshot_revision",
            ),
        )
        self.__post_init__()
        target = cast(TargetObservation, values["target"])
        canonical = cast(CanonicalObservation, values["canonical"])
        occupancy = cast(OccupancyObservation, values["occupancy"])
        certification = cast(CertificationObservation, values["certification"])
        return {
            "resolved_repository_digest": values["resolved_repository_digest"],
            "target": {
                "repository_id": target.repository_id,
                "target_ref": target.target_ref,
                "commit_sha": target.commit_sha,
                "tree_sha": target.tree_sha,
            },
            "canonical": {
                "repository_id": canonical.repository_id,
                "common_dir_id": canonical.common_dir_id,
                "worktree_id": canonical.worktree_id,
                "state": canonical.state.value,
                "private_wip": canonical.private_wip,
                "index_clean": canonical.index_clean,
            },
            "occupancy": {
                "repository_id": occupancy.repository_id,
                "target_ref": occupancy.target_ref,
                "other_worktree_count": occupancy.other_worktree_count,
                "state": occupancy.state.value,
            },
            "certification": {
                "repository_id": certification.repository_id,
                "target_ref": certification.target_ref,
                "generation_id": certification.generation_id,
                "certificate_digest": certification.certificate_digest,
                "base_sha": certification.base_sha,
                "expected_landing_base_sha": certification.expected_landing_base_sha,
                "synthetic_commit_sha": certification.synthetic_commit_sha,
                "generation_tree_sha": certification.generation_tree_sha,
                "landing_fence": certification.landing_fence,
                "certified": certification.certified,
            },
            "target_revision": values["target_revision"],
            "canonical_revision": values["canonical_revision"],
            "occupancy_revision": values["occupancy_revision"],
            "certification_revision": values["certification_revision"],
            "snapshot_revision": values["snapshot_revision"],
        }


@dataclass(frozen=True, slots=True)
class LandingValidationBarrier:
    """Authority result from the final reservation/lease/source barrier."""

    reservation: DurableLandingReservation
    snapshot: LandingStateSnapshot
    barrier_revision: str

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            LandingValidationBarrier,
            ("reservation", "snapshot", "barrier_revision"),
        )
        if type(values["reservation"]) is not DurableLandingReservation:
            raise LandingReservationError("barrier reservation is invalid")
        cast(DurableLandingReservation, values["reservation"]).__post_init__()
        if type(values["snapshot"]) is not LandingStateSnapshot:
            raise LandingReservationError("barrier snapshot is invalid")
        cast(LandingStateSnapshot, values["snapshot"]).__post_init__()
        _revision(values["barrier_revision"], "barrier revision")


@dataclass(frozen=True, slots=True)
class DurableLandingReservation:
    """Durable authority result; local projections cannot manufacture it."""

    reservation_id: str
    request_id: str
    invocation_id: str
    repository_id: str
    target_ref: str
    request_digest: str
    resolved_repository_digest: str
    common_dir_id: str
    worktree_id: str
    authority_revision: str
    controller_id: str
    owner_id: str
    tenant_id: str
    lease_epoch: int
    fence: str
    authority_epoch: int
    principal_id: str | None = None
    session_id: str | None = None
    reconciliation_lease_id: str | None = None
    reconciliation_lease_epoch: int | None = None
    reconciliation_lease_fence: str | None = None
    canonical_lease_id: str | None = None
    canonical_lease_epoch: int | None = None
    canonical_lease_fence: str | None = None
    authority_incarnation: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            DurableLandingReservation,
            (
                "reservation_id",
                "request_id",
                "invocation_id",
                "repository_id",
                "target_ref",
                "request_digest",
                "resolved_repository_digest",
                "common_dir_id",
                "worktree_id",
                "authority_revision",
                "controller_id",
                "owner_id",
                "tenant_id",
                "lease_epoch",
                "fence",
                "authority_epoch",
                "principal_id",
                "session_id",
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
                "authority_incarnation",
                "active",
            ),
        )
        _text(values["reservation_id"], "reservation_id", maximum=_MAX_ID_BYTES)
        _text(values["request_id"], "reservation request_id", maximum=_MAX_ID_BYTES)
        _text(
            values["invocation_id"], "reservation invocation_id", maximum=_MAX_ID_BYTES
        )
        _text(
            values["repository_id"], "reservation repository_id", maximum=_MAX_ID_BYTES
        )
        object.__setattr__(
            self, "target_ref", normalize_target_ref(values["target_ref"])
        )
        _digest(values["request_digest"], "reservation request_digest")
        _digest(values["resolved_repository_digest"], "resolved repository digest")
        _text(
            values["common_dir_id"], "reservation common_dir_id", maximum=_MAX_ID_BYTES
        )
        _text(values["worktree_id"], "reservation worktree_id", maximum=_MAX_ID_BYTES)
        _revision(values["authority_revision"], "reservation authority_revision")
        _text(
            values["controller_id"], "reservation controller_id", maximum=_MAX_ID_BYTES
        )
        _text(values["owner_id"], "reservation owner_id", maximum=_MAX_ID_BYTES)
        _text(values["tenant_id"], "reservation tenant_id", maximum=_MAX_ID_BYTES)
        _positive_int(values["lease_epoch"], "lease_epoch")
        _text(values["fence"], "reservation fence")
        _positive_int(values["authority_epoch"], "reservation authority_epoch")
        _optional_text(
            values["principal_id"], "reservation principal_id", maximum=_MAX_ID_BYTES
        )
        _optional_text(
            values["session_id"], "reservation session_id", maximum=_MAX_ID_BYTES
        )
        _optional_text(
            values["reconciliation_lease_id"],
            "reservation reconciliation lease id",
            maximum=_MAX_ID_BYTES,
        )
        if values["reconciliation_lease_epoch"] is not None:
            _positive_int(
                values["reconciliation_lease_epoch"],
                "reservation reconciliation lease epoch",
            )
        _optional_text(
            values["reconciliation_lease_fence"],
            "reservation reconciliation lease fence",
        )
        _optional_text(
            values["canonical_lease_id"],
            "reservation canonical lease id",
            maximum=_MAX_ID_BYTES,
        )
        if values["canonical_lease_epoch"] is not None:
            _positive_int(
                values["canonical_lease_epoch"],
                "reservation canonical lease epoch",
            )
        _optional_text(
            values["canonical_lease_fence"],
            "reservation canonical lease fence",
        )
        _optional_text(
            values["authority_incarnation"],
            "reservation authority incarnation",
            maximum=_MAX_ID_BYTES,
        )
        lease_values = (
            values["reconciliation_lease_id"],
            values["reconciliation_lease_epoch"],
            values["reconciliation_lease_fence"],
            values["canonical_lease_id"],
            values["canonical_lease_epoch"],
            values["canonical_lease_fence"],
        )
        if any(value is None for value in lease_values) and any(
            value is not None for value in lease_values
        ):
            raise LandingReservationError("reservation lease evidence is incomplete")
        if type(values["active"]) is not bool:
            raise LandingReservationError("reservation active is invalid")


@dataclass(frozen=True, slots=True)
class LandingReservationSnapshot:
    """Bounded immutable CP3 input captured after reservation and re-read."""

    reservation_id: str
    request_digest: str
    resolved_repository_digest: str
    repository_id: str
    target_ref: str
    expected_target_sha: str
    expected_base_sha: str
    observed_target_sha: str
    observed_target_tree_sha: str
    common_dir_id: str
    worktree_id: str
    authority_revision: str
    lease_epoch: int
    lease_fence: str
    tenant_id: str
    authority_epoch: int
    generation_id: str
    certificate_digest: str
    synthetic_commit_sha: str
    generation_tree_sha: str
    landing_fence: str
    target_worktree_count: int
    target_revision: str
    canonical_revision: str
    occupancy_revision: str
    certification_revision: str
    snapshot_revision: str
    barrier_revision: str
    reconciliation_lease_id: str
    reconciliation_lease_epoch: int
    reconciliation_lease_fence: str
    canonical_lease_id: str
    canonical_lease_epoch: int
    canonical_lease_fence: str
    authority_incarnation: str
    authority_attestation: str
    digest: str

    def _immutable_payload_unchecked(self) -> dict[str, object]:
        """Build the digest payload after the caller has validated fields."""

        values = _strict_dataclass_values(
            self,
            LandingReservationSnapshot,
            (
                "reservation_id",
                "request_digest",
                "resolved_repository_digest",
                "repository_id",
                "target_ref",
                "expected_target_sha",
                "expected_base_sha",
                "observed_target_sha",
                "observed_target_tree_sha",
                "common_dir_id",
                "worktree_id",
                "authority_revision",
                "lease_epoch",
                "lease_fence",
                "tenant_id",
                "authority_epoch",
                "generation_id",
                "certificate_digest",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "target_worktree_count",
                "target_revision",
                "canonical_revision",
                "occupancy_revision",
                "certification_revision",
                "snapshot_revision",
                "barrier_revision",
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
                "authority_incarnation",
                "authority_attestation",
            ),
        )
        return {
            name: values[name]
            for name in (
                "reservation_id",
                "request_digest",
                "resolved_repository_digest",
                "repository_id",
                "target_ref",
                "expected_target_sha",
                "expected_base_sha",
                "observed_target_sha",
                "observed_target_tree_sha",
                "common_dir_id",
                "worktree_id",
                "authority_revision",
                "lease_epoch",
                "lease_fence",
                "tenant_id",
                "authority_epoch",
                "generation_id",
                "certificate_digest",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "target_worktree_count",
                "target_revision",
                "canonical_revision",
                "occupancy_revision",
                "certification_revision",
                "snapshot_revision",
                "barrier_revision",
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
                "authority_incarnation",
                "authority_attestation",
            )
        }

    def immutable_payload(self) -> dict[str, object]:
        """Return the exact bounded fields covered by ``digest``."""

        self.__post_init__()
        return self._immutable_payload_unchecked()

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            LandingReservationSnapshot,
            (
                "reservation_id",
                "request_digest",
                "resolved_repository_digest",
                "repository_id",
                "target_ref",
                "expected_target_sha",
                "expected_base_sha",
                "observed_target_sha",
                "observed_target_tree_sha",
                "common_dir_id",
                "worktree_id",
                "authority_revision",
                "lease_epoch",
                "lease_fence",
                "tenant_id",
                "authority_epoch",
                "generation_id",
                "certificate_digest",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "target_worktree_count",
                "target_revision",
                "canonical_revision",
                "occupancy_revision",
                "certification_revision",
                "snapshot_revision",
                "barrier_revision",
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
                "authority_incarnation",
                "authority_attestation",
                "digest",
            ),
        )
        _text(
            values["reservation_id"], "snapshot reservation_id", maximum=_MAX_ID_BYTES
        )
        _digest(values["request_digest"], "snapshot request_digest")
        _digest(
            values["resolved_repository_digest"], "snapshot resolved repository digest"
        )
        _text(values["repository_id"], "snapshot repository_id", maximum=_MAX_ID_BYTES)
        object.__setattr__(
            self, "target_ref", normalize_target_ref(values["target_ref"])
        )
        for value, name in (
            (values["expected_target_sha"], "snapshot expected_target_sha"),
            (values["expected_base_sha"], "snapshot expected_base_sha"),
            (values["observed_target_sha"], "snapshot observed_target_sha"),
            (values["observed_target_tree_sha"], "snapshot observed_target_tree_sha"),
            (values["synthetic_commit_sha"], "snapshot synthetic_commit_sha"),
            (values["generation_tree_sha"], "snapshot generation_tree_sha"),
        ):
            _sha(value, name)
        if values["observed_target_sha"] != values["expected_target_sha"]:
            raise LandingReservationError(
                "snapshot target commit does not match expected"
            )
        _text(values["common_dir_id"], "snapshot common_dir_id", maximum=_MAX_ID_BYTES)
        _text(values["worktree_id"], "snapshot worktree_id", maximum=_MAX_ID_BYTES)
        _revision(values["authority_revision"], "snapshot authority_revision")
        _positive_int(values["lease_epoch"], "snapshot lease_epoch")
        _text(values["lease_fence"], "snapshot lease_fence")
        _text(values["tenant_id"], "snapshot tenant_id", maximum=_MAX_ID_BYTES)
        _positive_int(values["authority_epoch"], "snapshot authority_epoch")
        _text(values["generation_id"], "snapshot generation_id", maximum=_MAX_ID_BYTES)
        _digest(values["certificate_digest"], "snapshot certificate_digest")
        _text(values["landing_fence"], "snapshot landing_fence")
        _bounded_count(
            values["target_worktree_count"], "snapshot target_worktree_count"
        )
        _revision(values["target_revision"], "snapshot target_revision")
        _revision(values["canonical_revision"], "snapshot canonical_revision")
        _revision(values["occupancy_revision"], "snapshot occupancy_revision")
        _revision(values["certification_revision"], "snapshot certification_revision")
        _revision(values["snapshot_revision"], "snapshot revision")
        _revision(values["barrier_revision"], "snapshot barrier revision")
        _text(
            values["reconciliation_lease_id"],
            "snapshot reconciliation lease id",
            maximum=_MAX_ID_BYTES,
        )
        _positive_int(
            values["reconciliation_lease_epoch"],
            "snapshot reconciliation lease epoch",
        )
        _text(
            values["reconciliation_lease_fence"], "snapshot reconciliation lease fence"
        )
        _text(
            values["canonical_lease_id"],
            "snapshot canonical lease id",
            maximum=_MAX_ID_BYTES,
        )
        _positive_int(values["canonical_lease_epoch"], "snapshot canonical lease epoch")
        _text(values["canonical_lease_fence"], "snapshot canonical lease fence")
        _text(
            values["authority_incarnation"],
            "snapshot authority incarnation",
            maximum=_MAX_ID_BYTES,
        )
        _digest(values["authority_attestation"], "snapshot authority attestation")
        _digest(values["digest"], "snapshot digest")
        expected_digest = _snapshot_digest(
            {
                "schema": "rmdd-13-landing-reservation:v2",
                **self._immutable_payload_unchecked(),
            }
        )
        if values["digest"] != expected_digest:
            raise LandingReservationError("snapshot digest is invalid")


@dataclass(frozen=True, slots=True)
class LandingReservationResult:
    """Accepted/refused controller result with no ambiguous partial success."""

    accepted: bool
    refusal_code: LandingReservationRefusalCode | None = None
    detail: str = ""
    snapshot: LandingReservationSnapshot | None = None

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            LandingReservationResult,
            ("accepted", "refusal_code", "detail", "snapshot"),
        )
        if type(values["accepted"]) is not bool:
            raise LandingReservationError("accepted is invalid")
        if values["accepted"]:
            if (
                values["refusal_code"] is not None
                or type(values["snapshot"]) is not LandingReservationSnapshot
            ):
                raise LandingReservationError("accepted reservation result is invalid")
            cast(LandingReservationSnapshot, values["snapshot"]).__post_init__()
        elif (
            type(values["refusal_code"]) is not LandingReservationRefusalCode
            or values["snapshot"] is not None
        ):
            raise LandingReservationError("refused reservation result is invalid")
        if type(values["detail"]) is not str:
            raise LandingReservationError("reservation detail is invalid")
        detail = (
            cast(str, values["detail"])
            .encode("utf-8")[:_MAX_DETAIL_BYTES]
            .decode("utf-8", errors="ignore")
        )
        object.__setattr__(self, "detail", detail)

    @property
    def refused(self) -> bool:
        values = _strict_dataclass_values(self, LandingReservationResult, ("accepted",))
        if type(values["accepted"]) is not bool:
            raise LandingReservationError("accepted is invalid")
        return not cast(bool, values["accepted"])

    @property
    def code(self) -> LandingReservationRefusalCode | None:
        values = _strict_dataclass_values(
            self, LandingReservationResult, ("refusal_code",)
        )
        code = values["refusal_code"]
        if code is not None and type(code) is not LandingReservationRefusalCode:
            raise LandingReservationError("reservation refusal code is invalid")
        return cast(LandingReservationRefusalCode | None, code)


@dataclass(frozen=True, slots=True)
class _LandingAuthorityInput:
    """Internal immutable authority input; no public source can forge owner."""

    request_id: str
    invocation_id: str
    repository_id: str
    target_ref: str
    request_digest: str
    resolved_repository: ResolvedRepositoryIdentity
    controller: ControllerIdentity
    expected_target_sha: str
    expected_base_sha: str
    generation_id: str
    certificate_digest: str
    synthetic_commit_sha: str
    generation_tree_sha: str
    landing_fence: str
    expected_lease_epoch: int | None
    expected_lease_fence: str | None

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            _LandingAuthorityInput,
            (
                "request_id",
                "invocation_id",
                "repository_id",
                "target_ref",
                "request_digest",
                "resolved_repository",
                "controller",
                "expected_target_sha",
                "expected_base_sha",
                "generation_id",
                "certificate_digest",
                "synthetic_commit_sha",
                "generation_tree_sha",
                "landing_fence",
                "expected_lease_epoch",
                "expected_lease_fence",
            ),
        )
        _text(values["request_id"], "authority request_id", maximum=_MAX_ID_BYTES)
        _text(values["invocation_id"], "authority invocation_id", maximum=_MAX_ID_BYTES)
        _text(values["repository_id"], "authority repository_id", maximum=_MAX_ID_BYTES)
        normalize_target_ref(values["target_ref"])
        _digest(values["request_digest"], "authority request digest")
        if type(values["resolved_repository"]) is not ResolvedRepositoryIdentity:
            raise LandingReservationError("authority resolved repository is invalid")
        cast(ResolvedRepositoryIdentity, values["resolved_repository"]).__post_init__()
        if type(values["controller"]) is not ControllerIdentity:
            raise LandingReservationError("authority controller is invalid")
        cast(ControllerIdentity, values["controller"]).__post_init__()
        _sha(values["expected_target_sha"], "authority expected target sha")
        _sha(values["expected_base_sha"], "authority expected base sha")
        _text(values["generation_id"], "authority generation id", maximum=_MAX_ID_BYTES)
        _digest(values["certificate_digest"], "authority certificate digest")
        _sha(values["synthetic_commit_sha"], "authority synthetic commit sha")
        _sha(values["generation_tree_sha"], "authority generation tree sha")
        _text(values["landing_fence"], "authority landing fence")
        if values["expected_lease_epoch"] is None:
            if values["expected_lease_fence"] is not None:
                raise LandingReservationError(
                    "authority lease replay anchor is incomplete"
                )
        else:
            _positive_int(values["expected_lease_epoch"], "authority lease epoch")
            _text(values["expected_lease_fence"], "authority lease fence")


@dataclass(frozen=True, slots=True)
class _LeaseEvidence:
    """Opaque evidence for one already-held RMDD-26 arbitration pair."""

    reconciliation_lease_id: str
    reconciliation_lease_epoch: int
    reconciliation_lease_fence: str
    canonical_lease_id: str
    canonical_lease_epoch: int
    canonical_lease_fence: str

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            _LeaseEvidence,
            (
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
            ),
        )
        _text(values["reconciliation_lease_id"], "reconciliation lease id")
        _positive_int(
            values["reconciliation_lease_epoch"], "reconciliation lease epoch"
        )
        _text(values["reconciliation_lease_fence"], "reconciliation lease fence")
        _text(values["canonical_lease_id"], "canonical lease id")
        _positive_int(values["canonical_lease_epoch"], "canonical lease epoch")
        _text(values["canonical_lease_fence"], "canonical lease fence")

    def immutable_payload(self) -> dict[str, object]:
        return {
            "reconciliation_lease_id": self.reconciliation_lease_id,
            "reconciliation_lease_epoch": self.reconciliation_lease_epoch,
            "reconciliation_lease_fence": self.reconciliation_lease_fence,
            "canonical_lease_id": self.canonical_lease_id,
            "canonical_lease_epoch": self.canonical_lease_epoch,
            "canonical_lease_fence": self.canonical_lease_fence,
        }


@dataclass(frozen=True, slots=True)
class _HeldLandingContext:
    """Single captured identity used by the final barrier and attestation."""

    reservation: DurableLandingReservation
    repository: ResolvedRepositoryIdentity
    controller: ControllerIdentity
    request_digest: str
    lease_evidence: _LeaseEvidence
    authority_incarnation: str

    def immutable_payload(self) -> dict[str, object]:
        reservation = self.reservation
        return {
            "reservation_id": reservation.reservation_id,
            "request_id": reservation.request_id,
            "invocation_id": reservation.invocation_id,
            "repository_id": reservation.repository_id,
            "target_ref": reservation.target_ref,
            "request_digest": self.request_digest,
            "resolved_repository_digest": self.repository.digest(),
            "common_dir_id": self.repository.common_dir_id,
            "worktree_id": self.repository.worktree_id,
            "authority_revision": self.repository.authority_revision,
            "controller": self.controller.immutable_payload(),
            "lease_evidence": self.lease_evidence.immutable_payload(),
            "authority_incarnation": self.authority_incarnation,
        }


class _AuthorityRuntime:
    """Concrete native/test runtime owned by a sealed authority instance."""

    __slots__ = (
        "identity",
        "resolved",
        "states",
        "state_index",
        "on_before_hold",
        "on_after_reservation",
        "on_after_capture",
        "on_before_barrier",
        "on_after_barrier",
        "fail_release",
        "lease_lost",
        "lease_overrides",
        "held",
        "enter_count",
        "exit_count",
        "events",
    )

    def __init__(
        self,
        identity: ControllerIdentity | None = None,
        resolved: ResolvedRepositoryIdentity | None = None,
        states: list[LandingStateSnapshot] | None = None,
    ) -> None:
        self.identity = identity
        self.resolved = resolved
        self.states = states or []
        self.state_index = 0
        self.on_before_hold: Callable[[object], None] | None = None
        self.on_after_reservation: Callable[[object], None] | None = None
        self.on_after_capture: Callable[[object], None] | None = None
        self.on_before_barrier: Callable[[object], None] | None = None
        self.on_after_barrier: Callable[[object], None] | None = None
        self.fail_release = False
        self.lease_lost = False
        self.lease_overrides: dict[str, object] = {}
        self.held = False
        self.enter_count = 0
        self.exit_count = 0
        self.events: list[str] = []


class _ExistingReconciliationLease:
    """Default adapter composing RMDD-26 and the RMDD-12 merge lease."""

    def hold(
        self, canonical_path: str, *, operation: str
    ) -> AbstractContextManager[_LeaseEvidence]:
        @contextlib.contextmanager
        def _held() -> Iterator[_LeaseEvidence]:
            try:
                with hold_lease(
                    MERGE_LEASE,
                    operation=operation,
                    path=canonical_path,
                ) as record:
                    acquired_at = record.get("acquired_at", "unknown")
                    reconciliation_id = (
                        "reconciliation:"
                        + _snapshot_digest({"acquired_at": acquired_at})[:32]
                    )
                    reconciliation_fence = _snapshot_digest(
                        {"lease": reconciliation_id, "epoch": 1}
                    )
                    with hold_canonical_lease(canonical_path, note=operation):
                        canonical_id = (
                            "canonical:"
                            + _snapshot_digest(
                                {"path_digest": _path_digest(canonical_path)}
                            )[:32]
                        )
                        canonical_fence = _snapshot_digest(
                            {"lease": canonical_id, "epoch": 1}
                        )
                        yield _LeaseEvidence(
                            reconciliation_lease_id=reconciliation_id,
                            reconciliation_lease_epoch=1,
                            reconciliation_lease_fence=reconciliation_fence,
                            canonical_lease_id=canonical_id,
                            canonical_lease_epoch=1,
                            canonical_lease_fence=canonical_fence,
                        )
            except (BlockedByLease, LeaseUnavailable) as exc:
                raise LandingReservationUnavailable(
                    "canonical or reconciliation lease unavailable"
                ) from exc

        return _held()


def _durable_payload(value: DurableLandingReservation) -> dict[str, object]:
    """Read every durable reservation field for exact barrier comparison."""

    names = (
        "reservation_id",
        "request_id",
        "invocation_id",
        "repository_id",
        "target_ref",
        "request_digest",
        "resolved_repository_digest",
        "common_dir_id",
        "worktree_id",
        "authority_revision",
        "controller_id",
        "owner_id",
        "tenant_id",
        "lease_epoch",
        "fence",
        "authority_epoch",
        "principal_id",
        "session_id",
        "reconciliation_lease_id",
        "reconciliation_lease_epoch",
        "reconciliation_lease_fence",
        "canonical_lease_id",
        "canonical_lease_epoch",
        "canonical_lease_fence",
        "authority_incarnation",
        "active",
    )
    values = _strict_dataclass_values(value, DurableLandingReservation, names)
    value.__post_init__()
    return values


def _clone_state_snapshot(value: object) -> LandingStateSnapshot:
    """Reconstruct a state snapshot so provider aliases cannot cross the seam."""

    if type(value) is not LandingStateSnapshot:
        raise LandingReservationError("state snapshot is invalid")
    source = cast(LandingStateSnapshot, value)
    source.__post_init__()
    payload = source.immutable_payload()
    target = cast(dict[str, object], payload["target"])
    canonical = cast(dict[str, object], payload["canonical"])
    occupancy = cast(dict[str, object], payload["occupancy"])
    certification = cast(dict[str, object], payload["certification"])
    return LandingStateSnapshot(
        resolved_repository_digest=cast(str, payload["resolved_repository_digest"]),
        target=TargetObservation(
            repository_id=cast(str, target["repository_id"]),
            target_ref=cast(str, target["target_ref"]),
            commit_sha=cast(str, target["commit_sha"]),
            tree_sha=cast(str, target["tree_sha"]),
        ),
        canonical=CanonicalObservation(
            repository_id=cast(str, canonical["repository_id"]),
            common_dir_id=cast(str, canonical["common_dir_id"]),
            worktree_id=cast(str, canonical["worktree_id"]),
            state=CanonicalState(cast(str, canonical["state"])),
            private_wip=cast(bool, canonical["private_wip"]),
            index_clean=cast(bool, canonical["index_clean"]),
        ),
        occupancy=OccupancyObservation(
            repository_id=cast(str, occupancy["repository_id"]),
            target_ref=cast(str, occupancy["target_ref"]),
            other_worktree_count=cast(int, occupancy["other_worktree_count"]),
            state=OccupancyState(cast(str, occupancy["state"])),
        ),
        certification=CertificationObservation(
            repository_id=cast(str, certification["repository_id"]),
            target_ref=cast(str, certification["target_ref"]),
            generation_id=cast(str, certification["generation_id"]),
            certificate_digest=cast(str, certification["certificate_digest"]),
            base_sha=cast(str, certification["base_sha"]),
            expected_landing_base_sha=cast(
                str, certification["expected_landing_base_sha"]
            ),
            synthetic_commit_sha=cast(str, certification["synthetic_commit_sha"]),
            generation_tree_sha=cast(str, certification["generation_tree_sha"]),
            landing_fence=cast(str, certification["landing_fence"]),
            certified=cast(bool, certification["certified"]),
        ),
        target_revision=cast(str, payload["target_revision"]),
        canonical_revision=cast(str, payload["canonical_revision"]),
        occupancy_revision=cast(str, payload["occupancy_revision"]),
        certification_revision=cast(str, payload["certification_revision"]),
        snapshot_revision=cast(str, payload["snapshot_revision"]),
    )


def _clone_controller_identity(value: object) -> ControllerIdentity:
    """Copy trusted authentication identity before it enters held context."""

    values = _strict_dataclass_values(
        value,
        ControllerIdentity,
        (
            "controller_id",
            "owner_id",
            "tenant_id",
            "authority_epoch",
            "principal_id",
            "session_id",
        ),
    )
    result = ControllerIdentity(
        controller_id=cast(str, values["controller_id"]),
        owner_id=cast(str, values["owner_id"]),
        tenant_id=cast(str, values["tenant_id"]),
        authority_epoch=cast(int, values["authority_epoch"]),
        principal_id=cast(str | None, values["principal_id"]),
        session_id=cast(str | None, values["session_id"]),
    )
    result.__post_init__()
    return result


def _clone_resolved_identity(value: object) -> ResolvedRepositoryIdentity:
    """Copy canonical identity so provider aliases cannot mutate the context."""

    values = _strict_dataclass_values(
        value,
        ResolvedRepositoryIdentity,
        (
            "repository_id",
            "canonical_path",
            "common_dir_id",
            "worktree_id",
            "authority_revision",
        ),
    )
    result = ResolvedRepositoryIdentity(
        repository_id=cast(str, values["repository_id"]),
        canonical_path=cast(str, values["canonical_path"]),
        common_dir_id=cast(str, values["common_dir_id"]),
        worktree_id=cast(str, values["worktree_id"]),
        authority_revision=cast(str, values["authority_revision"]),
    )
    result.__post_init__()
    return result


class _BoundLandingAuthority:
    """Concrete sealed authority owning the complete landing critical section.

    The controller accepts this exact class only after the module-issued seal
    and issuance registry are verified.  It intentionally has no resolver,
    reader, lease, or barrier arguments: the native durable backend and the
    two RMDD-26 leases are private members of this one authority context.
    """

    __slots__ = (
        "_seal",
        "_runtime",
        "_incarnation",
        "_attestation_secret",
        "_gate",
        "_active",
        "_reservations",
        "_records",
        "_contexts",
        "_recovery",
        "_sequence",
        "_sequence_lock",
    )

    def __init__(self, *, _seal: object, _runtime: _AuthorityRuntime) -> None:
        if _seal is not _AUTHORITY_SEAL or type(_runtime) is not _AuthorityRuntime:
            raise TypeError("landing authority can only be created by its factory")
        self._seal = _seal
        self._runtime = _runtime
        self._incarnation = "authority:" + secrets.token_hex(16)
        self._attestation_secret = secrets.token_bytes(32)
        self._gate = Lock()
        self._active: set[tuple[str, str]] = set()
        self._reservations: dict[tuple[str, str], DurableLandingReservation] = {}
        self._records: dict[str, LandingReservationResult] = {}
        self._contexts: dict[str, _HeldLandingContext] = {}
        self._recovery: dict[str, str] = {}
        self._sequence = 0
        self._sequence_lock = Lock()
        _ISSUED_AUTHORITY_HANDLES[id(self)] = self

    def _sealed(self) -> bool:
        try:
            return (
                type(self) is _BoundLandingAuthority
                and _ISSUED_AUTHORITY_HANDLES.get(id(self)) is self
                and object.__getattribute__(self, "_seal") is _AUTHORITY_SEAL
            )
        except Exception:
            return False

    def __repr__(self) -> str:
        return "<bound landing authority>"

    def _hook(self, name: str) -> None:
        hook = getattr(self._runtime, name)
        if hook is not None:
            hook(self)

    def _identity(self) -> ControllerIdentity:
        value = self._runtime.identity
        if type(value) is not ControllerIdentity:
            raise LandingReservationUnavailable("authority identity unavailable")
        return _clone_controller_identity(value)

    def _resolved(self) -> ResolvedRepositoryIdentity:
        value = self._runtime.resolved
        if type(value) is not ResolvedRepositoryIdentity:
            raise LandingReservationUnavailable("authority repository unavailable")
        return _clone_resolved_identity(value)

    def _capture_state(self) -> LandingStateSnapshot:
        states = self._runtime.states
        if type(states) is not list or not states:
            raise LandingReservationSourceUnavailable("authority state unavailable")
        index = self._runtime.state_index
        if type(index) is not int or index < 0 or index >= len(states):
            raise LandingReservationSourceUnavailable("authority state unavailable")
        self._runtime.events.append("state-read")
        return _clone_state_snapshot(states[index])

    def _current_evidence(self, original: _LeaseEvidence) -> _LeaseEvidence:
        if self._runtime.lease_lost:
            raise LandingReservationStale("authority lease was lost")
        values: dict[str, object] = {}
        try:
            for name in (
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
            ):
                values[name] = self._runtime.lease_overrides.get(
                    name, getattr(original, name)
                )
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise LandingReservationStale("authority lease evidence changed") from exc
        try:
            evidence = _LeaseEvidence(
                reconciliation_lease_id=cast(str, values["reconciliation_lease_id"]),
                reconciliation_lease_epoch=cast(
                    int, values["reconciliation_lease_epoch"]
                ),
                reconciliation_lease_fence=cast(
                    str, values["reconciliation_lease_fence"]
                ),
                canonical_lease_id=cast(str, values["canonical_lease_id"]),
                canonical_lease_epoch=cast(int, values["canonical_lease_epoch"]),
                canonical_lease_fence=cast(str, values["canonical_lease_fence"]),
            )
            _text(evidence.reconciliation_lease_id, "lease id")
            _positive_int(evidence.reconciliation_lease_epoch, "lease epoch")
            _text(evidence.reconciliation_lease_fence, "lease fence")
            _text(evidence.canonical_lease_id, "lease id")
            _positive_int(evidence.canonical_lease_epoch, "lease epoch")
            _text(evidence.canonical_lease_fence, "lease fence")
            return evidence
        except (LandingReservationError, TypeError) as exc:
            raise LandingReservationStale("authority lease evidence changed") from exc

    def _context_matches(
        self,
        request: LandingReservationRequest,
        context: _HeldLandingContext,
        key: tuple[str, str],
    ) -> None:
        try:
            identity = self._identity()
        except (LandingReservationError, AttributeError, KeyError, TypeError) as exc:
            raise LandingReservationStale(
                "authority controller identity changed"
            ) from exc
        current_identity = identity.immutable_payload()
        original_identity = context.controller.immutable_payload()
        if current_identity["owner_id"] != original_identity["owner_id"]:
            raise LandingReservationOwnerMismatch("authority owner changed")
        if current_identity["tenant_id"] != original_identity["tenant_id"]:
            raise LandingReservationTenantMismatch("authority tenant changed")
        if current_identity["principal_id"] != original_identity["principal_id"]:
            raise LandingReservationPrincipalMismatch("authority principal changed")
        if current_identity["session_id"] != original_identity["session_id"]:
            raise LandingReservationSessionMismatch("authority session changed")
        if current_identity["authority_epoch"] != original_identity["authority_epoch"]:
            raise LandingReservationAuthorityEpochMismatch("authority epoch changed")
        if current_identity["controller_id"] != original_identity["controller_id"]:
            raise LandingReservationAuthorityEpochMismatch(
                "authority controller changed"
            )
        try:
            resolved = self._resolved()
        except (LandingReservationError, AttributeError, KeyError, TypeError) as exc:
            raise LandingReservationStale(
                "authority repository identity changed"
            ) from exc
        if resolved.immutable_payload() != context.repository.immutable_payload():
            raise LandingReservationStale("authority repository identity changed")
        current = self._reservations.get(key)
        replacement = self._runtime.lease_overrides.get("reservation")
        if replacement is not None:
            if type(replacement) is not DurableLandingReservation:
                raise LandingReservationStale("authority reservation was replaced")
            current = replacement
        if type(current) is not DurableLandingReservation:
            raise LandingReservationStale("authority reservation was lost")
        try:
            current_payload = _durable_payload(current)
            original_payload = _durable_payload(context.reservation)
        except (LandingReservationError, AttributeError, KeyError, TypeError) as exc:
            raise LandingReservationStale("authority reservation changed") from exc
        if current_payload != original_payload:
            raise LandingReservationStale("authority reservation changed")
        evidence = self._current_evidence(context.lease_evidence)
        if evidence != context.lease_evidence:
            raise LandingReservationStale("authority lease changed")
        if self._incarnation != context.authority_incarnation:
            raise LandingReservationStale("authority incarnation changed")
        if request.repository_id != key[0] or request.target_ref != key[1]:
            raise LandingReservationStale("authority target identity changed")

    def _attestation_message(
        self, snapshot: LandingReservationSnapshot, context: _HeldLandingContext
    ) -> bytes:
        payload = snapshot.immutable_payload()
        payload.pop("authority_attestation", None)
        return json.dumps(
            {"snapshot": payload, "context": context.immutable_payload()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")

    def _attest(
        self, snapshot: LandingReservationSnapshot, context: _HeldLandingContext
    ) -> str:
        return hmac.new(
            self._attestation_secret,
            self._attestation_message(snapshot, context),
            hashlib.sha256,
        ).hexdigest()

    @contextlib.contextmanager
    def _held_leases(
        self, repository: ResolvedRepositoryIdentity, operation: str
    ) -> Iterator[_LeaseEvidence]:
        body_failed = False
        entered = False
        try:
            with _ExistingReconciliationLease().hold(
                repository.canonical_path, operation=operation
            ) as evidence:
                if type(evidence) is not _LeaseEvidence:
                    raise LandingReservationError("lease evidence is invalid")
                self._runtime.held = True
                entered = True
                self._runtime.enter_count += 1
                self._runtime.events.append("lease-enter")
                try:
                    yield evidence
                except BaseException:
                    body_failed = True
                    raise
                finally:
                    self._runtime.held = False
                    self._runtime.exit_count += 1
                    self._runtime.events.append("lease-exit")
                    if self._runtime.fail_release:
                        raise LandingReservationRecoveryRequired(
                            "authority lease release requires recovery"
                        )
        except (BlockedByLease, LeaseUnavailable) as exc:
            raise LandingReservationUnavailable(
                "landing arbitration unavailable"
            ) from exc
        except OSError as exc:
            if entered and not body_failed:
                raise LandingReservationRecoveryRequired(
                    "authority lease release requires recovery"
                ) from exc
            raise LandingReservationUnavailable(
                "landing arbitration unavailable"
            ) from exc
        except LandingReservationRecoveryRequired:
            raise
        except Exception as exc:
            if entered and not body_failed:
                raise LandingReservationRecoveryRequired(
                    "authority lease release requires recovery"
                ) from exc
            raise

    def _make_snapshot(
        self,
        request: LandingReservationRequest,
        state: LandingStateSnapshot,
        context: _HeldLandingContext,
        barrier_revision: str,
    ) -> LandingReservationSnapshot:
        reservation = context.reservation
        evidence = context.lease_evidence
        certification = state.certification
        payload: dict[str, object] = {
            "reservation_id": reservation.reservation_id,
            "request_digest": context.request_digest,
            "resolved_repository_digest": context.repository.digest(),
            "repository_id": request.repository_id,
            "target_ref": request.target_ref,
            "expected_target_sha": request.expected_target_sha,
            "expected_base_sha": request.expected_base_sha,
            "observed_target_sha": state.target.commit_sha,
            "observed_target_tree_sha": state.target.tree_sha,
            "common_dir_id": state.canonical.common_dir_id,
            "worktree_id": state.canonical.worktree_id,
            "authority_revision": context.repository.authority_revision,
            "lease_epoch": reservation.lease_epoch,
            "lease_fence": reservation.fence,
            "tenant_id": context.controller.tenant_id,
            "authority_epoch": context.controller.authority_epoch,
            "generation_id": certification.generation_id,
            "certificate_digest": certification.certificate_digest,
            "synthetic_commit_sha": certification.synthetic_commit_sha,
            "generation_tree_sha": certification.generation_tree_sha,
            "landing_fence": certification.landing_fence,
            "target_worktree_count": state.occupancy.other_worktree_count,
            "target_revision": state.target_revision,
            "canonical_revision": state.canonical_revision,
            "occupancy_revision": state.occupancy_revision,
            "certification_revision": state.certification_revision,
            "snapshot_revision": state.snapshot_revision,
            "barrier_revision": barrier_revision,
            "reconciliation_lease_id": evidence.reconciliation_lease_id,
            "reconciliation_lease_epoch": evidence.reconciliation_lease_epoch,
            "reconciliation_lease_fence": evidence.reconciliation_lease_fence,
            "canonical_lease_id": evidence.canonical_lease_id,
            "canonical_lease_epoch": evidence.canonical_lease_epoch,
            "canonical_lease_fence": evidence.canonical_lease_fence,
            "authority_incarnation": context.authority_incarnation,
        }
        unsigned_payload = {**payload, "authority_attestation": "0" * 64}
        unsigned_digest = _snapshot_digest(
            {"schema": "rmdd-13-landing-reservation:v2", **unsigned_payload}
        )
        snapshot_type = cast(Any, LandingReservationSnapshot)
        unsigned = snapshot_type(
            **unsigned_payload,
            digest=unsigned_digest,
        )
        attestation = self._attest(unsigned, context)
        payload["authority_attestation"] = attestation
        digest = _snapshot_digest(
            {"schema": "rmdd-13-landing-reservation:v2", **payload}
        )
        return snapshot_type(
            **payload,
            digest=digest,
        )

    def _final_barrier(
        self,
        request: LandingReservationRequest,
        context: _HeldLandingContext,
        captured: LandingStateSnapshot,
        key: tuple[str, str],
    ) -> LandingStateSnapshot:
        self._hook("on_before_barrier")
        self._context_matches(request, context, key)
        latest = self._capture_state()
        if latest.immutable_payload() != captured.immutable_payload():
            raise LandingReservationStale("authoritative state changed")
        state_code = LandingReservationController._check_state(
            request, context.repository, latest
        )
        if state_code is not None:
            raise LandingReservationStale("authoritative state no longer landable")
        # Re-check every context and both lease identities at the end of the
        # barrier, after the last source read and before attestation.
        self._context_matches(request, context, key)
        return latest

    def _attempt(
        self,
        request: LandingReservationRequest,
        controller: ControllerIdentity,
        repository: ResolvedRepositoryIdentity,
        request_digest: str,
        key: tuple[str, str],
    ) -> LandingReservationResult:
        self._hook("on_before_hold")
        with self._held_leases(repository, "reserve certified landing") as evidence:
            self._runtime.events.append("reserve")
            with self._sequence_lock:
                self._sequence += 1
                sequence = self._sequence
            reservation = DurableLandingReservation(
                reservation_id=f"reservation:{sequence}",
                request_id=request.request_id,
                invocation_id=request.invocation_id,
                repository_id=request.repository_id,
                target_ref=request.target_ref,
                request_digest=request_digest,
                resolved_repository_digest=repository.digest(),
                common_dir_id=repository.common_dir_id,
                worktree_id=repository.worktree_id,
                authority_revision=repository.authority_revision,
                controller_id=controller.controller_id,
                owner_id=controller.owner_id,
                tenant_id=controller.tenant_id,
                lease_epoch=sequence,
                fence=_snapshot_digest(
                    {"reservation": sequence, "incarnation": self._incarnation}
                ),
                authority_epoch=controller.authority_epoch,
                principal_id=controller.principal_id,
                session_id=controller.session_id,
                reconciliation_lease_id=evidence.reconciliation_lease_id,
                reconciliation_lease_epoch=evidence.reconciliation_lease_epoch,
                reconciliation_lease_fence=evidence.reconciliation_lease_fence,
                canonical_lease_id=evidence.canonical_lease_id,
                canonical_lease_epoch=evidence.canonical_lease_epoch,
                canonical_lease_fence=evidence.canonical_lease_fence,
                authority_incarnation=self._incarnation,
            )
            reservation.__post_init__()
            if request.expected_lease_epoch is not None:
                if reservation.lease_epoch != request.expected_lease_epoch:
                    raise LandingReservationStale("reservation epoch is stale")
                if reservation.fence != request.expected_lease_fence:
                    raise LandingReservationFenceMismatch("reservation fence is stale")
            self._reservations[key] = reservation
            self._hook("on_after_reservation")
            context = _HeldLandingContext(
                reservation=reservation,
                repository=repository,
                controller=controller,
                request_digest=request_digest,
                lease_evidence=evidence,
                authority_incarnation=self._incarnation,
            )
            captured = self._capture_state()
            state_code = LandingReservationController._check_state(
                request, repository, captured
            )
            if state_code is not None:
                return _refuse(state_code)
            self._hook("on_after_capture")
            latest = self._final_barrier(request, context, captured, key)
            snapshot = self._make_snapshot(
                request, latest, context, f"barrier:{reservation.lease_epoch}"
            )
            result = LandingReservationResult(
                accepted=True,
                detail="landing reservation acquired and sealed barrier passed",
                snapshot=snapshot,
            )
            self._contexts[reservation.reservation_id] = context
            self._hook("on_after_barrier")
            return result

    def _record_recovery(self, key: tuple[str, str], detail: str) -> None:
        token = _snapshot_digest({"repo": key[0], "target": key[1]})
        self._recovery[token] = "recovery required"
        while len(self._recovery) > _MAX_RECOVERY_RECORDS:
            self._recovery.pop(next(iter(self._recovery)))

    def acquire_landing(
        self, request: LandingReservationRequest
    ) -> LandingReservationResult:
        """Run the complete reservation/barrier operation in one authority."""

        if not self._sealed():
            return _refuse(LandingReservationRefusalCode.AUTHORITY_INVALID)
        if type(request) is not LandingReservationRequest:
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)
        try:
            request.__post_init__()
        except TrustedReservationRuntimeError:
            raise
        except (LandingReservationError, AttributeError, KeyError, TypeError):
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)
        try:
            controller = self._identity()
            repository = self._resolved()
            if (
                repository.repository_id != request.repository_id
                or repository.canonical_path != request.repository.canonical_path
            ):
                return _refuse(LandingReservationRefusalCode.REPOSITORY_MISMATCH)
            request_digest = request.digest(repository, controller)
            authority_input = _LandingAuthorityInput(
                request_id=request.request_id,
                invocation_id=request.invocation_id,
                repository_id=request.repository_id,
                target_ref=request.target_ref,
                request_digest=request_digest,
                resolved_repository=repository,
                controller=controller,
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
            authority_input.__post_init__()
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationUnavailable:
            return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
        except (LandingReservationError, AttributeError, KeyError, TypeError):
            return _refuse(LandingReservationRefusalCode.AUTHORITY_INVALID)

        key = (request.repository_id, request.target_ref)
        if not self._gate.acquire(blocking=False):
            return _refuse(LandingReservationRefusalCode.RESERVATION_CONFLICT)
        try:
            existing = self._records.get(request_digest)
            if existing is not None:
                existing_snapshot = existing.snapshot
                if (
                    request.expected_lease_epoch is not None
                    and existing_snapshot is not None
                    and existing_snapshot.lease_epoch != request.expected_lease_epoch
                ):
                    return _refuse(LandingReservationRefusalCode.EPOCH_MISMATCH)
                if (
                    request.expected_lease_fence is not None
                    and existing_snapshot is not None
                    and existing_snapshot.lease_fence != request.expected_lease_fence
                ):
                    return _refuse(LandingReservationRefusalCode.FENCE_MISMATCH)
                return existing
            for prior in self._records.values():
                prior_snapshot = prior.snapshot
                if (
                    prior_snapshot is not None
                    and prior_snapshot.repository_id == request.repository_id
                    and prior_snapshot.target_ref == request.target_ref
                    and prior_snapshot.authority_epoch != controller.authority_epoch
                ):
                    return _refuse(
                        LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH
                    )
            if key in self._active or key in self._reservations:
                return _refuse(LandingReservationRefusalCode.RESERVATION_CONFLICT)
            self._active.add(key)
        finally:
            self._gate.release()

        success = False
        try:
            try:
                result = self._attempt(
                    request, controller, repository, request_digest, key
                )
            except TrustedReservationRuntimeError:
                raise
            except LandingReservationRecoveryRequired as exc:
                self._record_recovery(key, str(exc))
                return _refuse(LandingReservationRefusalCode.RECOVERY_REQUIRED)
            except LandingReservationConflict:
                return _refuse(LandingReservationRefusalCode.RESERVATION_CONFLICT)
            except LandingReservationOwnerMismatch:
                return _refuse(LandingReservationRefusalCode.OWNER_MISMATCH)
            except LandingReservationTenantMismatch:
                return _refuse(LandingReservationRefusalCode.TENANT_MISMATCH)
            except LandingReservationPrincipalMismatch:
                return _refuse(LandingReservationRefusalCode.PRINCIPAL_MISMATCH)
            except LandingReservationSessionMismatch:
                return _refuse(LandingReservationRefusalCode.SESSION_MISMATCH)
            except LandingReservationAuthorityEpochMismatch:
                return _refuse(LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH)
            except LandingReservationFenceMismatch:
                return _refuse(LandingReservationRefusalCode.FENCE_MISMATCH)
            except LandingReservationStale:
                return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
            except LandingReservationSourceUnavailable:
                return _refuse(LandingReservationRefusalCode.SOURCE_UNAVAILABLE)
            except LandingReservationUnavailable:
                return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)
            except LandingReservationError:
                return _refuse(LandingReservationRefusalCode.SOURCE_INVALID)
            except (
                ValueError,
                TypeError,
                OSError,
                KeyError,
                IndexError,
                AttributeError,
                UnicodeError,
            ):
                return _refuse(LandingReservationRefusalCode.SOURCE_UNAVAILABLE)
            if result.accepted:
                self._records[request_digest] = result
                success = True
            else:
                self._reservations.pop(key, None)
            return result
        except LandingReservationRecoveryRequired as exc:
            self._record_recovery(key, str(exc))
            self._reservations.pop(key, None)
            return _refuse(LandingReservationRefusalCode.RECOVERY_REQUIRED)
        finally:
            if not success:
                self._reservations.pop(key, None)
            self._active.discard(key)

    def verify_attested_snapshot(self, snapshot: LandingReservationSnapshot) -> bool:
        """Verify a CP3 snapshot against this authority's secret/context."""

        if not self._sealed() or type(snapshot) is not LandingReservationSnapshot:
            return False
        try:
            snapshot.__post_init__()
            context = self._contexts.get(snapshot.reservation_id)
            if context is None:
                return False
            result = self._records.get(snapshot.request_digest)
            if result is None or result.snapshot is None:
                return False
            if result.snapshot.immutable_payload() != snapshot.immutable_payload():
                return False
            expected = self._attest(snapshot, context)
            return hmac.compare_digest(expected, snapshot.authority_attestation)
        except (LandingReservationError, TypeError, ValueError):
            return False


def _new_bound_authority(runtime: _AuthorityRuntime) -> _BoundLandingAuthority:
    return _BoundLandingAuthority(_seal=_AUTHORITY_SEAL, _runtime=runtime)


def create_landing_authority() -> _BoundLandingAuthority:
    """Create a sealed native authority handle for production wiring."""

    return _new_bound_authority(_AuthorityRuntime())


def _create_test_authority(
    identity: ControllerIdentity,
    resolved: ResolvedRepositoryIdentity,
    states: list[LandingStateSnapshot],
) -> _BoundLandingAuthority:
    """Private exact-type test seam; not a public authority injection API."""

    return _new_bound_authority(
        _AuthorityRuntime(identity=identity, resolved=resolved, states=states)
    )


LandingReservationAuthority = _BoundLandingAuthority


def _validate_observation(value: object, expected: type[Any]) -> object:
    """Validate an exact observation before reconstructing the source snapshot."""

    if type(value) is not expected:
        raise LandingReservationError("state source returned an invalid observation")
    try:
        cast(Any, value).__post_init__()
    except TrustedReservationRuntimeError:
        raise
    except LandingReservationError:
        raise
    except (AttributeError, KeyError, IndexError, TypeError, UnicodeError) as exc:
        raise LandingReservationError(
            "state source returned an invalid observation"
        ) from exc
    except Exception as exc:
        raise LandingReservationError(
            "state source returned an invalid observation"
        ) from exc
    return value


def _snapshot_digest(payload: Mapping[str, object]) -> str:
    """Hash one bounded canonical mapping without exposing source values."""

    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _path_digest(path: str) -> str:
    """Hash a trusted path before it enters opaque lease evidence."""

    return hashlib.sha256(path.encode("utf-8")).hexdigest()


__all__ = [
    "CanonicalObservation",
    "CanonicalState",
    "CertificationObservation",
    "ControllerIdentity",
    "create_landing_authority",
    "DurableLandingReservation",
    "LandingReservationAuthority",
    "LandingReservationConflict",
    "LandingReservationController",
    "LandingReservationError",
    "LandingReservationRecoveryRequired",
    "LandingReservationAuthorityEpochMismatch",
    "LandingReservationFenceMismatch",
    "LandingReservationOwnerMismatch",
    "LandingReservationPrincipalMismatch",
    "LandingReservationRefusalCode",
    "LandingReservationRequest",
    "LandingReservationResult",
    "LandingReservationSnapshot",
    "LandingReservationStale",
    "LandingReservationTenantMismatch",
    "LandingReservationSessionMismatch",
    "LandingStateSnapshot",
    "LandingValidationBarrier",
    "LandingReservationUnavailable",
    "OccupancyObservation",
    "OccupancyState",
    "ResolvedRepositoryIdentity",
    "TargetObservation",
    "TrustedReservationRuntimeError",
    "normalize_target_ref",
    "reserve_landing",
]


class LandingReservationController:
    """Small public controller over one sealed authority capability.

    There is intentionally no protocol or duck-typed authority path here.  A
    controller can only operate a handle emitted by this module's factory;
    all repository resolution, authentication, lease acquisition, state reads,
    fencing, and attestation remain inside that authority.
    """

    def __init__(self, authority: _BoundLandingAuthority) -> None:
        if type(authority) is not _BoundLandingAuthority or not authority._sealed():
            raise TypeError("landing controller requires a sealed authority")
        self._authority = authority

    @staticmethod
    def _check_state(
        request: LandingReservationRequest,
        repository: ResolvedRepositoryIdentity,
        state: LandingStateSnapshot,
    ) -> LandingReservationRefusalCode | None:
        """Check one closed, canonical source snapshot against the request."""

        if type(request) is not LandingReservationRequest:
            return LandingReservationRefusalCode.REQUEST_INVALID
        if type(repository) is not ResolvedRepositoryIdentity:
            return LandingReservationRefusalCode.SOURCE_INVALID
        if type(state) is not LandingStateSnapshot:
            return LandingReservationRefusalCode.SOURCE_INVALID
        try:
            request.__post_init__()
            repository.__post_init__()
            state.__post_init__()
            if state.resolved_repository_digest != repository.digest():
                return LandingReservationRefusalCode.REPOSITORY_MISMATCH
            target = state.target
            canonical = state.canonical
            occupancy = state.occupancy
            certification = state.certification
            if (
                target.repository_id != request.repository_id
                or canonical.repository_id != request.repository_id
                or certification.repository_id != request.repository_id
            ):
                return LandingReservationRefusalCode.REPOSITORY_MISMATCH
            if target.target_ref != request.target_ref:
                return LandingReservationRefusalCode.TARGET_MISMATCH
            if target.commit_sha != request.expected_target_sha:
                return LandingReservationRefusalCode.TARGET_MOVED
            if (
                canonical.common_dir_id != repository.common_dir_id
                or canonical.worktree_id != repository.worktree_id
            ):
                return LandingReservationRefusalCode.CANONICAL_STATE_CHANGED
            if canonical.state is CanonicalState.UNKNOWN:
                return LandingReservationRefusalCode.CANONICAL_STATE_INVALID
            if (
                canonical.state is CanonicalState.PRIVATE_WIP
                or canonical.private_wip
                or not canonical.index_clean
            ):
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
            if (
                occupancy.state is not OccupancyState.FREE
                or occupancy.other_worktree_count
            ):
                return LandingReservationRefusalCode.TARGET_OCCUPIED
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
        except TrustedReservationRuntimeError:
            raise
        except (LandingReservationError, AttributeError, KeyError, TypeError):
            return LandingReservationRefusalCode.SOURCE_INVALID

    def reserve(self, request: LandingReservationRequest) -> LandingReservationResult:
        """Acquire one bounded reservation through the sealed authority."""

        return self._authority.acquire_landing(request)


def reserve_landing(
    request: LandingReservationRequest,
    *,
    authority: _BoundLandingAuthority,
) -> LandingReservationResult:
    """Functional adapter for the exact sealed authority handle."""

    return LandingReservationController(authority).reserve(request)
