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
_MAX_PATH_BYTES = 4096
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

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            CanonicalObservation,
            ("repository_id", "common_dir_id", "worktree_id", "state", "private_wip"),
        )
        _text(values["repository_id"], "canonical repository_id", maximum=_MAX_ID_BYTES)
        _text(values["common_dir_id"], "canonical common_dir_id", maximum=_MAX_ID_BYTES)
        _text(values["worktree_id"], "canonical worktree_id", maximum=_MAX_ID_BYTES)
        if type(values["state"]) is not CanonicalState:
            raise LandingReservationError("canonical state is invalid")
        if type(values["private_wip"]) is not bool:
            raise LandingReservationError("canonical private_wip is invalid")


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


class LandingReservationAuthority(Protocol):
    """Durable, authenticated reservation authority.

    Implementations are expected to bind these calls to Graph-OS/WorkItem or
    an equivalent cross-host authority.  A local projection may implement
    status reads, but must not be substituted for these methods.
    """

    def authenticate_controller(self, invocation_id: str) -> ControllerIdentity:
        """Resolve the invocation to a trusted controller/owner identity."""

    def resolve_repository(
        self, repository: RepositoryIdentity
    ) -> ResolvedRepositoryIdentity:
        """Resolve the exact canonical checkout from repository identity."""

    def hold_landing(
        self,
        repository: ResolvedRepositoryIdentity,
        target_ref: str,
        *,
        operation: str,
    ) -> AbstractContextManager[None]:
        """Acquire opaque reconciliation/canonical leases through authority."""

    def reserve_landing(
        self, request: _LandingAuthorityInput, controller: ControllerIdentity
    ) -> DurableLandingReservation:
        """Atomically reserve one repository+target and support replay."""

    def read_landing_snapshot(
        self,
        repository: ResolvedRepositoryIdentity,
        request: LandingReservationRequest,
    ) -> LandingStateSnapshot:
        """Read all bounded sources atomically or with source revisions."""

    def validate_landing_barrier(
        self,
        reservation: DurableLandingReservation,
        controller: ControllerIdentity,
        repository: ResolvedRepositoryIdentity,
        snapshot: LandingStateSnapshot,
    ) -> LandingValidationBarrier:
        """Revalidate reservation, both leases, and every source token atomically."""


class _LandingStateReader(Protocol):
    """Read-only source port owned by a trusted authority adapter."""

    def read_landing_snapshot(
        self,
        repository: ResolvedRepositoryIdentity,
        request: LandingReservationRequest,
    ) -> LandingStateSnapshot:
        """Read all sources with bounded revision tokens."""

    def validate_landing_barrier(
        self,
        repository: ResolvedRepositoryIdentity,
        reservation: DurableLandingReservation,
        controller: ControllerIdentity,
        snapshot: LandingStateSnapshot,
    ) -> LandingValidationBarrier:
        """Perform the final atomic source/revision barrier."""


class _ReconciliationLeasePort(Protocol):
    """Existing arbitration lease composition; no second queue is introduced."""

    def hold(
        self, canonical_path: str, *, operation: str
    ) -> AbstractContextManager[None]:
        """Hold the existing repo merge/canonical leases."""


class _ExistingReconciliationLease:
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
    repository: ResolvedRepositoryIdentity,
    request_digest: str,
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
        or lease.resolved_repository_digest != repository.digest()
        or lease.common_dir_id != repository.common_dir_id
        or lease.worktree_id != repository.worktree_id
        or lease.authority_revision != repository.authority_revision
    ):
        raise LandingReservationError("authority reservation identity does not match")
    if (
        lease.controller_id != controller.controller_id
        or lease.owner_id != controller.owner_id
    ):
        raise LandingReservationOwnerMismatch(
            "authority reservation owner does not match"
        )
    if lease.tenant_id != controller.tenant_id:
        raise LandingReservationTenantMismatch(
            "authority reservation tenant does not match"
        )
    if lease.authority_epoch != controller.authority_epoch:
        raise LandingReservationAuthorityEpochMismatch(
            "authority reservation epoch does not match"
        )
    if lease.principal_id != controller.principal_id:
        raise LandingReservationPrincipalMismatch(
            "authority reservation principal does not match"
        )
    if lease.session_id != controller.session_id:
        raise LandingReservationSessionMismatch(
            "authority reservation session does not match"
        )
    if lease.request_digest != request_digest:
        raise LandingReservationStale("authority reservation input changed")
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
    **kwargs: object,
) -> tuple[object | None, LandingReservationRefusalCode | None]:
    if not callable(call):
        return None, unavailable
    try:
        return call(*args, **kwargs), None
    except (
        LandingReservationConflict,
        LandingReservationStale,
        LandingReservationUnavailable,
    ):
        raise
    except OSError:
        return None, unavailable
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        return None, LandingReservationRefusalCode.AUTHORITY_INVALID


def _snapshot_digest(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class LandingReservationController:
    """Controller for CP2 reservation and one authority-owned final barrier."""

    def __init__(self, authority: LandingReservationAuthority) -> None:
        # The controller intentionally accepts no lease or reader argument.
        # Both are authority-owned capabilities and remain opaque here.
        self._authority = authority

    @staticmethod
    def _check_state(
        request: LandingReservationRequest,
        repository: ResolvedRepositoryIdentity,
        state: LandingStateSnapshot,
    ) -> LandingReservationRefusalCode | None:
        try:
            state.__post_init__()
            repository.__post_init__()
        except LandingReservationError:
            return LandingReservationRefusalCode.SOURCE_INVALID
        if state.resolved_repository_digest != repository.digest():
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        target = state.target
        canonical = state.canonical
        occupancy = state.occupancy
        certification = state.certification
        if target.repository_id != request.repository_id:
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        if target.target_ref != request.target_ref:
            return LandingReservationRefusalCode.TARGET_MISMATCH
        if target.commit_sha != request.expected_target_sha:
            return LandingReservationRefusalCode.TARGET_MOVED
        if canonical.repository_id != request.repository_id:
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        if (
            canonical.common_dir_id != repository.common_dir_id
            or canonical.worktree_id != repository.worktree_id
        ):
            return LandingReservationRefusalCode.CANONICAL_STATE_CHANGED
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
    def _read_snapshot(
        authority: LandingReservationAuthority,
        repository: ResolvedRepositoryIdentity,
        request: LandingReservationRequest,
    ) -> tuple[LandingStateSnapshot | None, LandingReservationRefusalCode | None]:
        call = _safe_method(authority, "read_landing_snapshot")
        if not callable(call):
            return None, LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
        try:
            value = call(repository, request)
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationUnavailable:
            return None, LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
        except OSError:
            return None, LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
        except Exception:
            return None, LandingReservationRefusalCode.SOURCE_UNAVAILABLE
        try:
            if type(value) is not LandingStateSnapshot:
                raise LandingReservationError("state snapshot is invalid")
            snapshot = cast(LandingStateSnapshot, value)
            snapshot.__post_init__()
            return snapshot, None
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationError:
            return None, LandingReservationRefusalCode.SOURCE_INVALID

    @staticmethod
    def _validate_barrier(
        value: object,
        request: LandingReservationRequest,
        controller: ControllerIdentity,
        repository: ResolvedRepositoryIdentity,
        request_digest: str,
        prior: LandingStateSnapshot,
    ) -> LandingValidationBarrier:
        if type(value) is not LandingValidationBarrier:
            raise LandingReservationError("authority barrier is invalid")
        barrier = cast(LandingValidationBarrier, value)
        barrier.__post_init__()
        lease = _validate_lease(
            barrier.reservation, request, controller, repository, request_digest
        )
        snapshot = barrier.snapshot
        for field_name in (
            "resolved_repository_digest",
            "target_revision",
            "canonical_revision",
            "occupancy_revision",
            "certification_revision",
            "snapshot_revision",
        ):
            if getattr(snapshot, field_name) != getattr(prior, field_name):
                raise LandingReservationStale("state changed before final barrier")
        if (
            lease.reservation_id != barrier.reservation.reservation_id
            or lease.lease_epoch != barrier.reservation.lease_epoch
            or lease.fence != barrier.reservation.fence
        ):
            raise LandingReservationStale("reservation changed at final barrier")
        return barrier

    @staticmethod
    def _snapshot_result(
        request: LandingReservationRequest,
        repository: ResolvedRepositoryIdentity,
        controller: ControllerIdentity,
        request_digest: str,
        barrier: LandingValidationBarrier,
    ) -> LandingReservationSnapshot:
        lease = barrier.reservation
        state = barrier.snapshot
        payload: dict[str, Any] = {
            "reservation_id": lease.reservation_id,
            "request_digest": request_digest,
            "resolved_repository_digest": repository.digest(),
            "repository_id": request.repository_id,
            "target_ref": request.target_ref,
            "expected_target_sha": request.expected_target_sha,
            "expected_base_sha": request.expected_base_sha,
            "observed_target_sha": state.target.commit_sha,
            "observed_target_tree_sha": state.target.tree_sha,
            "common_dir_id": state.canonical.common_dir_id,
            "worktree_id": state.canonical.worktree_id,
            "authority_revision": repository.authority_revision,
            "lease_epoch": lease.lease_epoch,
            "lease_fence": lease.fence,
            "tenant_id": controller.tenant_id,
            "authority_epoch": controller.authority_epoch,
            "generation_id": state.certification.generation_id,
            "certificate_digest": state.certification.certificate_digest,
            "synthetic_commit_sha": state.certification.synthetic_commit_sha,
            "generation_tree_sha": state.certification.generation_tree_sha,
            "landing_fence": state.certification.landing_fence,
            "target_worktree_count": state.occupancy.other_worktree_count,
            "target_revision": state.target_revision,
            "canonical_revision": state.canonical_revision,
            "occupancy_revision": state.occupancy_revision,
            "certification_revision": state.certification_revision,
            "snapshot_revision": state.snapshot_revision,
            "barrier_revision": barrier.barrier_revision,
        }
        digest = _snapshot_digest(
            {"schema": "rmdd-13-landing-reservation:v2", **payload}
        )
        return LandingReservationSnapshot(**payload, digest=digest)

    def reserve(self, request: LandingReservationRequest) -> LandingReservationResult:
        """Reserve and finalize one authority-owned revision barrier."""

        if type(request) is not LandingReservationRequest:
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)
        try:
            request.__post_init__()
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationError:
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)

        try:
            identity, code = _authority_call(
                _safe_method(self._authority, "authenticate_controller"),
                request.invocation_id,
            )
        except LandingReservationUnavailable:
            return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
        if code is not None:
            return _refuse(code)
        try:
            if type(identity) is not ControllerIdentity:
                raise LandingReservationError("controller identity is invalid")
            controller = cast(ControllerIdentity, identity)
            controller.__post_init__()
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationError:
            return _refuse(LandingReservationRefusalCode.AUTHORITY_INVALID)

        try:
            resolved_value, code = _authority_call(
                _safe_method(self._authority, "resolve_repository"), request.repository
            )
        except LandingReservationUnavailable:
            return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
        if code is not None:
            return _refuse(code)
        try:
            if type(resolved_value) is not ResolvedRepositoryIdentity:
                raise LandingReservationError("resolved repository is invalid")
            repository = cast(ResolvedRepositoryIdentity, resolved_value)
            repository.__post_init__()
            if (
                repository.repository_id != request.repository_id
                or repository.canonical_path != request.repository.canonical_path
            ):
                return _refuse(LandingReservationRefusalCode.REPOSITORY_MISMATCH)
            request_digest = request.digest(repository, controller)
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationError:
            return _refuse(LandingReservationRefusalCode.REPOSITORY_MISMATCH)

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
        try:
            authority_input.__post_init__()
        except TrustedReservationRuntimeError:
            raise
        except LandingReservationError:
            return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)

        try:
            lease_context, code = _authority_call(
                _safe_method(self._authority, "hold_landing"),
                repository,
                request.target_ref,
                operation="reserve certified landing",
            )
        except LandingReservationUnavailable:
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)
        if code is not None:
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)
        if lease_context is None:
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)

        try:
            with cast(AbstractContextManager[None], lease_context):
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
                    lease = _validate_lease(
                        durable, request, controller, repository, request_digest
                    )
                except LandingReservationOwnerMismatch:
                    return _refuse(LandingReservationRefusalCode.OWNER_MISMATCH)
                except LandingReservationTenantMismatch:
                    return _refuse(LandingReservationRefusalCode.TENANT_MISMATCH)
                except LandingReservationPrincipalMismatch:
                    return _refuse(LandingReservationRefusalCode.PRINCIPAL_MISMATCH)
                except LandingReservationSessionMismatch:
                    return _refuse(LandingReservationRefusalCode.SESSION_MISMATCH)
                except LandingReservationAuthorityEpochMismatch:
                    return _refuse(
                        LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH
                    )
                except LandingReservationFenceMismatch:
                    return _refuse(LandingReservationRefusalCode.FENCE_MISMATCH)
                except LandingReservationStale:
                    return _refuse(LandingReservationRefusalCode.EPOCH_MISMATCH)
                except LandingReservationError:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_INVALID)

                state, state_code = self._read_snapshot(
                    self._authority, repository, request
                )
                if state_code is not None:
                    return _refuse(state_code)
                if state is None:
                    return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
                state_check = self._check_state(request, repository, state)
                if state_check is not None:
                    return _refuse(state_check)

                barrier_call = _safe_method(self._authority, "validate_landing_barrier")
                if not callable(barrier_call):
                    return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
                try:
                    barrier_value, barrier_code = _authority_call(
                        barrier_call,
                        lease,
                        controller,
                        repository,
                        state,
                        unavailable=LandingReservationRefusalCode.RESERVATION_LOST,
                    )
                except LandingReservationStale:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                except LandingReservationUnavailable:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                if barrier_code is not None:
                    return _refuse(barrier_code)
                try:
                    barrier = self._validate_barrier(
                        barrier_value,
                        request,
                        controller,
                        repository,
                        request_digest,
                        state,
                    )
                except LandingReservationOwnerMismatch:
                    return _refuse(LandingReservationRefusalCode.OWNER_MISMATCH)
                except LandingReservationTenantMismatch:
                    return _refuse(LandingReservationRefusalCode.TENANT_MISMATCH)
                except LandingReservationPrincipalMismatch:
                    return _refuse(LandingReservationRefusalCode.PRINCIPAL_MISMATCH)
                except LandingReservationSessionMismatch:
                    return _refuse(LandingReservationRefusalCode.SESSION_MISMATCH)
                except LandingReservationAuthorityEpochMismatch:
                    return _refuse(
                        LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH
                    )
                except LandingReservationFenceMismatch:
                    return _refuse(LandingReservationRefusalCode.FENCE_MISMATCH)
                except LandingReservationStale:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                except LandingReservationError:
                    return _refuse(LandingReservationRefusalCode.RESERVATION_LOST)
                final_check = self._check_state(request, repository, barrier.snapshot)
                if final_check is not None:
                    return _refuse(final_check)
                snapshot = self._snapshot_result(
                    request, repository, controller, request_digest, barrier
                )
                return LandingReservationResult(
                    accepted=True,
                    detail="landing reservation acquired and final state barrier passed",
                    snapshot=snapshot,
                )
        except TrustedReservationRuntimeError:
            raise
        except (LandingReservationUnavailable, BlockedByLease):
            return _refuse(LandingReservationRefusalCode.LEASE_UNAVAILABLE)
        except (ValueError, TypeError, OSError):
            return _refuse(LandingReservationRefusalCode.SOURCE_UNAVAILABLE)


def reserve_landing(
    request: LandingReservationRequest,
    *,
    authority: LandingReservationAuthority,
) -> LandingReservationResult:
    """Functional adapter for the controller's reservation operation."""

    return LandingReservationController(authority).reserve(request)


__all__ = [
    "CanonicalObservation",
    "CanonicalState",
    "CertificationObservation",
    "ControllerIdentity",
    "DurableLandingReservation",
    "LandingReservationAuthority",
    "LandingReservationConflict",
    "LandingReservationController",
    "LandingReservationError",
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
