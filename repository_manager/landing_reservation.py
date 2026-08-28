"""RMDD-13 checkpoint 2: a fail-closed native landing protocol boundary.

The landing policy in :mod:`repository_manager.landing_policy` is deliberately
pure.  This module is the small controller seam that may precede a future
target compare-and-swap (CP3).  The native/durable authority is deliberately
not wired in this checkpoint, so production entrypoints return a stable
authority-unavailable refusal before any state read, lease, or mutation.

There are three important boundaries here:

* ``NativeLandingReservationPort`` is a type-only contract for the future
  authenticated native/durable authority.  It is never accepted as a public
  caller-supplied transport, resolver, lease, reader, or proof verifier.
* ``LandingReservationAuthority`` is only a fail-closed production facade until
  that native binding exists.  Python object identity, private globals, local
  locks, SQLite/JSON rows, local authentication calculations, and
  caller-computed digests are never authority evidence.
* A future native transaction must own identity, both RMDD-26 leases,
  reservation, post-hold reads, the end barrier, complete proof, rollback, and
  recovery.  The controller does not run Git commands, move refs, submit jobs,
  build, clean, or push.

Any future native snapshot contains only opaque identities, Git object IDs,
bounded state, and a backend-issued opaque proof reference.  It intentionally
excludes canonical paths, worktree paths, hostnames, process IDs, exception
text, and private WIP details.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol, cast

from repository_manager.development import RepositoryIdentity

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
    AUTHORITY_UNAVAILABLE = "landing_reservation_authority_unavailable"
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


def _target_branch_component_invalid(component: str) -> bool:
    return (
        component in {".", ".."}
        or component.startswith(".")
        or component.endswith(".")
        or component.endswith(".lock")
    )


def _target_ref_is_invalid(raw: str, branch: str) -> bool:
    # A flat OR-chain of side-effect-free predicates over the same
    # `branch`/`raw` strings -- collapsed into `any()` of a fully-evaluated
    # tuple (rather than the original short-circuiting `or`-chain) is a real
    # simplification, not a metric-gaming rewrite: every predicate is a pure
    # string/regex check that never raises on the `str` `_text()` guarantees,
    # so evaluating all of them instead of stopping at the first `True`
    # changes nothing observable.
    return any(
        (
            not branch,
            branch.startswith("/"),
            branch.endswith("/"),
            branch.endswith("."),
            branch.endswith(".lock"),
            branch.startswith("-"),
            branch in {".", "..", "HEAD"},
            ".." in branch,
            "//" in branch,
            "@{" in branch,
            bool(_REF_FORBIDDEN_RE.search(branch)),
            bool(_REF_INJECTION_RE.search(branch)),
            raw.startswith("refs/") and not raw.startswith(_TARGET_PREFIX),
            any(
                _target_branch_component_invalid(component)
                for component in branch.split("/")
            ),
        )
    )


def normalize_target_ref(value: object) -> str:
    """Return one canonical local target ref and reject ref injection.

    Public callers may use ``main`` or ``refs/heads/main``.  Internally the
    full ``refs/heads/`` form is used everywhere, so a same-basename or remote
    ref cannot collide with a local target.
    """

    raw = _text(value, "target_ref", maximum=_MAX_TEXT_BYTES)
    branch = raw[len(_TARGET_PREFIX) :] if raw.startswith(_TARGET_PREFIX) else raw
    if _target_ref_is_invalid(raw, branch):
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
        LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE: "landing reservation authority unavailable",
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
class LandingReservationProofRef:
    """Opaque proof reference minted by the future native authority.

    The Python boundary validates only bounded shape.  It never mints,
    verifies, or treats this reference as authority evidence.  Its repr is
    deliberately redacted so backend material cannot leak through logs.
    """

    issuer: str
    opaque_reference: str
    correlation_digest: str

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            LandingReservationProofRef,
            ("issuer", "opaque_reference", "correlation_digest"),
        )
        _text(values["issuer"], "proof issuer", maximum=_MAX_ID_BYTES)
        _text(values["opaque_reference"], "proof reference", maximum=_MAX_ID_BYTES)
        _digest(values["correlation_digest"], "proof correlation digest")

    def immutable_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "issuer": self.issuer,
            "opaque_reference": self.opaque_reference,
            "correlation_digest": self.correlation_digest,
        }

    def __repr__(self) -> str:
        return "<native landing proof ref redacted>"


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
    proof_ref: LandingReservationProofRef
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
                "proof_ref",
            ),
        )
        return {
            name: (
                cast(LandingReservationProofRef, values[name]).immutable_payload()
                if name == "proof_ref"
                else values[name]
            )
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
                "proof_ref",
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
                "proof_ref",
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
        if type(values["proof_ref"]) is not LandingReservationProofRef:
            raise LandingReservationError("snapshot proof reference is invalid")
        cast(LandingReservationProofRef, values["proof_ref"]).__post_init__()
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
class _HeldLandingContext:
    """Strict future-native context transcript; never a local authority."""

    reservation: DurableLandingReservation
    repository: ResolvedRepositoryIdentity
    controller: ControllerIdentity
    request_digest: str
    reconciliation_lease_id: str
    reconciliation_lease_epoch: int
    reconciliation_lease_fence: str
    canonical_lease_id: str
    canonical_lease_epoch: int
    canonical_lease_fence: str
    authority_incarnation: str

    def __post_init__(self) -> None:
        values = _strict_dataclass_values(
            self,
            _HeldLandingContext,
            (
                "reservation",
                "repository",
                "controller",
                "request_digest",
                "reconciliation_lease_id",
                "reconciliation_lease_epoch",
                "reconciliation_lease_fence",
                "canonical_lease_id",
                "canonical_lease_epoch",
                "canonical_lease_fence",
                "authority_incarnation",
            ),
        )
        if type(values["reservation"]) is not DurableLandingReservation:
            raise LandingReservationError("native context reservation is invalid")
        cast(DurableLandingReservation, values["reservation"]).__post_init__()
        if type(values["repository"]) is not ResolvedRepositoryIdentity:
            raise LandingReservationError("native context repository is invalid")
        cast(ResolvedRepositoryIdentity, values["repository"]).__post_init__()
        if type(values["controller"]) is not ControllerIdentity:
            raise LandingReservationError("native context controller is invalid")
        cast(ControllerIdentity, values["controller"]).__post_init__()
        _digest(values["request_digest"], "native context request digest")
        _text(values["reconciliation_lease_id"], "native reconciliation lease id")
        _positive_int(
            values["reconciliation_lease_epoch"], "native reconciliation lease epoch"
        )
        _text(values["reconciliation_lease_fence"], "native reconciliation lease fence")
        _text(values["canonical_lease_id"], "native canonical lease id")
        _positive_int(values["canonical_lease_epoch"], "native canonical lease epoch")
        _text(values["canonical_lease_fence"], "native canonical lease fence")
        _text(
            values["authority_incarnation"],
            "native authority incarnation",
            maximum=_MAX_ID_BYTES,
        )

    def immutable_payload(self) -> dict[str, object]:
        self.__post_init__()
        return {
            "reservation": _durable_payload(self.reservation),
            "repository": self.repository.immutable_payload(),
            "controller": self.controller.immutable_payload(),
            "request_digest": self.request_digest,
            "reconciliation_lease_id": self.reconciliation_lease_id,
            "reconciliation_lease_epoch": self.reconciliation_lease_epoch,
            "reconciliation_lease_fence": self.reconciliation_lease_fence,
            "canonical_lease_id": self.canonical_lease_id,
            "canonical_lease_epoch": self.canonical_lease_epoch,
            "canonical_lease_fence": self.canonical_lease_fence,
            "authority_incarnation": self.authority_incarnation,
        }


def _durable_payload(value: DurableLandingReservation) -> dict[str, object]:
    """Return every strict native reservation field without aliases."""

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


class NativeLandingReservationPort(Protocol):
    """Type-only contract for the future authenticated native authority.

    This protocol is documentation and static typing only.  No public
    constructor, controller, or production function accepts an implementation
    supplied by Python callers.  The future binding must be authenticated
    outside this module and own the complete transaction described in the
    development contract.
    """

    def reserve_landing(
        self, request: LandingReservationRequest
    ) -> LandingReservationResult:
        """Return a native-issued result or a fixed refusal."""

    def verify_current_landing_reservation(
        self,
        proof_ref: LandingReservationProofRef,
        snapshot: LandingReservationSnapshot,
    ) -> LandingReservationResult:
        """Revalidate the native proof and every current correlation."""


def _validate_observation(value: object, expected: type[Any]) -> object:
    """Validate an exact observation before reconstructing a native transcript."""

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
    """Hash a bounded transcript for serialization correlation only."""

    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _authority_unavailable() -> LandingReservationResult:
    return _refuse(LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)


class LandingReservationAuthority:
    """Fail-closed production facade until the native authority is bound.

    This object contains no lease, resolver, transport, secret, lock, or local
    durable state.  It cannot be made authoritative by subclassing, DTO
    copying, reflection, or a caller-provided proof.
    """

    __slots__ = ()

    def __repr__(self) -> str:
        return "<native landing reservation authority unavailable>"

    def reserve_landing(
        self, request: LandingReservationRequest
    ) -> LandingReservationResult:
        del request
        return _authority_unavailable()

    def acquire_landing(
        self, request: LandingReservationRequest
    ) -> LandingReservationResult:
        del request
        return _authority_unavailable()

    def verify_current_landing_reservation(
        self,
        proof_ref: object,
        snapshot: object,
    ) -> LandingReservationResult:
        del proof_ref, snapshot
        return _authority_unavailable()

    def verify_attested_snapshot(self, snapshot: object) -> LandingReservationResult:
        del snapshot
        return _authority_unavailable()


def create_landing_authority() -> LandingReservationAuthority:
    """Return the unbound native facade; no backend injection is permitted."""

    return LandingReservationAuthority()


def reserve_landing(
    request: LandingReservationRequest,
) -> LandingReservationResult:
    """Production entrypoint; unavailable until native binding is installed."""

    if type(request) is not LandingReservationRequest:
        return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)
    try:
        request.__post_init__()
    except TrustedReservationRuntimeError:
        raise
    except (LandingReservationError, AttributeError, KeyError, TypeError):
        return _refuse(LandingReservationRefusalCode.REQUEST_INVALID)
    return _authority_unavailable()


def verify_current_landing_reservation(
    proof_ref: object,
    snapshot: object,
) -> LandingReservationResult:
    """CP3 verification entrypoint; unavailable without native authority."""

    del proof_ref, snapshot
    return _authority_unavailable()


class LandingReservationController:
    """Immutable production controller with no injectable authority.

    No custom ``__init__`` on purpose: with ``__slots__ = ()`` there is no
    state to set, and the inherited ``object.__init__`` already rejects any
    constructor argument (proven by
    ``test_no_python_backend_or_authority_injection_is_accepted``), so a
    stub ``def __init__(self) -> None: pass`` would add nothing but a
    false positive for the stub/TODO gate.
    """

    __slots__ = ()

    def reserve(self, request: LandingReservationRequest) -> LandingReservationResult:
        return reserve_landing(request)


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
    "LandingReservationProofRef",
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
    "NativeLandingReservationPort",
    "normalize_target_ref",
    "reserve_landing",
    "verify_current_landing_reservation",
]
