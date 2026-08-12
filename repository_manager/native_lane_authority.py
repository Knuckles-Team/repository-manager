"""Production native ``LaneAuthority`` binding for RMDD-28.

RMDD-09's :mod:`repository_manager.lane_registry` defines the pure/local
policy this module must satisfy in production:

* :class:`~repository_manager.lane_registry.LaneAuthority` (the ``Protocol``
  ``LaneRegistry``/``NativeLaneAuthorityAdapter`` consume) -- ``allocate``,
  ``get``, ``list_records``, ``transition``, ``heartbeat``;
* :class:`~repository_manager.lane_registry.NativeLaneAuthorityAdapter`,
  which requires a ``native`` object exposing ``allocate_lane``, ``get_lane``,
  ``list_lanes``, ``transition_lane``, ``heartbeat_lane``.

This module supplies that ``native`` object, :class:`NativeLaneAuthority`,
bound to the RMDD-28 engine-native ``DevelopmentLaneHold`` protocol via AU's
narrow transport (``agent_utilities.orchestration.development_lane``). It is
deliberately narrower than RMDD-08's 1700-line
:mod:`repository_manager.native_reservations` -- that module is this one's
closest sibling and the template this follows for shape (no-fallback
constructor, strict field mapping, decision-driven result parsing) -- but
this v1 binding does not yet re-verify every immutable admission field on
every response the way that module does; see the module docstring residual
note near the bottom of this file and the RMDD-28 handoff for what is
intentionally deferred.

No fallback: :func:`NativeLaneAuthority.__init__` and
:func:`create_production_lane_registry` never construct
``FakeDurableLaneAuthority`` or any other local/SQLite/process-lock stand-in.
If the injected transport does not expose all eight native
``DevelopmentLane*`` operations, construction raises
:class:`NativeLaneAuthorityUnavailable` and callers must not catch it and
substitute a local approximation.

One WorkItem authority: the lane's ``lane.lifecycle`` WorkItem identity is
derived deterministically from ``lane_id`` (never a second generated id or a
parallel ledger), and every mutation re-reads its current attempt/lease/
fence from the durable WorkItem/hold authority rather than caching it
locally. The only "local" state this module ever touches is the read-only
best-effort SQLite projection RMDD-09's own ``LaneRegistry`` already owns
(``LaneRegistry._project``); this module writes nothing to it directly.
"""

from __future__ import annotations

import uuid
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

from repository_manager.lane_quota import LaneQuotaDecision
from repository_manager.lane_record import (
    LaneLifecycleState,
    LaneRecord,
)
from repository_manager.lane_registry import (
    LaneConflictError,
    LaneQuotaError,
    LaneRegistryError,
    LaneTransitionError,
    StaleLaneFence,
)

_WORK_ITEM_NAMESPACE = uuid.UUID("2f6a8f7e-9c3a-4d6a-8e6f-b1e6a7c1a1c8")
_LANE_LIFECYCLE_KIND = "repository.lane.lifecycle"
_LANE_INTENT_METADATA_KEY = "development_lane_intent"
_MAX_U64 = (1 << 64) - 1


class NativeLaneAuthorityUnavailable(RuntimeError):
    """The connected engine does not advertise the native lane-hold surface.

    Raised at construction (missing/incomplete transport) and at call time
    (a required correlation could not be established). Never caught and
    replaced by a local record -- callers must refuse, not degrade.
    """

    code = "native_development_lane_authority_unavailable"


class NativeLaneProtocolError(ValueError):
    """A native response failed the versioned RMDD-28 lane-hold contract."""


class DevelopmentLaneTransport(Protocol):
    """Narrow structural surface this adapter consumes.

    Matches ``agent_utilities.orchestration.development_lane.
    EngineNativeDevelopmentLaneTransport`` method-for-method so a focused
    test can inject a deterministic fake without importing the engine
    package, exactly as RMDD-08's ``NativeReservationClient`` does for
    ``native_reservations.py``.
    """

    def reserve(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def renew(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def observe(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def finish(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def cleanup_complete(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def query(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def status(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...
    def update_quota(self, request: Mapping[str, Any]) -> Mapping[str, Any]: ...


_REQUIRED_TRANSPORT_METHODS = (
    "reserve",
    "renew",
    "observe",
    "finish",
    "cleanup_complete",
    "query",
    "status",
    "update_quota",
)


class WorkItemLifecycleClaimer(Protocol):
    """Claims/reads the ``lane.lifecycle`` WorkItem correlation for one lane.

    Kept as an injected seam (mirroring RMDD-08's ``NativeFenceCodec``) so a
    focused test can supply a deterministic fake. :class:`AgentUtilitiesWorkItemClaimer`
    is the real production implementation, built on
    ``agent_utilities.orchestration.work_item``.
    """

    def claim_lifecycle(
        self,
        *,
        lane_id: str,
        request_key: str,
        repository_id: str,
        owner_id: str,
        session_id: str,
        tenant_ref: str,
        lane_intent: Mapping[str, Any],
        now: datetime,
    ) -> Mapping[str, Any]:
        """Return ``{work_item_id, owner_id, attempt, lease_epoch, fencing_token}``."""


def lane_work_item_id(lane_id: str) -> str:
    """Deterministic ``lane.lifecycle`` WorkItem id for one lane.

    Never a second generated identity: the same ``lane_id`` always resolves
    to the same WorkItem id, so this module never maintains its own
    lane-id -> work-item-id table.
    """

    return f"workitem:repository_manager:{uuid.uuid5(_WORK_ITEM_NAMESPACE, lane_id)}"


class AgentUtilitiesWorkItemClaimer:
    """Production :class:`WorkItemLifecycleClaimer` over the real WorkItem authority."""

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def claim_lifecycle(
        self,
        *,
        lane_id: str,
        request_key: str,
        repository_id: str,
        owner_id: str,
        session_id: str,
        tenant_ref: str,
        lane_intent: Mapping[str, Any],
        now: datetime,
    ) -> Mapping[str, Any]:
        from agent_utilities.orchestration.work_item import (
            claim_specific,
            submit_work_item_atomic,
        )

        work_item_id = lane_work_item_id(lane_id)
        submit_work_item_atomic(
            self._engine,
            work_item_id=work_item_id,
            kind=_LANE_LIFECYCLE_KIND,
            tenant=tenant_ref,
            payload_ref=lane_id,
            idempotency_key=f"{repository_id}:{request_key}",
            created_by=owner_id,
            metadata={_LANE_INTENT_METADATA_KEY: dict(lane_intent)},
            create_if_absent=True,
            now=now.timestamp(),
        )
        claimed = claim_specific(self._engine, work_item_id, now=now.timestamp())
        if claimed is None:
            raise NativeLaneProtocolError(
                "lane.lifecycle WorkItem could not be claimed for allocation"
            )
        return claimed


def _string(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise NativeLaneProtocolError(f"{name} must be a non-blank string")
    if len(value) > 256:
        raise NativeLaneProtocolError(f"{name} exceeds the 256-character bound")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > _MAX_U64
    ):
        raise NativeLaneProtocolError(
            f"{name} must be an integer in [{minimum}, {_MAX_U64}]"
        )
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise NativeLaneProtocolError(f"{name} must be a boolean")
    return value


def _mapping(value: object, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise NativeLaneProtocolError(f"{name} must be an object")
    return cast(Mapping[str, Any], value)


_ACCEPTED = {"accepted", "idempotent"}
_HOLD_STATE_MAP = {
    "allocating": LaneLifecycleState.ALLOCATING,
    "active": LaneLifecycleState.ACTIVE,
    "submitted": LaneLifecycleState.SUBMITTED,
    "released": LaneLifecycleState.LANDED,
    "expired": LaneLifecycleState.EXPIRED,
    "cleanup_pending": LaneLifecycleState.QUARANTINED,
    "cleaned": LaneLifecycleState.LANDED,
    "aborted": LaneLifecycleState.ABORTED,
}
_LOCAL_TO_TERMINAL = {
    LaneLifecycleState.LANDED: "succeeded",
    LaneLifecycleState.ABORTED: "failed",
    LaneLifecycleState.REJECTED: "cancelled",
    LaneLifecycleState.EXPIRED: "failed",
    LaneLifecycleState.QUARANTINED: "failed",
}


class NativeLaneAuthority:
    """Bind RMDD-09's ``native`` protocol to the RMDD-28 native transaction."""

    def __init__(
        self,
        transport: DevelopmentLaneTransport,
        claimer: WorkItemLifecycleClaimer | None,
        *,
        tenant_ref: str,
        host_ref: str,
        managed_root: str | Path = "/var/lib/repository-manager/lanes",
        quota_policy_name: str = "default",
        quota_policy_version: str = "1",
        clock: Callable[[], datetime] | Any | None = None,
    ) -> None:
        missing = [
            name
            for name in _REQUIRED_TRANSPORT_METHODS
            if not callable(getattr(transport, name, None))
        ]
        if missing:
            raise NativeLaneAuthorityUnavailable(NativeLaneAuthorityUnavailable.code)
        if claimer is None or not callable(getattr(claimer, "claim_lifecycle", None)):
            raise NativeLaneAuthorityUnavailable(NativeLaneAuthorityUnavailable.code)
        self._transport = transport
        self._claimer = claimer
        self._tenant_ref = _string(tenant_ref, name="tenant_ref")
        self._host_ref = _string(host_ref, name="host_ref")
        # The native hold intentionally never carries an absolute local path
        # (privacy contract; ``repository_id`` is an opaque hash and
        # ``worktree_locator`` is managed-root-relative). ``allocate_lane``
        # returns the operationally real path taken directly from the
        # caller's request. Every other read (get/list/transition/heartbeat)
        # has no such request to draw from, so it synthesizes a canonical,
        # deterministic path under ``managed_root`` -- stable identity
        # correlation, not the true Git checkout path. RMDD-09's own
        # best-effort SQLite projection (LaneRegistry._project, never an
        # authorization path) retains the real path from the original
        # allocate() call; this is a disclosed v1 residual, not a silent gap.
        self._managed_root = Path(managed_root)
        self._quota_policy_name = _string(quota_policy_name, name="quota_policy_name")
        self._quota_policy_version = _string(
            quota_policy_version, name="quota_policy_version"
        )
        self._clock = self._normalize_clock(clock)

    @classmethod
    def from_graph_client(
        cls,
        graph_client: Any,
        engine: Any,
        *,
        tenant_ref: str,
        host_ref: str,
        quota_policy_name: str = "default",
        quota_policy_version: str = "1",
        clock: Callable[[], datetime] | Any | None = None,
    ) -> NativeLaneAuthority:
        """Construct entirely from an injected native graph client + WorkItem engine.

        No path here constructs ``FakeDurableLaneAuthority`` or any other
        local approximation; a missing/incomplete ``development_lanes``
        namespace on ``graph_client`` fails closed via
        :class:`NativeLaneAuthorityUnavailable`.
        """

        from agent_utilities.orchestration.development_lane import (
            EngineNativeDevelopmentLaneTransport,
        )

        transport = EngineNativeDevelopmentLaneTransport(graph_client)
        claimer = AgentUtilitiesWorkItemClaimer(engine)
        return cls(
            transport,
            claimer,
            tenant_ref=tenant_ref,
            host_ref=host_ref,
            quota_policy_name=quota_policy_name,
            quota_policy_version=quota_policy_version,
            clock=clock,
        )

    @staticmethod
    def _normalize_clock(
        clock: Callable[[], datetime] | Any | None,
    ) -> Callable[[], datetime]:
        """Accept a zero-arg callable or a ``.now()``-style object.

        Matches ``LaneRegistry``/``FakeDurableLaneAuthority``'s own clock
        convention exactly, so the same test ``Clock`` double works
        unmodified against both the pure and native authorities.
        """

        if clock is None:
            return lambda: datetime.now(UTC)
        if callable(clock):
            return clock
        now_method = getattr(clock, "now", None)
        if callable(now_method):
            return now_method
        raise TypeError("clock must be callable or provide now()")

    def _now_ms(self, value: datetime | None = None) -> int:
        timestamp = value or self._clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise NativeLaneProtocolError("native clock must be timezone-aware")
        return int(timestamp.astimezone(UTC).timestamp() * 1000)

    def _call(self, method: str, request: Mapping[str, Any]) -> Mapping[str, Any]:
        function = getattr(self._transport, method, None)
        if not callable(function):
            raise NativeLaneAuthorityUnavailable(NativeLaneAuthorityUnavailable.code)
        try:
            result = function(request)
        except NativeLaneAuthorityUnavailable:
            raise
        except Exception as exc:
            if getattr(exc, "code", None) == NativeLaneAuthorityUnavailable.code:
                raise NativeLaneAuthorityUnavailable(
                    NativeLaneAuthorityUnavailable.code
                ) from exc
            raise
        return _mapping(result, name=f"native {method} result")

    # -- request builders ---------------------------------------------

    def _intent(
        self, request: Mapping[str, Any], *, resource_reservation_id: str
    ) -> dict[str, Any]:
        return {
            "schema_version": "1",
            "tenant_ref": self._tenant_ref,
            "request_id": _string(request["request_key"], name="request_key"),
            "lane_id": _string(request["lane_id"], name="lane_id"),
            "repository_id": _string(request["repository_id"], name="repository_id"),
            "base_ref": _string(request["base_ref"], name="base_ref"),
            "base_sha": _string(request.get("base_sha") or "unknown", name="base_sha"),
            "branch": _string(request["branch"], name="branch"),
            "host_target_kind": "local",
            "host_target_alias": None,
            "host_ref": self._host_ref,
            "resource_reservation_id": resource_reservation_id,
            "workspace_ref": _string(request["repository_id"], name="repository_id"),
            "worktree_locator": _string(request["worktree_path"], name="worktree_path"),
            "owner_id": _string(request["owner_id"], name="owner_id"),
            "session_id": _string(request["session_id"], name="session_id"),
            "fairness_group": "default",
            "quota_policy_name": self._quota_policy_name,
            "quota_policy_version": self._quota_policy_version,
            "predicted_disk_bytes": _integer(
                request["predicted_disk_bytes"], name="predicted_disk_bytes"
            ),
            "ttl_ms": _integer(request["ttl_seconds"], name="ttl_seconds") * 1000,
            "input_fingerprint": "v1:"
            + _string(
                request["input_digest"], name="input_digest", allow_empty=True
            ).rjust(64, "0")[:64],
        }

    # -- response mapping ------------------------------------------------

    def _record_from_hold(
        self, hold: Mapping[str, Any], *, request: Mapping[str, Any] | None = None
    ) -> LaneRecord:
        state_value = _string(hold.get("state"), name="hold.state")
        state = _HOLD_STATE_MAP.get(state_value)
        if state is None:
            raise NativeLaneProtocolError("native hold state is unknown")
        expires_ms = _integer(hold.get("expires_at_ms"), name="hold.expires_at_ms")
        renewed_ms = _integer(
            hold.get("last_renewed_at_ms"), name="hold.last_renewed_at_ms"
        )
        try:
            expires_at = datetime.fromtimestamp(expires_ms / 1000, UTC)
            heartbeat_at = datetime.fromtimestamp(renewed_ms / 1000, UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise NativeLaneProtocolError("native hold timestamp is invalid") from exc
        ttl_seconds = max(1, int((expires_ms - renewed_ms) / 1000))
        repository_path = (
            request["repository_path"]
            if request is not None
            else str(
                self._managed_root
                / "repo"
                / _string(hold.get("repository_id"), name="hold.repository_id")
            )
        )
        worktree_path = (
            request["worktree_path"]
            if request is not None
            else str(
                self._managed_root
                / _string(hold.get("worktree_locator"), name="hold.worktree_locator")
            )
        )
        created_ms = renewed_ms
        return LaneRecord(
            lane_id=_string(hold.get("lane_id"), name="hold.lane_id"),
            request_key=_string(hold.get("request_id"), name="hold.request_id"),
            input_digest=_string(
                hold.get("input_fingerprint"), name="hold.input_fingerprint"
            ),
            repository_id=_string(hold.get("repository_id"), name="hold.repository_id"),
            repository_path=str(repository_path),
            branch=_string(hold.get("branch"), name="hold.branch"),
            base_ref=_string(hold.get("base_ref"), name="hold.base_ref"),
            base_sha=_string(
                hold.get("base_sha"), name="hold.base_sha", allow_empty=True
            )
            or "",
            worktree_path=str(worktree_path),
            owner_id=_string(hold.get("owner_id"), name="hold.owner_id"),
            session_id=_string(hold.get("session_id"), name="hold.session_id"),
            host_id=self._host_ref,
            # The native hold_id is assigned once by the reserve transaction
            # and is unchanged by an idempotent replay (unlike the caller's
            # freshly generated per-call ``request["fence"]``, which
            # LaneRegistry.allocate() always mints even on a replay path) --
            # using it uniformly here is what makes replay return the exact
            # original record, matching FakeDurableLaneAuthority.
            fence=_string(hold.get("hold_id"), name="hold.hold_id"),
            attempt=_integer(hold.get("attempt"), name="hold.attempt", minimum=1),
            created_at=datetime.fromtimestamp(created_ms / 1000, UTC),
            heartbeat_at=heartbeat_at,
            expires_at=expires_at,
            ttl_seconds=ttl_seconds,
            predicted_disk_bytes=_integer(
                hold.get("predicted_disk_bytes"), name="hold.predicted_disk_bytes"
            ),
            observed_disk_bytes=_integer(
                hold.get("observed_disk_bytes"), name="hold.observed_disk_bytes"
            ),
            disk_budget_bytes=max(
                1,
                _integer(
                    hold.get("predicted_disk_bytes"), name="hold.predicted_disk_bytes"
                ),
            ),
            state=state,
            version=_integer(
                hold.get("hold_revision"), name="hold.hold_revision", minimum=1
            )
            or 1,
            last_transition=state_value,
        )

    # -- LaneAuthority "native" surface ----------------------------------

    def allocate_lane(self, request: Mapping[str, Any], *, now: datetime) -> LaneRecord:
        lane_id = _string(request["lane_id"], name="lane_id")
        # Idempotent replay: if a hold already exists for this exact lane_id,
        # the native reserve call below is itself idempotent on unchanged
        # input (RMDD-28 "Atomic allocation contract" #7/idempotency), so no
        # separate local pre-check is needed or wanted here.
        preliminary_intent = self._intent(request, resource_reservation_id="")
        claim = self._claimer.claim_lifecycle(
            lane_id=lane_id,
            request_key=_string(request["request_key"], name="request_key"),
            repository_id=_string(request["repository_id"], name="repository_id"),
            owner_id=_string(request["owner_id"], name="owner_id"),
            session_id=_string(request["session_id"], name="session_id"),
            tenant_ref=self._tenant_ref,
            lane_intent=preliminary_intent,
            now=now,
        )
        work_item_id = _string(claim.get("work_item_id"), name="claim.work_item_id")
        owner_id = _string(
            claim.get("lease_owner") or claim.get("owner_id"), name="claim.owner"
        )
        attempt = _integer(claim.get("attempt"), name="claim.attempt", minimum=1)
        lease_epoch = _integer(claim.get("lease_epoch"), name="claim.lease_epoch")
        fencing_token = _integer(claim.get("fencing_token"), name="claim.fencing_token")
        intent = self._intent(request, resource_reservation_id=work_item_id)
        wire = {
            "schema_version": "1",
            "tenant_ref": self._tenant_ref,
            "work_item_id": work_item_id,
            "owner_id": owner_id,
            "attempt": attempt,
            "lease_epoch": lease_epoch,
            "fencing_token": fencing_token,
            "work_item_fence": str(fencing_token),
            "intent": intent,
            "idempotency_key": _string(request["request_key"], name="request_key"),
            "now_ms": self._now_ms(now),
        }
        result = self._call("reserve", wire)
        decision = _string(result.get("decision"), name="result.decision")
        if decision == "quota":
            raise LaneQuotaError(
                LaneQuotaDecision(
                    admitted=False, scope="native", reason="native quota refused"
                )
            )
        if decision == "exclusivity" or decision == "conflict":
            raise LaneConflictError("repository branch or worktree is already reserved")
        if decision == "input_conflict":
            raise LaneConflictError(
                "idempotency key was reused with changed immutable lane input"
            )
        if decision not in _ACCEPTED:
            raise NativeLaneAuthorityUnavailable(
                f"native lane reserve refused: {decision}"
            )
        hold = result.get("hold")
        if hold is None:
            raise NativeLaneProtocolError("accepted native reserve lacks a hold")
        return self._record_from_hold(
            _mapping(hold, name="result.hold"), request=request
        )

    def get_lane(self, lane_id: str) -> LaneRecord | None:
        result = self._call(
            "status",
            {
                "schema_version": "1",
                "tenant_ref": self._tenant_ref,
                "hold_id": None,
                "lane_id": _string(lane_id, name="lane_id"),
                "work_item_id": None,
                "limit": 1,
                "cursor": None,
                "now_ms": self._now_ms(),
            },
        )
        holds = result.get("holds")
        if not isinstance(holds, list) or not holds:
            return None
        return self._record_from_hold(_mapping(holds[0], name="status.holds[0]"))

    def list_lanes(self) -> tuple[LaneRecord, ...]:
        records: list[LaneRecord] = []
        cursor: str | None = None
        for _ in range(1000):
            result = self._call(
                "status",
                {
                    "schema_version": "1",
                    "tenant_ref": self._tenant_ref,
                    "hold_id": None,
                    "lane_id": None,
                    "work_item_id": None,
                    "limit": 200,
                    "cursor": cursor,
                    "now_ms": self._now_ms(),
                },
            )
            holds = result.get("holds")
            if isinstance(holds, list):
                records.extend(
                    self._record_from_hold(_mapping(item, name="status.holds[]"))
                    for item in holds
                )
            if _boolean(result.get("complete"), name="status.complete"):
                break
            next_cursor = result.get("next_cursor")
            if not isinstance(next_cursor, str) or next_cursor == cursor:
                break
            cursor = next_cursor
        return tuple(sorted(records, key=lambda item: (item.created_at, item.lane_id)))

    def _current_hold(self, lane_id: str) -> Mapping[str, Any]:
        result = self._call(
            "status",
            {
                "schema_version": "1",
                "tenant_ref": self._tenant_ref,
                "hold_id": None,
                "lane_id": _string(lane_id, name="lane_id"),
                "work_item_id": None,
                "limit": 1,
                "cursor": None,
                "now_ms": self._now_ms(),
            },
        )
        holds = result.get("holds")
        if not isinstance(holds, list) or not holds:
            raise LaneRegistryError(f"unknown lane: {lane_id}")
        return _mapping(holds[0], name="status.holds[0]")

    def _authorize_hold(
        self, hold: Mapping[str, Any], *, owner_id: str | None, fence: str | None
    ) -> None:
        hold_id = _string(hold.get("hold_id"), name="hold.hold_id")
        hold_owner = _string(hold.get("owner_id"), name="hold.owner_id")
        if owner_id != hold_owner or fence != hold_id:
            raise StaleLaneFence("lane mutation refused: owner or fence is not current")

    def transition_lane(
        self,
        lane_id: str,
        operation: str,
        *,
        owner_id: str | None,
        fence: str | None,
        target: LaneLifecycleState | str | None = None,
        updates: Mapping[str, Any] | None = None,
        now: datetime,
    ) -> LaneRecord:
        hold = self._current_hold(lane_id)
        self._authorize_hold(hold, owner_id=owner_id, fence=fence)
        work_item_id = _string(hold.get("work_item_id"), name="hold.work_item_id")
        attempt = _integer(hold.get("attempt"), name="hold.attempt", minimum=1)
        lease_epoch = _integer(hold.get("lease_epoch"), name="hold.lease_epoch")
        fencing_token = _integer(hold.get("fencing_token"), name="hold.fencing_token")
        hold_id = _string(hold.get("hold_id"), name="hold.hold_id")
        hold_revision = _integer(
            hold.get("hold_revision"), name="hold.hold_revision", minimum=0
        )
        resolved_target = (
            LaneLifecycleState(target)
            if target is not None
            else {
                "activate": LaneLifecycleState.ACTIVE,
                "submit": LaneLifecycleState.SUBMITTED,
                "finish": LaneLifecycleState.LANDED,
                "abort": LaneLifecycleState.ABORTED,
                "expire": LaneLifecycleState.EXPIRED,
                "quarantine": LaneLifecycleState.QUARANTINED,
            }.get(operation)
        )
        if resolved_target is None:
            raise LaneTransitionError(f"unknown lane operation: {operation}")
        base = {
            "schema_version": "1",
            "tenant_ref": self._tenant_ref,
            "work_item_id": work_item_id,
            "owner_id": owner_id,
            "attempt": attempt,
            "lease_epoch": lease_epoch,
            "fencing_token": fencing_token,
            "work_item_fence": str(fencing_token),
            "hold_id": hold_id,
            "expected_hold_revision": hold_revision,
            "idempotency_key": f"{operation}:{hold_id}:{hold_revision}",
            "now_ms": self._now_ms(now),
        }
        if resolved_target in _LOCAL_TO_TERMINAL:
            base["terminal_state"] = _LOCAL_TO_TERMINAL[resolved_target]
            result = self._call("finish", base)
        elif resolved_target == LaneLifecycleState.ACTIVE:
            # ALLOCATING -> ACTIVE has no dedicated native verb; renew keeps
            # the hold current without inventing a synthetic finish/reserve.
            base["ttl_ms"] = (
                _integer((updates or {}).get("ttl_seconds", 3600), name="ttl_seconds")
                * 1000
            )
            result = self._call("renew", base)
        elif resolved_target == LaneLifecycleState.SUBMITTED:
            base["ttl_ms"] = (
                _integer((updates or {}).get("ttl_seconds", 3600), name="ttl_seconds")
                * 1000
            )
            result = self._call("renew", base)
        else:
            raise LaneTransitionError(
                f"illegal lane transition target: {resolved_target.value}"
            )
        decision = _string(result.get("decision"), name="result.decision")
        if decision in {"stale", "wrong_owner", "wrong_fence", "wrong_attempt"}:
            raise StaleLaneFence("lane mutation refused: owner or fence is not current")
        if decision == "terminal":
            raise LaneTransitionError(
                "illegal lane transition: hold is already terminal"
            )
        if decision not in _ACCEPTED:
            raise NativeLaneAuthorityUnavailable(
                f"native lane transition refused: {decision}"
            )
        new_hold = result.get("hold")
        if new_hold is None:
            raise NativeLaneProtocolError("accepted native transition lacks a hold")
        record = self._record_from_hold(_mapping(new_hold, name="result.hold"))
        return record.model_copy(
            update={"state": resolved_target, "last_transition": operation}
        )

    def heartbeat_lane(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        updates: Mapping[str, Any],
        now: datetime,
    ) -> LaneRecord:
        hold = self._current_hold(lane_id)
        self._authorize_hold(hold, owner_id=owner_id, fence=fence)
        work_item_id = _string(hold.get("work_item_id"), name="hold.work_item_id")
        attempt = _integer(hold.get("attempt"), name="hold.attempt", minimum=1)
        lease_epoch = _integer(hold.get("lease_epoch"), name="hold.lease_epoch")
        fencing_token = _integer(hold.get("fencing_token"), name="hold.fencing_token")
        hold_id = _string(hold.get("hold_id"), name="hold.hold_id")
        hold_revision = _integer(
            hold.get("hold_revision"), name="hold.hold_revision", minimum=0
        )
        observed = updates.get("observed_disk_bytes")
        if observed is not None:
            result = self._call(
                "observe",
                {
                    "schema_version": "1",
                    "tenant_ref": self._tenant_ref,
                    "work_item_id": work_item_id,
                    "owner_id": owner_id,
                    "attempt": attempt,
                    "lease_epoch": lease_epoch,
                    "fencing_token": fencing_token,
                    "work_item_fence": str(fencing_token),
                    "hold_id": hold_id,
                    "expected_hold_revision": hold_revision,
                    "observed_disk_bytes": _integer(
                        observed, name="observed_disk_bytes"
                    ),
                    "observation_revision": hold_revision + 1,
                    "idempotency_key": f"observe:{hold_id}:{hold_revision}",
                    "now_ms": self._now_ms(now),
                },
            )
        else:
            result = self._call(
                "renew",
                {
                    "schema_version": "1",
                    "tenant_ref": self._tenant_ref,
                    "work_item_id": work_item_id,
                    "owner_id": owner_id,
                    "attempt": attempt,
                    "lease_epoch": lease_epoch,
                    "fencing_token": fencing_token,
                    "work_item_fence": str(fencing_token),
                    "hold_id": hold_id,
                    "expected_hold_revision": hold_revision,
                    "ttl_ms": 3600 * 1000,
                    "idempotency_key": f"renew:{hold_id}:{hold_revision}",
                    "now_ms": self._now_ms(now),
                },
            )
        decision = _string(result.get("decision"), name="result.decision")
        if decision in {"stale", "wrong_owner", "wrong_fence", "wrong_attempt"}:
            raise StaleLaneFence("lane mutation refused: owner or fence is not current")
        if decision not in _ACCEPTED:
            raise NativeLaneAuthorityUnavailable(
                f"native lane heartbeat refused: {decision}"
            )
        new_hold = result.get("hold")
        if new_hold is None:
            raise NativeLaneProtocolError("accepted native heartbeat lacks a hold")
        return self._record_from_hold(_mapping(new_hold, name="result.hold"))


def create_production_lane_registry(
    graph_client: Any,
    engine: Any,
    *,
    tenant_ref: str,
    host_ref: str,
    store_path: str | None = None,
    quota_policy_name: str = "default",
    quota_policy_version: str = "1",
    clock: Callable[[], datetime] | Any | None = None,
    **registry_kwargs: Any,
) -> Any:
    """Build a :class:`~repository_manager.lane_registry.LaneRegistry` with no fallback.

    There is no path in this function that constructs
    ``FakeDurableLaneAuthority`` -- if ``graph_client``/``engine`` cannot
    supply the native RMDD-28 surface, ``NativeLaneAuthority.from_graph_client``
    raises :class:`NativeLaneAuthorityUnavailable` and construction stops
    there.
    """

    from repository_manager.lane_registry import (
        LaneRegistry,
        NativeLaneAuthorityAdapter,
    )

    native = NativeLaneAuthority.from_graph_client(
        graph_client,
        engine,
        tenant_ref=tenant_ref,
        host_ref=host_ref,
        quota_policy_name=quota_policy_name,
        quota_policy_version=quota_policy_version,
        clock=clock,
    )
    adapter = NativeLaneAuthorityAdapter(native)
    return LaneRegistry(store_path, authority=adapter, clock=clock, **registry_kwargs)


__all__ = [
    "AgentUtilitiesWorkItemClaimer",
    "DevelopmentLaneTransport",
    "NativeLaneAuthority",
    "NativeLaneAuthorityUnavailable",
    "NativeLaneProtocolError",
    "WorkItemLifecycleClaimer",
    "create_production_lane_registry",
    "lane_work_item_id",
]
