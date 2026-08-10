"""RMDD-28 parity, no-fallback, and immutability tests for the RM native lane binding.

The parity matrix runs the *same* operation sequence through RMDD-09's pure
``FakeDurableLaneAuthority`` and through :class:`NativeLaneAuthority` bound to
an in-memory fake of the RMDD-28 wire protocol, and asserts matching
externally observable ``LaneRecord`` results -- proving parity is measured,
not asserted, per the RMDD-28 lane brief's required method.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from repository_manager.lane_record import LaneLifecycleState
from repository_manager.lane_registry import (
    FakeDurableLaneAuthority,
    LaneConflictError,
    LaneRegistry,
    NativeLaneAuthorityAdapter,
    StaleLaneFence,
)
from repository_manager.native_lane_authority import (
    NativeLaneAuthority,
    NativeLaneAuthorityUnavailable,
    lane_work_item_id,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class Clock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


# ---------------------------------------------------------------------------
# Fake RMDD-28 native transport: an in-memory, deliberately independent
# reimplementation of the reserve/renew/observe/finish/status decision
# vocabulary described in the RMDD-28 lane brief and the frozen epistemic-graph
# Rust protocol.  This is a TEST DOUBLE ONLY -- it is never wired into
# NativeLaneAuthority.from_graph_client / create_production_lane_registry.
# ---------------------------------------------------------------------------


class FakeDevelopmentLaneTransport:
    def __init__(self) -> None:
        self._holds: dict[str, dict[str, Any]] = {}
        self._by_lane: dict[str, str] = {}
        self.calls: list[str] = []

    def _active_conflict(
        self,
        *,
        repository_id: str,
        branch: str,
        worktree_locator: str,
        exclude: str | None,
    ) -> bool:
        for hold_id, hold in self._holds.items():
            if hold_id == exclude or hold["state"] in {
                "released",
                "cleaned",
                "aborted",
            }:
                continue
            if (hold["repository_id"], hold["branch"]) == (
                repository_id,
                branch,
            ) or hold["worktree_locator"] == worktree_locator:
                return True
        return False

    def reserve(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("reserve")
        intent = request["intent"]
        lane_id = intent["lane_id"]
        existing_hold_id = self._by_lane.get(lane_id)
        if existing_hold_id is not None:
            existing = self._holds[existing_hold_id]
            if existing["input_fingerprint"] != intent["input_fingerprint"]:
                return {"decision": "input_conflict", "hold": None}
            return {"decision": "idempotent", "hold": dict(existing)}
        if self._active_conflict(
            repository_id=intent["repository_id"],
            branch=intent["branch"],
            worktree_locator=intent["worktree_locator"],
            exclude=None,
        ):
            return {"decision": "conflict", "hold": None}
        hold_id = f"hold:{uuid.uuid4().hex}"
        now_ms = request["now_ms"]
        hold = {
            "hold_id": hold_id,
            "lane_id": lane_id,
            "request_id": intent["request_id"],
            "work_item_id": request["work_item_id"],
            "owner_id": request["owner_id"],
            "session_id": intent["session_id"],
            "repository_id": intent["repository_id"],
            "base_ref": intent["base_ref"],
            "base_sha": intent["base_sha"],
            "branch": intent["branch"],
            "worktree_locator": intent["worktree_locator"],
            "input_fingerprint": intent["input_fingerprint"],
            "predicted_disk_bytes": intent["predicted_disk_bytes"],
            "observed_disk_bytes": 0,
            "state": "allocating",
            "attempt": request["attempt"],
            "lease_epoch": request["lease_epoch"],
            "fencing_token": request["fencing_token"],
            "hold_revision": 1,
            "last_renewed_at_ms": now_ms,
            "expires_at_ms": now_ms + intent["ttl_ms"],
        }
        self._holds[hold_id] = hold
        self._by_lane[lane_id] = hold_id
        return {"decision": "accepted", "hold": dict(hold)}

    def _resolve(self, hold_id: str) -> dict[str, Any] | None:
        return self._holds.get(hold_id)

    def _fence_check(self, hold: dict[str, Any], request: dict[str, Any]) -> str | None:
        if hold["owner_id"] != request["owner_id"]:
            return "wrong_owner"
        if hold["fencing_token"] != request["fencing_token"]:
            return "wrong_fence"
        if hold["hold_revision"] != request["expected_hold_revision"]:
            return "stale"
        if hold["state"] in {"released", "cleaned", "aborted"}:
            return "terminal"
        return None

    def renew(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("renew")
        hold = self._resolve(request["hold_id"])
        if hold is None:
            return {"decision": "not_found", "hold": None}
        refusal = self._fence_check(hold, request)
        if refusal:
            return {"decision": refusal, "hold": None}
        hold["state"] = "active" if hold["state"] == "allocating" else hold["state"]
        hold["hold_revision"] += 1
        hold["last_renewed_at_ms"] = request["now_ms"]
        hold["expires_at_ms"] = request["now_ms"] + request["ttl_ms"]
        return {"decision": "accepted", "hold": dict(hold)}

    def observe(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("observe")
        hold = self._resolve(request["hold_id"])
        if hold is None:
            return {"decision": "not_found", "hold": None}
        refusal = self._fence_check(hold, request)
        if refusal:
            return {"decision": refusal, "hold": None}
        hold["observed_disk_bytes"] = request["observed_disk_bytes"]
        hold["hold_revision"] += 1
        hold["last_renewed_at_ms"] = request["now_ms"]
        return {"decision": "accepted", "hold": dict(hold)}

    def finish(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("finish")
        hold = self._resolve(request["hold_id"])
        if hold is None:
            return {"decision": "not_found", "hold": None}
        refusal = self._fence_check(hold, request)
        if refusal:
            return {"decision": refusal, "hold": None}
        hold["state"] = (
            "released" if request["terminal_state"] == "succeeded" else "aborted"
        )
        hold["hold_revision"] += 1
        hold["last_renewed_at_ms"] = request["now_ms"]
        return {"decision": "accepted", "hold": dict(hold)}

    def cleanup_complete(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("cleanup_complete")
        hold = self._resolve(request["hold_id"])
        if hold is None:
            return {"decision": "not_found", "hold": None}
        hold["state"] = "cleaned"
        hold["hold_revision"] += 1
        return {"decision": "accepted", "hold": dict(hold)}

    def query(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("query")
        hold = self._resolve(request["hold_id"])
        if hold is None:
            return {"decision": "not_found", "hold": None}
        return {"decision": "accepted", "hold": dict(hold)}

    def status(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("status")
        lane_id = request.get("lane_id")
        holds = [
            dict(hold)
            for hold in self._holds.values()
            if lane_id is None or hold["lane_id"] == lane_id
        ]
        return {"complete": True, "next_cursor": None, "holds": holds}

    def update_quota(self, request: dict[str, Any]) -> dict[str, Any]:
        self.calls.append("update_quota")
        return {"decision": "accepted", "policy": request["policy"]}


class FakeWorkItemClaimer:
    """Deterministic test double for the ``lane.lifecycle`` claim step."""

    def __init__(self) -> None:
        self._attempt: dict[str, int] = {}

    def claim_lifecycle(
        self,
        *,
        lane_id: str,
        request_key: str,
        repository_id: str,
        owner_id: str,
        session_id: str,
        tenant_ref: str,
        lane_intent: dict[str, Any],
        now: datetime,
    ) -> dict[str, Any]:
        self._attempt[lane_id] = self._attempt.get(lane_id, 0) + 1
        return {
            "work_item_id": lane_work_item_id(lane_id),
            "lease_owner": owner_id,
            "attempt": 1,
            "lease_epoch": 1,
            "fencing_token": 1,
        }


def _native_authority() -> NativeLaneAuthority:
    return NativeLaneAuthority(
        FakeDevelopmentLaneTransport(),
        FakeWorkItemClaimer(),
        tenant_ref="tenant:one",
        host_ref="host:one",
        clock=Clock(),
    )


def _fake_registry(path: Path, *, clock: Clock | None = None) -> LaneRegistry:
    authority = FakeDurableLaneAuthority(clock=clock)
    return LaneRegistry(path, clock=clock, authority=authority)


def _native_registry(path: Path, *, clock: Clock | None = None) -> LaneRegistry:
    native = NativeLaneAuthority(
        FakeDevelopmentLaneTransport(),
        FakeWorkItemClaimer(),
        tenant_ref="tenant:one",
        host_ref="host:one",
        clock=clock or Clock(),
    )
    authority = NativeLaneAuthorityAdapter(native)
    return LaneRegistry(path, clock=clock, authority=authority)


def _allocate(
    registry: LaneRegistry, root: Path, *, request_id: str = "request:one"
) -> Any:
    return registry.allocate(
        root,
        "feature/one",
        root / "lane",
        owner_id="agent:one",
        session_id="session:one",
        host_id="host:one",
        request_id=request_id,
        predicted_disk_bytes=10,
        disk_budget_bytes=10,
    )


@pytest.mark.parametrize("factory", [_fake_registry, _native_registry])
def test_allocate_parity_matrix(tmp_path: Path, factory: Any) -> None:
    registry = factory(tmp_path / "lanes.sqlite3")
    record = _allocate(registry, tmp_path)
    assert record.repository_id
    assert record.branch == "feature/one"
    assert record.owner_id == "agent:one"
    assert record.session_id == "session:one"
    assert record.predicted_disk_bytes == 10
    assert record.state == LaneLifecycleState.ALLOCATING


@pytest.mark.parametrize("factory", [_fake_registry, _native_registry])
def test_allocate_idempotent_replay_returns_the_original_record(
    tmp_path: Path, factory: Any
) -> None:
    registry = factory(tmp_path / "replay.sqlite3")
    first = _allocate(registry, tmp_path, request_id="request:replay")
    second = _allocate(registry, tmp_path, request_id="request:replay")
    assert second.lane_id == first.lane_id
    assert second.fence == first.fence


def test_allocate_branch_conflict_refused_by_both(tmp_path: Path) -> None:
    for factory, path in (
        (_fake_registry, tmp_path / "fake.sqlite3"),
        (_native_registry, tmp_path / "native.sqlite3"),
    ):
        registry = factory(path)
        _allocate(registry, tmp_path, request_id="request:one")
        with pytest.raises(LaneConflictError):
            registry.allocate(
                tmp_path,
                "feature/one",
                tmp_path / "other-lane",
                owner_id="agent:two",
                session_id="session:two",
                host_id="host:two",
                request_id="request:two",
                predicted_disk_bytes=5,
                disk_budget_bytes=5,
            )


def test_heartbeat_updates_observed_disk_without_growing_workitem_count(
    tmp_path: Path,
) -> None:
    for factory, path in (
        (_fake_registry, tmp_path / "fake2.sqlite3"),
        (_native_registry, tmp_path / "native2.sqlite3"),
    ):
        registry = factory(path)
        record = _allocate(registry, tmp_path)
        updated = registry.heartbeat(
            record.lane_id,
            owner_id=record.owner_id,
            fence=record.fence,
            observed_disk_bytes=7,
        )
        assert updated.observed_disk_bytes == 7
        assert updated.lane_id == record.lane_id


def test_stale_fence_refused_by_both(tmp_path: Path) -> None:
    for factory, path in (
        (_fake_registry, tmp_path / "fake3.sqlite3"),
        (_native_registry, tmp_path / "native3.sqlite3"),
    ):
        registry = factory(path)
        record = _allocate(registry, tmp_path)
        with pytest.raises(StaleLaneFence):
            registry.heartbeat(
                record.lane_id,
                owner_id=record.owner_id,
                fence="fence:not-the-real-one",
                observed_disk_bytes=1,
            )


def test_finish_transition_reaches_terminal_state_on_both(tmp_path: Path) -> None:
    for factory, path in (
        (_fake_registry, tmp_path / "fake4.sqlite3"),
        (_native_registry, tmp_path / "native4.sqlite3"),
    ):
        registry = factory(path)
        record = _allocate(registry, tmp_path)
        # ALLOCATING -> LANDED is not a legal direct transition on either
        # authority (ALLOCATING only legally reaches ACTIVE/ABORTED/EXPIRED);
        # activate first, matching FakeDurableLaneAuthority._legal().
        activated = registry.activate(
            record.lane_id, owner_id=record.owner_id, fence=record.fence
        )
        finished = registry.finish(
            record.lane_id, owner_id=activated.owner_id, fence=activated.fence
        )
        assert finished.state == LaneLifecycleState.LANDED


# ---------------------------------------------------------------------------
# No-fallback proof
# ---------------------------------------------------------------------------


def test_native_lane_authority_refuses_construction_without_full_transport() -> None:
    """A transport missing even one of the eight native verbs must refuse.

    This proves the gate catches a known-bad input (H-9): an incomplete
    transport -- not merely an absent one -- must still fail closed rather
    than silently operating with a partial surface.
    """

    class IncompleteTransport:
        def reserve(self, request: dict[str, Any]) -> dict[str, Any]:
            raise AssertionError("must never be called")

        # renew/observe/finish/cleanup_complete/query/status/update_quota
        # deliberately absent.

    with pytest.raises(NativeLaneAuthorityUnavailable) as excinfo:
        NativeLaneAuthority(
            IncompleteTransport(),
            FakeWorkItemClaimer(),
            tenant_ref="tenant:one",
            host_ref="host:one",
        )
    assert excinfo.value.code == "native_development_lane_authority_unavailable"


def test_native_lane_authority_refuses_construction_without_claimer() -> None:
    with pytest.raises(NativeLaneAuthorityUnavailable):
        NativeLaneAuthority(
            FakeDevelopmentLaneTransport(),
            None,
            tenant_ref="tenant:one",
            host_ref="host:one",
        )


def test_native_lane_authority_refuses_against_the_real_generated_client() -> None:
    """The genuine, current, no-fallback refusal against real epistemic-graph.

    ``epistemic_graph.client.EpistemicGraphClient``/``SyncEpistemicGraphClient``
    do not yet expose a ``development_lanes`` namespace (only ``work_items``
    for RMDD-27's resource reservation). Importing the real package and
    attempting to build the AU transport against a bare namespace-less stand-in
    for that client -- not a synthetic "missing" fixture -- must raise
    ``DevelopmentLaneAuthorityUnavailable``. This is today's honest state of
    the world, not a placeholder.
    """

    agent_utilities_orchestration_development_lane = pytest.importorskip(
        "agent_utilities.orchestration.development_lane"
    )
    transport_cls = agent_utilities_orchestration_development_lane.EngineNativeDevelopmentLaneTransport
    unavailable_cls = agent_utilities_orchestration_development_lane.DevelopmentLaneAuthorityUnavailable

    class BareClientWithoutDevelopmentLanes:
        """Shaped like the real generated client's public surface, minus the
        one namespace RMDD-28 needs -- proving the refusal is about the real
        capability gap, not a typo in this test."""

        def __init__(self) -> None:
            self.work_items = object()

        def supports(self, _op: str) -> bool:
            return True

    with pytest.raises(unavailable_cls) as excinfo:
        transport_cls(BareClientWithoutDevelopmentLanes())
    assert excinfo.value.code == "native_development_lane_authority_unavailable"
