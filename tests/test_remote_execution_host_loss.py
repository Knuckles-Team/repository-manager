"""RMDD-15: mid-job host loss -- quarantine plus retry-safe reservation release.

Host loss, cancellation, and restart are acceptance-critical for this lane.
These tests prove ``HostLossReconciler`` quarantines a lost host so the
scheduler admits no further work there, and releases the held reservation
through the scheduler's own fenced ``release`` so a retry can be admitted
elsewhere without duplicate effect -- never by mutating capacity/reservation
state directly.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from repository_manager.capacity import (
    CapacityInventory,
    HostCapacity,
    HostState,
    ResourceVector,
)
from repository_manager.remote_execution.host_loss import (
    HostLossDecision,
    HostLossReconciler,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _FakeReservationReleasePort:
    """Records release calls and can be scripted to refuse (stale fence)."""

    def __init__(self, *, accept: bool = True) -> None:
        self.accept = accept
        self.calls: list[dict[str, object]] = []

    def release(
        self,
        reservation_id: str,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reason: str = "completed",
        at: datetime | None = None,
    ) -> bool:
        self.calls.append(
            {
                "reservation_id": reservation_id,
                "work_item_id": work_item_id,
                "attempt": attempt,
                "fence": fence,
                "reason": reason,
                "at": at,
            }
        )
        return self.accept


def _host(
    host_id: str,
    *,
    state: HostState = HostState.ACTIVE,
    heartbeat_at: datetime = NOW,
    heartbeat_ttl_seconds: int = 120,
) -> HostCapacity:
    return HostCapacity(
        host_id,
        ResourceVector(cpu_weight=4, memory_mib=4096, disk_mib=4096, process_slots=2),
        target_kind="inventory_alias",
        state=state,
        heartbeat_at=heartbeat_at,
        heartbeat_ttl_seconds=heartbeat_ttl_seconds,
    )


def test_a_fresh_healthy_host_is_a_no_op() -> None:
    capacity = CapacityInventory([_host("host:build-1")])
    release_port = _FakeReservationReleasePort()
    reconciler = HostLossReconciler(capacity, release_port=release_port)

    decision = reconciler.reconcile(
        host_id="host:build-1",
        reservation_id="reservation:abc",
        work_item_id="workitem:repository_manager:job-1",
        attempt=1,
        fence="fence:abc",
        now=NOW,
    )

    assert decision == HostLossDecision(
        host_id="host:build-1",
        reservation_id="reservation:abc",
        lost=False,
        quarantined=False,
        released=False,
        reason="host heartbeat is fresh; no loss detected",
    )
    assert release_port.calls == []
    fresh_host = capacity.get("host:build-1")
    assert fresh_host is not None
    assert fresh_host.state == HostState.ACTIVE


def test_a_stale_heartbeat_mid_job_quarantines_and_releases() -> None:
    """The central mid-job host-loss proof: a build that stopped heartbeating."""

    capacity = CapacityInventory(
        [_host("host:build-1", heartbeat_at=NOW - timedelta(seconds=600))]
    )
    release_port = _FakeReservationReleasePort()
    reconciler = HostLossReconciler(capacity, release_port=release_port)

    decision = reconciler.reconcile(
        host_id="host:build-1",
        reservation_id="reservation:abc",
        work_item_id="workitem:repository_manager:job-1",
        attempt=1,
        fence="fence:abc",
        reason="heartbeat_timeout",
        now=NOW,
    )

    assert decision.lost is True
    assert decision.quarantined is True
    assert decision.released is True
    quarantined_host = capacity.get("host:build-1")
    assert quarantined_host is not None
    assert quarantined_host.state == HostState.QUARANTINED
    assert release_port.calls == [
        {
            "reservation_id": "reservation:abc",
            "work_item_id": "workitem:repository_manager:job-1",
            "attempt": 1,
            "fence": "fence:abc",
            "reason": "heartbeat_timeout",
            "at": NOW,
        }
    ]


def test_an_unknown_host_is_lost_without_a_quarantine_transition() -> None:
    """No capacity record exists at all (e.g. already withdrawn)."""

    capacity = CapacityInventory()
    release_port = _FakeReservationReleasePort()
    reconciler = HostLossReconciler(capacity, release_port=release_port)

    decision = reconciler.reconcile(
        host_id="host:withdrawn",
        reservation_id="reservation:abc",
        work_item_id="workitem:repository_manager:job-1",
        attempt=1,
        fence="fence:abc",
        now=NOW,
    )

    assert decision.lost is True
    assert decision.quarantined is False  # nothing to transition
    assert decision.released is True
    assert len(release_port.calls) == 1


def test_an_already_quarantined_host_is_not_re_quarantined_but_release_is_retried() -> (
    None
):
    """A retry against an already-quarantined host must remain idempotent."""

    capacity = CapacityInventory([_host("host:build-1", state=HostState.QUARANTINED)])
    release_port = _FakeReservationReleasePort()
    reconciler = HostLossReconciler(capacity, release_port=release_port)

    decision = reconciler.reconcile(
        host_id="host:build-1",
        reservation_id="reservation:abc",
        work_item_id="workitem:repository_manager:job-1",
        attempt=1,
        fence="fence:abc",
        now=NOW,
    )

    assert decision.lost is True
    assert decision.quarantined is False  # already unavailable; not a new transition
    assert decision.released is True
    still_quarantined_host = capacity.get("host:build-1")
    assert still_quarantined_host is not None
    assert still_quarantined_host.state == HostState.QUARANTINED


def test_an_offline_host_is_treated_as_already_unavailable() -> None:
    capacity = CapacityInventory([_host("host:build-1", state=HostState.OFFLINE)])
    release_port = _FakeReservationReleasePort()
    reconciler = HostLossReconciler(capacity, release_port=release_port)

    decision = reconciler.reconcile(
        host_id="host:build-1",
        reservation_id="reservation:abc",
        work_item_id="workitem:repository_manager:job-1",
        attempt=1,
        fence="fence:abc",
        now=NOW,
    )
    assert decision.quarantined is False
    assert decision.lost is True


def test_a_stale_fence_release_reports_released_false_without_raising() -> None:
    """The scheduler's own fence discipline may refuse a stale/duplicate release.

    ``HostLossReconciler`` must surface that honestly (``released=False``)
    rather than assume success -- this is exactly the "restart cannot produce
    duplicate success" acceptance gate: a second, stale reconciliation attempt
    (e.g. after the WorkItem already advanced past this attempt) must not be
    reported as if it released anything.
    """

    capacity = CapacityInventory(
        [_host("host:build-1", heartbeat_at=NOW - timedelta(seconds=600))]
    )
    release_port = _FakeReservationReleasePort(accept=False)
    reconciler = HostLossReconciler(capacity, release_port=release_port)

    decision = reconciler.reconcile(
        host_id="host:build-1",
        reservation_id="reservation:abc",
        work_item_id="workitem:repository_manager:job-1",
        attempt=1,
        fence="fence:stale",
        now=NOW,
    )

    assert decision.lost is True
    assert decision.quarantined is True  # host transition still happens
    assert decision.released is False  # but the release itself was refused
    assert len(release_port.calls) == 1


def test_reconciler_never_mutates_capacity_or_reservation_state_directly() -> None:
    """Structural proof: the reconciler holds no reservation store reference.

    ``HostLossReconciler`` must only ever call the injected
    ``ReservationReleasePort`` -- never touch ``CapacityInventory`` reservation
    accounting directly -- so it can never become a second job ledger.
    """

    capacity = CapacityInventory([_host("host:build-1")])
    release_port = _FakeReservationReleasePort()
    reconciler = HostLossReconciler(capacity, release_port=release_port)
    assert not hasattr(reconciler, "_reservations")
    assert not hasattr(reconciler, "_reservation_store")
