"""Host-loss detection, quarantine, and retry-safe reservation release.

This module owns no WorkItem state and is not a second job ledger.  Given a
caller-owned :class:`repository_manager.capacity.CapacityInventory` (the same
instance the RMDD-08 scheduler uses -- never a copy) and an injected
reservation-release port shaped exactly like
``repository_manager.resource_scheduler.ResourceScheduler.release``, it
detects a stale or unknown host, quarantines it so the scheduler admits no
further work there, and requests release of the held reservation so a retry
can be admitted elsewhere without duplicate effect.  It never moves a target
branch and never acquires a controller-local filesystem lock.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from repository_manager.capacity import CapacityInventory, HostState


@runtime_checkable
class ReservationReleasePort(Protocol):
    """The exact frozen RMDD-08 ``ResourceScheduler.release`` shape.

    A real ``ResourceScheduler`` instance satisfies this structurally; tests
    inject a deterministic fake that records calls.
    """

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
        """Release a held reservation through its exact current fence."""


@dataclass(frozen=True)
class HostLossDecision:
    """The outcome of one host-loss reconciliation attempt."""

    host_id: str
    reservation_id: str
    lost: bool
    quarantined: bool
    released: bool
    reason: str


class HostLossReconciler:
    """Detect host loss and drive quarantine plus reservation release."""

    def __init__(
        self, capacity: CapacityInventory, *, release_port: ReservationReleasePort
    ) -> None:
        self._capacity = capacity
        self._release_port = release_port

    def reconcile(
        self,
        *,
        host_id: str,
        reservation_id: str,
        work_item_id: str,
        attempt: int,
        fence: str,
        reason: str = "host_loss_reconcile",
        now: datetime | None = None,
    ) -> HostLossDecision:
        """Quarantine a lost host and release its reservation, if any.

        A fresh, non-quarantined/offline host is a no-op: ``lost=False`` and
        neither quarantine nor release is attempted.  An unknown host (no
        capacity record at all -- e.g. it was already withdrawn) is treated
        as lost without a quarantine transition, since there is no capacity
        record to transition.
        """

        now = (now or datetime.now(UTC)).astimezone(UTC)
        view = self._capacity.get(host_id)
        if view is None:
            released = self._release_port.release(
                reservation_id,
                work_item_id=work_item_id,
                attempt=attempt,
                fence=fence,
                reason=reason,
                at=now,
            )
            return HostLossDecision(
                host_id=host_id,
                reservation_id=reservation_id,
                lost=True,
                quarantined=False,
                released=released,
                reason="host unknown to capacity inventory",
            )

        snapshot = self._capacity.snapshot(host_id, now=now)
        already_unavailable = snapshot.state in {
            HostState.QUARANTINED,
            HostState.OFFLINE,
        }
        if snapshot.heartbeat_fresh and not already_unavailable:
            return HostLossDecision(
                host_id=host_id,
                reservation_id=reservation_id,
                lost=False,
                quarantined=False,
                released=False,
                reason="host heartbeat is fresh; no loss detected",
            )

        quarantined = False
        if not already_unavailable:
            self._capacity.set_state(host_id, HostState.QUARANTINED)
            quarantined = True

        released = self._release_port.release(
            reservation_id,
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reason=reason,
            at=now,
        )
        return HostLossDecision(
            host_id=host_id,
            reservation_id=reservation_id,
            lost=True,
            quarantined=quarantined,
            released=released,
            reason="host heartbeat stale or host already quarantined/offline",
        )


__all__ = ["HostLossDecision", "HostLossReconciler", "ReservationReleasePort"]
