"""Characterization tests for CXA-FL-REPOSITORYMANAGER-03.

Pins one branch of ``InMemoryWorkItemReservationPort.atomic_reserve``
(``repository_manager/reservations.py``) that the existing
``tests/test_resource_scheduler.py`` suite exercises only above the exact
disk high-watermark boundary, never *at* it: the ``predicted_used >= high``
comparison in the native disk-hysteresis block.

Mutation-proof performed manually before writing this file (reverted before
commit): changing ``>=`` to ``>`` at that comparison left the full existing
``test_resource_scheduler.py`` suite green (58/58) -- i.e. that suite does not
pin the exact-equality boundary. This file closes that gap so the refactor
commit (extracting the disk-watermark block into
``_reserve_disk_watermark_check``) cannot silently flip the boundary.

This file is intentionally small and additive: it does not replace
``test_resource_scheduler.py`` as the primary characterization baseline for
``atomic_reserve`` (that suite's 58 tests already cover identity/idempotency,
concurrency, anti-affinity, exclusivity, fairness debt, and the disk-blocked
hysteresis path; both are run before and after the refactor and must stay
identical).
"""

from __future__ import annotations

from datetime import UTC, datetime

from repository_manager.capacity import CapacityView, HostState, ResourceVector
from repository_manager.reservations import (
    FenceDecision,
    InMemoryWorkItemReservationPort,
    ReservationRecord,
)

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _view(host_id: str, *, total_disk_mib: int) -> CapacityView:
    total = ResourceVector(8, 8_192, total_disk_mib, 8)
    return CapacityView(
        host_id=host_id,
        state=HostState.ACTIVE,
        labels=(),
        target_kind="local",
        total=total,
        live=ResourceVector(),
        reserved=ResourceVector(),
        available=total,
        heartbeat_at=NOW,
        heartbeat_fresh=True,
        observed_disk_free_mib=None,
        heartbeat_ttl_seconds=120,
    )


def _reservation(
    reservation_id: str,
    work_item_id: str,
    fence: str,
    *,
    host_id: str,
    disk_mib: int,
    disk_high_watermark_mib: int | None = None,
) -> ReservationRecord:
    return ReservationRecord(
        reservation_id=reservation_id,
        work_item_id=work_item_id,
        attempt=1,
        fence=fence,
        host_id=host_id,
        profile_name="light-check",
        requirement=ResourceVector(1, 256, disk_mib, 1),
        capacity_snapshot={},
        disk_high_watermark_mib=disk_high_watermark_mib,
        reserved_at=NOW,
        expires_at=NOW.replace(hour=13),
    )


def test_disk_high_watermark_blocks_at_exact_boundary_not_only_above_it() -> None:
    """predicted_used == high must block (>=), matching the source comment's
    intent that the high watermark is a hard ceiling, not merely a
    strictly-above threshold."""

    port = InMemoryWorkItemReservationPort(clock=lambda: NOW)
    port.register_capacity_view(_view("h1", total_disk_mib=1_000))

    port.claim("wi-a", fence="fa", now=NOW)
    first = _reservation(
        "r-a", "wi-a", "fa", host_id="h1", disk_mib=300, disk_high_watermark_mib=600
    )
    assert port.atomic_reserve(
        work_item_id="wi-a", attempt=1, fence="fa", reservation=first
    ) is FenceDecision.ACCEPTED

    # available.disk_mib is now 1000 - 300 = 700. A second reservation of
    # exactly 300 makes predicted_used = 1000 - 700 + 300 = 600, exactly at
    # the high watermark.
    port.claim("wi-b", fence="fb", now=NOW)
    second = _reservation(
        "r-b", "wi-b", "fb", host_id="h1", disk_mib=300, disk_high_watermark_mib=600
    )
    decision = port.atomic_reserve(
        work_item_id="wi-b", attempt=1, fence="fb", reservation=second
    )
    assert decision is FenceDecision.DISK
