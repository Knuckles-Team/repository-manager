"""Focused RMDD-08 scheduler, capacity, fence, and simulation tests."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from repository_manager.capacity import (
    CapacityInventory,
    HostCapacity,
    HostState,
    ResourceVector,
)
from repository_manager.development import ResourceRequest, TargetKind, TargetPolicy
from repository_manager.disk_policy import (
    DiskDecisionCode,
    DiskPolicy,
    DiskState,
    DiskWatermarks,
)
from repository_manager.fairness import (
    FairnessAuthority,
    FairnessPolicy,
    FairnessSelector,
    InMemoryFairnessState,
    JsonFairnessState,
    QueueCandidate,
)
from repository_manager.reservations import (
    FenceDecision,
    InMemoryReservationStore,
    InMemoryWorkItemReservationPort,
    JsonReservationStore,
    ReservationRecord,
)
from repository_manager.resource_profiles import (
    ResourceProfile,
    ResourceProfileError,
    ResourceProfileRegistry,
    default_resource_profiles,
)
from repository_manager.resource_scheduler import (
    AdmissionReason,
    AdmissionRequest,
    AdmissionStatus,
    ResourceScheduler,
    reservation_id_for,
)

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _port() -> InMemoryWorkItemReservationPort:
    return InMemoryWorkItemReservationPort(clock=lambda: NOW)


def _host(
    host_id: str,
    *,
    cpu: int = 16,
    memory: int = 32_768,
    disk: int = 100_000,
    processes: int = 16,
    labels: tuple[str, ...] = (),
    target_kind: str = "local",
    free_disk: int | None = None,
    state: HostState = HostState.ACTIVE,
    heartbeat_at: datetime = NOW,
    heartbeat_ttl_seconds: int = 120,
) -> HostCapacity:
    return HostCapacity(
        host_id,
        ResourceVector(cpu, memory, disk, processes),
        labels=labels,
        target_kind=target_kind,
        state=state,
        heartbeat_at=heartbeat_at,
        heartbeat_ttl_seconds=heartbeat_ttl_seconds,
        observed_disk_free_mib=free_disk,
    )


def _scheduler(
    *hosts: HostCapacity,
    port: InMemoryWorkItemReservationPort | None = None,
    store: InMemoryReservationStore | None = None,
    fairness: FairnessSelector | None = None,
    profiles: ResourceProfileRegistry | None = None,
) -> tuple[ResourceScheduler, InMemoryWorkItemReservationPort]:
    port = port or _port()
    scheduler = ResourceScheduler(
        capacity=CapacityInventory(hosts),
        work_item_port=port,
        reservation_store=store,
        fairness=fairness,
        profiles=profiles,
    )
    return scheduler, port


def _request(
    work_item_id: str,
    fence: str,
    *,
    resource_class: str = "light-check",
    attempt: int = 1,
    repository_id: str = "repo",
    branch: str = "lane",
    owner_id: str = "",
    tenant_id: str = "",
    reservation_id: str = "",
    now: datetime = NOW,
    resources: ResourceRequest | None = None,
) -> AdmissionRequest:
    return AdmissionRequest(
        work_item_id=work_item_id,
        attempt=attempt,
        fence=fence,
        resources=resources or ResourceRequest(resource_class=resource_class),
        repository_id=repository_id,
        branch=branch,
        owner_id=owner_id,
        tenant_id=tenant_id,
        reservation_id=reservation_id,
        enqueued_at=now,
    )


def test_in_memory_port_clock_controls_liveness_and_lifecycle_timestamps():
    current = [NOW]
    port = InMemoryWorkItemReservationPort(clock=lambda: current[0])
    scheduler, _ = _scheduler(
        _host("local", heartbeat_ttl_seconds=10),
        port=port,
    )
    port.claim("wi", fence="f", ttl_seconds=100_000)
    port.claim("expiring", fence="ef", ttl_seconds=10)
    port.claim("reclaim", fence="rf", ttl_seconds=10_000)
    assert port.is_current("expiring", 1, "ef")
    authoritative = port._authoritative_view("local")
    assert authoritative is not None
    assert authoritative.heartbeat_fresh

    decision = scheduler.admit(_request("wi", "f"), now=NOW)
    assert decision.admitted
    reclaim_decision = scheduler.admit(_request("reclaim", "rf"), now=NOW)
    assert reclaim_decision.admitted
    assert reclaim_decision.reservation is not None
    assert (
        port.atomic_reclaim(
            work_item_id="reclaim",
            attempt=1,
            fence="rf",
            reservation_id=reclaim_decision.reservation_id,
            reservation=reclaim_decision.reservation,
        )
        == FenceDecision.STALE
    )

    current[0] = NOW + timedelta(seconds=10_001)
    assert not port.is_current("expiring", 1, "ef")
    authoritative = port._authoritative_view("local")
    assert authoritative is not None
    assert not authoritative.heartbeat_fresh
    assert (
        port.atomic_reclaim(
            work_item_id="reclaim",
            attempt=1,
            fence="rf",
            reservation_id=reclaim_decision.reservation_id,
            reservation=reclaim_decision.reservation,
        )
        == FenceDecision.ACCEPTED
    )
    assert scheduler.release(
        decision.reservation_id,
        work_item_id="wi",
        attempt=1,
        fence="f",
    )
    native = port.query_reservation(
        reservation_id=decision.reservation_id,
        work_item_id="wi",
        attempt=1,
        fence="f",
        for_lifecycle=True,
    )
    assert isinstance(native, ReservationRecord)
    assert native.released_at == current[0]


def test_unknown_profile_fails_closed_before_native_reservation():
    scheduler, port = _scheduler(_host("local"))
    port.claim("wi", fence="f")
    decision = scheduler.admit(_request("wi", "f", resource_class="made-up"), now=NOW)
    assert decision.status == AdmissionStatus.REFUSED
    assert decision.reason_code == AdmissionReason.UNKNOWN_PROFILE
    assert port.link("missing") is None


def test_native_work_item_extension_rejects_metadata_policy_mismatch():
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port)
    expected = ResourceVector(1, 256, 256, 1)
    port.claim(
        "wi",
        fence="f",
        ttl_seconds=10_000,
        now=NOW,
        owner_id="owner-a",
        tenant_id="tenant-a",
        profile_name="light-check",
        requirement=expected,
        repository_id="repo",
        branch="lane",
        concurrency_key="light-check",
        fairness_group="default",
    )
    mismatch = scheduler.admit(
        _request(
            "wi",
            "f",
            owner_id="owner-b",
            tenant_id="tenant-a",
            resources=ResourceRequest(resource_class="light-check"),
        ),
        now=NOW,
    )
    assert mismatch.status == AdmissionStatus.DEFERRED
    assert mismatch.reason_code == AdmissionReason.FENCE_CONFLICT
    assert port.link(reservation_id_for("wi", 1)) is None
    assert scheduler.capacity.reserved_for("local") == ResourceVector()

    matching = scheduler.admit(
        _request(
            "wi",
            "f",
            owner_id="owner-a",
            tenant_id="tenant-a",
            resources=ResourceRequest(resource_class="light-check"),
        ),
        now=NOW,
    )
    assert matching.admitted


def test_profile_is_authoritative_for_concurrency_key_and_rejects_bad_watermarks():
    profiles = default_resource_profiles()
    request = ResourceRequest(resource_class="frontend-build")
    profile, merged = AdmissionRequest(
        "wi", 1, "f", resources=request
    ).profile_and_request(profiles)
    assert profile.name == "frontend-build"
    assert merged.concurrency_key == "frontend-build"
    assert merged.cpu_weight == 8
    with pytest.raises(ResourceProfileError):
        ResourceProfile("bad", disk_low_watermark_mib=9, disk_high_watermark_mib=8)


def test_live_capacity_guard_rejects_reversed_usage_and_negative_accounting():
    with pytest.raises(ValueError):
        HostCapacity("bad", ResourceVector(1, 1, 1, 1), live=ResourceVector(2, 0, 0, 0))
    inventory = CapacityInventory(
        [_host("local", cpu=2, memory=512, disk=100, processes=2)]
    )
    assert not inventory.try_reserve(
        "local",
        "too-heavy",
        ResourceVector(cpu_weight=3, memory_mib=1, disk_mib=1, process_slots=1),
        now=NOW,
    )
    assert inventory.reserved_for("local") == ResourceVector()


def test_insufficient_capacity_is_refused_before_any_executor_boundary():
    scheduler, port = _scheduler(
        _host("local", cpu=1, memory=512, disk=1_000, processes=1)
    )
    port.claim("wi", fence="f")
    request = _request(
        "wi",
        "f",
        resources=ResourceRequest(
            resource_class="light-check",
            cpu_weight=2,
            memory_mib=512,
            disk_mib=1,
            process_slots=1,
        ),
    )
    decision = scheduler.admit(request, now=NOW)
    assert decision.status == AdmissionStatus.DEFERRED
    assert decision.reason_code == AdmissionReason.CAPACITY
    assert scheduler.reservations.all() == ()
    assert scheduler.capacity.reserved_for("local") == ResourceVector()


def test_frontend_concurrency_limit_spans_repositories():
    scheduler, port = _scheduler(
        _host("local", cpu=64, memory=100_000, disk=100_000, processes=32)
    )
    port.claim("wi-a", fence="fa")
    port.claim("wi-b", fence="fb")
    first = scheduler.admit(
        _request("wi-a", "fa", resource_class="frontend-build", repository_id="a"),
        now=NOW,
    )
    second = scheduler.admit(
        _request("wi-b", "fb", resource_class="frontend-build", repository_id="b"),
        now=NOW,
    )
    assert first.admitted
    assert second.status == AdmissionStatus.DEFERRED
    assert second.reason_code == AdmissionReason.CONCURRENCY


def test_independent_light_jobs_fit_until_exact_capacity_and_release():
    scheduler, port = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4)
    )
    claims = []
    for index in range(4):
        work_item = f"wi-{index}"
        fence = f"f-{index}"
        port.claim(work_item, fence=fence)
        decision = scheduler.admit(
            _request(work_item, fence, reservation_id=f"r-{index}"), now=NOW
        )
        assert decision.admitted
        claims.append(decision.reservation_id)
    assert scheduler.capacity.reserved_for("local") == ResourceVector(4, 1024, 1024, 4)
    port.rotate("wi-0", attempt=2)
    # The old fence cannot release a newer attempt's reservation.
    record = scheduler.reservations.get(claims[0])
    assert record is not None
    assert not scheduler.release(claims[0], work_item_id="wi-0", attempt=1, fence="f-0")
    # The remaining three current claims can release exactly once.
    for index in range(1, 4):
        assert scheduler.release(
            claims[index], work_item_id=f"wi-{index}", attempt=1, fence=f"f-{index}"
        )
    assert scheduler.capacity.reserved_for("local") == ResourceVector(1, 256, 256, 1)


def test_native_capacity_fake_is_atomic_under_concurrent_claims():
    port = _port()
    scheduler, _ = _scheduler(
        _host("local", cpu=2, memory=512, disk=512, processes=2), port=port
    )
    requests = []
    for index in range(8):
        work_item, fence = f"wi-{index}", f"f-{index}"
        port.claim(work_item, fence=fence)
        requests.append(_request(work_item, fence, reservation_id=f"r-{index}"))
    with ThreadPoolExecutor(max_workers=8) as pool:
        decisions = tuple(pool.map(lambda req: scheduler.admit(req, now=NOW), requests))
    assert sum(decision.admitted for decision in decisions) == 2
    assert scheduler.capacity.reserved_for("local") == ResourceVector(2, 512, 512, 2)
    assert len(scheduler.reservations.all()) == 2


def test_native_policy_refusal_is_not_misreported_as_fence_conflict():
    class PolicyRefusingPort(InMemoryWorkItemReservationPort):
        def __init__(self):
            super().__init__(clock=lambda: NOW)

        def atomic_reserve(self, **kwargs):  # type: ignore[no-untyped-def]
            return FenceDecision.ANTI_AFFINITY

    port = PolicyRefusingPort()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", fence="f")
    decision = scheduler.admit(_request("wi", "f"), now=NOW)
    assert decision.status == AdmissionStatus.DEFERRED
    assert decision.reason_code == AdmissionReason.ANTI_AFFINITY
    assert "policy" not in decision.reason_code.value


@pytest.mark.parametrize(
    ("native_result", "expected_status", "expected_reason"),
    (
        (FenceDecision.ACCEPTED, AdmissionStatus.ADMITTED, AdmissionReason.ADMITTED),
        (
            FenceDecision.IDEMPOTENT,
            AdmissionStatus.ADMITTED,
            AdmissionReason.ADMITTED,
        ),
        (
            FenceDecision.STALE,
            AdmissionStatus.STALE_FENCE,
            AdmissionReason.STALE_FENCE,
        ),
        (
            FenceDecision.CONFLICT,
            AdmissionStatus.DEFERRED,
            AdmissionReason.FENCE_CONFLICT,
        ),
        (
            FenceDecision.INPUT_CONFLICT,
            AdmissionStatus.DEFERRED,
            AdmissionReason.FENCE_CONFLICT,
        ),
        (
            FenceDecision.CAPACITY,
            AdmissionStatus.DEFERRED,
            AdmissionReason.CAPACITY,
        ),
        (
            FenceDecision.POLICY,
            AdmissionStatus.DEFERRED,
            AdmissionReason.FENCE_CONFLICT,
        ),
        (
            FenceDecision.DRAINED,
            AdmissionStatus.DEFERRED,
            AdmissionReason.DRAINED,
        ),
        (
            FenceDecision.QUARANTINED,
            AdmissionStatus.DEFERRED,
            AdmissionReason.QUARANTINED,
        ),
        (
            FenceDecision.STALE_HOST,
            AdmissionStatus.DEFERRED,
            AdmissionReason.STALE_HOST,
        ),
        (
            FenceDecision.LABELS,
            AdmissionStatus.DEFERRED,
            AdmissionReason.LABELS,
        ),
        (
            FenceDecision.ANTI_AFFINITY,
            AdmissionStatus.DEFERRED,
            AdmissionReason.ANTI_AFFINITY,
        ),
        (
            FenceDecision.DISK,
            AdmissionStatus.DEFERRED,
            AdmissionReason.DISK_HIGH_WATERMARK,
        ),
        (
            FenceDecision.CONCURRENCY,
            AdmissionStatus.DEFERRED,
            AdmissionReason.CONCURRENCY,
        ),
        (
            FenceDecision.EXCLUSIVITY,
            AdmissionStatus.DEFERRED,
            AdmissionReason.EXCLUSIVITY,
        ),
        (
            FenceDecision.NOT_FOUND,
            AdmissionStatus.DEFERRED,
            AdmissionReason.NATIVE_NOT_FOUND,
        ),
    ),
)
def test_all_native_fence_decisions_preserve_scheduler_reason_vocabulary(
    native_result: FenceDecision,
    expected_status: AdmissionStatus,
    expected_reason: AdmissionReason,
):
    class DecisionPort(InMemoryWorkItemReservationPort):
        def __init__(self):
            super().__init__(clock=lambda: NOW)
            self.authoritative_record = None

        def query_reservation(self, **kwargs):  # type: ignore[no-untyped-def]
            if kwargs.get("expected") is not None:
                return self.authoritative_record or FenceDecision.NOT_FOUND
            return FenceDecision.NOT_FOUND

        def atomic_reserve(self, **kwargs):  # type: ignore[no-untyped-def]
            self.authoritative_record = kwargs["reservation"]
            return native_result

    port = DecisionPort()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", fence="f")
    decision = scheduler.admit(_request("wi", "f"), now=NOW)
    assert decision.status == expected_status
    assert decision.reason_code == expected_reason


def test_priority_aging_and_fairness_prevent_group_monopoly():
    selector = FairnessSelector(
        FairnessPolicy(aging_interval_seconds=10), state=InMemoryFairnessState()
    )
    candidates = [
        QueueCandidate(f"a-{i}", fairness_group="tenant-a", priority=0, enqueued_at=NOW)
        for i in range(5)
    ] + [QueueCandidate("b-0", fairness_group="tenant-b", priority=0, enqueued_at=NOW)]
    chosen = [selector.choose(candidates, now=NOW)[0].fairness_group]
    selector.state.record("tenant-a", 1)
    remaining = [
        candidate for candidate in candidates if candidate.candidate_id != "a-0"
    ]
    chosen.append(selector.choose(remaining, now=NOW)[0].fairness_group)
    assert chosen == ["tenant-a", "tenant-b"]
    aged = selector.rank(
        QueueCandidate("old", fairness_group="tenant-a", priority=0, enqueued_at=NOW),
        now=NOW + timedelta(seconds=30),
    )
    assert aged.effective_priority == 3


def test_fairness_debt_is_shared_and_survives_selector_recreation(tmp_path: Path):
    state = InMemoryFairnessState()
    first = FairnessSelector(state=state)
    second = FairnessSelector(state=state)
    candidates = (
        QueueCandidate("a", fairness_group="a", priority=100, enqueued_at=NOW),
        QueueCandidate("b", fairness_group="b", priority=0, enqueued_at=NOW),
    )
    assert first.choose(candidates, now=NOW)[0].candidate_id == "a"
    state.record("a", 1)
    # A fresh selector sees the native/shared debt and gives the other tenant
    # its turn despite the first tenant's higher priority.
    assert second.choose(candidates, now=NOW)[0].candidate_id == "b"
    assert not second.authoritative

    path = tmp_path / "fairness.json"
    durable = JsonFairnessState(path)
    durable_first = FairnessSelector(state=durable)
    durable_first.choose(candidates, now=NOW)
    durable.record("a", 1)
    durable_second = FairnessSelector(state=JsonFairnessState(path))
    assert durable_second.served("a") == 1
    assert not durable_second.authoritative
    assert durable_second.authority is FairnessAuthority.LOCAL_ADVISORY


def test_fairness_selection_is_pure_and_native_admission_records_once():
    state = InMemoryFairnessState()
    first_selector = FairnessSelector(state=state)
    second_selector = FairnessSelector(state=state)
    candidates = (
        QueueCandidate("a", fairness_group="tenant-a", enqueued_at=NOW),
        QueueCandidate("b", fairness_group="tenant-b", enqueued_at=NOW),
    )
    # Two replicas may make the same advisory choice; selection alone creates
    # no phantom debt and does not pretend to claim the WorkItem.
    assert first_selector.choose(candidates, now=NOW)[0].candidate_id == "a"
    assert second_selector.choose(candidates, now=NOW)[0].candidate_id == "a"
    assert state.served("tenant-a") == 0

    port = _port()
    deferred_selector = FairnessSelector(state=state)
    deferred, _ = _scheduler(
        _host("local", cpu=0, memory=256, disk=256, processes=0),
        port=port,
        fairness=deferred_selector,
    )
    port.claim("deferred", fence="df", ttl_seconds=10_000, now=NOW)
    deferred_request = _request(
        "deferred",
        "df",
        resources=ResourceRequest(
            resource_class="light-check",
            fairness_group="tenant-a",
        ),
    )
    assert not deferred.admit(deferred_request, now=NOW).admitted
    assert state.served("tenant-a") == 0

    admitted_port = _port()
    admitted_selector = FairnessSelector(state=state)
    admitted, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=admitted_port,
        fairness=admitted_selector,
    )
    admitted_port.claim("admitted", fence="af", ttl_seconds=10_000, now=NOW)
    admitted_request = _request(
        "admitted",
        "af",
        resources=ResourceRequest(
            resource_class="light-check",
            fairness_group="tenant-a",
        ),
    )
    first_admission = admitted.admit(admitted_request, now=NOW)
    assert first_admission.admitted
    assert state.served("tenant-a") == 2
    assert admitted.status()["fairness_authority"] == FairnessAuthority.SIMULATION.value
    assert admitted.status()["fairness_authoritative"] is False

    # A local idempotent retry and a missing-projection replica both observe
    # the existing native link; neither increments debt a second time.
    assert admitted.admit(admitted_request, now=NOW + timedelta(seconds=1)).admitted
    replica, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=admitted_port,
        store=InMemoryReservationStore(),
        fairness=FairnessSelector(state=state),
    )
    assert replica.admit(admitted_request, now=NOW + timedelta(seconds=2)).admitted
    assert state.served("tenant-a") == 2


def test_disk_high_watermark_blocks_then_low_watermark_reopens_without_flap():
    policy = DiskPolicy()
    watermarks = DiskWatermarks(low_mib=600, high_mib=800)
    blocked = policy.evaluate(
        "local", total_mib=1_000, free_mib=150, requested_mib=0, watermarks=watermarks
    )
    assert blocked.code == DiskDecisionCode.HIGH_WATERMARK
    assert policy.state("local") == DiskState.BLOCKED
    still_blocked = policy.evaluate(
        "local", total_mib=1_000, free_mib=350, requested_mib=0, watermarks=watermarks
    )
    assert still_blocked.code == DiskDecisionCode.HIGH_WATERMARK
    reopened = policy.evaluate(
        "local", total_mib=1_000, free_mib=450, requested_mib=0, watermarks=watermarks
    )
    assert reopened.code == DiskDecisionCode.ADMIT
    assert policy.state("local") == DiskState.OPEN


def test_disk_hysteresis_is_keyed_and_missing_watermarks_cannot_clear_native_policy():
    policy = DiskPolicy()
    blocked = policy.evaluate(
        "local",
        total_mib=1_000,
        free_mib=150,
        watermarks=DiskWatermarks(
            low_mib=600, high_mib=800, policy_key="light-check:v1"
        ),
    )
    assert blocked.code == DiskDecisionCode.HIGH_WATERMARK
    # An absent request policy reuses the native key's thresholds and remains
    # blocked; it cannot silently reset state by supplying no watermarks.
    absent = policy.evaluate(
        "local",
        total_mib=1_000,
        free_mib=350,
        watermarks=None,
        policy_key="light-check:v1",
    )
    assert absent.code == DiskDecisionCode.HIGH_WATERMARK
    # A different versioned policy has independent state and cannot clear the
    # blocked v1 key.
    different = policy.evaluate(
        "local",
        total_mib=1_000,
        free_mib=450,
        watermarks=DiskWatermarks(policy_key="light-check:v2"),
    )
    assert different.code == DiskDecisionCode.ADMIT
    assert policy.state("local", policy_key="light-check:v1") == DiskState.BLOCKED


def test_profile_watermarks_cannot_be_weakened_by_caller_overrides():
    profile = ResourceProfile(
        "guarded",
        disk_low_watermark_mib=500,
        disk_high_watermark_mib=800,
    )
    weakened = profile.merge_request(
        ResourceRequest(
            resource_class="guarded",
            disk_low_watermark_mib=700,
            disk_high_watermark_mib=900,
        )
    )
    assert (weakened.disk_low_watermark_mib, weakened.disk_high_watermark_mib) == (
        500,
        800,
    )
    tightened = profile.merge_request(
        ResourceRequest(
            resource_class="guarded",
            disk_low_watermark_mib=400,
            disk_high_watermark_mib=700,
        )
    )
    assert (tightened.disk_low_watermark_mib, tightened.disk_high_watermark_mib) == (
        400,
        700,
    )


def test_disk_prediction_keeps_existing_reservations_in_watermark_budget():
    scheduler, port = _scheduler(
        _host("local", cpu=8, memory=8_192, disk=1_000, processes=8)
    )
    resources = ResourceRequest(
        resource_class="light-check",
        cpu_weight=1,
        memory_mib=256,
        disk_mib=400,
        process_slots=1,
        disk_low_watermark_mib=500,
        disk_high_watermark_mib=700,
    )
    port.claim("wi-a", fence="fa")
    port.claim("wi-b", fence="fb")
    first = scheduler.admit(
        _request("wi-a", "fa", reservation_id="disk-a", resources=resources), now=NOW
    )
    second = scheduler.admit(
        _request("wi-b", "fb", reservation_id="disk-b", resources=resources), now=NOW
    )
    assert first.admitted
    assert second.status == AdmissionStatus.DEFERRED
    assert second.reason_code == AdmissionReason.DISK_HIGH_WATERMARK
    assert scheduler.release("disk-a", work_item_id="wi-a", attempt=1, fence="fa")
    # Usage is now below low watermark, so admission resumes deterministically.
    port.claim("wi-c", fence="fc")
    reopened = scheduler.admit(
        _request("wi-c", "fc", reservation_id="disk-c", resources=resources), now=NOW
    )
    assert reopened.admitted


def test_drained_and_quarantined_hosts_receive_no_new_work():
    scheduler, port = _scheduler(_host("drained"), _host("quarantined"))
    scheduler.capacity.set_state("drained", HostState.DRAINED)
    scheduler.capacity.set_state("quarantined", HostState.QUARANTINED)
    port.claim("wi", fence="f")
    decision = scheduler.admit(_request("wi", "f"), now=NOW)
    assert not decision.admitted
    assert decision.reason_code == AdmissionReason.DRAINED
    assert scheduler.reservations.all() == ()


def test_remote_target_selects_remote_record_without_local_lease():
    scheduler, port = _scheduler(
        _host("remote-a", target_kind="remote", labels=("nodejs",)),
        _host("local"),
    )
    port.claim("wi", fence="f")
    resources = ResourceRequest(
        resource_class="light-check",
        required_target=TargetPolicy(kind=TargetKind.INVENTORY_ALIAS, alias="remote-a"),
        host_labels=("nodejs",),
    )
    decision = scheduler.admit(_request("wi", "f", resources=resources), now=NOW)
    assert decision.admitted
    assert decision.host_id == "remote-a"
    assert decision.reservation is not None
    assert decision.reservation.selected_target.kind == TargetKind.INVENTORY_ALIAS


def test_expired_reservation_reclaims_only_when_native_fence_still_authorizes():
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", fence="f", ttl_seconds=10, now=NOW)
    decision = scheduler.admit(_request("wi", "f", reservation_id="r"), now=NOW)
    assert decision.admitted
    # The reservation itself expires; the native port permits reclaim only
    # because the linked WorkItem lease has expired, not because the scheduler
    # ignored its fence.
    expired = scheduler.reclaim_expired(now=NOW + timedelta(hours=1))
    assert expired == ("r",)
    assert scheduler.capacity.reserved_for("local") == ResourceVector()


def test_superseded_attempt_can_be_reclaimed_by_current_authority_not_stale_worker():
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", attempt=1, fence="f1", ttl_seconds=100, now=NOW)
    decision = scheduler.admit(_request("wi", "f1", reservation_id="old"), now=NOW)
    assert decision.admitted
    # Attempt 2 supersedes attempt 1.  The old worker cannot release; the
    # controller's current native authority can reclaim the old reservation.
    port.rotate("wi", attempt=2, now=NOW)
    assert not scheduler.release("old", work_item_id="wi", attempt=1, fence="f1")
    assert scheduler.reclaim_expired(now=NOW + timedelta(seconds=1)) == ("old",)
    assert scheduler.reclaim_expired(now=NOW + timedelta(seconds=1)) == ("old",)
    assert scheduler.capacity.reserved_for("local") == ResourceVector()


def test_json_store_survives_service_recreation(tmp_path: Path):
    path = tmp_path / "reservations.json"
    store = JsonReservationStore(path)
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port, store=store)
    port.claim("wi", fence="f")
    decision = scheduler.admit(_request("wi", "f", reservation_id="persisted"), now=NOW)
    assert decision.admitted
    recreated = JsonReservationStore(path)
    assert recreated.get("persisted") is not None
    assert recreated.get("persisted").state.value == "reserved"


def test_scheduler_recreation_rehydrates_capacity_and_releases_through_native_port(
    tmp_path: Path,
):
    path = tmp_path / "reservations.json"
    port = _port()
    first, _ = _scheduler(
        _host("local", cpu=2, memory=512, disk=512, processes=2),
        port=port,
        store=JsonReservationStore(path),
    )
    port.claim("wi", fence="f")
    decision = first.admit(_request("wi", "f", reservation_id="survive"), now=NOW)
    assert decision.admitted
    second_store = JsonReservationStore(path)
    second, _ = _scheduler(
        _host("local", cpu=2, memory=512, disk=512, processes=2),
        port=port,
        store=second_store,
    )
    assert second.capacity.reserved_for("local") == ResourceVector(1, 256, 256, 1)
    assert second.status(now=NOW)["reservations"]
    assert second.release("survive", work_item_id="wi", attempt=1, fence="f")
    assert second.capacity.reserved_for("local") == ResourceVector()


def test_reservation_identity_is_stable_per_work_item_attempt():
    assert reservation_id_for("wi", 1) == reservation_id_for("wi", 1)
    assert reservation_id_for("wi", 1) != reservation_id_for("wi", 2)
    assert reservation_id_for("wi", 1) != reservation_id_for("other", 1)


def test_native_idempotency_deduplicates_across_missing_projection_and_preserves_link():
    port = _port()
    first, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    request = _request("wi", "f", now=NOW)
    original = first.admit(request, now=NOW)
    assert original.admitted
    assert original.reservation is not None
    assert original.reservation_id == reservation_id_for("wi", 1)

    # A second service has neither the durable reservation row nor the local
    # capacity projection, but native WorkItem state is shared.
    second, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
        store=InMemoryReservationStore(),
    )
    duplicate = second.admit(request, now=NOW + timedelta(seconds=1))
    assert duplicate.admitted
    assert duplicate.reservation_id == original.reservation_id
    assert second.capacity.reserved_for("local") == original.reservation.requirement
    assert port.reserved_for("local") == original.reservation.requirement
    assert port.link(original.reservation_id) == ("wi", 1, "f")


def test_local_active_projection_never_authorizes_when_native_query_is_missing():
    class NativeQueryMissingPort(InMemoryWorkItemReservationPort):
        def query_reservation(self, **kwargs):  # type: ignore[no-untyped-def]
            return FenceDecision.NOT_FOUND

    port = NativeQueryMissingPort(clock=lambda: NOW)
    scheduler, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4), port=port
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    request = _request("wi", "f", now=NOW)
    first = scheduler.admit(request, now=NOW)
    assert first.admitted
    retry = scheduler.admit(request, now=NOW + timedelta(seconds=1))
    assert retry.status == AdmissionStatus.DEFERRED
    assert retry.reason_code == AdmissionReason.NATIVE_NOT_FOUND
    assert port.reserved_for("local") == first.reservation.requirement


def test_release_tombstone_and_local_projection_retry_are_idempotent():
    class FailOnceUpdateStore(InMemoryReservationStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def update(self, record):  # type: ignore[no-untyped-def]
            if self.fail:
                self.fail = False
                raise OSError("simulated release projection outage")
            super().update(record)

    port = _port()
    store = FailOnceUpdateStore()
    scheduler, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
        store=store,
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    admitted = scheduler.admit(_request("wi", "f"), now=NOW)
    assert admitted.admitted
    with pytest.raises(OSError, match="release projection outage"):
        scheduler.release(
            admitted.reservation_id, work_item_id="wi", attempt=1, fence="f"
        )
    assert port.link(admitted.reservation_id) is None
    assert port.reserved_for("local") == ResourceVector()
    assert scheduler.reservations.get(admitted.reservation_id).active

    assert scheduler.release(
        admitted.reservation_id, work_item_id="wi", attempt=1, fence="f"
    )
    released = scheduler.reservations.get(admitted.reservation_id)
    assert released is not None
    assert released.state.value == "released"
    assert (
        port.query_reservation(
            reservation_id=admitted.reservation_id,
            work_item_id="wi",
            attempt=1,
            fence="f",
        ).state.value
        == "released"
    )


def test_terminal_commit_can_release_exact_attempt_and_terminal_retry_is_idempotent():
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    admitted = scheduler.admit(_request("wi", "f"), now=NOW)
    assert admitted.admitted
    port.complete("wi")

    # The same attempt/fence may release after its WorkItem commits terminal;
    # a rotated older attempt may not.
    assert scheduler.release(
        admitted.reservation_id, work_item_id="wi", attempt=1, fence="f"
    )
    assert scheduler.release(
        admitted.reservation_id, work_item_id="wi", attempt=1, fence="f"
    )
    native = port.query_reservation(
        reservation_id=admitted.reservation_id,
        work_item_id="wi",
        attempt=1,
        fence="f",
        for_lifecycle=True,
    )
    assert native.state.value == "released"
    port.rotate("wi", attempt=2, now=NOW)
    assert not scheduler.release(
        admitted.reservation_id, work_item_id="wi", attempt=1, fence="f"
    )


def test_explain_only_is_non_executable_preview_without_native_reservation():
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    preview = scheduler.admit(_request("wi", "f"), now=NOW, explain_only=True)
    assert preview.status == AdmissionStatus.PREVIEW
    assert preview.reason_code == AdmissionReason.PREVIEW
    assert preview.preview
    assert not preview.admitted
    assert preview.reservation is None
    assert port.link(preview.reservation_id) is None


def test_reclaim_tombstone_and_local_projection_retry_are_idempotent():
    class FailOnceUpdateStore(InMemoryReservationStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def update(self, record):  # type: ignore[no-untyped-def]
            if self.fail:
                self.fail = False
                raise OSError("simulated reclaim projection outage")
            super().update(record)

    port = _port()
    store = FailOnceUpdateStore()
    scheduler, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
        store=store,
    )
    port.claim("wi", fence="f", ttl_seconds=10, now=NOW)
    admitted = scheduler.admit(_request("wi", "f"), now=NOW)
    assert admitted.admitted
    with pytest.raises(OSError, match="reclaim projection outage"):
        scheduler.reclaim_expired(now=NOW + timedelta(hours=1))
    assert port.link(admitted.reservation_id) is None
    assert port.reserved_for("local") == ResourceVector()
    assert scheduler.reclaim_expired(now=NOW + timedelta(hours=1)) == (
        admitted.reservation_id,
    )
    expired = scheduler.reservations.get(admitted.reservation_id)
    assert expired is not None
    assert expired.state.value == "expired"


def test_reclaim_input_conflict_is_atomic_and_preserves_native_hold():
    port = _port()
    scheduler, _ = _scheduler(_host("local"), port=port)
    port.claim("wi", fence="f", ttl_seconds=10, now=NOW)
    admitted = scheduler.admit(_request("wi", "f"), now=NOW)
    assert admitted.admitted
    assert admitted.reservation is not None
    original = admitted.reservation
    changed = replace(original, repository_id="different-repository")

    refusal = port.atomic_reclaim_expired(
        reservation=changed,
        now=NOW + timedelta(hours=1),
    )
    assert refusal == FenceDecision.INPUT_CONFLICT
    assert port.link(original.reservation_id) == ("wi", 1, "f")
    assert port.reserved_for("local") == original.requirement
    native = port.query_reservation(
        reservation_id=original.reservation_id,
        work_item_id="wi",
        attempt=1,
        fence="f",
        for_lifecycle=True,
    )
    assert native == original

    assert scheduler.reclaim_expired(now=NOW + timedelta(hours=1)) == (
        original.reservation_id,
    )
    expired = scheduler.reservations.get(admitted.reservation_id)
    assert expired is not None
    assert expired.state.value == "expired"
    # A reconciler retry after the local projection is already terminal is an
    # exact native idempotent success, not a second capacity release.
    assert scheduler.reclaim_expired(now=NOW + timedelta(hours=1)) == (
        admitted.reservation_id,
    )


def test_projection_write_failure_leaves_native_hold_for_retry_without_double_debt():
    class FailOnceStore(InMemoryReservationStore):
        def __init__(self) -> None:
            super().__init__()
            self.fail = True

        def put(self, record):  # type: ignore[no-untyped-def]
            if self.fail:
                self.fail = False
                raise OSError("simulated projection outage")
            super().put(record)

    port = _port()
    state = InMemoryFairnessState()
    store = FailOnceStore()
    scheduler, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
        store=store,
        fairness=FairnessSelector(state=state),
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    request = _request("wi", "f")
    with pytest.raises(OSError, match="projection outage"):
        scheduler.admit(request, now=NOW)
    reservation_id = reservation_id_for("wi", 1)
    assert port.link(reservation_id) == ("wi", 1, "f")
    assert port.reserved_for("local") == ResourceVector(1, 256, 256, 1)
    assert state.served("default") == 2
    assert scheduler.capacity.reserved_for("local") == ResourceVector()

    retry = scheduler.admit(request, now=NOW + timedelta(seconds=1))
    assert retry.admitted
    assert scheduler.capacity.reserved_for("local") == ResourceVector(1, 256, 256, 1)
    assert state.served("default") == 2


def test_changed_input_conflict_cannot_compensation_release_native_original():
    port = _port()
    first, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    original = first.admit(_request("wi", "f"), now=NOW)
    assert original.admitted
    assert original.reservation is not None
    original_requirement = original.reservation.requirement

    # This replica lacks the local row and projection.  The same stable
    # attempt identity with changed immutable input must be refused by native
    # CAS without releasing the original native capacity/link.
    changed = _request("wi", "f", repository_id="different-repository")
    second, _ = _scheduler(
        _host("local", cpu=4, memory=1_024, disk=1_024, processes=4),
        port=port,
        store=InMemoryReservationStore(),
    )
    decision = second.admit(changed, now=NOW + timedelta(seconds=1))
    assert decision.status == AdmissionStatus.DEFERRED
    assert decision.reason_code == AdmissionReason.FENCE_CONFLICT
    assert second.capacity.reserved_for("local") == ResourceVector()
    assert port.reserved_for("local") == original_requirement
    assert port.link(original.reservation_id) == ("wi", 1, "f")


def test_native_disk_policy_rechecks_held_disk_when_replica_projection_is_missing():
    port = _port()
    first, _ = _scheduler(
        _host("local", cpu=8, memory=8_192, disk=1_000, processes=8),
        port=port,
    )
    resources = ResourceRequest(
        resource_class="light-check",
        cpu_weight=1,
        memory_mib=256,
        disk_mib=400,
        process_slots=1,
        disk_low_watermark_mib=500,
        disk_high_watermark_mib=700,
    )
    port.claim("wi-a", fence="fa", ttl_seconds=10_000, now=NOW)
    first_decision = first.admit(_request("wi-a", "fa", resources=resources), now=NOW)
    assert first_decision.admitted

    second, _ = _scheduler(
        _host("local", cpu=8, memory=8_192, disk=1_000, processes=8),
        port=port,
        store=InMemoryReservationStore(),
    )
    port.claim("wi-b", fence="fb", ttl_seconds=10_000, now=NOW)
    second_decision = second.admit(_request("wi-b", "fb", resources=resources), now=NOW)
    assert second_decision.status == AdmissionStatus.DEFERRED
    assert second_decision.reason_code == AdmissionReason.DISK_HIGH_WATERMARK
    assert port.reserved_for("local").disk_mib == 400


def test_native_concurrency_and_exclusivity_recheck_missing_replica_projection():
    port = _port()
    first, _ = _scheduler(
        _host("local", cpu=64, memory=100_000, disk=100_000, processes=32),
        port=port,
    )
    port.claim("frontend-a", fence="fa", ttl_seconds=10_000, now=NOW)
    frontend = first.admit(
        _request("frontend-a", "fa", resource_class="frontend-build"),
        now=NOW,
    )
    assert frontend.admitted

    second, _ = _scheduler(
        _host("local", cpu=64, memory=100_000, disk=100_000, processes=32),
        port=port,
        store=InMemoryReservationStore(),
    )
    port.claim("frontend-b", fence="fb", ttl_seconds=10_000, now=NOW)
    blocked_frontend = second.admit(
        _request("frontend-b", "fb", resource_class="frontend-build"),
        now=NOW,
    )
    assert blocked_frontend.reason_code == AdmissionReason.CONCURRENCY

    exclusive_profiles = default_resource_profiles()
    exclusive_profiles.register(
        ResourceProfile(
            "exclusive-repo",
            cpu_weight=2,
            memory_mib=1_024,
            disk_mib=512,
            process_slots=1,
            concurrency_key="exclusive-repo",
            repository_exclusive=True,
        )
    )
    exclusive_port = _port()
    exclusive_first, _ = _scheduler(
        _host("local", cpu=64, memory=100_000, disk=100_000, processes=32),
        port=exclusive_port,
        profiles=exclusive_profiles,
    )
    exclusive_port.claim("merge-a", fence="ma", ttl_seconds=10_000, now=NOW)
    merged = exclusive_first.admit(
        _request(
            "merge-a", "ma", resource_class="exclusive-repo", repository_id="repo"
        ),
        now=NOW,
    )
    assert merged.admitted
    exclusive_second, _ = _scheduler(
        _host("local", cpu=64, memory=100_000, disk=100_000, processes=32),
        port=exclusive_port,
        store=InMemoryReservationStore(),
        profiles=exclusive_profiles,
    )
    exclusive_port.claim("merge-b", fence="mb", ttl_seconds=10_000, now=NOW)
    blocked_merge = exclusive_second.admit(
        _request(
            "merge-b",
            "mb",
            resource_class="exclusive-repo",
            repository_id="repo",
        ),
        now=NOW,
    )
    assert blocked_merge.reason_code == AdmissionReason.EXCLUSIVITY


@pytest.mark.parametrize(
    ("profile_name", "repository_exclusive", "branch_exclusive"),
    (
        ("global-repository", True, False),
        ("global-branch", False, True),
    ),
)
def test_repository_and_branch_exclusivity_are_global_across_hosts(
    profile_name: str, repository_exclusive: bool, branch_exclusive: bool
):
    profiles = default_resource_profiles()
    profiles.register(
        ResourceProfile(
            profile_name,
            cpu_weight=2,
            memory_mib=1_024,
            disk_mib=512,
            process_slots=1,
            concurrency_key=profile_name,
            repository_exclusive=repository_exclusive,
            branch_exclusive=branch_exclusive,
        )
    )
    port = _port()
    first, _ = _scheduler(
        _host("host-a"), _host("host-b"), port=port, profiles=profiles
    )
    port.claim("first", fence="f1", ttl_seconds=10_000, now=NOW)
    first_request = _request(
        "first",
        "f1",
        resource_class=profile_name,
        repository_id="repo",
        branch="main",
    )
    first_decision = first.admit(first_request, now=NOW)
    assert first_decision.admitted
    assert first_decision.host_id == "host-a"

    # The second scheduler has no local row.  Requiring host-b proves the
    # native repository/branch key is global, while anti-affinity remains
    # host-local.
    second, _ = _scheduler(
        _host("host-a"),
        _host("host-b"),
        port=port,
        store=InMemoryReservationStore(),
        profiles=profiles,
    )
    port.claim("second", fence="f2", ttl_seconds=10_000, now=NOW)
    second_request = _request(
        "second",
        "f2",
        resource_class=profile_name,
        repository_id="repo",
        branch="main",
        resources=ResourceRequest(
            resource_class=profile_name,
            required_target=TargetPolicy(
                kind=TargetKind.INVENTORY_ALIAS, alias="host-b"
            ),
        ),
    )
    blocked = second.admit(second_request, now=NOW)
    assert blocked.status == AdmissionStatus.DEFERRED
    assert blocked.reason_code == AdmissionReason.EXCLUSIVITY


def test_capacity_refresh_is_monotonic_and_preserves_held_accounting():
    inventory = CapacityInventory([_host("local", cpu=4, memory=1_024, disk=1_024)])
    held = ResourceVector(1, 256, 256, 1)
    assert inventory.try_reserve("local", "held", held, now=NOW)
    newer = _host(
        "local",
        cpu=4,
        memory=1_024,
        disk=1_024,
        heartbeat_at=NOW + timedelta(seconds=5),
    )
    newer = replace(newer, version=2, state=HostState.DRAINING)
    assert inventory.refresh(newer)
    assert inventory.get("local").version == 2
    assert inventory.reserved_for("local") == held

    stale = replace(newer, version=1, state=HostState.ACTIVE)
    assert not inventory.refresh(stale)
    assert inventory.get("local").version == 2
    assert inventory.get("local").state == HostState.DRAINING

    port = _port()
    scheduler, _ = _scheduler(
        _host("native", cpu=4, memory=1_024, disk=1_024), port=port
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    decision = scheduler.admit(_request("wi", "f"), now=NOW)
    assert decision.admitted
    authoritative = port._authoritative_view("native")  # simulation introspection
    assert authoritative is not None
    refresh = replace(
        authoritative,
        version=2,
        state=HostState.DRAINING,
        heartbeat_at=NOW + timedelta(seconds=5),
    )
    assert port.register_capacity_view(refresh)
    assert port.reserved_for("native") == decision.reservation.requirement
    stale_refresh = replace(refresh, version=1, state=HostState.ACTIVE)
    assert not port.register_capacity_view(stale_refresh)
    assert port._authoritative_view("native").state == HostState.DRAINING


def test_heartbeat_and_state_replays_are_monotonic_and_preserve_holds():
    inventory = CapacityInventory([_host("local", cpu=4, memory=1_024, disk=1_024)])
    held = ResourceVector(1, 256, 256, 1)
    assert inventory.try_reserve("local", "held", held, now=NOW)
    original = inventory.get("local")
    assert original is not None

    # Explicit equal/stale revisions are no-ops even when the payload differs.
    equal_heartbeat = inventory.heartbeat(
        "local",
        version=original.version,
        at=NOW + timedelta(minutes=1),
        live=ResourceVector(2, 0, 0, 0),
        observed_disk_free_mib=1,
    )
    assert equal_heartbeat == original
    assert inventory.reserved_for("local") == held

    implicit_heartbeat = inventory.heartbeat(
        "local",
        at=NOW + timedelta(seconds=1),
        live=ResourceVector(1, 0, 0, 0),
        observed_disk_free_mib=900,
    )
    assert implicit_heartbeat.version == original.version + 1
    assert implicit_heartbeat.live == ResourceVector(1, 0, 0, 0)
    assert inventory.reserved_for("local") == held

    stale_heartbeat = inventory.heartbeat(
        "local",
        version=original.version,
        at=NOW + timedelta(minutes=2),
        live=ResourceVector(3, 0, 0, 0),
        observed_disk_free_mib=2,
    )
    assert stale_heartbeat == implicit_heartbeat
    assert inventory.reserved_for("local") == held

    equal_state = inventory.set_state(
        "local", HostState.DRAINED, version=implicit_heartbeat.version
    )
    assert equal_state == implicit_heartbeat
    implicit_state = inventory.set_state("local", HostState.DRAINED)
    assert implicit_state.version == implicit_heartbeat.version + 1
    assert implicit_state.state is HostState.DRAINED
    stale_state = inventory.set_state(
        "local", HostState.ACTIVE, version=original.version
    )
    assert stale_state == implicit_state
    assert inventory.reserved_for("local") == held


@pytest.mark.parametrize(
    ("state", "heartbeat_at"),
    (
        (HostState.DRAINED, NOW),
        (HostState.QUARANTINED, NOW),
        (HostState.ACTIVE, NOW - timedelta(hours=1)),
    ),
)
def test_restart_restores_held_accounting_on_ineligible_host(
    tmp_path: Path, state: HostState, heartbeat_at: datetime
):
    path = tmp_path / f"{state.value}-reservations.json"
    port = _port()
    first, _ = _scheduler(
        _host("local", cpu=2, memory=512, disk=512, processes=2),
        port=port,
        store=JsonReservationStore(path),
    )
    port.claim("wi", fence="f", ttl_seconds=10_000, now=NOW)
    original = first.admit(_request("wi", "f"), now=NOW)
    assert original.admitted
    assert original.reservation is not None

    recreated, _ = _scheduler(
        _host(
            "local",
            cpu=2,
            memory=512,
            disk=512,
            processes=2,
            state=state,
            heartbeat_at=heartbeat_at,
            heartbeat_ttl_seconds=10,
            free_disk=0,
        ),
        port=port,
        store=JsonReservationStore(path),
    )
    assert recreated.capacity.reserved_for("local") == original.reservation.requirement
    host_status = recreated.status(now=NOW)["hosts"]
    assert isinstance(host_status, list)
    assert host_status[0]["state"] == state.value
    assert recreated.release(
        original.reservation_id,
        work_item_id="wi",
        attempt=1,
        fence="f",
        at=NOW,
    )
    assert recreated.capacity.reserved_for("local") == ResourceVector()
