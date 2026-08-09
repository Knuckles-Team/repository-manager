"""Focused RMDD-27 production reservation-port binding tests."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from repository_manager.capacity import ResourceVector
from repository_manager.development import ReservationState, TargetPolicy
from repository_manager.native_reservations import (
    DurableJobFenceCodec,
    NativeReservationProtocolError,
    NativeResourceReservationUnavailable,
    NativeWorkItemReservationPort,
)
from repository_manager.reservations import FenceDecision, ReservationRecord
from repository_manager.resource_profiles import (
    ResourceProfile,
    ResourceProfileRegistry,
    default_resource_profiles,
)
from repository_manager.resource_scheduler import create_production_resource_scheduler

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _resolved_test_profiles() -> ResourceProfileRegistry:
    return ResourceProfileRegistry(
        {
            "light-check": ResourceProfile(
                "light-check",
                concurrency_limit=4,
                required_labels=("linux",),
                anti_affinity=("gpu",),
                repository_exclusive=True,
                disk_low_watermark_mib=100,
                disk_high_watermark_mib=200,
            )
        }
    )


def _record(
    *, state: ReservationState = ReservationState.RESERVED, revision: int = 1
) -> ReservationRecord:
    return ReservationRecord(
        reservation_id="reservation:one",
        work_item_id="workitem:one",
        attempt=2,
        fence="11",
        host_id="host-a",
        profile_name="light-check",
        requirement=ResourceVector(2, 512, 1024, 2),
        capacity_snapshot={"version": 4},
        selected_target=TargetPolicy(),
        concurrency_key="light-check",
        concurrency_limit=4,
        repository_exclusive=True,
        branch_exclusive=False,
        required_labels=("linux",),
        disk_low_watermark_mib=100,
        disk_high_watermark_mib=200,
        disk_policy_key="light-check:v1",
        repository_id="repo-one",
        branch="main",
        owner_id="worker-one",
        tenant_id="tenant-one",
        fairness_group="default",
        fairness_cost=4,
        anti_affinity=("gpu",),
        reserved_at=NOW,
        expires_at=NOW + timedelta(minutes=10),
        state=state,
        revision=revision,
        input_fingerprint="v1:" + "a" * 64,
    )


def _native_record(
    record: ReservationRecord, *, revision: int | None = None
) -> dict[str, object]:
    state = {
        ReservationState.RESERVED: "reserved",
        ReservationState.RELEASED: "released",
        ReservationState.EXPIRED: "expired",
    }[record.state]
    return {
        "reservation_id": record.reservation_id,
        "tenant_ref": record.tenant_id,
        "owner_id": record.owner_id,
        "work_item_id": record.work_item_id,
        "fence": record.fence,
        "lease_epoch": 11,
        "fencing_token": 11,
        "attempt": record.attempt,
        "input_fingerprint": record.input_fingerprint,
        "host_ref": record.host_id,
        "profile_name": record.profile_name,
        "profile_version": "1",
        "requirement": record.requirement.as_dict(),
        "capacity_snapshot": {
            "cpu_weight": 2,
            "memory_mib": 512,
            "disk_mib": 1024,
            "process_slots": 2,
            "host_revision": 4,
        },
        "target_kind": "local",
        "target_alias": None,
        "selected_target": {
            "kind": "local",
            "alias": None,
            "capability_labels": list(record.selected_target.capability_labels),
        },
        "repository_id": record.repository_id,
        "branch": record.branch,
        "concurrency_key": record.concurrency_key,
        "concurrency_limit": record.concurrency_limit,
        "repository_exclusive": record.repository_exclusive,
        "branch_exclusive": record.branch_exclusive,
        "required_labels": list(record.required_labels),
        "anti_affinity": list(record.anti_affinity),
        "fairness_group": record.fairness_group,
        "fairness_cost": record.fairness_cost,
        "disk_low_watermark_mib": record.disk_low_watermark_mib,
        "disk_high_watermark_mib": record.disk_high_watermark_mib,
        "disk_policy_key": record.disk_policy_key,
        "reserved_at_ms": int(record.reserved_at.timestamp() * 1000),
        "expires_at_ms": int(record.expires_at.timestamp() * 1000),
        "state": state,
        "revision": revision or record.revision,
        "lifecycle_revision": revision or record.revision,
        "tombstone": record.state is not ReservationState.RESERVED,
    }


def _native_result(
    record: ReservationRecord,
    *,
    decision: str = "accepted",
    result_record: dict[str, object] | None = None,
    revision: int | None = None,
    changed: bool = True,
) -> dict[str, object]:
    result_record = (
        result_record
        if result_record is not None
        else _native_record(record, revision=revision)
    )
    state = str(result_record["state"])
    return {
        "schema_version": "1",
        "decision": decision,
        "reservation_id": record.reservation_id,
        "work_item_id": record.work_item_id,
        "attempt": record.attempt,
        "lease_epoch": 11,
        "fencing_token": 11,
        "lifecycle_revision": int(str(result_record["lifecycle_revision"])),
        "host_ref": record.host_id,
        "host_revision": 4,
        "record": result_record,
        "state": state,
        "held_cpu_weight": 0 if result_record["tombstone"] else 2,
        "held_memory_mib": 0 if result_record["tombstone"] else 512,
        "held_disk_mib": 0 if result_record["tombstone"] else 1024,
        "held_process_slots": 0 if result_record["tombstone"] else 2,
        "fairness_debt": 3,
        "tombstone": bool(result_record["tombstone"]),
        "changed_work_item_ids": [record.work_item_id] if changed else [],
    }


class FakeNativeClient:
    def __init__(self, record: ReservationRecord) -> None:
        self.record = record
        self.requests: list[tuple[str, Mapping[str, object]]] = []
        self.next_result: dict[str, object] | None = None

    def supports(self, operation: str) -> bool:
        return operation in {
            "ReserveWorkItemResources",
            "ReleaseWorkItemResources",
            "ReclaimWorkItemResources",
            "QueryWorkItemReservation",
            "ResourceReservationStatus",
            "UpdateResourceHost",
        }

    def reserve(self, request: Mapping[str, object]) -> dict[str, object]:
        self.requests.append(("reserve", request))
        return self.next_result or _native_result(self.record)

    def release(self, request: Mapping[str, object]) -> dict[str, object]:
        self.requests.append(("release", request))
        released = _record(state=ReservationState.RELEASED, revision=2)
        return _native_result(
            self.record, result_record=_native_record(released), revision=2
        )

    def reclaim(self, request: Mapping[str, object]) -> dict[str, object]:
        self.requests.append(("reclaim", request))
        expired = _record(state=ReservationState.EXPIRED, revision=2)
        return _native_result(
            self.record, result_record=_native_record(expired), revision=2
        )

    def query_reservation(self, request: Mapping[str, object]) -> dict[str, object]:
        self.requests.append(("query_reservation", request))
        return _native_result(self.record)

    def status(self, request: Mapping[str, object]) -> dict[str, object]:
        self.requests.append(("status", request))
        return {
            "schema_version": "1",
            "complete": True,
            "next_cursor": None,
            "host_ref": None,
            "host_revision": 4,
            "held_cpu_weight": 2,
            "held_memory_mib": 512,
            "held_disk_mib": 1024,
            "held_process_slots": 2,
            "fairness_debt": 3,
            "reservations": [],
            "orphan_count": 0,
            "superseded_count": 0,
        }

    def update_host(self, request: Mapping[str, object]) -> dict[str, object]:
        self.requests.append(("update_host", request))
        return {
            "schema_version": "1",
            "accepted": True,
            "reason": "accepted",
            "host_ref": request["host_ref"],
            "revision": request["revision"],
            "held_cpu_weight": 2,
            "held_memory_mib": 512,
            "held_disk_mib": 1024,
            "held_process_slots": 2,
            "draining": request["draining"],
            "quarantined": request["quarantined"],
        }


def _port(client: FakeNativeClient) -> NativeWorkItemReservationPort:
    return NativeWorkItemReservationPort(
        client,
        tenant_ref="tenant-one",
        owner_id="worker-one",
        fence_codec=DurableJobFenceCodec(),
        profiles=_resolved_test_profiles(),
        clock=lambda: NOW,
    )


def test_reserve_maps_full_request_and_fixed_decision() -> None:
    record = _record()
    client = FakeNativeClient(record)
    assert (
        _port(client).atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )
        is FenceDecision.ACCEPTED
    )
    operation, request = client.requests[-1]
    assert operation == "reserve"
    assert request["fence"] == record.fence
    assert request["lease_epoch"] == 11
    assert request["fencing_token"] == 11
    assert request["now_ms"] == int(NOW.timestamp() * 1000)
    assert request["input_fingerprint"] == record.input_fingerprint


def test_reserve_uses_current_clock_on_retry() -> None:
    record = _record()
    retry_now = NOW + timedelta(minutes=3)
    client = FakeNativeClient(record)
    port = NativeWorkItemReservationPort(
        client,
        tenant_ref=record.tenant_id,
        owner_id=record.owner_id,
        fence_codec=DurableJobFenceCodec(),
        profiles=_resolved_test_profiles(),
        clock=lambda: retry_now,
    )
    port.atomic_reserve(
        work_item_id=record.work_item_id,
        attempt=record.attempt,
        fence=record.fence,
        reservation=record,
    )
    assert client.requests[-1][1]["now_ms"] == int(retry_now.timestamp() * 1000)


def test_durable_job_fence_codec_preserves_deployed_projection() -> None:
    codec = DurableJobFenceCodec()
    codec.validate("11", 11, 11)
    with pytest.raises(NativeReservationProtocolError):
        codec.validate("f:7/11", 7, 11)


def test_is_current_uses_exact_native_query_correlations() -> None:
    record = _record()
    client = FakeNativeClient(record)
    assert _port(client).is_current(record.work_item_id, record.attempt, record.fence)
    operation, request = client.requests[-1]
    assert operation == "query_reservation"
    assert request["reservation_id"] is None
    assert request["owner_id"] == record.owner_id
    assert request["fence"] == record.fence
    assert request["attempt"] == record.attempt
    assert request["lease_epoch"] == 11
    assert request["fencing_token"] == 11
    assert request["input_fingerprint"] is None


def test_identity_mismatch_is_rejected_before_native_mutation() -> None:
    record = _record()
    client = FakeNativeClient(record)
    mismatched = replace(record, tenant_id="other-tenant")
    with pytest.raises(NativeReservationProtocolError, match="configured tenant"):
        _port(client).atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=mismatched,
        )
    assert client.requests == []


def test_missing_trusted_profile_fails_closed_before_mutation() -> None:
    record = _record()
    client = FakeNativeClient(record)
    port = NativeWorkItemReservationPort(
        client,
        tenant_ref=record.tenant_id,
        owner_id=record.owner_id,
        fence_codec=DurableJobFenceCodec(),
        profiles=ResourceProfileRegistry(),
        clock=lambda: NOW,
    )
    with pytest.raises(NativeResourceReservationUnavailable):
        port.atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )
    assert client.requests == []


@pytest.mark.parametrize(
    "changed",
    [
        {"requirement": ResourceVector(1, 128, 128, 1)},
        {"concurrency_key": "unsafe-key"},
        {"fairness_cost": 1},
        {"disk_policy_key": "light-check:v0"},
    ],
)
def test_direct_port_rejects_underdeclared_trusted_profile(
    changed: dict[str, object],
) -> None:
    record = _record()
    client = FakeNativeClient(record)
    underdeclared = replace(record, **changed)
    with pytest.raises(NativeReservationProtocolError, match="profile|RMDD-08"):
        _port(client).atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=underdeclared,
        )
    assert client.requests == []


def test_from_graph_client_requires_generated_sync_capability_probe() -> None:
    record = _record()
    client = FakeNativeClient(record)

    class NamespaceOnlyClient:
        work_items = client

    with pytest.raises(NativeResourceReservationUnavailable):
        NativeWorkItemReservationPort.from_graph_client(
            NamespaceOnlyClient(),
            tenant_ref=record.tenant_id,
            owner_id=record.owner_id,
            fence_codec=DurableJobFenceCodec(),
            profiles=default_resource_profiles(),
        )


def test_from_graph_client_binds_generated_sync_namespace() -> None:
    record = _record()
    native = FakeNativeClient(record)

    class GeneratedSyncClientShape:
        work_items = native

        @staticmethod
        def supports(operation: str) -> bool:
            return native.supports(operation)

    port = NativeWorkItemReservationPort.from_graph_client(
        GeneratedSyncClientShape(),
        tenant_ref=record.tenant_id,
        owner_id=record.owner_id,
        fence_codec=DurableJobFenceCodec(),
        profiles=default_resource_profiles(),
        clock=lambda: NOW,
    )
    assert port.is_current(record.work_item_id, record.attempt, record.fence)


def test_actual_generated_sync_wrapper_shape_binds_without_async_bridge() -> None:
    client_module = pytest.importorskip("epistemic_graph.client")
    sync_type = getattr(client_module, "SyncEpistemicGraphClient", None)
    if sync_type is None:
        pytest.skip("generated sync client is unavailable")
    record = _record()
    native = FakeNativeClient(record)
    generated = object.__new__(sync_type)
    generated.work_items = native
    generated.supports = native.supports
    port = NativeWorkItemReservationPort.from_graph_client(
        generated,
        tenant_ref=record.tenant_id,
        owner_id=record.owner_id,
        fence_codec=DurableJobFenceCodec(),
        profiles=default_resource_profiles(),
        clock=lambda: NOW,
    )
    assert port.is_current(record.work_item_id, record.attempt, record.fence)


def test_public_scheduler_factory_has_no_in_memory_fallback() -> None:
    record = _record()
    native = FakeNativeClient(record)

    class GeneratedSyncClientShape:
        work_items = native

        @staticmethod
        def supports(operation: str) -> bool:
            return native.supports(operation)

    scheduler = create_production_resource_scheduler(
        GeneratedSyncClientShape(),
        tenant_ref=record.tenant_id,
        owner_id=record.owner_id,
        fence_codec=DurableJobFenceCodec(),
        profiles=_resolved_test_profiles(),
        clock=lambda: NOW,
    )
    assert isinstance(scheduler.work_item_port, NativeWorkItemReservationPort)
    assert native.requests == []


def test_capacity_refusal_gets_fresh_key_before_later_acceptance() -> None:
    record = _record()

    class CapacityThenAccepted(FakeNativeClient):
        def reserve(self, request: Mapping[str, object]) -> dict[str, object]:
            self.requests.append(("reserve", request))
            if sum(operation == "reserve" for operation, _ in self.requests) == 1:
                refusal = _native_result(record)
                refusal.update(
                    {
                        "decision": "capacity",
                        "reservation_id": None,
                        "record": None,
                        "state": "absent",
                        "tombstone": False,
                        "changed_work_item_ids": [],
                    }
                )
                return refusal
            return _native_result(record)

    client = CapacityThenAccepted(record)
    port = _port(client)
    assert (
        port.atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )
        is FenceDecision.CAPACITY
    )
    assert (
        port.atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )
        is FenceDecision.ACCEPTED
    )
    keys = [
        request["idempotency_key"]
        for operation, request in client.requests
        if operation == "reserve"
    ]
    assert len(keys) == 2
    assert keys[0] != keys[1]


def test_query_reconstructs_native_record_and_preserves_revision() -> None:
    record = _record()
    native = _port(FakeNativeClient(record)).query_reservation(
        reservation_id=record.reservation_id,
        work_item_id=record.work_item_id,
        attempt=record.attempt,
        fence=record.fence,
        expected=record,
    )
    assert isinstance(native, ReservationRecord)
    assert native.revision == record.revision
    assert native.fence == record.fence
    assert native.requirement == record.requirement


def test_release_and_reclaim_require_native_tombstones() -> None:
    record = _record()
    client = FakeNativeClient(record)
    port = _port(client)
    assert (
        port.atomic_release(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation_id=record.reservation_id,
            reservation=record,
        )
        is FenceDecision.ACCEPTED
    )
    assert (
        port.atomic_reclaim(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation_id=record.reservation_id,
            reservation=record,
            now=NOW,
        )
        is FenceDecision.ACCEPTED
    )
    assert [operation for operation, _ in client.requests] == ["release", "reclaim"]


def test_release_uses_release_clock_not_reservation_start() -> None:
    record = _record()
    client = FakeNativeClient(record)
    release_now = NOW + timedelta(minutes=3)
    port = NativeWorkItemReservationPort(
        client,
        tenant_ref="tenant-one",
        owner_id="worker-one",
        fence_codec=DurableJobFenceCodec(),
        profiles=_resolved_test_profiles(),
        clock=lambda: release_now,
    )
    port.atomic_release(
        work_item_id=record.work_item_id,
        attempt=record.attempt,
        fence=record.fence,
        reservation_id=record.reservation_id,
        reservation=record,
    )
    assert client.requests[-1][1]["now_ms"] == int(release_now.timestamp() * 1000)


def test_mismatched_accepted_record_fails_closed_without_local_compensation() -> None:
    record = _record()
    client = FakeNativeClient(record)
    changed = _native_record(record)
    changed["owner_id"] = "other-worker"
    client.next_result = _native_result(record, result_record=changed)
    with pytest.raises(NativeReservationProtocolError, match="owner mismatch"):
        _port(client).atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )


def test_selected_target_observation_is_required_and_correlated() -> None:
    record = _record()
    client = FakeNativeClient(record)
    changed = _native_record(record)
    changed["selected_target"] = {
        "kind": "local",
        "alias": None,
        "capability_labels": ["unexpected"],
    }
    client.next_result = _native_result(record, result_record=changed)
    with pytest.raises(NativeReservationProtocolError, match="selected target"):
        _port(client).atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )


def test_old_engine_fails_closed_without_fixture_fallback() -> None:
    record = _record()

    class OldEngineClient(FakeNativeClient):
        def supports(self, _operation: str) -> bool:
            return False

    client = OldEngineClient(record)
    with pytest.raises(NativeResourceReservationUnavailable):
        _port(client).atomic_reserve(
            work_item_id=record.work_item_id,
            attempt=record.attempt,
            fence=record.fence,
            reservation=record,
        )


def test_status_pagination_rejects_unbounded_cursor() -> None:
    record = _record()

    class LoopingStatusClient(FakeNativeClient):
        def status(self, _request: Mapping[str, object]) -> dict[str, object]:
            return {
                "schema_version": "1",
                "complete": False,
                "next_cursor": "same",
                "host_ref": "host-a",
                "host_revision": 4,
                "held_cpu_weight": 0,
                "held_memory_mib": 0,
                "held_disk_mib": 0,
                "held_process_slots": 0,
                "fairness_debt": 0,
                "reservations": [],
                "orphan_count": 0,
                "superseded_count": 0,
            }

    client = LoopingStatusClient(record)
    with pytest.raises(NativeReservationProtocolError, match="cursor"):
        tuple(_port(client).status_pages(host_ref="host-a", max_pages=2))


def test_status_rejects_page_larger_than_limit() -> None:
    record = _record()

    class OversizedStatusClient(FakeNativeClient):
        def status(self, _request: Mapping[str, object]) -> dict[str, object]:
            return {
                "schema_version": "1",
                "complete": True,
                "next_cursor": None,
                "host_ref": "host-a",
                "host_revision": 4,
                "held_cpu_weight": 0,
                "held_memory_mib": 0,
                "held_disk_mib": 0,
                "held_process_slots": 0,
                "fairness_debt": 0,
                "reservations": [{}, {}],
                "orphan_count": 0,
                "superseded_count": 0,
            }

    with pytest.raises(NativeReservationProtocolError, match="reservations"):
        _port(OversizedStatusClient(record)).status(host_ref="host-a", limit=1)


def test_host_update_rejects_impossible_disk_telemetry() -> None:
    record = _record()
    with pytest.raises(ValueError, match="disk used"):
        _port(FakeNativeClient(record)).update_host(
            host_ref="host-a",
            revision=5,
            capacity=ResourceVector(4, 1024, 100, 4),
            observed=ResourceVector(1, 256, 10, 1),
            heartbeat_at=NOW,
            heartbeat_ttl_ms=120_000,
            target_kind="local",
            disk_used_mib=101,
            disk_capacity_mib=100,
        )


def test_host_update_maps_complete_native_telemetry_projection() -> None:
    record = _record()
    client = FakeNativeClient(record)
    _port(client).update_host(
        host_ref="host-a",
        revision=5,
        capacity=ResourceVector(4, 1024, 100, 4),
        observed=ResourceVector(1, 256, 10, 1),
        heartbeat_at=NOW,
        heartbeat_ttl_ms=120_000,
        target_kind="local",
        now=NOW,
    )
    request = client.requests[-1][1]
    assert request["heartbeat_ttl_ms"] == 120_000
    assert request["target_kind"] == "local"
    assert request["target_alias"] is None
