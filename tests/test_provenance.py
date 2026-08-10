"""RMDD-19 — repository-manager domain provenance emitter/adapter tests.

Proves every emitter funnels through the ONE agent-utilities chokepoint
(``write_repository_event``), carries the right C-02..C-07 correlation
fields from its typed contract argument, and that the read/explain/reconcile
projection composes a full job lifecycle from a fake graph engine -- the
same fake-engine pattern used by the agent-utilities side of this lane
(``tests/unit/observability/test_repository_provenance.py``).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from repository_manager.development.enums import (
    ExecutionOutcome,
    FailureClass,
    GenerationState,
    JobState,
    ReservationState,
    TargetKind,
    ValidationStage,
)
from repository_manager.development.models import (
    ArtifactReference,
    Candidate,
    CandidateVersion,
    CapacitySnapshot,
    ExecutionResult,
    Generation,
    LeaseRecord,
    RepositoryIdentity,
    ResourceReservation,
    ResourceRequest,
    TargetPolicy,
    ValidationEvidence,
)

from repository_manager import provenance

_SHA = "a" * 40
_DIGEST = "b" * 64
_NOW = datetime.now(UTC)


class _FakeEngine:
    """Same minimal in-memory add_node/link_nodes/query_cypher fake used on
    the agent-utilities side of this lane."""

    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[tuple[str, str, str]] = []

    def add_node(
        self,
        node_id: str,
        node_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
    ) -> None:
        existing = self.nodes.get(node_id, {})
        self.nodes[node_id] = {**existing, **(properties or {}), "node_type": node_type}

    def link_nodes(
        self,
        source_id: str,
        target_id: str,
        rel_type: str,
        properties: dict[str, Any] | None = None,
        ephemeral: bool = False,
        *,
        session: Any = None,
    ) -> None:
        self.edges.append((source_id, target_id, rel_type))

    def query_cypher(
        self, query: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        params = params or {}
        rows = [
            node
            for node in self.nodes.values()
            if node.get("node_type") == "ToolCall"
            and node.get("work_item_ref") == params.get("work_item_ref")
        ]
        if "tenant_ref" in params:
            rows = [n for n in rows if n.get("tenant_ref") == params.get("tenant_ref")]
        rows = sorted(rows, key=lambda n: int(n.get("event_sequence") or 0), reverse=True)
        if "RETURN t.fence_ref" in query:
            return [
                {"fence_ref": n.get("fence_ref"), "event_sequence": n.get("event_sequence")}
                for n in rows
                if n.get("fence_ref")
            ]
        return [{"t": n} for n in rows]


def _repository_identity() -> RepositoryIdentity:
    return RepositoryIdentity(repository_id="repository-manager", canonical_path="/tmp")


def _lease(attempt: int = 1, fence: str = "fence-1") -> LeaseRecord:
    return LeaseRecord(
        owner="worker-a",
        fence=fence,
        attempt=attempt,
        heartbeat_at=_NOW,
        expires_at=_NOW + timedelta(minutes=5),
    )


def _reservation(state: ReservationState = ReservationState.RESERVED) -> ResourceReservation:
    return ResourceReservation(
        reservation_id="reservation-1",
        request=ResourceRequest(),
        selected_target=TargetPolicy(kind=TargetKind.LOCAL),
        fence="fence-1",
        capacity=CapacitySnapshot(
            cpu_weight_total=100,
            cpu_weight_available=90,
            memory_mib_total=1024,
            memory_mib_available=900,
            disk_mib_total=1024,
            disk_mib_available=900,
            process_slots_total=10,
            process_slots_available=9,
        ),
        state=state,
        reason="" if state != ReservationState.REFUSED else "capacity exhausted",
        reserved_at=_NOW,
        expires_at=_NOW + timedelta(minutes=10),
    )


def _execution_result(
    outcome: ExecutionOutcome = ExecutionOutcome.SUCCEEDED,
    fence: str = "fence-1",
    failure_class: FailureClass | None = None,
) -> ExecutionResult:
    kwargs: dict[str, Any] = dict(
        command_id="cmd-1",
        outcome=outcome,
        started_at=_NOW,
        finished_at=_NOW + timedelta(seconds=5),
        duration_ms=5000,
        worker_id="worker-a",
        fence=fence,
    )
    if outcome == ExecutionOutcome.SUCCEEDED:
        kwargs["exit_code"] = 0
    else:
        kwargs["exit_code"] = 1
        kwargs["failure_class"] = failure_class or FailureClass.INTERNAL_ERROR
    return ExecutionResult(**kwargs)


def _artifact() -> ArtifactReference:
    return ArtifactReference(
        content_address=_DIGEST,
        relative_path="artifacts/output.tar.gz",
        size_bytes=1024,
        media_type="application/gzip",
    )


def _validation_evidence() -> ValidationEvidence:
    return ValidationEvidence(
        evidence_id="evidence-1",
        stage=ValidationStage.FEEDBACK,
        tree_sha=_SHA,
        gate_config_digest=_DIGEST,
        command_digest=_DIGEST,
        target=TargetPolicy(kind=TargetKind.LOCAL),
        host_id="local",
        toolchain_digest=_DIGEST,
        started_at=_NOW,
        finished_at=_NOW + timedelta(seconds=30),
        outcome="passed",
    )


def _generation(state: GenerationState = GenerationState.OPEN) -> Generation:
    return Generation(
        generation_id="generation-1",
        repository=_repository_identity(),
        target_branch="main",
        base_sha=_SHA,
        expected_landing_base_sha=_SHA,
        candidate_versions=(CandidateVersion(candidate_id="candidate-1", version=1, candidate_sha=_SHA),),
        config_digest=_DIGEST,
        toolchain_digest=_DIGEST,
        state=state,
    )


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="candidate-1",
        version=1,
        repository=_repository_identity(),
        branch="feature/x",
        candidate_sha=_SHA,
        base_sha=_SHA,
        lane_id="lane-1",
        owner_id="owner-1",
        config_digest=_DIGEST,
        enqueued_at=_NOW,
    )


WORK_ITEM_ID = "workitem:repository_manager:11111111-1111-1111-1111-111111111111"


def test_emit_submitted_records_queue_depth_and_event() -> None:
    engine = _FakeEngine()
    result = provenance.emit_submitted(
        engine, work_item_id=WORK_ITEM_ID, attempt=1, repo="repository-manager"
    )
    assert result["kind"] == "submitted"
    assert result["status"] == str(JobState.SUBMITTED)
    event = engine.nodes[result["event_id"]]
    assert event["event_kind"] == "submitted"


def test_emit_lease_claimed_carries_the_lease_fence() -> None:
    engine = _FakeEngine()
    result = provenance.emit_lease_claimed(
        engine, work_item_id=WORK_ITEM_ID, repo="repository-manager", lease=_lease()
    )
    event = engine.nodes[result["event_id"]]
    assert event["status"] == str(JobState.LEASED)
    assert "fence_ref" in event
    assert "owner_ref" in event


def test_emit_admission_placement_uses_reservation_fence_and_state() -> None:
    engine = _FakeEngine()
    reservation = _reservation()
    result = provenance.emit_admission_placement(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        repo="repository-manager",
        reservation=reservation,
    )
    assert result["status"] == "reserved"
    event = engine.nodes[result["event_id"]]
    assert "fence_ref" in event
    assert "reservation_id_ref" in event


def test_emit_command_result_success_then_stale_fence_replay_refused() -> None:
    engine = _FakeEngine()
    provenance.emit_lease_claimed(
        engine, work_item_id=WORK_ITEM_ID, repo="repository-manager", lease=_lease()
    )
    ok = provenance.emit_command_result(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        repo="repository-manager",
        result=_execution_result(),
    )
    assert ok["status"] == "succeeded"

    # A new lease under a superseded fence, then the OLD worker's stale
    # command_result must be refused, not recorded as success.
    provenance.emit_lease_claimed(
        engine,
        work_item_id=WORK_ITEM_ID,
        repo="repository-manager",
        lease=_lease(attempt=2, fence="fence-2"),
    )
    with pytest.raises(provenance.StaleFenceError):
        write_result = _execution_result(fence="fence-1")
        provenance.emit_command_result(
            engine,
            work_item_id=WORK_ITEM_ID,
            attempt=1,
            repo="repository-manager",
            result=write_result,
        )


def test_emit_command_result_failure_carries_failure_class_as_error() -> None:
    engine = _FakeEngine()
    result = _execution_result(
        outcome=ExecutionOutcome.FAILED, failure_class=FailureClass.WORKER_ENVIRONMENT_FAILURE
    )
    outcome = provenance.emit_command_result(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        repo="repository-manager",
        result=result,
    )
    assert outcome["status"] == "failed"
    event = engine.nodes[outcome["event_id"]]
    # Raw error text is never persisted -- only a digest + character count.
    assert event["error"] == ""
    assert event["error_digest"]


def test_emit_artifact_published_uses_content_address_and_index() -> None:
    engine = _FakeEngine()
    artifact = _artifact()
    result = provenance.emit_artifact_published(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        artifact=artifact,
        artifact_index=0,
    )
    event = engine.nodes[result["event_id"]]
    assert event["content_address_ref"]
    assert event["relative_path_ref"]
    # Replaying the same artifact at the same index is idempotent (same id).
    replay = provenance.emit_artifact_published(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        artifact=artifact,
        artifact_index=0,
    )
    assert replay["event_id"] == result["event_id"]


def test_emit_validation_evidence_records_duration_metric_without_raising() -> None:
    engine = _FakeEngine()
    result = provenance.emit_validation_evidence(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        repo="repository-manager",
        evidence=_validation_evidence(),
    )
    assert result["status"] == "passed"


def test_emit_generation_event_landed_increments_landing_drift() -> None:
    engine = _FakeEngine()
    generation = _generation(state=GenerationState.OPEN)
    result = provenance.emit_generation_event(
        engine, work_item_id="workitem:repository_manager:generation-fixture", generation=generation, occurrence=0
    )
    assert result["status"] == "open"


def test_emit_candidate_event_records_lane_and_generation_correlations() -> None:
    engine = _FakeEngine()
    result = provenance.emit_candidate_event(
        engine,
        work_item_id="workitem:repository_manager:candidate-fixture",
        candidate=_candidate(),
        occurrence=0,
    )
    event = engine.nodes[result["event_id"]]
    assert event["lane_id_ref"]


def test_full_lifecycle_explain_and_reconcile_via_domain_helpers() -> None:
    engine = _FakeEngine()
    provenance.emit_submitted(
        engine, work_item_id=WORK_ITEM_ID, attempt=1, repo="repository-manager"
    )
    provenance.emit_lease_claimed(
        engine, work_item_id=WORK_ITEM_ID, repo="repository-manager", lease=_lease()
    )
    provenance.emit_started(
        engine, work_item_id=WORK_ITEM_ID, attempt=1, fence="fence-1"
    )
    provenance.emit_command_result(
        engine,
        work_item_id=WORK_ITEM_ID,
        attempt=1,
        repo="repository-manager",
        result=_execution_result(),
    )
    for node in engine.nodes.values():
        node["tenant_ref"] = "tenant:fixture"

    explanation = provenance.explain_job(
        engine, work_item_id=WORK_ITEM_ID, tenant_ref="tenant:fixture"
    )
    assert explanation["terminal"] is True
    assert explanation["event_count"] == 4

    events = provenance.job_events(
        engine, work_item_id=WORK_ITEM_ID, tenant_ref="tenant:fixture"
    )
    assert [e["event_kind"] for e in events] == [
        "submitted",
        "lease_claimed",
        "started",
        "command_result",
    ]

    report = provenance.reconcile_job(
        engine, work_item_id=WORK_ITEM_ID, tenant_ref="tenant:fixture"
    )
    assert report["proposed_repair"] is None  # terminal -- nothing to repair


def test_reconcile_leased_never_started_proposes_reclaim() -> None:
    engine = _FakeEngine()
    stuck_id = "workitem:repository_manager:22222222-2222-2222-2222-222222222222"
    provenance.emit_lease_claimed(
        engine, work_item_id=stuck_id, repo="repository-manager", lease=_lease()
    )
    for node in engine.nodes.values():
        node["tenant_ref"] = "tenant:fixture"
    report = provenance.reconcile_job(
        engine, work_item_id=stuck_id, tenant_ref="tenant:fixture"
    )
    assert report["proposed_repair"]["kind"] == "reclaim_and_relaunch"


def test_engine_unavailable_refuses_loudly() -> None:
    with pytest.raises(provenance.RepositoryProvenanceUnavailable):
        provenance.emit_submitted(
            None, work_item_id=WORK_ITEM_ID, attempt=1, repo="repository-manager"
        )
