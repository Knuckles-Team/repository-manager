"""Focused contract tests for RMDD-01's effect-free v1 boundary."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from repository_manager.development import (
    C10_FAILURE_CODES,
    ArtifactReference,
    BuildKey,
    BuildOutcome,
    BuildResult,
    Candidate,
    CandidateVersion,
    CapacitySnapshot,
    ConsentPolicy,
    DependencyEdge,
    DevelopmentRequest,
    EvidenceOutcome,
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FloorRewrite,
    Generation,
    JobState,
    LaneReference,
    LaneState,
    LeaseRecord,
    LogReference,
    OperationKind,
    RefusalCode,
    ReleasePlanState,
    RepositoryIdentity,
    RepositoryJobResult,
    ResourceRequest,
    ResourceReservation,
    TargetKind,
    TargetPolicy,
    ValidationEvidence,
    ValidationPolicy,
    ValidationStage,
    WorkspaceProject,
    WorkspaceReleasePlan,
    canonical_digest,
    canonical_json,
    contract_schema_bundle,
    deserialize_contract,
    is_legal_transition,
    require_legal_transition,
    serialize_contract,
)

SHA0 = "0" * 40
SHA1 = "1" * 40
DIGEST0 = "0" * 64
DIGEST1 = "1" * 64
NOW = datetime(2026, 8, 8, 23, 0, tzinfo=UTC)


def _repository() -> RepositoryIdentity:
    return RepositoryIdentity(
        repository_id="repo:agent-webui",
        canonical_path="/home/apps/workspace/agent-packages/agent-webui",
        configured_roots=(
            "/home/apps/workspace",
            "/home/apps/worktrees/repository-manager",
        ),
    )


def _artifact() -> ArtifactReference:
    return ArtifactReference(
        content_address=DIGEST0,
        relative_path="dist/agent-webui.tgz",
        size_bytes=42,
        media_type="application/gzip",
    )


def _log() -> LogReference:
    return LogReference(
        content_address=DIGEST1,
        relative_path="logs/certification.log",
        size_bytes=12,
        tail_bytes=12,
    )


def _execution_result() -> ExecutionResult:
    return ExecutionResult(
        command_id="command:certification",
        outcome=ExecutionOutcome.SUCCEEDED,
        exit_code=0,
        started_at=NOW,
        finished_at=NOW + timedelta(seconds=2),
        duration_ms=2000,
        worker_id="worker:local-1",
        fence="fence:certification-1",
        stdout_tail="passed",
        log_refs=(_log(),),
        artifact_refs=(_artifact(),),
    )


def _candidate() -> Candidate:
    return Candidate(
        candidate_id="candidate:one",
        version=1,
        repository=_repository(),
        branch="feature/one",
        candidate_sha=SHA1,
        base_sha=SHA0,
        lane_id="lane:one",
        owner_id="agent:one",
        config_digest=DIGEST0,
        concept_claims=("concept:one",),
        enqueued_at=NOW,
    )


def _generation() -> Generation:
    return Generation(
        generation_id="generation:one",
        repository=_repository(),
        target_branch="main",
        base_sha=SHA0,
        expected_landing_base_sha=SHA0,
        candidate_versions=(
            CandidateVersion(
                candidate_id="candidate:one",
                version=1,
                candidate_sha=SHA1,
            ),
        ),
        config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
    )


def test_canonical_serialization_is_stable_for_mapping_order_and_round_trips() -> None:
    first = {"z": {"b", "a"}, "a": {"nested": 1}}
    second = {"a": {"nested": 1}, "z": {"a", "b"}}
    assert canonical_json(first) == canonical_json(second)
    assert canonical_digest(first) == canonical_digest(second)
    assert (
        ResourceRequest(
            host_labels=("nodejs", "linux"), anti_affinity=("frontend", "build")
        ).digest()
        == ResourceRequest(
            host_labels=("linux", "nodejs"), anti_affinity=("build", "frontend")
        ).digest()
    )

    models = [
        TargetPolicy(),
        _repository(),
        ValidationPolicy(
            stages=(ValidationStage.FEEDBACK, ValidationStage.INTEGRATION)
        ),
        ConsentPolicy(),
        ResourceRequest(),
        CapacitySnapshot(
            cpu_weight_total=8,
            cpu_weight_available=8,
            memory_mib_total=4096,
            memory_mib_available=4096,
            disk_mib_total=8192,
            disk_mib_available=8192,
            process_slots_total=4,
            process_slots_available=4,
        ),
        ResourceReservation(
            reservation_id="reservation:one",
            request=ResourceRequest(),
            selected_target=TargetPolicy(),
            fence="fence:one",
            capacity=CapacitySnapshot(
                cpu_weight_total=8,
                cpu_weight_available=8,
                memory_mib_total=4096,
                memory_mib_available=4096,
                disk_mib_total=8192,
                disk_mib_available=8192,
                process_slots_total=4,
                process_slots_available=4,
            ),
            reserved_at=NOW,
            expires_at=NOW + timedelta(minutes=5),
        ),
        _artifact(),
        _log(),
        ExecutionCommand(
            argv=("python", "-m", "pytest"), workdir="/home/apps/workspace"
        ),
        _execution_result(),
        ValidationEvidence(
            evidence_id="evidence:feedback",
            stage=ValidationStage.FEEDBACK,
            tree_sha=SHA0,
            gate_config_digest=DIGEST0,
            command_digest=DIGEST1,
            target=TargetPolicy(),
            host_id="host:local",
            toolchain_digest=DIGEST0,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            outcome=EvidenceOutcome.PASSED,
        ),
        _candidate(),
        _generation(),
        LaneReference(
            lane_id="lane:one",
            repository=_repository(),
            branch="feature/one",
            base_sha=SHA0,
            worktree_path="/home/apps/worktrees/repository-manager/rmdd-01-contracts-0808",
            owner_id="agent:one",
            session_id="session:one",
            created_at=NOW,
            heartbeat_at=NOW,
            expires_at=NOW + timedelta(hours=1),
            disk_budget_mib=1024,
            state=LaneState.ACTIVE,
        ),
        DevelopmentRequest(
            request_id="request:one",
            idempotency_key="idempotency:one",
            repository=_repository(),
            operation=OperationKind.CANDIDATE_SUBMIT,
            base_ref="main",
            base_sha=SHA0,
            lane_id="lane:one",
            candidate_id="candidate:one",
            owner_id="agent:one",
            session_id="session:one",
            tenant_id="tenant:one",
            fairness_group="fairness:one",
        ),
        BuildKey(
            repository=_repository(),
            tree_sha=SHA0,
            build_spec_digest=DIGEST0,
            toolchain_digest=DIGEST1,
            target_triple="linux-x86_64",
            artifact_contract_digest=DIGEST0,
        ),
    ]
    for model in models:
        payload = serialize_contract(model)
        assert json.loads(payload)["contract_version"] == "1"
        assert deserialize_contract(type(model), payload) == model


def test_build_job_and_lease_contracts_round_trip() -> None:
    key = BuildKey(
        repository=_repository(),
        tree_sha=SHA0,
        build_spec_digest=DIGEST0,
        toolchain_digest=DIGEST1,
        target_triple="linux-x86_64",
        artifact_contract_digest=DIGEST0,
    )
    build = BuildResult(
        key=key,
        outcome=BuildOutcome.PRODUCED_MISS,
        artifact_refs=(_artifact(),),
    )
    lease = LeaseRecord(
        owner="worker:local-1",
        fence="fence:one",
        attempt=1,
        heartbeat_at=NOW,
        expires_at=NOW + timedelta(minutes=1),
    )
    assert deserialize_contract(BuildResult, serialize_contract(build)) == build
    assert deserialize_contract(LeaseRecord, serialize_contract(lease)) == lease


def test_target_path_ref_and_command_boundaries_fail_closed() -> None:
    with pytest.raises(ValidationError, match="outside configured roots"):
        RepositoryIdentity(
            repository_id="repo:bad",
            canonical_path="/etc",
            configured_roots=("/home/apps/workspace",),
        )
    with pytest.raises(ValidationError, match="parent traversal"):
        RepositoryIdentity(
            repository_id="repo:bad",
            canonical_path="/home/apps/workspace/../etc",
        )
    with pytest.raises(ValidationError, match="named ref"):
        Candidate(
            candidate_id="candidate:bad-ref",
            version=1,
            repository=_repository(),
            branch=SHA0,
            candidate_sha=SHA1,
            base_sha=SHA0,
            lane_id="lane:bad-ref",
            owner_id="agent:bad-ref",
            config_digest=DIGEST0,
            enqueued_at=NOW,
        )
    with pytest.raises(ValidationError, match="sequence"):
        ExecutionCommand.model_validate(
            {"argv": "python -m pytest", "workdir": "/home/apps/workspace"}
        )
    with pytest.raises(ValidationError, match="inventory alias"):
        TargetPolicy(kind=TargetKind.INVENTORY_ALIAS)
    with pytest.raises(ValidationError, match="inventory alias only"):
        TargetPolicy(kind=TargetKind.INVENTORY_ALIAS, alias="user@host")
    with pytest.raises(ValidationError, match="extra"):
        TargetPolicy.model_validate(
            {
                "kind": TargetKind.INVENTORY_ALIAS,
                "alias": "builder-a",
                "hostname": "builder-a.example",
            }
        )


def test_certification_requires_exact_generation_and_differential_baseline() -> None:
    with pytest.raises(ValidationError, match="exact generation_id"):
        ValidationEvidence(
            evidence_id="evidence:certification",
            stage=ValidationStage.CERTIFICATION,
            tree_sha=SHA0,
            gate_config_digest=DIGEST0,
            command_digest=DIGEST1,
            target=TargetPolicy(),
            host_id="host:local",
            toolchain_digest=DIGEST0,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            outcome=EvidenceOutcome.PASSED,
        )
    with pytest.raises(ValidationError, match="baseline_tree_sha"):
        ValidationEvidence(
            evidence_id="evidence:diff",
            stage=ValidationStage.INTEGRATION,
            tree_sha=SHA0,
            gate_config_digest=DIGEST0,
            command_digest=DIGEST1,
            target=TargetPolicy(),
            host_id="host:local",
            toolchain_digest=DIGEST0,
            started_at=NOW,
            finished_at=NOW + timedelta(seconds=1),
            outcome=EvidenceOutcome.PASSED,
            differential=True,
        )


def test_generation_id_is_derived_from_ordered_membership() -> None:
    first = Generation.derive_id(
        repository_id="repo:one",
        target_branch="main",
        base_sha=SHA0,
        candidate_versions=(
            CandidateVersion(candidate_id="candidate:a", version=1, candidate_sha=SHA1),
            CandidateVersion(candidate_id="candidate:b", version=2, candidate_sha=SHA0),
        ),
        config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
    )
    reordered = Generation.derive_id(
        repository_id="repo:one",
        target_branch="main",
        base_sha=SHA0,
        candidate_versions=(
            CandidateVersion(candidate_id="candidate:b", version=2, candidate_sha=SHA0),
            CandidateVersion(candidate_id="candidate:a", version=1, candidate_sha=SHA1),
        ),
        config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
    )
    assert first.startswith("generation:")
    assert first != reordered


def test_job_result_requires_full_authority_ids_and_legal_terminal_fields() -> None:
    common = {
        "request_id": "request:one",
        "operation": OperationKind.BUILD,
        "repository": _repository(),
        "input_digest": DIGEST0,
        "config_digest": DIGEST1,
    }
    with pytest.raises(ValidationError, match="full namespaced UUID"):
        RepositoryJobResult.model_validate(
            {
                "job_id": "rmjob:123e4567-e89b-12d3-a456-42661417400x",
                "work_item_id": "workitem:repository_manager:123e4567-e89b-12d3-a456-426614174000",
                "state": JobState.READY,
                **common,
            }
        )
    with pytest.raises(ValidationError, match="succeeded job"):
        RepositoryJobResult.model_validate(
            {
                "job_id": "rmjob:123e4567-e89b-12d3-a456-426614174000",
                "work_item_id": "workitem:repository_manager:123e4567-e89b-12d3-a456-426614174000",
                "state": JobState.SUCCEEDED,
                **common,
            }
        )


def test_c10_codes_are_stable_and_match_machine_fixture() -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "development_contracts" / "v1.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["contract_version"] == "1"
    assert fixture["failure_classes"] == list(C10_FAILURE_CODES)
    assert set(fixture["refusal_codes"]) == {code.value for code in RefusalCode}


def test_schema_bundle_is_versioned_and_contains_public_contract_shapes() -> None:
    bundle = contract_schema_bundle(
        (
            ArtifactReference,
            DevelopmentRequest,
            ExecutionCommand,
            RepositoryJobResult,
        )
    )
    assert bundle["contract_name"] == "repository-development"
    assert bundle["contract_version"] == "1"
    assert set(bundle["models"]) == {
        "ArtifactReference",
        "DevelopmentRequest",
        "ExecutionCommand",
        "RepositoryJobResult",
    }
    assert "argv" in bundle["models"]["ExecutionCommand"]["properties"]


def test_workspace_release_plan_freezes_dag_and_rejects_cycles() -> None:
    project_a = WorkspaceProject(
        project_id="project:a",
        repository=RepositoryIdentity(
            repository_id="repo:a",
            canonical_path="/home/apps/workspace/projects/a",
            configured_roots=("/home/apps/workspace",),
        ),
        tree_sha=SHA0,
        current_version="1.0.0",
        next_version="1.1.0",
    )
    project_b = WorkspaceProject(
        project_id="project:b",
        repository=RepositoryIdentity(
            repository_id="repo:b",
            canonical_path="/home/apps/workspace/projects/b",
            configured_roots=("/home/apps/workspace",),
        ),
        tree_sha=SHA1,
        current_version="2.0.0",
        next_version="2.0.1",
    )
    edge = DependencyEdge(
        dependent_project_id="project:a",
        dependency_project_id="project:b",
        current_floor=">=2.0.0",
        next_floor=">=2.0.1",
    )
    rewrite = FloorRewrite(
        project_id="project:a",
        dependency_project_id="project:b",
        old_floor=">=2.0.0",
        new_floor=">=2.0.1",
    )
    consent = ConsentPolicy()
    digest = WorkspaceReleasePlan.derive_digest(
        workspace_id="workspace:main",
        selected_projects=("project:a", "project:b"),
        projects=(project_a, project_b),
        dependency_edges=(edge,),
        floor_rewrites=(rewrite,),
        validation_stages=(ValidationStage.FEEDBACK, ValidationStage.RELEASE),
        build_job_ids=(),
        push_job_ids=(),
        parallel_groups=(("project:b",), ("project:a",)),
        consent=consent,
    )
    plan = WorkspaceReleasePlan(
        plan_id="plan:one",
        workspace_id="workspace:main",
        selected_projects=("project:a", "project:b"),
        projects=(project_a, project_b),
        dependency_edges=(edge,),
        floor_rewrites=(rewrite,),
        validation_stages=(ValidationStage.FEEDBACK, ValidationStage.RELEASE),
        parallel_groups=(("project:b",), ("project:a",)),
        consent=consent,
        plan_digest=digest,
        state=ReleasePlanState.DRAFT,
        created_at=NOW,
    )
    assert deserialize_contract(WorkspaceReleasePlan, serialize_contract(plan)) == plan

    cycle = DependencyEdge(
        dependent_project_id="project:b",
        dependency_project_id="project:a",
        current_floor=">=1.0.0",
        next_floor=">=1.1.0",
    )
    cycle_digest = WorkspaceReleasePlan.derive_digest(
        workspace_id="workspace:main",
        selected_projects=("project:a", "project:b"),
        projects=(project_a, project_b),
        dependency_edges=(edge, cycle),
        floor_rewrites=(rewrite,),
        validation_stages=(ValidationStage.FEEDBACK, ValidationStage.RELEASE),
        build_job_ids=(),
        push_job_ids=(),
        parallel_groups=(("project:b",), ("project:a",)),
        consent=consent,
    )
    with pytest.raises(ValidationError, match="cycle"):
        WorkspaceReleasePlan(
            plan_id="plan:cycle",
            workspace_id="workspace:main",
            selected_projects=("project:a", "project:b"),
            projects=(project_a, project_b),
            dependency_edges=(edge, cycle),
            floor_rewrites=(rewrite,),
            validation_stages=(ValidationStage.FEEDBACK, ValidationStage.RELEASE),
            parallel_groups=(("project:b",), ("project:a",)),
            consent=consent,
            plan_digest=cycle_digest,
            created_at=NOW,
        )


def test_lifecycle_transition_table_rejects_terminal_and_cross_family_moves() -> None:
    assert is_legal_transition(JobState.SUBMITTED, JobState.READY)
    assert not is_legal_transition(JobState.SUCCEEDED, JobState.READY)
    assert not is_legal_transition(JobState.SUBMITTED, LaneState.ACTIVE)
    with pytest.raises(ValueError, match="illegal JobState transition"):
        require_legal_transition(JobState.SUCCEEDED, JobState.READY)
