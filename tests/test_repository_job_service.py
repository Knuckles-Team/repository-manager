"""Focused RMDD-06 application-service contract tests."""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace
from typing import Any

import pytest

import repository_manager.development.jobs as jobs
from repository_manager.development.enums import JobState
from repository_manager.development.jobs import (
    DurableJobView,
    FakeRepositoryJobPort,
    GraphRepositoryJobPort,
    JobAuthorization,
    JobFilters,
    JobSubmitResult,
    LegacyShadowAdapter,
    ReconciliationClass,
    ReconciliationObservation,
    RepositoryJobService,
    RepositoryJobServiceCode,
    RepositoryJobServiceError,
    decode_cursor,
    encode_cursor,
)
from repository_manager.development.models import DevelopmentRequest
from repository_manager.resource_profiles import default_resource_profiles

NOW = datetime(2026, 8, 9, 3, 0, tzinfo=UTC)


def _auth(*, tenant: str = "tenant-a", owner: str = "owner-a") -> JobAuthorization:
    return JobAuthorization(tenant_id=tenant, owner_id=owner, session_id="session-a")


def _request(
    *,
    tenant: str = "tenant-a",
    owner: str = "owner-a",
    key: str = "job-1",
    operation: str = "build",
    created_dependencies: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "contract_version": "1",
        "request_id": f"request:{key}",
        "idempotency_key": key,
        "operation": operation,
        "repository_id": "repo:agent-webui",
        "base_ref": "main",
        "base_sha": "a" * 40,
        "owner_id": owner,
        "session_id": "session-a",
        "tenant_id": tenant,
        "dependencies": list(created_dependencies),
        "priority": 20,
        "resources": {
            "resource_class": "frontend-build",
            "concurrency_key": "agent-webui",
            "cpu_weight": 4,
            "memory_mib": 1024,
            "disk_mib": 2048,
            "process_slots": 1,
            "host_labels": ["nodejs"],
            "preferred_target": {"kind": "local"},
            "fairness_group": "frontend",
        },
        "target": {"kind": "local"},
    }


def _service() -> tuple[RepositoryJobService, FakeRepositoryJobPort]:
    port = FakeRepositoryJobPort()
    return RepositoryJobService(port), port


def _mark_terminal(
    port: FakeRepositoryJobPort,
    job_id: str,
    state: JobState,
    *,
    attempt: int = 1,
    max_attempts: int = 3,
) -> DurableJobView:
    current = port.rows[job_id]
    updated = current.model_copy(
        update={
            "state": state,
            "attempt": attempt,
            "max_attempts": max_attempts,
            "updated_at": NOW,
            "completed_at": NOW,
            "error_ref": "error:test" if state != JobState.SUCCEEDED else None,
        }
    )
    port.rows[job_id] = updated
    return updated


def test_duplicate_submit_across_service_objects_is_one_durable_id() -> None:
    first_service, port = _service()
    second_service = RepositoryJobService(port)
    first = first_service.submit(_request(), auth=_auth(), now=NOW)
    second = second_service.submit(_request(), auth=_auth(), now=NOW)

    assert first.job.job_id == second.job.job_id
    assert first.deduplicated is False
    assert second.deduplicated is True
    assert port.submit_calls == 2  # authority arbitrates, not service memory
    assert len(port.rows) == 1


def test_changed_immutable_input_under_same_key_is_refused() -> None:
    service, _ = _service()
    service.submit(_request(), auth=_auth(), now=NOW)
    changed = _request()
    changed["base_sha"] = "b" * 40

    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.submit(changed, auth=_auth(), now=NOW)
    assert exc_info.value.code == RepositoryJobServiceCode.DUPLICATE.value


def test_declared_source_digest_cannot_mask_changed_immutable_input() -> None:
    service, _ = _service()
    original = _request(key="declared-digest-collision")
    original["input_digest"] = "d" * 64
    service.submit(original, auth=_auth(), now=NOW)

    changed = dict(original)
    changed["base_sha"] = "b" * 40

    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.submit(changed, auth=_auth(), now=NOW)
    assert exc_info.value.code == RepositoryJobServiceCode.DUPLICATE.value


def test_fake_normalizes_nested_repository_identity_from_development_contract() -> None:
    service, _ = _service()
    nested = _request(key="nested-repository")
    repository_id = nested.pop("repository_id")
    nested["repository"] = {"repository_id": repository_id}

    result = service.submit(nested, auth=_auth(), now=NOW)
    assert result.job.repository_id == repository_id
    assert result.job.kind == "repository.build"


def test_service_accepts_the_real_nested_development_request_model() -> None:
    service, _ = _service()
    payload = _request(key="nested-development-request")
    payload.pop("repository_id")
    payload["repository"] = {
        "repository_id": "repo:agent-webui",
        "canonical_path": "/home/apps/workspace/agent-packages/agents/agent-webui",
    }
    payload["fairness_group"] = "frontend"
    request = DevelopmentRequest.model_validate(payload)

    result = service.submit(request, auth=_auth(), now=NOW)
    assert result.job.repository_id == "repo:agent-webui"


def test_state_and_result_projection_survives_service_recreation() -> None:
    service, port = _service()
    submitted = service.submit(_request(), auth=_auth(), now=NOW)
    _mark_terminal(port, submitted.job.job_id, JobState.SUCCEEDED)

    restarted = RepositoryJobService(port)
    recovered = restarted.get(submitted.job.job_id, auth=_auth())
    assert recovered.state == JobState.SUCCEEDED
    assert recovered.error_ref is None
    assert recovered.completed_at == NOW


def test_dependency_submission_has_no_executor_or_future() -> None:
    service, port = _service()
    parent = service.submit(_request(key="parent"), auth=_auth(), now=NOW)
    child = service.submit(
        _request(key="child", created_dependencies=(parent.job.work_item_id,)),
        auth=_auth(),
        now=NOW,
    )

    assert child.job.state == JobState.SUBMITTED
    assert not hasattr(service, "_jobs")
    assert not hasattr(service, "_job_futures")
    assert not hasattr(port, "futures")


@pytest.mark.parametrize(
    ("state", "allowed"),
    [
        (JobState.SUBMITTED, True),
        (JobState.READY, True),
        (JobState.LEASED, True),
        (JobState.RUNNING, True),
        (JobState.CANCELLED, True),
        (JobState.SUCCEEDED, False),
        (JobState.FAILED, False),
        (JobState.DEAD_LETTER, False),
    ],
)
def test_cancel_lifecycle_rules(state: JobState, allowed: bool) -> None:
    service, port = _service()
    submitted = service.submit(_request(key=state.value), auth=_auth(), now=NOW)
    port.rows[submitted.job.job_id] = port.rows[submitted.job.job_id].model_copy(
        update={"state": state}
    )
    if allowed:
        result = service.cancel(submitted.job.job_id, auth=_auth(), now=NOW)
        assert result.job.state == JobState.CANCELLED
    else:
        with pytest.raises(RepositoryJobServiceError) as exc_info:
            service.cancel(submitted.job.job_id, auth=_auth(), now=NOW)
        assert exc_info.value.code == RepositoryJobServiceCode.INVALID_STATE.value


@pytest.mark.parametrize(
    ("state", "attempt", "max_attempts", "allowed"),
    [
        (JobState.FAILED, 1, 3, True),
        (JobState.DEAD_LETTER, 1, 3, True),
        (JobState.FAILED, 3, 3, False),
        (JobState.SUBMITTED, 0, 3, False),
        (JobState.READY, 0, 3, False),
        (JobState.LEASED, 1, 3, False),
        (JobState.RUNNING, 1, 3, False),
        (JobState.CANCELLED, 1, 3, False),
        (JobState.SUCCEEDED, 1, 3, False),
    ],
)
def test_retry_lifecycle_rules(
    state: JobState, attempt: int, max_attempts: int, allowed: bool
) -> None:
    service, port = _service()
    submitted = service.submit(
        _request(key=f"retry-{state.value}-{attempt}"), auth=_auth(), now=NOW
    )
    _mark_terminal(
        port,
        submitted.job.job_id,
        state,
        attempt=attempt,
        max_attempts=max_attempts,
    )
    if allowed:
        result = service.retry(submitted.job.job_id, auth=_auth(), now=NOW)
        assert result.retry_of == submitted.job.job_id
        assert result.job.job_id != submitted.job.job_id
        assert result.job.correlation_id == submitted.job.job_id
    else:
        with pytest.raises(RepositoryJobServiceError) as exc_info:
            service.retry(submitted.job.job_id, auth=_auth(), now=NOW)
        assert exc_info.value.code == RepositoryJobServiceCode.INVALID_STATE.value


def test_retry_chain_consumes_the_original_budget_without_replenishing_it() -> None:
    service, port = _service()
    original = service.submit(
        _request(key="retry-budget"), auth=_auth(), max_attempts=3, now=NOW
    )
    _mark_terminal(
        port, original.job.job_id, JobState.FAILED, attempt=1, max_attempts=3
    )

    first_retry = service.retry(original.job.job_id, auth=_auth(), now=NOW)
    assert port.rows[first_retry.job.job_id].max_attempts == 2
    _mark_terminal(
        port, first_retry.job.job_id, JobState.FAILED, attempt=1, max_attempts=2
    )

    second_retry = service.retry(first_retry.job.job_id, auth=_auth(), now=NOW)
    assert port.rows[second_retry.job.job_id].max_attempts == 1
    _mark_terminal(
        port, second_retry.job.job_id, JobState.FAILED, attempt=1, max_attempts=1
    )

    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.retry(second_retry.job.job_id, auth=_auth(), now=NOW)
    assert exc_info.value.code == RepositoryJobServiceCode.INVALID_STATE.value
    assert len(port.rows) == 3


def test_cross_tenant_and_owner_access_refuses_without_disclosing_state() -> None:
    service, _ = _service()
    submitted = service.submit(_request(), auth=_auth(), now=NOW)

    for auth in (_auth(tenant="tenant-b"), _auth(owner="owner-b")):
        with pytest.raises(RepositoryJobServiceError) as exc_info:
            service.get(submitted.job.job_id, auth=auth)
        assert exc_info.value.code == RepositoryJobServiceCode.UNAUTHORIZED.value


def test_public_invalid_inputs_keep_stable_service_codes() -> None:
    service, _ = _service()

    with pytest.raises(RepositoryJobServiceError) as submit_error:
        service.submit(object(), auth=_auth(), now=NOW)  # type: ignore[arg-type]
    assert submit_error.value.code == RepositoryJobServiceCode.INVALID_REQUEST.value

    with pytest.raises(RepositoryJobServiceError) as get_error:
        service.get("", auth=_auth())
    assert get_error.value.code == RepositoryJobServiceCode.INVALID_REQUEST.value


def test_list_is_owner_scoped_and_keyset_bounded() -> None:
    service, port = _service()
    for index in range(5):
        service.submit(
            _request(key=f"page-{index}"),
            auth=_auth(),
            now=NOW + timedelta(seconds=index),
        )
    service.submit(
        _request(tenant="tenant-a", owner="other", key="other"),
        auth=_auth(owner="other"),
        now=NOW,
    )

    first = service.list(auth=_auth(), filters=JobFilters(limit=2))
    assert len(first.items) <= 2
    assert first.scanned == 2
    assert first.next_cursor is not None
    assert all(item.owner_id == "owner-a" for item in first.items)
    second = service.list(
        auth=_auth(), filters=JobFilters(limit=2, cursor=first.next_cursor)
    )
    assert second.scanned == 2
    assert set(item.job_id for item in first.items).isdisjoint(
        item.job_id for item in second.items
    )
    assert all(item.owner_id == "owner-a" for item in second.items)
    assert len(port.rows) == 6


def test_cursor_cannot_be_replayed_across_tenants() -> None:
    token = encode_cursor("tenant-a", (1.0, "workitem:repository_manager:x"))
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        decode_cursor("tenant-b", token)
    assert exc_info.value.code == RepositoryJobServiceCode.UNAUTHORIZED.value


def test_reconcile_distinguishes_required_findings_and_is_preview_only() -> None:
    service, port = _service()
    submitted = service.submit(_request(), auth=_auth(), now=NOW)
    observation = ReconciliationObservation(
        job_id=submitted.job.job_id,
        worktree_present=False,
        process_present=True,
        process_last_heartbeat=NOW - timedelta(hours=1),
        observed_artifact_refs=("artifact:orphan",),
        expected_artifact_refs=("artifact:expected",),
        observed_fence="fence:stale",
        expected_fence="fence:current",
        observed_target_sha="b" * 40,
        expected_target_sha="a" * 40,
        observed_at=NOW,
    )
    result = service.reconcile(observation, auth=_auth(), now=NOW)
    finding = result.findings[0]
    assert finding.classifications == (
        ReconciliationClass.MISSING_WORKTREE,
        ReconciliationClass.STALE_PROCESS,
        ReconciliationClass.ORPHAN_ARTIFACT,
        ReconciliationClass.STALE_FENCE,
        ReconciliationClass.TARGET_DRIFT,
    )
    assert finding.repair is not None
    assert finding.repair.preview_only is True
    assert port.submit_calls == 1


@pytest.mark.parametrize("state", [JobState.LEASED, JobState.RUNNING])
def test_reconcile_explicit_missing_process_is_stale_for_active_jobs(
    state: JobState,
) -> None:
    service, port = _service()
    submitted = service.submit(
        _request(key=f"missing-process-{state}"), auth=_auth(), now=NOW
    )
    port.rows[submitted.job.job_id] = port.rows[submitted.job.job_id].model_copy(
        update={"state": state, "lease_owner": "worker-a", "lease_fence": "fence-a"}
    )

    result = service.reconcile(
        ReconciliationObservation(
            job_id=submitted.job.job_id,
            worktree_present=True,
            process_present=False,
            observed_at=NOW,
        ),
        auth=_auth(),
    )
    assert result.findings[0].classifications == (ReconciliationClass.STALE_PROCESS,)


def test_reconcile_unobserved_process_and_worktree_is_clean() -> None:
    service, _ = _service()
    submitted = service.submit(_request(key="clean"), auth=_auth(), now=NOW)

    result = service.reconcile(
        ReconciliationObservation(
            job_id=submitted.job.job_id,
            worktree_present=True,
            process_present=None,
            observed_at=NOW,
        ),
        auth=_auth(),
    )
    finding = result.findings[0]
    assert finding.classifications == (ReconciliationClass.CLEAN,)
    assert finding.repair is None


def test_reconcile_enqueues_idempotent_follow_up_without_inline_mutation() -> None:
    service, port = _service()
    submitted = service.submit(_request(), auth=_auth(), now=NOW)
    observation = ReconciliationObservation(
        job_id=submitted.job.job_id,
        worktree_present=False,
        observed_at=NOW,
    )
    first = service.reconcile(observation, auth=_auth(), enqueue_repairs=True, now=NOW)
    second = service.reconcile(
        observation.model_copy(
            update={
                "process_present": False,
                "observed_at": NOW + timedelta(seconds=1),
            }
        ),
        auth=_auth(),
        enqueue_repairs=True,
        now=NOW + timedelta(seconds=1),
    )
    first_repair = first.findings[0].repair
    second_repair = second.findings[0].repair
    assert first_repair is not None
    assert second_repair is not None
    assert first_repair.preview_only is False
    assert first_repair.enqueued_job_id == second_repair.enqueued_job_id
    assert first_repair.repair_id == second_repair.repair_id
    assert len(port.rows) == 2
    assert port.rows[submitted.job.job_id].state == JobState.READY
    assert not hasattr(port, "repair_plans")


def test_repair_mapping_is_accepted_by_the_merged_au_contract() -> None:
    authority = pytest.importorskip(
        "agent_utilities.orchestration.repository_work_item"
    )
    service, _ = _service()
    submitted = service.submit(_request(key="repair-contract"), auth=_auth(), now=NOW)
    preview = service.reconcile(
        ReconciliationObservation(
            job_id=submitted.job.job_id,
            worktree_present=False,
            observed_at=NOW,
        ),
        auth=_auth(),
    )
    proposal = preview.findings[0].repair
    assert proposal is not None

    request = jobs._repair_request_mapping(
        submitted.job,
        proposal,
        owner_id="owner-a",
        session_id="session-a",
    )
    typed = authority.RepositoryWorkItemRequest.from_contract(request)
    operation = getattr(typed.operation, "value", typed.operation)
    assert operation == "repair"
    assert typed.correlation_id == submitted.job.job_id
    assert typed.input_digest == request["input_digest"]
    assert typed.input_digest != submitted.job.input_digest
    assert request["repair_intent_digest"] == request["input_digest"]


def test_already_completed_effect_is_visible_and_does_not_disappear_as_clean() -> None:
    service, _ = _service()
    submitted = service.submit(_request(), auth=_auth(), now=NOW)
    result = service.reconcile(
        ReconciliationObservation(
            job_id=submitted.job.job_id,
            already_completed_effect=True,
            observed_at=NOW,
        ),
        auth=_auth(),
    )
    assert result.findings[0].classifications == (
        ReconciliationClass.ALREADY_COMPLETED_EFFECT,
    )


def test_shadow_mismatch_is_visible_and_never_changes_durable_view() -> None:
    service, port = _service()
    submitted = service.submit(_request(), auth=_auth(), now=NOW)
    mismatch = service.shadow_compare(
        submitted.job.job_id,
        {"status": "completed", "repo_name": "wrong-repo"},
        auth=_auth(),
    )
    assert mismatch is not None
    assert set(mismatch.fields) == {"repository_id", "state"}
    assert port.rows[submitted.job.job_id].state == JobState.READY
    assert LegacyShadowAdapter.compare(port.rows[submitted.job.job_id], {}) is None


def test_production_adapter_normalizes_the_merged_au_view_shape() -> None:
    """Pin the AU ``kind``/nested ``lease`` projection translation.

    The real AU model is deliberately not imported by the Repository Manager
    package at module import time.  This fixture uses the exact serialized
    fields from ``RepositoryWorkItemView`` and therefore proves the adapter's
    wire-shape boundary without starting a graph engine.
    """

    raw = {
        "contract_version": "1",
        "job_id": "rmjob:11111111-1111-1111-1111-111111111111",
        "work_item_id": "workitem:repository_manager:11111111-1111-1111-1111-111111111111",
        "request_id": "request:one",
        "operation": "build",
        "kind": "repository.build",
        "state": "running",
        "repository_id": "repo:agent-webui",
        "tenant_id": "tenant-a",
        "owner_id": "owner-a",
        "session_id": "session-a",
        "base_ref": "main",
        "base_sha": "a" * 40,
        "target_kind": "local",
        "input_digest": "b" * 64,
        "attempt": 1,
        "max_attempts": 3,
        "resource_class": "frontend-build",
        "lease": {
            "owner": "worker-a",
            "epoch": 2,
            "fencing_token": 2,
            "attempt": 1,
            "heartbeat_at": NOW.timestamp(),
            "expires_at": (NOW + timedelta(minutes=5)).timestamp(),
        },
    }
    view = GraphRepositoryJobPort._view(SimpleNamespace(model_dump=lambda **_: raw))
    assert view.operation == "build"
    assert view.kind == "repository.build"
    assert view.state == JobState.RUNNING
    assert view.lease_owner == "worker-a"
    assert view.lease_fence == "2"
    assert view.lease_expires_at == NOW + timedelta(minutes=5)


def test_production_adapter_accepts_the_actual_merged_au_view_model() -> None:
    authority = pytest.importorskip(
        "agent_utilities.orchestration.repository_work_item"
    )
    lease = authority.RepositoryLease(
        owner="worker-a",
        epoch=2,
        fencing_token=2,
        attempt=1,
        heartbeat_at=NOW.timestamp(),
        expires_at=(NOW + timedelta(minutes=5)).timestamp(),
    )
    au_view = authority.RepositoryWorkItemView(
        job_id="rmjob:11111111-1111-1111-1111-111111111111",
        work_item_id="workitem:repository_manager:11111111-1111-1111-1111-111111111111",
        request_id="request:one",
        operation="build",
        kind="repository.build",
        state="running",
        repository_id="repo:agent-webui",
        tenant_id="tenant-a",
        owner_id="owner-a",
        session_id="session-a",
        base_ref="main",
        base_sha="a" * 40,
        target_kind="local",
        input_digest="b" * 64,
        attempt=1,
        max_attempts=3,
        lease=lease,
    )
    view = GraphRepositoryJobPort._view(au_view)
    assert view.kind == "repository.build"
    assert view.lease_owner == "worker-a"
    assert view.lease_fence == "2"


def test_production_list_consumes_one_raw_page_and_returns_scanned_cursor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, fake = _service()
    first = service.submit(_request(key="raw-page-1"), auth=_auth(), now=NOW)
    second = service.submit(
        _request(key="raw-page-2"), auth=_auth(), now=NOW + timedelta(seconds=1)
    )
    rows = []
    for view in (fake.rows[first.job.job_id], fake.rows[second.job.job_id]):
        raw = view.model_dump(mode="json", exclude_none=False)
        raw.update(
            {
                "kind": "repository.build",
                "lease": None,
                "created_at": view.created_at.timestamp() if view.created_at else 0.0,
            }
        )
        rows.append(raw)
    calls: list[tuple[int, tuple[float, str] | None]] = []

    class Authority:
        _REPOSITORY_KINDS = ("repository.build",)

        @staticmethod
        def _repository_rows(*_args: object, **kwargs: Any) -> list[dict[str, object]]:
            calls.append((int(kwargs["limit"]), kwargs["cursor"]))
            cursor = kwargs["cursor"]
            if cursor is None:
                return rows[:1]
            return rows[1:]

        @staticmethod
        def _view_from_row(row: dict[str, object]) -> SimpleNamespace:
            return SimpleNamespace(model_dump=lambda **_: row)

        @staticmethod
        def _row_cursor(row: dict[str, object]) -> tuple[float, str]:
            return (float(str(row["created_at"])), str(row["work_item_id"]))

        @staticmethod
        def repository_work_item_kind(operation: object) -> SimpleNamespace:
            del operation
            return SimpleNamespace(value="repository.build")

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    port = GraphRepositoryJobPort(object())
    page = port.list_page(
        tenant_id="tenant-a", filters=JobFilters(limit=1), cursor=None
    )
    assert page.scanned == 1
    assert page.next_cursor is not None
    assert calls == [(1, None)]


def test_production_adapter_preserves_au_duplicate_conflict_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Authority:
        class RepositoryWorkItemConflict(ValueError):
            pass

        class RepositoryWorkItemError(ValueError):
            pass

        @staticmethod
        def submit_repository_work_item(*_args: object, **_kwargs: object) -> None:
            raise Authority.RepositoryWorkItemConflict("idempotency key conflict")

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        GraphRepositoryJobPort(object(), profiles=default_resource_profiles()).submit(
            _request(), now=NOW
        )
    assert exc_info.value.code == RepositoryJobServiceCode.DUPLICATE.value


def test_production_submission_requires_trusted_profile_registry() -> None:
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        GraphRepositoryJobPort(object())._resolved_submission(_request())
    assert exc_info.value.code == RepositoryJobServiceCode.INTERNAL.value


def test_production_submission_persists_resolved_admission_projection() -> None:
    prepared = GraphRepositoryJobPort(
        object(), profiles=default_resource_profiles()
    )._resolved_submission(_request())
    resources = prepared["resources"]
    assert isinstance(resources, dict)
    assert resources["resource_class"] == "frontend-build"
    assert resources["profile_version"] == "1"
    assert resources["concurrency_key"] == "frontend-build"
    assert resources["concurrency_limit"] == 1
    assert resources["repository_exclusive"] is False
    assert resources["branch_exclusive"] is False
    assert resources["disk_policy_key"] == "frontend-build:v1"
    assert resources["fairness_cost"] == 9
    assert (
        resources["resolved_profile_authority"]
        == "repository_manager:resource_profile_registry:v1"
    )
    assert (
        prepared["resolved_profile_authority"]
        == resources["resolved_profile_authority"]
    )
    assert "reservation_input_fingerprint" not in resources


def test_branch_exclusive_projection_preserves_only_explicit_branch() -> None:
    raw = _request()
    raw["branch"] = "release"
    raw_resources = dict(raw["resources"])
    raw_resources.update(
        {
            "resource_class": "merge-drain",
            "resolved_profile_authority": "caller-forged",
        }
    )
    raw["resources"] = raw_resources
    prepared = GraphRepositoryJobPort(
        object(), profiles=default_resource_profiles()
    )._resolved_submission(raw)
    assert prepared["branch"] == "release"
    resources = prepared["resources"]
    assert isinstance(resources, dict)
    assert resources["branch_exclusive"] is True
    assert (
        resources["resolved_profile_authority"]
        == "repository_manager:resource_profile_registry:v1"
    )


def test_production_adapter_preserves_au_base_moved_conflict_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Authority:
        class RepositoryWorkItemError(ValueError):
            pass

        @staticmethod
        def submit_repository_work_item(*_args: object, **_kwargs: object) -> None:
            raise Authority.RepositoryWorkItemError("base moved while admitting job")

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        GraphRepositoryJobPort(object(), profiles=default_resource_profiles()).submit(
            _request(), now=NOW
        )
    assert exc_info.value.code == RepositoryJobServiceCode.CONFLICT.value


def test_production_get_passes_authenticated_owner_to_au(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str | None] = []

    class Authority:
        @staticmethod
        def get_repository_work_item(*_args: object, **kwargs: object) -> None:
            owner_id = kwargs.get("owner_id")
            calls.append(owner_id if isinstance(owner_id, str) else None)
            return None

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    assert (
        GraphRepositoryJobPort(object()).get(
            "rmjob:11111111-1111-1111-1111-111111111111",
            tenant_id="tenant-a",
            owner_id="owner-a",
        )
        is None
    )
    assert calls == ["owner-a"]


def test_production_direct_cancel_denies_wrong_owner_before_native_mutation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, fake = _service()
    source = fake.submit(_request(key="direct-cancel-owner"), now=NOW).job
    raw = fake.rows[source.job_id].model_dump(mode="json", exclude_none=False)
    cancel_calls: list[str] = []

    class Authority:
        @staticmethod
        def get_repository_work_item(
            *_args: object, **_kwargs: object
        ) -> SimpleNamespace:
            return SimpleNamespace(model_dump=lambda **_: raw)

        @staticmethod
        def cancel_repository_work_item(*_args: object, **_kwargs: object) -> bool:
            cancel_calls.append("called")
            return True

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        GraphRepositoryJobPort(object()).cancel(
            source.job_id,
            tenant_id="tenant-a",
            owner_id="owner-b",
            reason="wrong owner",
            now=NOW,
        )
    assert exc_info.value.code == RepositoryJobServiceCode.UNAUTHORIZED.value
    assert cancel_calls == []


def test_authority_error_does_not_echo_secret_like_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Authority:
        class RepositoryWorkItemError(ValueError):
            pass

        @staticmethod
        def submit_repository_work_item(*_args: object, **_kwargs: object) -> None:
            raise Authority.RepositoryWorkItemError(
                "invalid input secret=4111111111111111"
            )

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        GraphRepositoryJobPort(object(), profiles=default_resource_profiles()).submit(
            _request(), now=NOW
        )
    assert exc_info.value.code == RepositoryJobServiceCode.INVALID_REQUEST.value
    assert "4111111111111111" not in str(exc_info.value)


def test_unexpected_port_error_is_normalized_to_internal_without_payload() -> None:
    class ExplodingPort(FakeRepositoryJobPort):
        def submit(
            self,
            request: Mapping[str, Any],
            *,
            max_attempts: int = 3,
            now: datetime | None = None,
        ) -> JobSubmitResult:
            del request, max_attempts, now
            raise RuntimeError("backend secret=4111111111111111")

    with pytest.raises(RepositoryJobServiceError) as exc_info:
        RepositoryJobService(ExplodingPort()).submit(_request(), auth=_auth(), now=NOW)
    assert exc_info.value.code == RepositoryJobServiceCode.INTERNAL.value
    assert "4111111111111111" not in str(exc_info.value)


def test_production_page_seam_matches_the_merged_au_contract() -> None:
    authority = pytest.importorskip(
        "agent_utilities.orchestration.repository_work_item"
    )
    parameters = inspect.signature(authority._repository_rows).parameters
    assert {
        "engine",
        "tenant",
        "limit",
        "kinds",
        "statuses",
        "cursor",
    }.issubset(parameters)
    assert (
        "owner_id" in inspect.signature(authority.get_repository_work_item).parameters
    )
    assert {"kind", "lease"}.issubset(authority.RepositoryWorkItemView.model_fields)
