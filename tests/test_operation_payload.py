"""RMDD-29 RM-side payload, exact-read, and retry preservation tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from repository_manager.development.enums import JobState
from repository_manager.development.jobs import (
    DurableJobView,
    FakeRepositoryJobPort,
    GraphRepositoryJobPort,
    JobAuthorization,
    ReconciliationObservation,
    RepositoryJobService,
    RepositoryJobServiceCode,
    RepositoryJobServiceError,
)
from repository_manager.development.payloads import (
    BuildExecutionDescriptor,
    RepositoryOperationPayload,
    canonical_payload_json,
)

NOW = datetime(2026, 8, 9, 3, 0, tzinfo=UTC)
FIXTURE = Path(__file__).parent / "fixtures" / "rmdd_29_operation_payload.json"


def _payload() -> dict[str, Any]:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _auth(*, tenant: str = "tenant-a", owner: str = "owner-a") -> JobAuthorization:
    return JobAuthorization(tenant_id=tenant, owner_id=owner, session_id="session-a")


def _request(
    payload: dict[str, Any] | None,
    *,
    key: str = "typed-build",
    operation: str = "build",
    repository_id: str = "repository-manager",
    base_sha: str = "0123456789abcdef0123456789abcdef01234567",
) -> dict[str, Any]:
    request: dict[str, Any] = {
        "contract_version": "1",
        "request_id": f"request:{key}",
        "idempotency_key": key,
        "operation": operation,
        "repository_id": repository_id,
        "base_ref": "main",
        "base_sha": base_sha,
        "owner_id": "owner-a",
        "session_id": "session-a",
        "tenant_id": "tenant-a",
        "resources": {"resource_class": "light-check"},
        "target": {"kind": "local"},
    }
    if payload is not None:
        request["operation_payload"] = payload
    return request


def _terminal(port: FakeRepositoryJobPort, job_id: str) -> DurableJobView:
    current = port.rows[job_id]
    updated = current.model_copy(
        update={
            "state": JobState.FAILED,
            "attempt": 1,
            "max_attempts": 3,
            "completed_at": NOW,
            "updated_at": NOW,
        }
    )
    port.rows[job_id] = updated
    return updated


def test_golden_payload_and_closed_type_adapter_match_au() -> None:
    raw = _payload()
    descriptor = BuildExecutionDescriptor.model_validate(raw)
    adapted = TypeAdapter(RepositoryOperationPayload).validate_python(raw)

    assert descriptor.payload_digest == raw["payload_digest"]
    assert descriptor.cache_key_digest == raw["cache_key_digest"]
    assert adapted.payload_digest == descriptor.payload_digest
    assert canonical_payload_json(descriptor) == canonical_payload_json(raw)

    with pytest.raises(ValidationError):
        TypeAdapter(RepositoryOperationPayload).validate_python(
            {**raw, "kind": "repository.unknown/v1"}
        )
    with pytest.raises(ValidationError):
        BuildExecutionDescriptor.model_validate({**raw, "unknown": "rejected"})


@pytest.mark.parametrize(
    "update",
    [
        {"workdir": "/tmp/out"},
        {"workdir": "../escape"},
        {"argv": ["bash", "-c", "echo unsafe"]},
        {"environment_refs": ["TOKEN=secret"]},
        {"artifact_patterns": ["../secret"]},
        {"repository_id": "https://user:password@example.invalid/repo"},
    ],
)
def test_descriptor_copy_update_revalidates_security(update: dict[str, Any]) -> None:
    descriptor = BuildExecutionDescriptor.model_validate(_payload())
    with pytest.raises((TypeError, ValueError, ValidationError)):
        descriptor.model_copy(update=update)


def test_fake_exact_read_is_private_atomic_and_restart_safe() -> None:
    port = FakeRepositoryJobPort()
    service = RepositoryJobService(port)
    submitted = service.submit(_request(_payload()), auth=_auth(), now=NOW)

    view_body = submitted.job.model_dump(mode="json", exclude_none=False)
    assert "operation_payload" not in view_body
    assert submitted.job.operation_payload_kind == "repository.build-execution/v1"
    assert submitted.job.operation_payload_version == "1"
    assert submitted.job.operation_payload_digest == _payload()["payload_digest"]
    assert (
        service.get_exact_execution_input(
            submitted.job.job_id, auth=_auth()
        ).payload_digest
        == _payload()["payload_digest"]
    )

    restarted = RepositoryJobService(
        FakeRepositoryJobPort.from_snapshot(port.snapshot())
    )
    exact = restarted.get_exact_execution_input(submitted.job.job_id, auth=_auth())
    assert exact is not None
    assert exact.payload_digest == _payload()["payload_digest"]

    for wrong_auth in (_auth(tenant="tenant-b"), _auth(owner="owner-b")):
        with pytest.raises(RepositoryJobServiceError) as exc_info:
            restarted.get_exact_execution_input(submitted.job.job_id, auth=wrong_auth)
        assert exc_info.value.code == RepositoryJobServiceCode.UNAUTHORIZED.value


def test_fake_typed_input_conflict_and_tamper_fail_closed() -> None:
    port = FakeRepositoryJobPort()
    service = RepositoryJobService(port)
    request = _request(_payload(), key="typed-conflict")
    submitted = service.submit(request, auth=_auth(), now=NOW)
    changed = BuildExecutionDescriptor.model_validate(_payload()).model_copy(
        update={"argv": ("cargo", "check")}
    )
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.submit(
            _request(
                changed.model_dump(mode="json", exclude_none=False),
                key="typed-conflict",
            ),
            auth=_auth(),
            now=NOW,
        )
    assert exc_info.value.code == RepositoryJobServiceCode.INPUT_CONFLICT.value

    tampered = _payload()
    tampered["feature_set"] = "tampered"
    port.execution_inputs[submitted.job.job_id] = tampered  # type: ignore[assignment]
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.get_exact_execution_input(submitted.job.job_id, auth=_auth())
    assert exc_info.value.code == RepositoryJobServiceCode.INPUT_CONFLICT.value


@pytest.mark.parametrize(
    "payload_request",
    [
        _request(_payload(), repository_id="other-repository", key="wrong-repository"),
        _request(
            _payload(),
            base_sha="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            key="wrong-base",
        ),
        _request(_payload(), operation="validation", key="wrong-operation"),
    ],
)
def test_fake_rejects_payload_binding_mismatch(payload_request: dict[str, Any]) -> None:
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        FakeRepositoryJobPort().submit(payload_request, now=NOW)
    assert exc_info.value.code == RepositoryJobServiceCode.INVALID_REQUEST.value


def test_legacy_build_exact_retry_and_repair_require_typed_payload() -> None:
    port = FakeRepositoryJobPort()
    service = RepositoryJobService(port)
    submitted = service.submit(
        _request(None, key="legacy-build"), auth=_auth(), now=NOW
    )

    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.get_exact_execution_input(submitted.job.job_id, auth=_auth())
    assert exc_info.value.code == (
        RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED.value
    )

    _terminal(port, submitted.job.job_id)
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.retry(submitted.job.job_id, auth=_auth(), now=NOW)
    assert exc_info.value.code == (
        RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED.value
    )

    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.reconcile(
            ReconciliationObservation(
                job_id=submitted.job.job_id,
                worktree_present=False,
                observed_at=NOW,
            ),
            auth=_auth(),
            enqueue_repairs=True,
            now=NOW,
        )
    assert exc_info.value.code == (
        RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_REQUIRED.value
    )


def test_service_revalidates_malformed_authority_summary() -> None:
    port = FakeRepositoryJobPort()
    service = RepositoryJobService(port)
    submitted = service.submit(_request(_payload()), auth=_auth(), now=NOW)
    port.rows[submitted.job.job_id] = submitted.job.model_copy(
        update={"operation_payload_kind": "unknown/v1"}
    )
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        service.get(submitted.job.job_id, auth=_auth())
    assert exc_info.value.code == RepositoryJobServiceCode.INTERNAL.value


def test_production_submit_and_exact_read_accept_foreign_au_model(monkeypatch) -> None:
    raw_payload = _payload()

    class ForeignAUPayload(BaseModel):
        model_config = ConfigDict(extra="allow")

    foreign_payload = ForeignAUPayload.model_validate(raw_payload)
    job_id = "rmjob:11111111-1111-1111-1111-111111111111"
    view_raw = {
        "contract_version": "1",
        "job_id": job_id,
        "work_item_id": "workitem:repository_manager:11111111-1111-1111-1111-111111111111",
        "request_id": "request:typed",
        "operation": "build",
        "kind": "repository.build",
        "state": "ready",
        "repository_id": "repository-manager",
        "tenant_id": "tenant-a",
        "owner_id": "owner-a",
        "session_id": "session-a",
        "base_ref": "main",
        "base_sha": raw_payload["base_sha"],
        "input_digest": "a" * 64,
        "operation_payload_kind": raw_payload["kind"],
        "operation_payload_version": raw_payload["schema_version"],
        "operation_payload_digest": raw_payload["payload_digest"],
    }
    calls: dict[str, Any] = {}

    class Authority:
        @staticmethod
        def submit_repository_work_item(*_args: object, **kwargs: object) -> object:
            calls["submit"] = kwargs
            return SimpleNamespace(
                model_dump=lambda **_: {"job_id": job_id, "deduplicated": False}
            )

        @staticmethod
        def get_repository_work_item(*_args: object, **kwargs: object) -> object:
            calls["get"] = kwargs
            return SimpleNamespace(model_dump=lambda **_: view_raw)

        @staticmethod
        def get_repository_operation_payload(
            *_args: object, **kwargs: object
        ) -> object:
            calls["exact"] = kwargs
            return foreign_payload

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    port = GraphRepositoryJobPort(object())
    monkeypatch.setattr(port, "_resolved_submission", lambda request: dict(request))
    result = port.submit(_request(raw_payload), now=NOW)

    assert result.job.operation_payload_digest == raw_payload["payload_digest"]
    assert "operation_payload" not in result.job.model_dump(mode="json")
    exact = port.get_exact_execution_input(
        job_id, tenant_id="tenant-a", owner_id="owner-a"
    )
    assert isinstance(exact, BuildExecutionDescriptor)
    assert exact.payload_digest == raw_payload["payload_digest"]
    assert calls["exact"] == {"tenant": "tenant-a", "owner_id": "owner-a"}


def test_unexpected_authority_exception_text_stays_internal(monkeypatch) -> None:
    class Authority:
        @staticmethod
        def submit_repository_work_item(*_args: object, **_kwargs: object) -> object:
            raise RuntimeError("input_conflict secret=payload-body")

    monkeypatch.setattr(
        GraphRepositoryJobPort,
        "_authority_module",
        staticmethod(lambda: Authority),
    )
    port = GraphRepositoryJobPort(object())
    monkeypatch.setattr(port, "_resolved_submission", lambda request: dict(request))
    with pytest.raises(RepositoryJobServiceError) as exc_info:
        port.submit(_request(_payload()), now=NOW)
    assert exc_info.value.code == RepositoryJobServiceCode.INTERNAL.value
    assert "payload-body" not in str(exc_info.value)
