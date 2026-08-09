"""Focused RMDD-09 lifecycle, quota, reconciliation, and reclamation tests."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager import lane_doctor
from repository_manager.lane_quota import (
    DiskAccountingProbe,
    LaneQuotaPolicy,
)
from repository_manager.lane_reclamation import (
    CleanupRefused,
    LaneReclaimer,
    ReconciliationClass,
)
from repository_manager.lane_record import LaneLifecycleState, repository_id_for
from repository_manager.lane_registry import (
    FakeDurableLaneAuthority,
    LaneAllocationDisabled,
    LaneConflictError,
    LaneQuotaError,
    LaneRegistry,
    LaneRegistryError,
    NativeLaneAuthorityAdapter,
    StaleLaneFence,
)
from repository_manager.worktree import WorktreeManager

NOW = datetime(2026, 8, 9, 15, 0, tzinfo=UTC)


class Clock:
    def __init__(self, value: datetime = NOW) -> None:
        self.value = value

    def now(self) -> datetime:
        return self.value


class Result:
    def __init__(self, ok: bool, data: str = "") -> None:
        self.status = "success" if ok else "error"
        self.data = data
        self.error = None


class ReadOnlyGit:
    def __init__(self, *, anchor: bool = True) -> None:
        self.anchor = anchor
        self.commands: list[str] = []

    def git_action(self, command: str, **_kwargs: object) -> Result:
        self.commands.append(command)
        if command.startswith("git status"):
            return Result(True, "")
        if command.startswith("git merge-base"):
            return Result(True, "")
        if command.startswith("git rev-parse --verify"):
            return Result(self.anchor, "a" * 40 if self.anchor else "")
        return Result(True, "")


class FakeWorktreeManager:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def remove(self, repo: str, branch: str, **_kwargs: object) -> dict[str, object]:
        self.calls.append((repo, branch))
        return {"ok": True, "branch_anchor": "refs/lane-backup/feature"}


class CleanupAuthority:
    def __init__(self) -> None:
        self.job_id = "cleanup:one"
        self.job_fence = "cleanup-fence:one"
        self.submissions = 0
        self.receipts: dict[str, dict[str, object]] = {}

    def submit(self, _plan: object) -> dict[str, str]:
        self.submissions += 1
        return {"job_id": self.job_id, "fence": self.job_fence}

    def is_current(self, job_id: str, **kwargs: object) -> bool:
        return (
            job_id == self.job_id
            and kwargs.get("job_fence") == self.job_fence
            and kwargs.get("lane_fence")
        )

    def record_cleanup_complete(self, payload: dict[str, object]) -> bool:
        self.receipts[str(payload["plan_id"])] = dict(payload)
        return True

    def get_removal_receipt(
        self, _job_id: str, *, plan_id: str
    ) -> dict[str, object] | None:
        return self.receipts.get(plan_id)


class ReceiptFailAuthority(CleanupAuthority):
    def __init__(self) -> None:
        super().__init__()
        self.fail_receipt = True
        self.pending: dict[str, object] | None = None

    def record_cleanup_complete(self, payload: dict[str, object]) -> bool:
        self.pending = dict(payload)
        if self.fail_receipt:
            return False
        return super().record_cleanup_complete(payload)


def _allocate(
    registry: LaneRegistry,
    root: Path,
    *,
    request_id: str = "request:one",
    branch: str = "feature/one",
    owner_id: str = "agent:one",
    predicted_disk_bytes: int = 10,
) -> object:
    return registry.allocate(
        root,
        branch,
        root / "lane",
        owner_id=owner_id,
        session_id="session:one",
        host_id="host:one",
        request_id=request_id,
        predicted_disk_bytes=predicted_disk_bytes,
        disk_budget_bytes=max(10, predicted_disk_bytes),
    )


def _registry(
    path: Path,
    *,
    clock: Clock | None = None,
    authority: FakeDurableLaneAuthority | None = None,
    quota: LaneQuotaPolicy | None = None,
) -> LaneRegistry:
    shared = authority or FakeDurableLaneAuthority(quota=quota, clock=clock)
    return LaneRegistry(path, quota=quota, clock=clock, authority=shared)


def test_repository_identity_includes_canonical_root_for_same_basename(
    tmp_path: Path,
) -> None:
    first = tmp_path / "one" / "repository-manager"
    second = tmp_path / "two" / "repository-manager"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    assert repository_id_for(first) != repository_id_for(second)
    assert repository_id_for(first) == repository_id_for(first / ".")


def test_duplicate_allocation_is_idempotent_but_changed_input_refuses(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "lanes.sqlite", clock=Clock())
    first = _allocate(registry, tmp_path / "repo")
    assert first.repository_basename == "repo"
    repeated = _allocate(registry, tmp_path / "repo")
    assert repeated.lane_id == first.lane_id
    assert repeated.fence == first.fence
    with pytest.raises(LaneConflictError):
        _allocate(registry, tmp_path / "repo", branch="feature/changed")


def test_projection_scopes_request_key_by_repository(tmp_path: Path) -> None:
    store = tmp_path / "lanes.sqlite"
    authority = FakeDurableLaneAuthority(clock=Clock())
    registry = LaneRegistry(store, clock=Clock(), authority=authority)
    first = _allocate(registry, tmp_path / "one", request_id="request:same")
    second = _allocate(registry, tmp_path / "two", request_id="request:same")
    assert first.repository_id != second.repository_id

    projection = LaneRegistry(store, clock=Clock())
    try:
        projected_ids = {record.lane_id for record in projection.list_records()}
    finally:
        projection.close()
    assert projected_ids == {first.lane_id, second.lane_id}


def test_worktree_allocation_derives_a_valid_default_request_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = _registry(tmp_path / "lanes.sqlite", clock=Clock())
    git = SimpleNamespace(path=str(tmp_path), project_map={"origin": str(repository)})
    manager = WorktreeManager(git, registry=registry)
    expected_path = manager.worktree_path("repo", "feature/default-key")
    monkeypatch.setattr(
        manager,
        "add",
        lambda *_args, **_kwargs: {
            "ok": True,
            "created": True,
            "path": expected_path,
            "branch": "feature/default-key",
        },
    )

    result = manager.allocate(
        "repo",
        "feature/default-key",
        owner_id="agent:one",
        session_id="session:one",
    )

    assert result["ok"] is True
    record = registry.require(str(result["lane_id"]))
    assert record.request_key.startswith("auto:")
    assert not any(ord(character) < 0x20 for character in record.request_key)


def test_lane_doctor_finish_never_borrows_projected_owner_or_fence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    registry = _registry(tmp_path / "lanes.sqlite", clock=Clock())
    record = _allocate(registry, repository)
    record = registry.activate(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
    )
    monkeypatch.setattr(
        lane_doctor,
        "diagnose",
        lambda *_args, **_kwargs: {"ok": True, "blocking": [], "checks": []},
    )

    result = lane_doctor.finish(
        repository,
        branch=record.branch,
        registry=registry,
        lane_id=record.lane_id,
    )

    assert result["ok"] is False
    assert result["stage"] == "registry"
    assert "explicit owner_id and fence" in result["reason"]
    assert registry.require(record.lane_id).state == LaneLifecycleState.ACTIVE


def test_managed_mutation_fails_closed_without_durable_authority(
    tmp_path: Path,
) -> None:
    registry = LaneRegistry(tmp_path / "lanes.sqlite", clock=Clock())
    with pytest.raises(LaneRegistryError, match="durable lane authority"):
        _allocate(registry, tmp_path / "repo")


def test_native_authority_adapter_fails_closed_without_atomic_lane_seam() -> None:
    with pytest.raises(LaneRegistryError, match="native durable lane authority"):
        NativeLaneAuthorityAdapter(object())


def test_independent_controllers_share_authoritative_identity_and_quota(
    tmp_path: Path,
) -> None:
    policy = LaneQuotaPolicy(max_per_agent=1, max_predicted_disk_bytes=100)
    authority = FakeDurableLaneAuthority(quota=policy, clock=Clock())
    first = _registry(
        tmp_path / "first.sqlite", authority=authority, quota=policy, clock=Clock()
    )
    second = _registry(
        tmp_path / "second.sqlite", authority=authority, quota=policy, clock=Clock()
    )
    allocated = _allocate(first, tmp_path / "repo")
    assert second.require(allocated.lane_id).fence == allocated.fence
    with pytest.raises(LaneConflictError):
        _allocate(second, tmp_path / "repo", request_id="request:two")
    with pytest.raises(LaneQuotaError):
        _allocate(
            second,
            tmp_path / "other",
            request_id="request:three",
            branch="feature/three",
        )


def test_branch_and_worktree_reservations_are_atomic(tmp_path: Path) -> None:
    registry = _registry(tmp_path / "lanes.sqlite", clock=Clock())
    _allocate(registry, tmp_path / "repo")
    with pytest.raises(LaneConflictError):
        _allocate(registry, tmp_path / "repo", request_id="request:two")
    with pytest.raises(LaneConflictError):
        registry.allocate(
            tmp_path / "other",
            "feature/two",
            tmp_path / "repo" / "lane",
            owner_id="agent:two",
            session_id="session:two",
            request_id="request:three",
        )


def test_restart_heartbeat_and_fence_refusal_with_injected_clock(
    tmp_path: Path,
) -> None:
    clock = Clock()
    path = tmp_path / "lanes.sqlite"
    authority = FakeDurableLaneAuthority(clock=clock)
    registry = _registry(path, clock=clock, authority=authority)
    record = _allocate(registry, tmp_path / "repo")
    record = registry.activate(
        record.lane_id, owner_id=record.owner_id or "", fence=record.fence
    )
    clock.value += timedelta(seconds=15)
    beat = registry.heartbeat(
        record.lane_id,
        owner_id=record.owner_id or "",
        fence=record.fence,
        observed_disk_bytes=42,
    )
    assert beat.observed_disk_bytes == 42
    with pytest.raises(StaleLaneFence):
        registry.finish(record.lane_id, owner_id="agent:other", fence=record.fence)
    registry.close()
    restarted = _registry(path, clock=clock, authority=authority)
    assert restarted.require(record.lane_id).heartbeat_at == beat.heartbeat_at


def test_quota_refusal_reports_exact_usage_before_create(tmp_path: Path) -> None:
    policy = LaneQuotaPolicy(max_per_agent=1, max_predicted_disk_bytes=100)
    authority = FakeDurableLaneAuthority(quota=policy, clock=Clock())
    registry = _registry(
        tmp_path / "lanes.sqlite",
        quota=policy,
        clock=Clock(),
        authority=authority,
    )
    _allocate(registry, tmp_path / "repo", predicted_disk_bytes=60)
    with pytest.raises(LaneQuotaError) as exc_info:
        _allocate(
            registry,
            tmp_path / "other",
            request_id="request:two",
            branch="feature/two",
            owner_id="agent:one",
            predicted_disk_bytes=20,
        )
    decision = exc_info.value.decision
    assert decision.scope == "agent"
    assert decision.usage.total_active == 1
    assert decision.usage.predicted_disk_bytes == 60


def test_disk_accounting_is_cached_bounded_and_does_not_follow_symlink(
    tmp_path: Path,
) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "small").write_bytes(b"1234")
    outside = tmp_path / "outside"
    outside.write_bytes(b"x" * 1000)
    (root / "link").symlink_to(outside)
    ticks = iter((1.0, 2.0, 3.0))
    probe = DiskAccountingProbe(ttl_seconds=5, clock=lambda: next(ticks))
    first = probe.measure(root)
    second = probe.measure(root)
    assert first.bytes == 4
    assert first.skipped_symlinks == 1
    assert second is first
    bounded = DiskAccountingProbe(max_entries=1).measure(root)
    assert bounded.bounded


def test_legacy_discovery_requires_explicit_adoption_and_rollback_retains_rows(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path / "lanes.sqlite", clock=Clock())
    legacy = registry.observe_legacy(
        tmp_path / "repo", "feature/legacy", tmp_path / "lane"
    )
    assert legacy.state == LaneLifecycleState.OBSERVED_LEGACY
    adopted = registry.adopt(
        legacy.lane_id,
        owner_id="agent:one",
        session_id="session:one",
        host_id="host:one",
        operator_id="operator:one",
    )
    assert adopted.state == LaneLifecycleState.ACTIVE
    assert adopted.fence != legacy.fence
    registry.set_allocation_enabled(False)
    with pytest.raises(LaneAllocationDisabled):
        _allocate(registry, tmp_path / "new", request_id="request:new")
    assert registry.require(adopted.lane_id).state == LaneLifecycleState.ACTIVE


def test_expiry_refuses_dirty_jobs_candidates_concepts_and_missing_anchor(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    lane_root = tmp_path / "repo"
    (lane_root / "lane").mkdir(parents=True)
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, lane_root)
    record = registry.activate(record.lane_id, owner_id="agent:one", fence=record.fence)
    # Make the lane expired while retaining the exact current owner/fence.
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    git = ReadOnlyGit(anchor=False)
    candidate = LaneReclaimer(registry, git=git, clock=clock).assess(
        record, now=clock.value
    )
    assert not candidate.eligible
    assert "backup_anchor" in candidate.refusal_codes

    blocked = record.model_copy(
        update={
            "active_job_ids": ("job:one",),
            "active_candidate_id": "candidate:one",
            "concept_ids": ("concept:one",),
        }
    )
    # The registry projection remains authoritative; this assertion documents
    # all three independent claim checks without mutating the durable row.
    checks = LaneReclaimer(registry, git=git, clock=clock).assess(
        blocked, now=clock.value
    )
    assert {"active_job", "active_candidate", "concept_claim"}.issubset(
        set(checks.refusal_codes)
    )


def test_reclamation_fails_closed_for_missing_and_failing_process_probes(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, tmp_path / "repo")
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    missing = LaneReclaimer(registry, git=ReadOnlyGit(), clock=clock).assess(record)
    assert not missing.eligible
    assert "live_process" in missing.refusal_codes
    failing = LaneReclaimer(
        registry,
        git=ReadOnlyGit(),
        process_probe=lambda _lane: (_ for _ in ()).throw(RuntimeError("probe")),
        clock=clock,
    ).assess(record)
    assert not failing.eligible
    assert "live_process" in failing.refusal_codes


def test_reclamation_refuses_unknown_occupancy_and_forged_anchor(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, tmp_path / "repo")
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    record = registry.record_cleanup_anchor(
        record.lane_id,
        "forged arbitrary anchor",
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    assessment = LaneReclaimer(
        registry,
        git=ReadOnlyGit(anchor=True),
        process_probe=lambda _lane: False,
        job_probe=lambda _lane: False,
        candidate_probe=lambda _lane: False,
        concept_probe=lambda _lane: False,
        occupancy_probe=lambda _lane: (_ for _ in ()).throw(RuntimeError("unknown")),
        clock=clock,
    ).assess(record)
    assert not assessment.eligible
    assert {"backup_anchor", "occupied"}.issubset(set(assessment.refusal_codes))


def test_stale_and_restart_cleanup_plans_require_current_durable_job(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    lane_root = tmp_path / "repo"
    (lane_root / "lane").mkdir(parents=True)
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, lane_root)
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    record = registry.record_cleanup_anchor(
        record.lane_id,
        "refs/lane-backup/feature",
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    cleanup = CleanupAuthority()
    manager = FakeWorktreeManager()
    first = LaneReclaimer(
        registry,
        worktree_manager=manager,
        git=ReadOnlyGit(),
        process_probe=lambda _lane: False,
        job_probe=lambda _lane: False,
        candidate_probe=lambda _lane: False,
        concept_probe=lambda _lane: False,
        occupancy_probe=lambda _lane: False,
        cleanup_authority=cleanup,
        clock=clock,
    )
    preview = first.plan_cleanup(
        record.lane_id, owner_id="agent:one", fence=record.fence
    )
    with pytest.raises(CleanupRefused):
        first.execute_cleanup(preview)
    durable = first.request_cleanup(preview, submit=cleanup.submit)
    cleanup.job_fence = "rotated-fence"
    with pytest.raises(CleanupRefused):
        first.execute_cleanup(durable)
    cleanup.job_fence = "cleanup-fence:one"
    restarted = LaneReclaimer(
        registry,
        worktree_manager=manager,
        git=ReadOnlyGit(),
        process_probe=lambda _lane: False,
        job_probe=lambda _lane: False,
        candidate_probe=lambda _lane: False,
        concept_probe=lambda _lane: False,
        occupancy_probe=lambda _lane: False,
        cleanup_authority=cleanup,
        clock=clock,
    )
    assert restarted.execute_cleanup(durable)["ok"]


def test_successful_cleanup_uses_guarded_remove_and_is_reconciliation_visible(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    path = tmp_path / "repo"
    path.mkdir()
    lane_path = path / "lane"
    lane_path.mkdir()
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, path)
    record = registry.activate(record.lane_id, owner_id="agent:one", fence=record.fence)
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    # A durable anchor is part of the authority record and avoids a Git write
    # in the read-only assessment.
    record = registry.record_cleanup_anchor(
        record.lane_id,
        "refs/lane-backup/feature",
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    manager = FakeWorktreeManager()
    cleanup_authority = CleanupAuthority()
    reclaimer = LaneReclaimer(
        registry,
        worktree_manager=manager,
        git=ReadOnlyGit(),
        process_probe=lambda _lane: False,
        job_probe=lambda _lane: False,
        candidate_probe=lambda _lane: False,
        concept_probe=lambda _lane: False,
        occupancy_probe=lambda _lane: False,
        cleanup_authority=cleanup_authority,
        clock=clock,
    )
    plan = reclaimer.plan_cleanup(
        record.lane_id, owner_id="agent:one", fence=record.fence
    )
    assert plan.ok
    assert plan.preview_only
    with pytest.raises(CleanupRefused):
        reclaimer.execute_cleanup(plan)
    plan = reclaimer.request_cleanup(plan, submit=cleanup_authority.submit)
    assert plan.executable
    result = reclaimer.execute_cleanup(plan)
    assert result["ok"]
    assert manager.calls == [(str(path.resolve()), "feature/one")]
    assert registry.require(record.lane_id).state == LaneLifecycleState.QUARANTINED
    assert reclaimer.execute_cleanup(plan)["idempotent"]

    findings = registry.reconcile(
        [{"path": str(lane_path.resolve()), "branch": "feature/one"}]
    )
    assert any(
        item.classification == ReconciliationClass.STATE_MISMATCH for item in findings
    )


def test_quarantined_lane_without_receipt_cannot_claim_removal(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    path = tmp_path / "repo"
    (path / "lane").mkdir(parents=True)
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, path)
    record = registry.activate(record.lane_id, owner_id="agent:one", fence=record.fence)
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    record = registry.record_cleanup_anchor(
        record.lane_id,
        "refs/lane-backup/feature",
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    authority = CleanupAuthority()
    manager = FakeWorktreeManager()
    reclaimer = LaneReclaimer(
        registry,
        worktree_manager=manager,
        git=ReadOnlyGit(),
        process_probe=lambda _lane: False,
        job_probe=lambda _lane: False,
        candidate_probe=lambda _lane: False,
        concept_probe=lambda _lane: False,
        occupancy_probe=lambda _lane: False,
        cleanup_authority=authority,
        clock=clock,
    )
    preview = reclaimer.plan_cleanup(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
    )
    registry.quarantine(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        reason="operator quarantine before cleanup receipt",
    )
    durable = reclaimer.request_cleanup(preview, submit=authority.submit)

    result = reclaimer.execute_cleanup(durable)

    assert result["ok"] is False
    assert result["reconciliation_pending"] is True
    assert result["removal_performed"] is False
    assert manager.calls == []
    assert (path / "lane").is_dir()
    assert registry.require(record.lane_id).state == LaneLifecycleState.QUARANTINED


def test_cleanup_never_claims_success_before_durable_receipt_and_retries_by_receipt(
    tmp_path: Path,
) -> None:
    clock = Clock(NOW + timedelta(hours=2))
    path = tmp_path / "repo"
    (path / "lane").mkdir(parents=True)
    registry = _registry(tmp_path / "lanes.sqlite", clock=clock)
    record = _allocate(registry, path)
    record = registry.heartbeat(
        record.lane_id,
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    record = registry.record_cleanup_anchor(
        record.lane_id,
        "refs/lane-backup/feature",
        owner_id="agent:one",
        fence=record.fence,
        now=NOW - timedelta(hours=2),
    )
    authority = ReceiptFailAuthority()
    manager = FakeWorktreeManager()
    reclaimer = LaneReclaimer(
        registry,
        worktree_manager=manager,
        git=ReadOnlyGit(),
        process_probe=lambda _lane: False,
        job_probe=lambda _lane: False,
        candidate_probe=lambda _lane: False,
        concept_probe=lambda _lane: False,
        occupancy_probe=lambda _lane: False,
        cleanup_authority=authority,
        clock=clock,
    )
    plan = reclaimer.request_cleanup(
        reclaimer.plan_cleanup(
            record.lane_id, owner_id="agent:one", fence=record.fence
        ),
        submit=authority.submit,
    )
    first = reclaimer.execute_cleanup(plan)
    assert not first["ok"]
    assert first["removal_performed"] is True
    assert registry.require(record.lane_id).state == LaneLifecycleState.EXPIRED
    authority.fail_receipt = False
    assert authority.pending is not None
    assert authority.record_cleanup_complete(authority.pending)
    second = reclaimer.execute_cleanup(plan)
    assert second["ok"]
    assert second["idempotent"] is True
    assert registry.require(record.lane_id).state == LaneLifecycleState.QUARANTINED
    assert reclaimer.execute_cleanup(plan)["receipt"] is True
