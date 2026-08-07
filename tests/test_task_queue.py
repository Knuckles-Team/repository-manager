"""Tests for the generic task/reservation ledger (CONCEPT:RM-TASK-LEDGER).

Every test drives a REAL git repository (the ledger's per-repo store lives
under that repo's shared ``--git-common-dir``) and REAL ``fcntl`` leases via
``agent_utilities.governance.lanes`` — the mechanism under test is file-backed
mutual exclusion plus fold ordering, and a mock of either would prove
nothing about it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repository_manager import task_queue as tq


def _run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(
        cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    root.mkdir()
    _run("git init -q -b main", root)
    _run("git config user.email t@t.io && git config user.name t", root)
    _run("git config commit.gpgsign false", root)
    (root / "README.md").write_text("hi\n")
    _run("git add -A && git commit -q -m init", root)
    return root


# ---------------------------------------------------------------------------
# fold-by-recorded_at — ported directly from the merge queue's D-CVG-9 proof
# ---------------------------------------------------------------------------
def test_resolve_latest_record_prefers_recorded_at_over_list_order():
    """This is the exact D-CVG-9 shape: FragmentStore.fold() groups records in
    LANE-NAME-alphabetical order, so `"canonical"` sorts before `"lane-foo"`
    regardless of which one wrote more recently. A naive `group[-1]` would
    therefore pick `lane-foo`'s STALE `queued` record over `canonical`'s
    fresher `landed` one — reviving a dead candidate / reporting `queued`
    forever. Assert BOTH: the real function gets it right, and the naive
    fallback it replaced would have gotten it wrong (so this test actually
    pins the fix rather than merely exercising the code).
    """
    group = [
        # "canonical" iterates first (alphabetically), but wrote SECOND.
        {
            "id": "x",
            "lane": "canonical",
            "state": "landed",
            "recorded_at": "2026-01-01T12:00:00+00:00",
        },
        # "lane-foo" iterates second, but wrote FIRST.
        {
            "id": "x",
            "lane": "lane-foo",
            "state": "queued",
            "recorded_at": "2026-01-01T10:00:00+00:00",
        },
    ]
    naive = group[-1]
    assert naive["state"] == "queued", (
        "fixture must reproduce the exact stale-pick shape"
    )

    resolved = tq.resolve_latest_record(group)
    assert resolved["state"] == "landed"
    assert resolved["lane"] == "canonical"


def test_resolve_latest_record_falls_back_to_list_order_without_timestamps():
    group = [{"id": "x", "state": "queued"}, {"id": "x", "state": "landed"}]
    assert tq.resolve_latest_record(group)["state"] == "landed"


# ---------------------------------------------------------------------------
# Task / TaskStore — the generic append-only record model
# ---------------------------------------------------------------------------
def test_enqueue_and_queued_round_trip(repo: Path):
    task = tq.enqueue_task("t1", "widget", repo="repo", payload={"a": 1}, path=repo)
    assert task.state == tq.QUEUED
    pending = tq.queued_tasks("widget", path=repo)
    assert [t.id for t in pending] == ["t1"]
    assert pending[0].payload == {"a": 1}


def test_record_state_supersedes_via_new_append_not_edit(repo: Path):
    task = tq.enqueue_task("t1", "widget", path=repo)
    tq.record_state(task, tq.DONE, "ok", path=repo)
    fragment = tq.task_store("widget", repo).fragment_for(tq.lane_scope(repo).lane)
    # Two records were APPENDED for the same id — never rewritten in place.
    assert len(fragment.read_text().splitlines()) == 2
    all_records = tq.all_tasks("widget", path=repo)
    assert len(all_records) == 1
    assert all_records[0].state == tq.DONE
    assert tq.queued_tasks("widget", path=repo) == []


def test_record_state_across_lanes_folds_to_the_terminal_one(repo: Path):
    """Same cross-lane scenario the merge queue's own tests pin: one lane
    enqueues, a DIFFERENT lane (a driver) records the terminal state, and the
    fold must resolve to the terminal one regardless of fragment write order.
    """
    scope = tq.lane_scope(repo)
    store = tq.task_store("widget", repo)
    store.append(
        {
            "id": "t1",
            "kind": "widget",
            "lane": "lane-foo",
            "state": tq.QUEUED,
            "enqueued_at": "2026-01-01T09:00:00+00:00",
            "recorded_at": "2026-01-01T09:00:00+00:00",
            "payload": {},
            "reason": "",
            "repo": "repo",
        },
        lane="lane-foo",
    )
    store.append(
        {
            "id": "t1",
            "kind": "widget",
            "lane": scope.lane,
            "state": tq.DONE,
            "enqueued_at": "2026-01-01T09:00:00+00:00",
            "recorded_at": "2026-01-01T09:05:00+00:00",
            "payload": {},
            "reason": "driver finished it",
            "repo": "repo",
        },
        lane=scope.lane,
    )
    tasks = tq.all_tasks("widget", path=repo)
    assert len(tasks) == 1
    assert tasks[0].state == tq.DONE


def test_withdraw_unknown_task_refuses(repo: Path):
    with pytest.raises(tq.TaskQueueError):
        tq.withdraw_task("widget", "nope", path=repo)


def test_withdraw_known_task(repo: Path):
    tq.enqueue_task("t1", "widget", path=repo)
    tq.withdraw_task("widget", "t1", reason="changed my mind", path=repo)
    assert tq.queued_tasks("widget", path=repo) == []
    assert tq.all_tasks("widget", path=repo)[0].state == tq.WITHDRAWN


# ---------------------------------------------------------------------------
# ExecutionClass — scope + policy, colocation gate, pool bound
# ---------------------------------------------------------------------------
def test_unknown_execution_class_refuses_rather_than_inventing_a_policy(repo: Path):
    with pytest.raises(tq.TaskQueueError, match="unknown execution class"):
        with tq.acquire("nonexistent-class", operation="x", path=repo, colocated=True):
            pass


def test_exclusive_lease_backed_class_requires_colocation_proof(repo: Path):
    """Default (`colocated=None`) and explicit `False` must BOTH refuse —
    silently trusting a lock whose exclusion may not hold over NFS is exactly
    the false-safety failure mode this gate exists to close.
    """
    with pytest.raises(tq.ColocationRequired):
        with tq.acquire("merge-drain", operation="x", path=repo):
            pass
    with pytest.raises(tq.ColocationRequired):
        with tq.acquire("merge-drain", operation="x", path=repo, colocated=False):
            pass


def test_exclusive_lease_backed_class_works_when_colocated_is_proven(repo: Path):
    with tq.acquire(
        "merge-drain", operation="x", path=repo, colocated=True
    ) as reservation:
        assert reservation.lease_name == "merge-drain"
        assert reservation.policy == tq.Policy.EXCLUSIVE
    # Released on exit — a second acquire must succeed.
    with tq.acquire("merge-drain", operation="x", path=repo, colocated=True):
        pass


def test_exclusive_class_second_holder_is_refused_while_first_holds(repo: Path):
    from agent_utilities.governance.lanes import LeaseUnavailable

    with tq.acquire("merge-drain", operation="x", path=repo, colocated=True):
        with pytest.raises(LeaseUnavailable):
            with tq.acquire("merge-drain", operation="y", path=repo, colocated=True):
                pass


def test_partition_policy_needs_no_colocation_and_never_takes_a_lease(repo: Path):
    tq.register_execution_class(
        tq.ExecutionClass(
            "test-partition", scope=tq.Scope.WORKTREE, policy=tq.Policy.PARTITION
        )
    )
    # No colocated= at all — PARTITION holds no shared lease, so the gate does
    # not apply.
    with tq.acquire("test-partition", operation="x", path=repo) as reservation:
        assert reservation.lease_name is None
        assert reservation.partition_dir is not None
        assert reservation.partition_dir.is_dir()


def test_pool_policy_caps_concurrent_holders_and_refuses_the_next(repo: Path):
    tq.register_execution_class(
        tq.ExecutionClass(
            "test-pool", scope=tq.Scope.REPO, policy=tq.Policy.POOL, pool_size=2
        )
    )
    with tq.acquire("test-pool", operation="a", path=repo, colocated=True) as r1:
        with tq.acquire("test-pool", operation="b", path=repo, colocated=True) as r2:
            assert {r1.slot, r2.slot} == {0, 1}
            with pytest.raises(tq.TaskQueueError, match="no free slot"):
                with tq.acquire("test-pool", operation="c", path=repo, colocated=True):
                    pass
    # Both slots released — a fresh acquire succeeds again.
    with tq.acquire("test-pool", operation="d", path=repo, colocated=True):
        pass


def test_class_status_reports_live_holder(repo: Path):
    with tq.acquire("merge-drain", operation="x", path=repo, colocated=True):
        report = tq.class_status("merge-drain", repo)
        assert report["holder"] is not None
        assert report["holder"]["operation"] == "x"
    report = tq.class_status("merge-drain", repo)
    assert report["holder"] is None


# ---------------------------------------------------------------------------
# D-CDX-12 — GLOBAL scope must genuinely be host-wide: two DIFFERENT
# repositories contending for the SAME GLOBAL class must exclude each other,
# not silently get two independent per-repository lease files.
# ---------------------------------------------------------------------------
@pytest.fixture
def isolated_global_arbitration(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect the host-wide GLOBAL-lease root to a throwaway directory so
    these tests can never collide with (or be polluted by) a REAL concurrent
    lane's actual `uv-sync` lock on this shared host.
    """
    root = tmp_path / "global-arbitration"

    def _fake_workspace_arbitration_dir() -> Path:
        root.mkdir(parents=True, exist_ok=True)
        return root

    monkeypatch.setattr(tq, "workspace_arbitration_dir", _fake_workspace_arbitration_dir)
    return root


@pytest.fixture
def two_repos(tmp_path: Path) -> tuple[Path, Path]:
    a = tmp_path / "repo-a"
    b = tmp_path / "repo-b"
    for r in (a, b):
        r.mkdir()
        _run("git init -q -b main", r)
        _run("git config user.email t@t.io && git config user.name t", r)
        _run("git config commit.gpgsign false", r)
        (r / "README.md").write_text("hi\n")
        _run("git add -A && git commit -q -m init", r)
    return a, b


def test_global_scope_lease_excludes_a_different_repository(
    two_repos: tuple[Path, Path], isolated_global_arbitration: Path
):
    """The literal D-CDX-12 reproduction: before the fix, `hold_lease`'s
    per-repository default meant two DIFFERENT repos got two DIFFERENT lease
    files for the SAME declared GLOBAL "uv-sync" class and could both hold
    "slot 0" at once. Prove the barrier now actually holds across repos.
    """
    repo_a, repo_b = two_repos
    assert tq.EXECUTION_CLASSES["uv-sync"].scope == tq.Scope.GLOBAL

    with tq.acquire("uv-sync", operation="sync in repo-a", path=repo_a, colocated=True):
        # repo_a already holds one of the shared pool's slots; every
        # REMAINING slot in that SAME host-wide pool is contended from
        # repo_b too.
        pool_size = tq.EXECUTION_CLASSES["uv-sync"].pool_size
        held = []
        for _ in range(pool_size - 1):
            cm = tq.acquire(
                "uv-sync", operation="sync in repo-b", path=repo_b, colocated=True
            )
            reservation = cm.__enter__()
            held.append((cm, reservation))
        with pytest.raises(tq.TaskQueueError, match="no free slot"):
            with tq.acquire(
                "uv-sync", operation="one too many", path=repo_b, colocated=True
            ):
                pass
        for cm, _r in held:
            cm.__exit__(None, None, None)

    # released — repo_b can now acquire every slot again.
    with tq.acquire("uv-sync", operation="sync in repo-b again", path=repo_b, colocated=True):
        pass


def test_global_scope_lease_file_lives_under_the_host_wide_root_not_either_repo(
    two_repos: tuple[Path, Path], isolated_global_arbitration: Path
):
    repo_a, repo_b = two_repos
    with tq.acquire("uv-sync", operation="x", path=repo_a, colocated=True):
        lease_files = list(isolated_global_arbitration.rglob("*.lease"))
        assert len(lease_files) == 1, lease_files
        # never under either repository's own --git-common-dir.
        for repo in (repo_a, repo_b):
            git_common = Path(tq.lane_scope(repo).common_dir)
            assert not str(lease_files[0]).startswith(str(git_common))


def test_repo_scoped_class_remains_independent_per_repository(
    two_repos: tuple[Path, Path],
):
    """Control case: a REPO-scoped class (e.g. "merge-drain") must NOT be
    shared across repositories — only GLOBAL classes are host-wide."""
    repo_a, repo_b = two_repos
    with tq.acquire("merge-drain", operation="x", path=repo_a, colocated=True):
        # a different repo's SAME class name is unaffected.
        with tq.acquire("merge-drain", operation="y", path=repo_b, colocated=True):
            pass


def test_global_class_status_reads_the_same_host_wide_holder_from_either_repo(
    two_repos: tuple[Path, Path], isolated_global_arbitration: Path
):
    repo_a, repo_b = two_repos
    with tq.acquire("uv-sync", operation="x", path=repo_a, colocated=True):
        status_from_a = tq.class_status("uv-sync", repo_a)
        status_from_b = tq.class_status("uv-sync", repo_b)
        assert status_from_a["slots"] == status_from_b["slots"]
        assert any(s["holder"] is not None for s in status_from_a["slots"])


# ---------------------------------------------------------------------------
# D-CDX-14 — owner identity must be persisted and readable from status, for
# BOTH a repo-scoped and a GLOBAL-scoped class.
# ---------------------------------------------------------------------------
def test_owner_identity_is_persisted_for_a_repo_scoped_class(repo: Path):
    owner = {"fleet": "codex", "session": "codex-session-1"}
    with tq.acquire(
        "merge-drain", operation="x", path=repo, colocated=True, owner=owner
    ) as reservation:
        assert reservation.owner == owner
        status = tq.class_status("merge-drain", repo)
        assert status["holder"]["owner"] == owner


def test_owner_identity_is_persisted_for_a_global_scoped_class(
    repo: Path, isolated_global_arbitration: Path
):
    owner = {"fleet": "claude", "session": "claude-session-1"}
    with tq.acquire(
        "uv-sync", operation="x", path=repo, colocated=True, owner=owner
    ) as reservation:
        assert reservation.owner == owner
        status = tq.class_status("uv-sync", repo)
        held = [s for s in status["slots"] if s["holder"] is not None]
        assert len(held) == 1
        assert held[0]["holder"]["owner"] == owner
