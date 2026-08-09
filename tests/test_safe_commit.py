"""RMDD-26 safe-commit qualification against disposable git repositories."""

from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from repository_manager import stash_guard, tree_repair
from repository_manager.safe_commit import safe_commit
from tests.fixtures.tree_mutation_hazards import unstaged_deletions_with_hook


def _git(
    args: list[str], path: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(path), capture_output=True, text=True, check=check
    )


def _repo(path: Path, branch: str) -> Path:
    path.mkdir()
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "safe@test"], path)
    _git(["config", "user.name", "safe"], path)
    (path / "tracked.txt").write_text("base\n")
    _git(["add", "-A"], path)
    _git(["commit", "-m", "base"], path)
    if branch != "main":
        lane = path.parent / f"{path.name}-{branch}"
        _git(["worktree", "add", str(lane), "-b", branch, "main"], path)
        return lane
    return path


def test_safe_commit_preserves_unstaged_deletion_and_proves_gate_snapshot(
    tmp_path: Path,
) -> None:
    """The exact deletion-plus-hook shape closes the staged-only window."""
    repo = unstaged_deletions_with_hook(tmp_path)
    gate_observations: list[list[str]] = []

    def gate(path: Path) -> bool:
        status = _git(["diff", "--name-only"], path).stdout.splitlines()
        gate_observations.append(status)
        return not status

    result = safe_commit(repo, "safe deletion", gate=gate)

    assert result["ok"] is True
    assert result["nothing_left_unstaged"] is True
    assert result["gate_invoked"] is True
    assert gate_observations == [[]]
    assert "must-delete.txt" in result["staged_paths"]
    assert "staged-change.txt" in result["staged_paths"]
    stat = _git(["show", "--format=", "--stat", "HEAD"], repo).stdout
    assert "must-delete.txt" in stat
    assert not (repo / "must-delete.txt").exists()


def test_same_fixture_bare_commit_loses_the_deletion_negative_control(
    tmp_path: Path,
) -> None:
    """The fixture is known-bad before the safe path is trusted."""
    repo = unstaged_deletions_with_hook(tmp_path)
    _git(["add", "staged-change.txt"], repo)
    _git(["commit", "-m", "unsafe staged-only commit"], repo)

    # The hook's staged-only restoration brings the unstaged deletion back.
    assert (repo / "must-delete.txt").exists()
    assert (
        "must-delete.txt"
        not in _git(["show", "--format=", "--name-only", "HEAD"], repo).stdout
    )
    assert _git(["status", "--porcelain"], repo).stdout.strip() == ""


def test_safe_commit_supports_an_explicit_configured_gate(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "configured-gate", "main")
    (repo / "tracked.txt").write_text("changed\n")
    called: list[Path] = []

    def gate(path: Path) -> dict[str, object]:
        called.append(path)
        return {"ok": True}

    result = safe_commit(repo, "configured gate", gate=gate)

    assert result["status"] == "success"
    assert result["gate_stage"] == "configured"
    assert called == [repo]


def test_safe_commit_can_create_an_explicitly_deferred_snapshot(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "deferred-snapshot", "main")
    (repo / "tracked.txt").write_text("deferred\n")
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 91\n")
    hook.chmod(0o755)

    result = safe_commit(repo, "deferred snapshot", defer_gate=True)

    assert result["ok"] is True
    assert result["gate_deferred"] is True
    assert result["gate_invoked"] is False
    assert result["gate_stage"] == "deferred"
    assert _git(["show", "--format=%s", "-s", "HEAD"], repo).stdout.strip() == (
        "deferred snapshot"
    )


def test_successful_commit_records_baseline_for_subsequent_diagnosis(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "baseline-after-commit", "main")
    (repo / "tracked.txt").write_text("new baseline\n")

    result = safe_commit(repo, "record baseline")

    assert result["status"] == "success"
    assert result["baseline_recorded"] is True
    assert result["baseline"]["head_sha"] == result["commit_sha"]
    diagnosis = tree_repair.diagnose(repo)
    assert diagnosis["finding"] == "clean"
    assert diagnosis["baseline"]["head_sha"] == result["commit_sha"]


def test_commit_reports_when_baseline_persistence_is_not_confirmed(
    tmp_path: Path, monkeypatch
) -> None:
    repo = _repo(tmp_path / "baseline-warning", "main")
    (repo / "tracked.txt").write_text("baseline warning\n")
    monkeypatch.setattr(
        tree_repair,
        "record_baseline",
        lambda path: {"ok": False, "error": "administrative directory is read-only"},
    )

    result = safe_commit(repo, "report baseline warning")

    assert result["status"] == "success"
    assert result["baseline_recorded"] is False
    assert "read-only" in result["baseline_error"]


def test_safe_commit_refuses_same_tree_lease_interleave(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "busy", "main")
    (repo / "tracked.txt").write_text("busy\n")

    with stash_guard.hold_tree_mutation_lease(repo, note="test holder"):
        result = safe_commit(repo, "must wait")

    assert result["status"] == "error"
    assert result["reason"] == "tree-mutation-busy"


def test_configured_gate_receives_environment_and_timeout(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "gate-env", "main")
    (repo / "tracked.txt").write_text("changed\n")
    gate = tmp_path / "gate.py"
    gate.write_text(
        "import os, sys\n"
        "sys.exit(0 if os.environ.get('RMDD_GATE_MARKER') == 'present' else 1)\n"
    )

    result = safe_commit(
        repo,
        "gate env",
        gate=[sys.executable, str(gate)],
        env={"RMDD_GATE_MARKER": "present"},
        timeout=7,
    )

    assert result["status"] == "success"


def test_two_safe_commits_in_different_worktrees_do_not_interact(
    tmp_path: Path,
) -> None:
    canonical = _repo(tmp_path / "parallel", "main")
    lane_a = canonical.parent / "parallel-lane-a"
    _git(["worktree", "add", str(lane_a), "-b", "lane-a", "main"], canonical)
    lane_b = canonical.parent / "parallel-lane-b"
    _git(["worktree", "add", str(lane_b), "-b", "lane-b", "main"], canonical)
    (lane_a / "tracked.txt").write_text("A\n")
    (lane_b / "tracked.txt").write_text("B\n")

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                lambda item: safe_commit(item[0], item[1]),
                [(lane_a, "lane A"), (lane_b, "lane B")],
            )
        )

    assert [result["status"] for result in results] == ["success", "success"]
    assert (
        _git(["show", "-s", "--format=%s", "HEAD"], lane_a).stdout.strip() == "lane A"
    )
    assert (
        _git(["show", "-s", "--format=%s", "HEAD"], lane_b).stdout.strip() == "lane B"
    )
