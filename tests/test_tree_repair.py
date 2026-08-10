"""RMDD-26 structural corruption diagnosis and guarded repair."""

from __future__ import annotations

import subprocess
from pathlib import Path

from repository_manager import tree_repair
from repository_manager.stash_guard import hold_tree_mutation_lease
from tests.fixtures.tree_mutation_hazards import core_bare_drift, truncated_index


def _git(
    args: list[str], path: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(path), capture_output=True, text=True, check=check
    )


def test_index_wipe_is_detected_repaired_and_content_is_unchanged(
    tmp_path: Path,
) -> None:
    repo = truncated_index(tmp_path, files=4634, retained=5)
    before = {
        path.relative_to(repo): path.read_bytes() for path in repo.glob("tracked-*.txt")
    }

    diagnosis = tree_repair.diagnose(repo)
    assert diagnosis["finding"] == "probable-index-wipe"
    assert diagnosis["evidence"]["baseline_count"] == 4635
    assert diagnosis["evidence"]["indexed_count"] == 5

    repaired = tree_repair.repair(repo, finding=diagnosis)
    assert repaired["ok"] is True
    assert repaired["content_preserved"] is True
    assert repaired["checksum_before"] == repaired["checksum_after"]
    assert len(_git(["ls-files"], repo).stdout.splitlines()) == 4635
    assert {
        path.relative_to(repo): path.read_bytes() for path in repo.glob("tracked-*.txt")
    } == before
    assert tree_repair.diagnose(repo)["finding"] == "clean"
    assert tree_repair.repair(repo, finding="clean")["status"] == "noop"


def test_core_bare_drift_is_repaired_without_touching_files(tmp_path: Path) -> None:
    repo = core_bare_drift(tmp_path)
    before = (repo / "README.md").read_bytes()
    assert _git(["status", "--porcelain"], repo, check=False).returncode != 0

    diagnosis = tree_repair.diagnose(repo)
    assert diagnosis["finding"] == "core-bare-drift"
    repaired = tree_repair.repair(repo, finding="core-bare-drift")

    assert repaired["ok"] is True
    assert repaired["content_preserved"] is True
    assert (repo / "README.md").read_bytes() == before
    assert (
        _git(["config", "--bool", "--get", "core.bare"], repo).stdout.strip() == "false"
    )
    assert _git(["status", "--porcelain"], repo).returncode == 0
    assert tree_repair.diagnose(repo)["finding"] == "clean"
    assert tree_repair.repair(repo, finding="clean")["status"] == "noop"


def test_forged_index_wipe_request_cannot_erase_healthy_staged_index(
    tmp_path: Path,
) -> None:
    repo = core_bare_drift(tmp_path)
    tree_repair.repair(repo, finding="core-bare-drift")
    (repo / "staged.txt").write_text("staged but not committed\n")
    _git(["add", "staged.txt"], repo)
    before = _git(["diff", "--cached", "--name-only"], repo).stdout

    refused = tree_repair.repair(repo, finding="probable-index-wipe")

    assert refused["ok"] is False
    assert refused["status"] == "refused"
    assert refused["actual_finding"] == "clean"
    assert refused["content_preserved"] is True
    assert _git(["diff", "--cached", "--name-only"], repo).stdout == before
    assert (repo / "staged.txt").read_text() == "staged but not committed\n"


def test_stale_index_wipe_finding_cannot_restore_a_legitimate_deletion(
    tmp_path: Path,
) -> None:
    repo = truncated_index(tmp_path, files=20, retained=1)
    stale = tree_repair.diagnose(repo)
    assert stale["finding"] == "probable-index-wipe"
    assert tree_repair.repair(repo, finding=stale)["ok"] is True
    for index in range(19):
        (repo / f"tracked-{index:04d}.txt").unlink()
    _git(["add", "-u"], repo)
    before = _git(["diff", "--cached", "--name-status"], repo).stdout

    refused = tree_repair.repair(repo, finding=stale)

    assert refused["ok"] is False
    assert refused["status"] == "refused"
    assert "trusted baseline" in refused["error"]
    assert _git(["diff", "--cached", "--name-status"], repo).stdout == before
    assert not (repo / "tracked-0000.txt").exists()


def test_repair_refuses_managed_interleave_while_lease_is_held(tmp_path: Path) -> None:
    repo = core_bare_drift(tmp_path)
    with hold_tree_mutation_lease(repo, note="test interleave"):
        refused = tree_repair.repair(repo, finding="core-bare-drift")
    assert refused["status"] == "refused"
    assert refused["reason"] == "tree-mutation-busy"
    assert (
        _git(["config", "--bool", "--get", "core.bare"], repo).stdout.strip() == "true"
    )


def test_one_hundred_clean_diagnoses_have_no_false_positive(tmp_path: Path) -> None:
    repo = core_bare_drift(tmp_path)
    tree_repair.repair(repo, finding="core-bare-drift")
    reports = [tree_repair.diagnose(repo) for _ in range(100)]

    assert all(report["finding"] == "clean" for report in reports)
    assert all(report["ok"] for report in reports)
