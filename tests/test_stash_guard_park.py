"""Qualification of the generalized lane-scoped stash guard."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repository_manager import stash_guard
from repository_manager.destructive_guard import _GitAdapter
from tests.fixtures.tree_mutation_hazards import shared_stash


def _git(
    args: list[str], path: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(path), capture_output=True, text=True, check=check
    )


def _repo(path: Path) -> Path:
    path.mkdir()
    _git(["init", "-b", "main"], path)
    _git(["config", "user.email", "stash@test"], path)
    _git(["config", "user.name", "stash"], path)
    (path / "tracked.txt").write_text("base\n")
    _git(["add", "-A"], path)
    _git(["commit", "-m", "base"], path)
    return path


def test_park_and_unpark_use_the_lane_private_ref_and_leave_shared_stash_empty(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "park")
    (repo / "tracked.txt").write_text("lane WIP\n")
    (repo / "untracked.txt").write_text("untracked WIP\n")

    parked = stash_guard.park(_GitAdapter(), str(repo), lane="lane-a")

    assert parked["ok"] is True
    assert parked["parked"] is True
    assert parked["ref"] == "refs/lane/lane-a/stash"
    assert _git(["status", "--porcelain"], repo).stdout.strip() == ""
    assert _git(["stash", "list"], repo).stdout.strip() == ""
    restored = stash_guard.unpark(
        _GitAdapter(), str(repo), ref=parked["ref"], lane="lane-a"
    )

    assert restored["ok"] is True
    assert (repo / "tracked.txt").read_text() == "lane WIP\n"
    assert (repo / "untracked.txt").read_text() == "untracked WIP\n"
    assert (
        _git(
            ["rev-parse", "--verify", "--quiet", parked["ref"]], repo, check=False
        ).returncode
        != 0
    )
    assert _git(["stash", "list"], repo).stdout.strip() == ""


def test_park_never_consumes_an_existing_shared_stash_entry(tmp_path: Path) -> None:
    repo = shared_stash(tmp_path)
    existing = _git(["stash", "list"], repo).stdout.strip()
    (repo / "README.md").write_text("new lane WIP\n")
    (repo / "new-untracked.txt").write_text("new\n")

    parked = stash_guard.park(_GitAdapter(), str(repo), lane="lane-independent")

    assert parked["ok"] is True
    assert _git(["stash", "list"], repo).stdout.strip() == existing
    restored = stash_guard.unpark(
        _GitAdapter(), str(repo), ref=parked["ref"], lane="lane-independent"
    )
    assert restored["ok"] is True
    assert (repo / "README.md").read_text() == "new lane WIP\n"
    assert (repo / "new-untracked.txt").read_text() == "new\n"
    assert (
        _git(
            ["rev-parse", "--verify", "--quiet", parked["ref"]], repo, check=False
        ).returncode
        != 0
    )
    assert _git(["stash", "list"], repo).stdout.strip() == existing


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_park_captures_hidden_tracked_wip_despite_empty_porcelain(
    tmp_path: Path, flag: str
) -> None:
    repo = _repo(tmp_path / flag.removeprefix("--"))
    (repo / "tracked.txt").write_text("hidden WIP\n")
    _git(["update-index", flag, "tracked.txt"], repo)
    (repo / "tracked.txt").write_text("hidden WIP\n")
    assert _git(["status", "--porcelain"], repo).stdout.strip() == ""

    parked = stash_guard.park(_GitAdapter(), str(repo), lane="hidden")

    assert parked["ok"] is True
    assert parked["parked"] is True
    assert (repo / "tracked.txt").read_text() == "base\n"
    restored = stash_guard.unpark(
        _GitAdapter(), str(repo), ref=parked["ref"], lane="hidden"
    )
    assert restored["ok"] is True
    assert (repo / "tracked.txt").read_text() == "hidden WIP\n"


def test_park_untracked_symlink_cleanup_does_not_follow_outside_tree(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "symlink")
    outside = tmp_path / "outside.txt"
    outside.write_text("outside must survive\n")
    (repo / "outside-link").symlink_to(outside)

    parked = stash_guard.park(_GitAdapter(), str(repo), lane="symlink")

    assert parked["ok"] is True
    assert outside.read_text() == "outside must survive\n"
    assert not (repo / "outside-link").exists()
    restored = stash_guard.unpark(
        _GitAdapter(), str(repo), ref=parked["ref"], lane="symlink"
    )
    assert restored["ok"] is True
    assert (repo / "outside-link").is_symlink()
    assert (repo / "outside-link").resolve() == outside.resolve()
