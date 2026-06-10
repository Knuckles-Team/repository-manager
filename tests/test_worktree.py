"""Tests for WorktreeManager (CONCEPT:RM-WORKTREE) against real git repos."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from repository_manager import worktree as wt_mod
from repository_manager.worktree import WorktreeManager


class FakeGit:
    """Minimal Git stand-in exposing the surface WorktreeManager uses:
    ``git_action`` (returns a GitResult-shaped object), ``project_map``, ``path``.
    """

    def __init__(self, workspace, project_map):
        self.path = workspace
        self.project_map = project_map

    def git_action(self, command, path=None, quiet=False, **_):
        p = subprocess.run(
            command, shell=True, cwd=path or self.path,
            capture_output=True, text=True,
        )
        out = (p.stdout + p.stderr).strip()
        return SimpleNamespace(
            status="success" if p.returncode == 0 else "error",
            data=out,
            error=None if p.returncode == 0
            else SimpleNamespace(message=out, code=p.returncode),
            metadata=SimpleNamespace(return_code=p.returncode),
        )


def _run(cmd, cwd):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True,
                   capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo 'myrepo' on main with one commit, plus an isolated
    WORKTREE_ROOT so tests never touch the real /home/apps/worktrees."""
    ws = tmp_path / "workspace"
    repo_path = ws / "myrepo"
    repo_path.mkdir(parents=True)
    _run("git init -b main", repo_path)
    _run("git config user.email t@t.io && git config user.name t", repo_path)
    (repo_path / "README.md").write_text("hello\n")
    _run("git add -A && git commit -q -m init", repo_path)

    monkeypatch.setattr(wt_mod, "WORKTREE_ROOT", str(tmp_path / "worktrees"))
    git = FakeGit(str(ws), {"git@x/myrepo.git": str(repo_path)})
    return SimpleNamespace(wm=WorktreeManager(git), path=str(repo_path))


def test_add_creates_worktree_on_branch(repo):
    res = repo.wm.add("myrepo", "feat-x")
    assert res["ok"] and res["created"]
    assert os.path.isdir(res["path"])
    # the worktree is checked out on feat-x
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD", shell=True, cwd=res["path"],
        capture_output=True, text=True,
    ).stdout.strip()
    assert branch == "feat-x"


def test_add_is_idempotent(repo):
    a = repo.wm.add("myrepo", "feat-x")
    b = repo.wm.add("myrepo", "feat-x")
    assert a["created"] is True
    assert b["created"] is False and b["status"] == "exists"


def test_list_reports_linked_worktree(repo):
    repo.wm.add("myrepo", "feat-x")
    listing = repo.wm.list_worktrees("myrepo")
    branches = {w.get("branch") for w in listing["worktrees"]}
    assert "feat-x" in branches
    linked = [w for w in listing["worktrees"] if w["branch"] == "feat-x"][0]
    assert linked["linked"] is True


def test_merge_back_to_main(repo):
    res = repo.wm.add("myrepo", "feat-x")
    (os.path.join(res["path"], "feature.txt"))
    open(os.path.join(res["path"], "feature.txt"), "w").write("x")
    _run("git add -A && git commit -q -m feat", res["path"])
    merged = repo.wm.merge("myrepo", "feat-x")
    assert merged["ok"] and not merged["conflict"]
    # canonical main now contains the feature file
    assert os.path.isfile(os.path.join(repo.path, "feature.txt"))


def test_remove_worktree(repo):
    res = repo.wm.add("myrepo", "feat-x")
    rm = repo.wm.remove("myrepo", "feat-x", force=True)
    assert rm["ok"]
    assert not os.path.isdir(res["path"])


def test_adopt_moves_wip_onto_branch(repo):
    # uncommitted WIP in the canonical checkout
    open(os.path.join(repo.path, "wip.txt"), "w").write("work in progress")
    res = repo.wm.add("myrepo", "feat-adopt", adopt=True)
    assert res["ok"] and res["adopted"]
    # WIP now lives in the worktree, and the canonical tree is clean
    assert os.path.isfile(os.path.join(res["path"], "wip.txt"))
    status = subprocess.run(
        "git status --porcelain", shell=True, cwd=repo.path,
        capture_output=True, text=True,
    ).stdout.strip()
    assert status == ""


def test_unknown_repo_errors(repo):
    res = repo.wm.add("nonexistent", "feat-x")
    assert res["ok"] is False and "not found" in res["error"]
