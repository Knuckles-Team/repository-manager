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
            command,
            shell=True,
            cwd=path or self.path,
            capture_output=True,
            text=True,
        )
        out = (p.stdout + p.stderr).strip()
        return SimpleNamespace(
            status="success" if p.returncode == 0 else "error",
            data=out,
            error=None
            if p.returncode == 0
            else SimpleNamespace(message=out, code=p.returncode),
            metadata=SimpleNamespace(return_code=p.returncode),
        )


def _run(cmd, cwd):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A real git repo 'myrepo' on main with one commit, plus an isolated
    WORKTREE_ROOT so tests never touch an operator worktree directory."""
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
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=res["path"],
        capture_output=True,
        text=True,
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
        "git status --porcelain",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert status == ""


def test_unknown_repo_errors(repo):
    res = repo.wm.add("nonexistent", "feat-x")
    assert res["ok"] is False and "not found" in res["error"]


# ── audit (CONCEPT:RM-WORKTREE-AUDIT) ─────────────────────────────────────


def _commit_in(path, name, msg, env=None):
    open(os.path.join(path, name), "w").write(name)
    full_env = {**os.environ, **(env or {})}
    subprocess.run(
        f"git add -A && git commit -q -m {msg}",
        shell=True,
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
        env=full_env,
    )


def _wt_by_branch(audit, branch):
    return next(w for w in audit["worktrees"] if w["branch"] == branch)


def test_audit_merged_worktree_is_safe_to_prune(repo):
    res = repo.wm.add("myrepo", "feat-merged")
    _commit_in(res["path"], "feature.txt", "feat")
    repo.wm.merge("myrepo", "feat-merged")  # --no-ff into main
    audit = repo.wm.audit("myrepo")
    w = _wt_by_branch(audit, "feat-merged")
    assert w["class"] == "merged"
    assert w["merged"] is True and w["dirty"] is False
    assert {"repo": "myrepo", "branch": "feat-merged", "class": "merged"} in (
        audit["safe_to_prune"]
    )


def test_audit_no_divergence_is_merged(repo):
    # a branch with no commits beyond main is an ancestor of main -> redundant.
    repo.wm.add("myrepo", "feat-empty")
    audit = repo.wm.audit("myrepo")
    assert _wt_by_branch(audit, "feat-empty")["class"] == "merged"


def test_audit_unmerged_ahead_is_active(repo):
    res = repo.wm.add("myrepo", "feat-active")
    _commit_in(res["path"], "wip.txt", "wip")  # ahead, not merged, just now
    audit = repo.wm.audit("myrepo")
    w = _wt_by_branch(audit, "feat-active")
    assert w["class"] == "active"
    assert w["ahead"] == 1 and w["merged"] is False
    assert {"repo": "myrepo", "branch": "feat-active"} in audit["do_not_disturb"]


def test_audit_dirty_worktree_is_active(repo):
    res = repo.wm.add("myrepo", "feat-dirty")
    open(os.path.join(res["path"], "uncommitted.txt"), "w").write("scratch")
    audit = repo.wm.audit("myrepo")
    w = _wt_by_branch(audit, "feat-dirty")
    assert w["dirty"] is True and w["class"] == "active"


def test_audit_detached_worktree_is_dangling(repo):
    res = repo.wm.add("myrepo", "feat-detach")
    _run("git checkout --detach", res["path"])
    audit = repo.wm.audit("myrepo")
    dangling = [w for w in audit["worktrees"] if w["class"] == "dangling"]
    assert any(w["path"] == res["path"] for w in dangling)


def test_audit_quiet_unmerged_branch_is_stale(repo):
    res = repo.wm.add("myrepo", "feat-stale")
    old = "2020-01-01T00:00:00"
    _commit_in(
        res["path"],
        "old.txt",
        "old",
        env={"GIT_AUTHOR_DATE": old, "GIT_COMMITTER_DATE": old},
    )
    audit = repo.wm.audit("myrepo", stale_days=14)
    w = _wt_by_branch(audit, "feat-stale")
    assert w["class"] == "stale"
    assert {"repo": "myrepo", "branch": "feat-stale"} in audit["review"]


def test_audit_reports_orphan_dir_without_pruning(repo):
    res = repo.wm.add("myrepo", "feat-merged")
    _commit_in(res["path"], "feature.txt", "feat")
    repo.wm.merge("myrepo", "feat-merged")
    # an untracked directory that looks like a worktree
    root = wt_mod.WORKTREE_ROOT
    ghost = os.path.join(root, "ghost")
    os.makedirs(ghost, exist_ok=True)
    open(os.path.join(ghost, ".git"), "w").write("gitdir: /nonexistent")

    audit = repo.wm.audit("myrepo", prune_merged=True)
    assert any(o["path"] == ghost for o in audit["orphans"])
    # prune removed the merged worktree but left the orphan dir intact
    assert os.path.isdir(ghost)
    assert not os.path.isdir(res["path"])
    assert any(p["branch"] == "feat-merged" for p in audit["pruned"])


def test_audit_prune_merged_keeps_active(repo):
    merged = repo.wm.add("myrepo", "feat-merged")
    _commit_in(merged["path"], "feature.txt", "feat")
    repo.wm.merge("myrepo", "feat-merged")
    active = repo.wm.add("myrepo", "feat-active")
    _commit_in(active["path"], "wip.txt", "wip")

    audit = repo.wm.audit("myrepo", prune_merged=True)
    assert not os.path.isdir(merged["path"])  # merged removed
    assert os.path.isdir(active["path"])  # active untouched
    assert any(k.get("branch") == "feat-active" for k in audit["kept"])
    # the deleted branch is gone, the active one survives
    branches = subprocess.run(
        "git branch --list",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout
    assert "feat-merged" not in branches
    assert "feat-active" in branches


def test_audit_canonical_repo_unpushed(repo, tmp_path):
    # give myrepo an origin, push main, then commit locally without pushing.
    bare = tmp_path / "origin.git"
    _run(f"git init --bare -b main {bare}", repo.path)
    _run(f"git remote add origin {bare}", repo.path)
    _run("git push -q origin main", repo.path)
    _commit_in(repo.path, "local.txt", "local")  # ahead of origin/main

    audit = repo.wm.audit("myrepo")
    rep = next(r for r in audit["repos"] if r["repo"] == "myrepo")
    assert rep["class"] == "unpushed"
    assert rep["ahead_origin"] == 1 and rep["no_upstream"] is False
    assert rep["base_unpushed"] is True


# ── release-flow hygiene wrapper (Git.worktree_hygiene) ───────────────────


def _real_git_repo(tmp_path, monkeypatch):
    """A real ``Git`` (not FakeGit) over one temp repo, with an isolated
    WORKTREE_ROOT — the surface the release pipeline calls worktree_hygiene on."""
    from repository_manager.repository_manager import Git

    ws = tmp_path / "workspace"
    repo_path = ws / "myrepo"
    repo_path.mkdir(parents=True)
    _run("git init -b main", repo_path)
    _run("git config user.email t@t.io && git config user.name t", repo_path)
    (repo_path / "README.md").write_text("hello\n")
    _run("git add -A && git commit -q -m init", repo_path)

    monkeypatch.setattr(wt_mod, "WORKTREE_ROOT", str(tmp_path / "worktrees"))
    git = Git(path=str(ws))
    git.project_map = {"git@x/myrepo.git": str(repo_path)}
    return git, WorktreeManager(git), str(repo_path)


def test_worktree_hygiene_reports_without_pruning(tmp_path, monkeypatch):
    git, wm, _ = _real_git_repo(tmp_path, monkeypatch)
    merged = wm.add("myrepo", "feat-merged")
    _commit_in(merged["path"], "f.txt", "feat")
    wm.merge("myrepo", "feat-merged")
    active = wm.add("myrepo", "feat-active")
    _commit_in(active["path"], "wip.txt", "wip")

    report = git.worktree_hygiene()  # default: read-only
    assert "pruned" not in report
    assert any(s["branch"] == "feat-merged" for s in report["safe_to_prune"])
    assert {"repo": "myrepo", "branch": "feat-active"} in report["do_not_disturb"]
    # nothing removed
    assert os.path.isdir(merged["path"]) and os.path.isdir(active["path"])


def test_worktree_hygiene_prune_removes_only_merged(tmp_path, monkeypatch):
    git, wm, _ = _real_git_repo(tmp_path, monkeypatch)
    merged = wm.add("myrepo", "feat-merged")
    _commit_in(merged["path"], "f.txt", "feat")
    wm.merge("myrepo", "feat-merged")
    active = wm.add("myrepo", "feat-active")
    _commit_in(active["path"], "wip.txt", "wip")

    result = git.worktree_hygiene(prune=True)
    assert not os.path.isdir(merged["path"])  # merged pruned
    assert os.path.isdir(active["path"])  # active untouched
    assert any(p["branch"] == "feat-merged" for p in result["pruned"])
