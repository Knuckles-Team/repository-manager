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

    def git_action(
        self,
        command,
        path=None,
        quiet=False,
        env=None,
        timeout=1800,
        raw_output=False,
    ):
        del env, timeout, raw_output
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


def test_add_absolute_repo_path_stays_beneath_worktree_root(repo):
    res = repo.wm.add(repo.path, "feat-absolute")

    assert res["ok"] and res["created"]
    expected = os.path.join(
        wt_mod.WORKTREE_ROOT,
        os.path.basename(repo.path),
        "feat-absolute",
    )
    assert res["path"] == expected
    assert os.path.commonpath([res["path"], wt_mod.WORKTREE_ROOT]) == os.path.abspath(
        wt_mod.WORKTREE_ROOT
    )
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=res["path"],
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "feat-absolute"


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


def test_list_linked_status_is_independent_of_configured_worktree_root(
    repo, monkeypatch
):
    """D-CDX-1: the local CLI and the deployed MCP server run with different
    ``WORKTREE_ROOT`` values by design. The SAME real linked worktree must
    report ``linked=True`` regardless of which root this process happens to
    be configured with — the split-brain was git-truth being overridden by a
    path-prefix guess tied to one process's own config."""
    made = repo.wm.add("myrepo", "feat-x")
    assert os.path.commonpath([made["path"], wt_mod.WORKTREE_ROOT]) == os.path.abspath(
        wt_mod.WORKTREE_ROOT
    )

    # point WORKTREE_ROOT somewhere totally unrelated to where the worktree
    # actually lives (simulating the other surface's differently configured
    # root) — `linked` must still be computed correctly from git itself.
    monkeypatch.setattr(wt_mod, "WORKTREE_ROOT", "/nonexistent/other-root")
    listing = repo.wm.list_worktrees("myrepo")
    linked = [w for w in listing["worktrees"] if w["branch"] == "feat-x"][0]
    assert linked["linked"] is True

    # and the canonical checkout itself must never be reported as linked,
    # regardless of WORKTREE_ROOT either.
    canonical_entry = [
        w for w in listing["worktrees"] if os.path.abspath(w["path"]) == repo.path
    ][0]
    assert canonical_entry["linked"] is False


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


# ── canonical-checkout guard (CONCEPT:RM-CANON-GUARD) ─────────────────────
# ``add`` parks the canonical checkout off the requested branch (if it holds
# it), and ``merge`` switches the canonical checkout onto ``into`` -- both are
# tree-mutating checkouts against the *canonical* tree. Prove they refuse and
# report instead of clobbering when that tree is dirty, and still work when
# it is clean, against a real git repo (not a mocked guard function).


def test_add_refuses_dirty_tracked_change_on_canonical(repo):
    _run("git checkout -b feat-y", repo.path)
    open(os.path.join(repo.path, "README.md"), "w").write("dirty change\n")
    res = repo.wm.add("myrepo", "feat-y")
    assert res["ok"] is False
    assert res["skipped"] is True
    assert res["reason"] == "dirty-canonical-checkout"
    # canonical untouched: still on feat-y, edit intact, no worktree created
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "feat-y"
    assert open(os.path.join(repo.path, "README.md")).read() == "dirty change\n"
    assert not os.path.isdir(repo.wm.worktree_path("myrepo", "feat-y"))


def test_add_refuses_dirty_untracked_file_on_canonical(repo):
    _run("git checkout -b feat-z", repo.path)
    open(os.path.join(repo.path, "scratch.txt"), "w").write("wip\n")
    res = repo.wm.add("myrepo", "feat-z")
    assert res["ok"] is False and res["reason"] == "dirty-canonical-checkout"
    assert os.path.isfile(os.path.join(repo.path, "scratch.txt"))  # not clobbered
    assert not os.path.isdir(repo.wm.worktree_path("myrepo", "feat-z"))


def test_add_parks_canonical_when_clean(repo):
    _run("git checkout -b feat-w", repo.path)
    res = repo.wm.add("myrepo", "feat-w")
    assert res["ok"] and res["created"]
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "main"  # parked off feat-w so the worktree could take it


def test_merge_refuses_dirty_tracked_change_on_canonical(repo):
    res = repo.wm.add("myrepo", "feat-x")
    open(os.path.join(res["path"], "feature.txt"), "w").write("x")
    _run("git add -A && git commit -q -m feat", res["path"])
    # canonical is on some other branch with uncommitted WIP (the exact
    # "someone working directly in canonical" hazard).
    _run("git checkout -b someone-else-branch", repo.path)
    open(os.path.join(repo.path, "README.md"), "w").write("someone's WIP\n")

    result = repo.wm.merge("myrepo", "feat-x")
    assert result["ok"] is False
    assert result["reason"] == "dirty-canonical-checkout"
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "someone-else-branch"  # never switched
    assert open(os.path.join(repo.path, "README.md")).read() == "someone's WIP\n"


def test_merge_refuses_dirty_untracked_file_on_canonical(repo):
    res = repo.wm.add("myrepo", "feat-x2")
    open(os.path.join(res["path"], "feature.txt"), "w").write("x")
    _run("git add -A && git commit -q -m feat", res["path"])
    _run("git checkout -b someone-else-branch2", repo.path)
    open(os.path.join(repo.path, "scratch.txt"), "w").write("wip\n")

    result = repo.wm.merge("myrepo", "feat-x2")
    assert result["ok"] is False and result["reason"] == "dirty-canonical-checkout"
    assert os.path.isfile(os.path.join(repo.path, "scratch.txt"))


def test_merge_switches_canonical_when_clean(repo):
    res = repo.wm.add("myrepo", "feat-x3")
    open(os.path.join(res["path"], "feature.txt"), "w").write("x")
    _run("git add -A && git commit -q -m feat", res["path"])
    _run("git checkout -b someone-else-branch3", repo.path)  # clean

    result = repo.wm.merge("myrepo", "feat-x3")
    assert result["ok"] and not result["conflict"]
    assert os.path.isfile(os.path.join(repo.path, "feature.txt"))
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "main"


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


def _branches(repo_path):
    return subprocess.run(
        "git branch --list",
        shell=True,
        cwd=repo_path,
        capture_output=True,
        text=True,
    ).stdout


def _ref(repo_path, ref):
    """What ``ref`` points at right now, or "" when it does not exist."""
    return subprocess.run(
        f"git rev-parse --verify --quiet {ref}",
        shell=True,
        cwd=repo_path,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _kept_reason(result, branch):
    return next(k["reason"] for k in result["kept"] if k.get("branch") == branch)


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


def test_audit_fresh_worktree_at_base_is_not_prunable(repo):
    """A worktree that has not committed yet sits exactly on base, so ``ahead``
    is 0 — the same reading a merged-back branch gives. It is the *start* of a
    lane, not the end of one, and must never be offered for pruning."""
    repo.wm.add("myrepo", "feat-empty")
    audit = repo.wm.audit("myrepo")
    w = _wt_by_branch(audit, "feat-empty")
    assert w["ahead"] == 0 and w["behind"] == 0
    assert w["at_base"] is True
    assert w["merged"] is False
    assert w["class"] == "active"
    assert not any(s["branch"] == "feat-empty" for s in audit["safe_to_prune"])


def test_prune_merged_leaves_a_fresh_worktree_and_its_branch_alone(repo):
    fresh = repo.wm.add("myrepo", "feat-empty")
    repo.wm.audit("myrepo", prune_merged=True)
    assert os.path.isdir(fresh["path"])
    assert "feat-empty" in _branches(repo.path)


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
    assert wm.merge("myrepo", "feat-merged")["ok"]
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
    assert wm.merge("myrepo", "feat-merged")["ok"]
    active = wm.add("myrepo", "feat-active")
    _commit_in(active["path"], "wip.txt", "wip")

    result = git.worktree_hygiene(prune=True)
    assert not os.path.isdir(merged["path"])  # merged pruned
    assert os.path.isdir(active["path"])  # active untouched
    assert any(p["branch"] == "feat-merged" for p in result["pruned"])


# ── prune safety (CONCEPT:RM-PRUNE-GUARD, registry D-FE-9) ────────────────────
# D-FE-9: an `agent-utilities`-scoped sweep removed a live lane's worktree AND
# ran `git branch -D` on its ref, leaving its commits reachable only as dangling
# objects. The branch was genuinely merged at that instant — the lane had merged
# an intermediate chunk back to main and kept working — so "merged" was never
# sufficient authorisation to destroy anything.


def _merged_worktree(wm, branch="feat-merged"):
    """A worktree whose branch really is merged back into main."""
    made = wm.add("myrepo", branch)
    _commit_in(made["path"], f"{branch}.txt", "feat")
    assert wm.merge("myrepo", branch)["ok"]
    return made


def test_prune_merged_still_removes_a_genuinely_merged_worktree(repo):
    """The feature itself must keep working: a real merge-back is reclaimed,
    branch ref included."""
    made = _merged_worktree(repo.wm)
    tip = _ref(repo.path, "refs/heads/feat-merged")
    result = repo.wm.audit("myrepo", prune_merged=True)
    entry = next(p for p in result["pruned"] if p["branch"] == "feat-merged")
    assert entry["ok"] is True
    assert entry["branch_deleted"] is True
    assert not os.path.isdir(made["path"])
    assert "feat-merged" not in _branches(repo.path)
    # the deleted tip is anchored, so even a wrong answer above stays recoverable
    assert entry["branch_anchor"] == "refs/lane-backup/feat-merged"
    assert _ref(repo.path, entry["branch_anchor"]) == tip


def test_prune_skips_a_worktree_whose_lane_holds_a_lease(repo):
    """The D-FE-9 shape: the branch is merged and the tree is clean because the
    lane is blocked inside a long `pre-commit` run — which the lane protocol
    announces as a lease at the scope every lane of this repo shares."""
    lanes = pytest.importorskip("agent_utilities.governance.lanes")
    made = _merged_worktree(repo.wm)

    with lanes.hold_lease(
        "precommit-all-files", operation="pre-commit run", path=made["path"]
    ):
        result = repo.wm.audit("myrepo", prune_merged=True)

    assert result["pruned"] == []
    assert "precommit-all-files" in _kept_reason(result, "feat-merged")
    assert os.path.isdir(made["path"])
    assert "feat-merged" in _branches(repo.path)


def test_prune_skips_a_worktree_the_lane_dirtied_after_the_scan(repo):
    """Classification happens in `audit()`, deletion later in `_prune_merged`.
    Work that lands inside that window must not be destroyed by a decision taken
    before it existed."""
    made = _merged_worktree(repo.wm)
    scan = repo.wm.audit("myrepo")  # read-only: classification only
    assert _wt_by_branch(scan, "feat-merged")["class"] == "merged"

    open(os.path.join(made["path"], "late.txt"), "w").write("landed mid-sweep")

    pruned, kept = repo.wm._prune_merged(scan["worktrees"], "main")
    assert pruned == []
    # the lane protocol answers first: a tree with uncommitted work is never
    # resettable by an actor that does not own it.
    assert "uncommitted work" in _kept_reason({"kept": kept}, "feat-merged")
    assert os.path.isdir(made["path"])
    assert os.path.isfile(os.path.join(made["path"], "late.txt"))


def test_prune_never_deletes_a_branch_that_committed_after_the_scan(repo):
    """The lethal race: two commits land between the `merged` reading and the
    deletion. Neither the directory nor — above all — the ref may be taken."""
    made = _merged_worktree(repo.wm)
    scan = repo.wm.audit("myrepo")
    assert _wt_by_branch(scan, "feat-merged")["class"] == "merged"

    _commit_in(made["path"], "after-1.txt", "after1")
    _commit_in(made["path"], "after-2.txt", "after2")
    head = subprocess.run(
        "git rev-parse HEAD",
        shell=True,
        cwd=made["path"],
        capture_output=True,
        text=True,
    ).stdout.strip()

    pruned, kept = repo.wm._prune_merged(scan["worktrees"], "main")
    assert pruned == []
    # clean tree, no lease held: only the re-derived state catches this.
    reason = _kept_reason({"kept": kept}, "feat-merged")
    assert "state changed since the audit scan" in reason and "ahead=2" in reason
    assert os.path.isdir(made["path"])
    assert "feat-merged" in _branches(repo.path)
    # the ref still points at the two new commits: nothing became garbage
    tip = subprocess.run(
        "git rev-parse feat-merged",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert tip == head


def test_prune_skips_a_locked_worktree(repo):
    made = _merged_worktree(repo.wm)
    _run(f"git worktree lock {made['path']}", repo.path)
    result = repo.wm.audit("myrepo", prune_merged=True)
    assert result["pruned"] == []
    assert "locked" in _kept_reason(result, "feat-merged")
    assert os.path.isdir(made["path"])
    assert "feat-merged" in _branches(repo.path)


def test_delete_merged_branch_refuses_an_unmerged_branch(repo):
    """The ref gate on its own: `git branch -D` would take this; the guard and
    git's own `-d` both refuse it."""
    made = repo.wm.add("myrepo", "feat-unmerged")
    _commit_in(made["path"], "wip.txt", "wip")
    deleted, reason, anchor = repo.wm._delete_merged_branch(
        repo.path, "feat-unmerged", "main"
    )
    assert deleted is False
    assert "not reachable from" in reason
    assert anchor == ""
    assert "feat-unmerged" in _branches(repo.path)
    # a refused delete leaves the ref namespace exactly as it found it
    assert _ref(repo.path, "refs/lane-backup/feat-unmerged") == ""


def test_remove_with_delete_branch_refuses_an_unmerged_branch(repo):
    """An explicit `remove --delete-branch` is still not authorisation to orphan
    commits; `force` covers the recoverable directory, never the ref."""
    made = repo.wm.add("myrepo", "feat-unmerged")
    _commit_in(made["path"], "wip.txt", "wip")
    result = repo.wm.remove("myrepo", "feat-unmerged", delete_branch=True)
    assert result["ok"] is True
    assert result["branch_deleted"] is False
    assert "not reachable from" in result["branch_kept_reason"]
    assert "feat-unmerged" in _branches(repo.path)
    assert not os.path.isdir(made["path"])  # directory gone, work recoverable


def test_remove_with_delete_branch_deletes_a_merged_branch(repo):
    _merged_worktree(repo.wm, "feat-done")
    result = repo.wm.remove("myrepo", "feat-done", delete_branch=True)
    assert result["branch_deleted"] is True
    assert "feat-done" not in _branches(repo.path)


def test_delete_anchors_the_tip_before_removing_the_ref(repo):
    """The anchor is taken at the moment of deletion, not at lane start, so it
    cannot preserve a commit the branch has since moved past."""
    made = _merged_worktree(repo.wm, "feat-anchor")
    # the branch advances after it was first observed, exactly as a live lane's
    # would; the anchor must follow the tip, not the earlier reading.
    early = _ref(repo.path, "refs/heads/feat-anchor")
    _commit_in(made["path"], "later.txt", "later")
    assert repo.wm.merge("myrepo", "feat-anchor")["ok"]
    late = _ref(repo.path, "refs/heads/feat-anchor")
    assert late != early

    result = repo.wm.remove("myrepo", "feat-anchor", delete_branch=True)
    assert result["branch_deleted"] is True
    assert result["branch_anchor"] == "refs/lane-backup/feat-anchor"
    assert _ref(repo.path, result["branch_anchor"]) == late
    # and the anchor is a real root: the tip is reachable, not dangling
    dangling = subprocess.run(
        "git fsck --dangling",
        shell=True,
        cwd=repo.path,
        capture_output=True,
        text=True,
    ).stdout
    assert late not in dangling


def test_delete_restores_a_pre_existing_anchor_when_git_refuses(repo):
    """A merged branch whose worktree is still checked out passes the ancestry
    gate and is then refused by git itself — the rollback path. Someone else's
    anchor must survive our failed attempt untouched."""
    _merged_worktree(repo.wm, "feat-held")  # worktree deliberately left in place
    _run("git update-ref refs/lane-backup/feat-held main", repo.path)
    prior = _ref(repo.path, "refs/lane-backup/feat-held")
    assert prior == _ref(repo.path, "refs/heads/main")

    deleted, reason, anchor = repo.wm._delete_merged_branch(
        repo.path, "feat-held", "main"
    )
    assert deleted is False and anchor == ""
    assert "used by worktree" in reason
    assert _ref(repo.path, "refs/lane-backup/feat-held") == prior
    assert "feat-held" in _branches(repo.path)


def test_delete_removes_only_the_anchor_it_created_when_git_refuses(repo):
    _merged_worktree(repo.wm, "feat-held2")  # worktree still checked out
    deleted, _reason, _anchor = repo.wm._delete_merged_branch(
        repo.path, "feat-held2", "main"
    )
    assert deleted is False
    # no anchor existed before, so none is left behind
    assert _ref(repo.path, "refs/lane-backup/feat-held2") == ""


# ---------------------------------------------------------------------------
# D-CDX-29 — sync() must rebase/merge onto the AUTHORITATIVE local base, never
# a possibly-stale `origin/<base>` remote-tracking ref. Proven both ways: the
# OLD behavior (rebase onto origin/<base>) silently drops a commit that only
# exists on local main; the NEW behavior (rebase onto the local branch ref,
# shared via the linked worktree's git dir) keeps it.
# ---------------------------------------------------------------------------


def _add_origin_with_stale_main(repo_path, tmp_root):
    """Give ``repo_path`` a real ``origin`` remote, then advance LOCAL main
    only — reproducing this workspace's actual state (local main ahead of a
    never-pushed origin/main) inside a disposable fixture repo."""
    bare = os.path.join(tmp_root, "origin.git")
    _run(f"git clone --bare {repo_path} {bare}", tmp_root)
    _run(f"git remote add origin {bare}", repo_path)
    _run("git fetch origin", repo_path)
    # advance LOCAL main only - never pushed, exactly like a merge-queue
    # landing in this workspace's all-local fast-forward model.
    _commit_in(repo_path, "landed.txt", "landed-only-on-local-main")
    landed_sha = _ref(repo_path, "refs/heads/main")
    origin_sha = _ref(repo_path, "refs/remotes/origin/main")
    assert landed_sha != origin_sha, "fixture must reproduce local-ahead-of-origin"
    return landed_sha


def test_sync_rebases_onto_local_main_not_stale_origin_main(repo, tmp_path):
    """NEW behavior: local main is authoritative. A lane branch forked BEFORE
    the landing must pick it up on sync, even though origin/main never saw it."""
    made = repo.wm.add("myrepo", "feat-sync")
    landed_sha = _add_origin_with_stale_main(repo.path, str(tmp_path))

    result = repo.wm.sync("myrepo", "feat-sync")
    assert result["ok"], result
    assert result["base_ref"] == "refs/heads/main"

    log = subprocess.run(
        "git log --format=%H", shell=True, cwd=made["path"],
        capture_output=True, text=True,
    ).stdout
    assert landed_sha in log.splitlines(), (
        "sync() must carry forward a commit that only exists on local main"
    )


def test_old_stale_origin_rebase_would_have_dropped_the_landed_commit(repo, tmp_path):
    """Same fixture, but re-run literally what the OLD code did (rebase onto
    origin/<base> after a best-effort fetch) to prove the failure this item
    was opened against was real, not hypothetical."""
    made = repo.wm.add("myrepo", "feat-sync-old")
    landed_sha = _add_origin_with_stale_main(repo.path, str(tmp_path))

    # the exact OLD implementation: fetch, then rebase onto origin/<base>.
    _run("git fetch origin main", made["path"])
    old_rebase = subprocess.run(
        "git rebase origin/main", shell=True, cwd=made["path"],
        capture_output=True, text=True,
    )
    assert old_rebase.returncode == 0, old_rebase.stdout + old_rebase.stderr

    log = subprocess.run(
        "git log --format=%H", shell=True, cwd=made["path"],
        capture_output=True, text=True,
    ).stdout
    assert landed_sha not in log.splitlines(), (
        "this reproduces the defect: the OLD rebase-onto-origin/main target "
        "silently drops a commit that only ever landed on local main"
    )


def test_sync_refuses_when_base_ref_is_not_resolvable_from_the_worktree(repo):
    """A worktree that is not genuinely linked to the canonical repo (refs not
    shared) must be refused rather than silently falling back to origin."""
    made = repo.wm.add("myrepo", "feat-detached")
    # simulate a broken/unlinked worktree: delete the local base ref entirely
    # is not realistic for a linked worktree (refs are shared), so instead
    # assert the refusal path by asking for a base that never existed.
    result = repo.wm.sync("myrepo", "feat-detached", base="no-such-base")
    assert result["ok"] is False
    assert "not resolvable" in result["error"]


# ---------------------------------------------------------------------------
# D-CDX-15 — no destructive worktree/branch mutation may bypass lane
# occupancy detection, even with force=True. Two primitives:
# `remove()` (hardened) and `reset_branch()` (new, the sanctioned way to move
# a lane branch's tip).
# ---------------------------------------------------------------------------


def test_remove_refuses_an_occupied_worktree_even_with_force(repo):
    """The exact D-CDX-15 shape: a lane just committed (so the worktree is
    clean) and holds a live lease - `force=True` must not bypass it."""
    lanes = pytest.importorskip("agent_utilities.governance.lanes")
    made = repo.wm.add("myrepo", "rm-buildbroker-0802")
    _commit_in(made["path"], "wip.txt", "fresh-wip-commit")

    with lanes.hold_lease(
        "build-broker-job", operation="running build", path=made["path"]
    ):
        result = repo.wm.remove("myrepo", "rm-buildbroker-0802", force=True)

    assert result["ok"] is False
    assert result["reason"] == "lane-occupied"
    assert os.path.isdir(made["path"]), "occupied worktree must survive"
    assert "rm-buildbroker-0802" in _branches(repo.path)


def test_remove_still_works_on_a_genuinely_abandoned_worktree(repo):
    """The guard must not become a universal refusal - an unoccupied worktree
    with no lease and no uncommitted work removes exactly as before."""
    made = repo.wm.add("myrepo", "feat-abandoned")
    result = repo.wm.remove("myrepo", "feat-abandoned", force=True)
    assert result["ok"] is True
    assert not os.path.isdir(made["path"])


def test_reset_branch_refuses_an_occupied_worktree_with_unmerged_commits(repo):
    """The literal acceptance criterion: an occupied worktree with unmerged
    commits is refused rather than reset."""
    lanes = pytest.importorskip("agent_utilities.governance.lanes")
    made = repo.wm.add("myrepo", "rm-buildbroker-0802")
    _commit_in(made["path"], "wip.txt", "five-wip-commits")
    tip_before = _ref(repo.path, "refs/heads/rm-buildbroker-0802")

    with lanes.hold_lease(
        "build-broker-job", operation="running build", path=made["path"]
    ):
        result = repo.wm.reset_branch("myrepo", "rm-buildbroker-0802", target="main")

    assert result["ok"] is False
    assert result["reason"] == "lane-occupied"
    assert _ref(repo.path, "refs/heads/rm-buildbroker-0802") == tip_before


def test_reset_branch_refuses_unmerged_commits_even_when_unoccupied(repo):
    """Occupancy is only half the picture: even with no live lease, resetting
    a branch that carries commits not reachable from the target would make
    them unreachable from any ref this branch still names - refused."""
    made = repo.wm.add("myrepo", "feat-unmerged-reset")
    _commit_in(made["path"], "wip.txt", "unlanded-work")
    tip_before = _ref(repo.path, "refs/heads/feat-unmerged-reset")

    result = repo.wm.reset_branch("myrepo", "feat-unmerged-reset", target="main")

    assert result["ok"] is False
    assert result["reason"] == "unmerged-commits"
    assert _ref(repo.path, "refs/heads/feat-unmerged-reset") == tip_before


def test_reset_branch_succeeds_when_branch_is_already_an_ancestor_of_target(repo):
    """The safe case: a branch with nothing unlanded (already merged or never
    diverged) can be reset without friction."""
    _merged_worktree(repo.wm, "feat-safe-reset")
    main_tip = _ref(repo.path, "refs/heads/main")

    result = repo.wm.reset_branch("myrepo", "feat-safe-reset", target="main")

    assert result["ok"] is True
    assert result["new_sha"] == main_tip
    assert _ref(repo.path, "refs/heads/feat-safe-reset") == main_tip
