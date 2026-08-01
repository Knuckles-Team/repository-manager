"""Tests for the private-ref stash guard (CONCEPT:RM-STASH-GUARD, registry
``D-CP-1``) against a real git fixture repo.

The hazard: ``refs/stash`` is one global stack per ``.git``, and many worktrees
(plus the canonical checkout) share a single ``.git``. The old
``WorktreeManager.add(..., adopt=True)`` flow did::

    git stash push -u -m "..."   # canonical
    ... (fetch, park checkout, `git worktree add` -- can take real wall-clock time)
    git stash pop                 # canonical or the new worktree

``git stash pop`` always takes ``stash@{0}`` -- whichever entry is on *top* of
the stack at that instant, regardless of who pushed it. Anything else that
pushes onto the shared stack in that window (another repository-manager
operation, or a human/tool running a raw ``git stash`` directly, which is
exactly why this workspace's lanes were told never to do that) lands on top
and gets popped instead, silently crossing two lanes' WIP or burying one of
them. ``stash_guard`` closes this by moving the capture onto a private ref
(``refs/rm-stash/<label>-<uuid>``) the instant it is made, under the
canonical checkout's cross-process lease, so nothing after that point ever
depends on stack order again.
"""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from types import SimpleNamespace

import pytest

from repository_manager import stash_guard
from repository_manager.canonical_guard import hold_canonical_lease


class FakeGit:
    """Minimal Git stand-in: real subprocess git, GitResult-shaped return."""

    def git_action(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict | None = None,
        timeout: int = 1800,
        raw_output: bool = False,
    ) -> SimpleNamespace:
        del env, timeout, raw_output
        p = subprocess.run(
            command, shell=True, cwd=path, capture_output=True, text=True
        )
        out = (p.stdout + p.stderr).strip()
        return SimpleNamespace(
            status="success" if p.returncode == 0 else "error",
            data=out,
            error=None if p.returncode == 0 else SimpleNamespace(message=out),
        )


class HookedGit(FakeGit):
    """``FakeGit`` that fires a callback once, right after a matching command
    runs, letting a test inject "what a concurrent actor did in this instant"
    deterministically -- no thread races, no flakiness."""

    def __init__(self, hooks: dict[str, Callable[[], None]]):
        self._hooks = dict(hooks)

    def git_action(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict | None = None,
        timeout: int = 1800,
        raw_output: bool = False,
    ) -> SimpleNamespace:
        res = super().git_action(
            command,
            path=path,
            quiet=quiet,
            env=env,
            timeout=timeout,
            raw_output=raw_output,
        )
        for prefix, hook in list(self._hooks.items()):
            if command.startswith(prefix):
                del self._hooks[prefix]  # fire once
                hook()
        return res


def _run(cmd, cwd, check=True):
    return subprocess.run(
        cmd, shell=True, cwd=cwd, check=check, capture_output=True, text=True
    )


def _status(cwd):
    return _run("git status --porcelain", cwd).stdout.strip()


def _stash_list(cwd):
    return _run("git stash list", cwd).stdout.strip()


def _ref_exists(cwd, ref):
    return (
        _run(f"git rev-parse --verify --quiet {ref}", cwd, check=False).returncode == 0
    )


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "myrepo"
    path.mkdir()
    _run("git init -b main", path)
    _run("git config user.email t@t.io && git config user.name t", path)
    (path / "a.txt").write_text("hello\n")
    _run("git add -A && git commit -q -m init", path)
    return str(path)


# ── capture_wip / apply_and_clear correctness ──────────────────────────────


def test_clean_tree_has_nothing_to_capture(repo):
    result = stash_guard.capture_wip(FakeGit(), repo, label="lane-a")
    assert result == {"ok": True, "ref": None, "error": None}


def test_capture_moves_tracked_and_untracked_wip_off_the_shared_stack(repo):
    (open(os.path.join(repo, "a.txt"), "w")).write("lane-a edit\n")
    open(os.path.join(repo, "untracked.txt"), "w").write("lane-a untracked\n")

    result = stash_guard.capture_wip(FakeGit(), repo, label="lane-a")

    assert result["ok"] is True
    assert result["ref"] is not None
    assert result["ref"].startswith("refs/rm-stash/lane-a-")
    # canonical tree is clean again
    assert _status(repo) == ""
    # and NOT sitting on the shared stack -- the whole point
    assert _stash_list(repo) == ""
    assert _ref_exists(repo, result["ref"])


def test_apply_and_clear_restores_wip_and_deletes_the_private_ref(repo):
    open(os.path.join(repo, "a.txt"), "w").write("lane-a edit\n")
    open(os.path.join(repo, "untracked.txt"), "w").write("lane-a untracked\n")
    git = FakeGit()
    captured = stash_guard.capture_wip(git, repo, label="lane-a")

    applied = stash_guard.apply_and_clear(git, repo, captured["ref"])

    assert applied == {"ok": True, "ref": captured["ref"], "error": None}
    assert open(os.path.join(repo, "a.txt")).read() == "lane-a edit\n"
    assert open(os.path.join(repo, "untracked.txt")).read() == "lane-a untracked\n"
    assert not _ref_exists(repo, captured["ref"])  # cleaned up


def test_apply_and_clear_leaves_the_ref_in_place_on_conflict(repo):
    """A failed apply must never lose the WIP -- the ref survives for manual
    recovery instead of being dropped."""
    open(os.path.join(repo, "a.txt"), "w").write("lane-a edit\n")
    git = FakeGit()
    captured = stash_guard.capture_wip(git, repo, label="lane-a")

    # something else now conflicts with the stashed change
    open(os.path.join(repo, "a.txt"), "w").write("someone else entirely\n")

    applied = stash_guard.apply_and_clear(git, repo, captured["ref"])

    assert applied["ok"] is False
    assert applied["ref"] == captured["ref"]
    assert _ref_exists(repo, captured["ref"])  # NOT dropped


def test_capture_refuses_when_the_canonical_lease_is_already_held(repo):
    open(os.path.join(repo, "a.txt"), "w").write("lane-a edit\n")
    with hold_canonical_lease(repo, note="someone else"):
        result = stash_guard.capture_wip(FakeGit(), repo, label="lane-a")
    assert result["ok"] is False
    assert result["ref"] is None
    assert "lease" in result["error"]
    # nothing was touched: the WIP is still sitting in the working tree
    assert open(os.path.join(repo, "a.txt")).read() == "lane-a edit\n"
    assert _stash_list(repo) == ""


# ── the hazard itself: shared refs/stash crosses WIP under the old pattern ─


def test_old_shared_stash_pattern_crosses_concurrent_wip(repo):
    """Reproduces the exact shape of the removed code:
    ``git stash push -u`` ... (a window where anything else can happen) ...
    ``git stash pop``. A concurrent actor's own ``git stash push`` landing in
    that window means the blind LIFO ``pop`` takes *their* WIP, not ours --
    and ours is left sitting on the stack, silently buried. This is run with
    raw git only (no stash_guard involved) to prove the underlying mechanism
    is the actual hazard, independent of any particular caller's code."""
    open(os.path.join(repo, "a.txt"), "w").write("lane-RM own WIP\n")
    open(os.path.join(repo, "rm-only.txt"), "w").write("belongs to RM\n")

    # 1. RM's own push (what the old code did first).
    _run('git stash push -u -m "RM WIP"', repo)
    assert _status(repo) == ""  # tree clean, RM's WIP presumed safe on the stack

    # 2. A concurrent actor -- another worktree/process sharing this .git, or
    #    a human running `git stash` directly -- pushes its own WIP in the
    #    window before RM gets back to `pop`.
    open(os.path.join(repo, "a.txt"), "w").write("interloper's WIP\n")
    open(os.path.join(repo, "interloper-only.txt"), "w").write("not RM's\n")
    _run('git stash push -u -m "interloper WIP"', repo)

    # 3. The old code's `pop` -- blind LIFO, takes whatever is on top now.
    _run("git stash pop", repo)

    # The tree now holds the INTERLOPER's WIP, not RM's -- crossed.
    assert open(os.path.join(repo, "a.txt")).read() == "interloper's WIP\n"
    assert os.path.isfile(os.path.join(repo, "interloper-only.txt"))
    assert not os.path.isfile(os.path.join(repo, "rm-only.txt"))
    # And RM's own WIP is still on the stack -- silently buried, not lost
    # outright, but the caller who thinks it "restored its own WIP" is wrong.
    assert "RM WIP" in _stash_list(repo)


def test_private_ref_pattern_is_immune_to_a_stash_push_that_lands_afterward(repo):
    """Same interloper, same shared stack -- but once RM's WIP is on its
    private ref, whatever the interloper does to `refs/stash` afterward is
    irrelevant: `apply_and_clear` names its own ref, never `stash@{0}`."""
    open(os.path.join(repo, "a.txt"), "w").write("lane-RM own WIP\n")
    open(os.path.join(repo, "rm-only.txt"), "w").write("belongs to RM\n")
    git = FakeGit()

    captured = stash_guard.capture_wip(git, repo, label="lane-RM")
    assert captured["ok"] is True
    assert _stash_list(repo) == ""  # nothing left on the shared stack

    # The interloper now pushes its own WIP onto the (now-empty) shared stack.
    open(os.path.join(repo, "a.txt"), "w").write("interloper's WIP\n")
    open(os.path.join(repo, "interloper-only.txt"), "w").write("not RM's\n")
    _run('git stash push -u -m "interloper WIP"', repo)

    # RM applies from its OWN ref, not from the shared stack.
    applied = stash_guard.apply_and_clear(git, repo, captured["ref"])
    assert applied["ok"] is True

    assert open(os.path.join(repo, "a.txt")).read() == "lane-RM own WIP\n"
    assert os.path.isfile(os.path.join(repo, "rm-only.txt"))
    # the interloper's own entry is untouched -- neither crossed nor dropped
    assert "interloper WIP" in _stash_list(repo)


def test_two_repository_manager_captures_on_the_same_canonical_never_cross(repo):
    """The scenario the ticket actually asks for: two *repository-manager*
    operations racing on worktrees that share one ``.git``. Lane B's own
    ``capture_wip`` call is injected to fire in the instant right after lane
    A's ``git stash push`` runs -- the tightest window the new code exposes.
    The canonical lease must serialize them: B is refused while A holds it,
    so A's push/rev-parse/store/drop always complete as one unit and B never
    sees a partially-captured state to collide with."""
    open(os.path.join(repo, "a.txt"), "w").write("lane-A WIP\n")
    open(os.path.join(repo, "a-only.txt"), "w").write("A\n")

    b_attempts: list[dict] = []

    def _lane_b_races_in():
        # Lane B tries to capture its own WIP on the SAME canonical while A
        # still holds the lease (A's `with hold_canonical_lease` body has not
        # exited yet -- we are firing from inside A's own push command).
        b_attempts.append(stash_guard.capture_wip(FakeGit(), repo, label="lane-B"))

    git = HookedGit({"git stash push": _lane_b_races_in})
    result_a = stash_guard.capture_wip(git, repo, label="lane-A")

    assert result_a["ok"] is True
    # B was refused -- serialized out, not allowed to interleave.
    assert len(b_attempts) == 1
    assert b_attempts[0]["ok"] is False
    assert "lease" in b_attempts[0]["error"] or "busy" in b_attempts[0]["error"]

    # A's own WIP landed on A's own ref, uncontaminated.
    applied_a = stash_guard.apply_and_clear(FakeGit(), repo, result_a["ref"])
    assert applied_a["ok"] is True
    assert open(os.path.join(repo, "a.txt")).read() == "lane-A WIP\n"
    assert os.path.isfile(os.path.join(repo, "a-only.txt"))

    # B can now retry once A has released the lease, and gets its own,
    # independent capture of whatever B's tree holds at that point (here:
    # nothing, since B never got to dirty its own tree in this scenario).
    retry_b = stash_guard.capture_wip(FakeGit(), repo, label="lane-B")
    assert retry_b["ok"] is True
