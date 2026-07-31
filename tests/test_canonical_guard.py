"""Tests for the canonical-checkout guard (CONCEPT:RM-CANON-GUARD) against a
real git fixture repo -- not a mock of the guard function, the actual git
state it inspects."""

import os
import subprocess
from types import SimpleNamespace

import pytest

from repository_manager.canonical_guard import (
    BlockedByLease,
    guarded_canonical_mutation,
    hold_canonical_lease,
)


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
        )


def _run(cmd, cwd):
    subprocess.run(cmd, shell=True, cwd=cwd, check=True, capture_output=True, text=True)


@pytest.fixture
def repo(tmp_path):
    path = tmp_path / "myrepo"
    path.mkdir()
    _run("git init -b main", path)
    _run("git config user.email t@t.io && git config user.name t", path)
    (path / "README.md").write_text("hello\n")
    _run("git add -A && git commit -q -m init", path)
    return str(path)


def test_clean_tree_proceeds(repo):
    with guarded_canonical_mutation(FakeGit(), repo, "myrepo", "test-action") as blocked:
        assert blocked is None


def test_tracked_modification_refuses_and_reports(repo, caplog):
    (open(os.path.join(repo, "README.md"), "w")).write("changed\n")
    with caplog.at_level("WARNING"):
        with guarded_canonical_mutation(
            FakeGit(), repo, "myrepo", "test-action"
        ) as blocked:
            assert blocked is not None
            assert blocked["ok"] is False
            assert blocked["skipped"] is True
            assert blocked["reason"] == "dirty-canonical-checkout"
            assert "myrepo" in blocked["error"]
            assert "README.md" in blocked["detail"]
    # loud + actionable: names the repo and the action refused, in the log
    assert any(
        "REFUSING" in r.message and "myrepo" in r.message and "test-action" in r.message
        for r in caplog.records
    )


def test_untracked_file_refuses_and_reports(repo, caplog):
    open(os.path.join(repo, "scratch.txt"), "w").write("wip\n")
    with caplog.at_level("WARNING"):
        with guarded_canonical_mutation(
            FakeGit(), repo, "myrepo", "test-action"
        ) as blocked:
            assert blocked is not None
            assert blocked["reason"] == "dirty-canonical-checkout"
            assert "scratch.txt" in blocked["detail"]
    assert any("REFUSING" in r.message and "myrepo" in r.message for r in caplog.records)


def test_untracked_and_tracked_together_still_refuses(repo):
    open(os.path.join(repo, "README.md"), "w").write("changed\n")
    open(os.path.join(repo, "scratch.txt"), "w").write("wip\n")
    with guarded_canonical_mutation(FakeGit(), repo, "myrepo", "test-action") as blocked:
        assert blocked is not None
        assert "README.md" in blocked["detail"]
        assert "scratch.txt" in blocked["detail"]


def test_nothing_mutated_when_refused(repo):
    """The whole point: no tree-mutating command may run once refused."""
    (open(os.path.join(repo, "README.md"), "w")).write("changed\n")
    _run("git branch other", repo)
    git = FakeGit()
    with guarded_canonical_mutation(git, repo, "myrepo", "checkout other") as blocked:
        if blocked is None:
            git.git_action("git checkout other", path=repo)
    branch = subprocess.run(
        "git rev-parse --abbrev-ref HEAD",
        shell=True,
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert branch == "main"  # never switched
    # the uncommitted edit is intact, not clobbered
    assert open(os.path.join(repo, "README.md")).read() == "changed\n"


def test_lease_blocks_concurrent_holder(repo, caplog):
    with hold_canonical_lease(repo, note="outer holder"):
        with pytest.raises(BlockedByLease):
            with hold_canonical_lease(repo, note="inner holder"):
                pass  # pragma: no cover - must not be reached


def test_guard_skips_when_lease_already_held(repo, caplog):
    with hold_canonical_lease(repo, note="someone else"):
        with caplog.at_level("WARNING"):
            with guarded_canonical_mutation(
                FakeGit(), repo, "myrepo", "test-action"
            ) as blocked:
                assert blocked is not None
                assert blocked["reason"] == "canonical-checkout-busy"
    assert any("lease" in r.message for r in caplog.records)


def test_lease_released_after_use(repo):
    """The lease must not linger once the `with` exits -- a second, later
    operation on the same canonical must be able to acquire it."""
    with hold_canonical_lease(repo, note="first"):
        pass
    with hold_canonical_lease(repo, note="second"):
        pass  # no BlockedByLease raised


def test_lease_file_lives_inside_dotgit_and_is_never_dirty(repo):
    with hold_canonical_lease(repo, note="x"):
        assert os.path.exists(os.path.join(repo, ".git", "repository-manager.lease"))
    status = subprocess.run(
        "git status --porcelain",
        shell=True,
        cwd=repo,
        capture_output=True,
        text=True,
    ).stdout.strip()
    assert status == ""  # never reported as untracked
