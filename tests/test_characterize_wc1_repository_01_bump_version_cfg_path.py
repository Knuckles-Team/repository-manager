"""Characterization tests for ``Git.bump_version``'s bump2version-configured
path (WC1-REPOSITORY-01).

The only pre-existing ``bump_version`` tests
(``tests/test_repository_manager.py::test_bump_version_fallback_*``) cover
exclusively the NO-config fallback branch. The real bump2version-configured
path -- pre-flight tag check, orphan-tag force re-bump, and the post-success
stage/commit/tag sequence the fleet has a documented failure history on
(``bump2version`` stages everything then fails to commit) -- had ZERO test
coverage before this lane. These tests pin that path's exact `git_action`
call sequence and returned `GitResult`, run once against the unmodified
function (record green) and once after the extract-method refactor
(require identical), per the brief's characterization discipline. Given how
sensitive this function is (release-path version-bump sequencing), the
refactor makes NO change to call order -- verified here by asserting the
`git_action` call sequence itself, not just the final result.
"""

import datetime
from unittest.mock import patch

import pytest

from repository_manager.models import GitMetadata, GitResult
from repository_manager.repository_manager import Git


def _meta(command="test"):
    return GitMetadata(
        command=command,
        workspace="/tmp",
        return_code=0,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    )


@pytest.fixture
def configured_project(tmp_path):
    """A project directory with a bump2version config (so bump_version takes
    the real bump2version path, not the no-config fallback)."""
    git = Git(path=str(tmp_path))
    proj_dir = tmp_path / "proj"
    proj_dir.mkdir()
    (proj_dir / ".bumpversion.cfg").write_text(
        "[bumpversion]\ncurrent_version = 1.0.0\n"
    )
    return git, proj_dir


def test_bump_version_dry_run_skips_preflight_and_finalize(configured_project):
    git, proj_dir = configured_project
    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append(command)
        return GitResult(
            status="success", data="new_version=1.0.1", metadata=_meta(command)
        )

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir), dry_run=True)

    assert res.status == "success"
    # dry_run => no pre-flight tag check call, no "--list" appended, no
    # post-success finalize (uv lock / add -u / status / commit / tag).
    assert calls == [
        "SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build bump2version patch --dry-run"
    ]


def test_bump_version_no_existing_tag_runs_full_sequence_in_order(configured_project):
    git, proj_dir = configured_project
    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append(command)
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="", metadata=_meta(command))
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=1.0.1", metadata=_meta(command)
            )
        if command == "git status --porcelain":
            return GitResult(
                status="success", data="M pyproject.toml\n", metadata=_meta(command)
            )
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir))

    assert res.status == "success"
    assert res.data == "new_version=1.0.1"
    # Exact call order pins the release-path sequence: pre-flight dry-run
    # list -> tag existence check -> real bump2version (with --list) -> (no
    # uv.lock present, so no `uv lock` call) -> add -u -> status --porcelain
    # -> commit --amend -> tag -f, in that order, never reordered.
    assert calls == [
        "bump2version patch --dry-run --list",
        "git tag -l v1.0.1",
        "SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build bump2version patch --list",
        "git add -u",
        "git status --porcelain",
        "SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build git commit --amend --no-edit",
        "git tag -f v1.0.1",
    ]


def test_bump_version_syncs_uv_lock_when_present(configured_project):
    git, proj_dir = configured_project
    (proj_dir / "uv.lock").write_text("# lock\n")
    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append(command)
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="", metadata=_meta(command))
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=2.0.0", metadata=_meta(command)
            )
        if command == "git status --porcelain":
            return GitResult(
                status="success", data="M uv.lock\n", metadata=_meta(command)
            )
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        git.bump_version(part="major", path=str(proj_dir))

    assert "uv lock" in calls
    # `uv lock` must run BEFORE `git add -u` (so the lock update itself gets staged).
    assert calls.index("uv lock") < calls.index("git add -u")


def test_bump_version_no_dirty_status_skips_commit_and_tag(configured_project):
    git, proj_dir = configured_project
    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append(command)
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="", metadata=_meta(command))
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=1.0.1", metadata=_meta(command)
            )
        if command == "git status --porcelain":
            return GitResult(
                status="success", data="", metadata=_meta(command)
            )  # clean
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir))

    assert res.status == "success"
    assert not any("commit --amend" in c for c in calls)
    assert not any(c.startswith("git tag -f") for c in calls)


def test_bump_version_existing_tag_no_force_is_skipped(configured_project):
    git, proj_dir = configured_project

    def fake_git_action(command, path, **kwargs):
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="v1.0.1", metadata=_meta(command))
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=1.0.1", metadata=_meta(command)
            )
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir), force=False)

    assert res.status == "skipped"
    assert "tag_exists=true" in res.data


def test_bump_version_existing_tag_force_but_on_remote_is_still_skipped(
    configured_project,
):
    """force=True does NOT override a tag that is already published -- only
    an orphan (local-only) tag may be deleted and re-bumped."""
    git, proj_dir = configured_project

    def fake_git_action(command, path, **kwargs):
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="v1.0.1", metadata=_meta(command))
        if command.startswith("git ls-remote --tags origin"):
            return GitResult(
                status="success", data="refs/tags/v1.0.1", metadata=_meta(command)
            )
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=1.0.1", metadata=_meta(command)
            )
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir), force=True)

    assert res.status == "skipped"
    assert "tag_exists=true" in res.data


def test_bump_version_existing_orphan_tag_force_deletes_and_rebumps(
    configured_project,
):
    git, proj_dir = configured_project
    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append(command)
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="v1.0.1", metadata=_meta(command))
        if command.startswith("git ls-remote --tags origin"):
            return GitResult(
                status="success", data="", metadata=_meta(command)
            )  # not on remote
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=1.0.1", metadata=_meta(command)
            )
        if command == "git status --porcelain":
            return GitResult(status="success", data="", metadata=_meta(command))
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir), force=True)

    assert res.status == "success"
    assert "git tag -d v1.0.1" in calls
    # The orphan tag is deleted BEFORE the real bump2version invocation.
    delete_idx = calls.index("git tag -d v1.0.1")
    bump_idx = next(
        i for i, c in enumerate(calls) if c.startswith("SKIP=") and "bump2version" in c
    )
    assert delete_idx < bump_idx


def test_bump_version_bump2version_failure_skips_finalize(configured_project):
    git, proj_dir = configured_project
    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append(command)
        if command.startswith("git tag -l "):
            return GitResult(status="success", data="", metadata=_meta(command))
        if "bump2version" in command and command.startswith("SKIP="):
            return GitResult(
                status="error", data="", error=None, metadata=_meta(command)
            )
        if "bump2version" in command:
            return GitResult(
                status="success", data="new_version=1.0.1", metadata=_meta(command)
            )
        return GitResult(status="success", data="", metadata=_meta(command))

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir))

    assert res.status == "error"
    assert not any("git add -u" == c for c in calls)


def test_bump_version_exception_is_caught_and_returns_error(configured_project):
    git, proj_dir = configured_project

    def fake_git_action(command, path, **kwargs):
        if command.startswith("git tag -l ") or "--dry-run --list" in command:
            return GitResult(status="success", data="", metadata=_meta(command))
        raise RuntimeError("boom")

    with patch.object(Git, "git_action", side_effect=fake_git_action):
        res = git.bump_version(part="patch", path=str(proj_dir))

    assert res.status == "error"
    assert res.error is not None
    assert res.error.message == "RuntimeError"
