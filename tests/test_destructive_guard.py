"""RMDD-26 destructive-verb refusal, snapshot, and concurrency tests."""

from __future__ import annotations

import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repository_manager import destructive_guard, stash_guard
from tests.fixtures.tree_mutation_hazards import (
    bare_destructive_attempt,
    mutation_race,
    tracked_destructive_loss,
)


def _git(
    args: list[str], path: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=str(path), capture_output=True, text=True, check=check
    )


def _audit(repo: Path, lane: str, operation: str, argv: list[str]) -> dict[str, str]:
    return {
        "actor": "test",
        "lane": lane,
        "operation": operation,
        "repository": str(repo.resolve()),
        "argv": shlex.join(argv),
    }


def test_classifier_covers_every_destructive_verb() -> None:
    cases = {
        ("git", "reset", "--hard", "HEAD~1"): "mixed-reset",
        ("git", "checkout", "."): "reviewed-path-operation",
        ("git", "checkout", "--", "."): "reviewed-path-operation",
        ("git", "checkout", "-f", "main"): "reviewed-path-operation",
        ("git", "restore", "."): "reviewed-path-operation",
        ("git", "clean", "-fd"): "park-unpark",
        ("git", "stash"): "park-unpark",
        ("git", "stash", "pop"): "park-unpark",
        ("git", "stash", "apply"): "park-unpark",
        ("git", "branch", "-D", "lane"): "guarded-prune",
        ("git", "branch", "-df", "lane"): "guarded-prune",
        ("git", "push", "--force"): "reviewed-fast-forward",
        ("git", "push", "-uf"): "reviewed-fast-forward",
        ("git", "push", "-d", "origin", "lane"): "reviewed-fast-forward",
        ("git", "push", "+main:main"): "reviewed-fast-forward",
        ("git", "push", "origin", ":refs/heads/lane"): "reviewed-fast-forward",
        ("git", "push", "--prune"): "reviewed-fast-forward",
    }
    for argv, alternative in cases.items():
        result = destructive_guard.classify(argv)
        assert result["dangerous"] is True
        assert result["alternative_code"] == alternative
        assert result["safer_alternative"]

    private = destructive_guard.classify(("git", "stash", "apply", "refs/lane/a/stash"))
    assert private["dangerous"] is True
    assert private["alternative_code"] == "park-unpark"
    private_show = destructive_guard.classify(
        ("git", "stash", "show", "refs/lane/a/stash")
    )
    assert private_show["dangerous"] is False
    assert (
        destructive_guard.classify(("git", "stash", "push", "refs/lane/a/stash"))[
            "dangerous"
        ]
        is True
    )
    assert (
        destructive_guard.classify(("git", "branch", "--delete", "lane"))["dangerous"]
        is False
    )
    for argv in (
        ("git", "checkout", "./"),
        ("git", "checkout", "--", "./"),
        ("git", "checkout", "*"),
        ("git", "restore", "--source=HEAD", "--worktree", "."),
    ):
        assert destructive_guard.classify(argv)["dangerous"] is True


def test_guard_fails_closed_for_non_git_malformed_and_alias_argv(
    tmp_path: Path, monkeypatch
) -> None:
    repo = bare_destructive_attempt(tmp_path)
    monkeypatch.setenv("GIT_CONFIG_COUNT", "1")
    monkeypatch.setenv("GIT_CONFIG_KEY_0", "alias.nuke")
    monkeypatch.setenv("GIT_CONFIG_VALUE_0", "reset --hard HEAD")
    commands = [
        [],
        ["rm", "-f", "must-survive.txt"],
        ["git"],
        ["git", "-C"],
        ["git", "nuke"],
        ["git", "-c", "alias.nuke=reset --hard", "nuke"],
        ["git", "-c", "alias.x=!rm -f must-survive.txt", "x"],
    ]

    for command in commands:
        result = destructive_guard.guard(command, path=repo)
        assert result["status"] == "refused"
        assert result["executed"] is False
        assert result["reason"] == "unsupported-git-argv"

    assert (repo / "must-survive.txt").exists()


def test_reset_hard_refuses_without_touching_dirty_tree(tmp_path: Path) -> None:
    repo = bare_destructive_attempt(tmp_path)
    before = (repo / "must-survive.txt").read_text()

    result = destructive_guard.guard(["git", "reset", "--hard", "HEAD~1"], path=repo)

    assert result["status"] == "refused"
    assert result["executed"] is False
    assert result["alternative_code"] == "mixed-reset"
    assert (repo / "must-survive.txt").read_text() == before
    assert result["snapshot_ref"] is None


def test_tracked_destructive_fixture_proves_the_bare_reset_hazard(
    tmp_path: Path,
) -> None:
    repo = tracked_destructive_loss(tmp_path)
    before = (repo / "README.md").read_text()

    _git(["reset", "--hard", "HEAD"], repo)

    assert (repo / "README.md").read_text() != before


@pytest.mark.parametrize("flag", ["--assume-unchanged", "--skip-worktree"])
def test_snapshot_captures_hidden_tracked_wip(tmp_path: Path, flag: str) -> None:
    repo = bare_destructive_attempt(tmp_path)
    original = (repo / "README.md").read_text()
    (repo / "README.md").write_text("hidden tracked WIP\n")
    _git(["update-index", flag, "README.md"], repo)
    assert "README.md" not in _git(["status", "--porcelain"], repo).stdout
    command = ["git", "reset", "--hard", "HEAD"]
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(repo, "hidden", "git reset --hard HEAD", command),
    )

    result = destructive_guard.guard(command, path=repo, lane="hidden", override=token)

    assert result["status"] == "success"
    assert result["park_ref"]
    assert (repo / "README.md").read_text() == original
    restored = stash_guard.unpark(
        destructive_guard._GitAdapter(), str(repo), ref=result["park_ref"]
    )
    assert restored["ok"] is True
    assert (repo / "README.md").read_text() == "hidden tracked WIP\n"


def test_bare_stash_refuses_and_redirects_to_private_park(tmp_path: Path) -> None:
    repo = bare_destructive_attempt(tmp_path)
    result = destructive_guard.guard(["git", "stash"], path=repo)

    assert result["status"] == "refused"
    assert result["alternative_code"] == "park-unpark"
    assert "park" in result["safer_alternative"]
    assert (
        _git(
            ["rev-parse", "--verify", "--quiet", "refs/stash"], repo, check=False
        ).returncode
        != 0
    )


def test_clean_override_requires_snapshot_and_restores_private_wip(
    tmp_path: Path,
) -> None:
    repo = bare_destructive_attempt(tmp_path)
    original = (repo / "must-survive.txt").read_text()
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(
            repo, "lane-clean", "git clean -fd", ["git", "clean", "-fd"]
        ),
    )

    result = destructive_guard.guard(
        ["git", "clean", "-fd"], path=repo, lane="lane-clean", override=token
    )

    assert result["status"] == "success"
    assert result["override_used"] is True
    assert result["snapshot_required"] is True
    assert result["snapshot_created_at"]
    assert result["triggering_operation"] == "git clean -fd"
    assert result["snapshot_ref"].startswith(
        "refs/lane-backup/pre-destructive/lane-clean-"
    )
    assert (
        _git(["rev-parse", result["snapshot_ref"]], repo).stdout.strip()
        == _git(["rev-parse", "HEAD"], repo).stdout.strip()
    )
    assert result["park_ref"]
    assert not (repo / "must-survive.txt").exists()

    restored = stash_guard.unpark(
        destructive_guard._GitAdapter(),  # The guard's fixed-argv adapter is intentional.
        str(repo),
        ref=result["park_ref"],
    )
    assert restored["ok"] is True
    assert (repo / "must-survive.txt").read_text() == original


def test_override_token_is_single_use_and_not_a_standing_config(tmp_path: Path) -> None:
    repo = bare_destructive_attempt(tmp_path)
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(
            repo, "lane-token", "git clean -fd", ["git", "clean", "-fd"]
        ),
    )
    first = destructive_guard.guard(
        ["git", "clean", "-fd"],
        path=repo,
        lane="lane-token",
        override=token,
        execute=False,
    )
    second = destructive_guard.guard(
        ["git", "clean", "-fd"],
        path=repo,
        lane="lane-token",
        override=token,
        execute=False,
    )

    assert first["status"] == "authorized"
    assert first["snapshot_ref"]
    assert second["status"] == "refused"
    assert second["reason"] == "override-single-use"


def test_override_token_is_bound_to_lane_and_operation(tmp_path: Path) -> None:
    repo = bare_destructive_attempt(tmp_path)
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(repo, "lane-a", "git clean -fd", ["git", "clean", "-fd"]),
    )

    wrong_lane = destructive_guard.guard(
        ["git", "clean", "-fd"], path=repo, lane="lane-b", override=token
    )
    wrong_operation = destructive_guard.guard(
        ["git", "reset", "--hard", "HEAD"], path=repo, lane="lane-a", override=token
    )

    assert wrong_lane["status"] == "refused"
    assert wrong_lane["reason"] == "override-scope-mismatch"
    assert wrong_operation["status"] == "refused"
    assert wrong_operation["reason"] == "override-scope-mismatch"


def test_override_token_is_bound_to_exact_repository(tmp_path: Path) -> None:
    repo_a = bare_destructive_attempt(tmp_path / "a")
    repo_b = bare_destructive_attempt(tmp_path / "b")
    command = ["git", "clean", "-fd"]
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(repo_a, "same-lane", "git clean -fd", command),
    )

    wrong_repository = destructive_guard.guard(
        command, path=repo_b, lane="same-lane", override=token, execute=False
    )

    assert wrong_repository["status"] == "refused"
    assert wrong_repository["reason"] == "override-scope-mismatch"
    right_repository = destructive_guard.guard(
        command, path=repo_a, lane="same-lane", override=token, execute=False
    )
    assert right_repository["status"] == "authorized"


def test_guard_refuses_alternate_git_targets_and_fake_git_executable(
    tmp_path: Path,
) -> None:
    repo = bare_destructive_attempt(tmp_path / "repo")
    alternate = bare_destructive_attempt(tmp_path / "alternate")
    fake = tmp_path / "fake" / "git"
    fake.parent.mkdir()
    fake.write_text("#!/bin/sh\ntouch marker\n")
    fake.chmod(0o755)

    commands = [
        ["git", "--git-dir", str(alternate / ".git"), "status"],
        ["git", "--work-tree", str(alternate), "status"],
        [str(fake), "status"],
    ]
    for command in commands:
        result = destructive_guard.guard(command, path=repo)
        assert result["status"] == "refused"
        assert result["reason"] == "unsupported-git-argv"
        assert result["executed"] is False
    assert not (repo / "marker").exists()


def test_tree_mutation_lease_spans_snapshot_and_execute(
    tmp_path: Path, monkeypatch
) -> None:
    repo = mutation_race(tmp_path)
    command = ["git", "clean", "-fd"]
    first_token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(repo, "race", "git clean -fd", command),
    )
    nested: list[dict] = []
    original_run = destructive_guard.subprocess.run

    def race_hook(argv, *args, **kwargs):
        if list(argv)[1:] == command[1:]:
            nested_token = destructive_guard.issue_override_token(
                authorization=lambda: True,
                audit_context=_audit(repo, "race", "git clean -fd", command),
            )
            nested.append(
                destructive_guard.guard(
                    command,
                    path=repo,
                    lane="race",
                    override=nested_token,
                )
            )
        return original_run(argv, *args, **kwargs)

    monkeypatch.setattr(destructive_guard.subprocess, "run", race_hook)
    result = destructive_guard.guard(
        command, path=repo, lane="race", override=first_token
    )

    assert result["status"] == "success"
    assert nested and nested[0]["reason"] == "tree-mutation-busy"


def test_ignored_clean_is_refused_even_with_override(tmp_path: Path) -> None:
    repo = bare_destructive_attempt(tmp_path)
    (repo / ".gitignore").write_text("ignored.txt\n")
    (repo / "ignored.txt").write_text("must survive\n")
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(
            repo, "lane-ignored", "git clean -fdx", ["git", "clean", "-fdx"]
        ),
    )

    result = destructive_guard.guard(
        ["git", "clean", "-fdx"], path=repo, lane="lane-ignored", override=token
    )

    assert result["status"] == "refused"
    assert result["reason"] == "ignored-clean-forbidden"
    assert (repo / "ignored.txt").read_text() == "must survive\n"


def test_failed_snapshot_consumes_the_override_token_fail_closed(
    tmp_path: Path,
) -> None:
    repo = bare_destructive_attempt(tmp_path)
    _git(["update-ref", "refs/lane/lane-fail/stash", "HEAD"], repo)
    token = destructive_guard.issue_override_token(
        authorization=lambda: True,
        audit_context=_audit(
            repo, "lane-fail", "git clean -fd", ["git", "clean", "-fd"]
        ),
    )

    failed = destructive_guard.guard(
        ["git", "clean", "-fd"], path=repo, lane="lane-fail", override=token
    )
    retried = destructive_guard.guard(
        ["git", "clean", "-fd"], path=repo, lane="lane-fail", override=token
    )

    assert failed["status"] == "refused"
    assert failed["reason"] == "snapshot-required"
    assert failed["override_consumed"] is True
    assert failed["snapshot_required"] is True
    assert failed["snapshot_created_at"]
    assert failed["triggering_operation"] == "git clean -fd"
    assert retried["reason"] == "override-single-use"


def test_override_token_minting_requires_live_authorization_and_audit_context(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("RM_DESTRUCTIVE_OVERRIDE", "1")
    try:
        destructive_guard.issue_override_token()
    except PermissionError:
        pass
    else:  # pragma: no cover - assertion branch
        raise AssertionError("standing environment configuration must not mint a token")
    with pytest.raises(PermissionError):
        destructive_guard.issue_override_token(
            authorization=lambda: True,
            audit_context={
                "actor": "test",
                "lane": "lane-only",
                "operation": "git clean -fd",
                "argv": "git clean -fd",
            },
        )
    with pytest.raises(PermissionError):
        destructive_guard.issue_override_token(
            authorization=lambda: True,
            audit_context={
                "actor": "test",
                "lane": "lane-only",
                "operation": "git clean -fd",
                "repository": str(tmp_path.resolve()),
            },
        )


def test_twenty_wrapped_operations_have_exact_refusal_count(tmp_path: Path) -> None:
    repositories = [
        bare_destructive_attempt(tmp_path / f"lane-{index}") for index in range(20)
    ]

    def run(index_and_repo: tuple[int, Path]) -> dict:
        index, repo = index_and_repo
        command = ["git", "clean", "-fd"] if index < 4 else ["git", "status"]
        return destructive_guard.guard(command, path=repo, lane=f"lane-{index}")

    with ThreadPoolExecutor(max_workers=20) as pool:
        results = list(pool.map(run, enumerate(repositories)))

    assert sum(result["status"] == "refused" for result in results) == 4
    assert sum(result["executed"] for result in results) == 16
    for index in range(4):
        assert (repositories[index] / "must-survive.txt").exists()
