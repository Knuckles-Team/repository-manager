"""Pre-push gate: run the repo's declared HEAVY (pre-push-stage) hooks before push.

``_gate_before_push`` now runs through ``repository_manager.gates.run_gate_stage``
with ``stage="heavy"`` -- the fix for GOC-60's blocking gap (this method's name
always promised "pre-push" but used to run pre-commit's default commit-stage
hooks). These tests assert the fixed call shape; ``tests/test_gates.py`` proves
the live ``--hook-stage`` firing behavior end to end against real ``pre-commit``.
"""

import contextlib
import subprocess
from functools import partial
from typing import Any
from unittest.mock import MagicMock, patch

from agent_utilities.governance.lanes import hold_lease

from repository_manager.models import GitError
from repository_manager.repository_manager import Git, GitResult

_HEAD_OID = "b" * 40
_UPSTREAM_OID = "a" * 40


def _git_success(data: str = "") -> GitResult:
    return GitResult(status="success", data=data, error=None, metadata=None)


def _git_failure(message: str = "not configured") -> GitResult:
    return GitResult(
        status="error",
        data="",
        error=GitError(message=message, code=1),
        metadata=None,
    )


def _call_command(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    command = kwargs.get("command", "")
    if isinstance(command, str) and command:
        return command
    first = args[0] if args else ""
    return first if isinstance(first, str) else ""


def _mock_git_action(ahead: str, *args: Any, **kwargs: Any) -> GitResult:
    """Return deterministic Git results for the gate's mocked manager."""
    command = _call_command(args, kwargs)
    if "rev-parse --verify" in command:
        oid = _UPSTREAM_OID if "refs/repository-manager/" in command else _HEAD_OID
        return _git_success(oid)
    if "rev-list --count" in command:
        return _git_success(ahead)
    for marker, result in (
        ("symbolic-ref --quiet --short HEAD", _git_success("main")),
        ("config --get-all branch.main.remote", _git_success("origin")),
        ("config --get-all branch.main.merge", _git_success("refs/heads/main")),
        ("git config --get-all", _git_failure()),
        ("remote get-url --push", _git_success("configured-push-url")),
        ("check-ref-format", _git_success()),
        ("merge-base --is-ancestor", _git_success()),
        ("git fetch", _git_success()),
        ("git update-ref -d", _git_success()),
        ("diff --name-only", _git_success("pyproject.toml\nfoo.py\n")),
        ("status --porcelain", _git_success()),
    ):
        if marker in command:
            return result
    return _git_success("Pushed")


def _ensure_test_repo(tmp_path):
    if not (tmp_path / ".git").exists():
        _run_git(tmp_path, "init", "--initial-branch=main")


def _git(tmp_path, ahead="1"):
    """A Git manager whose git_action is mocked; rev-list reports `ahead` commits."""
    _ensure_test_repo(tmp_path)
    m = Git(path=str(tmp_path))
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    m.git_action = MagicMock(  # type: ignore[method-assign]
        side_effect=partial(_mock_git_action, ahead)
    )
    return m


def _completed(returncode, stdout=""):
    return subprocess.CompletedProcess(
        args=[
            "pre-commit",
            "run",
            "--hook-stage",
            "pre-push",
            "--all-files",
            "--verbose",
        ],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


def _run_git(path, *args):
    return subprocess.run(
        ["git", *args],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _remote_candidate(tmp_path, *, remote_commits=1):
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    candidate = tmp_path / "candidate"
    _run_git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    seed.mkdir()
    _run_git(seed, "init", "--initial-branch=main")
    _run_git(seed, "config", "user.email", "tests@example.invalid")
    _run_git(seed, "config", "user.name", "Repository Manager Tests")
    (seed / "base.txt").write_text("base\n")
    _run_git(seed, "add", "base.txt")
    _run_git(seed, "commit", "-m", "base")
    base_oid = _run_git(seed, "rev-parse", "HEAD")
    _run_git(seed, "remote", "add", "origin", str(remote))
    _run_git(seed, "push", "-u", "origin", "main")
    if remote_commits > 1:
        (seed / "second.txt").write_text("second\n")
        _run_git(seed, "add", "second.txt")
        _run_git(seed, "commit", "-m", "second")
        _run_git(seed, "push", "origin", "main")
    snapshotted_oid = _run_git(seed, "rev-parse", "HEAD")

    _run_git(tmp_path, "clone", str(remote), str(candidate))
    _run_git(candidate, "config", "user.email", "tests@example.invalid")
    _run_git(candidate, "config", "user.name", "Repository Manager Tests")
    (candidate / "candidate.txt").write_text("candidate\n")
    _run_git(candidate, "add", "candidate.txt")
    _run_git(candidate, "commit", "-m", "candidate")
    return remote, seed, candidate, base_oid, snapshotted_oid


def test_gate_disabled_is_noop(tmp_path):
    m = _git(tmp_path)
    m.gate_before_push = False
    assert m._gate_before_push(str(tmp_path)) is None


def test_gate_skips_when_nothing_to_push(tmp_path):
    m = _git(tmp_path, ahead="0")
    m.gate_before_push = True
    with patch("repository_manager.gates._run_pre_commit") as rpc:
        assert m._gate_before_push(str(tmp_path)) is None
        rpc.assert_not_called()  # never even runs the gate on a no-op repo


def test_unpushed_check_refreshes_stale_tracking_ref_from_actual_upstream(tmp_path):
    """A bundle-carried ``origin/main`` must not make a real push look empty.

    A transported checkout can carry a remote-tracking ref that already points
    at its local candidate even though the actual remote is still at the base
    commit.  Counting ``@{u}..HEAD`` without contacting the remote then returns
    zero and used to skip the complete pre-push gate.
    """
    remote = tmp_path / "remote.git"
    seed = tmp_path / "seed"
    candidate = tmp_path / "candidate"

    _run_git(tmp_path, "init", "--bare", "--initial-branch=main", str(remote))
    seed.mkdir()
    _run_git(seed, "init", "--initial-branch=main")
    _run_git(seed, "config", "user.email", "tests@example.invalid")
    _run_git(seed, "config", "user.name", "Repository Manager Tests")
    (seed / "base.txt").write_text("base\n")
    _run_git(seed, "add", "base.txt")
    _run_git(seed, "commit", "-m", "base")
    _run_git(seed, "remote", "add", "origin", str(remote))
    _run_git(seed, "push", "-u", "origin", "main")
    _run_git(seed, "tag", "remote-only-tag")
    _run_git(seed, "push", "origin", "remote-only-tag")

    _run_git(tmp_path, "clone", str(remote), str(candidate))
    _run_git(candidate, "tag", "-d", "remote-only-tag")
    _run_git(candidate, "config", "user.email", "tests@example.invalid")
    _run_git(candidate, "config", "user.name", "Repository Manager Tests")
    (candidate / "candidate.txt").write_text("candidate\n")
    _run_git(candidate, "add", "candidate.txt")
    _run_git(candidate, "commit", "-m", "candidate")

    # Reproduce the bundle case: the tracking ref says the candidate is
    # published, but the remote itself still has only the base commit.
    _run_git(candidate, "update-ref", "refs/remotes/origin/main", "HEAD")
    assert _run_git(candidate, "rev-list", "--count", "@{u}..HEAD") == "0"
    assert _run_git(candidate, "rev-parse", "HEAD") != _run_git(
        remote, "rev-parse", "main"
    )

    manager = Git(path=str(candidate))
    original_action = manager.git_action
    commands = []
    fetch_head = candidate / ".git" / "FETCH_HEAD"
    local_head = _run_git(candidate, "rev-parse", "HEAD")

    def race_fetch_head(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        commands.append(command)
        result = original_action(*args, **kwargs)
        if "git fetch" in command:
            # Simulate another fetch overwriting the shared pseudoref after
            # ours. The snapshot must use its private ref/OID, never this file.
            fetch_head.write_text(f"{local_head}\t\tbranch 'racer'\n")
        return result

    manager.git_action = MagicMock(  # type: ignore[method-assign]
        side_effect=race_fetch_head
    )
    refresh = manager._refresh_upstream(str(candidate))

    assert refresh.status == "success", refresh
    assert refresh.upstream_oid == _run_git(remote, "rev-parse", "main")
    assert manager._has_unpushed_commits(str(candidate), refresh) is True
    assert manager._unpushed_changed_files(str(candidate), refresh) == ["candidate.txt"]
    assert fetch_head.read_text().startswith(local_head)
    assert _run_git(candidate, "rev-parse", "refs/remotes/origin/main") == local_head
    assert _run_git(candidate, "tag", "--list", "remote-only-tag") == ""
    assert (
        _run_git(
            candidate,
            "for-each-ref",
            "--format=%(refname)",
            "refs/repository-manager/push-upstream",
        )
        == ""
    )
    fetch_command = next(command for command in commands if "git fetch" in command)
    for flag in (
        "--no-tags",
        "--no-recurse-submodules",
        "--no-auto-maintenance",
        "--no-write-fetch-head",
        "--no-write-commit-graph",
        "--refmap=",
    ):
        assert flag in fetch_command

    (candidate / ".pre-commit-config.yaml").write_text("repos: []\n")
    with patch(
        "repository_manager.repository_manager.run_gate_stage",
        return_value=MagicMock(success=True),
    ) as run_gate:
        assert manager._gate_before_push(str(candidate)) is None
    assert run_gate.call_args.kwargs["files"] == ["candidate.txt"]


def test_refresh_uses_configured_remote_with_nonstandard_fetch_refspec(tmp_path):
    actual = tmp_path / "actual.git"
    wrong = tmp_path / "wrong.git"
    seed = tmp_path / "seed"
    candidate = tmp_path / "candidate"

    _run_git(tmp_path, "init", "--bare", "--initial-branch=main", str(actual))
    _run_git(tmp_path, "init", "--bare", "--initial-branch=main", str(wrong))
    seed.mkdir()
    _run_git(seed, "init", "--initial-branch=main")
    _run_git(seed, "config", "user.email", "tests@example.invalid")
    _run_git(seed, "config", "user.name", "Repository Manager Tests")
    (seed / "base.txt").write_text("base\n")
    _run_git(seed, "add", "base.txt")
    _run_git(seed, "commit", "-m", "base")
    _run_git(seed, "remote", "add", "origin", str(actual))
    _run_git(seed, "push", "origin", "main")

    _run_git(tmp_path, "clone", str(actual), str(candidate))
    _run_git(candidate, "remote", "rename", "origin", "upstream")
    _run_git(candidate, "remote", "add", "origin", str(wrong))
    _run_git(candidate, "config", "user.email", "tests@example.invalid")
    _run_git(candidate, "config", "user.name", "Repository Manager Tests")
    (candidate / "candidate.txt").write_text("candidate\n")
    _run_git(candidate, "add", "candidate.txt")
    _run_git(candidate, "commit", "-m", "candidate")
    _run_git(candidate, "config", "--unset-all", "remote.upstream.fetch")
    _run_git(
        candidate,
        "config",
        "--add",
        "remote.upstream.fetch",
        "+refs/heads/other:refs/remotes/upstream/not-main",
    )
    _run_git(candidate, "config", "branch.main.remote", "upstream")
    _run_git(candidate, "config", "branch.main.merge", "refs/heads/main")

    manager = Git(path=str(candidate))
    original_action = manager.git_action
    manager.git_action = MagicMock(  # type: ignore[method-assign]
        wraps=original_action
    )
    refresh = manager._refresh_upstream(str(candidate))

    assert refresh.status == "success", refresh
    assert refresh.upstream_oid == _run_git(actual, "rev-parse", "main")
    fetch_command = next(
        call.kwargs["command"]
        for call in manager.git_action.call_args_list
        if "git fetch" in call.kwargs.get("command", "")
    )
    assert refresh.push_remote == "upstream"
    assert str(actual) in fetch_command
    assert str(wrong) not in fetch_command


def test_push_remote_and_exact_refspec_override_other_defaults(tmp_path):
    manager = _git(tmp_path)
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "config --get-all branch.main.pushRemote" in command:
            return GitResult(status="success", data="publish", error=None)
        if "config --get-all remote.pushDefault" in command:
            return GitResult(status="success", data="mirror", error=None)
        if "config --get-all remote.publish.push" in command:
            return GitResult(
                status="success",
                data="HEAD:refs/heads/release",
                error=None,
            )
        if "remote get-url --push --all -- publish" in command:
            return GitResult(status="success", data="publish-url", error=None)
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "success"
    assert refresh.push_remote == "publish"
    assert refresh.push_target == "publish-url"
    assert refresh.destination_ref == "refs/heads/release"
    fetch_command = next(
        call.kwargs["command"]
        for call in manager.git_action.call_args_list
        if "git fetch" in call.kwargs.get("command", "")
    )
    assert "publish-url" in fetch_command
    assert "+refs/heads/release:refs/repository-manager/" in fetch_command


def test_multiple_push_refspecs_are_refused_as_ambiguous(tmp_path):
    manager = _git(tmp_path)
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "config --get-all remote.origin.push" in command:
            return GitResult(
                status="success",
                data="main:main\nmain:release\n",
                error=None,
            )
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "error"
    assert "multiple remote push refspecs" in (refresh.error or "")
    assert not any(
        "git fetch" in call.kwargs.get("command", "")
        for call in manager.git_action.call_args_list
    )


def test_push_default_matching_is_refused_as_ambiguous(tmp_path):
    manager = _git(tmp_path)
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "config --get-all push.default" in command:
            return GitResult(status="success", data="matching", error=None)
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "error"
    assert "does not select one exact branch" in (refresh.error or "")


def test_multiple_push_urls_are_refused_as_ambiguous(tmp_path):
    manager = _git(tmp_path)
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "remote get-url --push --all -- origin" in command:
            return GitResult(
                status="success", data="first-url\nsecond-url\n", error=None
            )
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "error"
    assert "multiple push remote URLs" in (refresh.error or "")


def test_local_upstream_is_snapshotted_without_fetch(tmp_path):
    _run_git(tmp_path, "init", "--initial-branch=main")
    _run_git(tmp_path, "config", "user.email", "tests@example.invalid")
    _run_git(tmp_path, "config", "user.name", "Repository Manager Tests")
    (tmp_path / "base.txt").write_text("base\n")
    _run_git(tmp_path, "add", "base.txt")
    _run_git(tmp_path, "commit", "-m", "base")
    _run_git(tmp_path, "switch", "-c", "feature")
    (tmp_path / "feature.txt").write_text("feature\n")
    _run_git(tmp_path, "add", "feature.txt")
    _run_git(tmp_path, "commit", "-m", "feature")
    _run_git(tmp_path, "config", "branch.feature.remote", ".")
    _run_git(tmp_path, "config", "branch.feature.merge", "refs/heads/main")
    _run_git(tmp_path, "config", "push.default", "upstream")

    manager = Git(path=str(tmp_path))
    original_action = manager.git_action
    manager.git_action = MagicMock(  # type: ignore[method-assign]
        wraps=original_action
    )
    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "noop"
    assert refresh.upstream_oid == _run_git(tmp_path, "rev-parse", "main")
    assert manager._has_unpushed_commits(str(tmp_path), refresh) is True
    assert manager._unpushed_changed_files(str(tmp_path), refresh) == ["feature.txt"]
    assert not any(
        "git fetch" in call.kwargs.get("command", "")
        for call in manager.git_action.call_args_list
    )


def test_refresh_rejects_head_or_branch_change_during_snapshot(tmp_path):
    manager = _git(tmp_path)
    original_action = manager.git_action
    branch_queries = 0

    def action(*args, **kwargs):
        nonlocal branch_queries
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "symbolic-ref --quiet --short HEAD" in command:
            branch_queries += 1
            if branch_queries > 1:
                return GitResult(status="success", data="other", error=None)
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "error"
    assert "HEAD or branch changed" in (refresh.error or "")


def test_refresh_exception_cleans_private_ref_and_redacts_secret(tmp_path):
    manager = _git(tmp_path)
    original_action = manager.git_action
    sensitive_value = "refresh-token-value"

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "rev-parse --verify" in command and "refs/repository-manager/" in command:
            raise RuntimeError(f"access_token={sensitive_value}")
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    refresh = manager._refresh_upstream(str(tmp_path))

    assert refresh.status == "error"
    assert sensitive_value not in (refresh.error or "")
    assert any(
        "git update-ref -d refs/repository-manager/push-upstream/"
        in call.kwargs.get("command", "")
        for call in manager.git_action.call_args_list
    )


def test_upstream_refresh_failure_aborts_push(tmp_path):
    manager = _git(tmp_path, ahead="0")
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "git fetch" in command:
            return GitResult(
                status="error",
                data="",
                error=GitError(message="remote unavailable", code=1),
            )
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    with patch("repository_manager.repository_manager.run_gate_stage") as run_gate:
        result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert result.error is not None
    assert "Actual upstream refresh did not complete" in result.error.message
    run_gate.assert_not_called()
    assert not any(
        command.startswith("git push")
        for command in (
            call.kwargs.get("command", "") for call in manager.git_action.call_args_list
        )
    )


def test_busy_cross_process_lease_aborts_push(tmp_path):
    manager = _git(tmp_path)

    with hold_lease(
        "pre-push-transaction",
        operation="competing operation",
        path=tmp_path,
    ):
        result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert result.error is not None
    assert "Actual upstream refresh did not complete" in result.error.message
    assert not any(
        "git fetch" in call.kwargs.get("command", "")
        or call.kwargs.get("command", "").startswith("git push")
        for call in manager.git_action.call_args_list
    )


def test_one_repo_lease_spans_gate_and_atomic_push(tmp_path, monkeypatch):
    manager = _git(tmp_path)
    original_action = manager.git_action
    active = False
    lease_kwargs = {}

    @contextlib.contextmanager
    def transaction_lease(name, **kwargs):
        nonlocal active
        assert name == "pre-push-transaction"
        lease_kwargs.update(kwargs)
        active = True
        try:
            yield {}
        finally:
            active = False

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if command.startswith("git push"):
            assert active
        return original_action(*args, **kwargs)

    def gate_result(*args, **kwargs):
        assert active
        return MagicMock(success=True)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]
    monkeypatch.setenv("RM_GATE_TIMEOUT_SECONDS", "1200")
    with (
        patch(
            "repository_manager.repository_manager.hold_lease",
            side_effect=transaction_lease,
        ),
        patch(
            "repository_manager.repository_manager.run_gate_stage",
            side_effect=gate_result,
        ),
    ):
        result = manager.push_project(str(tmp_path))

    assert result.status == "success"
    assert active is False
    assert lease_kwargs["ttl_seconds"] >= 5100
    push_command = next(
        call.kwargs["command"]
        for call in manager.git_action.call_args_list
        if call.kwargs.get("command", "").startswith("git push")
    )
    assert "--atomic" in push_command
    assert f"--force-with-lease=refs/heads/main:{_UPSTREAM_OID}" in push_command
    assert f"{_HEAD_OID}:refs/heads/main" in push_command
    assert "configured-push-url" in push_command


def test_gate_passes_lets_push_proceed(tmp_path):
    m = _git(tmp_path)
    m.gate_before_push = True
    with patch("repository_manager.gates._run_pre_commit", return_value=_completed(0)):
        assert m._gate_before_push(str(tmp_path)) is None


def test_gate_scopes_hooks_to_pushed_diff_and_uses_pre_push_stage(tmp_path):
    """Per-file hooks are scoped to the diff being pushed, AND run at pre-push."""
    m = _git(tmp_path)
    m.gate_before_push = True
    with patch(
        "repository_manager.gates._run_pre_commit", return_value=_completed(0)
    ) as rpc:
        m._gate_before_push(str(tmp_path))
        rpc.assert_called_once()
        assert rpc.call_args.args[1] == "pre-push"  # the literal fix under test
        assert rpc.call_args.kwargs.get("files") == ["pyproject.toml", "foo.py"]


def test_gate_runs_the_heavy_pre_push_stage(tmp_path):
    """The gate must request pre-commit's `pre-push` stage explicitly.

    The fleet's two-tier `.pre-commit-config.yaml` convention defaults to the
    lightweight `pre-commit` stage; the slow/heavy hooks are staged
    `[pre-push, manual]`. Omitting `--hook-stage pre-push` would silently
    re-run only the lightweight tier already enforced at commit time and
    never touch the hooks this gate exists to run (CONCEPT:RM-PUSH
    pre-push-gate-stage).
    """
    m = _git(tmp_path)
    m.gate_before_push = True
    with patch(
        "repository_manager.gates._run_pre_commit", return_value=_completed(0)
    ) as rpc:
        m._gate_before_push(str(tmp_path))
        rpc.assert_called_once()
        # ``hook_stage`` is the second POSITIONAL parameter of
        # ``gates._run_pre_commit`` and the caller passes it positionally, so a
        # kwargs-only read always returns None and the assertion can never fail
        # for the reason it is written to catch. Read whichever form was used.
        call = rpc.call_args
        hook_stage = call.kwargs.get(
            "hook_stage", call.args[1] if len(call.args) > 1 else None
        )
        assert hook_stage == "pre-push"


def test_gate_failure_aborts_push(tmp_path):
    m = _git(tmp_path)
    m.gate_before_push = True
    out = "ruff....................................................................Failed\n- hook id: ruff\n- duration: 0.1s\n"
    with patch(
        "repository_manager.gates._run_pre_commit", return_value=_completed(1, out)
    ):
        res = m.push_project(str(tmp_path))
    assert res.status == "error"
    assert "Pre-push gate failed" in res.error.message
    # the actual push must NOT have run once the gate failed
    pushed = any(
        "git push" in (c.kwargs.get("command", "") or (c.args[0] if c.args else ""))
        for c in m.git_action.call_args_list
    )
    assert not pushed


def test_gate_exception_aborts_push_and_redacts_secret(tmp_path, caplog):
    manager = _git(tmp_path)
    manager.gate_before_push = True
    sensitive_value = "gate-token-value"

    with patch(
        "repository_manager.repository_manager.run_gate_stage",
        side_effect=RuntimeError(f"access_token={sensitive_value}"),
    ):
        result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert result.error is not None
    assert "Pre-push gate did not complete" in result.error.message
    assert sensitive_value not in result.error.message
    assert sensitive_value not in caplog.text
    assert not any(
        "git push" in (call.kwargs.get("command", "") or "")
        for call in manager.git_action.call_args_list
    )


def test_gate_skipped_without_precommit_config(tmp_path):
    m = _git(tmp_path)
    (tmp_path / ".pre-commit-config.yaml").unlink()
    m.gate_before_push = True
    assert m._gate_before_push(str(tmp_path)) is None


def test_push_refuses_dirty_repository_without_implicit_commit(tmp_path):
    _run_git(tmp_path, "init", "--initial-branch=main")
    manager = Git(path=str(tmp_path))
    manager.git_action = MagicMock(  # type: ignore[method-assign]
        return_value=GitResult(status="success", data=" M changed.py", error=None)
    )

    result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert result.error and result.error.code == 409
    commands = [
        call.kwargs.get("command", "") for call in manager.git_action.call_args_list
    ]
    assert not any(
        "git add" in command or "git commit" in command for command in commands
    )
    assert not any("git push" in command for command in commands)


def test_diverged_push_never_rebases_or_unconditionally_forces(tmp_path):
    manager = _git(tmp_path)
    manager.gate_before_push = False
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if command.startswith("git push"):
            return GitResult(
                status="error",
                data="",
                error=GitError(message="non-fast-forward", code=1),
            )
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]
    result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert result.error and result.error.code == 409
    commands = [
        call.kwargs.get("command", "") for call in manager.git_action.call_args_list
    ]
    push_command = next(
        command for command in commands if command.startswith("git push")
    )
    assert "--force-with-lease=refs/heads/main:" in push_command
    assert " --force " not in f" {push_command} "
    assert not any("rebase" in command for command in commands)


def test_remote_advance_after_gate_fails_expected_old_push(tmp_path):
    remote, seed, candidate, _base_oid, snapshotted_oid = _remote_candidate(tmp_path)
    manager = Git(path=str(candidate))
    manager.gate_before_push = False
    original_action = manager.git_action
    advanced_oid = None

    def action(*args, **kwargs):
        nonlocal advanced_oid
        command = kwargs.get("command", "") or (args[0] if args else "")
        if command.startswith("git push") and advanced_oid is None:
            (seed / "advance.txt").write_text("advance\n")
            _run_git(seed, "add", "advance.txt")
            _run_git(seed, "commit", "-m", "advance")
            _run_git(seed, "push", "origin", "main")
            advanced_oid = _run_git(seed, "rev-parse", "HEAD")
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    result = manager.push_project(str(candidate))

    assert result.status == "error"
    assert advanced_oid is not None and advanced_oid != snapshotted_oid
    assert _run_git(remote, "rev-parse", "main") == advanced_oid


def test_remote_rewind_after_gate_fails_expected_old_push(tmp_path):
    remote, seed, candidate, base_oid, snapshotted_oid = _remote_candidate(
        tmp_path, remote_commits=2
    )
    manager = Git(path=str(candidate))
    manager.gate_before_push = False
    original_action = manager.git_action
    rewound = False

    def action(*args, **kwargs):
        nonlocal rewound
        command = kwargs.get("command", "") or (args[0] if args else "")
        if command.startswith("git push") and not rewound:
            _run_git(seed, "reset", "--hard", base_oid)
            _run_git(seed, "push", "--force", "origin", "main")
            rewound = True
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    result = manager.push_project(str(candidate))

    assert result.status == "error"
    assert snapshotted_oid != base_oid
    assert _run_git(remote, "rev-parse", "main") == base_oid


def test_release_tag_is_in_same_atomic_push_and_never_retried_without_tags(tmp_path):
    manager = _git(tmp_path)
    manager.gate_before_push = False
    (tmp_path / ".bumpversion.cfg").write_text(
        "[bumpversion]\ncurrent_version = 1.2.3\n"
    )
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "git tag -l v1.2.3" in command:
            return GitResult(status="success", data="v1.2.3", error=None)
        if "rev-parse --verify" in command and "refs/tags/v1.2.3" in command:
            return GitResult(status="success", data=_HEAD_OID, error=None)
        if command.startswith("git push"):
            return GitResult(
                status="error",
                data="tag already exists",
                error=GitError(message="tag already exists", code=1),
            )
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    push_commands = [
        call.kwargs["command"]
        for call in manager.git_action.call_args_list
        if call.kwargs.get("command", "").startswith("git push")
    ]
    assert len(push_commands) == 1
    assert "--atomic" in push_commands[0]
    assert "refs/tags/v1.2.3:refs/tags/v1.2.3" in push_commands[0]


def test_unresolved_current_release_tag_refuses_branch_push(tmp_path):
    manager = _git(tmp_path)
    manager.gate_before_push = False
    (tmp_path / ".bumpversion.cfg").write_text(
        "[bumpversion]\ncurrent_version = 1.2.3\n"
    )
    original_action = manager.git_action

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "git tag -l v1.2.3" in command:
            return GitResult(status="success", data="v1.2.3", error=None)
        if "rev-parse --verify" in command and "refs/tags/v1.2.3" in command:
            return GitResult(
                status="error",
                data="",
                error=GitError(message="tag disappeared", code=1),
            )
        return original_action(*args, **kwargs)

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]

    result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert "release tag is invalid or unresolved" in result.error.message
    assert not any(
        call.kwargs.get("command", "").startswith("git push")
        for call in manager.git_action.call_args_list
    )


def test_missing_toolchain_is_reported_as_unrunnable_not_as_a_defect(tmp_path):
    """A hook whose executable is absent never ran; saying "fix the gate" lies.

    This is the exact shape the repository-manager MCP pod produced on
    2026-08-21: no Rust toolchain in the container, so every one of
    epistemic-graph's cargo hooks "Failed" in seconds and the push was refused
    with a message that read as a quality verdict. Two investigation cycles
    went into looking for a defect that did not exist.
    """
    m = _git(tmp_path)
    m.gate_before_push = True
    out = (
        "cargo fmt...............................................................Failed\n"
        "- hook id: cargo-fmt\n"
        "- duration: 0.01s\n"
        "\n"
        "Executable `cargo` not found\n"
        "clippy..................................................................Failed\n"
        "- hook id: clippy\n"
        "- duration: 0.01s\n"
        "\n"
        "cargo: command not found\n"
    )
    with patch(
        "repository_manager.gates._run_pre_commit", return_value=_completed(1, out)
    ):
        res = m.push_project(str(tmp_path))

    assert res.status == "error"
    assert "CANNOT RUN" in res.error.message
    # ``hook_id`` here is pre-commit's display NAME (what the parser keys on),
    # not the ``- hook id:`` slug -- that is what the operator sees in the output.
    assert "cargo fmt" in res.error.message and "clippy" in res.error.message
    # It must still refuse the push -- an ungated push is worse than a confusing
    # message. Only the REASON changes.
    assert not any(
        "git push" in (c.kwargs.get("command", "") or (c.args[0] if c.args else ""))
        for c in m.git_action.call_args_list
    )


def test_one_real_failure_alongside_a_missing_toolchain_still_says_gate_failed(
    tmp_path,
):
    """The honest-reporting path is for a TOTAL environment gap, not a partial one.

    If even one hook actually ran and found something, there is a real verdict
    to report and it must not be softened into "this environment cannot gate".
    """
    m = _git(tmp_path)
    m.gate_before_push = True
    out = (
        "cargo fmt...............................................................Failed\n"
        "- hook id: cargo-fmt\n"
        "- duration: 0.01s\n"
        "\n"
        "Executable `cargo` not found\n"
        "ruff....................................................................Failed\n"
        "- hook id: ruff\n"
        "- duration: 0.2s\n"
        "\n"
        "foo.py:1:1: F401 `os` imported but unused\n"
    )
    with patch(
        "repository_manager.gates._run_pre_commit", return_value=_completed(1, out)
    ):
        res = m.push_project(str(tmp_path))

    assert res.status == "error"
    assert "Pre-push gate failed" in res.error.message
    assert "CANNOT RUN" not in res.error.message


def test_a_missing_data_file_is_not_miscredited_to_a_missing_toolchain(tmp_path):
    """`No such file or directory` is only a toolchain signal in its errno form.

    A gate that ran fine and failed because an input file was absent is a real
    verdict; reporting it as "install the toolchain" would send the reader to
    the wrong place.
    """
    m = _git(tmp_path)
    m.gate_before_push = True
    out = (
        "schema check............................................................Failed\n"
        "- hook id: schema-check\n"
        "- duration: 0.3s\n"
        "\n"
        "cat: config/schema.json: No such file or directory\n"
    )
    with patch(
        "repository_manager.gates._run_pre_commit", return_value=_completed(1, out)
    ):
        res = m.push_project(str(tmp_path))

    assert res.status == "error"
    assert "Pre-push gate failed" in res.error.message
    assert "CANNOT RUN" not in res.error.message
