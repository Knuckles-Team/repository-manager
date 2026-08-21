"""Pre-push gate: run the repo's declared HEAVY (pre-push-stage) hooks before push.

``_gate_before_push`` now runs through ``repository_manager.gates.run_gate_stage``
with ``stage="heavy"`` -- the fix for GOC-60's blocking gap (this method's name
always promised "pre-push" but used to run pre-commit's default commit-stage
hooks). These tests assert the fixed call shape; ``tests/test_gates.py`` proves
the live ``--hook-stage`` firing behavior end to end against real ``pre-commit``.
"""

import subprocess
from unittest.mock import MagicMock, patch

from repository_manager.models import GitError
from repository_manager.repository_manager import Git, GitResult


def _git(tmp_path, ahead="1"):
    """A Git manager whose git_action is mocked; rev-list reports `ahead` commits."""
    m = Git(path=str(tmp_path))
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")

    def side_effect(*args, **kwargs):
        cmd = kwargs.get("command", "") or (args[0] if args else "")
        if "rev-list --count" in cmd:
            return GitResult(status="success", data=ahead, error=None, metadata=None)
        if "diff --name-only" in cmd:
            return GitResult(
                status="success",
                data="pyproject.toml\nfoo.py\n",
                error=None,
                metadata=None,
            )
        if "status --porcelain" in cmd:
            return GitResult(status="success", data="", error=None, metadata=None)
        return GitResult(status="success", data="Pushed", error=None, metadata=None)

    m.git_action = MagicMock(side_effect=side_effect)  # type: ignore[method-assign]
    return m


def _completed(returncode, stdout=""):
    return subprocess.CompletedProcess(
        args=["pre-commit", "run", "--hook-stage", "pre-push", "--all-files", "--verbose"],
        returncode=returncode,
        stdout=stdout,
        stderr="",
    )


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


def test_gate_skipped_without_precommit_config(tmp_path):
    m = _git(tmp_path)
    (tmp_path / ".pre-commit-config.yaml").unlink()
    m.gate_before_push = True
    assert m._gate_before_push(str(tmp_path)) is None


def test_push_refuses_dirty_repository_without_implicit_commit(tmp_path):
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


def test_diverged_push_never_rebases_or_force_pushes(tmp_path):
    manager = Git(path=str(tmp_path))
    manager.gate_before_push = False

    def action(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        if "status --porcelain" in command:
            return GitResult(status="success", data="", error=None)
        return GitResult(
            status="error",
            data="",
            error=GitError(message="non-fast-forward", code=1),
        )

    manager.git_action = MagicMock(side_effect=action)  # type: ignore[method-assign]
    result = manager.push_project(str(tmp_path))

    assert result.status == "error"
    assert result.error and result.error.code == 409
    commands = [
        call.kwargs.get("command", "") for call in manager.git_action.call_args_list
    ]
    assert not any("rebase" in command or "--force" in command for command in commands)


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
