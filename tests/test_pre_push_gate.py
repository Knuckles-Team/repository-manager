"""Fast pre-push gate: run repo pre-commit gates (minus full pytest) before push."""

import subprocess
from unittest.mock import MagicMock, patch

from repository_manager.repository_manager import Git, GitResult


def _git(tmp_path, ahead="1"):
    """A Git manager whose git_action is mocked; rev-list reports `ahead` commits."""
    m = Git(path=str(tmp_path))
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")

    def side_effect(*args, **kwargs):
        cmd = kwargs.get("command", "") or (args[0] if args else "")
        if "rev-list --count" in cmd:
            return GitResult(status="success", data=ahead, error=None, metadata=None)
        if "status --porcelain" in cmd:
            return GitResult(status="success", data="", error=None, metadata=None)
        return GitResult(status="success", data="Pushed", error=None, metadata=None)

    m.git_action = MagicMock(side_effect=side_effect)  # type: ignore[method-assign]
    return m


def _completed(returncode, stdout=""):
    return subprocess.CompletedProcess(
        args=["pre-commit"], returncode=returncode, stdout=stdout, stderr=""
    )


def test_gate_disabled_is_noop(tmp_path):
    m = _git(tmp_path)
    m.gate_before_push = False
    assert m._gate_before_push(str(tmp_path)) is None


def test_gate_skips_when_nothing_to_push(tmp_path):
    m = _git(tmp_path, ahead="0")
    m.gate_before_push = True
    with patch("repository_manager.scanner.run_pre_commit") as rpc:
        assert m._gate_before_push(str(tmp_path)) is None
        rpc.assert_not_called()  # never even runs the gate on a no-op repo


def test_gate_passes_lets_push_proceed(tmp_path):
    m = _git(tmp_path)
    m.gate_before_push = True
    with patch("repository_manager.scanner.run_pre_commit", return_value=_completed(0)):
        assert m._gate_before_push(str(tmp_path)) is None


def test_gate_failure_aborts_push(tmp_path):
    m = _git(tmp_path)
    m.gate_before_push = True
    out = "ruff....................................................................Failed\n"
    with patch(
        "repository_manager.scanner.run_pre_commit", return_value=_completed(1, out)
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
