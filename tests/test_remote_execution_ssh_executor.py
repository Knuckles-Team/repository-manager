"""``TunnelSSHExecutor`` — the SSH primitive that actually exists.

Unlike ``test_remote_execution_executor.py`` (which exercises the
``tunnel_manager.remote_execution`` seam that lives only on an unmerged
tunnel-manager branch and is not installed here), these tests inject a fake
``Tunnel`` factory so they never require a real SSH connection or even a real
``tunnel_manager`` install to prove the executor's OWN logic (argv
construction, outcome mapping, cancellation/fence downgrade). The base-import
and refusal-without-tunnel-manager behavior is proven the same way
``test_remote_execution_optional_dependency.py`` proves it for the sibling
executor.
"""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from repository_manager.development import ExecutionCommand
from repository_manager.execution.cancellation import CancellationToken


def _clean_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return env


def test_base_import_succeeds_without_tunnel_manager() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import repository_manager.remote_execution.ssh_executor as m; "
            "print('OK', m._TUNNEL_MANAGER_IMPORT_ERROR is not None)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=_clean_env(),
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


tunnel_manager = pytest.importorskip(
    "tunnel_manager.tunnel_manager",
    reason="optional dependency: tunnel-manager (repository-manager's 'remote' extra)",
)

from repository_manager.remote_execution.ssh_executor import (  # noqa: E402
    TunnelSSHExecutor,
)


class _FakeTunnel:
    """Records the exact shell command dispatched; returns a scripted result."""

    def __init__(self, alias: str, *, responder=None) -> None:
        self.alias = alias
        self.calls: list[tuple[str, int | None]] = []
        self._responder = responder or (
            lambda cmd: SimpleNamespace(success=True, stdout="ok", stderr="")
        )

    def run_command(self, command: str, timeout: int | None = None):
        self.calls.append((command, timeout))
        return self._responder(command)


def _executor(fake: _FakeTunnel) -> TunnelSSHExecutor:
    return TunnelSSHExecutor("R820", tunnel_factory=lambda alias: fake)


def _command(**overrides: object) -> ExecutionCommand:
    base = ExecutionCommand(
        argv=("echo", "hello world"),
        workdir="/srv/repo",
        timeout_seconds=30,
    )
    return base.model_copy(update=overrides) if overrides else base


def test_dispatches_a_correctly_quoted_shell_command() -> None:
    fake = _FakeTunnel("R820")
    result = _executor(fake).run(_command())
    assert result.outcome.value == "succeeded"
    [(sent, timeout)] = fake.calls
    assert sent == "cd /srv/repo && echo 'hello world'"
    assert timeout == 30


def test_failed_remote_exit_is_reported_failed_not_swallowed() -> None:
    fake = _FakeTunnel(
        "R820",
        responder=lambda cmd: SimpleNamespace(success=False, stdout="", stderr="boom"),
    )
    result = _executor(fake).run(_command(argv=("false",)))
    assert result.outcome.value == "failed"
    assert "boom" in result.stderr_tail


def test_a_transport_exception_is_reported_failed_never_raised() -> None:
    def _raise(cmd: str):
        raise ConnectionError("no route to host")

    fake = _FakeTunnel("R820", responder=_raise)
    result = _executor(fake).run(_command())
    assert result.outcome.value == "failed"
    assert "R820" in result.stderr_tail


def test_cancelled_before_dispatch_never_reaches_the_tunnel() -> None:
    fake = _FakeTunnel("R820")
    token = CancellationToken()
    token.cancel()
    result = _executor(fake).run(_command(), cancellation=token)
    assert result.outcome.value == "cancelled"
    assert fake.calls == []


def test_a_success_that_raced_a_fence_loss_is_downgraded_never_published_as_success() -> (
    None
):
    fake = _FakeTunnel("R820")
    calls = {"n": 0}

    def _fence_check() -> bool:
        # Valid on the pre-dispatch check, invalid by the time the (already
        # blocking) remote call returns — the exact race this downgrade
        # exists to close.
        calls["n"] += 1
        return calls["n"] == 1

    result = _executor(fake).run(_command(), fence_check=_fence_check)
    assert result.outcome.value == "refused"
    assert fake.calls  # the command DID dispatch; only the report is downgraded


def test_constructing_without_tunnel_manager_refuses_cleanly(monkeypatch) -> None:
    import repository_manager.remote_execution.ssh_executor as m

    monkeypatch.setattr(m, "_TUNNEL_MANAGER_IMPORT_ERROR", ImportError("simulated"))
    with pytest.raises(m.RemoteSshExecutionUnavailableError):
        m.TunnelSSHExecutor("R820")
