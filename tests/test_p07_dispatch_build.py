"""P0.7 — ``rm_build ... host=`` dispatch: ``remote_worker_actions.dispatch_build``.

The literal success test this program names: dispatch a build to a NAMED
host and get back a real result, using the SAME `stage_source`/
`verify_source` primitives (via `TunnelSSHExecutor`, proven live against a
real host — see `ssh_executor.py`) rather than a hand-rolled `ssh ...
systemd-run`.

A fake `Tunnel` stands in for the real SSH transport here so this suite runs
in CI without network access; the actual SSH mechanism itself was validated
live against a real host (R820) during this work, outside pytest (network
access to a real lab host is not something a portable test suite can
require) — see the session report for that literal transcript.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager import remote_worker_actions as rwa
from repository_manager.remote_execution import ssh_executor as ssh_executor_module


@pytest.fixture(autouse=True)
def _isolated_capacity_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    rwa._CAPACITY_INVENTORY = None
    rwa._CAPACITY_STORE = None
    rwa._REGISTRY = None
    yield
    rwa._CAPACITY_INVENTORY = None
    rwa._CAPACITY_STORE = None
    rwa._REGISTRY = None


class _FakeTunnel:
    """Records every dispatched shell command; simulates a real git host."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        self.calls: list[str] = []
        self._materialized_sha: str | None = None

    def run_command(self, command: str, timeout: int | None = None):
        self.calls.append(command)
        if "git clone" in command:
            return SimpleNamespace(success=True, stdout="", stderr="Cloning...")
        if "git fetch" in command:
            return SimpleNamespace(success=True, stdout="", stderr="fetched")
        if "git checkout" in command:
            self._materialized_sha = _SHARED_SHA
            return SimpleNamespace(success=True, stdout="", stderr="checked out")
        if command.endswith("git status --porcelain"):
            return SimpleNamespace(success=True, stdout="", stderr="")
        if command.endswith("git rev-parse HEAD"):
            return SimpleNamespace(success=True, stdout=_SHARED_SHA, stderr="")
        if "echo built" in command:
            return SimpleNamespace(success=True, stdout="built ok", stderr="")
        return SimpleNamespace(success=False, stdout="", stderr=f"unhandled: {command}")


_SHARED_SHA = "a" * 40


@pytest.fixture(autouse=True)
def _fake_tunnel(monkeypatch: pytest.MonkeyPatch) -> _FakeTunnel:
    fake = _FakeTunnel()
    # Inject the factory consumed by TunnelSSHExecutor; patching the Tunnel
    # class itself is too late because _default_tunnel imports paramiko first.
    monkeypatch.setattr(ssh_executor_module, "_default_tunnel", lambda _alias: fake)
    monkeypatch.setattr(ssh_executor_module, "_TUNNEL_MANAGER_IMPORT_ERROR", None)
    return fake


def _register_host(host_id: str = "R820") -> None:
    result = rwa.dispatch(
        "register_worker",
        host_id=host_id,
        cpu_weight=8,
        memory_mib=32768,
        disk_mib=500_000,
        process_slots=4,
        inventory_alias=host_id,
        repository_roots={"epistemic-graph": "/srv/rm-build"},
        toolchains=["rust-1.95"],
    )
    assert result["ok"] is True


def test_dispatch_build_stages_and_runs_the_command_on_the_named_host(
    _fake_tunnel: _FakeTunnel,
) -> None:
    _register_host()

    result = rwa.dispatch(
        "dispatch_build",
        host_id="R820",
        repository_id="epistemic-graph",
        origin="/tmp/some/origin",
        tree_sha=_SHARED_SHA,
        command=("bash", "-c", "echo built"),
        workdir=".",
        cpu_weight=2,
        memory_mib=1024,
        disk_mib=1024,
        process_slots=1,
    )

    assert result["ok"] is True
    assert result["succeeded"] is True
    assert result["host_id"] == "R820"
    assert result["staged"]["tree_sha"] == _SHARED_SHA
    assert result["build"]["outcome"] == "succeeded"
    assert result["build"]["stdout_tail"] == "built ok"
    assert any("git clone" in call for call in _fake_tunnel.calls)
    assert any("git fetch" in call for call in _fake_tunnel.calls)
    assert any("git checkout" in call for call in _fake_tunnel.calls)
    assert any("echo built" in call for call in _fake_tunnel.calls)


def test_dispatch_build_refuses_an_unregistered_host() -> None:
    result = rwa.dispatch(
        "dispatch_build",
        host_id="unknown-host",
        repository_id="epistemic-graph",
        origin="/tmp/origin",
        tree_sha=_SHARED_SHA,
        command=("echo", "hi"),
    )
    assert result["ok"] is False
    assert "unknown remote worker host" in result["refused"]


def test_dispatch_build_refuses_a_repository_not_authorized_on_that_host() -> None:
    _register_host()
    result = rwa.dispatch(
        "dispatch_build",
        host_id="R820",
        repository_id="some-other-repo",
        origin="/tmp/origin",
        tree_sha=_SHARED_SHA,
        command=("echo", "hi"),
    )
    assert result["ok"] is False
    assert "no authorized root" in result["refused"]


def test_dispatch_build_refuses_when_the_host_is_draining(
    _fake_tunnel: _FakeTunnel,
) -> None:
    _register_host()
    from repository_manager.capacity import HostState

    rwa.capacity_inventory().set_state("R820", HostState.DRAINING)

    result = rwa.dispatch(
        "dispatch_build",
        host_id="R820",
        repository_id="epistemic-graph",
        origin="/tmp/origin",
        tree_sha=_SHARED_SHA,
        command=("echo", "hi"),
    )
    assert result["ok"] is False
    assert "draining" in result["refused"]
    # Refused before ever touching the transport.
    assert _fake_tunnel.calls == []
