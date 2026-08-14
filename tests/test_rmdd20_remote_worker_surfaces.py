"""RMDD-20: MCP/CLI parity and refusal proofs for the exposed remote-worker surface.

``rm_remote_workers`` (MCP) and ``--remote-workers`` (CLI) must invoke the
identical :func:`repository_manager.remote_worker_actions.dispatch`
application service.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from types import SimpleNamespace

import pytest

from repository_manager.cli_commands.remote_workers import run_remote_workers_cli
from repository_manager.mcp_server import get_mcp_instance
from repository_manager.remote_execution import ssh_executor as ssh_executor_module
from repository_manager.remote_execution.executor import RemoteExecutionUnavailableError


async def _rm_remote_workers_tool():
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    return next(t for t in tools if t.name == "rm_remote_workers")


# `.fn(...)` calls the raw Python function directly, bypassing FastMCP's own
# pydantic argument binding (which is what resolves each unset
# `Field(default=...)` to its real default over the wire) -- every optional
# parameter must be passed explicitly to mirror what a real MCP call
# produces.
_ALL_OPTIONAL_KWARGS: dict = {
    "host_id": None,
    "path": None,
    "cpu_weight": None,
    "memory_mib": None,
    "disk_mib": None,
    "process_slots": None,
    "labels": None,
    "inventory_alias": None,
    "repository_roots": None,
    "toolchains": None,
    "actor": None,
    "repository_id": None,
    "required_toolchain": None,
    "origin": None,
    "tree_sha": None,
    "parent_root": None,
    "worktree_name": None,
    "timeout_seconds": 1800,
    "execute_locally": False,
    "destination": None,
    "expected_sha": None,
    "root": None,
    "relative_path": None,
    "content_base64": None,
    "declared_digest": None,
    "source_description": None,
    "media_type": None,
    "kind": None,
    "reservation_id": None,
    "work_item_id": None,
    "attempt": None,
    "fence": None,
    "reason": None,
    "command": None,
    "workdir": None,
    "ctx": None,
}


async def _call_tool(tool, **overrides):
    kwargs = dict(_ALL_OPTIONAL_KWARGS)
    kwargs.update(overrides)
    return await tool.fn(**kwargs)


def _cli_result(capsys, action: str, **params) -> tuple[dict, int]:
    args = argparse.Namespace(
        remote_workers=action,
        remote_workers_params_json=json.dumps(params) if params else "",
    )
    exit_code = run_remote_workers_cli(args)
    captured = capsys.readouterr()
    return json.loads(captured.out), exit_code


@pytest.mark.anyio
async def test_mcp_and_cli_register_worker_are_identical(capsys):
    host_id = f"host:parity-{uuid.uuid4().hex[:8]}"
    tool = await _rm_remote_workers_tool()
    mcp_result = await _call_tool(
        tool,
        action="register_worker",
        host_id=host_id,
        cpu_weight=4,
        memory_mib=16384,
        disk_mib=51200,
        process_slots=2,
        inventory_alias="parity-alias",
        repository_roots={"repository-manager": "/opt/rm-worktrees"},
        toolchains=["python3.14"],
    )
    assert mcp_result == {
        "ok": True,
        "host_id": host_id,
        "capacity_registered": True,
        "profile_registered": True,
    }

    host_id_cli = f"host:parity-cli-{uuid.uuid4().hex[:8]}"
    cli_result, exit_code = _cli_result(
        capsys,
        "register_worker",
        host_id=host_id_cli,
        cpu_weight=4,
        memory_mib=16384,
        disk_mib=51200,
        process_slots=2,
        inventory_alias="parity-alias",
        repository_roots={"repository-manager": "/opt/rm-worktrees"},
        toolchains=["python3.14"],
    )
    assert cli_result == {
        "ok": True,
        "host_id": host_id_cli,
        "capacity_registered": True,
        "profile_registered": True,
    }
    assert exit_code == 0


class _FakeTunnel:
    """Records every dispatched shell command; simulates a real git host.

    Mirrors the fake used by ``tests/test_p07_dispatch_build.py`` for the
    ``remote_worker_actions.dispatch`` layer directly -- this one instead
    proves the two ADAPTERS (``rm_remote_workers`` MCP tool, ``--remote-
    workers`` CLI flag) both actually reach ``dispatch_build`` end to end.
    Before this test existed, neither adapter accepted the action at all:
    the CLI's ``choices=[...]`` list and the MCP tool's parameter set had
    not been updated when ``dispatch_build`` (P0.7) was added, so it was
    reachable only via a direct ``remote_worker_actions.dispatch(...)``
    import -- confirmed live against R820 outside pytest while diagnosing
    that host's unrelated NFS delegation-reaper pathology (see the session
    report), which is what surfaced this gap.
    """

    def __init__(self) -> None:
        self.calls: list[str] = []

    def run_command(self, command: str, timeout: int | None = None):
        del timeout
        self.calls.append(command)
        if "git clone" in command:
            return SimpleNamespace(success=True, stdout="", stderr="Cloning...")
        if "git fetch" in command:
            return SimpleNamespace(success=True, stdout="", stderr="fetched")
        if "git checkout" in command:
            return SimpleNamespace(success=True, stdout="", stderr="checked out")
        if command.endswith("git status --porcelain"):
            return SimpleNamespace(success=True, stdout="", stderr="")
        if command.endswith("git rev-parse HEAD"):
            return SimpleNamespace(success=True, stdout=_DISPATCH_BUILD_SHA, stderr="")
        if "echo parity-ok" in command:
            return SimpleNamespace(success=True, stdout="parity-ok", stderr="")
        return SimpleNamespace(success=False, stdout="", stderr=f"unhandled: {command}")


_DISPATCH_BUILD_SHA = "b" * 40


@pytest.fixture
def _fake_tunnel(monkeypatch: pytest.MonkeyPatch) -> _FakeTunnel:
    fake = _FakeTunnel()
    monkeypatch.setattr(ssh_executor_module, "Tunnel", lambda **_kw: fake)
    monkeypatch.setattr(ssh_executor_module, "_TUNNEL_MANAGER_IMPORT_ERROR", None)
    return fake


@pytest.mark.anyio
async def test_mcp_and_cli_dispatch_build_are_identical(capsys, _fake_tunnel):
    """``dispatch_build`` staging + build command must work identically via
    both adapters -- the exact case that regressed silently (see
    ``_FakeTunnel``'s docstring above) until this test and the matching
    ``choices``/parameter fixes existed."""

    host_id_mcp = f"host:dispatch-build-mcp-{uuid.uuid4().hex[:8]}"
    tool = await _rm_remote_workers_tool()
    register_mcp = await _call_tool(
        tool,
        action="register_worker",
        host_id=host_id_mcp,
        cpu_weight=4,
        memory_mib=16384,
        disk_mib=51200,
        process_slots=2,
        inventory_alias=host_id_mcp,
        repository_roots={"repository-manager": "/opt/rm-worktrees"},
        toolchains=["git"],
    )
    assert register_mcp["ok"] is True

    mcp_result = await _call_tool(
        tool,
        action="dispatch_build",
        host_id=host_id_mcp,
        repository_id="repository-manager",
        origin="/tmp/some/origin",
        tree_sha=_DISPATCH_BUILD_SHA,
        command=["bash", "-c", "echo parity-ok"],
        workdir=".",
        cpu_weight=1,
        memory_mib=128,
        disk_mib=512,
        process_slots=1,
    )
    assert mcp_result["ok"] is True
    assert mcp_result["succeeded"] is True
    assert mcp_result["build"]["stdout_tail"] == "parity-ok"
    assert mcp_result["staged"]["tree_sha"] == _DISPATCH_BUILD_SHA

    host_id_cli = f"host:dispatch-build-cli-{uuid.uuid4().hex[:8]}"
    register_cli, register_exit = _cli_result(
        capsys,
        "register_worker",
        host_id=host_id_cli,
        cpu_weight=4,
        memory_mib=16384,
        disk_mib=51200,
        process_slots=2,
        inventory_alias=host_id_cli,
        repository_roots={"repository-manager": "/opt/rm-worktrees"},
        toolchains=["git"],
    )
    assert register_exit == 0
    assert register_cli["ok"] is True

    cli_result, exit_code = _cli_result(
        capsys,
        "dispatch_build",
        host_id=host_id_cli,
        repository_id="repository-manager",
        origin="/tmp/some/origin",
        tree_sha=_DISPATCH_BUILD_SHA,
        command=["bash", "-c", "echo parity-ok"],
        workdir=".",
        cpu_weight=1,
        memory_mib=128,
        disk_mib=512,
        process_slots=1,
    )
    assert exit_code == 0
    assert cli_result["ok"] is True
    assert cli_result["succeeded"] is True
    assert cli_result["build"]["stdout_tail"] == "parity-ok"
    assert cli_result["staged"]["tree_sha"] == _DISPATCH_BUILD_SHA


@pytest.mark.anyio
async def test_mcp_and_cli_recheck_refusal_are_identical(capsys):
    """A host with no tunnel-manager available refuses identically both ways."""

    from repository_manager import remote_worker_actions

    host_id = f"host:recheck-{uuid.uuid4().hex[:8]}"
    remote_worker_actions.dispatch(
        "register_worker",
        host_id=host_id,
        cpu_weight=1,
        memory_mib=1,
        disk_mib=1,
        process_slots=1,
        inventory_alias="recheck-alias",
        repository_roots={"repository-manager": "/opt/rm-worktrees"},
        toolchains=[],
    )

    tool = await _rm_remote_workers_tool()
    mcp_result = await _call_tool(
        tool,
        action="recheck",
        host_id=host_id,
        actor="tester",
        repository_id="repository-manager",
    )
    cli_result, exit_code = _cli_result(
        capsys,
        "recheck",
        host_id=host_id,
        actor="tester",
        repository_id="repository-manager",
    )

    assert mcp_result == cli_result
    assert mcp_result["ok"] is False
    assert mcp_result["error_code"] == "dependency_blocked"
    assert exit_code == 1


@pytest.mark.anyio
async def test_mcp_and_cli_host_loss_reconcile_refusal_are_identical(capsys):
    """No live WorkItem-authoritative scheduler is wired -> honest refusal, both ways."""

    tool = await _rm_remote_workers_tool()
    mcp_result = await _call_tool(
        tool,
        action="host_loss_reconcile",
        host_id="host:none",
        reservation_id="res:1",
        work_item_id="wi:1",
        attempt=1,
        fence="f:1",
    )
    cli_result, exit_code = _cli_result(
        capsys,
        "host_loss_reconcile",
        host_id="host:none",
        reservation_id="res:1",
        work_item_id="wi:1",
        attempt=1,
        fence="f:1",
    )

    assert mcp_result == cli_result
    assert mcp_result["ok"] is False
    assert "no such live scheduler" in mcp_result["refused"]
    assert exit_code == 1


def test_recheck_refusal_preserves_tunnel_manager_import_error_as_cause():
    """Refusal proof: no live inventory resolver -> named refusal (H-12).

    ``_UnavailableInventoryResolver.resolve`` (``repository_manager/
    remote_worker_actions.py``) always refuses ``recheck_at_claim``, by
    design, in either of two states of the environment under test:

    * bare ``tunnel_manager`` is not importable at all -> refusal chains
      the real ``ImportError`` as its cause;
    * ``tunnel_manager`` IS installed (e.g. the published package present
      for other packages' needs, even without RMDD-14's unmerged
      ``remote_execution`` seam -- see ``remote_execution/README.md``) but
      this entrypoint has no configured inventory resolver -> refusal names
      that with nothing to chain (nothing was raised).

    Branch on which state actually holds here rather than assuming the
    first, matching the same fix already applied to the analogous
    concept-authority proof in ``test_rmdd20_concepts_surfaces.py``.
    """

    import importlib.util

    from repository_manager import remote_worker_actions
    from repository_manager.remote_execution import RemoteWorkerRegistryError

    tunnel_manager_present = importlib.util.find_spec("tunnel_manager") is not None

    host_id = f"host:cause-{uuid.uuid4().hex[:8]}"
    remote_worker_actions.dispatch(
        "register_worker",
        host_id=host_id,
        cpu_weight=1,
        memory_mib=1,
        disk_mib=1,
        process_slots=1,
        inventory_alias="cause-alias",
        repository_roots={"repository-manager": "/opt/rm-worktrees"},
        toolchains=[],
    )

    with pytest.raises(RemoteWorkerRegistryError) as excinfo:
        remote_worker_actions.remote_worker_registry().recheck_at_claim(
            host_id, actor="tester", repository_id="repository-manager"
        )

    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, RemoteExecutionUnavailableError)
    if tunnel_manager_present:
        assert excinfo.value.__cause__.__cause__ is None
        assert "no configured inventory resolver" in str(excinfo.value.__cause__)
    else:
        assert excinfo.value.__cause__.__cause__ is not None
        assert isinstance(excinfo.value.__cause__.__cause__, ImportError)


def test_cli_argument_parser_accepts_dispatch_build_as_a_choice(tmp_path):
    """The real ``argparse`` layer, not the bypassed test ``Namespace``.

    ``_cli_result``/``run_remote_workers_cli`` above never exercise
    ``ArgumentParser.parse_args`` -- they construct the ``Namespace``
    directly, which is exactly why the CLI's ``choices=[...]`` list falling
    out of sync with ``REMOTE_WORKER_ACTIONS`` (missing ``dispatch_build``,
    the P0.7 action) went unnoticed until it was proven live against R820.
    A real subprocess invocation is the only way to prove ``argparse``
    itself accepts the choice: before the fix this exited 2 with
    "invalid choice: 'dispatch_build'" before ever reaching ``dispatch()``;
    after the fix it reaches the real refusal path instead (no fake tunnel
    is wired in a subprocess, and the host is unregistered either way, so
    it refuses honestly rather than attempting a real SSH call).
    """

    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["XDG_STATE_HOME"] = str(tmp_path)

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "repository_manager.repository_manager",
            "--remote-workers",
            "dispatch_build",
            "--remote-workers-params-json",
            json.dumps(
                {
                    "host_id": "host:does-not-exist",
                    "repository_id": "repository-manager",
                    "origin": "/tmp/origin",
                    "tree_sha": "c" * 40,
                    "command": ["echo", "hi"],
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert "invalid choice" not in completed.stderr, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["ok"] is False
    assert "unknown remote worker host" in payload["refused"]
    assert completed.returncode == 1


def test_import_mcp_server_without_tunnel_manager():
    """``import repository_manager.mcp_server`` never hard-depends on RMDD-14's seam.

    Run in a fresh subprocess with no inherited ``PYTHONPATH`` — this is the
    same proof pattern RMDD-15's own
    ``tests/test_remote_execution_optional_dependency.py`` uses.

    Asserts ``tunnel_manager.remote_execution`` specifically, not bare
    ``tunnel_manager``, and does not assert bare ``tunnel_manager``'s
    absence as a precondition: some environments have the *published*
    tunnel-manager package installed (for other packages' needs) without
    RMDD-14's unmerged ``remote_execution`` seam (see
    ``test_remote_execution_optional_dependency.py``'s
    ``test_base_import_succeeds_without_tunnel_manager`` for the identical
    reasoning). The claim this test actually needs to prove -- that
    ``mcp_server`` imports cleanly regardless -- does not depend on that
    seam's absence either.
    """

    import os

    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import repository_manager.mcp_server as m; "
            "assert 'tunnel_manager.remote_execution' not in __import__('sys').modules; "
            "print('ok')",
        ],
        capture_output=True,
        text=True,
        timeout=60,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
