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

import pytest

from repository_manager.cli_commands.remote_workers import run_remote_workers_cli
from repository_manager.mcp_server import get_mcp_instance
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

    from repository_manager.remote_execution import RemoteWorkerRegistryError

    from repository_manager import remote_worker_actions

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
