"""RMDD-20: MCP/CLI parity and refusal proofs for the exposed concept surface.

``rm_concepts`` (MCP) and ``--concepts`` (CLI) must invoke the identical
:func:`repository_manager.concept_actions.dispatch` application service —
these tests prove that with a *behavioral* comparison (same input, same
output through both adapters), not by inspection, per the lane's "MCP/CLI
parity is the point of this lane" requirement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

import pytest

from repository_manager.cli_commands.concepts import run_concepts_cli
from repository_manager.concept_coordination import ConceptAuthorityUnavailable
from repository_manager.mcp_server import get_mcp_instance


async def _rm_concepts_tool():
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    return next(t for t in tools if t.name == "rm_concepts")


def _cli_result(capsys, **kwargs) -> tuple[dict, int]:
    defaults = {
        "concepts": None,
        "concepts_repo_root": ".",
        "concepts_tenant_ref": "",
        "concepts_lane_ref": "",
        "concepts_params_json": "",
    }
    defaults.update(kwargs)
    args = argparse.Namespace(**defaults)
    exit_code = run_concepts_cli(args)
    captured = capsys.readouterr()
    return json.loads(captured.out), exit_code


@pytest.mark.anyio
async def test_mcp_and_cli_reserve_refusal_are_identical(tmp_path, capsys):
    """Same reserve request through MCP and CLI -> the identical refusal.

    The concept authority (RMDD-16) is absent in this environment, so both
    adapters must refuse identically instead of one crashing and the other
    refusing, or the two disagreeing on the refusal shape.
    """

    tool = await _rm_concepts_tool()
    # `.fn(...)` calls the raw Python function directly, bypassing FastMCP's
    # own pydantic argument binding (which is what resolves each unset
    # `Field(default=...)` to its real default over the wire) -- so every
    # parameter is passed explicitly here to mirror what a real MCP call
    # produces. See `RM_MCP_OPTIONAL_KWARGS` below for the shared value set
    # `_cli_result` also uses, so both paths see identical input.
    mcp_result = await tool.fn(
        action="reserve",
        repo_root=str(tmp_path),
        tenant_ref="tenant:parity",
        lane_ref="lane:parity",
        reservation_id=None,
        owner_ref="owner:parity",
        expected_fence=None,
        concept_id="PARITY.foo",
        namespace="PARITY",
        repository_ref="repo:parity",
        request_key_ref="req:parity-1",
        purpose="parity test",
        design_ref=None,
        branch=None,
        base_sha=None,
        workitem_ref=None,
        run_trace_ref=None,
        provenance_refs=None,
        state=None,
        concept_prefix=None,
        limit=1000,
        cursor=None,
        candidate=None,
        generation=None,
        candidates=None,
        repo_path=None,
        source_tree_ish="HEAD",
        ctx=None,
    )

    params_json = json.dumps(
        {
            "concept_id": "PARITY.foo",
            "namespace": "PARITY",
            "repository_ref": "repo:parity",
            "owner_ref": "owner:parity",
            "request_key_ref": "req:parity-1",
            "purpose": "parity test",
        }
    )
    cli_result, exit_code = _cli_result(
        capsys,
        concepts="reserve",
        concepts_repo_root=str(tmp_path),
        concepts_tenant_ref="tenant:parity",
        concepts_lane_ref="lane:parity",
        concepts_params_json=params_json,
    )

    assert mcp_result == cli_result
    assert mcp_result["ok"] is False
    assert mcp_result["error_code"] == "dependency_blocked"
    assert exit_code == 1


@pytest.mark.anyio
async def test_mcp_and_cli_list_refusal_are_identical(tmp_path, capsys):
    tool = await _rm_concepts_tool()
    mcp_result = await tool.fn(
        action="list",
        repo_root=str(tmp_path),
        tenant_ref="tenant:parity",
        lane_ref="lane:parity",
        reservation_id=None,
        owner_ref=None,
        expected_fence=None,
        concept_id=None,
        namespace=None,
        repository_ref=None,
        request_key_ref=None,
        purpose=None,
        design_ref=None,
        branch=None,
        base_sha=None,
        workitem_ref=None,
        run_trace_ref=None,
        provenance_refs=None,
        state=None,
        concept_prefix=None,
        limit=1000,
        cursor=None,
        candidate=None,
        generation=None,
        candidates=None,
        repo_path=None,
        source_tree_ish="HEAD",
        ctx=None,
    )
    cli_result, exit_code = _cli_result(
        capsys,
        concepts="list",
        concepts_repo_root=str(tmp_path),
        concepts_tenant_ref="tenant:parity",
        concepts_lane_ref="lane:parity",
    )
    assert mcp_result == cli_result
    assert mcp_result["ok"] is False
    assert exit_code == 1


def test_concept_authority_refusal_names_module_and_preserves_cause(tmp_path):
    """Refusal proof: authority absent -> named refusal, cause preserved (H-12).

    Verified directly against ``agent_utilities.governance.concept_reservation``
    absent in this environment (repository-manager AGENTS.md optional
    dependency guardrail; confirmed absent by this same test run).
    """

    from repository_manager.concept_coordination import ConceptCoordinationActions

    actions = ConceptCoordinationActions(
        repo_root=tmp_path, tenant_ref="t", lane_ref="l"
    )
    with pytest.raises(ConceptAuthorityUnavailable) as excinfo:
        actions.get("concept-reservation:t:1")

    assert "agent_utilities.governance.concept_reservation" in str(excinfo.value)
    assert excinfo.value.__cause__ is not None
    assert isinstance(excinfo.value.__cause__, ImportError)


def test_import_mcp_server_without_concept_reservation_authority():
    """``import repository_manager.mcp_server`` succeeds with the authority absent.

    Run in a fresh subprocess to prove the *base install* imports cleanly —
    a module-level import of ``agent_utilities.governance.concept_reservation``
    anywhere in the package would abort collection, exactly the RMDD-19
    revert class this lane must not repeat.
    """

    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import agent_utilities.governance.concept_reservation as m; "
            "raise SystemExit('concept_reservation unexpectedly present: ' + m.__file__)",
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode != 0, (
        "expected agent_utilities.governance.concept_reservation to be absent "
        "in this environment; the RMDD-20 refusal proof requires it"
    )
    assert "No module named" in completed.stderr

    completed = subprocess.run(
        [sys.executable, "-c", "import repository_manager.mcp_server; print('ok')"],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stderr
    assert "ok" in completed.stdout
