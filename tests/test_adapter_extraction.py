"""Regression contracts for the RMDD-04 MCP and CLI adapter extraction."""

from __future__ import annotations

import asyncio
import hashlib
import json

from repository_manager.mcp_server import get_mcp_instance
from repository_manager.mcp_tools import MCP_TOOL_REGISTRY, extend_registry

EXPECTED_REGISTRY = (
    ("git_operations", "GIT_OPERATIONSTOOL"),
    ("misc", "MISCTOOL"),
    ("project_management", "PROJECT_MANAGEMENTTOOL"),
    ("workspace_management", "WORKSPACE_MANAGEMENTTOOL"),
)
EXPECTED_TOOL_NAMES = (
    "rm_git",
    "repository_ingest_repositories",
    "rm_lane",
    "rm_worktree",
    "rm_merge_queue",
    "rm_build",
    "rm_projects",
    "rm_workspace",
)

# SHA-256 of the ordered, normalized FastMCP catalog on the pre-extraction
# baseline.  This covers names, descriptions, tags, parameter schemas,
# defaults, and required fields without duplicating a 21-KB fixture here.
BASELINE_CATALOG_SHA256 = (
    "8b810e50a6749ead3f75ea9be23a32bec64a22b82544ccb6ec8d8818a1db5b38"
)


def _catalog_payload(tools: list) -> list[dict]:
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.parameters,
            "tags": sorted(tool.tags or []),
        }
        for tool in tools
    ]


def test_registry_order_and_extension_seam_are_deterministic() -> None:
    assert tuple((tag, env) for tag, env, _ in MCP_TOOL_REGISTRY) == EXPECTED_REGISTRY
    assert all(callable(registrar) for _, _, registrar in MCP_TOOL_REGISTRY)

    sentinel = ("future_domain", "FUTURE_DOMAINTOOL", lambda _mcp: None)
    extended = extend_registry(sentinel)
    assert tuple((tag, env) for tag, env, _ in extended) == (
        *EXPECTED_REGISTRY,
        ("future_domain", "FUTURE_DOMAINTOOL"),
    )
    assert len(MCP_TOOL_REGISTRY) == len(EXPECTED_REGISTRY)


def test_cli_entrypoint_remains_a_compatibility_shim() -> None:
    """The packaged CLI entrypoint delegates to the extracted parser module."""
    import repository_manager.repository_manager as repository_manager_module

    assert repository_manager_module.main.__module__ == (
        "repository_manager.repository_manager"
    )
    assert repository_manager_module._run_cli.__module__ == (
        "repository_manager.cli_commands.parser"
    )


def test_condensed_mcp_catalog_matches_pre_extraction_contract() -> None:
    mcp, _args, _middlewares, _registered_tags = get_mcp_instance()
    tools = asyncio.run(mcp.list_tools())
    payload = _catalog_payload(tools)
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert tuple(tool["name"] for tool in payload) == EXPECTED_TOOL_NAMES
    assert digest == BASELINE_CATALOG_SHA256
