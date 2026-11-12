"""Destructive actions are declared with the SDK and ask before they run."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from repository_manager.mcp_server import get_mcp_instance

DESTRUCTIVE_TOOLS = (
    "rm_git_cancel",
    "rm_worktree_remove",
    "rm_worktree_reset_branch",
    "rm_worktree_prune",
)


def _server():
    with patch.object(sys, "argv", ["repository-manager"]):
        return [x for x in get_mcp_instance() if hasattr(x, "list_tools")][0]


async def _tools():
    server = _server()
    return {tool.name: tool for tool in await server.list_tools()}


@pytest.mark.asyncio
async def test_destructive_actions_are_declared_and_take_confirm():
    tools = await _tools()
    for name in DESTRUCTIVE_TOOLS:
        properties = tools[name].parameters["properties"]
        assert tools[name].annotations.destructive_hint is True, name
        assert "confirm" in properties, name
        assert not [key for key in properties if key.startswith("allow_")], name
    assert tools["rm_worktree_list"].annotations.destructive_hint is False


def test_consent_policy_has_no_destructive_allow_flag():
    from repository_manager.development import ConsentPolicy

    assert "allow_destructive_cleanup" not in ConsentPolicy.model_fields


@pytest.mark.asyncio
async def test_worktree_remove_refuses_without_confirmation(monkeypatch):
    from repository_manager.mcp_tools import worktree

    called = []

    async def refuse(ctx, operation, arguments=None, *, confirm=False):
        called.append((operation, confirm))
        return {"cancelled": True, "operation": operation}

    monkeypatch.setattr("scion_sdk.mcp.destructive.guard_destructive", refuse)
    server = _server()
    tool = await server.get_tool("rm_worktree_remove")
    result = await tool.fn(repo="r", branch="b")
    assert result == {"cancelled": True, "operation": "remove"}
    assert called == [("remove", False)]
    assert worktree is not None
