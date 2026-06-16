"""Action-discovery contract for the action-routed MCP tools.

Each action-routed ``rm_*`` tool resolves its incoming ``action`` through the
shared ``agent_utilities.mcp_utilities.resolve_action`` helper against a module
-level canonical action set. That gives every tool ``list_actions`` discovery,
plural->singular aliases, and a rich did-you-mean error on unknown actions.
"""

from unittest.mock import MagicMock

import pytest
from agent_utilities.mcp_utilities import resolve_action

import repository_manager.mcp_server as mcp_server

ACTION_SETS = {
    "rm_git": mcp_server.RM_GIT_ACTIONS,
    "rm_workspace": mcp_server.RM_WORKSPACE_ACTIONS,
    "rm_worktree": mcp_server.RM_WORKTREE_ACTIONS,
    "rm_projects": mcp_server.RM_PROJECTS_ACTIONS,
}


@pytest.mark.parametrize("name,actions", list(ACTION_SETS.items()))
def test_list_actions_returns_names(name, actions):
    payload = resolve_action("list_actions", actions, service="repository-manager")
    assert isinstance(payload, dict)
    assert payload["service"] == "repository-manager"
    assert set(payload["actions"]) == set(actions)


@pytest.mark.parametrize("keyword", ["list_actions", "help", "actions"])
def test_discovery_keywords(keyword):
    payload = resolve_action(
        keyword, mcp_server.RM_GIT_ACTIONS, service="repository-manager"
    )
    assert isinstance(payload, dict)
    assert payload["actions"]


@pytest.mark.parametrize("name,actions", list(ACTION_SETS.items()))
def test_bogus_action_raises_with_list_actions_hint(name, actions):
    with pytest.raises(ValueError) as exc:
        resolve_action(
            "definitely_not_a_real_action", actions, service="repository-manager"
        )
    assert "list_actions" in str(exc.value)


def test_plural_alias_resolves_to_singular():
    # 'list' is a real action on rm_worktree; its plural should normalize.
    assert (
        resolve_action(
            "lists", mcp_server.RM_WORKTREE_ACTIONS, service="repository-manager"
        )
        == "list"
    )


def test_canonical_action_passes_through():
    assert (
        resolve_action(
            "validate", mcp_server.RM_PROJECTS_ACTIONS, service="repository-manager"
        )
        == "validate"
    )


def test_magicmock_client_is_not_required():
    # The helper is pure over the action set; a stand-in client is never touched.
    client = MagicMock()
    resolve_action(
        "install", mcp_server.RM_PROJECTS_ACTIONS, service="repository-manager"
    )
    client.assert_not_called()
