"""Thin MCP adapter modules for Repository Manager.

The adapters in this package only define FastMCP registrations and marshal
validated arguments to the existing Repository Manager application cores.  The
server entry point keeps the job state and compatibility helpers; adapters
obtain those dependencies through :class:`McpToolContext` so tests and future
surfaces can exercise the same implementation without importing a giant module.
"""

from repository_manager.mcp_tools.build import register_build_tools
from repository_manager.mcp_tools.git import register_git_operations_tools
from repository_manager.mcp_tools.lane import register_lane_tools
from repository_manager.mcp_tools.merge_queue import register_merge_queue_tools
from repository_manager.mcp_tools.misc import register_misc_tools
from repository_manager.mcp_tools.projects import register_project_tools
from repository_manager.mcp_tools.registry import (
    MCP_TOOL_REGISTRY,
    extend_registry,
    register_project_management_tools,
)
from repository_manager.mcp_tools.workspace import register_workspace_management_tools
from repository_manager.mcp_tools.worktree import register_worktree_tools

__all__ = [
    "MCP_TOOL_REGISTRY",
    "extend_registry",
    "register_build_tools",
    "register_git_operations_tools",
    "register_lane_tools",
    "register_merge_queue_tools",
    "register_misc_tools",
    "register_project_tools",
    "register_project_management_tools",
    "register_workspace_management_tools",
    "register_worktree_tools",
]
