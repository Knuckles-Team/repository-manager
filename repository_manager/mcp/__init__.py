"""MCP tool registration modules for repository-manager.

Auto-generated during ecosystem standardization.
Each domain has its own module with a register_*_tools function.
"""

from repository_manager.mcp.mcp_git_operations import register_git_operations_tools
from repository_manager.mcp.mcp_misc import register_misc_tools
from repository_manager.mcp.mcp_project_management import (
    register_project_management_tools,
)
from repository_manager.mcp.mcp_workspace_management import (
    register_workspace_management_tools,
)

__all__ = [
    "register_git_operations_tools",
    "register_misc_tools",
    "register_project_management_tools",
    "register_workspace_management_tools",
]
