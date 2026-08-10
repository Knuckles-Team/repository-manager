"""Deterministic MCP adapter registry.

The order is part of the adapter contract: ``register_tool_surface`` applies
the per-domain environment toggles in this order and the parity tests snapshot
the resulting condensed catalog.  New development tools should add a new
registrar here rather than editing ``mcp_server.py``.
"""

from __future__ import annotations

from typing import Any

from repository_manager.mcp_tools.context import from_server


def register_git_operations_tools(mcp: Any) -> None:
    from repository_manager.mcp_tools.git import (
        register_git_operations_tools as register,
    )

    register(mcp, context=from_server())


def register_misc_tools(mcp: Any) -> None:
    from repository_manager.mcp_tools.misc import register_misc_tools as register

    register(mcp, context=from_server())


def register_project_management_tools(mcp: Any) -> None:
    """Register the stable project-management domain in legacy order."""
    from repository_manager.mcp_tools.build import register_build_tools
    from repository_manager.mcp_tools.lane import register_lane_tools
    from repository_manager.mcp_tools.merge_queue import register_merge_queue_tools
    from repository_manager.mcp_tools.projects import register_project_tools
    from repository_manager.mcp_tools.worktree import register_worktree_tools

    context = from_server()
    register_lane_tools(mcp, context=context)
    register_worktree_tools(mcp, context=context)
    register_merge_queue_tools(mcp, context=context)
    register_build_tools(mcp, context=context)
    register_project_tools(mcp, context=context)


def register_development_surface_tools(mcp: Any) -> None:
    """Register RMDD-20's newest exposed development surfaces.

    A dedicated registrar (rather than folding into
    ``register_project_management_tools``) so the two lanes RMDD-20 exposes
    (RMDD-15 remote execution, RMDD-17 concept coordination) each land as
    one deterministic, independently toggleable entry.
    """
    from repository_manager.mcp_tools.concepts import register_concepts_tools
    from repository_manager.mcp_tools.remote_workers import (
        register_remote_workers_tools,
    )

    context = from_server()
    register_concepts_tools(mcp, context=context)
    register_remote_workers_tools(mcp, context=context)


def register_workspace_management_tools(mcp: Any) -> None:
    from repository_manager.mcp_tools.workspace import (
        register_workspace_management_tools as register,
    )

    register(mcp, context=from_server())


MCP_TOOL_REGISTRY: tuple[tuple[str, str, Any], ...] = (
    ("git_operations", "GIT_OPERATIONSTOOL", register_git_operations_tools),
    ("misc", "MISCTOOL", register_misc_tools),
    ("project_management", "PROJECT_MANAGEMENTTOOL", register_project_management_tools),
    (
        "workspace_management",
        "WORKSPACE_MANAGEMENTTOOL",
        register_workspace_management_tools,
    ),
    (
        "development_surfaces",
        "DEVELOPMENT_SURFACESTOOL",
        register_development_surface_tools,
    ),
)


def extend_registry(
    *entries: tuple[str, str, Any],
) -> tuple[tuple[str, str, Any], ...]:
    """Return a deterministic registry with additional adapter entries.

    Future MCP domains should compose their registrar into this seam rather
    than modifying ``mcp_server.py`` or relying on module-name discovery.  The
    built-in order and environment gates remain unchanged; callers own the
    ordering of their appended entries and can pass the result to
    ``register_tool_surface(tool_registry=...)``.
    """

    return MCP_TOOL_REGISTRY + tuple(entries)
