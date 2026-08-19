"""MCP adapter for the bounded documentation-readiness action core."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager import docs_readiness
from repository_manager.mcp_tools.context import McpToolContext, from_server


def register_docs_readiness_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register one action-routed, dry-run-first readiness tool."""

    adapter_context = context or from_server()

    @mcp.tool(tags={"workspace_management", "project_manager", "documentation"})
    async def rm_docs_readiness(
        action: str = Field(
            default="preview",
            description=(
                "Action: 'preview' (default), 'apply', or 'verify'; per-repository "
                "readiness config must already exist."
            ),
        ),
        repository: str | None = Field(
            default=None,
            description=(
                "Exact workspace.yml repository identity (for example "
                "agent-packages/agents/repository-manager); basenames and paths "
                "outside the manifest are refused. Readiness config must already "
                "be generated/adopted; this action does not make a repo rollout-ready."
            ),
        ),
        workspace: str | None = Field(
            default=None,
            description="Workspace root; defaults to the configured RM workspace.",
        ),
        manifest: str | None = Field(
            default=None,
            description="Workspace manifest; defaults to workspace.yml under workspace.",
        ),
        confirm: bool = Field(
            default=False,
            description="Required for apply; preview and verify are read-only.",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for compatibility."
        ),
    ) -> dict[str, Any]:
        """Preview/apply/verify artifacts after per-repository readiness setup."""

        del ctx
        return await run_blocking(
            docs_readiness.dispatch,
            action,
            workspace_root=(
                workspace
                if workspace is not None
                else adapter_context.default_workspace
            ),
            manifest_path=(
                manifest
                if manifest is not None
                else adapter_context.default_workspace_yml
            ),
            repository=repository,
            confirm=confirm,
        )


__all__ = ["register_docs_readiness_tools"]
