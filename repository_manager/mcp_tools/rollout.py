"""Dependency-rollout MCP adapter (RM-RELEASE-R009)."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastmcp import FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext


def register_rollout_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register ``rm_rollout``: the same core as ``repository-manager rollout``."""

    del context

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_rollout(
        action: str = Field(
            description=(
                "'plan' (select + read floors/PRs, write ledger), 'apply' (plan, then "
                "open one auto-merging PR per repo), 'status' (merged/failing/pending), "
                "'release' (tag v<version> on main after merge + floor met)."
            )
        ),
        dependency: str = Field(description="Dependency to roll, e.g. 'scion-sdk'."),
        floor: str = Field(description="New floor MAJOR.MINOR.PATCH, e.g. '0.6.4'."),
        extra: str | None = Field(
            default=None, description="Ensure this extra on the requirement."
        ),
        group: str = Field(
            default="agent-connectors",
            description="workspace.yml selector name or subdirectory prefix.",
        ),
        only: str | None = Field(
            default=None, description="Comma-separated repo names."
        ),
        jobs: int = Field(default=6, description="Bounded parallelism."),
        dry_run: bool = Field(
            default=True, description="Plan/apply without writing anything."
        ),
        manifest: str | None = Field(default=None, description="workspace.yml path."),
    ) -> dict[str, Any]:
        from repository_manager.rollout import run_rollout_action

        return await asyncio.to_thread(
            run_rollout_action,
            action,
            dependency=dependency,
            floor=floor,
            manifest_path=Path(manifest) if manifest else None,
            extra=extra,
            group=group,
            only=tuple(x.strip() for x in (only or "").split(",") if x.strip()),
            jobs=jobs,
            dry_run=dry_run,
        )
