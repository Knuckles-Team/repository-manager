"""Workspace-management MCP adapter."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_WORKSPACE_ACTIONS
from repository_manager.models import GitError, GitResult, WorkspaceConfig


def register_workspace_management_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register tools for core workspace setup and organization."""

    adapter_context = context or from_server()

    @mcp.tool(tags={"workspace_management"})
    async def rm_workspace(
        action: str = Field(
            description="Action: 'list', 'list_branches', 'setup', 'template', 'save', 'maintain', 'maintain_status'"
        ),
        yml_path: str | None = Field(
            default=None,
            description="Path to workspace.yml (for 'setup', 'template', 'save').",
        ),
        config_dict: dict[str, Any] | None = Field(
            default=None,
            description="Dictionary representation of WorkspaceConfig (for 'save').",
        ),
        part: str = Field(
            default="patch",
            description="Version part to bump for 'maintain' (major, minor, patch).",
        ),
        phase: int = Field(
            default=1, description="Starting phase number for 'maintain'."
        ),
        auto_start: bool = Field(
            default=True,
            description=(
                "For 'maintain': begin at the lowest phase with repository changes "
                "instead of always 'phase', cascading to every later phase and "
                "skipping unchanged upstream phases. Default True; set False to "
                "start at 'phase'. Ignored when 'projects' or 'force' is set."
            ),
        ),
        dry_run: bool = Field(
            default=False, description="Perform a dry run for 'maintain'."
        ),
        projects: str | None = Field(
            default=None,
            description=(
                "For 'maintain': comma-separated repo names to bump ONLY those "
                "(e.g. re-bump repos a prior run skipped) instead of the whole "
                "topological set. Restricts the bulk phase to these names."
            ),
        ),
        force: bool = Field(
            default=False,
            description=(
                "For 'maintain': bump even when no code changes are detected, and "
                "override an orphan local 'next-version' tag (delete it and "
                "re-bump) — only if that tag is NOT on the remote."
            ),
        ),
        use_default: bool = Field(
            default=True,
            description="Use the pre-filled package template for 'template'.",
        ),
        job_id: str | None = Field(
            default=None,
            description="Job ID to check status for 'maintain_status' action.",
        ),
        summary: bool = Field(
            default=True,
            description=(
                "For 'maintain_status': return a COMPACT roll-up (per-phase "
                "counts + failed set + active names + remaining count) instead of "
                "the full per-repo phase dump. Keeps the response inline at 200+ "
                "repos. Set False for the full per-repo detail."
            ),
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[str] | str | GitResult | dict[str, Any]:
        """Core workspace organization, configuration, and maintenance."""
        if not isinstance(summary, bool):
            summary = True
        if not isinstance(force, bool):
            force = False
        if not isinstance(auto_start, bool):
            auto_start = True

        git = adapter_context.get_git_instance()

        resolved = resolve_action(
            action, RM_WORKSPACE_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "list":
            return git.get_workspace_projects()

        if action == "list_branches":
            return await run_blocking(git.list_branches)

        if action == "setup":
            if not yml_path:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="yml_path required for 'setup'", code=1),
                )
            return await run_blocking(git.setup_from_yaml, yml_path)

        if action == "template":
            if not yml_path:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="yml_path required for 'template'", code=1),
                )
            return await run_blocking(
                git.generate_workspace_template,
                target_path=yml_path,
                use_default=use_default,
            )

        if action == "save":
            if not yml_path or not config_dict:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="yml_path and config_dict required for 'save'", code=1
                    ),
                )
            try:
                config = WorkspaceConfig(**config_dict)
                return await run_blocking(
                    git.save_workspace_config, yaml_path=yml_path, config=config
                )
            except Exception:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="Repository operation failed", code=1),
                )

        if action == "maintain":
            progress = {
                "current_phase": "Initializing Bumps",
                "progress": 0,
                "phases": {},
            }
            return adapter_context.submit_job(
                "maintain",
                git.maintain_projects,
                part=part,
                start_phase=phase,
                auto_start=auto_start,
                dry_run=dry_run,
                project_filter=projects or None,
                force=force,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )

        if action == "maintain_status":
            if not job_id:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="job_id required for 'maintain_status'", code=1
                    ),
                )
            return adapter_context.get_job_status(job_id, summary=summary)

        return f"Error: Unknown action '{action}'"
