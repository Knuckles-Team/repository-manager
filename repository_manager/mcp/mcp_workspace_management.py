"""MCP tools for workspace management operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from typing import Any

from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_server import _submit_job, get_git_instance
from repository_manager.models import GitResult, WorkspaceConfig


def register_workspace_management_tools(mcp: FastMCP):
    """Register tools for core workspace setup and organization."""

    @mcp.tool(tags={"workspace_management"})
    async def rm_workspace(
        action: str = Field(
            description="Action: 'list', 'list_branches', 'setup', 'template', 'save', 'maintain', 'remediate'"
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
        dry_run: bool = Field(
            default=False, description="Perform a dry run for 'maintain'."
        ),
        use_default: bool = Field(
            default=True,
            description="Use the pre-filled package template for 'template'.",
        ),
        repositories: str | None = Field(
            default=None,
            description="Comma-separated list of specific repositories to target for 'remediate'.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[str] | str | GitResult | dict:
        """Core workspace organization, configuration, and maintenance."""
        from repository_manager.models import GitError

        git = get_git_instance()

        if action == "list":
            return git.get_workspace_projects()

        if action == "list_branches":
            return git.list_branches()

        if action == "setup":
            if not yml_path:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="yml_path required for 'setup'", code=1),
                )
            return git.setup_from_yaml(yml_path)

        if action == "template":
            if not yml_path:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="yml_path required for 'template'", code=1),
                )
            return git.generate_workspace_template(
                target_path=yml_path, use_default=use_default
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
                return git.save_workspace_config(yaml_path=yml_path, config=config)
            except Exception as e:
                return GitResult(
                    status="error", data="", error=GitError(message=str(e), code=1)
                )

        if action == "maintain":
            progress = {
                "current_phase": "Initializing Bumps",
                "progress": 0,
                "phases": {},
            }
            return _submit_job(
                "maintain",
                git.maintain_projects,
                part=part,
                start_phase=phase,
                dry_run=dry_run,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )

        if action == "remediate":
            repos = []
            if repositories:
                repos = [r.strip() for r in repositories.split(",")]
            res = git.remediate_projects(repos)
            return f"Remediation complete. \nSuccess: {res['success']}\nErrors: {res['errors']}"

        return f"Error: Unknown action '{action}'"
