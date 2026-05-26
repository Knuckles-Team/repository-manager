"""MCP tools for git operations operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

import os

from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_server import _submit_job, get_git_instance
from repository_manager.models import GitResult


def register_git_operations_tools(mcp: FastMCP):
    @mcp.tool(
        tags={
            "workspace_management",
            "project_manager",
            "devops_engineer",
            "git_operations",
        }
    )
    async def rm_git(
        action: str = Field(
            description="Action: 'raw', 'clone', 'pull', 'push', 'phased_push'"
        ),
        command: str | None = Field(
            default=None,
            description="The Git command to execute for 'raw' action (e.g., 'git status')",
        ),
        path: str | None = Field(default=None, description="Path to execute in."),
        threads: int | None = Field(
            default=None, description="Parallel workers for bulk operations."
        ),
        phase: int | None = Field(
            default=1, description="Starting phase number for 'phased_push'. Default 1."
        ),
        target_project: str | None = Field(
            default=None,
            description="Optional specific project to push for 'phased_push'.",
        ),
        projects: str | None = Field(
            default=None,
            description="Optional comma-separated list of repository URLs to clone or directory names/paths to pull/push.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult | str | dict:
        """Bulk Git operations and arbitrary command execution."""
        from repository_manager.models import GitError

        git = get_git_instance(path=path, threads=threads)

        if action == "raw":
            if not command:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="command is required for 'raw' action", code=1
                    ),
                )
            return git.git_action(command=command, path=path)

        if action == "clone":
            urls = None
            if projects:
                urls = [url.strip() for url in projects.split(",") if url.strip()]
            return _submit_job("clone", git.clone_projects, projects=urls)

        if action == "pull":
            dirs = None
            if projects:
                dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if os.path.isabs(p):
                        dirs.append(p)
                    else:
                        dirs.append(os.path.abspath(os.path.join(git.path, p)))
            return _submit_job("pull", git.pull_projects, project_dirs=dirs)

        if action == "push":
            dirs = None
            if projects:
                dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if os.path.isabs(p):
                        dirs.append(p)
                    else:
                        dirs.append(os.path.abspath(os.path.join(git.path, p)))
            return _submit_job("push", git.push_projects, project_dirs=dirs)

        if action == "phased_push":
            progress = {
                "current_phase": "Initializing Pushes",
                "progress": 0,
                "phases": {},
            }
            return _submit_job(
                "phased_push",
                git.phased_push,
                start_phase=phase or 1,
                project_filter=target_project,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )

        return f"Error: Unknown action '{action}'"
