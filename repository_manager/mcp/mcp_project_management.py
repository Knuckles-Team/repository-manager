"""MCP tools for project management operations.

Auto-generated from mcp_server.py during ecosystem standardization.
"""

from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_server import (
    ValidationReport,
    _get_job_status,
    _jobs,
    _jobs_lock,
    _submit_job,
    get_git_instance,
)


def register_project_management_tools(mcp: FastMCP):
    """Register tools for the autonomous project harness."""

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_projects(
        action: str = Field(
            description="Action: 'install', 'build', 'validate', 'validate_status'"
        ),
        threads: int | None = Field(default=None, description="Parallel workers."),
        extra: str = Field(
            default="all", description="Install group (e.g. 'all') for 'install'."
        ),
        type: str = Field(
            default="all",
            description="Validation type: 'agent', 'mcp', or 'all' for 'validate'.",
        ),
        output_dir: str | None = Field(
            default=None,
            description="Directory to write the validation-reports for 'validate'.",
        ),
        generate_report: bool = Field(
            default=True,
            description="Generate validation report directory for 'validate'. Default True.",
        ),
        repositories: str | None = Field(
            default=None,
            description="Comma-separated list of specific repositories to target.",
        ),
        job_id: str | None = Field(
            default=None,
            description="Job ID to check status for 'validate_status' action.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str | ValidationReport | dict:
        """Bulk install, build, and validate Python projects.

        The 'validate' action submits validation as a background job and returns
        a job_id immediately.  Use 'validate_status' with that job_id to poll
        progress and retrieve results once complete.
        """
        git = get_git_instance(threads=threads)

        if repositories:
            repo_list = repositories.replace(" ", "").split(",")
            names_to_keep = set(repo_list)
            if git.project_map:
                filtered = {}
                for url, path in git.project_map.items():
                    name = url.split("/")[-1].replace(".git", "")
                    if name in names_to_keep:
                        filtered[url] = path
                git.project_map = filtered

        if action == "install":
            return _submit_job("install", git.install_projects, extra=extra)

        if action == "build":
            return _submit_job("build", git.build_projects)

        if action == "validate":
            with _jobs_lock:
                for existing_id, existing_job in _jobs.items():
                    if (
                        existing_job["action"] == "validate"
                        and existing_job["status"] == "running"
                    ):
                        return {
                            "status": "error",
                            "message": f"There is already another validation job in the queue with job_id '{existing_id}'.",
                        }

            repo_list_for_writer = (
                repositories.replace(" ", "").split(",") if repositories else None
            )
            # Shared progress dict — updated by validate_projects(), read by status poller
            progress = {"current_phase": "Initializing", "progress": 0, "phases": {}}
            return _submit_job(
                "validate",
                git.validate_projects,
                type=type,
                output_dir=output_dir,
                generate_report=generate_report,
                validated_repositories=repo_list_for_writer,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )

        if action == "validate_status":
            return _get_job_status(job_id)

        return f"Error: Unknown action '{action}'"
