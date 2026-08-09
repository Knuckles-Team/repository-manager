"""Git MCP adapter.

This module owns only the FastMCP schema and request marshalling for ``rm_git``.
All Git operations, background lifecycle, and path policy stay in their existing
application cores.
"""

from __future__ import annotations

import uuid
from typing import Any, cast

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_GIT_ACTIONS
from repository_manager.models import GitError, GitResult

logger = get_logger("RepositoryManagerServer")


def register_git_operations_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the behavior-preserving condensed Git tool."""

    adapter_context = context or from_server()

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
            description=(
                "Action: 'clone', 'enumerate', 'pull', 'push', 'phased_push', "
                "'add', 'commit', 'pre_commit', 'commit_code', 'status', or "
                "'cancel'. "
                "Use 'status' with job_id to poll any submitted rm_git job. "
                "Use 'cancel' with job_id to cancel a queued job; a running job "
                "is refused honestly because cooperative cancellation is not "
                "supported. "
                "Use 'commit_code', rather than separate add and commit jobs, "
                "for the ordered stage → gate → commit operation. Legacy 'raw' "
                "is permanently retired. 'enumerate' lists all repos across a "
                "GitLab instance/GitHub org into an ingest manifest "
                "(command=vcs, projects=groups/orgs)."
            )
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
        auto_start: bool = Field(
            default=True,
            description=(
                "For 'phased_push': begin at the lowest phase with unpushed work "
                "instead of always 'phase', skipping the inter-phase waits of "
                "unchanged upstream phases. Default True; set False to start at "
                "'phase'. Ignored when 'target_project' is set."
            ),
        ),
        projects: str | None = Field(
            default=None,
            description="Optional comma-separated list of repository URLs to clone or directory names/paths to pull/push/add/commit.",
        ),
        message: str | None = Field(
            default=None,
            description="Commit message for 'commit' / 'commit_code' actions.",
        ),
        run_precommit: bool = Field(
            default=True,
            description="For 'commit_code': run pre-commit hooks before committing. Default True.",
        ),
        job_id: str | None = Field(
            default=None,
            description="Background rm_git job id; required for action='status' or 'cancel'.",
        ),
        summary: bool = Field(
            default=True,
            description="For action='status': return compact progress by default; false includes full detail.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult | str | dict[str, Any]:
        """Typed bulk Git operations; arbitrary host commands are prohibited."""
        if not isinstance(auto_start, bool):
            auto_start = True

        resolved = resolve_action(action, RM_GIT_ACTIONS, service="repository-manager")
        if isinstance(resolved, dict):
            return resolved
        action = resolved

        if action == "status":
            if not job_id:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="job_id is required for 'status' action", code=1
                    ),
                )
            return adapter_context.get_job_status(job_id, summary=summary)

        if action == "cancel":
            if not job_id:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="job_id is required for 'cancel' action", code=1
                    ),
                )
            return adapter_context.cancel_job(job_id)

        git = adapter_context.get_git_instance(path=path, threads=threads)

        if action == "raw":
            del command
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=(
                        "Raw host commands are permanently retired; use a typed "
                        "repository-manager action through governed delegation."
                    ),
                    code=13,
                ),
            )

        if action == "enumerate":
            # Remote VCS enumeration for enterprise-scale ingestion.
            from repository_manager.kg_ingest import ingest_repositories
            from repository_manager.vcs_enumerator import (
                enumerate_github,
                enumerate_gitlab,
                write_manifest,
            )

            vcs = (command or "gitlab").strip().lower()
            scopes = (
                [s.strip() for s in projects.split(",") if s.strip()]
                if projects
                else None
            )
            run_id = uuid.uuid4().hex[:10]
            if vcs == "github":
                refs = await run_blocking(
                    enumerate_github, orgs=scopes, user=not scopes
                )
            else:
                refs = await run_blocking(enumerate_gitlab, groups=scopes)
            manifest_path = await run_blocking(write_manifest, refs, run_id)
            ingested = None
            try:
                ingested = await run_blocking(ingest_repositories, refs)
            except Exception as exc:  # noqa: BLE001 - ingestion is best-effort
                logger.debug("KG ingest skipped: %s", exc)
            return {
                "status": "ok",
                "vcs": vcs,
                "count": len(refs),
                "run_id": run_id,
                "manifest": manifest_path,
                "ingested": ingested,
            }

        if action == "clone":
            urls = None
            if projects:
                urls = [url.strip() for url in projects.split(",") if url.strip()]
            return adapter_context.submit_job(
                "clone", git.clone_projects, projects=urls
            )

        if action == "pull":
            pull_dirs: list[str] | None = None
            if projects:
                pull_dirs = []
                for project in projects.split(","):
                    project = project.strip()
                    if not project:
                        continue
                    pull_dirs.append(adapter_context.resolve_repo_dir(git, project))
            return adapter_context.submit_job(
                "pull", git.pull_projects, project_dirs=pull_dirs
            )

        if action == "push":
            push_dirs: list[str] | None = None
            if projects:
                push_dirs = []
                for project in projects.split(","):
                    project = project.strip()
                    if not project:
                        continue
                    push_dirs.append(adapter_context.resolve_repo_dir(git, project))
            return adapter_context.submit_job(
                "push", git.push_projects, project_dirs=push_dirs
            )

        if action == "add":
            add_dirs: list[str] | None = None
            if projects:
                add_dirs = []
                for project in projects.split(","):
                    project = project.strip()
                    if not project:
                        continue
                    add_dirs.append(adapter_context.resolve_repo_dir(git, project))
            return adapter_context.submit_job(
                "add", git.add_projects, project_dirs=add_dirs
            )

        if action == "commit":
            if not message:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="message is required for 'commit' action", code=1
                    ),
                )
            commit_dirs: list[str] | None = None
            if projects:
                commit_dirs = []
                for project in projects.split(","):
                    project = project.strip()
                    if not project:
                        continue
                    commit_dirs.append(adapter_context.resolve_repo_dir(git, project))
            return adapter_context.submit_job(
                "commit", git.commit_projects, message=message, project_dirs=commit_dirs
            )

        def resolve_dirs(spec: str | None) -> list[str] | None:
            if not spec:
                return None
            out: list[str] = []
            for project in spec.split(","):
                project = project.strip()
                if not project:
                    continue
                out.append(adapter_context.resolve_repo_dir(git, project))
            return out

        if action == "pre_commit":
            if not isinstance(run_precommit, bool):
                run_precommit = True
            pc_dirs = resolve_dirs(projects)
            return adapter_context.submit_job(
                "pre_commit", git.pre_commit_projects, projects=pc_dirs
            )

        if action == "commit_code":
            if not message:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="message is required for 'commit_code' action", code=1
                    ),
                )
            if not isinstance(run_precommit, bool):
                run_precommit = True
            commit_code_dirs = resolve_dirs(projects)
            if commit_code_dirs is None and path:
                target, error = adapter_context.resolve_commit_code_target(path)
                if error:
                    return GitResult(
                        status="error",
                        data="",
                        error=GitError(message=error, code=1),
                    )
                commit_code_dirs = [cast(str, target)]
            return adapter_context.submit_job(
                "commit_code",
                git.commit_code_projects,
                message=message,
                run_precommit=run_precommit,
                project_dirs=commit_code_dirs,
            )

        if action == "phased_push":
            progress = {
                "current_phase": "Initializing Pushes",
                "progress": 0,
                "phases": {},
            }
            return adapter_context.submit_job(
                "phased_push",
                git.phased_push,
                start_phase=phase or 1,
                project_filter=target_project,
                auto_start=auto_start,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )

        return f"Error: Unknown action '{action}'"
