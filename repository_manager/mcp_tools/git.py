"""Git MCP adapter.

This module owns only the FastMCP schema and request marshalling for ``rm_git``.
All Git operations, background lifecycle, and path policy stay in their existing
application cores.
"""

from __future__ import annotations

import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
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


def _resolve_project_dirs(
    adapter_context: McpToolContext, git: Any, spec: str | None
) -> list[str] | None:
    """Split a comma-separated project spec into resolved repo directories."""
    if not spec:
        return None
    dirs: list[str] = []
    for project in spec.split(","):
        project = project.strip()
        if not project:
            continue
        dirs.append(adapter_context.resolve_repo_dir(git, project))
    return dirs


@dataclass
class PhasedPushArgs:
    """Parameters specific to the 'phased_push' action."""

    phase: int | None
    target_project: str | None
    auto_start: bool


@dataclass
class GitActionArgs:
    """Bundled request parameters shared across the dict-dispatched actions."""

    command: str | None = None
    path: str | None = None
    projects: str | None = None
    message: str | None = None
    run_precommit: bool = True
    phased: PhasedPushArgs | None = None


async def _handle_raw(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> GitResult:
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


async def _handle_enumerate(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> dict[str, Any]:
    # Remote VCS enumeration for enterprise-scale ingestion.
    from repository_manager.kg_ingest import ingest_repositories
    from repository_manager.vcs_enumerator import (
        enumerate_github,
        enumerate_gitlab,
        write_manifest,
    )

    vcs = (args.command or "gitlab").strip().lower()
    scopes = (
        [s.strip() for s in args.projects.split(",") if s.strip()]
        if args.projects
        else None
    )
    run_id = uuid.uuid4().hex[:10]
    if vcs == "github":
        refs = await run_blocking(enumerate_github, orgs=scopes, user=not scopes)
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


async def _handle_clone(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    urls = None
    if args.projects:
        urls = [url.strip() for url in args.projects.split(",") if url.strip()]
    return adapter_context.submit_job("clone", git.clone_projects, projects=urls)


async def _handle_pull(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    dirs = _resolve_project_dirs(adapter_context, git, args.projects)
    return adapter_context.submit_job("pull", git.pull_projects, project_dirs=dirs)


async def _handle_push(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    dirs = _resolve_project_dirs(adapter_context, git, args.projects)
    return adapter_context.submit_job("push", git.push_projects, project_dirs=dirs)


async def _handle_add(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    dirs = _resolve_project_dirs(adapter_context, git, args.projects)
    return adapter_context.submit_job("add", git.add_projects, project_dirs=dirs)


async def _handle_commit(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    if not args.message:
        return GitResult(
            status="error",
            data="",
            error=GitError(message="message is required for 'commit' action", code=1),
        )
    dirs = _resolve_project_dirs(adapter_context, git, args.projects)
    return adapter_context.submit_job(
        "commit", git.commit_projects, message=args.message, project_dirs=dirs
    )


async def _handle_pre_commit(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    run_precommit = args.run_precommit
    if not isinstance(run_precommit, bool):
        run_precommit = True
    dirs = _resolve_project_dirs(adapter_context, git, args.projects)
    return adapter_context.submit_job(
        "pre_commit", git.pre_commit_projects, projects=dirs
    )


async def _handle_commit_code(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    if not args.message:
        return GitResult(
            status="error",
            data="",
            error=GitError(
                message="message is required for 'commit_code' action", code=1
            ),
        )
    run_precommit = args.run_precommit
    if not isinstance(run_precommit, bool):
        run_precommit = True
    dirs = _resolve_project_dirs(adapter_context, git, args.projects)
    if dirs is None and args.path:
        target, error = adapter_context.resolve_commit_code_target(args.path)
        if error:
            return GitResult(
                status="error", data="", error=GitError(message=error, code=1)
            )
        dirs = [cast(str, target)]
    return adapter_context.submit_job(
        "commit_code",
        git.commit_code_projects,
        message=args.message,
        run_precommit=run_precommit,
        project_dirs=dirs,
    )


async def _handle_phased_push(
    adapter_context: McpToolContext, git: Any, args: GitActionArgs
) -> Any:
    phased = args.phased or PhasedPushArgs(
        phase=1, target_project=None, auto_start=True
    )
    progress: dict[str, Any] = {
        "current_phase": "Initializing Pushes",
        "progress": 0,
        "phases": {},
    }
    return adapter_context.submit_job(
        "phased_push",
        git.phased_push,
        start_phase=phased.phase or 1,
        project_filter=phased.target_project,
        auto_start=phased.auto_start,
        progress=progress,
        _extra_job_data={"progress_detail": progress},
    )


_ACTION_HANDLERS: dict[
    str, Callable[[McpToolContext, Any, GitActionArgs], Awaitable[Any]]
] = {
    "raw": _handle_raw,
    "enumerate": _handle_enumerate,
    "clone": _handle_clone,
    "pull": _handle_pull,
    "push": _handle_push,
    "add": _handle_add,
    "commit": _handle_commit,
    "pre_commit": _handle_pre_commit,
    "commit_code": _handle_commit_code,
    "phased_push": _handle_phased_push,
}


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

        args = GitActionArgs(
            command=command,
            path=path,
            projects=projects,
            message=message,
            run_precommit=run_precommit,
            phased=PhasedPushArgs(
                phase=phase, target_project=target_project, auto_start=auto_start
            ),
        )
        handler = _ACTION_HANDLERS.get(action)
        if handler is not None:
            return await handler(adapter_context, git, args)

        return f"Error: Unknown action '{action}'"
