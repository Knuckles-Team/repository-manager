"""Workspace project-management MCP adapter."""

from __future__ import annotations

import os
from typing import Any

from fastmcp import Context, FastMCP
from pydantic import Field
from scion_sdk.mcp.action_dispatch import resolve_action

from repository_manager.development.fleet_release_gate import (
    require_legacy_release_route,
)
from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_PROJECTS_ACTIONS
from repository_manager.scan_models import RepoScanResult


def _require_project_release_route(action: str) -> None:
    if action in {"install", "build"}:
        require_legacy_release_route(action)


def _normalize_rm_projects_flags(
    failed_only: bool,
    summary: bool,
    force_revalidate: bool,
    commit_code: bool,
) -> tuple[bool, bool, bool, bool]:
    """Coerce each MCP-boundary flag to its documented default when a caller
    passed a non-bool value. Extracted verbatim from ``rm_projects``'s
    preamble — same defaults, same order.
    """
    if not isinstance(failed_only, bool):
        failed_only = False
    if not isinstance(summary, bool):
        summary = True
    if not isinstance(force_revalidate, bool):
        force_revalidate = False
    if not isinstance(commit_code, bool):
        commit_code = False
    return failed_only, summary, force_revalidate, commit_code


def _resolve_failed_only_repositories(
    adapter_context: McpToolContext,
    action: str,
    failed_only: bool,
    repositories: str | None,
) -> tuple[str | None, bool, dict[str, Any] | None]:
    """Apply the 'validate' remediation loop's failed_only targeting.

    Extracted verbatim from ``rm_projects``. Returns
    ``(repositories, force_revalidate_override, early_return_payload)``: the
    caller must return ``early_return_payload`` immediately when it is not
    ``None`` (the 'clean, nothing to re-validate' short-circuit); otherwise
    it must set ``force_revalidate = True`` when the override is truthy.
    """
    if action != "validate" or not failed_only or repositories:
        return repositories, False, None
    failed_repositories = adapter_context.last_failed_repos()
    if not failed_repositories:
        return (
            repositories,
            False,
            {
                "status": "clean",
                "message": "No previously-failed projects to re-validate.",
                "queued_count": 0,
            },
        )
    return ",".join(failed_repositories), True, None


def _filter_project_map_by_repositories(git: Any, repositories: str | None) -> None:
    """Narrow ``git.project_map`` to the comma-separated ``repositories``
    allowlist, in place. Extracted verbatim from ``rm_projects``.
    """
    if not repositories:
        return
    repo_list = repositories.replace(" ", "").split(",")
    names_to_keep = set(repo_list)
    if git.project_map:
        filtered = {}
        for url, path in git.project_map.items():
            name = url.split("/")[-1].replace(".git", "")
            if name in names_to_keep:
                filtered[url] = path
        git.project_map = filtered


def _compute_validate_targets(
    git: Any, repositories: str | None
) -> list[tuple[str, str]]:
    """Resolve the (repo_name, path) pairs 'validate' targets.

    Extracted verbatim from ``rm_projects``'s 'validate' branch — a second,
    redundant-but-harmless filter against ``repositories`` on top of
    whatever ``_filter_project_map_by_repositories`` already narrowed
    ``git.project_map`` to.
    """
    repo_list_for_writer = (
        repositories.replace(" ", "").split(",") if repositories else None
    )
    targets = []
    for _url, path in git.project_map.items():
        repo_name = os.path.basename(path)
        if repo_list_for_writer and repo_name not in repo_list_for_writer:
            continue
        targets.append((repo_name, path))
    return targets


def _index_running_and_completed_validate_jobs(
    jobs: dict[str, dict[str, Any]],
) -> tuple[dict[str, str], dict[str, tuple[str, Any]]]:
    """Index the background job table by repo for the most recent 'validate'
    job per repo, split into in-flight vs completed. Extracted verbatim from
    ``rm_projects``'s job-scan loop.
    """
    running_by_repo: dict[str, str] = {}
    completed_by_repo: dict[str, tuple[str, Any]] = {}
    for existing_id, job in jobs.items():
        if job.get("action") != "validate":
            continue
        repo_name = job.get("repo_name")
        if not repo_name:
            continue
        if job["status"] in ("running", "queued", "pending"):
            running_by_repo[repo_name] = existing_id
        elif job["status"] == "completed":
            completed_by_repo[repo_name] = (existing_id, job.get("result"))
    return running_by_repo, completed_by_repo


def _classify_existing_validate_job(
    repo_name: str,
    running_by_repo: dict[str, str],
    completed_by_repo: dict[str, tuple[str, Any]],
) -> tuple[str | None, str | None, Any]:
    """Return (status, job_id, job_result) for one repo's most recent
    'validate' job, or (None, None, None) if it has none. Extracted verbatim
    from ``rm_projects``'s per-target classification.
    """
    existing_job_id = running_by_repo.get(repo_name)
    if existing_job_id is not None:
        return "running", existing_job_id, None
    if repo_name in completed_by_repo:
        existing_job_id, existing_job_result = completed_by_repo[repo_name]
        return "completed", existing_job_id, existing_job_result
    return None, None, None


def _hook_failures(hooks: Any) -> list[str]:
    failures: list[str] = []
    for hook in hooks:
        if not getattr(hook, "passed", True):
            output = getattr(hook, "output", "").strip()
            failures.append(
                f"Hook '{hook.hook_id}' failed: {output}"
                if output
                else f"Hook '{hook.hook_id}' failed."
            )
    return failures


def _build_cache_summary(existing_job_result: Any) -> dict[str, Any]:
    """Derive the compact pass/fail cache summary for a completed validate
    job's cached result. Extracted verbatim from ``rm_projects``.
    """
    cache_summary: dict[str, Any] = {"passed": False, "failures": []}
    if not existing_job_result:
        return cache_summary
    if hasattr(existing_job_result, "success"):
        cache_summary["passed"] = existing_job_result.success
    if hasattr(existing_job_result, "hooks"):
        cache_summary["failures"] = _hook_failures(existing_job_result.hooks)
    if hasattr(existing_job_result, "error") and existing_job_result.error:
        cache_summary["failures"].append(existing_job_result.error)
    return cache_summary


def _scan_existing_validate_jobs(
    adapter_context: McpToolContext,
    targets: list[tuple[str, str]],
    force_revalidate: bool,
) -> dict[str, Any]:
    """Classify each target repo's most recent 'validate' job (if any) into
    the running/completed(cache-hit) buckets of a fresh result_payload.
    Extracted verbatim from ``rm_projects`` — same lock scope (held only for
    this scan, released before any job submission).
    """
    result_payload: dict[str, Any] = {"queued": {}, "running": {}, "completed": {}}
    with adapter_context.jobs_lock:
        running_by_repo, completed_by_repo = _index_running_and_completed_validate_jobs(
            adapter_context.jobs
        )
        for repo_name, _path in targets:
            status, job_id, job_result = _classify_existing_validate_job(
                repo_name, running_by_repo, completed_by_repo
            )
            if status == "running":
                result_payload["running"][repo_name] = job_id
            elif status == "completed" and not force_revalidate:
                result_payload["completed"][repo_name] = {
                    "job_id": job_id,
                    "summary": _build_cache_summary(job_result),
                }
    return result_payload


def _submit_missing_validate_jobs(
    adapter_context: McpToolContext,
    git: Any,
    targets: list[tuple[str, str]],
    result_payload: dict[str, Any],
) -> None:
    """Submit a 'validate' job for every target not already running/cached,
    recording each new job id into ``result_payload["queued"]`` in place.
    Extracted verbatim from ``rm_projects``.
    """
    for repo_name, path in targets:
        if (
            repo_name in result_payload["running"]
            or repo_name in result_payload["completed"]
        ):
            continue
        result = adapter_context.submit_job(
            "validate",
            git.validate_single_project,
            path,
            _extra_job_data={"repo_name": repo_name},
        )
        result_payload["queued"][repo_name] = result["job_id"]


def _collect_validation_job_ids(
    targets: list[tuple[str, str]], result_payload: dict[str, Any]
) -> list[str]:
    """Gather each target's validate job id (queued, running, or cached),
    in target order. Extracted verbatim from ``rm_projects``.
    """
    validation_job_ids: list[str] = []
    for repo_name, _path in targets:
        if repo_name in result_payload["queued"]:
            validation_job_ids.append(result_payload["queued"][repo_name])
        elif repo_name in result_payload["running"]:
            validation_job_ids.append(result_payload["running"][repo_name])
        elif repo_name in result_payload["completed"]:
            validation_job_ids.append(result_payload["completed"][repo_name]["job_id"])
    return validation_job_ids


def _submit_commit_code_job(
    adapter_context: McpToolContext,
    git: Any,
    targets: list[tuple[str, str]],
    validation_job_ids: list[str],
    commit_code: bool,
    commit_message: str | None,
    result_payload: dict[str, Any],
) -> list[str]:
    """Submit the pre-release commit_code job when requested, and return the
    dependency list the version bump should wait on (the commit job alone
    when submitted, else the raw validation ids). Extracted verbatim from
    ``rm_projects``.
    """
    bump_dependencies = validation_job_ids
    if commit_code:
        commit_dirs = [path for _name, path in targets]
        commit_result = adapter_context.submit_job(
            "commit_code",
            adapter_context.wait_for_jobs_and_run,
            validation_job_ids,
            True,
            git.commit_code_projects,
            message=commit_message
            or "chore: commit validated feature code (pre-release)",
            run_precommit=True,
            project_dirs=commit_dirs,
        )
        result_payload["commit_job_id"] = commit_result["job_id"]
        bump_dependencies = [commit_result["job_id"]]
    return bump_dependencies


def _build_validate_terse_summary(result_payload: dict[str, Any]) -> dict[str, Any]:
    """Build the compact roll-up returned when ``summary=True``. Extracted
    verbatim from ``rm_projects``.
    """
    terse: dict[str, Any] = {
        "status": "submitted",
        "queued_count": len(result_payload["queued"]),
        "running_count": len(result_payload["running"]),
        "completed_count": len(result_payload["completed"]),
        "queued_projects": list(result_payload["queued"].keys()),
    }
    for key in ("commit_job_id",):
        if key in result_payload:
            terse[key] = result_payload[key]
    terse["message"] = (
        "Validation submitted. Poll action='validate_status' "
        "(summary mode) for the compact roll-up."
    )
    return terse


def register_project_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register install, build, and validation adapters."""

    adapter_context = context or from_server()

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_projects(
        action: str = Field(
            description="Action: 'install', 'build', 'validate', 'validate_status'"
        ),
        threads: int | None = Field(default=None, description="Parallel workers."),
        extra: str = Field(
            default="all", description="Install group (e.g. 'all') for 'install'."
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
        commit_code: bool = Field(
            default=False,
            description=(
                "For 'validate': after validation passes, concurrently stage "
                "(git add -A), run pre-commit, and commit feature code across ALL "
                "targeted repos. Use with commit_message."
            ),
        ),
        commit_message: str | None = Field(
            default=None,
            description="Commit message for the commit_code step. Required when commit_code=True.",
        ),
        force_revalidate: bool = Field(
            default=False,
            description="If true, bypass validation cache and force re-validation of all projects.",
        ),
        failed_only: bool = Field(
            default=False,
            description=(
                "For 'validate': target ONLY repositories whose most-recent "
                "validation failed (the remediation loop). Ignored if "
                "'repositories' is given. Forces re-validation of that set."
            ),
        ),
        summary: bool = Field(
            default=True,
            description=(
                "For 'validate'/'validate_status': return a COMPACT roll-up "
                "(counts + failed set + running names) instead of the full "
                "per-job dump. Keeps responses inline-returnable at thousands of "
                "repos. Set False for the full per-job detail."
            ),
        ),
        job_id: str | None = Field(
            default=None,
            description="Job ID to check status for 'validate_status' action.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str | RepoScanResult | dict[str, Any]:
        """Bulk install, build, and validate Python projects.

        The 'validate' action submits validation as a background job and returns
        a job_id immediately.  Use 'validate_status' with that job_id to poll
        progress and retrieve results once complete.
        """
        git = adapter_context.get_git_instance(threads=threads)
        del output_dir, generate_report, ctx
        failed_only, summary, force_revalidate, commit_code = (
            _normalize_rm_projects_flags(
                failed_only, summary, force_revalidate, commit_code
            )
        )

        resolved = resolve_action(
            action, RM_PROJECTS_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        _require_project_release_route(action)

        repositories, force_revalidate_override, early_return = (
            _resolve_failed_only_repositories(
                adapter_context, action, failed_only, repositories
            )
        )
        if early_return is not None:
            return early_return
        if force_revalidate_override:
            force_revalidate = True

        _filter_project_map_by_repositories(git, repositories)

        if action == "install":
            return adapter_context.submit_job(
                "install", git.install_projects, extra=extra
            )

        if action == "build":
            return adapter_context.submit_job("build", git.build_projects)

        if action == "validate":
            targets = _compute_validate_targets(git, repositories)
            result_payload = _scan_existing_validate_jobs(
                adapter_context, targets, force_revalidate
            )
            _submit_missing_validate_jobs(adapter_context, git, targets, result_payload)
            validation_job_ids = _collect_validation_job_ids(targets, result_payload)

            _submit_commit_code_job(
                adapter_context,
                git,
                targets,
                validation_job_ids,
                commit_code,
                commit_message,
                result_payload,
            )

            if summary:
                return _build_validate_terse_summary(result_payload)
            return result_payload

        if action == "validate_status":
            return adapter_context.get_job_status(job_id, summary=summary)

        return f"Error: Unknown action '{action}'"
