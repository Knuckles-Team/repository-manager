#!/usr/bin/env python
import warnings

from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import os
import sys
import threading
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from agent_utilities.base_utilities import to_boolean, to_integer
from agent_utilities.mcp_utilities import (
    create_mcp_server,
)
from dotenv import find_dotenv, load_dotenv
from starlette.requests import Request
from starlette.responses import JSONResponse

from repository_manager.models import (
    GitResult,
    WorkspaceConfig,
)
from repository_manager.repository_manager import Git
from repository_manager.scan_models import RepoScanResult

__version__ = "1.26.0"

DEFAULT_WORKSPACE = os.environ.get(
    "REPOSITORY_MANAGER_WORKSPACE",
    os.environ.get("WORKSPACE_PATH", "/home/apps/workspace"),
)
DEFAULT_THREADS = to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "12"))
DEFAULT_WORKSPACE_YML = os.environ.get("WORKSPACE_YML", "workspace.yml")

logger = get_logger("RepositoryManagerServer")


# ---------------------------------------------------------------------------
# Unified Background Job Queue
# ---------------------------------------------------------------------------
import concurrent.futures

import psutil


def _get_max_workers():
    try:
        cpu_count = psutil.cpu_count(logical=True)
        if not cpu_count:
            return 4
        return max(1, int(cpu_count * 0.2))
    except Exception:
        return 4


_executor = concurrent.futures.ThreadPoolExecutor(max_workers=_get_max_workers())
_jobs: dict[str, dict[str, Any]] = {}

_jobs_lock = threading.RLock()


def _submit_job(
    action: str,
    func: Callable,
    *args: Any,
    _extra_job_data: dict | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Submit a function to run in the background.

    Returns a dict with ``status``, ``job_id``, and a human-readable
    ``message`` explaining how to poll for results.
    """
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now(timezone.utc).isoformat() + "Z"

    job_entry: dict[str, Any] = {
        "status": "running",
        "action": action,
        "started_at": now,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    if _extra_job_data:
        job_entry.update(_extra_job_data)

    with _jobs_lock:
        _jobs[job_id] = job_entry

    def _run() -> None:
        try:
            result = func(*args, **kwargs)
            with _jobs_lock:
                _jobs[job_id]["status"] = "completed"
                _jobs[job_id]["result"] = result
                _jobs[job_id]["completed_at"] = (
                    datetime.now(timezone.utc).isoformat() + "Z"
                )
        except Exception as e:
            with _jobs_lock:
                _jobs[job_id]["status"] = "failed"
                _jobs[job_id]["error"] = str(e)
                _jobs[job_id]["completed_at"] = (
                    datetime.now(timezone.utc).isoformat() + "Z"
                )

    _executor.submit(_run)

    return {
        "status": "submitted",
        "job_id": job_id,
        "message": (
            f"Job '{job_id}' ({action}) submitted. "
            f"Poll with the corresponding tool's status action using job_id='{job_id}'."
        ),
    }


def _get_job_status(job_id: str | None = None) -> dict[str, Any]:
    """Get the status of a specific job, or list all jobs."""
    if not job_id:
        with _jobs_lock:
            if not _jobs:
                return {"status": "empty", "message": "No background jobs found."}

            jobs_output = {}
            failed_projects = []
            running_projects = []

            for jid, j in _jobs.items():
                job_data = {
                    "status": j["status"],
                    "action": j["action"],
                    "started_at": j["started_at"],
                    "completed_at": j["completed_at"],
                }
                repo_name = None
                if "repo_name" in j:
                    repo_name = j["repo_name"]
                    job_data["repo_name"] = repo_name

                if (
                    j["status"] == "running"
                    or j["status"] == "queued"
                    or j["status"] == "pending"
                ):
                    if repo_name:
                        running_projects.append(repo_name)

                if j["status"] == "completed" and j.get("result"):
                    res = j["result"]
                    if hasattr(res, "success"):
                        summary = {"passed": res.success, "failures": []}
                        if hasattr(res, "hooks"):
                            for h in res.hooks:
                                if not getattr(h, "passed", True):
                                    out = getattr(h, "output", "").strip()
                                    summary["failures"].append(
                                        f"Hook '{h.hook_id}' failed: {out}"
                                        if out
                                        else f"Hook '{h.hook_id}' failed."
                                    )
                        if hasattr(res, "error") and res.error:
                            summary["failures"].append(res.error)
                        job_data["summary"] = summary
                        if not res.success and repo_name:
                            failed_projects.append(repo_name)

                if j["status"] == "failed" and j.get("error"):
                    job_data["error"] = j["error"]
                    if repo_name:
                        failed_projects.append(repo_name)

                jobs_output[jid] = job_data

            return {
                "jobs": jobs_output,
                "failed_projects": failed_projects,
                "failed_projects_csv": ",".join(failed_projects),
                "running_projects": running_projects,
            }

    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return {"status": "error", "message": f"Job '{job_id}' not found."}

        response: dict[str, Any] = {
            "job_id": job_id,
            "status": job["status"],
            "action": job["action"],
            "started_at": job["started_at"],
            "completed_at": job["completed_at"],
        }

        # Include live progress details if available
        if "progress_detail" in job:
            pd = job["progress_detail"]
            response["current_phase"] = pd.get("current_phase", "")
            response["progress"] = pd.get("progress", 0)
            response["phases"] = pd.get("phases", {})

            completed_projects = set()
            active_projects = set()
            remaining_projects = set()

            for phase_data in pd.get("phases", {}).values():
                repos_dict = phase_data.get("repos") or phase_data.get("details") or {}
                for repo_name, status in repos_dict.items():
                    if not isinstance(repo_name, str):
                        continue
                    if status in ("success", "failed", "error", "skipped", "skip"):
                        completed_projects.add(repo_name)
                    elif status == "running":
                        active_projects.add(repo_name)
                    elif status == "pending":
                        remaining_projects.add(repo_name)

            # Resolve overlaps across phases (completed > active > pending)
            for p in completed_projects | active_projects:
                remaining_projects.discard(p)
            for p in completed_projects:
                active_projects.discard(p)

            response["completed_projects"] = sorted(list(completed_projects))
            response["active_projects"] = sorted(list(active_projects))
            response["remaining_projects"] = sorted(list(remaining_projects))

        if job["status"] == "completed" and job["result"] is not None:
            if hasattr(job["result"], "to_markdown"):
                try:
                    response["summary"] = job["result"].to_markdown()
                    git = get_git_instance()
                    response["report_final_path"] = os.path.join(
                        git.path, "reports", "report_final.md"
                    )
                except Exception as e:
                    logger.error(
                        f"Failed to generate summary or locate report path in job status: {e}"
                    )

            if hasattr(job["result"], "model_dump"):
                try:
                    ts = job["result"]._format_timestamp_for_path()
                    summary_path = f"/home/apps/workspace/reports/validation-reports-{ts}/summary.md"
                    response["result"] = (
                        f"Validation completed. Check summary report at: {summary_path}"
                    )
                except Exception:
                    response["result"] = "Validation completed."
            else:
                response["result"] = str(job["result"])
        elif job["status"] == "failed":
            response["error"] = job["error"]

        return response


# ---------------------------------------------------------------------------
# Git instance factory
# ---------------------------------------------------------------------------


def get_git_instance(path: str | None = None, threads: int | None = None) -> Git:
    """Helper to get a Git instance with workpace YAML loaded."""
    workspace_path = path or DEFAULT_WORKSPACE
    git = Git(path=workspace_path, threads=threads)

    yml_path = os.path.join(workspace_path, DEFAULT_WORKSPACE_YML)
    if os.path.exists(yml_path):
        git.load_projects_from_yaml(yml_path)
    else:
        if path is not None:
            # If path was explicitly specified but workspace.yml is missing, discover projects
            git.discover_projects()
        else:
            # Fallback to the packaged version if the workspace-relative one isn't found
            from repository_manager.repository_manager import (
                DEFAULT_WORKSPACE_YML as PACKAGED_YML,
            )

            if os.path.exists(PACKAGED_YML):
                git.load_projects_from_yaml(PACKAGED_YML)
            else:
                git.discover_projects()

    return git


# ---------------------------------------------------------------------------
# MCP Tool Registration
# ---------------------------------------------------------------------------


def register_misc_tools(mcp: FastMCP):
    """Register miscellaneous tools like health check."""

    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})


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
            description="Action: 'raw', 'clone', 'pull', 'push', 'phased_push', 'add', 'commit'"
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
            description="Optional comma-separated list of repository URLs to clone or directory names/paths to pull/push/add/commit.",
        ),
        message: str | None = Field(
            default=None,
            description="Commit message for 'commit' action.",
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
            pull_dirs: list[str] | None = None
            if projects:
                pull_dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if os.path.isabs(p):
                        pull_dirs.append(p)
                    else:
                        pull_dirs.append(os.path.abspath(os.path.join(git.path, p)))
            return _submit_job("pull", git.pull_projects, project_dirs=pull_dirs)

        if action == "push":
            push_dirs: list[str] | None = None
            if projects:
                push_dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if os.path.isabs(p):
                        push_dirs.append(p)
                    else:
                        push_dirs.append(os.path.abspath(os.path.join(git.path, p)))
            return _submit_job("push", git.push_projects, project_dirs=push_dirs)

        if action == "add":
            add_dirs: list[str] | None = None
            if projects:
                add_dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if os.path.isabs(p):
                        add_dirs.append(p)
                    else:
                        add_dirs.append(os.path.abspath(os.path.join(git.path, p)))
            return _submit_job("add", git.add_projects, project_dirs=add_dirs)

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
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    if os.path.isabs(p):
                        commit_dirs.append(p)
                    else:
                        commit_dirs.append(os.path.abspath(os.path.join(git.path, p)))
            return _submit_job(
                "commit", git.commit_projects, message=message, project_dirs=commit_dirs
            )

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


def _validate_and_release_coordinator(
    git, targets, auto_bump, auto_push, bump_part, force_revalidate=False
):
    import time

    job_ids = {}

    # Check existing jobs for caching
    with _jobs_lock:
        for repo_name, path in targets:
            needs_validation = True

            if not force_revalidate:
                # Find the most recent job for this repo
                latest_job = None
                for _jid, j in _jobs.items():
                    if j["action"] == "validate" and j.get("repo_name") == repo_name:
                        latest_job = j

                if latest_job:
                    if latest_job["status"] == "completed" and latest_job.get("result"):
                        res = latest_job["result"]
                        if getattr(res, "success", False):
                            needs_validation = False
                            logger.info(
                                f"Skipping validation for {repo_name} (already passed)."
                            )

            if needs_validation:
                j_id = _submit_job(
                    "validate",
                    git.validate_single_project,
                    path,
                    _extra_job_data={"repo_name": repo_name},
                )
                job_ids[repo_name] = j_id["job_id"]
            else:
                job_ids[repo_name] = "ALREADY_PASSED"

    # Poll until all active validation jobs are completed
    while True:
        all_done = True
        for job_id in job_ids.values():
            if job_id == "ALREADY_PASSED":
                continue
            with _jobs_lock:
                status = _jobs[job_id]["status"]
            if status in ("running", "queued", "pending"):
                all_done = False
                break
        if all_done:
            break
        time.sleep(1)

    # Check if all passed
    passed = True
    for _repo_name, job_id in job_ids.items():
        if job_id == "ALREADY_PASSED":
            continue
        with _jobs_lock:
            job_data = _jobs[job_id]
            res = job_data.get("result")
            if not res or not getattr(res, "success", False):
                passed = False
                break

    release_results = {}
    if passed:
        logger.info("All validations passed. Queuing bump and push jobs...")
        if auto_bump:
            progress = {
                "current_phase": "Initializing Bumps",
                "progress": 0,
                "phases": {},
            }
            bump_job = _submit_job(
                "maintain",
                git.maintain_projects,
                part=bump_part,
                start_phase=1,
                dry_run=False,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )
            # Wait for bump to finish
            while True:
                with _jobs_lock:
                    b_status = _jobs[bump_job["job_id"]]["status"]
                if b_status not in ("running", "queued", "pending"):
                    break
                time.sleep(1)
            release_results["bump_job_id"] = bump_job["job_id"]

        if auto_push:
            progress = {
                "current_phase": "Initializing Pushes",
                "progress": 0,
                "phases": {},
            }
            push_job = _submit_job(
                "phased_push",
                git.phased_push,
                start_phase=1,
                project_filter=None,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )
            # Wait for push to finish
            while True:
                with _jobs_lock:
                    p_status = _jobs[push_job["job_id"]]["status"]
                if p_status not in ("running", "queued", "pending"):
                    break
                time.sleep(1)
            release_results["push_job_id"] = push_job["job_id"]

    return {
        "passed": passed,
        "release_results": release_results,
        "status": "completed" if passed else "deferred_due_to_failures",
    }


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
        auto_bump: bool = Field(
            default=False,
            description="Automatically run phased_bumpversion if validation passes.",
        ),
        auto_push: bool = Field(
            default=False,
            description="Automatically run phased_push if validation passes.",
        ),
        bump_part: str = Field(
            default="minor",
            description="The part of the version to bump (e.g. minor, patch, major) if auto_bump is used.",
        ),
        force_revalidate: bool = Field(
            default=False,
            description="If true, bypass validation cache and force re-validation of all projects.",
        ),
        job_id: str | None = Field(
            default=None,
            description="Job ID to check status for 'validate_status' action.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str | RepoScanResult | dict:
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
            result_payload: dict[str, Any] = {
                "queued": {},
                "running": {},
                "completed": {},
            }

            repo_list_for_writer = (
                repositories.replace(" ", "").split(",") if repositories else None
            )

            targets = []
            for _url, path in git.project_map.items():
                repo_name = os.path.basename(path)
                if repo_list_for_writer and repo_name not in repo_list_for_writer:
                    continue
                targets.append((repo_name, path))

            with _jobs_lock:
                for repo_name, _path in targets:
                    existing_job_id = None
                    existing_job_status = None
                    existing_job_result = None
                    for jid, j in _jobs.items():
                        if (
                            j["action"] == "validate"
                            and j.get("repo_name") == repo_name
                        ):
                            if j["status"] in ("running", "queued", "pending"):
                                existing_job_id = jid
                                existing_job_status = "running"
                                break
                            elif j["status"] == "completed":
                                existing_job_id = jid
                                existing_job_status = "completed"
                                existing_job_result = j.get("result")

                    if existing_job_status == "running":
                        result_payload["running"][repo_name] = existing_job_id
                    elif existing_job_status == "completed" and not force_revalidate:
                        summary: dict[str, Any] = {"passed": False, "failures": []}
                        if existing_job_result:
                            if hasattr(existing_job_result, "success"):
                                summary["passed"] = existing_job_result.success
                            if hasattr(existing_job_result, "hooks"):
                                summary["failures"] = []
                                for h in existing_job_result.hooks:
                                    if not getattr(h, "passed", True):
                                        out = getattr(h, "output", "").strip()
                                        summary["failures"].append(
                                            f"Hook '{h.hook_id}' failed: {out}"
                                            if out
                                            else f"Hook '{h.hook_id}' failed."
                                        )
                            if (
                                hasattr(existing_job_result, "error")
                                and existing_job_result.error
                            ):
                                summary["failures"].append(existing_job_result.error)

                        result_payload["completed"][repo_name] = {
                            "job_id": existing_job_id,
                            "summary": summary,
                        }
                    else:
                        # Queue new job
                        pass  # We will queue it below outside this block but wait, we need to do it without deadlock.

            if auto_bump or auto_push:
                coord_id = _submit_job(
                    "validate_and_release",
                    _validate_and_release_coordinator,
                    git=git,
                    targets=targets,
                    auto_bump=auto_bump,
                    auto_push=auto_push,
                    bump_part=bump_part,
                    force_revalidate=force_revalidate,
                )
                result_payload["coordinator_job_id"] = coord_id["job_id"]
            else:
                for repo_name, path in targets:
                    if (
                        repo_name in result_payload["running"]
                        or repo_name in result_payload["completed"]
                    ):
                        continue

                    res = _submit_job(
                        "validate",
                        git.validate_single_project,
                        path,
                        _extra_job_data={"repo_name": repo_name},
                    )
                    result_payload["queued"][repo_name] = res["job_id"]

            return result_payload

        if action == "validate_status":
            return _get_job_status(job_id)

        return f"Error: Unknown action '{action}'"


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize the MCP instance, args, and middlewares."""
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="RepositoryManager",
        version=__version__,
        instructions="Expert tool for managing hierarchical Git workspaces, engineering bulk operations, and documentation.",
    )

    registered_tags = []
    if to_boolean(os.getenv("MISCTOOL", "True")):
        register_misc_tools(mcp)
        registered_tags.append("misc")

    if to_boolean(os.getenv("GIT_OPERATIONSTOOL", "True")):
        register_git_operations_tools(mcp)
        registered_tags.append("git_operations")

    if to_boolean(os.getenv("WORKSPACE_MANAGEMENTTOOL", "True")):
        register_workspace_management_tools(mcp)
        register_project_management_tools(mcp)
        registered_tags.append("workspace_management")

    for mw in middlewares:
        mcp.add_middleware(mw)

    return mcp, args, middlewares, registered_tags


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags = get_mcp_instance()
    print(f"{'repository-manager'} MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Dynamic Tags Loaded: {len(set(registered_tags))}", file=sys.stderr)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()
