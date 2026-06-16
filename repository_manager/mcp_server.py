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

__version__ = "1.37.0"

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
    """Concurrency for the validation executor — bounded to a share of the host.

    Caps parallelism to **both** a CPU fraction and a RAM fraction (whichever is
    smaller) so a big workspace never oversubscribes the box (each validation
    runs pre-commit + pytest, which is CPU- and RAM-heavy). Defaults to **20%**
    of CPU and 20% of RAM. All env-tunable:

    * ``RM_MAX_WORKERS``     — explicit override (skips the computation).
    * ``RM_CPU_FRACTION``    — fraction of logical cores (default 0.20).
    * ``RM_RAM_FRACTION``    — fraction of total RAM to budget (default 0.20).
    * ``RM_WORKER_MEM_GB``   — assumed RAM per validation worker (default 1.5).
    (CONCEPT:RM-TOPOLOGY scale + host-throttle)
    """
    import os as _os

    override = _os.environ.get("RM_MAX_WORKERS")
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    def _frac(name: str, default: float) -> float:
        try:
            return float(_os.environ.get(name, default))
        except ValueError:
            return default

    try:
        cpu_count = psutil.cpu_count(logical=True) or 4
        cpu_workers = max(1, int(cpu_count * _frac("RM_CPU_FRACTION", 0.20)))
    except Exception:
        cpu_workers = 4

    try:
        total_gb = psutil.virtual_memory().total / (1024**3)
        per_worker = max(0.25, _frac("RM_WORKER_MEM_GB", 1.5))
        ram_workers = max(
            1, int((total_gb * _frac("RM_RAM_FRACTION", 0.20)) / per_worker)
        )
    except Exception:
        ram_workers = cpu_workers

    # Honor the tighter of the two caps so we stay under ~20% CPU AND ~20% RAM.
    return max(1, min(cpu_workers, ram_workers))


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


def _job_failures(j: dict[str, Any]) -> list[str]:
    """Extract the human-readable failure messages from a completed/failed job."""
    out: list[str] = []
    res = j.get("result")
    if res is not None and hasattr(res, "hooks"):
        for h in res.hooks:
            if not getattr(h, "passed", True):
                ho = getattr(h, "output", "").strip()
                out.append(
                    f"Hook '{h.hook_id}' failed: {ho}"
                    if ho
                    else f"Hook '{h.hook_id}' failed."
                )
    if res is not None and getattr(res, "error", None):
        out.append(res.error)
    if j.get("error"):
        out.append(j["error"])
    return out


def _job_passed(j: dict[str, Any]) -> bool:
    res = j.get("result")
    return bool(res is not None and getattr(res, "success", False))


def _latest_jobs() -> dict[str, dict[str, Any]]:
    """Return the deduplicated ``{job_id: job}`` set used for every roll-up.

    Repo-scoped jobs (those carrying ``repo_name``) collapse to the **most
    recent** job per repo, keyed by ``started_at``. This is the fix for the
    stale-roll-up bug: ``_jobs`` accumulates a new job_id for every validation
    of a repo across successive cascade runs, so a repo that FAILED early and
    then PASSED on a re-run used to appear in BOTH the passed and failed
    tallies — the historical failed job_id never cleared. Collapsing to the
    latest job per repo means a repo reflects only its current state, and stale
    'running' jobs from a superseded cascade stop inflating the running count.

    Workspace-wide orchestration jobs (phased bump/push, etc.) carry no
    ``repo_name`` and are each preserved verbatim. (CONCEPT:RM-TOPOLOGY)
    """
    latest_by_repo: dict[str, tuple[str, dict[str, Any]]] = {}
    out: dict[str, dict[str, Any]] = {}
    with _jobs_lock:
        for jid, j in _jobs.items():
            repo = j.get("repo_name")
            if not repo:
                out[jid] = j
                continue
            cur = latest_by_repo.get(repo)
            if cur is None or (j.get("started_at") or "") >= (
                cur[1].get("started_at") or ""
            ):
                latest_by_repo[repo] = (jid, j)
    for jid, j in latest_by_repo.values():
        out[jid] = j
    return out


def _reap_stale_jobs(max_age_seconds: float | None = None) -> None:
    """Flip jobs wedged in 'running' past a hard ceiling to a timeout failure.

    A validation worker whose subprocess hung (or whose host was starved by an
    oversized concurrent sweep) can otherwise sit in 'running' forever, making
    the roll-up look permanently frozen (e.g. ``626 completed / 45 running``
    unchanged for many minutes). This is a DISPLAY/accounting safety net — it
    does not, and cannot, kill the underlying thread; the per-command
    subprocess timeout (<=600s) is what actually frees the worker. The ceiling
    is deliberately set well above the longest legitimate per-repo validation
    so it never reaps a merely-slow repo. Env-tunable via
    ``RM_JOB_STALE_SECONDS`` (default 1800). (CONCEPT:RM-TOPOLOGY watchdog)
    """
    import os as _os

    if max_age_seconds is None:
        try:
            max_age_seconds = float(_os.environ.get("RM_JOB_STALE_SECONDS", 1800))
        except ValueError:
            max_age_seconds = 1800.0

    now = datetime.now(timezone.utc)
    with _jobs_lock:
        for j in _jobs.values():
            if j.get("status") not in ("running", "queued", "pending"):
                continue
            started = j.get("started_at")
            if not started:
                continue
            try:
                dt = datetime.fromisoformat(str(started).rstrip("Z"))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if (now - dt).total_seconds() > max_age_seconds:
                j["status"] = "failed"
                j["error"] = (
                    f"Job exceeded the {int(max_age_seconds)}s stale-job ceiling "
                    "and was reaped (worker wedged or host starved). The release "
                    "step is gated on validation, so re-run the failed set."
                )
                j["completed_at"] = now.isoformat() + "Z"


def _last_failed_repos() -> list[str]:
    """Repos whose most-recent validate job did NOT pass (for failed-only reruns).

    Uses the latest job per repo so a repo that was fixed and re-validated green
    drops out of the set. Powers the remediation loop's "re-validate only the
    failures" behavior. (CONCEPT:RM-TOPOLOGY)
    """
    return [
        str(repo)
        for repo, j in (
            (j.get("repo_name"), j)
            for j in _latest_jobs().values()
            if j.get("action") == "validate" and j.get("repo_name")
        )
        if j["status"] == "failed"
        or (j["status"] == "completed" and not _job_passed(j))
    ]


def _get_job_status(job_id: str | None = None, summary: bool = True) -> dict[str, Any]:
    """Get the status of a specific job, or a roll-up of all jobs.

    ``summary=True`` (default) returns a COMPACT roll-up — counts, the failed
    set with their failure detail, and the running names — but OMITS the full
    per-job record dict. This keeps the response small enough to return inline
    even at thousands of repositories (the full dump exceeds the MCP token limit
    and forces a file spill). ``summary=False`` adds the full ``jobs`` map.
    """
    if not job_id:
        # Self-heal first: reap wedged 'running' jobs, then roll up over the
        # LATEST job per repo (not every historical job_id) so stale failures
        # and stale running entries from superseded cascade runs never linger.
        _reap_stale_jobs()
        with _jobs_lock:
            if not _jobs:
                return {"status": "empty", "message": "No background jobs found."}

            counts = {"completed": 0, "running": 0, "failed": 0, "passed": 0}
            failed_projects: list[str] = []
            running_projects: list[str] = []
            failed_details: dict[str, Any] = {}
            jobs_output: dict[str, Any] = {}

            rollup_jobs = _latest_jobs()
            for jid, j in rollup_jobs.items():
                st = j["status"]
                repo_name = j.get("repo_name")
                if st in ("running", "queued", "pending"):
                    counts["running"] += 1
                    if repo_name:
                        running_projects.append(repo_name)
                elif st == "completed":
                    counts["completed"] += 1
                    if _job_passed(j):
                        counts["passed"] += 1
                    else:
                        counts["failed"] += 1
                        if repo_name:
                            failed_projects.append(repo_name)
                            failed_details[repo_name] = {
                                "job_id": jid,
                                "failures": _job_failures(j),
                            }
                elif st == "failed":
                    counts["failed"] += 1
                    if repo_name:
                        failed_projects.append(repo_name)
                        failed_details[repo_name] = {
                            "job_id": jid,
                            "failures": _job_failures(j),
                        }

                if not summary:
                    jd = {
                        "status": st,
                        "action": j["action"],
                        "started_at": j["started_at"],
                        "completed_at": j["completed_at"],
                    }
                    if repo_name:
                        jd["repo_name"] = repo_name
                    if st == "completed" and j.get("result") is not None:
                        jd["summary"] = {
                            "passed": _job_passed(j),
                            "failures": _job_failures(j),
                        }
                    elif st == "failed" and j.get("error"):
                        jd["error"] = j["error"]
                    jobs_output[jid] = jd

            counts["total"] = len(rollup_jobs)
            resp: dict[str, Any] = {
                "summary": counts,
                "failed_projects": failed_projects,
                "failed_projects_csv": ",".join(failed_projects),
                "failed_details": failed_details,
                "running_projects": running_projects,
            }
            if not summary:
                resp["jobs"] = jobs_output
            return resp

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

        # Include live progress details if available. At workspace scale (200+
        # repos) the full per-repo phase dicts blow past the MCP token limit
        # (and spill to a file), so the DEFAULT is a compact roll-up: per-phase
        # counts + the failed set + active names + remaining COUNT. summary=False
        # restores the full per-repo phase dicts + project lists.
        # (CONCEPT:RM-BUMP / RM-TOPOLOGY terse status)
        if "progress_detail" in job:
            pd = job["progress_detail"]
            response["current_phase"] = pd.get("current_phase", "")
            response["progress"] = pd.get("progress", 0)

            completed_projects = set()
            active_projects = set()
            remaining_projects = set()
            phase_failed: list[str] = []

            for phase_data in pd.get("phases", {}).values():
                repos_dict = phase_data.get("repos") or phase_data.get("details") or {}
                for repo_name, status in repos_dict.items():
                    if not isinstance(repo_name, str):
                        continue
                    if status in ("failed", "error"):
                        completed_projects.add(repo_name)
                        phase_failed.append(repo_name)
                    elif status in ("success", "skipped", "skip"):
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

            if summary:
                # Per-phase counts only (drop the big per-repo details/repos maps).
                phase_summary: dict[str, Any] = {}
                for pname, pdata in pd.get("phases", {}).items():
                    phase_summary[pname] = {
                        k: pdata.get(k)
                        for k in (
                            "status",
                            "total",
                            "processed",
                            "completed",
                            "success",
                            "failed",
                        )
                        if k in pdata
                    }
                response["phases"] = phase_summary
                response["counts"] = {
                    "completed": len(completed_projects),
                    "active": len(active_projects),
                    "remaining": len(remaining_projects),
                    "failed": len(phase_failed),
                }
                response["failed_projects"] = sorted(phase_failed)
                response["active_projects"] = sorted(active_projects)
            else:
                response["phases"] = pd.get("phases", {})
                response["completed_projects"] = sorted(completed_projects)
                response["active_projects"] = sorted(active_projects)
                response["remaining_projects"] = sorted(remaining_projects)
                response["failed_projects"] = sorted(phase_failed)

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


def _resolve_repo_dir(git: Git, spec: str) -> str:
    """Resolve a repo name / relative spec to its real on-disk directory.

    Honors the workspace's **nested** layout. The workspace.yml groups repos
    under subdirectories (e.g. ``agent-packages/agents/data-science-mcp``), so a
    bare name like ``agent-utilities`` lives at
    ``<ws>/agent-packages/agent-utilities`` — NOT the flat ``<ws>/agent-utilities``.
    ``validate`` already resolves via ``project_map``; the per-repo git
    sub-actions (pull/push/add/commit) historically flat-joined ``git.path + name``
    and so hit ``[Errno 2] No such file or directory`` on every nested repo (e.g.
    a standalone ``push projects=agent-utilities`` failing while ``validate``
    of the same repo passed). This makes path resolution consistent across
    actions by consulting ``project_map`` for bare names.

    Resolution order (first match wins):
      1. absolute path → used verbatim;
      2. relative path that already exists under ``git.path`` (flat repos, or an
         already-correct nested relative spec) → used as-is for back-compat;
      3. bare name matched by basename against ``project_map`` → its nested path;
      4. otherwise the flat join (preserves the prior behavior + error surface).
    """
    if os.path.isabs(spec):
        return spec
    flat = os.path.abspath(os.path.join(git.path, spec))
    if os.path.exists(flat):
        return flat
    base = os.path.basename(spec.rstrip("/"))
    for mapped in getattr(git, "project_map", {}).values():
        ap = os.path.abspath(os.path.expanduser(mapped))
        if os.path.basename(ap) == base:
            return ap
    return flat


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
            description="Action: 'raw', 'clone', 'enumerate', 'pull', 'push', 'phased_push', 'add', 'commit', 'pre_commit', 'commit_code'. 'enumerate' lists all repos across a GitLab instance/GitHub org into an ingest manifest (command=vcs, projects=groups/orgs)."
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
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult | str | dict:
        """Bulk Git operations and arbitrary command execution."""
        from repository_manager.models import GitError

        if not isinstance(auto_start, bool):
            auto_start = True

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

        if action == "enumerate":
            # Remote VCS enumeration for enterprise-scale ingestion (CONCEPT:KG-2.49):
            # list all repos across a GitLab instance / GitHub org into a JSON ingest
            # manifest the KG batch ingestor consumes. `command` selects the VCS
            # ('gitlab'|'github'); `projects` is a comma-separated group/org list.
            import uuid as _uuid

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
            run_id = _uuid.uuid4().hex[:10]
            if vcs == "github":
                refs = enumerate_github(orgs=scopes, user=not scopes)
            else:
                refs = enumerate_gitlab(groups=scopes)
            manifest_path = write_manifest(refs, run_id)
            return {
                "status": "ok",
                "vcs": vcs,
                "count": len(refs),
                "run_id": run_id,
                "manifest": manifest_path,
            }

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
                    pull_dirs.append(_resolve_repo_dir(git, p))
            return _submit_job("pull", git.pull_projects, project_dirs=pull_dirs)

        if action == "push":
            push_dirs: list[str] | None = None
            if projects:
                push_dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    push_dirs.append(_resolve_repo_dir(git, p))
            return _submit_job("push", git.push_projects, project_dirs=push_dirs)

        if action == "add":
            add_dirs: list[str] | None = None
            if projects:
                add_dirs = []
                for p in projects.split(","):
                    p = p.strip()
                    if not p:
                        continue
                    add_dirs.append(_resolve_repo_dir(git, p))
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
                    commit_dirs.append(_resolve_repo_dir(git, p))
            return _submit_job(
                "commit", git.commit_projects, message=message, project_dirs=commit_dirs
            )

        def _resolve_dirs(spec: str | None) -> list[str] | None:
            if not spec:
                return None
            out: list[str] = []
            for p in spec.split(","):
                p = p.strip()
                if not p:
                    continue
                out.append(_resolve_repo_dir(git, p))
            return out

        if action == "pre_commit":
            if not isinstance(run_precommit, bool):
                run_precommit = True
            pc_dirs = _resolve_dirs(projects)
            return _submit_job("pre_commit", git.pre_commit_projects, projects=pc_dirs)

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
            cc_dirs = _resolve_dirs(projects)
            return _submit_job(
                "commit_code",
                git.commit_code_projects,
                message=message,
                run_precommit=run_precommit,
                project_dirs=cc_dirs,
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
                auto_start=auto_start,
                progress=progress,
                _extra_job_data={"progress_detail": progress},
            )

        return f"Error: Unknown action '{action}'"


def register_workspace_management_tools(mcp: FastMCP):
    """Register tools for core workspace setup and organization."""

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
    ) -> list[str] | str | GitResult | dict:
        """Core workspace organization, configuration, and maintenance."""
        from repository_manager.models import GitError

        if not isinstance(summary, bool):
            summary = True
        if not isinstance(force, bool):
            force = False
        if not isinstance(auto_start, bool):
            auto_start = True

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
            return _get_job_status(job_id, summary=summary)

        return f"Error: Unknown action '{action}'"


def _wait_for_jobs_and_run(
    dependency_job_ids: list[str],
    success_required: bool,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    import time

    while True:
        all_done = True
        for job_id in dependency_job_ids:
            with _jobs_lock:
                status = _jobs.get(job_id, {}).get("status", "unknown")
            if status in ("running", "queued", "pending", "submitted"):
                all_done = False
                break
        if all_done:
            break
        time.sleep(1)

    if success_required:
        passed = True
        for job_id in dependency_job_ids:
            with _jobs_lock:
                job_data = _jobs.get(job_id, {})
                res = job_data.get("result")
                status = job_data.get("status")
                action = job_data.get("action")

                if status == "failed":
                    passed = False
                    break
                if action == "validate":
                    if not res or not getattr(res, "success", False):
                        passed = False
                        break
                elif action == "maintain":
                    if res == "Skipped due to dependency failures.":
                        passed = False
                        break

        if not passed:
            return "Skipped due to dependency failures."

    return func(*args, **kwargs)


def register_project_management_tools(mcp: FastMCP):
    """Register tools for the autonomous project harness."""

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_worktree(
        action: str = Field(
            description="Action: 'add', 'list', 'remove', 'merge', 'sync', 'prune', 'bulk_add', 'audit'."
        ),
        repo: str | None = Field(
            default=None,
            description="Repo basename (e.g. 'agent-utilities') or absolute path. Omit for 'list'/'prune' across all repos.",
        ),
        branch: str | None = Field(
            default=None,
            description="Worktree branch name (each session uses a distinct branch).",
        ),
        base: str = Field(
            default="main", description="Base branch to fork from / sync against."
        ),
        into: str = Field(default="main", description="Target branch for 'merge'."),
        adopt: bool = Field(
            default=False,
            description="For 'add': stash the canonical checkout's uncommitted WIP and replay it onto the new branch.",
        ),
        force: bool = Field(
            default=False,
            description="For 'remove': remove even if the worktree is dirty.",
        ),
        delete_branch: bool = Field(
            default=False, description="For 'remove': also delete the branch."
        ),
        strategy: str = Field(
            default="rebase", description="For 'sync': 'rebase' or 'merge'."
        ),
        stale_days: int = Field(
            default=14,
            description="For 'audit': an unmerged worktree quiet for longer than this many days is classified 'stale' (review) rather than 'active'.",
        ),
        prune_merged: bool = Field(
            default=False,
            description="For 'audit': DESTRUCTIVE. After classifying, remove every 'merged' worktree (and delete its branch) plus prune 'dangling' admin pointers. Never touches 'active'/'stale' work or orphaned directories.",
        ),
        repos: str | None = Field(
            default=None,
            description="For 'bulk_add': comma-separated repo basenames (default: every workspace repo).",
        ),
        path: str | None = Field(default=None, description="Workspace root override."),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict:
        """Manage git worktrees for concurrent multi-session development (CONCEPT:RM-WORKTREE).

        Each session works a repo in its own worktree on its own branch under
        ``/home/apps/worktrees/<repo>/<branch>`` (shared ``.git``, no re-clone),
        leaving the canonical checkout on its default branch so a working-tree
        reset never disturbs in-flight session work.
        """
        from repository_manager.worktree import WorktreeManager

        git = get_git_instance(path=path)
        wm = WorktreeManager(git)
        if action == "list":
            return wm.list_worktrees(repo=repo)
        if action == "prune":
            return wm.prune(repo=repo)
        if action == "audit":
            return wm.audit(
                repo=repo,
                base=base,
                stale_days=stale_days,
                prune_merged=prune_merged,
            )
        if action == "bulk_add":
            if branch is None:
                return {"ok": False, "error": "action 'bulk_add' requires 'branch'"}
            repo_list = [r.strip() for r in repos.split(",")] if repos else None
            return wm.bulk_add(branch, repos=repo_list, base=base)
        # add / remove / merge / sync all require a concrete repo + branch.
        if repo is None or branch is None:
            return {
                "ok": False,
                "error": f"action '{action}' requires 'repo' and 'branch'",
            }
        if action == "add":
            return wm.add(repo, branch, base=base, adopt=adopt)
        if action == "remove":
            return wm.remove(repo, branch, force=force, delete_branch=delete_branch)
        if action == "merge":
            return wm.merge(repo, branch, into=into)
        if action == "sync":
            return wm.sync(repo, branch, base=base, strategy=strategy)
        return {"ok": False, "error": f"unknown action: {action}"}

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
            description=(
                "Automatically run phased_bumpversion if validation passes. "
                "Begins at the lowest phase that has repository changes (and "
                "cascades to every later phase), skipping unchanged upstream "
                "phases."
            ),
        ),
        auto_push: bool = Field(
            default=False,
            description=(
                "Automatically run phased_push if validation passes. Begins at "
                "the lowest phase with unpushed work, skipping the inter-phase "
                "waits of unchanged upstream phases."
            ),
        ),
        bump_part: str = Field(
            default="minor",
            description="The part of the version to bump (e.g. minor, patch, major) if auto_bump is used.",
        ),
        prune_worktrees: bool = Field(
            default=False,
            description=(
                "For 'validate' with auto_bump/auto_push: after the release, prune "
                "session worktrees already merged into main (and dangling admin "
                "pointers). DESTRUCTIVE. Default False — the release still runs the "
                "audit and REPORTS the safe_to_prune/do_not_disturb classification "
                "under 'worktree_hygiene_job_id' WITHOUT deleting anything. Never "
                "touches active/in-flight or orphaned worktrees."
            ),
        ),
        commit_code: bool = Field(
            default=False,
            description=(
                "For 'validate': after validation passes and BEFORE the version "
                "bump, concurrently stage (git add -A), run pre-commit, and commit "
                "feature code across ALL targeted repos. Ensures untracked/new "
                "files are committed (not left for the push safety net). The bump "
                "then waits on this step. Use with commit_message."
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
    ) -> str | RepoScanResult | dict:
        """Bulk install, build, and validate Python projects.

        The 'validate' action submits validation as a background job and returns
        a job_id immediately.  Use 'validate_status' with that job_id to poll
        progress and retrieve results once complete.
        """
        git = get_git_instance(threads=threads)

        # Coerce Field defaults to real bools: when this tool is invoked directly
        # (bypassing FastMCP's default resolution, e.g. in unit tests), unset
        # bool params arrive as truthy ``FieldInfo`` objects. (CONCEPT:RM-TOPOLOGY)
        if not isinstance(failed_only, bool):
            failed_only = False
        if not isinstance(summary, bool):
            summary = True
        if not isinstance(force_revalidate, bool):
            force_revalidate = False
        if not isinstance(commit_code, bool):
            commit_code = False
        if not isinstance(prune_worktrees, bool):
            prune_worktrees = False

        # Remediation loop: ``failed_only`` re-validates ONLY the repos whose
        # most-recent validation failed (and forces past the cache). The final
        # ecosystem-wide sweep is a normal full validate (no failed_only).
        # (CONCEPT:RM-TOPOLOGY)
        if action == "validate" and failed_only and not repositories:
            _failed = _last_failed_repos()
            if not _failed:
                return {
                    "status": "clean",
                    "message": "No previously-failed projects to re-validate.",
                    "queued_count": 0,
                }
            repositories = ",".join(_failed)
            force_revalidate = True

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
                # O(jobs + targets): index the latest validate job per repo ONCE,
                # rather than re-scanning all jobs for every target (the former
                # O(targets x jobs) nested loop held _jobs_lock long enough to
                # starve the async event loop on a full-workspace run, so
                # validate_status RPCs timed out). Running takes precedence; else
                # the most-recent completed job wins.
                _running_by_repo: dict[str, str] = {}
                _completed_by_repo: dict[str, tuple[str, Any]] = {}
                for jid, j in _jobs.items():
                    if j.get("action") != "validate":
                        continue
                    rn = j.get("repo_name")
                    if not rn:
                        continue
                    if j["status"] in ("running", "queued", "pending"):
                        _running_by_repo[rn] = jid
                    elif j["status"] == "completed":
                        _completed_by_repo[rn] = (jid, j.get("result"))

                for repo_name, _path in targets:
                    existing_job_id = _running_by_repo.get(repo_name)
                    existing_job_result = None
                    if existing_job_id is not None:
                        existing_job_status = "running"
                    elif repo_name in _completed_by_repo:
                        existing_job_id, existing_job_result = _completed_by_repo[
                            repo_name
                        ]
                        existing_job_status = "completed"
                    else:
                        existing_job_status = None

                    if existing_job_status == "running":
                        result_payload["running"][repo_name] = existing_job_id
                    elif existing_job_status == "completed" and not force_revalidate:
                        cache_summary: dict[str, Any] = {
                            "passed": False,
                            "failures": [],
                        }
                        if existing_job_result:
                            if hasattr(existing_job_result, "success"):
                                cache_summary["passed"] = existing_job_result.success
                            if hasattr(existing_job_result, "hooks"):
                                cache_summary["failures"] = []
                                for h in existing_job_result.hooks:
                                    if not getattr(h, "passed", True):
                                        out = getattr(h, "output", "").strip()
                                        cache_summary["failures"].append(
                                            f"Hook '{h.hook_id}' failed: {out}"
                                            if out
                                            else f"Hook '{h.hook_id}' failed."
                                        )
                            if (
                                hasattr(existing_job_result, "error")
                                and existing_job_result.error
                            ):
                                cache_summary["failures"].append(
                                    existing_job_result.error
                                )

                        result_payload["completed"][repo_name] = {
                            "job_id": existing_job_id,
                            "summary": cache_summary,
                        }
                    else:
                        # Queue new job
                        pass  # We will queue it below outside this block but wait, we need to do it without deadlock.

            validation_job_ids = []
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

            for r_name in [t[0] for t in targets]:
                if r_name in result_payload["queued"]:
                    validation_job_ids.append(result_payload["queued"][r_name])
                elif r_name in result_payload["running"]:
                    validation_job_ids.append(result_payload["running"][r_name])
                elif r_name in result_payload["completed"]:
                    validation_job_ids.append(
                        result_payload["completed"][r_name]["job_id"]
                    )

            # Commit feature code (stage -A → pre-commit → commit) after validation
            # passes and before the bump, so untracked/new files are committed and
            # the bump operates on a clean tree. (CONCEPT:RM-TOPOLOGY)
            bump_dependencies = validation_job_ids
            if commit_code:
                commit_dirs = [path for _name, path in targets]
                res_commit = _submit_job(
                    "commit_code",
                    _wait_for_jobs_and_run,
                    validation_job_ids,
                    True,
                    git.commit_code_projects,
                    message=commit_message
                    or "chore: commit validated feature code (pre-release)",
                    run_precommit=True,
                    project_dirs=commit_dirs,
                )
                result_payload["commit_job_id"] = res_commit["job_id"]
                bump_dependencies = [res_commit["job_id"]]

            if auto_bump:
                progress = {
                    "current_phase": "Waiting for validation",
                    "progress": 0,
                    "phases": {},
                }
                res_bump = _submit_job(
                    "maintain",
                    _wait_for_jobs_and_run,
                    bump_dependencies,
                    True,
                    git.maintain_projects,
                    part=bump_part,
                    start_phase=1,
                    auto_start=True,
                    dry_run=False,
                    progress=progress,
                    _extra_job_data={"progress_detail": progress},
                )
                result_payload["bump_job_id"] = res_bump["job_id"]
                push_dependencies = [res_bump["job_id"]]
            else:
                push_dependencies = bump_dependencies

            if auto_push:
                progress = {
                    "current_phase": "Waiting for dependencies",
                    "progress": 0,
                    "phases": {},
                }
                res_push = _submit_job(
                    "phased_push",
                    _wait_for_jobs_and_run,
                    push_dependencies,
                    True,
                    git.phased_push,
                    start_phase=1,
                    auto_start=True,
                    project_filter=None,
                    progress=progress,
                    _extra_job_data={"progress_detail": progress},
                )
                result_payload["push_job_id"] = res_push["job_id"]

            # Worktree hygiene: after a release, audit the session worktrees and
            # (opt-in) prune the ones already merged into main. Report-only by
            # default — the result carries the safe_to_prune/do_not_disturb
            # classification and never deletes active/in-flight work. Gated on the
            # last release job; success_required=False so it still runs (and reports)
            # even when the bump/push was a no-op. (CONCEPT:RM-WORKTREE-AUDIT)
            if auto_bump or auto_push:
                if "push_job_id" in result_payload:
                    hygiene_deps = [result_payload["push_job_id"]]
                elif "bump_job_id" in result_payload:
                    hygiene_deps = [result_payload["bump_job_id"]]
                else:
                    hygiene_deps = bump_dependencies
                res_hygiene = _submit_job(
                    "worktree_hygiene",
                    _wait_for_jobs_and_run,
                    hygiene_deps,
                    False,
                    git.worktree_hygiene,
                    prune=prune_worktrees,
                )
                result_payload["worktree_hygiene_job_id"] = res_hygiene["job_id"]

            # Terse submission echo (default): at scale the full id↔name maps are
            # huge. Return counts + the small running/completed lists + release
            # job ids; full maps only when summary=False. (CONCEPT:RM-TOPOLOGY)
            if summary:
                terse: dict[str, Any] = {
                    "status": "submitted",
                    "queued_count": len(result_payload["queued"]),
                    "running_count": len(result_payload["running"]),
                    "completed_count": len(result_payload["completed"]),
                    "queued_projects": list(result_payload["queued"].keys()),
                }
                for k in (
                    "commit_job_id",
                    "bump_job_id",
                    "push_job_id",
                    "worktree_hygiene_job_id",
                ):
                    if k in result_payload:
                        terse[k] = result_payload[k]
                terse["message"] = (
                    "Validation submitted. Poll action='validate_status' "
                    "(summary mode) for the compact roll-up."
                )
                return terse

            return result_payload

        if action == "validate_status":
            return _get_job_status(job_id, summary=summary)

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
