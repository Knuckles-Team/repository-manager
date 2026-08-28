#!/usr/bin/env python
import warnings

from fastmcp.utilities.logging import get_logger

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
from datetime import UTC, datetime
from typing import Any

from agent_utilities.base_utilities import to_integer
from agent_utilities.core.config import load_config, setting
from agent_utilities.mcp.server_factory import create_mcp_server
from agent_utilities.mcp.verbose_tools import register_tool_surface

from repository_manager.mcp_tools import (
    MCP_TOOL_REGISTRY,
    register_docs_readiness_tools,
    register_git_operations_tools,
    register_misc_tools,
    register_project_management_tools,
    register_workspace_management_tools,
)
from repository_manager.mcp_tools.contracts import (
    RM_BUILD_ACTIONS,
    RM_DOCS_READINESS_ACTIONS,
    RM_GATES_ACTIONS,
    RM_GIT_ACTIONS,
    RM_MERGE_QUEUE_ACTIONS,
    RM_PROJECTS_ACTIONS,
    RM_WORKSPACE_ACTIONS,
    RM_WORKTREE_ACTIONS,
)
from repository_manager.models import GitResult
from repository_manager.repository_manager import Git

__version__ = "3.4.0"

DEFAULT_WORKSPACE = setting(
    "REPOSITORY_MANAGER_WORKSPACE",
    setting("WORKSPACE_PATH", os.getenv("AGENT_UTILITIES_WORKSPACE_ROOT", os.getcwd())),
)
DEFAULT_THREADS = to_integer(setting("REPOSITORY_MANAGER_THREADS", "12"))
DEFAULT_WORKSPACE_YML = setting("WORKSPACE_YML", "workspace.yml")

logger = get_logger("RepositoryManagerServer")

__all__ = [
    "RM_BUILD_ACTIONS",
    "RM_DOCS_READINESS_ACTIONS",
    "RM_GATES_ACTIONS",
    "RM_GIT_ACTIONS",
    "RM_MERGE_QUEUE_ACTIONS",
    "RM_PROJECTS_ACTIONS",
    "RM_WORKSPACE_ACTIONS",
    "RM_WORKTREE_ACTIONS",
    "register_git_operations_tools",
    "register_docs_readiness_tools",
    "register_misc_tools",
    "register_project_management_tools",
    "register_workspace_management_tools",
]


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
    override = setting("RM_MAX_WORKERS", None)
    if override:
        try:
            return max(1, int(override))
        except ValueError:
            pass

    def _frac(name: str, default: float) -> float:
        try:
            return float(setting(name, default))
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
_job_futures: dict[str, concurrent.futures.Future[Any]] = {}

_jobs_lock = threading.RLock()


def _submit_job(
    action: str,
    func: Callable,
    *args: Any,
    _extra_job_data: dict | None = None,
    **kwargs: Any,
) -> dict[str, str]:
    """Submit a function to run in the background.

    A successful response includes ``status``, ``job_id``, and a human-readable
    polling message. Executor refusal returns ``status=error`` without publishing
    a job id that cannot be polled or cancelled.
    """
    job_id = str(uuid.uuid4())[:8]
    now = datetime.now(UTC).isoformat() + "Z"

    job_entry: dict[str, Any] = {
        "status": "queued",
        "action": action,
        "submitted_at": now,
        "heartbeat_at": now,
        "started_at": None,
        "completed_at": None,
        "result": None,
        "error": None,
    }
    if _extra_job_data:
        job_entry.update(_extra_job_data)
    initial_progress = _progress_marker(job_entry)
    if initial_progress is not None:
        job_entry["_progress_marker"] = initial_progress

    def _run() -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None or job["status"] != "queued":
                return
            job["status"] = "running"
            started_at = datetime.now(UTC).isoformat() + "Z"
            job["started_at"] = started_at
            job["heartbeat_at"] = started_at
        try:
            result = func(*args, **kwargs)
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is not None and job["status"] == "running":
                    job["status"] = "completed"
                    job["result"] = result
                    job["completed_at"] = datetime.now(UTC).isoformat() + "Z"
        except Exception as exc:
            with _jobs_lock:
                job = _jobs.get(job_id)
                if job is not None and job["status"] == "running":
                    job["status"] = "failed"
                    job["error"] = (
                        f"Background repository operation raised {type(exc).__name__}."
                    )
                    job["completed_at"] = datetime.now(UTC).isoformat() + "Z"

    def _release_future(done: concurrent.futures.Future[Any]) -> None:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if done.cancelled() and job is not None and job["status"] == "queued":
                job["status"] = "cancelled"
                job["completed_at"] = datetime.now(UTC).isoformat() + "Z"
            _job_futures.pop(job_id, None)

    # Publish the record and its Future atomically under the same lock the worker
    # must acquire before transitioning out of ``queued``. No status/cancel caller
    # can therefore observe a queued job without its cancellation handle.
    with _jobs_lock:
        try:
            future = _executor.submit(_run)
        except Exception as exc:
            return {
                "status": "error",
                "message": (
                    f"Job ({action}) could not be submitted: {type(exc).__name__}."
                ),
            }
        _jobs[job_id] = job_entry
        _job_futures[job_id] = future
        future.add_done_callback(_release_future)

    return {
        "status": "submitted",
        "job_id": job_id,
        "message": (
            f"Job '{job_id}' ({action}) submitted. "
            f"Poll with the corresponding tool's status action using job_id='{job_id}'."
        ),
    }


def _cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued job, or honestly refuse once its worker has started."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return {"status": "error", "message": f"Job '{job_id}' not found."}

        status = str(job["status"])
        if status == "cancelled":
            return {"job_id": job_id, "status": status, "cancelled": True}
        if status in {"completed", "failed"}:
            return {
                "job_id": job_id,
                "status": status,
                "cancelled": False,
                "message": "Job is already terminal and cannot be cancelled.",
            }

        future = _job_futures.get(job_id)
        if status == "queued" and future is not None:
            # ``Future.cancel`` may return false when a pool worker has entered
            # our wrapper but is still blocked on ``_jobs_lock``. The operation
            # itself has not started at that point: marking the record cancelled
            # is sufficient because ``_run`` checks this state before calling it.
            future.cancel()
            job["status"] = "cancelled"
            job["completed_at"] = datetime.now(UTC).isoformat() + "Z"
            return {"job_id": job_id, "status": "cancelled", "cancelled": True}

        if status == "queued":
            return {
                "job_id": job_id,
                "status": "error",
                "cancelled": False,
                "message": "Queued job has no cancellation Future; lifecycle is invalid.",
            }

        return {
            "job_id": job_id,
            "status": str(job["status"]),
            "cancelled": False,
            "message": (
                "Job has started; cooperative cancellation is not supported, "
                "so it remains running."
            ),
        }


def _hook_failure_messages(res: Any) -> list[str]:
    """Extract per-hook failure messages from a result carrying a ``hooks`` list."""
    out: list[str] = []
    if res is None or not hasattr(res, "hooks"):
        return out
    for h in res.hooks:
        if not getattr(h, "passed", True):
            ho = getattr(h, "output", "").strip()
            out.append(
                f"Hook '{h.hook_id}' failed: {ho}"
                if ho
                else f"Hook '{h.hook_id}' failed."
            )
    return out


def _result_error_messages(res: Any) -> list[str]:
    """Extract error messages from a result or a list of results."""
    out: list[str] = []
    results = res if isinstance(res, list) else [res]
    for result in results:
        error = getattr(result, "error", None)
        if error:
            out.append(str(getattr(error, "message", error)))
    return out


def _job_failures(j: dict[str, Any]) -> list[str]:
    """Extract the human-readable failure messages from a completed/failed job."""
    res = j.get("result")
    out = _hook_failure_messages(res) + _result_error_messages(res)
    if j.get("error"):
        out.append(j["error"])
    return out


def _job_passed(j: dict[str, Any]) -> bool:
    """Return the operation outcome separately from its execution lifecycle."""
    res = j.get("result")
    if res is None:
        return False
    if isinstance(res, list):
        return all(
            isinstance(item, GitResult) and item.status in {"success", "skipped"}
            for item in res
        )
    if isinstance(res, GitResult):
        return res.status in {"success", "skipped"}
    if hasattr(res, "success"):
        return bool(res.success)
    return True


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


def _progress_marker(job: dict[str, Any]) -> tuple[Any, ...] | None:
    """Return a compact marker for observable long-running job progress."""
    progress = job.get("progress_detail")
    if not isinstance(progress, dict):
        return None
    phases = []
    for name, detail in sorted(progress.get("phases", {}).items()):
        if not isinstance(detail, dict):
            continue
        phases.append(
            (
                str(name),
                detail.get("status"),
                detail.get("processed"),
                detail.get("completed"),
                detail.get("success"),
                detail.get("failed"),
            )
        )
    return (
        progress.get("current_phase"),
        progress.get("progress"),
        tuple(phases),
    )


def _refresh_progress_heartbeats() -> None:
    """Advance heartbeats only when a job's observable progress changes."""
    now = datetime.now(UTC).isoformat() + "Z"
    with _jobs_lock:
        for job in _jobs.values():
            if job.get("status") not in {"running", "queued", "pending"}:
                continue
            marker = _progress_marker(job)
            if marker is None:
                continue
            previous = job.get("_progress_marker")
            job["_progress_marker"] = marker
            if previous is not None and marker != previous:
                job["heartbeat_at"] = now


def _resolve_stale_seconds(max_age_seconds: float | None) -> float:
    """Resolve the stale-job age ceiling, falling back to the env setting."""
    if max_age_seconds is not None:
        return max_age_seconds
    try:
        return float(setting("RM_JOB_STALE_SECONDS", 1800))
    except ValueError:
        return 1800.0


def _reap_job_if_stale(
    job_id: str, j: dict[str, Any], now: datetime, max_age_seconds: float
) -> None:
    """Fail one job record if it is orphaned and past the stale-age ceiling."""
    if j.get("status") not in ("running", "queued", "pending"):
        return
    future = _job_futures.get(job_id)
    if future is not None and not future.done():
        # A live worker/pending Future is stronger liveness evidence than
        # wall age. This is essential for phased_push, whose configured
        # inter-phase wait alone can legitimately reach 30 minutes.
        return
    heartbeat = j.get("heartbeat_at") or j.get("started_at") or j.get("submitted_at")
    if not heartbeat:
        return
    try:
        dt = datetime.fromisoformat(str(heartbeat).rstrip("Z"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
    except ValueError:
        return
    if (now - dt).total_seconds() > max_age_seconds:
        j["status"] = "failed"
        j["error"] = (
            f"Job exceeded the {int(max_age_seconds)}s stale-job ceiling "
            "without a live Future or progress heartbeat and was reaped "
            "as an orphaned record. The release step is gated on "
            "validation, so re-run the failed set."
        )
        j["completed_at"] = now.isoformat() + "Z"


def _reap_stale_jobs(max_age_seconds: float | None = None) -> None:
    """Fail old orphaned records without terminalizing live background work.

    A validation worker whose subprocess hung (or whose host was starved by an
    oversized concurrent sweep) can otherwise sit in 'running' forever, making
    the roll-up look permanently frozen (e.g. ``626 completed / 45 running``
    unchanged for many minutes). A pending/running Future is authoritative
    liveness evidence regardless of age; observable phased progress advances a
    heartbeat. Only a record with neither is eligible for age-based reaping.
    This is a DISPLAY/accounting safety net — it does not kill an underlying
    operation. Env-tunable via
    ``RM_JOB_STALE_SECONDS`` (default 1800). (CONCEPT:RM-TOPOLOGY watchdog)
    """
    max_age_seconds = _resolve_stale_seconds(max_age_seconds)
    _refresh_progress_heartbeats()
    now = datetime.now(UTC)
    with _jobs_lock:
        for job_id, j in _jobs.items():
            _reap_job_if_stale(job_id, j, now, max_age_seconds)


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


def _record_failed_job(
    jid: str,
    j: dict[str, Any],
    repo_name: str | None,
    failed_projects: list[str],
    failed_details: dict[str, Any],
) -> None:
    """Record one repo-scoped job's failure into the roll-up collections."""
    if not repo_name:
        return
    failed_projects.append(repo_name)
    failed_details[repo_name] = {"job_id": jid, "failures": _job_failures(j)}


def _tally_job_into_rollup(
    jid: str,
    j: dict[str, Any],
    counts: dict[str, int],
    failed_projects: list[str],
    running_projects: list[str],
    failed_details: dict[str, Any],
) -> None:
    """Tally one job's status into the roll-up counters/collections."""
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
            _record_failed_job(jid, j, repo_name, failed_projects, failed_details)
    elif st == "failed":
        counts["failed"] += 1
        _record_failed_job(jid, j, repo_name, failed_projects, failed_details)
    elif st == "cancelled":
        counts["cancelled"] += 1


def _job_status_row(jid: str, j: dict[str, Any]) -> dict[str, Any]:
    """Build one job's row for the non-summary roll-up ``jobs`` map."""
    st = j["status"]
    repo_name = j.get("repo_name")
    jd: dict[str, Any] = {
        "status": st,
        "action": j["action"],
        "submitted_at": j.get("submitted_at", j["started_at"]),
        "heartbeat_at": j.get("heartbeat_at", j["started_at"]),
        "started_at": j["started_at"],
        "completed_at": j["completed_at"],
    }
    if repo_name:
        jd["repo_name"] = repo_name
    if st == "completed" and j.get("result") is not None:
        jd["summary"] = {"passed": _job_passed(j), "failures": _job_failures(j)}
    elif st == "failed" and j.get("error"):
        jd["error"] = j["error"]
    return jd


def _job_status_rollup(summary: bool) -> dict[str, Any]:
    """Build the all-jobs roll-up: counts, failed/running sets, optional detail.

    Self-heals first (the caller reaps wedged 'running' jobs), then rolls up
    over the LATEST job per repo (not every historical job_id) so stale
    failures and stale running entries from superseded cascade runs never
    linger.
    """
    with _jobs_lock:
        if not _jobs:
            return {"status": "empty", "message": "No background jobs found."}

        counts = {
            "completed": 0,
            "running": 0,
            "failed": 0,
            "cancelled": 0,
            "passed": 0,
        }
        failed_projects: list[str] = []
        running_projects: list[str] = []
        failed_details: dict[str, Any] = {}
        jobs_output: dict[str, Any] = {}

        rollup_jobs = _latest_jobs()
        for jid, j in rollup_jobs.items():
            _tally_job_into_rollup(
                jid, j, counts, failed_projects, running_projects, failed_details
            )
            if not summary:
                jobs_output[jid] = _job_status_row(jid, j)

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


def _job_status_base_fields(job_id: str, job: dict[str, Any]) -> dict[str, Any]:
    """The status fields common to every single-job response."""
    return {
        "job_id": job_id,
        "status": job["status"],
        "action": job["action"],
        "submitted_at": job.get("submitted_at", job["started_at"]),
        "heartbeat_at": job.get("heartbeat_at", job["started_at"]),
        "started_at": job["started_at"],
        "completed_at": job["completed_at"],
    }


def _classify_repo_status(
    repo_name: str,
    status: Any,
    completed_projects: set[str],
    active_projects: set[str],
    remaining_projects: set[str],
    phase_failed: list[str],
) -> None:
    """Bucket one repo into the right set/list by its observed phase status."""
    if status in ("failed", "error"):
        completed_projects.add(repo_name)
        phase_failed.append(repo_name)
    elif status in ("success", "skipped", "skip"):
        completed_projects.add(repo_name)
    elif status == "running":
        active_projects.add(repo_name)
    elif status == "pending":
        remaining_projects.add(repo_name)


def _tally_progress_repos(
    pd: dict[str, Any],
) -> tuple[set[str], set[str], set[str], list[str]]:
    """Bucket every repo named across a job's phases by its observed status.

    Returns ``(completed_projects, active_projects, remaining_projects,
    phase_failed)``, with overlaps across phases resolved so a repo counts as
    only one of completed / active / remaining (completed > active > pending).
    """
    completed_projects: set[str] = set()
    active_projects: set[str] = set()
    remaining_projects: set[str] = set()
    phase_failed: list[str] = []

    for phase_data in pd.get("phases", {}).values():
        repos_dict = phase_data.get("repos") or phase_data.get("details") or {}
        for repo_name, status in repos_dict.items():
            if isinstance(repo_name, str):
                _classify_repo_status(
                    repo_name,
                    status,
                    completed_projects,
                    active_projects,
                    remaining_projects,
                    phase_failed,
                )

    for p in completed_projects | active_projects:
        remaining_projects.discard(p)
    for p in completed_projects:
        active_projects.discard(p)

    return completed_projects, active_projects, remaining_projects, phase_failed


def _summarize_phases(pd: dict[str, Any]) -> dict[str, Any]:
    """Per-phase counts only (drop the big per-repo details/repos maps)."""
    phase_summary: dict[str, Any] = {}
    for pname, pdata in pd.get("phases", {}).items():
        phase_summary[pname] = {
            k: pdata.get(k)
            for k in ("status", "total", "processed", "completed", "success", "failed")
            if k in pdata
        }
    return phase_summary


def _apply_progress_detail(
    response: dict[str, Any], pd: dict[str, Any], summary: bool
) -> None:
    """Populate ``response`` with progress/phase/project fields from ``progress_detail``.

    At workspace scale (200+ repos) the full per-repo phase dicts blow past
    the MCP token limit (and spill to a file), so the DEFAULT (``summary``)
    is a compact roll-up: per-phase counts + the failed set + active names +
    remaining COUNT. ``summary=False`` restores the full per-repo phase dicts
    + project lists. (CONCEPT:RM-BUMP / RM-TOPOLOGY terse status)
    """
    response["current_phase"] = pd.get("current_phase", "")
    response["progress"] = pd.get("progress", 0)

    completed_projects, active_projects, remaining_projects, phase_failed = (
        _tally_progress_repos(pd)
    )

    if summary:
        response["phases"] = _summarize_phases(pd)
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


def _apply_completed_outcome(response: dict[str, Any], job: dict[str, Any]) -> None:
    """Populate outcome/failures/summary/result fields for a completed job."""
    response["outcome"] = "succeeded" if _job_passed(job) else "failed"
    response["failures"] = _job_failures(job)
    result = job["result"]

    if hasattr(result, "to_markdown"):
        try:
            response["summary"] = result.to_markdown()
            response["report_final_path"] = "report://validation/report_final.md"
        except Exception as e:
            logger.error(
                "Failed to generate project summary: error_type=%s",
                type(e).__name__,
            )

    if hasattr(result, "model_dump"):
        try:
            ts = result._format_timestamp_for_path()
            summary_path = f"report://validation-reports-{ts}/summary.md"
            response["result"] = (
                f"Validation completed. Check summary report at: {summary_path}"
            )
        except Exception:
            response["result"] = "Validation completed."
    else:
        response["result"] = str(result)


def _apply_job_outcome(response: dict[str, Any], job: dict[str, Any]) -> None:
    """Populate the outcome-specific fields for a completed/failed/cancelled job."""
    if job["status"] == "completed" and job["result"] is not None:
        _apply_completed_outcome(response, job)
    elif job["status"] == "failed":
        response["outcome"] = "failed"
        response["error"] = job["error"]
    elif job["status"] == "cancelled":
        response["outcome"] = "cancelled"


def _job_status_single(job_id: str, summary: bool) -> dict[str, Any]:
    """Build the status response for one specific ``job_id``."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            return {"status": "error", "message": f"Job '{job_id}' not found."}

        response = _job_status_base_fields(job_id, job)

        if "progress_detail" in job:
            _apply_progress_detail(response, job["progress_detail"], summary)

        _apply_job_outcome(response, job)

        return response


def _get_job_status(job_id: str | None = None, summary: bool = True) -> dict[str, Any]:
    """Get the status of a specific job, or a roll-up of all jobs.

    ``summary=True`` (default) returns a COMPACT roll-up — counts, the failed
    set with their failure detail, and the running names — but OMITS the full
    per-job record dict. This keeps the response small enough to return inline
    even at thousands of repositories (the full dump exceeds the MCP token limit
    and forces a file spill). ``summary=False`` adds the full ``jobs`` map.
    """
    # Targeted polling must self-heal just like the roll-up. Otherwise callers
    # following the documented job_id path can observe a wedged ``running`` job
    # forever while only an unrelated global-status call reaps it.
    _reap_stale_jobs()
    if not job_id:
        return _job_status_rollup(summary)
    return _job_status_single(job_id, summary)


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


def _resolve_commit_code_target(path: str) -> tuple[str | None, str | None]:
    """Resolve an explicit ``path`` to the ONE commit_code target, or an error.

    D-CDX-60 — ``commit_code``'s ``path`` argument used to be spent entirely on
    constructing the ``Git`` workspace instance (:func:`get_git_instance`); the
    actual set of repos it committed in came only from ``projects``, which
    defaults to a full workspace fan-out
    (``Git.commit_code_projects(project_dirs=None)`` → ``self.project_map``).
    A caller naming an explicit isolated worktree — with no ``projects``
    filter — got a false-success job that quietly committed nothing there and
    fanned out elsewhere instead. This makes an explicit ``path`` (with no
    ``projects`` override) mean exactly what it says: commit code in this ONE
    repository/worktree, nothing else.

    Returns ``(resolved_path, None)`` on success or ``(None, error_message)``.
    Refuses a path outside the configured workspace root or worktree root (no
    escaping to arbitrary host paths) and a path that is not itself a git
    repository or linked worktree.
    """
    from repository_manager.worktree import WORKTREE_ROOT

    resolved = os.path.abspath(os.path.expanduser(path))
    roots = [os.path.abspath(DEFAULT_WORKSPACE), WORKTREE_ROOT]
    if not any(
        resolved == root or resolved.startswith(root + os.sep) for root in roots
    ):
        return None, (
            f"path {resolved!r} is outside the configured workspace roots "
            f"({', '.join(roots)}); refusing to commit_code there"
        )
    if not os.path.exists(os.path.join(resolved, ".git")):
        return None, (
            f"path {resolved!r} is not a git repository or linked worktree "
            "(no .git present)"
        )
    return resolved, None


# ---------------------------------------------------------------------------
# MCP adapter compatibility seam
# ---------------------------------------------------------------------------
# The public registrar names remain importable from this module while their
# FastMCP definitions live under repository_manager.mcp_tools. Keeping the
# names here preserves packaging and direct registration consumers; the actual
# server uses MCP_TOOL_REGISTRY below.


def _job_is_pending(job_id: str) -> bool:
    """Return True if a dependency job is still queued/running/unsettled."""
    with _jobs_lock:
        status = _jobs.get(job_id, {}).get("status", "unknown")
    return status in ("running", "queued", "pending", "submitted")


def _wait_until_jobs_settle(dependency_job_ids: list[str]) -> None:
    """Block until every dependency job has left a pending/running state."""
    import time

    while any(_job_is_pending(job_id) for job_id in dependency_job_ids):
        time.sleep(1)


def _dependency_job_passed(job_id: str) -> bool:
    """Return whether one settled dependency job counts as a pass."""
    with _jobs_lock:
        job_data = _jobs.get(job_id, {})
        result = job_data.get("result")
        status = job_data.get("status")
        action = job_data.get("action")

    if status == "failed":
        return False
    if action == "validate":
        return bool(result) and bool(getattr(result, "success", False))
    if action == "maintain":
        return result != "Skipped due to dependency failures."
    return True


def _wait_for_jobs_and_run(
    dependency_job_ids: list[str],
    success_required: bool,
    func: Callable,
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Run a dependent operation after its existing background jobs settle."""
    _wait_until_jobs_settle(dependency_job_ids)

    if success_required and not all(
        _dependency_job_passed(job_id) for job_id in dependency_job_ids
    ):
        return "Skipped due to dependency failures."

    return func(*args, **kwargs)


# ---------------------------------------------------------------------------
# Server bootstrap
# ---------------------------------------------------------------------------


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize the MCP instance, args, and middlewares."""
    load_config()

    args, mcp, middlewares = create_mcp_server(
        name="RepositoryManager",
        version=__version__,
        instructions="Expert tool for managing hierarchical Git workspaces, engineering bulk operations, and documentation.",
    )

    registered_tags = register_tool_surface(
        mcp,
        client_cls=Git,
        get_client=lambda: Git(),
        service="repository-manager",
        tool_registry=MCP_TOOL_REGISTRY,
    )

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
