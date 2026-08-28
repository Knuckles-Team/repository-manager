"""RMDD-20-shaped: the one MCP/CLI-shared gate-execution application service.

This module is the chokepoint both the ``rm_gates`` MCP tool
(:mod:`repository_manager.mcp_tools.gates`) and the ``--gate``/``--gate-retest``
CLI family (:mod:`repository_manager.cli_commands.parser`) call, following the
exact ``concept_actions.py``/``remote_worker_actions.py`` pattern: neither
front end constructs its own copy of this dispatch logic, so MCP and CLI can
never quietly diverge on what "run the gate" or "retest" actually means.

**``run``/``status``/``explain``/``profile`` are a pure refactor.** Their
bodies are ported VERBATIM out of what used to be inline in
``mcp_tools/gates.py``'s ``rm_gates`` tool function -- same branches, same
returned dict shapes, same messages. The only structural change is that every
MCP-specific dependency (the background job store, ``submit_job``, target
resolution against a live ``Git`` instance) is now a parameter/callable this
module is handed, rather than something it imports or reaches for on an
``adapter_context``. That is what lets the CLI drive the identical code path
with its own :class:`LocalJobStore` instead of a second, hand-copied
implementation of "submit N gate jobs and read them back".

**``retest`` is new, and it exists to close a measured, expensive gap.** On
2026-08-21 one repository's push cost roughly six hours because every one of
six failing pre-commit hooks was validated, on every fix, by re-running the
repo's ENTIRE 90-minute heavy wave -- there was no way to ask "just the hooks
that were failing." :mod:`repository_manager.gate_ledger` now durably records
what a wave found; ``retest`` is the first caller that reads it to decide
what actually needs to run again:

* No prior run recorded at all -> there is nothing to narrow against, so the
  FULL wave runs and the result says so plainly (``"baseline": "missing"``).
  Silently treating "never ran" as "ran clean" would be exactly the kind of
  fabricated evidence this codebase's C-10/H-12 conventions exist to forbid.
* A prior run recorded and nothing failing -> no job is submitted at all.
* A prior run recorded with failing hooks -> only those hook ids are
  requested, cutting a six-hook, 90-minute wave down to the handful of
  hooks that actually failed.
* Any of the ledger rows this decision would rely on is STALE (recorded
  against a different commit than the one on disk right now) -> the stale
  evidence is never trusted as a baseline; the target degrades to the full
  wave, same as "missing", and the response says which case it was.

A narrowed retest pass is deliberately never treated as proof the repo is
shippable on its own -- see :meth:`repository_manager.gate_ledger.GateLedger.is_shippable`'s
own docstring for the deadlock that survived 95 clean isolated runs. So when
``escalate`` (default ``True``) is set and every retested hook passes, a
SECOND job -- the full wave, ``trigger="retest-escalate"``,
``scope="full_wave"`` -- is submitted right behind it. That second submission
happens from *inside* the first job's own background thread the instant its
subprocess returns; no polling loop, no ``asyncio.gather``, no second
executor. ``submit_job``-style job stores in this codebase already run an
arbitrary callable in a worker thread (see ``mcp_server.py``'s
``_submit_job``), and calling ``submit_one`` again from within that callable
is just an ordinary (thread-safe) nested function call.

**Identity convention retest's ledger reads depend on.** Ledger rows are
keyed by ``(repo_id, stage)``. ``run_gate_stage`` (a concurrently-landed
change in ``gates.py``, since this module was first written) now records
every run itself, keyed by ``build_queue.stable_repository_id(repo_path)`` --
NOT a display basename, which two differently-located checkouts of the same
repo name would collide on. Retest's ledger reads use the exact same
``build_queue.stable_repository_id`` call for exactly that reason: a
different identity convention here would mean every retest reads a baseline
that never matches what ``run_gate_stage`` just wrote. The job-store-facing
``repo_name`` (``os.path.basename(path)``, used for ``_extra_job_data`` and
``rm_gates status``/``explain``/``profile`` lookups) is a SEPARATE, purely
cosmetic identity and is never used as a ledger key.

Neither ``asyncio.gather``, ``anyio``, nor ``ProcessPoolExecutor`` appear
anywhere in this module, and fastmcp's ``task=True`` is never used -- all
three are explicitly out of scope for this lane (no docket backend is
installed; ``task=True`` would silently no-op).
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - fixed argv only, never shell=True
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from threading import RLock
from typing import Any

from repository_manager import build_queue
from repository_manager.gate_ledger import GateLedger, default_gate_ledger
from repository_manager.gates import (
    GateStage,
    explain_gate_result,
    profile_gate_result,
    run_gate_stage,
)
from repository_manager.scan_models import RepoScanResult

__all__ = [
    "GATE_RUNNER_ACTIONS",
    "LocalJobStore",
    "dispatch",
    "escalating_run_gate_stage",
    "targets_from_project_map",
]

GATE_RUNNER_ACTIONS: tuple[str, ...] = (
    "run",
    "status",
    "explain",
    "profile",
    "retest",
    "audit_fail_fast",
    "xdist_plan",
    "xdist_apply",
)

#: The job-store ``action`` tag every gate job (run OR retest OR escalation)
#: carries, so a roll-up (``status``/``profile`` with no ``job_id``/``repo``)
#: is never diluted by install/build/validate jobs sharing the same store.
_GATE_JOB_ACTION = "gate"


def escalating_run_gate_stage(
    repo_path: str,
    stage: GateStage,
    hook_ids: list[str] | None,
    *,
    timeout: int | None,
    escalate_on_pass: bool,
    repo_name: str,
    submit_one: Callable[..., dict[str, Any]],
    same_node: bool = False,
    trigger: str = "retest",
    scope: str = "retest",
    colocated: bool | None = None,
    record: bool = True,
) -> RepoScanResult:
    """The callable a retest job actually runs in its background thread.

    Runs the requested (possibly narrowed) gate -- ``trigger``/``scope``/
    ``colocated``/``record`` are forwarded straight through to
    :func:`repository_manager.gates.run_gate_stage`, which now (as of the
    concurrent lane that wired :meth:`~repository_manager.gate_ledger.
    GateLedger.record_run` into it) actually records them: ``scope`` in
    particular is what :meth:`~repository_manager.gate_ledger.GateLedger.
    is_shippable` checks for ``"full_wave"``, so a narrowed retest job MUST
    be recorded with ``scope="retest"`` (never ``"full_wave"``) or it would
    silently satisfy a shippability check it never actually proved.

    If ``escalate_on_pass`` and every retested hook passed, submits the
    SECOND, full-wave job (``hook_ids=None, trigger="retest-escalate",
    scope="full_wave"``) right here, the instant the narrowed subprocess
    returns -- inside this same background thread, not a poller watching for
    it. Every job store in this codebase (the real MCP one and
    :class:`LocalJobStore`) is already thread-safe, so calling ``submit_one``
    again from here is an ordinary nested function call, not a new
    concurrency primitive.
    """
    result = run_gate_stage(
        repo_path,
        stage,
        hook_ids=hook_ids,
        timeout=timeout,
        trigger=trigger,
        scope=scope,
        colocated=colocated,
        record=record,
    )
    if escalate_on_pass and result.success:
        submit_one(
            repo_name,
            repo_path,
            hook_ids=None,
            trigger="retest-escalate",
            scope="full_wave",
            _escalate_on_pass=False,
            _same_node=same_node,
        )
    return result


# --------------------------------------------------------------------------
# Shared target resolution
# --------------------------------------------------------------------------


def targets_from_project_map(
    project_map: dict[str, str], repos: str | None
) -> list[tuple[str, str]]:
    """Filter a Git-style ``{url: path}`` project map to gate-eligible targets.

    A repo is eligible when it declares a ``.pre-commit-config.yaml`` -- the
    same rule ``mcp_tools/gates.py``'s ``_target_repos`` already enforced, now
    factored out so both front ends' "skip a repo with no config" means
    exactly one thing, not two.
    """
    wanted = (
        {r.strip() for r in repos.replace(" ", "").split(",") if r.strip()}
        if repos
        else None
    )
    targets: list[tuple[str, str]] = []
    for _url, path in project_map.items():
        repo_name = os.path.basename(path)
        if wanted and repo_name not in wanted:
            continue
        if not os.path.exists(os.path.join(path, ".pre-commit-config.yaml")):
            continue
        targets.append((repo_name, path))
    return targets


def _current_git_sha(repo_path: str) -> str:
    """Best-effort ``git rev-parse HEAD``; never raises, "" means unknown.

    An empty return is deliberately NOT treated by callers as "the tree has
    no commit" -- it is treated as "freshness cannot be proven", which
    degrades a retest to the full wave exactly like a ledger-reported stale
    row does. Silently skipping the staleness check (which is what
    :meth:`~repository_manager.gate_ledger.GateLedger.latest_hooks` does when
    handed ``git_sha=""``) would let an unresolved sha masquerade as "no
    staleness concern", which is the opposite of honest.
    """
    try:
        completed = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["git", "rev-parse", "HEAD"],
            cwd=repo_path,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception:  # noqa: BLE001 - unresolved sha is a legitimate outcome
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


# --------------------------------------------------------------------------
# Shared fan-out
# --------------------------------------------------------------------------


def _fan_out(
    targets: Sequence[tuple[str, str]],
    submit_one: Callable[..., Any],
    *,
    max_workers: int | None,
    **submit_kwargs: Any,
) -> dict[str, Any]:
    """Call ``submit_one(repo_name, path, **submit_kwargs)`` for every target
    on one bounded :class:`ThreadPoolExecutor`, returning ``{repo_name: result}``.

    The one shared fan-out both front ends now use -- it replaces the
    ``ThreadPoolExecutor`` that used to be duplicated inline in
    ``cli_commands/parser.py``'s ``--gate`` block. What "submitting" means
    differs by caller: for the MCP adapter ``submit_one`` just registers a
    background job and returns immediately (so this fan-out barely matters --
    each call is already fast); for the CLI's :class:`LocalJobStore`,
    ``submit_one`` runs the gate SYNCHRONOUSLY inline, so parallelizing these
    calls is exactly what makes the CLI's ``--gate``/``--gate-retest`` runs
    parallel across repos, same as before this refactor.
    """
    results: dict[str, Any] = {}
    if not targets:
        return results
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(submit_one, repo_name, path, **submit_kwargs): repo_name
            for repo_name, path in targets
        }
        for future, repo_name in futures.items():
            results[repo_name] = future.result()
    return results


# --------------------------------------------------------------------------
# CLI's own minimal job store
# --------------------------------------------------------------------------


class LocalJobStore:
    """In-process stand-in for the MCP job store, so the CLI can drive the
    exact same ``dispatch`` code path without an MCP server.

    ``submit_job`` deliberately runs ``func`` SYNCHRONOUSLY, inline, rather
    than handing it to yet another background executor -- when called
    through :func:`_fan_out`, many ``submit_job`` calls already run
    concurrently on that fan-out's own bounded pool, which is exactly the
    executor the CLI's old inline ``--gate`` block used directly. By the time
    a ``dispatch("run"/"retest", ...)`` call returns, every job it created
    here is already terminal -- there is no polling to do, and none is
    provided.
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.jobs_lock = RLock()
        self._counter = 0

    def submit_job(
        self,
        action: str,
        func: Callable[..., Any],
        *args: Any,
        _extra_job_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> dict[str, str]:
        with self.jobs_lock:
            self._counter += 1
            job_id = f"cli-{action}-{self._counter}"
            self.jobs[job_id] = {
                "action": action,
                "status": "running",
                "started_at": datetime.now(UTC).isoformat(),
                "result": None,
                "error": None,
                **(_extra_job_data or {}),
            }
        try:
            result = func(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - store the failure, never raise
            with self.jobs_lock:
                self.jobs[job_id]["status"] = "failed"
                self.jobs[job_id]["error"] = str(exc)
                self.jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
            return {"job_id": job_id}
        with self.jobs_lock:
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["completed_at"] = datetime.now(UTC).isoformat()
        return {"job_id": job_id}

    def get_job_status(self, job_id: str, *, summary: bool = True) -> dict[str, Any]:
        with self.jobs_lock:
            job = self.jobs.get(job_id)
            if job is None:
                return {"status": "error", "message": f"Job '{job_id}' not found."}
            if summary:
                return {"job_id": job_id, "status": job["status"]}
            return {"job_id": job_id, **job}


# --------------------------------------------------------------------------
# status/explain/profile shared helpers (ported verbatim from mcp_tools/gates.py)
# --------------------------------------------------------------------------


def _latest_gate_jobs(
    jobs: dict[str, dict[str, Any]], jobs_lock: RLock
) -> dict[str, dict[str, Any]]:
    """The latest ``action="gate"`` job per repo, keyed by job_id."""
    latest_by_repo: dict[str, tuple[str, dict[str, Any]]] = {}
    with jobs_lock:
        for jid, job in jobs.items():
            if job.get("action") != _GATE_JOB_ACTION:
                continue
            repo_name = job.get("repo_name")
            if not repo_name:
                continue
            cur = latest_by_repo.get(repo_name)
            if cur is None or (job.get("started_at") or "") >= (
                cur[1].get("started_at") or ""
            ):
                latest_by_repo[repo_name] = (jid, job)
    return {jid: job for jid, job in latest_by_repo.values()}


def _resolve_target_result(
    jobs: dict[str, dict[str, Any]],
    jobs_lock: RLock,
    *,
    job_id: str | None,
    repo: str | None,
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Resolve an explicit ``job_id`` or ``repo`` name to its gate job record.

    Returns ``(job_id, job, error_message)``.
    """
    if job_id:
        with jobs_lock:
            job = jobs.get(job_id)
        if job is None:
            return None, None, f"Job '{job_id}' not found."
        return job_id, job, None
    if repo:
        for jid, job in _latest_gate_jobs(jobs, jobs_lock).items():
            if job.get("repo_name") == repo:
                return jid, job, None
        return None, None, f"No gate job found for repo '{repo}'."
    return None, None, "Provide 'job_id' or 'repo'."


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def _dispatch_audit_fail_fast(kwargs: dict[str, Any]) -> dict[str, Any]:
    from repository_manager import fail_fast_audit

    return fail_fast_audit.dispatch(
        "check_fleet" if kwargs.get("fleet") else "check",
        **_fleet_config_kwargs(kwargs),
    )


def _dispatch_xdist_plan(kwargs: dict[str, Any]) -> dict[str, Any]:
    from repository_manager import xdist_rollout

    return xdist_rollout.dispatch("plan", **_fleet_config_kwargs(kwargs))


def _dispatch_xdist_apply(kwargs: dict[str, Any]) -> dict[str, Any]:
    from repository_manager import xdist_rollout

    return xdist_rollout.dispatch("apply", **_fleet_config_kwargs(kwargs))


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """Resolve and execute one gate action; MCP and CLI share this exactly.

    Unlike ``concept_actions.dispatch``/``remote_worker_actions.dispatch``,
    the result is NOT wrapped in ``{"ok": True, ...}`` -- this is a pure
    refactor of ``rm_gates``' existing action bodies, and preserving their
    exact historical return shape (``{"status": ...}``, ``{"summary": ...}``,
    etc.) is the whole point: behaviour must not drift.

    The fleet-configuration actions (``audit_fail_fast``, ``xdist_plan``,
    ``xdist_apply``) delegate straight to their own modules' ``dispatch``,
    which are already the shared CLI/MCP implementation layer -- the same
    shape ``concept_actions``/``remote_worker_actions`` use. They are routed
    through ``rm_gates`` rather than given a tool of their own because both
    answer questions ABOUT the gate configuration, and a reader looking for
    "why is this gate slow / can it stop early" should not have to know they
    live somewhere else.

    Unlike the five gate-execution actions, these DO return the ``{"ok":
    ...}`` envelope their own modules define. That inconsistency is
    deliberate and preserved rather than smoothed over: the five
    gate-execution actions are a verbatim port whose historical return shape
    callers already depend on, and rewrapping either side to match the other
    would change a contract to win a cosmetic point.
    """

    from agent_utilities.mcp.action_dispatch import resolve_action

    resolved = resolve_action(
        action, GATE_RUNNER_ACTIONS, service="repository-manager-gates"
    )
    if isinstance(resolved, dict):
        return resolved

    handler = _GATE_DISPATCH_TABLE.get(resolved)
    if handler is not None:
        return handler(kwargs)
    raise AssertionError(f"unhandled resolved gate action {resolved!r}")


def _fleet_config_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Translate `rm_gates`' repo selection into the config modules' arguments.

    `rm_gates` speaks `repos` (a comma-separated selector) because every one of
    its execution actions does; `fail_fast_audit`/`xdist_rollout` speak
    `repo_path`/`repo_paths`. Doing that translation HERE, once, keeps the two
    modules free of any knowledge of the MCP tool's argument conventions -- they
    stay plain libraries usable from a script, which is what let them be tested
    without a server in the first place.

    `dry_run` deliberately defaults to True for `xdist_apply`: a fleet-wide
    rewrite of 40 repositories' pre-commit configs is not something a caller
    should be able to trigger by omitting an argument.
    """

    resolved: dict[str, Any] = {}
    repo_paths = kwargs.get("repo_paths")
    if repo_paths is None and kwargs.get("repos"):
        repo_paths = [
            part.strip() for part in str(kwargs["repos"]).split(",") if part.strip()
        ]
    if repo_paths is not None:
        resolved["repo_paths"] = repo_paths
        if repo_paths:
            resolved["repo_path"] = repo_paths[0]
    if kwargs.get("repo_path") is not None:
        resolved["repo_path"] = kwargs["repo_path"]
    if kwargs.get("fleet_root") is not None:
        resolved["fleet_root"] = kwargs["fleet_root"]
    resolved["dry_run"] = bool(kwargs.get("dry_run", True))
    return resolved


def _dispatch_run(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Verbatim port of ``rm_gates``'s ``run`` branch.

    ``resolve_targets(threads, repos) -> [(repo_name, path), ...]`` and
    ``submit_one(repo_name, path, *, hook_ids=None, trigger="run",
    scope="full_wave") -> {"job_id": ...}`` are supplied by the caller (the
    MCP adapter's real background job store, or the CLI's
    :class:`LocalJobStore`).
    """
    resolve_targets: Callable[[int | None, str | None], list[tuple[str, str]]] = kwargs[
        "resolve_targets"
    ]
    submit_one: Callable[..., dict[str, Any]] = kwargs["submit_one"]
    stage = kwargs.get("stage", "fast")
    repos = kwargs.get("repos")
    threads = kwargs.get("threads")
    max_workers = kwargs.get("max_workers", threads)

    targets = resolve_targets(threads, repos)
    if not targets:
        return {
            "status": "clean",
            "message": "No repositories with a .pre-commit-config.yaml matched.",
            "queued_count": 0,
        }

    submitted = _fan_out(
        targets, submit_one, max_workers=max_workers, trigger="run", scope="full_wave"
    )
    job_ids = {repo_name: result["job_id"] for repo_name, result in submitted.items()}
    return {
        "status": "submitted",
        "stage": stage,
        "queued_count": len(job_ids),
        "jobs": job_ids,
        "message": (
            f"{len(job_ids)} {stage} gate job(s) submitted in parallel "
            "(one job per repo). Poll action='status'."
        ),
    }


def _single_job_status(
    jobs: dict[str, dict[str, Any]],
    jobs_lock: RLock,
    get_job_status: Callable[..., dict[str, Any]],
    job_id: str,
    summary: bool,
) -> dict[str, Any]:
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        return {"status": "error", "message": f"Job '{job_id}' not found."}
    return get_job_status(job_id, summary=summary)


def _record_failure_detail(
    jid: str,
    st: str,
    job: dict[str, Any],
    result: Any,
    job_repo_name: str,
    details: dict[str, Any],
    summary: bool,
) -> None:
    detail: dict[str, Any] = {"job_id": jid, "status": st}
    if isinstance(result, RepoScanResult):
        detail["failures"] = [
            f"Hook '{h.hook_id}' failed" for h in result.hooks if not h.passed
        ]
        detail["explain"] = explain_gate_result(result)
    elif job.get("error"):
        detail["error"] = job["error"]
    if not summary:
        details[job_repo_name] = detail
    elif summary:
        details[job_repo_name] = {"job_id": jid}


def _process_gate_job(
    jid: str,
    job: dict[str, Any],
    counts: dict[str, int],
    failed_repos: list[str],
    running_repos: list[str],
    details: dict[str, Any],
    summary: bool,
) -> None:
    st = job["status"]
    job_repo_name = job.get("repo_name")
    if st in ("running", "queued", "pending"):
        counts["running"] += 1
        if job_repo_name:
            running_repos.append(job_repo_name)
        return
    result = job.get("result")
    passed = isinstance(result, RepoScanResult) and result.success
    if st == "completed":
        counts["completed"] += 1
        counts["passed" if passed else "failed"] += 1
    elif st == "failed":
        counts["failed"] += 1
    if not passed and job_repo_name:
        failed_repos.append(job_repo_name)
        _record_failure_detail(jid, st, job, result, job_repo_name, details, summary)


def _gate_jobs_summary(
    gate_jobs: dict[str, dict[str, Any]], summary: bool
) -> dict[str, Any]:
    counts = {"completed": 0, "running": 0, "failed": 0, "passed": 0}
    failed_repos: list[str] = []
    running_repos: list[str] = []
    details: dict[str, Any] = {}
    for jid, job in gate_jobs.items():
        _process_gate_job(
            jid, job, counts, failed_repos, running_repos, details, summary
        )
    return {
        "summary": {**counts, "total": len(gate_jobs)},
        "failed_projects": sorted(failed_repos),
        "running_projects": sorted(running_repos),
        "failed_details": details,
    }


def _dispatch_status(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Verbatim port of ``rm_gates``'s ``status`` branch."""
    jobs: dict[str, dict[str, Any]] = kwargs["jobs"]
    jobs_lock: RLock = kwargs["jobs_lock"]
    get_job_status: Callable[..., dict[str, Any]] = kwargs["get_job_status"]
    job_id = kwargs.get("job_id")
    summary = kwargs.get("summary", True)

    if job_id:
        return _single_job_status(jobs, jobs_lock, get_job_status, job_id, summary)

    gate_jobs = _latest_gate_jobs(jobs, jobs_lock)
    if not gate_jobs:
        return {"status": "empty", "message": "No gate jobs found."}
    return _gate_jobs_summary(gate_jobs, summary)


def _dispatch_explain(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Verbatim port of ``rm_gates``'s ``explain`` branch."""
    jobs: dict[str, dict[str, Any]] = kwargs["jobs"]
    jobs_lock: RLock = kwargs["jobs_lock"]
    job_id = kwargs.get("job_id")
    repo = kwargs.get("repo")

    resolved_job_id, job, error = _resolve_target_result(
        jobs, jobs_lock, job_id=job_id, repo=repo
    )
    if error:
        return {"status": "error", "message": error}
    assert job is not None
    if job["status"] in ("running", "queued", "pending"):
        return {
            "status": job["status"],
            "job_id": resolved_job_id,
            "message": "Still running.",
        }
    result = job.get("result")
    if not isinstance(result, RepoScanResult):
        return {
            "status": job["status"],
            "job_id": resolved_job_id,
            "message": job.get("error") or "No gate result available.",
        }
    return {
        "job_id": resolved_job_id,
        "repo": job.get("repo_name"),
        "stage": job.get("stage"),
        "passed": result.success,
        "explain": explain_gate_result(result),
    }


def _dispatch_profile(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Verbatim port of ``rm_gates``'s ``profile`` branch."""
    jobs: dict[str, dict[str, Any]] = kwargs["jobs"]
    jobs_lock: RLock = kwargs["jobs_lock"]
    job_id = kwargs.get("job_id")
    repo = kwargs.get("repo")
    top_n = kwargs.get("top_n", 15)

    if job_id or repo:
        resolved_job_id, job, error = _resolve_target_result(
            jobs, jobs_lock, job_id=job_id, repo=repo
        )
        if error:
            return {"status": "error", "message": error}
        assert job is not None
        result = job.get("result")
        if not isinstance(result, RepoScanResult):
            return {
                "status": job["status"],
                "job_id": resolved_job_id,
                "message": "No completed gate result to profile yet.",
            }
        return {
            "job_id": resolved_job_id,
            "repo": job.get("repo_name"),
            "stage": job.get("stage"),
            "duration_s": result.duration_s,
            "hooks": profile_gate_result(result),
        }

    all_hooks: list[dict[str, Any]] = []
    per_repo: list[dict[str, Any]] = []
    for job in _latest_gate_jobs(jobs, jobs_lock).values():
        result = job.get("result")
        if not isinstance(result, RepoScanResult):
            continue
        job_repo_name = job.get("repo_name")
        per_repo.append(
            {
                "repo": job_repo_name,
                "stage": job.get("stage"),
                "duration_s": result.duration_s,
            }
        )
        for h in profile_gate_result(result):
            all_hooks.append({"repo": job_repo_name, **h})
    all_hooks.sort(key=lambda h: h.get("duration_s") or 0.0, reverse=True)
    per_repo.sort(key=lambda r: r.get("duration_s") or 0.0, reverse=True)
    return {
        "measured_gate_jobs": len(per_repo),
        "slowest_hooks": all_hooks[:top_n],
        "slowest_repos": per_repo[:top_n],
    }


# --------------------------------------------------------------------------
# retest
# --------------------------------------------------------------------------


def _retest_plan(
    *, repo_id: str, repo_path: str, stage: str, ledger: GateLedger
) -> dict[str, Any]:
    """Decide, from the ledger alone, what one repo's retest should do.

    Returns ``{"baseline", "hook_ids", "stale"}``. ``hook_ids is None`` means
    "run the full wave" (either nothing to narrow against, or narrowing
    would not be honest); a non-``None`` list -- possibly empty, meaning
    "nothing failing" -- means "narrow to exactly these hook ids, or submit
    nothing if the list is empty".
    """
    if not ledger.has_any_run(repo_id, stage):
        return {"baseline": "missing", "hook_ids": None, "stale": False}

    git_sha = _current_git_sha(repo_path)
    # An unresolved sha means freshness cannot be proven at all -- treat that
    # exactly like a stale row rather than silently skipping the check (see
    # `_current_git_sha`'s docstring).
    hooks = ledger.latest_hooks(repo_id, stage, git_sha=git_sha)
    stale = bool(git_sha) is False or any(h.stale for h in hooks)
    failing = [h for h in hooks if h.failed]

    if stale:
        # Report what the (untrustworthy) ledger nominally said, but never
        # act on it: force the full wave regardless of whether it looked
        # clean or failing.
        return {
            "baseline": "failing" if failing else "clean",
            "hook_ids": None,
            "stale": True,
        }
    if not failing:
        return {"baseline": "clean", "hook_ids": None, "stale": False}
    return {
        "baseline": "failing",
        "hook_ids": sorted(h.hook_id for h in failing),
        "stale": False,
    }


def _submit_retest_target(
    repo_name: str,
    path: str,
    *,
    plans: dict[str, dict[str, Any]],
    submit_gate: Callable[..., dict[str, Any]],
    stage: str,
    escalate: bool,
    same_node: bool,
) -> dict[str, Any]:
    # NOTE: this keyword-only param is named `submit_gate`, not `submit_one`
    # like its caller's local variable -- `_fan_out`'s OWN second positional
    # parameter is itself named `submit_one` (the callable to fan out, which
    # here is this very function), so passing a `submit_one=...` kwarg
    # through `_fan_out(..., **submit_kwargs)` would collide with that
    # positional binding (mypy caught this: "gets multiple values for
    # keyword argument 'submit_one'").
    plan = plans[repo_name]
    entry: dict[str, Any] = {
        "stage": stage,
        "baseline": plan["baseline"],
        "retest_hook_ids": plan["hook_ids"],
        "retest_job_id": None,
        "escalate": False,
        "stale": plan["stale"],
    }
    if plan["baseline"] == "clean" and not plan["stale"]:
        # Nothing failing and nothing stale to distrust -- no work to do.
        return entry

    hook_ids = plan["hook_ids"]  # None => full wave
    narrowed = hook_ids is not None and len(hook_ids) > 0
    submitted = submit_gate(
        repo_name,
        path,
        hook_ids=hook_ids,
        trigger="retest" if narrowed else "retest-full",
        scope="retest" if narrowed else "full_wave",
        _escalate_on_pass=(escalate and narrowed),
        _same_node=same_node,
    )
    entry["retest_job_id"] = submitted["job_id"]
    entry["escalate"] = bool(escalate and narrowed)
    return entry


def _build_retest_plans(
    targets: list[tuple[str, str]], stage: str, ledger: GateLedger
) -> dict[str, dict[str, Any]]:
    return {
        repo_name: _retest_plan(
            repo_id=build_queue.stable_repository_id(path),
            repo_path=path,
            stage=stage,
            ledger=ledger,
        )
        for repo_name, path in targets
    }


def _retest_summary(per_repo: dict[str, Any]) -> dict[str, Any]:
    submitted_count = sum(1 for v in per_repo.values() if v["retest_job_id"])
    stale_count = sum(1 for v in per_repo.values() if v["stale"])
    missing_count = sum(1 for v in per_repo.values() if v["baseline"] == "missing")
    return {
        "status": "submitted" if submitted_count else "clean",
        "targets": per_repo,
        "message": (
            f"{submitted_count} retest job(s) submitted across {len(per_repo)} "
            f"target(s) ({missing_count} with no prior baseline, {stale_count} "
            "with a stale baseline degraded to the full wave). Poll "
            "action='status'."
        ),
    }


def _dispatch_retest(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Narrow a repo's gate re-run to what the ledger says last failed.

    See this module's docstring for the full baseline/staleness/escalation
    contract. ``submit_one`` here is the SAME callable ``run`` uses, called
    with ``hook_ids``/``trigger``/``scope`` -- ``run_gate_stage`` itself now
    records each run into the ledger (see ``gates.py``'s module docstring),
    so this module never has to call
    :meth:`~repository_manager.gate_ledger.GateLedger.record_run` directly;
    it only has to supply honest ``trigger``/``scope`` values for that
    recording to mean anything (a narrowed retest MUST record
    ``scope="retest"``, never ``"full_wave"`` -- see
    :func:`escalating_run_gate_stage`'s docstring for why).

    ``same_node`` (default ``False``) is the CLI's ``--same-node`` colocation
    assertion. This function's own ledger READS never depend on it (whatever
    process calls ``dispatch`` reads its own local ``GateLedger`` directly --
    there is no cross-node read here to prove anything about). It IS load-
    bearing one layer down, though: each adapter's ``submit_one`` maps it
    onto ``run_gate_stage``'s ``colocated`` parameter, which gates whether a
    HEAVY-tier run against a Cargo-based repo takes the ``task_queue``
    ``"build"`` reservation that prevents concurrent ``cargo`` invocations
    from corrupting a shared ``CARGO_TARGET_DIR``. The MCP adapter always
    asserts colocation (the MCP server process IS the pinned same-node
    arbiter -- see ``task_queue.acquire``'s own docstring); the CLI asserts
    it only when the operator passes ``--same-node``.
    """
    resolve_targets: Callable[[int | None, str | None], list[tuple[str, str]]] = kwargs[
        "resolve_targets"
    ]
    submit_one: Callable[..., dict[str, Any]] = kwargs["submit_one"]
    stage = kwargs.get("stage", "fast")
    repos = kwargs.get("repos")
    threads = kwargs.get("threads")
    max_workers = kwargs.get("max_workers", threads)
    ledger: GateLedger = kwargs.get("gate_ledger") or default_gate_ledger()
    escalate = bool(kwargs.get("escalate", True))
    same_node = bool(kwargs.get("same_node", False))

    targets = resolve_targets(threads, repos)
    if not targets:
        return {
            "status": "clean",
            "targets": {},
            "message": "No repositories with a .pre-commit-config.yaml matched.",
        }

    plans = _build_retest_plans(targets, stage, ledger)

    per_repo = _fan_out(
        targets,
        _submit_retest_target,
        max_workers=max_workers,
        plans=plans,
        submit_gate=submit_one,
        stage=stage,
        escalate=escalate,
        same_node=same_node,
    )
    return _retest_summary(per_repo)


# Built here, after every handler above is defined, and looked up by name
# (not referenced until `dispatch()` is actually CALLED, well after module
# import completes) -- see `dispatch()`'s own docstring for why the five
# gate-execution actions and the three fleet-configuration actions return
# different shapes.
_GATE_DISPATCH_TABLE: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {
    "run": _dispatch_run,
    "status": _dispatch_status,
    "explain": _dispatch_explain,
    "profile": _dispatch_profile,
    "retest": _dispatch_retest,
    "audit_fail_fast": _dispatch_audit_fail_fast,
    "xdist_plan": _dispatch_xdist_plan,
    "xdist_apply": _dispatch_xdist_apply,
}
