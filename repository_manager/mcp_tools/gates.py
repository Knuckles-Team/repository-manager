"""``rm_gates`` -- the tool that drives the two-tier gate model (GOC-59/60).

``fast`` runs the repo's ``pre-commit``-stage hooks (formatters/linters, no
network/tests/compilers, target <=5s warm); ``heavy`` runs its ``pre-push``-stage
hooks (pytest, cargo, ``uv lock --check``, ...). Both map directly onto
:func:`repository_manager.gates.run_gate_stage`'s ``--hook-stage`` argument --
the single execution engine also used by ``rm_projects action=validate`` and
the automatic pre-push gate (``Git._gate_before_push``).

``run`` fans a background job out per repo onto the existing bounded executor
(``adapter_context.submit_job``, sized to 20% CPU / 20% RAM at
``mcp_server.py``) -- the same parallel job machinery ``rm_projects
action=validate`` uses, not a second pool. ``status``/``explain``/``profile``
all read back from that same job store, scoped to this tool's own
``action="gate"`` jobs so a gate roll-up is never diluted by install/build/
validate jobs sharing the same background-job store.
"""

from __future__ import annotations

import os
from typing import Any, Literal

from agent_utilities.mcp.action_dispatch import resolve_action
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.gates import (
    explain_gate_result,
    profile_gate_result,
    run_gate_stage,
)
from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_GATES_ACTIONS
from repository_manager.scan_models import RepoScanResult

_GATE_JOB_ACTION = "gate"


def _target_repos(
    adapter_context: McpToolContext, threads: int | None, repos: str | None
) -> list[tuple[str, str]]:
    """Resolve ``repos`` (comma-separated names/paths, or None = whole workspace)
    to ``[(repo_name, repo_path), ...]``, skipping repos with no pre-commit config."""
    git = adapter_context.get_git_instance(threads=threads)
    wanted = (
        {r.strip() for r in repos.replace(" ", "").split(",") if r.strip()}
        if repos
        else None
    )
    targets: list[tuple[str, str]] = []
    for _url, path in git.project_map.items():
        repo_name = os.path.basename(path)
        if wanted and repo_name not in wanted:
            continue
        if not os.path.exists(os.path.join(path, ".pre-commit-config.yaml")):
            continue
        targets.append((repo_name, path))
    return targets


def _latest_gate_jobs(adapter_context: McpToolContext) -> dict[str, dict[str, Any]]:
    """The latest ``action="gate"`` job per repo, keyed by job_id."""
    latest_by_repo: dict[str, tuple[str, dict[str, Any]]] = {}
    with adapter_context.jobs_lock:
        for jid, job in adapter_context.jobs.items():
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
    adapter_context: McpToolContext, *, job_id: str | None, repo: str | None
) -> tuple[str | None, dict[str, Any] | None, str | None]:
    """Resolve an explicit ``job_id`` or ``repo`` name to its gate job record.

    Returns ``(job_id, job, error_message)``.
    """
    if job_id:
        with adapter_context.jobs_lock:
            job = adapter_context.jobs.get(job_id)
        if job is None:
            return None, None, f"Job '{job_id}' not found."
        return job_id, job, None
    if repo:
        for jid, job in _latest_gate_jobs(adapter_context).items():
            if job.get("repo_name") == repo:
                return jid, job, None
        return None, None, f"No gate job found for repo '{repo}'."
    return None, None, "Provide 'job_id' or 'repo'."


def register_gates_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the ``rm_gates`` two-tier gate-execution adapter."""

    adapter_context = context or from_server()

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_gates(
        # `action` stays `str` (not `Literal`), matching every sibling rm_*
        # tool: `resolve_action` below accepts the discovery keywords
        # ('list_actions'/'help'/'actions') that a Literal-typed FastMCP
        # signature would reject at schema validation before ever reaching
        # this body. `stage` below (this tool's only other closed-set
        # argument, and not part of the discovery convention) IS Literal --
        # narrower gain, no discovery conflict.
        action: str = Field(
            description="Action: 'run' (submit fast/heavy gate jobs), "
            "'status' (roll-up + per-repo detail), 'explain' (condensed "
            "failure detail for one job/repo), 'profile' (measured per-hook "
            "timings)."
        ),
        stage: Literal["fast", "heavy"] = Field(
            default="fast",
            description=(
                "'fast' -> `pre-commit run --hook-stage pre-commit` (formatters/"
                "linters, no network/tests). 'heavy' -> `--hook-stage pre-push` "
                "(pytest, cargo, `uv lock --check`, ...). For 'run'."
            ),
        ),
        repos: str | None = Field(
            default=None,
            description=(
                "Comma-separated repo names or absolute paths to target. "
                "Omit for 'run' to target the whole workspace."
            ),
        ),
        threads: int | None = Field(default=None, description="Parallel workers."),
        timeout: int = Field(
            default=600, description="Per-repo pre-commit timeout in seconds."
        ),
        job_id: str | None = Field(
            default=None,
            description="Target one gate job for 'status'/'explain'/'profile'.",
        ),
        repo: str | None = Field(
            default=None,
            description=(
                "Target one repo's latest gate job for 'status'/'explain'/"
                "'profile' (alternative to job_id)."
            ),
        ),
        summary: bool = Field(
            default=True,
            description="'status': compact roll-up (counts + failed set) vs the full per-job detail.",
        ),
        top_n: int = Field(
            default=15,
            description="'profile' with no job_id/repo: how many slowest hooks to report fleet-wide.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict[str, Any]:
        """Run and inspect the two-tier (fast/heavy) pre-commit gate across repos.

        The single tool that can actually drive both tiers of the two-tier gate
        model: ``stage="fast"`` exercises a repo's ``pre-commit``-stage hooks,
        ``stage="heavy"`` exercises its ``pre-push``-stage hooks (pytest, cargo,
        ``uv lock --check``, ...) via ``pre-commit run --hook-stage <stage>``.
        """
        del ctx
        resolved = resolve_action(
            action, RM_GATES_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved  # type: ignore[assignment]

        if action == "run":
            targets = _target_repos(adapter_context, threads, repos)
            if not targets:
                return {
                    "status": "clean",
                    "message": "No repositories with a .pre-commit-config.yaml matched.",
                    "queued_count": 0,
                }
            job_ids: dict[str, str] = {}
            for repo_name, path in targets:
                submitted = adapter_context.submit_job(
                    _GATE_JOB_ACTION,
                    run_gate_stage,
                    path,
                    stage,
                    timeout=timeout,
                    _extra_job_data={"repo_name": repo_name, "stage": stage},
                )
                job_ids[repo_name] = submitted["job_id"]
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

        if action == "status":
            if job_id:
                with adapter_context.jobs_lock:
                    job = adapter_context.jobs.get(job_id)
                if job is None:
                    return {"status": "error", "message": f"Job '{job_id}' not found."}
                return adapter_context.get_job_status(job_id, summary=summary)

            gate_jobs = _latest_gate_jobs(adapter_context)
            if not gate_jobs:
                return {"status": "empty", "message": "No gate jobs found."}
            counts = {"completed": 0, "running": 0, "failed": 0, "passed": 0}
            failed_repos: list[str] = []
            running_repos: list[str] = []
            details: dict[str, Any] = {}
            for jid, job in gate_jobs.items():
                st = job["status"]
                job_repo_name = job.get("repo_name")
                if st in ("running", "queued", "pending"):
                    counts["running"] += 1
                    if job_repo_name:
                        running_repos.append(job_repo_name)
                    continue
                result = job.get("result")
                passed = isinstance(result, RepoScanResult) and result.success
                if st == "completed":
                    counts["completed"] += 1
                    counts["passed" if passed else "failed"] += 1
                elif st == "failed":
                    counts["failed"] += 1
                if not passed and job_repo_name:
                    failed_repos.append(job_repo_name)
                    detail: dict[str, Any] = {"job_id": jid, "status": st}
                    if isinstance(result, RepoScanResult):
                        detail["failures"] = [
                            f"Hook '{h.hook_id}' failed"
                            for h in result.hooks
                            if not h.passed
                        ]
                        detail["explain"] = explain_gate_result(result)
                    elif job.get("error"):
                        detail["error"] = job["error"]
                    if not summary:
                        details[job_repo_name] = detail
                    elif summary:
                        details[job_repo_name] = {"job_id": jid}
            return {
                "summary": {**counts, "total": len(gate_jobs)},
                "failed_projects": sorted(failed_repos),
                "running_projects": sorted(running_repos),
                "failed_details": details,
            }

        if action == "explain":
            resolved_job_id, job, error = _resolve_target_result(
                adapter_context, job_id=job_id, repo=repo
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

        if action == "profile":
            if job_id or repo:
                resolved_job_id, job, error = _resolve_target_result(
                    adapter_context, job_id=job_id, repo=repo
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

            # Fleet-wide: the slowest hooks across every completed gate job,
            # and every repo's total gate duration -- what to trim, at scale.
            all_hooks: list[dict[str, Any]] = []
            per_repo: list[dict[str, Any]] = []
            for job in _latest_gate_jobs(adapter_context).values():
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

        return {"status": "error", "message": f"Unhandled action '{action}'."}
