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

``retest`` narrows a re-run to whatever :mod:`repository_manager.gate_ledger`
recorded as still-failing for a repo/stage instead of re-running the whole
wave -- see :mod:`repository_manager.gate_runner`'s module docstring for the
full contract (baseline missing/clean/failing, staleness degradation,
escalation-on-pass).

This module is now a THIN adapter: every action body lives in
:mod:`repository_manager.gate_runner` (``gate_runner.dispatch``), the one
place MCP and the ``--gate``/``--gate-retest`` CLI both call, so the two
front ends can never quietly diverge on what any of these actions do.
"""

from __future__ import annotations

from typing import Any, Literal

from agent_utilities.mcp.action_dispatch import resolve_action
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager import gate_runner
from repository_manager.gate_ledger import default_gate_ledger
from repository_manager.gates import run_gate_stage
from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_GATES_ACTIONS

_GATE_JOB_ACTION = "gate"


def _target_repos(
    adapter_context: McpToolContext, threads: int | None, repos: str | None
) -> list[tuple[str, str]]:
    """Resolve ``repos`` (comma-separated names/paths, or None = whole workspace)
    to ``[(repo_name, repo_path), ...]``, skipping repos with no pre-commit config.

    Target resolution is inherently MCP-specific (it needs a live ``Git``
    instance from ``adapter_context.get_git_instance``), so it stays here and
    is handed into ``gate_runner.dispatch`` as the ``resolve_targets``
    callable -- the actual eligibility filter it applies is the ONE shared
    ``gate_runner.targets_from_project_map``, not a second copy of it.
    """
    git = adapter_context.get_git_instance(threads=threads)
    return gate_runner.targets_from_project_map(git.project_map, repos)


def _make_submit_one(adapter_context: McpToolContext, *, stage: str, timeout: int):
    """Build this tool's ``submit_one(repo_name, path, **kwargs)`` callable.

    Shared shape with the CLI's own closure in ``cli_commands/parser.py``:
    a plain gate submission runs ``run_gate_stage`` directly; a retest
    submission asked to escalate-on-pass runs
    ``gate_runner.escalating_run_gate_stage`` instead, which submits the
    full-wave follow-up job (via this SAME ``submit_one``, called again from
    inside that job's own background thread) the instant the narrowed run
    passes.

    ``colocated=True`` is always passed to ``run_gate_stage``: this IS the
    pinned repository-manager-mcp process, so it is unconditionally the
    same-node arbiter ``task_queue.acquire``'s own docstring describes --
    unlike the CLI, which only has proof of that when the operator passes
    ``--same-node`` (see ``cli_commands/parser.py``'s equivalent closure).
    """

    def submit_one(
        repo_name: str,
        path: str,
        *,
        hook_ids: list[str] | None = None,
        trigger: str = "run",
        scope: str = "full_wave",
        _escalate_on_pass: bool = False,
        _same_node: bool = False,
    ) -> dict[str, Any]:
        del _same_node  # the MCP server process is always the colocated arbiter
        extra_job_data = {
            "repo_name": repo_name,
            "stage": stage,
            "trigger": trigger,
            "scope": scope,
            "hook_ids_requested": list(hook_ids or []),
            "colocated": True,
        }
        if _escalate_on_pass:
            return adapter_context.submit_job(
                _GATE_JOB_ACTION,
                gate_runner.escalating_run_gate_stage,
                path,
                stage,
                hook_ids,
                timeout=timeout,
                escalate_on_pass=True,
                repo_name=repo_name,
                submit_one=submit_one,
                trigger=trigger,
                scope=scope,
                colocated=True,
                record=True,
                _extra_job_data=extra_job_data,
            )
        return adapter_context.submit_job(
            _GATE_JOB_ACTION,
            run_gate_stage,
            path,
            stage,
            timeout=timeout,
            hook_ids=hook_ids,
            trigger=trigger,
            scope=scope,
            colocated=True,
            record=True,
            _extra_job_data=extra_job_data,
        )

    return submit_one


def register_gates_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the ``rm_gates`` two-tier gate-execution adapter.

    No ``ToolAnnotations(read_only_hint=True)`` is set here even though
    ``status``/``explain``/``profile`` ARE read-only: ``rm_gates`` is one
    tool multiplexing five actions, two of which (``run``, ``retest``)
    submit background jobs -- annotations apply to the whole tool, and
    marking the entire tool read-only would misdescribe those two. Skipping
    the annotation is the honest choice here, not an oversight.
    """

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
            "'retest' (narrow a re-run to the ledger's last-failing hooks "
            "per repo, escalating to a full wave on an all-pass), 'status' "
            "(roll-up + per-repo detail), 'explain' (condensed failure "
            "detail for one job/repo), 'profile' (measured per-hook "
            "timings), 'audit_fail_fast' (static scan for hook entries that "
            "would stop at the first failure -- DETECTION only, it cannot "
            "rewrite a repo's own opaque entry text), 'xdist_plan' (which "
            "repos could run pytest in parallel and why the rest cannot), "
            "'xdist_apply' (perform that rollout; dry-run unless dry_run=False)."
        ),
        fleet: bool = Field(
            default=False,
            description=(
                "For 'audit_fail_fast': scan every repository in the workspace "
                "rather than the ones named by 'repos'."
            ),
        ),
        dry_run: bool = Field(
            default=True,
            description=(
                "For 'xdist_apply': report what would change without writing. "
                "Defaults to True on purpose -- rewriting ~40 repositories' "
                "pre-commit configs must be an explicit choice, never the "
                "consequence of omitting an argument."
            ),
        ),
        stage: Literal["fast", "heavy"] = Field(
            default="fast",
            description=(
                "'fast' -> `pre-commit run --hook-stage pre-commit` (formatters/"
                "linters, no network/tests). 'heavy' -> `--hook-stage pre-push` "
                "(pytest, cargo, `uv lock --check`, ...). For 'run'/'retest'."
            ),
        ),
        repos: str | None = Field(
            default=None,
            description=(
                "Comma-separated repo names or absolute paths to target. "
                "Omit for 'run'/'retest' to target the whole workspace."
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
        escalate: bool = Field(
            default=True,
            description=(
                "'retest' only: when a repo's narrowed retest passes every "
                "requested hook, also submit a SECOND job for that repo's "
                "full wave (a narrowed pass alone is never sufficient "
                "evidence of shippability -- see gate_ledger.is_shippable)."
            ),
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

        def resolve_targets(
            threads_: int | None, repos_: str | None
        ) -> list[tuple[str, str]]:
            return _target_repos(adapter_context, threads_, repos_)

        submit_one = _make_submit_one(adapter_context, stage=stage, timeout=timeout)

        if action in ("run", "retest"):
            return gate_runner.dispatch(
                action,
                resolve_targets=resolve_targets,
                submit_one=submit_one,
                stage=stage,
                repos=repos,
                threads=threads,
                gate_ledger=default_gate_ledger(),
                escalate=escalate,
            )

        if action in ("audit_fail_fast", "xdist_plan", "xdist_apply"):
            return gate_runner.dispatch(
                action,
                repos=repos,
                fleet=fleet,
                dry_run=dry_run,
            )

        return gate_runner.dispatch(
            action,
            jobs=adapter_context.jobs,
            jobs_lock=adapter_context.jobs_lock,
            get_job_status=adapter_context.get_job_status,
            job_id=job_id,
            repo=repo,
            summary=summary,
            top_n=top_n,
        )
