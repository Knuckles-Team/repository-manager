"""Merge-queue MCP adapter."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext
from repository_manager.mcp_tools.contracts import RM_MERGE_QUEUE_ACTIONS


def register_merge_queue_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the serialized merge-queue adapter."""

    del context

    @mcp.tool(tags={"workspace_management", "project_manager", "git_operations"})
    async def rm_merge_queue(
        action: str = Field(
            description=(
                "Action: 'enqueue' (offer a branch for landing), 'status' (queue "
                "depth, order, recent outcomes), 'withdraw' (pull a candidate "
                "back out), 'run' (drain a batch under the serializing lease), "
                "'config' (show and validate this repo's gate declaration)."
            )
        ),
        repo_path: str | None = Field(
            default=None,
            description=(
                "Any working tree of the target repository. This is what makes the "
                "queue cross-project: one server, N repositories, each with its own "
                "lease, candidate store, and gates. Defaults to the server's cwd."
            ),
        ),
        branch: str | None = Field(
            default=None,
            description="Candidate branch for enqueue/withdraw. Defaults to that tree's HEAD.",
        ),
        base: str | None = Field(
            default=None,
            description="Branch to land onto. Defaults to the repo's declared base.",
        ),
        reason: str | None = Field(
            default=None, description="Why a candidate is being withdrawn."
        ),
        batch_size: int = Field(
            default=0,
            description="Candidates gated together per run (0 = the repo's declared batch_size).",
        ),
        prune: bool = Field(
            default=True,
            description=(
                "Remove each landed candidate's worktree and branch through the "
                "guarded prune (merge-base re-checked at delete time, a "
                "refs/lane-backup anchor written first, `git branch -d` never -D, "
                "and a worktree holding uncommitted work is refused)."
            ),
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict[str, Any]:
        """Serialized merge queue for ANY git repository (CONCEPT:RM-MERGE-QUEUE).

        The canonical driver of parallel development: lanes take worktrees, work,
        then hand branches here, and the queue gates them **as merged**,
        fast-forwards the base, and prunes the worktree and branch.

        **The queue does not know what a gate is.** Gates are declared in the
        target repository's own ``.mergequeue.yaml`` — a command, a tier, a
        timeout, and how to compare its result against the base ref.
        agent-utilities declares pytest + ruff + its contract scripts;
        epistemic-graph declares ``cargo check --all-features`` + clippy; a docs
        repo could declare only a link check. A repository with no declaration is
        REFUSED rather than defaulted, because "declared no gates" and "has no
        queue configured" must not be the same value.

        Gating is **differential**: only a failure the base ref does not already
        produce blocks a candidate — the base is legitimately red here, and an
        absolute gate once deadlocked a queue and stranded 19 branches. A
        baseline that cannot be produced REFUSES the candidate; it never degrades
        to allow-all.

        ``run`` holds that repository's ``reconciliation-merge`` LEASE. If another
        runner holds it the call returns ``deferred: true`` with the holder —
        **defer, do not retry in a loop**.

        **``landed`` vs ``pushed`` (D-W3WPS-3).** Each outcome carries both,
        separately: ``landed`` means the declared base ref fast-forwarded in the
        LOCAL canonical checkout (proven by re-reading it); ``pushed`` means that
        commit also reached the configured remote via the repo's own gated push
        path (``_gate_before_push`` + ``push_project`` — the same pre-commit
        gates and GH013/divergent-remote handling a manual push gets). A push
        failure never fails the landing — ``landed: true, pushed: false`` is a
        legitimate, visible intermediate state (network blip, a diverged
        remote); check ``push_error`` on the outcome for why. The queue's own
        ``run`` summary also reports ``pushed``/``landed_unpushed`` counts.
        """
        from agent_utilities.governance.lanes import (
            LaneArbitrationError,
            LeaseUnavailable,
        )

        from repository_manager import merge_queue as merge_queue_core

        resolved = resolve_action(
            action, RM_MERGE_QUEUE_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        if ctx is not None:
            await ctx.info(f"merge-queue {resolved} on {repo_path or 'cwd'}")
        try:
            return await run_blocking(
                merge_queue_core.dispatch,
                resolved,
                path=repo_path,
                branch=branch or "",
                base=base or "",
                reason=reason or "",
                batch_size=batch_size,
                prune=prune,
            )
        except LeaseUnavailable as exc:
            return {
                "ok": False,
                "deferred": True,
                "holder": exc.holder,
                "error": (
                    "another runner holds this repository's reconciliation-merge "
                    "lease; the candidates stay queued — defer, do not retry"
                ),
            }
        except LaneArbitrationError as exc:
            return {"ok": False, "refused": str(exc), "error": str(exc)}
