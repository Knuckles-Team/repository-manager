"""Lane lifecycle MCP adapter."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext


def register_lane_tools(mcp: FastMCP, *, context: McpToolContext | None = None) -> None:
    """Register the lane lifecycle adapter."""

    del context

    @mcp.tool(tags={"workspace_management", "project_manager", "git_operations"})
    async def rm_lane(
        action: str = Field(
            description=(
                "Action: 'start' (open an isolated worktree and prove its "
                "partitions), 'doctor' (check an existing tree's isolation, "
                "mutates nothing), 'env' (the shell exports alone), 'finish' "
                "(preflight, then hand the branch to the merge queue)."
            )
        ),
        path: str | None = Field(
            default=None,
            description="The lane's working tree for doctor/env/finish (default: cwd).",
        ),
        repo: str | None = Field(
            default=None,
            description="Repo basename or absolute path. Required for 'start'.",
        ),
        branch: str | None = Field(
            default=None,
            description="Branch for 'start'/'finish' ('finish' defaults to that tree's HEAD).",
        ),
        base: str | None = Field(
            default=None,
            description="Base branch to fork from / land onto (default main).",
        ),
        force: bool = Field(
            default=False,
            description="For 'finish': enqueue despite a blocking preflight check. Recorded in the result.",
        ),
        env: dict[str, str] | None = Field(
            default=None,
            description=(
                "For 'doctor': the CALLER's own exported environment (at least "
                "PRE_COMMIT_HOME/CARGO_TARGET_DIR/PYTEST_ADDOPTS). Required for a "
                "truthful remote check (D-CDX-61) — this server process's own "
                "environment is not the lane process's, so omitting this "
                "evaluates the checks against server-internal state that has "
                "nothing to do with what the caller will actually run with."
            ),
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict[str, Any]:
        """The lane lifecycle for concurrent agents and humans (CONCEPT:RM-LANE-DOCTOR).

        Every check behind this tool exists because a specific collision destroyed
        real work while the rule that would have prevented it was written down, in
        prose, in front of the actor who broke it: editing a canonical checkout a
        background `git reset` then discarded; a `git stash` into the ONE
        `refs/stash` shared by ~54 worktrees; a shared `CARGO_TARGET_DIR` that
        corrupts rather than merely serializes; an unset `PRE_COMMIT_HOME` whose
        shared store swallows unstaged work; a foreign or stale worktree-local
        `.venv` worth ~167 phantom failures; and measuring a branch TIP that will
        never be the tree that lands. The exact all-extras environment managed by
        `scripts/uv_workspace.py` is accepted because it is lane-partitioned and
        synchronized before execution.

        `doctor` is the one to reach for when something behaves impossibly — a
        test failing that cannot fail, a build that will not go green, a merge
        that keeps being refused. It answers in under a second and mutates
        nothing.
        """
        del ctx
        from repository_manager import lane_doctor

        resolved = resolve_action(
            action, lane_doctor.ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        return await run_blocking(
            lane_doctor.dispatch,
            resolved,
            path=path,
            repo=repo,
            branch=branch,
            base=base,
            force=force,
            env=env,
        )
