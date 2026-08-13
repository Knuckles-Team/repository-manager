"""Content-addressed build MCP adapter."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext
from repository_manager.mcp_tools.contracts import RM_BUILD_ACTIONS


def register_build_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the build broker adapter."""

    del context

    @mcp.tool(tags={"workspace_management", "project_manager", "git_operations"})
    async def rm_build(
        action: str = Field(
            description=(
                "Action: 'request' (dedup-or-build — a second request for the "
                "same key waits on and reuses the first's artifacts instead of "
                "rebuilding), 'status' (computed cache key + execution-class "
                "occupancy, or a specific key's cache/task state), 'artifacts' "
                "(published files for a cache key), 'explain' (why a key did "
                "NOT hit cache — the differing component), 'gc' (bounded "
                "reclamation of old cache entries)."
            )
        ),
        repo_path: str | None = Field(
            default=None,
            description="Any working tree of the target repository. Defaults to the server's cwd.",
        ),
        spec: str | None = Field(
            default=None,
            description="Which declared build spec to use (.buildcache.yaml). Defaults to the repo's first.",
        ),
        key: str | None = Field(
            default=None,
            description="A cache key, for status/artifacts/explain.",
        ),
        wait_timeout: int = Field(
            default=60,
            description="Seconds 'request' waits on an in-flight build of the same key before building anyway.",
        ),
        keep_recent: int = Field(
            default=10,
            description="For gc: always keep this many most-recent cache entries.",
        ),
        max_age_days: int = Field(
            default=14,
            description="For gc: reclaim cache entries older than this (subject to keep_recent).",
        ),
        host: str | None = Field(
            default=None,
            description=(
                "For 'request': dispatch to this REGISTERED, authorized "
                "remote host (its rm_remote_workers 'register_worker' "
                "host_id) instead of building locally. Requires a clean "
                "local tree (its HEAD sha + origin are what gets staged and "
                "built remotely) and does not yet retrieve artifacts back — "
                "see the returned result's 'note'. Omit to build locally "
                "(the default, always colocated=True)."
            ),
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict[str, Any]:
        """Content-addressed build broker for ANY repository (CONCEPT:RM-TASK-LEDGER).

        Keyed by ``(repo, tree-sha, feature-set, toolchain-fingerprint,
        target-triple)``. The DEDUP is the value: N lanes asking for the same
        build today each pay for a full ``cargo``/``pnpm`` build into their own
        ``target-isolated``/``node_modules`` — measured at 21.7 GB across 4
        duplicate directories after pruning. This tool publishes named
        artifacts to a shared, checksummed cache instead.

        **Honest degradation**: a dirty tree or an unfingerprintable toolchain
        BUILDS — it never serves a possibly-stale artifact and never silently
        skips the cache as though it were consulted.

        **Co-location**: this tool always passes ``colocated=True`` to the
        underlying reservation, because being inside this pinned MCP server
        process IS the proof of same-node execution that a bare CLI invocation
        cannot provide — ``fcntl`` leases do not arbitrate across nodes, and
        the lease files live under an NFS-exported path. Route build requests
        through this tool rather than a bare local invocation when you are not
        certain you are on the pinned node.

        **Arbitration is advisory (D-CP-8)**: an agent running ``cargo build``
        directly bypasses this broker entirely.
        """
        from agent_utilities.governance.lanes import LaneArbitrationError

        from repository_manager import build_queue as build_queue_core

        resolved = resolve_action(
            action, RM_BUILD_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        if ctx is not None:
            await ctx.info(f"build {resolved} on {repo_path or 'cwd'}")
        try:
            return await run_blocking(
                build_queue_core.dispatch,
                resolved,
                path=repo_path,
                spec=spec or "",
                key=key or "",
                colocated=True,
                wait_timeout=wait_timeout,
                keep_recent=keep_recent,
                max_age_days=max_age_days,
                host=host,
            )
        except LaneArbitrationError as exc:
            return {"ok": False, "refused": str(exc), "error": str(exc)}
