"""Worktree management MCP adapter."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_WORKTREE_ACTIONS


def register_worktree_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the linked-worktree adapter."""

    adapter_context = context or from_server()

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_worktree(
        action: str = Field(
            description="Action: 'add', 'list', 'remove', 'merge', 'sync', 'prune', 'bulk_add', 'audit', 'reset_branch'."
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
        into: str = Field(
            default="main",
            description="Target branch for 'merge', or the reset target for 'reset_branch'.",
        ),
        adopt: bool = Field(
            default=False,
            description="For 'add': move the canonical checkout's uncommitted WIP onto the new branch via a private ref (never the shared refs/stash stack).",
        ),
        force: bool = Field(
            default=False,
            description=(
                "For 'remove': override git's own dirty-tree refusal once "
                "lane occupancy has already cleared (never bypasses "
                "occupancy itself, D-CDX-15)."
            ),
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
    ) -> dict[str, Any]:
        """Manage git worktrees for concurrent multi-session development (CONCEPT:RM-WORKTREE).

        Each session works a repo in its own worktree on its own branch under
        ``<WORKTREE_ROOT>/<repo>/<branch>`` (shared ``.git``, no re-clone),
        leaving the canonical checkout on its default branch so a working-tree
        reset never disturbs in-flight session work.
        """
        del ctx
        from repository_manager.worktree import WorktreeManager

        git = adapter_context.get_git_instance(path=path)
        worktree_manager = WorktreeManager(git)
        resolved = resolve_action(
            action, RM_WORKTREE_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        if action == "list":
            return await run_blocking(worktree_manager.list_worktrees, repo=repo)
        if action == "prune":
            return await run_blocking(worktree_manager.prune, repo=repo)
        if action == "audit":
            return await run_blocking(
                worktree_manager.audit,
                repo=repo,
                base=base,
                stale_days=stale_days,
                prune_merged=prune_merged,
            )
        if action == "bulk_add":
            if branch is None:
                return {"ok": False, "error": "action 'bulk_add' requires 'branch'"}
            repo_list = [item.strip() for item in repos.split(",")] if repos else None
            return await run_blocking(
                worktree_manager.bulk_add, branch, repos=repo_list, base=base
            )
        if repo is None or branch is None:
            return {
                "ok": False,
                "error": f"action '{action}' requires 'repo' and 'branch'",
            }
        if action == "add":
            return await run_blocking(
                worktree_manager.add, repo, branch, base=base, adopt=adopt
            )
        if action == "remove":
            return await run_blocking(
                worktree_manager.remove,
                repo,
                branch,
                force=force,
                delete_branch=delete_branch,
                base=base,
            )
        if action == "merge":
            return await run_blocking(worktree_manager.merge, repo, branch, into=into)
        if action == "reset_branch":
            return await run_blocking(
                worktree_manager.reset_branch, repo, branch, target=into
            )
        if action == "sync":
            return await run_blocking(
                worktree_manager.sync, repo, branch, base=base, strategy=strategy
            )
        return {"ok": False, "error": f"unknown action: {action}"}
