"""Worktree management MCP adapter."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agent_utilities.mcp.action_dispatch import resolve_action
from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_WORKTREE_ACTIONS


@dataclass
class _NoRepoRequest:
    """Bundled params for the actions that don't require repo+branch."""

    repo: str | None
    base: str
    stale_days: int
    prune_merged: bool
    branch: str | None
    repos: str | None


@dataclass
class _RepoBranchRequest:
    """Bundled params for the actions that require repo+branch."""

    repo: str
    branch: str
    base: str
    into: str
    adopt: bool
    force: bool
    delete_branch: bool
    strategy: str


async def _worktree_list(worktree_manager: Any, req: _NoRepoRequest) -> dict[str, Any]:
    return await run_blocking(worktree_manager.list_worktrees, repo=req.repo)


async def _worktree_prune(worktree_manager: Any, req: _NoRepoRequest) -> dict[str, Any]:
    return await run_blocking(worktree_manager.prune, repo=req.repo)


async def _worktree_audit(worktree_manager: Any, req: _NoRepoRequest) -> dict[str, Any]:
    return await run_blocking(
        worktree_manager.audit,
        repo=req.repo,
        base=req.base,
        stale_days=req.stale_days,
        prune_merged=req.prune_merged,
    )


async def _worktree_bulk_add(
    worktree_manager: Any, req: _NoRepoRequest
) -> dict[str, Any]:
    if req.branch is None:
        return {"ok": False, "error": "action 'bulk_add' requires 'branch'"}
    repo_list = [item.strip() for item in req.repos.split(",")] if req.repos else None
    return await run_blocking(
        worktree_manager.bulk_add, req.branch, repos=repo_list, base=req.base
    )


_NO_REPO_ACTIONS: dict[
    str, Callable[[Any, _NoRepoRequest], Awaitable[dict[str, Any]]]
] = {
    "list": _worktree_list,
    "prune": _worktree_prune,
    "audit": _worktree_audit,
    "bulk_add": _worktree_bulk_add,
}


async def _worktree_add(
    worktree_manager: Any, req: _RepoBranchRequest
) -> dict[str, Any]:
    return await run_blocking(
        worktree_manager.add, req.repo, req.branch, base=req.base, adopt=req.adopt
    )


async def _worktree_remove(
    worktree_manager: Any, req: _RepoBranchRequest
) -> dict[str, Any]:
    return await run_blocking(
        worktree_manager.remove,
        req.repo,
        req.branch,
        force=req.force,
        delete_branch=req.delete_branch,
        base=req.base,
    )


async def _worktree_merge(
    worktree_manager: Any, req: _RepoBranchRequest
) -> dict[str, Any]:
    return await run_blocking(
        worktree_manager.merge, req.repo, req.branch, into=req.into
    )


async def _worktree_reset_branch(
    worktree_manager: Any, req: _RepoBranchRequest
) -> dict[str, Any]:
    return await run_blocking(
        worktree_manager.reset_branch, req.repo, req.branch, target=req.into
    )


async def _worktree_sync(
    worktree_manager: Any, req: _RepoBranchRequest
) -> dict[str, Any]:
    return await run_blocking(
        worktree_manager.sync,
        req.repo,
        req.branch,
        base=req.base,
        strategy=req.strategy,
    )


_REPO_BRANCH_ACTIONS: dict[
    str, Callable[[Any, _RepoBranchRequest], Awaitable[dict[str, Any]]]
] = {
    "add": _worktree_add,
    "remove": _worktree_remove,
    "merge": _worktree_merge,
    "reset_branch": _worktree_reset_branch,
    "sync": _worktree_sync,
}


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

        no_repo_handler = _NO_REPO_ACTIONS.get(action)
        if no_repo_handler is not None:
            no_repo_req = _NoRepoRequest(
                repo=repo,
                base=base,
                stale_days=stale_days,
                prune_merged=prune_merged,
                branch=branch,
                repos=repos,
            )
            return await no_repo_handler(worktree_manager, no_repo_req)

        if repo is None or branch is None:
            return {
                "ok": False,
                "error": f"action '{action}' requires 'repo' and 'branch'",
            }

        repo_branch_handler = _REPO_BRANCH_ACTIONS.get(action)
        if repo_branch_handler is None:
            return {"ok": False, "error": f"unknown action: {action}"}
        repo_branch_req = _RepoBranchRequest(
            repo=repo,
            branch=branch,
            base=base,
            into=into,
            adopt=adopt,
            force=force,
            delete_branch=delete_branch,
            strategy=strategy,
        )
        return await repo_branch_handler(worktree_manager, repo_branch_req)
