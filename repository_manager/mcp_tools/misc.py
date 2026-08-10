"""Miscellaneous Repository Manager MCP adapters."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from repository_manager.mcp_tools.context import McpToolContext


def register_misc_tools(mcp: FastMCP, *, context: McpToolContext | None = None) -> None:
    """Register health and repository-ingestion adapters."""

    async def health_check(request: Request) -> JSONResponse:
        del request
        return JSONResponse({"status": "OK"})

    @mcp.tool(tags={"workspace_management", "knowledge_graph"})
    async def repository_ingest_repositories(
        vcs: str = Field(
            default="gitlab",
            description="Which VCS to enumerate: 'gitlab' or 'github'.",
        ),
        scopes: str | None = Field(
            default=None,
            description="Comma-separated GitLab groups / GitHub orgs to enumerate. Omit for the whole instance / the authenticated user.",
        ),
        max_repos: int | None = Field(
            default=None,
            description="Optional cap on the number of repositories enumerated + ingested.",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict[str, Any]:
        """Natively ingest git repositories into epistemic-graph as typed :GitRepository nodes.

        Enumerates repositories across a GitLab instance / GitHub org via the real VCS
        enumerator and pushes them (typed :GitRepository nodes with vcs/clone_url/web_url/
        default_branch provenance) into the knowledge graph via the fast engine client.
        Best-effort: returns ``{"ingested": None}`` when no engine is reachable.
        CONCEPT:AU-KG.ingest.enterprise-source-extractor.
        """
        from repository_manager.kg_ingest import ingest_repositories
        from repository_manager.vcs_enumerator import (
            enumerate_github,
            enumerate_gitlab,
        )

        which = (vcs or "gitlab").strip().lower()
        scope_list = (
            [s.strip() for s in scopes.split(",") if s.strip()] if scopes else None
        )
        if which == "github":
            refs = await run_blocking(
                enumerate_github,
                orgs=scope_list,
                user=not scope_list,
                max_repos=max_repos,
            )
        else:
            refs = await run_blocking(
                enumerate_gitlab, groups=scope_list, max_repos=max_repos
            )
        ingested = await run_blocking(ingest_repositories, refs)
        return {"vcs": which, "listed": len(refs), "ingested": ingested}
