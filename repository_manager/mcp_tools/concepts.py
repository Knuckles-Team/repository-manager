"""Concept-claim coordination MCP adapter (RMDD-20, exposing RMDD-17)."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext


def register_concepts_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the concept-claim coordination adapter."""

    del context

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_concepts(
        action: str = Field(
            description=(
                "Action: 'reserve' (create-if-absent a concept-id claim), "
                "'get' (fetch one claim), 'list' (page claims by "
                "namespace/state/prefix), 'materialize' (RESERVED -> "
                "MATERIALIZED), 'release' (abort a never-visible claim), "
                "'verify_candidate' (check one candidate's introduced concept "
                "markers against claims), 'verify_generation' (verify every "
                "candidate in a sealed generation and cross-candidate "
                "collisions), 'reconcile' (read-only drift report across "
                "authority/fragments/view/source)."
            )
        ),
        repo_root: str = Field(description="Repository working tree root."),
        tenant_ref: str = Field(description="Authenticated tenant scope."),
        lane_ref: str = Field(
            description="Lane/worktree identity for fragment provenance."
        ),
        reservation_id: str | None = Field(
            default=None, description="For get/release/materialize."
        ),
        owner_ref: str | None = Field(
            default=None,
            description="For release/materialize (must match the claim owner).",
        ),
        expected_fence: int | None = Field(
            default=None,
            description="For release/materialize: the claim's current fence.",
        ),
        concept_id: str | None = Field(default=None, description="For reserve."),
        namespace: str | None = Field(
            default=None,
            description="For reserve (required) or list (optional filter).",
        ),
        repository_ref: str | None = Field(default=None, description="For reserve."),
        request_key_ref: str | None = Field(
            default=None, description="For reserve: idempotency key."
        ),
        purpose: str | None = Field(default=None, description="For reserve."),
        design_ref: str | None = Field(
            default=None, description="For reserve, optional."
        ),
        branch: str | None = Field(default=None, description="For reserve, optional."),
        base_sha: str | None = Field(
            default=None, description="For reserve, optional."
        ),
        workitem_ref: str | None = Field(
            default=None, description="For reserve, optional."
        ),
        run_trace_ref: str | None = Field(
            default=None, description="For reserve, optional."
        ),
        provenance_refs: list[str] | None = Field(
            default=None, description="For reserve, optional."
        ),
        state: str | None = Field(
            default=None, description="For list: filter by claim state."
        ),
        concept_prefix: str | None = Field(
            default=None, description="For list: filter by concept-id prefix."
        ),
        limit: int = Field(default=1000, description="For list: page size."),
        cursor: str | None = Field(
            default=None, description="For list: opaque page cursor."
        ),
        candidate: dict[str, Any] | None = Field(
            default=None,
            description="For verify_candidate: a serialized Candidate contract payload.",
        ),
        generation: dict[str, Any] | None = Field(
            default=None,
            description="For verify_generation: a serialized Generation contract payload.",
        ),
        candidates: list[dict[str, Any]] | None = Field(
            default=None,
            description="For verify_generation: serialized Candidate contract payloads.",
        ),
        repo_path: str | None = Field(
            default=None,
            description="For verify_candidate/verify_generation: the Git repository to diff.",
        ),
        source_tree_ish: str = Field(
            default="HEAD",
            description="For reconcile: the tree-ish to scan for markers.",
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict[str, Any]:
        """Concept-id claim coordination against RMDD-16's central authority.

        Repository-manager never allocates a concept id itself: every action
        here is a thin, MCP/CLI-neutral consumer of an injected authority
        port. When the central authority (``agent_utilities.governance.
        concept_reservation``) is unavailable in this environment, every
        mutating action refuses with a named
        ``dependency_blocked``/``ConceptAuthorityUnavailable`` error instead
        of fabricating a local allocation -- it never mints or reuses an id
        on its own.
        """

        del ctx
        from repository_manager import concept_actions

        return await run_blocking(
            concept_actions.dispatch,
            action,
            repo_root=repo_root,
            tenant_ref=tenant_ref,
            lane_ref=lane_ref,
            reservation_id=reservation_id,
            owner_ref=owner_ref,
            expected_fence=expected_fence,
            concept_id=concept_id,
            namespace=namespace,
            repository_ref=repository_ref,
            request_key_ref=request_key_ref,
            purpose=purpose,
            design_ref=design_ref,
            branch=branch,
            base_sha=base_sha,
            workitem_ref=workitem_ref,
            run_trace_ref=run_trace_ref,
            provenance_refs=provenance_refs,
            state=state,
            concept_prefix=concept_prefix,
            limit=limit,
            cursor=cursor,
            candidate=candidate,
            generation=generation,
            candidates=candidates,
            repo_path=repo_path,
            source_tree_ish=source_tree_ish,
        )
