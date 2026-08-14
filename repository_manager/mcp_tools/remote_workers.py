"""Remote-worker MCP adapter (RMDD-20, exposing RMDD-15)."""

from __future__ import annotations

from typing import Any

from agent_utilities.mcp.concurrency import run_blocking
from fastmcp import Context, FastMCP
from pydantic import Field

from repository_manager.mcp_tools.context import McpToolContext


def register_remote_workers_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register the remote-worker registry/staging/artifact adapter."""

    del context

    @mcp.tool(tags={"workspace_management", "project_manager"})
    async def rm_remote_workers(
        action: str = Field(
            description=(
                "Action: 'register_worker' (declare a host's weighted "
                "capacity + authorized repository roots/toolchains, "
                "DURABLY -- survives an MCP server restart), "
                "'seed_from_inventory' (register a PLACEHOLDER capacity "
                "record, deliberately already-stale, for every host in "
                "tunnel-manager's SSH inventory not already registered -- "
                "never admits real work until a real register_worker/"
                "heartbeat confirms the host), "
                "'profile' (read a registered worker's declared capability), "
                "'recheck' (dispatch-time entitlement recheck before a "
                "claim -- refuses honestly without the optional "
                "tunnel-manager dependency), 'stage_source' (build the fixed "
                "clone/fetch/checkout command sequence for one immutable "
                "commit SHA, optionally executing+verifying it locally), "
                "'verify_source' (independently re-derive cleanliness/HEAD "
                "identity of an already-staged worktree), 'receive_artifact' "
                "(stream a base64 payload into the bounded, checksummed, "
                "content-addressed artifact/log store), 'host_loss_reconcile' "
                "(quarantine a lost host and release its reservation -- "
                "refuses honestly: no live WorkItem-authoritative resource "
                "scheduler is wired into this entrypoint yet)."
            )
        ),
        host_id: str | None = Field(default=None, description="Target host id."),
        path: str | None = Field(
            default=None,
            description=(
                "For seed_from_inventory: override the inventory.yaml path "
                "(default ~/.config/agent-utilities/inventory.yaml)."
            ),
        ),
        cpu_weight: int | None = Field(
            default=None, description="For register_worker."
        ),
        memory_mib: int | None = Field(
            default=None, description="For register_worker."
        ),
        disk_mib: int | None = Field(default=None, description="For register_worker."),
        process_slots: int | None = Field(
            default=None, description="For register_worker."
        ),
        labels: list[str] | None = Field(
            default=None, description="For register_worker."
        ),
        inventory_alias: str | None = Field(
            default=None,
            description="For register_worker/recheck: the tunnel-manager inventory alias.",
        ),
        repository_roots: dict[str, str] | None = Field(
            default=None,
            description="For register_worker: repository id -> authorized absolute worktree root.",
        ),
        toolchains: list[str] | None = Field(
            default=None, description="For register_worker."
        ),
        actor: str | None = Field(
            default=None, description="For recheck: the requesting actor."
        ),
        repository_id: str | None = Field(
            default=None, description="For recheck/stage_source/verify_source."
        ),
        required_toolchain: str | None = Field(
            default=None, description="For recheck."
        ),
        origin: str | None = Field(
            default=None, description="For stage_source: credential-free Git origin."
        ),
        tree_sha: str | None = Field(
            default=None,
            description="For stage_source: the full 40-hex immutable commit SHA.",
        ),
        parent_root: str | None = Field(
            default=None,
            description="For stage_source: the worker's authorized parent root.",
        ),
        worktree_name: str | None = Field(
            default=None, description="For stage_source."
        ),
        timeout_seconds: int = Field(default=1800, description="For stage_source."),
        execute_locally: bool = Field(
            default=False,
            description="For stage_source: also run+verify the commands via a local executor.",
        ),
        destination: str | None = Field(default=None, description="For verify_source."),
        expected_sha: str | None = Field(
            default=None, description="For verify_source."
        ),
        root: str | None = Field(
            default=None, description="For receive_artifact: store root."
        ),
        relative_path: str | None = Field(
            default=None, description="For receive_artifact."
        ),
        content_base64: str | None = Field(
            default=None, description="For receive_artifact."
        ),
        declared_digest: str | None = Field(
            default=None, description="For receive_artifact, optional."
        ),
        source_description: str | None = Field(
            default=None, description="For receive_artifact."
        ),
        media_type: str | None = Field(
            default=None, description="For receive_artifact, optional."
        ),
        kind: str | None = Field(
            default=None,
            description="For receive_artifact: 'artifact' (default) or 'log'.",
        ),
        reservation_id: str | None = Field(
            default=None, description="For host_loss_reconcile."
        ),
        work_item_id: str | None = Field(
            default=None, description="For host_loss_reconcile."
        ),
        attempt: int | None = Field(
            default=None, description="For host_loss_reconcile."
        ),
        fence: str | None = Field(default=None, description="For host_loss_reconcile."),
        reason: str | None = Field(
            default=None, description="For host_loss_reconcile, optional."
        ),
        ctx: Context | None = Field(
            default=None, description="MCP context for progress reporting"
        ),
    ) -> dict[str, Any]:
        """Remote worker registry, immutable source staging, and artifact transport.

        Composes RMDD-15's ``repository_manager.remote_execution`` package
        with the RMDD-08 ``CapacityInventory`` resource ledger. The optional
        ``tunnel-manager`` dependency is never required to import this tool:
        ``recheck``/live remote dispatch refuse with a named,
        cause-preserving error when it is absent instead of raising an
        ``ImportError``. ``host_loss_reconcile`` always refuses today: no
        live WorkItem-authoritative ``ResourceScheduler`` is wired into any
        repository-manager entrypoint yet, and this tool never substitutes a
        local in-memory reservation ledger for that authority.
        """

        del ctx
        from repository_manager import remote_worker_actions

        return await run_blocking(
            remote_worker_actions.dispatch,
            action,
            host_id=host_id,
            path=path,
            cpu_weight=cpu_weight,
            memory_mib=memory_mib,
            disk_mib=disk_mib,
            process_slots=process_slots,
            labels=labels,
            inventory_alias=inventory_alias,
            repository_roots=repository_roots,
            toolchains=toolchains,
            actor=actor,
            repository_id=repository_id,
            required_toolchain=required_toolchain,
            origin=origin,
            tree_sha=tree_sha,
            parent_root=parent_root,
            worktree_name=worktree_name,
            timeout_seconds=timeout_seconds,
            execute_locally=execute_locally,
            destination=destination,
            expected_sha=expected_sha,
            root=root,
            relative_path=relative_path,
            content_base64=content_base64,
            declared_digest=declared_digest,
            source_description=source_description,
            media_type=media_type,
            kind=kind,
            reservation_id=reservation_id,
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reason=reason,
        )
