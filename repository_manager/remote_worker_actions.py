"""RMDD-20: the shared MCP/CLI remote-worker application service.

Composes RMDD-15's :mod:`repository_manager.remote_execution` package with
the frozen RMDD-08 :class:`~repository_manager.capacity.CapacityInventory`
C-03 resource layer -- the exact pairing RMDD-15's own package docstring
calls for ("composed with ``ResourceScheduler.admit``/``.release``").

**What this module does and does not wire.** ``CapacityInventory`` is a
pure, process-local, thread-safe accounting structure (RMDD-08) -- never a
WorkItem authority, never a second job ledger -- so exactly one module-level
instance here is legitimate, and it is the SAME instance every action in
this module composes with
(:mod:`repository_manager.remote_execution.host_loss`'s own docstring: "the
same instance the scheduler uses -- never a copy"). A live,
WorkItem-authoritative ``ResourceScheduler.admit``/``.release`` requires
:meth:`repository_manager.native_reservations.NativeWorkItemReservationPort.from_graph_client`,
which needs a connected graph-os engine client. As of this lane
(2026-08-10), grepping the whole package for
``create_production_resource_scheduler``/``ResourceScheduler(`` finds this
construction wired into **no** MCP/CLI entrypoint anywhere in
repository-manager yet -- not for this lane, and not for RMDD-28's sibling
native lane authority either
(:mod:`repository_manager.native_lane_authority` is equally unwired).
Fabricating a local in-memory ``WorkItemReservationPort`` here to unblock a
demo would itself be exactly the "parallel job ledger / second store" the
lane's non-negotiable correctness constraints forbid. ``host_loss_reconcile``
therefore refuses honestly, naming that no live WorkItem-authoritative
resource scheduler is wired into this entrypoint -- this is deliberate, not
an oversight, and is exactly the "graph unavailable fails clearly/closed for
mutations without fabricated job success" behavior RMDD-20's required tests
call for.

Every action here degrades honestly, never an ``ImportError``, when the
optional ``tunnel-manager`` dependency is absent -- see
``_UnavailableInventoryResolver`` below and RMDD-15's own
``RemoteExecutionUnavailableError``.
"""

from __future__ import annotations

import base64
from pathlib import Path
from threading import RLock
from typing import Any

from repository_manager.capacity import CapacityInventory, HostCapacity, ResourceVector
from repository_manager.development import RefusalCode
from repository_manager.execution.executor import LocalExecutor
from repository_manager.remote_execution import (
    ArtifactStagingReceiver,
    ImmutableSourceStaging,
    RemoteWorkerProfile,
    RemoteWorkerRegistry,
    RemoteWorkerRegistryError,
)
from repository_manager.remote_execution.artifact_transport import ArtifactTransferError
from repository_manager.remote_execution.executor import RemoteExecutionUnavailableError
from repository_manager.remote_execution.source_staging import (
    DirtySourceError,
    SourceVerificationError,
)

__all__ = [
    "REMOTE_WORKER_ACTIONS",
    "capacity_inventory",
    "dispatch",
    "remote_worker_registry",
]

REMOTE_WORKER_ACTIONS: tuple[str, ...] = (
    "register_worker",
    "profile",
    "recheck",
    "stage_source",
    "verify_source",
    "receive_artifact",
    "host_loss_reconcile",
)

# One process-local capacity ledger (RMDD-08 C-03 accounting -- never a
# WorkItem authority). This is the SAME instance every action in this module
# composes with; a future lane that wires a live `ResourceScheduler` must
# consume this exact instance, never construct a second one.
_LOCK = RLock()
_CAPACITY_INVENTORY: CapacityInventory | None = None
_REGISTRY: RemoteWorkerRegistry | None = None


class _UnavailableInventoryResolver:
    """Structural ``InventoryResolver`` used while tunnel-manager is absent.

    Registration (``register_profile``/``profile``/``authorized_root``)
    never needs live inventory resolution -- only the dispatch-time
    ``recheck_at_claim`` entitlement check does -- so the registry stays
    constructible in the base install and the refusal is deferred to the
    one call that actually needs tunnel-manager, preserving the real
    ``ImportError`` as ``__cause__`` (H-12).
    """

    def resolve(self, alias: str, actor: object) -> Any:
        del alias, actor
        try:
            import tunnel_manager  # noqa: F401
        except ImportError as exc:
            raise RemoteExecutionUnavailableError(
                "remote worker entitlement recheck requires the optional "
                "'tunnel-manager' dependency (repository-manager's 'remote' "
                "extra), which is not installed in this environment"
            ) from exc
        raise RemoteExecutionUnavailableError(
            "tunnel-manager is installed but this entrypoint has no "
            "configured inventory resolver"
        )


def capacity_inventory() -> CapacityInventory:
    """The one process-local RMDD-08 capacity ledger (never a second copy)."""

    global _CAPACITY_INVENTORY
    with _LOCK:
        if _CAPACITY_INVENTORY is None:
            _CAPACITY_INVENTORY = CapacityInventory()
        return _CAPACITY_INVENTORY


def remote_worker_registry() -> RemoteWorkerRegistry:
    """The one process-local RMDD-15 registry, bound to ``capacity_inventory()``."""

    global _REGISTRY
    with _LOCK:
        if _REGISTRY is None:
            _REGISTRY = RemoteWorkerRegistry(
                capacity_inventory(),
                inventory_resolver=_UnavailableInventoryResolver(),
            )
        return _REGISTRY


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """Resolve and execute one remote-worker action; MCP and CLI share this.

    Returns ``{"ok": True, ...}`` on success or ``{"ok": False, "refused":
    ..., "error_code": ...}`` on a named refusal (C-10) -- no exception ever
    propagates as a bare traceback to either adapter, and every refusal
    preserves its cause (H-12).
    """

    from agent_utilities.mcp.action_dispatch import resolve_action

    resolved = resolve_action(
        action, REMOTE_WORKER_ACTIONS, service="repository-manager-remote-workers"
    )
    if isinstance(resolved, dict):
        return resolved
    try:
        result = _dispatch_resolved(resolved, kwargs)
    except (RemoteWorkerRegistryError, RemoteExecutionUnavailableError) as exc:
        return {
            "ok": False,
            "refused": str(exc),
            "error_code": RefusalCode.DEPENDENCY_BLOCKED.value,
        }
    except (DirtySourceError, SourceVerificationError) as exc:
        return {
            "ok": False,
            "refused": str(exc),
            "error_code": RefusalCode.VALIDATION_CANDIDATE_FAILURE.value,
        }
    except ArtifactTransferError as exc:
        return {
            "ok": False,
            "refused": str(exc),
            "error_code": RefusalCode.INVALID_REQUEST.value,
            "outcome": exc.outcome.value,
            "quarantine_path": exc.quarantine_path,
        }
    except (KeyError, LookupError, TypeError, AttributeError, ValueError) as exc:
        # A missing/malformed request field is an invalid request, not a
        # crash — named per C-10 rather than left to propagate as a bare
        # traceback (H-12: the original exception is still visible via
        # `str(exc)`, never discarded).
        return {
            "ok": False,
            "refused": f"invalid remote-worker request: {exc}",
            "error_code": RefusalCode.INVALID_REQUEST.value,
        }
    return {"ok": True, **result}


def _dispatch_resolved(resolved: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    if resolved == "register_worker":
        host = HostCapacity(
            host_id=kwargs["host_id"],
            total=ResourceVector(
                cpu_weight=kwargs.get("cpu_weight", 1),
                memory_mib=kwargs.get("memory_mib", 1),
                disk_mib=kwargs.get("disk_mib", 1),
                process_slots=kwargs.get("process_slots", 1),
            ),
            labels=tuple(kwargs.get("labels") or ()),
            target_kind="remote",
        )
        registered = capacity_inventory().register(host)
        profile = RemoteWorkerProfile(
            host_id=kwargs["host_id"],
            inventory_alias=kwargs["inventory_alias"],
            repository_roots=dict(kwargs.get("repository_roots") or {}),
            toolchains=tuple(kwargs.get("toolchains") or ()),
        )
        remote_worker_registry().register_profile(profile)
        return {
            "host_id": host.host_id,
            "capacity_registered": registered,
            "profile_registered": True,
        }

    if resolved == "profile":
        profile = remote_worker_registry().profile(kwargs["host_id"])
        return {
            "host_id": profile.host_id,
            "inventory_alias": profile.inventory_alias,
            "repository_roots": dict(profile.repository_roots),
            "toolchains": list(profile.toolchains),
        }

    if resolved == "recheck":
        target = remote_worker_registry().recheck_at_claim(
            kwargs["host_id"],
            actor=kwargs.get("actor", "repository-manager"),
            repository_id=kwargs["repository_id"],
            required_toolchain=kwargs.get("required_toolchain"),
        )
        return {"host_id": kwargs["host_id"], "authorized_target": repr(target)}

    if resolved == "stage_source":
        staging = ImmutableSourceStaging()
        clone, fetch, checkout = staging.stage_commands(
            origin=kwargs["origin"],
            tree_sha=kwargs["tree_sha"],
            parent_root=kwargs["parent_root"],
            worktree_name=kwargs["worktree_name"],
            timeout_seconds=kwargs.get("timeout_seconds", 1800),
        )
        commands = {
            "clone": clone.canonical_payload(),
            "fetch": fetch.canonical_payload(),
            "checkout": checkout.canonical_payload(),
        }
        if not kwargs.get("execute_locally", False):
            return {"commands": commands, "executed": False}

        executor = LocalExecutor(authorized_roots=kwargs["parent_root"])
        for index, command in enumerate((clone, fetch, checkout)):
            result = executor.run(
                command,
                command_id=f"command:stage-source:{index}",
                worker_id="worker:stage-source",
                fence=f"fence:stage-source:{kwargs['worktree_name']}",
            )
            if result.outcome.value != "succeeded":
                raise SourceVerificationError(
                    f"source staging step {index} did not succeed: "
                    f"{result.outcome.value}"
                )
        destination = f"{kwargs['parent_root'].rstrip('/')}/{kwargs['worktree_name']}"
        staged = staging.verify_staged_sha(
            executor,
            destination=destination,
            expected_sha=kwargs["tree_sha"],
            repository_id=kwargs.get("repository_id", kwargs["worktree_name"]),
        )
        return {
            "commands": commands,
            "executed": True,
            "staged": {
                "repository_id": staged.repository_id,
                "tree_sha": staged.tree_sha,
                "destination": staged.destination,
                "verified_at": staged.verified_at.isoformat(),
            },
        }

    if resolved == "verify_source":
        destination = kwargs["destination"]
        parent_root = str(Path(destination).parent)
        executor = LocalExecutor(authorized_roots=parent_root)
        staging = ImmutableSourceStaging()
        staged = staging.verify_staged_sha(
            executor,
            destination=destination,
            expected_sha=kwargs["expected_sha"],
            repository_id=kwargs["repository_id"],
        )
        return {
            "repository_id": staged.repository_id,
            "tree_sha": staged.tree_sha,
            "destination": staged.destination,
            "verified_at": staged.verified_at.isoformat(),
        }

    if resolved == "receive_artifact":
        receiver = ArtifactStagingReceiver(kwargs["root"])
        content = base64.b64decode(kwargs["content_base64"])
        method = (
            receiver.receive_log if kwargs.get("kind") == "log" else receiver.receive
        )
        receipt = method(
            kwargs["relative_path"],
            [content],
            declared_size=len(content),
            host_id=kwargs["host_id"],
            source_description=kwargs["source_description"],
            declared_digest=kwargs.get("declared_digest"),
            media_type=kwargs.get(
                "media_type",
                "text/plain"
                if kwargs.get("kind") == "log"
                else "application/octet-stream",
            ),
        )
        return {
            "reference": receipt.reference.canonical_payload(),
            "host_id": receipt.host_id,
            "outcome": receipt.outcome.value,
        }

    if resolved == "host_loss_reconcile":
        raise RemoteExecutionUnavailableError(
            "host-loss reconciliation requires a live WorkItem-authoritative "
            "ResourceScheduler.release (repository_manager.native_reservations."
            "NativeWorkItemReservationPort, bound to a connected graph-os "
            "engine client); no such live scheduler is wired into this "
            "MCP/CLI entrypoint yet, and this module never substitutes a "
            "local in-memory reservation ledger for that authority"
        )

    raise AssertionError(f"unhandled resolved remote-worker action {resolved!r}")
