"""Remote worker capability registry: labels, roots, toolchains, and health.

This module never duplicates the C-03 weighted capacity ledger.  It composes
with a caller-owned :class:`repository_manager.capacity.CapacityInventory`
(the *same* instance the scheduler uses -- never a second copy) and adds only
the metadata a remote worker needs beyond weighted resources: its authorized
per-repository worktree root and declared toolchains, plus the dispatch-time
entitlement recheck against tunnel-manager's frozen ``HostInventory``.  It
never registers WorkItem state and is not a second job ledger.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from threading import RLock
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from repository_manager.capacity import CapacityInventory, HostState

if TYPE_CHECKING:  # pragma: no cover - import-time optional dependency guard
    # ``tunnel-manager`` is an optional dependency (see executor.py for the
    # full rationale); this module only ever uses ``AuthorizedTarget`` as a
    # type annotation, so a deferred/type-checking-only import keeps
    # ``import repository_manager.remote_execution.registry`` working with
    # or without tunnel-manager installed.
    from tunnel_manager.remote_execution import AuthorizedTarget


class RemoteWorkerRegistryError(ValueError):
    """A worker capability, freshness, or entitlement recheck failed."""


@runtime_checkable
class InventoryResolver(Protocol):
    """The exact frozen RMDD-14 entitlement seam this registry reuses."""

    def resolve(self, alias: str, actor: object) -> AuthorizedTarget:
        """Resolve an inventory alias to an opaque authorized target view."""


@dataclass(frozen=True)
class RemoteWorkerProfile:
    """Declared capability of one registered remote worker host.

    ``repository_roots`` maps a repository identity to the absolute worktree
    root authorized on that host -- never a shared NFS path, never inferred
    from caller input.  ``inventory_alias`` is the tunnel-manager alias this
    host resolves through; it is never a hostname, address, or connection
    string.
    """

    host_id: str
    inventory_alias: str
    repository_roots: Mapping[str, str] = field(default_factory=dict)
    toolchains: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.host_id or self.host_id.strip() != self.host_id:
            raise RemoteWorkerRegistryError("host_id must be non-blank")
        if not self.inventory_alias or self.inventory_alias.strip() != (
            self.inventory_alias
        ):
            raise RemoteWorkerRegistryError("inventory_alias must be non-blank")
        for repository_id, root in self.repository_roots.items():
            if not repository_id or not root or not root.startswith("/"):
                raise RemoteWorkerRegistryError(
                    f"authorized root for repository {repository_id!r} must be "
                    "a non-blank absolute path"
                )


class RemoteWorkerRegistry:
    """Registers worker capability metadata and reauthorizes at claim time."""

    def __init__(
        self, capacity: CapacityInventory, *, inventory_resolver: InventoryResolver
    ) -> None:
        self._capacity = capacity
        self._resolver = inventory_resolver
        self._lock = RLock()
        self._profiles: dict[str, RemoteWorkerProfile] = {}

    def register_profile(self, profile: RemoteWorkerProfile) -> None:
        """Attach capability metadata to an already-registered capacity host.

        Capacity registration remains :class:`CapacityInventory`'s job; this
        method refuses a profile for a host with no capacity record so the
        registry can never authorize a host the scheduler cannot see.
        """

        with self._lock:
            if self._capacity.get(profile.host_id) is None:
                raise RemoteWorkerRegistryError(
                    f"host {profile.host_id!r} has no registered capacity; "
                    "register capacity before a worker profile"
                )
            self._profiles[profile.host_id] = profile

    def profile(self, host_id: str) -> RemoteWorkerProfile:
        with self._lock:
            profile = self._profiles.get(host_id)
        if profile is None:
            raise RemoteWorkerRegistryError(f"unknown remote worker host {host_id!r}")
        return profile

    def authorized_root(self, host_id: str, repository_id: str) -> str:
        """Return the authorized worktree root, refusing an unmapped pair."""

        profile = self.profile(host_id)
        root = profile.repository_roots.get(repository_id)
        if root is None:
            raise RemoteWorkerRegistryError(
                f"host {host_id!r} has no authorized root for repository "
                f"{repository_id!r}"
            )
        return root

    def recheck_at_claim(
        self,
        host_id: str,
        *,
        actor: object,
        repository_id: str,
        required_toolchain: str | None = None,
        now: datetime | None = None,
    ) -> AuthorizedTarget:
        """Reauthorize a scheduler-selected host immediately before dispatch.

        Every check is redone fresh -- capacity freshness/drain/quarantine,
        declared toolchain, authorized root, and, last so a revoked
        entitlement always wins, the inventory alias entitlement itself.
        Passing this method is a precondition for constructing a
        :class:`repository_manager.remote_execution.executor.RemoteWorkerExecutor`;
        it never grants execution by itself and never mutates capacity.
        """

        profile = self.profile(host_id)
        if self._capacity.get(host_id) is None:
            raise RemoteWorkerRegistryError(f"host {host_id!r} capacity was withdrawn")
        snapshot = self._capacity.snapshot(host_id, now=now)
        if snapshot.state in {HostState.DRAINED, HostState.DRAINING}:
            raise RemoteWorkerRegistryError(
                f"host {host_id!r} is draining/drained and cannot claim new work"
            )
        if snapshot.state in {HostState.QUARANTINED, HostState.OFFLINE}:
            raise RemoteWorkerRegistryError(
                f"host {host_id!r} is quarantined/offline and cannot claim new work"
            )
        if not snapshot.heartbeat_fresh:
            raise RemoteWorkerRegistryError(
                f"host {host_id!r} heartbeat is stale and cannot claim new work"
            )
        if repository_id not in profile.repository_roots:
            raise RemoteWorkerRegistryError(
                f"host {host_id!r} has no authorized root for repository "
                f"{repository_id!r}"
            )
        if (
            required_toolchain is not None
            and required_toolchain not in profile.toolchains
        ):
            raise RemoteWorkerRegistryError(
                f"host {host_id!r} does not declare required toolchain "
                f"{required_toolchain!r}"
            )
        try:
            return self._resolver.resolve(profile.inventory_alias, actor)
        except RemoteWorkerRegistryError:
            raise
        except Exception as exc:
            raise RemoteWorkerRegistryError(
                f"remote target authorization failed for host {host_id!r}"
            ) from exc


__all__ = [
    "InventoryResolver",
    "RemoteWorkerProfile",
    "RemoteWorkerRegistry",
    "RemoteWorkerRegistryError",
]
