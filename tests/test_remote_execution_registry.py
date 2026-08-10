"""RMDD-15: remote worker registry -- unauthorized-host refusal at claim time.

``RemoteWorkerRegistry.recheck_at_claim`` is the precondition every dispatch
must pass immediately before a :class:`RemoteWorkerExecutor` is constructed.
These tests never import ``tunnel_manager``: ``AuthorizedTarget`` is used only
as a type annotation in ``registry.py`` (a ``TYPE_CHECKING``-only import), so
a lightweight fake resolver satisfying the structural ``InventoryResolver``
protocol is sufficient and keeps this suite runnable without the optional
dependency.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from repository_manager.capacity import CapacityInventory, HostCapacity, HostState, ResourceVector
from repository_manager.remote_execution.registry import (
    RemoteWorkerProfile,
    RemoteWorkerRegistry,
    RemoteWorkerRegistryError,
)

NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


class _FakeAuthorizedTarget:
    def __init__(self, alias: str) -> None:
        self.alias = alias


class _FakeInventoryResolver:
    """Structural fake satisfying ``InventoryResolver`` without tunnel_manager."""

    def __init__(self, authorized_aliases: set[str]) -> None:
        self.authorized_aliases = set(authorized_aliases)
        self.calls: list[tuple[str, object]] = []

    def resolve(self, alias: str, actor: object) -> _FakeAuthorizedTarget:
        self.calls.append((alias, actor))
        if alias not in self.authorized_aliases:
            raise RemoteWorkerRegistryError("remote target authorization failed")
        return _FakeAuthorizedTarget(alias)

    def revoke(self, alias: str) -> None:
        self.authorized_aliases.discard(alias)


def _host(
    host_id: str,
    *,
    state: HostState = HostState.ACTIVE,
    heartbeat_at: datetime = NOW,
    heartbeat_ttl_seconds: int = 120,
    target_kind: str = "inventory_alias",
) -> HostCapacity:
    return HostCapacity(
        host_id,
        ResourceVector(cpu_weight=4, memory_mib=4096, disk_mib=4096, process_slots=2),
        target_kind=target_kind,
        state=state,
        heartbeat_at=heartbeat_at,
        heartbeat_ttl_seconds=heartbeat_ttl_seconds,
    )


def _registry(
    *hosts: HostCapacity, authorized_aliases: set[str] | None = None
) -> tuple[RemoteWorkerRegistry, _FakeInventoryResolver]:
    capacity = CapacityInventory(hosts)
    resolver = _FakeInventoryResolver(authorized_aliases or set())
    return RemoteWorkerRegistry(capacity, inventory_resolver=resolver), resolver


def test_register_profile_refuses_a_host_with_no_capacity_record() -> None:
    registry, _ = _registry()
    with pytest.raises(RemoteWorkerRegistryError):
        registry.register_profile(
            RemoteWorkerProfile(
                host_id="host:build-1",
                inventory_alias="build-1",
                repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
            )
        )


def test_recheck_at_claim_succeeds_for_a_healthy_authorized_host() -> None:
    registry, resolver = _registry(
        _host("host:build-1"), authorized_aliases={"build-1"}
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
            toolchains=("python3.13",),
        )
    )
    target = registry.recheck_at_claim(
        "host:build-1",
        actor=object(),
        repository_id="repository:demo",
        required_toolchain="python3.13",
        now=NOW,
    )
    assert target.alias == "build-1"
    assert len(resolver.calls) == 1
    assert resolver.calls[0][0] == "build-1"


def test_recheck_at_claim_refuses_an_unregistered_host() -> None:
    registry, _ = _registry()
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:unknown", actor=object(), repository_id="repository:demo", now=NOW
        )


def test_recheck_at_claim_refuses_a_drained_host() -> None:
    registry, _ = _registry(
        _host("host:build-1", state=HostState.DRAINED),
        authorized_aliases={"build-1"},
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
        )
    )
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1", actor=object(), repository_id="repository:demo", now=NOW
        )


def test_recheck_at_claim_refuses_a_quarantined_host() -> None:
    registry, _ = _registry(
        _host("host:build-1", state=HostState.QUARANTINED),
        authorized_aliases={"build-1"},
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
        )
    )
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1", actor=object(), repository_id="repository:demo", now=NOW
        )


def test_recheck_at_claim_refuses_a_stale_heartbeat() -> None:
    registry, _ = _registry(
        _host(
            "host:build-1",
            heartbeat_at=NOW - timedelta(seconds=600),
            heartbeat_ttl_seconds=120,
        ),
        authorized_aliases={"build-1"},
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
        )
    )
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1", actor=object(), repository_id="repository:demo", now=NOW
        )


def test_recheck_at_claim_refuses_an_unauthorized_repository() -> None:
    registry, _ = _registry(
        _host("host:build-1"), authorized_aliases={"build-1"}
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:other": "/srv/remote-worktrees/other"},
        )
    )
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1", actor=object(), repository_id="repository:demo", now=NOW
        )


def test_recheck_at_claim_refuses_a_missing_required_toolchain() -> None:
    registry, _ = _registry(
        _host("host:build-1"), authorized_aliases={"build-1"}
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
            toolchains=("go1.24",),
        )
    )
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1",
            actor=object(),
            repository_id="repository:demo",
            required_toolchain="rust-1.80",
            now=NOW,
        )


def test_recheck_at_claim_refuses_a_revoked_entitlement_even_though_capacity_is_healthy() -> (
    None
):
    """The critical unauthorized-host proof: a mid-flight entitlement revocation.

    Capacity/toolchain/root checks all pass; only the alias entitlement was
    revoked between plan time and claim time.  ``recheck_at_claim`` always
    re-resolves last, so a revoked entitlement always wins over a stale local
    "it was fine a minute ago" assumption.
    """

    registry, resolver = _registry(
        _host("host:build-1"), authorized_aliases={"build-1"}
    )
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
        )
    )
    # Prove it succeeds before revocation.
    registry.recheck_at_claim(
        "host:build-1", actor=object(), repository_id="repository:demo", now=NOW
    )
    resolver.revoke("build-1")
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1", actor=object(), repository_id="repository:demo", now=NOW
        )


def test_register_profile_and_recheck_both_require_a_live_capacity_record() -> None:
    """A host profile can never be registered, or reused, without capacity.

    ``register_profile`` refuses up front for a host with no capacity record
    (proven separately above); this proves ``recheck_at_claim`` independently
    re-checks ``capacity.get(host_id)`` rather than trusting that a profile
    already exists -- a profile for a host whose capacity was withdrawn after
    registration must still refuse at claim time, not just at registration
    time.
    """

    capacity = CapacityInventory([_host("host:build-1")])
    resolver = _FakeInventoryResolver({"build-1"})
    registry = RemoteWorkerRegistry(capacity, inventory_resolver=resolver)
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
        )
    )
    # Withdraw the host's own capacity record's freshness by moving past its
    # heartbeat TTL rather than mutating CapacityInventory's private state --
    # this exercises the same "capacity is no longer live" refusal path
    # through the inventory's own public heartbeat/TTL semantics.
    stale_at = NOW + timedelta(seconds=600)
    with pytest.raises(RemoteWorkerRegistryError):
        registry.recheck_at_claim(
            "host:build-1",
            actor=object(),
            repository_id="repository:demo",
            now=stale_at,
        )


def test_authorized_root_refuses_an_unmapped_repository() -> None:
    registry, _ = _registry(_host("host:build-1"), authorized_aliases={"build-1"})
    registry.register_profile(
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": "/srv/remote-worktrees/demo"},
        )
    )
    assert (
        registry.authorized_root("host:build-1", "repository:demo")
        == "/srv/remote-worktrees/demo"
    )
    with pytest.raises(RemoteWorkerRegistryError):
        registry.authorized_root("host:build-1", "repository:other")


@pytest.mark.parametrize("bad_root", ["relative/path", ""])
def test_profile_refuses_a_non_absolute_authorized_root(bad_root: str) -> None:
    with pytest.raises(RemoteWorkerRegistryError):
        RemoteWorkerProfile(
            host_id="host:build-1",
            inventory_alias="build-1",
            repository_roots={"repository:demo": bad_root},
        )
