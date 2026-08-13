"""Seed host capacity from the fleet's existing SSH inventory
(CONCEPT:RM-CAPACITY-SEED).

P0.7 — "seed it from the existing tunnel-manager host inventory rather than
requiring hand registration." ``~/.config/agent-utilities/inventory.yaml``
already lists every real host (``r510``, ``r710``, ``r820``, ``rw710``,
``gb10``, ...) — but it is tunnel-manager's own SSH inventory, a DIFFERENT
system with a different purpose (host discovery + SSH connection material),
not a capacity ledger. Per the explicit instruction not to couple the two,
this module only READS the plain YAML file — it does not import
``tunnel_manager``, does not touch its models, and does not become a second
copy of its inventory. It adapts what that file already says (a host exists,
reachable via SSH alias ``<name>``) into the ONE thing
:class:`repository_manager.capacity.HostCapacity` needs to exist at all: a
declared, versioned capacity record.

**Honesty about what a seed is not.** The inventory file carries no CPU/RAM/
disk numbers — those live on the actual machine, not in a static YAML file.
A seed is therefore a conservative PLACEHOLDER capacity, deliberately marked
already-stale (``heartbeat_at`` set far in the past) so
:meth:`~repository_manager.capacity.CapacityInventory.can_fit` refuses to
admit real work against it until an operator sends a real heartbeat/recheck
— seeding must never be mistaken for having actually measured the host.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from repository_manager.capacity import HostCapacity, HostState, ResourceVector

__all__ = [
    "InventoryHost",
    "SeedResult",
    "default_inventory_path",
    "parse_inventory_hosts",
    "seed_capacity",
]

#: Conservative placeholder capacity for a host whose real specs have not yet
#: been measured. Deliberately small: a seed must never let an unmeasured
#: host look more capable than it is — `can_fit` is refused by construction
#: (see `_SEED_EPOCH`) until a real heartbeat replaces this anyway, but a
#: small default also bounds the damage of a future caller that only checks
#: `HostCapacity.total` and skips freshness for some other purpose.
_SEED_CPU_WEIGHT = 4
_SEED_MEMORY_MIB = 8192
_SEED_DISK_MIB = 51_200
_SEED_PROCESS_SLOTS = 2

#: A seed has never actually heartbeat-ed. Anchoring it here (rather than
#: `datetime.now(UTC)`) means `HostCapacity.is_fresh()` reads FALSE the
#: instant it is registered — a seed cannot masquerade as a live observation.
_SEED_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


def default_inventory_path() -> Path:
    """``~/.config/agent-utilities/inventory.yaml`` (or ``$AGENT_UTILITIES_CONFIG_DIR``).

    Mirrors ``find_workspace_manifest``-style resolution without importing
    agent-utilities' config module, so this stays a plain file read.
    """

    config_dir = os.getenv(
        "AGENT_UTILITIES_CONFIG_DIR",
        str(Path.home() / ".config" / "agent-utilities"),
    )
    return Path(config_dir) / "inventory.yaml"


@dataclass(frozen=True)
class InventoryHost:
    """One host as tunnel-manager's inventory describes it -- SSH alias only."""

    alias: str
    group: str
    role: str = ""


def parse_inventory_hosts(path: Path | str | None = None) -> tuple[InventoryHost, ...]:
    """Read the ansible-shaped inventory YAML and return its host aliases.

    Never raises on a missing/empty file -- an absent inventory means "seed
    nothing", not an error; this module has no opinion on whether the file
    should exist.
    """

    target = Path(path) if path is not None else default_inventory_path()
    if not target.is_file():
        return ()
    try:
        data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return ()
    if not isinstance(data, dict):
        return ()

    hosts: list[InventoryHost] = []
    children = (data.get("all") or {}).get("children") or {}
    if not isinstance(children, dict):
        return ()
    for group_name, group in children.items():
        if not isinstance(group, dict):
            continue
        group_hosts = group.get("hosts")
        if not isinstance(group_hosts, dict):
            continue
        for alias, attrs in group_hosts.items():
            if not isinstance(alias, str) or not alias.strip():
                continue
            role = ""
            if isinstance(attrs, dict):
                role = str(attrs.get("role") or "")
            hosts.append(InventoryHost(alias=alias, group=str(group_name), role=role))
    return tuple(hosts)


@dataclass(frozen=True)
class SeedResult:
    seeded: tuple[str, ...]
    already_registered: tuple[str, ...]
    inventory_path: str
    inventory_host_count: int


def seed_capacity(
    inventory: Any,
    *,
    path: Path | str | None = None,
) -> SeedResult:
    """Register a placeholder :class:`HostCapacity` for every inventory host
    not already registered on *inventory* (a live
    :class:`~repository_manager.capacity.CapacityInventory`).

    Never overwrites an existing registration -- ``CapacityInventory.register``
    already refuses a stale/duplicate version, and this passes ``version=1``
    for every seed, so a host an operator has already registered (and is
    therefore at ``version >= 1`` with real measured capacity) is left alone.
    """

    resolved_path = Path(path) if path is not None else default_inventory_path()
    inventory_hosts = parse_inventory_hosts(resolved_path)
    seeded: list[str] = []
    already: list[str] = []
    for host in inventory_hosts:
        if inventory.get(host.alias) is not None:
            already.append(host.alias)
            continue
        labels = ("seed",) if not host.role else ("seed", host.role)
        record = HostCapacity(
            host_id=host.alias,
            total=ResourceVector(
                cpu_weight=_SEED_CPU_WEIGHT,
                memory_mib=_SEED_MEMORY_MIB,
                disk_mib=_SEED_DISK_MIB,
                process_slots=_SEED_PROCESS_SLOTS,
            ),
            labels=labels,
            target_kind="inventory_alias",
            state=HostState.ACTIVE,
            heartbeat_at=_SEED_EPOCH,
            version=1,
        )
        if inventory.register(record):
            seeded.append(host.alias)
    return SeedResult(
        seeded=tuple(seeded),
        already_registered=tuple(already),
        inventory_path=str(resolved_path),
        inventory_host_count=len(inventory_hosts),
    )
