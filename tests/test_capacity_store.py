"""``CapacityStore`` — the durable projection backing the capacity ledger."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path

from repository_manager.capacity import HostCapacity, HostState, ResourceVector
from repository_manager.capacity_store import CapacityStore


def _host(host_id: str = "r820", version: int = 1, **overrides: object) -> HostCapacity:
    host = HostCapacity(
        host_id=host_id,
        total=ResourceVector(cpu_weight=32, memory_mib=131072, disk_mib=2_000_000),
        labels=("gpu",),
        target_kind="inventory_alias",
        version=version,
    )
    return replace(host, **overrides) if overrides else host  # type: ignore[arg-type]


def test_save_and_load_round_trips_a_host(tmp_path: Path) -> None:
    store = CapacityStore(tmp_path / "capacity.sqlite3")
    store.save(_host())
    loaded = store.load_all()
    assert len(loaded) == 1
    assert loaded[0].host_id == "r820"
    assert loaded[0].total.cpu_weight == 32
    assert loaded[0].labels == ("gpu",)
    assert loaded[0].target_kind == "inventory_alias"


def test_survives_a_process_restart_a_fresh_store_instance_still_sees_it(
    tmp_path: Path,
) -> None:
    """The literal claim under test: a NEW `CapacityStore` object (standing
    in for a fresh process) opened against the SAME path sees what an
    earlier one persisted — the bad state ("registering forgot on restart")
    is the one this proves is now impossible.
    """

    path = tmp_path / "capacity.sqlite3"
    CapacityStore(path).save(_host())
    reopened = CapacityStore(path)
    assert [h.host_id for h in reopened.load_all()] == ["r820"]


def test_a_stale_version_write_is_a_no_op(tmp_path: Path) -> None:
    store = CapacityStore(tmp_path / "capacity.sqlite3")
    store.save(_host(version=3))
    store.save(_host(version=2))  # older revision must not roll back
    [loaded] = store.load_all()
    assert loaded.version == 3


def test_a_newer_version_overwrites(tmp_path: Path) -> None:
    store = CapacityStore(tmp_path / "capacity.sqlite3")
    store.save(_host(version=1, state=HostState.ACTIVE))
    store.save(_host(version=2, state=HostState.DRAINING))
    [loaded] = store.load_all()
    assert loaded.version == 2
    assert loaded.state == HostState.DRAINING


def test_the_real_last_known_heartbeat_is_restored_not_a_fresh_one(
    tmp_path: Path,
) -> None:
    """A host that has not actually reported since before a restart must
    still read as STALE afterward — restoring a synthetic fresh heartbeat
    would fabricate liveness the host never sent (fail-closed on an
    unconfirmed observation).
    """

    old = datetime(2020, 1, 1, tzinfo=UTC)
    store = CapacityStore(tmp_path / "capacity.sqlite3")
    store.save(_host(heartbeat_at=old, heartbeat_ttl_seconds=120))
    [loaded] = store.load_all()
    assert loaded.heartbeat_at == old
    assert not loaded.is_fresh(datetime.now(UTC))


def test_multiple_hosts_round_trip_independently(tmp_path: Path) -> None:
    store = CapacityStore(tmp_path / "capacity.sqlite3")
    store.save(_host("r510"))
    store.save(_host("r710"))
    store.save(_host("r820"))
    assert sorted(h.host_id for h in store.load_all()) == ["r510", "r710", "r820"]
