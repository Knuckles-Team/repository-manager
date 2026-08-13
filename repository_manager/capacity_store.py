"""Durable projection of the host capacity ledger (CONCEPT:RM-CAPACITY-STORE).

P0.7 — "``register_worker`` writes to a process-local in-memory singleton that
dies with the process." Verified in :mod:`repository_manager.remote_worker_actions`:
``_CAPACITY_INVENTORY``/``_REGISTRY`` are plain module globals with no backing
store at all, so an MCP server restart (routine — see this repo's own
container-image work) silently forgets every registered host and every
worker profile, and any caller who registered R820 an hour ago has to do it
again with no signal that anything was lost.

This module gives the ledger the SAME durability shape
:class:`repository_manager.lane_registry.LaneRegistry` already uses for
lanes — a local SQLite projection at
``${XDG_STATE_HOME:-~/.local/state}/repository-manager/capacity.sqlite3`` —
not a second, different persistence mechanism. It is honestly a PROJECTION,
exactly like the lane registry's own: :class:`CapacityInventory` (RMDD-08)
is deliberately "a pure, process-local, thread-safe accounting structure...
never a WorkItem authority", so this store's job is only to survive a
restart of that accounting, not to become a distributed authority it was
never designed to be. Written on every accepted mutation
(``register``/``refresh``/``heartbeat``/``set_state``); read once at process
start to rehydrate the in-memory :class:`~repository_manager.capacity.CapacityInventory`.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from threading import RLock

from repository_manager.capacity import HostCapacity, HostState, ResourceVector

__all__ = ["CapacityStore"]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (
    host_id TEXT PRIMARY KEY,
    payload TEXT NOT NULL,
    version INTEGER NOT NULL,
    updated_at TEXT NOT NULL
);
"""


class CapacityStore:
    """SQLite-backed projection of every registered :class:`HostCapacity`."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store_path = self._resolve_store_path(store_path)
        self._lock = RLock()
        self._connection = self._open(self.store_path)
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._connection.commit()

    @staticmethod
    def _resolve_store_path(store_path: str | Path | None) -> str:
        if store_path is None:
            state_root = Path(
                os.getenv("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
            )
            path = state_root / "repository-manager" / "capacity.sqlite3"
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        if str(store_path) == ":memory:":
            return ":memory:"
        path = Path(store_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.resolve(strict=False))

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        connection = sqlite3.connect(
            path, timeout=30, isolation_level=None, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        if path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def save(self, host: HostCapacity) -> None:
        """Persist (or monotonically refresh) one host's projection.

        Mirrors :meth:`repository_manager.capacity.CapacityInventory.register`'s
        own monotonic-version rule so the store can never regress a host to
        an older revision than what a concurrent reader already accepted in
        memory — the store is a mirror of that acceptance, never a second
        decision-maker.
        """

        payload = json.dumps(
            {
                "host_id": host.host_id,
                "total": host.total.as_dict(),
                "labels": list(host.labels),
                "target_kind": host.target_kind,
                "state": host.state.value,
                "heartbeat_at": host.heartbeat_at.isoformat(),
                "heartbeat_ttl_seconds": host.heartbeat_ttl_seconds,
                "live": host.live.as_dict(),
                "observed_disk_free_mib": host.observed_disk_free_mib,
                "version": host.version,
            },
            sort_keys=True,
        )
        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT version FROM hosts WHERE host_id=?", (host.host_id,)
                ).fetchone()
                if row is not None and int(row["version"]) >= host.version:
                    self._connection.rollback()
                    return
                self._connection.execute(
                    """INSERT INTO hosts(host_id, payload, version, updated_at)
                       VALUES (?, ?, ?, ?)
                       ON CONFLICT(host_id) DO UPDATE SET
                         payload=excluded.payload,
                         version=excluded.version,
                         updated_at=excluded.updated_at""",
                    (
                        host.host_id,
                        payload,
                        host.version,
                        host.heartbeat_at.isoformat(),
                    ),
                )
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                # A projection outage must never look like a lost host to a
                # caller who already has the in-memory registration accepted
                # (same "best-effort projection" contract lane_registry uses).
                raise

    def load_all(self) -> tuple[HostCapacity, ...]:
        """Rehydrate every persisted host as a :class:`HostCapacity`."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT payload FROM hosts ORDER BY host_id"
            ).fetchall()
        hosts: list[HostCapacity] = []
        for row in rows:
            data = json.loads(row["payload"])
            hosts.append(
                HostCapacity(
                    host_id=data["host_id"],
                    total=ResourceVector(**data["total"]),
                    labels=tuple(data["labels"]),
                    target_kind=data["target_kind"],
                    state=HostState(data["state"]),
                    # The REAL last-known heartbeat is restored, not a fresh
                    # one minted at load time — a host that has not actually
                    # reported since before the restart must still read as
                    # stale (fail closed) until it heartbeats again, never be
                    # given a synthetic proof of liveness it did not send.
                    heartbeat_at=datetime.fromisoformat(data["heartbeat_at"]),
                    live=ResourceVector(**data["live"]),
                    observed_disk_free_mib=data["observed_disk_free_mib"],
                    heartbeat_ttl_seconds=data["heartbeat_ttl_seconds"],
                    version=data["version"],
                )
            )
        return tuple(hosts)
