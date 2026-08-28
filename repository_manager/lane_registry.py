"""Durable, fenced lane registry.

The registry is deliberately smaller than the RMDD-06 WorkItem authority.
Production callers inject a shared durable :class:`LaneAuthority`; SQLite is
only a local, best-effort status projection and can never authorize an effect.
All managed mutations require the current owner and fence once a lane is
managed.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from collections.abc import Callable, Iterable, Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Protocol, runtime_checkable

from .lane_quota import LaneQuota, LaneQuotaDecision, LaneQuotaPolicy
from .lane_record import (
    ACTIVE_STATES,
    LaneLifecycleState,
    LaneRecord,
    lane_id_for,
    new_fence,
    repository_id_for,
)

ACTIVE_SQL_STATES = tuple(state.value for state in ACTIVE_STATES)
_ACTIVE_SQL = "(" + ",".join("?" for _ in ACTIVE_SQL_STATES) + ")"


class LaneRegistryError(ValueError):
    """Base class for stable lane registry refusals."""


class LaneConflictError(LaneRegistryError):
    """An immutable identity or uniqueness key conflicts with durable state."""


class LaneAuthorizationError(LaneRegistryError):
    """Owner or fencing token does not match the current lane attempt."""


class LaneTransitionError(LaneRegistryError):
    """Lifecycle transition is not legal for the current state."""


class LaneQuotaError(LaneRegistryError):
    """Admission was refused by a count or disk quota."""

    def __init__(self, decision: LaneQuotaDecision):
        self.decision = decision
        super().__init__(
            f"lane quota refused ({decision.scope or 'quota'}): {decision.reason}"
        )


class LaneAllocationDisabled(LaneRegistryError):
    """New managed allocations are disabled during rollback."""


class StaleLaneFence(LaneAuthorizationError):
    """A stale owner/fence attempted an effect on a lane."""


@runtime_checkable
class LaneAuthority(Protocol):
    """Durable authority required for every managed lane effect.

    A graph-os/WorkItem-backed implementation supplies this protocol in
    production.  The registry never treats its SQLite projection or a process
    lock as evidence for any method below.
    """

    def allocate(self, request: Mapping[str, Any], *, now: datetime) -> LaneRecord: ...

    def get(self, lane_id: str) -> LaneRecord | None: ...

    def list_records(self) -> tuple[LaneRecord, ...]: ...

    def transition(
        self,
        lane_id: str,
        operation: str,
        *,
        owner_id: str | None,
        fence: str | None,
        target: LaneLifecycleState | str | None = None,
        updates: Mapping[str, Any] | None = None,
        now: datetime,
    ) -> LaneRecord: ...

    def heartbeat(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        updates: Mapping[str, Any],
        now: datetime,
    ) -> LaneRecord: ...


class FakeDurableLaneAuthority:
    """Explicit shared authority for tests and local projection qualification.

    This class is intentionally named ``Fake``: its in-memory lock is useful
    for deterministic tests, but it is not a cross-host guarantee.  Production
    callers must inject a graph-os/WorkItem-backed implementation of
    :class:`LaneAuthority`.
    """

    def __init__(
        self,
        *,
        quota: LaneQuotaPolicy | LaneQuota | None = None,
        clock: Callable[[], datetime] | Any | None = None,
        job_authority: Any | None = None,
    ) -> None:
        self._records: dict[str, LaneRecord] = {}
        self._request_index: dict[tuple[str, str], str] = {}
        self._events: dict[str, list[dict[str, Any]]] = {}
        self._lock = RLock()
        self._quota = quota if isinstance(quota, LaneQuota) else LaneQuota(quota)
        self._clock = clock
        self._allocation_enabled = True
        self.job_authority = job_authority

    def _now(self, value: datetime | None = None) -> datetime:
        if value is not None:
            current = value
        elif self._clock is None:
            current = datetime.now(UTC)
        elif callable(self._clock):
            current = self._clock()
        else:
            current = self._clock.now()
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("authority clock must return an aware datetime")
        return current.astimezone(UTC)

    @staticmethod
    def _active(record: LaneRecord) -> bool:
        return record.state in {
            LaneLifecycleState.ALLOCATING,
            LaneLifecycleState.ACTIVE,
            LaneLifecycleState.SUBMITTED,
            LaneLifecycleState.EXPIRED,
        }

    @staticmethod
    def _legal(current: LaneLifecycleState, target: LaneLifecycleState) -> bool:
        return target in {
            LaneLifecycleState.ALLOCATING: {
                LaneLifecycleState.ACTIVE,
                LaneLifecycleState.ABORTED,
                LaneLifecycleState.EXPIRED,
            },
            LaneLifecycleState.ACTIVE: {
                LaneLifecycleState.SUBMITTED,
                LaneLifecycleState.LANDED,
                LaneLifecycleState.ABORTED,
                LaneLifecycleState.EXPIRED,
                LaneLifecycleState.QUARANTINED,
            },
            LaneLifecycleState.SUBMITTED: {
                LaneLifecycleState.LANDED,
                LaneLifecycleState.REJECTED,
                LaneLifecycleState.ABORTED,
                LaneLifecycleState.EXPIRED,
                LaneLifecycleState.QUARANTINED,
            },
            LaneLifecycleState.EXPIRED: {LaneLifecycleState.QUARANTINED},
            LaneLifecycleState.OBSERVED_LEGACY: {LaneLifecycleState.ACTIVE},
            LaneLifecycleState.QUARANTINED: set(),
            LaneLifecycleState.LANDED: set(),
            LaneLifecycleState.ABORTED: set(),
            LaneLifecycleState.REJECTED: set(),
        }.get(current, set())

    def _authorize(
        self, record: LaneRecord, owner_id: str | None, fence: str | None
    ) -> None:
        if record.state == LaneLifecycleState.OBSERVED_LEGACY:
            raise LaneAuthorizationError("observed legacy lane has no managed owner")
        if owner_id != record.owner_id or fence != record.fence:
            raise StaleLaneFence("lane mutation refused: owner or fence is not current")
        current = getattr(self.job_authority, "is_current", None)
        if callable(current):
            for job_id in record.active_job_ids:
                try:
                    valid = current(job_id, record.attempt, record.fence)
                except TypeError:
                    valid = current(job_id, record.fence)
                if not valid:
                    raise StaleLaneFence("lane WorkItem fence is no longer current")

    def _event(self, record: LaneRecord, operation: str, actor_id: str | None) -> None:
        self._events.setdefault(record.lane_id, []).append(
            {
                "lane_id": record.lane_id,
                "operation": operation,
                "actor_id": actor_id,
                "fence": record.fence,
                "at": record.heartbeat_at.isoformat(),
            }
        )

    def allocation_enabled(self) -> bool:
        with self._lock:
            return self._allocation_enabled

    @property
    def quota(self) -> LaneQuota:
        """Authoritative quota object used by this explicit test authority."""

        return self._quota

    def set_allocation_enabled(self, enabled: bool) -> bool:
        with self._lock:
            self._allocation_enabled = bool(enabled)
            return self._allocation_enabled

    def _idempotent_existing(
        self, index_key: tuple[str, str], digest: str
    ) -> LaneRecord | None:
        existing_id = self._request_index.get(index_key)
        if existing_id is None:
            return None
        existing = self._records[existing_id]
        if existing.input_digest != digest:
            raise LaneConflictError(
                "idempotency key was reused with changed immutable lane input"
            )
        return existing

    def _check_no_conflict(
        self, repository_id: str, branch: str, worktree: str
    ) -> None:
        if any(
            self._active(item)
            and (
                (item.repository_id, item.branch) == (repository_id, branch)
                or item.worktree_path == worktree
            )
            for item in self._records.values()
        ):
            raise LaneConflictError("repository branch or worktree is already reserved")

    def _admit_quota(self, request: Mapping[str, Any], repository_id: str) -> None:
        decision = self._quota.check(
            (item for item in self._records.values() if self._active(item)),
            owner_id=str(request["owner_id"]),
            session_id=str(request["session_id"]),
            repository_id=repository_id,
            host_id=str(request.get("host_id") or "") or None,
            predicted_disk_bytes=int(request["predicted_disk_bytes"]),
        )
        if not decision.admitted:
            raise LaneQuotaError(decision)

    @staticmethod
    def _build_allocated_record(
        request: Mapping[str, Any], now: datetime
    ) -> LaneRecord:
        return LaneRecord(
            lane_id=str(request["lane_id"]),
            request_key=str(request["request_key"]),
            input_digest=str(request["input_digest"]),
            repository_id=str(request["repository_id"]),
            repository_path=str(request["repository_path"]),
            repository_origin=request.get("repository_origin"),
            branch=str(request["branch"]),
            base_ref=str(request["base_ref"]),
            base_sha=str(request.get("base_sha") or ""),
            worktree_path=str(request["worktree_path"]),
            owner_id=str(request["owner_id"]),
            session_id=str(request["session_id"]),
            host_id=str(request.get("host_id") or "") or None,
            fence=str(request["fence"]),
            created_at=now,
            heartbeat_at=now,
            expires_at=now + timedelta(seconds=int(request["ttl_seconds"])),
            ttl_seconds=int(request["ttl_seconds"]),
            predicted_disk_bytes=int(request["predicted_disk_bytes"]),
            disk_budget_bytes=int(request["disk_budget_bytes"]),
            concept_ids=tuple(request.get("concept_ids") or ()),
            active_job_ids=tuple(request.get("active_job_ids") or ()),
            active_candidate_id=request.get("active_candidate_id"),
            state=LaneLifecycleState.ALLOCATING,
            last_transition="allocate",
        )

    def allocate(self, request: Mapping[str, Any], *, now: datetime) -> LaneRecord:
        with self._lock:
            key = str(request["request_key"])
            repository_id = str(request["repository_id"])
            index_key = (repository_id, key)
            existing = self._idempotent_existing(
                index_key, str(request["input_digest"])
            )
            if existing is not None:
                return existing
            if not self._allocation_enabled:
                raise LaneAllocationDisabled(
                    "managed lane allocation is disabled; existing records are retained"
                )
            self._check_no_conflict(
                repository_id, str(request["branch"]), str(request["worktree_path"])
            )
            self._admit_quota(request, repository_id)
            record = self._build_allocated_record(request, now)
            self._records[record.lane_id] = record
            self._request_index[index_key] = record.lane_id
            self._event(record, "allocate", record.owner_id)
            return record

    def get(self, lane_id: str) -> LaneRecord | None:
        with self._lock:
            return self._records.get(lane_id)

    def list_records(self) -> tuple[LaneRecord, ...]:
        with self._lock:
            return tuple(
                sorted(
                    self._records.values(),
                    key=lambda item: (item.created_at, item.lane_id),
                )
            )

    def transition(
        self,
        lane_id: str,
        operation: str,
        *,
        owner_id: str | None,
        fence: str | None,
        target: LaneLifecycleState | str | None = None,
        updates: Mapping[str, Any] | None = None,
        now: datetime,
    ) -> LaneRecord:
        with self._lock:
            record = self._records.get(lane_id)
            if record is None:
                raise LaneRegistryError(f"unknown lane: {lane_id}")
            self._authorize(record, owner_id, fence)
            target = (
                LaneLifecycleState(target)
                if target is not None
                else {
                    "activate": LaneLifecycleState.ACTIVE,
                    "submit": LaneLifecycleState.SUBMITTED,
                    "finish": LaneLifecycleState.LANDED,
                    "abort": LaneLifecycleState.ABORTED,
                    "expire": LaneLifecycleState.EXPIRED,
                    "quarantine": LaneLifecycleState.QUARANTINED,
                }.get(operation)
            )
            if target is None:
                raise LaneTransitionError(f"unknown lane operation: {operation}")
            if record.state == target:
                return record
            if not self._legal(record.state, target):
                raise LaneTransitionError(
                    f"illegal lane transition: {record.state.value} -> {target.value}"
                )
            values = dict(updates or {})
            values.update(
                {
                    "state": target,
                    "version": record.version + 1,
                    "heartbeat_at": now,
                    "expires_at": now + timedelta(seconds=record.ttl_seconds),
                    "last_transition": operation,
                }
            )
            updated = record.model_copy(update=values)
            self._records[lane_id] = updated
            self._event(updated, operation, owner_id)
            return updated

    def heartbeat(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        updates: Mapping[str, Any],
        now: datetime,
    ) -> LaneRecord:
        with self._lock:
            record = self._records.get(lane_id)
            if record is None:
                raise LaneRegistryError(f"unknown lane: {lane_id}")
            self._authorize(record, owner_id, fence)
            if record.state not in {
                LaneLifecycleState.ALLOCATING,
                LaneLifecycleState.ACTIVE,
                LaneLifecycleState.SUBMITTED,
            }:
                raise LaneTransitionError(
                    f"lane in state {record.state.value} cannot heartbeat"
                )
            values = dict(updates)
            values.update(
                {
                    "version": record.version + 1,
                    "heartbeat_at": now,
                    "expires_at": now + timedelta(seconds=record.ttl_seconds),
                    "last_transition": "heartbeat",
                }
            )
            updated = record.model_copy(update=values)
            self._records[lane_id] = updated
            self._event(updated, "heartbeat", owner_id)
            return updated

    def observe_legacy(
        self, request: Mapping[str, Any], *, now: datetime
    ) -> LaneRecord:
        with self._lock:
            key = str(request["request_key"])
            index_key = (str(request["repository_id"]), key)
            existing_id = self._request_index.get(index_key)
            if existing_id is not None:
                existing = self._records[existing_id]
                if existing.input_digest != request["input_digest"]:
                    raise LaneConflictError(
                        "legacy observation key conflicts with durable input"
                    )
                return existing
            record = LaneRecord(
                lane_id=str(request["lane_id"]),
                request_key=key,
                input_digest=str(request["input_digest"]),
                repository_id=str(request["repository_id"]),
                repository_path=str(request["repository_path"]),
                repository_origin=request.get("repository_origin"),
                branch=str(request["branch"]),
                base_ref=str(request["base_ref"]),
                base_sha=str(request.get("base_sha") or ""),
                worktree_path=str(request["worktree_path"]),
                fence=str(request["fence"]),
                created_at=now,
                heartbeat_at=now,
                expires_at=now + timedelta(days=3650),
                ttl_seconds=3650 * 86400,
                disk_budget_bytes=1,
                state=LaneLifecycleState.OBSERVED_LEGACY,
                last_transition="observe_legacy",
            )
            self._records[record.lane_id] = record
            self._request_index[index_key] = record.lane_id
            self._event(record, "observe_legacy", None)
            return record

    @staticmethod
    def _adopted_record(
        record: LaneRecord,
        owner_id: str,
        session_id: str,
        host_id: str,
        now: datetime,
    ) -> LaneRecord:
        return record.model_copy(
            update={
                "owner_id": owner_id,
                "session_id": session_id,
                "host_id": host_id or None,
                "fence": new_fence(),
                "state": LaneLifecycleState.ACTIVE,
                "heartbeat_at": now,
                "expires_at": now + timedelta(seconds=record.ttl_seconds),
                "version": record.version + 1,
                "last_transition": "adopt",
            }
        )

    def _check_adopt_conflict(self, lane_id: str, record: LaneRecord) -> None:
        if any(
            self._active(item)
            and item.lane_id != lane_id
            and (
                (item.repository_id, item.branch)
                == (record.repository_id, record.branch)
                or item.worktree_path == record.worktree_path
            )
            for item in self._records.values()
        ):
            raise LaneConflictError("repository branch or worktree is already reserved")

    def adopt(
        self,
        lane_id: str,
        *,
        owner_id: str,
        session_id: str,
        host_id: str,
        operator_id: str,
        now: datetime,
    ) -> LaneRecord:
        with self._lock:
            record = self._records.get(lane_id)
            if record is None:
                raise LaneRegistryError(f"unknown lane: {lane_id}")
            if record.state != LaneLifecycleState.OBSERVED_LEGACY:
                if record.owner_id == owner_id and record.session_id == session_id:
                    return record
                raise LaneConflictError("only observed legacy lanes may be adopted")
            updated = self._adopted_record(record, owner_id, session_id, host_id, now)
            self._check_adopt_conflict(lane_id, record)
            self._records[lane_id] = updated
            self._event(updated, "adopt", operator_id)
            return updated

    def record_cleanup_anchor(
        self,
        lane_id: str,
        anchor: str,
        *,
        owner_id: str,
        fence: str,
        now: datetime,
    ) -> LaneRecord:
        with self._lock:
            record = self._records.get(lane_id)
            if record is None:
                raise LaneRegistryError(f"unknown lane: {lane_id}")
            self._authorize(record, owner_id, fence)
            updated = record.model_copy(
                update={
                    "cleanup_anchors": tuple(
                        sorted(set(record.cleanup_anchors + (anchor,)))
                    ),
                    "version": record.version + 1,
                    "last_transition": "anchor",
                    "heartbeat_at": now,
                    "expires_at": now + timedelta(seconds=record.ttl_seconds),
                }
            )
            self._records[lane_id] = updated
            self._event(updated, "anchor", owner_id)
            return updated

    def events(self, lane_id: str) -> tuple[dict[str, Any], ...]:
        with self._lock:
            return tuple(self._events.get(lane_id, ()))


class NativeLaneAuthorityAdapter:
    """Narrow adapter for the future/native RMDD lane authority contract.

    RMDD-06's current ``RepositoryJobService`` is a job submission/read API;
    it does not expose an atomic lane reservation, fenced transition, or
    cross-host quota transaction.  This adapter therefore accepts only an
    explicitly supplied native authority that implements those typed atomic
    operations.  It never encodes lane state into arbitrary job fields and it
    fails closed at startup when the native seam is absent.

    The native object is expected to be backed by the engine/AU durable
    WorkItem transaction, and to expose ``allocate_lane``, ``get_lane``,
    ``list_lanes``, ``transition_lane`` and ``heartbeat_lane``.  Those methods
    must return :class:`LaneRecord` values and perform uniqueness, quota,
    idempotency, and fencing atomically in the authority.
    """

    _REQUIRED = (
        "allocate_lane",
        "get_lane",
        "list_lanes",
        "transition_lane",
        "heartbeat_lane",
    )

    def __init__(self, native: Any) -> None:
        missing = [
            name for name in self._REQUIRED if not callable(getattr(native, name, None))
        ]
        if missing:
            raise LaneRegistryError(
                "native durable lane authority is unavailable; missing: "
                + ", ".join(missing)
            )
        self.native = native

    def allocate(self, request: Mapping[str, Any], *, now: datetime) -> LaneRecord:
        return self.native.allocate_lane(request, now=now)

    def get(self, lane_id: str) -> LaneRecord | None:
        return self.native.get_lane(lane_id)

    def list_records(self) -> tuple[LaneRecord, ...]:
        return tuple(self.native.list_lanes())

    def transition(
        self,
        lane_id: str,
        operation: str,
        *,
        owner_id: str | None,
        fence: str | None,
        target: LaneLifecycleState | str | None = None,
        updates: Mapping[str, Any] | None = None,
        now: datetime,
    ) -> LaneRecord:
        return self.native.transition_lane(
            lane_id,
            operation,
            owner_id=owner_id,
            fence=fence,
            target=target,
            updates=updates,
            now=now,
        )

    def heartbeat(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        updates: Mapping[str, Any],
        now: datetime,
    ) -> LaneRecord:
        return self.native.heartbeat_lane(
            lane_id,
            owner_id=owner_id,
            fence=fence,
            updates=updates,
            now=now,
        )

    def __getattr__(self, name: str) -> Any:
        # Optional native operations (adopt, legacy observation, anchors,
        # rollback control, cleanup jobs) stay on the same authority object;
        # missing methods are surfaced by LaneRegistry as a fail-closed error.
        return getattr(self.native, name)


RepositoryJobLaneAuthority = NativeLaneAuthorityAdapter


class LaneRegistry:
    """Controller over a durable lane authority and local SQLite projection."""

    _SCHEMA = """
    CREATE TABLE IF NOT EXISTS lanes (
        lane_id TEXT PRIMARY KEY,
        request_key TEXT NOT NULL,
        input_digest TEXT NOT NULL,
        repository_id TEXT NOT NULL,
        repository_path TEXT NOT NULL,
        branch TEXT NOT NULL,
        worktree_path TEXT NOT NULL,
        owner_id TEXT,
        session_id TEXT,
        host_id TEXT,
        state TEXT NOT NULL,
        fence TEXT NOT NULL,
        version INTEGER NOT NULL,
        heartbeat_at TEXT NOT NULL,
        expires_at TEXT NOT NULL,
        predicted_disk_bytes INTEGER NOT NULL,
        observed_disk_bytes INTEGER NOT NULL,
        payload TEXT NOT NULL
    );
    CREATE UNIQUE INDEX IF NOT EXISTS lanes_request_identity
      ON lanes(repository_id, request_key);
    CREATE UNIQUE INDEX IF NOT EXISTS lanes_active_branch
      ON lanes(repository_id, branch)
      WHERE state IN ('allocating', 'active', 'submitted', 'expired');
    CREATE UNIQUE INDEX IF NOT EXISTS lanes_active_worktree
      ON lanes(worktree_path)
      WHERE state IN ('allocating', 'active', 'submitted', 'expired');
    CREATE INDEX IF NOT EXISTS lanes_repository_state
      ON lanes(repository_id, state);
    CREATE INDEX IF NOT EXISTS lanes_expiry
      ON lanes(expires_at, state);
    CREATE TABLE IF NOT EXISTS lane_events (
        event_id INTEGER PRIMARY KEY AUTOINCREMENT,
        lane_id TEXT NOT NULL,
        operation TEXT NOT NULL,
        actor_id TEXT,
        fence TEXT,
        at TEXT NOT NULL,
        detail TEXT NOT NULL
    );
    CREATE TABLE IF NOT EXISTS lane_meta (
        key TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    INSERT OR IGNORE INTO lane_meta(key, value)
      VALUES ('allocation_enabled', 'true');
    """

    def __init__(
        self,
        store_path: str | Path | None = None,
        *,
        quota: LaneQuotaPolicy | LaneQuota | None = None,
        clock: Callable[[], datetime] | Any | None = None,
        job_authority: Any | None = None,
        authority: LaneAuthority | Any | None = None,
        durable_authority: LaneAuthority | Any | None = None,
    ) -> None:
        # ``authority`` is intentionally explicit.  A local SQLite file is a
        # projection for status/restart tests only; it is never allowed to
        # authorize a managed allocation or transition.
        if (
            authority is not None
            and durable_authority is not None
            and authority is not durable_authority
        ):
            raise ValueError("authority and durable_authority must identify one object")
        self.authority = (
            durable_authority if durable_authority is not None else authority
        )
        self.store_path = self._resolve_store_path(store_path)
        self._connection = self._open(self.store_path)
        self._lock = RLock()
        self.quota = quota if isinstance(quota, LaneQuota) else LaneQuota(quota)
        self._clock = clock
        self.job_authority = job_authority
        with self._lock:
            self._connection.executescript(self._SCHEMA)
            self._connection.commit()

    def _require_authority(self) -> Any:
        if self.authority is None:
            raise LaneRegistryError(
                "durable lane authority is required for managed lane mutations"
            )
        return self.authority

    def _project(self, record: LaneRecord) -> None:
        """Best-effort local projection; never an authorization path."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                row = self._connection.execute(
                    "SELECT lane_id FROM lanes WHERE lane_id=?", (record.lane_id,)
                ).fetchone()
                if row is None:
                    self._connection.execute(
                        """INSERT INTO lanes(
                           lane_id, request_key, input_digest, repository_id,
                           repository_path, branch, worktree_path, owner_id,
                           session_id, host_id, state, fence, version,
                           heartbeat_at, expires_at, predicted_disk_bytes,
                           observed_disk_bytes, payload)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            record.lane_id,
                            record.request_key,
                            record.input_digest,
                            record.repository_id,
                            record.repository_path,
                            record.branch,
                            record.worktree_path,
                            record.owner_id,
                            record.session_id,
                            record.host_id,
                            record.state.value,
                            record.fence,
                            record.version,
                            record.heartbeat_at.isoformat(),
                            record.expires_at.isoformat(),
                            record.predicted_disk_bytes,
                            record.observed_disk_bytes,
                            record.canonical_json(),
                        ),
                    )
                else:
                    self._write(self._connection, record)
                self._connection.commit()
            except Exception:
                self._connection.rollback()
                # A projection outage must not turn an already-authorized
                # durable effect into a second authorization attempt.
                return

    def _authority_records(self) -> tuple[LaneRecord, ...]:
        authority = self._require_authority()
        values = authority.list_records()
        records = tuple(values)
        for record in records:
            self._project(record)
        return records

    @staticmethod
    def _resolve_store_path(store_path: str | Path | None) -> str:
        if store_path is None:
            state_root = Path(
                os.getenv(
                    "XDG_STATE_HOME",
                    str(Path.home() / ".local" / "state"),
                )
            )
            path = state_root / "repository-manager" / "lanes.sqlite3"
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
            path,
            timeout=30,
            isolation_level=None,
            check_same_thread=False,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        if path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def close(self) -> None:
        with self._lock:
            self._connection.close()

    def __enter__(self) -> LaneRegistry:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def _now(self, value: datetime | None = None) -> datetime:
        if value is not None:
            current = value
        elif self._clock is None:
            current = datetime.now(UTC)
        elif callable(self._clock):
            current = self._clock()
        elif callable(getattr(self._clock, "now", None)):
            current = self._clock.now()
        else:
            raise TypeError("clock must be callable or provide now()")
        if not isinstance(current, datetime):
            raise TypeError("clock must return a datetime")
        if current.tzinfo is None or current.utcoffset() is None:
            raise ValueError("registry clock must return an aware datetime")
        return current.astimezone(UTC)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """Hold the process lock and a SQLite write transaction."""

        with self._lock:
            self._connection.execute("BEGIN IMMEDIATE")
            try:
                yield self._connection
            except Exception:
                self._connection.rollback()
                raise
            else:
                self._connection.commit()

    @staticmethod
    def _iso(value: datetime) -> str:
        return value.astimezone(UTC).isoformat()

    @staticmethod
    def _parse_time(value: str) -> datetime:
        return datetime.fromisoformat(value).astimezone(UTC)

    @staticmethod
    def _json(value: object) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _repository_from_mapping(
        values: Mapping[str, Any],
        repository_path: str | Path | None,
        origin: str | None,
    ) -> tuple[str, str, str | None]:
        path_value = (
            repository_path
            or values.get("canonical_path")
            or values.get("repository_path")
        )
        if not path_value:
            raise ValueError("repository identity requires canonical_path")
        canonical = str(Path(path_value).expanduser().resolve(strict=False))
        repository_value = values.get("repository_id")
        origin_value = values.get("origin") if origin is None else origin
        return (
            str(repository_value or repository_id_for(canonical, origin=origin_value)),
            canonical,
            None if origin_value is None else str(origin_value),
        )

    @staticmethod
    def _repository_from_object(
        repository: Any, origin: str | None
    ) -> tuple[str, str, str | None]:
        canonical = str(
            Path(repository.canonical_path).expanduser().resolve(strict=False)
        )
        repository_value = str(getattr(repository, "repository_id", ""))
        origin_value = (
            origin if origin is not None else getattr(repository, "origin", None)
        )
        return (
            repository_value or repository_id_for(canonical, origin=origin_value),
            canonical,
            None if origin_value is None else str(origin_value),
        )

    @staticmethod
    def _repository_from_path(
        repository: str | Path | Any,
        repository_path: str | Path | None,
        origin: str | None,
    ) -> tuple[str, str, str | None]:
        canonical_value = repository_path or repository
        canonical = str(Path(canonical_value).expanduser().resolve(strict=False))
        return repository_id_for(canonical, origin=origin), canonical, origin

    @staticmethod
    def _repository(
        repository: str | Path | Mapping[str, Any] | Any,
        *,
        repository_path: str | Path | None = None,
        origin: str | None = None,
    ) -> tuple[str, str, str | None]:
        if isinstance(repository, Mapping):
            return LaneRegistry._repository_from_mapping(
                repository, repository_path, origin
            )
        if hasattr(repository, "canonical_path"):
            return LaneRegistry._repository_from_object(repository, origin)
        return LaneRegistry._repository_from_path(repository, repository_path, origin)

    def _row(self, lane_id: str) -> sqlite3.Row | None:
        return self._connection.execute(
            "SELECT payload FROM lanes WHERE lane_id = ?", (lane_id,)
        ).fetchone()

    def get(self, lane_id: str) -> LaneRecord | None:
        """Return one durable record, or ``None`` for an unknown lane."""

        if self.authority is not None:
            record = self.authority.get(lane_id)
            if record is not None:
                self._project(record)
            return record
        with self._lock:
            row = self._row(lane_id)
        return None if row is None else LaneRecord.model_validate_json(row["payload"])

    def require(self, lane_id: str) -> LaneRecord:
        record = self.get(lane_id)
        if record is None:
            raise LaneRegistryError(f"unknown lane: {lane_id}")
        return record

    @staticmethod
    def _filter_records_by_repository(
        records: list[LaneRecord], repository_id: str | None
    ) -> list[LaneRecord]:
        if repository_id is None:
            return records
        return [item for item in records if item.repository_id == repository_id]

    @staticmethod
    def _filter_records_by_owner(
        records: list[LaneRecord], owner_id: str | None
    ) -> list[LaneRecord]:
        if owner_id is None:
            return records
        return [item for item in records if item.owner_id == owner_id]

    @staticmethod
    def _filter_records_by_state(
        records: list[LaneRecord],
        states: Iterable[LaneLifecycleState | str] | None,
        include_terminal: bool,
    ) -> list[LaneRecord]:
        normalized_states = tuple(LaneLifecycleState(state) for state in states or ())
        if normalized_states:
            return [item for item in records if item.state in normalized_states]
        if not include_terminal:
            return [item for item in records if item.state in ACTIVE_STATES]
        return records

    def _list_records_from_authority(
        self,
        *,
        repository_id: str | None,
        owner_id: str | None,
        states: Iterable[LaneLifecycleState | str] | None,
        include_terminal: bool,
    ) -> tuple[LaneRecord, ...]:
        records = list(self._authority_records())
        records = self._filter_records_by_repository(records, repository_id)
        records = self._filter_records_by_owner(records, owner_id)
        records = self._filter_records_by_state(records, states, include_terminal)
        return tuple(sorted(records, key=lambda item: (item.created_at, item.lane_id)))

    @staticmethod
    def _list_records_where(
        *,
        repository_id: str | None,
        owner_id: str | None,
        states: Iterable[LaneLifecycleState | str] | None,
        include_terminal: bool,
    ) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        values: list[Any] = []
        if repository_id is not None:
            clauses.append("repository_id = ?")
            values.append(repository_id)
        if owner_id is not None:
            clauses.append("owner_id = ?")
            values.append(owner_id)
        normalized_state_values: tuple[str, ...] = tuple(
            LaneLifecycleState(state).value for state in states or ()
        )
        if normalized_state_values:
            clauses.append(
                "state IN (" + ",".join("?" for _ in normalized_state_values) + ")"
            )
            values.extend(normalized_state_values)
        elif not include_terminal:
            clauses.append("state IN " + _ACTIVE_SQL)
            values.extend(ACTIVE_SQL_STATES)
        where = " WHERE " + " AND ".join(clauses) if clauses else ""
        return where, values

    def _list_records_from_sqlite(
        self,
        *,
        repository_id: str | None,
        owner_id: str | None,
        states: Iterable[LaneLifecycleState | str] | None,
        include_terminal: bool,
    ) -> tuple[LaneRecord, ...]:
        where, values = self._list_records_where(
            repository_id=repository_id,
            owner_id=owner_id,
            states=states,
            include_terminal=include_terminal,
        )
        with self._lock:
            rows = self._connection.execute(
                # ``created_at`` lives in the immutable JSON payload; rowid
                # preserves projection insertion order without pretending the
                # local table is the durable authority.
                #
                # ``where`` is built only from fixed literal clause
                # fragments ("repository_id = ?", "owner_id = ?", "state IN
                # (...)") joined by a fixed " AND " -- no caller-controlled
                # string is ever concatenated into the SQL text.  Every
                # actual value is bound through ``?`` placeholders via
                # ``values`` below, so this is parameterized, not
                # string-interpolated; bandit's B608 check is purely
                # syntactic and cannot tell the two apart.
                "SELECT payload FROM lanes" + where + " ORDER BY rowid, lane_id",  # nosec B608
                values,
            ).fetchall()
        return tuple(LaneRecord.model_validate_json(row["payload"]) for row in rows)

    def list_records(
        self,
        *,
        repository_id: str | None = None,
        owner_id: str | None = None,
        states: Iterable[LaneLifecycleState | str] | None = None,
        include_terminal: bool = True,
    ) -> tuple[LaneRecord, ...]:
        if self.authority is not None:
            return self._list_records_from_authority(
                repository_id=repository_id,
                owner_id=owner_id,
                states=states,
                include_terminal=include_terminal,
            )
        return self._list_records_from_sqlite(
            repository_id=repository_id,
            owner_id=owner_id,
            states=states,
            include_terminal=include_terminal,
        )

    def status(self, lane_id: str) -> LaneRecord:
        """Read-only status projection used by bounded adapters."""

        return self.require(lane_id)

    def allocation_enabled(self) -> bool:
        if self.authority is not None:
            try:
                enabled = getattr(self.authority, "allocation_enabled", None)
            except AttributeError:
                enabled = None
            if callable(enabled):
                return bool(enabled())
            return True
        with self._lock:
            row = self._connection.execute(
                "SELECT value FROM lane_meta WHERE key = 'allocation_enabled'"
            ).fetchone()
        return row is not None and row["value"].lower() == "true"

    def set_allocation_enabled(self, enabled: bool) -> bool:
        """Persist the rollback switch without touching existing records."""

        authority = self._require_authority()
        try:
            setter = getattr(authority, "set_allocation_enabled", None)
        except AttributeError:
            setter = None
        if not callable(setter):
            raise LaneRegistryError(
                "durable authority does not support rollback control"
            )
        result = bool(setter(enabled))
        with self._lock:
            self._connection.execute(
                "UPDATE lane_meta SET value = ? WHERE key = 'allocation_enabled'",
                ("true" if result else "false",),
            )
            self._connection.commit()
        return result

    @staticmethod
    def _immutable_payload(
        *,
        repository_id: str,
        repository_path: str,
        repository_origin: str | None,
        branch: str,
        base_ref: str,
        base_sha: str,
        worktree_path: str,
        owner_id: str | None,
        session_id: str | None,
        host_id: str | None,
        ttl_seconds: int,
        predicted_disk_bytes: int,
        disk_budget_bytes: int,
        concept_ids: tuple[str, ...],
        active_job_ids: tuple[str, ...],
        active_candidate_id: str | None,
    ) -> dict[str, Any]:
        return {
            "repository_id": repository_id,
            "repository_path": repository_path,
            "repository_origin": repository_origin,
            "branch": branch,
            "base_ref": base_ref,
            "base_sha": base_sha,
            "worktree_path": worktree_path,
            "owner_id": owner_id,
            "session_id": session_id,
            "host_id": host_id,
            "ttl_seconds": ttl_seconds,
            "predicted_disk_bytes": predicted_disk_bytes,
            "disk_budget_bytes": disk_budget_bytes,
            "concept_ids": concept_ids,
            "active_job_ids": active_job_ids,
            "active_candidate_id": active_candidate_id,
        }

    @staticmethod
    def _digest(payload: Mapping[str, Any]) -> str:
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _quota_records(self, connection: sqlite3.Connection) -> tuple[LaneRecord, ...]:
        # ``_ACTIVE_SQL`` is the fixed "(?,?,...)" placeholder string built
        # from ``ACTIVE_STATES`` (a fixed enum, never caller input); the
        # actual state values are bound through it via ``ACTIVE_SQL_STATES``
        # below, so this is parameterized, not string-interpolated -- the
        # same false-positive shape as the other B608 finding in this file.
        rows = connection.execute(
            "SELECT payload FROM lanes WHERE state IN " + _ACTIVE_SQL,  # nosec B608
            ACTIVE_SQL_STATES,
        ).fetchall()
        return tuple(LaneRecord.model_validate_json(row["payload"]) for row in rows)

    @staticmethod
    def _write(connection: sqlite3.Connection, record: LaneRecord) -> None:
        payload = record.canonical_json()
        connection.execute(
            """UPDATE lanes SET input_digest=?, repository_id=?, repository_path=?,
               branch=?, worktree_path=?, owner_id=?, session_id=?, host_id=?,
               state=?, fence=?, version=?, heartbeat_at=?, expires_at=?,
               predicted_disk_bytes=?, observed_disk_bytes=?, payload=?
               WHERE lane_id=?""",
            (
                record.input_digest,
                record.repository_id,
                record.repository_path,
                record.branch,
                record.worktree_path,
                record.owner_id,
                record.session_id,
                record.host_id,
                record.state.value,
                record.fence,
                record.version,
                record.heartbeat_at.isoformat(),
                record.expires_at.isoformat(),
                record.predicted_disk_bytes,
                record.observed_disk_bytes,
                payload,
                record.lane_id,
            ),
        )

    @staticmethod
    def _event(
        connection: sqlite3.Connection,
        record: LaneRecord,
        operation: str,
        *,
        actor_id: str | None = None,
        detail: Mapping[str, Any] | None = None,
    ) -> None:
        connection.execute(
            """INSERT INTO lane_events(lane_id, operation, actor_id, fence, at, detail)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.lane_id,
                operation,
                actor_id,
                record.fence,
                record.heartbeat_at.isoformat(),
                json.dumps(dict(detail or {}), sort_keys=True, separators=(",", ":")),
            ),
        )

    def events(self, lane_id: str) -> tuple[dict[str, Any], ...]:
        if self.authority is not None:
            try:
                provider = getattr(self.authority, "events", None)
            except AttributeError:
                provider = None
            if not callable(provider):
                raise LaneRegistryError("durable authority does not expose lane events")
            return tuple(provider(lane_id))
        with self._lock:
            rows = self._connection.execute(
                """SELECT event_id, operation, actor_id, fence, at, detail
                   FROM lane_events WHERE lane_id=? ORDER BY event_id""",
                (lane_id,),
            ).fetchall()
        return tuple(
            {
                "event_id": row["event_id"],
                "lane_id": lane_id,
                "operation": row["operation"],
                "actor_id": row["actor_id"],
                "fence": row["fence"],
                "at": row["at"],
                "detail": json.loads(row["detail"]),
            }
            for row in rows
        )

    @staticmethod
    def _validate_allocate_actors(branch: str, owner_id: str, session_id: str) -> None:
        if not branch or branch.strip() != branch:
            raise ValueError("branch must be non-blank")
        if not owner_id or not session_id:
            raise ValueError("owner_id and session_id are required")

    @staticmethod
    def _validate_allocate_limits(ttl_seconds: int, predicted_disk_bytes: int) -> None:
        if ttl_seconds < 1 or predicted_disk_bytes < 0:
            raise ValueError("ttl_seconds must be positive and disk bytes non-negative")

    @staticmethod
    def _resolve_request_key(
        idempotency_key: str | None, request_id: str | None
    ) -> str:
        request_key = idempotency_key or request_id
        if not request_key:
            raise ValueError("request_id or idempotency_key is required")
        return request_key

    @staticmethod
    def _resolve_worktree(
        worktree_path: str | Path | None, canonical: str, branch: str
    ) -> str:
        if worktree_path is not None:
            return str(Path(worktree_path).expanduser().resolve(strict=False))
        return str(
            (Path(canonical).parent / ".repository-manager-lanes" / branch).resolve()
        )

    @staticmethod
    def _normalize_allocate_inputs(
        concept_ids: Iterable[str],
        active_job_ids: Iterable[str],
        predicted_disk_bytes: int,
        disk_budget_bytes: int | None,
    ) -> tuple[tuple[str, ...], tuple[str, ...], int]:
        concept_tuple = tuple(sorted(set(concept_ids)))
        job_tuple = tuple(sorted(set(active_job_ids)))
        budget = max(
            1,
            disk_budget_bytes
            if disk_budget_bytes is not None
            else predicted_disk_bytes,
        )
        return concept_tuple, job_tuple, budget

    def allocate(
        self,
        repository: str | Path | Mapping[str, Any] | Any,
        branch: str,
        worktree_path: str | Path | None = None,
        *,
        owner_id: str,
        session_id: str,
        host_id: str = "",
        request_id: str | None = None,
        idempotency_key: str | None = None,
        base_ref: str = "main",
        base_sha: str = "",
        repository_path: str | Path | None = None,
        origin: str | None = None,
        ttl_seconds: int = 3600,
        predicted_disk_bytes: int = 0,
        disk_budget_bytes: int | None = None,
        concept_ids: Iterable[str] = (),
        active_job_ids: Iterable[str] = (),
        active_candidate_id: str | None = None,
        now: datetime | None = None,
    ) -> LaneRecord:
        """Reserve a lane atomically before any worktree is created.

        Repeating the same request with the same immutable input returns the
        original row.  Reusing its request key with changed input is a hard
        conflict, even when the first row is terminal.
        """

        self._validate_allocate_actors(branch, owner_id, session_id)
        self._validate_allocate_limits(ttl_seconds, predicted_disk_bytes)
        request_key = self._resolve_request_key(idempotency_key, request_id)
        repository_value, canonical, repository_origin = self._repository(
            repository, repository_path=repository_path, origin=origin
        )
        worktree = self._resolve_worktree(worktree_path, canonical, branch)
        concept_tuple, job_tuple, budget = self._normalize_allocate_inputs(
            concept_ids, active_job_ids, predicted_disk_bytes, disk_budget_bytes
        )
        immutable = self._immutable_payload(
            repository_id=repository_value,
            repository_path=canonical,
            repository_origin=repository_origin,
            branch=branch,
            base_ref=base_ref,
            base_sha=base_sha,
            worktree_path=worktree,
            owner_id=owner_id,
            session_id=session_id,
            host_id=host_id or None,
            ttl_seconds=ttl_seconds,
            predicted_disk_bytes=predicted_disk_bytes,
            disk_budget_bytes=budget,
            concept_ids=concept_tuple,
            active_job_ids=job_tuple,
            active_candidate_id=active_candidate_id,
        )
        digest = self._digest(immutable)
        current = self._now(now)
        lane_id = lane_id_for(repository_value, str(request_key))
        authority = self._require_authority()
        authority_request = {
            **immutable,
            "request_key": str(request_key),
            "input_digest": digest,
            "lane_id": lane_id,
            "fence": new_fence(),
        }
        allocated = authority.allocate(authority_request, now=current)
        self._project(allocated)
        return allocated

    @staticmethod
    def _job_fence_is_valid(
        current: Callable[..., Any], record: LaneRecord, job_id: str
    ) -> bool:
        try:
            return bool(current(job_id, record.attempt, record.fence))
        except TypeError:
            return bool(current(job_id, record.fence))

    def _check_job_fence_current(self, record: LaneRecord) -> None:
        if self.job_authority is None:
            return
        current = getattr(self.job_authority, "is_current", None)
        if not callable(current):
            return
        for job_id in record.active_job_ids:
            if not self._job_fence_is_valid(current, record, job_id):
                raise StaleLaneFence("lane WorkItem fence is no longer current")

    def _authorize(
        self,
        record: LaneRecord,
        *,
        owner_id: str | None,
        fence: str | None,
        operator: bool = False,
    ) -> None:
        if self.authority is None:
            raise LaneRegistryError(
                "durable lane authority is required for managed lane effects"
            )
        if record.state == LaneLifecycleState.OBSERVED_LEGACY and not operator:
            raise LaneAuthorizationError("observed legacy lane has no managed owner")
        if not operator and (owner_id != record.owner_id or fence != record.fence):
            raise StaleLaneFence(
                "lane mutation refused: owner or fencing token is not current"
            )
        self._check_job_fence_current(record)

    @staticmethod
    def _legal(current: LaneLifecycleState, target: LaneLifecycleState) -> bool:
        legal = {
            LaneLifecycleState.ALLOCATING: {
                LaneLifecycleState.ACTIVE,
                LaneLifecycleState.ABORTED,
                LaneLifecycleState.EXPIRED,
            },
            LaneLifecycleState.ACTIVE: {
                LaneLifecycleState.SUBMITTED,
                LaneLifecycleState.LANDED,
                LaneLifecycleState.ABORTED,
                LaneLifecycleState.EXPIRED,
                LaneLifecycleState.QUARANTINED,
            },
            LaneLifecycleState.SUBMITTED: {
                LaneLifecycleState.LANDED,
                LaneLifecycleState.REJECTED,
                LaneLifecycleState.ABORTED,
                LaneLifecycleState.EXPIRED,
                LaneLifecycleState.QUARANTINED,
            },
            LaneLifecycleState.EXPIRED: {LaneLifecycleState.QUARANTINED},
            LaneLifecycleState.QUARANTINED: set(),
            LaneLifecycleState.LANDED: set(),
            LaneLifecycleState.ABORTED: set(),
            LaneLifecycleState.REJECTED: set(),
            LaneLifecycleState.OBSERVED_LEGACY: {LaneLifecycleState.ACTIVE},
        }
        return target in legal.get(current, set())

    def _transition(
        self,
        lane_id: str,
        target: LaneLifecycleState,
        *,
        operation: str,
        owner_id: str | None,
        fence: str | None,
        now: datetime | None = None,
        updates: Mapping[str, Any] | None = None,
        operator: bool = False,
    ) -> LaneRecord:
        current_time = self._now(now)
        authority = self._require_authority()
        updated = authority.transition(
            lane_id,
            operation,
            owner_id=owner_id,
            fence=fence,
            target=target,
            updates=updates,
            now=current_time,
        )
        self._project(updated)
        return updated

    def activate(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        worktree_path: str | Path | None = None,
        now: datetime | None = None,
    ) -> LaneRecord:
        updates = {}
        if worktree_path is not None:
            updates["worktree_path"] = str(
                Path(worktree_path).expanduser().resolve(strict=False)
            )
        return self._transition(
            lane_id,
            LaneLifecycleState.ACTIVE,
            operation="activate",
            owner_id=owner_id,
            fence=fence,
            now=now,
            updates=updates,
        )

    def heartbeat(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        observed_disk_bytes: int | None = None,
        active_job_ids: Iterable[str] | None = None,
        active_candidate_id: str | None = None,
        now: datetime | None = None,
    ) -> LaneRecord:
        current_time = self._now(now)
        record = self.require(lane_id)
        observed = (
            record.observed_disk_bytes
            if observed_disk_bytes is None
            else max(0, observed_disk_bytes)
        )
        updates: dict[str, Any] = {"observed_disk_bytes": observed}
        if active_job_ids is not None:
            updates["active_job_ids"] = tuple(sorted(set(active_job_ids)))
        if active_candidate_id is not None:
            updates["active_candidate_id"] = active_candidate_id
        authority = self._require_authority()
        updated = authority.heartbeat(
            lane_id,
            owner_id=owner_id,
            fence=fence,
            updates=updates,
            now=current_time,
        )
        self._project(updated)
        return updated

    def submit(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        active_candidate_id: str | None = None,
        now: datetime | None = None,
    ) -> LaneRecord:
        return self._transition(
            lane_id,
            LaneLifecycleState.SUBMITTED,
            operation="submit",
            owner_id=owner_id,
            fence=fence,
            now=now,
            updates={"active_candidate_id": active_candidate_id},
        )

    def finish(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        state: LaneLifecycleState | str = LaneLifecycleState.LANDED,
        now: datetime | None = None,
    ) -> LaneRecord:
        target = LaneLifecycleState(state)
        if target not in {
            LaneLifecycleState.LANDED,
            LaneLifecycleState.REJECTED,
        }:
            raise ValueError("finish target must be landed or rejected")
        return self._transition(
            lane_id,
            target,
            operation="finish",
            owner_id=owner_id,
            fence=fence,
            now=now,
            updates={"active_job_ids": (), "active_candidate_id": None},
        )

    def abort(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        reason: str = "lane aborted",
        now: datetime | None = None,
    ) -> LaneRecord:
        return self._transition(
            lane_id,
            LaneLifecycleState.ABORTED,
            operation="abort",
            owner_id=owner_id,
            fence=fence,
            now=now,
            updates={
                "active_job_ids": (),
                "active_candidate_id": None,
                "last_error": reason,
            },
        )

    def expire(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        reason: str = "lane lease expired",
        now: datetime | None = None,
    ) -> LaneRecord:
        return self._transition(
            lane_id,
            LaneLifecycleState.EXPIRED,
            operation="expire",
            owner_id=owner_id,
            fence=fence,
            now=now,
            updates={"last_error": reason},
        )

    def quarantine(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        reason: str = "lane quarantined",
        now: datetime | None = None,
    ) -> LaneRecord:
        return self._transition(
            lane_id,
            LaneLifecycleState.QUARANTINED,
            operation="quarantine",
            owner_id=owner_id,
            fence=fence,
            now=now,
            updates={
                "active_job_ids": (),
                "active_candidate_id": None,
                "last_error": reason,
            },
        )

    def adopt(
        self,
        lane_id: str,
        *,
        owner_id: str,
        session_id: str,
        host_id: str = "",
        operator_id: str | None = None,
        now: datetime | None = None,
    ) -> LaneRecord:
        """Explicitly claim an ``observed_legacy`` record."""

        if not operator_id:
            raise LaneAuthorizationError(
                "legacy adoption requires explicit operator_id"
            )
        current_time = self._now(now)
        authority = self._require_authority()
        try:
            adopter = getattr(authority, "adopt", None)
        except AttributeError:
            adopter = None
        if not callable(adopter):
            raise LaneRegistryError(
                "durable authority does not support legacy adoption"
            )
        updated = adopter(
            lane_id,
            owner_id=owner_id,
            session_id=session_id,
            host_id=host_id,
            operator_id=operator_id,
            now=current_time,
        )
        self._project(updated)
        return updated

    def observe_legacy(
        self,
        repository: str | Path | Mapping[str, Any] | Any,
        branch: str,
        worktree_path: str | Path,
        *,
        base_ref: str = "main",
        base_sha: str = "",
        request_key: str | None = None,
        now: datetime | None = None,
    ) -> LaneRecord:
        """Record an existing worktree without claiming ownership."""

        repository_value, canonical, repository_origin = self._repository(repository)
        worktree = str(Path(worktree_path).expanduser().resolve(strict=False))
        key = request_key or f"legacy:{repository_value}:{worktree}"
        digest = self._digest(
            {
                "repository_id": repository_value,
                "repository_path": canonical,
                "branch": branch,
                "worktree_path": worktree,
                "base_ref": base_ref,
                "base_sha": base_sha,
            }
        )
        current = self._now(now)
        lane_id = lane_id_for(repository_value, key)
        authority = self._require_authority()
        try:
            observer = getattr(authority, "observe_legacy", None)
        except AttributeError:
            observer = None
        if not callable(observer):
            raise LaneRegistryError(
                "durable authority does not support legacy observation"
            )
        record = observer(
            {
                "lane_id": lane_id,
                "request_key": key,
                "input_digest": digest,
                "repository_id": repository_value,
                "repository_path": canonical,
                "repository_origin": repository_origin,
                "branch": branch,
                "base_ref": base_ref,
                "base_sha": base_sha,
                "worktree_path": worktree,
                "fence": new_fence(),
            },
            now=current,
        )
        self._project(record)
        return record

    def discover_legacy(
        self,
        worktrees: Iterable[Mapping[str, Any]],
        *,
        repository: str | Path | Mapping[str, Any] | Any | None = None,
        now: datetime | None = None,
    ) -> tuple[LaneRecord, ...]:
        """Discover current worktrees as unowned ``observed_legacy`` rows."""

        records: list[LaneRecord] = []
        for item in worktrees:
            path = item.get("path")
            branch = item.get("branch")
            if not path or not branch or branch == "(detached)":
                continue
            source = repository or item.get("repository_path") or item.get("repo")
            if not source:
                continue
            records.append(
                self.observe_legacy(
                    source,
                    str(branch),
                    str(path),
                    now=now,
                )
            )
        return tuple(records)

    def update_observed_disk(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        observed_disk_bytes: int,
        now: datetime | None = None,
    ) -> LaneRecord:
        return self.heartbeat(
            lane_id,
            owner_id=owner_id,
            fence=fence,
            observed_disk_bytes=observed_disk_bytes,
            now=now,
        )

    def record_cleanup_anchor(
        self,
        lane_id: str,
        anchor: str,
        *,
        owner_id: str,
        fence: str,
        now: datetime | None = None,
    ) -> LaneRecord:
        """Persist a reachable recovery ref under the current lane fence."""

        if not anchor or anchor.strip() != anchor:
            raise ValueError("cleanup anchor must be non-blank")
        current_time = self._now(now)
        authority = self._require_authority()
        try:
            recorder = getattr(authority, "record_cleanup_anchor", None)
        except AttributeError:
            recorder = None
        if not callable(recorder):
            raise LaneRegistryError(
                "durable authority does not support cleanup anchors"
            )
        updated = recorder(
            lane_id,
            anchor,
            owner_id=owner_id,
            fence=fence,
            now=current_time,
        )
        self._project(updated)
        return updated

    def reconcile(
        self, observed_worktrees: Iterable[Mapping[str, Any]]
    ) -> tuple[Any, ...]:
        """Run the read-only registry/Git reconciliation adapter."""

        from .lane_reclamation import LaneReclaimer

        return LaneReclaimer(self).reconcile(observed_worktrees)

    def quota_status(self) -> dict[str, object]:
        authority_quota = getattr(self.authority, "quota", None)
        quota = (
            authority_quota if isinstance(authority_quota, LaneQuota) else self.quota
        )
        usage = quota.usage(self.list_records(include_terminal=False))
        return {
            "usage": usage.as_dict(),
            "policy": quota.policy.__dict__.copy(),
            "allocation_enabled": self.allocation_enabled(),
        }

    def safe_commit(
        self, lane_id: str, *, owner_id: str, fence: str, **kwargs: Any
    ) -> dict[str, Any]:
        """Route lane commits through the C-12 primitive."""

        record = self.require(lane_id)
        self._authorize(record, owner_id=owner_id, fence=fence)
        from repository_manager.safe_commit import safe_commit

        return safe_commit(record.worktree_path, **kwargs)

    def park(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        git: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route WIP parking through the private-ref C-12 primitive."""

        record = self.require(lane_id)
        self._authorize(record, owner_id=owner_id, fence=fence)
        from repository_manager.stash_guard import park

        return park(git, record.worktree_path, lane=record.lane_id, **kwargs)

    def unpark(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        git: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route WIP restoration through the private-ref C-12 primitive."""

        record = self.require(lane_id)
        self._authorize(record, owner_id=owner_id, fence=fence)
        from repository_manager.stash_guard import unpark

        return unpark(git, record.worktree_path, lane=record.lane_id, **kwargs)

    def destructive_guard(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        argv: Iterable[str],
        override: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Route guarded Git mutation through the C-12 primitive."""

        record = self.require(lane_id)
        self._authorize(record, owner_id=owner_id, fence=fence)
        from repository_manager.destructive_guard import guard

        return guard(
            list(argv),
            path=record.worktree_path,
            lane=record.lane_id,
            override=override,
            **kwargs,
        )


__all__ = [
    "FakeDurableLaneAuthority",
    "InMemoryLaneAuthority",
    "LaneAllocationDisabled",
    "LaneAuthorizationError",
    "LaneConflictError",
    "LaneAuthority",
    "LaneQuotaError",
    "LaneRegistry",
    "LaneRegistryError",
    "LaneTransitionError",
    "NativeLaneAuthorityAdapter",
    "RepositoryJobLaneAuthority",
    "StaleLaneFence",
]


InMemoryLaneAuthority = FakeDurableLaneAuthority
