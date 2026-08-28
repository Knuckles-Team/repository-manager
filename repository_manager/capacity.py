"""Host capacity inventory and atomic weighted accounting.

The inventory is deliberately independent of process execution.  It records
what a worker has declared and what the scheduler has reserved; it never
discovers hosts, opens SSH, starts a command, or takes a filesystem lease.  A
remote worker is therefore represented by the same host record as a local
worker without accidentally acquiring a local ``flock``.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, replace
from dataclasses import field as dataclass_field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from threading import RLock

from repository_manager.development import CapacitySnapshot


class HostState(StrEnum):
    """Admission disposition of a worker host."""

    ACTIVE = "active"
    DRAINING = "draining"
    DRAINED = "drained"
    QUARANTINED = "quarantined"
    OFFLINE = "offline"


class CapacityError(ValueError):
    """Capacity inventory input or accounting error."""


def _validate_positive_int(value: int, message: str) -> None:
    """Raise ``CapacityError(message)`` unless ``value`` is a positive int.

    ``bool`` is rejected even though it is technically an ``int`` subclass.
    """

    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise CapacityError(message)


@dataclass(frozen=True)
class ResourceVector:
    """Four weighted dimensions used for admission and accounting."""

    cpu_weight: int = 0
    memory_mib: int = 0
    disk_mib: int = 0
    process_slots: int = 0

    def __post_init__(self) -> None:
        for name in ("cpu_weight", "memory_mib", "disk_mib", "process_slots"):
            value = getattr(self, name)
            if isinstance(value, bool) or value < 0:
                raise CapacityError(f"{name} must be a non-negative integer")

    def __add__(self, other: ResourceVector) -> ResourceVector:
        return ResourceVector(
            self.cpu_weight + other.cpu_weight,
            self.memory_mib + other.memory_mib,
            self.disk_mib + other.disk_mib,
            self.process_slots + other.process_slots,
        )

    def __sub__(self, other: ResourceVector) -> ResourceVector:
        result = ResourceVector(
            self.cpu_weight - other.cpu_weight,
            self.memory_mib - other.memory_mib,
            self.disk_mib - other.disk_mib,
            self.process_slots - other.process_slots,
        )
        if (
            min(
                result.cpu_weight,
                result.memory_mib,
                result.disk_mib,
                result.process_slots,
            )
            < 0
        ):
            raise CapacityError("resource vector accounting would become negative")
        return result

    def fits(self, requirement: ResourceVector) -> bool:
        return (
            self.cpu_weight >= requirement.cpu_weight
            and self.memory_mib >= requirement.memory_mib
            and self.disk_mib >= requirement.disk_mib
            and self.process_slots >= requirement.process_slots
        )

    def as_dict(self) -> dict[str, int]:
        return {
            "cpu_weight": self.cpu_weight,
            "memory_mib": self.memory_mib,
            "disk_mib": self.disk_mib,
            "process_slots": self.process_slots,
        }


@dataclass(frozen=True)
class HostCapacity:
    """A heartbeat-bearing host capacity record.

    ``observed_disk_free_mib`` is the most recent filesystem observation.  It
    may be lower than ``total.disk_mib - live.disk_mib`` because unrelated
    activity can consume the disk; admission uses the lower of those values.
    ``reserved`` is scheduler-owned accounting and is always subtracted again
    from the observed free value.
    """

    host_id: str
    total: ResourceVector
    labels: tuple[str, ...] = ()
    target_kind: str = "local"
    state: HostState = HostState.ACTIVE
    heartbeat_at: datetime = dataclass_field(default_factory=lambda: datetime.now(UTC))
    heartbeat_ttl_seconds: int = 120
    live: ResourceVector = ResourceVector()
    observed_disk_free_mib: int | None = None
    # Host policy/telemetry revisions are monotonic.  A refresh with an older
    # revision must never roll back a newer drain, label, heartbeat, or disk
    # observation while reservations are held.
    version: int = 1

    def __post_init__(self) -> None:
        self._validate_identity()
        self._validate_telemetry_bounds()
        self._validate_labels()
        self._validate_usage()

    def _validate_identity(self) -> None:
        if not self.host_id or self.host_id.strip() != self.host_id:
            raise CapacityError("host_id must be non-blank")
        if self.target_kind not in {"local", "remote", "inventory_alias"}:
            raise CapacityError(f"unknown target_kind {self.target_kind!r}")

    def _validate_telemetry_bounds(self) -> None:
        if self.heartbeat_ttl_seconds < 1:
            raise CapacityError("heartbeat_ttl_seconds must be positive")
        _validate_positive_int(self.version, "host capacity version must be positive")

    def _validate_labels(self) -> None:
        if any(not label.strip() for label in self.labels):
            raise CapacityError("host labels must be non-blank")

    def _validate_usage(self) -> None:
        if not self.total.fits(self.live):
            raise CapacityError("live usage cannot exceed host capacity")
        if self.observed_disk_free_mib is not None and self.observed_disk_free_mib < 0:
            raise CapacityError("observed_disk_free_mib cannot be negative")

    @property
    def is_remote(self) -> bool:
        return self.target_kind in {"remote", "inventory_alias"}

    def is_fresh(self, now: datetime | None = None) -> bool:
        now = (now or datetime.now(UTC)).astimezone(UTC)
        return now - self.heartbeat_at <= timedelta(seconds=self.heartbeat_ttl_seconds)


@dataclass(frozen=True)
class CapacityView:
    """Read-only accounting returned to explain/status callers."""

    host_id: str
    state: HostState
    labels: tuple[str, ...]
    target_kind: str
    total: ResourceVector
    live: ResourceVector
    reserved: ResourceVector
    available: ResourceVector
    heartbeat_at: datetime
    heartbeat_fresh: bool
    observed_disk_free_mib: int | None
    heartbeat_ttl_seconds: int = 120
    version: int = 1

    def __post_init__(self) -> None:
        _validate_positive_int(self.version, "capacity view version must be positive")

    @property
    def is_remote(self) -> bool:
        return self.target_kind in {"remote", "inventory_alias"}

    def to_contract(self) -> CapacitySnapshot:
        """Project the view onto the frozen C-03 ``CapacitySnapshot``."""

        return CapacitySnapshot(
            cpu_weight_total=self.total.cpu_weight,
            cpu_weight_available=self.available.cpu_weight,
            memory_mib_total=self.total.memory_mib,
            memory_mib_available=self.available.memory_mib,
            disk_mib_total=self.total.disk_mib,
            disk_mib_available=self.available.disk_mib,
            process_slots_total=self.total.process_slots,
            process_slots_available=self.available.process_slots,
        )

    def as_dict(self) -> dict[str, object]:
        return {
            "host_id": self.host_id,
            "state": self.state.value,
            "labels": list(self.labels),
            "target_kind": self.target_kind,
            "total": self.total.as_dict(),
            "live": self.live.as_dict(),
            "reserved": self.reserved.as_dict(),
            "available": self.available.as_dict(),
            "heartbeat_at": self.heartbeat_at.isoformat(),
            "heartbeat_fresh": self.heartbeat_fresh,
            "observed_disk_free_mib": self.observed_disk_free_mib,
            "heartbeat_ttl_seconds": self.heartbeat_ttl_seconds,
            "version": self.version,
        }


def _next_monotonic_version(
    current_version: int, version: int | None, error_message: str
) -> int | None:
    """Resolve the next monotonic revision for a refresh call.

    Returns ``current_version + 1`` when ``version`` is omitted (the local
    convenience meaning "next"). When ``version`` is given, it must be a
    positive int; returns ``None`` when it is not newer than
    ``current_version`` (an idempotent no-op the caller should short-circuit
    on), otherwise returns ``version`` unchanged.
    """

    if version is None:
        return current_version + 1
    _validate_positive_int(version, error_message)
    if version <= current_version:
        return None
    return version


class CapacityInventory:
    """Thread-safe host inventory with exact, non-negative reservation totals."""

    def __init__(self, hosts: Iterable[HostCapacity] = ()) -> None:
        self._lock = RLock()
        self._hosts: dict[str, HostCapacity] = {}
        self._reserved: dict[str, dict[str, ResourceVector]] = {}
        for host in hosts:
            self.register(host)

    def register(self, host: HostCapacity, *, replace: bool = False) -> bool:
        """Register or monotonically refresh one host projection.

        ``replace`` is retained for source compatibility, but it cannot make
        an older revision authoritative.  Equal or stale revisions are a
        no-op; callers can use the boolean to distinguish an accepted refresh.
        Held reservation accounting is intentionally preserved across an
        accepted refresh.
        """

        with self._lock:
            current = self._hosts.get(host.host_id)
            if current is not None:
                if host.version <= current.version:
                    return False
                held = self._reserved_total(host.host_id)
                if not host.total.fits(host.live + held):
                    raise CapacityError(
                        f"host {host.host_id!r} refresh would invalidate held capacity"
                    )
            self._hosts[host.host_id] = host
            self._reserved.setdefault(host.host_id, {})
            return True

    def refresh(self, host: HostCapacity) -> bool:
        """Apply a monotonic host policy/telemetry refresh."""

        return self.register(host, replace=True)

    def get(self, host_id: str) -> HostCapacity | None:
        with self._lock:
            return self._hosts.get(host_id)

    def require(self, host_id: str) -> HostCapacity:
        host = self.get(host_id)
        if host is None:
            raise CapacityError(
                f"unknown host {host_id!r}; capacity must be registered"
            )
        return host

    def heartbeat(
        self,
        host_id: str,
        *,
        at: datetime | None = None,
        live: ResourceVector | None = None,
        observed_disk_free_mib: int | None = None,
        version: int | None = None,
    ) -> HostCapacity:
        """Apply a heartbeat only at a newer monotonic host revision.

        Omitting ``version`` is a local convenience meaning ``current + 1``.
        Explicit equal/stale revisions are idempotent no-ops and cannot replay
        different telemetry into an existing revision.
        """

        with self._lock:
            host = self.require(host_id)
            next_version = _next_monotonic_version(
                host.version, version, "heartbeat version must be positive"
            )
            if next_version is None:
                return host
            next_live = live if live is not None else host.live
            if not host.total.fits(next_live + self._reserved_total(host_id)):
                raise CapacityError(
                    f"heartbeat for host {host_id!r} would invalidate held capacity"
                )
            updated = replace(
                host,
                heartbeat_at=(at or datetime.now(UTC)).astimezone(UTC),
                live=next_live,
                observed_disk_free_mib=(
                    observed_disk_free_mib
                    if observed_disk_free_mib is not None
                    else host.observed_disk_free_mib
                ),
                version=next_version,
            )
            self._hosts[host_id] = updated
            return updated

    def set_state(
        self, host_id: str, state: HostState, *, version: int | None = None
    ) -> HostCapacity:
        """Apply a host-state transition at a newer monotonic revision."""

        with self._lock:
            host = self.require(host_id)
            next_version = _next_monotonic_version(
                host.version, version, "state version must be positive"
            )
            if next_version is None:
                return host
            updated = replace(
                host,
                state=HostState(state),
                version=next_version,
            )
            self._hosts[host_id] = updated
            return updated

    def reserved_for(self, host_id: str) -> ResourceVector:
        with self._lock:
            return self._reserved_total(host_id)

    def reservation_ids(self, host_id: str = "") -> tuple[str, ...]:
        with self._lock:
            if host_id:
                return tuple(sorted(self._reserved.get(host_id, {})))
            return tuple(
                sorted(
                    reservation_id
                    for records in self._reserved.values()
                    for reservation_id in records
                )
            )

    def snapshot(self, host_id: str, *, now: datetime | None = None) -> CapacityView:
        with self._lock:
            host = self.require(host_id)
            reserved = self._reserved_total(host_id)
            # Disk observations include unrelated host activity.  Reservations
            # are subtracted from both the declared and observed view so a
            # prediction can never admit beyond either boundary.
            declared_available = host.total - host.live
            if host.observed_disk_free_mib is None:
                disk_free = declared_available.disk_mib
            else:
                disk_free = min(
                    declared_available.disk_mib, host.observed_disk_free_mib
                )
            available = ResourceVector(
                cpu_weight=declared_available.cpu_weight - reserved.cpu_weight,
                memory_mib=declared_available.memory_mib - reserved.memory_mib,
                disk_mib=max(0, disk_free - reserved.disk_mib),
                process_slots=declared_available.process_slots - reserved.process_slots,
            )
            # ``ResourceVector.__sub__`` protects CPU/memory/process, while a
            # stale disk observation is clamped at zero intentionally.
            available = ResourceVector(
                max(0, available.cpu_weight),
                max(0, available.memory_mib),
                available.disk_mib,
                max(0, available.process_slots),
            )
            return CapacityView(
                host_id=host.host_id,
                state=host.state,
                labels=host.labels,
                target_kind=host.target_kind,
                total=host.total,
                live=host.live,
                reserved=reserved,
                available=available,
                heartbeat_at=host.heartbeat_at,
                heartbeat_fresh=host.is_fresh(now),
                observed_disk_free_mib=host.observed_disk_free_mib,
                heartbeat_ttl_seconds=host.heartbeat_ttl_seconds,
                version=host.version,
            )

    def can_fit(
        self,
        host_id: str,
        requirement: ResourceVector,
        *,
        now: datetime | None = None,
    ) -> tuple[bool, str]:
        view = self.snapshot(host_id, now=now)
        if view.state in {HostState.DRAINED, HostState.DRAINING}:
            return False, f"host {host_id!r} is {view.state.value}"
        if view.state in {HostState.QUARANTINED, HostState.OFFLINE}:
            return False, f"host {host_id!r} is {view.state.value}"
        if not view.heartbeat_fresh:
            return False, f"host {host_id!r} heartbeat is stale"
        if not view.available.fits(requirement):
            return False, "insufficient weighted CPU/memory/disk/process capacity"
        return True, "capacity available"

    def try_reserve(
        self,
        host_id: str,
        reservation_id: str,
        requirement: ResourceVector,
        *,
        now: datetime | None = None,
    ) -> bool:
        """Atomically consume capacity for one reservation ID.

        This method only accounts for scheduler-owned capacity.  The caller
        must pair it with a WorkItem-native CAS; it is not a replacement for
        that fence authority.
        """

        with self._lock:
            if reservation_id in self._reserved.get(host_id, {}):
                return self._reserved[host_id][reservation_id] == requirement
            fits, _ = self.can_fit(host_id, requirement, now=now)
            if not fits:
                return False
            self._reserved.setdefault(host_id, {})[reservation_id] = requirement
            return True

    def restore_reservation(
        self,
        host_id: str,
        reservation_id: str,
        requirement: ResourceVector,
    ) -> bool:
        """Restore durable held accounting without admission eligibility checks.

        A service restart must be able to explain and release a reservation
        that was valid when acquired even if its host has since gone stale,
        draining, or quarantined.  This path deliberately bypasses heartbeat,
        host-state, and observed-disk checks used by :meth:`try_reserve`; it
        still rejects duplicate IDs with different accounting and prevents
        the declared live-plus-held total from exceeding host capacity.  The
        native WorkItem reservation port remains the distributed authority.
        """

        with self._lock:
            host = self.require(host_id)
            records = self._reserved.setdefault(host_id, {})
            current = records.get(reservation_id)
            if current is not None:
                if current != requirement:
                    raise CapacityError(
                        f"reservation {reservation_id!r} accounting does not match restore"
                    )
                return True
            held = self._reserved_total(host_id) + requirement
            if not host.total.fits(host.live + held):
                raise CapacityError(
                    f"reservation {reservation_id!r} exceeds host capacity during restore"
                )
            records[reservation_id] = requirement
            return True

    def release(
        self,
        host_id: str,
        reservation_id: str,
        requirement: ResourceVector | None = None,
    ) -> bool:
        """Release an exact reservation, refusing mismatched accounting."""

        with self._lock:
            records = self._reserved.get(host_id, {})
            current = records.get(reservation_id)
            if current is None:
                return False
            if requirement is not None and current != requirement:
                raise CapacityError(
                    f"reservation {reservation_id!r} accounting does not match release"
                )
            del records[reservation_id]
            return True

    def views(self, *, now: datetime | None = None) -> tuple[CapacityView, ...]:
        with self._lock:
            return tuple(
                self.snapshot(host_id, now=now) for host_id in sorted(self._hosts)
            )

    def _reserved_total(self, host_id: str) -> ResourceVector:
        records = self._reserved.get(host_id, {})
        total = ResourceVector()
        for vector in records.values():
            total = total + vector
        return total


__all__ = [
    "CapacityError",
    "CapacityInventory",
    "CapacityView",
    "HostCapacity",
    "HostState",
    "ResourceVector",
]
