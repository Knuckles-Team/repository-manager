"""WorkItem-fenced durable resource reservations.

The scheduler owns these records, but it does not own job state.  Every
mutation is first authorized by an injected native WorkItem reservation port
which compares the current ``(work_item_id, attempt, fence)`` in its own
transaction/CAS.  The in-memory implementation below is a deterministic test
port; production graph-os adapters can implement the same protocol without
changing scheduler policy.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from repository_manager.capacity import CapacityView, HostState, ResourceVector
from repository_manager.development import (
    ReservationState,
    TargetPolicy,
)
from repository_manager.fairness import FairnessStatePort


class ReservationError(ValueError):
    """Malformed reservation or a persistence/authority refusal."""


class FenceDecision(StrEnum):
    ACCEPTED = "accepted"
    IDEMPOTENT = "idempotent"
    STALE = "stale"
    CONFLICT = "conflict"
    INPUT_CONFLICT = "input_conflict"
    CAPACITY = "capacity"
    POLICY = "policy"
    DRAINED = "drained"
    QUARANTINED = "quarantined"
    STALE_HOST = "stale_host"
    LABELS = "labels"
    ANTI_AFFINITY = "anti_affinity"
    DISK = "disk"
    CONCURRENCY = "concurrency"
    EXCLUSIVITY = "exclusivity"
    NOT_FOUND = "not_found"


@dataclass(frozen=True)
class WorkItemClaim:
    """Native claim plus optional expected immutable admission extension.

    Empty extension fields are permissive defaults for small simulations; a
    production graph adapter must source these values from the WorkItem rather
    than trusting caller-supplied reservation metadata.  ``terminal`` stops
    new admission while retaining the exact attempt/fence for one release or
    lifecycle reconciliation operation.
    """

    work_item_id: str
    attempt: int
    fence: str
    heartbeat_at: datetime
    expires_at: datetime
    owner_id: str = ""
    tenant_id: str = ""
    profile_name: str = ""
    requirement: ResourceVector | None = None
    repository_id: str = ""
    branch: str = ""
    concurrency_key: str = ""
    fairness_group: str = ""
    target: TargetPolicy | None = None
    host_id: str = ""
    input_fingerprint: str = ""
    terminal: bool = False

    def matches_reservation(self, reservation: ReservationRecord) -> bool:
        """Validate caller input against WorkItem-owned admission extension."""

        return (
            (not self.owner_id or self.owner_id == reservation.owner_id)
            and (not self.tenant_id or self.tenant_id == reservation.tenant_id)
            and (not self.profile_name or self.profile_name == reservation.profile_name)
            and (
                self.requirement is None or self.requirement == reservation.requirement
            )
            and (
                not self.repository_id
                or self.repository_id == reservation.repository_id
            )
            and (not self.branch or self.branch == reservation.branch)
            and (
                not self.concurrency_key
                or self.concurrency_key == reservation.concurrency_key
            )
            and (
                not self.fairness_group
                or self.fairness_group == reservation.fairness_group
            )
            and (not self.host_id or self.host_id == reservation.host_id)
            and (self.target is None or self.target == reservation.selected_target)
            and (
                not self.input_fingerprint
                or self.input_fingerprint == reservation.input_fingerprint
            )
        )

    def is_live(self, now: datetime | None = None) -> bool:
        return (now or datetime.now(UTC)).astimezone(UTC) < self.expires_at


@dataclass(frozen=True)
class ReservationRecord:
    """Immutable view of one scheduler-owned reservation revision."""

    reservation_id: str
    work_item_id: str
    attempt: int
    fence: str
    host_id: str
    profile_name: str
    requirement: ResourceVector
    capacity_snapshot: dict[str, Any]
    selected_target: TargetPolicy = TargetPolicy()
    concurrency_key: str = ""
    concurrency_limit: int | None = None
    repository_exclusive: bool = False
    branch_exclusive: bool = False
    required_labels: tuple[str, ...] = ()
    disk_low_watermark_mib: int | None = None
    disk_high_watermark_mib: int | None = None
    # Native disk hysteresis is keyed by host plus this profile/policy
    # revision.  It is immutable so a retry cannot clear another policy's
    # blocked state by omitting watermarks.
    disk_policy_key: str = "default"
    repository_id: str = ""
    branch: str = ""
    owner_id: str = ""
    tenant_id: str = ""
    fairness_group: str = "default"
    fairness_cost: int = 1
    anti_affinity: tuple[str, ...] = ()
    reserved_at: datetime = datetime.min.replace(tzinfo=UTC)
    expires_at: datetime = datetime.min.replace(tzinfo=UTC)
    state: ReservationState = ReservationState.RESERVED
    reason: str = ""
    released_at: datetime | None = None
    revision: int = 1
    # Hash of the immutable WorkItem admission input.  Capacity observations
    # and reservation timestamps are intentionally excluded so a retry from a
    # replica with a missing local projection can deduplicate against the
    # native record without pretending a fresh admission occurred.
    input_fingerprint: str = ""

    def __post_init__(self) -> None:
        if not self.reservation_id.strip():
            raise ReservationError("reservation_id must be non-blank")
        if not self.work_item_id.strip() or not self.fence.strip():
            raise ReservationError("work_item_id and fence must be non-blank")
        if self.attempt < 1:
            raise ReservationError("attempt must be positive")
        if self.expires_at <= self.reserved_at:
            raise ReservationError("reservation expiry must be after reservation time")
        if self.revision < 1:
            raise ReservationError("reservation revision must be positive")
        if self.fairness_cost < 1:
            raise ReservationError("fairness_cost must be positive")
        if self.input_fingerprint and not self.input_fingerprint.startswith("v1:"):
            raise ReservationError("input_fingerprint must use the v1 namespace")
        if not self.disk_policy_key.strip():
            raise ReservationError("disk_policy_key must be non-blank")
        if self.concurrency_limit is not None and self.concurrency_limit < 1:
            raise ReservationError("concurrency_limit must be positive")
        if (
            self.disk_low_watermark_mib is not None
            and self.disk_high_watermark_mib is not None
            and self.disk_low_watermark_mib > self.disk_high_watermark_mib
        ):
            raise ReservationError(
                "disk_low_watermark_mib must not exceed disk_high_watermark_mib"
            )
        object.__setattr__(
            self, "anti_affinity", tuple(sorted(set(self.anti_affinity)))
        )
        object.__setattr__(
            self, "required_labels", tuple(sorted(set(self.required_labels)))
        )

    @property
    def active(self) -> bool:
        return self.state is ReservationState.RESERVED

    def expired(self, now: datetime | None = None) -> bool:
        return (
            self.active
            and (now or datetime.now(UTC)).astimezone(UTC) >= self.expires_at
        )

    def same_immutable_input(self, other: ReservationRecord) -> bool:
        """Compare admission input while ignoring mutable/presentation fields.

        ``capacity_snapshot`` and absolute reservation timestamps are
        projections or mutable bookkeeping; the requested TTL remains part of
        the comparison through the timestamp duration.  Lifecycle state,
        reason, release time, and revision are also excluded.  Everything that
        can change what WorkItem was admitted, where, or with which policy
        remains part of this comparison.  The fingerprint is an additional
        canonical request-level guard.
        """

        if self.input_fingerprint or other.input_fingerprint:
            if self.input_fingerprint != other.input_fingerprint:
                return False
        return (
            self.reservation_id,
            self.work_item_id,
            self.attempt,
            self.fence,
            self.host_id,
            self.profile_name,
            self.requirement,
            self.selected_target.model_dump(mode="json"),
            self.concurrency_key,
            self.concurrency_limit,
            self.repository_exclusive,
            self.branch_exclusive,
            self.required_labels,
            self.disk_low_watermark_mib,
            self.disk_high_watermark_mib,
            self.disk_policy_key,
            self.repository_id,
            self.branch,
            self.owner_id,
            self.tenant_id,
            self.fairness_group,
            self.fairness_cost,
            self.anti_affinity,
            self.expires_at - self.reserved_at,
        ) == (
            other.reservation_id,
            other.work_item_id,
            other.attempt,
            other.fence,
            other.host_id,
            other.profile_name,
            other.requirement,
            other.selected_target.model_dump(mode="json"),
            other.concurrency_key,
            other.concurrency_limit,
            other.repository_exclusive,
            other.branch_exclusive,
            other.required_labels,
            other.disk_low_watermark_mib,
            other.disk_high_watermark_mib,
            other.disk_policy_key,
            other.repository_id,
            other.branch,
            other.owner_id,
            other.tenant_id,
            other.fairness_group,
            other.fairness_cost,
            other.anti_affinity,
            other.expires_at - other.reserved_at,
        )

    def with_state(
        self,
        state: ReservationState,
        *,
        reason: str = "",
        at: datetime | None = None,
    ) -> ReservationRecord:
        released_at = (
            (at or datetime.now(UTC)).astimezone(UTC)
            if state in {ReservationState.RELEASED, ReservationState.EXPIRED}
            else None
        )
        return replace(
            self,
            state=state,
            reason=reason,
            released_at=released_at,
            revision=self.revision + 1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "reservation_id": self.reservation_id,
            "work_item_id": self.work_item_id,
            "attempt": self.attempt,
            "fence": self.fence,
            "host_id": self.host_id,
            "profile_name": self.profile_name,
            "requirement": self.requirement.as_dict(),
            "capacity_snapshot": self.capacity_snapshot,
            "selected_target": self.selected_target.model_dump(mode="json"),
            "concurrency_key": self.concurrency_key,
            "concurrency_limit": self.concurrency_limit,
            "repository_exclusive": self.repository_exclusive,
            "branch_exclusive": self.branch_exclusive,
            "required_labels": list(self.required_labels),
            "disk_low_watermark_mib": self.disk_low_watermark_mib,
            "disk_high_watermark_mib": self.disk_high_watermark_mib,
            "disk_policy_key": self.disk_policy_key,
            "repository_id": self.repository_id,
            "branch": self.branch,
            "owner_id": self.owner_id,
            "tenant_id": self.tenant_id,
            "fairness_group": self.fairness_group,
            "fairness_cost": self.fairness_cost,
            "anti_affinity": list(self.anti_affinity),
            "reserved_at": self.reserved_at.astimezone(UTC).isoformat(),
            "expires_at": self.expires_at.astimezone(UTC).isoformat(),
            "state": self.state.value,
            "reason": self.reason,
            "released_at": self.released_at.astimezone(UTC).isoformat()
            if self.released_at
            else None,
            "revision": self.revision,
            "input_fingerprint": self.input_fingerprint,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ReservationRecord:
        requirement = value.get("requirement") or {}
        selected = value.get("selected_target") or {"kind": "local"}
        return cls(
            reservation_id=str(value["reservation_id"]),
            work_item_id=str(value["work_item_id"]),
            attempt=int(value["attempt"]),
            fence=str(value["fence"]),
            host_id=str(value["host_id"]),
            profile_name=str(value["profile_name"]),
            requirement=ResourceVector(
                cpu_weight=int(requirement.get("cpu_weight", 0)),
                memory_mib=int(requirement.get("memory_mib", 0)),
                disk_mib=int(requirement.get("disk_mib", 0)),
                process_slots=int(requirement.get("process_slots", 0)),
            ),
            capacity_snapshot=dict(value.get("capacity_snapshot") or {}),
            selected_target=TargetPolicy.model_validate(selected),
            concurrency_key=str(value.get("concurrency_key", "")),
            concurrency_limit=(
                int(value["concurrency_limit"])
                if value.get("concurrency_limit") is not None
                else None
            ),
            repository_exclusive=bool(value.get("repository_exclusive", False)),
            branch_exclusive=bool(value.get("branch_exclusive", False)),
            required_labels=tuple(
                str(item) for item in value.get("required_labels", ())
            ),
            disk_low_watermark_mib=(
                int(value["disk_low_watermark_mib"])
                if value.get("disk_low_watermark_mib") is not None
                else None
            ),
            disk_high_watermark_mib=(
                int(value["disk_high_watermark_mib"])
                if value.get("disk_high_watermark_mib") is not None
                else None
            ),
            disk_policy_key=str(value.get("disk_policy_key", "default")),
            repository_id=str(value.get("repository_id", "")),
            branch=str(value.get("branch", "")),
            owner_id=str(value.get("owner_id", "")),
            tenant_id=str(value.get("tenant_id", "")),
            fairness_group=str(value.get("fairness_group", "default")),
            fairness_cost=int(value.get("fairness_cost", 1)),
            anti_affinity=tuple(str(item) for item in value.get("anti_affinity", ())),
            reserved_at=datetime.fromisoformat(str(value["reserved_at"])).astimezone(
                UTC
            ),
            expires_at=datetime.fromisoformat(str(value["expires_at"])).astimezone(UTC),
            state=ReservationState(str(value.get("state", ReservationState.RESERVED))),
            reason=str(value.get("reason", "")),
            released_at=(
                datetime.fromisoformat(str(value["released_at"])).astimezone(UTC)
                if value.get("released_at")
                else None
            ),
            revision=int(value.get("revision", 1)),
            input_fingerprint=str(value.get("input_fingerprint", "")),
        )


class WorkItemReservationPort(Protocol):
    """Native atomic WorkItem/CAS seam required by scheduler mutations.

    Local reservation rows are mirrors.  An ADMITTED retry or execution
    handoff must use :meth:`query_reservation` (or an equivalent native
    transaction) to revalidate the current fence, immutable WorkItem input,
    and lifecycle revision before it can proceed.
    """

    def is_current(self, work_item_id: str, attempt: int, fence: str) -> bool:
        """Return whether this attempt/fence currently owns the WorkItem lease."""

    def query_reservation(
        self,
        *,
        reservation_id: str,
        work_item_id: str,
        attempt: int,
        fence: str,
        expected: ReservationRecord | None = None,
        for_lifecycle: bool = False,
        controller: bool = False,
    ) -> ReservationRecord | FenceDecision | None:
        """Exact native lookup/revalidation, including retained tombstones.

        ``NOT_FOUND`` and lifecycle records are not admission.  With
        ``for_lifecycle=True``, a terminal exact attempt may return a retained
        tombstone for mirror repair; a controller may reconcile an expired
        tombstone after a newer attempt supersedes it, but that result never
        authorizes execution.
        A production implementation must compare the current WorkItem
        extension and return the native immutable reservation revision in the
        same authority that owns the fence.
        """

    def atomic_reserve(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation: ReservationRecord,
        expected_capacity: CapacityView | None = None,
    ) -> FenceDecision | bool:
        """Create the link and consume capacity in one native transaction.

        An exact retry of an already-linked reservation must return
        ``FenceDecision.IDEMPOTENT`` after comparing the full immutable
        reservation input.  A different input for the same WorkItem attempt
        must return ``CONFLICT`` and must not release or mutate the existing
        link.  Native implementations must enforce one active reservation per
        ``(work_item_id, attempt)`` even when a caller supplies an arbitrary
        reservation ID.

        If native fairness state is enabled, the transaction must add
        ``reservation.fairness_cost`` to ``reservation.fairness_group`` in the
        same WorkItem CAS.  An idempotent retry must not add debt again.

        ``expected_capacity`` is an advisory snapshot for optimistic
        concurrency diagnostics.  A production implementation MUST re-read
        authoritative host state (capacity, heartbeat, drain/quarantine,
        labels, anti-affinity, disk watermarks), concurrency, and
        exclusivity records inside the same CAS/transaction that verifies the
        WorkItem fence; it must not trust this local snapshot.  Refusals should
        return a specific ``FenceDecision`` policy code rather than collapsing
        a capacity race into a fence error.
        """

    def atomic_release(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
        reservation: ReservationRecord | None = None,
    ) -> FenceDecision | bool:
        """Release capacity and retain a native lifecycle tombstone/revision.

        The native mutation precedes local mirror update.  Repeating the call
        after a local failure must return ``IDEMPOTENT`` for the same immutable
        input rather than decrementing capacity or fairness a second time.  An
        exact current attempt may release after its WorkItem reaches terminal
        state; a rotated older attempt remains stale.
        """

    def atomic_reclaim(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
        reservation: ReservationRecord | None = None,
        now: datetime | None = None,
    ) -> FenceDecision | bool:
        """Reclaim under native lease rules and retain a tombstone/revision.

        Reclaim is idempotent across local projection recovery and must be
        conditional on current native controller/attempt semantics.
        """

    def atomic_reclaim_expired(
        self, *, reservation: ReservationRecord, now: datetime
    ) -> FenceDecision | bool:
        """Reclaim an expired/superseded link from current native state.

        A reconciler, rather than the stale worker, calls this operation.  The
        native transaction must prove that the reservation TTL/heartbeat is
        expired or that a newer WorkItem attempt superseded it before releasing
        capacity.  The released/expired row remains queryable as a native
        lifecycle tombstone so local reconciliation can safely retry.
        """


class InMemoryWorkItemReservationPort:
    """Native-transaction test fixture with restart-safe claim semantics.

    It intentionally requires an explicit :meth:`claim` before a reservation
    can be attached.  Tests can rotate the fence or expire the claim to prove
    stale workers cannot release or reclaim a newer attempt.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._lock = RLock()
        self._clock = clock or (lambda: datetime.now(UTC))
        self._claims: dict[str, WorkItemClaim] = {}
        self._links: dict[str, tuple[str, int, str]] = {}
        self._attempt_links: dict[tuple[str, int], str] = {}
        self._records: dict[str, ReservationRecord] = {}
        self._capacity: dict[str, ResourceVector] = {}
        self._native_hosts: dict[str, CapacityView] = {}
        self._reserved_capacity: dict[str, dict[str, ResourceVector]] = {}
        self._disk_blocked: dict[tuple[str, str], bool] = {}
        self._disk_watermarks: dict[tuple[str, str], tuple[int | None, int | None]] = {}
        self._fairness_state: FairnessStatePort | None = None

    def _now(self) -> datetime:
        """Return the fixture's authoritative UTC time for liveness checks."""

        return self._clock().astimezone(UTC)

    def register_capacity(self, host_id: str, total: ResourceVector) -> bool:
        """Configure a complete default authoritative host for simulations."""

        with self._lock:
            now = self._now()
            return self.register_capacity_view(
                CapacityView(
                    host_id=host_id,
                    state=HostState.ACTIVE,
                    labels=(),
                    target_kind="local",
                    total=total,
                    live=ResourceVector(),
                    reserved=ResourceVector(),
                    available=total,
                    heartbeat_at=now,
                    heartbeat_fresh=True,
                    observed_disk_free_mib=None,
                    heartbeat_ttl_seconds=120,
                )
            )

    def register_capacity_view(self, view: CapacityView) -> bool:
        """Register authoritative policy/telemetry, not caller projections.

        Updates are monotonic by host revision.  Held native reservations are
        retained and are revalidated against a smaller refreshed total before
        the refresh is accepted.
        """

        with self._lock:
            current = self._native_hosts.get(view.host_id)
            if current is not None:
                if view.version <= current.version:
                    return False
                held = self._reserved_capacity.get(view.host_id, {})
                held_total = ResourceVector()
                for vector in held.values():
                    held_total = held_total + vector
                if not view.total.fits(view.live + held_total):
                    return False
            self._native_hosts[view.host_id] = view
            self._capacity[view.host_id] = view.total
            self._reserved_capacity.setdefault(view.host_id, {})
            return True

    def register_fairness_state(self, state: FairnessStatePort | None) -> None:
        """Bind the native fairness debt used by atomic reservations."""

        with self._lock:
            if self._fairness_state is None:
                self._fairness_state = state

    def _authoritative_view(self, host_id: str) -> CapacityView | None:
        base = self._native_hosts.get(host_id)
        if base is None:
            return None
        reserved = ResourceVector()
        for vector in self._reserved_capacity.get(host_id, {}).values():
            reserved = reserved + vector
        declared_available = base.total - base.live
        if base.observed_disk_free_mib is None:
            disk_free = declared_available.disk_mib
        else:
            disk_free = min(declared_available.disk_mib, base.observed_disk_free_mib)
        available = ResourceVector(
            max(0, declared_available.cpu_weight - reserved.cpu_weight),
            max(0, declared_available.memory_mib - reserved.memory_mib),
            max(0, disk_free - reserved.disk_mib),
            max(0, declared_available.process_slots - reserved.process_slots),
        )
        return CapacityView(
            host_id=base.host_id,
            state=base.state,
            labels=base.labels,
            target_kind=base.target_kind,
            total=base.total,
            live=base.live,
            reserved=reserved,
            available=available,
            heartbeat_at=base.heartbeat_at,
            heartbeat_fresh=self._now() - base.heartbeat_at
            <= timedelta(seconds=base.heartbeat_ttl_seconds),
            observed_disk_free_mib=base.observed_disk_free_mib,
            heartbeat_ttl_seconds=base.heartbeat_ttl_seconds,
            version=base.version,
        )

    def claim(
        self,
        work_item_id: str,
        *,
        attempt: int = 1,
        fence: str | None = None,
        ttl_seconds: int = 120,
        now: datetime | None = None,
        owner_id: str = "",
        tenant_id: str = "",
        profile_name: str = "",
        requirement: ResourceVector | None = None,
        repository_id: str = "",
        branch: str = "",
        concurrency_key: str = "",
        fairness_group: str = "",
        target: TargetPolicy | None = None,
        host_id: str = "",
        input_fingerprint: str = "",
    ) -> WorkItemClaim:
        if attempt < 1 or ttl_seconds < 1:
            raise ReservationError("attempt and ttl_seconds must be positive")
        at = (now or self._now()).astimezone(UTC)
        claim = WorkItemClaim(
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence or f"fence:{uuid.uuid4().hex}",
            heartbeat_at=at,
            expires_at=at + timedelta(seconds=ttl_seconds),
            owner_id=owner_id,
            tenant_id=tenant_id,
            profile_name=profile_name,
            requirement=requirement,
            repository_id=repository_id,
            branch=branch,
            concurrency_key=concurrency_key,
            fairness_group=fairness_group,
            target=target,
            host_id=host_id,
            input_fingerprint=input_fingerprint,
        )
        with self._lock:
            self._claims[work_item_id] = claim
        return claim

    def rotate(
        self,
        work_item_id: str,
        *,
        attempt: int | None = None,
        now: datetime | None = None,
    ) -> WorkItemClaim:
        with self._lock:
            current = self._claims.get(work_item_id)
            if current is None:
                raise ReservationError(f"WorkItem {work_item_id!r} is not claimed")
            return self.claim(
                work_item_id,
                attempt=(attempt if attempt is not None else current.attempt + 1),
                now=now,
                owner_id=current.owner_id,
                tenant_id=current.tenant_id,
                profile_name=current.profile_name,
                requirement=current.requirement,
                repository_id=current.repository_id,
                branch=current.branch,
                concurrency_key=current.concurrency_key,
                fairness_group=current.fairness_group,
                target=current.target,
                host_id=current.host_id,
                input_fingerprint=current.input_fingerprint,
            )

    def expire(self, work_item_id: str, *, now: datetime | None = None) -> None:
        with self._lock:
            current = self._claims.get(work_item_id)
            if current is None:
                return
            at = (now or self._now()).astimezone(UTC)
            self._claims[work_item_id] = replace(current, expires_at=at)

    def complete(self, work_item_id: str) -> None:
        """Mark the current WorkItem attempt terminal without changing fence.

        A terminal commit is still the same native attempt for one exact
        release/reconciliation operation, but it is no longer eligible for new
        execution admission.
        """

        with self._lock:
            current = self._claims.get(work_item_id)
            if current is not None:
                self._claims[work_item_id] = replace(current, terminal=True)

    def is_current(self, work_item_id: str, attempt: int, fence: str) -> bool:
        with self._lock:
            current = self._claims.get(work_item_id)
            return bool(
                current
                and current.attempt == attempt
                and current.fence == fence
                and not current.terminal
                and current.is_live(self._now())
            )

    def query_reservation(
        self,
        *,
        reservation_id: str,
        work_item_id: str,
        attempt: int,
        fence: str,
        expected: ReservationRecord | None = None,
        for_lifecycle: bool = False,
        controller: bool = False,
    ) -> ReservationRecord | FenceDecision | None:
        """Re-read the native link and retained lifecycle revision.

        This fixture intentionally performs all checks under one lock to model
        the graph transaction required by RMDD-27.  A local row or capacity
        mirror is never consulted as authorization.
        """

        with self._lock:
            record = self._records.get(reservation_id)
            if record is None:
                return FenceDecision.NOT_FOUND
            current_claim = self._claims.get(work_item_id)
            if current_claim is not None and (
                current_claim.attempt != attempt or current_claim.fence != fence
            ):
                if not (
                    for_lifecycle
                    and controller
                    and not record.active
                    and current_claim.attempt > record.attempt
                ):
                    return FenceDecision.STALE
            if not for_lifecycle:
                if not self.is_current(work_item_id, attempt, fence):
                    return FenceDecision.STALE
            elif current_claim is None and record.active:
                # A lifecycle query may repair a retained tombstone after the
                # claim row is terminal/compacted, but it cannot authorize an
                # active record with no current WorkItem state.
                return FenceDecision.STALE
            if (
                record.work_item_id,
                record.attempt,
                record.fence,
            ) != (work_item_id, attempt, fence):
                return FenceDecision.CONFLICT
            if current_claim is not None and not current_claim.matches_reservation(
                record
            ):
                return FenceDecision.INPUT_CONFLICT
            if expected is not None and not record.same_immutable_input(expected):
                return FenceDecision.INPUT_CONFLICT
            return record

    def atomic_reserve(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation: ReservationRecord,
        expected_capacity: CapacityView | None = None,
    ) -> FenceDecision:
        with self._lock:
            if not self.is_current(work_item_id, attempt, fence):
                return FenceDecision.STALE
            if (
                reservation.work_item_id,
                reservation.attempt,
                reservation.fence,
            ) != (work_item_id, attempt, fence):
                return FenceDecision.CONFLICT
            claim = self._claims.get(work_item_id)
            if claim is None or not claim.matches_reservation(reservation):
                return FenceDecision.INPUT_CONFLICT
            attempt_key = (work_item_id, attempt)
            attempt_reservation_id = self._attempt_links.get(attempt_key)
            if attempt_reservation_id is not None:
                existing = self._records.get(attempt_reservation_id)
                if (
                    attempt_reservation_id == reservation.reservation_id
                    and existing is not None
                    and existing.same_immutable_input(reservation)
                ):
                    return FenceDecision.IDEMPOTENT
                return FenceDecision.CONFLICT
            historical = self._records.get(reservation.reservation_id)
            if historical is not None:
                # Retained release/reclaim tombstones make a same-attempt
                # lifecycle terminal.  A caller cannot resurrect it with a
                # new local projection or a different reservation ID.
                return FenceDecision.CONFLICT
            linked = self._links.get(reservation.reservation_id)
            if linked is not None:
                existing = self._records.get(reservation.reservation_id)
                if (
                    linked == (work_item_id, attempt, fence)
                    and existing is not None
                    and existing.same_immutable_input(reservation)
                ):
                    return FenceDecision.IDEMPOTENT
                return FenceDecision.CONFLICT
            # The caller's expected_capacity is an optimistic diagnostic only.
            # This fixture re-reads the complete registered native view and
            # recomputes availability with native-held reservations, modeling
            # the graph transaction required in production.
            authoritative = self._authoritative_view(reservation.host_id)
            if authoritative is None:
                return FenceDecision.POLICY
            if authoritative.state in {HostState.DRAINING, HostState.DRAINED}:
                return FenceDecision.DRAINED
            if authoritative.state in {HostState.QUARANTINED, HostState.OFFLINE}:
                return FenceDecision.QUARANTINED
            if not authoritative.heartbeat_fresh:
                return FenceDecision.STALE_HOST
            if not set(reservation.required_labels).issubset(authoritative.labels):
                return FenceDecision.LABELS
            active = tuple(record for record in self._records.values() if record.active)
            if reservation.concurrency_limit is not None:
                count = sum(
                    record.concurrency_key == reservation.concurrency_key
                    for record in active
                )
                if count >= reservation.concurrency_limit:
                    return FenceDecision.CONCURRENCY
            for record in active:
                if record.host_id == reservation.host_id and set(
                    record.anti_affinity
                ).intersection(reservation.anti_affinity):
                    return FenceDecision.ANTI_AFFINITY
            disk_key = (reservation.host_id, reservation.disk_policy_key)
            configured = self._disk_watermarks.get(disk_key)
            if configured is None or (
                configured == (None, None)
                and (
                    reservation.disk_low_watermark_mib is not None
                    or reservation.disk_high_watermark_mib is not None
                )
            ):
                configured_high = reservation.disk_high_watermark_mib
                configured_low = reservation.disk_low_watermark_mib
                if configured_high is not None and configured_low is None:
                    configured_low = configured_high
                configured = (
                    configured_low,
                    configured_high,
                )
                self._disk_watermarks[disk_key] = configured
            low, high = configured
            blocked = self._disk_blocked.get(disk_key, False)
            if high is not None or blocked:
                predicted_used = (
                    authoritative.total.disk_mib
                    - authoritative.available.disk_mib
                    + reservation.requirement.disk_mib
                )
                if blocked:
                    if low is None or predicted_used > low:
                        return FenceDecision.DISK
                    self._disk_blocked[disk_key] = False
                elif high is not None and predicted_used >= high:
                    self._disk_blocked[disk_key] = True
                    return FenceDecision.DISK
            if not authoritative.available.fits(reservation.requirement):
                return FenceDecision.CAPACITY
            self._reserved_capacity[reservation.host_id][reservation.reservation_id] = (
                reservation.requirement
            )
            for record in active:
                # Anti-affinity is deliberately host-local.  Repository and
                # branch exclusivity are global keys across every host.
                if record.host_id == reservation.host_id and set(
                    record.anti_affinity
                ).intersection(reservation.anti_affinity):
                    self._reserved_capacity[reservation.host_id].pop(
                        reservation.reservation_id, None
                    )
                    return FenceDecision.ANTI_AFFINITY
                if (
                    (reservation.repository_exclusive or record.repository_exclusive)
                    and reservation.repository_id
                    and record.repository_id == reservation.repository_id
                ):
                    self._reserved_capacity[reservation.host_id].pop(
                        reservation.reservation_id, None
                    )
                    return FenceDecision.EXCLUSIVITY
                if (
                    (reservation.branch_exclusive or record.branch_exclusive)
                    and reservation.repository_id
                    and reservation.branch
                    and (record.repository_id, record.branch)
                    == (reservation.repository_id, reservation.branch)
                ):
                    self._reserved_capacity[reservation.host_id].pop(
                        reservation.reservation_id, None
                    )
                    return FenceDecision.EXCLUSIVITY
            if self._fairness_state is not None:
                try:
                    self._fairness_state.record(
                        reservation.fairness_group, reservation.fairness_cost
                    )
                except Exception:
                    self._reserved_capacity[reservation.host_id].pop(
                        reservation.reservation_id, None
                    )
                    raise
            self._records[reservation.reservation_id] = reservation
            self._links[reservation.reservation_id] = (work_item_id, attempt, fence)
            self._attempt_links[attempt_key] = reservation.reservation_id
            return FenceDecision.ACCEPTED

    def atomic_release(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
        reservation: ReservationRecord | None = None,
    ) -> FenceDecision:
        with self._lock:
            current_claim = self._claims.get(work_item_id)
            if current_claim is None or (
                current_claim.attempt != attempt or current_claim.fence != fence
            ):
                return FenceDecision.STALE
            release_at = self._now()
            if not current_claim.terminal and not current_claim.is_live(release_at):
                return FenceDecision.STALE
            current = self._records.get(reservation_id)
            if current is None:
                return FenceDecision.NOT_FOUND
            if reservation is not None and not current.same_immutable_input(
                reservation
            ):
                return FenceDecision.INPUT_CONFLICT
            link = self._links.get(reservation_id)
            expected_link = (work_item_id, attempt, fence)
            if link is None:
                if (
                    link is None
                    and current.state is ReservationState.RELEASED
                    and (
                        current.work_item_id,
                        current.attempt,
                        current.fence,
                    )
                    == expected_link
                ):
                    # Native lifecycle tombstone/revision makes retries safe
                    # after a local projection update failed.
                    return FenceDecision.IDEMPOTENT
                return FenceDecision.CONFLICT
            if link != expected_link:
                return FenceDecision.CONFLICT
            self._links.pop(reservation_id, None)
            self._records[reservation_id] = current.with_state(
                ReservationState.RELEASED,
                reason="native release committed",
                at=release_at,
            )
            held = current
            self._attempt_links[(work_item_id, attempt)] = reservation_id
            self._reserved_capacity.get(held.host_id, {}).pop(reservation_id, None)
            return FenceDecision.ACCEPTED

    def atomic_reclaim_expired(
        self, *, reservation: ReservationRecord, now: datetime
    ) -> FenceDecision:
        """Reclaim a superseded or expired record from current authority."""

        with self._lock:
            link = self._links.get(reservation.reservation_id)
            current = self._claims.get(reservation.work_item_id)
            current_record = self._records.get(reservation.reservation_id)
            if (
                link is None
                and current_record is not None
                and current_record.state is ReservationState.EXPIRED
                and current_record.same_immutable_input(reservation)
            ):
                return FenceDecision.IDEMPOTENT
            if link is None or current is None:
                return FenceDecision.CONFLICT
            if link != (
                reservation.work_item_id,
                reservation.attempt,
                reservation.fence,
            ):
                return FenceDecision.CONFLICT
            # Validate the complete immutable input before touching the link,
            # record, attempt index, or capacity.  A refused reclaim must be
            # observationally atomic even when its caller has a stale local
            # projection with changed repository/policy metadata.
            if current_record is None:
                return FenceDecision.CONFLICT
            if not current_record.same_immutable_input(reservation):
                return FenceDecision.INPUT_CONFLICT
            check_at = now.astimezone(UTC)
            same_attempt_expired = (
                current.attempt == reservation.attempt
                and current.fence == reservation.fence
                and current.expires_at <= check_at
            )
            superseded = current.attempt > reservation.attempt
            if not same_attempt_expired and not superseded:
                return FenceDecision.STALE
            next_record = current_record.with_state(
                ReservationState.EXPIRED,
                reason="native expired/superseded reclaim committed",
                at=now,
            )
            self._links.pop(reservation.reservation_id, None)
            self._records[reservation.reservation_id] = next_record
            self._attempt_links[(reservation.work_item_id, reservation.attempt)] = (
                reservation.reservation_id
            )
            self._reserved_capacity.get(current_record.host_id, {}).pop(
                reservation.reservation_id, None
            )
            return FenceDecision.ACCEPTED

    def atomic_reclaim(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
        reservation: ReservationRecord | None = None,
        now: datetime | None = None,
    ) -> FenceDecision:
        # A native implementation may allow reclaim after lease expiry using a
        # conditional WorkItem transaction.  The old fence is accepted here
        # only when it is still the linked claim AND that exact claim has
        # expired; a newer attempt/fence cannot be reclaimed by a stale worker.
        with self._lock:
            current = self._claims.get(work_item_id)
            current_record = self._records.get(reservation_id)
            if (
                self._links.get(reservation_id) is None
                and current_record is not None
                and current_record.state is ReservationState.EXPIRED
                and (
                    reservation is None
                    or current_record.same_immutable_input(reservation)
                )
            ):
                return FenceDecision.IDEMPOTENT
            if current is None or self._links.get(reservation_id) != (
                work_item_id,
                attempt,
                fence,
            ):
                return FenceDecision.CONFLICT
            check_at = (now or self._now()).astimezone(UTC)
            if current.attempt != attempt or current.fence != fence:
                return FenceDecision.STALE
            if current.expires_at > check_at:
                return FenceDecision.STALE
            if current_record is None or (
                reservation is not None
                and not current_record.same_immutable_input(reservation)
            ):
                return FenceDecision.INPUT_CONFLICT
            self._links.pop(reservation_id, None)
            self._records[reservation_id] = current_record.with_state(
                ReservationState.EXPIRED,
                reason="native expired reclaim committed",
                at=check_at,
            )
            self._attempt_links[(work_item_id, attempt)] = reservation_id
            held = current_record
            if held is not None:
                self._reserved_capacity.get(held.host_id, {}).pop(reservation_id, None)
            return FenceDecision.ACCEPTED

    def link(self, reservation_id: str) -> tuple[str, int, str] | None:
        with self._lock:
            return self._links.get(reservation_id)

    def reserved_for(self, host_id: str) -> ResourceVector:
        """Return authoritative held capacity for deterministic simulations."""

        with self._lock:
            total = ResourceVector()
            for vector in self._reserved_capacity.get(host_id, {}).values():
                total = total + vector
            return total


class ReservationStore(Protocol):
    """Persistence port for scheduler-owned reservation records."""

    def put(self, record: ReservationRecord) -> None:
        """Persist a new or idempotently repeated record."""

    def get(self, reservation_id: str) -> ReservationRecord | None:
        """Load the latest revision for a reservation."""

    def update(self, record: ReservationRecord) -> None:
        """Persist a later state revision."""

    def all(self) -> tuple[ReservationRecord, ...]:
        """Return all latest reservation records."""


class InMemoryReservationStore:
    """Thread-safe durable-process fixture used by deterministic simulations."""

    def __init__(self, records: Iterable[ReservationRecord] = ()) -> None:
        self._lock = RLock()
        self._records: dict[str, ReservationRecord] = {}
        for record in records:
            self.put(record)

    def put(self, record: ReservationRecord) -> None:
        with self._lock:
            current = self._records.get(record.reservation_id)
            if current is not None and current != record:
                raise ReservationError(
                    f"reservation {record.reservation_id!r} already exists"
                )
            self._records[record.reservation_id] = record

    def get(self, reservation_id: str) -> ReservationRecord | None:
        with self._lock:
            return self._records.get(reservation_id)

    def update(self, record: ReservationRecord) -> None:
        with self._lock:
            current = self._records.get(record.reservation_id)
            if current is None:
                raise ReservationError(
                    f"reservation {record.reservation_id!r} is missing"
                )
            if record.revision <= current.revision:
                raise ReservationError("reservation revisions must increase")
            self._records[record.reservation_id] = record

    def all(self) -> tuple[ReservationRecord, ...]:
        with self._lock:
            return tuple(self._records.values())


class JsonReservationStore(InMemoryReservationStore):
    """Small restart-safe JSON store for local development and simulations.

    The scheduler remains responsible for WorkItem-native fencing; this file
    only persists the scheduler's linked reservation view.  Writes use a lock
    file and atomic replacement, and every latest revision is loaded on object
    construction so service recreation does not lose held capacity.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file_lock = self.path.with_suffix(self.path.suffix + ".lock")
        super().__init__(self._load())

    def put(self, record: ReservationRecord) -> None:
        super().put(record)
        self._flush()

    def update(self, record: ReservationRecord) -> None:
        super().update(record)
        self._flush()

    def _load(self) -> tuple[ReservationRecord, ...]:
        if not self.path.exists():
            return ()
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ReservationError(
                f"cannot load reservation store {self.path}: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise ReservationError("reservation store root must be a list")
        return tuple(ReservationRecord.from_dict(dict(item)) for item in data)

    def _flush(self) -> None:
        fd = os.open(self._file_lock, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            payload = (
                json.dumps(
                    [record.to_dict() for record in self.all()],
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=self.path.parent, delete=False
            ) as temp:
                temp.write(payload)
                temp.flush()
                os.fsync(temp.fileno())
                temp_path = Path(temp.name)
            temp_path.replace(self.path)
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


__all__ = [
    "FenceDecision",
    "InMemoryReservationStore",
    "InMemoryWorkItemReservationPort",
    "JsonReservationStore",
    "ReservationError",
    "ReservationRecord",
    "ReservationStore",
    "WorkItemClaim",
    "WorkItemReservationPort",
]
