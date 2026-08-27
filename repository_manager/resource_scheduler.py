"""Unified weighted resource admission for Repository Manager.

``ResourceScheduler`` is intentionally an admission service, not a worker or a
job queue.  A WorkItem is created and leased by graph-os; this module accepts a
leased request, chooses a live host, and attaches one durable reservation to
that WorkItem through a native fenced port.  Consumers later release the
reservation after their WorkItem transition.  No subprocess, Git operation,
SSH connection, or local filesystem lease is acquired here.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any, cast

from repository_manager.capacity import (
    CapacityInventory,
    CapacityView,
    HostState,
    ResourceVector,
)
from repository_manager.development import (
    ReservationState,
    ResourceRequest,
    TargetKind,
    TargetPolicy,
)
from repository_manager.disk_policy import (
    DiskDecision,
    DiskDecisionCode,
    DiskPolicy,
    DiskWatermarks,
)
from repository_manager.fairness import FairnessSelector, QueueCandidate
from repository_manager.reservations import (
    FenceDecision,
    InMemoryReservationStore,
    ReservationRecord,
    ReservationStore,
    WorkItemReservationPort,
)
from repository_manager.resource_profiles import (
    ResourceProfile,
    ResourceProfileRegistry,
    UnknownResourceProfileError,
    default_resource_profiles,
)


class AdmissionStatus(StrEnum):
    ADMITTED = "admitted"
    PREVIEW = "preview"
    DEFERRED = "deferred"
    REFUSED = "refused"
    STALE_FENCE = "stale_fence"


class AdmissionReason(StrEnum):
    ADMITTED = "admitted"
    PREVIEW = "preview_only"
    UNKNOWN_PROFILE = "unknown_profile"
    DEADLINE_EXPIRED = "queue_deadline_expired"
    STALE_FENCE = "stale_work_item_fence"
    NO_HOST = "no_eligible_host"
    CAPACITY = "capacity_insufficient"
    DISK_HIGH_WATERMARK = "disk_high_watermark"
    DISK_INSUFFICIENT = "disk_insufficient"
    DRAINED = "host_drained"
    QUARANTINED = "host_quarantined"
    STALE_HOST = "host_heartbeat_stale"
    EXCLUSIVITY = "reservation_exclusivity_conflict"
    CONCURRENCY = "concurrency_limit"
    FENCE_CONFLICT = "reservation_fence_conflict"
    LABELS = "host_labels"
    ANTI_AFFINITY = "anti_affinity"
    NATIVE_QUERY_REQUIRED = "native_query_required"
    NATIVE_NOT_FOUND = "native_reservation_not_found"


@dataclass(frozen=True)
class AdmissionRequest:
    """WorkItem-backed request presented to the scheduler."""

    work_item_id: str
    attempt: int
    fence: str
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    job_id: str = ""
    repository_id: str = ""
    branch: str = ""
    owner_id: str = ""
    tenant_id: str = ""
    enqueued_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    reservation_id: str = ""
    ttl_seconds: int = 900
    profile_name: str = ""

    def __post_init__(self) -> None:
        if not self.work_item_id.strip() or not self.fence.strip():
            raise ValueError("work_item_id and fence must be non-blank")
        if self.attempt < 1 or self.ttl_seconds < 1:
            raise ValueError("attempt and ttl_seconds must be positive")

    @property
    def id(self) -> str:
        return self.job_id or self.work_item_id

    def profile_and_request(
        self, registry: ResourceProfileRegistry
    ) -> tuple[ResourceProfile, ResourceRequest]:
        profile_name = self.profile_name or self.resources.resource_class
        profile = registry.resolve(profile_name)
        request = self.resources
        if request.resource_class != profile.name:
            request = request.model_copy(update={"resource_class": profile.name})
        return profile, profile.merge_request(request)


ResourceAdmissionRequest = AdmissionRequest


def reservation_id_for(work_item_id: str, attempt: int) -> str:
    """Return the stable native reservation identity for one WorkItem attempt."""

    if not work_item_id.strip() or attempt < 1:
        raise ValueError("work_item_id must be non-blank and attempt positive")
    material = f"repository-manager:reservation:v1:{work_item_id}\0{attempt}"
    digest = hashlib.sha256(material.encode("utf-8")).hexdigest()
    return f"reservation:{digest}"


def _reservation_input_fingerprint(
    request: AdmissionRequest,
    profile: ResourceProfile,
    resources: ResourceRequest,
    reservation_id: str,
) -> str:
    """Hash all retry-stable admission input, excluding observations/timestamps."""

    payload = {
        "version": "v1",
        "reservation_id": reservation_id,
        "work_item_id": request.work_item_id,
        "attempt": request.attempt,
        "fence": request.fence,
        "profile": profile.name,
        "profile_version": profile.profile_version,
        "job_id": request.job_id,
        "repository_id": request.repository_id,
        "branch": request.branch,
        "owner_id": request.owner_id,
        "tenant_id": request.tenant_id,
        "ttl_seconds": request.ttl_seconds,
        "resources": resources.model_dump(mode="json"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"v1:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class AdmissionDecision:
    status: AdmissionStatus
    reason_code: AdmissionReason
    reason: str
    request_id: str
    host_id: str = ""
    reservation_id: str = ""
    reservation: ReservationRecord | None = None
    capacity: CapacityView | None = None
    disk: DiskDecision | None = None
    considered_hosts: tuple[str, ...] = ()
    explanations: tuple[str, ...] = ()

    @property
    def admitted(self) -> bool:
        # A preview intentionally has no native reservation evidence and must
        # never be mistaken for an executable handoff by status-only callers.
        return (
            self.status == AdmissionStatus.ADMITTED
            and self.reservation is not None
            and self.reservation.active
        )

    @property
    def preview(self) -> bool:
        return self.status is AdmissionStatus.PREVIEW

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status.value,
            "reason_code": self.reason_code.value,
            "reason": self.reason,
            "request_id": self.request_id,
            "host_id": self.host_id,
            "reservation_id": self.reservation_id,
            "reservation": self.reservation.to_dict() if self.reservation else None,
            "capacity": self.capacity.as_dict() if self.capacity else None,
            "disk": self.disk.as_dict() if self.disk else None,
            "considered_hosts": list(self.considered_hosts),
            "explanations": list(self.explanations),
        }


class ResourceScheduler:
    """Pure-fit + fenced-reservation admission service."""

    def __init__(
        self,
        *,
        profiles: ResourceProfileRegistry | None = None,
        capacity: CapacityInventory | None = None,
        reservation_store: ReservationStore | None = None,
        work_item_port: WorkItemReservationPort,
        disk_policy: DiskPolicy | None = None,
        fairness: FairnessSelector | None = None,
    ) -> None:
        self.profiles = profiles or default_resource_profiles()
        self.capacity = capacity or CapacityInventory()
        self.reservations = reservation_store or InMemoryReservationStore()
        self.work_item_port = work_item_port
        self.disk_policy = disk_policy or DiskPolicy()
        self.fairness = fairness or FairnessSelector()
        self._sync_native_capacity()
        self._hydrate_capacity_mirror()

    def admit(
        self,
        request: AdmissionRequest,
        *,
        now: datetime | None = None,
        explain_only: bool = False,
    ) -> AdmissionDecision:
        """Admit one leased WorkItem or return an actionable deferral/refusal."""

        now = (now or datetime.now(UTC)).astimezone(UTC)
        resolved = self._resolve_profile_and_deadline(request, now)
        if isinstance(resolved, AdmissionDecision):
            return resolved
        profile, resources = resolved

        reservation_id = request.reservation_id or reservation_id_for(
            request.work_item_id, request.attempt
        )
        input_fingerprint = _reservation_input_fingerprint(
            request, profile, resources, reservation_id
        )
        existing = self.reservations.get(reservation_id)
        native_query = getattr(self.work_item_port, "query_reservation", None)
        decision = self._admit_via_reservation_identity(
            request, reservation_id, input_fingerprint, existing, native_query, now
        )
        if decision is not None:
            return decision

        requirement = ResourceVector(
            cpu_weight=resources.cpu_weight,
            memory_mib=resources.memory_mib,
            disk_mib=resources.disk_mib,
            process_slots=resources.process_slots,
        )
        disk_policy_key = f"{profile.name}:v{profile.profile_version}"
        active = self._active_reservations()
        hosts = self._eligible_hosts(resources, now=now)
        host_results, reasons = self._evaluate_candidate_hosts(
            profile,
            request,
            resources,
            requirement,
            reservation_id,
            disk_policy_key,
            active,
            hosts,
            now=now,
            explain_only=explain_only,
        )

        eligible = self._filter_eligible_hosts(host_results)
        if not eligible:
            return self._deferred_decision_for_no_eligible_host(
                request, host_results, reasons, hosts
            )

        # Host ordering is deterministic and respects preferred target before
        # remaining eligible hosts.  Fairness for queued jobs is handled by
        # ``select``; a single admission must not mutate queue counters.
        eligible.sort(key=lambda item: self._host_sort_key(resources, item[0]))
        for view, disk_or_none, _ in eligible:
            if disk_or_none is None:
                continue
            decision = self._attempt_reservation_on_host(
                request,
                profile,
                resources,
                requirement,
                reservation_id,
                input_fingerprint,
                disk_policy_key,
                view,
                disk_or_none,
                now,
                explain_only,
                native_query,
                reasons,
                eligible,
            )
            if decision is not None:
                return decision

        return self._decision(
            request,
            AdmissionStatus.DEFERRED,
            AdmissionReason.CAPACITY,
            "capacity changed while trying eligible hosts; retry through WorkItem queue",
            considered_hosts=tuple(item[0].host_id for item in eligible),
            explanations=tuple(reasons),
        )

    def _active_reservations(self) -> tuple[ReservationRecord, ...]:
        return tuple(record for record in self.reservations.all() if record.active)

    @staticmethod
    def _filter_eligible_hosts(
        host_results: list[tuple[CapacityView, DiskDecision | None, str]],
    ) -> list[tuple[CapacityView, DiskDecision | None, str]]:
        return [item for item in host_results if item[2] == "eligible"]

    def _resolve_profile_and_deadline(
        self, request: AdmissionRequest, now: datetime
    ) -> AdmissionDecision | tuple[ResourceProfile, ResourceRequest]:
        """Resolve the resource profile and enforce the queue deadline.

        Extracted verbatim from the head of ``admit`` (pure extract-method,
        no reordering): resolves the profile/request pair or returns the
        same refusal/deadline decisions ``admit`` returned inline before.
        """

        try:
            profile, resources = request.profile_and_request(self.profiles)
        except UnknownResourceProfileError as exc:
            return self._decision(
                request,
                AdmissionStatus.REFUSED,
                AdmissionReason.UNKNOWN_PROFILE,
                str(exc),
            )
        except ValueError as exc:
            return self._decision(
                request,
                AdmissionStatus.REFUSED,
                AdmissionReason.UNKNOWN_PROFILE,
                str(exc),
            )

        if resources.queue_deadline is not None and now > resources.queue_deadline:
            return self._decision(
                request,
                AdmissionStatus.REFUSED,
                AdmissionReason.DEADLINE_EXPIRED,
                "queue deadline elapsed before admission",
            )
        return profile, resources

    def _admit_via_reservation_identity(
        self,
        request: AdmissionRequest,
        reservation_id: str,
        input_fingerprint: str,
        existing: ReservationRecord | None,
        native_query: Any,
        now: datetime,
    ) -> AdmissionDecision | None:
        """Resolve admission from an existing/native reservation identity.

        Extracted verbatim from the middle of ``admit`` (pure extract-method,
        same conditions in the same order).  Returns ``None`` only when
        ``admit`` must fall through to a fresh host search, exactly as the
        original inline code did.
        """

        if existing is None and callable(native_query):
            decision = self._admit_via_native_query_for_missing_local(
                request, reservation_id, native_query, input_fingerprint, now
            )
            if decision is not None:
                return decision
        if existing is not None:
            return self._admit_via_existing_local_reservation(
                request, reservation_id, input_fingerprint, existing, now
            )
        if not self.work_item_port.is_current(
            request.work_item_id, request.attempt, request.fence
        ):
            return self._decision(
                request,
                AdmissionStatus.STALE_FENCE,
                AdmissionReason.STALE_FENCE,
                "WorkItem lease/fence is not current; admission did not mutate capacity",
            )
        return None

    def _admit_via_native_query_for_missing_local(
        self,
        request: AdmissionRequest,
        reservation_id: str,
        native_query: Any,
        input_fingerprint: str,
        now: datetime,
    ) -> AdmissionDecision | None:
        # Stable reservation identities are queried before placement.  A
        # replica with no local row must discover an existing native hold
        # rather than selecting a speculative host and treating an
        # IDEMPOTENT CAS result as permission to persist that candidate.
        native_existing = native_query(
            reservation_id=reservation_id,
            work_item_id=request.work_item_id,
            attempt=request.attempt,
            fence=request.fence,
        )
        if isinstance(native_existing, ReservationRecord):
            if (
                native_existing.active
                and native_existing.input_fingerprint == input_fingerprint
            ):
                self._mirror_native_record(native_existing)
                return self._decision(
                    request,
                    AdmissionStatus.ADMITTED,
                    AdmissionReason.ADMITTED,
                    "native reservation revalidated and local projection rebuilt",
                    host_id=native_existing.host_id,
                    reservation_id=native_existing.reservation_id,
                    reservation=native_existing,
                    capacity=(
                        self.capacity.snapshot(native_existing.host_id, now=now)
                        if self.capacity.get(native_existing.host_id) is not None
                        else None
                    ),
                    considered_hosts=(native_existing.host_id,),
                )
            return self._decision(
                request,
                AdmissionStatus.DEFERRED,
                AdmissionReason.FENCE_CONFLICT,
                "native reservation lifecycle/input conflicts with this retry",
                reservation_id=reservation_id,
            )
        if native_existing not in {None, FenceDecision.NOT_FOUND}:
            return self._decision(
                request,
                AdmissionStatus.STALE_FENCE
                if native_existing == FenceDecision.STALE
                else AdmissionStatus.DEFERRED,
                self._native_query_reason(native_existing),
                "native reservation query did not authorize placement",
                reservation_id=reservation_id,
            )
        return None

    def _admit_via_existing_local_reservation(
        self,
        request: AdmissionRequest,
        reservation_id: str,
        input_fingerprint: str,
        existing: ReservationRecord,
        now: datetime,
    ) -> AdmissionDecision:
        same_claim = (
            existing.work_item_id,
            existing.attempt,
            existing.fence,
        ) == (request.work_item_id, request.attempt, request.fence)
        if same_claim and existing.input_fingerprint == input_fingerprint:
            # A local row is only a projection.  Even an apparently active
            # row must be exact-queried against native WorkItem authority
            # before an ADMITTED retry can hand work to an executor.
            query = getattr(self.work_item_port, "query_reservation", None)
            if not callable(query):
                return self._decision(
                    request,
                    AdmissionStatus.DEFERRED,
                    AdmissionReason.NATIVE_QUERY_REQUIRED,
                    "native reservation revalidation is required; local projection "
                    "cannot authorize execution",
                    reservation_id=reservation_id,
                )
            native = query(
                reservation_id=reservation_id,
                work_item_id=request.work_item_id,
                attempt=request.attempt,
                fence=request.fence,
                expected=existing,
            )
            if isinstance(native, ReservationRecord):
                if native.active and native.input_fingerprint == input_fingerprint:
                    self._mirror_native_record(native)
                    native_capacity = (
                        self.capacity.snapshot(native.host_id, now=now)
                        if self.capacity.get(native.host_id) is not None
                        else None
                    )
                    return self._decision(
                        request,
                        AdmissionStatus.ADMITTED,
                        AdmissionReason.ADMITTED,
                        "native reservation revalidated for current WorkItem fence",
                        host_id=native.host_id,
                        reservation_id=native.reservation_id,
                        reservation=native,
                        capacity=native_capacity,
                        considered_hosts=(native.host_id,),
                    )
                return self._decision(
                    request,
                    AdmissionStatus.DEFERRED,
                    AdmissionReason.FENCE_CONFLICT,
                    "native reservation lifecycle is not active; local projection "
                    "cannot authorize execution",
                    reservation_id=reservation_id,
                )
            query_reason = self._native_query_reason(native)
            return self._decision(
                request,
                AdmissionStatus.STALE_FENCE
                if native == FenceDecision.STALE
                else AdmissionStatus.DEFERRED,
                query_reason,
                "native reservation revalidation did not authorize this retry",
                reservation_id=reservation_id,
            )
        return self._decision(
            request,
            AdmissionStatus.DEFERRED,
            AdmissionReason.FENCE_CONFLICT,
            "reservation identity is already attached to a different immutable input",
            reservation_id=reservation_id,
        )

    def _evaluate_candidate_hosts(
        self,
        profile: ResourceProfile,
        request: AdmissionRequest,
        resources: ResourceRequest,
        requirement: ResourceVector,
        reservation_id: str,
        disk_policy_key: str,
        active: tuple[ReservationRecord, ...],
        hosts: list[CapacityView],
        *,
        now: datetime,
        explain_only: bool,
    ) -> tuple[list[tuple[CapacityView, DiskDecision | None, str]], list[str]]:
        """Classify each candidate host, in order, exactly as ``admit`` did.

        This still calls ``self.disk_policy.evaluate(...)`` once per host in
        the same host order as the original inline loop -- that call has
        side effects (GC/mutate the disk policy's predicted-usage ledger),
        so the per-host iteration order and count are preserved exactly.
        """

        host_results: list[tuple[CapacityView, DiskDecision | None, str]] = []
        reasons: list[str] = []
        for view in hosts:
            if view.state in {HostState.DRAINED, HostState.DRAINING}:
                reasons.append(f"{view.host_id}: {AdmissionReason.DRAINED.value}")
                continue
            if view.state in {HostState.QUARANTINED, HostState.OFFLINE}:
                reasons.append(f"{view.host_id}: {AdmissionReason.QUARANTINED.value}")
                continue
            if not view.heartbeat_fresh:
                reasons.append(f"{view.host_id}: {AdmissionReason.STALE_HOST.value}")
                continue
            conflict = self._conflict(profile, request, resources, view.host_id, active)
            if conflict:
                reasons.append(f"{view.host_id}: {conflict}")
                continue
            # ``available`` already subtracts scheduler reservations.  Using
            # ``available + reserved`` here would hide outstanding predicted
            # reservations from the watermark calculation and admit the next
            # heavy job into the same disk budget.
            free_disk = view.available.disk_mib
            total_disk = view.total.disk_mib
            disk = self.disk_policy.evaluate(
                view.host_id,
                total_mib=total_disk,
                free_mib=free_disk,
                requested_mib=requirement.disk_mib,
                watermarks=DiskWatermarks(
                    low_mib=resources.disk_low_watermark_mib,
                    high_mib=resources.disk_high_watermark_mib,
                    policy_key=disk_policy_key,
                ),
                reservation_id=reservation_id,
                request_gc=not explain_only,
                mutate=not explain_only,
                policy_key=disk_policy_key,
            )
            if not disk.admitted:
                reasons.append(f"{view.host_id}: {disk.code.value}")
                host_results.append((view, disk, disk.reason))
                continue
            if not view.available.fits(requirement):
                reasons.append(f"{view.host_id}: {AdmissionReason.CAPACITY.value}")
                host_results.append((view, disk, "insufficient weighted capacity"))
                continue
            host_results.append((view, disk, "eligible"))
        return host_results, reasons

    def _deferred_decision_for_no_eligible_host(
        self,
        request: AdmissionRequest,
        host_results: list[tuple[CapacityView, DiskDecision | None, str]],
        reasons: list[str],
        hosts: list[CapacityView],
    ) -> AdmissionDecision:
        disk_failure = next(
            (
                item
                for item in host_results
                if item[1] and item[1].code == DiskDecisionCode.HIGH_WATERMARK
            ),
            None,
        )
        if disk_failure:
            disk_failure_decision = disk_failure[1]
            assert disk_failure_decision is not None
            return self._decision(
                request,
                AdmissionStatus.DEFERRED,
                AdmissionReason.DISK_HIGH_WATERMARK,
                disk_failure_decision.reason,
                host_id=disk_failure[0].host_id,
                capacity=disk_failure[0],
                disk=disk_failure_decision,
                considered_hosts=tuple(view.host_id for view in hosts),
                explanations=tuple(reasons),
            )
        reason_code = self._no_host_reason_code(host_results, reasons)
        return self._decision(
            request,
            AdmissionStatus.DEFERRED,
            reason_code,
            "no eligible host currently satisfies admission policy",
            considered_hosts=tuple(view.host_id for view in hosts),
            explanations=tuple(reasons),
        )

    @staticmethod
    def _no_host_reason_code(
        host_results: list[tuple[CapacityView, DiskDecision | None, str]],
        reasons: list[str],
    ) -> AdmissionReason:
        """Same elif cascade as the original inline code, as sequential ifs.

        An elif chain runs only the first true branch; an if/return chain in
        the same order has identical behaviour, so this is a pure
        extract-method with no semantic change.  Split into two functions,
        called in sequence, purely to keep each below the complexity cap.
        """

        if host_results and any(
            item[1] and item[1].code == DiskDecisionCode.INSUFFICIENT_FREE
            for item in host_results
        ):
            return AdmissionReason.DISK_INSUFFICIENT
        if any("concurrency" in reason for reason in reasons):
            return AdmissionReason.CONCURRENCY
        if any("exclusive" in reason for reason in reasons):
            return AdmissionReason.EXCLUSIVITY
        return ResourceScheduler._no_host_reason_code_from_health(reasons)

    @staticmethod
    def _no_host_reason_code_from_health(reasons: list[str]) -> AdmissionReason:
        if any(AdmissionReason.CAPACITY.value in reason for reason in reasons):
            return AdmissionReason.CAPACITY
        if any(AdmissionReason.DRAINED.value in reason for reason in reasons):
            return AdmissionReason.DRAINED
        if any(AdmissionReason.QUARANTINED.value in reason for reason in reasons):
            return AdmissionReason.QUARANTINED
        if any(AdmissionReason.STALE_HOST.value in reason for reason in reasons):
            return AdmissionReason.STALE_HOST
        return AdmissionReason.NO_HOST

    def _build_reservation_record(
        self,
        request: AdmissionRequest,
        profile: ResourceProfile,
        resources: ResourceRequest,
        requirement: ResourceVector,
        reservation_id: str,
        input_fingerprint: str,
        disk_policy_key: str,
        view: CapacityView,
        now: datetime,
    ) -> ReservationRecord:
        reserved_at = now
        return ReservationRecord(
            reservation_id=reservation_id,
            work_item_id=request.work_item_id,
            attempt=request.attempt,
            fence=request.fence,
            host_id=view.host_id,
            profile_name=profile.name,
            requirement=requirement,
            capacity_snapshot=view.as_dict(),
            selected_target=TargetPolicy(
                kind=(
                    TargetKind.INVENTORY_ALIAS
                    if view.is_remote
                    else TargetKind.LOCAL
                ),
                alias=view.host_id if view.is_remote else None,
                capability_labels=view.labels,
            ),
            concurrency_key=resources.concurrency_key or profile.concurrency_key,
            concurrency_limit=profile.concurrency_limit,
            repository_exclusive=profile.repository_exclusive,
            branch_exclusive=profile.branch_exclusive,
            required_labels=resources.host_labels,
            disk_low_watermark_mib=resources.disk_low_watermark_mib,
            disk_high_watermark_mib=resources.disk_high_watermark_mib,
            disk_policy_key=disk_policy_key,
            repository_id=request.repository_id,
            branch=request.branch,
            owner_id=request.owner_id,
            tenant_id=request.tenant_id,
            fairness_group=resources.fairness_group,
            fairness_cost=max(1, resources.cpu_weight + resources.process_slots),
            anti_affinity=tuple(
                sorted(set(resources.anti_affinity).union(profile.anti_affinity))
            ),
            reserved_at=reserved_at,
            expires_at=reserved_at + timedelta(seconds=request.ttl_seconds),
            input_fingerprint=input_fingerprint,
        )

    def _compensate_release_if_new(
        self,
        local_was_held: bool,
        host_id: str,
        reservation_id: str,
        requirement: ResourceVector,
    ) -> None:
        """Undo ``try_reserve`` only if this admission attempt created it.

        Identical to the ``if not local_was_held: self.capacity.release(...)``
        guard that was inlined at four call sites in the original ``admit``;
        factored out verbatim (same condition, same call, same arguments) to
        avoid repeating it four times in the decomposed code.
        """

        if not local_was_held:
            self.capacity.release(host_id, reservation_id, requirement)

    def _reconcile_idempotent_native_result(
        self,
        native: Any,
        request: AdmissionRequest,
        reservation_id: str,
        input_fingerprint: str,
        record: ReservationRecord,
        view: CapacityView,
        requirement: ResourceVector,
        local_was_held: bool,
        native_query: Any,
        disk: DiskDecision,
        reasons: list[str],
        eligible: list[tuple[CapacityView, DiskDecision | None, str]],
    ) -> tuple[ReservationRecord, AdmissionDecision | None]:
        """Handle an ``atomic_reserve`` IDEMPOTENT result, verbatim.

        Returns ``(record, None)`` to continue with ``record`` (unchanged
        unless the native row is revalidated) or ``(record, decision)`` to
        make ``admit`` return immediately, exactly as the original inline
        code did.
        """

        if native != FenceDecision.IDEMPOTENT:
            return record, None
        if not callable(native_query):
            # A legacy port that cannot return the authoritative record
            # cannot safely authorize an execution handoff from an
            # idempotent result.  Leave native capacity held for a later
            # RMDD-27 reconciliation rather than trusting the candidate.
            self._compensate_release_if_new(
                local_was_held, view.host_id, reservation_id, requirement
            )
            return record, self._decision(
                request,
                AdmissionStatus.DEFERRED,
                AdmissionReason.NATIVE_QUERY_REQUIRED,
                "native IDEMPOTENT result requires an exact reservation query",
                host_id=view.host_id,
                reservation_id=reservation_id,
                capacity=view,
                disk=disk,
                considered_hosts=tuple(item[0].host_id for item in eligible),
                explanations=tuple(reasons),
            )
        native_record = native_query(
            reservation_id=reservation_id,
            work_item_id=request.work_item_id,
            attempt=request.attempt,
            fence=request.fence,
            expected=record,
        )
        if not isinstance(native_record, ReservationRecord) or not (
            native_record.active
            and native_record.input_fingerprint == input_fingerprint
        ):
            self._compensate_release_if_new(
                local_was_held, view.host_id, reservation_id, requirement
            )
            return record, self._decision(
                request,
                AdmissionStatus.DEFERRED,
                self._native_query_reason(native_record),
                "native IDEMPOTENT result could not be revalidated",
                host_id=view.host_id,
                reservation_id=reservation_id,
            )
        self._mirror_native_record(native_record)
        return native_record, None

    def _attempt_reservation_on_host(
        self,
        request: AdmissionRequest,
        profile: ResourceProfile,
        resources: ResourceRequest,
        requirement: ResourceVector,
        reservation_id: str,
        input_fingerprint: str,
        disk_policy_key: str,
        view: CapacityView,
        disk: DiskDecision,
        now: datetime,
        explain_only: bool,
        native_query: Any,
        reasons: list[str],
        eligible: list[tuple[CapacityView, DiskDecision | None, str]],
    ) -> AdmissionDecision | None:
        """Try to admit onto one already-eligible host, verbatim.

        Extracted from the body of the original ``for view, disk_or_none, _
        in eligible:`` loop.  Returns ``None`` only where the original had a
        bare ``continue`` (capacity changed during ``try_reserve``) so the
        caller's loop moves on to the next host in the same order; every
        other path returns the same ``AdmissionDecision`` the inline code
        returned, in the same order, with the same side effects.
        """

        record = self._build_reservation_record(
            request,
            profile,
            resources,
            requirement,
            reservation_id,
            input_fingerprint,
            disk_policy_key,
            view,
            now,
        )
        if explain_only:
            return self._decision(
                request,
                AdmissionStatus.PREVIEW,
                AdmissionReason.PREVIEW,
                "request fits; explain-only mode did not mutate capacity or WorkItem",
                host_id=view.host_id,
                reservation_id=reservation_id,
                capacity=view,
                disk=disk,
                considered_hosts=tuple(item[0].host_id for item in eligible),
                explanations=tuple(reasons),
            )
        local_was_held = reservation_id in self.capacity.reservation_ids(
            view.host_id
        )
        if not self.capacity.try_reserve(
            view.host_id, reservation_id, requirement, now=now
        ):
            reasons.append(f"{view.host_id}: capacity changed during admission")
            return None
        native = self.work_item_port.atomic_reserve(
            work_item_id=request.work_item_id,
            attempt=request.attempt,
            fence=request.fence,
            reservation=record,
            expected_capacity=view,
        )
        if not self._accepted(native):
            self._compensate_release_if_new(
                local_was_held, view.host_id, reservation_id, requirement
            )
            native_reason = self._native_reason(native)
            return self._decision(
                request,
                AdmissionStatus.STALE_FENCE
                if native == FenceDecision.STALE
                else AdmissionStatus.DEFERRED,
                native_reason,
                f"native WorkItem reservation transaction refused admission: {native}",
                host_id=view.host_id,
                reservation_id=reservation_id,
                capacity=view,
                disk=disk,
                considered_hosts=tuple(item[0].host_id for item in eligible),
                explanations=tuple(reasons),
            )
        record, early_decision = self._reconcile_idempotent_native_result(
            native,
            request,
            reservation_id,
            input_fingerprint,
            record,
            view,
            requirement,
            local_was_held,
            native_query,
            disk,
            reasons,
            eligible,
        )
        if early_decision is not None:
            return early_decision
        try:
            self.reservations.put(record)
        except Exception:
            # The local row is only a projection.  Never use normal
            # release as compensation: native admission may already have
            # recorded fairness debt, and release cannot atomically undo
            # that historical service.  Leave the native reservation held
            # and let a deterministic retry/reconciler rebuild the mirror.
            self._compensate_release_if_new(
                local_was_held, view.host_id, reservation_id, requirement
            )
            raise
        return self._decision(
            request,
            AdmissionStatus.ADMITTED,
            AdmissionReason.ADMITTED,
            "resource capacity atomically reserved against the current WorkItem fence",
            host_id=view.host_id,
            reservation_id=reservation_id,
            reservation=record,
            capacity=self.capacity.snapshot(view.host_id, now=now),
            disk=disk,
            considered_hosts=tuple(item[0].host_id for item in eligible),
            explanations=tuple(reasons),
        )


    def release(
        self,
        reservation_id: str,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reason: str = "completed",
        at: datetime | None = None,
    ) -> bool:
        """Release through the exact current/terminal WorkItem fence."""

        record = self.reservations.get(reservation_id)
        if record is None:
            return False
        if (record.work_item_id, record.attempt, record.fence) != (
            work_item_id,
            attempt,
            fence,
        ):
            return False
        if not record.active:
            # An exact retry after the local terminal projection succeeded is
            # idempotent success only when native lifecycle authority confirms
            # the same release tombstone.  A local state row alone is never
            # enough to report completion.
            query = getattr(self.work_item_port, "query_reservation", None)
            if record.state is not ReservationState.RELEASED or not callable(query):
                return False
            native_terminal = query(
                reservation_id=reservation_id,
                work_item_id=work_item_id,
                attempt=attempt,
                fence=fence,
                expected=record,
                for_lifecycle=True,
            )
            return bool(
                isinstance(native_terminal, ReservationRecord)
                and native_terminal.state is ReservationState.RELEASED
                and native_terminal.revision >= record.revision
                and native_terminal.same_immutable_input(record)
            )
        native = self.work_item_port.atomic_release(
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reservation_id=reservation_id,
            reservation=record,
        )
        if not self._accepted(native):
            return False
        self.capacity.release(record.host_id, record.reservation_id, record.requirement)
        self.reservations.update(
            record.with_state(ReservationState.RELEASED, reason=reason, at=at)
        )
        return True

    def reclaim_expired(
        self,
        *,
        now: datetime | None = None,
        limit: int = 100,
    ) -> tuple[str, ...]:
        """Reclaim expired records only when the native port authorizes it."""

        now = (now or datetime.now(UTC)).astimezone(UTC)
        reclaimed: list[str] = []
        reclaim_expired = getattr(self.work_item_port, "atomic_reclaim_expired", None)
        query_native = getattr(self.work_item_port, "query_reservation", None)
        for record in sorted(
            self.reservations.all(), key=lambda item: item.reservation_id
        ):
            if len(reclaimed) >= limit:
                break
            if not record.active:
                if (
                    record.state is ReservationState.EXPIRED
                    and callable(reclaim_expired)
                    and callable(query_native)
                ):
                    native_terminal = query_native(
                        reservation_id=record.reservation_id,
                        work_item_id=record.work_item_id,
                        attempt=record.attempt,
                        fence=record.fence,
                        expected=record,
                        for_lifecycle=True,
                        controller=True,
                    )
                    if (
                        isinstance(native_terminal, ReservationRecord)
                        and native_terminal.state is ReservationState.EXPIRED
                        and native_terminal.revision >= record.revision
                        and native_terminal.same_immutable_input(record)
                    ):
                        reclaimed.append(record.reservation_id)
                continue
            if not record.expired(now) and not callable(reclaim_expired):
                continue
            if callable(reclaim_expired):
                native = reclaim_expired(reservation=record, now=now)
            else:
                native = self.work_item_port.atomic_reclaim(
                    work_item_id=record.work_item_id,
                    attempt=record.attempt,
                    fence=record.fence,
                    reservation_id=record.reservation_id,
                    reservation=record,
                    now=now,
                )
            if not self._accepted(native):
                continue
            self.capacity.release(
                record.host_id, record.reservation_id, record.requirement
            )
            self.reservations.update(
                record.with_state(
                    ReservationState.EXPIRED,
                    reason="reservation TTL elapsed and native fence authority permitted reclaim",
                    at=now,
                )
            )
            reclaimed.append(record.reservation_id)
        return tuple(reclaimed)

    def select(
        self,
        requests: Iterable[AdmissionRequest],
        *,
        limit: int = 1,
        now: datetime | None = None,
    ) -> tuple[AdmissionRequest, ...]:
        """Fairly order queued requests without changing WorkItem state."""

        now = (now or datetime.now(UTC)).astimezone(UTC)
        requests = tuple(requests)
        candidates: list[QueueCandidate] = []
        for request in requests:
            try:
                profile, resources = request.profile_and_request(self.profiles)
            except ValueError:
                continue
            candidates.append(
                QueueCandidate(
                    candidate_id=request.id,
                    fairness_group=resources.fairness_group
                    or profile.default_fairness_group,
                    priority=resources.priority,
                    enqueued_at=request.enqueued_at,
                    cost=max(1, resources.cpu_weight + resources.process_slots),
                )
            )
        chosen = self.fairness.choose(candidates, limit=limit, now=now)
        by_id = {request.id: request for request in requests}
        return tuple(by_id[candidate.candidate_id] for candidate in chosen)

    def status(self, *, now: datetime | None = None) -> dict[str, object]:
        return {
            "profiles": list(self.profiles.names()),
            "fairness_authoritative": self.fairness.authoritative,
            "fairness_authority": self.fairness.authority.value,
            "hosts": [view.as_dict() for view in self.capacity.views(now=now)],
            "reservations": [record.to_dict() for record in self.reservations.all()],
        }

    def _sync_native_capacity(self) -> None:
        """Seed deterministic native fixtures with host policy and telemetry.

        Real graph adapters do not need this optional hook: they already own
        the capacity records and revalidate them in their transaction.  The
        hook is intentionally capability-detected; a native fixture retains
        its first registered view so a later replica cannot overwrite
        authority with a local projection.
        """

        register_fairness = getattr(
            self.work_item_port, "register_fairness_state", None
        )
        if callable(register_fairness):
            register_fairness(self.fairness.state)
        views = self.capacity.views()
        register_view = getattr(self.work_item_port, "register_capacity_view", None)
        if callable(register_view):
            for view in views:
                register_view(view)
            return
        register = getattr(self.work_item_port, "register_capacity", None)
        if callable(register):
            for view in views:
                register(view.host_id, view.total)

    def _hydrate_capacity_mirror(self) -> None:
        """Rebuild local accounting from durable reservation projections.

        This is deliberately a mirror operation.  The native port remains the
        authority and must reconcile its own records during service restart;
        a corrupt or incomplete local projection cannot grant admission.
        """

        for record in self.reservations.all():
            if not record.active or self.capacity.get(record.host_id) is None:
                continue
            try:
                self.capacity.restore_reservation(
                    record.host_id,
                    record.reservation_id,
                    record.requirement,
                )
            except ValueError as exc:
                raise ValueError(
                    f"durable reservation {record.reservation_id!r} exceeds local "
                    "capacity mirror; native reconciliation is required"
                ) from exc

    def _mirror_native_record(self, record: ReservationRecord) -> None:
        """Rebuild local projections from an already-authorized native row.

        This helper never grants authority.  Native query/CAS has already
        validated the WorkItem fence; local persistence failures leave the
        native hold intact and are repaired by a deterministic retry.
        """

        if self.capacity.get(record.host_id) is not None:
            self.capacity.restore_reservation(
                record.host_id,
                record.reservation_id,
                record.requirement,
            )
        local = self.reservations.get(record.reservation_id)
        if local is None:
            self.reservations.put(record)
        elif not local.same_immutable_input(record):
            raise ValueError(
                f"local reservation {record.reservation_id!r} conflicts with native row"
            )

    def _eligible_hosts(
        self, resources: ResourceRequest, *, now: datetime | None = None
    ) -> list[CapacityView]:
        required = resources.required_target
        preferred = resources.preferred_target
        required_alias = (
            required.alias if required and required.kind != TargetKind.LOCAL else None
        )
        preferred_alias = (
            preferred.alias if preferred.kind != TargetKind.LOCAL else None
        )
        views = list(self.capacity.views(now=now))
        if required_alias:
            views = [view for view in views if view.host_id == required_alias]
        if resources.host_labels:
            labels = set(resources.host_labels)
            views = [view for view in views if labels.issubset(view.labels)]
        # A local request cannot be silently routed to a remote inventory host
        # unless its required/preferred target explicitly names that alias.
        if required_alias is None and preferred_alias is None:
            views = [view for view in views if not view.is_remote]
        return sorted(
            views,
            key=lambda view: (
                0 if preferred_alias and view.host_id == preferred_alias else 1,
                view.host_id,
            ),
        )

    def _host_sort_key(
        self, resources: ResourceRequest, view: CapacityView
    ) -> tuple[int, int, str]:
        preferred = resources.preferred_target.alias
        return (
            0 if preferred and preferred == view.host_id else 1,
            -view.available.cpu_weight,
            view.host_id,
        )

    def _conflict(
        self,
        profile: ResourceProfile,
        request: AdmissionRequest,
        resources: ResourceRequest,
        host_id: str,
        active: Iterable[ReservationRecord],
    ) -> str:
        records = tuple(active)
        if profile.concurrency_limit is not None:
            count = sum(
                record.concurrency_key
                == (resources.concurrency_key or profile.concurrency_key)
                for record in records
            )
            if count >= profile.concurrency_limit:
                return f"concurrency limit for {profile.concurrency_key!r}"
        for record in records:
            if record.host_id == host_id and set(record.anti_affinity).intersection(
                set(resources.anti_affinity).union(profile.anti_affinity)
            ):
                return f"anti-affinity conflict with {record.reservation_id!r}"
            if (
                (profile.repository_exclusive or record.repository_exclusive)
                and request.repository_id
                and record.repository_id == request.repository_id
            ):
                return f"repository exclusive reservation {record.reservation_id!r}"
            if (
                (profile.branch_exclusive or record.branch_exclusive)
                and request.repository_id
                and request.branch
                and (
                    record.repository_id,
                    record.branch,
                )
                == (request.repository_id, request.branch)
            ):
                return f"branch exclusive reservation {record.reservation_id!r}"
        return ""

    @staticmethod
    def _accepted(value: FenceDecision | bool | str) -> bool:
        return (
            value is True
            or value == FenceDecision.ACCEPTED
            or value == FenceDecision.IDEMPOTENT
            or value == FenceDecision.ACCEPTED.value
            or value == FenceDecision.IDEMPOTENT.value
        )

    @staticmethod
    def _native_reason(value: FenceDecision | bool | str | None) -> AdmissionReason:
        if value == FenceDecision.STALE:
            return AdmissionReason.STALE_FENCE
        mapping = {
            FenceDecision.INPUT_CONFLICT: AdmissionReason.FENCE_CONFLICT,
            FenceDecision.CAPACITY: AdmissionReason.CAPACITY,
            FenceDecision.DISK: AdmissionReason.DISK_HIGH_WATERMARK,
            FenceDecision.CONCURRENCY: AdmissionReason.CONCURRENCY,
            FenceDecision.EXCLUSIVITY: AdmissionReason.EXCLUSIVITY,
            FenceDecision.DRAINED: AdmissionReason.DRAINED,
            FenceDecision.QUARANTINED: AdmissionReason.QUARANTINED,
            FenceDecision.STALE_HOST: AdmissionReason.STALE_HOST,
            FenceDecision.LABELS: AdmissionReason.LABELS,
            FenceDecision.ANTI_AFFINITY: AdmissionReason.ANTI_AFFINITY,
            FenceDecision.NOT_FOUND: AdmissionReason.NATIVE_NOT_FOUND,
        }
        if isinstance(value, FenceDecision):
            return mapping.get(value, AdmissionReason.FENCE_CONFLICT)
        return AdmissionReason.FENCE_CONFLICT

    @staticmethod
    def _native_query_reason(
        value: FenceDecision | bool | str | None,
    ) -> AdmissionReason:
        """Map an exact-query refusal without treating policy as fence loss."""

        if value is None:
            return AdmissionReason.NATIVE_NOT_FOUND
        return ResourceScheduler._native_reason(value)

    @staticmethod
    def _decision(
        request: AdmissionRequest,
        status: AdmissionStatus,
        reason_code: AdmissionReason,
        reason: str,
        *,
        host_id: str = "",
        reservation_id: str = "",
        reservation: ReservationRecord | None = None,
        capacity: CapacityView | None = None,
        disk: DiskDecision | None = None,
        considered_hosts: tuple[str, ...] = (),
        explanations: tuple[str, ...] = (),
    ) -> AdmissionDecision:
        return AdmissionDecision(
            status=status,
            reason_code=reason_code,
            reason=reason,
            request_id=request.id,
            host_id=host_id,
            reservation_id=reservation_id,
            reservation=reservation,
            capacity=capacity,
            disk=disk,
            considered_hosts=considered_hosts,
            explanations=explanations,
        )


def create_production_resource_scheduler(
    graph_client: object, **kwargs: Any
) -> ResourceScheduler:
    """Public construction seam for the native RMDD-27 scheduler binding."""

    from repository_manager.native_reservations import (
        create_production_resource_scheduler as _create_native_scheduler,
    )

    return cast(ResourceScheduler, _create_native_scheduler(graph_client, **kwargs))


__all__ = [
    "AdmissionDecision",
    "AdmissionReason",
    "AdmissionRequest",
    "AdmissionStatus",
    "ResourceAdmissionRequest",
    "ResourceScheduler",
    "create_production_resource_scheduler",
]
