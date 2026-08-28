"""Production WorkItem reservation binding for the RMDD-08 scheduler.

The scheduler keeps a small local projection for fit/explain operations, but
the native client in this module is the only authority for a mutation or an
execution handoff.  The client is injected deliberately: importing this
module does not discover a socket, create a fixture, or silently select an
older engine.

The native v1 result contains a complete immutable reservation record.  A
record is accepted only after its WorkItem, lease, fence, tenant, owner, and
revision correlations have been checked.  In particular, a malformed
``accepted`` response is a protocol failure; it is never compensated with a
local release because the native transaction may already have committed.
"""

from __future__ import annotations

import inspect
import uuid
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol, cast

from repository_manager.capacity import CapacityView, ResourceVector
from repository_manager.development import ReservationState, TargetKind, TargetPolicy
from repository_manager.reservations import (
    FenceDecision,
    ReservationRecord,
    WorkItemReservationPort,
)
from repository_manager.resource_profiles import ResourceProfileRegistry


class NativeResourceReservationUnavailable(RuntimeError):
    """The connected engine does not advertise the required native methods."""

    code = "native_resource_reservation_unavailable"


class NativeReservationProtocolError(ValueError):
    """A native response failed the versioned reservation contract."""


class NativeFenceCodec(Protocol):
    """Validate the native numeric lease pair against RM's opaque fence.

    A production codec may encode the exact deployed fence for the early
    native WorkItem query, but must also validate every returned numeric/direct
    pair.  It must never silently coerce a lossy ``DurableJobView.lease_fence``
    value into a fabricated epoch.  The direct opaque ``fence`` field in the
    corrected native projection remains mandatory whenever a result carries it.
    """

    def encode(self, fence: str) -> Mapping[str, int]:
        """Return the exact native numeric pair for an RM fence."""

    def validate(self, fence: str, lease_epoch: int, fencing_token: int) -> None:
        """Raise when the native lease pair is not this exact RM fence."""


class DurableJobFenceCodec:
    """Codec for the currently deployed :class:`DurableJobView` projection.

    AU's current WorkItem claim sets ``lease_epoch`` and ``fencing_token`` to
    the same value, while RM's ``DurableJobView.lease_fence`` is the exact
    decimal string of that token.  This strict codec preserves that deployed
    contract and rejects any lossy or composite-looking value.  A future
    versioned composite fence must ship a new codec and projection together;
    it must not be silently accepted here.
    """

    def encode(self, fence: str) -> Mapping[str, int]:
        value = _string(fence, name="fence")
        if not value.isdecimal() or str(int(value)) != value:
            raise NativeReservationProtocolError(
                "DurableJobView fence must be canonical decimal fencing token"
            )
        token = int(value)
        return {"lease_epoch": token, "fencing_token": token}

    def validate(self, fence: str, lease_epoch: int, fencing_token: int) -> None:
        if (
            _string(fence, name="fence") != str(fencing_token)
            or lease_epoch != fencing_token
        ):
            raise NativeReservationProtocolError(
                "DurableJobView fence must equal its paired native fencing token"
            )


NativeResult = Mapping[str, object]


class NativeReservationClient(Protocol):
    """Narrow client surface consumed by :class:`NativeWorkItemReservationPort`.

    The sibling generated client exposes these methods on its ``work_items``
    namespace.  Keeping this protocol structural lets focused tests inject a
    deterministic fake without importing the engine package.
    """

    def supports(self, operation: str) -> bool:
        """Return the server-advertised capability for one native operation."""

    def reserve(self, request: Mapping[str, object]) -> NativeResult:
        """Invoke ``ReserveWorkItemResources``."""

    def release(self, request: Mapping[str, object]) -> NativeResult:
        """Invoke ``ReleaseWorkItemResources``."""

    def reclaim(self, request: Mapping[str, object]) -> NativeResult:
        """Invoke ``ReclaimWorkItemResources``."""

    def query_reservation(self, request: Mapping[str, object]) -> NativeResult:
        """Invoke ``QueryWorkItemReservation``."""

    def status(self, request: Mapping[str, object]) -> NativeResult:
        """Invoke ``ResourceReservationStatus``."""

    def update_host(self, request: Mapping[str, object]) -> NativeResult:
        """Invoke ``UpdateResourceHost``."""


_OPERATION_NAMES = (
    "ReserveWorkItemResources",
    "ReleaseWorkItemResources",
    "ReclaimWorkItemResources",
    "QueryWorkItemReservation",
    "ResourceReservationStatus",
    "UpdateResourceHost",
)
_DECISIONS = {member.value: member for member in FenceDecision}
_STATE_MAP = {
    "reserved": ReservationState.RESERVED,
    "released": ReservationState.RELEASED,
    "reclaimed": ReservationState.EXPIRED,
    "expired": ReservationState.EXPIRED,
    "superseded": ReservationState.EXPIRED,
}
_FIXED_STATUS_REASONS = {"accepted", "stale_host", "conflict", "not_found"}
_MAX_U64 = (1 << 64) - 1


def _mapping(value: object, *, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise NativeReservationProtocolError(f"{name} must be an object")
    return cast(Mapping[str, object], value)


def _string(value: object, *, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise NativeReservationProtocolError(f"{name} must be a non-blank string")
    if len(value) > 256:
        raise NativeReservationProtocolError(f"{name} exceeds the 256-character bound")
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < minimum
        or value > _MAX_U64
    ):
        raise NativeReservationProtocolError(
            f"{name} must be an integer in [{minimum}, {_MAX_U64}]"
        )
    return value


def _boolean(value: object, *, name: str) -> bool:
    if not isinstance(value, bool):
        raise NativeReservationProtocolError(f"{name} must be a boolean")
    return value


def _list(value: object, *, name: str, maximum: int = 128) -> list[object]:
    if not isinstance(value, list):
        raise NativeReservationProtocolError(f"{name} must be a list")
    if len(value) > maximum:
        raise NativeReservationProtocolError(f"{name} exceeds the {maximum}-item bound")
    return value


def _target_fields(kind: object, alias: object, *, name: str) -> tuple[str, str | None]:
    """Validate the two native target fields as one correlated value."""

    target_kind = _string(kind, name=f"{name}.target_kind")
    if target_kind == "remote":
        target_kind = "inventory_alias"
    if target_kind not in {"local", "inventory_alias"}:
        raise NativeReservationProtocolError(
            f"{name}.target_kind is not a supported native target"
        )
    if target_kind == "local":
        if alias is not None:
            raise NativeReservationProtocolError(
                f"{name}.target_alias must be null for a local target"
            )
        return target_kind, None
    return target_kind, _string(alias, name=f"{name}.target_alias")


def _sync_value(value: object, *, name: str) -> object:
    """Reject an async result without creating a second event loop."""

    if inspect.isawaitable(value):
        close = getattr(value, "close", None)
        if callable(close):
            close()
        raise NativeResourceReservationUnavailable(
            f"native sync port received an async {name}"
        )
    return value


@dataclass(frozen=True)
class _StatusRequestFilters:
    """Bundled optional filters ``_validate_status_request_filters`` checks.

    Bundled so the validation helper stays under the fleet's parameter cap;
    ``_status_request`` itself keeps its existing keyword-only signature
    (pre-existing debt, not widened here).
    """

    work_item_id: str | None
    reservation_id: str | None
    host_ref: str | None
    owner_id: str | None
    fence: str | None
    attempt: int | None
    lease_epoch: int | None
    fencing_token: int | None
    input_fingerprint: str | None
    cursor: str | None


@dataclass(frozen=True)
class _ReservationRequestContext:
    """Derived/validated values ``_build_reservation_payload`` assembles.

    Bundled so the builder helper stays under the fleet's parameter cap.
    """

    tenant: str
    owner: str
    requirement: dict[str, int]
    target_kind: str
    target_alias: str | None
    profile_version: str
    input_fingerprint: str
    required_labels: list[str]
    anti_affinity: list[str]
    reserved_at_ms: int
    expires_at_ms: int
    host_revision: int | None
    now: datetime | None
    operation: str


@dataclass(frozen=True)
class _HostUpdateRequest:
    """Bundled inputs for ``NativeWorkItemReservationPort.update_host``.

    Bundled so the payload-building helper stays under the fleet's
    parameter cap; ``update_host`` itself keeps its existing keyword-only
    signature (pre-existing debt, not widened here).
    """

    host_ref: str
    revision: int
    capacity: ResourceVector
    observed: ResourceVector
    heartbeat_at: datetime
    heartbeat_ttl_ms: int
    target_kind: str
    target_alias: str | None
    now: datetime | None
    draining: bool
    quarantined: bool
    labels: Sequence[str]
    disk_used_mib: int
    disk_capacity_mib: int


class NativeWorkItemReservationPort(WorkItemReservationPort):
    """Bind RMDD-08's synchronous port to the native reservation client."""

    def __init__(
        self,
        client: NativeReservationClient,
        *,
        tenant_ref: str,
        owner_id: str,
        fence_codec: NativeFenceCodec,
        profiles: ResourceProfileRegistry,
        capability_probe: Callable[[str], bool] | None = None,
        operation_key_factory: Callable[[str, str], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._client = client
        self._tenant_ref = _string(tenant_ref, name="tenant_ref")
        self._owner_id = _string(owner_id, name="owner_id")
        self._fence_codec = fence_codec
        self._profiles = profiles
        self._capability_probe = capability_probe or getattr(client, "supports", None)
        self._operation_key_factory = operation_key_factory or (
            lambda operation, reservation_id: (
                f"repository-manager:{operation}:v1:{reservation_id}:{uuid.uuid4().hex}"
            )
        )
        self._clock = clock or (lambda: datetime.now(UTC))
        self._capabilities_checked = False

    @classmethod
    def from_graph_client(
        cls,
        graph_client: object,
        *,
        tenant_ref: str,
        owner_id: str,
        fence_codec: NativeFenceCodec,
        profiles: ResourceProfileRegistry,
        operation_key_factory: Callable[[str, str], str] | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> NativeWorkItemReservationPort:
        """Construct from a synchronous engine client.

        ``profiles`` is the trusted RM ``ResourceProfileRegistry`` used to
        resolve the profile version sent as a mutation assertion.  Native
        reserve/release/reclaim independently re-read the immutable WorkItem
        declaration and reject drift in one transaction.
        """

        namespace = getattr(graph_client, "work_items", None)
        if namespace is None:
            raise NativeResourceReservationUnavailable(
                NativeResourceReservationUnavailable.code
            )
        probe = getattr(graph_client, "supports", None)
        if not callable(probe):
            # The real sync generated wrapper must expose the negotiated
            # capability probe explicitly.  A namespace-only or async client
            # is not a production binding and is rejected at construction.
            raise NativeResourceReservationUnavailable(
                NativeResourceReservationUnavailable.code
            )
        return cls(
            cast(NativeReservationClient, namespace),
            tenant_ref=tenant_ref,
            owner_id=owner_id,
            fence_codec=fence_codec,
            profiles=profiles,
            capability_probe=probe,
            operation_key_factory=operation_key_factory,
            clock=clock,
        )

    def _ensure_capabilities(self) -> None:
        if self._capabilities_checked:
            return
        if not callable(self._capability_probe):
            raise NativeResourceReservationUnavailable(
                NativeResourceReservationUnavailable.code
            )
        try:
            for operation in _OPERATION_NAMES:
                supported = _sync_value(
                    self._capability_probe(operation), name="capability probe"
                )
                if supported is not True:
                    raise NativeResourceReservationUnavailable(
                        NativeResourceReservationUnavailable.code
                    )
        except NativeResourceReservationUnavailable:
            raise
        except Exception as exc:
            raise NativeResourceReservationUnavailable(
                NativeResourceReservationUnavailable.code
            ) from exc
        self._capabilities_checked = True

    def _call(self, method: str, request: Mapping[str, object]) -> NativeResult:
        self._ensure_capabilities()
        function = getattr(self._client, method, None)
        if not callable(function):
            raise NativeResourceReservationUnavailable(
                NativeResourceReservationUnavailable.code
            )
        try:
            result = _sync_value(function(request), name="client result")
        except NativeResourceReservationUnavailable:
            raise
        except Exception as exc:
            if getattr(exc, "code", None) == NativeResourceReservationUnavailable.code:
                raise NativeResourceReservationUnavailable(
                    NativeResourceReservationUnavailable.code
                ) from exc
            raise
        return _mapping(result, name=f"native {method} result")

    def _now_ms(self, value: datetime | None = None) -> int:
        timestamp = value or self._clock()
        if timestamp.tzinfo is None or timestamp.utcoffset() is None:
            raise NativeReservationProtocolError("native clock must be timezone-aware")
        timestamp = timestamp.astimezone(UTC)
        milliseconds = int(timestamp.timestamp() * 1000)
        if milliseconds < 0:
            raise NativeReservationProtocolError(
                "native clock must not be before epoch"
            )
        return milliseconds

    def _fence_fields(
        self, fence: str, authority: Mapping[str, object] | None = None
    ) -> dict[str, object]:
        """Project a native lease pair, validating any authoritative read."""

        _string(fence, name="fence")
        if authority is None:
            try:
                encoded = self._fence_codec.encode(fence)
            except NativeResourceReservationUnavailable:
                raise
            except Exception as exc:
                raise NativeReservationProtocolError(
                    "native fence codec rejected the RM fence"
                ) from exc
            fields = _mapping(
                _sync_value(encoded, name="fence codec"), name="fence codec result"
            )
            lease_epoch = _integer(fields.get("lease_epoch"), name="lease_epoch")
            fencing_token = _integer(fields.get("fencing_token"), name="fencing_token")
        else:
            direct = _string(
                authority.get("fence", authority.get("lease_fence")),
                name="WorkItem fence",
            )
            if direct != fence:
                raise NativeReservationProtocolError("WorkItem fence mismatch")
            lease_epoch = _integer(authority.get("lease_epoch"), name="lease_epoch")
            fencing_token = _integer(
                authority.get("fencing_token"), name="fencing_token"
            )
        self._validate_fence(fence, lease_epoch, fencing_token)
        return {
            "fence": fence,
            "lease_epoch": lease_epoch,
            "fencing_token": fencing_token,
        }

    def _validate_fence(self, fence: str, lease_epoch: int, fencing_token: int) -> None:
        validator = getattr(self._fence_codec, "validate", None)
        if not callable(validator):
            raise NativeResourceReservationUnavailable(
                "native fence codec must validate the composite WorkItem fence"
            )
        try:
            result = validator(fence, lease_epoch, fencing_token)
        except NativeResourceReservationUnavailable:
            raise
        except Exception as exc:
            raise NativeReservationProtocolError(
                "native fence codec rejected the WorkItem fence"
            ) from exc
        _sync_value(result, name="fence codec")

    def _decode_fence(self, value: Mapping[str, object]) -> str:
        lease_epoch = _integer(value.get("lease_epoch"), name="result.lease_epoch")
        fencing_token = _integer(
            value.get("fencing_token"), name="result.fencing_token"
        )
        direct = _string(value.get("fence"), name="result.fence")
        self._validate_fence(direct, lease_epoch, fencing_token)
        return direct

    def _identity(self, record: ReservationRecord) -> tuple[str, str]:
        tenant = _string(record.tenant_id, name="reservation tenant_id")
        owner = _string(record.owner_id, name="reservation owner_id")
        if tenant != self._tenant_ref:
            raise NativeReservationProtocolError(
                "configured tenant does not match reservation tenant"
            )
        if owner != self._owner_id:
            raise NativeReservationProtocolError(
                "configured owner does not match reservation owner"
            )
        return tenant, owner

    @staticmethod
    def _host_revision(record: ReservationRecord) -> int | None:
        for name in ("host_revision", "version"):
            value = record.capacity_snapshot.get(name)
            if isinstance(value, int) and not isinstance(value, bool):
                return value
        return None

    def _trusted_profile(self, profile_name: str) -> Any:
        try:
            profile = self._profiles.resolve(profile_name)
        except Exception as exc:
            raise NativeResourceReservationUnavailable(
                f"trusted RM profile {profile_name!r} is unavailable"
            ) from exc
        version = getattr(profile, "profile_version", None)
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise NativeReservationProtocolError(
                "trusted RM profile version is invalid"
            )
        return profile

    def _validate_profile(self, record: ReservationRecord) -> str:
        """Assert that a resolved RM record did not under-declare its profile."""

        profile = self._trusted_profile(record.profile_name)
        self._validate_profile_resources(record, profile)
        self._validate_profile_policy_fields(record, profile)
        self._validate_profile_watermarks(record, profile)
        self._validate_profile_fairness_cost(record)
        return _string(str(profile.profile_version), name="profile_version")

    @staticmethod
    def _validate_profile_resources(record: ReservationRecord, profile: Any) -> None:
        for field in ("cpu_weight", "memory_mib", "disk_mib", "process_slots"):
            if getattr(record.requirement, field) < getattr(profile, field):
                raise NativeReservationProtocolError(
                    f"reservation requirement under-declares profile {field}"
                )

    @staticmethod
    def _validate_profile_policy_fields(
        record: ReservationRecord, profile: Any
    ) -> None:
        if record.concurrency_key != profile.concurrency_key:
            raise NativeReservationProtocolError(
                "reservation concurrency_key does not match trusted profile"
            )
        if record.concurrency_limit != profile.concurrency_limit:
            raise NativeReservationProtocolError(
                "reservation concurrency_limit does not match trusted profile"
            )
        if record.repository_exclusive != profile.repository_exclusive:
            raise NativeReservationProtocolError(
                "reservation repository_exclusive does not match trusted profile"
            )
        if record.branch_exclusive != profile.branch_exclusive:
            raise NativeReservationProtocolError(
                "reservation branch_exclusive does not match trusted profile"
            )
        if not set(profile.required_labels).issubset(record.required_labels):
            raise NativeReservationProtocolError(
                "reservation required_labels under-declare trusted profile"
            )
        if not set(profile.anti_affinity).issubset(record.anti_affinity):
            raise NativeReservationProtocolError(
                "reservation anti_affinity under-declares trusted profile"
            )

    @staticmethod
    def _validate_profile_watermarks(record: ReservationRecord, profile: Any) -> None:
        for field in ("disk_low_watermark_mib", "disk_high_watermark_mib"):
            profile_value = getattr(profile, field)
            record_value = getattr(record, field)
            if profile_value is not None and (
                record_value is None or record_value > profile_value
            ):
                raise NativeReservationProtocolError(
                    f"reservation {field} weakens trusted profile"
                )
        expected_policy_key = f"{profile.name}:v{profile.profile_version}"
        if record.disk_policy_key != expected_policy_key:
            raise NativeReservationProtocolError(
                "reservation disk_policy_key does not match trusted profile"
            )

    @staticmethod
    def _validate_profile_fairness_cost(record: ReservationRecord) -> None:
        expected_fairness_cost = max(
            1, record.requirement.cpu_weight + record.requirement.process_slots
        )
        if record.fairness_cost != expected_fairness_cost:
            raise NativeReservationProtocolError(
                "reservation fairness_cost does not match RMDD-08 rule"
            )

    def _request(
        self,
        record: ReservationRecord,
        *,
        now: datetime | None,
        operation: str,
    ) -> dict[str, object]:
        tenant, owner = self._identity(record)
        host_revision = self._host_revision(record)
        reserved_at_ms = self._now_ms(record.reserved_at)
        expires_at_ms = self._now_ms(record.expires_at)
        requirement = self._validate_reservation_requirement(record)
        required_labels, anti_affinity = self._validate_reservation_labels(record)
        target_kind, target_alias = _target_fields(
            record.selected_target.kind.value,
            record.selected_target.alias,
            name="reservation",
        )
        profile_version = self._validate_profile(record)
        input_fingerprint = self._validate_reservation_fingerprint(record)
        self._validate_reservation_concurrency_fields(record)
        self._validate_reservation_watermarks(record)
        ctx = _ReservationRequestContext(
            tenant=tenant,
            owner=owner,
            requirement=requirement,
            target_kind=target_kind,
            target_alias=target_alias,
            profile_version=profile_version,
            input_fingerprint=input_fingerprint,
            required_labels=required_labels,
            anti_affinity=anti_affinity,
            reserved_at_ms=reserved_at_ms,
            expires_at_ms=expires_at_ms,
            host_revision=host_revision,
            now=now,
            operation=operation,
        )
        payload = self._build_reservation_payload(record, ctx)
        if expires_at_ms <= reserved_at_ms:
            raise NativeReservationProtocolError(
                "reservation expiry must be after start"
            )
        return payload

    @staticmethod
    def _validate_reservation_requirement(
        record: ReservationRecord,
    ) -> dict[str, int]:
        requirement = record.requirement.as_dict()
        for field, requirement_value in requirement.items():
            _integer(
                requirement_value,
                name=f"reservation.requirement.{field}",
                minimum=1,
            )
        return requirement

    @staticmethod
    def _validate_reservation_labels(
        record: ReservationRecord,
    ) -> tuple[list[str], list[str]]:
        required_labels = [
            _string(item, name="reservation.required_label")
            for item in record.required_labels
        ]
        anti_affinity = [
            _string(item, name="reservation.anti_affinity")
            for item in record.anti_affinity
        ]
        if len(required_labels) > 128 or len(required_labels) != len(
            set(required_labels)
        ):
            raise NativeReservationProtocolError(
                "reservation required_labels must be unique and bounded"
            )
        if len(anti_affinity) > 128 or len(anti_affinity) != len(set(anti_affinity)):
            raise NativeReservationProtocolError(
                "reservation anti_affinity must be unique and bounded"
            )
        return required_labels, anti_affinity

    @staticmethod
    def _validate_reservation_fingerprint(record: ReservationRecord) -> str:
        input_fingerprint = _string(
            record.input_fingerprint, name="reservation.input_fingerprint"
        )
        if len(input_fingerprint) != 67 or not input_fingerprint.startswith("v1:"):
            raise NativeReservationProtocolError(
                "reservation.input_fingerprint must use v1 namespace"
            )
        return input_fingerprint

    @staticmethod
    def _validate_reservation_concurrency_fields(record: ReservationRecord) -> None:
        if record.concurrency_limit is not None:
            _integer(
                record.concurrency_limit,
                name="reservation.concurrency_limit",
                minimum=1,
            )
        _boolean(record.repository_exclusive, name="reservation.repository_exclusive")
        _boolean(record.branch_exclusive, name="reservation.branch_exclusive")
        _integer(record.fairness_cost, name="reservation.fairness_cost", minimum=1)

    @staticmethod
    def _validate_reservation_watermarks(record: ReservationRecord) -> None:
        for field, watermark_value in (
            ("disk_low_watermark_mib", record.disk_low_watermark_mib),
            ("disk_high_watermark_mib", record.disk_high_watermark_mib),
        ):
            if watermark_value is not None:
                _integer(watermark_value, name=f"reservation.{field}")
        if (
            record.disk_low_watermark_mib is not None
            and record.disk_high_watermark_mib is not None
            and record.disk_low_watermark_mib > record.disk_high_watermark_mib
        ):
            raise NativeReservationProtocolError(
                "reservation disk low watermark exceeds high watermark"
            )

    def _build_reservation_payload(
        self, record: ReservationRecord, ctx: _ReservationRequestContext
    ) -> dict[str, object]:
        return {
            "schema_version": "1",
            "tenant_ref": ctx.tenant,
            "work_item_id": record.work_item_id,
            "owner_id": ctx.owner,
            **self._fence_fields(record.fence),
            "attempt": record.attempt,
            "reservation_id": record.reservation_id,
            "input_fingerprint": ctx.input_fingerprint,
            "profile_name": _string(
                record.profile_name, name="reservation.profile_name"
            ),
            "profile_version": ctx.profile_version,
            "host_ref": record.host_id,
            "requirement": ctx.requirement,
            "target_kind": ctx.target_kind,
            "target_alias": ctx.target_alias,
            "repository_id": _string(
                record.repository_id, name="reservation.repository_id"
            ),
            "branch": _string(record.branch, name="reservation.branch"),
            "concurrency_key": _string(
                record.concurrency_key, name="reservation.concurrency_key"
            ),
            "concurrency_limit": record.concurrency_limit,
            "repository_exclusive": record.repository_exclusive,
            "branch_exclusive": record.branch_exclusive,
            "required_labels": ctx.required_labels,
            "anti_affinity": ctx.anti_affinity,
            "fairness_group": _string(
                record.fairness_group, name="reservation.fairness_group"
            ),
            "fairness_cost": record.fairness_cost,
            "disk_low_watermark_mib": record.disk_low_watermark_mib,
            "disk_high_watermark_mib": record.disk_high_watermark_mib,
            "disk_policy_key": _string(
                record.disk_policy_key, name="reservation.disk_policy_key"
            ),
            "reserved_at_ms": ctx.reserved_at_ms,
            "expires_at_ms": ctx.expires_at_ms,
            "idempotency_key": _string(
                self._operation_key_factory(ctx.operation, record.reservation_id),
                name="idempotency_key",
            ),
            "now_ms": self._now_ms(ctx.now),
            "expected_host_revision": ctx.host_revision,
            "expected_lifecycle_revision": record.revision,
        }

    def _status_request(
        self,
        *,
        work_item_id: str | None,
        reservation_id: str | None,
        host_ref: str | None,
        owner_id: str | None,
        fence: str | None,
        attempt: int | None,
        lease_epoch: int | None,
        fencing_token: int | None,
        input_fingerprint: str | None,
        limit: int,
        cursor: str | None,
        now: datetime | None,
    ) -> dict[str, object]:
        if limit < 1 or limit > 1000:
            raise ValueError("native status limit must be between 1 and 1000")
        self._validate_status_request_filters(
            _StatusRequestFilters(
                work_item_id=work_item_id,
                reservation_id=reservation_id,
                host_ref=host_ref,
                owner_id=owner_id,
                fence=fence,
                attempt=attempt,
                lease_epoch=lease_epoch,
                fencing_token=fencing_token,
                input_fingerprint=input_fingerprint,
                cursor=cursor,
            )
        )
        return {
            "schema_version": "1",
            "tenant_ref": self._tenant_ref,
            "work_item_id": work_item_id,
            "reservation_id": reservation_id,
            "host_ref": host_ref,
            "owner_id": owner_id,
            "fence": fence,
            "attempt": attempt,
            "lease_epoch": lease_epoch,
            "fencing_token": fencing_token,
            "input_fingerprint": input_fingerprint,
            "limit": limit,
            "cursor": cursor,
            "now_ms": self._now_ms(now),
        }

    @staticmethod
    def _validate_status_string_filters(filters: _StatusRequestFilters) -> None:
        for field, value in (
            ("work_item_id", filters.work_item_id),
            ("reservation_id", filters.reservation_id),
            ("host_ref", filters.host_ref),
            ("owner_id", filters.owner_id),
            ("fence", filters.fence),
            ("cursor", filters.cursor),
        ):
            if value is not None:
                _string(value, name=f"status.{field}")
        if filters.attempt is not None:
            _integer(filters.attempt, name="status.attempt", minimum=1)
        if filters.lease_epoch is not None:
            _integer(filters.lease_epoch, name="status.lease_epoch")
        if filters.fencing_token is not None:
            _integer(filters.fencing_token, name="status.fencing_token")

    @staticmethod
    def _validate_status_fingerprint_filter(filters: _StatusRequestFilters) -> None:
        if filters.input_fingerprint is None:
            return
        fingerprint = _string(
            filters.input_fingerprint, name="status.input_fingerprint"
        )
        if len(fingerprint) != 67 or not fingerprint.startswith("v1:"):
            raise NativeReservationProtocolError(
                "status.input_fingerprint must use v1 namespace"
            )

    @staticmethod
    def _validate_status_request_filters(filters: _StatusRequestFilters) -> None:
        NativeWorkItemReservationPort._validate_status_string_filters(filters)
        NativeWorkItemReservationPort._validate_status_fingerprint_filter(filters)
        if (
            filters.work_item_id is None
            and filters.reservation_id is None
            and filters.host_ref is None
        ):
            raise ValueError(
                "native status requires a WorkItem, reservation, or host filter"
            )

    def _validate_result_top_level(
        self, value: NativeResult, *, work_item_id: str, attempt: int, fence: str
    ) -> tuple[FenceDecision, int, int]:
        if value.get("schema_version") != "1":
            raise NativeReservationProtocolError(
                "native result schema_version must be 1"
            )
        decision_value = _string(value.get("decision"), name="result.decision")
        decision = _DECISIONS.get(decision_value)
        if decision is None:
            raise NativeReservationProtocolError("native result decision is unknown")
        if (
            _string(value.get("work_item_id"), name="result.work_item_id")
            != work_item_id
        ):
            raise NativeReservationProtocolError(
                "native result WorkItem correlation mismatch"
            )
        if _integer(value.get("attempt"), name="result.attempt", minimum=1) != attempt:
            raise NativeReservationProtocolError("native result attempt mismatch")
        result_epoch = _integer(value.get("lease_epoch"), name="result.lease_epoch")
        result_token = _integer(value.get("fencing_token"), name="result.fencing_token")
        direct_fence = value.get("fence")
        if direct_fence is not None:
            direct_fence = _string(direct_fence, name="result.fence")
            self._validate_fence(direct_fence, result_epoch, result_token)
            if direct_fence != fence:
                raise NativeReservationProtocolError("native result fence mismatch")
        return decision, result_epoch, result_token

    @staticmethod
    def _validate_result_state_fields(value: NativeResult) -> tuple[str, bool]:
        top_state = _string(value.get("state"), name="result.state")
        if top_state not in {
            "reserved",
            "released",
            "reclaimed",
            "expired",
            "superseded",
            "absent",
        }:
            raise NativeReservationProtocolError("native result state is unknown")
        top_tombstone = _boolean(value.get("tombstone"), name="result.tombstone")
        if top_tombstone != (
            top_state in {"released", "reclaimed", "expired", "superseded"}
        ):
            raise NativeReservationProtocolError(
                "native result state/tombstone mismatch"
            )
        for field in (
            "lifecycle_revision",
            "host_revision",
            "held_cpu_weight",
            "held_memory_mib",
            "held_disk_mib",
            "held_process_slots",
            "fairness_debt",
        ):
            _integer(value.get(field), name=f"result.{field}")
        return top_state, top_tombstone

    @staticmethod
    def _validate_result_reservation_and_changed(
        value: NativeResult,
        *,
        decision: FenceDecision,
        reservation_id: str,
        work_item_id: str,
        require_changed_id: bool,
    ) -> None:
        result_reservation_id = value.get("reservation_id")
        if (
            decision in {FenceDecision.ACCEPTED, FenceDecision.IDEMPOTENT}
            and result_reservation_id is None
        ):
            raise NativeReservationProtocolError(
                "accepted native result lacks reservation identity"
            )
        if (
            result_reservation_id is not None
            and _string(result_reservation_id, name="result.reservation_id")
            != reservation_id
        ):
            raise NativeReservationProtocolError("native result reservation mismatch")
        changed = _list(
            value.get("changed_work_item_ids"), name="result.changed_work_item_ids"
        )
        changed_strings = [
            _string(item, name="changed_work_item_id") for item in changed
        ]
        if len(changed_strings) != len(set(changed_strings)):
            raise NativeReservationProtocolError("native changed ids must be unique")
        if (
            require_changed_id
            and decision is FenceDecision.ACCEPTED
            and work_item_id not in changed_strings
        ):
            raise NativeReservationProtocolError(
                "native result omitted changed WorkItem"
            )

    @staticmethod
    def _validate_missing_record(
        *,
        decision: FenceDecision,
        expected_fence: tuple[int, int] | None,
        result_epoch: int,
        result_token: int,
    ) -> None:
        if (
            expected_fence is not None
            and (result_epoch, result_token) != expected_fence
        ):
            raise NativeReservationProtocolError(
                "native result fence correlation mismatch"
            )
        if decision is FenceDecision.NOT_FOUND and expected_fence is None:
            raise NativeReservationProtocolError(
                "native not-found fence correlation mismatch"
            )

    def _parse_result_record(
        self,
        record_value: object,
        *,
        result_epoch: int,
        result_token: int,
    ) -> tuple[ReservationRecord, Mapping[str, object]]:
        record_payload = _mapping(record_value, name="native reservation record")
        record = self._record_from_native(record_payload)
        record_epoch = _integer(
            record_payload.get("lease_epoch"), name="record.lease_epoch"
        )
        record_token = _integer(
            record_payload.get("fencing_token"), name="record.fencing_token"
        )
        if (record_epoch, record_token) != (result_epoch, result_token):
            raise NativeReservationProtocolError(
                "native result and reservation fence numbers disagree"
            )
        return record, record_payload

    @staticmethod
    def _validate_expected_admission(
        record_payload: Mapping[str, object],
        expected_admission: Mapping[str, object],
    ) -> None:
        for field in (
            "profile_name",
            "profile_version",
            "requirement",
            "target_kind",
            "target_alias",
            "repository_id",
            "branch",
            "concurrency_key",
            "concurrency_limit",
            "repository_exclusive",
            "branch_exclusive",
            "required_labels",
            "anti_affinity",
            "fairness_group",
            "fairness_cost",
            "disk_low_watermark_mib",
            "disk_high_watermark_mib",
            "disk_policy_key",
            "input_fingerprint",
        ):
            expected_value = expected_admission.get(field)
            native_value = record_payload.get(field)
            if field in {"required_labels", "anti_affinity"}:
                expected_value = list(cast(Sequence[object], expected_value or ()))
            if expected_value != native_value:
                raise NativeReservationProtocolError(
                    f"native immutable admission field mismatch: {field}"
                )

    def _validate_record_correlation(
        self,
        record: ReservationRecord,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
    ) -> None:
        if (record.work_item_id, record.attempt, record.fence) != (
            work_item_id,
            attempt,
            fence,
        ):
            raise NativeReservationProtocolError(
                "native reservation correlation mismatch"
            )
        if record.reservation_id != reservation_id:
            raise NativeReservationProtocolError("native reservation identity mismatch")
        if record.tenant_id != self._tenant_ref:
            raise NativeReservationProtocolError("native reservation tenant mismatch")
        if self._owner_id and record.owner_id != self._owner_id:
            raise NativeReservationProtocolError("native reservation owner mismatch")

    @staticmethod
    def _validate_record_state_consistency(
        value: NativeResult,
        record: ReservationRecord,
        record_payload: Mapping[str, object],
        *,
        top_state: str,
        top_tombstone: bool,
    ) -> None:
        if top_state != record_payload.get("state"):
            raise NativeReservationProtocolError("native result state mismatch")
        if top_tombstone != (record.state is not ReservationState.RESERVED):
            raise NativeReservationProtocolError("native result tombstone mismatch")
        if _boolean(record_payload.get("tombstone"), name="record.tombstone") != (
            record.state is not ReservationState.RESERVED
        ):
            raise NativeReservationProtocolError("native record tombstone mismatch")
        top_revision = _integer(
            value.get("lifecycle_revision"), name="result.lifecycle_revision"
        )
        native_lifecycle = _integer(
            record_payload.get("lifecycle_revision"),
            name="record.lifecycle_revision",
        )
        if top_revision != native_lifecycle:
            raise NativeReservationProtocolError("native lifecycle revision mismatch")

    def _validate_host_correlation(
        self, value: NativeResult, record: ReservationRecord
    ) -> None:
        host_ref = value.get("host_ref")
        if host_ref is None:
            raise NativeReservationProtocolError("native result lacks host correlation")
        if _string(host_ref, name="result.host_ref") != record.host_id:
            raise NativeReservationProtocolError("native host correlation mismatch")
        host_revision = _integer(
            value.get("host_revision"), name="result.host_revision"
        )
        if host_revision != self._host_revision(record):
            raise NativeReservationProtocolError("native host revision mismatch")

    def _result(
        self,
        value: NativeResult,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
        expected: ReservationRecord | None,
        require_changed_id: bool,
        expected_admission: Mapping[str, object] | None = None,
        expected_fence: tuple[int, int] | None = None,
    ) -> tuple[FenceDecision, ReservationRecord | None]:
        decision, result_epoch, result_token = self._validate_result_top_level(
            value, work_item_id=work_item_id, attempt=attempt, fence=fence
        )
        top_state, top_tombstone = self._validate_result_state_fields(value)
        self._validate_result_reservation_and_changed(
            value,
            decision=decision,
            reservation_id=reservation_id,
            work_item_id=work_item_id,
            require_changed_id=require_changed_id,
        )
        record_value = value.get("record")
        if record_value is None:
            self._validate_missing_record(
                decision=decision,
                expected_fence=expected_fence,
                result_epoch=result_epoch,
                result_token=result_token,
            )
            return decision, None
        record, record_payload = self._parse_result_record(
            record_value, result_epoch=result_epoch, result_token=result_token
        )
        if expected_admission is not None:
            self._validate_expected_admission(record_payload, expected_admission)
        self._validate_record_correlation(
            record,
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reservation_id=reservation_id,
        )
        self._validate_record_state_consistency(
            value,
            record,
            record_payload,
            top_state=top_state,
            top_tombstone=top_tombstone,
        )
        self._validate_host_correlation(value, record)
        if expected is not None and not self._same_immutable(expected, record):
            raise NativeReservationProtocolError(
                "native immutable reservation mismatch"
            )
        if expected is not None and not self._same_selected_target(expected, record):
            raise NativeReservationProtocolError(
                "native selected target observation mismatch"
            )
        return decision, record

    @staticmethod
    def _same_immutable(expected: ReservationRecord, native: ReservationRecord) -> bool:
        """Compare immutable target/policy values without rewriting observations.

        ``selected_target.capability_labels`` is a host observation captured
        by RM's fit/rank policy.  Native ``required_labels`` are immutable
        admission policy, so comparing the two as if they were the same field
        was both lossy and silently changed the caller's expected value.
        """

        if expected.input_fingerprint or native.input_fingerprint:
            if expected.input_fingerprint != native.input_fingerprint:
                return False
        return (
            expected.reservation_id,
            expected.work_item_id,
            expected.attempt,
            expected.fence,
            expected.host_id,
            expected.profile_name,
            expected.requirement,
            expected.selected_target.kind,
            expected.selected_target.alias,
            expected.concurrency_key,
            expected.concurrency_limit,
            expected.repository_exclusive,
            expected.branch_exclusive,
            expected.required_labels,
            expected.disk_low_watermark_mib,
            expected.disk_high_watermark_mib,
            expected.disk_policy_key,
            expected.repository_id,
            expected.branch,
            expected.owner_id,
            expected.tenant_id,
            expected.fairness_group,
            expected.fairness_cost,
            expected.anti_affinity,
            expected.expires_at - expected.reserved_at,
        ) == (
            native.reservation_id,
            native.work_item_id,
            native.attempt,
            native.fence,
            native.host_id,
            native.profile_name,
            native.requirement,
            native.selected_target.kind,
            native.selected_target.alias,
            native.concurrency_key,
            native.concurrency_limit,
            native.repository_exclusive,
            native.branch_exclusive,
            native.required_labels,
            native.disk_low_watermark_mib,
            native.disk_high_watermark_mib,
            native.disk_policy_key,
            native.repository_id,
            native.branch,
            native.owner_id,
            native.tenant_id,
            native.fairness_group,
            native.fairness_cost,
            native.anti_affinity,
            native.expires_at - native.reserved_at,
        )

    @staticmethod
    def _same_selected_target(
        expected: ReservationRecord, native: ReservationRecord
    ) -> bool:
        """Compare native placement observations without treating labels as policy."""

        return (
            expected.selected_target.kind == native.selected_target.kind
            and expected.selected_target.alias == native.selected_target.alias
            and expected.selected_target.capability_labels
            == native.selected_target.capability_labels
        )

    @staticmethod
    def _parse_native_requirement(value: Mapping[str, object]) -> ResourceVector:
        requirement_value = _mapping(
            value.get("requirement"), name="record.requirement"
        )
        return ResourceVector(
            cpu_weight=_integer(
                requirement_value.get("cpu_weight"),
                name="requirement.cpu_weight",
                minimum=1,
            ),
            memory_mib=_integer(
                requirement_value.get("memory_mib"),
                name="requirement.memory_mib",
                minimum=1,
            ),
            disk_mib=_integer(
                requirement_value.get("disk_mib"),
                name="requirement.disk_mib",
                minimum=1,
            ),
            process_slots=_integer(
                requirement_value.get("process_slots"),
                name="requirement.process_slots",
                minimum=1,
            ),
        )

    @staticmethod
    def _parse_native_capacity_snapshot(
        value: Mapping[str, object],
    ) -> dict[str, int]:
        snapshot_value = _mapping(
            value.get("capacity_snapshot"), name="record.capacity_snapshot"
        )
        return {
            "cpu_weight": _integer(
                snapshot_value.get("cpu_weight"), name="capacity_snapshot.cpu_weight"
            ),
            "memory_mib": _integer(
                snapshot_value.get("memory_mib"), name="capacity_snapshot.memory_mib"
            ),
            "disk_mib": _integer(
                snapshot_value.get("disk_mib"), name="capacity_snapshot.disk_mib"
            ),
            "process_slots": _integer(
                snapshot_value.get("process_slots"),
                name="capacity_snapshot.process_slots",
            ),
            "host_revision": _integer(
                snapshot_value.get("host_revision"),
                name="capacity_snapshot.host_revision",
            ),
        }

    @staticmethod
    def _parse_native_selected_target(value: Mapping[str, object]) -> TargetPolicy:
        _target_fields(
            value.get("target_kind"),
            value.get("target_alias"),
            name="record",
        )
        selected_target_value = _mapping(
            value.get("selected_target"), name="record.selected_target"
        )
        selected_kind_value, selected_alias = _target_fields(
            selected_target_value.get("kind"),
            selected_target_value.get("alias"),
            name="record.selected_target",
        )
        selected_labels = tuple(
            _string(item, name="record.selected_target.capability_label")
            for item in _list(
                selected_target_value.get("capability_labels"),
                name="record.selected_target.capability_labels",
            )
        )
        if len(selected_labels) != len(set(selected_labels)):
            raise NativeReservationProtocolError(
                "native selected target capability labels must be unique"
            )
        try:
            target_kind = TargetKind(selected_kind_value)
            return TargetPolicy(
                kind=target_kind,
                alias=selected_alias,
                # Native v1 carries selected-host capability observations in
                # this nested snapshot; keep them distinct from required
                # admission labels below.
                capability_labels=selected_labels,
            )
        except ValueError as exc:
            raise NativeReservationProtocolError(
                "native target policy is invalid"
            ) from exc

    @staticmethod
    def _parse_native_timestamps(
        value: Mapping[str, object],
    ) -> tuple[datetime, datetime]:
        reserved_ms = _integer(
            value.get("reserved_at_ms"), name="record.reserved_at_ms"
        )
        expires_ms = _integer(
            value.get("expires_at_ms"), name="record.expires_at_ms", minimum=1
        )
        try:
            reserved_at = datetime.fromtimestamp(reserved_ms / 1000, UTC)
            expires_at = datetime.fromtimestamp(expires_ms / 1000, UTC)
        except (OverflowError, OSError, ValueError) as exc:
            raise NativeReservationProtocolError(
                "native reservation timestamp is invalid"
            ) from exc
        if expires_at <= reserved_at:
            raise NativeReservationProtocolError("native reservation expiry is invalid")
        return reserved_at, expires_at

    @staticmethod
    def _parse_native_labels(
        value: Mapping[str, object],
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        labels = tuple(
            _string(item, name="record.required_label")
            for item in _list(
                value.get("required_labels"), name="record.required_labels"
            )
        )
        anti_affinity = tuple(
            _string(item, name="record.anti_affinity")
            for item in _list(value.get("anti_affinity"), name="record.anti_affinity")
        )
        if len(labels) != len(set(labels)):
            raise NativeReservationProtocolError(
                "native record required_labels must be unique"
            )
        if len(anti_affinity) != len(set(anti_affinity)):
            raise NativeReservationProtocolError(
                "native record anti_affinity must be unique"
            )
        return labels, anti_affinity

    def _record_from_native(self, value: Mapping[str, object]) -> ReservationRecord:
        reservation_id = _string(
            value.get("reservation_id"), name="record.reservation_id"
        )
        work_item_id = _string(value.get("work_item_id"), name="record.work_item_id")
        attempt = _integer(value.get("attempt"), name="record.attempt", minimum=1)
        fence = self._decode_fence(value)
        requirement = self._parse_native_requirement(value)
        snapshot = self._parse_native_capacity_snapshot(value)
        selected_target = self._parse_native_selected_target(value)
        state_value = _string(value.get("state"), name="record.state")
        if state_value not in _STATE_MAP:
            raise NativeReservationProtocolError("native reservation state is unknown")
        reserved_at, expires_at = self._parse_native_timestamps(value)
        labels, anti_affinity = self._parse_native_labels(value)
        # The request target is an immutable policy assertion; the nested
        # selected_target is the native host-selection observation.  Both are
        # parsed above, while RM's ReservationRecord retains the latter.
        return ReservationRecord(
            reservation_id=reservation_id,
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            host_id=_string(value.get("host_ref"), name="record.host_ref"),
            profile_name=_string(value.get("profile_name"), name="record.profile_name"),
            requirement=requirement,
            capacity_snapshot=snapshot,
            selected_target=selected_target,
            concurrency_key=_string(
                value.get("concurrency_key"), name="record.concurrency_key"
            ),
            concurrency_limit=(
                _integer(
                    value.get("concurrency_limit"),
                    name="record.concurrency_limit",
                    minimum=1,
                )
                if value.get("concurrency_limit") is not None
                else None
            ),
            repository_exclusive=_boolean(
                value.get("repository_exclusive"), name="record.repository_exclusive"
            ),
            branch_exclusive=_boolean(
                value.get("branch_exclusive"), name="record.branch_exclusive"
            ),
            required_labels=labels,
            disk_low_watermark_mib=(
                _integer(
                    value.get("disk_low_watermark_mib"),
                    name="record.disk_low_watermark_mib",
                )
                if value.get("disk_low_watermark_mib") is not None
                else None
            ),
            disk_high_watermark_mib=(
                _integer(
                    value.get("disk_high_watermark_mib"),
                    name="record.disk_high_watermark_mib",
                )
                if value.get("disk_high_watermark_mib") is not None
                else None
            ),
            disk_policy_key=_string(
                value.get("disk_policy_key"), name="record.disk_policy_key"
            ),
            repository_id=_string(
                value.get("repository_id"), name="record.repository_id"
            ),
            branch=_string(value.get("branch"), name="record.branch"),
            owner_id=_string(value.get("owner_id"), name="record.owner_id"),
            tenant_id=_string(value.get("tenant_ref"), name="record.tenant_ref"),
            fairness_group=_string(
                value.get("fairness_group"), name="record.fairness_group"
            ),
            fairness_cost=_integer(
                value.get("fairness_cost"), name="record.fairness_cost", minimum=1
            ),
            anti_affinity=anti_affinity,
            reserved_at=reserved_at,
            expires_at=expires_at,
            state=_STATE_MAP[state_value],
            revision=_integer(value.get("revision"), name="record.revision", minimum=1),
            input_fingerprint=_string(
                value.get("input_fingerprint"), name="record.input_fingerprint"
            ),
        )

    def _query_current_work_item(
        self, *, work_item_id: str, attempt: int, fence: str
    ) -> FenceDecision:
        fence_fields = self._fence_fields(fence)
        request = self._status_request(
            work_item_id=work_item_id,
            reservation_id=None,
            host_ref=None,
            owner_id=self._owner_id,
            fence=fence,
            attempt=attempt,
            lease_epoch=cast(int, fence_fields["lease_epoch"]),
            fencing_token=cast(int, fence_fields["fencing_token"]),
            input_fingerprint=None,
            limit=1,
            cursor=None,
            now=None,
        )
        result = self._call("query_reservation", request)
        decision = self._validate_current_result_correlation(
            result,
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            fence_fields=fence_fields,
        )
        tombstone = self._resolve_current_tombstone(
            result, work_item_id=work_item_id, attempt=attempt, fence=fence
        )
        if tombstone:
            return FenceDecision.STALE
        return decision

    def _validate_current_result_correlation(
        self,
        result: NativeResult,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        fence_fields: Mapping[str, object],
    ) -> FenceDecision:
        if result.get("schema_version") != "1":
            raise NativeReservationProtocolError(
                "native current WorkItem schema_version must be 1"
            )
        decision_value = _string(result.get("decision"), name="result.decision")
        decision = _DECISIONS.get(decision_value)
        if decision is None:
            raise NativeReservationProtocolError("native current decision is unknown")
        if (
            _string(result.get("work_item_id"), name="result.work_item_id")
            != work_item_id
        ):
            raise NativeReservationProtocolError("native current WorkItem mismatch")
        if _integer(result.get("attempt"), name="result.attempt", minimum=1) != attempt:
            raise NativeReservationProtocolError("native current attempt mismatch")
        result_pair = (
            _integer(result.get("lease_epoch"), name="result.lease_epoch"),
            _integer(result.get("fencing_token"), name="result.fencing_token"),
        )
        if result_pair != (
            cast(int, fence_fields["lease_epoch"]),
            cast(int, fence_fields["fencing_token"]),
        ):
            raise NativeReservationProtocolError("native current fence mismatch")
        direct = result.get("fence")
        if direct is not None:
            direct_value = _string(direct, name="result.fence")
            self._validate_fence(direct_value, *result_pair)
            if direct_value != fence:
                raise NativeReservationProtocolError("native current fence mismatch")
        return decision

    def _resolve_current_tombstone(
        self, result: NativeResult, *, work_item_id: str, attempt: int, fence: str
    ) -> bool:
        tombstone = _boolean(result.get("tombstone"), name="result.tombstone")
        record_value = result.get("record")
        if record_value is None:
            return tombstone
        record_payload = _mapping(record_value, name="native current record")
        record = self._record_from_native(record_payload)
        if (
            record.work_item_id,
            record.attempt,
            record.fence,
            record.tenant_id,
            record.owner_id,
        ) != (work_item_id, attempt, fence, self._tenant_ref, self._owner_id):
            raise NativeReservationProtocolError(
                "native current record correlation mismatch"
            )
        return record.state is not ReservationState.RESERVED

    def is_current(self, work_item_id: str, attempt: int, fence: str) -> bool:
        try:
            decision = self._query_current_work_item(
                work_item_id=work_item_id, attempt=attempt, fence=fence
            )
        except NativeResourceReservationUnavailable:
            raise
        except NativeReservationProtocolError:
            return False
        return decision in {FenceDecision.ACCEPTED, FenceDecision.IDEMPOTENT}

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
        del (
            controller
        )  # Native authority, not a local controller flag, decides reclaim.
        if expected is not None and (
            expected.reservation_id,
            expected.work_item_id,
            expected.attempt,
            expected.fence,
        ) != (reservation_id, work_item_id, attempt, fence):
            return FenceDecision.CONFLICT
        if expected is not None:
            self._identity(expected)
        fence_fields = self._fence_fields(fence)
        expected_payload = (
            self._request(expected, now=self._clock(), operation="query")
            if expected is not None
            else None
        )
        request = self._status_request(
            work_item_id=work_item_id,
            reservation_id=reservation_id,
            host_ref=None,
            owner_id=self._owner_id,
            fence=fence,
            attempt=attempt,
            lease_epoch=cast(int, fence_fields["lease_epoch"]),
            fencing_token=cast(int, fence_fields["fencing_token"]),
            input_fingerprint=(
                expected.input_fingerprint if expected is not None else None
            ),
            limit=1,
            cursor=None,
            now=None,
        )
        decision, record = self._result(
            self._call("query_reservation", request),
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reservation_id=reservation_id,
            expected=expected,
            require_changed_id=False,
            expected_admission=expected_payload,
            expected_fence=(
                cast(int, fence_fields["lease_epoch"]),
                cast(int, fence_fields["fencing_token"]),
            ),
        )
        if record is None:
            return decision
        if not for_lifecycle and record.state is not ReservationState.RESERVED:
            return FenceDecision.CONFLICT
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
        del expected_capacity
        if (reservation.work_item_id, reservation.attempt, reservation.fence) != (
            work_item_id,
            attempt,
            fence,
        ):
            return FenceDecision.CONFLICT
        request = self._request(reservation, now=self._clock(), operation="reserve")
        decision, record = self._result(
            self._call("reserve", request),
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reservation_id=reservation.reservation_id,
            expected=reservation,
            require_changed_id=True,
            expected_admission=request,
            expected_fence=(
                cast(int, request["lease_epoch"]),
                cast(int, request["fencing_token"]),
            ),
        )
        if decision in {FenceDecision.ACCEPTED, FenceDecision.IDEMPOTENT} and (
            record is None or record.state is not ReservationState.RESERVED
        ):
            raise NativeReservationProtocolError(
                "native admission did not return an active record"
            )
        return decision

    def atomic_release(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        reservation_id: str,
        reservation: ReservationRecord | None = None,
    ) -> FenceDecision:
        if reservation is None:
            raise NativeReservationProtocolError(
                "release requires the immutable reservation"
            )
        if reservation.reservation_id != reservation_id:
            return FenceDecision.CONFLICT
        request = self._request(reservation, now=self._clock(), operation="release")
        decision, record = self._result(
            self._call("release", request),
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reservation_id=reservation_id,
            expected=reservation,
            require_changed_id=True,
            expected_admission=request,
            expected_fence=(
                cast(int, request["lease_epoch"]),
                cast(int, request["fencing_token"]),
            ),
        )
        if decision in {FenceDecision.ACCEPTED, FenceDecision.IDEMPOTENT} and (
            record is None or record.state is not ReservationState.RELEASED
        ):
            raise NativeReservationProtocolError(
                "native release did not return its tombstone"
            )
        return decision

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
        if reservation is None:
            raise NativeReservationProtocolError(
                "reclaim requires the immutable reservation"
            )
        if reservation.reservation_id != reservation_id:
            return FenceDecision.CONFLICT
        request = self._request(reservation, now=now, operation="reclaim")
        decision, record = self._result(
            self._call("reclaim", request),
            work_item_id=work_item_id,
            attempt=attempt,
            fence=fence,
            reservation_id=reservation_id,
            expected=reservation,
            require_changed_id=True,
            expected_admission=request,
            expected_fence=(
                cast(int, request["lease_epoch"]),
                cast(int, request["fencing_token"]),
            ),
        )
        if decision in {FenceDecision.ACCEPTED, FenceDecision.IDEMPOTENT} and (
            record is None or record.state is not ReservationState.EXPIRED
        ):
            raise NativeReservationProtocolError(
                "native reclaim did not return its tombstone"
            )
        return decision

    def atomic_reclaim_expired(
        self, *, reservation: ReservationRecord, now: datetime
    ) -> FenceDecision:
        return self.atomic_reclaim(
            work_item_id=reservation.work_item_id,
            attempt=reservation.attempt,
            fence=reservation.fence,
            reservation_id=reservation.reservation_id,
            reservation=reservation,
            now=now,
        )

    def status(
        self,
        *,
        work_item_id: str | None = None,
        reservation_id: str | None = None,
        host_ref: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
        now: datetime | None = None,
    ) -> NativeResult:
        result = self._call(
            "status",
            self._status_request(
                work_item_id=work_item_id,
                reservation_id=reservation_id,
                host_ref=host_ref,
                owner_id=self._owner_id,
                fence=None,
                attempt=None,
                lease_epoch=None,
                fencing_token=None,
                input_fingerprint=None,
                limit=limit,
                cursor=cursor,
                now=now,
            ),
        )
        self._validate_status_result(
            result,
            limit=limit,
            work_item_id=work_item_id,
            reservation_id=reservation_id,
            host_ref=host_ref,
        )
        return result

    @staticmethod
    def _validate_status_pagination(result: NativeResult) -> None:
        if result.get("schema_version") != "1":
            raise NativeReservationProtocolError(
                "native status schema_version must be 1"
            )
        complete = _boolean(result.get("complete"), name="status.complete")
        next_cursor = result.get("next_cursor")
        if next_cursor is not None:
            _string(next_cursor, name="status.next_cursor")
        if complete and next_cursor is not None:
            raise NativeReservationProtocolError("complete status page has a cursor")
        if not complete and next_cursor is None:
            raise NativeReservationProtocolError("incomplete status page has no cursor")

    @staticmethod
    def _validate_status_host_and_fields(
        result: NativeResult, *, host_ref: str | None
    ) -> None:
        result_host_ref = result.get("host_ref")
        if result_host_ref is not None:
            _string(result_host_ref, name="status.host_ref")
        if host_ref is not None and result_host_ref != host_ref:
            raise NativeReservationProtocolError("native status host filter mismatch")
        _integer(result.get("host_revision"), name="status.host_revision")
        for field in (
            "held_cpu_weight",
            "held_memory_mib",
            "held_disk_mib",
            "held_process_slots",
            "fairness_debt",
            "orphan_count",
            "superseded_count",
        ):
            _integer(result.get(field), name=f"status.{field}")

    @staticmethod
    def _validate_status_top_level(
        result: NativeResult, *, host_ref: str | None
    ) -> None:
        NativeWorkItemReservationPort._validate_status_pagination(result)
        NativeWorkItemReservationPort._validate_status_host_and_fields(
            result, host_ref=host_ref
        )

    @staticmethod
    def _validate_status_reservation_summary(
        reservation: object,
        index: int,
        *,
        work_item_id: str | None,
        reservation_id: str | None,
        host_ref: str | None,
    ) -> None:
        summary = _mapping(reservation, name=f"status.reservations[{index}]")
        for field in (
            "reservation_id",
            "work_item_id",
            "host_ref",
            "profile_name",
        ):
            _string(
                summary.get(field),
                name=f"status.reservations[{index}].{field}",
            )
        _integer(
            summary.get("attempt"),
            name=f"status.reservations[{index}].attempt",
            minimum=1,
        )
        state = _string(
            summary.get("state"),
            name=f"status.reservations[{index}].state",
        )
        if state not in {
            "reserved",
            "released",
            "reclaimed",
            "expired",
            "superseded",
            "absent",
        }:
            raise NativeReservationProtocolError(
                f"status.reservations[{index}].state is unknown"
            )
        _integer(
            summary.get("revision"),
            name=f"status.reservations[{index}].revision",
            minimum=1,
        )
        _integer(
            summary.get("expires_at_ms"),
            name=f"status.reservations[{index}].expires_at_ms",
        )
        for field in (
            "held_cpu_weight",
            "held_memory_mib",
            "held_disk_mib",
            "held_process_slots",
        ):
            _integer(
                summary.get(field),
                name=f"status.reservations[{index}].{field}",
            )
        _boolean(
            summary.get("tombstone"),
            name=f"status.reservations[{index}].tombstone",
        )
        if work_item_id is not None and summary.get("work_item_id") != work_item_id:
            raise NativeReservationProtocolError(
                f"status.reservations[{index}] WorkItem filter mismatch"
            )
        if (
            reservation_id is not None
            and summary.get("reservation_id") != reservation_id
        ):
            raise NativeReservationProtocolError(
                f"status.reservations[{index}] reservation filter mismatch"
            )
        if host_ref is not None and summary.get("host_ref") != host_ref:
            raise NativeReservationProtocolError(
                f"status.reservations[{index}] host filter mismatch"
            )

    @staticmethod
    def _validate_status_result(
        result: NativeResult,
        *,
        limit: int,
        work_item_id: str | None = None,
        reservation_id: str | None = None,
        host_ref: str | None = None,
    ) -> None:
        NativeWorkItemReservationPort._validate_status_top_level(
            result, host_ref=host_ref
        )
        reservations = _list(
            result.get("reservations"), name="status.reservations", maximum=limit
        )
        for index, reservation in enumerate(reservations):
            NativeWorkItemReservationPort._validate_status_reservation_summary(
                reservation,
                index,
                work_item_id=work_item_id,
                reservation_id=reservation_id,
                host_ref=host_ref,
            )

    def status_pages(
        self,
        *,
        work_item_id: str | None = None,
        reservation_id: str | None = None,
        host_ref: str | None = None,
        limit: int = 100,
        now: datetime | None = None,
        max_pages: int = 100,
    ) -> Iterator[NativeResult]:
        if max_pages < 1 or max_pages > 1000:
            raise ValueError("max_pages must be between 1 and 1000")
        cursor: str | None = None
        seen: set[str] = set()
        for _ in range(max_pages):
            page = self.status(
                work_item_id=work_item_id,
                reservation_id=reservation_id,
                host_ref=host_ref,
                limit=limit,
                cursor=cursor,
                now=now,
            )
            if page.get("schema_version") != "1":
                raise NativeReservationProtocolError(
                    "native status schema_version must be 1"
                )
            self._validate_status_result(
                page,
                limit=limit,
                work_item_id=work_item_id,
                reservation_id=reservation_id,
                host_ref=host_ref,
            )
            complete, next_cursor = self._validate_status_page_cursor(page)
            yield page
            if complete:
                return
            cursor = self._advance_status_cursor(next_cursor, cursor, seen)
        raise NativeReservationProtocolError("native status pagination exceeded bound")

    @staticmethod
    def _validate_status_page_cursor(page: NativeResult) -> tuple[bool, str | None]:
        complete = _boolean(page.get("complete"), name="status.complete")
        next_cursor = page.get("next_cursor")
        if next_cursor is not None:
            next_cursor = _string(next_cursor, name="status.next_cursor")
        if complete and next_cursor is not None:
            raise NativeReservationProtocolError("complete status page has a cursor")
        if not complete and next_cursor is None:
            raise NativeReservationProtocolError("incomplete status page has no cursor")
        return complete, next_cursor

    @staticmethod
    def _advance_status_cursor(
        next_cursor: str | None, cursor: str | None, seen: set[str]
    ) -> str:
        assert isinstance(next_cursor, str)
        if next_cursor in seen or next_cursor == cursor:
            raise NativeReservationProtocolError("status cursor did not advance")
        seen.add(next_cursor)
        return next_cursor

    def update_host(
        self,
        *,
        host_ref: str,
        revision: int,
        capacity: ResourceVector,
        observed: ResourceVector,
        heartbeat_at: datetime,
        heartbeat_ttl_ms: int,
        target_kind: str,
        target_alias: str | None = None,
        now: datetime | None = None,
        draining: bool = False,
        quarantined: bool = False,
        labels: Sequence[str] = (),
        disk_used_mib: int = 0,
        disk_capacity_mib: int = 0,
    ) -> NativeResult:
        request = _HostUpdateRequest(
            host_ref=host_ref,
            revision=revision,
            capacity=capacity,
            observed=observed,
            heartbeat_at=heartbeat_at,
            heartbeat_ttl_ms=heartbeat_ttl_ms,
            target_kind=target_kind,
            target_alias=target_alias,
            now=now,
            draining=draining,
            quarantined=quarantined,
            labels=labels,
            disk_used_mib=disk_used_mib,
            disk_capacity_mib=disk_capacity_mib,
        )
        payload = self._build_update_host_payload(request)
        result = self._call("update_host", payload)
        self._validate_update_host_result(
            result,
            host_ref=host_ref,
            revision=revision,
            draining=draining,
            quarantined=quarantined,
        )
        return result

    def _build_update_host_payload(
        self, request: _HostUpdateRequest
    ) -> dict[str, object]:
        _integer(request.revision, name="host revision", minimum=1)
        _integer(request.heartbeat_ttl_ms, name="heartbeat TTL", minimum=1)
        if request.disk_used_mib < 0 or request.disk_capacity_mib < 0:
            raise ValueError("disk telemetry must be non-negative")
        if request.disk_used_mib > request.disk_capacity_mib:
            raise ValueError("disk used telemetry must not exceed disk capacity")
        if not isinstance(request.draining, bool) or not isinstance(
            request.quarantined, bool
        ):
            raise ValueError("host drain/quarantine telemetry must be boolean")
        bounded_labels = [_string(item, name="host label") for item in request.labels]
        if len(bounded_labels) > 128 or len(set(bounded_labels)) != len(bounded_labels):
            raise ValueError("host labels must be unique and bounded")
        native_target_kind, native_target_alias = _target_fields(
            request.target_kind, request.target_alias, name="host"
        )
        return {
            "schema_version": "1",
            "tenant_ref": self._tenant_ref,
            "host_ref": _string(request.host_ref, name="host_ref"),
            "revision": request.revision,
            "capacity": request.capacity.as_dict(),
            "observed": request.observed.as_dict(),
            "heartbeat_at_ms": self._now_ms(request.heartbeat_at),
            "heartbeat_ttl_ms": request.heartbeat_ttl_ms,
            "now_ms": self._now_ms(request.now),
            "draining": request.draining,
            "quarantined": request.quarantined,
            "labels": sorted(bounded_labels),
            "target_kind": native_target_kind,
            "target_alias": native_target_alias,
            "disk_used_mib": request.disk_used_mib,
            "disk_capacity_mib": request.disk_capacity_mib,
        }

    @staticmethod
    def _validate_update_host_result(
        result: NativeResult,
        *,
        host_ref: str,
        revision: int,
        draining: bool,
        quarantined: bool,
    ) -> None:
        if result.get("schema_version") != "1":
            raise NativeReservationProtocolError(
                "native host result schema_version must be 1"
            )
        if (
            _string(result.get("reason"), name="host result.reason")
            not in _FIXED_STATUS_REASONS
        ):
            raise NativeReservationProtocolError("native host result reason is unknown")
        if _string(result.get("host_ref"), name="host result.host_ref") != host_ref:
            raise NativeReservationProtocolError("native host correlation mismatch")
        accepted = _boolean(result.get("accepted"), name="host result.accepted")
        if accepted != (result.get("reason") == "accepted"):
            raise NativeReservationProtocolError(
                "native host acceptance and reason disagree"
            )
        result_revision = _integer(
            result.get("revision"), name="host result.revision", minimum=1
        )
        if accepted and result_revision != revision:
            raise NativeReservationProtocolError(
                "native accepted host revision does not match request"
            )
        for field in (
            "held_cpu_weight",
            "held_memory_mib",
            "held_disk_mib",
            "held_process_slots",
        ):
            _integer(result.get(field), name=f"host result.{field}")
        result_draining = _boolean(result.get("draining"), name="host result.draining")
        result_quarantined = _boolean(
            result.get("quarantined"), name="host result.quarantined"
        )
        if accepted and (result_draining, result_quarantined) != (
            draining,
            quarantined,
        ):
            raise NativeReservationProtocolError(
                "native accepted host policy does not match request"
            )


def create_production_resource_scheduler(
    graph_client: object,
    *,
    tenant_ref: str,
    owner_id: str,
    fence_codec: NativeFenceCodec,
    profiles: ResourceProfileRegistry,
    operation_key_factory: Callable[[str, str], str] | None = None,
    clock: Callable[[], datetime] | None = None,
    **scheduler_kwargs: Any,
) -> Any:
    """Build a scheduler only from an explicitly injected native graph client.

    ``ResourceScheduler`` itself remains constructor-injected so deterministic
    tests can continue to choose their fixture.  This production helper has no
    path that creates ``InMemoryWorkItemReservationPort``.
    """

    from repository_manager.resource_scheduler import ResourceScheduler

    port = NativeWorkItemReservationPort.from_graph_client(
        graph_client,
        tenant_ref=tenant_ref,
        owner_id=owner_id,
        fence_codec=fence_codec,
        profiles=profiles,
        operation_key_factory=operation_key_factory,
        clock=clock,
    )
    return ResourceScheduler(profiles=profiles, work_item_port=port, **scheduler_kwargs)


__all__ = [
    "DurableJobFenceCodec",
    "NativeFenceCodec",
    "NativeReservationClient",
    "NativeReservationProtocolError",
    "NativeResourceReservationUnavailable",
    "NativeWorkItemReservationPort",
    "create_production_resource_scheduler",
]
