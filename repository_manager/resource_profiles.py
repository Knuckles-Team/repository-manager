"""Versioned resource profiles used by the Repository Manager scheduler.

Profiles are declarations, not reservations.  A profile supplies conservative
minimum requirements and placement policy for an operation; the scheduler
combines those requirements with the request made by the WorkItem owner before
it attempts a fenced reservation.  Keeping the registry separate from
``task_queue`` lets old local execution classes remain a compatibility adapter
while new consumers share one admission policy.

The registry deliberately fails closed.  An absent profile is not interpreted
as ``light-check``: doing so would turn a typo in a heavy build declaration into
an unbounded process launch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from repository_manager.development import ResourceRequest


class ResourceProfileError(ValueError):
    """Base error for malformed or unavailable resource profiles."""


class UnknownResourceProfileError(ResourceProfileError):
    """Raised when a request names a profile that is not declared."""


@dataclass(frozen=True)
class ResourceProfile:
    """One immutable, versioned admission policy.

    Numeric fields are minimums.  A caller may request more, but never less
    than a profile's safe default.  ``concurrency_key`` and the two exclusivity
    flags describe scheduler-owned reservation keys; they do not create a
    second job lifecycle.
    """

    name: str
    profile_version: int = 1
    cpu_weight: int = 1
    memory_mib: int = 256
    disk_mib: int = 256
    process_slots: int = 1
    concurrency_key: str = ""
    concurrency_limit: int | None = None
    required_labels: tuple[str, ...] = ()
    anti_affinity: tuple[str, ...] = ()
    repository_exclusive: bool = False
    branch_exclusive: bool = False
    # Watermarks are expressed as used MiB.  Admission is blocked at high and
    # resumes only at low, which gives hysteresis when usage oscillates.
    disk_low_watermark_mib: int | None = None
    disk_high_watermark_mib: int | None = None
    default_priority: int = 0
    default_fairness_group: str = "default"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or self.name.strip() != self.name:
            raise ResourceProfileError("profile name must be non-blank")
        if self.profile_version < 1:
            raise ResourceProfileError("profile_version must be positive")
        for field_name in ("cpu_weight", "memory_mib", "disk_mib", "process_slots"):
            value = getattr(self, field_name)
            if isinstance(value, bool) or value < 1:
                raise ResourceProfileError(f"{field_name} must be a positive integer")
        if self.concurrency_limit is not None and self.concurrency_limit < 1:
            raise ResourceProfileError("concurrency_limit must be positive")
        if self.disk_low_watermark_mib is not None and self.disk_low_watermark_mib < 0:
            raise ResourceProfileError("disk_low_watermark_mib cannot be negative")
        if (
            self.disk_high_watermark_mib is not None
            and self.disk_high_watermark_mib < 0
        ):
            raise ResourceProfileError("disk_high_watermark_mib cannot be negative")
        if (
            self.disk_low_watermark_mib is not None
            and self.disk_high_watermark_mib is not None
            and self.disk_low_watermark_mib > self.disk_high_watermark_mib
        ):
            raise ResourceProfileError(
                "disk_low_watermark_mib must not exceed disk_high_watermark_mib"
            )
        for field_name in (
            "required_labels",
            "anti_affinity",
        ):
            values = tuple(getattr(self, field_name))
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ResourceProfileError(f"{field_name} entries must be non-blank")
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        if not self.concurrency_key:
            object.__setattr__(self, "concurrency_key", self.name)
        if not self.default_fairness_group.strip():
            raise ResourceProfileError("default_fairness_group must be non-blank")

    def request_defaults(self) -> ResourceRequest:
        """Return the profile's typed minimum request."""

        return ResourceRequest(
            resource_class=self.name,
            concurrency_key=self.concurrency_key,
            cpu_weight=self.cpu_weight,
            memory_mib=self.memory_mib,
            disk_mib=self.disk_mib,
            process_slots=self.process_slots,
            host_labels=self.required_labels,
            anti_affinity=self.anti_affinity,
            priority=self.default_priority,
            fairness_group=self.default_fairness_group,
            disk_low_watermark_mib=self.disk_low_watermark_mib,
            disk_high_watermark_mib=self.disk_high_watermark_mib,
        )

    def merge_request(self, request: ResourceRequest) -> ResourceRequest:
        """Apply this profile's conservative minimums to *request*.

        The profile name is authoritative.  Numeric values and labels are
        widened rather than replaced, so a caller cannot under-declare a heavy
        operation while still receiving its profile identity.
        """

        if request.resource_class != self.name:
            raise ResourceProfileError(
                f"request resource_class {request.resource_class!r} does not match "
                f"profile {self.name!r}"
            )
        labels = tuple(sorted(set(self.required_labels).union(request.host_labels)))
        anti_affinity = tuple(
            sorted(set(self.anti_affinity).union(request.anti_affinity))
        )
        # Watermarks are safety limits, not caller-owned hints.  A request may
        # tighten a profile (lower high watermark or lower low watermark), but
        # it may never weaken the profile by raising the reopen threshold or
        # clearing its high threshold.  Profile defaults therefore remain the
        # floor for a native policy key.
        if self.disk_low_watermark_mib is None:
            disk_low = request.disk_low_watermark_mib
        elif request.disk_low_watermark_mib is None:
            disk_low = self.disk_low_watermark_mib
        else:
            disk_low = min(self.disk_low_watermark_mib, request.disk_low_watermark_mib)
        if self.disk_high_watermark_mib is None:
            disk_high = request.disk_high_watermark_mib
        elif request.disk_high_watermark_mib is None:
            disk_high = self.disk_high_watermark_mib
        else:
            disk_high = min(
                self.disk_high_watermark_mib, request.disk_high_watermark_mib
            )
        if disk_low is not None and disk_high is not None and disk_low > disk_high:
            raise ResourceProfileError(
                "caller disk watermarks cannot weaken or invalidate profile "
                f"{self.name!r}"
            )
        return request.model_copy(
            update={
                # The profile is the authority for the concurrency domain.
                # A frozen ResourceRequest defaults to ``light-check`` even
                # when a caller only changes resource_class, so preserving
                # that value here would silently bypass heavy-class limits.
                "concurrency_key": self.concurrency_key,
                "cpu_weight": max(request.cpu_weight, self.cpu_weight),
                "memory_mib": max(request.memory_mib, self.memory_mib),
                "disk_mib": max(request.disk_mib, self.disk_mib),
                "process_slots": max(request.process_slots, self.process_slots),
                "host_labels": labels,
                "anti_affinity": anti_affinity,
                "priority": max(request.priority, self.default_priority),
                "fairness_group": request.fairness_group or self.default_fairness_group,
                "disk_low_watermark_mib": disk_low,
                "disk_high_watermark_mib": disk_high,
            }
        )


class ResourceProfileRegistry:
    """Thread-safe versioned profile registry.

    Registration is explicit and replacement is opt-in.  This makes a profile
    update observable and prevents one plugin from silently changing admission
    policy for already queued jobs.
    """

    def __init__(self, profiles: Mapping[str, ResourceProfile] | None = None) -> None:
        self._lock = RLock()
        self._profiles: dict[str, ResourceProfile] = dict(profiles or {})

    def register(self, profile: ResourceProfile, *, replace: bool = False) -> None:
        with self._lock:
            current = self._profiles.get(profile.name)
            if current is not None and not replace:
                raise ResourceProfileError(
                    f"resource profile {profile.name!r} is already registered; "
                    "use replace=True for an explicit policy change"
                )
            if (
                current is not None
                and profile.profile_version <= current.profile_version
            ):
                raise ResourceProfileError(
                    f"profile {profile.name!r} version {profile.profile_version} must "
                    f"advance beyond {current.profile_version}"
                )
            self._profiles[profile.name] = profile

    def resolve(self, name: str) -> ResourceProfile:
        with self._lock:
            profile = self._profiles.get(name)
        if profile is None:
            raise UnknownResourceProfileError(
                f"unknown resource profile {name!r}; declared profiles: "
                f"{sorted(self.names())}"
            )
        return profile

    def get(
        self, name: str, default: ResourceProfile | None = None
    ) -> ResourceProfile | None:
        with self._lock:
            return self._profiles.get(name, default)

    def names(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._profiles))

    def snapshot(self) -> dict[str, ResourceProfile]:
        with self._lock:
            return dict(self._profiles)


def default_resource_profiles() -> ResourceProfileRegistry:
    """Build the conservative v1 profile set.

    The returned registry is new per caller so tests and future configuration
    overlays cannot mutate global policy.  Heavy classes intentionally require
    substantial tokens; a host with no measured capacity therefore refuses them
    before an executor can create a process.
    """

    profiles = (
        ResourceProfile(
            "light-check",
            cpu_weight=1,
            memory_mib=256,
            disk_mib=256,
            process_slots=1,
            concurrency_key="light-check",
        ),
        ResourceProfile(
            "frontend-build",
            cpu_weight=8,
            memory_mib=8_192,
            disk_mib=4_096,
            process_slots=1,
            concurrency_key="frontend-build",
            concurrency_limit=1,
            anti_affinity=("frontend-build",),
        ),
        ResourceProfile(
            "rust-build",
            cpu_weight=8,
            memory_mib=16_384,
            disk_mib=16_384,
            process_slots=2,
            concurrency_key="rust-build",
        ),
        ResourceProfile(
            "pre-commit",
            cpu_weight=4,
            memory_mib=2_048,
            disk_mib=1_024,
            process_slots=2,
            concurrency_key="pre-commit",
        ),
        ResourceProfile(
            "merge-drain",
            cpu_weight=2,
            memory_mib=1_024,
            disk_mib=512,
            process_slots=1,
            concurrency_key="merge-drain",
            concurrency_limit=1,
            repository_exclusive=True,
            branch_exclusive=True,
        ),
        ResourceProfile(
            "workspace-release",
            cpu_weight=4,
            memory_mib=4_096,
            disk_mib=2_048,
            process_slots=2,
            concurrency_key="workspace-release",
            concurrency_limit=1,
            repository_exclusive=True,
        ),
    )
    return ResourceProfileRegistry({profile.name: profile for profile in profiles})


# A read-mostly convenience snapshot for callers that do not need an isolated
# registry.  ``default_resource_profiles()`` remains the preferred constructor
# for tests/config overlays so policy replacement cannot leak between services.
DEFAULT_RESOURCE_PROFILES = default_resource_profiles()


__all__ = [
    "DEFAULT_RESOURCE_PROFILES",
    "ResourceProfile",
    "ResourceProfileError",
    "ResourceProfileRegistry",
    "UnknownResourceProfileError",
    "default_resource_profiles",
]
