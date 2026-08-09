"""Lane count and disk quota policy with bounded filesystem accounting."""

from __future__ import annotations

import os
import stat
import time
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock

from .lane_record import LaneLifecycleState, LaneRecord


@dataclass(frozen=True)
class DiskUsage:
    """Bounded size observation for one lane path."""

    path: str
    bytes: int = 0
    entries: int = 0
    directories: int = 0
    skipped_symlinks: int = 0
    skipped_errors: int = 0
    bounded: bool = False
    observed_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def size_bytes(self) -> int:
        return self.bytes

    @property
    def observed_disk_bytes(self) -> int:
        return self.bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "bytes": self.bytes,
            "size_bytes": self.bytes,
            "entries": self.entries,
            "directories": self.directories,
            "skipped_symlinks": self.skipped_symlinks,
            "skipped_errors": self.skipped_errors,
            "bounded": self.bounded,
            "observed_at": self.observed_at.isoformat(),
        }


class DiskAccountingProbe:
    """Cacheable, bounded, no-follow-links disk accounting.

    The probe walks with ``os.scandir`` and requests ``follow_symlinks=False``
    for every test and stat operation.  A symlink is counted as a skipped entry,
    never as the size of its target.  Limits are hard bounds: the probe stops
    once either the entry or depth budget is exhausted and reports ``bounded``.
    """

    def __init__(
        self,
        *,
        ttl_seconds: float = 5.0,
        max_entries: int = 100_000,
        max_depth: int = 32,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if ttl_seconds < 0:
            raise ValueError("disk probe cache TTL cannot be negative")
        if max_entries < 1 or max_depth < 0:
            raise ValueError("disk probe bounds must be positive")
        self.ttl_seconds = float(ttl_seconds)
        self.max_entries = int(max_entries)
        self.max_depth = int(max_depth)
        self._clock = clock or time.monotonic
        self._cache: dict[str, tuple[float, DiskUsage]] = {}
        self._lock = RLock()

    def measure(
        self,
        path: str | Path,
        *,
        now: datetime | None = None,
        refresh: bool = False,
    ) -> DiskUsage:
        """Measure *path* without following links, returning a cached result."""

        candidate = Path(path).expanduser()
        key = str(candidate.absolute())
        tick = self._clock()
        with self._lock:
            cached = self._cache.get(key)
            if (
                cached is not None
                and not refresh
                and tick - cached[0] <= self.ttl_seconds
            ):
                return cached[1]
        result = self._measure_uncached(candidate, now=now)
        with self._lock:
            self._cache[key] = (tick, result)
        return result

    def invalidate(self, path: str | Path | None = None) -> None:
        """Invalidate one path or the complete bounded cache."""

        with self._lock:
            if path is None:
                self._cache.clear()
            else:
                self._cache.pop(str(Path(path).expanduser().absolute()), None)

    def _measure_uncached(self, root: Path, *, now: datetime | None) -> DiskUsage:
        observed_at = (now or datetime.now(UTC)).astimezone(UTC)
        if root.is_symlink():
            return DiskUsage(
                path=str(root), skipped_symlinks=1, observed_at=observed_at
            )
        if not root.exists():
            return DiskUsage(path=str(root), observed_at=observed_at)

        total = entries = directories = skipped_symlinks = skipped_errors = 0
        bounded = False
        pending: list[tuple[Path, int]] = [(root, 0)]
        while pending:
            current, depth = pending.pop()
            try:
                with os.scandir(current) as iterator:
                    for entry in iterator:
                        if entries >= self.max_entries:
                            bounded = True
                            break
                        entries += 1
                        try:
                            if entry.is_symlink():
                                skipped_symlinks += 1
                                continue
                            if entry.is_dir(follow_symlinks=False):
                                directories += 1
                                if depth >= self.max_depth:
                                    bounded = True
                                else:
                                    pending.append((Path(entry.path), depth + 1))
                                continue
                            info = entry.stat(follow_symlinks=False)
                            if stat.S_ISREG(info.st_mode):
                                total += max(0, int(info.st_size))
                        except (OSError, ValueError):
                            skipped_errors += 1
                if entries >= self.max_entries and pending:
                    bounded = True
            except (OSError, ValueError):
                skipped_errors += 1
        return DiskUsage(
            path=str(root),
            bytes=total,
            entries=entries,
            directories=directories,
            skipped_symlinks=skipped_symlinks,
            skipped_errors=skipped_errors,
            bounded=bounded,
            observed_at=observed_at,
        )


@dataclass(frozen=True)
class LaneQuotaPolicy:
    """Per-scope active lane and disk limits.

    ``None`` means unlimited for that dimension.  Counts and bytes are kept
    separate so an admission refusal can explain the exact current usage.
    """

    max_per_agent: int | None = None
    max_per_session: int | None = None
    max_per_repository: int | None = None
    max_per_host: int | None = None
    max_predicted_disk_bytes: int | None = None
    max_observed_disk_bytes: int | None = None
    max_total_active: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "max_per_agent",
            "max_per_session",
            "max_per_repository",
            "max_per_host",
            "max_predicted_disk_bytes",
            "max_observed_disk_bytes",
            "max_total_active",
        ):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")

    @classmethod
    def unlimited(cls) -> LaneQuotaPolicy:
        return cls()


QuotaPolicy = LaneQuotaPolicy


@dataclass(frozen=True)
class LaneQuotaUsage:
    """Exact active lane counts and disk totals at admission time."""

    total_active: int = 0
    by_agent: Mapping[str, int] = field(default_factory=dict)
    by_session: Mapping[str, int] = field(default_factory=dict)
    by_repository: Mapping[str, int] = field(default_factory=dict)
    by_host: Mapping[str, int] = field(default_factory=dict)
    predicted_disk_bytes: int = 0
    observed_disk_bytes: int = 0

    @property
    def predicted_bytes(self) -> int:
        return self.predicted_disk_bytes

    @property
    def observed_bytes(self) -> int:
        return self.observed_disk_bytes

    def as_dict(self) -> dict[str, object]:
        return {
            "total_active": self.total_active,
            "by_agent": dict(self.by_agent),
            "by_session": dict(self.by_session),
            "by_repository": dict(self.by_repository),
            "by_host": dict(self.by_host),
            "predicted_disk_bytes": self.predicted_disk_bytes,
            "observed_disk_bytes": self.observed_disk_bytes,
        }


@dataclass(frozen=True)
class LaneQuotaDecision:
    """Admission result with usage and the first violated quota."""

    admitted: bool
    reason: str = "admitted"
    scope: str = ""
    requested_predicted_disk_bytes: int = 0
    usage: LaneQuotaUsage = field(default_factory=LaneQuotaUsage)
    policy: LaneQuotaPolicy = field(default_factory=LaneQuotaPolicy)

    @property
    def ok(self) -> bool:
        return self.admitted

    def as_dict(self) -> dict[str, object]:
        return {
            "admitted": self.admitted,
            "ok": self.admitted,
            "reason": self.reason,
            "scope": self.scope,
            "requested_predicted_disk_bytes": self.requested_predicted_disk_bytes,
            "usage": self.usage.as_dict(),
            "policy": self.policy.__dict__.copy(),
        }


class LaneQuotaExceeded(ValueError):
    """Admission refused before a worktree is created."""

    def __init__(self, decision: LaneQuotaDecision):
        self.decision = decision
        super().__init__(
            f"lane quota refused ({decision.scope or 'quota'}): {decision.reason}"
        )


class LaneQuota:
    """Pure quota calculations over a sequence of lane records."""

    def __init__(self, policy: LaneQuotaPolicy | None = None) -> None:
        self.policy = policy or LaneQuotaPolicy.unlimited()

    @staticmethod
    def usage(records: Iterable[LaneRecord]) -> LaneQuotaUsage:
        total = 0
        predicted = observed = 0
        agent: dict[str, int] = {}
        session: dict[str, int] = {}
        repository: dict[str, int] = {}
        host: dict[str, int] = {}
        for record in records:
            if record.state not in {
                LaneLifecycleState.ALLOCATING,
                LaneLifecycleState.ACTIVE,
                LaneLifecycleState.SUBMITTED,
                LaneLifecycleState.EXPIRED,
            }:
                continue
            total += 1
            predicted += record.predicted_disk_bytes
            observed += record.observed_disk_bytes
            for values, key in (
                (agent, record.owner_id),
                (session, record.session_id),
                (repository, record.repository_id),
                (host, record.host_id),
            ):
                if key:
                    values[key] = values.get(key, 0) + 1
        return LaneQuotaUsage(
            total_active=total,
            by_agent=agent,
            by_session=session,
            by_repository=repository,
            by_host=host,
            predicted_disk_bytes=predicted,
            observed_disk_bytes=observed,
        )

    def check(
        self,
        records: Iterable[LaneRecord],
        *,
        owner_id: str | None,
        session_id: str | None,
        repository_id: str,
        host_id: str | None,
        predicted_disk_bytes: int,
    ) -> LaneQuotaDecision:
        if predicted_disk_bytes < 0:
            raise ValueError("predicted disk bytes must be non-negative")
        current = self.usage(records)
        next_total = current.total_active + 1
        scopes = (
            (
                "agent",
                self.policy.max_per_agent,
                (current.by_agent.get(owner_id or "", 0) + 1),
            ),
            (
                "session",
                self.policy.max_per_session,
                (current.by_session.get(session_id or "", 0) + 1),
            ),
            (
                "repository",
                self.policy.max_per_repository,
                (current.by_repository.get(repository_id, 0) + 1),
            ),
            (
                "host",
                self.policy.max_per_host,
                (current.by_host.get(host_id or "", 0) + 1),
            ),
            ("total", self.policy.max_total_active, next_total),
        )
        for scope, limit, value in scopes:
            if limit is not None and value > limit:
                return LaneQuotaDecision(
                    False,
                    reason=f"{scope} lane count {value} exceeds limit {limit}",
                    scope=scope,
                    requested_predicted_disk_bytes=predicted_disk_bytes,
                    usage=current,
                    policy=self.policy,
                )
        predicted = current.predicted_disk_bytes + predicted_disk_bytes
        if (
            self.policy.max_predicted_disk_bytes is not None
            and predicted > self.policy.max_predicted_disk_bytes
        ):
            return LaneQuotaDecision(
                False,
                reason=(
                    f"predicted disk bytes {predicted} exceeds limit "
                    f"{self.policy.max_predicted_disk_bytes}"
                ),
                scope="predicted_disk",
                requested_predicted_disk_bytes=predicted_disk_bytes,
                usage=current,
                policy=self.policy,
            )
        if self.policy.max_observed_disk_bytes is not None and (
            current.observed_disk_bytes > self.policy.max_observed_disk_bytes
        ):
            return LaneQuotaDecision(
                False,
                reason=(
                    f"observed disk bytes {current.observed_disk_bytes} exceeds "
                    f"limit {self.policy.max_observed_disk_bytes}"
                ),
                scope="observed_disk",
                requested_predicted_disk_bytes=predicted_disk_bytes,
                usage=current,
                policy=self.policy,
            )
        return LaneQuotaDecision(
            True,
            requested_predicted_disk_bytes=predicted_disk_bytes,
            usage=current,
            policy=self.policy,
        )


__all__ = [
    "DiskAccountingProbe",
    "DiskUsage",
    "LaneQuota",
    "LaneQuotaDecision",
    "LaneQuotaExceeded",
    "LaneQuotaPolicy",
    "LaneQuotaUsage",
    "QuotaPolicy",
]
