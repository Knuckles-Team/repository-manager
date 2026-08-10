"""Deterministic priority aging and fairness-group selection.

Admission requests can be individually well behaved and still starve a tenant
when a high-volume agent keeps winning a simple priority sort.  This module
keeps queue selection pure and deterministic: effective priority ages waiting
requests, while a weighted round-robin group score prevents one fairness group
from monopolizing the host.  The scheduler records selection counts, not a
second job state machine.
"""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock
from typing import Protocol


class FairnessError(ValueError):
    """Malformed fairness input."""


class FairnessAuthority(StrEnum):
    """Truth level of a fairness state implementation."""

    LOCAL_ADVISORY = "local_advisory"
    SIMULATION = "simulation"
    NATIVE = "native"


@dataclass(frozen=True)
class FairnessPolicy:
    aging_interval_seconds: int = 60
    aging_points_per_interval: int = 1
    max_aging_points: int = 10_000
    default_group_weight: int = 1
    group_weights: Mapping[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.aging_interval_seconds < 1:
            raise FairnessError("aging_interval_seconds must be positive")
        if self.aging_points_per_interval < 0:
            raise FairnessError("aging_points_per_interval cannot be negative")
        if self.max_aging_points < 0:
            raise FairnessError("max_aging_points cannot be negative")
        if self.default_group_weight < 1:
            raise FairnessError("default_group_weight must be positive")
        weights = dict(self.group_weights or {})
        if any(not key.strip() or value < 1 for key, value in weights.items()):
            raise FairnessError(
                "fairness group names must be non-blank and weights positive"
            )
        object.__setattr__(self, "group_weights", weights)

    def weight(self, group: str) -> int:
        return int(self.group_weights.get(group, self.default_group_weight))


@dataclass(frozen=True)
class QueueCandidate:
    """Minimal immutable input to the fairness selector."""

    candidate_id: str
    fairness_group: str = "default"
    priority: int = 0
    enqueued_at: datetime = datetime.min.replace(tzinfo=UTC)
    cost: int = 1

    def __post_init__(self) -> None:
        if not self.candidate_id.strip():
            raise FairnessError("candidate_id must be non-blank")
        if not self.fairness_group.strip():
            raise FairnessError("fairness_group must be non-blank")
        if self.priority < 0 or self.cost < 1:
            raise FairnessError("priority must be non-negative and cost positive")


@dataclass(frozen=True)
class SchedulingRank:
    candidate_id: str
    fairness_group: str
    effective_priority: int
    age_intervals: int
    served_units: int
    group_weight: int
    score: float

    def as_dict(self) -> dict[str, object]:
        return {
            "candidate_id": self.candidate_id,
            "fairness_group": self.fairness_group,
            "effective_priority": self.effective_priority,
            "age_intervals": self.age_intervals,
            "served_units": self.served_units,
            "group_weight": self.group_weight,
            "score": self.score,
        }


class FairnessStatePort(Protocol):
    """Durable/native fairness debt port.

    Production implementations must update service debt atomically with the
    scheduler's WorkItem claim/reservation transaction.  Selection itself is
    advisory and never mutates this state.
    """

    @property
    def authority(self) -> FairnessAuthority:
        """Declare whether this state is native, simulation, or local only."""

    def served(self, group: str) -> int:
        """Return durable service units for a fairness group."""

    def record(self, group: str, units: int) -> None:
        """Atomically add service units to a fairness group."""

    def reset(self, group: str | None = None) -> None:
        """Reset state only for test/reconciliation control paths."""


class InMemoryFairnessState:
    """Shared native-state simulation for multiple scheduler replicas."""

    authority = FairnessAuthority.SIMULATION

    def __init__(self) -> None:
        self._lock = RLock()
        self._served: dict[str, int] = {}

    def served(self, group: str) -> int:
        with self._lock:
            return self._served.get(group, 0)

    def record(self, group: str, units: int) -> None:
        if units < 1:
            raise FairnessError("fairness service units must be positive")
        with self._lock:
            self._served[group] = self._served.get(group, 0) + units

    def reset(self, group: str | None = None) -> None:
        with self._lock:
            if group is None:
                self._served.clear()
            else:
                self._served.pop(group, None)


class JsonFairnessState:
    """Restart-safe local state fixture; not distributed enforcement."""

    authority = FairnessAuthority.LOCAL_ADVISORY

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def served(self, group: str) -> int:
        with self._locked():
            return int(self._load().get(group, 0))

    def record(self, group: str, units: int) -> None:
        if units < 1:
            raise FairnessError("fairness service units must be positive")
        with self._locked():
            data = self._load()
            data[group] = int(data.get(group, 0)) + units
            self._write(data)

    def reset(self, group: str | None = None) -> None:
        with self._locked():
            data = self._load()
            if group is None:
                data.clear()
            else:
                data.pop(group, None)
            self._write(data)

    def _load(self) -> dict[str, int]:
        if not self.path.exists():
            return {}
        value = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise FairnessError("fairness state must be a JSON object")
        return {str(key): int(item) for key, item in value.items()}

    def _write(self, value: dict[str, int]) -> None:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=self.path.parent, delete=False
        ) as temp:
            json.dump(value, temp, sort_keys=True)
            temp.write("\n")
            temp.flush()
            os.fsync(temp.fileno())
            temporary = Path(temp.name)
        temporary.replace(self.path)

    @contextmanager
    def _locked(self) -> Iterator[None]:
        fd = os.open(self._lock_path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


class FairnessSelector:
    """Weighted selector with explicit durable-state/advisory disposition."""

    def __init__(
        self,
        policy: FairnessPolicy | None = None,
        *,
        state: FairnessStatePort | None = None,
    ) -> None:
        self.policy = policy or FairnessPolicy()
        self._lock = RLock()
        self._state = state or InMemoryFairnessState()
        self.authoritative = self._state.authority is FairnessAuthority.NATIVE

    @property
    def authority(self) -> FairnessAuthority:
        return self._state.authority

    def served(self, group: str) -> int:
        return self._state.served(group)

    @property
    def state(self) -> FairnessStatePort:
        """Expose state for the native admission adapter binding seam."""

        return self._state

    def reset(self, group: str | None = None) -> None:
        self._state.reset(group)

    def rank(
        self,
        candidate: QueueCandidate,
        *,
        now: datetime | None = None,
        served_units: int | None = None,
    ) -> SchedulingRank:
        now = (now or datetime.now(UTC)).astimezone(UTC)
        enqueued = candidate.enqueued_at.astimezone(UTC)
        age_seconds = max(0, int((now - enqueued).total_seconds()))
        age_intervals = age_seconds // self.policy.aging_interval_seconds
        age_points = min(
            self.policy.max_aging_points,
            age_intervals * self.policy.aging_points_per_interval,
        )
        effective = candidate.priority + age_points
        served = (
            self.served(candidate.fairness_group)
            if served_units is None
            else served_units
        )
        weight = self.policy.weight(candidate.fairness_group)
        # Higher priority wins; the normalized service debt is subtracted so
        # a group that has consumed more work yields to an equally eligible
        # group.  Cost is included to prevent a stream of heavy jobs hiding
        # behind many tiny requests.
        score = float(effective) - (served + candidate.cost) / weight
        return SchedulingRank(
            candidate_id=candidate.candidate_id,
            fairness_group=candidate.fairness_group,
            effective_priority=effective,
            age_intervals=age_intervals,
            served_units=served,
            group_weight=weight,
            score=score,
        )

    def choose(
        self,
        candidates: Iterable[QueueCandidate],
        *,
        limit: int = 1,
        now: datetime | None = None,
    ) -> tuple[QueueCandidate, ...]:
        """Choose at most *limit* candidates without mutating fairness state.

        The first key is effective priority.  The second is weighted service
        debt.  Candidate ID is the final tie-breaker, making simulations and
        restart/replay behavior reproducible.  Native WorkItem admission must
        record service only after a successful claim/reservation.
        """

        if limit < 1:
            return ()
        pool = list(candidates)
        chosen: list[QueueCandidate] = []
        remaining = {candidate.candidate_id: candidate for candidate in pool}
        for _ in range(min(limit, len(pool))):
            if not remaining:
                break
            ranked = [self.rank(candidate, now=now) for candidate in remaining.values()]
            groups: dict[str, list[SchedulingRank]] = {}
            for rank in ranked:
                groups.setdefault(rank.fairness_group, []).append(rank)
            # Pick the least-served group first.  Priority still orders jobs
            # within that group, and breaks a fresh-group tie, but a producer
            # with a constant high priority cannot monopolize another group.
            group_choice = min(
                groups,
                key=lambda group: (
                    self.served(group) / self.policy.weight(group),
                    -max(rank.effective_priority for rank in groups[group]),
                    group,
                ),
            )
            best = max(
                groups[group_choice],
                key=lambda rank: (
                    rank.effective_priority,
                    rank.score,
                    -rank.served_units,
                ),
            )
            tied = [
                rank
                for rank in groups[group_choice]
                if (
                    rank.effective_priority,
                    rank.score,
                    rank.served_units,
                )
                == (best.effective_priority, best.score, best.served_units)
            ]
            best = min(tied, key=lambda rank: rank.candidate_id)
            candidate = remaining.pop(best.candidate_id)
            chosen.append(candidate)
        return tuple(chosen)

    def explain(
        self,
        candidates: Iterable[QueueCandidate],
        *,
        now: datetime | None = None,
    ) -> tuple[SchedulingRank, ...]:
        """Return all ranks without changing fairness counters."""

        return tuple(
            sorted(
                (self.rank(candidate, now=now) for candidate in candidates),
                key=lambda rank: (
                    -rank.effective_priority,
                    -rank.score,
                    rank.served_units,
                    rank.candidate_id,
                ),
            )
        )


__all__ = [
    "FairnessAuthority",
    "FairnessError",
    "FairnessPolicy",
    "FairnessStatePort",
    "FairnessSelector",
    "InMemoryFairnessState",
    "JsonFairnessState",
    "QueueCandidate",
    "SchedulingRank",
]
