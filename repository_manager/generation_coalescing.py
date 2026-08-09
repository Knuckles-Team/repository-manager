"""Pure compatibility and bounded candidate-generation selection.

The coalescer consumes immutable :class:`CandidateSnapshot` values and never
looks at Git, a mutable branch, a scheduler, or a worker.  A sealed membership
set is therefore a plain value: a candidate arriving after the seal is returned
as ``late`` for the next generation rather than being appended to the old one.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timedelta

from repository_manager.candidate_generation import (
    CandidateSnapshot,
    GenerationRecord,
    generation_record,
    timestamp_value,
)
from repository_manager.development import CandidateVersion, Generation, TargetPolicy


@dataclass(frozen=True)
class CompatibilityKey:
    """All immutable inputs that must agree before candidates can coalesce."""

    repository_id: str
    target_branch: str
    base_sha: str
    config_digest: str
    toolchain_digest: str
    resource_digest: str
    build_target: str
    concept_claims: tuple[str, ...]
    incompatibility_labels: tuple[str, ...]
    target: TargetPolicy


@dataclass(frozen=True)
class CoalescingResult:
    """Deterministic selection result for one queue observation."""

    batches: tuple[tuple[CandidateSnapshot, ...], ...]
    late: tuple[CandidateSnapshot, ...]
    waiting: tuple[CandidateSnapshot, ...]
    sealed_at: datetime

    @property
    def selected(self) -> tuple[CandidateSnapshot, ...]:
        return tuple(member for batch in self.batches for member in batch)


def compatibility_key(candidate: CandidateSnapshot) -> CompatibilityKey:
    """Return the complete immutable grouping key for one candidate."""

    return CompatibilityKey(
        repository_id=candidate.repository.repository_id,
        target_branch=candidate.target_branch,
        base_sha=candidate.base_sha,
        config_digest=candidate.config_digest,
        toolchain_digest=candidate.toolchain_digest,
        resource_digest=candidate.resource_digest,
        build_target=candidate.build_target,
        concept_claims=candidate.concept_claims,
        incompatibility_labels=candidate.incompatibility_labels,
        target=candidate.target,
    )


def candidates_compatible(
    left: CandidateSnapshot,
    right: CandidateSnapshot,
    *,
    base_ancestor: Callable[[str, str], bool] | None = None,
) -> bool:
    """Return whether two immutable snapshots may share one generation.

    Equal base SHAs are the normal fast path.  A caller that has already
    performed a read-only ancestry check may provide it for a pair of distinct
    but compatible base snapshots.  No ancestry lookup is performed here.
    """

    if left.incompatibility_labels or right.incompatibility_labels:
        return False
    left_key = compatibility_key(left)
    right_key = compatibility_key(right)
    if left_key == right_key:
        return True
    if (
        left.repository.repository_id != right.repository.repository_id
        or left.target_branch != right.target_branch
        or left.config_digest != right.config_digest
        or left.toolchain_digest != right.toolchain_digest
        or left.resource_digest != right.resource_digest
        or left.build_target != right.build_target
        or left.concept_claims != right.concept_claims
        or left.incompatibility_labels != right.incompatibility_labels
        or left.target != right.target
    ):
        return False
    if base_ancestor is None:
        return False
    return base_ancestor(left.base_sha, right.base_sha) or base_ancestor(
        right.base_sha, left.base_sha
    )


def _seconds(value: float | int | timedelta) -> float:
    if isinstance(value, timedelta):
        return max(0.0, value.total_seconds())
    return max(0.0, float(value))


def _candidate_sort_key(
    candidate: CandidateSnapshot,
) -> tuple[datetime, str, str, int, str]:
    return (
        timestamp_value(candidate.enqueued_at),
        candidate.branch,
        candidate.candidate_id,
        candidate.version,
        candidate.candidate_sha,
    )


def _ordered_unique(
    candidates: Iterable[CandidateSnapshot],
) -> tuple[CandidateSnapshot, ...]:
    by_id: dict[str, CandidateSnapshot] = {}
    for candidate in candidates:
        previous = by_id.get(candidate.candidate_id)
        if previous is None or candidate.version > previous.version:
            by_id[candidate.candidate_id] = candidate
        elif (
            candidate.version == previous.version
            and candidate.immutable_digest() != previous.immutable_digest()
        ):
            raise ValueError(
                f"candidate version {candidate.record_id} has conflicting inputs"
            )
    return tuple(sorted(by_id.values(), key=_candidate_sort_key))


def select_batches(
    candidates: Iterable[CandidateSnapshot],
    *,
    now: datetime,
    debounce: float | int | timedelta = 0,
    maximum_age: float | int | timedelta = 0,
    batch_size: int = 8,
    sealed_at: datetime | None = None,
    base_ancestor: Callable[[str, str], bool] | None = None,
) -> CoalescingResult:
    """Select mature, compatible batches in deterministic order.

    ``debounce`` holds a fresh candidate briefly so a burst can coalesce;
    ``maximum_age`` is a hard upper bound that makes the oldest candidate
    eligible even if the debounce window has not elapsed.  A candidate newer
    than ``sealed_at`` is never selected into that sealed generation.
    """

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    current = timestamp_value(now)
    seal = timestamp_value(sealed_at) if sealed_at is not None else current
    unique = _ordered_unique(candidates)
    late = tuple(item for item in unique if timestamp_value(item.enqueued_at) > seal)
    eligible: list[CandidateSnapshot] = []
    waiting: list[CandidateSnapshot] = []
    debounce_seconds = _seconds(debounce)
    maximum_age_seconds = _seconds(maximum_age)
    for candidate in unique:
        if candidate in late:
            continue
        age = max(
            0.0, (current - timestamp_value(candidate.enqueued_at)).total_seconds()
        )
        mature = age >= debounce_seconds
        forced = maximum_age_seconds > 0 and age >= maximum_age_seconds
        if mature or forced:
            eligible.append(candidate)
        else:
            waiting.append(candidate)

    batches: list[tuple[CandidateSnapshot, ...]] = []
    remaining = list(eligible)
    while remaining:
        seed = remaining.pop(0)
        batch = [seed]
        retained: list[CandidateSnapshot] = []
        for candidate in remaining:
            if len(batch) < batch_size and all(
                candidates_compatible(member, candidate, base_ancestor=base_ancestor)
                for member in batch
            ):
                batch.append(candidate)
            else:
                retained.append(candidate)
        remaining = retained
        batches.append(tuple(batch))
    return CoalescingResult(
        batches=tuple(batches),
        late=late,
        waiting=tuple(waiting),
        sealed_at=seal,
    )


def generation_id_for(
    members: Iterable[CandidateSnapshot], *, target_branch: str | None = None
) -> str:
    """Derive the stable generation identity from ordered immutable members."""

    ordered = tuple(sorted(members, key=_candidate_sort_key))
    if not ordered:
        raise ValueError("cannot derive an ID for an empty generation")
    first = ordered[0]
    branch = target_branch or first.target_branch
    return Generation.derive_id(
        repository_id=first.repository.repository_id,
        target_branch=branch,
        base_sha=first.base_sha,
        candidate_versions=tuple(
            CandidateVersion(
                candidate_id=item.candidate_id,
                version=index,
                candidate_sha=item.candidate_sha,
            )
            for index, item in enumerate(ordered, start=1)
        ),
        config_digest=first.config_digest,
        toolchain_digest=first.toolchain_digest,
    )


def seal_generation(
    members: Iterable[CandidateSnapshot],
    *,
    sealed_at: datetime,
    target_branch: str | None = None,
    target: TargetPolicy | None = None,
) -> GenerationRecord:
    """Seal one selected batch; membership is fixed from this point onward."""

    ordered = tuple(sorted(members, key=_candidate_sort_key))
    if not ordered:
        raise ValueError("cannot seal an empty generation")
    branch = target_branch or ordered[0].target_branch
    return generation_record(
        ordered,
        target_branch=branch,
        target=target,
        sealed_at=sealed_at,
    )


def next_generation_candidates(
    candidates: Iterable[CandidateSnapshot], *, sealed_at: datetime
) -> tuple[CandidateSnapshot, ...]:
    """Return candidates arriving after a seal for the next generation."""

    seal = timestamp_value(sealed_at)
    return tuple(
        candidate
        for candidate in _ordered_unique(candidates)
        if timestamp_value(candidate.enqueued_at) > seal
    )


__all__ = [
    "CoalescingResult",
    "CompatibilityKey",
    "candidates_compatible",
    "compatibility_key",
    "generation_id_for",
    "next_generation_candidates",
    "seal_generation",
    "select_batches",
]
