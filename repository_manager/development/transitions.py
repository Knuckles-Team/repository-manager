"""Legal lifecycle transitions for versioned repository-development states."""

from __future__ import annotations

from enum import StrEnum
from typing import TypeVar

from .enums import (
    CandidateState,
    GenerationState,
    JobState,
    LaneState,
    ReleasePlanState,
    ReservationState,
)

StateT = TypeVar("StateT", bound=StrEnum)

_TRANSITIONS: dict[type[StrEnum], dict[StrEnum, frozenset[StrEnum]]] = {
    JobState: {
        JobState.SUBMITTED: frozenset({JobState.READY, JobState.CANCELLED}),
        JobState.READY: frozenset({JobState.LEASED, JobState.CANCELLED}),
        JobState.LEASED: frozenset(
            {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED}
        ),
        JobState.RUNNING: frozenset(
            {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.DEAD_LETTER,
            }
        ),
        JobState.FAILED: frozenset({JobState.READY, JobState.DEAD_LETTER}),
        JobState.SUCCEEDED: frozenset(),
        JobState.CANCELLED: frozenset(),
        JobState.DEAD_LETTER: frozenset(),
    },
    LaneState: {
        LaneState.ALLOCATING: frozenset(
            {LaneState.ACTIVE, LaneState.ABORTED, LaneState.EXPIRED}
        ),
        LaneState.ACTIVE: frozenset(
            {
                LaneState.SUBMITTED,
                LaneState.ABORTED,
                LaneState.EXPIRED,
                LaneState.QUARANTINED,
            }
        ),
        LaneState.SUBMITTED: frozenset(
            {LaneState.LANDED, LaneState.REJECTED, LaneState.ABORTED}
        ),
        LaneState.EXPIRED: frozenset({LaneState.QUARANTINED}),
        LaneState.LANDED: frozenset(),
        LaneState.ABORTED: frozenset(),
        LaneState.REJECTED: frozenset(),
        LaneState.QUARANTINED: frozenset(),
    },
    CandidateState: {
        CandidateState.QUEUED: frozenset(
            {
                CandidateState.VALIDATING,
                CandidateState.WITHDRAWN,
                CandidateState.REJECTED,
            }
        ),
        CandidateState.VALIDATING: frozenset(
            {
                CandidateState.READY,
                CandidateState.FAILED,
                CandidateState.REJECTED,
                CandidateState.WITHDRAWN,
            }
        ),
        CandidateState.READY: frozenset(
            {CandidateState.LANDING, CandidateState.REJECTED, CandidateState.WITHDRAWN}
        ),
        CandidateState.LANDING: frozenset(
            {CandidateState.LANDED, CandidateState.REJECTED, CandidateState.FAILED}
        ),
        CandidateState.LANDED: frozenset(),
        CandidateState.REJECTED: frozenset(),
        CandidateState.WITHDRAWN: frozenset(),
        CandidateState.FAILED: frozenset(),
    },
    GenerationState: {
        GenerationState.OPEN: frozenset(
            {GenerationState.SEALED, GenerationState.REJECTED, GenerationState.EXPIRED}
        ),
        GenerationState.SEALED: frozenset(
            {
                GenerationState.INTEGRATING,
                GenerationState.REJECTED,
                GenerationState.EXPIRED,
            }
        ),
        GenerationState.INTEGRATING: frozenset(
            {
                GenerationState.CERTIFIED,
                GenerationState.REJECTED,
                GenerationState.EXPIRED,
            }
        ),
        GenerationState.CERTIFIED: frozenset(
            {GenerationState.LANDING, GenerationState.REJECTED, GenerationState.EXPIRED}
        ),
        GenerationState.LANDING: frozenset(
            {GenerationState.LANDED, GenerationState.REJECTED}
        ),
        GenerationState.LANDED: frozenset(),
        GenerationState.REJECTED: frozenset(),
        GenerationState.EXPIRED: frozenset(),
    },
    ReservationState: {
        ReservationState.RESERVED: frozenset(
            {ReservationState.RELEASED, ReservationState.EXPIRED}
        ),
        ReservationState.RELEASED: frozenset(),
        ReservationState.EXPIRED: frozenset(),
        ReservationState.REFUSED: frozenset(),
    },
    ReleasePlanState: {
        ReleasePlanState.DRAFT: frozenset(
            {ReleasePlanState.FROZEN, ReleasePlanState.REJECTED}
        ),
        ReleasePlanState.FROZEN: frozenset(
            {ReleasePlanState.APPLIED, ReleasePlanState.REJECTED}
        ),
        ReleasePlanState.APPLIED: frozenset(),
        ReleasePlanState.REJECTED: frozenset(),
    },
}


def is_legal_transition(current: StateT, target: StateT) -> bool:
    """Return whether *current* may advance to *target* in its state family."""

    if type(current) is not type(target):
        return False
    return target in _TRANSITIONS.get(type(current), {}).get(current, frozenset())


def require_legal_transition(current: StateT, target: StateT) -> None:
    """Raise a stable validation error when a lifecycle transition is illegal."""

    if not is_legal_transition(current, target):
        raise ValueError(
            f"illegal {type(current).__name__} transition: {current} -> {target}"
        )


__all__ = ["is_legal_transition", "require_legal_transition"]
