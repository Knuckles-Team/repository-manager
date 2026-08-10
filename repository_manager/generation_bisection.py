"""Pure result classification and deterministic generation bisection.

The bisection planner never executes a gate or changes a candidate.  It turns
one immutable attempt result into an explicit decision so a controller can
retry an environment failure, split an attributable failure, or reject one
candidate with evidence.  Parent/child lineage is represented by generation
IDs and is therefore restart-safe when appended to the existing queue record
authority.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import StrEnum

from repository_manager.development import FailureClass


class FailureKind(StrEnum):
    """Pure bisection classes; values remain descriptive, not executable."""

    NONE = "none"
    CANDIDATE = "candidate"
    ENVIRONMENT = "environment"
    CANCELLATION = "cancellation"
    OPAQUE = "opaque"


class DecisionAction(StrEnum):
    """The controller action implied by one attempt result."""

    ACCEPT = "accept"
    RETRY = "retry"
    SPLIT = "split"
    REJECT = "reject"


@dataclass(frozen=True)
class AttemptResult:
    """Immutable result of one generation evaluation."""

    generation_id: str
    member_ids: tuple[str, ...]
    passed: bool
    failure_class: str | FailureClass | None = None
    detail: str = ""
    evidence_ids: tuple[str, ...] = ()
    attempt: int = 1
    attempt_budget: int = 3

    def __post_init__(self) -> None:
        if not self.generation_id:
            raise ValueError("attempt requires a generation_id")
        if not self.member_ids:
            raise ValueError("attempt requires at least one member")
        if len(set(self.member_ids)) != len(self.member_ids):
            raise ValueError("attempt member IDs must be unique")
        if self.attempt < 1:
            raise ValueError("attempt number must be positive")
        if self.attempt_budget < 1:
            raise ValueError("attempt budget must be positive")
        object.__setattr__(self, "member_ids", tuple(self.member_ids))
        object.__setattr__(self, "evidence_ids", tuple(sorted(set(self.evidence_ids))))


@dataclass(frozen=True)
class BisectionDecision:
    """A deterministic next step and its explicit attribution metadata."""

    action: DecisionAction
    kind: FailureKind
    parent_generation_id: str
    member_ids: tuple[str, ...]
    left_member_ids: tuple[str, ...] = ()
    right_member_ids: tuple[str, ...] = ()
    accepted_member_ids: tuple[str, ...] = ()
    rejected_member_ids: tuple[str, ...] = ()
    retry_member_ids: tuple[str, ...] = ()
    reused_evidence_ids: tuple[str, ...] = ()
    quarantined: bool = False
    reason: str = ""

    @property
    def child_member_sets(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            members
            for members in (self.left_member_ids, self.right_member_ids)
            if members
        )


def classify_failure(
    *,
    passed: bool,
    failure_class: str | FailureClass | None = None,
) -> FailureKind:
    """Classify only an explicit stable failure code.

    Human-readable detail is intentionally not consulted: it is untrusted
    evidence and must never turn an opaque worker failure into a source-code
    rejection or a bisection decision.
    """

    if passed:
        return FailureKind.NONE if failure_class is None else FailureKind.OPAQUE
    if isinstance(failure_class, FailureClass):
        value = failure_class
    elif isinstance(failure_class, str):
        try:
            value = FailureClass(failure_class)
        except ValueError:
            return FailureKind.OPAQUE
    else:
        return FailureKind.OPAQUE
    if value in {
        FailureClass.WORKER_ENVIRONMENT_FAILURE,
        FailureClass.CAPACITY_DISK_DEFERRED,
    }:
        return FailureKind.ENVIRONMENT
    if value == FailureClass.CANCELLED_DEADLINE:
        return FailureKind.CANCELLATION
    if value == FailureClass.VALIDATION_CANDIDATE_FAILURE:
        return FailureKind.CANDIDATE
    return FailureKind.OPAQUE


def _split_members(
    member_ids: tuple[str, ...],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    middle = len(member_ids) // 2
    if middle < 1:
        raise ValueError("cannot split a singleton attempt")
    return member_ids[:middle], member_ids[middle:]


def decide(attempt: AttemptResult) -> BisectionDecision:
    """Choose accept/retry/split/reject for one immutable attempt."""

    kind = classify_failure(
        passed=attempt.passed,
        failure_class=attempt.failure_class,
    )
    if attempt.passed and kind == FailureKind.NONE:
        return BisectionDecision(
            action=DecisionAction.ACCEPT,
            kind=FailureKind.NONE,
            parent_generation_id=attempt.generation_id,
            member_ids=attempt.member_ids,
            accepted_member_ids=attempt.member_ids,
            reused_evidence_ids=attempt.evidence_ids,
            reason="generation passed; exact evidence is reusable for the same membership",
        )
    if kind != FailureKind.CANDIDATE:
        exhausted = attempt.attempt >= attempt.attempt_budget
        quarantine = kind == FailureKind.ENVIRONMENT and exhausted
        if kind == FailureKind.OPAQUE:
            reason = (
                "unknown failure code: retry unchanged generation"
                if not exhausted
                else "unknown failure code exhausted retry budget: quarantine and reconcile"
            )
        elif kind == FailureKind.ENVIRONMENT:
            reason = (
                "environment failure: retry the unchanged generation"
                if not exhausted
                else "environment failure exhausted retry budget: quarantine and reconcile"
            )
        else:
            reason = "cancellation is not a success; retry or apply group cancellation policy"
        return BisectionDecision(
            action=DecisionAction.RETRY,
            kind=kind,
            parent_generation_id=attempt.generation_id,
            member_ids=attempt.member_ids,
            retry_member_ids=attempt.member_ids,
            quarantined=quarantine or (kind == FailureKind.OPAQUE and exhausted),
            reason=reason,
        )
    if len(attempt.member_ids) == 1:
        return BisectionDecision(
            action=DecisionAction.REJECT,
            kind=kind,
            parent_generation_id=attempt.generation_id,
            member_ids=attempt.member_ids,
            rejected_member_ids=attempt.member_ids,
            reason=attempt.detail or "single candidate failed generation evaluation",
        )
    left, right = _split_members(attempt.member_ids)
    return BisectionDecision(
        action=DecisionAction.SPLIT,
        kind=kind,
        parent_generation_id=attempt.generation_id,
        member_ids=attempt.member_ids,
        left_member_ids=left,
        right_member_ids=right,
        reason=attempt.detail or "failure requires deterministic attribution bisection",
    )


def reusable_evidence(
    attempt: AttemptResult, member_ids: Iterable[str]
) -> tuple[str, ...]:
    """Reuse evidence only when the exact ordered membership is unchanged."""

    requested = tuple(member_ids)
    if attempt.passed and requested == attempt.member_ids:
        return attempt.evidence_ids
    return ()


@dataclass(frozen=True)
class LineageEdge:
    """Typed parent-to-child relation persisted with bisection results."""

    parent_generation_id: str
    child_generation_id: str
    relation: str = "bisection"

    def __post_init__(self) -> None:
        if not self.parent_generation_id or not self.child_generation_id:
            raise ValueError("lineage edges require parent and child generation IDs")
        if not self.relation:
            raise ValueError("lineage edges require a relation")

    def to_record(self) -> dict[str, str]:
        return {
            "parent_generation_id": self.parent_generation_id,
            "child_generation_id": self.child_generation_id,
            "relation": self.relation,
        }


def child_lineage(
    parent_generation_id: str,
    child_generation_ids: Iterable[str],
    *,
    relation: str = "bisection",
) -> tuple[LineageEdge, ...]:
    """Return stable typed parent-to-child edges without dropping the parent."""

    if not parent_generation_id:
        raise ValueError("lineage edges require a parent generation ID")
    if not relation:
        raise ValueError("lineage edges require a relation")
    return tuple(
        LineageEdge(parent_generation_id, child_id, relation)
        for child_id in sorted(
            set(str(value) for value in child_generation_ids if value)
        )
    )


__all__ = [
    "AttemptResult",
    "BisectionDecision",
    "DecisionAction",
    "FailureKind",
    "LineageEdge",
    "child_lineage",
    "classify_failure",
    "decide",
    "reusable_evidence",
]
