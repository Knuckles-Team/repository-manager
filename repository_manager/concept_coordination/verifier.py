"""Pure candidate/generation concept verification (deliverable 5).

Requires a matching active/landed claim, or a pre-existing base ID, for every
concept marker a candidate introduces — with exact refusal evidence naming
which IDs failed and why (missing claim, wrong owner, wrong repository,
expired, released/tombstoned). This module performs no I/O and mutates
nothing: it is handed already-scanned marker sets
(:mod:`~repository_manager.concept_coordination.scanner`) and already-fetched
claim records, and returns a classification.

**On the RMDD-12 "generation hook" seam:** the lane brief describes RMDD-17
registering this verifier "through a frozen seam" that RMDD-12/RMDD-20 own.
As of this lane (2026-08-10), no such registration point exists anywhere in
repository-manager on ``main`` — verified by an exhaustive grep for
``verify_hook``/``register_verif*``/``register_hook`` and by reading
``candidate_generation.py``/``merge_queue.py`` end to end; nothing calls
:func:`repository_manager.generation_bisection.decide` outside tests. This
module therefore does **not** edit ``candidate_generation.py``,
``generation_bisection.py``, or ``merge_queue.py`` (forbidden edits: "merge
generation internals", "MCP/CLI entrypoints") — it produces
:class:`~repository_manager.generation_bisection.AttemptResult`, the *exact*
existing typed result shape ``decide()`` already consumes, purely as data.
Wiring a call site is RMDD-12/RMDD-20's seam to build; this module is ready
to be called the moment it exists.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

from repository_manager.development import FailureClass
from repository_manager.generation_bisection import AttemptResult

from .models import ConceptClaimRecord
from .state import ConceptClaimState

__all__ = [
    "ConceptVerificationOutcome",
    "GenerationConceptVerification",
    "verify_candidate_concepts",
    "verify_generation_concepts",
]

_VALID_ACTIVE_STATES = frozenset(
    {
        ConceptClaimState.RESERVED,
        ConceptClaimState.MATERIALIZED,
        ConceptClaimState.LANDED,
    }
)


@dataclass(frozen=True)
class ConceptVerificationOutcome:
    """Classification of one candidate's introduced concept markers."""

    candidate_id: str
    checked_ids: tuple[str, ...]
    pre_existing_ids: tuple[str, ...]
    missing_claims: tuple[str, ...] = ()
    wrong_owner: tuple[str, ...] = ()
    wrong_repo: tuple[str, ...] = ()
    expired: tuple[str, ...] = ()
    released_or_tombstoned: tuple[str, ...] = ()

    @property
    def passed(self) -> bool:
        return not (
            self.missing_claims
            or self.wrong_owner
            or self.wrong_repo
            or self.expired
            or self.released_or_tombstoned
        )

    @property
    def detail(self) -> str:
        if self.passed:
            return f"{self.candidate_id}: every introduced concept id has a valid claim"
        parts = []
        if self.missing_claims:
            parts.append(f"missing_claims={list(self.missing_claims)}")
        if self.wrong_owner:
            parts.append(f"wrong_owner={list(self.wrong_owner)}")
        if self.wrong_repo:
            parts.append(f"wrong_repo={list(self.wrong_repo)}")
        if self.expired:
            parts.append(f"expired={list(self.expired)}")
        if self.released_or_tombstoned:
            parts.append(f"released_or_tombstoned={list(self.released_or_tombstoned)}")
        return f"{self.candidate_id}: " + "; ".join(parts)


def verify_candidate_concepts(
    *,
    candidate_id: str,
    repository_ref: str,
    owner_ref: str,
    introduced: tuple[str, ...],
    base_marker_ids: frozenset[str],
    claims_by_concept_id: Mapping[str, ConceptClaimRecord],
) -> ConceptVerificationOutcome:
    """Classify one candidate's introduced concept markers against claims.

    ``base_marker_ids`` is the set of concept IDs already present at the
    candidate's immutable base — those never require a new claim
    ("pre-existing base ID does not require a new claim").
    """

    pre_existing: list[str] = []
    missing: list[str] = []
    wrong_owner: list[str] = []
    wrong_repo: list[str] = []
    expired: list[str] = []
    released_or_tombstoned: list[str] = []

    for concept_id in introduced:
        if concept_id in base_marker_ids:
            pre_existing.append(concept_id)
            continue
        claim = claims_by_concept_id.get(concept_id)
        if claim is None:
            missing.append(concept_id)
            continue
        if claim.repository_ref != repository_ref:
            wrong_repo.append(concept_id)
            continue
        if claim.owner_ref != owner_ref:
            wrong_owner.append(concept_id)
            continue
        if claim.state is ConceptClaimState.EXPIRED:
            expired.append(concept_id)
            continue
        if claim.state in (ConceptClaimState.RELEASED, ConceptClaimState.TOMBSTONED):
            released_or_tombstoned.append(concept_id)
            continue
        if claim.state not in _VALID_ACTIVE_STATES:
            missing.append(concept_id)

    return ConceptVerificationOutcome(
        candidate_id=candidate_id,
        checked_ids=tuple(introduced),
        pre_existing_ids=tuple(sorted(pre_existing)),
        missing_claims=tuple(sorted(missing)),
        wrong_owner=tuple(sorted(wrong_owner)),
        wrong_repo=tuple(sorted(wrong_repo)),
        expired=tuple(sorted(expired)),
        released_or_tombstoned=tuple(sorted(released_or_tombstoned)),
    )


@dataclass(frozen=True)
class GenerationConceptVerification:
    """Aggregate verification across every candidate in one generation."""

    generation_id: str
    per_candidate: Mapping[str, ConceptVerificationOutcome] = field(
        default_factory=dict
    )
    collisions: Mapping[str, tuple[str, ...]] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.collisions and all(
            outcome.passed for outcome in self.per_candidate.values()
        )

    @property
    def detail(self) -> str:
        if self.passed:
            return (
                f"{self.generation_id}: all "
                f"{len(self.per_candidate)} member(s) have valid concept claims"
            )
        parts = [
            outcome.detail
            for outcome in self.per_candidate.values()
            if not outcome.passed
        ]
        if self.collisions:
            parts.append(f"cross_candidate_collisions={dict(self.collisions)}")
        return f"{self.generation_id}: " + " | ".join(parts)

    def to_attempt_result(
        self, *, attempt: int = 1, attempt_budget: int = 3
    ) -> AttemptResult:
        """Project onto the existing typed bisection input, data only.

        Cross-candidate collisions and per-candidate concept failures are
        both attributable to the offending candidate(s) — never an opaque
        worker/environment failure — so ``decide()`` correctly routes a
        failure to SPLIT/REJECT rather than a blind RETRY.
        """

        return AttemptResult(
            generation_id=self.generation_id,
            member_ids=tuple(sorted(self.per_candidate)),
            passed=self.passed,
            failure_class=(
                None if self.passed else FailureClass.VALIDATION_CANDIDATE_FAILURE
            ),
            detail=self.detail,
            attempt=attempt,
            attempt_budget=attempt_budget,
        )


def verify_generation_concepts(
    *,
    generation_id: str,
    introduced_by_candidate: Mapping[str, tuple[str, ...]],
    repository_ref: str,
    owner_by_candidate: Mapping[str, str],
    base_marker_ids: frozenset[str],
    claims_by_concept_id: Mapping[str, ConceptClaimRecord],
) -> GenerationConceptVerification:
    """Verify every candidate in a generation and report cross-candidate collisions."""

    from .scanner import find_cross_candidate_collisions

    collisions = find_cross_candidate_collisions(introduced_by_candidate)
    per_candidate = {
        candidate_id: verify_candidate_concepts(
            candidate_id=candidate_id,
            repository_ref=repository_ref,
            owner_ref=owner_by_candidate[candidate_id],
            introduced=introduced,
            base_marker_ids=base_marker_ids,
            claims_by_concept_id=claims_by_concept_id,
        )
        for candidate_id, introduced in introduced_by_candidate.items()
    }
    return GenerationConceptVerification(
        generation_id=generation_id,
        per_candidate=per_candidate,
        collisions=collisions,
    )
