"""Pure candidate/generation concept verification tests (deliverable 5).

Covers every required refusal from the lane brief: new unclaimed / wrong
owner / wrong repo / expired / released id refuses; a pre-existing base id
never requires a new claim; a combined generation with distinct valid claims
passes and preserves per-member attribution; a cross-candidate collision
refuses. Also proves composition with the existing, untouched
``generation_bisection.decide`` (RMDD-12) — the real bisection planner, not a
reimplementation.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from repository_manager.concept_coordination.models import ConceptClaimRecord
from repository_manager.concept_coordination.state import (
    ConceptClaimState,
    ConceptClaimVisibility,
)
from repository_manager.concept_coordination.verifier import (
    verify_candidate_concepts,
    verify_generation_concepts,
)
from repository_manager.generation_bisection import DecisionAction, decide

_NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(hours=1)
_PAST = _NOW - timedelta(hours=1)


def _claim(
    concept_id: str,
    *,
    repository_ref: str = "repo-1",
    owner_ref: str = "owner-1",
    state: ConceptClaimState = ConceptClaimState.RESERVED,
    expires_at: datetime = _LATER,
) -> ConceptClaimRecord:
    return ConceptClaimRecord(
        reservation_id=f"concept-reservation:{concept_id}",
        concept_id=concept_id,
        tenant_ref="tenant-1",
        repository_ref=repository_ref,
        lane_ref="lane-a",
        owner_ref=owner_ref,
        state=state,
        visibility=ConceptClaimVisibility.PRIVATE,
        fence=1,
        created_at=_PAST,
        expires_at=expires_at,
        transitioned_at=_NOW,
    )


def test_new_id_with_valid_claim_passes() -> None:
    claim = _claim("RM-OS.test.ok")
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.ok",),
        base_marker_ids=frozenset(),
        claims_by_concept_id={"RM-OS.test.ok": claim},
    )
    assert outcome.passed
    assert outcome.missing_claims == ()


def test_new_id_with_no_claim_refuses() -> None:
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.missing",),
        base_marker_ids=frozenset(),
        claims_by_concept_id={},
    )
    assert not outcome.passed
    assert outcome.missing_claims == ("RM-OS.test.missing",)


def test_new_id_wrong_owner_refuses() -> None:
    claim = _claim("RM-OS.test.owner", owner_ref="someone-else")
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.owner",),
        base_marker_ids=frozenset(),
        claims_by_concept_id={"RM-OS.test.owner": claim},
    )
    assert not outcome.passed
    assert outcome.wrong_owner == ("RM-OS.test.owner",)


def test_new_id_wrong_repo_refuses() -> None:
    claim = _claim("RM-OS.test.repo", repository_ref="some-other-repo")
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.repo",),
        base_marker_ids=frozenset(),
        claims_by_concept_id={"RM-OS.test.repo": claim},
    )
    assert not outcome.passed
    assert outcome.wrong_repo == ("RM-OS.test.repo",)


def test_expired_id_refuses() -> None:
    claim = _claim("RM-OS.test.expired", state=ConceptClaimState.EXPIRED)
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.expired",),
        base_marker_ids=frozenset(),
        claims_by_concept_id={"RM-OS.test.expired": claim},
    )
    assert not outcome.passed
    assert outcome.expired == ("RM-OS.test.expired",)


def test_released_id_refuses() -> None:
    claim = _claim("RM-OS.test.released", state=ConceptClaimState.RELEASED)
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.released",),
        base_marker_ids=frozenset(),
        claims_by_concept_id={"RM-OS.test.released": claim},
    )
    assert not outcome.passed
    assert outcome.released_or_tombstoned == ("RM-OS.test.released",)


def test_pre_existing_base_id_does_not_require_a_new_claim() -> None:
    outcome = verify_candidate_concepts(
        candidate_id="cand-1",
        repository_ref="repo-1",
        owner_ref="owner-1",
        introduced=("RM-OS.test.old",),
        base_marker_ids=frozenset({"RM-OS.test.old"}),
        claims_by_concept_id={},  # no claim needed/available and that's fine
    )
    assert outcome.passed
    assert outcome.pre_existing_ids == ("RM-OS.test.old",)


def test_generation_with_distinct_valid_claims_passes_and_preserves_attribution() -> (
    None
):
    claim_a = _claim("RM-OS.test.gen-a")
    claim_b = _claim("RM-OS.test.gen-b")
    result = verify_generation_concepts(
        generation_id="gen-1",
        introduced_by_candidate={
            "cand-a": ("RM-OS.test.gen-a",),
            "cand-b": ("RM-OS.test.gen-b",),
        },
        repository_ref="repo-1",
        owner_by_candidate={"cand-a": "owner-1", "cand-b": "owner-1"},
        base_marker_ids=frozenset(),
        claims_by_concept_id={
            "RM-OS.test.gen-a": claim_a,
            "RM-OS.test.gen-b": claim_b,
        },
    )
    assert result.passed
    # member attribution preserved: each candidate's own outcome is visible.
    assert set(result.per_candidate) == {"cand-a", "cand-b"}
    assert result.per_candidate["cand-a"].checked_ids == ("RM-OS.test.gen-a",)
    assert result.per_candidate["cand-b"].checked_ids == ("RM-OS.test.gen-b",)

    attempt = result.to_attempt_result()
    decision = decide(attempt)
    assert decision.action is DecisionAction.ACCEPT
    assert decision.accepted_member_ids == ("cand-a", "cand-b")


def test_cross_candidate_collision_refuses_and_rejects_via_bisection() -> None:
    claim = _claim("RM-OS.test.collide")
    result = verify_generation_concepts(
        generation_id="gen-2",
        introduced_by_candidate={
            "cand-a": ("RM-OS.test.collide",),
            "cand-b": ("RM-OS.test.collide",),
        },
        repository_ref="repo-1",
        owner_by_candidate={"cand-a": "owner-1", "cand-b": "owner-1"},
        base_marker_ids=frozenset(),
        claims_by_concept_id={"RM-OS.test.collide": claim},
    )
    assert not result.passed
    assert result.collisions == {"RM-OS.test.collide": ("cand-a", "cand-b")}

    attempt = result.to_attempt_result()
    decision = decide(attempt)
    # Attributable to the candidates, never treated as an opaque
    # worker/environment retry.
    assert decision.action in (DecisionAction.REJECT, DecisionAction.SPLIT)
