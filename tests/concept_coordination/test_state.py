"""Prove the local state mirror cannot silently drift from RMDD-16's authority.

The authority (:mod:`repository_manager.governance.concept_reservation`) lives
in this package since OQ-3, so the comparison always runs against the real enum.
"""

from __future__ import annotations

from repository_manager.concept_coordination.state import (
    ConceptClaimState,
    ConceptClaimVisibility,
)
from repository_manager.governance import concept_reservation


def test_claim_state_values_match_the_real_authority() -> None:
    real_state = concept_reservation.ConceptReservationState
    real_visibility = concept_reservation.ConceptReservationVisibility
    assert {member.value for member in ConceptClaimState} == {
        member.value for member in real_state
    }
    assert {member.value for member in ConceptClaimVisibility} == {
        member.value for member in real_visibility
    }
