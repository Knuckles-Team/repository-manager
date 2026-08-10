"""Prove the local state mirror cannot silently drift from RMDD-16's authority.

Skips (with a named reason, never a silent pass) when
``agent_utilities.governance.concept_reservation`` is not importable — which
is the real state of ``agent-utilities`` ``main`` as of this lane. The moment
that module lands on both repositories' ``main`` branches, this test starts
asserting real equality instead of skipping.
"""

from __future__ import annotations

import importlib

import pytest

from repository_manager.concept_coordination.state import (
    ConceptClaimState,
    ConceptClaimVisibility,
)


def test_claim_state_values_match_the_real_authority_when_importable() -> None:
    try:
        module = importlib.import_module(
            "agent_utilities.governance.concept_reservation"
        )
    except ImportError:
        pytest.skip(
            "agent_utilities.governance.concept_reservation is not on this "
            "agent-utilities checkout's main (see client.py docstring for the "
            "exact commits) — cannot compare against the real enum yet."
        )
        return
    real_state = module.ConceptReservationState
    real_visibility = module.ConceptReservationVisibility
    assert {member.value for member in ConceptClaimState} == {
        member.value for member in real_state
    }
    assert {member.value for member in ConceptClaimVisibility} == {
        member.value for member in real_visibility
    }
