"""Action-core lifecycle tests: reserve/materialize/release/land + refusals.

Uses the injected :class:`FakeConceptAuthority` test double (never the real
RMDD-16 authority — see ``client.py`` for why that is not importable in this
environment). Required-evidence tests from the lane brief:

* two lanes/hosts requesting the same ID -> one claim, one actionable refusal;
* repeated reserve/materialize is idempotent (across a fresh action-core
  instance, standing in for "across restart" — the fake authority instance is
  the durable state, exactly as the real authority would be);
* abort cannot release a landed/visible claim;
* stale fence cannot transition a claim.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from repository_manager.concept_coordination.action_core import (
    ConceptCoordinationActions,
)
from repository_manager.concept_coordination.errors import (
    ConceptClaimConflict,
    ConceptClaimFenceConflict,
    ConceptClaimIdUnavailable,
)
from repository_manager.concept_coordination.models import ConceptClaimRequest
from repository_manager.concept_coordination.state import ConceptClaimState
from tests.concept_coordination.fakes import FakeConceptAuthority

_NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _request(
    *,
    concept_id: str,
    request_key_ref: str,
    owner_ref: str = "pref_owner_" + "a" * 64,
    repository_ref: str = "pref_repo_" + "b" * 64,
    lane_ref: str = "lane-a",
    tenant_ref: str = "tenant-1",
    namespace: str | None = None,
) -> ConceptClaimRequest:
    return ConceptClaimRequest(
        tenant_ref=tenant_ref,
        concept_id=concept_id,
        namespace=namespace or concept_id.rsplit(".", 1)[0],
        repository_ref=repository_ref,
        lane_ref=lane_ref,
        owner_ref=owner_ref,
        request_key_ref=request_key_ref,
        purpose="test claim",
        created_at=_NOW,
        expires_at=_NOW + timedelta(hours=1),
    )


def _actions(
    tmp_path: Path, *, lane_ref: str, authority: FakeConceptAuthority
) -> ConceptCoordinationActions:
    return ConceptCoordinationActions(
        repo_root=tmp_path,
        tenant_ref="tenant-1",
        lane_ref=lane_ref,
        authority=authority,
    )


def test_reserve_and_materialize_are_idempotent(tmp_path: Path) -> None:
    authority = FakeConceptAuthority()
    actions = _actions(tmp_path, lane_ref="lane-a", authority=authority)
    request = _request(concept_id="RM-OS.test.alpha", request_key_ref="req-1")

    first = actions.reserve(request)
    second = actions.reserve(request)  # same request key + concept id: replay
    assert first.reservation_id == second.reservation_id
    assert first.state is ConceptClaimState.RESERVED

    materialized_once = actions.materialize(
        first.reservation_id, owner_ref=request.owner_ref, expected_fence=first.fence
    )
    # "across restart": a fresh ConceptCoordinationActions bound to the same
    # durable authority instance, exactly as a real restart reconnects to the
    # same durable graph-os authority.
    restarted = _actions(tmp_path, lane_ref="lane-a", authority=authority)
    materialized_again = restarted.materialize(
        first.reservation_id, owner_ref=request.owner_ref, expected_fence=first.fence
    )
    assert materialized_once.reservation_id == materialized_again.reservation_id
    assert materialized_again.state is ConceptClaimState.MATERIALIZED

    fragment_path = tmp_path / "docs" / "concept_claims.d" / "lane-a.yaml"
    assert fragment_path.exists()
    view_path = tmp_path / "docs" / "concept_claims.yaml"
    assert view_path.exists()


def test_two_lanes_same_id_one_claim_one_refusal(tmp_path: Path) -> None:
    authority = FakeConceptAuthority()
    lane_a = _actions(tmp_path, lane_ref="lane-a", authority=authority)
    lane_b = _actions(tmp_path, lane_ref="lane-b", authority=authority)

    request_a = _request(
        concept_id="RM-OS.test.dup",
        request_key_ref="req-a",
        owner_ref="pref_owner_" + "a" * 64,
    )
    request_b = _request(
        concept_id="RM-OS.test.dup",
        request_key_ref="req-b",
        owner_ref="pref_owner_" + "c" * 64,
        lane_ref="lane-b",
    )

    winner = lane_a.reserve(request_a)
    assert winner.concept_id == "RM-OS.test.dup"

    with pytest.raises(ConceptClaimIdUnavailable) as excinfo:
        lane_b.reserve(request_b)
    assert "RM-OS.test.dup" in str(excinfo.value)


def test_stale_fence_cannot_transition(tmp_path: Path) -> None:
    authority = FakeConceptAuthority()
    actions = _actions(tmp_path, lane_ref="lane-a", authority=authority)
    request = _request(concept_id="RM-OS.test.fence", request_key_ref="req-fence")
    record = actions.reserve(request)

    with pytest.raises(ConceptClaimFenceConflict):
        authority.transition(
            record.reservation_id,
            tenant_ref="tenant-1",
            owner_ref=request.owner_ref,
            expected_fence=999,  # stale/wrong fence
            target=ConceptClaimState.MATERIALIZED,
        )


def test_abort_cannot_release_a_landed_claim(tmp_path: Path) -> None:
    authority = FakeConceptAuthority()
    actions = _actions(tmp_path, lane_ref="lane-a", authority=authority)
    request = _request(concept_id="RM-OS.test.landed", request_key_ref="req-landed")
    record = actions.reserve(request)
    materialized = actions.materialize(
        record.reservation_id, owner_ref=request.owner_ref, expected_fence=record.fence
    )
    landed = actions.land(
        record.reservation_id,
        owner_ref=request.owner_ref,
        expected_fence=materialized.fence,
    )
    assert landed.state is ConceptClaimState.LANDED

    with pytest.raises(ConceptClaimConflict) as excinfo:
        actions.release(
            record.reservation_id,
            owner_ref=request.owner_ref,
            expected_fence=landed.fence,
        )
    assert "landed" in str(excinfo.value)


def test_release_of_a_reserved_claim_succeeds_and_can_be_reclaimed(
    tmp_path: Path,
) -> None:
    authority = FakeConceptAuthority()
    actions = _actions(tmp_path, lane_ref="lane-a", authority=authority)
    request = _request(concept_id="RM-OS.test.releaseme", request_key_ref="req-rel")
    record = actions.reserve(request)

    released = actions.release(
        record.reservation_id, owner_ref=request.owner_ref, expected_fence=record.fence
    )
    assert released.state is ConceptClaimState.RELEASED

    # A released id becomes available again through a NEW request (never a
    # silent reuse of the old reservation_id/fence).
    another = _request(
        concept_id="RM-OS.test.releaseme",
        request_key_ref="req-rel-2",
        owner_ref="pref_owner_" + "d" * 64,
    )
    reclaimed = actions.reserve(another)
    assert reclaimed.reservation_id != record.reservation_id
