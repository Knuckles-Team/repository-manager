"""Visibility promotion on transition — mirrors RMDD-16's authority exactly.

Found during root review: ``visibility`` was accepted by both
``ConceptAuthorityPort.transition`` and ``FakeConceptAuthority.transition``
but never applied (vulture 100%-confidence unused-variable on both). That was
a real functional gap, not decoration: RMDD-16's authority (commit
``bbb09765``) never lets visibility regress, floors it on
MATERIALIZED/LANDED, and forces the *effective* target to TOMBSTONED the
moment requested visibility reaches REPOSITORY or above — "landed/externally
visible/tombstoned ID is never reused" (RMDD-16 required test list) depends
on exactly this rule. These tests prove the fix is real behavior, not merely
a name reference that satisfies a linter.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from repository_manager.concept_coordination.models import ConceptClaimRequest
from repository_manager.concept_coordination.state import (
    ConceptClaimState,
    ConceptClaimVisibility,
)
from tests.concept_coordination.fakes import FakeConceptAuthority

_NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)


def _request(concept_id: str, request_key_ref: str) -> ConceptClaimRequest:
    return ConceptClaimRequest(
        tenant_ref="tenant-1",
        concept_id=concept_id,
        namespace=concept_id.rsplit(".", 1)[0],
        repository_ref="repo-1",
        lane_ref="lane-a",
        owner_ref="owner-1",
        request_key_ref=request_key_ref,
        purpose="test claim",
        created_at=_NOW,
        expires_at=_NOW + timedelta(hours=1),
    )


def test_materialize_floors_visibility_at_fragment() -> None:
    authority = FakeConceptAuthority()
    record = authority.reserve(_request("RM-OS.test.vis-materialize", "req-1"))
    assert record.visibility is ConceptClaimVisibility.PRIVATE

    materialized = authority.transition(
        record.reservation_id,
        tenant_ref="tenant-1",
        owner_ref="owner-1",
        expected_fence=record.fence,
        target=ConceptClaimState.MATERIALIZED,
    )
    assert materialized.visibility is ConceptClaimVisibility.FRAGMENT


def test_land_floors_visibility_at_repository() -> None:
    authority = FakeConceptAuthority()
    record = authority.reserve(_request("RM-OS.test.vis-land", "req-2"))
    materialized = authority.transition(
        record.reservation_id,
        tenant_ref="tenant-1",
        owner_ref="owner-1",
        expected_fence=record.fence,
        target=ConceptClaimState.MATERIALIZED,
    )
    landed = authority.transition(
        record.reservation_id,
        tenant_ref="tenant-1",
        owner_ref="owner-1",
        expected_fence=materialized.fence,
        target=ConceptClaimState.LANDED,
    )
    assert landed.visibility is ConceptClaimVisibility.REPOSITORY


def test_repository_visibility_forces_effective_target_to_tombstoned() -> None:
    """A claim already externally/repository-visible is never released back."""

    authority = FakeConceptAuthority()
    record = authority.reserve(_request("RM-OS.test.vis-escalate", "req-3"))
    materialized = authority.transition(
        record.reservation_id,
        tenant_ref="tenant-1",
        owner_ref="owner-1",
        expected_fence=record.fence,
        target=ConceptClaimState.MATERIALIZED,
    )

    # Caller asks to RELEASE it, but also asserts REPOSITORY visibility
    # (e.g. reconciliation found the marker already landed in source). The
    # authority must never honor "release" once that visible — it silently
    # upgrades the effective target to TOMBSTONED instead.
    result = authority.transition(
        record.reservation_id,
        tenant_ref="tenant-1",
        owner_ref="owner-1",
        expected_fence=materialized.fence,
        target=ConceptClaimState.RELEASED,
        visibility=ConceptClaimVisibility.REPOSITORY,
    )
    assert result.state is ConceptClaimState.TOMBSTONED
    assert result.visibility is ConceptClaimVisibility.EXTERNAL


def test_visibility_never_regresses_on_a_lower_explicit_value() -> None:
    authority = FakeConceptAuthority()
    record = authority.reserve(_request("RM-OS.test.vis-monotonic", "req-4"))
    materialized = authority.transition(
        record.reservation_id,
        tenant_ref="tenant-1",
        owner_ref="owner-1",
        expected_fence=record.fence,
        target=ConceptClaimState.MATERIALIZED,
        visibility=ConceptClaimVisibility.PRIVATE,  # weaker than the FRAGMENT floor
    )
    assert materialized.visibility is ConceptClaimVisibility.FRAGMENT
