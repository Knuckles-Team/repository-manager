"""Typed request/record contracts consumed by RMDD-17's concept action core.

These reuse repository-manager's existing typed primitives
(:class:`~repository_manager.development.ContractModel`,
``OpaqueId``, ``GitSha``, ``GitRef``, ``UtcDateTime`` — frozen, additively
versioned, C-01-shaped) unmodified rather than inventing a second set. The
field names deliberately mirror RMDD-16's
``agent_utilities.governance.concept_reservation.ConceptReservationRequest``/
``ConceptReservationRecord`` (``tenant_ref``, ``concept_id``, ``namespace``,
``repository_ref``, ``lane_ref``, ``owner_ref``, ``request_key_ref``,
``design_ref``, ``created_at``, ``expires_at``) so a future adapter maps
between the two without a field-renaming layer — see ``client.py`` for why
that adapter is not implemented in this lane.
"""

from __future__ import annotations

from pydantic import Field, StrictInt, StrictStr, model_validator

from repository_manager.development import (
    ContractModel,
    GitRef,
    GitSha,
    OpaqueId,
    UtcDateTime,
)

from .state import ConceptClaimState, ConceptClaimVisibility

__all__ = ["ConceptClaimRecord", "ConceptClaimRequest"]


class ConceptClaimRequest(ContractModel):
    """Immutable input to a concept-claim reservation (deliverable 2).

    Carries authenticated owner, repository identity, lane, branch/base, and
    purpose/design plus WorkItem/RunTrace correlation, so a reservation is
    never anonymous or context-free.
    """

    tenant_ref: OpaqueId
    concept_id: OpaqueId
    namespace: OpaqueId
    repository_ref: OpaqueId
    lane_ref: OpaqueId
    owner_ref: OpaqueId
    request_key_ref: OpaqueId
    purpose: StrictStr
    design_ref: OpaqueId | None = None
    branch: GitRef | None = None
    base_sha: GitSha | None = None
    workitem_ref: OpaqueId | None = None
    run_trace_ref: OpaqueId | None = None
    created_at: UtcDateTime
    expires_at: UtcDateTime
    provenance_refs: tuple[OpaqueId, ...] = ()

    @model_validator(mode="after")
    def _validate_request(self) -> ConceptClaimRequest:
        if not self.concept_id.startswith(self.namespace + "."):
            raise ValueError("namespace does not contain the requested concept id")
        if self.expires_at <= self.created_at:
            raise ValueError("expires_at must be after created_at")
        if not self.purpose.strip():
            raise ValueError("purpose must be non-blank")
        return self


class ConceptClaimRecord(ContractModel):
    """Durable claim record returned/queried through the action core."""

    reservation_id: OpaqueId
    concept_id: OpaqueId
    tenant_ref: OpaqueId
    repository_ref: OpaqueId
    lane_ref: OpaqueId
    owner_ref: OpaqueId
    state: ConceptClaimState
    visibility: ConceptClaimVisibility
    fence: StrictInt = Field(ge=1)
    created_at: UtcDateTime
    expires_at: UtcDateTime
    transitioned_at: UtcDateTime
    candidate_id: OpaqueId | None = None
    generation_id: OpaqueId | None = None
