"""Test-only fake concept authority.

**This is test scaffolding, never production code.** It behaviorally mirrors
the documented contract of RMDD-16's own
``agent_utilities.governance.concept_reservation.FixtureConceptReservationAuthority``
(create-if-absent uniqueness, request-key idempotency scoped to
``(tenant_ref, request_key_ref, concept_id)``, fenced same-owner-idempotent
transitions, the RESERVED/MATERIALIZED/LANDED/RELEASED/EXPIRED/TOMBSTONED
state machine) — read directly from that module's source at commit
``bbb09765`` (see ``repository_manager/concept_coordination/client.py`` for
why that module cannot be imported here). It advertises
``authoritative = False`` for the same reason RMDD-16's own fixture does: a
thread-local dict is not cross-host coordination, and nothing in this
codebase may treat it as one.

RMDD-17's production code (``client.py``) never imports or constructs this
class — only tests do.
"""

from __future__ import annotations

import threading
from datetime import UTC, datetime

from repository_manager.concept_coordination.errors import (
    ConceptClaimFenceConflict,
    ConceptClaimIdUnavailable,
    ConceptClaimNotFound,
    ConceptClaimUnauthorized,
)
from repository_manager.concept_coordination.models import (
    ConceptClaimRecord,
    ConceptClaimRequest,
)
from repository_manager.concept_coordination.state import (
    VISIBILITY_RANK,
    ConceptClaimState,
    ConceptClaimVisibility,
    is_legal_claim_transition,
    minimum_visibility_for_target,
    strongest_visibility,
)

__all__ = ["FakeConceptAuthority"]


class FakeConceptAuthority:
    """Thread-safe in-memory double; never a production/cross-host authority."""

    authoritative = False

    def __init__(self) -> None:
        self._records: dict[str, ConceptClaimRecord] = {}
        self._by_request_key: dict[tuple[str, str, str], str] = {}
        self._lock = threading.RLock()

    def _require(self, reservation_id: str, tenant_ref: str) -> ConceptClaimRecord:
        record = self._records.get(reservation_id)
        if record is None:
            raise ConceptClaimNotFound(
                f"concept reservation {reservation_id} was not found"
            )
        if record.tenant_ref != tenant_ref:
            raise ConceptClaimUnauthorized(
                f"concept reservation {reservation_id} belongs to another tenant"
            )
        return record

    def reserve(self, request: ConceptClaimRequest) -> ConceptClaimRecord:
        with self._lock:
            key = (request.tenant_ref, request.request_key_ref, request.concept_id)
            previous_id = self._by_request_key.get(key)
            if previous_id:
                previous = self._records[previous_id]
                if previous.concept_id != request.concept_id:
                    raise ConceptClaimIdUnavailable(
                        "request key was reused with a different concept id"
                    )
                return previous
            for existing in self._records.values():
                if existing.concept_id == request.concept_id and existing.state not in (
                    ConceptClaimState.RELEASED,
                    ConceptClaimState.EXPIRED,
                ):
                    raise ConceptClaimIdUnavailable(
                        f"concept id is already claimed: {request.concept_id}"
                    )
            reservation_id = (
                f"concept-reservation:{request.tenant_ref}:{len(self._records) + 1}"
            )
            record = ConceptClaimRecord(
                reservation_id=reservation_id,
                concept_id=request.concept_id,
                tenant_ref=request.tenant_ref,
                repository_ref=request.repository_ref,
                lane_ref=request.lane_ref,
                owner_ref=request.owner_ref,
                state=ConceptClaimState.RESERVED,
                visibility=ConceptClaimVisibility.PRIVATE,
                fence=1,
                created_at=request.created_at,
                expires_at=request.expires_at,
                transitioned_at=request.created_at,
            )
            self._records[reservation_id] = record
            self._by_request_key[key] = reservation_id
            return record

    def get(self, reservation_id: str, *, tenant_ref: str) -> ConceptClaimRecord:
        with self._lock:
            return self._require(reservation_id, tenant_ref)

    def list(
        self,
        *,
        tenant_ref: str,
        namespace: str | None = None,
        state: ConceptClaimState | None = None,
        concept_prefix: str | None = None,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> tuple[list[ConceptClaimRecord], str | None]:
        offset = int(cursor) if cursor else 0
        with self._lock:
            rows = [r for r in self._records.values() if r.tenant_ref == tenant_ref]
            if state is not None:
                rows = [r for r in rows if r.state is state]
            if concept_prefix is not None:
                rows = [r for r in rows if r.concept_id.startswith(concept_prefix)]
            rows.sort(key=lambda r: r.reservation_id)
            page = rows[offset : offset + limit]
            next_cursor = str(offset + limit) if offset + limit < len(rows) else None
            return page, next_cursor

    def transition(
        self,
        reservation_id: str,
        *,
        tenant_ref: str,
        owner_ref: str,
        expected_fence: int,
        target: ConceptClaimState,
        visibility: ConceptClaimVisibility | None = None,
    ) -> ConceptClaimRecord:
        """Fenced lifecycle mutation with monotonic visibility promotion.

        Mirrors RMDD-16's authority exactly (``bbb09765``): visibility never
        regresses (``strongest_visibility``), reaching MATERIALIZED/LANDED
        floors it at FRAGMENT/REPOSITORY, and — critically — a caller-supplied
        ``visibility`` that already reaches REPOSITORY or above forces the
        *effective* target to TOMBSTONED regardless of what ``target`` asked
        for: once a claim is externally/repository visible it is terminal,
        never merely "landed but still mutable".
        """

        with self._lock:
            record = self._require(reservation_id, tenant_ref)
            requested_visibility = strongest_visibility(
                record.visibility,
                visibility if visibility is not None else record.visibility,
            )
            effective_target = target
            if (
                target is not ConceptClaimState.TOMBSTONED
                and VISIBILITY_RANK[requested_visibility]
                >= VISIBILITY_RANK[ConceptClaimVisibility.REPOSITORY]
            ):
                effective_target = ConceptClaimState.TOMBSTONED
            # Same-owner idempotent replay: caller presents the fence one
            # past the transition it already caused.
            if (
                record.owner_ref == owner_ref
                and record.state is effective_target
                and record.fence == expected_fence + 1
            ):
                return record
            if record.owner_ref != owner_ref or record.fence != expected_fence:
                raise ConceptClaimFenceConflict(
                    f"concept reservation {reservation_id} owner or fence is stale"
                )
            if not is_legal_claim_transition(record.state, effective_target):
                raise ConceptClaimFenceConflict(
                    f"cannot transition {record.state.value} to {effective_target.value}"
                )
            minimum = minimum_visibility_for_target(effective_target)
            next_visibility = (
                strongest_visibility(requested_visibility, minimum)
                if minimum is not None
                else requested_visibility
            )
            updated = record.model_copy(
                update={
                    "state": effective_target,
                    "visibility": next_visibility,
                    "fence": record.fence + 1,
                    "transitioned_at": datetime.now(UTC),
                }
            )
            self._records[reservation_id] = updated
            return updated
