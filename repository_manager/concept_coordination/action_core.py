"""MCP/CLI-neutral concept coordination action core (deliverable 1).

One class, ``ConceptCoordinationActions``, implements
reserve/list/get/release/materialize/verify/reconcile against an injected
:class:`~repository_manager.concept_coordination.port.ConceptAuthorityPort`.
It contains no MCP tool registration and no CLI argument parsing — those
entrypoints are RMDD-20's owned seam (lane brief, "Forbidden edits": "MCP/CLI
entrypoints ... do not register tools there"). RMDD-20 constructs this class
with a real injected authority and exposes its methods.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from .client import resolve_default_authority
from .errors import ConceptClaimConflict, ConceptClaimUnauthorized
from .fragments import regenerate_view, write_fragment_row
from .lifecycle import abort_claims, land_claims
from .models import ConceptClaimRecord, ConceptClaimRequest
from .port import ConceptAuthorityPort
from .reconcile import ConceptReconciliationReport
from .reconcile import reconcile as reconcile_claims
from .scanner import diff_new_concept_ids, scan_tree_concept_markers
from .state import ConceptClaimState
from .verifier import (
    ConceptVerificationOutcome,
    GenerationConceptVerification,
    verify_candidate_concepts,
    verify_generation_concepts,
)

if TYPE_CHECKING:
    from repository_manager.development import Candidate, Generation

__all__ = ["ConceptCoordinationActions"]

_MAX_LIST_PAGES = 1024


class ConceptCoordinationActions:
    """Concept-claim lifecycle actions scoped to one repository/lane/tenant."""

    def __init__(
        self,
        *,
        repo_root: Path,
        tenant_ref: str,
        lane_ref: str,
        authority: ConceptAuthorityPort | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self.tenant_ref = tenant_ref
        self.lane_ref = lane_ref
        self._injected_authority = authority

    def _authority(self) -> ConceptAuthorityPort:
        if self._injected_authority is not None:
            return self._injected_authority
        return resolve_default_authority()

    # -- reserve / materialize -------------------------------------------------

    def reserve(self, request: ConceptClaimRequest) -> ConceptClaimRecord:
        """Create-if-absent a claim; idempotent on repeated request key."""

        record = self._authority().reserve(request)
        write_fragment_row(self.repo_root, self.lane_ref, record.canonical_payload())
        regenerate_view(self.repo_root)
        return record

    def materialize(
        self, reservation_id: str, *, owner_ref: str, expected_fence: int
    ) -> ConceptClaimRecord:
        """Advance RESERVED -> MATERIALIZED; idempotent if already there."""

        authority = self._authority()
        current = authority.get(reservation_id, tenant_ref=self.tenant_ref)
        if current.owner_ref != owner_ref:
            raise ConceptClaimUnauthorized(
                f"concept reservation {reservation_id} belongs to another owner"
            )
        if current.state is ConceptClaimState.MATERIALIZED:
            record = current
        elif current.state is ConceptClaimState.RESERVED:
            record = authority.transition(
                reservation_id,
                tenant_ref=self.tenant_ref,
                owner_ref=owner_ref,
                expected_fence=expected_fence,
                target=ConceptClaimState.MATERIALIZED,
            )
        else:
            raise ConceptClaimConflict(
                f"cannot materialize a {current.state.value} reservation"
            )
        write_fragment_row(self.repo_root, self.lane_ref, record.canonical_payload())
        regenerate_view(self.repo_root)
        return record

    # -- read --------------------------------------------------------------

    def get(self, reservation_id: str) -> ConceptClaimRecord:
        return self._authority().get(reservation_id, tenant_ref=self.tenant_ref)

    def list(
        self,
        *,
        namespace: str | None = None,
        state: ConceptClaimState | None = None,
        concept_prefix: str | None = None,
        limit: int = 1000,
        cursor: str | None = None,
    ) -> tuple[list[ConceptClaimRecord], str | None]:
        return self._authority().list(
            tenant_ref=self.tenant_ref,
            namespace=namespace,
            state=state,
            concept_prefix=concept_prefix,
            limit=limit,
            cursor=cursor,
        )

    def _list_all(self, **kwargs: object) -> Sequence[ConceptClaimRecord]:
        # NOTE: annotated ``Sequence``, not ``list`` — this class also defines
        # a method named ``list``, which shadows the builtin for every
        # class-body-level annotation written after it (mypy [valid-type]).
        rows: list[ConceptClaimRecord] = []
        cursor: str | None = None
        for _ in range(_MAX_LIST_PAGES):
            page, cursor = self.list(cursor=cursor, **kwargs)  # type: ignore[arg-type]
            rows.extend(page)
            if cursor is None:
                return rows
        raise ConceptClaimConflict(
            "concept claim listing did not terminate within the bounded page count"
        )

    # -- release / land ------------------------------------------------------

    def release(
        self, reservation_id: str, *, owner_ref: str, expected_fence: int
    ) -> ConceptClaimRecord:
        """Release a never-visible claim; refuses on a landed/visible one."""

        authority = self._authority()
        current = authority.get(reservation_id, tenant_ref=self.tenant_ref)
        (released,) = abort_claims(
            authority,
            tenant_ref=self.tenant_ref,
            owner_ref=owner_ref,
            records=[current],
            repo_root=self.repo_root,
            lane_ref=self.lane_ref,
        )
        regenerate_view(self.repo_root)
        return released

    def land(
        self, reservation_id: str, *, owner_ref: str, expected_fence: int
    ) -> ConceptClaimRecord:
        """Mark one claim permanent (deliverable 6, single-claim form)."""

        authority = self._authority()
        current = authority.get(reservation_id, tenant_ref=self.tenant_ref)
        (landed,) = land_claims(
            authority,
            tenant_ref=self.tenant_ref,
            owner_ref=owner_ref,
            records=[current],
            repo_root=self.repo_root,
            lane_ref=self.lane_ref,
        )
        regenerate_view(self.repo_root)
        return landed

    # -- verify (deliverable 5) ---------------------------------------------

    def verify_candidate(
        self, candidate: Candidate, *, repo_path: Path
    ) -> ConceptVerificationOutcome:
        """Verify one candidate's introduced concept markers against claims."""

        scan = diff_new_concept_ids(
            repo_path,
            base_sha=candidate.base_sha,
            candidate_sha=candidate.candidate_sha,
        )
        claims_by_id: dict[str, ConceptClaimRecord] = {}
        for concept_id in scan.introduced:
            try:
                records, _ = self.list(concept_prefix=concept_id, limit=1)
            except ConceptClaimConflict:
                records = []
            for record in records:
                if record.concept_id == concept_id:
                    claims_by_id[concept_id] = record
        return verify_candidate_concepts(
            candidate_id=candidate.candidate_id,
            repository_ref=candidate.repository.repository_id,
            owner_ref=candidate.owner_id,
            introduced=scan.introduced,
            base_marker_ids=frozenset(),
            claims_by_concept_id=claims_by_id,
        )

    def verify_generation(
        self,
        generation: Generation,
        candidates: Sequence[Candidate],
        *,
        repo_path: Path,
    ) -> GenerationConceptVerification:
        """Verify every candidate in one sealed generation."""

        introduced_by_candidate: dict[str, tuple[str, ...]] = {}
        owner_by_candidate: dict[str, str] = {}
        claims_by_id: dict[str, ConceptClaimRecord] = {}
        for candidate in candidates:
            scan = diff_new_concept_ids(
                repo_path,
                base_sha=candidate.base_sha,
                candidate_sha=candidate.candidate_sha,
            )
            introduced_by_candidate[candidate.candidate_id] = scan.introduced
            owner_by_candidate[candidate.candidate_id] = candidate.owner_id
            for concept_id in scan.introduced:
                try:
                    records, _ = self.list(concept_prefix=concept_id, limit=1)
                except ConceptClaimConflict:
                    records = []
                for record in records:
                    if record.concept_id == concept_id:
                        claims_by_id[concept_id] = record
        return verify_generation_concepts(
            generation_id=generation.generation_id,
            introduced_by_candidate=introduced_by_candidate,
            repository_ref=generation.repository.repository_id,
            owner_by_candidate=owner_by_candidate,
            base_marker_ids=frozenset(),
            claims_by_concept_id=claims_by_id,
        )

    # -- reconcile (deliverable 7) -------------------------------------------

    def reconcile(
        self, *, source_tree_ish: str = "HEAD"
    ) -> ConceptReconciliationReport:
        """Read-only reconciliation across authority, fragments, view, source."""

        central = self._list_all()
        source_markers = frozenset(
            scan_tree_concept_markers(self.repo_root, source_tree_ish)
        )
        return reconcile_claims(
            repo_root=self.repo_root,
            central_claims=central,
            source_marker_ids=source_markers,
        )
