"""RMDD-17 — repository concept tools and candidate/merge concept verification.

Public surface for the action core, verifier, scanner, reconciliation, and
fragment/audit-projection helpers. This package registers no MCP tool and no
CLI command (RMDD-20 owns those entrypoints) — it is imported directly by a
caller (an entrypoint, or a test) that constructs
:class:`ConceptCoordinationActions` with an injected
:class:`~repository_manager.concept_coordination.port.ConceptAuthorityPort`.

See ``action_core.py``'s and ``client.py``'s module docstrings for the exact
frozen interfaces this package consumes unmodified (RMDD-09's
``LaneRecord``/``lane_id_for``, RMDD-12's ``Candidate``/``Generation``/
``AttemptResult``/``decide``, RMDD-16's documented but not-yet-``main``
``ConceptReservationAuthority`` contract) and the residuals this lane leaves
for RMDD-12/RMDD-16/RMDD-20/RMDD-23.
"""

from __future__ import annotations

from .action_core import ConceptCoordinationActions
from .client import AUTHORITY_MODULE, resolve_default_authority
from .errors import (
    ConceptAuthorityUnavailable,
    ConceptClaimConflict,
    ConceptClaimFenceConflict,
    ConceptClaimIdUnavailable,
    ConceptClaimNotFound,
    ConceptClaimUnauthorized,
    ConceptCoordinationError,
    ConceptVerificationRefused,
)
from .models import ConceptClaimRecord, ConceptClaimRequest
from .port import ConceptAuthorityPort
from .reconcile import ConceptReconciliationReport, reconcile
from .scanner import (
    ConceptScanResult,
    diff_new_concept_ids,
    extract_markers,
    find_cross_candidate_collisions,
    scan_tree_concept_markers,
)
from .state import ConceptClaimState, ConceptClaimVisibility, is_legal_claim_transition
from .verifier import (
    ConceptVerificationOutcome,
    GenerationConceptVerification,
    verify_candidate_concepts,
    verify_generation_concepts,
)

__all__ = [
    "AUTHORITY_MODULE",
    "ConceptAuthorityPort",
    "ConceptAuthorityUnavailable",
    "ConceptClaimConflict",
    "ConceptClaimFenceConflict",
    "ConceptClaimIdUnavailable",
    "ConceptClaimNotFound",
    "ConceptClaimRecord",
    "ConceptClaimRequest",
    "ConceptClaimState",
    "ConceptClaimUnauthorized",
    "ConceptClaimVisibility",
    "ConceptCoordinationActions",
    "ConceptCoordinationError",
    "ConceptReconciliationReport",
    "ConceptScanResult",
    "ConceptVerificationOutcome",
    "ConceptVerificationRefused",
    "GenerationConceptVerification",
    "diff_new_concept_ids",
    "extract_markers",
    "find_cross_candidate_collisions",
    "is_legal_claim_transition",
    "reconcile",
    "resolve_default_authority",
    "scan_tree_concept_markers",
    "verify_candidate_concepts",
    "verify_generation_concepts",
]
