"""Read-only reconciliation across authority, fragments, view, and source (deliverable 7).

Compares central-authority claims, this repository's fragment projection,
the generated folded view, and the concept markers actually present in
source, and classifies drift. **Never mutates, never reallocates, never
reconciles by minting or reusing an ID** — a mismatch is reported, not
silently repaired, matching RMDD-16's own ``reconcile_projection`` contract
("never write") and the lane brief's "no ID reuse/fallback weakens RMDD-16
policy."
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .fragments import fold_fragments, read_fragments, verify_generated_view_is_fold
from .models import ConceptClaimRecord
from .state import ConceptClaimState

__all__ = ["ConceptReconciliationReport", "reconcile"]

_PERMANENT_STATES = frozenset({ConceptClaimState.LANDED, ConceptClaimState.TOMBSTONED})


@dataclass(frozen=True)
class ConceptReconciliationReport:
    """Classification result; every field is evidence, never an action taken."""

    matches: tuple[str, ...]
    missing_fragment: tuple[str, ...]
    orphan_fragment: tuple[str, ...]
    used_but_unclaimed: tuple[str, ...]
    landed_not_permanent: tuple[str, ...]
    generated_view_matches_fold: bool
    generated_view_drift_reasons: tuple[str, ...]

    @property
    def clean(self) -> bool:
        return (
            not self.missing_fragment
            and not self.orphan_fragment
            and not self.used_but_unclaimed
            and not self.landed_not_permanent
            and self.generated_view_matches_fold
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "matches": list(self.matches),
            "missing_fragment": list(self.missing_fragment),
            "orphan_fragment": list(self.orphan_fragment),
            "used_but_unclaimed": list(self.used_but_unclaimed),
            "landed_not_permanent": list(self.landed_not_permanent),
            "generated_view_matches_fold": self.generated_view_matches_fold,
            "generated_view_drift_reasons": list(self.generated_view_drift_reasons),
            "clean": self.clean,
        }


def reconcile(
    *,
    repo_root: Path,
    central_claims: Iterable[ConceptClaimRecord],
    source_marker_ids: frozenset[str],
) -> ConceptReconciliationReport:
    """Compare central claims, the fragment projection, the view, and source.

    ``source_marker_ids`` is a repository-wide scan (not a base/candidate
    diff — see :mod:`~repository_manager.concept_coordination.scanner` for
    that), typically the union of ``scan_tree_concept_markers`` over the
    current landed target.
    """

    fragments = fold_fragments(read_fragments(repo_root))
    fragment_ids = {row["concept_id"] for row in fragments}
    central_by_id = {record.concept_id: record for record in central_claims}
    central_ids = set(central_by_id)

    missing_fragment = sorted(central_ids - fragment_ids)
    orphan_fragment = sorted(fragment_ids - central_ids)
    used_but_unclaimed = sorted(source_marker_ids - central_ids)
    landed_not_permanent = sorted(
        concept_id
        for concept_id in source_marker_ids & central_ids
        if central_by_id[concept_id].state not in _PERMANENT_STATES
    )
    drift_free_ids = (
        central_ids
        - set(missing_fragment)
        - set(orphan_fragment)
        - set(landed_not_permanent)
    )
    matches = sorted(
        concept_id for concept_id in drift_free_ids if concept_id in fragment_ids
    )

    view_ok, view_reasons = verify_generated_view_is_fold(repo_root)

    return ConceptReconciliationReport(
        matches=tuple(matches),
        missing_fragment=tuple(missing_fragment),
        orphan_fragment=tuple(orphan_fragment),
        used_but_unclaimed=tuple(used_but_unclaimed),
        landed_not_permanent=tuple(landed_not_permanent),
        generated_view_matches_fold=view_ok,
        generated_view_drift_reasons=tuple(view_reasons),
    )
