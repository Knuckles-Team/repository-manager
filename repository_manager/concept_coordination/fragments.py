"""Repository-side concept-claim fragments: the audit projection, not a ledger.

Mirrors the append-only-fragment / folded-generated-view pattern already
established for concept markers in
``agent_utilities.governance.concept_allocator`` (per-lane fragment +
``regenerate_view()``) and for merge-queue records via
``agent_utilities.governance.lanes.FragmentStore``. Repository-manager has no
concept-claim ledger of its own on ``main`` today (verified: no
``docs/concept_reservations*`` path exists in this repository — only
agent-utilities has one, for its own markers), so this module creates one
scoped to *this* repository, deliberately named ``concept_claims`` (not
``concept_reservations``) to avoid any confusion with agent-utilities' own
file of that name in a different repository.

**This is a read/audit projection of RMDD-16's central authority, never a
second source of truth or a fallback allocator.** A fragment row records what
a claim *result* was (as returned by the injected
:class:`~repository_manager.concept_coordination.port.ConceptAuthorityPort`);
nothing here decides whether an ID is available. Each lane appends only to
its own fragment file (``docs/concept_claims.d/<lane>.yaml``); the generated
view (``docs/concept_claims.yaml``) is always a pure fold of the fragments,
regenerated, never hand-edited — ``verify_generated_view_is_fold`` proves
drift is detectable.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import yaml

from .state import ConceptClaimState

__all__ = [
    "fold_fragments",
    "fragment_dir",
    "read_fragments",
    "regenerate_view",
    "verify_generated_view_is_fold",
    "view_path",
    "write_fragment_row",
]

_FRAGMENT_DIRNAME = "concept_claims.d"
_VIEW_FILENAME = "concept_claims.yaml"

# Higher rank wins when folding two rows for the same concept_id: a landed
# claim is never displaced by a stale "reserved" row re-read from a slower
# fragment file. Mirrors RMDD-16's own state ordering; TOMBSTONED is terminal.
_STATE_RANK: dict[str, int] = {
    ConceptClaimState.RESERVED.value: 0,
    ConceptClaimState.MATERIALIZED.value: 1,
    ConceptClaimState.EXPIRED.value: 1,
    ConceptClaimState.RELEASED.value: 1,
    ConceptClaimState.LANDED.value: 2,
    ConceptClaimState.TOMBSTONED.value: 3,
}


def fragment_dir(repo_root: Path) -> Path:
    return Path(repo_root) / "docs" / _FRAGMENT_DIRNAME


def view_path(repo_root: Path) -> Path:
    return Path(repo_root) / "docs" / _VIEW_FILENAME


def _load_yaml_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    if not loaded:
        return []
    if not isinstance(loaded, list):
        raise ValueError(f"{path} must contain a YAML list of claim rows")
    return list(loaded)


def write_fragment_row(repo_root: Path, lane_ref: str, row: Mapping[str, Any]) -> Path:
    """Append one row to *this lane's own* fragment file only.

    Never touches another lane's fragment file and never rewrites the
    generated view directly — call :func:`regenerate_view` afterward.
    """

    if not lane_ref or "/" in lane_ref or "\\" in lane_ref:
        raise ValueError("lane_ref must be a bare, path-safe identifier")
    directory = fragment_dir(repo_root)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{lane_ref}.yaml"
    rows = _load_yaml_list(path)
    rows.append(dict(row))
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(rows, handle, sort_keys=True, default_flow_style=False)
    return path


def read_fragments(repo_root: Path) -> list[dict[str, Any]]:
    """Read every lane's fragment file (never writes)."""

    directory = fragment_dir(repo_root)
    if not directory.exists():
        return []
    rows: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.yaml")):
        rows.extend(_load_yaml_list(path))
    return rows


def fold_fragments(rows: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Deterministically union rows by ``concept_id``, keeping the furthest state."""

    folded: dict[str, dict[str, Any]] = {}
    for row in rows:
        concept_id = row.get("concept_id")
        if not concept_id:
            raise ValueError("fragment row is missing concept_id")
        current = folded.get(concept_id)
        if current is None:
            folded[concept_id] = dict(row)
            continue
        current_rank = _STATE_RANK.get(current.get("state", ""), -1)
        candidate_rank = _STATE_RANK.get(row.get("state", ""), -1)
        if candidate_rank > current_rank or (
            candidate_rank == current_rank
            and str(row.get("transitioned_at", ""))
            > str(current.get("transitioned_at", ""))
        ):
            folded[concept_id] = dict(row)
    return [folded[key] for key in sorted(folded)]


def regenerate_view(repo_root: Path) -> list[dict[str, Any]]:
    """Fold every fragment and (re)write the generated view; return the fold."""

    folded = fold_fragments(read_fragments(repo_root))
    path = view_path(repo_root)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(folded, handle, sort_keys=True, default_flow_style=False)
    return folded


def _missing_and_extra_reasons(
    expected_ids: set[str], on_disk_ids: set[str]
) -> list[str]:
    reasons: list[str] = []
    for missing in sorted(expected_ids - on_disk_ids):
        reasons.append(f"{missing}: present in fold, missing from generated view")
    for extra in sorted(on_disk_ids - expected_ids):
        reasons.append(f"{extra}: present in generated view, absent from fold")
    return reasons


def _mismatched_shared_reasons(
    expected: list[dict[str, Any]],
    on_disk: list[dict[str, Any]],
    shared_ids: set[str],
) -> list[str]:
    reasons: list[str] = []
    for shared in sorted(shared_ids):
        exp_row = next(row for row in expected if row["concept_id"] == shared)
        disk_row = next(row for row in on_disk if row.get("concept_id") == shared)
        if exp_row != disk_row:
            reasons.append(f"{shared}: generated view does not match the fragment fold")
    return reasons


def verify_generated_view_is_fold(repo_root: Path) -> tuple[bool, list[str]]:
    """Return whether the on-disk generated view is exactly the current fold.

    Used to detect hand-editing of ``docs/concept_claims.yaml`` (H-20:
    "regenerate, never hand-pick") — proven against a deliberately drifted
    fixture in ``tests/concept_coordination/test_fragments.py``.
    """

    expected = fold_fragments(read_fragments(repo_root))
    on_disk = _load_yaml_list(view_path(repo_root))
    if expected == on_disk:
        return True, []
    expected_ids = {row["concept_id"] for row in expected}
    on_disk_ids: set[str] = {
        str(row["concept_id"]) for row in on_disk if row.get("concept_id") is not None
    }
    reasons = _missing_and_extra_reasons(expected_ids, on_disk_ids)
    reasons.extend(
        _mismatched_shared_reasons(expected, on_disk, expected_ids & on_disk_ids)
    )
    return False, reasons
