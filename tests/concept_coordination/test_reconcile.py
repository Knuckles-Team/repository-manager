"""Read-only reconciliation tests (deliverable 7).

Covers every required classification from the lane brief: missing fragment,
orphan central claim, generated-view drift, used-but-unclaimed, and
landed-not-permanent — all without any reallocation (reconcile() never calls
an authority's ``reserve``/``transition``, only reads).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from repository_manager.concept_coordination.fragments import (
    fragment_dir,
    regenerate_view,
    view_path,
    write_fragment_row,
)
from repository_manager.concept_coordination.models import ConceptClaimRecord
from repository_manager.concept_coordination.reconcile import reconcile
from repository_manager.concept_coordination.state import (
    ConceptClaimState,
    ConceptClaimVisibility,
)

_NOW = datetime(2026, 8, 10, 12, 0, tzinfo=UTC)
_LATER = _NOW + timedelta(hours=1)


def _claim(concept_id: str, state: ConceptClaimState) -> ConceptClaimRecord:
    return ConceptClaimRecord(
        reservation_id=f"concept-reservation:{concept_id}",
        concept_id=concept_id,
        tenant_ref="tenant-1",
        repository_ref="repo-1",
        lane_ref="lane-a",
        owner_ref="owner-1",
        state=state,
        visibility=ConceptClaimVisibility.PRIVATE,
        fence=1,
        created_at=_NOW,
        expires_at=_LATER,
        transitioned_at=_NOW,
    )


def test_reconcile_detects_missing_fragment(tmp_path: Path) -> None:
    claim = _claim("RM-OS.test.missing-fragment", ConceptClaimState.RESERVED)
    report = reconcile(
        repo_root=tmp_path, central_claims=[claim], source_marker_ids=frozenset()
    )
    assert report.missing_fragment == ("RM-OS.test.missing-fragment",)
    assert not report.clean


def test_reconcile_detects_orphan_central_claim(tmp_path: Path) -> None:
    # A fragment row exists with no matching central claim at all.
    write_fragment_row(
        tmp_path,
        "lane-a",
        {
            "concept_id": "RM-OS.test.orphan",
            "reservation_id": "concept-reservation:RM-OS.test.orphan",
            "state": "reserved",
            "transitioned_at": "2026-08-10T00:00:00Z",
        },
    )
    report = reconcile(
        repo_root=tmp_path, central_claims=[], source_marker_ids=frozenset()
    )
    assert report.orphan_fragment == ("RM-OS.test.orphan",)
    assert not report.clean


def test_reconcile_detects_generated_view_drift(tmp_path: Path) -> None:
    claim = _claim("RM-OS.test.drift", ConceptClaimState.LANDED)
    write_fragment_row(tmp_path, "lane-a", claim.canonical_payload())
    regenerate_view(tmp_path)
    # Hand-edit the view after regenerating it — forbidden per H-20.
    view_path(tmp_path).write_text("[]\n", encoding="utf-8")

    report = reconcile(
        repo_root=tmp_path,
        central_claims=[claim],
        source_marker_ids=frozenset({"RM-OS.test.drift"}),
    )
    assert report.generated_view_matches_fold is False
    assert report.generated_view_drift_reasons
    assert not report.clean


def test_reconcile_detects_used_but_unclaimed(tmp_path: Path) -> None:
    report = reconcile(
        repo_root=tmp_path,
        central_claims=[],
        source_marker_ids=frozenset({"RM-OS.test.unclaimed"}),
    )
    assert report.used_but_unclaimed == ("RM-OS.test.unclaimed",)
    assert not report.clean


def test_reconcile_detects_landed_not_permanent(tmp_path: Path) -> None:
    claim = _claim("RM-OS.test.stale-state", ConceptClaimState.RESERVED)
    write_fragment_row(tmp_path, "lane-a", claim.canonical_payload())
    regenerate_view(tmp_path)

    report = reconcile(
        repo_root=tmp_path,
        central_claims=[claim],
        source_marker_ids=frozenset({"RM-OS.test.stale-state"}),
    )
    assert report.landed_not_permanent == ("RM-OS.test.stale-state",)
    assert not report.clean


def test_reconcile_is_clean_when_everything_agrees(tmp_path: Path) -> None:
    claim = _claim("RM-OS.test.clean", ConceptClaimState.LANDED)
    write_fragment_row(tmp_path, "lane-a", claim.canonical_payload())
    regenerate_view(tmp_path)

    report = reconcile(
        repo_root=tmp_path,
        central_claims=[claim],
        source_marker_ids=frozenset({"RM-OS.test.clean"}),
    )
    assert report.clean
    assert report.matches == ("RM-OS.test.clean",)


def test_reconcile_never_writes_anything(tmp_path: Path) -> None:
    """Read-only: reconcile() must never mutate fragments or the view."""

    claim = _claim("RM-OS.test.readonly", ConceptClaimState.LANDED)
    write_fragment_row(tmp_path, "lane-a", claim.canonical_payload())
    regenerate_view(tmp_path)
    before_fragment = (fragment_dir(tmp_path) / "lane-a.yaml").read_text(
        encoding="utf-8"
    )
    before_view = view_path(tmp_path).read_text(encoding="utf-8")

    reconcile(
        repo_root=tmp_path,
        central_claims=[claim],
        source_marker_ids=frozenset({"RM-OS.test.readonly"}),
    )

    assert (fragment_dir(tmp_path) / "lane-a.yaml").read_text(
        encoding="utf-8"
    ) == before_fragment
    assert view_path(tmp_path).read_text(encoding="utf-8") == before_view
