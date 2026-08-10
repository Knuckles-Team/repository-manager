"""Fragment / generated-view audit-projection tests.

Proves the append-only-fragment + folded-generated-view pattern (mirrored
from agent-utilities' own concept marker ledger) actually works, and proves
the drift detector catches a hand-edited generated view (H-9: prove a gate
catches a known-bad input, mirroring H-20 "regenerate, never hand-pick").
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from repository_manager.concept_coordination.fragments import (
    fold_fragments,
    fragment_dir,
    read_fragments,
    regenerate_view,
    verify_generated_view_is_fold,
    view_path,
    write_fragment_row,
)


def _row(
    concept_id: str,
    state: str = "reserved",
    transitioned_at: str = "2026-08-10T00:00:00Z",
) -> dict:
    return {
        "concept_id": concept_id,
        "reservation_id": f"concept-reservation:{concept_id}",
        "state": state,
        "transitioned_at": transitioned_at,
    }


def test_write_fragment_row_appends_to_only_its_own_lane_file(tmp_path: Path) -> None:
    write_fragment_row(tmp_path, "lane-a", _row("RM-OS.test.one"))
    write_fragment_row(tmp_path, "lane-b", _row("RM-OS.test.two"))

    assert (fragment_dir(tmp_path) / "lane-a.yaml").exists()
    assert (fragment_dir(tmp_path) / "lane-b.yaml").exists()
    lane_a_rows = read_fragments(tmp_path)
    ids = {row["concept_id"] for row in lane_a_rows}
    assert ids == {"RM-OS.test.one", "RM-OS.test.two"}


def test_write_fragment_row_rejects_path_unsafe_lane_ref(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        write_fragment_row(tmp_path, "../escape", _row("RM-OS.test.x"))


def test_fold_fragments_keeps_the_furthest_state(tmp_path: Path) -> None:
    rows = [
        _row(
            "RM-OS.test.dup", state="reserved", transitioned_at="2026-08-10T00:00:00Z"
        ),
        _row("RM-OS.test.dup", state="landed", transitioned_at="2026-08-10T01:00:00Z"),
    ]
    folded = fold_fragments(rows)
    assert len(folded) == 1
    assert folded[0]["state"] == "landed"


def test_regenerate_view_writes_the_exact_fold(tmp_path: Path) -> None:
    write_fragment_row(tmp_path, "lane-a", _row("RM-OS.test.one"))
    folded = regenerate_view(tmp_path)
    on_disk = yaml.safe_load(view_path(tmp_path).read_text(encoding="utf-8"))
    assert on_disk == folded
    ok, reasons = verify_generated_view_is_fold(tmp_path)
    assert ok is True
    assert reasons == []


def test_verify_generated_view_is_fold_catches_a_hand_edit(tmp_path: Path) -> None:
    """H-9: prove the drift gate flags a known-bad input, not just green paths."""

    write_fragment_row(tmp_path, "lane-a", _row("RM-OS.test.one"))
    regenerate_view(tmp_path)

    # Simulate exactly the forbidden action (H-20): hand-editing the
    # generated view instead of regenerating it from fragments.
    view_path(tmp_path).write_text(
        yaml.safe_dump([_row("RM-OS.test.one", state="landed")]), encoding="utf-8"
    )

    ok, reasons = verify_generated_view_is_fold(tmp_path)
    assert ok is False
    assert any("RM-OS.test.one" in reason for reason in reasons)
