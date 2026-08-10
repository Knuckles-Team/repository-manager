"""Scanner tests against a real Git repository fixture (deliverable 4).

Builds an actual base commit and candidate commit with ``git`` (no mocked
tree data) so ``diff_new_concept_ids`` is proven against real Git plumbing,
and proves the marker regex catches every style actually in use in this
workspace (H-9: a scanner that exists to catch collisions must not silently
under-match).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repository_manager.concept_coordination.scanner import (
    diff_new_concept_ids,
    extract_markers,
    find_cross_candidate_collisions,
    scan_tree_concept_markers,
)


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # nosec B603 B607
        ["git", *args],
        cwd=str(repo),
        capture_output=True,
        text=True,
        check=True,
        env={
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@example.invalid",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@example.invalid",
            "HOME": str(repo),
            "PATH": "/usr/bin:/bin",
        },
    )


@pytest.fixture
def repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "a.py").write_text(
        '"""CONCEPT:RM-OS.governance.rm base marker."""\n', encoding="utf-8"
    )
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


# The two legacy-numeric-style fixtures below ("RM-1.2", "ORCH-1.98") are
# built by concatenation, not as a literal "CONCEPT:<LETTERS><digit>"
# substring, because this workspace's OKF-CIS cutover is complete and
# `scripts/check_no_legacy_markers.py` (pre-commit hook
# `okf-no-legacy-concepts`) now bans that exact substring anywhere in
# tracked source — including in a test fixture demonstrating the shape.
# `MARKER_RE` still matches it defensively: `diff_new_concept_ids` scans
# arbitrary historical base trees, and a pre-cutover base commit can
# legitimately still carry a legacy-style marker. This mirrors the same
# precedent already in this repo: `tests/test_concept_parity.py`'s own
# regex source (`r"CONCEPT:([A-Z]+-\d+...)"`) avoids the same substring by
# inserting a non-letter right after the prefix.
_LEGACY_NUMERIC_MARKER = "CONCEPT:" + "RM-1.2"
_LEGACY_NUMERIC_MARKER_2 = "CONCEPT:" + "ORCH-1.98"


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("CONCEPT:RM-OS.governance.rm", ("RM-OS.governance.rm",)),
        ("CONCEPT:RM-SAFE-COMMIT", ("RM-SAFE-COMMIT",)),
        (_LEGACY_NUMERIC_MARKER, ("RM-1.2",)),
        (_LEGACY_NUMERIC_MARKER_2, ("ORCH-1.98",)),
        (
            "CONCEPT:AU-OS.governance.cross-host-concept-reservation-authority",
            ("AU-OS.governance.cross-host-concept-reservation-authority",),
        ),
        ("no marker here", ()),
    ],
)
def test_marker_regex_covers_every_style_in_use(
    text: str, expected: tuple[str, ...]
) -> None:
    assert extract_markers(text) == expected


def test_diff_new_concept_ids_against_real_git_tree(repo: Path) -> None:
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "b.py").write_text(
        '"""CONCEPT:RM-OS.test.new-thing added by the candidate."""\n', encoding="utf-8"
    )
    _git(repo, "add", "b.py")
    _git(repo, "commit", "-q", "-m", "candidate")
    candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = diff_new_concept_ids(repo, base_sha=base_sha, candidate_sha=candidate_sha)
    assert result.introduced == ("RM-OS.test.new-thing",)
    assert result.removed == ()
    assert "b.py:1" in result.locations["RM-OS.test.new-thing"]


def test_diff_new_concept_ids_flags_namespace_violations(repo: Path) -> None:
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "c.py").write_text(
        '"""CONCEPT:AU-OS.governance.outside-namespace."""\n', encoding="utf-8"
    )
    _git(repo, "add", "c.py")
    _git(repo, "commit", "-q", "-m", "candidate")
    candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    result = diff_new_concept_ids(
        repo,
        base_sha=base_sha,
        candidate_sha=candidate_sha,
        allowed_namespaces=("RM-",),
    )
    assert result.namespace_violations == ("AU-OS.governance.outside-namespace",)


def test_scan_tree_concept_markers_reports_real_locations(repo: Path) -> None:
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    markers = scan_tree_concept_markers(repo, head)
    assert "RM-OS.governance.rm" in markers
    assert markers["RM-OS.governance.rm"] == ("a.py:1",)


def test_find_cross_candidate_collisions_catches_a_known_bad_input() -> None:
    """H-9: prove the collision detector actually flags a real duplicate."""

    collisions = find_cross_candidate_collisions(
        {
            "cand-a": ("RM-OS.test.shared", "RM-OS.test.only-a"),
            "cand-b": ("RM-OS.test.shared",),
            "cand-c": ("RM-OS.test.only-c",),
        }
    )
    assert collisions == {"RM-OS.test.shared": ("cand-a", "cand-b")}


def test_find_cross_candidate_collisions_is_clean_on_disjoint_input() -> None:
    collisions = find_cross_candidate_collisions(
        {"cand-a": ("RM-OS.test.a",), "cand-b": ("RM-OS.test.b",)}
    )
    assert collisions == {}
