"""Candidate concept-marker scanner (deliverable 4).

Identifies concept IDs newly introduced by a candidate relative to its
immutable base tree, using this ecosystem's ``CONCEPT:<ID>`` marker
convention. Two marker grammars are live on ``main`` today (verified by
reading both live files, not assumed):

* dotted OKF style — ``agent_utilities.governance.concept_hierarchy.OKF_MARKER_RE``,
  e.g. ``CONCEPT:AU-OS.governance.cross-host-concept-reservation-authority``
  and this repository's own ``CONCEPT:RM-OS.governance.rm``;
* flat slug-only style, no digits at all, also in this repository's
  ``docs/concepts.md`` (``CONCEPT:RM-SAFE-COMMIT``, ``CONCEPT:RM-DESTRUCTIVE-GUARD``).

A third, flat **numeric** style (letters immediately followed by a digit,
e.g. what ``tests/test_concept_parity.py``'s own regex still matches on
legacy ``ORCH-``/``KG-``/``AHE-``/``ECO-``/``OS-`` prefixes) is retired: the
OKF-CIS cutover is complete, and ``scripts/check_no_legacy_markers.py``
(pre-commit hook ``okf-no-legacy-concepts``) now refuses any commit
introducing that exact substring anywhere in tracked source. ``MARKER_RE``
below still matches it, deliberately: ``diff_new_concept_ids`` scans
arbitrary immutable historical trees, and a pre-cutover base commit can
legitimately still carry a legacy-style marker — the scanner must not
silently miss it there. This is a defensive superset for *reading* history,
never license to *write* a new legacy-style marker (see
``tests/concept_coordination/test_scanner.py`` for how its own fixtures stay
compliant with the retirement).  ``MARKER_RE`` matching neither existing
narrower regex on its own (``OKF_MARKER_RE`` would miss the slug-only style)
is why a scanner meant to catch collisions is safer over-matching than
under-matching (H-9).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ConceptScanResult",
    "MARKER_RE",
    "diff_new_concept_ids",
    "extract_markers",
    "find_cross_candidate_collisions",
    "scan_tree_concept_markers",
]

MARKER_RE = re.compile(
    r"CONCEPT:([A-Z][A-Za-z0-9]*(?:-[A-Za-z0-9]+)*(?:\.[A-Za-z0-9][A-Za-z0-9-]*)*)"
)


def extract_markers(text: str) -> tuple[str, ...]:
    """Return every distinct ``CONCEPT:<ID>`` marker found in ``text``."""

    return tuple(sorted(set(MARKER_RE.findall(text))))


def _git_grep_tree(repo_path: Path, tree_ish: str) -> str:
    """Return ``git grep`` output for CONCEPT markers in one immutable tree.

    Fixed argv, never ``shell=True`` (repository-manager execution
    convention). A tree with no matches is not an error (``git grep`` exits 1
    when nothing matches); any other nonzero exit is a real refusal.
    """

    result = subprocess.run(  # nosec B603 B607
        ["git", "grep", "-I", "-n", "-e", "CONCEPT:", tree_ish, "--"],
        cwd=str(repo_path),
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode not in (0, 1):
        raise RuntimeError(
            f"git grep failed for {tree_ish!r} in {repo_path}: {result.stderr.strip()}"
        )
    return result.stdout


def scan_tree_concept_markers(
    repo_path: Path, tree_ish: str
) -> dict[str, tuple[str, ...]]:
    """Return ``{concept_id: (path:line, ...)}`` for one immutable Git tree."""

    locations: dict[str, list[str]] = {}
    for line in _git_grep_tree(repo_path, tree_ish).splitlines():
        # `git grep` output for a tree-ish is `<tree>:<path>:<lineno>:<text>`.
        parts = line.split(":", 3)
        if len(parts) < 4:
            continue
        _tree, path, lineno, text = parts
        for concept_id in extract_markers(text):
            locations.setdefault(concept_id, []).append(f"{path}:{lineno}")
    return {key: tuple(sorted(value)) for key, value in locations.items()}


@dataclass(frozen=True)
class ConceptScanResult:
    """New/removed markers between an immutable base and candidate tree."""

    base_sha: str
    candidate_sha: str
    introduced: tuple[str, ...]
    removed: tuple[str, ...]
    namespace_violations: tuple[str, ...]
    locations: Mapping[str, tuple[str, ...]]

    def as_dict(self) -> dict[str, object]:
        return {
            "base_sha": self.base_sha,
            "candidate_sha": self.candidate_sha,
            "introduced": list(self.introduced),
            "removed": list(self.removed),
            "namespace_violations": list(self.namespace_violations),
            "locations": {key: list(value) for key, value in self.locations.items()},
        }


def diff_new_concept_ids(
    repo_path: Path,
    *,
    base_sha: str,
    candidate_sha: str,
    allowed_namespaces: tuple[str, ...] = (),
) -> ConceptScanResult:
    """Scan a candidate tree for concept IDs introduced relative to its base.

    ``allowed_namespaces``, when non-empty, is a whitelist of accepted
    prefixes (e.g. ``("RM-",)``); an introduced ID outside every configured
    prefix is reported in ``namespace_violations`` rather than silently
    accepted or silently dropped.
    """

    base_markers = scan_tree_concept_markers(repo_path, base_sha)
    candidate_markers = scan_tree_concept_markers(repo_path, candidate_sha)
    introduced = sorted(set(candidate_markers) - set(base_markers))
    removed = sorted(set(base_markers) - set(candidate_markers))
    violations = (
        sorted(
            concept_id
            for concept_id in introduced
            if not any(concept_id.startswith(prefix) for prefix in allowed_namespaces)
        )
        if allowed_namespaces
        else ()
    )
    return ConceptScanResult(
        base_sha=base_sha,
        candidate_sha=candidate_sha,
        introduced=tuple(introduced),
        removed=tuple(removed),
        namespace_violations=tuple(violations),
        locations=candidate_markers,
    )


def find_cross_candidate_collisions(
    introduced_by_candidate: Mapping[str, Iterable[str]],
) -> dict[str, tuple[str, ...]]:
    """Return ``{concept_id: (candidate_id, ...)}`` for IDs >1 candidate introduces.

    Used at generation-formation time: two candidates in the same generation
    each introducing the same brand-new concept ID is a collision the
    authority's own uniqueness check would also catch, but reporting it here
    (deliverable 4, "duplicate ... errors") gives an actionable refusal
    before either candidate reaches the authority.
    """

    owners: dict[str, list[str]] = {}
    for candidate_id, concept_ids in introduced_by_candidate.items():
        for concept_id in concept_ids:
            owners.setdefault(concept_id, []).append(candidate_id)
    return {
        concept_id: tuple(sorted(candidates))
        for concept_id, candidates in owners.items()
        if len(candidates) > 1
    }
