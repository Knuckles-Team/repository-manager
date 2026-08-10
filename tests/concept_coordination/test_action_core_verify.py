"""``ConceptCoordinationActions.verify_candidate`` against real git + real RMDD-12 types.

Uses ``repository_manager.development.Candidate``/``RepositoryIdentity``
unmodified (frozen interfaces this lane consumes, per the lane brief's
Method step 1) and a real git repository fixture. Proves the required
"authority outage" evidence precisely: a no-concept candidate is unaffected
by an unreachable authority, while a candidate that introduces a concept id
fails closed.
"""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from repository_manager.concept_coordination.action_core import (
    ConceptCoordinationActions,
)
from repository_manager.concept_coordination.errors import ConceptAuthorityUnavailable
from repository_manager.development import Candidate, CandidateState, RepositoryIdentity


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
    (tmp_path / "a.py").write_text('"""nothing here."""\n', encoding="utf-8")
    _git(tmp_path, "add", "a.py")
    _git(tmp_path, "commit", "-q", "-m", "base")
    return tmp_path


def _candidate(repo: Path, *, base_sha: str, candidate_sha: str) -> Candidate:
    identity = RepositoryIdentity(repository_id="repo-1", canonical_path=str(repo))
    return Candidate(
        candidate_id="cand-1",
        version=1,
        repository=identity,
        branch="feature/x",
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        lane_id="lane-a",
        owner_id="owner-1",
        config_digest="a" * 64,
        enqueued_at=datetime(2026, 8, 10, 12, 0, tzinfo=UTC),
        state=CandidateState.QUEUED,
    )


def test_no_concept_candidate_never_touches_the_authority(repo: Path) -> None:
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "b.py").write_text('"""still nothing."""\n', encoding="utf-8")
    _git(repo, "add", "b.py")
    _git(repo, "commit", "-q", "-m", "candidate, no concept markers")
    candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    # No authority injected at all: if verify_candidate touched it, this
    # would raise ConceptAuthorityUnavailable from resolve_default_authority().
    actions = ConceptCoordinationActions(
        repo_root=repo, tenant_ref="tenant-1", lane_ref="lane-a", authority=None
    )
    outcome = actions.verify_candidate(
        _candidate(repo, base_sha=base_sha, candidate_sha=candidate_sha), repo_path=repo
    )
    assert outcome.passed
    assert outcome.checked_ids == ()


def test_new_concept_candidate_fails_closed_when_authority_unreachable(
    repo: Path,
) -> None:
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()
    (repo / "b.py").write_text(
        '"""CONCEPT:RM-OS.test.needs-authority."""\n', encoding="utf-8"
    )
    _git(repo, "add", "b.py")
    _git(repo, "commit", "-q", "-m", "candidate introduces a concept id")
    candidate_sha = _git(repo, "rev-parse", "HEAD").stdout.strip()

    actions = ConceptCoordinationActions(
        repo_root=repo, tenant_ref="tenant-1", lane_ref="lane-a", authority=None
    )
    with pytest.raises(ConceptAuthorityUnavailable):
        actions.verify_candidate(
            _candidate(repo, base_sha=base_sha, candidate_sha=candidate_sha),
            repo_path=repo,
        )
