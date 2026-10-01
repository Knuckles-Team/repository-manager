"""RM-IDENTITY-R002: reconcile diverged history before identity rewrite.

:func:`check_history_consistency` answers one question, read-only: does
every pair of local branches in a repository share a common ancestor? A
repository holding two or more genuinely disjoint histories (an orphan
branch with no merge-base to the rest) is not coherent enough for a rewrite
plan to reason about, since the plan's digests and mapping assume one
connected commit graph. :func:`admit_for_rewrite_plan` is the fail-closed
gate :mod:`.discovery`/:mod:`.preview` callers run first.

Reconciling a *specific* diverged repository — the spec names
``agent-webui`` — is an operational step this module only gates, never
performs: this package touches no repository other than the one path it is
given, and never resolves, merges, or rewrites a real divergence itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

from repository_manager.history_identity.discovery import (
    GitCommandError,
    GitRunner,
    SubprocessGitRunner,
)


class HistoryConsistencyError(RuntimeError):
    """The repository's branches do not share one consistent commit history."""


@dataclass(frozen=True)
class DivergedBranchPair:
    """Two local branches with no common ancestor."""

    left: str
    right: str


@dataclass(frozen=True)
class ConsistencyReport:
    """Whether every pair of a repository's local branches shares a common ancestor."""

    repo_id: str
    branches: tuple[str, ...]
    diverged_pairs: tuple[DivergedBranchPair, ...]

    @property
    def consistent(self) -> bool:
        return not self.diverged_pairs


def _branches(repo: Path, runner: GitRunner) -> list[str]:
    raw = runner.run(repo, ("for-each-ref", "--format=%(refname:short)", "refs/heads/"))
    return [line for line in raw.splitlines() if line.strip()]


def _shares_common_ancestor(
    repo: Path, runner: GitRunner, left: str, right: str
) -> bool:
    try:
        runner.run(repo, ("merge-base", left, right))
    except GitCommandError:
        return False
    return True


def check_history_consistency(
    repo: Path, *, repo_id: str | None = None, runner: GitRunner | None = None
) -> ConsistencyReport:
    """Whether every pair of local branches in ``repo`` shares a common ancestor."""
    active_runner = runner or SubprocessGitRunner()
    branches = _branches(repo, active_runner)
    diverged = tuple(
        DivergedBranchPair(left=left, right=right)
        for left, right in combinations(branches, 2)
        if not _shares_common_ancestor(repo, active_runner, left, right)
    )
    return ConsistencyReport(
        repo_id=repo_id or repo.name, branches=tuple(branches), diverged_pairs=diverged
    )


def admit_for_rewrite_plan(report: ConsistencyReport) -> None:
    """Raise :class:`HistoryConsistencyError` unless ``report`` is fully consistent."""
    if report.consistent:
        return
    pairs = ", ".join(f"{pair.left}<->{pair.right}" for pair in report.diverged_pairs)
    raise HistoryConsistencyError(
        f"{report.repo_id} has diverged branch histories with no common ancestor: {pairs}"
    )
