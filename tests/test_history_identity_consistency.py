"""Tests for RM-IDENTITY-R002: reconcile diverged history before identity rewrite."""

from __future__ import annotations

import pytest

from repository_manager.history_identity.consistency import (
    HistoryConsistencyError,
    admit_for_rewrite_plan,
    check_history_consistency,
)
from tests.history_identity_git_fixtures import commit, init_repo, run_git


def test_check_history_consistency_passes_for_a_normal_branch(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    run_git(repo, "branch", "topic")

    report = check_history_consistency(repo)

    assert report.consistent is True
    assert set(report.branches) == {"main", "topic"}
    admit_for_rewrite_plan(report)  # does not raise


def test_check_history_consistency_detects_an_orphan_branch(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    run_git(repo, "checkout", "-q", "--orphan", "disconnected")
    run_git(repo, "reset", "-q", "--hard")
    commit(repo, "disconnected root", filename="other.txt")

    report = check_history_consistency(repo)

    assert report.consistent is False
    pair_names = {(pair.left, pair.right) for pair in report.diverged_pairs}
    assert ("disconnected", "main") in pair_names or ("main", "disconnected") in pair_names


def test_admit_for_rewrite_plan_raises_on_a_diverged_report(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    run_git(repo, "checkout", "-q", "--orphan", "disconnected")
    run_git(repo, "reset", "-q", "--hard")
    commit(repo, "disconnected root", filename="other.txt")

    report = check_history_consistency(repo)

    with pytest.raises(HistoryConsistencyError, match="diverged branch histories"):
        admit_for_rewrite_plan(report)
