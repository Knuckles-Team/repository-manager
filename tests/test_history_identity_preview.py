"""Tests for RM-IDENTITY-02: deterministic dry-run identity rewrite plan."""

from __future__ import annotations

import json

import pytest

from repository_manager.history_identity.discovery import discover
from repository_manager.history_identity.policy import (
    Identity,
    IdentityPolicy,
    build_policy,
)
from repository_manager.history_identity.preview import generate_preview
from tests.history_identity_git_fixtures import AUTHOR_A, AUTHOR_B, commit, init_repo

_EXPIRY = "2099-01-01T00:00:00+00:00"


def _allowlist_path(tmp_path):
    path = tmp_path / "allowlist.json"
    path.write_text(
        json.dumps(
            {
                "version": "1",
                "identities": [{"name": "Canonical A", "email": "canonical-a@example.test"}],
            }
        )
    )
    return path


def _policy_mapping_author_a(tmp_path) -> IdentityPolicy:
    old_a = Identity(name=AUTHOR_A[0], email=AUTHOR_A[1])
    canonical = Identity(name="Canonical A", email="canonical-a@example.test")
    return build_policy(
        version="1", aliases=((old_a, canonical),), allowlist_path=_allowlist_path(tmp_path)
    )


def test_generate_preview_is_deterministic_across_runs(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    discovery = discover(repo)
    policy = _policy_mapping_author_a(tmp_path)

    first = generate_preview(repo, discovery, policy, expiry=_EXPIRY)
    second = generate_preview(repo, discovery, policy, expiry=_EXPIRY)

    assert first.digest == second.digest
    assert first.rewritten_commit_count == 1
    assert first.unmapped_commit_count == 0


def test_generate_preview_digest_changes_when_source_mutates(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    policy = _policy_mapping_author_a(tmp_path)

    before_discovery = discover(repo)
    before = generate_preview(repo, before_discovery, policy, expiry=_EXPIRY)

    commit(repo, "second", author=AUTHOR_A)
    after_discovery = discover(repo)
    after = generate_preview(repo, after_discovery, policy, expiry=_EXPIRY)

    assert before.digest != after.digest
    assert before.discovery_digest != after.discovery_digest


def test_generate_preview_surfaces_collision_warning_for_unmapped_identity(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    commit(repo, "second", author=AUTHOR_B, filename="other.txt")
    discovery = discover(repo)
    policy = _policy_mapping_author_a(tmp_path)

    preview = generate_preview(repo, discovery, policy, expiry=_EXPIRY)

    assert preview.rewritten_commit_count == 1
    assert preview.unmapped_commit_count == 1
    assert any(AUTHOR_B[1] in warning for warning in preview.collision_warnings)


def test_generate_preview_folds_policy_digest_and_is_sensitive_to_policy_change(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    discovery = discover(repo)
    mapped_policy = _policy_mapping_author_a(tmp_path)
    empty_policy = IdentityPolicy(
        version="1", aliases=(), canonical_identities=mapped_policy.canonical_identities
    )

    mapped_preview = generate_preview(repo, discovery, mapped_policy, expiry=_EXPIRY)
    empty_preview = generate_preview(repo, discovery, empty_policy, expiry=_EXPIRY)

    assert mapped_preview.policy_digest != empty_preview.policy_digest
    assert mapped_preview.digest != empty_preview.digest


@pytest.mark.parametrize("remotes", [(), ("origin",)])
def test_generate_preview_folds_remotes_into_digest(tmp_path, remotes):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    discovery = discover(repo)
    policy = _policy_mapping_author_a(tmp_path)

    preview = generate_preview(repo, discovery, policy, remotes=remotes, expiry=_EXPIRY)

    assert preview.remotes == remotes
