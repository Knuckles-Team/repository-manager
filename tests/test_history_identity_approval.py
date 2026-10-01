"""Tests for RM-IDENTITY-03: signed approval required before any rewrite.

Every negative case proves ``verify_approval`` *raises* (zero mutating
effects follow -- there is no execution path downstream of a rejected
approval in this package) rather than returning a falsy value a caller could
ignore.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from repository_manager.history_identity.approval import (
    ApprovalBinding,
    ApprovalError,
    sign_approval,
    verify_approval,
)

_KEY = b"test-only-fixture-key-not-a-real-secret"


def _binding(
    *,
    repository_ids: tuple[str, ...] = ("repo-a", "repo-b"),
    source_digest: str = "a" * 64,
    policy_digest: str = "b" * 64,
    remotes: tuple[str, ...] = ("origin",),
    expires_at: str = "2099-01-01T00:00:00+00:00",
) -> ApprovalBinding:
    return ApprovalBinding(
        repository_ids=repository_ids,
        source_digest=source_digest,
        policy_digest=policy_digest,
        remotes=remotes,
        expires_at=expires_at,
    )


def test_verify_approval_accepts_a_valid_matching_signature():
    binding = _binding()
    approval = sign_approval(binding, signer="maintainer", key=_KEY)

    verified = verify_approval(approval, binding, key=_KEY)

    assert verified is approval


def test_verify_approval_rejects_missing_approval():
    with pytest.raises(ApprovalError, match="no approval supplied"):
        verify_approval(None, _binding(), key=_KEY)


def test_verify_approval_rejects_expired_approval():
    binding = _binding(expires_at="2000-01-01T00:00:00+00:00")
    approval = sign_approval(binding, signer="maintainer", key=_KEY)

    with pytest.raises(ApprovalError, match="expired"):
        verify_approval(approval, binding, key=_KEY)


def test_verify_approval_rejects_an_approval_near_its_expiry_boundary():
    expires_at = datetime.now(UTC) + timedelta(seconds=1)
    binding = _binding(expires_at=expires_at.isoformat())
    approval = sign_approval(binding, signer="maintainer", key=_KEY)

    with pytest.raises(ApprovalError, match="expired"):
        verify_approval(approval, binding, key=_KEY, now=expires_at + timedelta(seconds=1))


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: replace(b, repository_ids=("repo-a", "repo-c")),
        lambda b: replace(b, source_digest="c" * 64),
        lambda b: replace(b, policy_digest="d" * 64),
        lambda b: replace(b, remotes=("origin", "fork")),
        lambda b: replace(b, expires_at="2100-01-01T00:00:00+00:00"),
    ],
)
def test_verify_approval_rejects_a_binding_that_no_longer_matches_the_current_run(mutate):
    original = _binding()
    approval = sign_approval(original, signer="maintainer", key=_KEY)
    current_expected = mutate(original)

    with pytest.raises(ApprovalError, match="not bound to the exact"):
        verify_approval(approval, current_expected, key=_KEY)


def test_verify_approval_rejects_a_forged_signature():
    binding = _binding()
    approval = sign_approval(binding, signer="maintainer", key=_KEY)
    forged = replace(approval, signature_hex="0" * 64)

    with pytest.raises(ApprovalError, match="does not verify"):
        verify_approval(forged, binding, key=_KEY)


def test_verify_approval_rejects_the_wrong_key():
    binding = _binding()
    approval = sign_approval(binding, signer="maintainer", key=_KEY)

    with pytest.raises(ApprovalError, match="does not verify"):
        verify_approval(approval, binding, key=b"a-different-fixture-key")
