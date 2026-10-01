"""RM-IDENTITY-03: signed approval required before any rewrite.

An :class:`ApprovalBinding` pins exactly the five facts plan.md requires: the
repository set, the source (preview) digest, the policy digest, the target
remotes, and an expiry. :func:`verify_approval` is fail-closed by
construction — every rejection path raises :class:`ApprovalError` before
returning, so there is no truthy/falsy result a caller could misread as
success. Nothing in this module mutates a repository; it only decides
whether a purported approval would authorize a rewrite plan that does not
exist yet in this package (see the package docstring).
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime


class ApprovalError(RuntimeError):
    """The approval is missing, expired, forged, or bound to a different plan."""


@dataclass(frozen=True)
class ApprovalBinding:
    """What an approval must match exactly: repository set, digests, remotes, expiry."""

    repository_ids: tuple[str, ...]
    source_digest: str
    policy_digest: str
    remotes: tuple[str, ...]
    expires_at: str

    def canonical_bytes(self) -> bytes:
        payload = {
            "repository_ids": sorted(self.repository_ids),
            "source_digest": self.source_digest,
            "policy_digest": self.policy_digest,
            "remotes": sorted(self.remotes),
            "expires_at": self.expires_at,
        }
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )


@dataclass(frozen=True)
class SignedApproval:
    """An authenticated approval: a binding plus its signer and signature."""

    binding: ApprovalBinding
    signer: str
    signature_hex: str


def _mac(binding: ApprovalBinding, signer: str, key: bytes) -> str:
    return hmac.new(
        key, binding.canonical_bytes() + signer.encode("utf-8"), hashlib.sha256
    ).hexdigest()


def sign_approval(
    binding: ApprovalBinding, *, signer: str, key: bytes
) -> SignedApproval:
    """Produce the authorized maintainer's signed approval for ``binding``.

    ``key`` is the maintainer's authentication secret (an HMAC key — the
    spec's "signed or otherwise authenticated" leaves the exact mechanism to
    a later owner decision; HMAC keeps verification dependency-free and
    testable now). The key itself is never stored on the result.
    """
    return SignedApproval(
        binding=binding, signer=signer, signature_hex=_mac(binding, signer, key)
    )


def _reject_if_unbound(
    approval: SignedApproval | None, expected: ApprovalBinding
) -> SignedApproval:
    if approval is None:
        raise ApprovalError("no approval supplied")
    if approval.binding != expected:
        raise ApprovalError(
            "approval is not bound to the exact repository set, source digest, policy digest, "
            "remotes, and expiry of this run"
        )
    return approval


def _reject_if_expired(approval: SignedApproval, *, now: datetime) -> None:
    expires_at = datetime.fromisoformat(approval.binding.expires_at)
    if expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=UTC)
    if now >= expires_at:
        raise ApprovalError(f"approval expired at {approval.binding.expires_at}")


def _reject_if_forged(approval: SignedApproval, *, key: bytes) -> None:
    expected_signature = _mac(approval.binding, approval.signer, key)
    if not hmac.compare_digest(expected_signature, approval.signature_hex):
        raise ApprovalError("approval signature does not verify")


def verify_approval(
    approval: SignedApproval | None,
    expected: ApprovalBinding,
    *,
    key: bytes,
    now: datetime | None = None,
) -> SignedApproval:
    """Return ``approval`` if it is a valid, unexpired, exact-binding match; otherwise raise.

    Checks binding, expiry, then signature in that order so a caller's error
    message always names the first thing wrong; all three must pass for a
    rewrite plan to ever be admitted.
    """
    bound = _reject_if_unbound(approval, expected)
    _reject_if_expired(bound, now=now or datetime.now(UTC))
    _reject_if_forged(bound, key=key)
    return bound
