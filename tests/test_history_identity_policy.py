"""Tests for the ``IdentityPolicy`` container (RM-IDENTITY-02's policy input).

No real fleet allowlist is read here -- every allowlist file is a disposable
fixture written under ``tmp_path``, matching "this repository consumes it, it
does not carry a second copy": the actual fleet identity list is
``pipelines_hooks/identity/fleet-default-allowlist.json``, external to this
package.
"""

from __future__ import annotations

import json

import pytest

from repository_manager.history_identity.policy import (
    Identity,
    IdentityPolicy,
    IdentityPolicyError,
    build_policy,
    configured_allowlist_path,
    load_canonical_identities,
)

_VALID_ALLOWLIST = {
    "version": "1",
    "identities": [
        {"name": "Operator Example", "email": "operator@example.test"},
        {"name": "Claude", "email": "noreply@anthropic.test"},
    ],
}


def _write_allowlist(tmp_path, payload=_VALID_ALLOWLIST):
    path = tmp_path / "allowlist.json"
    path.write_text(json.dumps(payload))
    return path


def test_load_canonical_identities_reads_valid_allowlist(tmp_path):
    path = _write_allowlist(tmp_path)

    identities = load_canonical_identities(path)

    assert identities == (
        Identity(name="Operator Example", email="operator@example.test"),
        Identity(name="Claude", email="noreply@anthropic.test"),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"version": "1", "identities": []},
        {"version": "1", "identities": [{"name": "X"}]},
        {"version": "1", "identities": [{"name": "", "email": "x@example.test"}]},
        {"version": "1", "identities": [{"name": "X", "email": "not-an-email"}]},
        {"identities": [{"name": "X", "email": "x@example.test"}]},
        {"version": "", "identities": [{"name": "X", "email": "x@example.test"}]},
    ],
)
def test_load_canonical_identities_rejects_malformed_allowlist(tmp_path, payload):
    path = _write_allowlist(tmp_path, payload)

    with pytest.raises(IdentityPolicyError):
        load_canonical_identities(path)


def test_load_canonical_identities_rejects_oversized_allowlist(tmp_path):
    payload = {
        "version": "1",
        "identities": [{"name": f"user{i}", "email": f"user{i}@example.test"} for i in range(300)],
    }
    path = _write_allowlist(tmp_path, payload)

    with pytest.raises(IdentityPolicyError, match="maximum identity count"):
        load_canonical_identities(path)


def test_configured_allowlist_path_prefers_explicit(tmp_path, monkeypatch):
    monkeypatch.setenv("COMMIT_IDENTITY_ALLOWLIST", str(tmp_path / "from-env.json"))
    explicit = tmp_path / "explicit.json"

    assert configured_allowlist_path(explicit=explicit) == explicit


def test_configured_allowlist_path_falls_back_to_env(tmp_path, monkeypatch):
    env_path = tmp_path / "from-env.json"
    monkeypatch.setenv("COMMIT_IDENTITY_ALLOWLIST", str(env_path))

    assert configured_allowlist_path() == env_path


def test_configured_allowlist_path_raises_when_unconfigured(monkeypatch):
    monkeypatch.delenv("COMMIT_IDENTITY_ALLOWLIST", raising=False)

    with pytest.raises(IdentityPolicyError, match="no fleet commit-identity allowlist configured"):
        configured_allowlist_path()


def test_build_policy_loads_canonical_identities_from_allowlist(tmp_path):
    path = _write_allowlist(tmp_path)
    old = Identity(name="old name", email="old@example.test")
    canonical = Identity(name="Claude", email="noreply@anthropic.test")

    policy = build_policy(version="1", aliases=((old, canonical),), allowlist_path=path)

    assert policy.resolve("old name", "old@example.test") == canonical
    assert policy.resolve("unknown", "unknown@example.test") is None


def test_identity_policy_rejects_alias_target_outside_allowlist(tmp_path):
    canonical_identities = load_canonical_identities(_write_allowlist(tmp_path))
    old = Identity(name="old", email="old@example.test")
    outsider = Identity(name="Not Canonical", email="outsider@example.test")

    with pytest.raises(IdentityPolicyError, match="not in the canonical allowlist"):
        IdentityPolicy(
            version="1",
            aliases=((old, outsider),),
            canonical_identities=canonical_identities,
        )


def test_identity_policy_rejects_non_refuse_unknown_rule(tmp_path):
    canonical_identities = load_canonical_identities(_write_allowlist(tmp_path))

    with pytest.raises(IdentityPolicyError, match="must be 'refuse'"):
        IdentityPolicy(
            version="1",
            aliases=(),
            canonical_identities=canonical_identities,
            unknown_identity_rule="map-to-operator",
        )


def test_identity_policy_digest_is_deterministic_and_order_independent(tmp_path):
    canonical_identities = load_canonical_identities(_write_allowlist(tmp_path))
    claude = Identity(name="Claude", email="noreply@anthropic.test")
    old_one = Identity(name="old one", email="one@example.test")
    old_two = Identity(name="old two", email="two@example.test")

    forward = IdentityPolicy(
        version="1",
        aliases=((old_one, claude), (old_two, claude)),
        canonical_identities=canonical_identities,
    )
    reversed_order = IdentityPolicy(
        version="1",
        aliases=((old_two, claude), (old_one, claude)),
        canonical_identities=canonical_identities,
    )

    assert forward.digest() == reversed_order.digest()


def test_identity_policy_digest_changes_with_content(tmp_path):
    canonical_identities = load_canonical_identities(_write_allowlist(tmp_path))
    claude = Identity(name="Claude", email="noreply@anthropic.test")
    old = Identity(name="old", email="old@example.test")

    with_alias = IdentityPolicy(
        version="1", aliases=((old, claude),), canonical_identities=canonical_identities
    )
    without_alias = IdentityPolicy(version="1", aliases=(), canonical_identities=canonical_identities)

    assert with_alias.digest() != without_alias.digest()
