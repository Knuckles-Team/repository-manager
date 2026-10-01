"""``IdentityPolicy/v1`` (RM-IDENTITY-02's mapping input; plan.md data model).

An :class:`IdentityPolicy` is an exact old-identity -> canonical-identity
mapping plus the canonical allowlist its targets must be drawn from, and
nothing else — no fuzzy name matching, no private inventory lookup (the
spec's explicit "never infer identity from a fuzzy name match or private
inventory"). The *content* of a real fleet policy (which historical aliases
map to which canonical identity, and what happens to an identity nobody
mapped) is the owner decision ``specs/fleet-git-identity/spec.md`` records as
still open; this module only supplies the typed, digestible container and
the loader for the one external allowlist file the fleet already maintains.

That allowlist is ``pipelines_hooks.identity``'s commit-identity file
(schema: ``{"version": ..., "identities": [{"name": ..., "email": ...}]}``).
This module reads that exact file by path — the same
``COMMIT_IDENTITY_ALLOWLIST`` precedence pipelines' own gate uses — rather
than carrying a second, independently maintained copy of the identity list.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The pipelines gate's own env var; reused so one setting configures both
#: the commit-time gate and this package's canonical-identity source.
ALLOWLIST_PATH_ENV = "COMMIT_IDENTITY_ALLOWLIST"
MAX_ALLOWLIST_BYTES = 64 * 1024
MAX_IDENTITIES = 256

#: The spec's safe default ("the safe default is refusal"); no other value is
#: accepted until an owner decision approves a different unknown-identity rule.
REFUSE_UNKNOWN = "refuse"


class IdentityPolicyError(ValueError):
    """The configured allowlist, aliases, or unknown-identity rule is invalid."""


@dataclass(frozen=True)
class Identity:
    """One exact author/committer identity: a name and an email, verbatim."""

    name: str
    email: str

    def as_dict(self) -> dict[str, str]:
        return {"name": self.name, "email": self.email}


def _validated_identity(raw: Any, *, where: str) -> Identity:
    if not isinstance(raw, dict) or set(raw) != {"name", "email"}:
        raise IdentityPolicyError(f"{where} must have exactly 'name' and 'email'")
    name, email = raw["name"], raw["email"]
    if not isinstance(name, str) or not name.strip():
        raise IdentityPolicyError(f"{where} name must be a non-empty string")
    if not isinstance(email, str) or not email.strip() or "@" not in email:
        raise IdentityPolicyError(f"{where} email must be a non-empty address")
    return Identity(name=name, email=email)


def _read_bounded_json(path: Path, max_bytes: int) -> Any:
    try:
        raw_bytes = path.read_bytes()
    except OSError as exc:
        raise IdentityPolicyError(f"cannot read allowlist {path}: {exc}") from exc
    if len(raw_bytes) > max_bytes:
        raise IdentityPolicyError(f"allowlist {path} exceeds {max_bytes} bytes")
    try:
        return json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise IdentityPolicyError(f"allowlist {path} is not valid JSON: {exc}") from exc


def load_canonical_identities(path: Path) -> tuple[Identity, ...]:
    """The validated canonical identities from the fleet allowlist at ``path``.

    Same bounded, versioned schema and the same caps pipelines' own loader
    enforces (at most :data:`MAX_IDENTITIES` entries, at most
    :data:`MAX_ALLOWLIST_BYTES`); a malformed or empty file is rejected
    rather than silently widening who can be a canonical rewrite target.
    """
    raw = _read_bounded_json(path, MAX_ALLOWLIST_BYTES)
    if not isinstance(raw, dict) or set(raw) != {"version", "identities"}:
        raise IdentityPolicyError(
            "allowlist must contain exactly 'version' and 'identities'"
        )
    if not isinstance(raw["version"], str) or not raw["version"].strip():
        raise IdentityPolicyError("allowlist version must be a non-empty string")
    identities = raw["identities"]
    if not isinstance(identities, list) or not identities:
        raise IdentityPolicyError("allowlist identities must be a non-empty list")
    if len(identities) > MAX_IDENTITIES:
        raise IdentityPolicyError("allowlist exceeds the maximum identity count")
    return tuple(
        _validated_identity(item, where="allowlist entry") for item in identities
    )


def configured_allowlist_path(*, explicit: Path | None = None) -> Path:
    """Where the fleet allowlist is read from: ``explicit``, else the env setting.

    Unlike pipelines' own resolver this carries no packaged fleet-default
    fallback — the actual identity list lives in exactly one place (the
    pipelines hook package); a caller that has not configured either raises
    rather than guessing a path.
    """
    if explicit is not None:
        return explicit
    configured = os.environ.get(ALLOWLIST_PATH_ENV)
    if configured:
        return Path(configured)
    raise IdentityPolicyError(
        f"no fleet commit-identity allowlist configured: pass explicit= or set {ALLOWLIST_PATH_ENV} "
        "to the path of pipelines_hooks/identity/fleet-default-allowlist.json (or a repository-local "
        "override following the same schema)"
    )


@dataclass(frozen=True)
class IdentityPolicy:
    """``IdentityPolicy/v1``: exact aliases, the canonical allowlist, and the digest.

    ``aliases`` is an exact ``(old, canonical)`` pair list; ``canonical``
    must appear in ``canonical_identities`` (the allowlist this policy was
    built from) — a policy cannot invent a rewrite target outside the fleet
    allowlist. ``unknown_identity_rule`` is fixed to
    :data:`REFUSE_UNKNOWN`: an identity with no alias entry is never rewritten
    by :meth:`resolve`, and :mod:`.preview` surfaces that as a collision
    warning rather than silently passing it through or guessing.
    """

    version: str
    aliases: tuple[tuple[Identity, Identity], ...]
    canonical_identities: tuple[Identity, ...]
    unknown_identity_rule: str = REFUSE_UNKNOWN
    _by_old: dict[tuple[str, str], Identity] = field(
        init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        if self.unknown_identity_rule != REFUSE_UNKNOWN:
            raise IdentityPolicyError(
                f"unknown_identity_rule must be {REFUSE_UNKNOWN!r} until an owner decision "
                "approves a different rule"
            )
        canonical_set = set(self.canonical_identities)
        by_old: dict[tuple[str, str], Identity] = {}
        for old, canonical in self.aliases:
            if canonical not in canonical_set:
                raise IdentityPolicyError(
                    f"alias target {canonical.as_dict()} is not in the canonical allowlist"
                )
            key = (old.name, old.email)
            if key in by_old and by_old[key] != canonical:
                raise IdentityPolicyError(
                    f"alias {old.as_dict()} maps to two different canonical identities"
                )
            by_old[key] = canonical
        object.__setattr__(self, "_by_old", by_old)

    def resolve(self, name: str, email: str) -> Identity | None:
        """The canonical identity for an exact ``(name, email)`` alias, or ``None``."""
        return self._by_old.get((name, email))

    def digest(self) -> str:
        """A stable sha256 over this policy's exact content (plan.md's policy digest)."""
        payload = {
            "version": self.version,
            "unknown_identity_rule": self.unknown_identity_rule,
            "canonical_identities": sorted(
                (i.as_dict() for i in self.canonical_identities),
                key=lambda d: (d["name"], d["email"]),
            ),
            "aliases": sorted(
                (
                    [old.as_dict(), canonical.as_dict()]
                    for old, canonical in self.aliases
                ),
                key=lambda pair: (pair[0]["name"], pair[0]["email"]),
            ),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(blob).hexdigest()


def build_policy(
    *,
    version: str,
    aliases: tuple[tuple[Identity, Identity], ...],
    allowlist_path: Path | None = None,
) -> IdentityPolicy:
    """Build an :class:`IdentityPolicy` whose canonical identities come from the fleet allowlist.

    ``allowlist_path`` defaults to :func:`configured_allowlist_path`. This is
    the one place production code should construct a policy, so the
    canonical-identity source is never duplicated at a second call site.
    """
    path = configured_allowlist_path(explicit=allowlist_path)
    canonical_identities = load_canonical_identities(path)
    return IdentityPolicy(
        version=version,
        aliases=aliases,
        canonical_identities=canonical_identities,
    )
