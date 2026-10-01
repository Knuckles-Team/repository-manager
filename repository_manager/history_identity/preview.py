"""RM-IDENTITY-02: deterministic dry-run identity rewrite plan (``HistoryPreview/v1``).

:func:`generate_preview` reads a repository's commit authors/committers
(never writes anything) and maps each through an injected
:class:`~repository_manager.history_identity.policy.IdentityPolicy`,
producing a :class:`HistoryPreview` with counts, the affected identities, and
a single deterministic digest. Running this twice against the same
``discovery``/``policy`` (the same source state) yields an identical digest;
:mod:`.approval` binds an approval to exactly this digest, so any commit,
ref, or policy change on the source invalidates a previously granted
approval rather than silently reusing it against moved history.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from repository_manager.history_identity.discovery import (
    GitRunner,
    RepositoryDiscovery,
    SubprocessGitRunner,
)
from repository_manager.history_identity.policy import Identity, IdentityPolicy

_COMMIT_FORMAT = "%H%x09%an%x09%ae%x09%cn%x09%ce"


@dataclass(frozen=True)
class CommitIdentityMapping:
    """One commit's old author/committer identity and its resolved canonical identity.

    ``author_new``/``committer_new`` is ``None`` when the policy has no
    alias for that exact old identity — an unmapped identity, surfaced as a
    collision warning rather than silently left as-is or guessed.
    """

    sha: str
    author_old: Identity
    author_new: Identity | None
    committer_old: Identity
    committer_new: Identity | None

    def as_dict(self) -> dict[str, object]:
        return {
            "sha": self.sha,
            "author_old": self.author_old.as_dict(),
            "author_new": self.author_new.as_dict() if self.author_new else None,
            "committer_old": self.committer_old.as_dict(),
            "committer_new": self.committer_new.as_dict()
            if self.committer_new
            else None,
        }

    @property
    def fully_mapped(self) -> bool:
        return self.author_new is not None and self.committer_new is not None


@dataclass(frozen=True)
class HistoryPreview:
    """A deterministic dry-run plan: everything an approval (RM-IDENTITY-03) binds to."""

    repo_id: str
    discovery_digest: str
    policy_digest: str
    remotes: tuple[str, ...]
    expiry: str
    mappings: tuple[CommitIdentityMapping, ...]
    signed_shas: tuple[str, ...]
    collision_warnings: tuple[str, ...]
    digest: str

    @property
    def rewritten_commit_count(self) -> int:
        return sum(1 for m in self.mappings if m.fully_mapped)

    @property
    def unmapped_commit_count(self) -> int:
        return sum(1 for m in self.mappings if not m.fully_mapped)


def _enumerate_commits(
    repo: Path, runner: GitRunner
) -> list[tuple[str, Identity, Identity]]:
    raw = runner.run(
        repo, ("log", "--all", "--date-order", f"--format={_COMMIT_FORMAT}")
    )
    commits: list[tuple[str, Identity, Identity]] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        sha, author_name, author_email, committer_name, committer_email = line.split(
            "\t"
        )
        commits.append(
            (
                sha,
                Identity(name=author_name, email=author_email),
                Identity(name=committer_name, email=committer_email),
            )
        )
    return commits


def _build_mapping(
    policy: IdentityPolicy, sha: str, author: Identity, committer: Identity
) -> CommitIdentityMapping:
    return CommitIdentityMapping(
        sha=sha,
        author_old=author,
        author_new=policy.resolve(author.name, author.email),
        committer_old=committer,
        committer_new=policy.resolve(committer.name, committer.email),
    )


def _collision_warnings(mappings: tuple[CommitIdentityMapping, ...]) -> tuple[str, ...]:
    warnings: list[str] = []
    for mapping in mappings:
        if mapping.author_new is None:
            warnings.append(
                f"{mapping.sha}: author {mapping.author_old.as_dict()} has no canonical mapping"
            )
        if mapping.committer_new is None:
            warnings.append(
                f"{mapping.sha}: committer {mapping.committer_old.as_dict()} has no canonical mapping"
            )
    return tuple(warnings)


def _digest(
    *,
    repo_id: str,
    discovery_digest: str,
    policy_digest: str,
    remotes: tuple[str, ...],
    expiry: str,
    mappings: tuple[CommitIdentityMapping, ...],
    signed_shas: tuple[str, ...],
) -> str:
    payload = {
        "repo_id": repo_id,
        "discovery_digest": discovery_digest,
        "policy_digest": policy_digest,
        "remotes": sorted(remotes),
        "expiry": expiry,
        "mappings": [m.as_dict() for m in mappings],
        "signed_shas": sorted(signed_shas),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def generate_preview(
    repo: Path,
    discovery: RepositoryDiscovery,
    policy: IdentityPolicy,
    *,
    remotes: tuple[str, ...] = (),
    expiry: str,
    runner: GitRunner | None = None,
) -> HistoryPreview:
    """Build a deterministic dry-run plan for ``repo`` given ``discovery`` and ``policy``.

    ``discovery`` must have been produced by
    :func:`repository_manager.history_identity.discovery.discover` against
    the same ``repo`` — its digest is folded into this preview's digest so a
    ref change invalidates the preview even if no commit identity changed.
    """
    active_runner = runner or SubprocessGitRunner()
    commits = _enumerate_commits(repo, active_runner)
    mappings = tuple(
        sorted(
            (
                _build_mapping(policy, sha, author, committer)
                for sha, author, committer in commits
            ),
            key=lambda mapping: mapping.sha,
        )
    )
    signed_shas = tuple(
        sorted({record.target_sha for record in discovery.refs if record.signed})
    )
    policy_digest = policy.digest()
    return HistoryPreview(
        repo_id=discovery.repo_id,
        discovery_digest=discovery.digest,
        policy_digest=policy_digest,
        remotes=tuple(remotes),
        expiry=expiry,
        mappings=mappings,
        signed_shas=signed_shas,
        collision_warnings=_collision_warnings(mappings),
        digest=_digest(
            repo_id=discovery.repo_id,
            discovery_digest=discovery.digest,
            policy_digest=policy_digest,
            remotes=remotes,
            expiry=expiry,
            mappings=mappings,
            signed_shas=signed_shas,
        ),
    )
