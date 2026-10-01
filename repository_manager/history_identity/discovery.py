"""RM-IDENTITY-01: complete ref and tag discovery before any rewrite.

:func:`discover` enumerates every local branch, annotated and lightweight
tag, remote-tracking ref, and note in a repository, records which referenced
commits or tag objects carry a GPG signature (the "signed-object inventory"
later phases need — this module never verifies a signature, only notes its
presence), and refuses outright — raising :class:`DiscoveryError`, never
returning a partial result — when:

* the repository's on-disk ref set holds anything this enumeration cannot
  classify (a "hidden ref": a ref under ``refs/`` outside the known
  heads/tags/remotes/notes namespaces, for example leftovers from a previous
  rewrite at ``refs/original/*``), or
* the repository's history is not what it appears to be (a shallow clone, or
  live ``refs/replace/*``/graft entries that make resolved commit content
  depend on something outside the ref database itself), or
* a configured remote cannot be reached to confirm its advertised refs.

A caller gets either a complete :class:`RepositoryDiscovery` or an exception;
there is no silent partial-discovery return value for a later phase to
misread as complete.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

#: Ref name prefixes this enumeration understands. Anything under ``refs/``
#: that does not start with one of these is a "hidden ref" and refuses.
_KNOWN_REF_PREFIXES: tuple[tuple[str, str], ...] = (
    ("refs/heads/", "branch"),
    ("refs/tags/", "tag"),
    ("refs/remotes/", "remote-tracking"),
    ("refs/notes/", "note"),
)

#: Presence of any of these means resolved history depends on something
#: outside the plain ref database — divergent state this tool refuses to plan
#: a rewrite against.
_DIVERGENT_REF_PREFIXES: tuple[str, ...] = ("refs/replace/", "refs/original/")


class GitCommandError(RuntimeError):
    """A git invocation exited non-zero. Raised by :class:`GitRunner` implementations only."""


class DiscoveryError(RuntimeError):
    """Discovery is incomplete, divergent, or a configured remote is unreachable."""


class GitRunner(Protocol):
    """Structural dependency :func:`discover` needs: run git in one repository."""

    def run(self, repo: Path, args: tuple[str, ...]) -> str: ...


@dataclass(frozen=True)
class SubprocessGitRunner:
    """The real runner: ``git -C <repo> <args>``, scoped to exactly one repository.

    Never touches any repository other than the one path it is called with —
    no global config, no other checkout, no network beyond the explicit
    ``ls-remote`` reachability check :func:`discover` issues per remote.
    """

    timeout_seconds: int = 60

    def run(self, repo: Path, args: tuple[str, ...]) -> str:
        argv = ["git", "-C", str(repo), *args]
        result = subprocess.run(  # fixed argv, no shell
            argv,
            capture_output=True,
            text=True,
            timeout=self.timeout_seconds,
            check=False,
        )
        if result.returncode != 0:
            raise GitCommandError(
                f"git {' '.join(args)} failed in {repo}: {result.stderr.strip()}"
            )
        return result.stdout


@dataclass(frozen=True)
class RefRecord:
    """One discovered ref: its name, resolved commit/tag object, kind, and signed flag."""

    name: str
    target_sha: str
    kind: str
    signed: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "target_sha": self.target_sha,
            "kind": self.kind,
            "signed": self.signed,
        }


@dataclass(frozen=True)
class RepositoryDiscovery:
    """A complete, digestible enumeration of one repository's refs and remotes."""

    repo_id: str
    refs: tuple[RefRecord, ...]
    remotes: tuple[str, ...]
    digest: str


def _ref_kind(name: str) -> str | None:
    for prefix, kind in _KNOWN_REF_PREFIXES:
        if name.startswith(prefix):
            return kind
    return None


def _reject_divergent_state(repo: Path, runner: GitRunner) -> None:
    if (repo / ".git" / "shallow").exists():
        raise DiscoveryError(f"{repo} is a shallow clone; refusing incomplete history")
    if (repo / ".git" / "info" / "grafts").exists():
        raise DiscoveryError(
            f"{repo} has a legacy grafts file; refusing divergent history"
        )
    raw = runner.run(repo, ("for-each-ref", "--format=%(refname)", "refs/"))
    for name in raw.splitlines():
        if name.startswith(_DIVERGENT_REF_PREFIXES):
            raise DiscoveryError(f"{repo} has a divergent ref {name!r}; refusing")


def _is_signed(repo: Path, runner: GitRunner, sha: str, kind: str) -> bool:
    body = runner.run(repo, ("cat-file", "-p", sha))
    if kind == "tag":
        return (
            "-----BEGIN PGP SIGNATURE-----" in body
            or "-----BEGIN SSH SIGNATURE-----" in body
        )
    return any(line.startswith("gpgsig ") for line in body.splitlines())


def _enumerate_refs(repo: Path, runner: GitRunner) -> list[RefRecord]:
    raw = runner.run(
        repo, ("for-each-ref", "--format=%(refname)%09%(objectname)", "refs/")
    )
    records: list[RefRecord] = []
    for line in raw.splitlines():
        if not line.strip():
            continue
        name, _, sha = line.partition("\t")
        kind = _ref_kind(name)
        if kind is None:
            raise DiscoveryError(
                f"{repo} has a hidden/unclassified ref {name!r}; refusing"
            )
        try:
            signed = _is_signed(repo, runner, sha, kind)
        except GitCommandError as exc:
            raise DiscoveryError(
                f"{repo} ref {name!r} points to an unreadable object: {exc}"
            ) from exc
        records.append(RefRecord(name=name, target_sha=sha, kind=kind, signed=signed))
    return records


def _enumerate_and_verify_remotes(repo: Path, runner: GitRunner) -> list[str]:
    names = [
        line for line in runner.run(repo, ("remote",)).splitlines() if line.strip()
    ]
    for name in names:
        try:
            runner.run(repo, ("ls-remote", "--exit-code", name))
        except GitCommandError as exc:
            raise DiscoveryError(
                f"remote {name!r} of {repo} is unreachable: {exc}"
            ) from exc
    return names


def _digest(refs: list[RefRecord], remotes: list[str]) -> str:
    payload = {
        "refs": sorted((r.as_dict() for r in refs), key=lambda d: str(d["name"])),
        "remotes": sorted(remotes),
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def discover(
    repo: Path,
    *,
    repo_id: str | None = None,
    runner: GitRunner | None = None,
) -> RepositoryDiscovery:
    """Enumerate ``repo``'s complete ref/tag/remote set, or raise :class:`DiscoveryError`.

    ``runner`` is injectable so tests exercise refusal paths (an unreadable
    remote, a hidden ref) without needing a second real process per case; the
    default is :class:`SubprocessGitRunner` scoped to ``repo`` alone.
    """
    active_runner = runner or SubprocessGitRunner()
    _reject_divergent_state(repo, active_runner)
    refs = _enumerate_refs(repo, active_runner)
    remotes = _enumerate_and_verify_remotes(repo, active_runner)
    return RepositoryDiscovery(
        repo_id=repo_id or repo.name,
        refs=tuple(refs),
        remotes=tuple(remotes),
        digest=_digest(refs, remotes),
    )
