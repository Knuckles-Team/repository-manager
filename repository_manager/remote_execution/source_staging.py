"""Immutable source materialization: commit/generation only, never dirty state.

Given a repository origin and an immutable full Git SHA, this module builds
the fixed argv command sequence needed to materialize exactly that tree under
an authorized worktree root -- on a local or remote
``CommandExecutor``-shaped worker, through the same ``run`` seam RMDD-07
already froze -- and independently re-derives cleanliness and HEAD identity
afterward.  It never trusts a caller's claim that a source is clean or that a
checkout landed at the right commit; both are always re-verified by running
``git status``/``git rev-parse`` through the same executor and reading their
result, never by inspecting the filesystem directly from this process.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from repository_manager.development import ExecutionCommand, ExecutionOutcome

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DirtySourceError(ValueError):
    """The source lane has uncommitted changes; it cannot be remote-executed
    as though it were committed."""


class SourceVerificationError(ValueError):
    """The materialized worktree does not match the requested immutable SHA,
    or its state could not be independently re-derived."""


@dataclass(frozen=True)
class StagedSource:
    """Proof that ``tree_sha`` was independently verified at ``destination``."""

    repository_id: str
    tree_sha: str
    destination: str
    verified_at: datetime


class ImmutableSourceStaging:
    """Build fixed-argv staging commands and independently verify their result."""

    def __init__(self, *, git_binary: str = "git") -> None:
        if (
            not git_binary
            or any(char.isspace() for char in git_binary)
            or ("/" in git_binary and not git_binary.startswith("/"))
        ):
            raise ValueError("git_binary must be a bare command name or absolute path")
        self.git_binary = git_binary

    @staticmethod
    def require_immutable_sha(value: str) -> str:
        """Refuse anything that is not a full 40-hex commit SHA.

        A branch name, tag, or short SHA is a *moving* or ambiguous
        reference; this lane's contract is explicit that repository input is
        an immutable commit/generation, never a mutable ref.
        """

        if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
            raise ValueError(
                "source materialization requires a full 40-hex commit SHA, "
                "not a branch, tag, or short ref"
            )
        return value

    @staticmethod
    def _require_credential_free_origin(origin: str) -> str:
        if not origin or " " in origin or "\t" in origin:
            raise ValueError("origin must be a single space-free URL or path")
        if "@" in origin and "://" in origin:
            # scheme://user[:pass]@host is a credential-bearing URL form.
            scheme_split = origin.split("://", 1)[1]
            if "@" in scheme_split:
                raise ValueError("origin must not embed credentials")
        return origin

    @staticmethod
    def _require_safe_component(name: str) -> str:
        if not _SAFE_COMPONENT_RE.fullmatch(name):
            raise ValueError(
                "worktree_name must be one safe path component with no "
                "traversal or separator characters"
            )
        return name

    def check_local_source_clean(
        self,
        executor: object,
        *,
        workdir: str,
        expected_sha: str,
        command_id: str = "command:source-check",
        worker_id: str = "worker:source-check",
        fence: str = "fence:source-check",
    ) -> None:
        """Refuse to proceed if the local lane source is dirty or mismatched.

        This is the pre-flight gate a caller runs against the *local* lane
        worktree before ever building a remote dispatch plan from it.  It
        never inspects the filesystem itself -- ``git status``/``git
        rev-parse`` are executed through the caller's own local
        ``CommandExecutor`` and only their results are trusted.
        """

        sha = self.require_immutable_sha(expected_sha)
        status_command = ExecutionCommand(
            argv=(self.git_binary, "status", "--porcelain"),
            workdir=workdir,
            timeout_seconds=60,
        )
        status_result = executor.run(
            status_command,
            command_id=f"{command_id}:status",
            worker_id=worker_id,
            fence=fence,
        )
        if status_result.outcome != ExecutionOutcome.SUCCEEDED:
            raise SourceVerificationError(
                "could not determine local source cleanliness"
            )
        if status_result.stdout_tail.strip():
            raise DirtySourceError(
                "local lane worktree has uncommitted changes; refusing to "
                "remote-execute dirty state as if it were committed"
            )
        head_command = ExecutionCommand(
            argv=(self.git_binary, "rev-parse", "HEAD"),
            workdir=workdir,
            timeout_seconds=60,
        )
        head_result = executor.run(
            head_command,
            command_id=f"{command_id}:head",
            worker_id=worker_id,
            fence=fence,
        )
        if (
            head_result.outcome != ExecutionOutcome.SUCCEEDED
            or head_result.stdout_tail.strip() != sha
        ):
            raise SourceVerificationError(
                "local worktree HEAD does not match the requested immutable SHA"
            )

    def stage_commands(
        self,
        *,
        origin: str,
        tree_sha: str,
        parent_root: str,
        worktree_name: str,
        timeout_seconds: int = 1800,
    ) -> tuple[ExecutionCommand, ExecutionCommand, ExecutionCommand]:
        """Return the fixed clone/fetch/checkout sequence for one immutable SHA.

        The caller runs each command in order through its chosen
        ``CommandExecutor`` (local or remote) and must call
        :meth:`verify_staged_sha` afterward -- constructing these commands is
        never itself proof of materialization.  ``destination`` is always a
        new directory under ``parent_root`` (the worker's authorized,
        non-shared root), never a path the caller can redirect outside it.
        """

        sha = self.require_immutable_sha(tree_sha)
        origin = self._require_credential_free_origin(origin)
        worktree_name = self._require_safe_component(worktree_name)
        destination = f"{parent_root.rstrip('/')}/{worktree_name}"

        clone = ExecutionCommand(
            argv=(self.git_binary, "clone", "--no-checkout", "--", origin, destination),
            workdir=parent_root,
            timeout_seconds=timeout_seconds,
        )
        fetch = ExecutionCommand(
            argv=(self.git_binary, "fetch", "--depth", "1", "origin", sha),
            workdir=destination,
            timeout_seconds=timeout_seconds,
        )
        checkout = ExecutionCommand(
            argv=(self.git_binary, "checkout", "--detach", sha),
            workdir=destination,
            timeout_seconds=timeout_seconds,
        )
        return clone, fetch, checkout

    def verify_staged_sha(
        self,
        executor: object,
        *,
        destination: str,
        expected_sha: str,
        repository_id: str,
        command_id: str = "command:source-verify",
        worker_id: str = "worker:source-verify",
        fence: str = "fence:source-verify",
    ) -> StagedSource:
        """Independently re-derive cleanliness and HEAD identity after staging.

        Raises :class:`SourceVerificationError` on any mismatch, including a
        materialized worktree that is unexpectedly dirty -- a staged
        immutable commit must never carry uncommitted state either.
        """

        sha = self.require_immutable_sha(expected_sha)
        status_command = ExecutionCommand(
            argv=(self.git_binary, "status", "--porcelain"),
            workdir=destination,
            timeout_seconds=60,
        )
        status_result = executor.run(
            status_command,
            command_id=f"{command_id}:status",
            worker_id=worker_id,
            fence=fence,
        )
        if status_result.outcome != ExecutionOutcome.SUCCEEDED:
            raise SourceVerificationError(
                "could not verify materialized worktree cleanliness"
            )
        if status_result.stdout_tail.strip():
            raise SourceVerificationError(
                "materialized worktree is unexpectedly dirty after checkout"
            )
        head_command = ExecutionCommand(
            argv=(self.git_binary, "rev-parse", "HEAD"),
            workdir=destination,
            timeout_seconds=60,
        )
        head_result = executor.run(
            head_command,
            command_id=f"{command_id}:head",
            worker_id=worker_id,
            fence=fence,
        )
        if head_result.outcome != ExecutionOutcome.SUCCEEDED:
            raise SourceVerificationError("could not read materialized worktree HEAD")
        materialized_sha = head_result.stdout_tail.strip()
        if materialized_sha != sha:
            raise SourceVerificationError(
                f"materialized HEAD {materialized_sha!r} does not match "
                f"requested {sha!r}"
            )
        return StagedSource(
            repository_id=repository_id,
            tree_sha=sha,
            destination=destination,
            verified_at=datetime.now(UTC),
        )


__all__ = [
    "DirtySourceError",
    "ImmutableSourceStaging",
    "SourceVerificationError",
    "StagedSource",
]
