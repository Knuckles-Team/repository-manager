"""RMDD-15: immutable source materialization -- dirty-source refusal.

The lane's non-negotiable constraint: "Repository input is an immutable
commit/generation. Dirty lane state must never be remote-executed as if
committed." ``ImmutableSourceStaging`` never inspects the filesystem itself
-- ``git status``/``git rev-parse`` are executed through the caller's own
``CommandExecutor`` (local or remote) and only their *results* are trusted,
so these tests use a scripted fake executor that never spawns a real
subprocess.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from repository_manager.development import (
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FailureClass,
)
from repository_manager.remote_execution.source_staging import (
    DirtySourceError,
    ImmutableSourceStaging,
    SourceVerificationError,
    StagedSource,
)

_CLEAN_SHA = "a" * 40
_OTHER_SHA = "b" * 40


class _ScriptedExecutor:
    """Return a scripted result keyed by the trailing argv token.

    ``status``/``rev-parse`` are dispatched with distinguishable final argv
    elements (``--porcelain`` / ``HEAD``), so this fake keys on
    ``command.argv[-1]`` without needing to special-case call order.
    """

    def __init__(self, responses: dict[str, ExecutionOutcome | str]) -> None:
        # responses maps argv[-1] -> either an ExecutionOutcome (failure) or
        # a stdout string (success).
        self._responses = responses
        self.commands: list[ExecutionCommand] = []

    def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
        self.commands.append(command)
        key = command.argv[-1]
        response = self._responses.get(key, "")
        now = datetime.now(UTC)
        if isinstance(response, ExecutionOutcome) and response != (
            ExecutionOutcome.SUCCEEDED
        ):
            return ExecutionResult(
                command_id=str(kwargs.get("command_id", "command:x")),
                outcome=response,
                started_at=now,
                finished_at=now,
                duration_ms=0,
                worker_id=str(kwargs.get("worker_id", "worker:x")),
                fence=str(kwargs.get("fence", "fence:x")),
                failure_class=FailureClass.WORKER_ENVIRONMENT_FAILURE,
            )
        stdout = response if isinstance(response, str) else ""
        return ExecutionResult(
            command_id=str(kwargs.get("command_id", "command:x")),
            outcome=ExecutionOutcome.SUCCEEDED,
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            worker_id=str(kwargs.get("worker_id", "worker:x")),
            fence=str(kwargs.get("fence", "fence:x")),
            stdout_tail=stdout,
        )


def test_require_immutable_sha_accepts_a_full_40_hex_sha() -> None:
    staging = ImmutableSourceStaging()
    assert staging.require_immutable_sha(_CLEAN_SHA) == _CLEAN_SHA


@pytest.mark.parametrize(
    "mutable_ref",
    [
        "main",
        "refs/heads/main",
        "HEAD",
        "v1.2.3",
        _CLEAN_SHA[:12],  # short SHA
        _CLEAN_SHA.upper(),  # uppercase hex is not the canonical form
        "",
        "not-a-sha at all",
    ],
)
def test_require_immutable_sha_refuses_a_mutable_or_ambiguous_ref(
    mutable_ref: str,
) -> None:
    staging = ImmutableSourceStaging()
    with pytest.raises(ValueError):
        staging.require_immutable_sha(mutable_ref)


def test_check_local_source_clean_refuses_dirty_lane_state() -> None:
    """A caller must never remote-execute uncommitted lane changes as if committed."""

    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor(
        {"--porcelain": " M repository_manager/some_file.py\n"}
    )
    with pytest.raises(DirtySourceError):
        staging.check_local_source_clean(
            executor, workdir="/srv/lane/repo", expected_sha=_CLEAN_SHA
        )


def test_check_local_source_clean_refuses_head_mismatch() -> None:
    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor({"--porcelain": "", "HEAD": _OTHER_SHA})
    with pytest.raises(SourceVerificationError):
        staging.check_local_source_clean(
            executor, workdir="/srv/lane/repo", expected_sha=_CLEAN_SHA
        )


def test_check_local_source_clean_accepts_clean_matching_lane() -> None:
    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor({"--porcelain": "", "HEAD": _CLEAN_SHA})
    # Must not raise.
    staging.check_local_source_clean(
        executor, workdir="/srv/lane/repo", expected_sha=_CLEAN_SHA
    )


def test_check_local_source_clean_refuses_when_status_cannot_be_determined() -> None:
    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor({"--porcelain": ExecutionOutcome.FAILED})
    with pytest.raises(SourceVerificationError):
        staging.check_local_source_clean(
            executor, workdir="/srv/lane/repo", expected_sha=_CLEAN_SHA
        )


def test_stage_commands_refuses_a_mutable_sha() -> None:
    staging = ImmutableSourceStaging()
    with pytest.raises(ValueError):
        staging.stage_commands(
            origin="/srv/origin/repo.git",
            tree_sha="main",
            parent_root="/srv/remote-worktrees",
            worktree_name="job-1",
        )


def test_stage_commands_refuses_credential_bearing_origin() -> None:
    staging = ImmutableSourceStaging()
    with pytest.raises(ValueError):
        staging.stage_commands(
            origin="https://user:hunter2@example.com/repo.git",
            tree_sha=_CLEAN_SHA,
            parent_root="/srv/remote-worktrees",
            worktree_name="job-1",
        )


@pytest.mark.parametrize(
    "unsafe_worktree_name",
    ["../escape", "job/../../etc", "/absolute", "job;rm -rf /", ""],
)
def test_stage_commands_refuses_unsafe_worktree_name(unsafe_worktree_name: str) -> None:
    staging = ImmutableSourceStaging()
    with pytest.raises(ValueError):
        staging.stage_commands(
            origin="/srv/origin/repo.git",
            tree_sha=_CLEAN_SHA,
            parent_root="/srv/remote-worktrees",
            worktree_name=unsafe_worktree_name,
        )


def test_stage_commands_destination_is_always_under_parent_root() -> None:
    staging = ImmutableSourceStaging()
    clone, fetch, checkout = staging.stage_commands(
        origin="/srv/origin/repo.git",
        tree_sha=_CLEAN_SHA,
        parent_root="/srv/remote-worktrees",
        worktree_name="job-1",
    )
    assert clone.argv[-1] == "/srv/remote-worktrees/job-1"
    assert fetch.workdir == "/srv/remote-worktrees/job-1"
    assert checkout.workdir == "/srv/remote-worktrees/job-1"
    assert checkout.argv == ("git", "checkout", "--detach", _CLEAN_SHA)


def test_verify_staged_sha_refuses_a_dirty_materialized_worktree() -> None:
    """Even a freshly staged immutable commit must never carry uncommitted state."""

    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor({"--porcelain": "?? unexpected-file\n"})
    with pytest.raises(SourceVerificationError):
        staging.verify_staged_sha(
            executor,
            destination="/srv/remote-worktrees/job-1",
            expected_sha=_CLEAN_SHA,
            repository_id="repository:demo",
        )


def test_verify_staged_sha_refuses_head_mismatch_after_checkout() -> None:
    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor({"--porcelain": "", "HEAD": _OTHER_SHA})
    with pytest.raises(SourceVerificationError):
        staging.verify_staged_sha(
            executor,
            destination="/srv/remote-worktrees/job-1",
            expected_sha=_CLEAN_SHA,
            repository_id="repository:demo",
        )


def test_verify_staged_sha_returns_proof_on_a_clean_exact_match() -> None:
    staging = ImmutableSourceStaging()
    executor = _ScriptedExecutor({"--porcelain": "", "HEAD": _CLEAN_SHA})
    proof = staging.verify_staged_sha(
        executor,
        destination="/srv/remote-worktrees/job-1",
        expected_sha=_CLEAN_SHA,
        repository_id="repository:demo",
    )
    assert isinstance(proof, StagedSource)
    assert proof.tree_sha == _CLEAN_SHA
    assert proof.destination == "/srv/remote-worktrees/job-1"
    assert proof.repository_id == "repository:demo"
