"""``RemoteWorkerExecutor``: a ``CommandExecutor``-shaped seam over RMDD-14.

This module composes exactly two frozen contracts without modifying either:

* :class:`repository_manager.execution.executor.CommandExecutor` -- the RMDD-07
  ``run(command, ...) -> ExecutionResult`` shape every worker implements.
* :mod:`tunnel_manager.remote_execution` -- the RMDD-14 entitlement-aware,
  alias-only, structured remote execution seam.

The frozen TM transport (``TunnelCommandExecutor.execute``) is a single
blocking call: it opens a connection, runs one fixed command, and closes.  It
has no streaming/cancellation hook, and this lane may not add one to
tunnel-manager.  Cancellation and heartbeat-driven termination are therefore
realized *cooperatively*, the same idiom :class:`LocalExecutor` uses for its
own child process, but projected onto the filesystem: a fixed, single-purpose
``touch <marker>`` command is dispatched over a **second** authorized
connection to the same target, and the installed remote worker bootstrap is
contracted (not implemented here -- see ``README.md``) to poll for that
marker at its own heartbeat cadence and terminate its process tree on sight.
This is honestly weaker than local in-process termination and is documented
as such; what is proven here is that the correct fixed command is dispatched
and that a result already flagged cancelled/fenced is never published as a
success.
"""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import ValidationError

from repository_manager.development import ExecutionCommand, ExecutionResult
from repository_manager.development import ExecutionOutcome as RmExecutionOutcome
from repository_manager.development import FailureClass as RmFailureClass
from repository_manager.execution.cancellation import CancellationToken
from repository_manager.execution.executor import (
    LogSink,
    PublicationDecision,
    PublicationPort,
)

# ``tunnel-manager`` is an optional dependency (the "remote" extra) that this
# lane cannot yet declare as resolvable: RMDD-14's ``remote_execution`` seam
# lives only on tunnel-manager's own unmerged integration branch and predates
# every published PyPI release, so no version constraint here would actually
# resolve.  Applying the dependency without an operator-approved source is
# exactly the "new dependency ... must stop at its decision gate until
# separately approved" case in CONTRACT-FREEZES.md's review rule, so the
# import is guarded per the repository's own try/except ImportError
# guardrail convention (AGENTS.md "Code Style & Conventions") rather than
# applied unconditionally.  ``import repository_manager.remote_execution``
# must succeed in the base install; only actually building or running a
# remote dispatch requires tunnel-manager to be present.
try:
    from tunnel_manager.connection_security import max_output_bytes, max_transfer_bytes
    from tunnel_manager.remote_execution import (
        AuthorizedTarget,
        RemoteCommandRequest,
        RemoteExecutionContext,
        RemoteExecutionResult,
    )
    from tunnel_manager.remote_execution import ExecutionOutcome as _TmExecutionOutcome
    from tunnel_manager.remote_execution import FailureClass as _TmFailureClass

    _TUNNEL_MANAGER_IMPORT_ERROR: ImportError | None = None
except ImportError as _tunnel_manager_import_error:  # pragma: no cover - exercised
    # by test_optional_dependency.py without the extra installed.
    AuthorizedTarget = RemoteCommandRequest = None  # type: ignore[assignment,misc]
    RemoteExecutionContext = RemoteExecutionResult = None  # type: ignore[assignment,misc]
    _TmExecutionOutcome = _TmFailureClass = None  # type: ignore[assignment]
    max_output_bytes = max_transfer_bytes = None  # type: ignore[assignment]
    _TUNNEL_MANAGER_IMPORT_ERROR = _tunnel_manager_import_error

logger = logging.getLogger(__name__)


class RemoteExecutionUnavailableError(RuntimeError):
    """Raised when a remote dispatch is attempted without tunnel-manager.

    The optional ``tunnel-manager`` dependency is not installed (or does not
    yet ship RMDD-14's ``remote_execution`` module).  Local execution is
    unaffected; this is refused, not silently degraded.
    """


def _require_tunnel_manager() -> None:
    if _TUNNEL_MANAGER_IMPORT_ERROR is not None:
        raise RemoteExecutionUnavailableError(
            "remote worker dispatch requires the optional 'tunnel-manager' "
            "dependency (repository-manager's 'remote' extra), which is not "
            "installed in this environment"
        ) from _TUNNEL_MANAGER_IMPORT_ERROR


def _outcome_map() -> dict[str, RmExecutionOutcome]:
    _require_tunnel_manager()
    return {member.value: member for member in RmExecutionOutcome}


def _failure_map() -> dict[str, RmFailureClass]:
    _require_tunnel_manager()
    return {member.value: member for member in RmFailureClass}


@runtime_checkable
class RemoteExecutorPort(Protocol):
    """The exact frozen RMDD-14 seam this executor drives.

    Production code injects ``tunnel_manager.remote_execution.TunnelCommandExecutor``
    directly; tests inject a deterministic fake with the identical call
    shape.  This module never constructs a ``Tunnel``/``HostConfig`` itself.
    """

    def execute(
        self,
        target: AuthorizedTarget,
        request: RemoteCommandRequest,
        actor: object,
        *,
        context: RemoteExecutionContext,
    ) -> RemoteExecutionResult:
        """Execute one structured command through an authorized alias."""


def to_remote_request(command: ExecutionCommand) -> RemoteCommandRequest:
    """Translate the RM C-04 command into the frozen TM wire request.

    Both models validate independently; an unsafe translation simply fails
    TM's own pydantic validation and is refused before dispatch, exactly as a
    directly caller-built request would be.

    Output/artifact bounds are clamped to TM's own local transport policy
    (never widened) before construction.  Without this, every
    default-constructed ``ExecutionCommand`` fails translation outright: RM's
    C-04 default ``max_artifact_bytes`` is 1 GiB while TM's
    ``max_transfer_bytes()`` policy defaults to 256 MiB, and
    ``RemoteCommandRequest`` refuses any request that exceeds its transport
    policy rather than silently truncating it.  A caller-supplied bound
    tighter than TM's policy is preserved as-is; only a bound *looser* than
    TM's own ceiling is narrowed, so this can only make a dispatch more
    conservative, never less.
    """

    _require_tunnel_manager()
    return RemoteCommandRequest(
        argv=command.argv,
        workdir=command.workdir,
        environment_refs=command.environment_refs,
        timeout_seconds=command.timeout_seconds,
        max_stdout_bytes=min(command.max_stdout_bytes, max_output_bytes()),
        max_stderr_bytes=min(command.max_stderr_bytes, max_output_bytes()),
        max_artifact_bytes=min(command.max_artifact_bytes, max_transfer_bytes()),
        heartbeat_interval_seconds=command.heartbeat_interval_seconds,
        cancellation_channel=command.cancellation_channel,
    )


def from_remote_result(result: RemoteExecutionResult) -> ExecutionResult:
    """Translate the frozen TM wire result into the RM C-04 execution result.

    TM's ``ExecutionOutcome``/``FailureClass`` are independently defined
    (tunnel-manager must not import repository-manager) but both frozen
    contracts pin identical wire *values*, so this is a value-level round
    trip rather than a heuristic remap.
    """

    outcome_map = _outcome_map()
    failure_map = _failure_map()
    outcome = outcome_map[result.outcome.value]
    failure_class = (
        failure_map[result.failure_class.value]
        if result.failure_class is not None
        else None
    )
    return ExecutionResult(
        command_id=result.command_id,
        outcome=outcome,
        exit_code=result.exit_code,
        signal=result.signal,
        started_at=result.started_at,
        finished_at=result.finished_at,
        duration_ms=result.duration_ms,
        worker_id=result.worker_id,
        fence=result.fence,
        stdout_tail=result.stdout_tail,
        stderr_tail=result.stderr_tail,
        failure_class=failure_class,
        cleanup_ok=result.cleanup_ok,
    )


def cancellation_marker_path(workdir: str, fence: str) -> str:
    """Return the deterministic per-attempt cooperative cancellation marker.

    The bootstrap contract (``README.md``) requires the remote worker to poll
    for this exact path under its own ``workdir`` and self-terminate its
    process tree on sight.  The path is derived only from already-validated
    opaque identifiers, never from caller-supplied text.
    """

    safe_fence = "".join(char if char.isalnum() else "_" for char in fence)
    return f"{workdir.rstrip('/')}/.rm-cancel-{safe_fence}"


class RemoteWorkerExecutor:
    """Run one C-04 command on one pre-authorized remote inventory target.

    Bound to one :class:`AuthorizedTarget` (produced by
    ``tunnel_manager.remote_execution.HostInventory.resolve``, never a raw
    host string) and one actor for its lifetime, mirroring how
    :class:`LocalExecutor` is bound to its authorized worktree roots.  Root
    and toolchain authorization for the target is the caller's
    responsibility (:mod:`repository_manager.remote_execution.registry`
    performs it at claim time, immediately before construction) -- this class
    only translates, dispatches, and applies fence/cancellation discipline.
    It never acquires a controller-local filesystem lock and never runs a
    Git branch-moving command.
    """

    def __init__(
        self,
        target: AuthorizedTarget,
        actor: object,
        *,
        port: RemoteExecutorPort,
        worker_id: str = "worker:remote",
        poll_interval_seconds: float = 0.2,
    ) -> None:
        _require_tunnel_manager()
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self.target = target
        self.actor = actor
        self.port = port
        self.worker_id = worker_id
        self.poll_interval_seconds = poll_interval_seconds

    def run(
        self,
        command: ExecutionCommand,
        *,
        command_id: str = "command:remote",
        worker_id: str | None = None,
        fence: str = "fence:remote",
        cancellation: CancellationToken | None = None,
        fence_check: object | None = None,
        heartbeat: object | None = None,
        log_sink: LogSink | None = None,
        publisher: PublicationPort | None = None,
    ) -> ExecutionResult:
        """Execute a command remotely with cooperative cancellation and fencing."""

        effective_worker = worker_id or self.worker_id
        token = cancellation or CancellationToken()
        checker = fence_check or (lambda: True)
        started_at = datetime.now(UTC)

        if token.is_cancelled():
            return self._refused(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.CANCELLED,
                RmFailureClass.CANCELLED_DEADLINE,
                started_at,
                log_sink,
                "cancelled before remote dispatch",
            )
        if not self._checker_ok(checker):
            return self._refused(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.REFUSED,
                RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT,
                started_at,
                log_sink,
                "fence invalid before remote dispatch",
            )

        try:
            request = to_remote_request(command)
        except ValidationError as exc:
            return self._refused(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.REFUSED,
                RmFailureClass.INVALID_REQUEST,
                started_at,
                log_sink,
                f"command failed remote translation: {exc.error_count()} error(s)",
            )

        context = RemoteExecutionContext(
            command_id=command_id, worker_id=effective_worker, fence=fence
        )

        outcome_box: list[RemoteExecutionResult] = []
        error_box: list[BaseException] = []

        def _dispatch() -> None:
            try:
                outcome_box.append(
                    self.port.execute(self.target, request, self.actor, context=context)
                )
            except BaseException as exc:  # noqa: BLE001 - reported, never swallowed
                error_box.append(exc)

        thread = threading.Thread(
            target=_dispatch, name="repository-manager-remote-dispatch", daemon=True
        )
        thread.start()
        # Track *which* check first failed, not just that one did: a lost
        # fence and an operator cancellation are different failure classes
        # (parity with LocalExecutor's own poll loop, which distinguishes
        # cancellation/fence-loss/heartbeat-failure into three different
        # outcomes rather than collapsing them all into "cancelled").  Once
        # one check fails the marker is sent and the reason is latched; later
        # checks never overwrite an already-latched reason.
        cancel_reason: str | None = None
        while thread.is_alive():
            if cancel_reason is None:
                if token.is_cancelled():
                    cancel_reason = "cancelled"
                elif not self._checker_ok(checker):
                    cancel_reason = "fence"
                elif not self._heartbeat_ok(heartbeat):
                    cancel_reason = "heartbeat"
                if cancel_reason is not None:
                    self._send_cancellation_marker(command.workdir, fence, context)
            thread.join(timeout=self.poll_interval_seconds)

        if error_box:
            dispatch_error = error_box[0]
            return self._refused(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.FAILED,
                RmFailureClass.WORKER_ENVIRONMENT_FAILURE,
                started_at,
                log_sink,
                f"remote dispatch raised {type(dispatch_error).__name__}",
            )

        result = from_remote_result(outcome_box[0])

        _CANCEL_REASON_OUTCOMES: dict[
            str, tuple[RmExecutionOutcome, RmFailureClass]
        ] = {
            "cancelled": (
                RmExecutionOutcome.CANCELLED,
                RmFailureClass.CANCELLED_DEADLINE,
            ),
            "fence": (
                RmExecutionOutcome.REFUSED,
                RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT,
            ),
            "heartbeat": (
                RmExecutionOutcome.REFUSED,
                RmFailureClass.WORKER_ENVIRONMENT_FAILURE,
            ),
        }
        if cancel_reason is not None and result.outcome == RmExecutionOutcome.SUCCEEDED:
            # A cancellation/fence-loss/heartbeat-failure marker was already
            # sent to the remote worker before it reported success; never
            # publish a success that raced that request (parity with
            # LocalExecutor's own post-hoc downgrade).
            downgraded_outcome, downgraded_failure = _CANCEL_REASON_OUTCOMES[
                cancel_reason
            ]
            result = result.model_copy(
                update={
                    "outcome": downgraded_outcome,
                    "failure_class": downgraded_failure,
                }
            )
        elif result.outcome == RmExecutionOutcome.SUCCEEDED and not self._checker_ok(
            checker
        ):
            result = result.model_copy(
                update={
                    "outcome": RmExecutionOutcome.REFUSED,
                    "failure_class": RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT,
                }
            )

        if log_sink is not None:
            self._emit_to_log_sink(log_sink, result)

        if result.outcome == RmExecutionOutcome.SUCCEEDED and publisher is not None:
            try:
                decision = publisher.publish(result, fence=fence)
            except Exception:  # noqa: BLE001 - defensive publication boundary
                result = result.model_copy(
                    update={
                        "outcome": RmExecutionOutcome.FAILED,
                        "failure_class": RmFailureClass.WORKER_ENVIRONMENT_FAILURE,
                    }
                )
            else:
                if decision != PublicationDecision.ACCEPTED:
                    result = result.model_copy(
                        update={
                            "outcome": RmExecutionOutcome.REFUSED,
                            "failure_class": (
                                RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT
                            ),
                        }
                    )

        return result

    def _send_cancellation_marker(
        self, workdir: str, fence: str, context: RemoteExecutionContext
    ) -> None:
        """Best-effort dispatch of the fixed cooperative cancellation marker.

        This opens a second authorized connection to the same target and
        never touches the primary in-flight dispatch.  A failure here is
        swallowed only in the sense that it does not raise past this
        boundary -- it cannot change the primary result, which is always
        derived from the primary dispatch outcome.
        """

        marker_request = RemoteCommandRequest(
            argv=("touch", cancellation_marker_path(workdir, fence)),
            workdir=workdir,
            timeout_seconds=30,
            heartbeat_interval_seconds=30,
        )
        marker_context = RemoteExecutionContext(
            command_id=f"{context.command_id}:cancel",
            worker_id=context.worker_id,
            fence=context.fence,
        )
        try:
            self.port.execute(
                self.target, marker_request, self.actor, context=marker_context
            )
        except Exception:  # noqa: BLE001 - best-effort; primary result is authoritative
            logger.debug(
                "best-effort cancellation marker dispatch failed for %s",
                context.command_id,
                exc_info=True,
            )

    @staticmethod
    def _checker_ok(checker: object) -> bool:
        try:
            return bool(checker())  # type: ignore[operator]
        except Exception:
            return False

    @staticmethod
    def _heartbeat_ok(heartbeat: object | None) -> bool:
        if heartbeat is None:
            return True
        try:
            return heartbeat() is not False  # type: ignore[operator]
        except Exception:
            return False

    @staticmethod
    def _emit_to_log_sink(log_sink: LogSink, result: ExecutionResult) -> None:
        """Feed the final bounded tails through an injected sink once.

        The frozen TM transport is blocking, not streaming, so this is a
        post-hoc emission rather than the incremental capture
        ``LocalExecutor`` performs; it exists only so a caller-supplied log
        sink observes a consistent write/close or abort lifecycle.
        """

        try:
            if result.stdout_tail:
                log_sink.write("stdout", result.stdout_tail.encode("utf-8"))
            if result.stderr_tail:
                log_sink.write("stderr", result.stderr_tail.encode("utf-8"))
            if result.outcome == RmExecutionOutcome.REFUSED:
                log_sink.abort()
            else:
                log_sink.close()
        except Exception:  # noqa: BLE001 - sink failures must not mask the result
            logger.debug("log sink emission failed", exc_info=True)

    def _refused(
        self,
        command_id: str,
        worker_id: str,
        fence: str,
        outcome: RmExecutionOutcome,
        failure_class: RmFailureClass,
        started_at: datetime,
        log_sink: LogSink | None,
        note: str,
    ) -> ExecutionResult:
        finished_at = datetime.now(UTC)
        result = ExecutionResult(
            command_id=command_id,
            outcome=outcome,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=max(0, int((finished_at - started_at).total_seconds() * 1000)),
            worker_id=worker_id,
            fence=fence,
            stderr_tail=note,
            failure_class=failure_class,
        )
        if log_sink is not None:
            self._emit_to_log_sink(log_sink, result)
        return result


__all__ = [
    "RemoteExecutionUnavailableError",
    "RemoteExecutorPort",
    "RemoteWorkerExecutor",
    "cancellation_marker_path",
    "from_remote_result",
    "to_remote_request",
]
