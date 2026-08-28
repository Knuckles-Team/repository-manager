"""Local fixed-argv execution with bounded output, cancellation, and fencing."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

from repository_manager.development import (
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FailureClass,
    RefusalCode,
)

from .bounded_log import BoundedLogSink, LogSink, RedactingLogSink, StreamName
from .cancellation import CancellationToken
from .process_supervisor import ProcessLike, ProcessSupervisor

_SHELL_META = re.compile(r"[\x00\r\n\t;&|<>$`(){}\[\]`\\]")
_SENSITIVE_ENV_NAME = re.compile(
    r"(?i)(?:token|secret|password|passwd|api[_-]?key|private[_-]?key|"
    r"authorization|credential|access[_-]?key|database[_-]?url|dsn|"
    r"connection[_-]?string|(?:^|[_-])pat(?:$|[_-]))"
)
_CREDENTIAL_URL = re.compile(r"(?i)^[a-z][a-z0-9+.-]*://[^\s/:@]+(?::[^\s@]*)?@")
_OPERATIONAL_ENV_NAMES = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LOGNAME",
        "SHELL",
        "LANG",
        "LC_ALL",
        "LC_CTYPE",
        "LC_MESSAGES",
        "TZ",
        "TMPDIR",
        "TMP",
        "TEMP",
        "XDG_CONFIG_HOME",
        "XDG_CACHE_HOME",
        "XDG_DATA_HOME",
        "XDG_STATE_HOME",
        "XDG_RUNTIME_DIR",
        "CARGO_HOME",
        "RUSTUP_HOME",
        "CARGO_TARGET_DIR",
        "VIRTUAL_ENV",
        "UV_CACHE_DIR",
        "GOPATH",
        "GOMODCACHE",
        "GOCACHE",
        "NPM_CONFIG_USERCONFIG",
        "npm_config_cache",
        # Windows process lookup/toolchain compatibility.
        "SYSTEMROOT",
        "PATHEXT",
    }
)


class Clock(Protocol):
    """Clock/sleep port used to keep supervisor tests deterministic."""

    def monotonic(self) -> float:
        """Return a monotonic timestamp."""

    def now(self) -> datetime:
        """Return an aware UTC timestamp."""

    def sleep(self, seconds: float) -> None:
        """Sleep or advance the injected clock."""


class RealClock:
    """Production wall and monotonic clock."""

    def monotonic(self) -> float:
        """Return the process monotonic clock."""

        return time.monotonic()

    def now(self) -> datetime:
        """Return the current aware UTC time."""

        return datetime.now(UTC)

    def sleep(self, seconds: float) -> None:
        """Sleep without busy-spinning the child supervisor."""

        time.sleep(seconds)


class ApprovedEnvironment:
    """Materialize approved environment references and track redactions.

    The default inherited environment is a small operational allowlist rather
    than the controller's complete environment.  It preserves process lookup,
    locale, temporary directories, home/XDG paths, and common tool caches while
    excluding ambient credentials.  Callers can provide an explicit mapping for
    secret references; only requested references are materialized and every
    selected value is returned separately so output capture can redact it before
    persistence or display.
    """

    def __init__(
        self,
        values: Mapping[str, str] | None = None,
        *,
        inherit: bool = True,
    ) -> None:
        self._values = dict(values or {})
        self._inherit = inherit

    def materialize(
        self, references: Sequence[str]
    ) -> tuple[dict[str, str], tuple[str, ...]]:
        """Resolve approved references or fail closed before process creation."""

        environment = self._inherited_environment()
        secrets = self._resolve_references(environment, references)
        secrets.extend(self._scan_sensitive_values(environment))
        return environment, tuple(secret for secret in secrets if secret)

    def _inherited_environment(self) -> dict[str, str]:
        if not self._inherit:
            return {}
        return {
            name: os.environ[name]
            for name in _OPERATIONAL_ENV_NAMES
            if name in os.environ
        }

    def _resolve_references(
        self, environment: dict[str, str], references: Sequence[str]
    ) -> list[str]:
        secrets: list[str] = []
        for reference in references:
            if reference in self._values:
                value = self._values[reference]
            elif reference in environment:
                value = environment[reference]
            else:
                raise ValueError(
                    f"approved environment reference is unavailable: {reference}"
                )
            environment[reference] = value
            secrets.append(value)
        return secrets

    @staticmethod
    def _scan_sensitive_values(environment: Mapping[str, str]) -> list[str]:
        return [
            value
            for name, value in environment.items()
            if _SENSITIVE_ENV_NAME.search(name) or _CREDENTIAL_URL.match(value)
        ]


@runtime_checkable
class CommandExecutor(Protocol):
    """Reusable executor boundary consumed by local and later remote workers."""

    def run(
        self,
        command: ExecutionCommand,
        *,
        command_id: str = "command:local",
        worker_id: str = "worker:local",
        fence: str = "fence:local",
        cancellation: CancellationToken | None = None,
        fence_check: Callable[[], bool] | None = None,
        heartbeat: Callable[[], bool | None] | None = None,
        log_sink: LogSink | None = None,
        publisher: PublicationPort | None = None,
    ) -> ExecutionResult:
        """Execute one frozen command and return its structured result."""


class ExecutionRefused(ValueError):
    """A command failed pre-launch validation and was not spawned."""

    def __init__(self, code: RefusalCode, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class PublicationDecision(StrEnum):
    """Result of an atomic fence-aware execution publication attempt."""

    ACCEPTED = "accepted"
    FENCED = "fenced"


class PublicationPort(Protocol):
    """Atomic WorkItem/result publication boundary.

    Implementations must compare ``fence`` with the currently leased WorkItem
    and publish the supplied result in the same transaction/CAS.  They return
    ``FENCED`` without publishing when the lease moved or expired; a boolean
    pre-check by the executor is only an optimization and is never the
    publication guarantee.
    """

    def publish(self, result: ExecutionResult, *, fence: str) -> PublicationDecision:
        """Atomically accept the result or reject it as stale."""


@dataclass(frozen=True)
class _RunIdentity:
    """Command/worker/fence identity plus start timestamps for one run."""

    command_id: str
    worker_id: str
    fence: str
    started_at: datetime
    started_mono: float


@dataclass(frozen=True)
class _LoopWatch:
    """Cancellation/fence/heartbeat/reader-error signals watched while a
    child process is running."""

    token: CancellationToken
    checker: Callable[[], bool]
    heartbeat: Callable[[], bool | None] | None
    sink_error: list[BaseException]


@dataclass(frozen=True)
class _RunState:
    """The outcome/failure-class/cleanup-ok trio threaded through post-exit
    reconciliation, sink close-out, and publication."""

    outcome: ExecutionOutcome
    failure_class: FailureClass | None
    cleanup_ok: bool


class LocalExecutor:
    """Run one command inside a canonical authorized worktree.

    No process is created until the command type, argv shape, workdir, limits,
    environment references, initial cancellation state, and initial fence have
    all passed validation.  A fence loss, heartbeat failure, timeout, or
    cancellation terminates the complete child process group and quarantines
    output so a late worker cannot publish a success.
    """

    def __init__(
        self,
        authorized_roots: Sequence[str | Path] | str | Path,
        *,
        worker_id: str = "worker:local",
        environment: Mapping[str, str] | None = None,
        inherit_environment: bool = True,
        supervisor: ProcessSupervisor | None = None,
        clock: Clock | None = None,
    ) -> None:
        roots = (
            (authorized_roots,)
            if isinstance(authorized_roots, (str, Path))
            else authorized_roots
        )
        self.authorized_roots = tuple(
            self._canonical_root(Path(root)) for root in roots
        )
        if not self.authorized_roots:
            raise ValueError("at least one authorized worktree root is required")
        if not worker_id.strip():
            raise ValueError("worker_id must be non-blank")
        self.worker_id = worker_id
        self.environment = ApprovedEnvironment(
            environment,
            inherit=inherit_environment,
        )
        self.clock = clock or RealClock()
        self.supervisor = supervisor or ProcessSupervisor(
            monotonic=self.clock.monotonic,
            sleep=self.clock.sleep,
        )

    def validate(self, command: ExecutionCommand) -> None:
        """Validate a command and raise :class:`ExecutionRefused` on failure."""

        try:
            self._validate(command)
        except ExecutionRefused:
            raise
        except (TypeError, ValueError) as exc:
            raise ExecutionRefused(RefusalCode.INVALID_REQUEST, str(exc)) from exc

    def run(
        self,
        command: ExecutionCommand,
        *,
        command_id: str = "command:local",
        worker_id: str | None = None,
        fence: str = "fence:local",
        cancellation: CancellationToken | None = None,
        fence_check: Callable[[], bool] | None = None,
        heartbeat: Callable[[], bool | None] | None = None,
        log_sink: LogSink | None = None,
        publisher: PublicationPort | None = None,
    ) -> ExecutionResult:
        """Execute a command with bounded cancellation and fence checks."""

        identity = _RunIdentity(
            command_id=command_id,
            worker_id=worker_id or self.worker_id,
            fence=fence,
            started_at=self.clock.now(),
            started_mono=self.clock.monotonic(),
        )
        token = cancellation or CancellationToken()
        checker = fence_check or (lambda: True)

        if not self._identity_is_well_formed(identity):
            invalid_identity = _RunIdentity(
                command_id="command:invalid",
                worker_id=self.worker_id,
                fence="fence:invalid",
                started_at=identity.started_at,
                started_mono=identity.started_mono,
            )
            return self._result(
                invalid_identity,
                ExecutionOutcome.REFUSED,
                FailureClass.INVALID_REQUEST,
                None,
            )

        environment, secrets, prep_failure = self._prepare_environment(command)
        if prep_failure is not None or environment is None:
            return self._result(
                identity,
                ExecutionOutcome.REFUSED,
                prep_failure or FailureClass.WORKER_ENVIRONMENT_FAILURE,
                None,
            )

        sink = self._build_log_sink(command, log_sink, secrets)

        early_refusal = self._pre_spawn_refusal(token, checker, sink)
        if early_refusal is not None:
            outcome, failure_class = early_refusal
            return self._finish_without_process(identity, outcome, failure_class, sink)

        process = self._try_spawn(command, environment)
        if process is None:
            sink.abort()
            return self._finish_without_process(
                identity,
                ExecutionOutcome.FAILED,
                FailureClass.WORKER_ENVIRONMENT_FAILURE,
                sink,
            )

        sink_error: list[BaseException] = []
        readers = self._start_readers(process, sink, sink_error)
        watch = _LoopWatch(
            token=token, checker=checker, heartbeat=heartbeat, sink_error=sink_error
        )
        state = self._run_until_exit(process, command, watch)

        returncode, cleanup_ok = self._finalize_process(
            process, readers, state.cleanup_ok
        )
        state = _RunState(state.outcome, state.failure_class, cleanup_ok)
        state, signal_number, exit_code = self._reconcile_outcome(
            state, watch, returncode
        )
        state = self._close_sink(sink, state)
        state = self._publish_result(publisher, identity, state, sink, returncode)

        return self._result(
            identity,
            state.outcome,
            state.failure_class,
            sink,
            exit_code=exit_code,
            signal_number=signal_number,
            cleanup_ok=state.cleanup_ok,
        )

    def execute(self, command: ExecutionCommand, **kwargs: object) -> ExecutionResult:
        """Alias for adapters that use an imperative ``execute`` verb."""

        return self.run(command, **kwargs)  # type: ignore[arg-type]

    @staticmethod
    def _identity_is_well_formed(identity: _RunIdentity) -> bool:
        return all(
            isinstance(value, str) and value.strip()
            for value in (identity.command_id, identity.worker_id, identity.fence)
        )

    def _prepare_environment(
        self, command: ExecutionCommand
    ) -> tuple[dict[str, str] | None, tuple[str, ...], FailureClass | None]:
        """Validate the command and materialize its environment, or report why not."""

        try:
            self.validate(command)
            environment, secrets = self.environment.materialize(
                command.environment_refs
            )
        except ExecutionRefused:
            return None, (), FailureClass.INVALID_REQUEST
        except ValueError:
            return None, (), FailureClass.WORKER_ENVIRONMENT_FAILURE
        return environment, secrets, None

    @staticmethod
    def _build_log_sink(
        command: ExecutionCommand,
        log_sink: LogSink | None,
        secrets: tuple[str, ...],
    ) -> LogSink:
        if log_sink is not None:
            return RedactingLogSink(log_sink, secrets)
        return BoundedLogSink(
            max_stdout_bytes=command.max_stdout_bytes,
            max_stderr_bytes=command.max_stderr_bytes,
            terminal_tail_bytes=min(
                max(command.max_stdout_bytes, command.max_stderr_bytes), 64 * 1024
            ),
            redactions=secrets,
        )

    def _pre_spawn_refusal(
        self,
        token: CancellationToken,
        checker: Callable[[], bool],
        sink: LogSink,
    ) -> tuple[ExecutionOutcome, FailureClass] | None:
        """Return the refusal outcome if the run must not spawn, else None."""

        if token.is_cancelled():
            return ExecutionOutcome.CANCELLED, FailureClass.CANCELLED_DEADLINE
        if not self._fence_is_valid(checker):
            sink.abort()
            return ExecutionOutcome.REFUSED, FailureClass.STALE_FENCE_DUPLICATE_EFFECT
        return None

    def _try_spawn(
        self, command: ExecutionCommand, environment: Mapping[str, str]
    ) -> ProcessLike | None:
        try:
            return self.supervisor.spawn(
                command.argv,
                cwd=Path(command.workdir),
                env=environment,
            )
        except (OSError, ValueError):
            return None

    def _run_until_exit(
        self,
        process: ProcessLike,
        command: ExecutionCommand,
        watch: _LoopWatch,
    ) -> _RunState:
        """Poll the child until it exits or a watched signal forces termination."""

        outcome = ExecutionOutcome.SUCCEEDED
        failure_class: FailureClass | None = None
        cleanup_ok = True
        next_heartbeat = self.clock.monotonic() + command.heartbeat_interval_seconds
        deadline = self.clock.monotonic() + command.timeout_seconds

        while process.poll() is None:
            now = self.clock.monotonic()
            terminate, loop_outcome, loop_failure, next_heartbeat = (
                self._next_loop_state(
                    now,
                    deadline,
                    next_heartbeat,
                    command.heartbeat_interval_seconds,
                    watch,
                )
            )
            if terminate:
                outcome = loop_outcome or outcome
                failure_class = loop_failure
                report = self.supervisor.terminate(process)
                cleanup_ok = report.cleanup_ok
                break
            self.clock.sleep(
                min(self.supervisor.poll_interval, max(0.0, deadline - now))
            )

        return _RunState(outcome, failure_class, cleanup_ok)

    def _next_loop_state(
        self,
        now: float,
        deadline: float,
        next_heartbeat: float,
        heartbeat_interval: float,
        watch: _LoopWatch,
    ) -> tuple[bool, ExecutionOutcome | None, FailureClass | None, float]:
        """Decide whether to keep polling or terminate, for one loop tick."""

        if watch.sink_error:
            return (
                True,
                ExecutionOutcome.FAILED,
                FailureClass.WORKER_ENVIRONMENT_FAILURE,
                next_heartbeat,
            )
        if watch.token.is_cancelled():
            return (
                True,
                ExecutionOutcome.CANCELLED,
                FailureClass.CANCELLED_DEADLINE,
                next_heartbeat,
            )
        if not self._fence_is_valid(watch.checker):
            return (
                True,
                ExecutionOutcome.REFUSED,
                FailureClass.STALE_FENCE_DUPLICATE_EFFECT,
                next_heartbeat,
            )
        if now >= deadline:
            return (
                True,
                ExecutionOutcome.TIMED_OUT,
                FailureClass.CANCELLED_DEADLINE,
                next_heartbeat,
            )
        if now >= next_heartbeat:
            if not self._heartbeat_is_valid(watch.heartbeat):
                return (
                    True,
                    ExecutionOutcome.REFUSED,
                    FailureClass.WORKER_ENVIRONMENT_FAILURE,
                    next_heartbeat,
                )
            return False, None, None, now + heartbeat_interval
        return False, None, None, next_heartbeat

    def _finalize_process(
        self,
        process: ProcessLike,
        readers: tuple[threading.Thread, ...],
        cleanup_ok: bool,
    ) -> tuple[int | None, bool]:
        """Force-terminate a still-running child, collect its exit code, and
        join output readers."""

        if process.poll() is None:
            report = self.supervisor.terminate(process)
            cleanup_ok = cleanup_ok and report.cleanup_ok
        returncode: int | None
        try:
            returncode = process.wait(timeout=0)
        except (ChildProcessError, OSError, TimeoutError):
            returncode = process.returncode
            cleanup_ok = False

        for reader in readers:
            reader.join(timeout=self.supervisor.termination_kill_seconds)
            if reader.is_alive():
                cleanup_ok = False

        return returncode, cleanup_ok

    def _reconcile_outcome(
        self,
        state: _RunState,
        watch: _LoopWatch,
        returncode: int | None,
    ) -> tuple[_RunState, int | None, int | None]:
        """Downgrade a SUCCEEDED outcome once the exit code and post-exit
        signals are known, and compute the final signal/exit-code pair."""

        outcome, failure_class, signal_number, exit_code = self._reconcile_exit_status(
            state.outcome, state.failure_class, watch.sink_error, returncode
        )
        outcome, failure_class = self._reconcile_post_exit_signals(
            outcome, failure_class, watch, state.cleanup_ok
        )
        return (
            _RunState(outcome, failure_class, state.cleanup_ok),
            signal_number,
            exit_code,
        )

    @staticmethod
    def _reconcile_exit_status(
        outcome: ExecutionOutcome,
        failure_class: FailureClass | None,
        sink_error: list[BaseException],
        returncode: int | None,
    ) -> tuple[ExecutionOutcome, FailureClass | None, int | None, int | None]:
        if sink_error and outcome == ExecutionOutcome.SUCCEEDED:
            outcome = ExecutionOutcome.FAILED
            failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE
        signal_number = (
            -returncode if returncode is not None and returncode < 0 else None
        )
        exit_code = returncode if returncode is not None and returncode >= 0 else None
        if outcome == ExecutionOutcome.SUCCEEDED and returncode != 0:
            outcome = ExecutionOutcome.FAILED
            failure_class = (
                FailureClass.WORKER_ENVIRONMENT_FAILURE
                if signal_number is not None
                else FailureClass.VALIDATION_CANDIDATE_FAILURE
            )
        return outcome, failure_class, signal_number, exit_code

    def _reconcile_post_exit_signals(
        self,
        outcome: ExecutionOutcome,
        failure_class: FailureClass | None,
        watch: _LoopWatch,
        cleanup_ok: bool,
    ) -> tuple[ExecutionOutcome, FailureClass | None]:
        if outcome == ExecutionOutcome.SUCCEEDED and watch.token.is_cancelled():
            outcome = ExecutionOutcome.CANCELLED
            failure_class = FailureClass.CANCELLED_DEADLINE
        if outcome == ExecutionOutcome.SUCCEEDED and not self._fence_is_valid(
            watch.checker
        ):
            outcome = ExecutionOutcome.REFUSED
            failure_class = FailureClass.STALE_FENCE_DUPLICATE_EFFECT
        if outcome == ExecutionOutcome.SUCCEEDED and not cleanup_ok:
            outcome = ExecutionOutcome.FAILED
            failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE
        return outcome, failure_class

    @staticmethod
    def _close_sink(sink: LogSink, state: _RunState) -> _RunState:
        try:
            if state.outcome == ExecutionOutcome.REFUSED:
                sink.abort()
            else:
                sink.close()
        except Exception:  # pragma: no cover - defensive sink boundary
            return _RunState(
                ExecutionOutcome.FAILED, FailureClass.WORKER_ENVIRONMENT_FAILURE, False
            )
        return state

    def _publish_result(
        self,
        publisher: PublicationPort | None,
        identity: _RunIdentity,
        state: _RunState,
        sink: LogSink,
        returncode: int | None,
    ) -> _RunState:
        """Publish a SUCCEEDED result atomically, downgrading it if the fence
        moved or publication itself failed."""

        if state.outcome != ExecutionOutcome.SUCCEEDED or publisher is None:
            return state

        publication_result = self._result(
            identity,
            state.outcome,
            None,
            sink,
            exit_code=returncode if returncode == 0 else None,
            signal_number=None,
            cleanup_ok=state.cleanup_ok,
        )
        try:
            decision = publisher.publish(publication_result, fence=identity.fence)
        except Exception:  # pragma: no cover - defensive publication boundary
            return _RunState(
                ExecutionOutcome.FAILED, FailureClass.WORKER_ENVIRONMENT_FAILURE, False
            )

        if decision != PublicationDecision.ACCEPTED:
            cleanup_ok = state.cleanup_ok
            try:
                sink.abort()
            except Exception:  # pragma: no cover - defensive sink boundary
                cleanup_ok = False
            return _RunState(
                ExecutionOutcome.REFUSED,
                FailureClass.STALE_FENCE_DUPLICATE_EFFECT,
                cleanup_ok,
            )
        return state

    def _validate(self, command: ExecutionCommand) -> None:
        if not isinstance(command, ExecutionCommand):
            raise ExecutionRefused(
                RefusalCode.INVALID_REQUEST,
                "executor accepts only ExecutionCommand, not a shell string",
            )
        self._validate_argv(command.argv)
        self._validate_resource_limits(command)
        self._validate_workdir(command)

    @staticmethod
    def _validate_argv(argv: Sequence[str]) -> None:
        LocalExecutor._validate_argv_content(argv)
        LocalExecutor._validate_executable_token(argv[0])

    @staticmethod
    def _validate_argv_content(argv: Sequence[str]) -> None:
        if isinstance(argv, (str, bytes)) or not argv:
            raise ExecutionRefused(
                RefusalCode.SHELL_COMMAND_FORBIDDEN,
                "executor requires a non-empty fixed argv sequence",
            )
        if any("\x00" in item for item in argv):
            raise ExecutionRefused(
                RefusalCode.SHELL_COMMAND_FORBIDDEN,
                "NUL bytes are not valid in fixed argv",
            )
        if any(any(ord(char) < 0x20 for char in item) for item in argv):
            raise ExecutionRefused(
                RefusalCode.SHELL_COMMAND_FORBIDDEN,
                "control characters are not valid in fixed argv",
            )

    @staticmethod
    def _validate_executable_token(executable: str) -> None:
        if not executable.strip() or any(char.isspace() for char in executable):
            raise ExecutionRefused(
                RefusalCode.SHELL_COMMAND_FORBIDDEN,
                "the executable must be one argv token, not a shell string",
            )
        if _SHELL_META.search(executable):
            raise ExecutionRefused(
                RefusalCode.SHELL_COMMAND_FORBIDDEN,
                "shell escaping/control syntax is not permitted in executable argv",
            )

    @staticmethod
    def _validate_resource_limits(command: ExecutionCommand) -> None:
        if command.timeout_seconds <= 0 or command.heartbeat_interval_seconds <= 0:
            raise ExecutionRefused(
                RefusalCode.RESOURCE_LIMIT_INVALID,
                "timeout and heartbeat limits must be positive",
            )
        if any(
            bound < 0
            for bound in (
                command.max_stdout_bytes,
                command.max_stderr_bytes,
                command.max_artifact_bytes,
            )
        ):
            raise ExecutionRefused(
                RefusalCode.RESOURCE_LIMIT_INVALID,
                "output and artifact limits must be non-negative",
            )

    def _validate_workdir(self, command: ExecutionCommand) -> None:
        workdir = self._canonical_root(Path(command.workdir))
        if not workdir.is_dir():
            raise ExecutionRefused(
                RefusalCode.PATH_OUTSIDE_CONFIGURED_ROOT,
                "authorized execution workdir does not exist as a directory",
            )
        if not any(
            root == workdir or root in workdir.parents for root in self.authorized_roots
        ):
            raise ExecutionRefused(
                RefusalCode.PATH_OUTSIDE_CONFIGURED_ROOT,
                "execution workdir is outside configured authorized roots",
            )
        if command.argv[0] in {".", ".."}:
            raise ExecutionRefused(
                RefusalCode.INVALID_REQUEST,
                "executable token is not runnable",
            )

    @staticmethod
    def _canonical_root(path: Path) -> Path:
        return path.expanduser().resolve(strict=False)

    @staticmethod
    def _fence_is_valid(checker: Callable[[], bool]) -> bool:
        try:
            return bool(checker())
        except Exception:
            return False

    @staticmethod
    def _heartbeat_is_valid(heartbeat: Callable[[], bool | None] | None) -> bool:
        if heartbeat is None:
            return True
        try:
            return heartbeat() is not False
        except Exception:
            return False

    @staticmethod
    def _start_readers(
        process: ProcessLike,
        sink: LogSink,
        sink_error: list[BaseException],
    ) -> tuple[threading.Thread, ...]:
        readers: list[threading.Thread] = []
        for stream_name in ("stdout", "stderr"):
            stream = getattr(process, stream_name, None)
            if stream is None:
                continue

            def _drain(
                stream_name: StreamName = stream_name,
                stream: object = stream,
            ) -> None:
                try:
                    while True:
                        chunk = stream.read(4096)  # type: ignore[attr-defined]
                        if not chunk:
                            break
                        if isinstance(chunk, str):
                            chunk = chunk.encode("utf-8", errors="replace")
                        sink.write(stream_name, chunk)
                except BaseException as exc:  # thread errors must reach supervisor
                    sink_error.append(exc)

            reader = threading.Thread(
                target=_drain,
                name=f"repository-manager-{stream_name}-drain",
                daemon=True,
            )
            reader.start()
            readers.append(reader)
        return tuple(readers)

    def _finish_without_process(
        self,
        identity: _RunIdentity,
        outcome: ExecutionOutcome,
        failure_class: FailureClass,
        sink: LogSink,
    ) -> ExecutionResult:
        sink.close()
        return self._result(identity, outcome, failure_class, sink)

    def _result(
        self,
        identity: _RunIdentity,
        outcome: ExecutionOutcome,
        failure_class: FailureClass | None,
        log_sink: LogSink | None,
        *,
        exit_code: int | None = None,
        signal_number: int | None = None,
        cleanup_ok: bool = True,
    ) -> ExecutionResult:
        finished_at = self.clock.now()
        duration_ms = max(
            0,
            int(round((self.clock.monotonic() - identity.started_mono) * 1000)),
        )
        return ExecutionResult(
            command_id=identity.command_id,
            outcome=outcome,
            exit_code=exit_code,
            signal=signal_number,
            started_at=identity.started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            worker_id=identity.worker_id,
            fence=identity.fence,
            stdout_tail=log_sink.tail_text("stdout") if log_sink else "",
            stderr_tail=log_sink.tail_text("stderr") if log_sink else "",
            failure_class=failure_class,
            cleanup_ok=cleanup_ok,
        )


__all__ = [
    "ApprovedEnvironment",
    "Clock",
    "CommandExecutor",
    "ExecutionRefused",
    "LocalExecutor",
    "PublicationDecision",
    "PublicationPort",
    "RealClock",
]
