"""Local fixed-argv execution with bounded output, cancellation, and fencing."""

from __future__ import annotations

import os
import re
import threading
import time
from collections.abc import Callable, Mapping, Sequence
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

        environment = (
            {
                name: os.environ[name]
                for name in _OPERATIONAL_ENV_NAMES
                if name in os.environ
            }
            if self._inherit
            else {}
        )
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

        for name, value in environment.items():
            if _SENSITIVE_ENV_NAME.search(name) or _CREDENTIAL_URL.match(value):
                secrets.append(value)
        return environment, tuple(secret for secret in secrets if secret)


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

        started_at = self.clock.now()
        started_mono = self.clock.monotonic()
        effective_worker = worker_id or self.worker_id
        token = cancellation or CancellationToken()
        checker = fence_check or (lambda: True)

        if not all(
            isinstance(value, str) and value.strip()
            for value in (command_id, effective_worker, fence)
        ):
            return self._result(
                command_id="command:invalid",
                worker_id=self.worker_id,
                fence="fence:invalid",
                outcome=ExecutionOutcome.REFUSED,
                failure_class=FailureClass.INVALID_REQUEST,
                started_at=started_at,
                started_mono=started_mono,
                log_sink=None,
            )

        try:
            self.validate(command)
            environment, secrets = self.environment.materialize(
                command.environment_refs
            )
        except ExecutionRefused:
            return self._result(
                command_id=command_id,
                worker_id=effective_worker,
                fence=fence,
                outcome=ExecutionOutcome.REFUSED,
                failure_class=FailureClass.INVALID_REQUEST,
                started_at=started_at,
                started_mono=started_mono,
                log_sink=None,
            )
        except ValueError:
            return self._result(
                command_id=command_id,
                worker_id=effective_worker,
                fence=fence,
                outcome=ExecutionOutcome.REFUSED,
                failure_class=FailureClass.WORKER_ENVIRONMENT_FAILURE,
                started_at=started_at,
                started_mono=started_mono,
                log_sink=None,
            )

        sink: LogSink = log_sink or BoundedLogSink(
            max_stdout_bytes=command.max_stdout_bytes,
            max_stderr_bytes=command.max_stderr_bytes,
            terminal_tail_bytes=min(
                max(command.max_stdout_bytes, command.max_stderr_bytes), 64 * 1024
            ),
            redactions=secrets,
        )
        if log_sink is not None:
            sink = RedactingLogSink(log_sink, secrets)

        if token.is_cancelled():
            return self._finish_without_process(
                command_id,
                effective_worker,
                fence,
                ExecutionOutcome.CANCELLED,
                FailureClass.CANCELLED_DEADLINE,
                started_at,
                started_mono,
                sink,
            )
        if not self._fence_is_valid(checker):
            sink.abort()
            return self._finish_without_process(
                command_id,
                effective_worker,
                fence,
                ExecutionOutcome.REFUSED,
                FailureClass.STALE_FENCE_DUPLICATE_EFFECT,
                started_at,
                started_mono,
                sink,
            )

        try:
            process = self.supervisor.spawn(
                command.argv,
                cwd=Path(command.workdir),
                env=environment,
            )
        except (OSError, ValueError):
            sink.abort()
            return self._finish_without_process(
                command_id,
                effective_worker,
                fence,
                ExecutionOutcome.FAILED,
                FailureClass.WORKER_ENVIRONMENT_FAILURE,
                started_at,
                started_mono,
                sink,
            )

        sink_error: list[BaseException] = []
        readers = self._start_readers(process, sink, sink_error)
        outcome = ExecutionOutcome.SUCCEEDED
        failure_class: FailureClass | None = None
        cleanup_ok = True
        termination_requested = False
        next_heartbeat = self.clock.monotonic() + command.heartbeat_interval_seconds
        deadline = self.clock.monotonic() + command.timeout_seconds

        while process.poll() is None:
            now = self.clock.monotonic()
            if sink_error:
                outcome = ExecutionOutcome.FAILED
                failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE
                termination_requested = True
            elif token.is_cancelled():
                outcome = ExecutionOutcome.CANCELLED
                failure_class = FailureClass.CANCELLED_DEADLINE
                termination_requested = True
            elif not self._fence_is_valid(checker):
                outcome = ExecutionOutcome.REFUSED
                failure_class = FailureClass.STALE_FENCE_DUPLICATE_EFFECT
                termination_requested = True
            elif now >= deadline:
                outcome = ExecutionOutcome.TIMED_OUT
                failure_class = FailureClass.CANCELLED_DEADLINE
                termination_requested = True
            elif now >= next_heartbeat:
                if not self._heartbeat_is_valid(heartbeat):
                    outcome = ExecutionOutcome.REFUSED
                    failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE
                    termination_requested = True
                else:
                    next_heartbeat = now + command.heartbeat_interval_seconds

            if termination_requested:
                report = self.supervisor.terminate(process)
                cleanup_ok = report.cleanup_ok
                break
            self.clock.sleep(
                min(self.supervisor.poll_interval, max(0.0, deadline - now))
            )

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
        if outcome == ExecutionOutcome.SUCCEEDED and token.is_cancelled():
            outcome = ExecutionOutcome.CANCELLED
            failure_class = FailureClass.CANCELLED_DEADLINE
        if outcome == ExecutionOutcome.SUCCEEDED and not self._fence_is_valid(checker):
            outcome = ExecutionOutcome.REFUSED
            failure_class = FailureClass.STALE_FENCE_DUPLICATE_EFFECT
        if outcome == ExecutionOutcome.SUCCEEDED and not cleanup_ok:
            outcome = ExecutionOutcome.FAILED
            failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE

        try:
            if outcome == ExecutionOutcome.REFUSED:
                sink.abort()
            else:
                sink.close()
        except Exception:  # pragma: no cover - defensive sink boundary
            outcome = ExecutionOutcome.FAILED
            failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE
            cleanup_ok = False

        if outcome == ExecutionOutcome.SUCCEEDED and publisher is not None:
            publication_result = self._result(
                command_id=command_id,
                worker_id=effective_worker,
                fence=fence,
                outcome=outcome,
                failure_class=None,
                started_at=started_at,
                started_mono=started_mono,
                log_sink=sink,
                exit_code=returncode if returncode == 0 else None,
                signal_number=None,
                cleanup_ok=cleanup_ok,
            )
            try:
                decision = publisher.publish(
                    publication_result,
                    fence=fence,
                )
            except Exception:  # pragma: no cover - defensive publication boundary
                outcome = ExecutionOutcome.FAILED
                failure_class = FailureClass.WORKER_ENVIRONMENT_FAILURE
                cleanup_ok = False
            else:
                if decision != PublicationDecision.ACCEPTED:
                    outcome = ExecutionOutcome.REFUSED
                    failure_class = FailureClass.STALE_FENCE_DUPLICATE_EFFECT
                    try:
                        sink.abort()
                    except Exception:  # pragma: no cover - defensive sink boundary
                        cleanup_ok = False

        return self._result(
            command_id=command_id,
            worker_id=effective_worker,
            fence=fence,
            outcome=outcome,
            failure_class=failure_class,
            started_at=started_at,
            started_mono=started_mono,
            log_sink=sink,
            exit_code=exit_code,
            signal_number=signal_number,
            cleanup_ok=cleanup_ok,
        )

    def execute(self, command: ExecutionCommand, **kwargs: object) -> ExecutionResult:
        """Alias for adapters that use an imperative ``execute`` verb."""

        return self.run(command, **kwargs)  # type: ignore[arg-type]

    def _validate(self, command: ExecutionCommand) -> None:
        if not isinstance(command, ExecutionCommand):
            raise ExecutionRefused(
                RefusalCode.INVALID_REQUEST,
                "executor accepts only ExecutionCommand, not a shell string",
            )
        argv = command.argv
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
        executable = argv[0]
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

        if executable in {".", ".."}:
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
        command_id: str,
        worker_id: str,
        fence: str,
        outcome: ExecutionOutcome,
        failure_class: FailureClass,
        started_at: datetime,
        started_mono: float,
        sink: LogSink,
    ) -> ExecutionResult:
        sink.close()
        return self._result(
            command_id=command_id,
            worker_id=worker_id,
            fence=fence,
            outcome=outcome,
            failure_class=failure_class,
            started_at=started_at,
            started_mono=started_mono,
            log_sink=sink,
        )

    def _result(
        self,
        *,
        command_id: str,
        worker_id: str,
        fence: str,
        outcome: ExecutionOutcome,
        failure_class: FailureClass | None,
        started_at: datetime,
        started_mono: float,
        log_sink: LogSink | None,
        exit_code: int | None = None,
        signal_number: int | None = None,
        cleanup_ok: bool = True,
    ) -> ExecutionResult:
        finished_at = self.clock.now()
        duration_ms = max(
            0,
            int(round((self.clock.monotonic() - started_mono) * 1000)),
        )
        return ExecutionResult(
            command_id=command_id,
            outcome=outcome,
            exit_code=exit_code,
            signal=signal_number,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            worker_id=worker_id,
            fence=fence,
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
