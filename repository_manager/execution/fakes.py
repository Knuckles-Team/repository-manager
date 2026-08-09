"""Deterministic fixtures for executor and scheduler integration tests."""

from __future__ import annotations

import io
import signal
from datetime import UTC, datetime, timedelta
from typing import Any

from repository_manager.development import (
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FailureClass,
)

from .cancellation import CancellationToken


class FakeClock:
    """A manually advanced clock implementing the local executor clock port."""

    def __init__(self, *, start: datetime | None = None) -> None:
        self._now = (start or datetime(2026, 1, 1, tzinfo=UTC)).astimezone(UTC)
        self._monotonic = 0.0

    def monotonic(self) -> float:
        """Return the fake monotonic time."""

        return self._monotonic

    def now(self) -> datetime:
        """Return the fake aware UTC time."""

        return self._now

    def sleep(self, seconds: float) -> None:
        """Advance rather than blocking the test process."""

        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        """Advance both clocks by a non-negative interval."""

        if seconds < 0:
            raise ValueError("fake clock cannot move backwards")
        self._monotonic += seconds
        self._now += timedelta(seconds=seconds)


class FakeProcess:
    """Small Popen-shaped process used to exercise supervisor cleanup paths."""

    _next_pid = 20_000

    def __init__(
        self,
        *,
        returncode: int | None = 0,
        stdout: bytes = b"",
        stderr: bytes = b"",
    ) -> None:
        self.pid = self._next_pid
        type(self)._next_pid += 1
        self.returncode = returncode
        self.stdout = io.BytesIO(stdout)
        self.stderr = io.BytesIO(stderr)
        self.term_signals: list[int] = []
        self.wait_calls = 0

    def poll(self) -> int | None:
        """Return the configured process state."""

        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        """Return the configured exit code or raise while still running."""

        self.wait_calls += 1
        if self.returncode is None:
            raise TimeoutError("fake process remains running")
        return self.returncode

    def send_signal(self, sig: int) -> None:
        """Record a signal and terminate the fake process."""

        self.term_signals.append(sig)
        self.returncode = -sig

    def terminate(self) -> None:
        """Record a TERM request."""

        self.send_signal(signal.SIGTERM)

    def kill(self) -> None:
        """Record a KILL request."""

        self.send_signal(signal.SIGKILL)


class FakeExecutor:
    """Predictable command executor for later service/scheduler tests."""

    def __init__(self, result: ExecutionResult | None = None) -> None:
        self.result = result
        self.commands: list[ExecutionCommand] = []
        self.cancellations: list[CancellationToken | None] = []

    def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
        """Record a command and return a configured or successful result."""

        self.commands.append(command)
        self.cancellations.append(kwargs.get("cancellation"))
        if self.result is not None:
            return self.result
        now = datetime.now(UTC)
        return ExecutionResult(
            command_id=str(kwargs.get("command_id", "command:fake")),
            outcome=ExecutionOutcome.SUCCEEDED,
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            worker_id=str(kwargs.get("worker_id", "worker:fake")),
            fence=str(kwargs.get("fence", "fence:fake")),
        )

    def failed(self, *, command_id: str = "command:fake") -> ExecutionResult:
        """Return a deterministic worker-failure result for fixtures."""

        now = datetime.now(UTC)
        return ExecutionResult(
            command_id=command_id,
            outcome=ExecutionOutcome.FAILED,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            worker_id="worker:fake",
            fence="fence:fake",
            failure_class=FailureClass.WORKER_ENVIRONMENT_FAILURE,
        )


__all__ = ["FakeClock", "FakeExecutor", "FakeProcess"]
