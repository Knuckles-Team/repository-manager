"""Process-group lifecycle management for local executor attempts."""

from __future__ import annotations

import os
import signal
import subprocess  # nosec B404 - argv is supplied as a sequence and shell is disabled
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast


class ProcessLike(Protocol):
    """Subset of ``subprocess.Popen`` used by the supervisor and fakes."""

    pid: int
    returncode: int | None

    def poll(self) -> int | None:
        """Return the exit code when the process has terminated."""

    def wait(self, timeout: float | None = None) -> int:
        """Wait for and return the process exit code."""

    def send_signal(self, sig: int) -> None:
        """Send a signal to the process."""

    def terminate(self) -> None:
        """Request process termination."""

    def kill(self) -> None:
        """Force process termination."""


@dataclass(frozen=True, slots=True)
class TerminationReport:
    """Evidence from a bounded termination escalation."""

    term_sent: bool
    kill_sent: bool
    reaped: bool
    error: str | None = None

    @property
    def cleanup_ok(self) -> bool:
        """Whether the process was reaped without a supervisor error."""

        return self.reaped and self.error is None


PopenFactory = Callable[..., ProcessLike]


class ProcessSupervisor:
    """Start and terminate one process group without shell interpretation.

    POSIX children start a new session, making the process PID the process-group
    leader.  Termination then targets the complete descendant group rather than
    only the direct child.  Every wait is bounded; a stubborn group receives a
    final ``SIGKILL`` and the result reports whether reaping succeeded.
    """

    def __init__(
        self,
        *,
        poll_interval: float = 0.05,
        termination_grace_seconds: float = 2.0,
        termination_kill_seconds: float = 2.0,
        monotonic: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
        popen_factory: PopenFactory | None = None,
    ) -> None:
        if poll_interval <= 0:
            raise ValueError("poll_interval must be positive")
        if termination_grace_seconds <= 0 or termination_kill_seconds <= 0:
            raise ValueError("termination deadlines must be positive")
        self.poll_interval = poll_interval
        self.termination_grace_seconds = termination_grace_seconds
        self.termination_kill_seconds = termination_kill_seconds
        self._monotonic = monotonic if monotonic is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep
        self._popen_factory = cast(PopenFactory, popen_factory or subprocess.Popen)

    def spawn(
        self,
        argv: Sequence[str],
        *,
        cwd: Path,
        env: Mapping[str, str],
    ) -> ProcessLike:
        """Spawn one fixed-argv process in its own process group."""

        if isinstance(argv, (str, bytes)) or not argv:
            raise ValueError("spawn requires a non-empty argv sequence")
        if any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("spawn argv entries must be non-empty strings")
        kwargs: dict[str, object] = {
            "cwd": str(cwd),
            "env": dict(env),
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.PIPE,
            "stderr": subprocess.PIPE,
            "shell": False,
            "close_fds": True,
            "text": False,
        }
        if os.name == "posix":
            kwargs["start_new_session"] = True
        else:
            kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
        return self._popen_factory(tuple(argv), **kwargs)

    def terminate(self, process: ProcessLike) -> TerminationReport:
        """Escalate TERM then KILL and reap the complete process group."""

        term_sent = False
        kill_sent = False
        errors: list[str] = []

        if process.poll() is None:
            try:
                self._signal_group(process, signal.SIGTERM)
                term_sent = True
            except ProcessLookupError:
                # The process exited between poll and signal.  Reaping below
                # still provides the authoritative cleanup result.
                pass
            except OSError as exc:
                errors.append(f"term:{type(exc).__name__}")

        # Escalate even when the direct parent exited during the grace
        # window: a descendant may have ignored TERM while retaining the
        # process-group membership and output pipes.
        self._wait_until_exit(process, self.termination_grace_seconds)
        try:
            self._signal_group(process, signal.SIGKILL)
            kill_sent = True
        except ProcessLookupError:
            pass
        except OSError as exc:
            errors.append(f"kill:{type(exc).__name__}")
        self._wait_until_exit(process, self.termination_kill_seconds)

        reaped = process.poll() is not None
        if reaped:
            try:
                process.wait(timeout=0)
            except (subprocess.TimeoutExpired, ChildProcessError):
                reaped = False
            except OSError as exc:
                errors.append(f"wait:{type(exc).__name__}")
                reaped = False
        return TerminationReport(
            term_sent=term_sent,
            kill_sent=kill_sent,
            reaped=reaped,
            error=";".join(errors) if errors else None,
        )

    def _signal_group(self, process: ProcessLike, sig: int) -> None:
        if os.name == "posix":
            try:
                os.killpg(os.getpgid(process.pid), sig)
            except ProcessLookupError:
                # Injectable fake processes and a child that exits between
                # lookup and signal still receive the bounded fallback.
                if sig == signal.SIGTERM:
                    process.terminate()
                else:
                    process.kill()
        elif sig == signal.SIGTERM:
            process.terminate()
        else:
            process.kill()

    def _wait_until_exit(self, process: ProcessLike, timeout: float) -> bool:
        deadline = float(self._monotonic()) + timeout
        while process.poll() is None and float(self._monotonic()) < deadline:
            remaining = max(0.0, deadline - float(self._monotonic()))
            self._sleep(min(self.poll_interval, remaining))
        return process.poll() is not None


__all__ = ["ProcessLike", "ProcessSupervisor", "TerminationReport"]
