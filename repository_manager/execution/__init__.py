"""Bounded, fenced local command execution.

The execution package is deliberately independent of scheduling, WorkItems, Git,
MCP, and remote transport.  It accepts the frozen development contracts and
returns the same structured result for every local worker attempt.
"""

from .bounded_log import (
    BoundedLogSink,
    LogSink,
    LogSinkClosed,
    LogSnapshot,
    RedactingLogSink,
    StreamingRedactor,
    StreamName,
)
from .cancellation import CancellationSnapshot, CancellationToken
from .executor import (
    ApprovedEnvironment,
    Clock,
    CommandExecutor,
    ExecutionRefused,
    LocalExecutor,
    RealClock,
)
from .fakes import FakeClock, FakeExecutor, FakeProcess
from .process_supervisor import ProcessLike, ProcessSupervisor, TerminationReport

__all__ = [
    "ApprovedEnvironment",
    "BoundedLogSink",
    "CancellationSnapshot",
    "CancellationToken",
    "Clock",
    "CommandExecutor",
    "ExecutionRefused",
    "FakeClock",
    "FakeExecutor",
    "FakeProcess",
    "LocalExecutor",
    "LogSink",
    "LogSinkClosed",
    "LogSnapshot",
    "RedactingLogSink",
    "ProcessLike",
    "ProcessSupervisor",
    "RealClock",
    "StreamName",
    "StreamingRedactor",
    "TerminationReport",
]
