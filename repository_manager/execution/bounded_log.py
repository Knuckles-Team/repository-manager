"""Bounded output capture and streaming for local execution.

Only a short redacted terminal tail is retained in memory.  Output can be
forwarded to an injected sink while it is read, but the executor never needs to
accumulate a command's complete stdout or stderr in its process memory.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Literal, Protocol

StreamName = Literal["stdout", "stderr"]
ChunkWriter = Callable[[StreamName, bytes], None]


class LogSinkClosed(RuntimeError):
    """Raised when a closed log sink receives another chunk."""


class LogSink(Protocol):
    """The small output port used by :class:`LocalExecutor`."""

    def write(self, stream: StreamName, chunk: bytes) -> None:
        """Consume one already-read output chunk."""

    def close(self) -> None:
        """Finish a normal or cancelled capture."""

    def abort(self) -> None:
        """Quarantine output that must not be published."""

    def tail_text(self, stream: StreamName) -> str:
        """Return a bounded, display-safe terminal tail."""


@dataclass(frozen=True, slots=True)
class LogSnapshot:
    """Observable counters for one bounded output stream."""

    stream: StreamName
    total_bytes: int
    retained_bytes: int
    discarded_bytes: int
    truncated: bool
    tail: bytes
    content_address: str


@dataclass(slots=True)
class _StreamState:
    max_bytes: int
    tail_bytes: int
    total_bytes: int = 0
    retained_bytes: int = 0
    discarded_bytes: int = 0
    digest: hashlib._Hash = field(default_factory=hashlib.sha256)
    tail: bytearray = field(default_factory=bytearray)


class BoundedLogSink:
    """Stream output to a callback while retaining only bounded terminal tails.

    ``max_stdout_bytes`` and ``max_stderr_bytes`` bound bytes sent to the
    external writer.  The tail ring continues to retain only the latest
    ``terminal_tail_bytes`` after that cap, so a noisy process cannot grow this
    object's memory.  Redactions are applied before either the writer or the
    terminal tail sees a chunk.
    """

    def __init__(
        self,
        *,
        max_stdout_bytes: int = 64 * 1024,
        max_stderr_bytes: int = 64 * 1024,
        terminal_tail_bytes: int = 8 * 1024,
        writer: ChunkWriter | None = None,
        redactions: Iterable[str] = (),
    ) -> None:
        if max_stdout_bytes < 0 or max_stderr_bytes < 0:
            raise ValueError("log byte bounds must be non-negative")
        if terminal_tail_bytes < 0:
            raise ValueError("terminal_tail_bytes must be non-negative")
        self._writer = writer
        self._closed = False
        self._aborted = False
        self._redactions: tuple[bytes, ...] = ()
        self.add_redactions(redactions)
        self._streams: dict[StreamName, _StreamState] = {
            "stdout": _StreamState(
                max_bytes=max_stdout_bytes,
                tail_bytes=terminal_tail_bytes,
            ),
            "stderr": _StreamState(
                max_bytes=max_stderr_bytes,
                tail_bytes=terminal_tail_bytes,
            ),
        }

    @property
    def aborted(self) -> bool:
        """Whether this capture was quarantined after a fence loss."""

        return self._aborted

    @property
    def closed(self) -> bool:
        """Whether no further chunks may be accepted."""

        return self._closed

    def add_redactions(self, values: Iterable[str]) -> None:
        """Add secret values to the byte-level redaction set.

        Empty values are ignored because replacing them would match every
        position.  Longer values are applied first to avoid exposing a suffix
        of a longer credential.
        """

        encoded = [value.encode("utf-8") for value in values if value]
        self._redactions = tuple(
            sorted(set((*self._redactions, *encoded)), key=len, reverse=True)
        )

    def write(self, stream: StreamName, chunk: bytes) -> None:
        """Consume a chunk without allowing memory to grow with output size."""

        if self._closed:
            raise LogSinkClosed("cannot write to a closed log sink")
        if stream not in self._streams:
            raise ValueError(f"unknown output stream: {stream!r}")
        if not isinstance(chunk, bytes):
            raise TypeError("log chunks must be bytes")
        if not chunk or self._aborted:
            return

        state = self._streams[stream]
        redacted = self._redact(chunk)
        state.total_bytes += len(chunk)
        state.digest.update(redacted)

        if state.tail_bytes:
            state.tail.extend(redacted)
            if len(state.tail) > state.tail_bytes:
                del state.tail[: len(state.tail) - state.tail_bytes]

        remaining = max(0, state.max_bytes - state.retained_bytes)
        retained = redacted[:remaining]
        if retained:
            if self._writer is not None:
                self._writer(stream, retained)
            state.retained_bytes += len(retained)
        discarded = len(redacted) - len(retained)
        state.discarded_bytes += discarded

    def close(self) -> None:
        """Close the capture; closing twice is harmless."""

        self._closed = True

    def abort(self) -> None:
        """Stop forwarding output while retaining the bounded diagnostic tail."""

        self._aborted = True

    def tail_bytes(self, stream: StreamName) -> bytes:
        """Return a copy of the bounded redacted terminal tail."""

        return bytes(self._streams[stream].tail)

    def tail_text(self, stream: StreamName) -> str:
        """Return the terminal tail as replacement-safe UTF-8 text."""

        return self.tail_bytes(stream).decode("utf-8", errors="replace")

    def snapshot(self, stream: StreamName) -> LogSnapshot:
        """Return immutable counters and a digest for one stream."""

        state = self._streams[stream]
        return LogSnapshot(
            stream=stream,
            total_bytes=state.total_bytes,
            retained_bytes=state.retained_bytes,
            discarded_bytes=state.discarded_bytes,
            truncated=state.discarded_bytes > 0,
            tail=self.tail_bytes(stream),
            content_address=state.digest.hexdigest(),
        )

    def _redact(self, chunk: bytes) -> bytes:
        redacted = chunk
        for secret in self._redactions:
            redacted = redacted.replace(secret, b"[REDACTED]")
        return redacted


__all__ = [
    "BoundedLogSink",
    "LogSink",
    "LogSinkClosed",
    "LogSnapshot",
    "StreamName",
]
