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

_REPLACEMENT_CANDIDATES = (
    b"[REDACTED]",
    b"<redacted>",
    b"[REMOVED]",
    b"<removed>",
)


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


class StreamingRedactor:
    """Replace secret byte patterns without leaking across chunk boundaries.

    A normal chunk boundary is not a security boundary: a reader may return
    ``b"sec"`` and ``b"ret"`` for one secret.  This scanner retains only the
    longest suffix that could still be a prefix of a secret and emits the safe
    prefix immediately.  ``flush`` is the only operation that releases that
    overlap; ``abort`` drops it, which is required when a stale fence
    quarantines a result.
    """

    def __init__(self, patterns: Iterable[bytes] = ()) -> None:
        self._patterns: tuple[bytes, ...] = ()
        self._pending = bytearray()
        self._closed = False
        self._aborted = False
        self._replacement = b""
        self.add_patterns(patterns)

    @property
    def pending_bytes(self) -> int:
        """Return the bounded overlap currently held for a future chunk."""

        return len(self._pending)

    @property
    def max_pattern_length(self) -> int:
        """Return the longest configured secret length."""

        return len(self._patterns[0]) if self._patterns else 0

    def add_patterns(self, patterns: Iterable[bytes]) -> None:
        """Add non-empty byte patterns before or during a capture."""

        values: list[bytes] = []
        for pattern in patterns:
            if not isinstance(pattern, bytes):
                raise TypeError("redaction patterns must be bytes")
            if pattern:
                values.append(pattern)
        self._patterns = tuple(
            sorted(set((*self._patterns, *values)), key=len, reverse=True)
        )
        self._replacement = self._choose_replacement()

    def feed(self, chunk: bytes) -> bytes:
        """Consume one chunk and return bytes proven safe to emit now."""

        if self._closed:
            raise RuntimeError("cannot feed a closed redactor")
        if self._aborted:
            return b""
        if not isinstance(chunk, bytes):
            raise TypeError("redactor chunks must be bytes")
        if not self._patterns:
            return chunk

        emitted = bytearray()
        for byte in chunk:
            self._pending.append(byte)
            self._drain(emitted, final=False)
        return bytes(emitted)

    def flush(self) -> bytes:
        """Flush safe pending bytes after normal stream completion."""

        if self._closed:
            return b""
        self._closed = True
        if self._aborted:
            self._pending.clear()
            return b""
        emitted = bytearray()
        self._drain(emitted, final=True)
        return bytes(emitted)

    def abort(self) -> None:
        """Quarantine and discard pending overlap without releasing it."""

        self._aborted = True
        self._pending.clear()

    def _drain(self, emitted: bytearray, *, final: bool) -> None:
        while self._pending:
            match = self._find_match()
            if match is not None:
                start, end = match
                emitted.extend(self._pending[:start])
                emitted.extend(self._replacement)
                del self._pending[:end]
                continue
            if final:
                emitted.extend(self._pending)
                self._pending.clear()
                return
            suffix_length = self._longest_prefix_suffix()
            safe_length = len(self._pending) - suffix_length
            if safe_length:
                emitted.extend(self._pending[:safe_length])
                del self._pending[:safe_length]
            return

    def _find_match(self) -> tuple[int, int] | None:
        data = bytes(self._pending)
        for start in range(len(data)):
            for pattern in self._patterns:
                if data.startswith(pattern, start):
                    return start, start + len(pattern)
        return None

    def _longest_prefix_suffix(self) -> int:
        data = bytes(self._pending)
        for length in range(min(len(data), self.max_pattern_length - 1), 0, -1):
            suffix = data[-length:]
            if any(pattern.startswith(suffix) for pattern in self._patterns):
                return length
        return 0

    def _choose_replacement(self) -> bytes:
        """Choose a marker that cannot itself contain/cross a secret."""

        for candidate in _REPLACEMENT_CANDIDATES:
            if self._marker_is_safe(candidate):
                return candidate
        for value in range(256):
            candidate = bytes((value,))
            if self._marker_is_safe(candidate):
                return candidate
        # This degenerate case means every byte is itself a secret.  No
        # non-empty marker can be safe; dropping the match is fail-closed.
        return b""

    def _marker_is_safe(self, marker: bytes) -> bool:
        for pattern in self._patterns:
            if pattern in marker:
                return False
            # A marker embedded in a secret could be completed by arbitrary
            # bytes before and after a replacement, recreating the secret in
            # the redacted stream.  Reject that case as well.
            if marker in pattern:
                return False
            for length in range(1, min(len(pattern), len(marker))):
                if marker[-length:] == pattern[:length]:
                    return False
                if marker[:length] == pattern[-length:]:
                    return False
        return True


class RedactingLogSink:
    """Apply boundary-safe redaction before forwarding to another log sink."""

    def __init__(self, sink: LogSink, redactions: Iterable[str]) -> None:
        self._sink = sink
        patterns = tuple(value.encode("utf-8") for value in redactions if value)
        self._redactors = {
            "stdout": StreamingRedactor(patterns),
            "stderr": StreamingRedactor(patterns),
        }

    def write(self, stream: StreamName, chunk: bytes) -> None:
        """Forward only bytes proven not to contain a configured secret."""

        emitted = self._redactors[stream].feed(chunk)
        if emitted:
            self._sink.write(stream, emitted)

    def close(self) -> None:
        """Flush overlap safely, then close the wrapped sink."""

        for stream in ("stdout", "stderr"):
            redactor = self._redactors[stream]
            emitted = redactor.flush()
            if emitted:
                self._sink.write(stream, emitted)
        self._sink.close()

    def abort(self) -> None:
        """Drop overlap and quarantine the wrapped sink."""

        for redactor in self._redactors.values():
            redactor.abort()
        self._sink.abort()

    def tail_text(self, stream: StreamName) -> str:
        """Return the wrapped sink's bounded terminal tail."""

        return self._sink.tail_text(stream)


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
    redactor: StreamingRedactor = field(default_factory=StreamingRedactor)


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
                redactor=StreamingRedactor(self._redactions),
            ),
            "stderr": _StreamState(
                max_bytes=max_stderr_bytes,
                tail_bytes=terminal_tail_bytes,
                redactor=StreamingRedactor(self._redactions),
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
        for state in getattr(self, "_streams", {}).values():
            state.redactor.add_patterns(self._redactions)

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
        state.total_bytes += len(chunk)
        self._accept(stream, state, state.redactor.feed(chunk))

    def _accept(self, stream: StreamName, state: _StreamState, redacted: bytes) -> None:
        if not redacted:
            return
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

        if self._closed:
            return
        if not self._aborted:
            for stream, state in self._streams.items():
                self._accept(stream, state, state.redactor.flush())
        self._closed = True

    def abort(self) -> None:
        """Stop forwarding output while retaining the bounded diagnostic tail."""

        self._aborted = True
        for state in self._streams.values():
            state.redactor.abort()

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


__all__ = [
    "BoundedLogSink",
    "LogSink",
    "LogSinkClosed",
    "LogSnapshot",
    "RedactingLogSink",
    "StreamName",
    "StreamingRedactor",
]
