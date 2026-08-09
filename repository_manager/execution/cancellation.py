"""Cooperative cancellation primitives for one executor attempt."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from datetime import UTC, datetime


@dataclass(frozen=True, slots=True)
class CancellationSnapshot:
    """A consistent view of a cancellation request."""

    cancelled: bool
    reason: str | None
    requested_at: datetime | None


class CancellationToken:
    """Thread-safe, idempotent cancellation source.

    The executor owns process termination; this object only carries the
    request across the caller/supervisor boundary.  Repeated ``cancel`` calls
    do not replace the first reason, which makes recovery and auditing stable.
    """

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._requested_at: datetime | None = None

    def cancel(self, reason: str = "cancelled") -> bool:
        """Request cancellation and return ``True`` only for the first request."""

        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("cancellation reason must be a non-blank string")
        with self._lock:
            if self._event.is_set():
                return False
            self._reason = reason.strip()
            self._requested_at = datetime.now(UTC)
            self._event.set()
            return True

    def is_cancelled(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for cancellation, returning the event state."""

        return self._event.wait(timeout)

    def snapshot(self) -> CancellationSnapshot:
        """Return cancellation state and its first-request metadata."""

        with self._lock:
            return CancellationSnapshot(
                cancelled=self._event.is_set(),
                reason=self._reason,
                requested_at=self._requested_at,
            )


__all__ = ["CancellationSnapshot", "CancellationToken"]
