"""Disk admission hysteresis and bounded garbage-collection requests.

This module never deletes files.  It turns a host observation and a predicted
reservation into a deterministic decision and, when configured, asks an
injected bounded GC planner to produce a plan for a later operator-controlled
stage.  Refusal happens before a caller reaches materialization or process
creation.

Watermarks are *used* MiB: crossing ``high`` blocks new admission; a blocked
host remains blocked until usage falls to ``low``.  Keeping the state per host
prevents the familiar high/low oscillation from admitting every other request
while a disk is hovering around a threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from threading import RLock
from typing import Protocol


class DiskState(StrEnum):
    OPEN = "open"
    BLOCKED = "blocked"


class DiskDecisionCode(StrEnum):
    ADMIT = "admit"
    HIGH_WATERMARK = "disk_high_watermark"
    INSUFFICIENT_FREE = "insufficient_free_disk"
    INVALID_WATERMARKS = "invalid_disk_watermarks"


@dataclass(frozen=True)
class DiskWatermarks:
    """Used-space thresholds, in MiB."""

    low_mib: int | None = None
    high_mib: int | None = None
    # Supplied by the versioned resource profile.  It is not derived from
    # optional request values, so another policy cannot clear this state.
    policy_key: str = "default"

    def __post_init__(self) -> None:
        for name in ("low_mib", "high_mib"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or value < 0):
                raise ValueError(f"{name} must be a non-negative integer or None")
        if (
            self.low_mib is not None
            and self.high_mib is not None
            and self.low_mib > self.high_mib
        ):
            raise ValueError("low disk watermark must not exceed high disk watermark")
        if not self.policy_key.strip():
            raise ValueError("policy_key must be non-blank")


@dataclass(frozen=True)
class DiskObservation:
    """One host disk observation used in an admission explanation."""

    total_mib: int
    free_mib: int

    def __post_init__(self) -> None:
        if self.total_mib < 0 or self.free_mib < 0:
            raise ValueError("disk totals and free space cannot be negative")
        if self.free_mib > self.total_mib:
            raise ValueError("free disk cannot exceed total disk")

    @property
    def used_mib(self) -> int:
        return self.total_mib - self.free_mib


@dataclass(frozen=True)
class DiskDecision:
    code: DiskDecisionCode
    state: DiskState
    host_id: str
    observed: DiskObservation
    predicted_used_mib: int
    predicted_free_mib: int
    reason: str
    gc_requested_mib: int = 0

    @property
    def admitted(self) -> bool:
        return self.code == DiskDecisionCode.ADMIT

    def as_dict(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "state": self.state.value,
            "host_id": self.host_id,
            "total_mib": self.observed.total_mib,
            "free_mib": self.observed.free_mib,
            "predicted_used_mib": self.predicted_used_mib,
            "predicted_free_mib": self.predicted_free_mib,
            "reason": self.reason,
            "gc_requested_mib": self.gc_requested_mib,
        }


@dataclass(frozen=True)
class GCRequest:
    """A bounded request for a later, separately authorized GC stage."""

    host_id: str
    requested_mib: int
    reason: str
    reservation_id: str = ""


class GCRequestPort(Protocol):
    """Port for planning GC; implementations must not delete here."""

    def request_gc(self, request: GCRequest) -> None:
        """Record or enqueue a bounded GC request."""


class DiskPolicy:
    """Native-policy-keyed disk state machine with high/low hysteresis.

    The in-memory map is a simulation/local projection.  A graph-backed port
    must persist the same ``(host_id, policy_key)`` state with host policy
    revision so a request with absent or changed watermarks cannot reopen a
    blocked native host.
    """

    def __init__(self, *, gc_requester: GCRequestPort | None = None) -> None:
        self._lock = RLock()
        self._states: dict[tuple[str, str], DiskState] = {}
        self._watermarks: dict[tuple[str, str], DiskWatermarks] = {}
        self._gc_requester = gc_requester

    def state(self, host_id: str, *, policy_key: str = "default") -> DiskState:
        with self._lock:
            return self._states.get((host_id, policy_key), DiskState.OPEN)

    def reset(self, host_id: str, *, policy_key: str | None = None) -> None:
        with self._lock:
            keys = (
                [key for key in self._states if key[0] == host_id]
                if policy_key is None
                else [(host_id, policy_key)]
            )
            for key in keys:
                self._states.pop(key, None)
                self._watermarks.pop(key, None)

    def evaluate(
        self,
        host_id: str,
        *,
        total_mib: int,
        free_mib: int,
        requested_mib: int = 0,
        watermarks: DiskWatermarks | None = None,
        reservation_id: str = "",
        request_gc: bool = True,
        mutate: bool = True,
        policy_key: str = "default",
    ) -> DiskDecision:
        """Evaluate an admission without mutating files or starting work."""

        watermarks = watermarks or DiskWatermarks(policy_key=policy_key)
        policy_key = watermarks.policy_key
        key = (host_id, policy_key)
        observation = DiskObservation(total_mib, free_mib)
        if requested_mib < 0:
            raise ValueError("requested_mib cannot be negative")
        if watermarks.high_mib is not None and watermarks.low_mib is None:
            low: int | None = watermarks.high_mib
        else:
            low = watermarks.low_mib
        high = watermarks.high_mib
        if low != watermarks.low_mib:
            watermarks = DiskWatermarks(
                low_mib=low, high_mib=high, policy_key=policy_key
            )
        if low is not None and high is not None and low > high:
            return DiskDecision(
                DiskDecisionCode.INVALID_WATERMARKS,
                self.state(host_id, policy_key=policy_key),
                host_id,
                observation,
                observation.used_mib + requested_mib,
                observation.free_mib - requested_mib,
                "low disk watermark exceeds high watermark",
            )

        predicted_used = observation.used_mib + requested_mib
        predicted_free = observation.free_mib - requested_mib
        if predicted_free < 0:
            if mutate:
                with self._lock:
                    self._states[key] = DiskState.BLOCKED
                    self._watermarks.setdefault(key, watermarks)
            decision = DiskDecision(
                DiskDecisionCode.INSUFFICIENT_FREE,
                DiskState.BLOCKED,
                host_id,
                observation,
                predicted_used,
                predicted_free,
                "predicted reservation exceeds observed free disk",
                max(0, requested_mib - observation.free_mib),
            )
            if mutate:
                self._maybe_request_gc(decision, reservation_id, request_gc)
            return decision

        with self._lock:
            # A policy key owns its first accepted thresholds.  A missing or
            # divergent request cannot silently clear native hysteresis; a
            # production implementation persists this alongside host policy.
            authoritative = self._watermarks.get(key)
            if authoritative is None:
                authoritative = watermarks
                if mutate:
                    self._watermarks[key] = authoritative
            else:
                watermarks = authoritative
            low = watermarks.low_mib
            high = watermarks.high_mib
            current = self._states.get(key, DiskState.OPEN)
            if current == DiskState.BLOCKED:
                # No high-watermark policy means a hard free-space refusal is
                # the only reason to remain blocked; otherwise reopen.
                if low is not None and predicted_used > low:
                    state = DiskState.BLOCKED
                elif low is None:
                    state = DiskState.OPEN
                else:
                    state = DiskState.OPEN
            else:
                state = (
                    DiskState.BLOCKED
                    if high is not None and predicted_used >= high
                    else DiskState.OPEN
                )
            if mutate:
                self._states[key] = state

        if state == DiskState.BLOCKED:
            decision = DiskDecision(
                DiskDecisionCode.HIGH_WATERMARK,
                state,
                host_id,
                observation,
                predicted_used,
                predicted_free,
                (
                    f"predicted used disk {predicted_used} MiB reaches high watermark "
                    f"{high} MiB; admission deferred until usage is <= {low} MiB"
                ),
                max(0, predicted_used - (low or predicted_used)),
            )
            if mutate:
                self._maybe_request_gc(decision, reservation_id, request_gc)
            return decision

        return DiskDecision(
            DiskDecisionCode.ADMIT,
            state,
            host_id,
            observation,
            predicted_used,
            predicted_free,
            "disk watermarks and predicted reservation permit admission",
        )

    def _maybe_request_gc(
        self, decision: DiskDecision, reservation_id: str, enabled: bool
    ) -> None:
        if not enabled or self._gc_requester is None or decision.gc_requested_mib <= 0:
            return
        self._gc_requester.request_gc(
            GCRequest(
                host_id=decision.host_id,
                requested_mib=decision.gc_requested_mib,
                reason=decision.reason,
                reservation_id=reservation_id,
            )
        )


__all__ = [
    "DiskDecision",
    "DiskDecisionCode",
    "DiskObservation",
    "DiskPolicy",
    "DiskState",
    "DiskWatermarks",
    "GCRequest",
    "GCRequestPort",
]
