"""Pluggable authority/store/queue ports for materialization (RM-MATERIALIZE-001).

:mod:`repository_manager.materialization.service` depends on these three
abstract boundaries rather than on a live approval service, a live receipt
store, or the real merge queue directly, so that:

* ``repository_manager``'s own test suite can qualify every acceptance rule
  (RM-MATERIALIZE-01 through 06) against disposable Git fixtures and the
  fakes below, with no live service required (plan.md's "Implementations
  must not require a particular operator environment"); and
* RM-MATERIALIZE-06 (no direct Git fallback when a dependency is
  unavailable) is a fault *you can inject*: raise the matching
  ``*Unavailable`` error from a fake and assert the service never reaches a
  Git mutation.

The real adapters at the bottom wire the ports to this package's existing
merge queue (:mod:`repository_manager.merge_queue`); a live approval
authority and a durable receipt store are intentionally left to whatever
composes this service in production (a graph client, a KV store, ...) --
inventing one here would be the "second... registry or policy path" the
build-lane instructions warn against.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Protocol

from repository_manager import merge_queue
from repository_manager.materialization.models import (
    ApprovalProof,
    ChangeProposal,
    MaterializationReceipt,
)

__all__ = [
    "ApprovalAuthority",
    "ApprovalAuthorityUnavailable",
    "ApprovalVerdict",
    "FakeApprovalAuthority",
    "GitMergeQueueAdapter",
    "InMemoryReceiptStore",
    "MaterializationReceiptStore",
    "MergeQueuePort",
    "MergeQueueUnavailable",
    "QueueSubmission",
    "ReceiptStoreUnavailable",
]


class ApprovalAuthorityUnavailable(RuntimeError):
    """The external approval authority could not be reached."""


class ReceiptStoreUnavailable(RuntimeError):
    """The receipt store could not be read or written."""


class MergeQueueUnavailable(RuntimeError):
    """The merge queue could not accept a submission right now."""


@dataclass(frozen=True, slots=True)
class ApprovalVerdict:
    """The authority's own authenticity/caller-authorization verdict."""

    valid: bool
    reasons: tuple[str, ...] = ()


class ApprovalAuthority(Protocol):
    """Verifies ``approval.signature`` and the caller's right to submit it.

    Structural binding (proposal digest, repository/tenant/actor match,
    expiry) is checked by the service itself before this is ever called; this
    port answers only "is this signature genuine, from an authorized caller".
    """

    def verify(
        self, proposal: ChangeProposal, approval: ApprovalProof, *, now: datetime
    ) -> ApprovalVerdict: ...


@dataclass
class FakeApprovalAuthority:
    """Deterministic authority for tests: an allowlist of accepted signatures.

    ``accepted_signatures=None`` accepts any structurally-complete signature
    (useful when a test is only exercising the service's own binding checks).
    ``raise_unavailable=True`` makes every call raise
    :class:`ApprovalAuthorityUnavailable`, for RM-MATERIALIZE-06 fault
    injection.
    """

    accepted_signatures: frozenset[str] | None = None
    raise_unavailable: bool = False

    def verify(
        self, proposal: ChangeProposal, approval: ApprovalProof, *, now: datetime
    ) -> ApprovalVerdict:
        del proposal, now
        if self.raise_unavailable:
            raise ApprovalAuthorityUnavailable("fake approval authority is unavailable")
        if (
            self.accepted_signatures is None
            or approval.signature in self.accepted_signatures
        ):
            return ApprovalVerdict(valid=True)
        return ApprovalVerdict(valid=False, reasons=("signature not recognized",))


class MaterializationReceiptStore(Protocol):
    """Durable, idempotency-keyed storage for :class:`MaterializationReceipt`."""

    def get(self, idempotency_key: str) -> MaterializationReceipt | None: ...

    def put(self, idempotency_key: str, receipt: MaterializationReceipt) -> None: ...


@dataclass
class InMemoryReceiptStore:
    """Process-local receipt store: the real shape for tests, never for prod."""

    raise_unavailable: bool = False
    _rows: dict[str, MaterializationReceipt] = field(default_factory=dict)

    def get(self, idempotency_key: str) -> MaterializationReceipt | None:
        if self.raise_unavailable:
            raise ReceiptStoreUnavailable("fake receipt store is unavailable")
        return self._rows.get(idempotency_key)

    def put(self, idempotency_key: str, receipt: MaterializationReceipt) -> None:
        if self.raise_unavailable:
            raise ReceiptStoreUnavailable("fake receipt store is unavailable")
        self._rows[idempotency_key] = receipt


@dataclass(frozen=True, slots=True)
class QueueSubmission:
    """What the queue port reports back for one submission attempt."""

    accepted: bool
    detail: str = ""
    queue_depth: int | None = None


class MergeQueuePort(Protocol):
    """Submits a committed candidate; never reports it as merged."""

    def submit(
        self, *, worktree_path: str, branch: str, base: str
    ) -> QueueSubmission: ...

    def status(self, *, worktree_path: str, branch: str) -> str: ...


@dataclass
class FakeMergeQueuePort:
    """Deterministic queue port for tests: accept/reject/raise on demand."""

    accept: bool = True
    detail: str = ""
    raise_unavailable: bool = False
    reconciled_status: str = merge_queue.QUEUED

    def submit(self, *, worktree_path: str, branch: str, base: str) -> QueueSubmission:
        del worktree_path, branch, base
        if self.raise_unavailable:
            raise MergeQueueUnavailable("fake merge queue is unavailable")
        return QueueSubmission(accepted=self.accept, detail=self.detail)

    def status(self, *, worktree_path: str, branch: str) -> str:
        del worktree_path, branch
        if self.raise_unavailable:
            raise MergeQueueUnavailable("fake merge queue is unavailable")
        return self.reconciled_status


@dataclass
class GitMergeQueueAdapter:
    """Wires :class:`MergeQueuePort` to this package's real merge queue.

    Submission is enqueue-only (CONCEPT:RM-MERGE-QUEUE's own contract): the
    candidate is appended to the repo's queue store and drained later by
    ``merge-queue-runner.timer``.  This adapter never claims the candidate
    landed -- only that the queue accepted the submission.
    """

    def submit(self, *, worktree_path: str, branch: str, base: str) -> QueueSubmission:
        try:
            result = merge_queue.enqueue(
                branch,
                base=base,
                worktree=Path(worktree_path),
                path=Path(worktree_path),
            )
        except merge_queue.MergeQueueError as exc:
            raise MergeQueueUnavailable(str(exc)) from exc
        return QueueSubmission(
            accepted=bool(result.get("enqueued")),
            detail=str(result.get("note", "")),
            queue_depth=result.get("queue_depth"),
        )

    def status(self, *, worktree_path: str, branch: str) -> str:
        """Reconcile against the queue's own report (D-MQR-7: queued != landed).

        ``merge-queue-runner.timer`` drains the queue asynchronously; this
        reads whatever it has recorded so far rather than draining itself
        (a manual drain races that scheduler for no benefit -- see
        :func:`merge_queue.enqueue`'s own note).
        """

        try:
            report = merge_queue.queue_report(path=Path(worktree_path))
        except merge_queue.MergeQueueError as exc:
            raise MergeQueueUnavailable(str(exc)) from exc
        for record in report.get("queued", []):
            if record.get("id") == branch:
                return merge_queue.QUEUED
        for record in report.get("recent", []):
            if record.get("id") == branch:
                return str(record.get("state", merge_queue.QUEUED))
        return merge_queue.QUEUED
