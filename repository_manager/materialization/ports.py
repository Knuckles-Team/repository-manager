"""Pluggable authority/store/queue ports for materialization (RM-MATERIALIZE-001).

:mod:`repository_manager.materialization.service` depends on these three
abstract boundaries rather than on a live approval service, a live receipt
store, or the real merge queue directly, so that callers can supply their own
implementation without a second Git-mutating path, and so RM-MATERIALIZE-06
(no direct Git fallback when a dependency is unavailable) is a fault a caller
can inject by raising the matching ``*Unavailable`` error from their own
implementation.

Only :class:`GitMergeQueueAdapter` below is a real implementation -- it wires
:class:`MergeQueuePort` to this package's existing merge queue
(:mod:`repository_manager.merge_queue`). There is today no real, non-test
:class:`ApprovalAuthority` or :class:`MaterializationReceiptStore` in this
repository or its dependencies (see
``tests/materialization/fakes.py``'s module docstring for exactly what a real
one would need to call). Test doubles for all three ports live under
``tests/materialization/fakes.py``, never here: this module holds the
contracts and the one real adapter only.
"""

from __future__ import annotations

from dataclasses import dataclass
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
    "GitMergeQueueAdapter",
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


class MaterializationReceiptStore(Protocol):
    """Durable, idempotency-keyed storage for :class:`MaterializationReceipt`."""

    def get(self, idempotency_key: str) -> MaterializationReceipt | None: ...

    def put(self, idempotency_key: str, receipt: MaterializationReceipt) -> None: ...


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
