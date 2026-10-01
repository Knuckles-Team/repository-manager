"""Test-only fakes for the three materialization ports.

**This is test scaffolding, never production code.** Nothing under
``repository_manager/`` may import from here; see
:mod:`repository_manager.materialization.ports` for the protocols these
fakes implement and the one real adapter (:class:`GitMergeQueueAdapter`)
that is NOT a fake.

Blocker this records (for the open claim, not a guess to fill in): there is
no real, non-fake :class:`~repository_manager.materialization.ports.ApprovalAuthority`
or :class:`~repository_manager.materialization.ports.MaterializationReceiptStore`
implementation anywhere in this repository or in its two dependencies
(``agent-utilities``, ``agent-connector-sdk``) today -- a repo-wide search for
``ApprovalAuthority``, ``ApprovalProof``, ``MaterializationReceipt``, and
``receipt store`` across all three finds nothing beyond this package itself.
RM-MATERIALIZE-001's own ``spec.md`` ("Boundaries") says why: "The external
approval authority owns approval facts; repository-manager verifies a bounded
proof... A graph index may store the receipt but cannot manufacture a Git
commit... Related public specs in OTHER repositories may describe approval
storage and orchestration." Both ports are therefore meant to be satisfied by
a service that lives outside this repository's dependency closure -- most
plausibly the graph/KG-backed governance layer referenced throughout this
workspace (an approval-proof verification call and a receipt read/write call
exposed as an MCP tool or RPC) -- and nothing in the current
``agent-utilities``/``agent-connector-sdk`` surface names that call today. The
closest in-repo analog is :mod:`repository_manager.concept_coordination`'s
claim/reservation authority (fenced, idempotency-keyed, create-if-absent),
but it governs concept-ID claims, not patch approval or receipt delivery, and
is not a drop-in. Writing a real adapter against a guessed method name would
be worse than no adapter; this stays a recorded blocker until the real
service and its exact verify/get/put calls are named.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from repository_manager import merge_queue
from repository_manager.materialization.models import (
    ApprovalProof,
    ChangeProposal,
    MaterializationReceipt,
)
from repository_manager.materialization.ports import (
    ApprovalAuthorityUnavailable,
    ApprovalVerdict,
    MergeQueueUnavailable,
    QueueSubmission,
    ReceiptStoreUnavailable,
)

__all__ = ["FakeApprovalAuthority", "FakeMergeQueuePort", "InMemoryReceiptStore"]


@dataclass
class FakeApprovalAuthority:
    """Deterministic authority for tests: an allowlist of accepted signatures.

    ``accepted_signatures=None`` accepts any structurally-complete signature
    (useful when a test is only exercising the service's own binding checks).
    ``raise_unavailable=True`` makes every call raise
    :class:`~repository_manager.materialization.ports.ApprovalAuthorityUnavailable`,
    for RM-MATERIALIZE-06 fault injection.
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
