"""The governed proposal-to-Git materializer (RM-MATERIALIZE-001).

:class:`MaterializationService` is the ONLY path from an approved
:class:`~repository_manager.materialization.models.ChangeProposal` to a Git
commit in this package: it composes the existing, already-gated primitives
(:mod:`repository_manager.worktree`, :mod:`repository_manager.safe_commit`,
:mod:`repository_manager.merge_queue`) behind one admission boundary rather
than adding a second Git-mutating surface. No other function in this package
creates a worktree, applies a patch, or commits on a proposal's behalf
(RM-MATERIALIZE-06).

Sequence (plan.md): verify request/approval -> check base and allowlist ->
create worktree -> apply patch -> validate exact tree -> explicit
stage/commit -> queue -> (later, via :meth:`MaterializationService.reconcile`)
reconcile the hosted result. Every step before a worktree exists can only
refuse; every step after an isolated worktree exists can refuse AND clean
that worktree up, but the canonical checkout's own tree is read, never
written.
"""

from __future__ import annotations

import logging
import subprocess  # nosec B404 - fixed-argv git invocations only
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from repository_manager import merge_queue
from repository_manager.materialization.models import (
    ApprovalProof,
    ChangeProposal,
    MaterializationErrorCode,
    MaterializationReceipt,
    MaterializationState,
)
from repository_manager.materialization.patch import PatchRejected, apply_patch
from repository_manager.materialization.ports import (
    ApprovalAuthority,
    ApprovalAuthorityUnavailable,
    MaterializationReceiptStore,
    MergeQueuePort,
    MergeQueueUnavailable,
    ReceiptStoreUnavailable,
)
from repository_manager.safe_commit import safe_commit
from repository_manager.worktree import WorktreeManager

logger = logging.getLogger(__name__)

__all__ = [
    "MaterializationService",
    "RepositoryManagerUnavailable",
    "check_binding",
]


class RepositoryManagerUnavailable(RuntimeError):
    """Worktree creation itself failed (RM-MATERIALIZE-06's "RM adapter")."""


#: Any of these means a DEPENDENCY, not the proposal's content, is the
#: problem; :meth:`MaterializationService.materialize` turns every one of
#: them into a ``PENDING``/``UNCERTAIN_DELIVERY`` receipt, never a refusal.
_UNAVAILABLE_ERRORS = (
    ApprovalAuthorityUnavailable,
    ReceiptStoreUnavailable,
    MergeQueueUnavailable,
    RepositoryManagerUnavailable,
)


def check_binding(
    proposal: ChangeProposal, approval: ApprovalProof, *, now: datetime
) -> tuple[str, ...]:
    """Pure structural checks: no I/O, no signature authenticity.

    Catches an approval bound to a DIFFERENT proposal (any field changed
    changes ``proposal.digest``), a wrong repository/tenant/actor, or an
    expired proposal/approval -- every RM-MATERIALIZE-01 negative fixture
    except forged-signature detection, which only
    :class:`~repository_manager.materialization.ports.ApprovalAuthority` can
    answer.
    """

    reasons: list[str] = []
    if now >= approval.expires_at:
        reasons.append("approval has expired")
    if now >= proposal.expires_at:
        reasons.append("proposal has expired")
    if approval.proposal_digest != proposal.digest:
        reasons.append("approval is not bound to this exact proposal")
    if approval.repository_id != proposal.repository_id:
        reasons.append("approval repository does not match the proposal")
    if approval.tenant_id != proposal.tenant_id:
        reasons.append("approval tenant does not match the proposal")
    if approval.actor_id != proposal.decision_identity:
        reasons.append("approval actor does not match the proposal decision identity")
    return tuple(reasons)


class _MaterializeAbort(Exception):
    """One phase of :meth:`MaterializationService.materialize` refused."""

    def __init__(self, code: MaterializationErrorCode, detail: str) -> None:
        super().__init__(detail)
        self.code = code
        self.detail = detail


@dataclass
class _Attempt:
    """Fields threaded through one ``materialize`` call's phases."""

    proposal: ChangeProposal
    approval: ApprovalProof
    idempotency_key: str
    now: datetime
    worktree_path: Path | None = None
    branch: str = ""
    commit_sha: str | None = None
    tree_sha: str | None = None
    queue_accepted: bool | None = None
    queue_detail: str = ""


def _current_ref_sha(canonical: Path, ref: str) -> str | None:
    result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, git from PATH
        ["git", "rev-parse", "--verify", "--quiet", ref],
        cwd=str(canonical),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _tree_sha(worktree_path: Path) -> str | None:
    result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, git from PATH
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=str(worktree_path),
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


class MaterializationService:
    """Composes worktree, patch, commit, and queue behind one admission gate."""

    def __init__(
        self,
        *,
        worktrees: WorktreeManager,
        approval_authority: ApprovalAuthority,
        receipt_store: MaterializationReceiptStore,
        queue: MergeQueuePort,
        gate: Sequence[str] | Callable[[Path], Any] | None = None,
    ) -> None:
        self._worktrees = worktrees
        self._approval_authority = approval_authority
        self._receipt_store = receipt_store
        self._queue = queue
        self._gate = gate

    def materialize(
        self,
        proposal: ChangeProposal,
        approval: ApprovalProof,
        *,
        idempotency_key: str | None = None,
        now: datetime | None = None,
    ) -> MaterializationReceipt:
        """Admit, apply, validate, commit, and submit one approved proposal."""

        attempt = _Attempt(
            proposal=proposal,
            approval=approval,
            idempotency_key=idempotency_key or proposal.proposal_id,
            now=now or datetime.now(UTC),
        )
        try:
            replay = self._check_idempotent_replay(attempt)
            if replay is not None:
                return replay
            self._admit(attempt)
            self._create_worktree(attempt)
            self._apply_and_commit(attempt)
            self._submit(attempt)
        except _UNAVAILABLE_ERRORS as exc:
            return self._pending_receipt(attempt, str(exc))
        except _MaterializeAbort as exc:
            return self._finish(attempt, exc)
        return self._finish(attempt, None)

    def reconcile(self, idempotency_key: str) -> MaterializationReceipt:
        """Ask the queue what became of an already-submitted candidate.

        Never performs a Git mutation; only re-reads the queue's own report
        (``merge-queue-runner.timer`` drains it asynchronously) and persists
        an updated receipt. A receipt not currently ``QUEUED`` is returned
        unchanged -- there is nothing left to reconcile.
        """

        try:
            receipt = self._receipt_store.get(idempotency_key)
        except _UNAVAILABLE_ERRORS as exc:
            raise ReceiptStoreUnavailable(str(exc)) from exc
        if receipt is None:
            raise KeyError(f"no receipt recorded for {idempotency_key!r}")
        if receipt.state is not MaterializationState.QUEUED:
            return receipt
        return self._reconcile_queued(receipt)

    # ── admission phases ────────────────────────────────────────────────
    def _check_idempotent_replay(
        self, attempt: _Attempt
    ) -> MaterializationReceipt | None:
        existing = self._receipt_store.get(attempt.idempotency_key)
        if existing is None:
            return None
        if existing.proposal_digest == attempt.proposal.digest:
            return existing
        raise _MaterializeAbort(
            MaterializationErrorCode.UNAUTHORIZED,
            f"idempotency key {attempt.idempotency_key!r} is already bound to a "
            "different proposal",
        )

    def _admit(self, attempt: _Attempt) -> None:
        proposal, approval = attempt.proposal, attempt.approval
        reasons = check_binding(proposal, approval, now=attempt.now)
        if reasons:
            code = (
                MaterializationErrorCode.EXPIRED_APPROVAL
                if any("expired" in reason for reason in reasons)
                else MaterializationErrorCode.UNAUTHORIZED
            )
            raise _MaterializeAbort(code, "; ".join(reasons))
        verdict = self._approval_authority.verify(proposal, approval, now=attempt.now)
        if not verdict.valid:
            raise _MaterializeAbort(
                MaterializationErrorCode.UNAUTHORIZED,
                "; ".join(verdict.reasons) or "approval authority rejected the request",
            )

    def _create_worktree(self, attempt: _Attempt) -> None:
        proposal = attempt.proposal
        canonical = self._worktrees.resolve_repo(proposal.repository_id)
        current = (
            _current_ref_sha(Path(canonical), proposal.base_ref) if canonical else None
        )
        if canonical is None or current != proposal.base_sha:
            detail = (
                f"repository {proposal.repository_id!r} is not registered"
                if canonical is None
                else (
                    f"{proposal.base_ref!r} is now {current or '<unresolvable>'}, not "
                    f"the approved base {proposal.base_sha!r}"
                )
            )
            raise _MaterializeAbort(MaterializationErrorCode.STALE_BASE, detail)
        branch = f"materialize/{proposal.proposal_id}"
        added = self._worktrees.add(
            proposal.repository_id, branch, base=proposal.base_sha
        )
        if not added.get("ok"):
            raise RepositoryManagerUnavailable(
                f"could not create an isolated worktree: {added.get('error', added)}"
            )
        attempt.worktree_path = Path(added["path"])
        attempt.branch = branch

    def _apply_and_commit(self, attempt: _Attempt) -> None:
        proposal = attempt.proposal
        assert attempt.worktree_path is not None
        try:
            apply_patch(
                attempt.worktree_path, proposal.patch_text, proposal.approved_paths
            )
        except PatchRejected as exc:
            self._abandon_worktree(attempt)
            raise _MaterializeAbort(
                MaterializationErrorCode.PATH_REJECTED, str(exc)
            ) from exc
        message = f"materialize {proposal.proposal_id} (approved by {proposal.decision_identity})"
        result = safe_commit(
            attempt.worktree_path,
            message,
            paths=list(proposal.approved_paths),
            gate=self._gate,
        )
        if not result.get("ok"):
            self._abandon_worktree(attempt)
            raise _MaterializeAbort(
                MaterializationErrorCode.GATE_FAILED,
                str(result.get("error") or "declared validation gate failed"),
            )
        attempt.commit_sha = result.get("commit_sha")
        attempt.tree_sha = _tree_sha(attempt.worktree_path)

    def _submit(self, attempt: _Attempt) -> None:
        proposal = attempt.proposal
        assert attempt.worktree_path is not None
        submission = self._queue.submit(
            worktree_path=str(attempt.worktree_path),
            branch=attempt.branch,
            base=proposal.base_ref,
        )
        attempt.queue_accepted = submission.accepted
        attempt.queue_detail = submission.detail
        if not submission.accepted:
            raise _MaterializeAbort(
                MaterializationErrorCode.QUEUE_CONFLICT,
                submission.detail or "the merge queue refused the candidate",
            )

    def _abandon_worktree(self, attempt: _Attempt) -> None:
        if attempt.worktree_path is None:
            return
        proposal = attempt.proposal
        try:
            self._worktrees.remove(
                proposal.repository_id,
                attempt.branch,
                force=True,
                delete_branch=True,
                base=proposal.base_ref,
            )
        except Exception:  # pragma: no cover - best-effort cleanup only
            logger.warning(
                "could not remove abandoned materialization worktree %s",
                attempt.worktree_path,
                exc_info=True,
            )

    # ── receipt assembly ─────────────────────────────────────────────────
    def _build_receipt(
        self,
        attempt: _Attempt,
        *,
        state: MaterializationState,
        error_code: MaterializationErrorCode | None,
        error_detail: str,
    ) -> MaterializationReceipt:
        receipt = MaterializationReceipt(
            idempotency_key=attempt.idempotency_key,
            proposal_id=attempt.proposal.proposal_id,
            proposal_digest=attempt.proposal.digest,
            approval_id=attempt.approval.approval_id,
            approval_digest=attempt.approval.digest,
            repository_id=attempt.proposal.repository_id,
            tenant_id=attempt.proposal.tenant_id,
            base_sha=attempt.proposal.base_sha,
            actor_id=attempt.proposal.decision_identity,
            gate_profile=attempt.proposal.validation_profile,
            state=state,
            created_at=attempt.now,
            updated_at=datetime.now(UTC),
            commit_sha=attempt.commit_sha,
            tree_sha=attempt.tree_sha,
            queue_branch=attempt.branch or None,
            queue_accepted=attempt.queue_accepted,
            queue_detail=attempt.queue_detail,
            error_code=error_code,
            error_detail=error_detail,
        )
        self._persist(attempt.idempotency_key, receipt)
        return receipt

    def _finish(
        self, attempt: _Attempt, abort: _MaterializeAbort | None
    ) -> MaterializationReceipt:
        if abort is None:
            return self._build_receipt(
                attempt,
                state=MaterializationState.QUEUED,
                error_code=None,
                error_detail="",
            )
        return self._build_receipt(
            attempt,
            state=MaterializationState.REFUSED,
            error_code=abort.code,
            error_detail=abort.detail,
        )

    def _pending_receipt(
        self, attempt: _Attempt, detail: str
    ) -> MaterializationReceipt:
        return self._build_receipt(
            attempt,
            state=MaterializationState.PENDING,
            error_code=MaterializationErrorCode.UNCERTAIN_DELIVERY,
            error_detail=detail,
        )

    def _persist(self, key: str, receipt: MaterializationReceipt) -> None:
        try:
            self._receipt_store.put(key, receipt)
        except _UNAVAILABLE_ERRORS:
            # The receipt is still returned to the caller; a replay will see
            # the same store failure and keep answering PENDING rather than
            # silently losing this outcome.
            logger.warning("could not persist materialization receipt for %s", key)

    # ── reconciliation ───────────────────────────────────────────────────
    def _reconcile_queued(
        self, receipt: MaterializationReceipt
    ) -> MaterializationReceipt:
        assert receipt.queue_branch is not None
        canonical = self._worktrees.resolve_repo(receipt.repository_id)
        try:
            status = self._queue.status(
                worktree_path=str(canonical or ""), branch=receipt.queue_branch
            )
        except _UNAVAILABLE_ERRORS as exc:
            return self._store_updated(
                receipt,
                state=MaterializationState.PENDING,
                error_code=MaterializationErrorCode.UNCERTAIN_DELIVERY,
                error_detail=str(exc),
            )
        if status == merge_queue.LANDED:
            return self._store_updated(
                receipt,
                state=MaterializationState.MERGED,
                error_code=None,
                error_detail="",
            )
        if status == merge_queue.REJECTED:
            return self._store_updated(
                receipt,
                state=MaterializationState.REFUSED,
                error_code=MaterializationErrorCode.QUEUE_CONFLICT,
                error_detail="the merge queue rejected the candidate",
            )
        return receipt

    def _store_updated(
        self,
        receipt: MaterializationReceipt,
        *,
        state: MaterializationState,
        error_code: MaterializationErrorCode | None,
        error_detail: str,
    ) -> MaterializationReceipt:
        updated = receipt.model_copy(
            update={
                "state": state,
                "error_code": error_code,
                "error_detail": error_detail,
                "updated_at": datetime.now(UTC),
            }
        )
        self._persist(receipt.idempotency_key, updated)
        return updated
