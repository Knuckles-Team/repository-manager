"""Governed proposal-to-Git materialization (RM-MATERIALIZE-001).

An approved change proposal becomes an auditable Git candidate only through
:class:`MaterializationService`: it creates an isolated worktree at the
approved base, applies only the approved paths, runs the declared
validation, commits through :func:`repository_manager.safe_commit.safe_commit`
with an explicit path allowlist, and submits the result to the existing
merge queue -- never claiming it landed. See ``specs/proposal-materialization/``
for the full contract. Test doubles for the three pluggable ports live under
``tests/materialization/fakes.py``, never in this package.
"""

from __future__ import annotations

from repository_manager.materialization.models import (
    MAX_PATCH_BYTES,
    ApprovalProof,
    ChangeProposal,
    MaterializationErrorCode,
    MaterializationReceipt,
    MaterializationState,
)
from repository_manager.materialization.patch import PatchRejected, apply_patch, run_git
from repository_manager.materialization.ports import (
    ApprovalAuthority,
    ApprovalAuthorityUnavailable,
    ApprovalVerdict,
    GitMergeQueueAdapter,
    MaterializationReceiptStore,
    MergeQueuePort,
    MergeQueueUnavailable,
    QueueSubmission,
    ReceiptStoreUnavailable,
)
from repository_manager.materialization.service import (
    MaterializationService,
    RepositoryManagerUnavailable,
    check_binding,
)

__all__ = [
    "MAX_PATCH_BYTES",
    "ApprovalAuthority",
    "ApprovalAuthorityUnavailable",
    "ApprovalProof",
    "ApprovalVerdict",
    "ChangeProposal",
    "GitMergeQueueAdapter",
    "MaterializationErrorCode",
    "MaterializationReceipt",
    "MaterializationReceiptStore",
    "MaterializationService",
    "MaterializationState",
    "MergeQueuePort",
    "MergeQueueUnavailable",
    "PatchRejected",
    "QueueSubmission",
    "ReceiptStoreUnavailable",
    "RepositoryManagerUnavailable",
    "apply_patch",
    "check_binding",
    "run_git",
]
