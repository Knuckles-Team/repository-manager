"""Qualification tests for RM-MATERIALIZE-001 against disposable git repos.

Every test drives a REAL temporary Git repository through
:class:`~repository_manager.materialization.service.MaterializationService`,
with fakes standing in only for the three pluggable ports (approval
authority, receipt store, merge queue) -- exactly plan.md's "local fake
approval and temporary Git repository fixtures cover every rule".
"""

from __future__ import annotations

import hashlib
import shlex
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repository_manager import merge_queue
from repository_manager import worktree as wt_mod
from repository_manager.development.serialization import (
    deserialize_contract,
    serialize_contract,
)
from repository_manager.materialization import (
    ApprovalProof,
    ChangeProposal,
    MaterializationErrorCode,
    MaterializationReceipt,
    MaterializationService,
    MaterializationState,
    PatchRejected,
    apply_patch,
    check_binding,
)
from repository_manager.worktree import WorktreeManager
from tests.materialization.fakes import (
    FakeApprovalAuthority,
    FakeMergeQueuePort,
    InMemoryReceiptStore,
)

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=UTC)
FUTURE = NOW + timedelta(hours=1)
PAST = NOW - timedelta(hours=1)


class FakeGit:
    """Minimal Git stand-in exposing the surface WorktreeManager uses."""

    def __init__(self, workspace: str) -> None:
        self.path = workspace
        self.project_map: dict[str, str] = {}

    def git_action(
        self, command, path=None, quiet=False, env=None, timeout=1800, raw_output=False
    ):
        del env, timeout, raw_output
        proc = subprocess.run(
            shlex.split(command),  # same parsing as Git.git_action
            cwd=path or self.path,
            capture_output=True,
            text=True,
        )
        out = (proc.stdout + proc.stderr).strip()
        return SimpleNamespace(
            status="success" if proc.returncode == 0 else "error",
            data=out,
            error=(
                None
                if proc.returncode == 0
                else SimpleNamespace(message=out, code=proc.returncode)
            ),
        )


def _git(
    args: list[str], cwd: Path, *, check: bool = True
) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=check
    )


def _repo(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, name: str = "repo") -> Path:
    repo = tmp_path / name
    repo.mkdir()
    _git(["init", "-b", "main"], repo)
    _git(["config", "user.email", "materialize@test"], repo)
    _git(["config", "user.name", "materialize"], repo)
    (repo / "allowed.txt").write_text("base\n")
    (repo / "other.txt").write_text("base\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "base"], repo)
    monkeypatch.setattr(wt_mod, "WORKTREE_ROOT", str(tmp_path / "worktrees"))
    return repo


def _base_sha(repo: Path) -> str:
    return _git(["rev-parse", "main"], repo).stdout.strip()


def _make_patch(
    repo: Path,
    tmp_path: Path,
    changes: dict[str, str | None],
    *,
    label: str,
    symlink: bool = False,
    binary: bool = False,
) -> str:
    """Build a real unified diff by editing a disposable detached worktree."""

    scratch = tmp_path / f"scratch-{label}"
    _git(["worktree", "add", "--detach", str(scratch), "main"], repo)
    try:
        for rel, content in changes.items():
            target = scratch / rel
            if content is None:
                target.unlink()
            elif symlink:
                target.unlink(missing_ok=True)
                target.symlink_to(content)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(
                    content.encode("latin-1") if binary else content.encode()
                )
        diff_args = ["diff", "--no-color"] + (["--binary"] if binary else [])
        diff = _git(diff_args, scratch, check=False)
        return diff.stdout
    finally:
        _git(["worktree", "remove", "--force", str(scratch)], repo, check=False)


def _proposal(
    repo: Path,
    *,
    patch_text: str,
    approved_paths: tuple[str, ...] = ("allowed.txt",),
    proposal_id: str = "proposal-1",
    base_sha: str | None = None,
    repository_id: str | None = None,
    expires_at: datetime = FUTURE,
) -> ChangeProposal:
    return ChangeProposal(
        proposal_id=proposal_id,
        repository_id=repository_id or str(repo),
        tenant_id="tenant-a",
        base_sha=base_sha or _base_sha(repo),
        approved_paths=approved_paths,
        patch_digest=hashlib.sha256(patch_text.encode()).hexdigest(),
        patch_text=patch_text,
        decision_identity="reviewer-a",
        validation_profile="default",
        expires_at=expires_at,
    )


def _approval(proposal: ChangeProposal, **overrides: Any) -> ApprovalProof:
    fields: dict[str, Any] = {
        "approval_id": "approval-1",
        "proposal_digest": proposal.digest,
        "repository_id": proposal.repository_id,
        "tenant_id": proposal.tenant_id,
        "actor_id": proposal.decision_identity,
        "issued_at": NOW,
        "expires_at": FUTURE,
        "signature": "signed-1",
    }
    fields.update(overrides)
    return ApprovalProof(**fields)


def _service(
    repo: Path,
    *,
    authority=None,
    store=None,
    queue=None,
    gate=None,
) -> tuple[MaterializationService, InMemoryReceiptStore]:
    store = store or InMemoryReceiptStore()
    service = MaterializationService(
        worktrees=WorktreeManager(FakeGit(str(repo.parent))),
        approval_authority=authority or FakeApprovalAuthority(),
        receipt_store=store,
        queue=queue or FakeMergeQueuePort(),
        gate=gate,
    )
    return service, store


def _simple_patch(repo: Path, tmp_path: Path) -> str:
    return _make_patch(repo, tmp_path, {"allowed.txt": "changed\n"}, label="simple")


# ── RM-MATERIALIZE-01: approval verification gates every request ───────────


def test_expired_approval_refuses_without_any_worktree(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal, expires_at=PAST)
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.EXPIRED_APPROVAL
    assert receipt.commit_sha is None
    assert not (Path(wt_mod.WORKTREE_ROOT)).exists()
    assert _git(["status", "--porcelain"], repo).stdout == ""


def test_approval_bound_to_a_different_proposal_is_unauthorized(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    tampered = proposal.model_copy(update={"decision_identity": "someone-else"})
    approval = _approval(proposal)  # still bound to the ORIGINAL digest
    service, _ = _service(repo)

    receipt = service.materialize(tampered, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.UNAUTHORIZED
    assert "not bound" in receipt.error_detail
    assert not Path(wt_mod.WORKTREE_ROOT).exists()


def test_approval_for_a_different_repository_is_unauthorized(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal, repository_id="some-other-repo")
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.UNAUTHORIZED
    assert not Path(wt_mod.WORKTREE_ROOT).exists()


def test_approval_from_the_wrong_actor_is_unauthorized(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal, actor_id="not-the-reviewer")
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.error_code is MaterializationErrorCode.UNAUTHORIZED


def test_approval_authority_rejection_is_unauthorized(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(
        repo, authority=FakeApprovalAuthority(accepted_signatures=frozenset())
    )

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.UNAUTHORIZED
    assert not Path(wt_mod.WORKTREE_ROOT).exists()


def test_check_binding_is_pure_and_lists_every_structural_reason():
    repo_a = ChangeProposal(
        proposal_id="p",
        repository_id="repo-a",
        tenant_id="tenant-a",
        base_sha="a" * 40,
        approved_paths=("x.txt",),
        patch_digest=hashlib.sha256(b"diff").hexdigest(),
        patch_text="diff",
        decision_identity="reviewer",
        validation_profile="default",
        expires_at=FUTURE,
    )
    approval = ApprovalProof(
        approval_id="appr",
        proposal_digest="0" * 64,
        repository_id="repo-b",
        tenant_id="tenant-b",
        actor_id="someone-else",
        issued_at=NOW,
        expires_at=PAST,
        signature="sig",
    )

    reasons = check_binding(repo_a, approval, now=NOW)

    assert len(reasons) == 5  # expired approval, digest, repo, tenant, actor


# ── RM-MATERIALIZE-02: isolated, allowlist-restricted patch application ────


def test_valid_patch_touches_only_the_approved_path(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(repo, gate=lambda path: True)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.QUEUED
    assert receipt.commit_sha is not None
    changed = _git(
        ["show", "--format=", "--name-only", receipt.commit_sha], repo
    ).stdout.split()
    assert changed == ["allowed.txt"]


def test_patch_touching_an_unapproved_path_is_rejected(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    patch = _make_patch(repo, tmp_path, {"other.txt": "changed\n"}, label="unapproved")
    proposal = _proposal(repo, patch_text=patch, approved_paths=("allowed.txt",))
    approval = _approval(proposal)
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.PATH_REJECTED
    assert receipt.commit_sha is None
    assert _git(["status", "--porcelain"], repo).stdout == ""


def test_a_failed_worktree_cleanup_is_logged_and_does_not_mask_the_refusal(
    tmp_path, monkeypatch, caplog
):
    repo = _repo(tmp_path, monkeypatch)
    patch = _make_patch(repo, tmp_path, {"other.txt": "changed\n"}, label="cleanup")
    proposal = _proposal(repo, patch_text=patch, approved_paths=("allowed.txt",))
    service, _ = _service(repo)

    def _refuse_removal(*_args: Any, **_kwargs: Any) -> None:
        raise OSError("worktree is busy")

    monkeypatch.setattr(service._worktrees, "remove", _refuse_removal)
    with caplog.at_level("WARNING"):
        receipt = service.materialize(proposal, _approval(proposal), now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.PATH_REJECTED
    assert "could not remove abandoned materialization worktree" in caplog.text


def test_patch_header_with_traversal_is_rejected_statically():
    traversal_patch = (
        "diff --git a/../outside.txt b/../outside.txt\n"
        "--- a/../outside.txt\n"
        "+++ b/../outside.txt\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
    )
    with pytest.raises(PatchRejected, match="unsafe component"):
        apply_patch(Path("/nonexistent"), traversal_patch, ("allowed.txt",))


def test_oversized_patch_is_rejected_before_any_git_mutation_can_start():
    huge = "x" * (2 * 1024 * 1024 + 1)
    with pytest.raises(ValueError, match="exceeds"):
        ChangeProposal(
            proposal_id="p",
            repository_id="repo",
            tenant_id="tenant",
            base_sha="a" * 40,
            approved_paths=("x.txt",),
            patch_digest=hashlib.sha256(huge.encode()).hexdigest(),
            patch_text=huge,
            decision_identity="reviewer",
            validation_profile="default",
            expires_at=FUTURE,
        )


def test_stale_base_is_rejected_before_any_worktree_is_created(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    stale_base = _base_sha(repo)
    (repo / "allowed.txt").write_text("someone landed first\n")
    _git(["add", "-A"], repo)
    _git(["commit", "-m", "advances main past the approved base"], repo)

    proposal = _proposal(
        repo, patch_text=_simple_patch(repo, tmp_path), base_sha=stale_base
    )
    approval = _approval(proposal)
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.STALE_BASE
    assert not Path(wt_mod.WORKTREE_ROOT).exists()


def test_unknown_repository_is_rejected_as_stale_base(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(
        repo,
        patch_text=_simple_patch(repo, tmp_path),
        repository_id=str(tmp_path / "does-not-exist"),
    )
    approval = _approval(proposal)
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.error_code is MaterializationErrorCode.STALE_BASE


def test_patch_declaring_a_new_symlink_is_rejected(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    patch = _make_patch(
        repo, tmp_path, {"allowed.txt": "/etc/passwd"}, label="symlink", symlink=True
    )
    proposal = _proposal(repo, patch_text=patch)
    approval = _approval(proposal)
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.PATH_REJECTED
    assert "symlink" in receipt.error_detail


def test_binary_patch_content_is_rejected(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    patch = _make_patch(
        repo, tmp_path, {"allowed.txt": "\x00\x01binary"}, label="binary", binary=True
    )
    proposal = _proposal(repo, patch_text=patch)
    approval = _approval(proposal)
    service, _ = _service(repo)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.PATH_REJECTED
    assert "binary" in receipt.error_detail


# ── RM-MATERIALIZE-03: validation gates and explicit staging precede commit ─


def test_a_failing_gate_blocks_both_commit_and_queue(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    base_before = _base_sha(repo)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(repo, gate=lambda path: False, queue=FakeMergeQueuePort())

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.GATE_FAILED
    assert receipt.commit_sha is None
    assert receipt.queue_accepted is None
    assert _base_sha(repo) == base_before  # `main` never moved


def test_a_passing_gate_commits_with_exactly_the_reviewed_paths(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    observed: list[list[str]] = []

    def gate(path: Path) -> bool:
        observed.append(_git(["diff", "--cached", "--name-only"], path).stdout.split())
        return True

    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(repo, gate=gate)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.QUEUED
    assert observed == [["allowed.txt"]]


# ── RM-MATERIALIZE-04: queue submission never claims landing ───────────────


def test_queue_acceptance_is_reported_as_queued_never_merged(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(
        repo, gate=lambda path: True, queue=FakeMergeQueuePort(accept=True)
    )

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.QUEUED
    assert receipt.state is not MaterializationState.MERGED
    assert receipt.queue_accepted is True


def test_queue_conflict_is_reported_as_refused_not_success(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(
        repo,
        gate=lambda path: True,
        queue=FakeMergeQueuePort(accept=False, detail="branch conflicts with base"),
    )

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.REFUSED
    assert receipt.error_code is MaterializationErrorCode.QUEUE_CONFLICT
    # The commit itself still exists -- only the QUEUE step was refused.
    assert receipt.commit_sha is not None


def test_reconcile_promotes_a_landed_candidate_to_merged(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    queue = FakeMergeQueuePort(accept=True)
    service, store = _service(repo, gate=lambda path: True, queue=queue)
    receipt = service.materialize(proposal, approval, now=NOW)
    assert receipt.state is MaterializationState.QUEUED

    queue.reconciled_status = merge_queue.LANDED
    reconciled = service.reconcile(receipt.idempotency_key)

    assert reconciled.state is MaterializationState.MERGED
    stored = store.get(receipt.idempotency_key)
    assert stored is not None
    assert stored.state is MaterializationState.MERGED


def test_reconcile_reports_a_rejected_candidate_as_refused(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    queue = FakeMergeQueuePort(accept=True)
    service, _ = _service(repo, gate=lambda path: True, queue=queue)
    receipt = service.materialize(proposal, approval, now=NOW)

    queue.reconciled_status = merge_queue.REJECTED
    reconciled = service.reconcile(receipt.idempotency_key)

    assert reconciled.state is MaterializationState.REFUSED
    assert reconciled.error_code is MaterializationErrorCode.QUEUE_CONFLICT


# ── RM-MATERIALIZE-05: an immutable receipt binds every outcome ────────────


def test_a_replayed_request_returns_the_same_receipt_without_a_second_commit(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, monkeypatch)
    base_before = _base_sha(repo)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(repo, gate=lambda path: True)

    first = service.materialize(proposal, approval, now=NOW, idempotency_key="same-key")
    second = service.materialize(
        proposal, approval, now=NOW, idempotency_key="same-key"
    )

    assert first == second
    assert _base_sha(repo) == base_before  # nothing landed on `main`
    ahead_of_base = _git(
        ["rev-list", "--count", f"main..{first.queue_branch}"], repo
    ).stdout.strip()
    assert ahead_of_base == "1"  # exactly one commit landed on the candidate branch


def test_an_altered_replay_under_the_same_key_is_rejected(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    patch_a = _simple_patch(repo, tmp_path)
    proposal_a = _proposal(repo, patch_text=patch_a, proposal_id="proposal-a")
    approval_a = _approval(proposal_a)
    service, _ = _service(repo, gate=lambda path: True)
    first = service.materialize(
        proposal_a, approval_a, now=NOW, idempotency_key="shared-key"
    )
    assert first.state is MaterializationState.QUEUED

    patch_b = _make_patch(
        repo, tmp_path, {"allowed.txt": "a different change\n"}, label="altered"
    )
    proposal_b = _proposal(repo, patch_text=patch_b, proposal_id="proposal-b")
    approval_b = _approval(proposal_b)

    second = service.materialize(
        proposal_b, approval_b, now=NOW, idempotency_key="shared-key"
    )

    assert second.state is MaterializationState.REFUSED
    assert second.error_code is MaterializationErrorCode.UNAUTHORIZED
    assert "already bound" in second.error_detail


def test_materialization_receipt_contract_round_trips_through_canonical_serialization(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(repo, gate=lambda path: True)

    receipt = service.materialize(proposal, approval, now=NOW)
    payload = serialize_contract(receipt)
    restored = deserialize_contract(MaterializationReceipt, payload)

    assert restored == receipt
    assert restored.digest == receipt.digest
    assert receipt.commit_sha is not None
    # The receipt's commit_sha is independently verifiable against real Git.
    assert (
        _git(["cat-file", "-e", receipt.commit_sha], repo, check=False).returncode == 0
    )


# ── RM-MATERIALIZE-06: no direct Git fallback when a dependency is down ────


def test_unavailable_approval_authority_leaves_the_request_pending(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(repo, authority=FakeApprovalAuthority(raise_unavailable=True))

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.PENDING
    assert receipt.error_code is MaterializationErrorCode.UNCERTAIN_DELIVERY
    assert receipt.commit_sha is None
    assert not Path(wt_mod.WORKTREE_ROOT).exists()


def test_unavailable_receipt_store_leaves_the_request_pending(tmp_path, monkeypatch):
    repo = _repo(tmp_path, monkeypatch)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    broken_store = InMemoryReceiptStore(raise_unavailable=True)
    service, _ = _service(repo, store=broken_store)

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.PENDING
    assert receipt.error_code is MaterializationErrorCode.UNCERTAIN_DELIVERY
    assert not Path(wt_mod.WORKTREE_ROOT).exists()


def test_unavailable_merge_queue_leaves_a_committed_candidate_pending(
    tmp_path, monkeypatch
):
    repo = _repo(tmp_path, monkeypatch)
    base_before = _base_sha(repo)
    proposal = _proposal(repo, patch_text=_simple_patch(repo, tmp_path))
    approval = _approval(proposal)
    service, _ = _service(
        repo, gate=lambda path: True, queue=FakeMergeQueuePort(raise_unavailable=True)
    )

    receipt = service.materialize(proposal, approval, now=NOW)

    assert receipt.state is MaterializationState.PENDING
    assert receipt.error_code is MaterializationErrorCode.UNCERTAIN_DELIVERY
    # The commit DID happen (an isolated worktree, never the canonical tree);
    # only the QUEUE submission -- which this receipt reports uncertain -- is
    # in doubt. The canonical repo's own `main` is still untouched.
    assert receipt.commit_sha is not None
    assert _base_sha(repo) == base_before
