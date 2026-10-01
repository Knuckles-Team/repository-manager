# RM-MATERIALIZE-001 architecture

## Reuse inventory

Use `repository_manager/worktree.py` for worktree creation and identity, `safe_commit.py` for gated explicit commits, `merge_queue.py` and `merge_queue_runner.py` for promotion, `validation_policy.py` and `validation/` for gate profiles/evidence, and `lane_registry.py` for lease/fence lifecycle. CLI/MCP adapters should call one application service rather than implement Git operations twice. No current method is claimed to satisfy the whole proposal protocol; the missing boundary is the authenticated proposal adapter and immutable receipt/reconciliation service.

## Contract and data model

`ChangeProposal/v1` contains immutable IDs, repository and tenant scope, base SHA, patch bytes/digest, allowed relative paths, approval identity/digest, requested validation profile, and expiry. Admission validates schema and approval before worktree creation. The materializer records state `PENDING → APPLIED → VALIDATED → COMMITTED → QUEUED → MERGED` or a typed terminal/refusal state; every transition stores a monotonic event and idempotency key. `MaterializationReceipt/v1` contains proposal digest, approval digest, base SHA, commit SHA, tree SHA, gate config and result digests, queue record/result, actor, timestamps, and a canonical digest/signature. Redact tokens and private filesystem paths from the public receipt.

## Sequence

`verify request/approval → reserve lane → check base and allowlist → create worktree → apply patch → validate exact tree → explicit stage/commit → queue → reconcile hosted result → emit receipt`. A delivered receipt can be checked against `git cat-file` and queue records; uncertain remote or graph delivery remains pending until reconciliation. A stale base requires a new approval, never implicit rebase. Rollback removes only uncommitted isolated worktrees after preserving diagnostic evidence; committed candidates remain reachable by ref/anchor.

## Contributor quickstart

Clone the public repository and run `uv sync --extra test`; run `uv run pytest tests/test_safe_commit.py tests/test_merge_queue.py tests/test_lane_registry.py`. Add protocol tests with temporary Git repos and a fake approval verifier. Supply a contributor-owned repository path in a temporary manifest for manual rehearsal. Tokens, a live graph, and an existing merge queue daemon are unnecessary for ordinary PR validation.

## Interface and migration

Expose one typed RM materialization operation through both CLI and MCP; a remote A2A adapter can wrap it after authentication. Keep errors stable: `UNAUTHORIZED`, `EXPIRED_APPROVAL`, `STALE_BASE`, `PATH_REJECTED`, `GATE_FAILED`, `QUEUE_CONFLICT`, and `UNCERTAIN_DELIVERY`. Migrate callers by switching to the typed operation, then remove direct Git publication routes. Observe latency, refusal reason, receipt reconciliation lag, and queue outcome without logging patch secrets.
