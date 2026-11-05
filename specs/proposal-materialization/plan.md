# RM-MATERIALIZE — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

One service, `MaterializationService` in `repository_manager/materialization/service.py`, is the only path from an approved `ChangeProposal` to a Git commit. It composes the existing `WorktreeManager`, `safe_commit` and `merge_queue` behind one admission boundary and returns an immutable `MaterializationReceipt`. The service is built and tested; what remains is the production entry points (a CLI verb and an MCP tool) and the receipt fields of the target design. Reuse candidates found by searching the code first: `repository_manager/worktree.py`, `repository_manager/safe_commit.py`, `repository_manager/merge_queue.py`, `repository_manager/merge_queue_runner.py` and `repository_manager/development/serialization.py` for the canonical digest.

```text
CLI verb / MCP tool (planned)
  -> MaterializationService.materialize     repository_manager/materialization/service.py:156
  -> _admit -> ApprovalAuthority.verify     service.py:220, ports.py:77
  -> _create_worktree -> WorktreeManager.add
  -> _apply_and_commit -> apply_patch, safe_commit   patch.py:116, safe_commit.py:613
  -> _submit -> GitMergeQueueAdapter.submit -> merge_queue.enqueue   ports.py:119
```

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: pydantic for the frozen contract models; the in-repo `worktree`, `safe_commit` and `merge_queue` modules; the system `git` binary through `run_git` (`patch.py:37`).
- **Storage**: none in this repository; the receipt goes through the injected `MaterializationReceiptStore` port.
- **Testing**: `pytest` under `tests/materialization/` with `FakeGit` and fakes for the three ports; the planned adapter tests under `tests/`.
- **Target Platform**: Linux and macOS hosts; no operator-specific environment.
- **Project Type**: Library with a CLI and an MCP adapter.
- **Performance Goals**: N/A; admission refusals perform no Git process.
- **Constraints**: patch at most 2 MiB (`MAX_PATCH_BYTES`); a refusal before the worktree leaves the canonical checkout untouched; no direct Git fallback.
- **Scale/Scope**: 7 rows, 3 user stories, 7 `FR-` labels and 3 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. The service reuses the existing worktree, commit and queue modules and adds no second Git publisher.
- **II. Complete specification before construction**: pass. Rows carry caller chains, shapes and tests; the unbuilt pieces are marked planned.
- **III. Observable integration and evidence**: pass. Tests enter at `materialize` and cover refusal, idempotency and recovery.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
repository_manager/materialization/__init__.py     existing: public exports
repository_manager/materialization/models.py       existing: ChangeProposal, ApprovalProof, MaterializationReceipt, MaterializationState, MaterializationErrorCode
repository_manager/materialization/patch.py        existing: apply_patch, run_git
repository_manager/materialization/ports.py        existing: ApprovalAuthority, MaterializationReceiptStore, MergeQueuePort, GitMergeQueueAdapter
repository_manager/materialization/service.py      existing: MaterializationService, check_binding
repository_manager/cli_commands/materialization.py new: run_materialize_cli (planned)
repository_manager/mcp_tools/materialization.py    new: register_materialization_tools (planned)
tests/materialization/                             existing: fakes.py, test_service.py
tests/test_materialization_cli.py                  new (planned)
tests/test_materialization_mcp.py                  new (planned)
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The adapter names are accepted as this plan lists them. The open choices (the gate-profile binding and the receipt additions) are owner decisions recorded in the pull request, not unknowns of the build.

## Phase 1: Design

1. **Admission before effect.** `check_binding` is pure; `ApprovalAuthority.verify` answers only authenticity and caller authorization. Nothing touches Git before both pass.
2. **Refuse or pend, never half-apply.** A content problem yields a `refused` receipt and removes the isolated worktree; a dependency problem yields a `pending` receipt with `uncertain_delivery`.
3. **Queue, never land.** `merged` is written only by `reconcile` after the queue reports `LANDED`.
4. **Replay is a lookup.** The idempotency key defaults to the proposal ID; a stored receipt with the same `proposal_digest` is returned and a different digest is refused.
5. **Stable error vocabulary.** `unauthorized`, `expired_approval`, `stale_base`, `path_rejected`, `gate_failed`, `queue_conflict` and `uncertain_delivery` are the values of `MaterializationErrorCode`.
6. **A stale base needs a new approval.** The service never rebases.

Data entities are the three frozen models in `models.py`; the wire shapes are `repository.change-proposal/v1`, `repository.approval-proof/v1` and `repository.materialization-receipt/v1`.

## Phase 2: Task approach

`tasks.md` holds one task per leaf row, producer first: admission (R002), worktree and patch (R003), commit (R004), queue (R005), receipt and replay (R006), pending behavior (R007), then the entry points (R001). No task runs in parallel because each edits `service.py` or depends on its output.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests/materialization` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit` and `vulture` pass without a suppression; git is invoked through `run_git` so no literal argv sits in a `subprocess.run` call.
- jscpd and dupehound are not configured repository hooks; run them where available and report a missing tool.
- The no-duplicate-authority check finds no second function that creates a worktree, applies a patch or commits for a proposal.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

Today the chain starts at `MaterializationService.materialize` (`repository_manager/materialization/service.py:156`) and no production module calls it; the callers are tests. The planned entries are `run_materialize_cli`, dispatched next to `run_merge_queue_cli` at `repository_manager/cli_commands/parser.py:781`, and `register_materialization_tools`, added as an entry of `MCP_TOOL_REGISTRY` at `repository_manager/mcp_tools/registry.py:105`. From `materialize` the chain is `_admit` (`service.py:220`), `_create_worktree` (:237), `_apply_and_commit` (:264), `_submit` (:292) and `_finish` (:361). Effects: a worktree under the shared worktree root, one commit, one queue entry through `merge_queue.enqueue`, one receipt through the injected store. Obsolete paths to delete: any caller route that publishes a proposal with direct Git commands.

## Cross-repo seams

none. The approval authority and the receipt store are injected ports (`ApprovalAuthority`, `MaterializationReceiptStore`); the owner rows of the services that implement them are an owner decision and are cited here once defined.
