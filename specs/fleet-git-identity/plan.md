# RM-IDENTITY — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The read-only half is built: discovery, a deterministic preview, approval verification and a branch-consistency check, reached through the `history-identity` verb of `repository-manager-governance`. The rewrite half (verifier, quarantined transaction, lease-protected publication, queue coordination) is specified and unbuilt, and attaches to the same verb through one `run_rewrite`. Reuse candidates found by searching the code first: `repository_manager/history_identity/` for discovery, policy and approval; `repository_manager/merge_queue.py` for pause and resume (its action table has no pause today); `repository_manager/worktree.py` for isolation; `repository_manager/development/workspace_release.py` for dependency-ordered coordination and digest patterns.

```text
repository-manager-governance history-identity    repository_manager/governance/cli.py:156
  -> _history_identity                            governance/cli.py:566
  -> run_preview                                  repository_manager/history_identity/cli.py:223
  -> discover / generate_preview / verify_approval  discovery.py:194, preview.py:157, approval.py:106
  -> run_rewrite (planned)                        repository_manager/history_identity/rewrite.py
```

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: the system `git` binary through the injected `GitRunner` (`SubprocessGitRunner`); the standard library `hmac` and `hashlib`; the commit-identity allowlist file owned by the fleet gate.
- **Storage**: Git refs and objects; backup refs and a content-addressed object map for the planned half.
- **Testing**: `pytest` with temporary repositories from `tests/history_identity_git_fixtures.py`.
- **Target Platform**: Linux and macOS hosts with Git.
- **Project Type**: Library with a governance CLI verb.
- **Performance Goals**: N/A; the preview is linear in the number of commits.
- **Constraints**: the preview mutates nothing; the allowlist is at most 64 KiB and 256 identities (`policy.py`); an unknown identity is refused.
- **Scale/Scope**: 9 rows, 3 user stories, 7 `FR-` labels and 4 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. One policy type and one Git runner protocol; the allowlist is read from the gate's file, not copied.
- **II. Complete specification before construction**: pass. Unbuilt rows are marked planned with exact attach points.
- **III. Observable integration and evidence**: pass. CLI tests enter through the real parser.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
repository_manager/history_identity/discovery.py    existing: discover, GitRunner, RepositoryDiscovery
repository_manager/history_identity/policy.py       existing: IdentityPolicy, build_policy
repository_manager/history_identity/preview.py      existing: generate_preview, HistoryPreview
repository_manager/history_identity/approval.py     existing: ApprovalBinding, SignedApproval, verify_approval
repository_manager/history_identity/consistency.py  existing: check_history_consistency, admit_for_rewrite_plan
repository_manager/history_identity/cli.py          existing: run_preview
repository_manager/governance/cli.py                existing: history-identity verb
repository_manager/history_identity/rewrite.py      new: run_rewrite (planned)
repository_manager/history_identity/verify.py       new: verify_rewrite (planned)
repository_manager/history_identity/transaction.py  new: stage_and_update (planned)
repository_manager/history_identity/publish.py      new: publish_remotes (planned)
repository_manager/merge_queue.py                   existing: gains pause and resume actions (planned)
tests/test_history_identity_*.py                    existing and new (planned)
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The open choices (the approval mechanism beyond a shared-key HMAC, the migration notice channel and wording, the pull-request refresh mechanism) are owner decisions recorded in the pull request.

## Phase 1: Design

1. **Read-only first.** Discovery, preview and approval never write; `test_history_identity_cli_mutates_nothing` proves it.
2. **One digest chain.** The discovery digest and the policy digest fold into the plan digest; the approval binds to the combined source digest, the policy digest, the remotes and the expiry, so any change restarts preview and approval.
3. **Exact matching.** `IdentityPolicy.resolve` matches the exact name and email; the unknown-identity rule is fixed to `refuse`.
4. **Independent verification.** The planned verifier shares no code with the rewrite engine.
5. **Atomic local update, leased remote update.** One `git update-ref --stdin` transaction locally; one `--force-with-lease` per ref and remote.
6. **Recovery.** Backup refs are kept; a recovery replays the old SHA from the backups with leases after a new approval.

Data entities are the existing `IdentityPolicy`, `HistoryPreview`, `ApprovalBinding` and the planned per-remote receipt.

## Phase 2: Task approach

`tasks.md` holds one task per leaf row, producer first: discovery, consistency, preview, approval, then verifier, transaction, publication, queue coordination and the composing `run_rewrite`. The planned modules are independent files and the verifier task runs before the transaction task.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests -k history_identity` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit` and `vulture` pass without a suppression.
- jscpd and dupehound are not configured repository hooks; run them where available and report a missing tool.
- The no-duplicate-authority check finds one policy evaluator and one rewrite transaction service.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

The chain starts at the shell verb `repository-manager-governance history-identity`: parsed at `repository_manager/governance/cli.py:156`, dispatched through `_COMMAND_HANDLERS` (:593) to `_history_identity` (:566), then `run_preview` (`repository_manager/history_identity/cli.py:223`), which calls `check_history_consistency`, `discover`, `generate_preview` and, with `--approval`, `verify_approval`. The effect boundary today is a JSON report and an exit code. The planned apply mode calls `run_rewrite` after `verify_approval` and ends at Git refs and remotes. Obsolete paths to delete: none.

## Cross-repo seams

none. The commit-identity allowlist is read from the file that the fleet commit-identity gate owns; the owning row in the gate repository is an owner decision and is cited here once defined.
