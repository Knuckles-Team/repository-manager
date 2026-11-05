# RM-GOVERNANCE — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

One repository-manager authority owns lane lifecycle, worktree identity, queue admission and concept reservations; other packages call it through bounded APIs. The core is built and tested: `repository_manager/lane_registry.py` and `repository_manager/native_lane_authority.py` (lanes), `repository_manager/worktree.py` (isolated checkouts), `repository_manager/safe_commit.py` (gated commits), `repository_manager/merge_queue.py` and `repository_manager/merge_queue_runner.py` (promotion), `repository_manager/concept_actions.py` and `repository_manager/governance/` (reservations and the lane guard), `repository_manager/gates.py` (gate stages). Several rows are policy rows about gates this repository does not run itself; they stay SPECIFIED with planned tests and the owner decisions listed in the pull request. The host-leased heavy gate (`RM-GOVERNANCE-R023`) extends the existing per-machine lease in `repository_manager/governance/lanes.py`.

```text
repository-manager-governance / MCP tools   repository_manager/governance/cli.py, repository_manager/mcp_tools/
  -> LaneRegistry.allocate -> NativeLaneAuthorityAdapter   lane_registry.py:1317, :555
  -> WorktreeManager.add                                   worktree.py:223
  -> safe_commit                                           safe_commit.py:613
  -> merge_queue.enqueue / run_queue                       merge_queue.py:802, :3583
  -> WorktreeManager.audit / prune / remove                worktree.py:1548, :1076, :771
```

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: the system `git` binary, pydantic, `scion-sdk[mcp]` for the MCP surface; no consumer runtime dependency for the exported hook.
- **Storage**: durable lane and queue state through the lane authority adapter; lease files under the lane lease directory; the gate ledger in SQLite (`repository_manager/gate_ledger.py`).
- **Testing**: `pytest` with disposable Git repositories, fake clocks and fake authority adapters (`FakeDurableLaneAuthority`); the Windows branches run on any host through monkeypatched `portable_fs` functions and on `windows-latest`.
- **Target Platform**: Linux and macOS, and Windows for the portable subset.
- **Project Type**: Library with a CLI and an MCP adapter.
- **Performance Goals**: N/A.
- **Constraints**: refusals change no state; no `git stash` on a shared repository; no implicit staging of all paths; paths resolve canonically and symlink escapes are rejected before any Git operation.
- **Scale/Scope**: 37 leaf rows and 1 rollup, 4 user stories, 11 `FR-` labels and 3 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. One lane registry, one queue and one concept authority; adapters, not duplicates.
- **II. Complete specification before construction**: pass. Each row names its caller chain, shape and test; unbuilt rows are marked planned.
- **III. Observable integration and evidence**: pass. Tests cover refusal, stale-fence, crash recovery and idempotent cleanup.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
repository_manager/lane_registry.py              existing: LaneRegistry, lane authorities
repository_manager/native_lane_authority.py      existing: NativeLaneAuthority
repository_manager/worktree.py                   existing: WorktreeManager
repository_manager/safe_commit.py                existing: safe_commit
repository_manager/merge_queue.py                existing: enqueue, run_queue, dispatch
repository_manager/merge_queue_runner.py         existing: queue drain scheduling
repository_manager/concept_actions.py            existing: concept dispatch
repository_manager/governance/lanes.py           existing: leases, lane scope; gains HostLease (planned)
repository_manager/governance/lane_guard.py      existing: exported lane-guard hook
repository_manager/governance/concept_reservation.py  existing: reservation types
repository_manager/gates.py                      existing: run_gate_stage
.pre-commit-hooks.yaml                           existing: phase-direction, lane-guard
tests/governance/, tests/test_worktree.py, tests/test_safe_commit.py, tests/test_merge_queue.py   existing
tests/test_resource_lease.py                     new (planned)
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The gates this repository does not run (the readme-deploy-block hook, the complexity, KISS, clone and secret-history gates, the status page and extracted-tree verification) are owned by the pipelines gates spec (`PIPE-GATES-R076` to `PIPE-GATES-R084`); the retired rows here point to them.

## Phase 1: Design

1. **Request path.** Validate repository identity, actor and base; allocate and fence the lane; create the worktree; stage explicit paths and run the gates; commit with a gate receipt; admit to the queue; record the merge result; release with a cleanup anchor. Durable lane state records actor, repository, lease, state, commit, tree and gate receipt.
2. **Git is authority.** Git remains the authority for refs and bytes; a concept reservation is bound to the candidate and is checked again at commit and at promotion; expiry is not authorization to reuse an unverified candidate.
3. **Migration.** An adapter makes old callers use this repository's lane, queue and concept contracts; one entry point moves at a time; the old producer is deleted once import searches and contract tests show no callers; stable CLI and MCP action names are kept and changed failures use typed error translation. [`RM-CONNECTOR-R001`](../connector-boundary/spec.md) moves the connector imports onto their legal package APIs after governance is available and does not import an agent orchestration package to reach Git governance.
4. **Security and recovery.** Resolve canonical paths and reject symlink escapes before any Git operation; a caller cannot bypass a fence failure by switching between CLI and MCP; queue and cleanup actions are idempotent and replay from the durable record plus Git ref state, not from an in-memory success flag; uncertain state is preserved until reconciliation proves it safe.
5. **Lane guard export.** `lane-guard` in `.pre-commit-hooks.yaml` is a `language: python` hook that runs `repository_manager.governance.lane_guard` without duplicating policy and without the governance CLI's different mutation-admission check. The hook carries no `stages:` key, so consumers keep their stages. Adopters pin a published revision that contains the hook and add no runtime dependency.
6. **Host-leased heavy gates.** Data flow: inventory capacity, then a per-host lease slot, then the remote gate run, then release. Slot accounting stays in this repository's lease; a second scheduler that picks hosts is rejected because it duplicates the engine's placement decision. The inventory file stays owned by tunnel-manager.
7. **Validation hygiene.** Qualification of `RM-GOVERNANCE-R022` uses current schema-versioned queue fixtures; legacy-compatibility tests assert their expected warnings; the spec marker is registered so requirement-bound tests stay warning-free.

Data entities are the existing lane, lease and reservation types; `HostLease` is the one planned type.

## Phase 2: Task approach

`tasks.md` holds one task per leaf row, producer first: the single authority, lanes, worktrees, commits, queue, cleanup, reservations, hook export, then the gate and host-lease rows. The host-lease rows run in series because they edit `repository_manager/governance/lanes.py`; the planned test files are independent.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests/test_lane_registry.py tests/test_native_lane_authority.py tests/test_safe_commit.py tests/test_merge_queue.py tests/test_merge_queue_runner.py tests/governance` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit` and `vulture` pass without a suppression; `check-import-safety` passes on the Windows job.
- jscpd and dupehound are not configured repository hooks; record their availability instead of asserting a pass.
- The no-duplicate-authority check finds one lane registry, one queue and one concept claim authority.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

Entry points are the `repository-manager-governance` console script (`pyproject.toml:55`, `repository_manager/governance/cli.py`), the `repository-manager` CLI and the MCP tools in `repository_manager/mcp_tools/` (`lane.py`, `worktree.py`, `merge_queue.py`, `git.py`). They call `LaneRegistry` (`repository_manager/lane_registry.py:648`), `WorktreeManager` (`repository_manager/worktree.py:140`), `safe_commit` (`repository_manager/safe_commit.py:613`), `merge_queue.dispatch` (`repository_manager/merge_queue.py:3621`) and `concept_actions.dispatch` (`repository_manager/concept_actions.py:124`). Effects: Git worktrees and refs, durable lane records, queue records and lease files. Obsolete paths to delete: any caller of the retired lane module, which the import scan of `tests/governance/test_no_parallel_lane_authority.py` already forbids.

## Cross-repo seams

- Meristem run loop and repository operations: `MERISTEM-RUN-R016`, provider row `RM-GOVERNANCE-R001`, contract in `contracts/meristem-repository-manager.md`.
