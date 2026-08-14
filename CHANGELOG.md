# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Dependency-readiness gate (CONCEPT:RM-DEP-READY)** — `repository_manager/dependency_readiness.py`,
  a pluggable artifact-availability predicate (`IndexBackend` protocol; `PyPISimpleIndexBackend`
  default, over the PEP 503/691 Simple Repository API — never a hardcoded `pypi.org`, honors
  `UV_INDEX_URL`/`PIP_INDEX_URL`/`[tool.uv.index]`) closing the blind-sleep gap in `phased_push`:
  **Layer 1** — a new `[manual, pre-push]` local hook (`dependency-readiness`) reads a repo's own
  declared PEP 440 constraints on other fleet packages (scoped via `workspace.yml`, excluding the
  non-package `images`/`services` subdirectory trees) and fails the push when none of the
  configured index's published versions satisfy one, naming the package, the declared constraint,
  and what's actually available — proven live against `agent-utilities`' real, currently-unsatisfiable
  `epistemic-graph[full]>=2.23.2,<3.0.0` (PyPI holds only `2.23.0`) and `agent-utilities>=2.0.0,<3.0.0`
  (PyPI holds only `1.26.4`) constraints. **Layer 2 (gate-driven, not polling)** — `Git.phased_push`
  decides a phase transition by RUNNING each downstream repo's own pre-push gate
  (`dependency_readiness.await_gate_readiness` → `gates.run_gate_stage(path, "heavy",
  hook_ids=["dependency-readiness"])` — the SAME call `Git._gate_before_push` makes before that
  repo's own real push), retrying with bounded exponential backoff up to a `wait_minutes` ceiling
  and **aborting the wave** (never advancing) if every attempt still fails, naming the specific
  failing repo and constraint. `wait_minutes` is kept, repurposed from a sleep duration into that
  retry ceiling, so existing `workspace.yml` manifests need no migration. A downstream repo that
  hasn't yet adopted the hook (gradual fleet rollout) is treated as unverifiable and blocks, never
  silently passes. Supersedes an earlier poll-the-package-index-directly design
  (`dependency_readiness.await_constraints`, removed) that duplicated Layer 1's own check in a
  second implementation — one mechanism now decides both "is this phase transition ready" and
  "will this repo's own push succeed", because they're the same call. `gates.run_gate_stage` gained
  a `hook_ids` parameter so a retry loop can rerun one fast network-bound hook instead of a repo's
  entire heavy suite (pytest, mypy, ...) on every attempt. `Git.phased_push`'s `bulk_push` phases
  (e.g. "Phase 5: Agents") also gained a scope guard — they resolve against the WHOLE workspace
  manifest's project map, which also holds every `images/`/`services/` infra repo, so bulk_push now
  skips those trees by construction, plus a newly-wired declarative `exclude` (fnmatch-on-name)
  phase field for explicit exclusions. Distinguishes index-unreachable from version-absent in every
  Layer-1 message, with a brief retry for the publish-but-not-yet-CDN-visible window, and one
  loud/audited override (`RM_DEPENDENCY_READINESS_OVERRIDE_REASON`) — never a silent skip. Fleet
  rollout via `scripts/sweep_dependency_readiness_hook.py` (same indentation-safe injection
  technique as the two-tier model's own sweep) — **a prerequisite for the gate-driven barrier to
  ever pass on a real fleet push**, since a downstream repo without the hook is treated as
  unverifiable, not silently ready.
- **Universal merge queue (CONCEPT:RM-MERGE-QUEUE)** — `repository_manager/merge_queue.py`, a
  **repo-agnostic** serialized merge queue: per-repo append-only candidate store, differential
  gating against a base ref, regenerate-on-land, guarded prune, the `reconciliation-merge` lease,
  fold-by-`recorded_at` ordering, optimistic batching with bisection, and fast-forward-only landing
  under BOTH `lanes.guarded_tree_mutation` and `canonical_guard.guarded_canonical_mutation`.
  Gates are **declarative and per repository** (`.mergequeue.yaml`) — the queue never knows what a
  gate is, only how to run one and compare its result against the base ref. Ported from
  `agent_utilities.governance.merge_queue`, which could serve only agent-utilities; the au queue is
  untouched and stays live (see `docs/merge-queue.md` for the migration plan).
- **`--merge-queue {enqueue,status,withdraw,run,config}` CLI verbs** and the **`rm_merge_queue` MCP
  tool**, both thin marshallers over one `merge_queue.dispatch()` action core. Exit 75
  (`EX_TEMPFAIL`) means another runner holds the repository's lease — defer, do not retry.
- **`mergequeue_presets/`** — shipped gate declarations for `epistemic-graph`
  (`cargo check --all-features` + clippy; the proof of genericity beyond Python — eg has had **no**
  merge queue at all) and `agent-utilities` (pytest + ruff + contract scripts). Presets are inert
  until copied into a repository root, which is what makes adoption per-repo and reversible.
- **`WorktreeManager.delete_merged_branch()`** — public entry to the guarded ref deletion
  (merge-base re-checked at delete time, `refs/lane-backup/<branch>` anchor, `git branch -d` never
  `-D`) so the queue reuses that guard instead of reimplementing it (D-ORC-21).
- **Landing writes the DECLARED base ref, and proves it moved (D-RMD-1)** — `land()` previously ran
  `git merge --ff-only` in the canonical checkout, which merges into whatever `HEAD` is rather than
  into the candidate's declared `base`. It now fully qualifies the ref (`refs/heads/<base>`), moves
  it by `merge --ff-only` only when the canonical checkout genuinely holds that branch and otherwise
  by a compare-and-swap `git update-ref <ref> <new> <expected-old>`, enforces fast-forward-only
  against the BASE ref rather than `HEAD`, and refuses when the base is checked out in another
  worktree. It then **re-reads the ref and asserts it holds the computed commit before anything is
  reported `landed`** — the durable half, which catches a wrong write target even with the bug still
  in place. Previously the queue could report `landed` while the base never moved, after which the
  guarded prune deleted the branch *as landed*: silent, positive, and self-erasing. A non-existent
  base ref is now refused once at `run_queue` entry with an actionable message.
- **Pre-commit data-loss guard (D-ORC-37)** — `refuse_precommit_on_dirty_tree()` refuses to run any
  pre-commit gate against a tree holding uncommitted work (pre-commit checks unstaged changes out
  of the tree while hooks run; a crash in that window loses them, D-OB-12), and every gate run gets
  a partitioned `PRE_COMMIT_HOME` so a crash can never orphan another lane's patch and the store's
  SQLite `db.db` can never lock against another lane's.

### Added
- **`rm_worktree audit` action (CONCEPT:RM-WORKTREE-AUDIT)** — classifies each worktree / repo git
  state as merged (safe to prune), active (in-flight), stale, or dangling, with per-repo unpushed
  detection and orphan listing. Read-only by default; `prune_merged` is opt-in.
- **Audit-aware worktree hygiene in the release flow** — `Git.worktree_hygiene()` is chained as a
  report-only step into `rm_projects validate` (`auto_bump`/`auto_push`); `prune_worktrees=true`
  switches it to audit-aware cleanup that replaces the prior blind worktree reaping.
- **Fast pre-push gate** — `_gate_before_push` runs the repo's own CI gates (`pre-commit run
  --all-files` with `SKIP=pytest`) before each repo's push so a `--no-verify` phased commit can't
  ship a commit the repo's CI then rejects. No-op for repos with nothing to push or no
  `.pre-commit-config.yaml`; a tooling/env failure never blocks a push (only a real hook failure
  does); toggle with `RM_GATE_BEFORE_PUSH=false` (default on).

### Changed
- **Pre-push gate scopes per-file hooks to the pushed diff** — runs `pre-commit run --files
  <git diff @{u}..HEAD>` instead of `--all-files`, so ruff/mypy no longer re-check the whole repo on
  every push (`always_run` guardrail gates still run fully; falls back to `--all-files` when the diff
  can't be computed).

### Fixed
- **Worktree prune could delete an active lane's branch ref (CONCEPT:RM-PRUNE-GUARD, `D-FE-9`)** —
  `rm_worktree audit --prune-merged` treated a `merged` classification as authorisation to remove a
  worktree and run `git branch -D`. It removed a live lane's `agent-utilities` worktree *and* its
  branch mid-run; the commits survived only as dangling objects. `merged` was a correct reading —
  the lane had merged an intermediate chunk back to `main` and kept working — which is the point:
  mergedness is not vacancy, and a scan-time classification is not a delete-time authorisation.
  Now: `merged` additionally requires `behind > 0` (so a worktree still sitting on `base` — a lane
  that has not started — is `active`, reported as `at_base`, never prunable); every removal runs
  inside `agent_utilities.governance.lanes.guarded_tree_mutation` with `_branch_state` re-derived
  under that lease; git's own `worktree lock` is honoured; and branch deletion goes through
  `_delete_merged_branch`, which re-asks `git merge-base --is-ancestor` on the exact tip at deletion
  time, anchors it at `refs/lane-backup/<branch>`, and uses `git branch -d` — never `-D` — so git
  re-decides reachability under its own ref lock. `rm_worktree remove --delete-branch` uses the same
  gate (`--force` covers the recoverable directory, never the ref) and both report `branch_anchor` /
  `branch_kept_reason`.
- **O(jobs+targets) job indexing in `validate`** — the action scanned all `_jobs` for every target
  (O(targets × jobs)) while holding `_jobs_lock` inside the async handler; on a full-workspace run
  that synchronous scan blocked the event loop long enough that concurrent `validate_status` RPCs
  exceeded the MCP client's 300s timeout (the "validator unresponsive" / session-recycle). Now builds
  a per-repo latest-job index once so status stays responsive.
- **`merged` derived from ahead-count** — drops the failing `--is-ancestor` probe that logged
  "Command failed" on every unmerged worktree (`ahead==0` is exactly the is-ancestor result; `merged`
  stays False when the count call fails, which is safer).

## [1.3.55] - 2026-04-29

### Added
- Initial release
