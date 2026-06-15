# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
