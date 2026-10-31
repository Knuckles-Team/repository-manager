# RM-RELEASE architecture and implementation plan

## Existing wiring to reuse

`repository_manager/workspace_manifest.py` validates and selects portable repository entries. `repository_manager/dependency_readiness.py::check_phase_direction` already checks manifest phase direction. `repository_manager/development/workspace_release.py` defines `DependencyGraph`, `WorkspaceReleasePlan`, and digest rules; `workspace_release_plan.py::build_frozen_release_plan` builds the decision artifact. `repository_manager/rollout.py` (CLI `repository-manager rollout`, MCP `rm_rollout`) is the fleet dependency-floor execution entry point. Extend those seams; do not add a second planner or hidden release order.

## Data flow and authority

`manifest + source observations → bounded census → classified graph → frozen plan/digest → admission → staged executor → immutable stage receipts → downstream readiness`. Manifest is a portable input; on-disk Git repositories and hosted artifacts are execution evidence. Stage receipts bind source tree, plan digest, gate result, artifact digest, and remote result. The executor verifies a plan again immediately before each effect; a retry reuses the same idempotency key and reconciles remote state before repeating it.

## Design decisions

- Model repository and package identity separately; dependency edges carry ecosystem, group, confidence, and source reference. Preserve both declared and observed edges until a reviewed decision resolves disagreement.
- Sort canonical payloads before hashing; use one digest serializer for CLI, MCP, and Git routes.
- Scope a requested subset by transitive prerequisites and source universe. Recompute on changed manifests or source commits; reject a stale frozen plan.
- Keep all live routes on the old executor until the new admission path passes identical conformance tests. Switch the route registry atomically.
- Expose structured diagnostics for unsupported formats, unknown aliases, cycle members, missing evidence, partial publish, and rollback uncertainty. Never convert these into an empty success plan.

## Contributor setup

Clone this public repository, install Python from `pyproject.toml` with `uv sync --extra test`, and run `uv run pytest tests/test_workspace_release_dag.py tests/test_workspace_release_plan.py tests/test_phase_direction.py tests/test_rollout.py`. The fixtures under `tests/fixtures/workspace_release/` provide a small independent fleet. Inspect `repository_manager/workspace.yml` as a portable seed example; replace its repository URLs and selectors for personal deployments. No running graph service or configured forge token is needed for fixture tests.

## Compatibility and security

Existing manifest selectors remain readable during migration. Reject traversal, symlinks out of the selected root, URL-shaped IDs, and untrusted artifact evidence before executing commands. Keep credentials in user environment or forge adapters, never in the plan or receipt. The public plan format needs a version and a migration test before changing fields.

## Amendments

2026-10-22: RM-RELEASE-R008 narrows package publication to bump2version-generated v-prefixed tags. Reuse the canonical Python pipeline with require-tag enabled; do not duplicate or bypass its frozen-source/conflicting-tag guard. Source CI remains independent. The next release uses the normal bumper and publishes its matching commit/tag together after validation; existing release tags remain immutable.

2026-10-30: RM-RELEASE-R009 retires phased bump/push. Root cause: `phased_bumpversion`/`phased_push` committed and pushed straight to `main`, but connectors now land only through PRs with auto-merge, required CI and owner-only merges; its `maintenance.phases` still named pre-rename packages (Epistemic Graph, agent-connector-sdk); and it bumped versions without ever raising a dependency floor or relocking, so it could not roll a dependency at all. `repository_manager/rollout.py` replaces it: forge-read planning, an XDG-state JSONL ledger keyed by (dependency, floor, repo) for resume, per-repo worktrees off `origin/<default>`, a verified relock, the repository's bumpversion config, `pipelines-hook spec-status --write` so the PR carries the current status schema, PRs with `--auto --rebase`, and a separate idempotent tag step. All process calls go through an injectable runner; only pathlib/subprocess/threading are used, so it runs on Windows.
