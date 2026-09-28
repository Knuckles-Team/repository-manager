# RF-016 architecture and implementation plan

## Existing wiring to reuse

`repository_manager/workspace_manifest.py` validates and selects portable repository entries. `repository_manager/dependency_readiness.py::check_phase_direction` already checks manifest phase direction. `repository_manager/development/workspace_release.py` defines `DependencyGraph`, `WorkspaceReleasePlan`, and digest rules; `workspace_release_plan.py::build_frozen_release_plan` builds the decision artifact. `repository_manager/merge_queue_runner.py::run_phased_push` and `repository_manager/mcp_tools/git.py` are execution entry points. Extend those seams; do not add a second planner or hidden release order.

## Data flow and authority

`manifest + source observations → bounded census → classified graph → frozen plan/digest → admission → staged executor → immutable stage receipts → downstream readiness`. Manifest is a portable input; on-disk Git repositories and hosted artifacts are execution evidence. Stage receipts bind source tree, plan digest, gate result, artifact digest, and remote result. The executor verifies a plan again immediately before each effect; a retry reuses the same idempotency key and reconciles remote state before repeating it.

## Design decisions

- Model repository and package identity separately; dependency edges carry ecosystem, group, confidence, and source reference. Preserve both declared and observed edges until a reviewed decision resolves disagreement.
- Sort canonical payloads before hashing; use one digest serializer for CLI, MCP, and Git routes.
- Scope a requested subset by transitive prerequisites and source universe. Recompute on changed manifests or source commits; reject a stale frozen plan.
- Keep all live routes on the old executor until the new admission path passes identical conformance tests. Switch the route registry atomically.
- Expose structured diagnostics for unsupported formats, unknown aliases, cycle members, missing evidence, partial publish, and rollback uncertainty. Never convert these into an empty success plan.

## Contributor setup

Clone this public repository, install Python from `pyproject.toml` with `uv sync --extra test`, and run `uv run pytest tests/test_workspace_release_dag.py tests/test_workspace_release_plan.py tests/test_phase_direction.py tests/test_phased_push.py`. The fixtures under `tests/fixtures/workspace_release/` provide a small independent fleet. Inspect `repository_manager/workspace.yml` as a portable seed example; replace its repository URLs and selectors for personal deployments. No running graph service or configured forge token is needed for fixture tests.

## Compatibility and security

Existing manifest selectors remain readable during migration. Reject traversal, symlinks out of the selected root, URL-shaped IDs, and untrusted artifact evidence before executing commands. Keep credentials in user environment or forge adapters, never in the plan or receipt. The public plan format needs a version and a migration test before changing fields.
