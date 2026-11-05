# RM-RELEASE — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The planner is built and tested as a library: a bounded census, a classified dependency graph, a frozen plan with one digest, and a fleet release gate that refuses while activation is closed. The dependency-floor rollout (`repository-manager rollout`, MCP `rm_rollout`) is built and wired to a CLI verb and an MCP tool. The staged executor and the atomic route switch are not built. Reuse candidates found by searching the code first: `repository_manager/workspace_manifest.py` (portable repository entries), `repository_manager/dependency_readiness.py` (`check_phase_direction`, `check_constraint`), `repository_manager/development/workspace_release.py` (`DependencyGraph`, `WorkspaceReleasePlan`, digest rules), `repository_manager/development/workspace_release_plan.py` (`build_frozen_release_plan`) and `repository_manager/rollout.py`. Extend those seams; do not add a second planner or a hidden release order.

```text
manifest + source observations
  -> capture_census / capture_complete_manifest_census   development/fleet_census.py:551, fleet_manifest.py:418
  -> build_dependency_graph                              development/workspace_release.py:1967
  -> build_frozen_release_plan                           development/workspace_release_plan.py:4513
  -> prepare_fleet_release                               development/fleet_release_gate.py:250
rollout CLI / MCP -> run_rollout_action                  rollout.py:962
```

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: the standard library plus pydantic; the system `git`, `gh`, `uv` and `pre-commit` binaries behind the injectable `Runner` of `rollout.py`.
- **Storage**: an append-only JSONL ledger under the XDG state directory for the rollout; none for the planner.
- **Testing**: `pytest` with fixtures under `tests/fixtures/workspace_release/`, temporary Git repositories and fake index, forge and runner adapters; no network.
- **Target Platform**: Linux, macOS and Windows (the rollout uses only `pathlib`, `subprocess` and `threading`).
- **Project Type**: Library with a CLI verb and an MCP tool.
- **Performance Goals**: N/A.
- **Constraints**: every refusal precedes any mutation; a plan is re-verified immediately before each effect; credentials stay in the user environment or forge adapters and never enter a plan or receipt.
- **Scale/Scope**: 9 rows, 3 user stories, 7 `FR-` labels and 3 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. One graph builder, one digest serializer and one route admission path.
- **II. Complete specification before construction**: pass. Each row names its caller chain, shape and test; unbuilt pieces are stated.
- **III. Observable integration and evidence**: pass. Tests cover refusal, retry and stale-plan cases.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
repository_manager/workspace_manifest.py                 existing: portable manifest selection
repository_manager/development/fleet_census.py           existing: capture_census, CensusReceipt
repository_manager/development/fleet_manifest.py         existing: manifest census and mirrors
repository_manager/development/fleet_source_universe.py  existing: independent source proof
repository_manager/development/workspace_release.py      existing: DependencyGraph, WorkspaceReleasePlan, plan_digest
repository_manager/development/workspace_release_plan.py existing: build_frozen_release_plan
repository_manager/development/fleet_release_gate.py     existing: activation gate, stage permits
repository_manager/development/fleet_evidence_provider.py existing: ReadOnlyFleetEvidenceProvider
repository_manager/dependency_readiness.py               existing: wheel rule, phase direction
repository_manager/rollout.py                            existing: Rollout, run_rollout_action
repository_manager/cli_commands/rollout.py               existing: run_rollout_cli
repository_manager/mcp_tools/rollout.py                  existing: rm_rollout
.github/workflows/connector.yml                          existing: tag-only publication
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The staged executor and the atomic route registry switch are unbuilt work, not unknowns.

## Phase 1: Design

1. **Data flow.** Manifest and source observations feed a bounded census, a classified graph, a frozen plan and digest, admission, a staged executor, immutable stage receipts and downstream readiness. The manifest is a portable input; on-disk Git repositories and hosted artifacts are execution evidence. The executor verifies a plan again immediately before each effect; a retry reuses the same idempotency key and reconciles remote state before repeating it.
2. **Identity.** Repository identity and package identity are modeled separately; dependency edges carry ecosystem, group, confidence and source reference. Declared and observed edges are preserved until a reviewed decision resolves a disagreement.
3. **One digest.** Canonical payloads are sorted before hashing and one serializer serves the CLI, MCP and Git routes.
4. **Scope and staleness.** A requested subset is scoped by transitive prerequisites and source universe; a changed manifest or source commit recomputes the plan and a stale frozen plan is rejected.
5. **Route switch.** Live routes stay on the old executor until the new admission path passes identical conformance tests; the route registry switches atomically.
6. **Diagnostics.** Unsupported formats, unknown aliases, cycle members, missing evidence, partial publish and rollback uncertainty are structured diagnostics and never an empty success plan.
7. **Security.** Traversal, symlinks out of the selected root, URL-shaped identifiers and untrusted artifact evidence are rejected before any command runs; the public plan format is versioned and changes need a migration test.
8. **Tag-only publication.** The repository calls the canonical connector pipeline, which publishes only for pushed `v*` tags with its frozen-source and conflicting-tag guard; source CI stays independent. The bumper creates the tag and its commit together after validation.
9. **Rollout.** `repository_manager/rollout.py` is the one fleet dependency-floor executor; nothing pushes to main. It plans from the forge, keeps a ledger keyed by (dependency, floor, repository) for resume, works in per-repository worktrees off `origin/<default>`, verifies the relock, uses the repository's bumpversion config, regenerates spec status, opens pull requests with `--auto --rebase` and tags in a separate idempotent step.

Data entities are the existing frozen plan, census, order and permit types.

## Phase 2: Task approach

`tasks.md` holds one task per leaf row, producer first: census, graph and digest, then execution gating, route admission and receipts; the tag-only publication and the rollout are independent and run in parallel.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests/test_workspace_release_dag.py tests/test_workspace_release_plan.py tests/test_phase_direction.py tests/test_rollout.py tests/test_fleet_release_gate.py` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit` and `vulture` pass without a suppression.
- jscpd and dupehound are not configured repository hooks; record that gap.
- The no-duplicate-authority check finds one graph builder and one route admission path.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

Planner: no CLI or MCP verb exists yet; the library chain is `capture_complete_manifest_census` (`repository_manager/development/fleet_manifest.py:418`), `build_dependency_graph` (`repository_manager/development/workspace_release.py:1967`), `build_frozen_release_plan` (`repository_manager/development/workspace_release_plan.py:4513`) and `prepare_fleet_release` (`repository_manager/development/fleet_release_gate.py:250`), reached through `ReadOnlyFleetEvidenceProvider` (`repository_manager/development/fleet_evidence_provider.py:33`). Legacy routes call `require_legacy_release_route` (`fleet_release_gate.py:38`). Rollout: `repository_manager/cli_commands/parser.py:46` to `run_rollout_cli` (`repository_manager/cli_commands/rollout.py:57`) and the MCP tool `rm_rollout` (`repository_manager/mcp_tools/rollout.py:23`) to `run_rollout_action` (`repository_manager/rollout.py:962`). Effects: pull requests and tags on the forge, a ledger file. Obsolete paths to delete: the legacy project-map and phase routes after the atomic route switch.

## Cross-repo seams

- `PIPE-RELEASE-R008` downstream readiness wheel rule, provider: pipelines, consumer: `RM-RELEASE-R005`, contract in `contracts/downstream-readiness.md`.
