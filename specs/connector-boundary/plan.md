# RM-CONNECTOR — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The migration is done in the code: the tool registrars in `repository_manager/mcp_tools/` and the server in `repository_manager/mcp_server.py` use only public `scion_sdk` modules, and governance is hosted in `repository_manager/governance/`. What remains is proof that stays true: a machine-readable import inventory and a package-wide retired-name scan, both planned. Reuse candidates found by searching the code first: `scripts/check_import_safety.py` (an `ast` walk over the package), `tests/governance/test_no_parallel_lane_authority.py` (the retired-import scan) and `tests/test_adapter_extraction.py` (the tool catalog golden). Do not add a local copy of SDK types or governance helpers.

```text
repository-manager-mcp   pyproject.toml:53
  -> mcp_server           repository_manager/mcp_server.py:1002
  -> get_mcp_instance     mcp_server.py:973
  -> create_mcp_server / register_tool_surface   scion_sdk.mcp
  -> mcp_tools/*.py registrars -> LaneRegistry, WorktreeManager, merge_queue, concept_actions
```

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: `scion-sdk[mcp]` (`pyproject.toml:12`), fastmcp, pydantic.
- **Storage**: N/A.
- **Testing**: `pytest` with fake MCP and SDK clients and temporary Git repositories.
- **Target Platform**: Linux, macOS and Windows.
- **Project Type**: Connector.
- **Performance Goals**: N/A.
- **Constraints**: keep public action schemas stable or version them; preserve typed failure codes and reject absent credentials before effects; start with only declared dependencies installed; log capability names and error classes, never credentials or private paths.
- **Scale/Scope**: 6 rows, 2 user stories, 5 `FR-` labels and 2 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. SDK for transport, this repository for governance, no local copies.
- **II. Complete specification before construction**: pass. Each row names its caller chain, shape and test; the two unbuilt checks are marked planned.
- **III. Observable integration and evidence**: pass. Tests start the real server and scan real modules.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
repository_manager/mcp_server.py                    existing: get_mcp_instance
repository_manager/mcp_tools/                       existing: tool registrars over scion_sdk.mcp
repository_manager/governance/                      existing: governance authority
scripts/check_import_safety.py                      existing: ast import walk
scripts/check_import_inventory.py                   new: import inventory (planned)
scripts/import_inventory.json                       new: one record per import (planned)
tests/governance/test_no_parallel_lane_authority.py existing: retired-import scan
tests/test_adapter_extraction.py                    existing: catalog golden
tests/test_import_inventory.py                      new (planned)
tests/test_retired_names.py                         new (planned)
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The class names and file location of the import inventory are an owner decision recorded in the pull request.

## Phase 1: Design

1. **Dependency order.** Governance first (`RM-GOVERNANCE-R001`), connector transport second, orchestration last. At each cut, run package import and the tool catalog golden.
2. **Inventory.** One record per import with module, symbol, caller and class; a dynamic import with a non-literal name is a failure.
3. **Direction.** Reject an import cycle from repository-manager through orchestration back to repository-manager with a dependency-direction test.
4. **Compatibility.** Keep public action schemas stable or version them and publish a migration note in this repository; delete a compatibility path when no caller remains.
5. **Isolation.** Test startup with only declared dependencies installed; a private sibling checkout must not mask missing package metadata.

Data entities: none beyond the inventory records.

## Phase 2: Task approach

`tasks.md` holds one task per row in dependency order: governance, transport, the first connector row, the inventory, the contract golden and the removal scan.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests/test_mcp_registration.py tests/test_adapter_extraction.py tests/governance/test_no_parallel_lane_authority.py` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit`, `vulture` and `check-import-safety` pass without a suppression.
- jscpd and dupehound are not configured repository hooks; report the tool gap.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

`repository-manager-mcp` (`pyproject.toml:53`), `mcp_server` (`repository_manager/mcp_server.py:1002`), `get_mcp_instance` (:973), `create_mcp_server` (:977) and `register_tool_surface` (:988) from `scion_sdk.mcp`, then `register_split_action_tools` over `MCP_TOOL_REGISTRY` (:994), whose registrars call the governance, worktree, queue and reservation services. Effects: Git worktrees and refs, lane and queue records. Obsolete paths to delete: none remain; the retired governance module is imported by no production module.

## Cross-repo seams

none. The connector transport is the public `scion_sdk` API; the owner row of that surface in the SDK repository is an owner decision and is cited here once defined.
