# RM-CONNECTOR-001 architecture

## Current wiring and migration

Start at `repository_manager/mcp_server.py`, `mcp_tools/`, `agent_server.py`, `development/`, and the package's import graph. The tool entry points already delegate Git and queue effects to `safe_commit.py`, `worktree.py`, `merge_queue.py`, and `lane_registry.py`; preserve those seams. Use the connector SDK only for connector contracts and transport, and use public orchestration APIs only where an agent decision is needed. Do not add a local copy of SDK types or governance helpers.

Create a bounded import inventory with module, symbol, caller, owner, proposed replacement, and test. Migrate by domain in dependency order: governance first ([`RM-GOVERNANCE-R001`](../development-governance/spec.md)), connector transport second, orchestration last. At each cut, run package import and tool schema goldens. Delete compatibility paths when no callers remain. Reject an import cycle from RM through orchestration back to RM with a dependency-direction test.

## Contributor setup

Clone this repository, install with `uv sync --extra test`, and run `uv run pytest tests/test_agent_integration.py tests/test_lane_registry.py tests/test_safe_commit.py`. Use fake MCP/SDK clients and temporary Git repositories for tests. A contributor may configure their own graph or hosted forge for optional manual testing, but none is required to validate the package. Do not read a machine-specific inventory or private registry to infer the contract.

## Compatibility and observability

Keep public action schemas stable or version them and publish a migration note in this repository. Preserve typed failure codes and reject absent credentials before effects. Test startup with only declared dependencies installed; a private sibling checkout must not mask missing package metadata. Log capability names and error classes, never credentials or private paths.
