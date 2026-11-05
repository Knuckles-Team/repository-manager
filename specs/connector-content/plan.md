# CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT — Implementation plan

Governing spec: [spec.md](spec.md). Rows: [requirements.md](requirements.md).

## Summary

The connector declares its identity once in `pyproject.toml` and builds its server with `create_mcp_server`, which serves the packaged skills, ontology and prompts. The index tool, the hydration tool and the graph-backed code reads are planned tool registrars that join `MCP_TOOL_REGISTRY` and delegate to the SDK repository transport. Reuse candidates found by searching the code first: `repository_manager/mcp_tools/registry.py` (the registry seam), `repository_manager/kg_ingest.py` (the existing ingest mapping), `repository_manager/workspace_manifest.py` (the repository registry), and the SDK modules `scion_sdk/repository/transport.py` and `scion_sdk/repository/local_git.py`.

## Technical Context

- **Language/Version**: Python 3.14 standard, floor 3.12 (`requires-python >=3.12,<3.15`).
- **Primary Dependencies**: `scion-sdk[mcp]` (server build, content, repository transport), fastmcp, pydantic.
- **Storage**: none; engine records are written by the SDK only.
- **Testing**: `pytest` with a fastmcp `Client` against the built server; the planned tests use fixture repositories and a recorded engine.
- **Target Platform**: Linux and macOS hosts running the MCP server over stdio or HTTP.
- **Project Type**: Connector.
- **Performance Goals**: N/A; paging bounds belong to the SDK.
- **Constraints**: the connector never calls the engine's index operation itself (`SCION-REPOSITORY-TRANSPORT-R002`); tools that write to the engine carry no read-only hint.
- **Scale/Scope**: 4 rows, 3 user stories, 4 `FR-` labels and 2 `SC-` labels.

## Constitution Check

- **I. Single owner and reusable architecture**: pass. The SDK owns state, paging and transport; the connector adds Git access only.
- **II. Complete specification before construction**: pass. Unbuilt rows are marked planned with exact attach points.
- **III. Observable integration and evidence**: pass. Tests enter through the built server.
- **IV. Quality and release discipline**: pass. Gates are listed under Build contract.
- **V. Amendment and status**: pass. Status is computed by the generator.

## Project Structure

```text
pyproject.toml                                      existing: scion_sdk.connector entry point, repository-manager-mcp script
repository_manager/mcp_server.py                    existing: get_mcp_instance builds the server with ConnectorContent
repository_manager/mcp_tools/registry.py            existing: MCP_TOOL_REGISTRY gains three entries (planned)
repository_manager/mcp_tools/repository_index.py    new: register_repository_index_tools (planned)
repository_manager/mcp_tools/repository_hydrate.py  new: register_repository_hydrate_tools (planned)
repository_manager/mcp_tools/code_reads.py          new: register_code_read_tools (planned)
repository_manager/kg_ingest.py                     existing: ingest_repositories moves to SourceIngest (planned)
tests/test_default_content_surfaces.py              existing
tests/test_index_repository_served.py               new (planned)
tests/test_repository_hydrate_served.py             new (planned)
tests/test_graph_reads.py                           new (planned)
```

## Phase 0: Research

No unknowns that block planning; no `research.md`. The open choices (the `RepositoryIndexState` type name in the SDK, the code-read tool names and whether the server drops its explicit `content=` argument) are owner decisions recorded in the pull request.

## Phase 1: Design

1. **Identity once.** The `scion_sdk.connector` entry point is the only declaration of the fleet name.
2. **Content is data.** Skills, ontology and prompts are package data; the server registers them from the package root.
3. **SDK for hydration.** `rm_repository_hydrate` builds a `LocalGitRepositoryProvider` and calls `index_repository_snapshot`; it never calls the engine index operation directly.
4. **Graph first for reads.** The engine answers when it holds the repository; the clone answers otherwise.
5. **Honest hints.** A tool that writes to the engine carries no read-only hint.

Data entities are SDK types (`RepositoryIndexReceipt`, `RepositoryRevision`); this spec adds no type.

## Phase 2: Task approach

`tasks.md` holds one task per leaf row, producer first: content (R001), index (R002), hydration (R003, depends on R002), graph reads (R004, depends on R003). The three registrars are independent files.

## Complexity Tracking

none

## Build contract

- `uv run pytest tests/test_default_content_surfaces.py` passes; the repository `pytest` hook runs `tests` with `-m "not slow and not integration and not native"`.
- The `complexity-staged` hook (`scripts/check_complexity_staged.py`) finds no new function over cyclomatic 10 or cognitive 15, and none regressed.
- `ruff-check`, `ruff-format`, `mypy`, `bandit` and `vulture` pass without a suppression; the `check-cli-help` hook still runs `mcp_server --help`.
- The connector standard hook (`connector-standard`) and the tool-schema fingerprint file `repository_manager/connectors/tool_schema_fingerprints.json` are updated with each new tool.
- Spec rows pass `spec-standard`, `plan-conformance` and `spec-decomposition` with zero findings.

## Production caller chain

The server starts at the console script `repository-manager-mcp` (`pyproject.toml:53`), `mcp_server` (`repository_manager/mcp_server.py:1002`), `get_mcp_instance` (:973), `create_mcp_server` with `ConnectorContent` (:981), `register_tool_surface` (:988) and `register_split_action_tools(mcp, list(MCP_TOOL_REGISTRY))` (:994). Planned tools attach as entries of `MCP_TOOL_REGISTRY` (`repository_manager/mcp_tools/registry.py:105`) and end at the SDK transport (`scion_sdk/repository/transport.py:96`) and the engine client of the caller. Obsolete paths to delete: none.

## Cross-repo seams

- `SCION-CONNECTOR-CONTROL-R071` and `SCION-CONNECTOR-CONTROL-R073` (identity and default content), provider: scion-sdk, consumer: this connector.
- `SCION-SOURCE-INGEST-R004.15` (the worker passes the index state to the connector tool), provider: this connector, consumer: scion-sdk.
- `SCION-REPOSITORY-TRANSPORT-R001` and `SCION-REPOSITORY-TRANSPORT-R002` (hydration through the SDK only), provider: scion-sdk, consumer: this connector.
