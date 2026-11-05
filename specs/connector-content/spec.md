# CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT — The connector serves its packaged content

Component: Connector content and served tools

Owner: repository-manager. Cross-repo IDs: `SCION-CONNECTOR-CONTROL-R071`, `SCION-CONNECTOR-CONTROL-R073`, `SCION-SOURCE-INGEST-R004.15`, `SCION-REPOSITORY-TRANSPORT-R001`, `SCION-REPOSITORY-TRANSPORT-R002`, `MEDULLA-SDLC-R012`, `MEDULLA-SCM-R004`.

## Outcome

The `repository-manager-mcp` server serves its packaged skills, prompts and ontology to any client with no wiring of its own, and serves the repository index and hydration tools and the graph-backed code reads that the fleet needs from this connector.

## Scope

Owner: `repository-manager`, one external system: the Git repositories of the registered workspace. The code is `repository_manager/mcp_server.py` (server build), `repository_manager/mcp_tools/` (tool registrars) and `pyproject.toml` (connector identity). The SDK owns state, paging, receipts and the transport; the engine owns the graph. Non-goals: the connector holds no engine call of its own and no second paging or state implementation.

Why a spec: the connector's content and served tools are owned here and nowhere else. Search: `git grep -n "CONNECTOR-CONTENT" -- specs` finds only this spec.

## User Scenarios & Testing

### User Story 1 - A client lists the packaged content (Priority: P1)

The user is any MCP client. The outcome is the skills, ontology and prompts of the package without extra configuration.

**Acceptance scenarios**

1. **Given** the package is installed, **When** a client calls `resources/list`, **Then** the result includes `skill://` and `ontology://repository-manager/` entries.
2. **Given** the package carries a prompt file, **When** a client calls `prompts/list`, **Then** the prompt is listed.

Independent test: the bound test of `CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R001`.

### User Story 2 - The SDK indexes and hydrates a repository through the connector (Priority: P2)

The user is the connector worker or the arbora fleet caller. The outcome is changed-file index pages and a hydrated revision with a receipt.

**Acceptance scenarios**

1. **Given** a fixture repository with two edited files and a previous index state, **When** the worker calls `index_repository`, **Then** the page lists those two files only.
2. **Given** a registered repository and a resolvable ref, **When** a caller calls `rm_repository_hydrate`, **Then** the receipt names the revision, the batch count and the manifest digest.
3. **Given** an unregistered repository or an unresolvable ref, **When** a caller calls `rm_repository_hydrate`, **Then** the tool refuses with `REPOSITORY_UNKNOWN` or `REF_UNRESOLVED` and writes nothing.

Independent test: the planned tests of `CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R002` and `CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R003`.

### User Story 3 - Code reads come from the graph (Priority: P3)

The user is an agent reading a repository's code. The outcome is a read from the engine, not from a local clone, when the engine holds the repository.

**Acceptance scenarios**

1. **Given** the engine holds the repository, **When** the agent reads a tree or blame, **Then** the answer comes from the engine and no local clone is touched.
2. **Given** the engine does not hold the repository, **When** the agent reads a tree, **Then** the registered clone answers.

Independent test: the planned test of `CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R004`.

## Requirements

- **FR-001**: The server lists the packaged skills, ontology and prompts (`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R001`).
- **FR-002**: The index tool returns changed files only for a previous state (`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R002`).
- **FR-003**: The hydrate tool indexes a revision through the SDK transport (`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R003`).
- **FR-004**: Code reads prefer the graph and fall back to a local clone only for an absent repository (`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R004`).

## Success Criteria

- **SC-001**: A fresh server lists every packaged skill and ontology resource with no configuration (`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R001`).
- **SC-002**: A second index call with unchanged files returns no pages (`CONN-REPOSITORY-MANAGER-CONNECTOR-CONTENT-R002`).

## Edge Cases

- The package ships no prompt file: the prompt listing is empty and the test asserts only files that exist.
- A ref resolves to a revision whose tree is empty: the receipt reports zero batches.
- The engine is unreachable during a read: the tool reports the failure and does not fall back silently to a stale clone.

## Assumptions

- The SDK default content registration and the connector identity entry point stay as declared by `SCION-CONNECTOR-CONTROL-R071` and `SCION-CONNECTOR-CONTROL-R073`.
- The caller supplies the engine client; the connector never holds engine credentials.
