# RM-CONNECTOR — Repository-manager connector boundary migration

Component: Connector boundary

**Owner:** repository-manager for its own connector and package imports. See [requirements.md](requirements.md) for the definition of every requirement ID. This is the repository-manager slice of a broader connector migration across the fleet; the local contract is complete here.

## Outcome

The repository-manager connector continues to expose its Git, worktree, release and lane tools while consuming connector transport through the public scion-sdk API and orchestration through the public API it actually needs. Governance remains in repository-manager under [`RM-GOVERNANCE-R001`](../development-governance/spec.md). A contributor can build and test this package from a fresh clone without an existing fleet or live graph.

## Scope

Owner: `repository-manager`. The code is `repository_manager/mcp_server.py`, `repository_manager/mcp_tools/` and the package import graph; the dependency is `scion-sdk[mcp]` in `pyproject.toml`. Git operation authority is repository-manager, connector transport is the SDK, and orchestration owns only agent decisions. This spec does not certify the other connector repositories. A missing SDK or orchestration dependency produces a clear startup or call error for the affected capability, never a silent substitution with a private module.

Why a spec: no other spec owns this repository's import boundary. Search: `git grep -n "import" -- specs` finds only this spec and the connector-content spec, which owns served content.

## User Scenarios & Testing

### User Story 1 - Serve the tools through the public SDK (Priority: P1)

The user is an MCP client. The outcome is the same tool catalog as before the migration, served through the public SDK.

**Acceptance scenarios**

1. **Given** the package is installed with only its declared dependencies, **When** the server starts, **Then** every tool registers and the catalog digest equals the baseline.
2. **Given** a tool action name or argument changes, **When** the golden comparison runs, **Then** the change fails unless it carries a versioned migration.

Independent test: the bound tests of `RM-CONNECTOR-R001` and `RM-CONNECTOR-R005`.

### User Story 2 - Keep imports in their owner's package (Priority: P1)

The user is a maintainer. The outcome is proof that governance stays local and no retired module is imported.

**Acceptance scenarios**

1. **Given** the production modules, **When** the import scan runs, **Then** none imports the retired governance module.
2. **Given** the orchestration package, **When** the inventory runs, **Then** every import has exactly one disposition and none is dynamic.

Independent test: the bound tests of `RM-CONNECTOR-R003`, `RM-CONNECTOR-R004` and `RM-CONNECTOR-R006` and the planned test of `RM-CONNECTOR-R002`.

## Requirements

- **FR-001**: Connector transport is reached only through the public SDK (`RM-CONNECTOR-R001`, `RM-CONNECTOR-R003`).
- **FR-002**: Every orchestration import has one disposition (`RM-CONNECTOR-R002`).
- **FR-003**: Governance stays under repository-manager authority (`RM-CONNECTOR-R004`).
- **FR-004**: The public MCP and CLI contract is preserved (`RM-CONNECTOR-R005`).
- **FR-005**: Dead adapters and fallback imports are removed (`RM-CONNECTOR-R006`).

## Success Criteria

- **SC-001**: The tool catalog digest equals the baseline after any import change (`RM-CONNECTOR-R005`).
- **SC-002**: No production module imports a retired module (`RM-CONNECTOR-R006`).

## Edge Cases

- An optional SDK extra is missing: the affected capability reports a typed startup error and no private module substitutes.
- A sibling checkout of the SDK exists on the path: package metadata, not the checkout, decides what is installed.
- An import cycle from repository-manager through orchestration back to repository-manager is rejected by a dependency-direction test.

## Assumptions

- The public SDK API is the only supported connector transport.
- Tests use fake MCP and SDK clients and temporary Git repositories; no hosted forge is required.
