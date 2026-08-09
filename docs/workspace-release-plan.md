# Workspace release-plan checkpoint 1

`repository_manager.development.workspace_release` is the pure C-11 planning
boundary for the workspace release DAG. It records canonical repository and
package identities, versions, dependency floors, edge source/confidence, immutable
parallel groups, and the digestable release-plan shape. A repository identity
is a workspace-relative path (`repo:agent-packages/example`), never a URL,
directory basename, or display name. A package identity includes that full
repository identity, ecosystem, and normalized package name, so equal package
basenames in two repositories cannot silently collapse into one owner.

`workspace_metadata` reads only declarative `pyproject.toml`, `Cargo.toml`, and
`package.json` files. Before parsing, each file is checked for regular-file
status, symlinks, and a byte limit. Parsed values are checked for nesting,
string, collection, package, and dependency bounds. JSON duplicate keys,
unsupported overlay fields, static-version conflicts, unsupported floor syntax,
ambiguous package owners, missing projects/packages, duplicate edges, and cycles
are refusal diagnostics. No build backend, package manager, subprocess, network
client, or project code is invoked.

Path metadata and overlays are read-only inputs. `OverlayInput` is an explicit,
strict in-memory schema for edges and version sources that cannot safely be
inferred from package metadata. Existing `maintenance.phases` declarations are
available through `phase_manifest_from_mapping` as an immutable compatibility
view; their historical bare references are retained for later shadow comparison
and are never used as canonical graph keys.

Checkpoint 1 intentionally does not rewrite floors, execute validation/build/
landing/push stages, create WorkItems, edit `workspace.yml`, or wire MCP/CLI
surfaces. Those effects belong to later RMDD-18 checkpoints and their owning
integration lanes.
