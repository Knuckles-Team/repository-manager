# Workspace release-plan checkpoints 1–2

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
ambiguous package owners (including competing explicit overlay owners), missing
projects/packages, duplicate edge/rewrite provenance, and cycles
are refusal diagnostics. No build backend, package manager, subprocess, network
client, or project code is invoked.

Frozen plans require every edge and floor rewrite endpoint to be present in the
frozen package inventory. Their project groups and stage dependencies are
complete, duplicate-free, and compared with the deterministic dependency-first
topological order. Same-repository package edges remain in the package graph
without creating a project-level self-cycle. Node ranges keep package names
separate from exact/caret/tilde floors; Cargo bare versions use caret semantics.

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

## Selected-change closure and phase shadowing (checkpoint 2)

`repository_manager.development.workspace_selection` derives a frozen selected
subgraph from explicit canonical `changed_projects` and optional explicit
`selected_projects`. `InclusionMode.NONE`, `DIRECT`, and `TRANSITIVE` independently
control dependency/upstream and dependent/downstream closure. Every known project
gets a deterministic explanation: changed/explicit roots and traversal witnesses
for included projects, or an explicit excluded reason. Package edges, including
same-project package edges, are retained; only cross-project edges participate in
the project DAG. Groups are dependency-first and deterministic, so independent
projects remain parallel.

Unknown roots, duplicate or contradictory policy IDs, malformed graph edges, and
cycles fail closed before a closure is returned. Policy and closure collections,
references, explanations, and digests are bounded and immutable. The closure
digest and explanations are independent of input project/edge iteration order.

`derive_phase_view` and `compare_legacy_phases` project a `LegacyPhaseManifest` into the same canonical
identity space without rewriting it. Canonical references resolve directly;
historical bare references resolve only for a unique basename owner. Same-basename
repositories produce an `AMBIGUOUS_PROJECT` diagnostic. The read-only report
contains derived/manual phase views, membership/order/bulk-flag diagnostics,
exact equality, and a deterministic SHA-256 report digest. Legacy phase reference
order is preserved for comparison, while duplicate references are refused by the
bounded manifest reader. No comparator path executes code, invokes a subprocess or
network, or mutates a manifest.

Checkpoint 2 still does not rewrite floors, plan versions, execute stages, create
WorkItems, restart/resume, edit workspace manifests, or wire MCP/CLI surfaces.
