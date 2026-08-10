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
freezes the complete known-project cross-project edge evidence and verifies that
selected membership, directional reasons, and witnesses agree with it; a digest
alone is not treated as authenticity. The closure digest and explanations are
independent of input project/edge iteration order.

`derive_phase_view` and `compare_legacy_phases` project a `LegacyPhaseManifest`
into the same canonical identity space without rewriting it. Canonical references
resolve directly; historical bare references resolve only for a unique basename
owner. Same-basename repositories produce an `AMBIGUOUS_PROJECT` diagnostic. The
read-only report contains derived/manual phase views, membership/order/bulk-flag
diagnostics, wait-time diagnostics, exact equality, and a deterministic SHA-256
report digest. Phase number, membership/order, bulk flags, and `wait_minutes` are
semantic; phase `name` is intentionally a display-only label and is not used to
claim equivalence. Legacy phase reference order is preserved for comparison,
while duplicate references are refused by the bounded manifest reader. Trailing
derived/manual phases are reported individually. Membership diagnostics carry a
bounded count, full-sequence digest, and prefix; diagnostic accumulation has a
small deterministic overflow summary. No comparator path executes code, invokes a
subprocess or network, or mutates a manifest.

Checkpoint 2 still does not rewrite floors, plan versions, execute stages, create
WorkItems, restart/resume, edit workspace manifests, or wire MCP/CLI surfaces.

## Version and floor previews (checkpoint 3)

`repository_manager.development.workspace_versions` consumes only the verified
`DependencyGraph` and `SelectedChangeClosure`, plus immutable site descriptors
from a declarative metadata reader. A site names its relative metadata file,
selector, representation (`python`, `rust`, or `node`), exact old literal, and
an explicit version or floor policy. The planner binds each resulting preview
to the owning project tree SHA, package identity, graph digest, selection
digest, and one stable plan digest. It never opens the file, invokes a build
backend, or writes the proposed new text.

Version transitions are explicit `major`, `minor`, `patch`, or `exact` policies;
stable three-component SemVer is required. Floor transitions are explicit
`range`, `compatible`, `caret`, `tilde`, or `exact` policies. The output keeps
dependency-first package batches, emits one preview per rewrite site, and
retains already-satisfied floors as explicit no-op evidence.

Synthetic preview (illustrative only; it is not an instruction to edit a
workspace file):

```json
{
  "project_id": "repo:services/consumer",
  "package": "repo:services/consumer::python:consumer",
  "dependency": "repo:services/library::python:library",
  "file_path": "pyproject.toml",
  "source_sha": "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
  "old_text": ">=1.4.0",
  "new_text": ">=1.5.0",
  "old_normalized": ">=1.4.0",
  "new_normalized": ">=1.5.0",
  "reason": "transitive_minimum",
  "witness": [
    "edge:repo:services/consumer::python:consumer->repo:services/library::python:library",
    "dependency-next:repo:services/library::python:library=1.5.0",
    "topological-batch:0"
  ],
  "graph_digest": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "selection_digest": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "plan_digest": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd"
}
```

The preview is evidence for a later mutation owner, not a mutation request.

## Frozen release plan and stage preview (checkpoint 4)

`repository_manager.development.workspace_release_plan` is the pure C-11 freeze
boundary for the later release/mutation lanes. `freeze_release_plan` consumes
only the verified `DependencyGraph`, `SelectedChangeClosure`, and checkpoint-3
`VersionPlan`, plus explicit bounded base/source/generation and profile inputs.
It copies canonical repository/package identities, selected dependency-first
groups, every selected project's immutable tree SHA, the version/floor preview
digests, profile digests, and an exact SHA-256 `plan_digest`. No checkout is
opened and no command, package manager, Git operation, WorkItem, or network
client is available from this module.

The resulting `StagePreview` records are a deterministic dependency DAG:

```text
validate(repo) -> bump(repo) -> local-land(repo) -> build(repo) -> package(repo)
                         \-- bump also waits for each selected upstream bump
```

Independent projects share a topological group and remain parallel. A stage
stores its immutable tree/base/generation/graph/selection/version-plan inputs,
the relevant version/floor preview digests, profile digests, deterministic
stage ID, deterministic input digest, and sorted upstream IDs. Failure behavior
is declarative (`block_dependents`); this checkpoint does not inspect or run a
stage. The optional push stages are a separate consent-gated extension and are
created only with a `PushConsentReference` containing both an opaque reference
and immutable digest. A boolean flag alone can never create a push stage.

`FrozenReleasePlan.validate()` recomputes nested evidence and the exact plan
preimage, while `validate_against(graph, selection)` additionally revalidates
the current graph, closure, and checkpoint-3 previews. These paths are intended
to reject dataclass/Pydantic copy/construct and `object.__new__` forgeries,
cleared or stale digests, changed source/tree/base/generation/profile/preview
fields, reordered dependencies, unknown stage dependencies, cycles, and graph
or selection drift. Inputs are exact builtin bounded containers and all refusal
messages are privacy-safe.

The corrective CP4 contract also freezes a `ReleaseDecisionContext`: normalized
target branch plus opaque name/digest references for the release profile,
candidate, certificate, immutable config/toolchain/preview-command and artifact
contract. Each stage carries the context digest and its resource profile,
retry-policy/count, and timeout policy/seconds; changing any one decision field
therefore changes the plan and stage identities. These are references only and
do not grant execution authority.

Source and base SHAs are independently required exact builtin strings; omitted
or falsey aliases are never substituted. Push flags are descriptive selectors,
not authorization: explicit `False` conflicts with consent, contradictory
aliases refuse before construction, and an accepted plan hashes exactly the
consent value it returns. A push stage is present only when its immutable
`PushConsentReference` is present.

Graph and closure provenance are snapshotted from exact bounded builtin
containers before sorting or hashing. `validate()` re-materializes that source
evidence and derives the complete required stage composition and dependencies,
so rehashing a semantically altered DAG does not make it valid. Malformed
containers and provider-data normalization errors are fixed privacy-safe
refusals; trusted `RuntimeError` failures remain visible to callers.

The snapshot also rebuilds the complete package inventory and dependency
topology from the immutable project/package records. Duplicate or orphan
records, unknown endpoints, inconsistent project edges/groups, and changed
metadata-edge provenance are refused even when a caller recomputes every
reported digest. Checkpoint-3 version/floor sites, previews, nested policies,
and omission-valued text are copied into exact immutable records before the
version planner is re-run against the snapshotted graph and closure. A changed
site, source SHA, floor/version text, or semantic preview cannot be authorized
by rehashing the nested and outer plans.

Target branches use a single local `refs/heads/...` representation after
bounded Git-branch validation; ref aliases, traversal, URL/scheme, controls,
and ambiguous branch syntax are refused without invoking Git. The CP3
validation boundary normalizes only documented malformed-data exceptions, so a
trusted planner `RuntimeError` remains observable to its caller.
