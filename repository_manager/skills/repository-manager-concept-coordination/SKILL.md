---
name: repository-manager-concept-coordination
skill_type: skill
description: >-
  Reserve, inspect, release, and materialize a CONCEPT:ID claim, and verify a
  candidate's or generation's introduced concept markers against those claims, via
  the repository-manager `rm_concepts` MCP tool / `--concepts` CLI family. Routes
  agents away from hand-writing a `CONCEPT:` marker without first reserving its id,
  which is exactly how two concurrent lanes collide on the same identifier. Use
  before writing any new `CONCEPT:` marker, or to verify a candidate/generation's
  markers before it lands. ⚠ On `main` as of this lane, every mutating action
  refuses honestly with `ConceptAuthorityUnavailable` — read the fail-closed section
  before assuming this tool allocates anything. Do NOT use for the lane/worktree
  lifecycle itself (repository-manager-lane-lifecycle) or for landing a branch
  (repository-manager-merge-and-reconcile).
license: MIT
tags: [repository-manager, concepts, reservation, verification, mcp]
metadata:
  author: Genius
  version: '1.0.0'
---
# Repository Manager — Concept Coordination

`rm_concepts` is repository-manager's thin, MCP/CLI-neutral consumer of RMDD-16's
central concept-id claim authority (`agent_utilities.governance.
concept_reservation`). Repository-manager **never allocates a concept id itself** —
every action here is a claim/verification call against an injected authority port.

## ⚠ Read this before anything else: the authority is unavailable on `main` today
As of this lane (verified 2026-08-10 against `agent-packages/repository-manager`
`main` at `fdce825`), RMDD-16's authority module is **not an ancestor of
agent-utilities `main`**. Every mutating `rm_concepts` action —
`reserve`/`get`/`list`/`release`/`materialize`/`verify_candidate`/`verify_generation`
— therefore refuses with a named `ConceptAuthorityUnavailable` refusal
(`error_code: "dependency_blocked"` from the shared `RefusalCode`/`FailureClass`
vocabulary), preserving the real `ImportError` as its cause. This is not a bug and
not this skill's cue to route around it:

- **Never** substitute a local, in-memory, or fixture allocator for the real
  authority to "unblock" a demo — `repository_manager.concept_actions.
  build_default_concept_authority()` is a deliberate best-effort probe that returns
  `None` (never a fabricated authority) whenever the module is absent, exposes no
  documented live-construction entrypoint, or construction itself raises.
- **Report the refusal truthfully** — `{"ok": false, "refused": "...", "error_code":
  "dependency_blocked", "unreachable": ...}` — rather than describing concept
  coordination as working.
- This will change the moment RMDD-16's authority lands on agent-utilities `main`;
  re-verify this section against the live server before trusting it stale.

## When to use
- Before writing a new `CONCEPT:` marker in code/docs, so two sessions never
  collide on the same identifier (`reserve`).
- Checking or listing existing claims (`get` / `list`).
- Releasing a claim that was never actually used (`release`).
- Promoting a claim from reserved to in-use (`materialize`).
- Verifying one candidate's or one sealed generation's introduced `CONCEPT:` markers
  against reserved claims, including cross-candidate collisions
  (`verify_candidate` / `verify_generation`).
- A read-only drift report across the authority/fragment/view/source layers
  (`reconcile`).

## When NOT to use
- Opening/checking/finishing a lane → `repository-manager-lane-lifecycle`.
- Landing a branch or resolving a merge conflict →
  `repository-manager-merge-and-reconcile`.
- The generation/candidate lifecycle beyond marker verification →
  `repository-manager-candidate-certification`.

## Tools & actions
| Condensed tool | Actions |
|----------------|---------|
| `rm_concepts` | `reserve`, `list`, `get`, `release`, `materialize`, `verify_candidate`, `verify_generation`, `reconcile` |

CLI: `repository-manager --concepts {reserve,list,get,release,materialize,verify_candidate,verify_generation,reconcile} --concepts-repo-root <path> --concepts-tenant-ref <ref> --concepts-lane-ref <ref> --concepts-params-json '<json>'`.

### Required top-level fields (every action)
- `repo_root` — repository working tree root. MCP: required, no default. CLI:
  `--concepts-repo-root`, defaults to `.`.
- `tenant_ref` — authenticated tenant scope. MCP: required. CLI:
  `--concepts-tenant-ref`, defaults to `""`.
- `lane_ref` — lane/worktree identity for fragment provenance. MCP: required. CLI:
  `--concepts-lane-ref`, defaults to `""`.

### Per-action fields (pass via `--concepts-params-json` on the CLI)
| Action | Required | Optional |
|---|---|---|
| `reserve` | `concept_id`, `namespace`, `repository_ref`, `owner_ref`, `request_key_ref`, `purpose` | `design_ref`, `branch`, `base_sha`, `workitem_ref`, `run_trace_ref`, `provenance_refs` (defaults to a 7-day reservation window) |
| `get` | `reservation_id` | — |
| `list` | — | `namespace`, `state`, `concept_prefix`, `limit` (default 1000), `cursor` |
| `release` | `reservation_id`, `owner_ref`, `expected_fence` | — |
| `materialize` | `reservation_id`, `owner_ref`, `expected_fence` | — |
| `verify_candidate` | `candidate` (a serialized `Candidate` contract payload), `repo_path` | — |
| `verify_generation` | `generation` (a serialized `Generation` payload), `candidates` (list of `Candidate` payloads), `repo_path` | — |
| `reconcile` | — | `source_tree_ish` (default `"HEAD"`) |

`owner_ref` + `expected_fence` on `release`/`materialize` are an optimistic-concurrency
pair: the fence must match the claim's current fence, so a stale caller is refused
rather than silently clobbering a concurrent update.

## Recipes
Reserve a concept id before writing its marker (MCP form):
```
rm_concepts(action="reserve", repo_root=".", tenant_ref="<tenant>", lane_ref="<lane>",
            concept_id="RM-EXAMPLE.thing", namespace="RM-EXAMPLE",
            repository_ref="repository-manager", owner_ref="<you>",
            request_key_ref="<idempotency-key>", purpose="short reason")
```
Same, over the CLI:
```
repository-manager --concepts reserve --concepts-repo-root . \
  --concepts-tenant-ref <tenant> --concepts-lane-ref <lane> \
  --concepts-params-json '{"concept_id":"RM-EXAMPLE.thing","namespace":"RM-EXAMPLE","repository_ref":"repository-manager","owner_ref":"<you>","request_key_ref":"<key>","purpose":"short reason"}'
```
Verify one candidate's introduced markers before it lands:
```
rm_concepts(action="verify_candidate", repo_root=".", tenant_ref="<tenant>", lane_ref="<lane>",
            candidate=<Candidate payload>, repo_path="/path/to/repo")
```
Read-only drift report:
```
rm_concepts(action="reconcile", repo_root=".", tenant_ref="<tenant>", lane_ref="<lane>")
```

## Result shape
Every action returns `{"ok": true, ...}` on success or `{"ok": false, "refused":
"...", "error_code": "...", "unreachable": ...}` on a named refusal — a
`ConceptCoordinationError` never propagates as a bare traceback to either adapter. A
missing/malformed field (e.g. a required kwarg omitted) is reported as
`error_code: "invalid_request"`, distinct from `"dependency_blocked"` (the
authority-unavailable case above).

## Gotchas
- Every mutating action refuses today (see the box above) — do not build a workflow
  that assumes `reserve` succeeds without checking `ok`.
- `list`/`reconcile` are read-only and do not depend on the authority in the same
  way as the mutating actions — still check `ok`.
- `verify_candidate`/`verify_generation` take **serialized contract payloads**
  (`Candidate`/`Generation`), not free-form dicts — an incompatible shape is an
  `invalid_request`, not a silent pass.
- Any tool's live action set is self-discoverable: `rm_concepts(action="list_actions")`.

## Related
- `repository-manager-development-lifecycle` — step 1 (plan) calls `reserve` before
  starting a lane that will introduce a new concept marker.
- `repository-manager-candidate-certification` — the generation-lifecycle context
  `verify_candidate`/`verify_generation` run inside.
- Mechanism: `repository_manager/concept_actions.py`,
  `repository_manager/concept_coordination/`, `repository_manager/mcp_tools/concepts.py`,
  `repository_manager/cli_commands/concepts.py`.
