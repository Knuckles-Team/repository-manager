# RM-RELEASE — Dependency aware fleet release

**Owner:** repository-manager. **Delivery state:** BUILDING; source exists, fleet activation and acceptance are unverified. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for current delivery state.

## Outcome

A maintainer can release a selected set of repositories in an order derived from declared and observed dependency edges. A contributor can reproduce the planner with fixtures in a fresh checkout, without access to any operator infrastructure. The release refuses incomplete evidence, cycles, stale inputs, and partial publication before mutating a repository.

## Users and scenarios

1. A release operator selects repository IDs from a portable manifest and receives a frozen plan with deterministic stages, a digest, and a reason for each edge.
2. A contributor changes dependency metadata and sees the changed plan or an actionable diagnostic before any push.
3. A recovery operator inspects a failed run and sees which immutable stages completed, which did not, and whether retry is safe.

## Requirements

| ID | Contract | Acceptance |
|---|---|---|
| RM-RELEASE-R002 | The census identifies each selected repository, package, version source, dependency group, and production source observation for Python, Rust, and JavaScript/TypeScript. Missing or contradictory metadata is explicit. Documentation readiness derives expected agent identities from the caller's canonical manifest and rejects identity drift before generation. | A fixture with an omitted repository or source yields an incomplete receipt and no executable plan. |
| RM-RELEASE-R003 | Build a directed graph with classified runtime, build, development, optional, deployment, and frontend edges. Reject cycles in each declared release scope; never silently drop an edge to make the graph acyclic. | A two-node cycle and a cross-phase upward edge produce stable, typed diagnostics. |
| RM-RELEASE-R004 | Bind selected repository identities, source revisions, manifest and census digests, edge evidence, stage order, and version-floor rewrites into an immutable plan digest. | Reordering input data does not change the digest; changing a relevant source revision does. |
| RM-RELEASE-R005 | Execute setup, install, validation, build, publication, and downstream readiness in dependency order. Mutations require a complete, current plan and trusted publication evidence. | A missing artifact, failed CI check, or stale predecessor blocks dependents before their mutation. |
| RM-RELEASE-R006 | Git, CLI, and MCP release routes use the same admission and executor contract. A failed activation preserves the prior route with no partially switched entry point. | Each route yields the same stage and refusal result from a shared fixture. |
| RM-RELEASE-R009 | Roll a dependency floor across the fleet deterministically and resumably: select, plan into a ledger, apply one auto-merging PR per repository (floor, verified relock, patch bump, CHANGELOG, spec-status regen, pre-commit), track, and tag after merge. Replaces the retired phased bump/push. | `tests/test_rollout.py`: resume idempotency, unresolved lock fails the row, existing PR not duplicated, tag only after merge and floor met. |
| RM-RELEASE-R007 | Each stage records exact artifact and repository identity, outcome, retry key, and rollback or repair action. | Injected push failure gives a reconciliable partial state and never reports success. |

## Boundaries and failure behavior

repository-manager owns census, ordering, Git execution, and receipts. Package managers and forges remain external systems; adapters return evidence rather than becoming alternate planners. No feature requires a pre-existing local fleet, private registry, or live service. Offline fixtures cover all logic; optional live qualification may use a contributor's own repositories. Unknown identity, private path, unverified version, missing mirror, unsupported syntax, uncertain publication, and timeout fail closed with a diagnostic.

## Trace and evidence

This directory is the complete contribution contract for **RM-RELEASE-R001**. Related public repository contracts may use the same ID, but this spec stands alone. Source implementation is not acceptance. Mark **LANDED** only after the exact code commit reaches `main`; mark **ACCEPTED** only after the tests in [test-spec.md](test-spec.md), full applicable quality gates, and a checked-in release receipt have passed for that commit. Record commit, CI run, and artifact digest here when verified.

Requirement IDs are defined in [requirements.md](requirements.md); delivery state per ID is in `status.json`.
