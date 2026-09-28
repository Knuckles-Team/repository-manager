# EH-483 — Repository-manager connector boundary migration

**Owner:** repository-manager for its own connector and package imports. **Delivery state:** BUILT source carrier, not verified landed or accepted. **Train:** T5. This is the RM slice of a wider connector fleet migration; the local contract is complete here.

## Outcome

The repository-manager agent continues to expose its existing Git, worktree, release, and lane tools while consuming agent orchestration through the public API it actually needs. Governance remains in repository-manager under EH-512. A contributor can build and test this package from a fresh clone without an existing fleet or live graph.

## Requirements and acceptance

| ID | Contract | Acceptance |
|---|---|---|
| EH-483-01 | Inventory every production and test import from the agent orchestration package; classify each as orchestration, connector transport, governance, or obsolete. | Machine-readable import inventory has one disposition for every matching import; no unclassified dynamic import. |
| EH-483-02 | Replace connector transport imports with the public SDK API and keep orchestration imports only at the orchestrator boundary. | Import and contract tests execute the RM tool entry points without reaching retired private modules. |
| EH-483-03 | Move/keep development governance imports on EH-512's repository-manager authority; do not route back through orchestration. | Lane, queue, reservation, and safe-commit tests use RM implementations. |
| EH-483-04 | Preserve public MCP/CLI action names, typed arguments, validation, and error categories during migration. | Golden tool schema and representative success/refusal responses remain compatible or carry explicit versioned migration. |
| EH-483-05 | Remove dead adapters, duplicate publishers, and fallback imports after the new path passes. | Source scan finds no active retired import; package build and smoke tests pass. |

## Failures and boundaries

Missing SDK or orchestration dependency produces a clear startup or call error for the affected capability, not silent substitution with old private modules. Git operation authority is RM; connector transport is SDK; orchestration owns only agent decisions. This spec does not certify the other connector repositories. Record exact main commit, import inventory, package artifact, and CI results before LANDED/ACCEPTED; the existence of a branch or spec is insufficient.
