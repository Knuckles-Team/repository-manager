# RM-MATERIALIZE-001 — Governed proposal to Git materialization

**Owner:** repository-manager for Git effects. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for current delivery state. This spec also supports a graph service that stores approval and receipts, and an orchestration service that requests execution; their public contracts are supplementary to the complete RM contract below.

## Purpose and actors

An approved change proposal becomes an auditable Git candidate only through repository-manager. A reviewer approves a bounded patch against a specified base; an executor materializes it; an auditor verifies the exact commit, gates, queue outcome, and immutable receipt. A proposal can remain pending safely when any authority or remote effect is unavailable.

## Functional requirements

| ID | Contract | Acceptance |
|---|---|---|
| RM-MATERIALIZE-01 | Accept a versioned `ChangeProposal` with proposal ID, repository ID, exact base SHA, approved path allowlist, patch digest, decision identity, and expiry. Verify caller authorization and signed/verified approval before creating a worktree. | Missing, expired, altered, or wrong-repository approval causes no Git mutation. |
| RM-MATERIALIZE-02 | Resolve the repository from a portable registry and create an isolated real worktree at the approved base. Apply only approved paths; reject path traversal, symlink escapes, binary surprises, oversized patches, and ref drift. | A valid patch changes only allowed paths; each invalid patch leaves the canonical repository untouched. |
| RM-MATERIALIZE-03 | Run declared local validation against the resulting tree, stage an explicit path list, and create one commit with approved attribution and exact tree identity. | Gate failure, changed tree, or unreviewed file prevents commit and queue admission. |
| RM-MATERIALIZE-04 | Submit the candidate to the existing merge queue; never claim landing before the queue verifies base, gates, and merge result. | Conflict or red hosted CI produces pending/failed outcome, not success. |
| RM-MATERIALIZE-05 | Return an immutable `MaterializationReceipt/v1` bound to proposal, approval, base, resulting commit/tree, gate profile/results, queue result, and idempotency key. Reconcile uncertain delivery before retry. | Repeated same request returns/reconciles one candidate; altered replay is rejected. |
| RM-MATERIALIZE-06 | No caller has a direct-Git fallback. If repository-manager, authorization, or receipt delivery is unavailable, the request remains pending. | Fault injection cannot cause an ungoverned branch or a false resolved proposal. |

## Boundaries

Git is authoritative for source bytes and history; repository-manager owns worktrees, gates, commits, and merge queue submission. The external approval authority owns approval facts; repository-manager verifies a bounded proof and returns its own authenticated execution result. A graph index may store the receipt but cannot manufacture a Git commit. Implementations must not require a particular operator environment: local fake approval and temporary Git repository fixtures cover every rule. Real forge qualification is optional until merge acceptance.

## Trace and status

Stable ID **RM-MATERIALIZE-R001** covers this materialization boundary. Related public specs in other repositories may describe approval storage and orchestration, but all RM acceptance is here. Mark LANDED after exact source commit reaches main; mark ACCEPTED after negative tests, full gates, a queue rehearsal, and the receipt is independently verified against Git at that commit. A design document or draft receipt never counts as built or accepted.
