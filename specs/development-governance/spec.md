# RM-GOVERNANCE-001 — Repository development governance

**Owner:** repository-manager. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for current delivery state. The related connector migration [`RM-CONNECTOR-R001`](../connector-boundary/spec.md) must consume this authority before its repository-manager half is accepted.

## Outcome and stories

A contributor can open an isolated branch, reserve shared concepts, run validation, commit only reviewed paths, enter a merge queue, and release the branch without corrupting another contributor's work. A maintainer can inspect durable lane and queue state, recover after process loss, and prune only a branch proven safe. The tools run against contributor-owned Git repositories; a central fleet and live graph are optional integrations.

## Functional requirements

| ID | Contract | Acceptance |
|---|---|---|
| RM-GOVERNANCE-01 | One repository-manager authority owns lane lifecycle, worktree identity, queue admission, and concept reservations. Other packages call it through bounded APIs. | Source search and import tests find no active parallel Git/lane authority. |
| RM-GOVERNANCE-02 | Allocate a lane with repository ID, base commit, owner, lease/fence, path allowlist, and lifecycle state. Enforce legal transitions and reject stale tokens. | Concurrent allocation conflict and stale heartbeat fail without changing state. |
| RM-GOVERNANCE-03 | Create a real Git worktree without modifying shared Git config; serialize operations that touch the common repository and keep per-lane state isolated. | Multiple worktrees remain usable; shared `core.bare` remains false. |
| RM-GOVERNANCE-04 | A commit stages only explicit reviewed paths, runs declared local gates, binds tree and gate identity, and enters the queue only from a clean, authorized lane. | An unrelated untracked file remains uncommitted; a red gate or changed tree refuses admission. |
| RM-GOVERNANCE-05 | Queue promotion rechecks base and validation identity, records outcome, and handles conflict or process death without reporting a false merge. | Stale base, conflicting patch, and timeout preserve the candidate for recovery. |
| RM-GOVERNANCE-06 | Cleanup proves merge containment or retains a durable recovery anchor; a dirty/unmerged lane is never deleted by default. | Unsafe prune returns a refusal; safe prune removes only its own worktree and merged branch. |
| RM-GOVERNANCE-07 | Concept reservations are scoped to repository and candidate identity, expire or release safely, and cannot be silently reused by a different actor. | Duplicate/stale claims fail with typed conflict and a reconciliation path. |

## Scope and boundaries

This spec governs development activity and Git bytes. Agent orchestration may request work but does not mutate Git directly. A graph service can index receipts but is not the source of Git truth. The public standalone path uses local Git repositories, fake authority adapters, and temporary directories; no operator inventory is required. Inputs that name paths outside the allowed root, mismatched repository identity, missing authorization, stale fence, or uncertain merge state fail closed.

## Trace and evidence

Stable IDs: RM-GOVERNANCE-R001 is this authority move; RM-CONNECTOR-R001 is the dependent repository-manager connector import migration. The latter is not covered merely by moving governance modules. Source import and a local commit are BUILDING/BUILT evidence only. Record exact main commit and CI run to mark LANDED; mark ACCEPTED only after the tests here, live entry point import checks, queue rehearsal, and quality gates pass at that commit.

Requirement IDs are defined in [requirements.md](requirements.md); delivery state per ID is in `status.json`.
