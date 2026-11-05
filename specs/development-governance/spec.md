# RM-GOVERNANCE — Repository development governance

Component: Development governance

**Owner:** repository-manager. See [requirements.md](requirements.md) for the definition of every requirement ID. The related connector migration [`RM-CONNECTOR-R001`](../connector-boundary/spec.md) consumes this authority before its repository-manager half is accepted.

## Outcome

A contributor can open an isolated branch, reserve shared concepts, run validation, commit only reviewed paths, enter a merge queue, and release the branch without corrupting another contributor's work. A maintainer can inspect durable lane and queue state, recover after process loss, and prune only a branch proven safe. The tools run against contributor-owned Git repositories; a central fleet and live graph are optional integrations.

## Scope

Owner: `repository-manager`. The code is `repository_manager/governance/` (lanes, lane guard, concept reservation), `repository_manager/lane_registry.py`, `repository_manager/native_lane_authority.py`, `repository_manager/worktree.py`, `repository_manager/safe_commit.py`, `repository_manager/merge_queue.py`, `repository_manager/merge_queue_runner.py`, `repository_manager/concept_actions.py` and `repository_manager/gates.py`. This spec governs development activity and Git bytes. Agent orchestration may request work but does not mutate Git directly. A graph service can index receipts and is not the source of Git truth. Inputs that name paths outside the allowed root, a mismatched repository identity, missing authorization, a stale fence or an uncertain merge state fail closed.

Why a spec: no other spec owns lane lifecycle, worktree identity, queue admission and concept reservation. Search: `git grep -n "lane" -- specs` finds only this spec.

## User Scenarios & Testing

### User Story 1 - Work in an isolated branch and commit reviewed paths (Priority: P1)

The user is a contributor. The outcome is a real worktree, a gated commit of exactly the reviewed paths and a queue entry, with no change to another contributor's checkout.

**Acceptance scenarios**

1. **Given** a repository, **When** the contributor allocates a lane and adds a worktree, **Then** the lane record holds repository, base, owner, lease and allowlist, and `core.bare` stays false.
2. **Given** a lane with an unrelated untracked file, **When** the contributor commits two reviewed paths, **Then** only those paths are committed and the tree and gate identity are recorded.
3. **Given** a red gate or a changed tree, **When** the contributor enqueues the commit, **Then** admission is refused.

Independent test: the bound tests of `RM-GOVERNANCE-R025`, `RM-GOVERNANCE-R026` and `RM-GOVERNANCE-R027`.

### User Story 2 - Promote through the queue and recover (Priority: P1)

The user is a maintainer. The outcome is a merge result that is never reported falsely and a candidate that survives a crash.

**Acceptance scenarios**

1. **Given** a clean candidate, **When** the queue drains, **Then** the base and validation identity are rechecked and the outcome is recorded.
2. **Given** a stale base, a conflict or a killed drain, **When** the queue runs, **Then** the candidate is preserved for recovery and no merge is reported.
3. **Given** a dirty or unmerged branch, **When** cleanup runs, **Then** it is refused and a merged branch removes only its own checkout.

Independent test: the bound tests of `RM-GOVERNANCE-R028` and `RM-GOVERNANCE-R029`.

### User Story 3 - Reserve concepts and guard the canonical checkout (Priority: P2)

The user is a contributor in a consumer repository. The outcome is a typed conflict for a duplicate claim and a refused edit of the canonical checkout.

**Acceptance scenarios**

1. **Given** a reservation held by one actor, **When** another actor claims it, **Then** a typed conflict names a reconciliation path.
2. **Given** a consumer repository with the `lane-guard` hook, **When** a commit edits the canonical checkout, **Then** the hook refuses it.

Independent test: the bound tests of `RM-GOVERNANCE-R030` and `RM-GOVERNANCE-R031`.

### User Story 4 - Keep the gates honest (Priority: P3)

The user is a maintainer. The outcome is quality gates that fail closed and a heavy gate that runs on a leased slot.

**Acceptance scenarios**

1. **Given** one red hook among green hooks, **When** the gate stage runs, **Then** the stage fails and every hook's verdict is visible.
2. **Given** every inventory host at capacity, **When** a heavy gate starts, **Then** it refuses with a typed reason.

Independent test: the bound tests of `RM-GOVERNANCE-R019` and the planned tests of `RM-GOVERNANCE-R023`.

## Requirements

- **FR-001**: One authority owns lane lifecycle, worktree identity, queue admission and concept reservations (`RM-GOVERNANCE-R001`, `RM-GOVERNANCE-R024`).
- **FR-002**: Lanes carry ownership, lease and lifecycle and reject stale tokens (`RM-GOVERNANCE-R025`).
- **FR-003**: Worktrees never corrupt shared Git configuration (`RM-GOVERNANCE-R026`).
- **FR-004**: Commits stage reviewed paths only and bind gate identity (`RM-GOVERNANCE-R027`).
- **FR-005**: Promotion rechecks identity and never reports a false merge (`RM-GOVERNANCE-R028`).
- **FR-006**: Cleanup proves containment first (`RM-GOVERNANCE-R029`, `RM-GOVERNANCE-R003`).
- **FR-007**: Concept reservations are scoped and not silently reusable (`RM-GOVERNANCE-R030`).
- **FR-008**: The lane guard is exported as a consumer hook (`RM-GOVERNANCE-R031`).
- **FR-009**: The package runs on Windows (`RM-GOVERNANCE-R032`).
- **FR-010**: Quality gates fail closed and pass together (`RM-GOVERNANCE-R019`, `RM-GOVERNANCE-R022`).
- **FR-012**: The generated code-enhancement report uses the standard hyphenated name (`RM-GOVERNANCE-R033`).
- **FR-011**: Heavy gates run on a leased inventory host slot (`RM-GOVERNANCE-R023`).

## Success Criteria

- **SC-001**: No second Git, lane or reservation authority exists outside this package (`RM-GOVERNANCE-R024`).
- **SC-002**: Every refusal fixture leaves lane, queue and Git state unchanged (`RM-GOVERNANCE-R025`, `RM-GOVERNANCE-R028`).
- **SC-003**: No unmerged or dirty checkout is deleted by default (`RM-GOVERNANCE-R029`).

## Edge Cases

- Two lanes allocate the same branch at once: one wins and the other gets a typed conflict.
- A lease holder dies: the lease is reclaimed after its time-to-live and a cross-host expiry is not reclaimed locally.
- A hook runs from a linked worktree outside the workspace tree: paths still resolve.
- A candidate's base moves while it waits: promotion rechecks and preserves it.

## Assumptions

- The public path uses local Git repositories, fake authority adapters and temporary directories; no operator inventory is required.
- Git remains the authority for refs and bytes.
