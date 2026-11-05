# RM-MATERIALIZE — Governed proposal to Git materialization

Component: Materialization

**Owner:** repository-manager, sole owner of the Git effects. See [requirements.md](requirements.md) for the definition of every requirement ID. A graph service that stores approval and receipts and an orchestration service that requests execution are consumers; their public contracts are supplementary to the complete contract here.

## Outcome

An approved change proposal becomes an auditable Git candidate only through repository-manager. A reviewer approves a bounded patch against a specified base; an executor materializes it; an auditor verifies the exact commit, gates, queue outcome and immutable receipt. A proposal stays pending when any authority or remote effect is unavailable.

## Scope

Owner: `repository-manager`. The code is the package `repository_manager/materialization/` (`models.py`, `patch.py`, `ports.py`, `service.py`). It composes `repository_manager/worktree.py`, `repository_manager/safe_commit.py` and `repository_manager/merge_queue.py`; it adds no second Git-mutating surface. Non-goals: repository-manager never stores approval facts, never rebases a stale base and never reports a candidate merged before the queue verifies it.

Why a spec: no other spec owns the proposal-to-commit boundary. Search: `git grep -n "ChangeProposal" -- specs` finds only this spec.

## User Scenarios & Testing

### User Story 1 - Materialize an approved proposal into a queued candidate (Priority: P1)

The user is an executor holding an approved proposal. The outcome is one commit on an isolated branch, submitted to the merge queue, with a receipt that names the exact commit and tree.

**Acceptance scenarios**

1. **Given** a proposal whose approval is bound to its digest and whose base is current, **When** the executor materializes it, **Then** one commit exists on `materialize/<proposal_id>` touching only the approved paths and the receipt state is `queued`.
2. **Given** the declared gate fails, **When** the executor materializes the proposal, **Then** no commit and no queue entry exist and the receipt is `refused` with `gate_failed`.
3. **Given** the queue later reports the candidate landed, **When** the executor reconciles, **Then** the receipt becomes `merged`.

Independent test: the bound tests of `RM-MATERIALIZE-R001`, `RM-MATERIALIZE-R004` and `RM-MATERIALIZE-R005`.

### User Story 2 - Refuse an unapproved or unsafe request without a Git effect (Priority: P1)

The user is an auditor. The outcome is proof that an invalid request leaves the canonical repository untouched.

**Acceptance scenarios**

1. **Given** an expired, altered or wrong-repository approval, **When** the request arrives, **Then** the receipt is `refused` and no worktree exists.
2. **Given** a patch with a traversal path, a symlink, binary content or a path outside the allowlist, **When** the request arrives, **Then** the receipt is `refused` with `path_rejected` and the worktree is removed.
3. **Given** the base moved after approval, **When** the request arrives, **Then** the receipt is `refused` with `stale_base` and a new approval is required.

Independent test: the bound tests of `RM-MATERIALIZE-R002` and `RM-MATERIALIZE-R003`.

### User Story 3 - Retry safely and never bypass the service (Priority: P2)

The user is an orchestrator that loses a response or a dependency. The outcome is one candidate per request and no ungoverned branch.

**Acceptance scenarios**

1. **Given** a receipt exists for an idempotency key, **When** the same request is repeated, **Then** the stored receipt returns and no second commit exists.
2. **Given** the same key with an altered proposal, **When** the request arrives, **Then** it is refused.
3. **Given** the approval authority, the receipt store or the queue is unavailable, **When** the request arrives, **Then** the receipt is `pending` with `uncertain_delivery`.

Independent test: the bound tests of `RM-MATERIALIZE-R006` and `RM-MATERIALIZE-R007`.

## Requirements

- **FR-001**: A proposal is admitted only after its approval binds to it and the approval authority verifies it (`RM-MATERIALIZE-R002`).
- **FR-002**: A patch applies in an isolated worktree restricted to the approved paths (`RM-MATERIALIZE-R003`).
- **FR-003**: One commit follows the declared gate and an explicit path list (`RM-MATERIALIZE-R004`).
- **FR-004**: The candidate enters the existing merge queue and is never reported merged before verification (`RM-MATERIALIZE-R005`).
- **FR-005**: An immutable receipt records every outcome and replays idempotently (`RM-MATERIALIZE-R006`).
- **FR-006**: No direct Git fallback exists and an unavailable dependency leaves the request pending (`RM-MATERIALIZE-R007`).
- **FR-007**: One service composes the whole path and is reachable from a CLI verb and an MCP tool (`RM-MATERIALIZE-R001`).

## Success Criteria

- **SC-001**: Every refusal fixture leaves the canonical repository with no new ref and no changed file (`RM-MATERIALIZE-R002`, `RM-MATERIALIZE-R003`).
- **SC-002**: A repeated identical request yields one commit and one receipt digest (`RM-MATERIALIZE-R006`).
- **SC-003**: Fault injection of each port yields `pending` and no commit through another path (`RM-MATERIALIZE-R007`).

## Edge Cases

- The proposal expires between admission and queue submission: the receipt stays `queued` and the queue verifies the base.
- The queue accepts the candidate and the receipt store is down: the receipt returns to the caller and a replay answers `pending` until the store returns.
- The patch applies cleanly and changes a path outside the allowlist only after rename detection: the numstat re-check rejects it.
- The base branch moved to a descendant of the approved base: the request is `stale_base`; no implicit rebase happens.

## Assumptions

- Git is authoritative for source bytes and history; the approval authority owns approval facts and repository-manager verifies a bounded proof.
- A local fake approval authority and temporary Git repositories cover every rule; a real forge qualification is optional until merge acceptance.
- The patch is text and at most 2 MiB (`MAX_PATCH_BYTES`).

## Boundaries

Git is authoritative for source bytes and history; repository-manager owns worktrees, gates, commits and merge queue submission. The external approval authority owns approval facts. A graph index may store the receipt and cannot manufacture a Git commit. Implementations never require a particular operator environment.
