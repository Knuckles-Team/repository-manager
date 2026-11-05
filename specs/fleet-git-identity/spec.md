# RM-IDENTITY — Fleet Git identity and history reconciliation

Component: History identity

**Owner:** repository-manager, the sole history materializer. See [requirements.md](requirements.md) for the definition of every requirement ID.

## Outcome

An authorized maintainer can preview and, after explicit approval, rewrite author and committer identities across a selected set of public repositories while preserving every commit's tree bytes and a recoverable old-ref map. The operation covers branches and tags, coordinates remote publication, and provides a verifiable rollback plan. Every contributor can run the preview and proof against disposable Git fixtures without credentials or a fleet checkout.

## Scope

Owner: `repository-manager`. The built half is the package `repository_manager/history_identity/` (`discovery.py`, `policy.py`, `preview.py`, `approval.py`, `consistency.py`, `cli.py`), reached through the `history-identity` verb of `repository-manager-governance` (`repository_manager/governance/cli.py`). The rewrite, backup, publication and queue-coordination half is specified and unbuilt. Non-goals: the tool never lifts branch protection, never infers identity from a fuzzy name match and never rewrites an identity that no approved rule matches.

Why a spec: no other spec owns history rewriting. Search: `git grep -n "history" -- specs` finds only this spec.

## Canonical identity policy

The approved identity policy declares exact old-name and old-email match rules and the canonical identity each maps to. The canonical identities come from the single allowlist file that the fleet commit-identity gate already owns (`COMMIT_IDENTITY_ALLOWLIST`); `build_policy` reads it and the package carries no second copy. The classes are the human operator and the coding agents; the exact aliases are reviewed and checked into the public policy before any execution.

Recorded rules:

1. **Authority.** repository-manager is the only tool permitted to rewrite history. The approving authority is the repository owner; every run needs the owner's signed approval bound to the exact repository set, source ref digests, policy digest, target remotes and expiry (`RM-IDENTITY-R005`).
2. **Unknown and external identities are refused.** Only identities that match an explicit rule in the approved policy are rewritten. A commit or tag by any other identity is left unchanged and the preview reports it (`RM-IDENTITY-R004`).
3. **Signed objects and protected branches.** Signed commits and tags are not rewritten unless the approval for that run explicitly authorizes re-signing; otherwise the run reports them (`RM-IDENTITY-R006`). A protected ref that rejects the lease-guarded update is reported as refused for that remote (`RM-IDENTITY-R008`).

## User Scenarios & Testing

### User Story 1 - Preview a fleet identity rewrite (Priority: P1)

The user is an authorized maintainer. The outcome is a deterministic plan that names every identity mapping and every unmapped identity, with no change to any repository.

**Acceptance scenarios**

1. **Given** a repository whose refs, tags and remotes are all readable, **When** the maintainer runs `history-identity` with a policy, **Then** the report lists the plan digest, the mapped and unmapped commit counts and writes nothing.
2. **Given** the same source and policy, **When** the maintainer runs the preview twice, **Then** both plan digests are equal.
3. **Given** a hidden ref, a shallow clone, an unreachable remote or diverged branches, **When** the maintainer runs the preview, **Then** the run refuses with exit code 1 and names the cause.

Independent test: the bound tests of `RM-IDENTITY-R002`, `RM-IDENTITY-R003` and `RM-IDENTITY-R004`.

### User Story 2 - Approve a plan exactly (Priority: P1)

The user is the repository owner. The outcome is an approval that is valid only for the exact plan it was given for.

**Acceptance scenarios**

1. **Given** an approval signed for a plan, **When** the maintainer verifies it against the same plan before expiry, **Then** it is accepted.
2. **Given** an expired, forged, missing or differently bound approval, **When** the maintainer verifies it, **Then** it is refused and nothing mutates.

Independent test: the bound test of `RM-IDENTITY-R005`.

### User Story 3 - Rewrite, publish and recover (Priority: P2)

The user is the maintainer after approval. The outcome is rewritten history whose trees match, with backups, per-remote receipts and a paused queue until verification.

**Acceptance scenarios**

1. **Given** an approved plan, **When** the rewrite runs, **Then** every new commit has its old tree, parents, message and timestamp, and backup refs restore the old history.
2. **Given** one remote drifted, **When** publication runs, **Then** the receipt is partial and no overwrite of the drift occurs.
3. **Given** a queued candidate based on old history, **When** the queue resumes after cutover, **Then** the candidate is refused.

Independent test: the planned tests of `RM-IDENTITY-R001` and `RM-IDENTITY-R006` to `RM-IDENTITY-R009`.

## Requirements

- **FR-001**: The preview enumerates every ref and refuses incomplete or diverged discovery (`RM-IDENTITY-R002`, `RM-IDENTITY-R003`).
- **FR-002**: The preview is deterministic and approval binds to its digests (`RM-IDENTITY-R004`, `RM-IDENTITY-R005`).
- **FR-003**: The rewrite preserves trees and topology and is verified independently (`RM-IDENTITY-R006`).
- **FR-004**: Backup refs and a quarantine stage precede an atomic local update (`RM-IDENTITY-R007`).
- **FR-005**: Publication is lease-protected with per-remote receipts (`RM-IDENTITY-R008`).
- **FR-006**: Merge queues pause during a rewrite and resume only after verification (`RM-IDENTITY-R009`).
- **FR-007**: The end-to-end rewrite composes all of the above (`RM-IDENTITY-R001`).

## Success Criteria

- **SC-001**: Two previews of an unchanged source produce one plan digest (`RM-IDENTITY-R004`).
- **SC-002**: Every old and new tree object pair matches after a rewrite (`RM-IDENTITY-R006`).
- **SC-003**: A failure injected before the local update leaves every original ref intact (`RM-IDENTITY-R007`).
- **SC-004**: No partial publication is reported as fleet success (`RM-IDENTITY-R008`).

## Edge Cases

- A repository has no remote: discovery returns an empty remote list and publication has nothing to do.
- A commit's author is mapped and its committer is not: the commit counts as unmapped and the preview warns.
- A signed tag points at a rewritten commit: the tag is reported and left unchanged unless the approval authorizes re-signing.
- The source changes between preview and execution: the approval no longer matches and the run restarts at preview.

## Assumptions

- Git objects and refs are authoritative; receipts index what was proven.
- Contributors run every proof against temporary repositories with synthetic users and a local bare remote; a real fleet run is an operational event separate from code acceptance.
- The approval mechanism is a shared-key HMAC until the owner chooses another authenticated mechanism.
