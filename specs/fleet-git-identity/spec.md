# RM-IDENTITY-001 — Fleet Git identity and history reconciliation

**Candidate owner:** repository-manager. **Owner decision:** OPEN; RM is the proposed execution home because it owns Git worktrees and release ordering, but repository governance must approve the authority before implementation or execution. See [requirements.md](requirements.md) for the definition of every requirement ID and [status.json](status.json) for current delivery state.

## Outcome

An authorized maintainer can preview and, after explicit approval, rewrite author and committer identities across a selected set of public repositories while preserving every commit's tree bytes and a recoverable old-ref map. The operation covers branches and tags, coordinates remote publication, and provides a verifiable rollback plan. Every contributor can run the preview and proof against disposable Git fixtures without credentials or a fleet checkout.

## Canonical identity policy

The approved identity manifest declares exact old-email/name match rules and the canonical identity for three classes: human operator, Claude, and Codex. Claude variants map to `Claude <noreply@anthropic.com>`; Codex variants map to `Codex <codex@users.noreply.github.com>`. Unclassified identities require a review decision before rewriting; the historical request proposed mapping them to the operator, but doing so may misattribute external contributions. The operator's canonical name/email and exact alias list must be reviewed and checked into this repository's public policy before execution. Never infer identity from a fuzzy name match or private inventory.

## Requirements

| ID | Contract | Acceptance |
|---|---|---|
| RM-IDENTITY-01 | Enumerate a selected public repository set, all local refs and annotated/lightweight tags, remote tracking refs, and signed objects; reject incomplete or divergent discovery. | Fixture with hidden ref, detached tag, or inaccessible remote cannot execute. |
| RM-IDENTITY-02 | Produce a deterministic dry-run identity map and old→new commit/tag/ref plan with counts, affected authors/committers, unchanged object tree digests, and collision warnings. | Repeating preview on same source yields the same digest; changed source invalidates approval. |
| RM-IDENTITY-03 | Require signed or otherwise authenticated approval bound to repository set, source ref digests, policy digest, target remotes, and expiry. | Missing, expired, or changed approval has zero mutating effects. |
| RM-IDENTITY-04 | Preserve byte-identical tree objects for every rewritten commit and preserve topology, parent order, messages, commit timestamps, tag payload semantics where possible; signed tags/commits require an explicit re-signing decision. | Independent verifier compares every old/new tree and graph invariant. |
| RM-IDENTITY-05 | Create immutable, collision-safe backup refs and a durable object map before any ref update. Stage all rewritten refs in quarantine, verify reachability, then update each repository atomically. | Simulated failure before update leaves old refs intact; after update, backups reconstruct old refs. |
| RM-IDENTITY-06 | Publish to each configured remote with lease-protected ref updates and a per-remote outcome receipt; never treat a partial push as fleet success. | Remote drift, rejected protected branch, or one remote outage yields partial/pending status and reconciliation instructions. |
| RM-IDENTITY-07 | Make downstream coordination explicit: pause merge queues, notify/refresh open PR branches and forks, resynchronize release baselines, and resume only after remote and CI verification. | Rehearsal proves no queued candidate is merged against stale history. |

## Decisions that must be resolved before build or execution

1. Confirm RM as the sole history materializer and name the approving authority. A spec reviewer must record the decision in this file.
2. Approve the public identity manifest, especially treatment of unknown and external contributor identities. The safe default is refusal.
3. Approve signed tag/commit handling and protected branch permissions for each remote.
4. Decide how contributors and open PRs receive a migration notice and recover their branches.

## State and evidence

No history rewrite is claimed. `status.json` remains SPECIFIED/NOT_AUDITED until owner and policy decisions are recorded; source code, if built, still requires exact-main commit and test evidence before LANDED, and a fully verified rehearsal/publication receipt before ACCEPTED. Record hashes and public CI/PR URLs here, not machine-specific paths.
