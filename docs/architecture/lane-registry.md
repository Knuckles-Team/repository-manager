# Durable lane registry boundary

RMDD-09 models one linked worktree as a fenced lane. `LaneRecord` carries the
collision-safe repository identity, branch/worktree reservation, owner/session,
heartbeat/TTL, quota observations, active claims, recovery anchors, and
lifecycle state. Repository identity includes the canonical repository path;
the display basename is never an identity key.

## Authority boundary

`LaneRegistry` is a bounded controller and SQLite is only a local projection for
status and diagnostics. A registry constructed without an injected durable
`LaneAuthority` refuses every managed mutation. The projection, a process lock,
or a local quota snapshot cannot authorize allocation, heartbeat, adoption,
transition, expiry, quarantine, or cleanup.

The production adapter is `NativeLaneAuthorityAdapter`. It accepts only the
typed native authority seam that atomically implements `allocate_lane`,
`get_lane`, `list_lanes`, `transition_lane`, and `heartbeat_lane`; optional
adoption, anchor, rollback, and cleanup-job methods remain on that same
authority. RMDD-06's current `RepositoryJobService` does not yet expose this
atomic lane transaction, so the adapter fails closed until the engine/AU native
lane lease/quota contract is available. `FakeDurableLaneAuthority` is explicitly
test-only and is useful for two-controller/restart tests when both controllers
share the same fake object.

The lifecycle is fenced and owner-bound:

```text
allocating -> active -> submitted -> landed
     |          |          |          |
     +-------> aborted/expired -> quarantined
                              \-> rejected (finish only)
observed_legacy --(operator adoption)--> active
```

Repeated operations with the current fence are idempotent where the target is
already reached. A stale owner/fence is refused. Branch and worktree uniqueness,
quota admission, and idempotency must be performed by the injected authority in
one durable transaction before `WorktreeManager.add` is called.

## Reclamation

Assessment is read-only. It fails closed when process liveness, occupancy, Git,
job/candidate/concept, or recovery-anchor evidence is missing or unavailable. A
recorded anchor is not trusted merely because it is nonempty: it must resolve to
an exact ref/commit and prove that the lane tip is an ancestor of the anchor.

`request_cleanup(submit=None)` is preview-only. Execution requires a separate
durable cleanup WorkItem/job id and lease fence, supplied by a cleanup authority;
an in-memory callback or preview cannot authorize removal. Execution re-reads
the authoritative lane and cleanup job, then rechecks process, claims, Git,
anchor, occupancy, and fence immediately before delegating to
`WorktreeManager.remove`. Removal remains guarded; failure leaves the lane
expired for reconciliation. A guarded remove is not reported complete until an
exact durable removal receipt and quarantine transition are persisted. A later
run can reconcile that receipt without trying to remove an already-removed path
again.

## Migration and rollback

Discovery creates `observed_legacy` records without claiming ownership. Adoption
requires an explicit operator and a fresh durable uniqueness check. Rollback is
the durable allocation switch: it stops new allocations while retaining existing
records and worktrees. No cleanup path assumes the registry projection is
available or silently falls back to local SQLite truth.

Reconciliation compares durable records with a read-only worktree listing and
classifies managed, observed-legacy, missing/path/branch/state mismatches, and
orphan worktrees. Repairs are separate fenced jobs and are not direct filesystem
mutations.
