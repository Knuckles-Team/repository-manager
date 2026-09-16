# Weighted resource scheduler

Repository Manager admits development work through one weighted policy surface.
The scheduler is deliberately an admission service: graph-os WorkItems remain
the only durable job, lease, attempt, and fence authority, and no scheduler
method starts a subprocess, opens SSH, mutates Git, or acquires a local
filesystem lease.

## Profiles

`repository_manager.resource_profiles.default_resource_profiles()` registers
the conservative v1 classes:

| Profile | CPU weight | Admission memory MiB | Runtime high/max MiB | Process slots | Concurrency |
| --- | ---: | ---: | ---: | ---: | --- |
| `light-check` | 1 | 256 | 2,048 / 4,096 | 1 | independent |
| `frontend-build` | 8 | 8,192 | 20,480 / 24,576 | 1 | one heavy producer |
| `rust-build` | 8 | 16,384 | 32,768 / 49,152 | 2 | one heavy producer |
| `pre-commit` | 4 | 2,048 | 8,192 / 12,288 | 2 | weighted host capacity |
| `merge-drain` | 2 | 1,024 | 4,096 / 6,144 | 1 | repository/branch exclusive |
| `workspace-release` | 4 | 4,096 | 2,048 / 4,096 | 2 | repository exclusive |

Profiles are versioned and unknown names refuse.  A request can ask for more
than the profile minimum, but never less.  The profile owns its concurrency
key; a caller cannot accidentally turn a frontend build into a `light-check`
reservation by relying on the default request value.

## Local process resource guard

`repository_manager.resource_guard` is the local execution boundary used by
the build broker, gate runner, and generic Repository Manager merge-queue drain.
It mirrors the admission and containment rules used by the fleet build
dispatcher while leaving remote transport to the dispatcher. A command whose
shape invokes Cargo always receives `rust-build` limits, even if its manifest
says `light-check` or the gate is labelled fast.

Before process creation the guard atomically reserves the single host-user
heavy slot in the same reservation authority used by `dispatch_build.sh`, then
records memory PSI, available memory, free swap, user-cgroup
headroom and OOM counters. It also checks free bytes and free inodes for the
explicit target, disk-backed `TMPDIR`, and the configured workspace and system
temporary probes. A high watermark
closes admission and the guard reopens only after the low PSI watermark is
reached. Refusal is a deferral and never a code-quality verdict.

Admitted build and gate commands run in a transient user systemd scope with
`MemoryHigh`, `MemoryMax`, `MemorySwapMax`, CPU quota, task count, runtime, and
`OOMPolicy` limits. The guard records each scope's systemd control group and
releases its host reservation only after the scope is empty; if systemd removes
the empty group first, its terminal inactive/dead state is checked. Missing or
uncertain closure retains the reservation, and stale recovery requires a
concrete empty control group. A merge-queue drain keeps its child in the installed
service cgroup instead of creating a sibling scope; admission verifies finite
memory, swap, CPU, and task controls across that cgroup's ancestors, and the
runner continues to monitor every descendant for cgroup escape. Cargo receives
`CARGO_BUILD_JOBS=2`, `CARGO_INCREMENTAL=0`, the guarded target directory, and
bounded thread-pool settings. Targets and temporary files default under
`/var/tmp/repository-manager-resource-guard`; operators may point
`RM_RESOURCE_GUARD_ROOT` at another local disk-backed filesystem. Queue child
processes receive `AU_LANE_TEMP_ROOT` and `PYTEST_XDIST_AUTO_NUM_WORKERS=2` so
the agent-utilities lane resolver owns disk-backed temporary paths and pytest
fanout; when that root is present, the generic queue verifies that the actual
gate environment's `CARGO_TARGET_DIR` and `TMPDIR` remain below it instead of
silently replacing the admitted paths with another allocator. Both the
materialized merged worktree and its gate scratch directory stay below that
same admitted `TMPDIR`; Cargo gate argv is checked before spawn so an explicit
`--target-dir` or `-j` value cannot escape the admitted target or job cap. Both the
admission and completion observations travel with the result
for guarded build and gate commands; queue admission refusals carry their
pre-launch evidence in the deferred error. Stdout and stderr are drained
concurrently and retain only a 64 KiB
terminal tail; byte counts, truncation state, and SHA-256 digests preserve
useful proof without letting verbose compilers grow the Repository Manager
process memory.

The generic `repository-manager --merge-queue run` entrypoint is the admission
boundary used by the supervised timer child. An unsupervised direct CLI/API
drain is refused before any queue work because only the runner supplies the
real per-repository wall-clock deadline (bounded by the `merge-drain` profile)
and descendant cleanup. The separate
native agent-utilities queue implementation remains outside this package's
dependency boundary and must apply its own equivalent admission policy before
invoking other work.

## Admission and fencing

`ResourceScheduler.admit()` performs deterministic profile resolution, deadline
checking, target/label/anti-affinity filtering, disk hysteresis, and weighted
fit/rank.  Before a consumer may execute, the scheduler calls the injected
`WorkItemReservationPort.atomic_reserve()` with the exact WorkItem ID, attempt,
and fence.

`explain_only=True` returns `PREVIEW`/`preview_only`, not `ADMITTED`, and has no
reservation evidence.  Callers must require an active native reservation before
starting execution; a fit explanation is never a handoff credential.

An active local row is never an execution credential.  Every retry or handoff
that could return `ADMITTED` first calls the port's exact
`query_reservation()` (or an equivalent native transaction), which rechecks the
current fence, immutable WorkItem admission extension, reservation identity,
and lifecycle revision.  A missing native row, stale fence, changed input, or
released/expired tombstone is deferred; the scheduler does not revive it from
the local store.  A replica with no local row can rebuild its mirror only from
the native record returned by that query.

The production port must re-read and atomically update WorkItem-linked host
capacity, concurrency, repository/branch exclusivity, and reservation records
inside the same native CAS/transaction.  The `CapacityInventory`,
`InMemoryReservationStore`, and `JsonReservationStore` are local mirrors and
deterministic test fixtures; their locks and files are never distributed truth.
Release and reclaim use the symmetric native operations and only then update
the local accounting mirror.  Native authority retains a released/expired
tombstone and monotonically increasing revision, so a local update failure can
retry `IDEMPOTENT` without returning capacity or fairness debt twice.  A stale
worker cannot release or reclaim a newer attempt's reservation.

Reservation IDs are deterministic for a WorkItem attempt when callers do not
provide one (the work item ID plus attempt, namespaced and hashed).  The native
transaction enforces one active reservation per attempt and compares the full
immutable admission input before returning an idempotent result.  A changed
request therefore refuses without compensation-releasing the original native
link.  Request timestamps and capacity observations are projections; the
retry-stable input fingerprint and requested TTL remain part of the comparison.

On service recreation, active durable records are replayed through the
inventory's restore path.  Restore verifies nonnegative accounting and the
declared capacity bound but intentionally bypasses new-admission heartbeat,
drain/quarantine, and observed-disk checks.  A held reservation remains
explainable and releasable while its host is stale or ineligible; those host
policies block only new work.

Remote targets are represented by authorized inventory aliases and never take a
local `fcntl.flock`.  The old `task_queue.acquire()` path remains a same-node
compatibility adapter only; it is not a substitute for a WorkItem-backed
reservation.

The RMDD-08 package intentionally stops at the injected native-port contract;
it does not claim to ship a graph-os adapter.  The production binding/dependency
lane must persist a WorkItem-linked reservation extension and implement these
single-transaction verbs:

1. `reserve_or_deduplicate`: verify the current WorkItem fence plus immutable
   owner/tenant/profile/requirement/target/repository/concurrency/fairness
   extension, enforce one active reservation per WorkItem attempt, re-read
   host policy/capacity, update capacity and fairness debt, and return
   `accepted`, `idempotent`, or a typed refusal.
2. `query_reservation`: exact-read the immutable linked record and current
   lifecycle revision under the WorkItem fence; return active records or
   retained release/expiry tombstones, never a local projection.
3. `release_if_current`: verify the current or terminal exact attempt/fence
   and immutable reservation identity, then unlink, return capacity, and
   retain a release tombstone; a newer attempt remains stale.
4. `reclaim_if_expired_or_superseded`: let current controller authority
   reclaim only after TTL expiry or attempt supersession.

The native schema must retain the immutable input fingerprint, WorkItem
`(id, attempt, fence)`, host/profile/requirement, fairness group/cost,
capacity snapshot, and lifecycle revision.  The in-memory port exercises this
contract; JSON stores are restart fixtures only.

Repository and branch exclusivity are global keys across all registered hosts;
anti-affinity is intentionally host-local.  Host capacity, labels, heartbeat,
drain/quarantine state, and observed disk telemetry are versioned.  A refresh
is accepted only when its host revision advances and it preserves held
reservations; stale/equal updates are no-ops and a refresh that cannot account
for existing holds is rejected.

## Disk and fairness policy

Disk watermarks are used-space MiB.  Crossing the high watermark defers new
admission and may emit a bounded GC request; no deletion occurs here.  A
blocked host remains blocked until usage falls to the low watermark, so
intermediate observations do not flap.  Hysteresis is keyed by native host and
versioned profile policy (`host_id`, `profile:vN`), not by optional request
watermarks.  Missing or weaker caller thresholds cannot clear a blocked native
policy; profiles retain authority over their safety limits.

Queue selection ages waiting priority deterministically and accounts service by
fairness group.  Weighted selection prevents a continuously producing tenant
from monopolizing all admission slots while preserving priority and stable ID
tie-breaking.  The default in-process selector is explicitly advisory;
selection never mutates debt.  Only the native WorkItem reservation
transaction records service after a successful claim.  Production replicas
must inject a state port whose authority is explicitly `native`/distributed;
the JSON fixture is `local_advisory` and the in-memory fixture is
`simulation`, even when shared by multiple test replicas.  The JSON/in-memory
fairness stores are fixtures, just like the local reservation mirrors.

## Migration

Existing `ExecutionClass` names remain available through `task_queue.py`, now
carrying a profile/concurrency compatibility mapping.  Consumers should first
run shadow admission and compare explanations, then submit WorkItem-backed
requests to `ResourceScheduler`; local execution-class leases can be removed
only after all consumers use the native reservation port.  Rollback stops new
reservations, reconciles open native records, and returns consumers to the
compatibility adapter without abandoning held capacity.
