# Repository job application service

RMDD-06 provides one domain-facing application service for Repository Manager's
long-running development operations. The service is deliberately transport
agnostic: MCP, CLI, and later worker adapters inject the same
`RepositoryJobService` instead of maintaining independent job state.

## Authority boundary

`RepositoryJobService` accepts a verified `JobAuthorization` and an injected
`RepositoryJobPort`. The production `GraphRepositoryJobPort` lazily delegates
to `agent_utilities.orchestration.repository_work_item`, whose graph-os
WorkItem is the only durable state, dependency, lease, retry, fence, and
result authority. The service owns no `_jobs`, `_job_futures`, worker thread,
future, SQLite table, Docket queue, Redis queue, or Git/process mutation.

`FakeRepositoryJobPort` has the same structural seam for tests. Recreating the
service with the same port reads the durable-shaped records again; it does not
restore state from service memory.

## Authorized operations

All reads and mutations carry an authenticated tenant and owner. A missing or
foreign job is refused with `unauthorized_target` rather than revealing whether
another tenant owns its handle. Owner filters on list calls are forced to the
authenticated owner, and the production cancel port repeats an owner-scoped
read immediately before its native tenant mutation. The service maps authority conflicts and state refusals
to the stable RMDD/C-10 codes exposed in `RepositoryJobServiceError.code`.

The lifecycle policy is intentionally small:

| Operation | Allowed states | Result |
| --- | --- | --- |
| `cancel` | `submitted`, `ready`, `leased`, `running` | Durable cancel; repeated cancel is idempotent. |
| `cancel` | `succeeded`, `failed`, `dead-letter` | `invalid_state_combination`. |
| `retry` | terminal `failed`/`dead-letter` with budget remaining | Deterministic new WorkItem attempt; old record is unchanged. |
| `retry` | active, succeeded, cancelled, or exhausted | `invalid_state_combination`. |

The retry idempotency key is derived from the durable source job ID and attempt
number. The follow-up carries only `old.max_attempts - old.attempt`,
so chained retries cannot replenish the original attempt budget. Two service
instances therefore converge on one durable follow-up WorkItem without a
process-local lock.

## Bounded listing

`JobPage` always reports `scanned`, `exhausted`, and an optional opaque
`rmpage:v1:` continuation. The cursor contains a tenant digest and the
`(created_at, work_item_id)` keyset tie-breaker. A port reads at most the
requested native page in one call; the production AU seam filters tenant,
kind, and native state in the query while owner and richer domain filters are
applied to that one page. A non-matching owner row still advances the cursor,
so post-page filters never trigger an unbounded scan. Callers explicitly pass
`JobFilters.cursor` to continue.

## Reconciliation and shadow mode

`ReconciliationObservation` is read-only input from a worker/host observer.
The service reports deterministic classifications:

`missing_worktree`, `stale_process`, `orphan_artifact`, `stale_fence`,
`target_drift`, `already_completed_effect`, or `clean`.

An explicit `process_present=False` is a stale-process finding for leased or
running jobs; `None` means that the observer did not inspect the process and
does not create a finding. This distinction keeps partial probes fail-closed.

The response includes a stable, previewable `RepairProposal`. With explicit
`enqueue_repairs=True`, the port submits one idempotent `operation=repair`
WorkItem. No Git checkout, process signal, artifact deletion, branch move, or
other mutation happens inline; the later repair worker must execute the
proposal under the normal scheduler/lease/consent boundaries. The repair
request binds a versioned repair-intent digest and the source correlation; a
repair worker must re-observe the correlated durable job instead of trusting a
stale observation snapshot. There is no second proposal store.

`LegacyShadowAdapter` compares selected legacy MCP fields with the durable view
and returns a `ShadowMismatch`. It never writes the legacy record and never
uses legacy memory as authority. RMDD-20 may use this hook while switching
entrypoints to the service.

## Downstream integration

RMDD-05 projects durable views to FastMCP tasks. RMDD-08 claims WorkItems and
applies resource admission. RMDD-10/11/12/18/20 consume the service through
the injected port; they must not reach into the AU adapter's private helpers or
recreate lifecycle policy. The narrow production listing adapter uses AU's
native bounded keyset page primitive until a public AU continuation API is
available; the service contract remains explicit and bounded.
