"""CONCEPT:RM-PROVENANCE — RMDD-19 repository-development provenance emitter/adapter.

Thin, additive, repository-development-shaped wrappers over
``agent_utilities.observability.repository_provenance.write_repository_event``
-- the ONE write chokepoint that module owns (RMDD-19 lane brief: "Emit
through existing graph authority; do not introduce another store"). Every
function in this module funnels through that single entrypoint; this module
adds no second graph-write path of its own.

**Not wired in yet.** RMDD-19's forbidden-edit boundary excludes WorkItem
core (``repository_manager/development/*.py``), MCP/CLI entrypoints
(``mcp_server.py``, ``repository_manager.py``), and domain build/merge code
(``build_service.py``, ``merge_queue.py``, ``landing_policy.py``, ...) -- the
actual repository-job/lease/build/validation/landing call sites. RMDD-20
("Final MCP/CLI development surfaces and parity") owns calling these
functions from those call sites; see the RMDD-19 handoff for the exact
caller inventory this module's shapes were designed against.

Every emitter accepts the ALREADY-VALIDATED C-02..C-07 contract types from
``repository_manager.development.models``/``.enums`` (read-only consumption
of a frozen, closed-dependency contract -- this module does not construct or
mutate them) so a future caller can pass its existing typed record straight
through without re-deriving fields.

**Occurrence is caller-supplied and must be derived deterministically from
the event's own immutable identity** (e.g. an index into an already-ordered
tuple), never a runtime counter -- see the module docstring on
``write_repository_event`` for why: idempotency across a retry/restart
depends on it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from agent_utilities.observability import gateway_metrics as _metrics
from agent_utilities.observability.repository_provenance import (
    RepositoryProvenanceUnavailable,
    StaleFenceError,
    explain_repository_job,
    query_repository_provenance,
    reconciliation_report,
    write_repository_event,
)

from repository_manager.development.enums import JobState
from repository_manager.development.models import (
    ArtifactReference,
    Candidate,
    ExecutionResult,
    Generation,
    LeaseRecord,
    ResourceReservation,
    ValidationEvidence,
)

__all__ = [
    "RepositoryProvenanceUnavailable",
    "StaleFenceError",
    "emit_submitted",
    "emit_dependency_ready",
    "emit_lease_claimed",
    "emit_admission_placement",
    "emit_started",
    "emit_heartbeat",
    "emit_checkpoint",
    "emit_cancelled",
    "emit_retried",
    "emit_dead_lettered",
    "emit_command_result",
    "emit_artifact_published",
    "emit_validation_evidence",
    "emit_candidate_event",
    "emit_generation_event",
    "emit_bisection_event",
    "emit_concept_event",
    "emit_landing_push",
    "emit_gc_reconcile",
    "explain_job",
    "reconcile_job",
    "job_events",
]


def _timestamp(value: Any) -> str | None:
    """Best-effort UTC ISO-8601 stamp from a contract model's own datetime field."""

    if value is None:
        return None
    try:
        return value.strftime("%Y-%m-%dT%H:%M:%SZ")
    except AttributeError:
        return str(value)


# --- submit / dependency-ready --------------------------------------------


def emit_submitted(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    priority_class: str = "default",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 ``submitted`` transition."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="submitted",
        occurrence=0,
        status=str(JobState.SUBMITTED),
        timestamp=timestamp,
        correlations={"repo": repo},
    )
    _metrics.REPOSITORY_JOB_QUEUE_DEPTH.labels(
        repo=repo, priority_class=priority_class
    ).inc()
    return result


def emit_dependency_ready(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    dependency_ids: Sequence[str],
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 dependency-satisfied transition into ``ready``."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="dependency_ready",
        occurrence=0,
        status=str(JobState.READY),
        timestamp=timestamp,
        payload={"dependency_count": len(dependency_ids)},
        correlations={"dependency_ids": ",".join(sorted(dependency_ids))},
    )


# --- lease/claim, admission/placement --------------------------------------


def emit_lease_claimed(
    engine: Any,
    *,
    work_item_id: str,
    repo: str,
    lease: LeaseRecord,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 lease/fence attachment (:class:`LeaseRecord`)."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=lease.attempt,
        kind="lease_claimed",
        occurrence=0,
        status=str(JobState.LEASED),
        timestamp=timestamp or _timestamp(lease.heartbeat_at),
        payload={"expires_at": _timestamp(lease.expires_at)},
        correlations={"owner": lease.owner, "repo": repo},
        fence=lease.fence,
    )


def emit_admission_placement(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    reservation: ResourceReservation,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-03 admission result (:class:`ResourceReservation`), distinct from an estimate."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="admission_placement",
        occurrence=0,
        status=reservation.state.value,
        timestamp=timestamp or _timestamp(reservation.reserved_at),
        payload={
            "resource_class": reservation.request.resource_class,
            "selected_target_kind": reservation.selected_target.kind.value,
        },
        error=reservation.reason,
        correlations={
            "reservation_id": reservation.reservation_id,
            "resource_class": reservation.request.resource_class,
            "repo": repo,
        },
        fence=reservation.fence,
    )
    host_alias = reservation.selected_target.alias or reservation.selected_target.kind.value
    gauge = _metrics.REPOSITORY_CAPACITY_RESERVATIONS.labels(
        resource_class=reservation.request.resource_class, host_alias=host_alias
    )
    if reservation.state.value == "reserved":
        gauge.inc()
    elif reservation.state.value in {"released", "expired"}:
        gauge.dec()
    return result


# --- start / heartbeat / checkpoint ----------------------------------------


def emit_started(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    fence: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 ``running`` transition."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="started",
        occurrence=0,
        status=str(JobState.RUNNING),
        timestamp=timestamp,
        fence=fence,
    )


def emit_heartbeat(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    fence: str,
    heartbeat_index: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 heartbeat/lease-renewal. ``heartbeat_index`` must be the lease's own
    durable renewal count -- never a process-local counter."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="heartbeat",
        occurrence=heartbeat_index,
        status=str(JobState.RUNNING),
        timestamp=timestamp,
        fence=fence,
    )


def emit_checkpoint(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    fence: str,
    checkpoint: str,
    checkpoint_index: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 checkpoint attachment. ``checkpoint_index`` is the worker's own
    ordered checkpoint sequence, not a runtime counter."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="checkpoint",
        occurrence=checkpoint_index,
        status=str(JobState.RUNNING),
        timestamp=timestamp,
        correlations={"checkpoint": checkpoint},
        fence=fence,
    )


# --- cancel / retry / dead-letter ------------------------------------------


def emit_cancelled(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    reason: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 ``cancelled`` transition."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="cancelled",
        occurrence=0,
        status=str(JobState.CANCELLED),
        timestamp=timestamp,
        error=reason,
    )


def emit_retried(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    retry_class: str,
    reason: str = "",
    timestamp: str | None = None,
) -> dict[str, Any]:
    """A retry transition -- ``attempt`` is the NEW attempt number this retry starts."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="retried",
        occurrence=0,
        status=str(JobState.READY),
        timestamp=timestamp,
        error=reason,
        correlations={"repo": repo, "retry_class": retry_class},
    )
    _metrics.REPOSITORY_JOB_RETRIES.labels(repo=repo, retry_class=retry_class).inc()
    return result


def emit_dead_lettered(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    reason: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-02 ``dead-letter`` terminal transition."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="dead_lettered",
        occurrence=0,
        status=str(JobState.DEAD_LETTER),
        timestamp=timestamp,
        error=reason,
        correlations={"repo": repo},
    )
    _metrics.REPOSITORY_JOB_RETRIES.labels(repo=repo, retry_class="dead_letter").inc()
    return result


# --- command result / artifact / validation --------------------------------


def emit_command_result(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    result: ExecutionResult,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-04 :class:`ExecutionResult`. This is the terminal-effect kind the
    fence guard protects -- ``result.fence`` must be the WorkItem's CURRENT
    fence or the write is refused with :class:`StaleFenceError`."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="command_result",
        occurrence=0,
        status=result.outcome.value,
        timestamp=timestamp or _timestamp(result.finished_at),
        payload={
            "duration_ms": result.duration_ms,
            "exit_code": result.exit_code,
            "signal": result.signal,
            "stdout_tail": result.stdout_tail,
            "stderr_tail": result.stderr_tail,
        },
        error=result.failure_class.value if result.failure_class else "",
        correlations={
            "command_id": result.command_id,
            "worker_id": result.worker_id,
            "repo": repo,
        },
        fence=result.fence,
    )


def emit_artifact_published(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    artifact: ArtifactReference,
    artifact_index: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-05 artifact publication. ``artifact_index`` is the index of this
    artifact within its producer's own immutable ``artifact_refs`` tuple --
    deterministic and therefore idempotent across a replay of the same
    result."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="artifact_published",
        occurrence=artifact_index,
        status="published",
        timestamp=timestamp,
        payload={
            "size_bytes": artifact.size_bytes,
            "media_type": artifact.media_type,
        },
        correlations={
            "content_address": artifact.content_address,
            "relative_path": artifact.relative_path,
        },
    )


def emit_validation_evidence(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    evidence: ValidationEvidence,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """C-06 :class:`ValidationEvidence` for one stage/gate against one tree."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="validation_certificate",
        occurrence=0,
        status=evidence.outcome.value,
        timestamp=timestamp or _timestamp(evidence.finished_at),
        payload={
            "differential": evidence.differential,
            "failure_count": len(evidence.failure_ids),
        },
        correlations={
            "evidence_id": evidence.evidence_id,
            "generation_id": evidence.generation_id or "",
            "tree_sha": evidence.tree_sha,
            "repo": repo,
        },
    )
    duration_seconds = max(
        0.0,
        (evidence.finished_at - evidence.started_at).total_seconds(),
    )
    _metrics.REPOSITORY_VALIDATION_DURATION.labels(
        repo=repo, stage=evidence.stage.value
    ).observe(duration_seconds)
    return result


# --- candidate / generation / bisection / concept --------------------------


def emit_candidate_event(
    engine: Any,
    *,
    work_item_id: str,
    candidate: Candidate,
    occurrence: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """One :class:`Candidate` lifecycle transition. ``occurrence`` should be
    a durable transition count the caller already tracks (e.g. state-change
    index), not a runtime counter."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=candidate.version,
        kind="candidate_event",
        occurrence=occurrence,
        status=candidate.state.value,
        timestamp=timestamp or _timestamp(candidate.enqueued_at),
        error=candidate.reason,
        correlations={
            "candidate_id": candidate.candidate_id,
            "lane_id": candidate.lane_id,
            "generation_id": candidate.generation_id or "",
        },
    )


def emit_generation_event(
    engine: Any,
    *,
    work_item_id: str,
    generation: Generation,
    occurrence: int,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """One :class:`Generation` lifecycle transition (open/sealed/certified/landing/...)."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="generation_event",
        occurrence=occurrence,
        status=generation.state.value,
        timestamp=timestamp,
        error=generation.reason,
        correlations={
            "generation_id": generation.generation_id,
            "target_branch": generation.target_branch,
        },
        fence=generation.landing_fence or "",
    )
    if generation.state.value == "landed":
        _metrics.REPOSITORY_LANDING_DRIFT.labels(
            repo=generation.repository.repository_id, drift_kind="landed"
        ).inc()
    elif generation.state.value == "rejected":
        _metrics.REPOSITORY_LANDING_DRIFT.labels(
            repo=generation.repository.repository_id, drift_kind="rejected"
        ).inc()
    return result


def emit_bisection_event(
    engine: Any,
    *,
    work_item_id: str,
    repo: str,
    generation_id: str,
    outcome: str,
    isolated_candidate_id: str = "",
    occurrence: int = 0,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """A failure-bisection run against a sealed generation."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="bisection_event",
        occurrence=occurrence,
        status=outcome,
        timestamp=timestamp,
        correlations={
            "generation_id": generation_id,
            "isolated_candidate_id": isolated_candidate_id,
            "repo": repo,
        },
    )
    _metrics.REPOSITORY_BISECTION_RUNS.labels(repo=repo, outcome=outcome).inc()
    return result


def emit_concept_event(
    engine: Any,
    *,
    work_item_id: str,
    concept_id: str,
    action: str,
    occurrence: int = 0,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """A concept-coordination action (reserve/release/reconcile) tied to this job."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="concept_event",
        occurrence=occurrence,
        status=action,
        timestamp=timestamp,
        correlations={"concept_id": concept_id},
    )


# --- landing / GC-reconcile --------------------------------------------


def emit_landing_push(
    engine: Any,
    *,
    work_item_id: str,
    attempt: int,
    repo: str,
    landing_fence: str,
    outcome: str,
    target_branch: str,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """Fenced target-branch landing result -- also a terminal-effect kind
    the fence guard protects."""

    result = write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=attempt,
        kind="landing_push",
        occurrence=0,
        status=outcome,
        timestamp=timestamp,
        correlations={"target_branch": target_branch, "repo": repo},
        fence=landing_fence,
    )
    if outcome == "target_moved":
        _metrics.REPOSITORY_LANDING_DRIFT.labels(
            repo=repo, drift_kind="target_moved"
        ).inc()
    return result


def emit_gc_reconcile(
    engine: Any,
    *,
    work_item_id: str,
    repo: str,
    reconcile_kind: str,
    findings: Mapping[str, Any] | None = None,
    occurrence: int = 0,
    timestamp: str | None = None,
) -> dict[str, Any]:
    """A GC/reconciliation pass over one WorkItem's lane/build/cleanup state."""

    return write_repository_event(
        engine,
        work_item_id=work_item_id,
        attempt=1,
        kind="gc_reconcile",
        occurrence=occurrence,
        status=reconcile_kind,
        timestamp=timestamp,
        payload=dict(findings) if findings else None,
        correlations={"repo": repo},
    )


# --- read / explain / reconcile projection (backs rm_jobs status/logs/reconcile) --


def explain_job(
    engine: Any, *, work_item_id: str, tenant_ref: str | None = None
) -> dict[str, Any]:
    """Operator-facing explanation for one WorkItem, from provenance alone."""

    return explain_repository_job(engine, work_item_id=work_item_id, tenant_ref=tenant_ref)


def reconcile_job(
    engine: Any, *, work_item_id: str, tenant_ref: str | None = None
) -> dict[str, Any]:
    """Observed-facts-to-proposed-repair projection for one WorkItem."""

    return reconciliation_report(engine, work_item_id=work_item_id, tenant_ref=tenant_ref)


def job_events(
    engine: Any, *, work_item_id: str, tenant_ref: str | None = None, limit: int = 500
) -> list[dict[str, Any]]:
    """Raw ordered event stream for one WorkItem (the ``rm_jobs logs`` projection)."""

    return query_repository_provenance(
        engine, work_item_id=work_item_id, tenant_ref=tenant_ref, limit=limit
    )
