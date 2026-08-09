"""Durable-shaped staged validation planning and execution.

``ValidationRunner`` is an application seam, not a replacement scheduler or
executor.  It creates one immutable job description per selected gate and
submits those descriptions to an injected WorkItem authority.  A separate
resource adapter admits each job before the fixed-argv executor is called.
Tests use the explicit fake adapters in this module; production callers must
provide adapters backed by graph-os WorkItems and RMDD-27 native reservation
authority.  This lane has no production claim/terminal WorkItem adapter or
executor fence heartbeat yet, so these durable-shaped seams cannot be called
production execution until RMDD-27/RMDD-29 closes that binding.

The runner never mutates a target branch, pushes, or lands.  For feedback and
integration runs against a live lane tree it invokes RMDD-26 ``safe_commit``
when the tree is dirty, then verifies the resulting immutable SHA before any
gate is submitted.  That makes the pre-commit staged-files hazard unreachable
for this lane's own stage-0/1 path.
"""

from __future__ import annotations

import os
import re
import selectors
import subprocess  # nosec B404 - all calls below use fixed argv
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from repository_manager.development import (
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    ResourceRequest,
    ValidationStage,
)
from repository_manager.development.serialization import canonical_digest
from repository_manager.execution.cancellation import CancellationToken
from repository_manager.safe_commit import safe_commit
from repository_manager.validation_evidence import (
    BaselineCache,
    BaselineObservation,
    EvidenceError,
    EvidenceOutcome,
    GateEvidence,
    ValidationCertificate,
    ValidationFailureClass,
    compare_failure_signals,
    verify_certificate,
)
from repository_manager.validation_policy import (
    BaselineMode,
    GateMode,
    TimeoutPolicy,
    ValidationGate,
    ValidationPolicyError,
    ValidationProfile,
)


class ValidationRunnerError(ValueError):
    """A validation request, authority, or execution seam refused work."""


class ValidationAuthorityUnavailable(ValidationRunnerError):
    """No durable WorkItem submitter was supplied for a run."""


class ValidationPreparationError(ValidationRunnerError):
    """The exact worktree could not be prepared and verified."""


class ResourceAdmissionUnavailable(ValidationRunnerError):
    """The resource adapter could not make an authoritative decision."""


_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_SAFE_ID = re.compile(r"^[^\x00\r\n]+$")
_JOB_NAMESPACE = uuid.UUID("b20d5b4e-3c0d-4d07-85e8-1e2358f768aa")
_MAX_GIT_OUTPUT_BYTES = 1 * 1024 * 1024


def _opaque(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationRunnerError(f"{field_name} must be a non-blank string")
    if not _SAFE_ID.fullmatch(value):
        raise ValidationRunnerError(f"{field_name} contains control characters")
    return value


def _sha(value: object, field_name: str) -> str:
    value = _opaque(value, field_name)
    if not _SHA.fullmatch(value):
        raise ValidationRunnerError(f"{field_name} must be a 40-character Git SHA")
    return value


def _digest(value: object, field_name: str) -> str:
    value = _opaque(value, field_name)
    if not _DIGEST.fullmatch(value):
        raise ValidationRunnerError(f"{field_name} must be a 64-character digest")
    return value


def _utc(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValidationRunnerError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


@dataclass(frozen=True, slots=True)
class ValidationRequest:
    """Immutable input for one staged validation workflow."""

    request_id: str
    repository_id: str
    tree_sha: str
    tree_path: str
    profile: ValidationProfile
    stages: tuple[ValidationStage, ...]
    config_digest: str
    toolchain_digest: str
    target_host: str
    generation_id: str | None = None
    base_sha: str | None = None
    changed_paths: tuple[str, ...] | None = None
    snapshot_dirty_lane_tree: bool = True
    resource_digest: str = ""
    owner_id: str = ""
    tenant_id: str = ""
    lane_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "request_id", _opaque(self.request_id, "request_id"))
        object.__setattr__(
            self, "repository_id", _opaque(self.repository_id, "repository_id")
        )
        object.__setattr__(self, "tree_sha", _sha(self.tree_sha, "tree_sha"))
        path = Path(self.tree_path)
        if not path.is_absolute() or ".." in path.parts:
            raise ValidationRunnerError("tree_path must be absolute and traversal-free")
        object.__setattr__(self, "tree_path", str(path))
        if not self.stages:
            raise ValidationRunnerError(
                "validation request must include at least one stage"
            )
        stage_order = {stage: index for index, stage in enumerate(ValidationStage)}
        if len(set(self.stages)) != len(self.stages) or tuple(
            stage_order[stage] for stage in self.stages
        ) != tuple(sorted(stage_order[stage] for stage in self.stages)):
            raise ValidationRunnerError(
                "validation request stages must be ordered and unique"
            )
        object.__setattr__(
            self, "config_digest", _digest(self.config_digest, "config_digest")
        )
        object.__setattr__(
            self, "toolchain_digest", _digest(self.toolchain_digest, "toolchain_digest")
        )
        object.__setattr__(
            self, "target_host", _opaque(self.target_host, "target_host")
        )
        if self.generation_id is not None:
            object.__setattr__(
                self, "generation_id", _opaque(self.generation_id, "generation_id")
            )
        if self.base_sha is not None:
            object.__setattr__(self, "base_sha", _sha(self.base_sha, "base_sha"))
        if self.changed_paths is not None:
            paths = []
            for item in self.changed_paths:
                item = _opaque(item, "changed path")
                path_item = Path(item)
                if path_item.is_absolute() or ".." in path_item.parts:
                    raise ValidationRunnerError(
                        "changed paths must remain inside the worktree"
                    )
                paths.append(item)
            object.__setattr__(self, "changed_paths", tuple(sorted(set(paths))))
        if self.resource_digest:
            object.__setattr__(
                self,
                "resource_digest",
                _digest(self.resource_digest, "resource_digest"),
            )
        else:
            object.__setattr__(
                self,
                "resource_digest",
                canonical_digest(
                    {
                        "profile": self.profile.digest,
                        "gates": tuple(
                            gate.resource_digest for gate in self.profile.gates
                        ),
                    }
                ),
            )
        if self.owner_id:
            object.__setattr__(self, "owner_id", _opaque(self.owner_id, "owner_id"))
        if self.tenant_id:
            object.__setattr__(self, "tenant_id", _opaque(self.tenant_id, "tenant_id"))
        if self.lane_id is not None:
            object.__setattr__(self, "lane_id", _opaque(self.lane_id, "lane_id"))


@dataclass(frozen=True, slots=True)
class ValidationJob:
    """Typed immutable WorkItem submission payload for one gate."""

    job_id: str
    request_id: str
    repository_id: str
    gate_name: str
    stage: ValidationStage
    tree_sha: str
    base_sha: str | None
    worktree_path: str
    command: tuple[str, ...]
    dependencies: tuple[str, ...]
    gate_config_digest: str
    profile_digest: str
    command_digest: str
    toolchain_digest: str
    resource_digest: str
    resource_policy_digest: str
    target_host: str
    generation_id: str | None
    baseline_mode: BaselineMode
    timeout_seconds: int
    resource_request: ResourceRequest

    def __post_init__(self) -> None:
        for name in (
            "job_id",
            "request_id",
            "repository_id",
            "gate_name",
            "worktree_path",
            "target_host",
        ):
            object.__setattr__(self, name, _opaque(getattr(self, name), name))
        object.__setattr__(self, "tree_sha", _sha(self.tree_sha, "tree_sha"))
        if self.base_sha is not None:
            object.__setattr__(self, "base_sha", _sha(self.base_sha, "base_sha"))
        object.__setattr__(
            self,
            "gate_config_digest",
            _digest(self.gate_config_digest, "gate_config_digest"),
        )
        object.__setattr__(
            self, "profile_digest", _digest(self.profile_digest, "profile_digest")
        )
        object.__setattr__(
            self, "command_digest", _digest(self.command_digest, "command_digest")
        )
        object.__setattr__(
            self, "toolchain_digest", _digest(self.toolchain_digest, "toolchain_digest")
        )
        object.__setattr__(
            self, "resource_digest", _digest(self.resource_digest, "resource_digest")
        )
        object.__setattr__(
            self,
            "resource_policy_digest",
            _digest(self.resource_policy_digest, "resource_policy_digest"),
        )
        if self.generation_id is not None:
            object.__setattr__(
                self, "generation_id", _opaque(self.generation_id, "generation_id")
            )
        if self.timeout_seconds < 1:
            raise ValidationRunnerError("job timeout_seconds must be positive")
        if not self.command:
            raise ValidationRunnerError("job command must not be empty")
        object.__setattr__(
            self,
            "command",
            tuple(_opaque(item, "command argv") for item in self.command),
        )
        deps = tuple(_opaque(item, "dependency job id") for item in self.dependencies)
        object.__setattr__(self, "dependencies", tuple(sorted(set(deps))))

    @property
    def input_digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "request_id": self.request_id,
            "repository_id": self.repository_id,
            "gate_name": self.gate_name,
            "stage": self.stage,
            "tree_sha": self.tree_sha,
            "base_sha": self.base_sha,
            "worktree_path": self.worktree_path,
            "command": self.command,
            "dependencies": self.dependencies,
            "gate_config_digest": self.gate_config_digest,
            "profile_digest": self.profile_digest,
            "command_digest": self.command_digest,
            "toolchain_digest": self.toolchain_digest,
            "resource_digest": self.resource_digest,
            "resource_policy_digest": self.resource_policy_digest,
            "target_host": self.target_host,
            "generation_id": self.generation_id,
            "baseline_mode": self.baseline_mode,
            "timeout_seconds": self.timeout_seconds,
            "resource_request": self.resource_request.model_dump(
                mode="json", exclude_none=False
            ),
        }


@dataclass(frozen=True, slots=True)
class ValidationPlan:
    """Sealed gate DAG; membership never changes during execution."""

    request: ValidationRequest
    gates: tuple[ValidationGate, ...]
    jobs: tuple[ValidationJob, ...]
    plan_digest: str


@dataclass(frozen=True, slots=True)
class SubmittedValidationJob:
    """The durable authority's acknowledgement of a gate WorkItem."""

    job_id: str
    state: str = "submitted"
    input_digest: str = ""


@dataclass(frozen=True, slots=True)
class PreparedValidation:
    """Preparation result, including an explicitly deferred snapshot marker."""

    request: ValidationRequest
    snapshot_gate_deferred: bool = False


@runtime_checkable
class ValidationJobAuthority(Protocol):
    """Adapter to graph-os WorkItem authority; no local job store is implied."""

    def submit(self, job: ValidationJob) -> SubmittedValidationJob:
        """Persist one immutable job with its dependency IDs."""

    def cancel(self, job_id: str, *, reason: str) -> bool:
        """Cancel a durable job cooperatively."""


@dataclass(frozen=True, slots=True)
class ResourceLease:
    """Admission proof handed to an executor attempt."""

    reservation_id: str
    fence: str
    host_id: str
    resource_digest: str


@runtime_checkable
class ResourceAdmission(Protocol):
    """Adapter to RMDD-08/RMDD-27 admission authority."""

    def reserve(self, job: ValidationJob) -> ResourceLease | None:
        """Return a durable reservation or refuse before process creation."""

    def release(self, lease: ResourceLease, *, outcome: EvidenceOutcome) -> bool:
        """Release the exact reservation/fence after every attempt."""


@runtime_checkable
class ValidationExecutor(Protocol):
    """Fixed-argv execution seam shared with local and future remote workers."""

    def run(self, command: ExecutionCommand, **kwargs: Any) -> ExecutionResult:
        """Run one command under the supplied fence and cancellation token."""


@runtime_checkable
class BaselineProvider(Protocol):
    """Materialize and execute the same gate against an immutable base SHA."""

    def run(self, job: ValidationJob, command: ExecutionCommand) -> BaselineObservation:
        """Return readable failure signals or an explicit unreadable result."""


@dataclass(frozen=True, slots=True)
class ValidationHandoff:
    """Typed post-land/release handoff without performing the downstream effect."""

    from_stage: ValidationStage
    next_stage: ValidationStage
    repository_id: str
    generation_id: str | None
    tree_sha: str
    evidence_digests: tuple[str, ...]
    required: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ValidationRunResult:
    """Run projection containing evidence and an optional certificate."""

    request: ValidationRequest
    plan: ValidationPlan | None
    submitted: tuple[SubmittedValidationJob, ...] = ()
    evidence: tuple[GateEvidence, ...] = ()
    certificate: ValidationCertificate | None = None
    preparation_error: str | None = None
    snapshot_gate_deferred: bool = False

    @property
    def ok(self) -> bool:
        if self.preparation_error or self.plan is None:
            return False
        blocking = {
            gate.name: gate
            for gate in self.plan.gates
            if gate.mode is GateMode.BLOCKING
        }
        by_name = {item.gate_name: item for item in self.evidence}
        return bool(blocking) and all(
            by_name.get(name) is not None
            and by_name[name].outcome is EvidenceOutcome.PASSED
            for name in blocking
        )

    @property
    def landable(self) -> bool:
        if self.certificate is None:
            return False
        certification = tuple(
            item
            for item in self.evidence
            if item.stage is ValidationStage.CERTIFICATION
        )
        return verify_certificate(self.certificate, certification).valid


class FakeValidationJobAuthority:
    """Explicit test-only WorkItem adapter preserving immutable submissions."""

    def __init__(self) -> None:
        self.jobs: list[ValidationJob] = []
        self.cancelled: list[tuple[str, str]] = []

    def submit(self, job: ValidationJob) -> SubmittedValidationJob:
        existing = next((item for item in self.jobs if item.job_id == job.job_id), None)
        if existing is not None:
            if existing.input_digest != job.input_digest:
                raise ValidationRunnerError("job id conflicts with immutable input")
            return SubmittedValidationJob(
                job_id=job.job_id, input_digest=job.input_digest
            )
        if any(
            dep not in {item.job_id for item in self.jobs} for dep in job.dependencies
        ):
            raise ValidationRunnerError("job dependency has not been submitted")
        self.jobs.append(job)
        return SubmittedValidationJob(job_id=job.job_id, input_digest=job.input_digest)

    def cancel(self, job_id: str, *, reason: str) -> bool:
        self.cancelled.append((job_id, reason))
        return True


class FailClosedResourceAdmission:
    """Default production-safe adapter until RMDD-27 native authority is wired."""

    def reserve(self, job: ValidationJob) -> ResourceLease | None:
        return None

    def release(self, lease: ResourceLease, *, outcome: EvidenceOutcome) -> bool:
        return False


class LocalTestAdmission:
    """Small deterministic admission fake; never claim this as distributed authority."""

    def __init__(self, *, allow: bool = True, host_id: str = "host:test") -> None:
        self.allow = allow
        self.host_id = host_id
        self.reserved: list[ResourceLease] = []
        self.released: list[tuple[ResourceLease, EvidenceOutcome]] = []

    def reserve(self, job: ValidationJob) -> ResourceLease | None:
        if not self.allow:
            return None
        lease = ResourceLease(
            reservation_id=f"reservation:{job.job_id}",
            fence=f"fence:{job.job_id}",
            host_id=self.host_id,
            resource_digest=job.resource_digest,
        )
        self.reserved.append(lease)
        return lease

    def release(self, lease: ResourceLease, *, outcome: EvidenceOutcome) -> bool:
        self.released.append((lease, outcome))
        return True


class UnreadableBaselineProvider:
    """Default differential provider: fail closed instead of allowing-all."""

    def __init__(self, *, reason: str = "no baseline provider was configured") -> None:
        self.reason = reason

    def run(self, job: ValidationJob, command: ExecutionCommand) -> BaselineObservation:
        return BaselineObservation(
            readable=False, tree_sha=job.tree_sha, detail=self.reason
        )


class ValidationRunner:
    """Plan, submit, and execute staged validation through injected authorities."""

    def __init__(
        self,
        *,
        job_authority: ValidationJobAuthority | None = None,
        resource_admission: ResourceAdmission | None = None,
        executor: ValidationExecutor | None = None,
        baseline_provider: BaselineProvider | None = None,
        baseline_cache: BaselineCache | None = None,
        safe_commit_fn: Callable[..., Mapping[str, Any]] = safe_commit,
        worker_id: str = "worker:validation",
    ) -> None:
        self.job_authority = job_authority
        self.resource_admission = resource_admission or FailClosedResourceAdmission()
        self.executor = executor
        self.baseline_provider = baseline_provider or UnreadableBaselineProvider()
        self.baseline_cache = baseline_cache or BaselineCache()
        self.safe_commit_fn = safe_commit_fn
        self.worker_id = _opaque(worker_id, "worker_id")

    def plan(self, request: ValidationRequest) -> ValidationPlan:
        """Resolve changed paths and seal a dependency-linked gate DAG."""

        try:
            selected = list(
                request.profile.gates_for(request.changed_paths, stages=request.stages)
            )
        except (KeyError, ValueError, ValidationPolicyError) as exc:
            raise ValidationRunnerError(str(exc)) from exc
        by_name = {gate.name: gate for gate in selected}
        # Artifact dependencies are part of the contract even when a path
        # selector would otherwise omit the producer.
        pending = list(selected)
        while pending:
            gate = pending.pop()
            for dependency in gate.artifact_dependencies:
                source = next(
                    (
                        candidate
                        for candidate in request.profile.gates
                        if candidate.name == dependency
                    ),
                    None,
                )
                if source is None:
                    raise ValidationRunnerError(
                        f"gate {gate.name!r} dependency {dependency!r} is not declared"
                    )
                if source.name not in by_name:
                    by_name[source.name] = source
                    pending.append(source)
        gates = tuple(gate for gate in request.profile.gates if gate.name in by_name)
        jobs: list[ValidationJob] = []
        by_gate: dict[str, ValidationJob] = {}
        stage_order = {stage: index for index, stage in enumerate(ValidationStage)}
        for gate in gates:
            digest_material = {
                "request_id": request.request_id,
                "repository_id": request.repository_id,
                "tree_sha": request.tree_sha,
                "gate": gate.canonical_payload(),
                "config_digest": request.config_digest,
                "profile_digest": request.profile.digest,
                "toolchain_digest": request.toolchain_digest,
                "target_host": request.target_host,
                "generation_id": request.generation_id,
            }
            job_uuid = uuid.uuid5(_JOB_NAMESPACE, canonical_digest(digest_material))
            job_id = f"rmjob:{job_uuid}"
            dependencies = set(gate.artifact_dependencies)
            dependencies.update(
                previous.name
                for previous in gates
                if stage_order[previous.stage] < stage_order[gate.stage]
            )
            dependency_ids = tuple(
                sorted(by_gate[name].job_id for name in dependencies if name in by_gate)
            )
            job = ValidationJob(
                job_id=job_id,
                request_id=request.request_id,
                repository_id=request.repository_id,
                gate_name=gate.name,
                stage=gate.stage,
                tree_sha=request.tree_sha,
                worktree_path=request.tree_path,
                command=gate.command,
                dependencies=dependency_ids,
                gate_config_digest=request.config_digest,
                profile_digest=request.profile.digest,
                command_digest=canonical_digest(gate.command),
                toolchain_digest=request.toolchain_digest,
                resource_digest=gate.resource_digest,
                resource_policy_digest=request.resource_digest,
                target_host=request.target_host,
                base_sha=request.base_sha,
                generation_id=request.generation_id,
                baseline_mode=gate.baseline_mode,
                timeout_seconds=gate.timeout_seconds,
                resource_request=gate.resources,
            )
            jobs.append(job)
            by_gate[gate.name] = job
        plan_digest = canonical_digest(
            {
                "request": {
                    "request_id": request.request_id,
                    "repository_id": request.repository_id,
                    "tree_sha": request.tree_sha,
                    "tree_path": request.tree_path,
                    "stages": request.stages,
                    "config_digest": request.config_digest,
                    "profile_digest": request.profile.digest,
                    "toolchain_digest": request.toolchain_digest,
                    "target_host": request.target_host,
                    "generation_id": request.generation_id,
                    "base_sha": request.base_sha,
                    "changed_paths": request.changed_paths,
                },
                "jobs": tuple(job.canonical_payload() for job in jobs),
            }
        )
        return ValidationPlan(
            request=request, gates=gates, jobs=tuple(jobs), plan_digest=plan_digest
        )

    def submit(self, plan: ValidationPlan) -> tuple[SubmittedValidationJob, ...]:
        """Submit every sealed job to the one durable WorkItem authority."""

        if self.job_authority is None:
            raise ValidationAuthorityUnavailable(
                "validation requires a graph-os WorkItem authority; no local job store is allowed"
            )
        submitted: list[SubmittedValidationJob] = []
        try:
            for job in plan.jobs:
                result = self.job_authority.submit(job)
                if result.job_id != job.job_id:
                    raise ValidationRunnerError(
                        f"durable authority changed immutable job ID for {job.gate_name}"
                    )
                if result.input_digest != job.input_digest:
                    raise ValidationRunnerError(
                        f"durable authority changed immutable input for {job.gate_name}"
                    )
                submitted.append(result)
        except Exception as exc:
            cancellation_failures: list[str] = []
            for previous in submitted:
                try:
                    if not self.job_authority.cancel(
                        previous.job_id,
                        reason="validation submission failed; canceling prefix",
                    ):
                        cancellation_failures.append(previous.job_id)
                except Exception as cancel_exc:
                    cancellation_failures.append(
                        f"{previous.job_id} ({type(cancel_exc).__name__})"
                    )
            detail = (
                f"validation job submission failed after {len(submitted)} job(s): "
                f"{type(exc).__name__}: {exc}"
            )
            if cancellation_failures:
                detail += "; submission reconciliation failed for job(s): " + ", ".join(
                    cancellation_failures
                )
            elif submitted:
                detail += "; submitted prefix was cooperatively canceled"
            raise ValidationRunnerError(detail) from exc
        return tuple(submitted)

    def run(
        self,
        request: ValidationRequest,
        *,
        cancellation: CancellationToken | None = None,
    ) -> ValidationRunResult:
        """Prepare, submit, and execute a staged plan without landing or pushing."""

        token = cancellation or CancellationToken()
        try:
            preparation = self._prepare_request(request)
            prepared = preparation.request
            plan = self.plan(prepared)
        except (
            ValidationRunnerError,
            ValidationPreparationError,
            OSError,
            ValueError,
        ) as exc:
            return ValidationRunResult(
                request=request,
                plan=None,
                preparation_error=str(exc),
            )
        if self.executor is None:
            return ValidationRunResult(
                request=prepared,
                plan=plan,
                preparation_error="no fixed-argv validation executor was configured",
                snapshot_gate_deferred=preparation.snapshot_gate_deferred,
            )
        try:
            submitted = self.submit(plan)
        except (
            ValidationRunnerError,
            ValidationPreparationError,
            OSError,
            ValueError,
        ) as exc:
            return ValidationRunResult(
                request=prepared,
                plan=plan,
                preparation_error=str(exc),
                snapshot_gate_deferred=preparation.snapshot_gate_deferred,
            )
        evidence: list[GateEvidence] = []
        failed_jobs: set[str] = set()
        by_name = {gate.name: gate for gate in plan.gates}
        for job in plan.jobs:
            gate = by_name[job.gate_name]
            if any(dep in failed_jobs for dep in job.dependencies):
                item = self._blocked_evidence(job, gate)
                if preparation.snapshot_gate_deferred:
                    item = replace(item, snapshot_gate_deferred=True)
                evidence.append(item)
                failed_jobs.add(job.job_id)
                continue
            item = self._execute_gate(job, gate, token)
            if preparation.snapshot_gate_deferred:
                item = replace(
                    item,
                    snapshot_gate_deferred=True,
                    snapshot_gate_replayed=(
                        gate.runs_precommit and item.outcome is EvidenceOutcome.PASSED
                    ),
                )
            evidence.append(item)
            if (
                item.outcome is not EvidenceOutcome.PASSED
                and gate.mode is GateMode.BLOCKING
            ):
                failed_jobs.add(job.job_id)
        certificate = self._certificate_if_ready(prepared, plan, evidence)
        return ValidationRunResult(
            request=prepared,
            plan=plan,
            submitted=submitted,
            evidence=tuple(evidence),
            certificate=certificate,
            snapshot_gate_deferred=preparation.snapshot_gate_deferred,
        )

    def _prepare_request(self, request: ValidationRequest) -> PreparedValidation:
        tree = Path(request.tree_path)
        if not tree.exists() or not tree.is_dir():
            raise ValidationPreparationError(f"validation tree does not exist: {tree}")
        snapshot_gate_deferred = False
        resolved = tree.resolve(strict=True)
        if resolved != tree:
            raise ValidationPreparationError(
                "validation tree path resolves through a symlink; refusing ambiguous worktree"
            )
        current = self._git_output(["rev-parse", "--show-toplevel"], tree)
        if current != str(tree):
            raise ValidationPreparationError(
                "tree path is not the Git worktree top-level"
            )
        head = self._git_output(["rev-parse", "HEAD"], tree)
        if head != request.tree_sha:
            raise ValidationPreparationError(
                f"tree SHA moved before validation: expected {request.tree_sha}, found {head}"
            )
        status = self._git_output(["status", "--porcelain"], tree, allow_error=True)
        if status is None:
            raise ValidationPreparationError("could not inspect worktree status")
        if status and any(
            stage in request.stages
            for stage in (ValidationStage.FEEDBACK, ValidationStage.INTEGRATION)
        ):
            if not request.snapshot_dirty_lane_tree:
                raise ValidationPreparationError(
                    "stage-0/1 validation refuses a dirty tree without safe_commit"
                )
            installed_hook = self._installed_precommit_hook(tree)
            if installed_hook is None:
                raise ValidationPreparationError(
                    "could not inspect Git pre-commit hook path before snapshot"
                )
            has_config = (tree / ".pre-commit-config.yaml").is_file()
            if installed_hook and not has_config:
                raise ValidationPreparationError(
                    "installed Git pre-commit hook has no admitted profile gate"
                )
            if has_config:
                result = self.safe_commit_fn(
                    tree,
                    f"RMDD-11 validation snapshot {request.request_id}",
                    defer_gate=True,
                )
                snapshot_gate_deferred = True
            else:
                result = self.safe_commit_fn(
                    tree,
                    f"RMDD-11 validation snapshot {request.request_id}",
                )
            if not result.get("ok"):
                raise ValidationPreparationError(
                    str(
                        result.get("error")
                        or "safe_commit refused the validation snapshot"
                    )
                )
            if has_config and result.get("gate_deferred") is not True:
                raise ValidationPreparationError(
                    "configured snapshot must explicitly defer its gate"
                )
            snapshot_sha = str(result.get("commit_sha") or "")
            if not _SHA.fullmatch(snapshot_sha):
                raise ValidationPreparationError(
                    "safe_commit did not return an immutable commit SHA"
                )
            request = replace(request, tree_sha=snapshot_sha)
        status_after = self._git_output(
            ["status", "--porcelain"], tree, allow_error=True
        )
        if status_after is None or status_after:
            raise ValidationPreparationError(
                "validation gate requires a clean committed tree after preparation"
            )
        head_after = self._git_output(["rev-parse", "HEAD"], tree)
        if head_after != request.tree_sha:
            raise ValidationPreparationError(
                "worktree changed during validation preparation"
            )
        changed_paths = self._derive_changed_paths(request, tree)
        return PreparedValidation(
            request=replace(request, changed_paths=changed_paths),
            snapshot_gate_deferred=snapshot_gate_deferred,
        )

    @classmethod
    def _installed_precommit_hook(cls, tree: Path) -> bool | None:
        """Return whether Git would run a pre-commit hook during ``git commit``."""

        hooks = cls._git_output(
            ["rev-parse", "--git-path", "hooks"], tree, allow_error=True
        )
        if hooks is None:
            return None
        hook_path = Path(hooks)
        if not hook_path.is_absolute():
            hook_path = tree / hook_path
        return (hook_path / "pre-commit").is_file()

    @staticmethod
    def _bounded_git(
        args: Sequence[str],
        tree: Path,
        *,
        max_bytes: int = _MAX_GIT_OUTPUT_BYTES,
        timeout_seconds: float = 30.0,
    ) -> tuple[int, bytes, bytes] | None:
        """Run fixed-argv Git with bounded, streaming stdout/stderr."""

        try:
            process = subprocess.Popen(  # nosec B603 - argv is constructed locally
                ["git", *args],
                cwd=str(tree),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
            )
        except OSError:
            return None
        if process.stdout is None or process.stderr is None:
            process.kill()
            process.wait()
            return None
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ, "stdout")
        selector.register(process.stderr, selectors.EVENT_READ, "stderr")
        buffers: dict[str, bytearray] = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + timeout_seconds
        overflow = False
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    process.kill()
                    process.wait()
                    return None
                events = selector.select(timeout=min(remaining, 0.25))
                if not events:
                    continue
                for key, _ in events:
                    chunk = os.read(key.fd, 64 * 1024)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    stream = str(key.data)
                    if len(buffers[stream]) + len(chunk) > max_bytes:
                        overflow = True
                        process.kill()
                        break
                    buffers[stream].extend(chunk)
                if overflow:
                    break
        except OSError:
            process.kill()
            process.wait()
            return None
        finally:
            selector.close()
        if overflow:
            process.wait()
            return None
        try:
            returncode = process.wait(timeout=max(0.0, deadline - time.monotonic()))
        except (OSError, subprocess.TimeoutExpired):
            process.kill()
            process.wait()
            return None
        return returncode, bytes(buffers["stdout"]), bytes(buffers["stderr"])

    @classmethod
    def _git_output(
        cls, args: list[str], tree: Path, *, allow_error: bool = False
    ) -> str | None:
        result = cls._bounded_git(args, tree)
        if result is None:
            return None
        returncode, stdout, stderr = result
        if returncode != 0:
            if allow_error:
                return None
            detail = stderr.decode("utf-8", "replace").strip()
            raise ValidationPreparationError(
                f"git {' '.join(args)} refused validation: {detail}"
            )
        return stdout.decode("utf-8", "surrogateescape").strip()

    @classmethod
    def _derive_changed_paths(
        cls, request: ValidationRequest, tree: Path
    ) -> tuple[str, ...] | None:
        """Derive changed paths from immutable Git objects, fail-open to all gates."""

        if request.base_sha is None:
            return None
        result = cls._bounded_git(
            [
                "diff",
                "--name-status",
                "--find-renames",
                "--find-copies",
                "--no-ext-diff",
                "-z",
                request.base_sha,
                request.tree_sha,
                "--",
            ],
            tree,
        )
        if result is None:
            return None
        returncode, stdout, _ = result
        if returncode != 0:
            return None
        fields = stdout.split(b"\0")
        if fields and fields[-1] == b"":
            fields.pop()
        paths: list[str] = []
        index = 0
        try:
            while index < len(fields):
                status = fields[index].decode("ascii")
                index += 1
                required = 2 if status.startswith(("R", "C")) else 1
                if index + required > len(fields):
                    return None
                for raw_path in fields[index : index + required]:
                    path = raw_path.decode("utf-8", "surrogateescape")
                    candidate = Path(path)
                    if not path or candidate.is_absolute() or ".." in candidate.parts:
                        return None
                    paths.append(path)
                index += required
        except (UnicodeError, ValueError):
            return None
        return tuple(sorted(set(paths)))

    def _execute_gate(
        self,
        job: ValidationJob,
        gate: ValidationGate,
        cancellation: CancellationToken,
    ) -> GateEvidence:
        started = datetime.now(UTC)
        if cancellation.is_cancelled():
            return self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.CANCELLED,
                failure_class=ValidationFailureClass.CANCELLATION,
                detail=cancellation.snapshot().reason
                or "validation cancelled before admission",
            )
        tree = Path(job.worktree_path)
        try:
            current = self._git_output(["rev-parse", "HEAD"], tree)
            if current != job.tree_sha:
                return self._simple_evidence(
                    job,
                    gate,
                    started,
                    outcome=EvidenceOutcome.REFUSED,
                    failure_class=ValidationFailureClass.STALE_TREE,
                    detail="worktree HEAD no longer matches the sealed job tree SHA",
                )
            status = self._git_output(["status", "--porcelain"], tree, allow_error=True)
            if status is None or status:
                return self._simple_evidence(
                    job,
                    gate,
                    started,
                    outcome=EvidenceOutcome.REFUSED,
                    failure_class=ValidationFailureClass.STALE_TREE,
                    detail="gate worktree is dirty or status is unreadable",
                )
            command = ExecutionCommand(
                argv=gate.command,
                workdir=str(tree),
                environment_refs=gate.command_env_refs,
                timeout_seconds=gate.timeout_seconds,
            )
        except (ValidationPreparationError, ValueError) as exc:
            return self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.REFUSED,
                failure_class=ValidationFailureClass.INVALID_REQUEST,
                detail=str(exc),
            )
        try:
            lease = self.resource_admission.reserve(job)
        except Exception as exc:
            return self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.DEFERRED,
                failure_class=ValidationFailureClass.RESOURCE,
                detail=(
                    "resource authority refused before process creation: "
                    f"{type(exc).__name__}"
                ),
            )
        if lease is None:
            return self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.DEFERRED,
                failure_class=ValidationFailureClass.RESOURCE,
                detail="resource admission refused before process creation",
            )
        if lease.host_id != job.target_host:
            evidence = self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.REFUSED,
                failure_class=ValidationFailureClass.STALE_FENCE,
                detail=(
                    "resource reservation host does not match the immutable target: "
                    f"expected {job.target_host}, got {lease.host_id}"
                ),
            )
            try:
                release_ok = self.resource_admission.release(
                    lease, outcome=evidence.outcome
                )
            except Exception:
                release_ok = False
            if not release_ok:
                evidence = replace(
                    evidence,
                    failure_class=ValidationFailureClass.RECONCILIATION,
                    detail=evidence.detail + "; resource release was not confirmed",
                )
            return evidence
        if lease.resource_digest != job.resource_digest:
            evidence = self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.REFUSED,
                failure_class=ValidationFailureClass.STALE_FENCE,
                detail="resource reservation does not match the immutable gate request",
            )
            try:
                release_ok = self.resource_admission.release(
                    lease, outcome=evidence.outcome
                )
            except Exception:
                release_ok = False
            if not release_ok:
                evidence = replace(
                    evidence,
                    failure_class=ValidationFailureClass.RECONCILIATION,
                    detail=evidence.detail + "; resource release was not confirmed",
                )
            return evidence
        result: ExecutionResult | None = None
        executor = self.executor
        if executor is None:
            evidence = self._simple_evidence(
                job,
                gate,
                started,
                outcome=EvidenceOutcome.FAILED,
                failure_class=ValidationFailureClass.ENVIRONMENT,
                detail="no fixed-argv validation executor was configured",
            )
            try:
                release_ok = self.resource_admission.release(
                    lease, outcome=evidence.outcome
                )
            except Exception:
                release_ok = False
            if not release_ok:
                evidence = replace(
                    evidence,
                    outcome=EvidenceOutcome.REFUSED,
                    failure_class=ValidationFailureClass.RECONCILIATION,
                    detail=evidence.detail + "; resource release was not confirmed",
                )
            return evidence
        evidence = self._simple_evidence(
            job,
            gate,
            started,
            outcome=EvidenceOutcome.FAILED,
            failure_class=ValidationFailureClass.ENVIRONMENT,
            detail="validation evidence was not produced",
        )
        release_ok = False
        try:
            try:
                result = executor.run(
                    command,
                    command_id=job.job_id,
                    worker_id=self.worker_id,
                    fence=lease.fence,
                    cancellation=cancellation,
                )
            except Exception as exc:  # executor boundary classifies worker faults
                evidence = self._simple_evidence(
                    job,
                    gate,
                    started,
                    outcome=EvidenceOutcome.FAILED,
                    failure_class=ValidationFailureClass.ENVIRONMENT,
                    detail=f"executor raised {type(exc).__name__}: {exc}",
                )
            else:
                if result.command_id != job.job_id or result.fence != lease.fence:
                    evidence = self._simple_evidence(
                        job,
                        gate,
                        started,
                        outcome=EvidenceOutcome.REFUSED,
                        failure_class=ValidationFailureClass.STALE_FENCE,
                        detail="executor result command or fence does not match the leased job",
                    )
                else:
                    try:
                        current_after = self._git_output(["rev-parse", "HEAD"], tree)
                        status_after = self._git_output(
                            ["status", "--porcelain"], tree, allow_error=True
                        )
                        if (
                            current_after != job.tree_sha
                            or status_after is None
                            or status_after
                        ):
                            evidence = self._simple_evidence(
                                job,
                                gate,
                                started,
                                outcome=EvidenceOutcome.REFUSED,
                                failure_class=ValidationFailureClass.STALE_TREE,
                                detail="worktree changed while the validation command was running",
                            )
                        else:
                            evidence = self._evidence_from_result(
                                job, gate, command, result, started
                            )
                    except Exception as exc:
                        evidence = self._simple_evidence(
                            job,
                            gate,
                            started,
                            outcome=EvidenceOutcome.FAILED,
                            failure_class=ValidationFailureClass.ENVIRONMENT,
                            detail=(
                                "validation evidence collection raised "
                                f"{type(exc).__name__}: {exc}"
                            ),
                        )
        finally:
            try:
                release_ok = self.resource_admission.release(
                    lease, outcome=evidence.outcome
                )
            except Exception:
                release_ok = False
        if not release_ok:
            # A successful gate with an unreleased reservation is not safe
            # evidence: it can strand capacity or permit a duplicate effect.
            evidence = replace(
                evidence,
                outcome=EvidenceOutcome.REFUSED,
                failure_class=ValidationFailureClass.RECONCILIATION,
                detail=(evidence.detail + "; " if evidence.detail else "")
                + "resource release was not confirmed",
            )
        return evidence

    @staticmethod
    def _outcome_from_result(result: ExecutionResult) -> EvidenceOutcome:
        return {
            ExecutionOutcome.SUCCEEDED: EvidenceOutcome.PASSED,
            ExecutionOutcome.FAILED: EvidenceOutcome.FAILED,
            ExecutionOutcome.CANCELLED: EvidenceOutcome.CANCELLED,
            ExecutionOutcome.TIMED_OUT: EvidenceOutcome.TIMED_OUT,
            ExecutionOutcome.REFUSED: EvidenceOutcome.REFUSED,
        }[result.outcome]

    def _evidence_from_result(
        self,
        job: ValidationJob,
        gate: ValidationGate,
        command: ExecutionCommand,
        result: ExecutionResult,
        started: datetime,
    ) -> GateEvidence:
        outcome = self._outcome_from_result(result)
        if (
            outcome is EvidenceOutcome.TIMED_OUT
            and gate.timeout_policy is TimeoutPolicy.DEFER
        ):
            outcome = EvidenceOutcome.DEFERRED
        finished = result.finished_at
        failure_class = self._failure_class(result)
        candidate_ids = self._failure_ids(gate, result)
        baseline: BaselineObservation | None = None
        verdict = None
        if gate.baseline_mode is BaselineMode.DIFFERENTIAL:
            if not self._base_sha_available(job):
                baseline = BaselineObservation(
                    readable=False,
                    tree_sha=job.tree_sha,
                    detail="request did not identify an immutable base SHA",
                )
            else:
                cache_identity = {
                    "base_sha": self._base_sha(job),
                    "gate_config_digest": job.gate_config_digest,
                    "command_digest": job.command_digest,
                    "toolchain_digest": job.toolchain_digest,
                    "target_host": job.target_host,
                }
                baseline = self._baseline_cache_get(cache_identity)
                if baseline is None:
                    baseline = self.baseline_provider.run(job, command)
                    if baseline.tree_sha != self._base_sha(job):
                        baseline = BaselineObservation(
                            readable=False,
                            tree_sha=self._base_sha(job),
                            detail="baseline provider returned a different tree SHA",
                        )
                    elif baseline.toolchain_digest != job.toolchain_digest:
                        baseline = BaselineObservation(
                            readable=False,
                            tree_sha=self._base_sha(job),
                            detail=(
                                "baseline provider did not return the exact "
                                "toolchain digest"
                            ),
                        )
                    self._baseline_cache_put(baseline, cache_identity)
            if outcome in {EvidenceOutcome.PASSED, EvidenceOutcome.FAILED}:
                verdict = compare_failure_signals(
                    mode=gate.baseline_mode,
                    baseline=baseline,
                    candidate_exit_code=result.exit_code,
                    candidate_failure_ids=candidate_ids,
                )
                if verdict.ok:
                    outcome = EvidenceOutcome.PASSED
                    failure_class = None
                else:
                    outcome = (
                        EvidenceOutcome.REFUSED
                        if not verdict.baseline_readable
                        else EvidenceOutcome.FAILED
                    )
                    failure_class = (
                        ValidationFailureClass.BASELINE_UNPRODUCIBLE
                        if not verdict.baseline_readable
                        else ValidationFailureClass.CODE
                    )
        detail = result.stderr_tail or result.stdout_tail
        if verdict is not None:
            detail = verdict.detail + (f"; {detail}" if detail else "")
        return GateEvidence(
            evidence_id=f"evidence:{job.job_id}",
            gate_name=gate.name,
            stage=gate.stage,
            tree_sha=job.tree_sha,
            generation_id=job.generation_id,
            gate_config_digest=job.gate_config_digest,
            profile_digest=job.profile_digest,
            command_digest=job.command_digest,
            target_host=job.target_host,
            toolchain_digest=job.toolchain_digest,
            resource_digest=job.resource_policy_digest,
            started_at=result.started_at if result.started_at else started,
            finished_at=finished,
            outcome=outcome,
            failure_class=failure_class,
            job_id=job.job_id,
            dependency_job_ids=job.dependencies,
            baseline_tree_sha=baseline.tree_sha if baseline is not None else None,
            baseline_readable=baseline.readable if baseline is not None else None,
            differential=gate.baseline_mode is BaselineMode.DIFFERENTIAL,
            failure_ids=()
            if outcome is EvidenceOutcome.PASSED
            else (
                verdict.new_failure_ids if verdict is not None else tuple(candidate_ids)
            ),
            pre_existing_failure_ids=verdict.pre_existing_failure_ids
            if verdict
            else (),
            fixed_failure_ids=verdict.fixed_failure_ids if verdict else (),
            log_refs=tuple(item.content_address for item in result.log_refs),
            artifact_refs=tuple(item.content_address for item in result.artifact_refs),
            stdout_tail=result.stdout_tail,
            stderr_tail=result.stderr_tail,
            exit_code=result.exit_code,
            detail=detail[:4000],
        )

    @staticmethod
    def _base_sha_available(job: ValidationJob) -> bool:
        return bool(job.base_sha)

    @staticmethod
    def _base_sha(job: ValidationJob) -> str:
        return job.base_sha or job.tree_sha

    def _baseline_cache_get(
        self, identity: Mapping[str, str]
    ) -> BaselineObservation | None:
        return self.baseline_cache.get(**dict(identity))

    def _baseline_cache_put(
        self, observation: BaselineObservation, identity: Mapping[str, str]
    ) -> None:
        self.baseline_cache.put(observation, **dict(identity))

    @staticmethod
    def _failure_class(result: ExecutionResult) -> ValidationFailureClass | None:
        if result.outcome is ExecutionOutcome.CANCELLED:
            return ValidationFailureClass.CANCELLATION
        if result.outcome is ExecutionOutcome.TIMED_OUT:
            return ValidationFailureClass.TIMEOUT
        if result.outcome is ExecutionOutcome.REFUSED:
            return ValidationFailureClass.STALE_FENCE
        if result.failure_class is None:
            return (
                None
                if result.outcome is ExecutionOutcome.SUCCEEDED
                else ValidationFailureClass.CODE
            )
        if result.failure_class.value == "worker_environment_failure":
            return ValidationFailureClass.ENVIRONMENT
        if result.failure_class.value == "stale_fence_duplicate_effect":
            return ValidationFailureClass.STALE_FENCE
        if result.failure_class.value == "cancelled_deadline":
            return ValidationFailureClass.CANCELLATION
        return ValidationFailureClass.CODE

    @staticmethod
    def _failure_ids(gate: ValidationGate, result: ExecutionResult) -> tuple[str, ...]:
        text = f"{result.stdout_tail}\n{result.stderr_tail}"
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if gate.compare == "pytest-ids":
            ids = []
            for line in lines:
                if "FAILED " in line:
                    ids.append(line.split("FAILED ", 1)[1].split(" - ", 1)[0].strip())
            return tuple(sorted(set(ids)))
        if gate.compare == "exit":
            return ()
        # Keep diagnostics bounded and deterministic.  The full output remains
        # in the executor's bounded log/artifact references.
        return tuple(sorted(set(lines[-64:])))

    def _simple_evidence(
        self,
        job: ValidationJob,
        gate: ValidationGate,
        started: datetime,
        *,
        outcome: EvidenceOutcome,
        failure_class: ValidationFailureClass | None,
        detail: str,
    ) -> GateEvidence:
        finished = datetime.now(UTC)
        baseline_sha = (
            self._base_sha(job)
            if gate.baseline_mode is BaselineMode.DIFFERENTIAL
            else None
        )
        return GateEvidence(
            evidence_id=f"evidence:{job.job_id}",
            gate_name=gate.name,
            stage=gate.stage,
            tree_sha=job.tree_sha,
            generation_id=job.generation_id,
            gate_config_digest=job.gate_config_digest,
            profile_digest=job.profile_digest,
            command_digest=job.command_digest,
            target_host=job.target_host,
            toolchain_digest=job.toolchain_digest,
            resource_digest=job.resource_policy_digest,
            started_at=started,
            finished_at=finished,
            outcome=outcome,
            failure_class=failure_class,
            job_id=job.job_id,
            dependency_job_ids=job.dependencies,
            baseline_tree_sha=baseline_sha,
            baseline_readable=None,
            differential=gate.baseline_mode is BaselineMode.DIFFERENTIAL,
            detail=detail,
        )

    def _blocked_evidence(
        self, job: ValidationJob, gate: ValidationGate
    ) -> GateEvidence:
        return self._simple_evidence(
            job,
            gate,
            datetime.now(UTC),
            outcome=EvidenceOutcome.SKIPPED,
            failure_class=ValidationFailureClass.DEPENDENCY,
            detail="blocking dependency did not pass; gate was not executed",
        )

    def _certificate_if_ready(
        self,
        request: ValidationRequest,
        plan: ValidationPlan,
        evidence: Sequence[GateEvidence],
    ) -> ValidationCertificate | None:
        cert_gates = tuple(
            gate for gate in plan.gates if gate.stage is ValidationStage.CERTIFICATION
        )
        if not cert_gates or request.generation_id is None:
            return None
        blocking = tuple(
            gate.name for gate in cert_gates if gate.mode is GateMode.BLOCKING
        )
        records = tuple(
            item for item in evidence if item.stage is ValidationStage.CERTIFICATION
        )
        if not blocking or any(
            next((item for item in records if item.gate_name == name), None) is None
            for name in blocking
        ):
            return None
        if any(
            item.outcome is not EvidenceOutcome.PASSED
            for item in records
            if item.gate_name in blocking
        ):
            return None
        try:
            return ValidationCertificate.issue(
                certificate_id=f"certificate:{request.generation_id}:{request.tree_sha}",
                generation_id=request.generation_id,
                tree_sha=request.tree_sha,
                gate_config_digest=request.config_digest,
                toolchain_digest=request.toolchain_digest,
                target_host=request.target_host,
                resource_digest=request.resource_digest,
                blocking_gate_names=blocking,
                evidence=records,
                issued_at=datetime.now(UTC),
                profile_digest=request.profile.digest,
            )
        except EvidenceError:
            return None

    @staticmethod
    def post_land_smoke_handoff(
        request: ValidationRequest,
        *,
        tree_sha: str,
        evidence: Sequence[GateEvidence],
    ) -> ValidationHandoff:
        return ValidationHandoff(
            from_stage=ValidationStage.SMOKE,
            next_stage=ValidationStage.RELEASE,
            repository_id=request.repository_id,
            generation_id=request.generation_id,
            tree_sha=_sha(tree_sha, "tree_sha"),
            evidence_digests=tuple(item.digest for item in evidence),
            required=True,
            reason="post-land smoke evidence is ready for the frozen release DAG",
        )

    @staticmethod
    def release_handoff(
        request: ValidationRequest,
        *,
        tree_sha: str,
        evidence: Sequence[GateEvidence],
    ) -> ValidationHandoff:
        return ValidationHandoff(
            from_stage=ValidationStage.RELEASE,
            next_stage=ValidationStage.RELEASE,
            repository_id=request.repository_id,
            generation_id=request.generation_id,
            tree_sha=_sha(tree_sha, "tree_sha"),
            evidence_digests=tuple(item.digest for item in evidence),
            required=True,
            reason="release evidence is a handoff only; workspace mutation and push remain separate",
        )


__all__ = [
    "BaselineProvider",
    "FailClosedResourceAdmission",
    "FakeValidationJobAuthority",
    "LocalTestAdmission",
    "ResourceAdmission",
    "ResourceAdmissionUnavailable",
    "ResourceLease",
    "PreparedValidation",
    "SubmittedValidationJob",
    "ValidationAuthorityUnavailable",
    "ValidationExecutor",
    "ValidationHandoff",
    "ValidationJob",
    "ValidationJobAuthority",
    "ValidationPlan",
    "ValidationPreparationError",
    "ValidationRequest",
    "ValidationRunResult",
    "ValidationRunner",
    "ValidationRunnerError",
    "UnreadableBaselineProvider",
]
