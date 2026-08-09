"""Durable build worker using WorkItem fencing and RMDD-08 admission."""

from __future__ import annotations

import hashlib
import os
import stat
import subprocess
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any, Protocol

from repository_manager import build_queue as bq
from repository_manager.build_artifacts import (
    ArtifactFenceLost,
    ArtifactStoreError,
    BuildArtifactStore,
)
from repository_manager.build_service import (
    BuildExecutionDescriptor,
    descriptor_input_digest,
)
from repository_manager.config_schema import load_yaml_mapping_text
from repository_manager.development import (
    DurableJobView,
    ExecutionCommand,
    ExecutionOutcome,
    FailureClass,
    JobState,
    ResourceRequest,
)
from repository_manager.execution import (
    CancellationToken,
    CommandExecutor,
    LocalExecutor,
)
from repository_manager.resource_scheduler import (
    AdmissionDecision,
    AdmissionRequest,
    AdmissionStatus,
    ResourceScheduler,
)


class BuildWorkerError(RuntimeError):
    """A worker could not claim, admit, execute, or publish a build."""


_MAX_CONFIG_BYTES = 1 << 20
_DEFAULT_STALE_STAGE_AGE_SECONDS = 24 * 60 * 60


def _read_bounded_regular_file(path: Path, limit: int) -> bytes:
    """Read one bounded config snapshot without following a symlink swap."""

    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise BuildWorkerError("build config is not a regular file")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise BuildWorkerError("build config exceeds the durable bound")
        return raw
    except OSError as exc:
        raise BuildWorkerError("build config could not be read") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _config_snapshot(
    scope: Any,
    *,
    base_sha: str,
    dirty: bool,
) -> tuple[bq.BuildConfig, str]:
    """Load the config from the submitted snapshot, not a mutable checkout."""

    if dirty:
        path = scope.tree / bq.CONFIG_FILENAME
        try:
            raw = _read_bounded_regular_file(path, _MAX_CONFIG_BYTES)
        except BuildWorkerError as exc:
            raise BuildWorkerError(
                "dirty build config could not be read for execution verification"
            ) from exc
        source = str(path)
    else:
        process = subprocess.Popen(
            ["git", "show", f"{base_sha}:{bq.CONFIG_FILENAME}"],
            cwd=str(scope.main_tree),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        try:
            assert process.stdout is not None
            raw = process.stdout.read(_MAX_CONFIG_BYTES + 1)
            if len(raw) > _MAX_CONFIG_BYTES:
                process.kill()
                process.wait(timeout=5)
                raise BuildWorkerError("build config exceeds the durable bound")
            returncode = process.wait(timeout=30)
        except (OSError, subprocess.SubprocessError) as exc:
            try:
                process.kill()
            except OSError:
                pass
            try:
                process.wait(timeout=5)
            except (OSError, subprocess.SubprocessError):
                pass
            raise BuildWorkerError(
                "submitted build SHA config could not be read"
            ) from exc
        if returncode != 0:
            raise BuildWorkerError(
                "submitted build SHA does not contain a build config"
            )
        source = f"{base_sha}:{bq.CONFIG_FILENAME}"
    try:
        text = raw.decode("utf-8")
        config = bq.parse_config(
            load_yaml_mapping_text(text, source=source), source=source
        )
    except (UnicodeDecodeError, bq.BuildQueueError, ValueError) as exc:
        raise BuildWorkerError("submitted build config could not be parsed") from exc
    return config, hashlib.sha256(raw).hexdigest()


class BuildAuthority(Protocol):
    """Durable WorkItem lifecycle operations consumed by the worker."""

    def get(self, job_id: str) -> DurableJobView | Mapping[str, Any] | None: ...

    def claim(self, job_id: str, *, token: str) -> Mapping[str, Any] | None: ...

    def heartbeat(self, job_id: str, claim: Mapping[str, Any]) -> bool: ...

    def is_current(self, job_id: str, claim: Mapping[str, Any]) -> bool: ...

    def terminal_matches(
        self, job_id: str, claim: Mapping[str, Any], *, result_ref: str
    ) -> bool: ...

    def get_build_descriptor(
        self, job_id: str
    ) -> BuildExecutionDescriptor | Mapping[str, Any] | None: ...

    def commit(
        self,
        job_id: str,
        claim: Mapping[str, Any],
        *,
        outcome: str,
        result_ref: str | None = None,
        error_ref: str | None = None,
        failure_class: str | None = None,
        refusal_code: str | None = None,
        retryable: bool = True,
    ) -> Any: ...

    def cancel(self, job_id: str, *, reason: str) -> bool: ...


def _claim_fence(claim: Mapping[str, Any]) -> str:
    value = claim.get("fence", claim.get("fencing_token", claim.get("fence_token")))
    if value is None or not str(value).strip():
        raise BuildWorkerError("durable build claim did not include a fencing token")
    return str(value)


def _claim_attempt(claim: Mapping[str, Any], view: DurableJobView) -> int:
    value = claim.get("attempt", view.attempt or 1)
    try:
        return max(1, int(value))
    except (TypeError, ValueError) as exc:
        raise BuildWorkerError(
            "durable build claim included an invalid attempt"
        ) from exc


def _validate_claim_identity(
    job_id: str, view: DurableJobView, claim: Mapping[str, Any]
) -> None:
    """Require the native claim to identify the exact WorkItem being run."""

    if str(claim.get("job_id") or "") != view.job_id or view.job_id != job_id:
        raise BuildWorkerError("durable build claim job identity does not match")
    if str(claim.get("work_item_id") or "") != view.work_item_id:
        raise BuildWorkerError("durable build claim WorkItem identity does not match")
    try:
        claim_attempt = int(claim.get("attempt", 0))
    except (TypeError, ValueError) as exc:
        raise BuildWorkerError(
            "durable build claim included an invalid attempt"
        ) from exc
    if claim_attempt < 1 or (view.attempt > 0 and claim_attempt != view.attempt):
        raise BuildWorkerError("durable build claim attempt does not match")
    claim_fence = _claim_fence(claim)
    if view.lease_fence is not None and view.lease_fence != claim_fence:
        raise BuildWorkerError("durable build claim fence does not match")


def _result_ref(key: str, fence: str) -> str:
    """Bind the terminal result reference to the committing fence."""

    return f"build-manifest:{key}:fence:{fence}"


def _as_view(value: DurableJobView | Mapping[str, Any] | None) -> DurableJobView:
    if value is None:
        raise BuildWorkerError("durable build WorkItem was not found")
    if isinstance(value, DurableJobView):
        return value
    return DurableJobView.model_validate(value)


def _key_from_components(components: Mapping[str, Any]) -> bq.CacheKey:
    required = (
        "repo",
        "spec",
        "tree_sha",
        "feature_set",
        "toolchain_fingerprint",
        "target_triple",
        "config_digest",
        "spec_digest",
        "generation_id",
        "generation_digest",
        "key_version",
    )
    if any(not isinstance(components.get(name), str) for name in required):
        raise BuildWorkerError("durable build descriptor has incomplete key components")
    if components["key_version"] != "v2":
        raise BuildWorkerError("durable build descriptor does not carry a v2 key")
    expected_generation = bq._generation_digest(components["generation_id"])  # noqa: SLF001
    if components["generation_digest"] != expected_generation:
        raise BuildWorkerError(
            "durable build descriptor generation digest does not match its ID"
        )
    if not components["config_digest"] or not components["spec_digest"]:
        raise BuildWorkerError("durable build descriptor is missing v2 input digests")
    return bq.CacheKey(
        repo=components["repo"],
        spec=components["spec"],
        tree_sha=components["tree_sha"],
        feature_set=components["feature_set"],
        toolchain_fingerprint=components["toolchain_fingerprint"],
        target_triple=components["target_triple"],
        config_digest=components["config_digest"],
        spec_digest=components["spec_digest"],
        generation_id=components["generation_id"],
        generation_digest=components["generation_digest"],
        key_version=components["key_version"],
    )


def _paths_tree_sha_at(tree: Path, head_sha: str, paths: tuple[str, ...]) -> str:
    """Hash the declared cache-key paths from the exact submitted HEAD."""

    if not paths:
        return bq._require_git(  # noqa: SLF001
            ["rev-parse", f"{head_sha}^{{tree}}"], tree
        )
    listing = bq._require_git(  # noqa: SLF001
        ["ls-tree", "-r", head_sha, "--", *paths], tree
    )
    return hashlib.sha256(listing.encode()).hexdigest()[:32]


def _descriptor_spec(descriptor: Mapping[str, Any]) -> bq.BuildSpec:
    command = descriptor.get("command")
    toolchain_command = descriptor.get("toolchain_command")
    artifacts = descriptor.get("artifacts")
    workdir = descriptor.get("workdir")
    name = descriptor.get("spec")
    if (
        not isinstance(name, str)
        or not name.strip()
        or not isinstance(command, list | tuple)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or not isinstance(toolchain_command, list | tuple)
        or any(not isinstance(item, str) or not item for item in toolchain_command)
        or not isinstance(artifacts, list | tuple)
        or not artifacts
        or any(not isinstance(item, str) or not item for item in artifacts)
        or not isinstance(workdir, str)
        or Path(workdir).is_absolute()
        or ".." in Path(workdir).parts
    ):
        raise BuildWorkerError("durable build descriptor has invalid execution paths")
    try:
        timeout = int(descriptor.get("timeout", 0))
    except (TypeError, ValueError) as exc:
        raise BuildWorkerError("durable build descriptor has invalid timeout") from exc
    if timeout < 1:
        raise BuildWorkerError("durable build descriptor has invalid timeout")
    return bq.BuildSpec(
        name=name,
        command=tuple(command),
        toolchain_fingerprint=tuple(toolchain_command),
        workdir=workdir,
        artifacts=tuple(artifacts),
        timeout=timeout,
        resource_class=str(descriptor.get("resource_class") or "light-check"),
    )


def _verify_toolchain_fingerprint(
    build_tree: Path,
    spec: bq.BuildSpec,
    descriptor: Mapping[str, Any],
    *,
    cacheable: bool,
) -> str | None:
    """Recompute the toolchain on the immutable materialized build tree.

    A cacheable descriptor must reproduce the fingerprint incorporated into
    its v2 key.  An uncacheable descriptor still executes the probe so the
    worker observes the submitted toolchain, but it intentionally does not
    turn a degraded request into a cache hit when the probe is unavailable or
    has changed.
    """

    expected = str(descriptor.get("toolchain_fingerprint") or "")
    if not spec.toolchain_fingerprint:
        if cacheable and expected != "unpinned":
            raise BuildWorkerError(
                "persisted unpinned toolchain fingerprint disagrees with the spec"
            )
        return None
    try:
        actual = bq._toolchain_fingerprint(build_tree, spec)  # noqa: SLF001
    except (OSError, subprocess.SubprocessError):
        actual = None
    if cacheable and (actual is None or actual != expected):
        raise BuildWorkerError(
            "worker toolchain fingerprint does not match the durable build key"
        )
    return actual


def _terminal_matches(
    authority: BuildAuthority,
    job_id: str,
    claim: Mapping[str, Any],
    *,
    result_ref: str,
) -> bool:
    checker = getattr(authority, "terminal_matches", None)
    if not callable(checker):
        # A generic current-fence check proves only that a live lease exists;
        # it is intentionally insufficient for finalization.
        return False
    try:
        return bool(checker(job_id, claim, result_ref=result_ref))
    except Exception:
        return False


class GraphBuildAuthority:
    """Production adapter over the existing Agent Utilities WorkItem verbs."""

    def __init__(self, engine: Any, *, tenant_id: str, token: str) -> None:
        if not tenant_id.strip() or not token.strip():
            raise ValueError("tenant_id and token must be non-blank")
        self.engine = engine
        self.tenant_id = tenant_id
        self.token = token

    @staticmethod
    def _authority() -> Any:
        from agent_utilities.orchestration import repository_work_item as authority

        return authority

    def get(self, job_id: str) -> DurableJobView | None:
        authority = self._authority()
        view = authority.get_repository_work_item(
            self.engine, job_id, tenant=self.tenant_id
        )
        if view is None:
            return None
        from repository_manager.development.jobs import GraphRepositoryJobPort

        return GraphRepositoryJobPort._view(view)  # noqa: SLF001 - adapter boundary

    def get_build_descriptor(
        self, job_id: str
    ) -> BuildExecutionDescriptor | Mapping[str, Any] | None:
        """Read the typed RMDD build extension, never arbitrary WorkItem fields."""

        authority = self._authority()
        getter = getattr(authority, "get_repository_build_descriptor", None)
        if callable(getter):
            return getter(self.engine, job_id, tenant=self.tenant_id)
        view = authority.get_repository_work_item(
            self.engine, job_id, tenant=self.tenant_id
        )
        return getattr(view, "build_descriptor", None) if view is not None else None

    def claim(self, job_id: str, *, token: str) -> Mapping[str, Any] | None:
        authority = self._authority()
        return authority.claim_repository_work_item(
            self.engine,
            job_id,
            tenant=self.tenant_id,
            token=token,
        )

    def heartbeat(self, job_id: str, claim: Mapping[str, Any]) -> bool:
        authority = self._authority()
        return bool(
            authority.heartbeat_repository_work_item(
                self.engine,
                job_id,
                claim,
                tenant=self.tenant_id,
            )
        )

    def is_current(self, job_id: str, claim: Mapping[str, Any]) -> bool:
        view = self.get(job_id)
        if view is None or view.state not in {JobState.LEASED, JobState.RUNNING}:
            return False
        current = view.lease_fence
        return current is not None and current == _claim_fence(claim)

    def terminal_matches(
        self, job_id: str, claim: Mapping[str, Any], *, result_ref: str
    ) -> bool:
        """Prove terminal success for this exact durable WorkItem/fence."""

        view = self.get(job_id)
        if view is None or view.job_id != str(claim.get("job_id") or job_id):
            return False
        if view.work_item_id != str(claim.get("work_item_id") or ""):
            return False
        try:
            expected_attempt = int(claim.get("attempt", 0))
        except (TypeError, ValueError):
            return False
        if view.attempt != expected_attempt or view.state is not JobState.SUCCEEDED:
            return False
        if view.result_ref != result_ref:
            return False
        # AU's terminal projection clears the live lease in some backends.
        # Read the raw native row as well so a terminal proof is still bound
        # to the exact fencing token that committed the result.
        authority = self._authority()
        get_row = getattr(authority, "get_work_item", None)
        if not callable(get_row):
            return False
        row = get_row(self.engine, view.work_item_id)
        if not isinstance(row, Mapping):
            return False
        raw_fence = row.get("fencing_token")
        durable_result = str(row.get("result_ref") or "")
        result_fence = result_ref.rsplit(":fence:", 1)[-1]
        if result_fence != _claim_fence(claim):
            return False
        return (
            row.get("id") == view.work_item_id
            and (raw_fence is None or str(raw_fence) == _claim_fence(claim))
            and durable_result == result_ref
        )

    def claim_next(self, *, kind: str, token: str) -> Mapping[str, Any] | None:
        authority = self._authority()
        return authority.claim_next_repository_work_item(
            self.engine,
            tenant=self.tenant_id,
            kind=kind,
            token=token,
        )

    def commit(
        self,
        job_id: str,
        claim: Mapping[str, Any],
        *,
        outcome: str,
        result_ref: str | None = None,
        error_ref: str | None = None,
        failure_class: str | None = None,
        refusal_code: str | None = None,
        retryable: bool = True,
    ) -> Any:
        authority = self._authority()
        return authority.commit_repository_work_item(
            self.engine,
            job_id,
            claim,
            tenant=self.tenant_id,
            outcome=outcome,
            result_ref=result_ref,
            error_ref=error_ref,
            failure_class=failure_class,
            refusal_code=refusal_code,
            retryable=retryable,
        )

    def cancel(self, job_id: str, *, reason: str) -> bool:
        authority = self._authority()
        return bool(
            authority.cancel_repository_work_item(
                self.engine,
                job_id,
                tenant=self.tenant_id,
                reason=reason,
            )
        )


class BuildWorker:
    """Claim, admit, execute, and publish one durable build WorkItem."""

    def __init__(
        self,
        authority: BuildAuthority,
        scheduler: ResourceScheduler | None,
        *,
        artifact_store: BuildArtifactStore | None = None,
        executor: CommandExecutor | None = None,
        worker_id: str = "worker:repository-manager-build",
        token_factory: Callable[[], str] | None = None,
        lease_ttl_seconds: int = 900,
        stale_stage_age_seconds: int = _DEFAULT_STALE_STAGE_AGE_SECONDS,
    ) -> None:
        if not callable(getattr(authority, "claim", None)):
            raise TypeError("BuildWorker requires a durable WorkItem authority")
        if not worker_id.strip():
            raise ValueError("worker_id must be non-blank")
        if lease_ttl_seconds < 1:
            raise ValueError("lease_ttl_seconds must be positive")
        if stale_stage_age_seconds < 0:
            raise ValueError("stale_stage_age_seconds must be non-negative")
        self.authority = authority
        self.scheduler = scheduler
        self.artifact_store = artifact_store
        self.executor = executor
        self.worker_id = worker_id
        self.token_factory = token_factory or (lambda: worker_id)
        self.lease_ttl_seconds = lease_ttl_seconds
        self.stale_stage_age_seconds = stale_stage_age_seconds
        self._cancellations: dict[str, CancellationToken] = {}
        self._release_errors: dict[str, str] = {}
        self._release_contexts: dict[str, dict[str, Any]] = {}

    def release_failures(self) -> dict[str, str]:
        """Return reservation-release failures for scheduler reconciliation."""

        return dict(self._release_errors)

    def reconcile_releases(self) -> dict[str, Any]:
        """Retry held reservations without rerunning their WorkItems."""

        if self.scheduler is None:
            return {
                "ok": False,
                "reconciled": [],
                "pending": sorted(self._release_errors),
            }
        reconciled: list[str] = []
        for reservation_id, context in tuple(self._release_contexts.items()):
            try:
                released = self.scheduler.release(
                    reservation_id,
                    **context,
                    reason="build worker release reconciliation",
                )
            except Exception as exc:
                self._release_errors[reservation_id] = str(exc)
                continue
            if released is False:
                self._release_errors[reservation_id] = (
                    "scheduler did not confirm exact reservation release"
                )
                continue
            self._release_errors.pop(reservation_id, None)
            self._release_contexts.pop(reservation_id, None)
            reconciled.append(reservation_id)
        return {
            "ok": not self._release_errors,
            "reconciled": reconciled,
            "pending": sorted(self._release_errors),
        }

    def _current_view(self, job_id: str) -> DurableJobView | None:
        """Read current authority state for idempotent terminal reconciliation."""

        try:
            value = self.authority.get(job_id)
        except Exception:
            return None
        try:
            return _as_view(value)
        except Exception:
            return None

    def run_next(
        self,
        *,
        repo_path: Path | str,
        spec_name: str = "",
        token: str | None = None,
    ) -> dict[str, Any] | None:
        """Claim the next build through an authority-specific queue verb."""

        claim_next = getattr(self.authority, "claim_next", None)
        if not callable(claim_next):
            raise BuildWorkerError("authority does not expose claim_next; use run_job")
        claim = claim_next(
            kind="repository.build",
            token=token or self.token_factory(),
        )
        if claim is None:
            return None
        job_id = str(claim.get("job_id") or "")
        if not job_id:
            raise BuildWorkerError("claim_next returned no repository job ID")
        return self.run_job(
            job_id,
            repo_path=repo_path,
            spec_name=spec_name,
            claim=claim,
        )

    def run_job(
        self,
        job_id: str,
        *,
        repo_path: Path | str,
        spec_name: str = "",
        claim: Mapping[str, Any] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Run one job and surface reservation reconciliation explicitly."""

        result = self._run_job(
            job_id,
            repo_path=repo_path,
            spec_name=spec_name,
            claim=claim,
            cancellation=cancellation,
        )
        reservation_id = result.get("reservation_id")
        if reservation_id and reservation_id in self._release_errors:
            result = dict(result)
            result.update(
                {
                    "release_pending": True,
                    "reconciliation_pending": True,
                    "release_error": self._release_errors[reservation_id],
                }
            )
        return result

    def _run_job(
        self,
        job_id: str,
        *,
        repo_path: Path | str,
        spec_name: str = "",
        claim: Mapping[str, Any] | None = None,
        cancellation: CancellationToken | None = None,
    ) -> dict[str, Any]:
        """Run one job; all heavyweight effects follow resource admission."""
        view = _as_view(self.authority.get(job_id))
        if view.state in {JobState.CANCELLED, JobState.SUCCEEDED}:
            return self._refusal(job_id, "work_item_terminal", view=view)
        supplied_claim = claim or self.authority.claim(
            job_id, token=self.token_factory()
        )
        actual_claim = dict(supplied_claim or {})
        if not actual_claim:
            return self._refusal(job_id, "work_item_not_claimable", view=view)
        _validate_claim_identity(job_id, view, actual_claim)
        fence = _claim_fence(actual_claim)
        attempt = _claim_attempt(actual_claim, view)
        token = cancellation or CancellationToken()
        self._cancellations[job_id] = token
        reservation_id: str | None = None
        store: BuildArtifactStore | None = None
        key: bq.CacheKey | None = None
        published = False
        terminal_committed = False
        try:
            scope, spec, key, descriptor = self._execution_plan(
                view, repo_path=repo_path, spec_name=spec_name
            )
            if self.scheduler is None:
                return self._terminal_refusal(
                    job_id,
                    actual_claim,
                    view,
                    code="resource_scheduler_required",
                )
            admission = self._admit(view, attempt, fence)
            raw_status = getattr(admission, "status", None)
            try:
                admission_status = AdmissionStatus(raw_status)
            except (TypeError, ValueError):
                admission_status = None
            if admission_status is AdmissionStatus.DEFERRED:
                return self._deferred_admission(
                    job_id,
                    view,
                    admission,
                    retryable=True,
                )
            if admission_status is AdmissionStatus.STALE_FENCE:
                return self._deferred_admission(
                    job_id,
                    view,
                    admission,
                    retryable=False,
                    stale_fence=True,
                )
            if not admission.admitted:
                return self._terminal_refusal(
                    job_id,
                    actual_claim,
                    view,
                    code=admission.reason_code.value,
                )
            reservation_id = admission.reservation_id
            if token.is_cancelled():
                return self._terminal_cancel(
                    job_id,
                    actual_claim,
                    view,
                    "cancelled before materialization",
                    reservation_id=reservation_id,
                )

            store = self.artifact_store or BuildArtifactStore(repo_path=scope.tree)
            if key is None:
                return self._run_degraded(
                    job_id,
                    actual_claim,
                    view,
                    scope,
                    spec,
                    descriptor,
                    token,
                    reservation_id=reservation_id,
                )

            with bq.materialized(scope.tree, view.base_sha, scope=scope) as build_tree:
                _verify_toolchain_fingerprint(
                    build_tree,
                    spec,
                    descriptor,
                    cacheable=True,
                )
                command = ExecutionCommand(
                    argv=spec.command,
                    workdir=str((build_tree / spec.workdir).resolve()),
                    timeout_seconds=spec.timeout,
                    heartbeat_interval_seconds=min(30, max(1, spec.timeout // 4)),
                )
                executor = self.executor or LocalExecutor(
                    build_tree,
                    worker_id=self.worker_id,
                )
                result = executor.run(
                    command,
                    command_id=f"build:{key.digest}",
                    worker_id=self.worker_id,
                    fence=fence,
                    cancellation=token,
                    fence_check=lambda: self.authority.is_current(job_id, actual_claim),
                    heartbeat=lambda: self.authority.heartbeat(job_id, actual_claim),
                )
                if result.outcome != ExecutionOutcome.SUCCEEDED:
                    return self._commit_execution_failure(
                        job_id, actual_claim, view, result, reservation_id
                    )
                staged = store.stage(
                    build_tree,
                    workdir=spec.workdir,
                    patterns=spec.artifacts,
                    key=key.digest,
                    attempt=attempt,
                    fence=fence,
                    job_id=view.job_id,
                    work_item_id=view.work_item_id,
                    generation_id=view.generation_id,
                    max_artifacts=1024,
                    max_bytes=(
                        view.disk_mib * 1024 * 1024 if view.disk_mib > 0 else None
                    ),
                )
                store.publish(
                    staged,
                    fence_check=lambda: self.authority.is_current(job_id, actual_claim),
                )
                published = True
            result_ref = _result_ref(key.digest, fence)
            try:
                commit_result = self.authority.commit(
                    job_id,
                    actual_claim,
                    outcome="succeeded",
                    result_ref=result_ref,
                    retryable=False,
                )
            except Exception as exc:
                if _terminal_matches(
                    self.authority,
                    job_id,
                    actual_claim,
                    result_ref=result_ref,
                ):
                    terminal_committed = True
                    return self._reconciliation_pending(
                        job_id,
                        view,
                        key=key,
                        error=str(exc),
                        reservation_id=reservation_id,
                    )
                raise
            if str(commit_result) not in {"None", "committed", "noop", "succeeded"}:
                if _terminal_matches(
                    self.authority,
                    job_id,
                    actual_claim,
                    result_ref=result_ref,
                ):
                    terminal_committed = True
                    return self._reconciliation_pending(
                        job_id,
                        view,
                        key=key,
                        error=f"durable WorkItem success commit returned {commit_result!r}",
                        reservation_id=reservation_id,
                    )
                raise BuildWorkerError(
                    f"durable WorkItem success commit returned {commit_result!r}"
                )
            terminal_committed = True
            # A restart between terminal commit and this metadata update is
            # safe: recovery can finalize the already-published checksummed
            # bytes after observing exact durable terminal evidence.
            manifest = store.finalize(
                key.digest,
                fence=fence,
                terminal_check=lambda: _terminal_matches(
                    self.authority,
                    job_id,
                    actual_claim,
                    result_ref=result_ref,
                ),
                job_id=view.job_id,
                work_item_id=view.work_item_id,
                attempt=attempt,
            )
            return {
                "ok": True,
                "job_id": job_id,
                "work_item_id": view.work_item_id,
                "state": "succeeded",
                "cached": False,
                "key": key.digest,
                "components": key.components(),
                "artifacts": manifest.get("artifacts", []),
                "reservation_id": reservation_id,
                "published_before_commit": published,
            }
        except ArtifactFenceLost as exc:
            if terminal_committed:
                return self._reconciliation_pending(
                    job_id,
                    view,
                    key=key,
                    error=str(exc),
                    reservation_id=reservation_id,
                )
            self._quarantine_stale_publication(
                store,
                key=key,
                job_id=job_id,
                view=view,
            )
            return self._terminal_refusal(
                job_id,
                actual_claim,
                view,
                code="stale_fence",
                error=str(exc),
                reservation_id=reservation_id,
            )
        except (ArtifactStoreError, bq.BuildQueueError) as exc:
            if terminal_committed:
                return self._reconciliation_pending(
                    job_id,
                    view,
                    key=key,
                    error=str(exc),
                    reservation_id=reservation_id,
                )
            self._quarantine_stale_publication(
                store,
                key=key,
                job_id=job_id,
                view=view,
            )
            return self._terminal_refusal(
                job_id,
                actual_claim,
                view,
                code="artifact_publication_failed",
                error=str(exc),
                reservation_id=reservation_id,
            )
        except Exception as exc:  # noqa: BLE001 - terminalize unexpected worker failures
            if terminal_committed:
                return self._reconciliation_pending(
                    job_id,
                    view,
                    key=key,
                    error=str(exc),
                    reservation_id=reservation_id,
                )
            self._quarantine_stale_publication(
                store,
                key=key,
                job_id=job_id,
                view=view,
            )
            return self._terminal_refusal(
                job_id,
                actual_claim,
                view,
                code="worker_environment_failure",
                error=str(exc),
                reservation_id=reservation_id,
            )
        finally:
            if reservation_id and self.scheduler is not None:
                release_context = {
                    "work_item_id": view.work_item_id,
                    "attempt": attempt,
                    "fence": fence,
                }
                self._release_contexts[reservation_id] = release_context
                try:
                    released = self.scheduler.release(
                        reservation_id,
                        **release_context,
                        reason="build worker finished",
                    )
                except Exception as exc:
                    # Native reservation authority remains the source of truth;
                    # retain an observable repair signal while keeping the
                    # reservation held safely until scheduler reconciliation.
                    self._release_errors[reservation_id] = str(exc)
                else:
                    # ResourceScheduler returns False when native lifecycle
                    # authority refuses or cannot prove the exact release.
                    # Treat that as held-and-reconcilable, never as success.
                    if released is False:
                        self._release_errors[reservation_id] = (
                            "scheduler did not confirm exact reservation release"
                        )
                    else:
                        self._release_errors.pop(reservation_id, None)
                        self._release_contexts.pop(reservation_id, None)
            self._cancellations.pop(job_id, None)

    def _quarantine_stale_publication(
        self,
        store: BuildArtifactStore | None,
        *,
        key: bq.CacheKey | None,
        job_id: str,
        view: DurableJobView,
    ) -> None:
        """Remove a stale publication only with its own durable identity proof."""

        if store is None or key is None:
            return

        def authority_proves_stale(
            manifest: Mapping[str, Any],
        ) -> Mapping[str, Any]:
            manifest_job = str(manifest.get("job_id") or "")
            manifest_work_item = str(manifest.get("work_item_id") or "")
            manifest_fence = str(manifest.get("fence") or "")
            try:
                manifest_attempt = int(manifest.get("attempt", 0))
            except (TypeError, ValueError):
                return {}
            if (
                manifest_job != view.job_id
                or manifest_work_item != view.work_item_id
                or manifest_attempt < 1
                or not manifest_fence
            ):
                return {}
            proof = {
                "job_id": manifest_job,
                "work_item_id": manifest_work_item,
                "attempt": manifest_attempt,
                "fence": manifest_fence,
                "stale": True,
            }
            current = self.authority.get(job_id)
            if current is None:
                return {}
            current_view = _as_view(current)
            if current_view.work_item_id != view.work_item_id:
                return {}
            expected_result = _result_ref(key.digest, manifest_fence)
            if (
                current_view.state is JobState.SUCCEEDED
                and current_view.result_ref == expected_result
            ):
                return {}
            if current_view.state in {
                JobState.SUBMITTED,
                JobState.READY,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.DEAD_LETTER,
            }:
                return proof
            stale = current_view.attempt != manifest_attempt or (
                current_view.lease_fence != manifest_fence
            )
            return proof if stale else {}

        try:
            store.quarantine_if(
                key.digest,
                authority_check=authority_proves_stale,
                reason="stale-fence-publication",
            )
        except ArtifactStoreError:
            # Quarantine is destructive; an authority/filesystem error leaves
            # the bytes in place for a later reconciler to inspect.
            return

    def _quarantine_terminal_invalid(
        self,
        store: BuildArtifactStore,
        *,
        key: bq.CacheKey,
        manifest: Mapping[str, Any],
        view: DurableJobView,
        reason: str,
    ) -> None:
        """Quarantine a terminal entry only after exact owner proof."""

        def proves_invalid(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
            try:
                candidate_attempt = int(candidate.get("attempt", 0))
            except (TypeError, ValueError):
                return {}
            if (
                candidate.get("job_id") != view.job_id
                or candidate.get("work_item_id") != view.work_item_id
                or candidate_attempt != view.attempt
            ):
                return {}
            current = self._current_view(view.job_id)
            if current is None or current.work_item_id != view.work_item_id:
                return {}
            if current.state not in {
                JobState.SUCCEEDED,
                JobState.FAILED,
                JobState.CANCELLED,
                JobState.DEAD_LETTER,
            }:
                return {}
            proof = {
                "job_id": candidate["job_id"],
                "work_item_id": candidate["work_item_id"],
                "attempt": candidate_attempt,
                "fence": candidate.get("fence"),
                "stale": True,
            }
            if not isinstance(proof["fence"], str) or not proof["fence"]:
                return {}
            if current.state is JobState.SUCCEEDED:
                # A matching terminal result still proves that this exact
                # owner/attempt owns the corrupt bytes; quarantine is the
                # repair action, not a second success decision.
                return proof
            return proof

        try:
            store.quarantine_if(
                key.digest,
                authority_check=proves_invalid,
                reason=reason,
            )
        except ArtifactStoreError:
            return

    @staticmethod
    def _reconciliation_pending(
        job_id: str,
        view: DurableJobView,
        *,
        key: bq.CacheKey | None,
        error: str,
        reservation_id: str | None = None,
    ) -> dict[str, Any]:
        """Report the durable success without contradicting it as a failure."""

        return {
            "ok": False,
            "job_id": job_id,
            "work_item_id": view.work_item_id,
            "state": JobState.SUCCEEDED.value,
            "durable_terminal": True,
            "reconciliation_pending": True,
            "key": key.digest if key is not None else None,
            "error": error,
            "reservation_id": reservation_id,
        }

    def _execution_plan(
        self,
        view: DurableJobView,
        *,
        repo_path: Path | str,
        spec_name: str,
    ) -> tuple[Any, bq.BuildSpec, bq.CacheKey | None, Mapping[str, Any]]:
        scope = bq.lane_scope(repo_path)
        if view.repository_id != bq.stable_repository_id(scope.main_tree):
            raise BuildWorkerError(
                "worker repository identity does not match the WorkItem"
            )
        get_descriptor = getattr(self.authority, "get_build_descriptor", None)
        if not callable(get_descriptor):
            raise BuildWorkerError(
                "durable build authority lacks the typed execution descriptor extension"
            )
        raw_descriptor = get_descriptor(view.job_id)
        if raw_descriptor is None:
            raise BuildWorkerError(
                "durable build WorkItem has no typed execution descriptor; resubmit"
            )
        try:
            descriptor_model = (
                raw_descriptor
                if isinstance(raw_descriptor, BuildExecutionDescriptor)
                else BuildExecutionDescriptor.model_validate(raw_descriptor)
            )
        except ValueError as exc:
            raise BuildWorkerError(
                "durable build execution descriptor is invalid"
            ) from exc
        descriptor = descriptor_model.model_dump(mode="python")
        if descriptor is None:
            raise BuildWorkerError(
                "durable build WorkItem has no persisted execution descriptor; resubmit"
            )
        if view.operation != "build":
            raise BuildWorkerError(
                "durable build WorkItem operation is not a build operation"
            )
        if descriptor.get("repository_id") != view.repository_id:
            raise BuildWorkerError(
                "build descriptor repository identity disagrees with WorkItem"
            )
        if descriptor.get("base_sha") != view.base_sha:
            raise BuildWorkerError(
                "build descriptor SHA disagrees with WorkItem authority"
            )
        if descriptor.get("head_sha") != view.base_sha:
            raise BuildWorkerError(
                "build descriptor HEAD disagrees with WorkItem authority"
            )
        if descriptor.get("resource_class") != view.resource_class:
            raise BuildWorkerError(
                "build descriptor resource profile disagrees with WorkItem"
            )
        descriptor_generation = descriptor.get("generation_id") or None
        view_generation = view.generation_id or None
        if descriptor_generation != view_generation:
            raise BuildWorkerError(
                "build descriptor generation disagrees with WorkItem authority"
            )
        if (
            view.config_digest is not None
            and descriptor.get("config_digest") != view.config_digest
        ):
            raise BuildWorkerError(
                "build descriptor config digest disagrees with WorkItem authority"
            )
        # ``DurableJobView.input_digest`` is AU's canonical whole-request
        # digest, while the typed build extension's input_digest covers the
        # executable descriptor body.  They are intentionally different
        # contracts until RMDD-29 adds a typed projection; do not manufacture
        # equivalence or accept an untyped field smuggled through correlation.
        if descriptor.get("input_digest") != descriptor_input_digest(descriptor):
            raise BuildWorkerError(
                "persisted build descriptor input digest does not match its body"
            )
        persisted_spec = str(descriptor.get("spec") or "")
        if spec_name and spec_name != persisted_spec:
            raise BuildWorkerError(
                "worker spec selection disagrees with persisted descriptor"
            )
        dirty = descriptor.get("degraded_reason") == "dirty-tree"
        if dirty:
            raise BuildWorkerError(
                "dirty durable builds require an immutable typed snapshot; "
                "durable snapshot authority is not installed"
            )
        config, config_digest = _config_snapshot(
            scope,
            base_sha=view.base_sha,
            dirty=dirty,
        )
        if descriptor.get("config_digest") != config_digest:
            raise BuildWorkerError(
                "persisted build config digest does not match the submitted snapshot"
            )
        try:
            snapshot_spec = config.spec(persisted_spec)
        except bq.BuildQueueError as exc:
            raise BuildWorkerError(
                "submitted build config does not contain the persisted spec"
            ) from exc
        if descriptor.get("spec_digest") != bq._spec_digest(snapshot_spec):  # noqa: SLF001
            raise BuildWorkerError(
                "persisted build spec digest does not match the submitted snapshot"
            )
        descriptor_spec = _descriptor_spec(descriptor)
        if (
            descriptor_spec.command != snapshot_spec.command
            or descriptor_spec.toolchain_fingerprint
            != snapshot_spec.toolchain_fingerprint
            or descriptor_spec.workdir != snapshot_spec.workdir
            or descriptor_spec.artifacts != snapshot_spec.artifacts
            or descriptor_spec.timeout != snapshot_spec.timeout
            or descriptor_spec.resource_class != snapshot_spec.resource_class
        ):
            raise BuildWorkerError(
                "persisted build execution fields disagree with the submitted snapshot"
            )
        spec = snapshot_spec
        cacheable = descriptor.get("cacheable") is True
        key: bq.CacheKey | None = None
        if cacheable:
            components = descriptor.get("key_components")
            if not isinstance(components, Mapping):
                raise BuildWorkerError(
                    "cacheable descriptor has no immutable key components"
                )
            key = _key_from_components(components)
            if descriptor.get("cache_key") != key.digest:
                raise BuildWorkerError(
                    "persisted build key does not match its components"
                )
            if descriptor.get("config_digest") != key.config_digest:
                raise BuildWorkerError(
                    "persisted build config digest does not match its key"
                )
            if descriptor.get("spec_digest") != key.spec_digest:
                raise BuildWorkerError(
                    "persisted build spec digest does not match its key"
                )
            if descriptor.get("generation_id") != key.generation_id:
                raise BuildWorkerError(
                    "persisted build generation does not match its key"
                )
            if descriptor.get("toolchain_fingerprint") != key.toolchain_fingerprint:
                raise BuildWorkerError(
                    "persisted build toolchain fingerprint does not match its key"
                )
            try:
                submitted_tree_sha = _paths_tree_sha_at(
                    scope.main_tree,
                    view.base_sha,
                    snapshot_spec.cache_key_paths,
                )
            except (bq.BuildQueueError, OSError) as exc:
                raise BuildWorkerError(
                    "submitted build SHA tree could not be verified"
                ) from exc
            if key.tree_sha != submitted_tree_sha:
                raise BuildWorkerError(
                    "persisted build key tree digest disagrees with submitted SHA"
                )
        elif descriptor.get("toolchain_fingerprint") != "unavailable":
            raise BuildWorkerError(
                "uncacheable build descriptor lacks an unavailable toolchain marker"
            )
        return scope, spec, key, descriptor

    def cancel(self, job_id: str, *, reason: str = "cancelled by owner") -> bool:
        token = self._cancellations.get(job_id)
        if token is not None:
            token.cancel(reason)
        return bool(self.authority.cancel(job_id, reason=reason))

    def _staging_authority_probe(
        self, stage_manifest: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        """Prove that one old stage belongs to a terminal or stale claim."""

        stage_job = str(stage_manifest.get("job_id") or "")
        stage_work_item = str(stage_manifest.get("work_item_id") or "")
        stage_fence = str(stage_manifest.get("fence") or "")
        try:
            stage_attempt = int(stage_manifest.get("attempt", 0))
        except (TypeError, ValueError):
            return {}
        if not stage_job or not stage_work_item or not stage_fence or stage_attempt < 1:
            return {}
        try:
            current = _as_view(self.authority.get(stage_job))
        except Exception:
            return {}
        if current.job_id != stage_job or current.work_item_id != stage_work_item:
            return {}
        if current.state is JobState.SUCCEEDED:
            # A successful WorkItem may still be between durable terminal
            # commit and manifest finalization.  Preserve its stage until the
            # normal terminal-evidence recovery path has inspected it.
            stale = False
        elif current.state in {
            JobState.FAILED,
            JobState.CANCELLED,
            JobState.DEAD_LETTER,
        }:
            stale = True
        elif current.state in {JobState.LEASED, JobState.RUNNING}:
            # A paused but still-owned producer must retain its private stage.
            stale = not (
                current.attempt == stage_attempt and current.lease_fence == stage_fence
            )
        elif current.state in {JobState.SUBMITTED, JobState.READY}:
            # No active lease can consume this private stage anymore.
            stale = True
        else:
            stale = False
        if not stale:
            return {}
        return {
            "job_id": stage_job,
            "work_item_id": stage_work_item,
            "attempt": stage_attempt,
            "fence": stage_fence,
            "stale": True,
        }

    def _reconcile_staging(self, store: BuildArtifactStore) -> dict[str, Any]:
        """Boundedly reconcile stages using exact durable WorkItem evidence."""

        try:
            return store.reconcile_staging(
                max_age_seconds=self.stale_stage_age_seconds,
                authority_probe=self._staging_authority_probe,
            )
        except Exception as exc:  # pragma: no cover - defensive authority boundary
            return {"removed": [], "kept": [], "errors": [str(exc)]}

    def recover(
        self,
        job_id: str,
        *,
        repo_path: Path | str,
        spec_name: str = "",
    ) -> dict[str, Any]:
        """Reconcile a restart without trusting a local running marker."""

        view = _as_view(self.authority.get(job_id))
        scope, spec, key, descriptor = self._execution_plan(
            view, repo_path=repo_path, spec_name=spec_name
        )
        store = self.artifact_store or BuildArtifactStore(repo_path=scope.tree)
        staging_reconciliation = self._reconcile_staging(store)

        def with_reconciliation(result: Mapping[str, Any]) -> dict[str, Any]:
            enriched = dict(result)
            enriched.setdefault("staging_reconciliation", staging_reconciliation)
            return enriched

        if key is None:
            if view.state is JobState.SUCCEEDED:
                return with_reconciliation(
                    {
                        "ok": True,
                        "recovered": True,
                        "state": view.state.value,
                        "degraded": True,
                        "degraded_reason": descriptor.get("degraded_reason"),
                    }
                )
            return with_reconciliation(
                self.run_job(job_id, repo_path=repo_path, spec_name=spec_name)
            )
        manifest = store.read_manifest(key.digest)
        manifest_fence = str(manifest.get("fence") or "") if manifest else ""
        result_ref = _result_ref(key.digest, manifest_fence)
        if view.state is JobState.SUCCEEDED:
            if (
                manifest
                and store.validate_manifest(
                    manifest, require_committed=False, expected_key=key.digest
                )
                and _terminal_matches(
                    self.authority,
                    job_id,
                    {
                        "job_id": view.job_id,
                        "work_item_id": view.work_item_id,
                        "attempt": view.attempt,
                        "fence": manifest.get("fence", ""),
                    },
                    result_ref=result_ref,
                )
            ):
                # This is the crash boundary after durable terminal commit and
                # before the manifest commit marker.  No claim or subprocess is
                # allowed for an already-succeeded WorkItem.
                try:
                    committed = store.finalize(
                        key.digest,
                        fence=str(manifest.get("fence") or ""),
                        terminal_check=lambda: _terminal_matches(
                            self.authority,
                            job_id,
                            {
                                "job_id": view.job_id,
                                "work_item_id": view.work_item_id,
                                "attempt": view.attempt,
                                "fence": manifest.get("fence", ""),
                            },
                            result_ref=result_ref,
                        ),
                        job_id=view.job_id,
                        work_item_id=view.work_item_id,
                        attempt=view.attempt,
                    )
                except (ArtifactFenceLost, ArtifactStoreError) as exc:
                    return with_reconciliation(
                        self._reconciliation_pending(
                            job_id,
                            view,
                            key=key,
                            error=str(exc),
                        )
                    )
                return with_reconciliation(
                    {
                        "ok": True,
                        "recovered": True,
                        "state": view.state.value,
                        "key": key.digest,
                        "artifacts": committed.get("artifacts", []),
                    }
                )
            if manifest:
                self._quarantine_terminal_invalid(
                    store,
                    key=key,
                    manifest=manifest,
                    view=view,
                    reason="terminal-evidence-mismatch",
                )
            return with_reconciliation(
                {
                    "ok": False,
                    "recovered": False,
                    "state": view.state.value,
                    "key": key.digest,
                    "error": "succeeded WorkItem has no exact published artifact evidence",
                }
            )
        if view.state in {JobState.CANCELLED, JobState.FAILED, JobState.DEAD_LETTER}:
            if manifest:
                self._quarantine_terminal_invalid(
                    store,
                    key=key,
                    manifest=manifest,
                    view=view,
                    reason="terminal-without-success",
                )
            return with_reconciliation(
                {
                    "ok": False,
                    "recovered": False,
                    "state": view.state.value,
                    "key": key.digest,
                    "error": "terminal WorkItem cannot be rerun",
                }
            )
        if manifest and not store.validate_manifest(
            manifest, require_committed=False, expected_key=key.digest
        ):
            self._quarantine_stale_publication(
                store,
                key=key,
                job_id=job_id,
                view=view,
            )
        # A stale lease is reclaimed by the native claim on a retry.  No local
        # process marker is promoted to success.
        return with_reconciliation(
            self.run_job(job_id, repo_path=repo_path, spec_name=spec_name)
        )

    reclaim = recover

    @staticmethod
    def _deferred_admission(
        job_id: str,
        view: DurableJobView,
        admission: Any,
        *,
        retryable: bool,
        stale_fence: bool = False,
    ) -> dict[str, Any]:
        """Leave a deferred claim durable for the native queue to retry.

        The worker performs one bounded attempt and returns.  For a capacity
        defer, the current native lease expires/reclaims through the queue;
        this worker has no typed native-defer verb and must not busy-loop.
        A stale fence is not retryable on this claim: only a fresh native
        claim may be attempted.
        """

        reason_code = getattr(getattr(admission, "reason_code", None), "value", None)
        if reason_code is None:
            reason_code = str(getattr(admission, "reason_code", "deferred"))
        status = getattr(getattr(admission, "status", None), "value", None)
        if status is None:
            status = str(getattr(admission, "status", "deferred"))
        return {
            "ok": False,
            "job_id": job_id,
            "work_item_id": view.work_item_id,
            "state": view.state.value,
            "deferred": not stale_fence,
            "stale_fence": stale_fence,
            "retryable": retryable,
            "admission_status": status,
            "reason_code": reason_code,
            "error": str(getattr(admission, "reason", "resource admission deferred")),
            "defer_mode": "lease-reclaim" if not stale_fence else None,
            "reconciliation_required": stale_fence,
            "reservation_id": None,
        }

    def _admit(
        self,
        view: DurableJobView,
        attempt: int,
        fence: str,
    ) -> AdmissionDecision:
        resources = ResourceRequest(
            resource_class=view.resource_class,
            concurrency_key=view.concurrency_key,
            cpu_weight=view.cpu_weight,
            memory_mib=view.memory_mib,
            disk_mib=view.disk_mib,
            process_slots=view.process_slots,
            host_labels=view.host_labels,
            preferred_target=view.preferred_target,
            required_target=view.required_target,
            anti_affinity=view.anti_affinity,
            priority=view.priority,
            fairness_group=view.fairness_group,
            queue_deadline=view.queue_deadline,
            disk_low_watermark_mib=view.disk_low_watermark_mib,
            disk_high_watermark_mib=view.disk_high_watermark_mib,
        )
        return self.scheduler.admit(
            AdmissionRequest(
                work_item_id=view.work_item_id,
                attempt=attempt,
                fence=fence,
                resources=resources,
                job_id=view.job_id,
                repository_id=view.repository_id,
                branch=view.base_ref,
                owner_id=view.owner_id,
                tenant_id=view.tenant_id,
                ttl_seconds=self.lease_ttl_seconds,
                profile_name=view.resource_class,
            )
        )

    def _run_degraded(
        self,
        job_id: str,
        claim: Mapping[str, Any],
        view: DurableJobView,
        scope: Any,
        spec: bq.BuildSpec,
        descriptor: Mapping[str, Any],
        token: CancellationToken,
        reservation_id: str | None,
    ) -> dict[str, Any]:
        """Build an uncacheable request without publishing cache bytes.

        A dirty request is only allowed from a managed noncanonical lane.  A
        clean-but-unfingerprintable request still materializes the submitted
        SHA so a restart cannot accidentally execute a mutable checkout.
        """

        from contextlib import nullcontext

        dirty = descriptor.get("degraded_reason") == "dirty-tree"
        if dirty:
            raise BuildWorkerError(
                "dirty durable builds require an immutable typed snapshot; "
                "durable snapshot authority is not installed"
            )
        fence = _claim_fence(claim)
        context = (
            nullcontext(scope.tree)
            if dirty
            else bq.materialized(scope.tree, view.base_sha, scope=scope)
        )
        with context as build_tree:
            _verify_toolchain_fingerprint(
                build_tree,
                spec,
                descriptor,
                cacheable=False,
            )
            executor = self.executor or LocalExecutor(
                build_tree, worker_id=self.worker_id
            )
            command = ExecutionCommand(
                argv=spec.command,
                workdir=str((build_tree / spec.workdir).resolve()),
                timeout_seconds=spec.timeout,
                heartbeat_interval_seconds=min(30, max(1, spec.timeout // 4)),
            )
            result = executor.run(
                command,
                command_id=f"build:degraded:{job_id}",
                worker_id=self.worker_id,
                fence=fence,
                cancellation=token,
                fence_check=lambda: self.authority.is_current(job_id, claim),
                heartbeat=lambda: self.authority.heartbeat(job_id, claim),
            )
        if result.outcome == ExecutionOutcome.SUCCEEDED:
            result_ref = f"build-degraded:{job_id}:fence:{fence}"
            try:
                self.authority.commit(
                    job_id,
                    claim,
                    outcome="succeeded",
                    result_ref=result_ref,
                    retryable=False,
                )
            except Exception as exc:
                # A successful commit followed by a lost response is a
                # durable-success boundary, not permission to emit failure.
                if _terminal_matches(
                    self.authority, job_id, claim, result_ref=result_ref
                ):
                    return self._reconciliation_pending(
                        job_id,
                        view,
                        key=None,
                        error=str(exc),
                        reservation_id=reservation_id,
                    )
                raise
            state = "succeeded"
            ok = True
        else:
            state = (
                "cancelled"
                if result.outcome is ExecutionOutcome.CANCELLED
                else "failed"
            )
            current = self._current_view(job_id)
            already_cancelled = (
                state == "cancelled"
                and current is not None
                and current.state is JobState.CANCELLED
            )
            if not already_cancelled:
                if state == "cancelled":
                    self.authority.commit(
                        job_id,
                        claim,
                        outcome="cancelled",
                        refusal_code=FailureClass.CANCELLED_DEADLINE.value,
                        retryable=False,
                    )
                else:
                    self.authority.commit(
                        job_id,
                        claim,
                        outcome="failed",
                        failure_class=(
                            result.failure_class or FailureClass.INTERNAL_ERROR
                        ).value,
                        retryable=True,
                    )
            ok = False
        return {
            "ok": ok,
            "job_id": job_id,
            "state": state,
            "degraded": True,
            "degraded_reason": descriptor.get(
                "degraded_reason", "dirty-tree-or-unfingerprintable-toolchain"
            ),
            "cached": False,
            "reservation_id": reservation_id,
        }

    def _commit_execution_failure(
        self,
        job_id: str,
        claim: Mapping[str, Any],
        view: DurableJobView,
        result: Any,
        reservation_id: str | None,
    ) -> dict[str, Any]:
        failure = result.failure_class or FailureClass.INTERNAL_ERROR
        cancelled = result.outcome == ExecutionOutcome.CANCELLED
        current = self._current_view(job_id)
        already_cancelled = (
            cancelled and current is not None and current.state is JobState.CANCELLED
        )
        if not already_cancelled:
            if cancelled:
                self.authority.commit(
                    job_id,
                    claim,
                    outcome="cancelled",
                    refusal_code=FailureClass.CANCELLED_DEADLINE.value,
                    retryable=False,
                )
            else:
                self.authority.commit(
                    job_id,
                    claim,
                    outcome="failed",
                    failure_class=failure.value,
                    retryable=True,
                )
        return {
            "ok": False,
            "job_id": job_id,
            "state": "cancelled" if cancelled else "failed",
            "error": failure.value,
            "execution": result.model_dump(mode="json")
            if hasattr(result, "model_dump")
            else result,
            "reservation_id": reservation_id,
        }

    def _terminal_cancel(
        self,
        job_id: str,
        claim: Mapping[str, Any],
        view: DurableJobView,
        reason: str,
        reservation_id: str | None = None,
    ) -> dict[str, Any]:
        current = self._current_view(job_id)
        if current is None or current.state is not JobState.CANCELLED:
            self.authority.commit(
                job_id,
                claim,
                outcome="cancelled",
                error_ref=reason,
                refusal_code=FailureClass.CANCELLED_DEADLINE.value,
                retryable=False,
            )
        return {
            "ok": False,
            "job_id": job_id,
            "state": "cancelled",
            "error": reason,
            "reservation_id": reservation_id,
        }

    def _terminal_refusal(
        self,
        job_id: str,
        claim: Mapping[str, Any],
        view: DurableJobView,
        *,
        code: str,
        error: str = "",
        reservation_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            self.authority.commit(
                job_id,
                claim,
                outcome="failed",
                # AU intentionally rejects a result carrying both fields.  A
                # refusal code is the canonical terminal classification here;
                # execution failures use failure_class in their own path.
                refusal_code=code,
                error_ref=error or code,
                retryable=True,
            )
        except Exception as exc:
            # The WorkItem authority remains the source of truth.  If it did
            # not accept terminalization, report reconciliation rather than
            # claiming a failed terminal state that was never durable.
            return {
                "ok": False,
                "job_id": job_id,
                "work_item_id": view.work_item_id,
                "state": view.state.value,
                "terminalization_pending": True,
                "error": str(exc),
                "refusal_code": code,
                "reservation_id": reservation_id,
            }
        return {
            "ok": False,
            "job_id": job_id,
            "work_item_id": view.work_item_id,
            "state": "failed",
            "error": error or code,
            "refusal_code": code,
            "reservation_id": reservation_id,
        }

    @staticmethod
    def _refusal(job_id: str, reason: str, *, view: DurableJobView) -> dict[str, Any]:
        return {
            "ok": False,
            "job_id": job_id,
            "work_item_id": view.work_item_id,
            "state": view.state.value,
            "error": reason,
        }


__all__ = [
    "BuildAuthority",
    "BuildWorker",
    "BuildWorkerError",
    "GraphBuildAuthority",
]
