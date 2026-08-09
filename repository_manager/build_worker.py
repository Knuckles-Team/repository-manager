"""Durable build worker using WorkItem fencing and RMDD-08 admission."""

from __future__ import annotations

import hashlib
import json
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
from repository_manager.config_schema import load_yaml_mapping_text
from repository_manager.development import (
    DurableJobView,
    ExecutionCommand,
    ExecutionOutcome,
    FailureClass,
    JobState,
    ResourceRequest,
)
from repository_manager.development.jobs import (
    RepositoryJobServiceCode,
    RepositoryJobServiceError,
)
from repository_manager.development.payloads import (
    BuildExecutionDescriptor,
    operation_payload_from_mapping,
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

    def __init__(self, message: str, *, refusal_code: str | None = None) -> None:
        super().__init__(message)
        self.refusal_code = refusal_code


_MAX_CONFIG_BYTES = 1 << 20
_DEFAULT_STALE_STAGE_AGE_SECONDS = 24 * 60 * 60
_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE = "typed_execution_payload_authority_unavailable"
_EXECUTION_INPUT_AUTHORITY_MESSAGE = (
    "typed repository execution input authority is unavailable"
)
_AUTHORITY_COMMIT_FAILURE_MESSAGE = "durable WorkItem authority commit failed"
_WORKER_AUTHORITY_FAILURE_MESSAGE = "durable build worker authority operation failed"


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

    def execution_input_authority_available(self) -> bool: ...

    def is_current(self, job_id: str, claim: Mapping[str, Any]) -> bool: ...

    def terminal_matches(
        self, job_id: str, claim: Mapping[str, Any], *, result_ref: str
    ) -> bool: ...

    def get_exact_execution_input(
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


def _execution_input_authority_available(authority: object) -> bool:
    """Read the explicit pre-native availability marker without side effects."""

    marker = getattr(authority, "execution_input_authority_available", None)
    if not callable(marker):
        return False
    try:
        return bool(marker())
    except Exception:
        return False


def _authority_unavailable_result(job_id: str) -> dict[str, Any]:
    """Return the stable pre-native refusal before touching durable state."""

    return {
        "ok": False,
        "job_id": job_id,
        "state": "failed",
        "error": _EXECUTION_INPUT_AUTHORITY_MESSAGE,
        "refusal_code": _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
    }


def _require_current_fence(
    authority: BuildAuthority, job_id: str, claim: Mapping[str, Any]
) -> bool:
    """Require a current mutation fence after the atomic exact-input read."""

    try:
        current = bool(authority.is_current(job_id, claim))
    except Exception as exc:
        raise BuildWorkerError(
            _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
        ) from exc
    if not current:
        raise BuildWorkerError(
            _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
        )
    return True


def _authority_refusal_code(code: str) -> str:
    """Normalize explicit fence-denial statuses without parsing error text."""

    if code in {
        _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
        "fenced",
        "stale_fence",
    }:
        return _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
    return code


def _authority_exception_code(error: BaseException, *, fallback: str) -> str:
    """Map only trusted structured authority errors to the stable refusal code."""

    if isinstance(error, RepositoryJobServiceError):
        if error.code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE:
            return _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
    if isinstance(error, BuildWorkerError):
        if error.refusal_code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE:
            return _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
    return _authority_refusal_code(fallback)


def _commit_result_refusal_code(result: object) -> str | None:
    """Recognize the native fenced result without exposing its representation."""

    if isinstance(result, str) and result.strip().lower() == "fenced":
        return _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
    return None


def _raise_on_fenced_commit_result(result: object) -> None:
    refusal_code = _commit_result_refusal_code(result)
    if refusal_code is not None:
        raise BuildWorkerError(
            _EXECUTION_INPUT_AUTHORITY_MESSAGE,
            refusal_code=refusal_code,
        )


def _safe_authority_error(code: str) -> str:
    if code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE:
        return _EXECUTION_INPUT_AUTHORITY_MESSAGE
    return _AUTHORITY_COMMIT_FAILURE_MESSAGE


def _result_ref(key: str, fence: str) -> str:
    """Bind the terminal result reference to the committing fence."""

    return f"build-manifest:{key}:fence:{fence}"


def _degraded_result_ref(job_id: str, fence: str) -> str:
    """Bind a degraded terminal result to its exact job and fence."""

    return f"build-degraded:{job_id}:fence:{fence}"


def _degraded_fence(result_ref: str | None, job_id: str) -> str | None:
    """Extract a strictly job-bound degraded result fence."""

    if not isinstance(result_ref, str):
        return None
    prefix = f"build-degraded:{job_id}:fence:"
    if not result_ref.startswith(prefix):
        return None
    fence = result_ref[len(prefix) :]
    return fence if fence.strip() == fence and fence else None


def _manifest_key_from_result_ref(result_ref: str | None) -> str | None:
    """Extract only the bounded cache address from terminal evidence."""

    if not isinstance(result_ref, str) or not result_ref.startswith("build-manifest:"):
        return None
    value = result_ref[len("build-manifest:") :]
    key, marker, fence = value.partition(":fence:")
    if (
        marker != ":fence:"
        or len(key) != 35
        or not key.startswith("v2:")
        or any(char not in "0123456789abcdef" for char in key[3:])
        or not fence
        or fence.strip() != fence
    ):
        return None
    return key


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


def _execution_digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_contract_digest(spec: bq.BuildSpec) -> str:
    return _execution_digest(
        {
            "patterns": list(spec.artifact_contract.patterns),
            "required": spec.artifact_contract.required,
            "publish": spec.artifact_contract.publish,
            "retention": spec.artifact_contract.retention,
        }
    )


def _verify_toolchain_fingerprint(
    build_tree: Path,
    spec: bq.BuildSpec,
    descriptor: BuildExecutionDescriptor,
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

    components = {item.name: item.value for item in descriptor.cache_key_components}
    expected = components.get("toolchain_fingerprint", "")
    expected_digest = _execution_digest(
        {"toolchain_fingerprint": expected if cacheable else "unavailable"}
    )
    if descriptor.toolchain_digest != expected_digest:
        raise BuildWorkerError(
            "persisted build toolchain digest does not match its cache identity"
        )
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

    def __init__(
        self,
        engine: Any,
        *,
        tenant_id: str,
        token: str,
    ) -> None:
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

    def execution_input_authority_available(self) -> bool:
        """EG-native atomic exact-input authority is not installed yet."""

        return False

    def get_exact_execution_input(self, job_id: str) -> BuildExecutionDescriptor | None:
        """Refuse until EG supplies one atomic authorized exact-input read."""

        raise RepositoryJobServiceError(
            RepositoryJobServiceCode.TYPED_EXECUTION_PAYLOAD_AUTHORITY_UNAVAILABLE,
            "typed repository execution input authority is unavailable",
            job_id=job_id,
        )

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
        del job_id, claim
        return False

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

        if not _execution_input_authority_available(self.authority):
            return _authority_unavailable_result("")
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
        if not _execution_input_authority_available(self.authority):
            return _authority_unavailable_result(job_id)
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
            scope, spec, key, payload = self._execution_plan(
                view,
                repo_path=repo_path,
                spec_name=spec_name,
                claim=actual_claim,
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
                return self._terminal_refusal(
                    job_id,
                    actual_claim,
                    view,
                    code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
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
                    payload,
                    token,
                    reservation_id=reservation_id,
                )

            with bq.materialized(scope.tree, view.base_sha, scope=scope) as build_tree:
                _verify_toolchain_fingerprint(
                    build_tree,
                    spec,
                    payload,
                    cacheable=True,
                )
                command = ExecutionCommand(
                    argv=payload.argv,
                    workdir=str((build_tree / payload.workdir).resolve()),
                    environment_refs=payload.environment_refs,
                    timeout_seconds=payload.timeout_seconds,
                    heartbeat_interval_seconds=min(
                        30, max(1, payload.timeout_seconds // 4)
                    ),
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
                    fence_check=lambda: _require_current_fence(
                        self.authority, job_id, actual_claim
                    ),
                    heartbeat=lambda: self.authority.heartbeat(job_id, actual_claim),
                )
                if result.outcome != ExecutionOutcome.SUCCEEDED:
                    return self._commit_execution_failure(
                        job_id, actual_claim, view, result, reservation_id
                    )
                staged = store.stage(
                    build_tree,
                    workdir=payload.workdir,
                    patterns=payload.artifact_patterns,
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
                    fence_check=lambda: _require_current_fence(
                        self.authority, job_id, actual_claim
                    ),
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
                        error=_AUTHORITY_COMMIT_FAILURE_MESSAGE,
                        reservation_id=reservation_id,
                    )
                code = _authority_exception_code(
                    exc, fallback="worker_environment_failure"
                )
                raise BuildWorkerError(
                    _safe_authority_error(code),
                    refusal_code=code,
                ) from exc
            commit_accepted = commit_result is None or (
                isinstance(commit_result, str)
                and commit_result in {"None", "committed", "noop", "succeeded"}
            )
            if not commit_accepted:
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
                        error=_AUTHORITY_COMMIT_FAILURE_MESSAGE,
                        reservation_id=reservation_id,
                    )
                commit_refusal_code = _commit_result_refusal_code(commit_result)
                if commit_refusal_code is not None:
                    raise BuildWorkerError(
                        _EXECUTION_INPUT_AUTHORITY_MESSAGE,
                        refusal_code=commit_refusal_code,
                    )
                raise BuildWorkerError(
                    _AUTHORITY_COMMIT_FAILURE_MESSAGE,
                    refusal_code="worker_environment_failure",
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
                code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
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
        except BuildWorkerError as exc:
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
            refusal_code = _authority_refusal_code(
                exc.refusal_code or "worker_environment_failure"
            )
            return self._terminal_refusal(
                job_id,
                actual_claim,
                view,
                code=refusal_code,
                error=(
                    _EXECUTION_INPUT_AUTHORITY_MESSAGE
                    if refusal_code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
                    else str(exc)
                ),
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
            code = _authority_exception_code(exc, fallback="worker_environment_failure")
            return self._terminal_refusal(
                job_id,
                actual_claim,
                view,
                code=code,
                error=_safe_authority_error(code)
                if code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
                else _WORKER_AUTHORITY_FAILURE_MESSAGE,
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
        key: bq.CacheKey | str,
        manifest: Mapping[str, Any],
        view: DurableJobView,
        reason: str,
    ) -> None:
        """Quarantine a terminal entry only after exact owner proof."""

        key_digest = key if isinstance(key, str) else key.digest

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
                key_digest,
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
        key: bq.CacheKey | str | None,
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
            "key": (
                key if isinstance(key, str) else key.digest if key is not None else None
            ),
            "error": error,
            "reservation_id": reservation_id,
        }

    def _execution_plan(
        self,
        view: DurableJobView,
        *,
        repo_path: Path | str,
        spec_name: str,
        claim: Mapping[str, Any] | None = None,
    ) -> tuple[
        Any,
        bq.BuildSpec,
        bq.CacheKey | None,
        BuildExecutionDescriptor,
    ]:
        if not _execution_input_authority_available(self.authority):
            raise BuildWorkerError(
                _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            )
        scope = bq.lane_scope(repo_path)
        if view.repository_id != bq.stable_repository_id(scope.main_tree):
            raise BuildWorkerError(
                "worker repository identity does not match the WorkItem"
            )
        if view.operation != "build":
            raise BuildWorkerError(
                "durable build WorkItem operation is not a build operation"
            )
        if claim is None:
            # Every executable-input read follows a current worker claim.  A
            # terminal recovery never calls this planner without a live claim.
            raise BuildWorkerError(
                _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            )
        _validate_claim_identity(view.job_id, view, claim)
        get_exact = getattr(self.authority, "get_exact_execution_input", None)
        if not callable(get_exact):
            raise BuildWorkerError(
                _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            )
        try:
            # This is the sole executable-input authority operation.  The
            # production adapter must perform authentication, currentness,
            # and private-row/body access atomically in native code.  Python
            # intentionally has no separate issue/verify/read capability API.
            raw_payload = get_exact(view.job_id)
        except RepositoryJobServiceError as exc:
            code = exc.code
            if code == "typed_execution_payload_required":
                raise BuildWorkerError(
                    "typed_execution_payload_required: resubmit build",
                    refusal_code="invalid_request",
                ) from exc
            if code == "input_conflict":
                raise BuildWorkerError(
                    "input_conflict: persisted build payload is invalid",
                    refusal_code="invalid_request",
                ) from exc
            if code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE:
                raise BuildWorkerError(
                    _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                    refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                ) from exc
            raise BuildWorkerError(
                _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            ) from exc
        except Exception as exc:
            raise BuildWorkerError(
                _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
                refusal_code=_EXECUTION_INPUT_AUTHORITY_UNAVAILABLE,
            ) from exc
        if raw_payload is None:
            raise BuildWorkerError(
                "typed_execution_payload_required: resubmit build",
                refusal_code="invalid_request",
            )
        try:
            payload = operation_payload_from_mapping(raw_payload)
        except (TypeError, ValueError) as exc:
            raise BuildWorkerError(
                "durable build operation payload is invalid"
            ) from exc
        # A native atomic read authenticates the private payload.  This second
        # check is only the post-read mutation-fence proof; it is never used to
        # authorize the preceding private read.
        _require_current_fence(self.authority, view.job_id, claim)
        if payload.repository_id != view.repository_id:
            raise BuildWorkerError(
                "build payload repository identity disagrees with WorkItem"
            )
        if payload.base_sha != view.base_sha:
            raise BuildWorkerError(
                "build payload SHA disagrees with WorkItem authority"
            )
        expected_profile = (
            f"repository_manager:resource_profile:{view.resource_class}:v1"
        )
        if payload.profile_ref != expected_profile:
            raise BuildWorkerError(
                "build payload resource profile disagrees with WorkItem"
            )
        if payload.execution_policy_ref != "repository.build-policy:v1":
            raise BuildWorkerError(
                "build payload execution policy is not the approved policy"
            )
        if view.target_kind != "local":
            raise BuildWorkerError(
                "build WorkItem target is not the submitted local target"
            )
        descriptor_generation = payload.generation_id
        view_generation = view.generation_id or None
        if descriptor_generation != view_generation:
            raise BuildWorkerError(
                "build payload generation disagrees with WorkItem authority"
            )
        if (
            view.config_digest is not None
            and payload.config_digest != view.config_digest
        ):
            raise BuildWorkerError(
                "build payload config digest disagrees with WorkItem authority"
            )
        persisted_spec = payload.build_spec_name
        if spec_name and spec_name != persisted_spec:
            raise BuildWorkerError(
                "worker spec selection disagrees with persisted payload"
            )
        dirty = payload.degraded_reason == "dirty-tree"
        if dirty:
            raise BuildWorkerError(
                "dirty durable builds require an immutable typed snapshot; "
                "durable snapshot authority is not installed"
            )
        config, config_digest = _config_snapshot(
            scope,
            base_sha=payload.base_sha,
            dirty=dirty,
        )
        if payload.config_digest != config_digest:
            raise BuildWorkerError(
                "build payload config digest does not match the submitted snapshot"
            )
        try:
            snapshot_spec = config.spec(persisted_spec)
        except bq.BuildQueueError as exc:
            raise BuildWorkerError(
                "submitted build config does not contain the persisted spec"
            ) from exc
        if payload.spec_digest != bq._spec_digest(snapshot_spec):  # noqa: SLF001
            raise BuildWorkerError(
                "build payload spec digest does not match the submitted snapshot"
            )
        if (
            payload.argv != snapshot_spec.command
            or payload.workdir != snapshot_spec.workdir
            or payload.artifact_patterns
            != tuple(sorted(dict.fromkeys(snapshot_spec.artifacts)))
            or payload.timeout_seconds != snapshot_spec.timeout
            or payload.feature_set != " ".join(snapshot_spec.command)
            or payload.target_triple != bq._target_triple(snapshot_spec)  # noqa: SLF001
            or payload.artifact_contract_digest
            != _artifact_contract_digest(snapshot_spec)
            or payload.profile_ref
            != f"repository_manager:resource_profile:{snapshot_spec.resource_class}:v1"
        ):
            raise BuildWorkerError(
                "build payload execution fields disagree with the submitted snapshot"
            )
        spec = snapshot_spec
        cacheable = payload.cacheable is True
        key: bq.CacheKey | None = None
        if cacheable:
            components = {
                item.name: item.value for item in payload.cache_key_components
            }
            key = _key_from_components(components)
            expected_components = {
                "repo": view.repository_id,
                "spec": snapshot_spec.name,
                "tree_sha": payload.tree_sha,
                "feature_set": " ".join(snapshot_spec.command),
                "target_triple": bq._target_triple(snapshot_spec),  # noqa: SLF001
            }
            for component, expected in expected_components.items():
                if getattr(key, component) != expected:
                    raise BuildWorkerError(
                        "persisted build key "
                        f"{component} component disagrees with the submitted snapshot"
                    )
            if payload.cache_key_digest != key.digest:
                raise BuildWorkerError(
                    "build payload cache key does not match its components"
                )
            if payload.config_digest != key.config_digest:
                raise BuildWorkerError(
                    "build payload config digest does not match its key"
                )
            if payload.spec_digest != key.spec_digest:
                raise BuildWorkerError(
                    "build payload spec digest does not match its key"
                )
            if (payload.generation_id or "") != key.generation_id:
                raise BuildWorkerError(
                    "build payload generation does not match its key"
                )
            try:
                submitted_tree_sha = _paths_tree_sha_at(
                    scope.main_tree,
                    payload.base_sha,
                    snapshot_spec.cache_key_paths,
                )
            except (bq.BuildQueueError, OSError) as exc:
                raise BuildWorkerError(
                    "submitted build SHA tree could not be verified"
                ) from exc
            if (
                key.tree_sha != submitted_tree_sha
                or payload.tree_sha != submitted_tree_sha
            ):
                raise BuildWorkerError(
                    "build payload tree digest disagrees with submitted SHA"
                )
        else:
            try:
                submitted_tree_sha = bq._require_git(  # noqa: SLF001
                    ["rev-parse", f"{payload.base_sha}^{{tree}}"], scope.main_tree
                )
            except (bq.BuildQueueError, OSError) as exc:
                raise BuildWorkerError(
                    "submitted build SHA tree could not be verified"
                ) from exc
            if payload.tree_sha != submitted_tree_sha:
                raise BuildWorkerError(
                    "uncacheable build payload tree digest disagrees with submitted SHA"
                )
        if not cacheable and payload.cache_key_digest is not None:
            raise BuildWorkerError(
                "uncacheable build payload unexpectedly carries a cache key"
            )
        components = {item.name: item.value for item in payload.cache_key_components}
        expected_toolchain = (
            components["toolchain_fingerprint"] if cacheable else "unavailable"
        )
        if payload.toolchain_digest != _execution_digest(
            {"toolchain_fingerprint": expected_toolchain}
        ):
            raise BuildWorkerError(
                "build payload toolchain digest disagrees with its key"
            )
        return scope, spec, key, payload

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

        if not _execution_input_authority_available(self.authority):
            return _authority_unavailable_result(job_id)
        view = _as_view(self.authority.get(job_id))
        scope = bq.lane_scope(repo_path)
        if view.repository_id != bq.stable_repository_id(scope.main_tree):
            raise BuildWorkerError(
                "worker repository identity does not match the WorkItem"
            )
        store = self.artifact_store or BuildArtifactStore(repo_path=scope.tree)
        staging_reconciliation = self._reconcile_staging(store)

        def with_reconciliation(result: Mapping[str, Any]) -> dict[str, Any]:
            enriched = dict(result)
            enriched.setdefault("staging_reconciliation", staging_reconciliation)
            return enriched

        if view.state is not JobState.SUCCEEDED:
            if view.state in {
                JobState.CANCELLED,
                JobState.FAILED,
                JobState.DEAD_LETTER,
            }:
                return with_reconciliation(
                    {
                        "ok": False,
                        "recovered": False,
                        "state": view.state.value,
                        "error": "terminal WorkItem cannot be rerun",
                    }
                )
            # A live/reclaimable job is claimed before its exact payload is
            # fetched.  This keeps stale workers from reading or executing
            # executable input under an old fence.
            return with_reconciliation(
                self.run_job(job_id, repo_path=repo_path, spec_name=spec_name)
            )

        # Terminal recovery never opens the executable payload.  Degraded
        # success is proved solely by its job-bound result reference; a
        # cacheable success carries only the cache address in its terminal
        # evidence, which is sufficient to reconcile the artifact manifest.
        degraded_result_ref = view.result_ref
        degraded_fence = _degraded_fence(degraded_result_ref, job_id)
        if degraded_fence is not None:
            if (
                view.job_id != job_id
                or view.attempt < 1
                or (view.lease_fence is not None and view.lease_fence != degraded_fence)
                or not _terminal_matches(
                    self.authority,
                    job_id,
                    {
                        "job_id": view.job_id,
                        "work_item_id": view.work_item_id,
                        "attempt": view.attempt,
                        "fence": degraded_fence,
                    },
                    result_ref=degraded_result_ref or "",
                )
            ):
                return with_reconciliation(
                    {
                        "ok": False,
                        "recovered": False,
                        "state": view.state.value,
                        "degraded": True,
                        "error": (
                            "succeeded degraded WorkItem has no exact "
                            "job/fence terminal result proof"
                        ),
                    }
                )
            return with_reconciliation(
                {
                    "ok": True,
                    "recovered": True,
                    "state": view.state.value,
                    "degraded": True,
                    "result_ref": degraded_result_ref,
                }
            )

        key_digest = _manifest_key_from_result_ref(view.result_ref)
        if key_digest is None:
            return with_reconciliation(
                {
                    "ok": False,
                    "recovered": False,
                    "state": view.state.value,
                    "error": (
                        "succeeded WorkItem has no exact job/fence terminal "
                        "artifact reference"
                    ),
                }
            )
        manifest = store.read_manifest(key_digest)
        manifest_fence = str(manifest.get("fence") or "") if manifest else ""
        result_ref = _result_ref(key_digest, manifest_fence)
        if (
            manifest
            and store.validate_manifest(
                manifest, require_committed=False, expected_key=key_digest
            )
            and view.result_ref == result_ref
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
                    key_digest,
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
                        key=key_digest,
                        error=str(exc),
                    )
                )
            return with_reconciliation(
                {
                    "ok": True,
                    "recovered": True,
                    "state": view.state.value,
                    "key": key_digest,
                    "artifacts": committed.get("artifacts", []),
                }
            )
        if manifest:
            self._quarantine_terminal_invalid(
                store,
                key=key_digest,
                manifest=manifest,
                view=view,
                reason="terminal-evidence-mismatch",
            )
        return with_reconciliation(
            {
                "ok": False,
                "recovered": False,
                "state": view.state.value,
                "key": key_digest,
                "error": "succeeded WorkItem has no exact published artifact evidence",
            }
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
        scheduler = self.scheduler
        if scheduler is None:
            raise BuildWorkerError("resource scheduler is required")
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
        return scheduler.admit(
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
        payload: BuildExecutionDescriptor,
        token: CancellationToken,
        reservation_id: str | None,
    ) -> dict[str, Any]:
        """Build an uncacheable request without publishing cache bytes.

        A dirty request is only allowed from a managed noncanonical lane.  A
        clean-but-unfingerprintable request still materializes the submitted
        SHA so a restart cannot accidentally execute a mutable checkout.
        """

        from contextlib import nullcontext

        dirty = payload.degraded_reason == "dirty-tree"
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
                payload,
                cacheable=False,
            )
            executor = self.executor or LocalExecutor(
                build_tree, worker_id=self.worker_id
            )
            command = ExecutionCommand(
                argv=payload.argv,
                workdir=str((build_tree / payload.workdir).resolve()),
                environment_refs=payload.environment_refs,
                timeout_seconds=payload.timeout_seconds,
                heartbeat_interval_seconds=min(
                    30, max(1, payload.timeout_seconds // 4)
                ),
            )
            result = executor.run(
                command,
                command_id=f"build:degraded:{job_id}",
                worker_id=self.worker_id,
                fence=fence,
                cancellation=token,
                fence_check=lambda: _require_current_fence(
                    self.authority, job_id, claim
                ),
                heartbeat=lambda: self.authority.heartbeat(job_id, claim),
            )
        result_ref: str | None = None
        if result.outcome == ExecutionOutcome.SUCCEEDED:
            result_ref = _degraded_result_ref(job_id, fence)
            try:
                commit_result = self.authority.commit(
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
                        error=_AUTHORITY_COMMIT_FAILURE_MESSAGE,
                        reservation_id=reservation_id,
                    )
                code = _authority_exception_code(
                    exc, fallback="worker_environment_failure"
                )
                raise BuildWorkerError(
                    _safe_authority_error(code),
                    refusal_code=code,
                ) from exc
            commit_accepted = commit_result is None or (
                isinstance(commit_result, str)
                and commit_result in {"None", "committed", "noop", "succeeded"}
            )
            if not commit_accepted:
                if _terminal_matches(
                    self.authority, job_id, claim, result_ref=result_ref
                ):
                    return self._reconciliation_pending(
                        job_id,
                        view,
                        key=None,
                        error=_AUTHORITY_COMMIT_FAILURE_MESSAGE,
                        reservation_id=reservation_id,
                    )
                commit_refusal_code = _commit_result_refusal_code(commit_result)
                if commit_refusal_code is not None:
                    raise BuildWorkerError(
                        _EXECUTION_INPUT_AUTHORITY_MESSAGE,
                        refusal_code=commit_refusal_code,
                    )
                raise BuildWorkerError(
                    _AUTHORITY_COMMIT_FAILURE_MESSAGE,
                    refusal_code="worker_environment_failure",
                )
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
                    commit_result = self.authority.commit(
                        job_id,
                        claim,
                        outcome="cancelled",
                        refusal_code=FailureClass.CANCELLED_DEADLINE.value,
                        retryable=False,
                    )
                    _raise_on_fenced_commit_result(commit_result)
                else:
                    commit_result = self.authority.commit(
                        job_id,
                        claim,
                        outcome="failed",
                        failure_class=(
                            result.failure_class or FailureClass.INTERNAL_ERROR
                        ).value,
                        retryable=True,
                    )
                    _raise_on_fenced_commit_result(commit_result)
            ok = False
        return {
            "ok": ok,
            "job_id": job_id,
            "state": state,
            "degraded": True,
            "degraded_reason": payload.degraded_reason
            or "dirty-tree-or-unfingerprintable-toolchain",
            "cached": False,
            "result_ref": result_ref,
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
                commit_result = self.authority.commit(
                    job_id,
                    claim,
                    outcome="cancelled",
                    refusal_code=FailureClass.CANCELLED_DEADLINE.value,
                    retryable=False,
                )
                _raise_on_fenced_commit_result(commit_result)
            else:
                commit_result = self.authority.commit(
                    job_id,
                    claim,
                    outcome="failed",
                    failure_class=failure.value,
                    retryable=True,
                )
                _raise_on_fenced_commit_result(commit_result)
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
            commit_result = self.authority.commit(
                job_id,
                claim,
                outcome="cancelled",
                error_ref=reason,
                refusal_code=FailureClass.CANCELLED_DEADLINE.value,
                retryable=False,
            )
            _raise_on_fenced_commit_result(commit_result)
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
        refusal_code = _authority_refusal_code(code)
        try:
            commit_result = self.authority.commit(
                job_id,
                claim,
                outcome="failed",
                # AU intentionally rejects a result carrying both fields.  A
                # refusal code is the canonical terminal classification here;
                # execution failures use failure_class in their own path.
                refusal_code=refusal_code,
                error_ref=error or refusal_code,
                retryable=True,
            )
            _raise_on_fenced_commit_result(commit_result)
        except Exception as exc:
            # The WorkItem authority remains the source of truth.  If it did
            # not accept terminalization, report reconciliation rather than
            # claiming a failed terminal state that was never durable.
            normalized_code = _authority_exception_code(exc, fallback=refusal_code)
            return {
                "ok": False,
                "job_id": job_id,
                "work_item_id": view.work_item_id,
                "state": view.state.value,
                "terminalization_pending": True,
                "error": _safe_authority_error(normalized_code),
                "refusal_code": normalized_code,
                "reservation_id": reservation_id,
            }
        safe_error = (
            _EXECUTION_INPUT_AUTHORITY_MESSAGE
            if refusal_code == _EXECUTION_INPUT_AUTHORITY_UNAVAILABLE
            else error or refusal_code
        )
        return {
            "ok": False,
            "job_id": job_id,
            "work_item_id": view.work_item_id,
            "state": "failed",
            "error": safe_error,
            "refusal_code": refusal_code,
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
