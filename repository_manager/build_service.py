"""Durable build submission and cache lookup.

``BuildService`` is deliberately small: it computes the immutable build key,
checks a validated artifact manifest, and submits a cache miss through the
existing :class:`RepositoryJobService`.  It does not start a process, acquire
capacity, or maintain a local job map.  Those effects belong to
``BuildWorker`` and the frozen scheduler/WorkItem seams.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from repository_manager import build_queue as bq
from repository_manager.build_artifacts import BuildArtifactStore
from repository_manager.development import (
    JobAuthorization,
    JobState,
    TargetKind,
)
from repository_manager.development.jobs import (
    JobSubmitResult,
    RepositoryJobService,
)
from repository_manager.development.payloads import (
    BuildExecutionDescriptor,
    RepositoryCacheKeyComponent,
    operation_payload_from_mapping,
)
from repository_manager.resource_profiles import default_resource_profiles


class BuildServiceError(RuntimeError):
    """A durable build request could not be constructed or submitted."""


_DIRTY_SNAPSHOT_MAX_BYTES = 1 << 20


def _terminate_process(process: subprocess.Popen[bytes]) -> None:
    """Best-effort cleanup for a bounded producer that exceeded its budget."""

    try:
        if process.poll() is None:
            process.kill()
    except OSError:
        pass
    try:
        process.wait(timeout=5)
    except (OSError, subprocess.SubprocessError):
        pass


def _bounded_git_output(command: tuple[str, ...], root: Path, limit: int) -> bytes:
    """Collect at most ``limit`` bytes, terminating a noisy Git command."""

    process = subprocess.Popen(
        list(command),
        cwd=str(root),
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )
    try:
        assert process.stdout is not None
        output = process.stdout.read(limit + 1)
        if len(output) > limit:
            _terminate_process(process)
            raise BuildServiceError("dirty build snapshot exceeds the durable bound")
        returncode = process.wait(timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        _terminate_process(process)
        raise BuildServiceError("dirty build snapshot could not be collected") from exc
    finally:
        if process.stdout is not None:
            process.stdout.close()
    if returncode != 0:
        raise BuildServiceError("dirty build snapshot could not be collected")
    return output


def _read_bounded_untracked(path: Path, limit: int) -> bytes:
    """Read one untracked regular file through a no-follow descriptor."""

    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        initial = os.fstat(descriptor)
        if not stat.S_ISREG(initial.st_mode):
            raise BuildServiceError("dirty build snapshot found a non-regular file")
        if initial.st_size > limit:
            raise BuildServiceError("dirty build snapshot exceeds the durable bound")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            content = handle.read(limit + 1)
        if len(content) > limit:
            raise BuildServiceError("dirty build snapshot exceeds the durable bound")
        final = os.stat(path, follow_symlinks=False)
        if (
            final.st_dev != initial.st_dev
            or final.st_ino != initial.st_ino
            or final.st_size != len(content)
        ):
            raise BuildServiceError("dirty build snapshot file changed while reading")
        return content
    except OSError as exc:
        raise BuildServiceError(
            "dirty build snapshot could not read an untracked file"
        ) from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _snapshot_untracked_entry(
    root: Path, raw_name: bytes, total: int
) -> tuple[bytes, int]:
    """One untracked-file entry for the dirty snapshot.

    Returns ``(part_bytes, new_total)``. Raises :class:`BuildServiceError` on
    any bound violation or unsafe path — never silently truncates or skips.
    """
    try:
        relative = raw_name.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise BuildServiceError(
            "dirty build snapshot contains invalid path data"
        ) from exc
    candidate = root / relative
    try:
        candidate.resolve(strict=False).relative_to(root)
    except ValueError as exc:
        raise BuildServiceError(
            "dirty build snapshot path escapes the repository"
        ) from exc
    try:
        candidate_stat = os.lstat(candidate)
    except OSError as exc:
        raise BuildServiceError(
            "dirty build snapshot could not inspect an untracked path"
        ) from exc
    if stat.S_ISLNK(candidate_stat.st_mode):
        try:
            content = os.readlink(candidate).encode("utf-8", "surrogateescape")
        except OSError as exc:
            raise BuildServiceError(
                "dirty build snapshot could not read an untracked symlink"
            ) from exc
        if (
            total + len(relative.encode("utf-8")) + len(content)
            > _DIRTY_SNAPSHOT_MAX_BYTES
        ):
            raise BuildServiceError("dirty build snapshot exceeds the durable bound")
    elif stat.S_ISREG(candidate_stat.st_mode):
        remaining = _DIRTY_SNAPSHOT_MAX_BYTES - total - len(relative.encode("utf-8"))
        content = _read_bounded_untracked(candidate, remaining)
    else:
        content = b"<non-regular>"
    new_total = total + len(relative.encode("utf-8")) + len(content)
    if new_total > _DIRTY_SNAPSHOT_MAX_BYTES:
        raise BuildServiceError("dirty build snapshot exceeds the durable bound")
    return relative.encode("utf-8") + b"\0" + content, new_total


def dirty_snapshot_digest(tree: Path | str) -> str:
    """Hash a bounded dirty-tree snapshot for honest uncacheable execution."""

    root = Path(tree).resolve(strict=True)
    commands = (
        ("git", "status", "--porcelain=v1", "-z", "--untracked-files=all"),
        ("git", "diff", "--binary", "--no-ext-diff", "--no-textconv", "HEAD", "--"),
        ("git", "ls-files", "--others", "--exclude-standard", "-z"),
    )
    parts: list[bytes] = []
    outputs: list[bytes] = []
    total = 0
    for command in commands:
        output = _bounded_git_output(command, root, _DIRTY_SNAPSHOT_MAX_BYTES - total)
        total += len(output)
        outputs.append(output)
        parts.append(command[1].encode("utf-8") + b"\0" + output)

    untracked = outputs[-1].split(b"\0")
    for raw_name in untracked:
        if not raw_name:
            continue
        part, total = _snapshot_untracked_entry(root, raw_name, total)
        parts.append(part)
    return hashlib.sha256(b"".join(parts)).hexdigest()


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _artifact_contract_digest(spec: bq.BuildSpec) -> str:
    """Digest only the declared artifact contract, not mutable build state."""

    return _digest(
        {
            "patterns": list(spec.artifact_contract.patterns),
            "required": spec.artifact_contract.required,
            "publish": spec.artifact_contract.publish,
            "retention": spec.artifact_contract.retention,
        }
    )


def _toolchain_digest(fingerprint: str) -> str:
    """Bind the diagnostic toolchain fingerprint to the typed payload."""

    return _digest({"toolchain_fingerprint": fingerprint})


@dataclass(frozen=True)
class _CurrentBuildInputs:
    """Freshly recomputed build inputs, for comparison against a submitted
    typed payload in :func:`_assert_identity_and_contract_unchanged` and its
    siblings — see :func:`_recompute_current_build_inputs`.
    """

    head: str
    spec: bq.BuildSpec
    repo_id: str
    key: bq.CacheKey
    config_digest: str
    spec_digest: str


def _recompute_current_build_inputs(
    tree: Path, spec_name: str, generation_id: str | None
) -> _CurrentBuildInputs:
    try:
        current_head = bq._require_git(["rev-parse", "HEAD"], tree)  # noqa: SLF001
        current_config = bq.load_config(tree)
        current_spec = current_config.spec(spec_name)
        current_repo_id = bq.stable_repository_id(bq.lane_scope(tree).main_tree)
        current_key = bq.compute_cache_key(
            tree,
            current_spec,
            repo_name=current_repo_id,
            generation_id=generation_id,
        )
        current_config_digest = bq._config_digest(tree)  # noqa: SLF001
        current_spec_digest = bq._spec_digest(current_spec)  # noqa: SLF001
    except Exception as exc:
        if isinstance(exc, BuildServiceError):
            raise
        raise BuildServiceError(
            "build submission inputs could not be revalidated"
        ) from exc
    return _CurrentBuildInputs(
        head=current_head,
        spec=current_spec,
        repo_id=current_repo_id,
        key=current_key,
        config_digest=current_config_digest,
        spec_digest=current_spec_digest,
    )


def _assert_identity_unchanged(payload: Any, current: _CurrentBuildInputs) -> None:
    if payload.repository_id != current.repo_id:
        raise BuildServiceError("build submission repository identity changed")
    if payload.base_sha != current.head:
        raise BuildServiceError(
            "build submission HEAD changed before durable submission"
        )
    if payload.config_digest != current.config_digest:
        raise BuildServiceError(
            "build submission config changed before durable submission"
        )
    if payload.spec_digest != current.spec_digest:
        raise BuildServiceError(
            "build submission spec changed before durable submission"
        )


def _assert_spec_contract_unchanged(payload: Any, current: _CurrentBuildInputs) -> None:
    if payload.build_spec_name != current.spec.name:
        raise BuildServiceError(
            "build submission spec changed before durable submission"
        )
    if payload.argv != current.spec.command:
        raise BuildServiceError(
            "build submission command changed before durable submission"
        )
    if payload.workdir != current.spec.workdir:
        raise BuildServiceError(
            "build submission workdir changed before durable submission"
        )
    normalized_artifacts = tuple(sorted(dict.fromkeys(current.spec.artifacts)))
    if payload.artifact_patterns != normalized_artifacts:
        raise BuildServiceError(
            "build submission artifact contract changed before durable submission"
        )
    if payload.timeout_seconds != current.spec.timeout:
        raise BuildServiceError(
            "build submission timeout changed before durable submission"
        )
    if payload.feature_set != " ".join(current.spec.command):
        raise BuildServiceError(
            "build submission feature set changed before durable submission"
        )
    if payload.target_triple != bq._target_triple(current.spec):  # noqa: SLF001
        raise BuildServiceError(
            "build submission target changed before durable submission"
        )


def _assert_identity_and_contract_unchanged(
    payload: Any, current: _CurrentBuildInputs
) -> None:
    _assert_identity_unchanged(payload, current)
    _assert_spec_contract_unchanged(payload, current)


def _assert_generation_and_policy_unchanged(
    payload: Any,
    current: _CurrentBuildInputs,
    key: bq.CacheKey,
    generation_id: str | None,
) -> None:
    if payload.generation_id != (generation_id or current.key.generation_id or None):
        raise BuildServiceError(
            "build submission generation changed before durable submission"
        )
    if payload.artifact_contract_digest != _artifact_contract_digest(current.spec):
        raise BuildServiceError(
            "build submission artifact contract changed before durable submission"
        )
    expected_toolchain = (
        current.key.toolchain_fingerprint if current.key.computable else "unavailable"
    )
    if payload.toolchain_digest != _toolchain_digest(expected_toolchain):
        raise BuildServiceError(
            "build submission toolchain identity changed before durable submission"
        )
    if payload.execution_policy_ref != "repository.build-policy:v1":
        raise BuildServiceError(
            "build submission execution policy changed before durable submission"
        )
    if payload.cacheable != key.computable:
        raise BuildServiceError(
            "build submission cacheability changed before durable submission"
        )
    if payload.profile_ref != (
        f"repository_manager:resource_profile:{current.spec.resource_class}:v1"
    ):
        raise BuildServiceError(
            "build submission resource profile changed before durable submission"
        )


def _assert_cache_key_components_unchanged(
    payload: Any, current: _CurrentBuildInputs, key: bq.CacheKey
) -> None:
    if current.key.components() != key.components():
        raise BuildServiceError(
            "build submission key inputs changed before durable submission"
        )
    current_components = current.key.components()
    payload_components = {
        component.name: component.value for component in payload.cache_key_components
    }
    if payload_components != current_components:
        raise BuildServiceError(
            "build submission key inputs changed before durable submission"
        )


def _assert_tree_identity_unchanged(
    payload: Any, current: _CurrentBuildInputs, tree: Path
) -> None:
    if payload.cacheable is True:
        if payload.cache_key_digest != current.key.digest:
            raise BuildServiceError(
                "build submission cache key changed before durable submission"
            )
        if payload.tree_sha != current.key.tree_sha:
            raise BuildServiceError(
                "build submission tree key changed before durable submission"
            )
        return
    try:
        submitted_tree_sha = bq._require_git(  # noqa: SLF001
            ["rev-parse", f"{current.head}^{{tree}}"], tree
        )
    except Exception as exc:
        raise BuildServiceError(
            "build submission tree identity could not be revalidated"
        ) from exc
    if payload.tree_sha != submitted_tree_sha:
        raise BuildServiceError(
            "build submission tree identity changed before durable submission"
        )
    if payload.degraded_reason != current.key.degraded_reason:
        raise BuildServiceError(
            "build submission degradation state changed before durable submission"
        )


def _revalidate_head_and_key(
    tree: Path, spec: bq.BuildSpec, generation_id: str | None, key: bq.CacheKey
) -> tuple[str, bq.CacheKey, bq.BuildSpec, bq.BuildConfig]:
    """Re-resolve HEAD/spec/key immediately before durable submission.

    Raises :class:`BuildServiceError` if the tree is dirty, the inputs can't
    be revalidated, or the key's components have drifted since ``key`` was
    computed. Returns ``(base_sha, current_key, current_spec, current_config)``
    — the caller rebinds its own ``key``/``spec``/``config`` to these, exactly
    as the inline version did.
    """
    try:
        if bq._tree_is_dirty(tree):  # noqa: SLF001
            raise BuildServiceError(
                "build submission tree changed or became dirty before durable submission"
            )
        base_sha = bq._require_git(["rev-parse", "HEAD"], tree)  # noqa: SLF001
    except Exception as exc:
        if isinstance(exc, BuildServiceError):
            raise
        raise BuildServiceError(
            "build submission could not resolve immutable HEAD"
        ) from exc
    try:
        current_config = bq.load_config(tree)
        current_spec = current_config.spec(spec.name)
        current_key = bq.compute_cache_key(
            tree,
            current_spec,
            repo_name=bq.stable_repository_id(bq.lane_scope(tree).main_tree),
            generation_id=generation_id,
        )
    except Exception as exc:
        if isinstance(exc, BuildServiceError):
            raise
        raise BuildServiceError(
            "build submission inputs could not be revalidated"
        ) from exc
    if current_key.components() != key.components():
        raise BuildServiceError(
            "build submission key inputs changed before durable submission"
        )
    return base_sha, current_key, current_spec, current_config


def _build_resource_payload(spec: bq.BuildSpec) -> dict[str, Any]:
    profile = default_resource_profiles().get(spec.resource_class)
    if profile is None:
        raise BuildServiceError(
            f"build spec {spec.name!r} names unknown resource profile {spec.resource_class!r}"
        )
    resources = spec.resources
    preferred_target: dict[str, Any] = {"kind": "local"}
    required_target: dict[str, Any] | None = None
    if spec.placement.preferred_host:
        preferred_target = {
            "kind": "inventory_alias",
            "alias": spec.placement.preferred_host,
        }
    if spec.placement.required_host:
        required_target = {
            "kind": "inventory_alias",
            "alias": spec.placement.required_host,
        }
    return {
        "resource_class": spec.resource_class,
        "concurrency_key": profile.concurrency_key,
        "cpu_weight": max(1, resources.cpu_weight, profile.cpu_weight),
        "memory_mib": max(1, resources.memory_mb or profile.memory_mib),
        "disk_mib": max(1, resources.disk_mb or profile.disk_mib),
        "process_slots": max(1, resources.process_slots, profile.process_slots),
        "host_labels": sorted(
            set(spec.placement.required_labels).union(profile.required_labels)
        ),
        "preferred_target": preferred_target,
        "required_target": required_target,
        "anti_affinity": sorted(
            set(spec.placement.anti_affinity).union(profile.anti_affinity)
        ),
        "fairness_group": profile.default_fairness_group,
        "disk_low_watermark_mib": profile.disk_low_watermark_mib,
        "disk_high_watermark_mib": profile.disk_high_watermark_mib,
    }


def _resolve_submission_tree_sha(
    tree: Path, base_sha: str, key: bq.CacheKey, cacheable: bool
) -> tuple[str, str]:
    """``(tree_sha, toolchain_fingerprint)`` for the durable payload."""
    if cacheable:
        return key.tree_sha, key.toolchain_fingerprint
    try:
        tree_sha = bq._require_git(  # noqa: SLF001
            ["rev-parse", f"{base_sha}^{{tree}}"], tree
        )
    except Exception as exc:
        raise BuildServiceError(
            "uncacheable build payload could not resolve submitted tree"
        ) from exc
    return tree_sha, "unavailable"


@dataclass(frozen=True)
class _DescriptorInputs:
    repository_id: str
    base_sha: str
    tree_sha: str
    generation_id: str | None
    spec: bq.BuildSpec
    spec_digest: str
    config_digest: str
    toolchain_fingerprint: str
    key: bq.CacheKey
    cacheable: bool


def _build_execution_descriptor(
    inputs: _DescriptorInputs,
) -> BuildExecutionDescriptor:
    try:
        return BuildExecutionDescriptor(
            repository_id=inputs.repository_id,
            base_sha=inputs.base_sha,
            tree_sha=inputs.tree_sha,
            generation_id=inputs.generation_id or inputs.key.generation_id or None,
            build_spec_name=inputs.spec.name,
            spec_digest=inputs.spec_digest,
            config_digest=inputs.config_digest,
            toolchain_digest=_toolchain_digest(inputs.toolchain_fingerprint),
            artifact_contract_digest=_artifact_contract_digest(inputs.spec),
            feature_set=" ".join(inputs.spec.command),
            target_triple=bq._target_triple(inputs.spec),  # noqa: SLF001
            cache_key_components=tuple(
                sorted(
                    (
                        RepositoryCacheKeyComponent(name=name, value=value)
                        for name, value in inputs.key.components().items()
                    ),
                    key=lambda item: item.name,
                )
            ),
            cache_key_digest=inputs.key.digest if inputs.cacheable else None,
            argv=inputs.spec.command,
            workdir=inputs.spec.workdir,
            timeout_seconds=inputs.spec.timeout,
            artifact_patterns=inputs.spec.artifacts,
            environment_refs=(),
            execution_policy_ref="repository.build-policy:v1",
            profile_ref=f"repository_manager:resource_profile:{inputs.spec.resource_class}:v1",
            cacheable=inputs.cacheable,
            degraded_reason=inputs.key.degraded_reason,
        )
    except (TypeError, ValueError) as exc:
        raise BuildServiceError("build execution payload is invalid") from exc


def _validate_job_service_shape(job_service: Any) -> None:
    if not isinstance(job_service, RepositoryJobService):
        # Keep this boundary structural for test doubles and for the
        # production graph adapter's delayed import, while failing early for
        # an accidental task queue or executor injection.
        if not callable(getattr(job_service, "submit", None)):
            raise TypeError("BuildService requires a RepositoryJobService-like object")


def _stale_candidate_ids(
    candidate: Mapping[str, Any],
) -> tuple[str, str, str, int] | None:
    """``(job_id, work_item_id, fence, attempt)`` or ``None`` if malformed."""
    job_id = str(candidate.get("job_id") or "")
    work_item_id = str(candidate.get("work_item_id") or "")
    fence = str(candidate.get("fence") or "")
    try:
        attempt = int(candidate.get("attempt", 0))
    except (TypeError, ValueError):
        return None
    if not job_id or not work_item_id or not fence or attempt < 1:
        return None
    return job_id, work_item_id, fence, attempt


def _staleness_proof(
    view: Any, key: str, job_id: str, work_item_id: str, fence: str, attempt: int
) -> Mapping[str, Any]:
    proof = {
        "job_id": job_id,
        "work_item_id": work_item_id,
        "attempt": attempt,
        "fence": fence,
        "stale": True,
    }
    if view.state is JobState.SUCCEEDED:
        expected = f"build-manifest:{key}:fence:{fence}"
        return proof if view.result_ref != expected else {}
    if view.state in {
        JobState.FAILED,
        JobState.CANCELLED,
        JobState.DEAD_LETTER,
    }:
        return proof
    stale = view.attempt != attempt or (
        view.lease_fence is not None and view.lease_fence != fence
    )
    return proof if stale else {}


def _resolve_service_identity(
    tenant_id: str | None, owner_id: str | None, auth: JobAuthorization | None
) -> tuple[str, str]:
    if auth is not None:
        if tenant_id is not None and tenant_id != auth.tenant_id:
            raise ValueError("tenant_id disagrees with the authenticated authorization")
        if owner_id is not None and owner_id != auth.owner_id:
            raise ValueError("owner_id disagrees with the authenticated authorization")
        return auth.tenant_id, auth.owner_id
    tenant = tenant_id or "repository-manager"
    owner = owner_id or "repository-manager"
    if not tenant.strip() or not owner.strip():
        raise ValueError("tenant_id and owner_id must be non-blank")
    return tenant, owner


def _legacy_cache_hit(
    store: BuildArtifactStore, key: bq.CacheKey
) -> dict[str, Any] | None:
    """A validated legacy (v1) manifest hit, or ``None``.

    Read legacy bytes in place. Never write a v2 alias that points into the
    legacy directory: legacy GC must remain independent of v2 ownership.
    """
    legacy = store.read_manifest(key.legacy_digest)
    if legacy is None or not store.validate_manifest(
        legacy, require_committed=False, expected_key=key.legacy_digest
    ):
        return None
    return {
        "ok": True,
        "cached": True,
        "degraded": False,
        "outcome": "hit",
        "key": key.digest,
        "components": key.components(),
        "artifacts": legacy.get("artifacts", []),
        "built_at": legacy.get("built_at"),
        "migrated_from": key.legacy_digest,
    }


def _local_cache_hit(
    manifest: Mapping[str, Any], key: bq.CacheKey, migrated_from: str | None
) -> dict[str, Any]:
    return {
        "ok": True,
        "cached": True,
        "degraded": False,
        "outcome": "hit",
        "key": key.digest,
        "components": key.components(),
        "artifacts": manifest.get("artifacts", []),
        "built_at": manifest.get("built_at"),
        **({"migrated_from": migrated_from} if migrated_from else {}),
    }


class BuildService:
    """Cache-aware durable build request service.

    A valid cache hit returns directly and never calls the job service.  A
    miss submits one idempotent ``build`` WorkItem and returns its durable ID;
    separate service objects therefore deduplicate through the same authority
    instead of coordinating through process-local futures.
    """

    def __init__(
        self,
        # Structural, not nominal: the isinstance-or-callable check below is
        # the actual boundary, deliberately kept permissive for test doubles
        # and the production graph adapter's delayed import.
        job_service: RepositoryJobService | Any,
        *,
        tenant_id: str | None = None,
        owner_id: str | None = None,
        session_id: str = "build-request",
        auth: JobAuthorization | None = None,
        artifact_store: BuildArtifactStore | None = None,
        max_attempts: int = 3,
    ) -> None:
        _validate_job_service_shape(job_service)
        tenant, owner = _resolve_service_identity(tenant_id, owner_id, auth)
        if not 1 <= max_attempts <= 100:
            raise ValueError("max_attempts must be between 1 and 100")
        self.job_service = job_service
        self.auth = auth or JobAuthorization(tenant_id=tenant, owner_id=owner)
        if not session_id.strip():
            raise ValueError("session_id must be non-blank")
        self.session_id = session_id
        self.artifact_store = artifact_store
        self.max_attempts = max_attempts

    def key(
        self,
        *,
        repo_path: Path | str | None = None,
        spec_name: str = "",
        generation_id: str | None = None,
    ) -> tuple[bq.CacheKey, bq.BuildSpec, Any, Path]:
        scope = bq.lane_scope(repo_path)
        config = bq.load_config(scope.tree)
        spec = config.spec(spec_name)
        key = bq.compute_cache_key(
            scope.tree,
            spec,
            repo_name=bq.stable_repository_id(scope.main_tree),
            generation_id=generation_id,
        )
        return key, spec, config, scope.tree

    def submit(
        self,
        *,
        repo_path: Path | str | None = None,
        spec_name: str = "",
        generation_id: str | None = None,
        wait_timeout: int = 0,
    ) -> dict[str, Any]:
        """Return a direct cache hit or an immediate durable job handle."""

        del wait_timeout  # Durable submission never blocks on the producer.
        key, spec, config, tree = self.key(
            repo_path=repo_path,
            spec_name=spec_name,
            generation_id=generation_id,
        )
        if not key.computable:
            return self._submit_uncacheable(
                key=key,
                spec=spec,
                config=config,
                tree=tree,
                generation_id=generation_id,
            )

        store = self._store(tree)
        manifest = store.read_manifest(key.digest)
        manifest, quarantined = self._quarantine_if_corrupt(store, key, manifest)
        migrated_from: str | None = None
        if manifest is None:
            legacy_hit = _legacy_cache_hit(store, key)
            if legacy_hit is not None:
                return legacy_hit
        if manifest is not None and store.validate_manifest(
            manifest, expected_key=key.digest
        ):
            return _local_cache_hit(manifest, key, migrated_from)

        request = self._request_mapping(
            key=key,
            spec=spec,
            config=config,
            tree=tree,
            generation_id=generation_id,
            cacheable=True,
        )
        self._assert_request_inputs_current(
            request,
            key=key,
            tree=tree,
            spec_name=spec.name,
            generation_id=generation_id,
        )
        result = self._submit_durable(request)
        return self._submission_result(
            result,
            key=key,
            spec=spec,
            degraded=False,
            quarantined=quarantined,
        )

    request = submit

    def status(
        self,
        *,
        job_id: str | None = None,
        repo_path: Path | str | None = None,
        key: str = "",
        spec_name: str = "",
        generation_id: str | None = None,
    ) -> dict[str, Any]:
        """Return durable status or cache status without local job state."""

        if job_id:
            view = self.job_service.get(job_id, auth=self.auth)
            return {"ok": True, "durable": True, "job": view.model_dump(mode="json")}
        computed, _spec, _config, _tree = self.key(
            repo_path=repo_path,
            spec_name=spec_name,
            generation_id=generation_id,
        )
        report = bq.status(repo_path=repo_path, key=key, spec_name=spec_name)
        report["key_version"] = computed.key_version
        report["components"] = computed.components()
        return report

    def cancel(
        self,
        job_id: str,
        *,
        reason: str = "cancelled by owner",
        wait_only: bool = False,
    ) -> dict[str, Any]:
        del reason, wait_only
        # A BuildService handle may be a deduplicated waiter.  Its ordinary
        # cancel operation is therefore local-only and can never cancel the
        # shared producer.  Producer/admin cancellation is deliberately a
        # separate authenticated operation below.
        return {
            "ok": True,
            "durable": True,
            "job_id": job_id,
            "wait_cancelled": True,
            "producer_cancelled": False,
        }

    def cancel_producer(
        self, job_id: str, *, reason: str = "cancelled by owner"
    ) -> dict[str, Any]:
        """Cancel through the authenticated owner/admin control surface."""

        result = self.job_service.cancel(job_id, auth=self.auth, reason=reason)
        return {"ok": True, "durable": True, **result.model_dump(mode="json")}

    def _store(self, tree: Path) -> BuildArtifactStore:
        return self.artifact_store or BuildArtifactStore(repo_path=tree)

    def _quarantine_if_corrupt(
        self,
        store: BuildArtifactStore,
        key: bq.CacheKey,
        manifest: dict[str, Any] | None,
    ) -> tuple[dict[str, Any] | None, bool]:
        """``(manifest, quarantined)``.

        A published-but-not-committed manifest is a normal crash window owned
        by the durable producer.  Never delete it merely because this
        controller did not observe the terminal marker; quarantine requires
        an authority proof that this exact owner is stale or terminally
        invalid.
        """
        if manifest is None or store.validate_manifest(
            manifest, expected_key=key.digest
        ):
            return manifest, False
        quarantined = self._quarantine_if_stale(
            store, key.digest, manifest, reason="checksum-or-manifest-corrupt"
        )
        if quarantined:
            return None, True
        return manifest, False

    def _quarantine_if_stale(
        self,
        store: BuildArtifactStore,
        key: str,
        manifest: Mapping[str, Any],
        *,
        reason: str,
    ) -> bool:
        """Quarantine only with exact durable owner/fence evidence."""

        def proves_stale(candidate: Mapping[str, Any]) -> Mapping[str, Any]:
            ids = _stale_candidate_ids(candidate)
            if ids is None:
                return {}
            job_id, work_item_id, fence, attempt = ids
            try:
                view = self.job_service.get(job_id, auth=self.auth)
            except Exception:
                return {}
            if view is None or view.work_item_id != work_item_id:
                return {}
            return _staleness_proof(view, key, job_id, work_item_id, fence, attempt)

        return (
            store.quarantine_if(
                key,
                authority_check=proves_stale,
                reason=reason,
            )
            is not None
        )

    def _submit_durable(self, request: Mapping[str, Any]) -> JobSubmitResult:
        """Submit one exact typed payload through the normal job service."""

        raw_payload = request.get("operation_payload")
        try:
            payload_model = operation_payload_from_mapping(raw_payload)
        except (TypeError, ValueError) as exc:
            raise BuildServiceError(
                "durable build operation payload is invalid"
            ) from exc
        payload = dict(request)
        payload["operation_payload"] = payload_model.model_dump(
            mode="json", exclude_none=False
        )
        # The ordinary service persists the sibling extension atomically with
        # the WorkItem.  Never move executable bytes into correlation or a
        # second descriptor-specific submission API.
        return self.job_service.submit(
            payload,
            auth=self.auth,
            max_attempts=self.max_attempts,
        )

    def _submit_uncacheable(
        self,
        *,
        key: bq.CacheKey,
        spec: bq.BuildSpec,
        config: bq.BuildConfig,
        tree: Path,
        generation_id: str | None,
    ) -> dict[str, Any]:
        if key.degraded_reason == "dirty-tree":
            # A digest is evidence about bytes observed at one instant, not
            # an immutable execution snapshot.  Until the typed authority can
            # persist a bounded managed-lane snapshot, running a dirty tree
            # would let an editor change compiler inputs after admission (and
            # make restart recovery execute different bytes).  Keep the
            # compatibility/local queue path separate and fail the durable
            # path closed.
            raise BuildServiceError(
                "dirty durable builds require an immutable typed snapshot; "
                "durable snapshot authority is not installed"
            )
        request = self._request_mapping(
            key=key,
            spec=spec,
            config=config,
            tree=tree,
            generation_id=generation_id,
            cacheable=False,
        )
        self._assert_request_inputs_current(
            request,
            key=key,
            tree=tree,
            spec_name=spec.name,
            generation_id=generation_id,
        )
        result = self._submit_durable(request)
        value = self._submission_result(
            result,
            key=key,
            spec=spec,
            degraded=True,
            quarantined=False,
        )
        value["degraded_reason"] = key.degraded_reason
        value["uncacheable"] = True
        return value

    def _request_mapping(
        self,
        *,
        key: bq.CacheKey,
        spec: bq.BuildSpec,
        config: bq.BuildConfig,
        tree: Path,
        generation_id: str | None,
        cacheable: bool,
    ) -> dict[str, Any]:
        base_sha, key, spec, config = _revalidate_head_and_key(
            tree, spec, generation_id, key
        )
        resource_payload = _build_resource_payload(spec)
        scope = bq.lane_scope(tree)
        repository_id = bq.stable_repository_id(scope.main_tree)
        config_digest = bq._config_digest(tree)  # noqa: SLF001
        spec_digest = bq._spec_digest(spec)  # noqa: SLF001
        tree_sha, toolchain_fingerprint = _resolve_submission_tree_sha(
            tree, base_sha, key, cacheable
        )
        payload_model = _build_execution_descriptor(
            _DescriptorInputs(
                repository_id=repository_id,
                base_sha=base_sha,
                tree_sha=tree_sha,
                generation_id=generation_id,
                spec=spec,
                spec_digest=spec_digest,
                config_digest=config_digest,
                toolchain_fingerprint=toolchain_fingerprint,
                key=key,
                cacheable=cacheable,
            )
        )
        # For cacheable requests the C-05 key is the idempotency identity. For
        # degraded requests the complete typed body is the identity, so a
        # changed executable input cannot reuse the same WorkItem.
        identity = payload_model.cache_key_digest or payload_model.payload_digest
        assert identity is not None
        idempotency_key = f"build:{identity}"
        request_id = f"build:{identity}:request"
        return {
            "contract_version": "1",
            "request_id": request_id,
            "idempotency_key": idempotency_key,
            "operation": "build",
            "repository_id": repository_id,
            "base_ref": config.base,
            "base_sha": base_sha,
            "owner_id": self.auth.owner_id,
            "session_id": self.session_id,
            "tenant_id": self.auth.tenant_id,
            "generation_id": generation_id,
            "config_digest": config_digest,
            "input_digest": payload_model.payload_digest,
            # Public correlation remains a bounded opaque source relationship;
            # executable input is exclusively the typed sibling below.
            "correlation_id": request_id,
            "operation_payload": payload_model.model_dump(
                mode="json", exclude_none=False
            ),
            "resources": resource_payload,
            "target": {"kind": TargetKind.LOCAL.value},
            "consent": {},
        }

    def _assert_request_inputs_current(
        self,
        request: Mapping[str, Any],
        *,
        key: bq.CacheKey,
        tree: Path,
        spec_name: str,
        generation_id: str | None,
    ) -> None:
        """Recheck immutable inputs immediately before typed submission."""

        try:
            payload = operation_payload_from_mapping(request.get("operation_payload"))
        except (TypeError, ValueError) as exc:
            raise BuildServiceError(
                "durable build request has no valid typed operation payload"
            ) from exc
        if bq._tree_is_dirty(tree):  # noqa: SLF001
            raise BuildServiceError(
                "build submission tree changed or became dirty before durable submission"
            )
        current = _recompute_current_build_inputs(tree, spec_name, generation_id)
        _assert_identity_and_contract_unchanged(payload, current)
        _assert_generation_and_policy_unchanged(payload, current, key, generation_id)
        _assert_cache_key_components_unchanged(payload, current, key)
        _assert_tree_identity_unchanged(payload, current, tree)

    @staticmethod
    def _submission_result(
        result: JobSubmitResult,
        *,
        key: bq.CacheKey,
        spec: bq.BuildSpec,
        degraded: bool,
        quarantined: bool,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "cached": False,
            "durable": True,
            "queued": True,
            "degraded": degraded,
            "outcome": "degraded_uncacheable" if degraded else "produced_miss",
            "key": key.digest if key.computable else None,
            "components": key.components(),
            "spec": spec.name,
            "job_id": result.job.job_id,
            "work_item_id": result.job.work_item_id,
            "state": result.job.state.value,
            "deduplicated": result.deduplicated,
            "waiter": result.deduplicated,
            "quarantined": quarantined,
        }


# Names used by downstream lanes during the RMDD-10 handoff.
BuildRequestService = BuildService


__all__ = [
    "BuildExecutionDescriptor",
    "BuildRequestService",
    "BuildService",
    "BuildServiceError",
    "dirty_snapshot_digest",
]
