"""Durable build submission and cache lookup.

``BuildService`` is deliberately small: it computes the immutable build key,
checks a validated artifact manifest, and submits a cache miss through the
existing :class:`RepositoryJobService`.  It does not start a process, acquire
capacity, or maintain a local job map.  Those effects belong to
``BuildWorker`` and the frozen scheduler/WorkItem seams.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import stat
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    field_validator,
    model_validator,
)

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
from repository_manager.resource_profiles import default_resource_profiles


class BuildServiceError(RuntimeError):
    """A durable build request could not be constructed or submitted."""


_DESCRIPTOR_PREFIX = "build-descriptor:v1:"
_DESCRIPTOR_MAX_BYTES = 12_288
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
                raise BuildServiceError(
                    "dirty build snapshot exceeds the durable bound"
                )
        elif stat.S_ISREG(candidate_stat.st_mode):
            remaining = (
                _DIRTY_SNAPSHOT_MAX_BYTES - total - len(relative.encode("utf-8"))
            )
            content = _read_bounded_untracked(candidate, remaining)
        else:
            content = b"<non-regular>"
        total += len(relative.encode("utf-8")) + len(content)
        if total > _DIRTY_SNAPSHOT_MAX_BYTES:
            raise BuildServiceError("dirty build snapshot exceeds the durable bound")
        parts.append(relative.encode("utf-8") + b"\0" + content)
    return hashlib.sha256(b"".join(parts)).hexdigest()


class BuildExecutionDescriptor(BaseModel):
    """Bounded typed execution input owned by the durable build extension.

    This model is passed through the future typed Repository WorkItem
    extension, not smuggled into arbitrary WorkItem fields.  Until the
    authority exposes that extension, :class:`BuildService` refuses durable
    submission rather than pretending a correlation string is executable
    state.
    """

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    version: StrictInt = Field(default=1, ge=1, le=1)
    repository_id: StrictStr
    base_sha: StrictStr
    # ``tree_sha`` in the v2 key may intentionally cover only declared
    # cache-key paths.  Keep the exact submitted HEAD separately so a
    # mutation between key computation and durable submission cannot change
    # the immutable execution snapshot underneath the request.
    head_sha: StrictStr
    spec: StrictStr
    command: tuple[StrictStr, ...]
    toolchain_command: tuple[StrictStr, ...] = ()
    workdir: StrictStr
    artifacts: tuple[StrictStr, ...]
    timeout: StrictInt = Field(ge=1, le=86_400)
    resource_class: StrictStr = "light-check"
    cacheable: StrictBool
    cache_key: StrictStr | None = None
    legacy_cache_key: StrictStr | None = None
    key_components: Mapping[str, StrictStr] = Field(default_factory=dict)
    generation_id: StrictStr | None = None
    config_digest: StrictStr | None = None
    spec_digest: StrictStr
    input_digest: StrictStr
    toolchain_fingerprint: StrictStr
    dirty_snapshot_digest: StrictStr | None = None
    degraded_reason: StrictStr = ""

    @field_validator("command", "artifacts", mode="before")
    @classmethod
    def normalize_sequences(cls, value: object) -> tuple[str, ...]:
        if not isinstance(value, (tuple, list)):
            raise ValueError("build descriptor sequences must be lists or tuples")
        result = tuple(value)
        if not result or any(not isinstance(item, str) or not item for item in result):
            raise ValueError(
                "build descriptor sequences must contain non-empty strings"
            )
        return result

    @field_validator("toolchain_command", mode="before")
    @classmethod
    def normalize_toolchain_command(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if not isinstance(value, (tuple, list)):
            raise ValueError("toolchain command must be a list or tuple")
        result = tuple(value)
        if any(not isinstance(item, str) or not item for item in result):
            raise ValueError("toolchain command entries must be non-empty strings")
        return result

    @field_validator("workdir")
    @classmethod
    def validate_workdir(cls, value: str) -> str:
        path = Path(value)
        if not value or path.is_absolute() or ".." in path.parts:
            raise ValueError("build descriptor workdir must be relative and contained")
        return value

    @model_validator(mode="after")
    def validate_cache_contract(self) -> BuildExecutionDescriptor:
        if self.cacheable and not self.cache_key:
            raise ValueError("cacheable build descriptor requires cache_key")
        if not self.cacheable and self.cache_key is not None:
            raise ValueError("uncacheable build descriptor must not carry cache_key")
        if len(self.command) > 128 or len(self.toolchain_command) > 128:
            raise ValueError("build descriptor command is too large")
        if len(self.artifacts) > 128 or len(self.key_components) > 64:
            raise ValueError("build descriptor is too large")
        if (
            len(
                json.dumps(
                    self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
                ).encode("utf-8")
            )
            > _DESCRIPTOR_MAX_BYTES
        ):
            raise ValueError("build descriptor exceeds the durable bound")
        return self


def encode_build_descriptor(value: Mapping[str, Any]) -> str:
    """Return a bounded candidate envelope for local migration experiments.

    Production submission deliberately does not use this representation:
    the typed WorkItem build extension owns descriptor persistence.
    """

    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    encoded = base64.urlsafe_b64encode(payload).decode("ascii").rstrip("=")
    result = _DESCRIPTOR_PREFIX + encoded
    if len(result.encode("utf-8")) > _DESCRIPTOR_MAX_BYTES:
        raise BuildServiceError("build execution descriptor exceeds the durable bound")
    return result


def decode_build_descriptor(value: str | None) -> dict[str, Any] | None:
    """Decode and validate a persisted build descriptor, failing closed."""

    if not isinstance(value, str) or not value.startswith(_DESCRIPTOR_PREFIX):
        return None
    if len(value.encode("utf-8")) > _DESCRIPTOR_MAX_BYTES:
        raise BuildServiceError("persisted build execution descriptor is too large")
    encoded = value[len(_DESCRIPTOR_PREFIX) :]
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        decoded = json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BuildServiceError(
            "persisted build execution descriptor is invalid"
        ) from exc
    if not isinstance(decoded, dict) or decoded.get("version") != 1:
        raise BuildServiceError(
            "persisted build execution descriptor has an unknown version"
        )
    return decoded


def _digest(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def descriptor_input_digest(descriptor: Mapping[str, Any]) -> str:
    """Recompute the immutable descriptor input digest at worker boundaries."""

    immutable = {
        "key": dict(descriptor.get("key_components") or {}),
        "repository_id": descriptor.get("repository_id"),
        "base_sha": descriptor.get("base_sha"),
        "head_sha": descriptor.get("head_sha"),
        "config_digest": descriptor.get("config_digest"),
        "spec_digest": descriptor.get("spec_digest"),
        "command": list(descriptor.get("command") or ()),
        "toolchain_command": list(descriptor.get("toolchain_command") or ()),
        "workdir": descriptor.get("workdir"),
        "artifacts": list(descriptor.get("artifacts") or ()),
        "timeout": descriptor.get("timeout"),
        "resource_class": descriptor.get("resource_class"),
        "cacheable": descriptor.get("cacheable"),
        "generation_id": descriptor.get("generation_id"),
        "dirty_snapshot_digest": descriptor.get("dirty_snapshot_digest"),
        "degraded_reason": descriptor.get("degraded_reason"),
    }
    return _digest(immutable)


class BuildService:
    """Cache-aware durable build request service.

    A valid cache hit returns directly and never calls the job service.  A
    miss submits one idempotent ``build`` WorkItem and returns its durable ID;
    separate service objects therefore deduplicate through the same authority
    instead of coordinating through process-local futures.
    """

    def __init__(
        self,
        job_service: RepositoryJobService,
        *,
        tenant_id: str | None = None,
        owner_id: str | None = None,
        session_id: str = "build-request",
        auth: JobAuthorization | None = None,
        artifact_store: BuildArtifactStore | None = None,
        max_attempts: int = 3,
    ) -> None:
        if not isinstance(job_service, RepositoryJobService):
            # Keep this boundary structural for test doubles and for the
            # production graph adapter's delayed import, while failing early
            # for an accidental task queue or executor injection.
            if not callable(getattr(job_service, "submit", None)):
                raise TypeError(
                    "BuildService requires a RepositoryJobService-like object"
                )
        if auth is not None:
            if tenant_id is not None and tenant_id != auth.tenant_id:
                raise ValueError(
                    "tenant_id disagrees with the authenticated authorization"
                )
            if owner_id is not None and owner_id != auth.owner_id:
                raise ValueError(
                    "owner_id disagrees with the authenticated authorization"
                )
            tenant = auth.tenant_id
            owner = auth.owner_id
        else:
            tenant = tenant_id or "repository-manager"
            owner = owner_id or "repository-manager"
            if not tenant.strip() or not owner.strip():
                raise ValueError("tenant_id and owner_id must be non-blank")
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
        quarantined = False
        if manifest is not None and not store.validate_manifest(
            manifest, expected_key=key.digest
        ):
            # A published-but-not-committed manifest is a normal crash window
            # owned by the durable producer.  Never delete it merely because
            # this controller did not observe the terminal marker; quarantine
            # requires an authority proof that this exact owner is stale or
            # terminally invalid.
            quarantined = self._quarantine_if_stale(
                store, key.digest, manifest, reason="checksum-or-manifest-corrupt"
            )
            if quarantined:
                manifest = None
        migrated_from: str | None = None
        if manifest is None:
            legacy = store.read_manifest(key.legacy_digest)
            if legacy is not None and store.validate_manifest(
                legacy, require_committed=False, expected_key=key.legacy_digest
            ):
                # Read legacy bytes in place.  Do not write a v2 alias that
                # points into the legacy directory: legacy GC must remain
                # independent of v2 ownership.
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
        if manifest is not None and store.validate_manifest(
            manifest, expected_key=key.digest
        ):
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
            job_id = str(candidate.get("job_id") or "")
            work_item_id = str(candidate.get("work_item_id") or "")
            fence = str(candidate.get("fence") or "")
            try:
                attempt = int(candidate.get("attempt", 0))
            except (TypeError, ValueError):
                return {}
            if not job_id or not work_item_id or not fence or attempt < 1:
                return {}
            try:
                view = self.job_service.get(job_id, auth=self.auth)
            except Exception:
                return {}
            if view is None or view.work_item_id != work_item_id:
                return {}
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

        return (
            store.quarantine_if(
                key,
                authority_check=proves_stale,
                reason=reason,
            )
            is not None
        )

    def _submit_durable(self, request: Mapping[str, Any]) -> JobSubmitResult:
        """Submit only through the typed build WorkItem extension."""

        raw_descriptor = request.get("build_descriptor")
        if not isinstance(raw_descriptor, Mapping):
            raise BuildServiceError(
                "durable build request has no typed execution descriptor"
            )
        try:
            descriptor = BuildExecutionDescriptor.model_validate(raw_descriptor)
        except ValueError as exc:
            raise BuildServiceError(
                "durable build execution descriptor is invalid"
            ) from exc
        submit_build = getattr(self.job_service, "submit_build", None)
        if not callable(submit_build):
            raise BuildServiceError(
                "RepositoryJobService lacks the typed build descriptor extension; "
                "durable build submission is fail-closed until that contract is installed"
            )
        payload = dict(request)
        # The descriptor is a typed extension payload, never an arbitrary
        # WorkItem/request field.  Keep it local for validation and hand it to
        # the extension through its explicit parameter only.
        payload.pop("build_descriptor", None)
        return submit_build(
            payload,
            descriptor=descriptor,
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
        key = current_key
        spec = current_spec
        config = current_config
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
        resource_payload = {
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
        scope = bq.lane_scope(tree)
        repository_id = bq.stable_repository_id(scope.main_tree)
        dirty_digest = (
            dirty_snapshot_digest(tree) if key.degraded_reason == "dirty-tree" else None
        )
        immutable = {
            "key": key.components(),
            "repository_id": repository_id,
            "base_sha": base_sha,
            "head_sha": base_sha,
            "command": list(spec.command),
            "toolchain_command": list(spec.toolchain_fingerprint),
            "workdir": spec.workdir,
            "artifacts": list(spec.artifacts),
            "timeout": spec.timeout,
            "resource_class": spec.resource_class,
            "cacheable": cacheable,
            "generation_id": key.generation_id,
            "dirty_snapshot_digest": dirty_digest,
            "config_digest": bq._config_digest(tree),  # noqa: SLF001
            "spec_digest": bq._spec_digest(spec),  # noqa: SLF001
            "degraded_reason": key.degraded_reason,
        }
        immutable_digest = _digest(immutable)
        idempotency_key = f"build:{key.digest if cacheable else immutable_digest}"
        request_id = f"build:{key.digest if cacheable else immutable_digest}:request"
        descriptor = BuildExecutionDescriptor(
            repository_id=repository_id,
            base_sha=base_sha,
            head_sha=base_sha,
            spec=spec.name,
            command=spec.command,
            toolchain_command=spec.toolchain_fingerprint,
            workdir=spec.workdir,
            artifacts=spec.artifacts,
            timeout=spec.timeout,
            resource_class=spec.resource_class,
            cacheable=cacheable,
            cache_key=key.digest if cacheable else None,
            legacy_cache_key=key.legacy_digest if cacheable else None,
            key_components=key.components(),
            generation_id=key.generation_id,
            # Keep the exact execution snapshot bound even for degraded jobs;
            # cache keys omit this field only when they cannot be trusted.
            config_digest=bq._config_digest(tree),  # noqa: SLF001
            spec_digest=key.spec_digest or bq._spec_digest(spec),  # noqa: SLF001
            input_digest=immutable_digest,
            toolchain_fingerprint=key.toolchain_fingerprint or "unavailable",
            dirty_snapshot_digest=dirty_digest,
            degraded_reason=key.degraded_reason,
        )
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
            "config_digest": key.config_digest or None,
            "input_digest": immutable_digest,
            # Public correlation remains a small request identity.  The
            # descriptor is handed separately to the typed build extension;
            # current AU projections intentionally drop arbitrary fields.
            "correlation_id": request_id,
            "build_descriptor": descriptor.model_dump(mode="json"),
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

        descriptor = request.get("build_descriptor")
        if not isinstance(descriptor, Mapping):
            raise BuildServiceError("durable build request has no typed descriptor")
        if bq._tree_is_dirty(tree):  # noqa: SLF001
            raise BuildServiceError(
                "build submission tree changed or became dirty before durable submission"
            )
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
        if descriptor.get("repository_id") != current_repo_id:
            raise BuildServiceError("build submission repository identity changed")
        if (
            descriptor.get("base_sha") != current_head
            or descriptor.get("head_sha") != current_head
        ):
            raise BuildServiceError(
                "build submission HEAD changed before durable submission"
            )
        if descriptor.get("config_digest") != current_config_digest:
            raise BuildServiceError(
                "build submission config changed before durable submission"
            )
        if descriptor.get("spec_digest") != current_spec_digest:
            raise BuildServiceError(
                "build submission spec changed before durable submission"
            )
        if current_key.components() != key.components():
            raise BuildServiceError(
                "build submission key inputs changed before durable submission"
            )
        if descriptor.get("key_components") != current_key.components():
            raise BuildServiceError(
                "build submission key inputs changed before durable submission"
            )
        if descriptor.get("cacheable") is True:
            if descriptor.get("cache_key") != current_key.digest:
                raise BuildServiceError(
                    "build submission cache key changed before durable submission"
                )
        elif descriptor.get("degraded_reason") != current_key.degraded_reason:
            raise BuildServiceError(
                "build submission degradation state changed before durable submission"
            )

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
    "decode_build_descriptor",
    "descriptor_input_digest",
    "dirty_snapshot_digest",
    "encode_build_descriptor",
]
