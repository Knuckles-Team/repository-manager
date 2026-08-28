"""A **content-addressed build broker** — DELIVERABLE 2 of CONCEPT:RM-TASK-LEDGER,
the second consumer of :mod:`repository_manager.task_queue` (the first is
:mod:`repository_manager.merge_queue`).

**The problem, measured.** Every lane that touches a compiled repository
builds it into its own ``target-isolated/`` (or ``node_modules/``) because a
shared build dir both serialises AND corrupts concurrent worktree builds (see
``task_queue``'s module docstring). The current design — one partition per
lane — is *safe* but maximally wasteful: N lanes asking for the SAME build
each pay for a full build. Measured: 4 ``target-isolated/`` dirs, 21.7 GB,
after pruning.

**The resolution.** Corruption comes from *concurrent writers*, not from
reuse. A build keyed by ``(repo, tree-sha, feature-set, toolchain-fingerprint,
target-triple)`` is deterministic enough that two requests for the *same* key
are asking the same question — the second should **wait on and reuse** the
first's answer rather than repeat the work. That is where the 21.7 GB and the
hours actually go; DEDUP, not serialization, is the value this module adds.
Serialization of the actual build step is still needed (two DIFFERENT builds
racing for the SAME lane's target dir corrupts exactly as before), and that
part is delegated to :mod:`task_queue`'s ``"build"`` :class:`ExecutionClass`
(``POOL``, capacity 4) — this module never touches a lease directly.

**The artifact is the product, not the target dir.** A pod needs
``numeric.abi3.so`` *in the mounted tree* — a compiled artifact cannot
propagate via a source mount, and ``*.so`` is gitignored deliberately (a
committed platform binary would ride into wheels for every OS and make
``check_wheel_completeness.py`` pass falsely). So a cache hit here publishes
NAMED ARTIFACTS to a content-addressed store; the target dir that produced
them is a throwaway implementation detail nobody outside this module touches.
Reuses, rather than reimplements, the epistemic-graph prior art:
``epistemic-graph/scripts/build_numeric_kernel.py`` /
``check_mounted_kernel.py`` are the reference for what "the produced artifact"
means for that repo's preset.

**Honest degradation, specifically.** When the cache key cannot be computed —
a dirty tree, an unknown/unfingerprintable toolchain — this module BUILDS,
never serves a possibly-stale artifact and never silently skips the cache as
if it were a hit. ``degraded`` is named in every result that took this path,
mirroring the merge queue's refusal-over-allow-all rule for an unproducible
baseline.

**Never runs in, or writes to, a canonical checkout.** A clean, cacheable
request materializes its declared commit into a throwaway detached worktree
via :func:`repository_manager.merge_queue.materialized` (the same primitive
the merge queue already trusts for this exact property) and builds there. A
dirty-tree (degraded) request builds directly in the CALLER's own tree — never
the canonical checkout, honored via ``lane_scope``.

**Arbitration is advisory (D-CP-8).** An agent that runs bare ``cargo build``
bypasses this broker entirely and is back to its own ``target-isolated/``.
This module does not, and cannot, prevent that — only make going through it
worth doing.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
import subprocess  # nosec B404 - fixed argv, never shell=True
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_utilities.governance.lanes import (
    LaneArbitrationError,
    LeaseUnavailable,
    hold_lease,
    lane_scope,
    partitioned_paths,
)

from repository_manager import task_queue as tq
from repository_manager.config_schema import (
    ArtifactContract,
    ConfigSchemaError,
    Placement,
    ResourceRequest,
    load_yaml_mapping_text,
    parse_build_config,
)
from repository_manager.disk_policy import DiskDecision, DiskPolicy, DiskWatermarks
from repository_manager.merge_queue import (
    _now,  # reuse the same UTC-isoformat timestamp helper merge_queue uses
    _require_git,
    _run_git,
    materialized,
)
from repository_manager.test_commands import ensure_no_fail_fast

CONFIG_FILENAME = ".buildcache.yaml"
ARTIFACT_STORE_DIRNAME = "build-cache"
EXECUTION_CLASS = "build"
# Resolved once so subprocess calls never hand a partial executable path to
# the OS (matches stash_guard.py/destructive_guard.py's convention).
_TRUSTED_GIT = shutil.which("git") or "git"
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_CONFIG_BYTES = 1 << 20
_MAX_LEGACY_ARTIFACTS = 4096
_MAX_LEGACY_ARTIFACT_BYTES = 64 * 1024 * 1024 * 1024
_MAX_LEGACY_SCAN_ENTRIES = 10_000
_MAX_LEGACY_SCAN_FILES = 100_000


class BuildQueueError(LaneArbitrationError):
    """A build-broker operation refused, carrying the reason a caller must act on."""


def _read_bounded_regular_file(path: Path, limit: int) -> bytes:
    """Read one bounded regular file without following a symlink swap."""

    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise BuildQueueError(f"file is not regular: {path}")
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read(limit + 1)
        if len(raw) > limit:
            raise BuildQueueError(f"file exceeds the durable bound: {path}")
        return raw
    except OSError as exc:
        raise BuildQueueError(f"could not read {path}: {exc}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _reject_symlink_path(path: Path) -> None:
    """Reject symlink components before a migration-only file operation."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise BuildQueueError(f"symlink component is not allowed: {current}")


def _copy_bounded_stream(
    source_handle: Any, destination_handle: Any, max_bytes: int | None
) -> int:
    """Stream ``source_handle`` into ``destination_handle`` 1MiB at a time,
    enforcing ``max_bytes``, then durably flush the destination.
    """
    copied = 0
    while True:
        chunk = source_handle.read(1 << 20)
        if not chunk:
            break
        copied += len(chunk)
        if max_bytes is not None and copied > max_bytes:
            raise BuildQueueError("legacy artifacts exceed the bounded migration size")
        destination_handle.write(chunk)
    destination_handle.flush()
    os.fsync(destination_handle.fileno())
    return copied


def _validate_legacy_source_stat(
    source: Path, source_stat: Any, max_bytes: int | None
) -> None:
    if not stat.S_ISREG(source_stat.st_mode):
        raise BuildQueueError(f"legacy artifact is not a regular file: {source}")
    if max_bytes is not None and source_stat.st_size > max_bytes:
        raise BuildQueueError("legacy artifacts exceed the bounded migration size")


def _open_legacy_destination_descriptor(destination: Path) -> int:
    _reject_symlink_path(destination.parent)
    destination.parent.mkdir(parents=True, exist_ok=True)
    return os.open(
        destination,
        os.O_WRONLY | os.O_CREAT | os.O_TRUNC | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )


def _verify_legacy_copy_unchanged(source: Path, source_stat: Any, copied: int) -> None:
    final_source = os.stat(source, follow_symlinks=False)
    if (
        final_source.st_dev != source_stat.st_dev
        or final_source.st_ino != source_stat.st_ino
        or final_source.st_size != copied
    ):
        raise BuildQueueError("legacy artifact changed while it was copied")


def _copy_legacy_file_no_follow(
    source: Path,
    destination: Path,
    *,
    max_bytes: int | None = None,
) -> int:
    """Copy a compatibility artifact through bounded no-follow descriptors."""

    _reject_symlink_path(source)
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        source_stat = os.fstat(source_descriptor)
        _validate_legacy_source_stat(source, source_stat, max_bytes)
        destination_descriptor = _open_legacy_destination_descriptor(destination)
        with os.fdopen(source_descriptor, "rb") as source_handle:
            source_descriptor = None
            with os.fdopen(destination_descriptor, "wb") as destination_handle:
                destination_descriptor = None
                copied = _copy_bounded_stream(
                    source_handle, destination_handle, max_bytes
                )
        _verify_legacy_copy_unchanged(source, source_stat, copied)
        return copied
    except OSError as exc:
        raise BuildQueueError(f"could not copy legacy artifact {source}") from exc
    finally:
        for descriptor in (source_descriptor, destination_descriptor):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


@dataclass
class _LegacyTreeScanState:
    pending: list[Path]
    entries: int = 0
    files: int = 0
    total: int = 0


def _account_legacy_tree_entry(child: Any, state: _LegacyTreeScanState) -> None:
    state.entries += 1
    if state.entries > _MAX_LEGACY_SCAN_ENTRIES:
        raise BuildQueueError("legacy artifact scan exceeds its entry bound")
    if child.is_symlink():
        raise BuildQueueError("legacy artifact scan found a symlink")
    if child.is_dir(follow_symlinks=False):
        state.pending.append(Path(child.path))
        return
    if not child.is_file(follow_symlinks=False):
        raise BuildQueueError("legacy artifact scan found a non-regular entry")
    state.files += 1
    if state.files > _MAX_LEGACY_SCAN_FILES:
        raise BuildQueueError("legacy artifact scan exceeds its file bound")
    state.total += child.stat(follow_symlinks=False).st_size
    if state.total > _MAX_LEGACY_ARTIFACT_BYTES:
        raise BuildQueueError("legacy artifact scan exceeds its byte bound")


def _scan_legacy_tree_dir(current: Path, state: _LegacyTreeScanState) -> None:
    try:
        iterator = os.scandir(current)
    except OSError as exc:
        raise BuildQueueError(
            "legacy artifact scan could not read a directory"
        ) from exc
    with iterator:
        for child in iterator:
            _account_legacy_tree_entry(child, state)


def _bounded_legacy_tree_size(path: Path) -> int:
    """Bounded no-follow byte accounting for compatibility GC."""

    _reject_symlink_path(path)
    if path.is_symlink() or not path.is_dir():
        raise BuildQueueError("legacy artifact entry is not a directory")
    state = _LegacyTreeScanState(pending=[path])
    while state.pending:
        current = state.pending.pop()
        _scan_legacy_tree_dir(current, state)
    return state.total


# ---------------------------------------------------------------------------
# Declaration — a repository's own .buildcache.yaml, mirroring .mergequeue.yaml
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class BuildSpec:
    """One declared build (a repo can declare more than one, e.g. debug/release)."""

    name: str
    #: argv, never a shell string — same injection-surface rule as GateSpec.
    command: tuple[str, ...]
    #: Where the command runs, relative to the materialized/working tree root.
    workdir: str = "."
    #: Command whose stdout fingerprints the toolchain. Absent = unfingerprintable
    #: => every request for this spec is DEGRADED (never cached).
    toolchain_fingerprint: tuple[str, ...] = ()
    #: Paths (relative to the tree root) that participate in the cache key.
    #: Empty = the WHOLE tree (safe default; a repo narrows this deliberately).
    cache_key_paths: tuple[str, ...] = ()
    #: Glob patterns (relative to workdir) naming the files THIS build must
    #: produce. Every pattern must match >=1 file or the build is a FAILURE —
    #: a build that silently produces nothing is worse than one that errors.
    artifacts: tuple[str, ...] = ()
    timeout: int = 3600
    #: Overrides ``platform.machine()-sys.platform`` when the toolchain cross-
    #: compiles (e.g. a maturin abi3 wheel target).
    target_triple: str = ""
    source: str = ""
    #: Scheduler-facing resource and placement declaration.  The build broker
    #: does not admit these yet; later scheduler lanes consume the typed value.
    resource_class: str = "light-check"
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    disk_estimate_mb: int = 0
    placement: Placement = field(default_factory=Placement)
    artifact_contract: ArtifactContract = field(default_factory=ArtifactContract)
    #: Validation stage and generation compatibility are metadata-only until
    #: the staged validation/build lanes consume them.
    stage: str = "feedback"
    generation_compatible: bool = True


@dataclass(frozen=True)
class BuildConfig:
    base: str = "main"
    specs: tuple[BuildSpec, ...] = ()
    source: str = ""
    schema_version: int = 2

    def spec(self, name: str = "") -> BuildSpec:
        if not self.specs:
            raise BuildQueueError(
                f"{self.source or CONFIG_FILENAME} declares no build specs"
            )
        if not name:
            return self.specs[0]
        for candidate in self.specs:
            if candidate.name == name:
                return candidate
        raise BuildQueueError(
            f"{self.source or CONFIG_FILENAME} has no spec named {name!r}; "
            f"declared: {[s.name for s in self.specs]}"
        )


def _as_argv(value: Any, *, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise BuildQueueError(
            f"{where}: a command must be a LIST of argv items, not a string "
            f"({value!r}) — never run through a shell."
        )
    if not isinstance(value, list | tuple) or not value:
        raise BuildQueueError(f"{where}: a command must be a non-empty list of strings")
    return tuple(str(v) for v in value)


def parse_config(data: dict[str, Any], *, source: str = "") -> BuildConfig:
    try:
        schema = parse_build_config(data, source=source)
    except ConfigSchemaError as exc:
        raise BuildQueueError(str(exc)) from exc
    return BuildConfig(
        base=schema.base,
        specs=tuple(
            BuildSpec(
                name=spec.name,
                command=spec.command,
                workdir=spec.workdir,
                toolchain_fingerprint=spec.toolchain_fingerprint,
                cache_key_paths=spec.cache_key_paths,
                artifacts=spec.artifacts,
                timeout=spec.timeout,
                target_triple=spec.target_triple,
                resource_class=spec.resource_class,
                resources=spec.resources,
                disk_estimate_mb=spec.disk_estimate_mb,
                placement=spec.placement,
                artifact_contract=spec.artifact_contract,
                stage=spec.stage,
                generation_compatible=spec.generation_compatible,
                source=source,
            )
            for spec in schema.specs
        ),
        schema_version=schema.schema_version,
        source=source,
    )


def load_config(tree: Path | str) -> BuildConfig:
    tree = Path(tree)
    config_path = tree / CONFIG_FILENAME
    if not config_path.exists():
        raise BuildQueueError(
            f"{tree} has no {CONFIG_FILENAME} — a repository with no build "
            "declaration is refused rather than defaulted (an absent config "
            "and 'this repo builds nothing' must not be the same value)."
        )
    try:
        raw = _read_bounded_regular_file(config_path, _MAX_CONFIG_BYTES)
        data = load_yaml_mapping_text(raw.decode("utf-8"), source=str(config_path))
        return parse_config(data, source=str(config_path))
    except (ConfigSchemaError, UnicodeDecodeError, BuildQueueError) as exc:
        raise BuildQueueError(str(exc)) from exc


# ---------------------------------------------------------------------------
# Cache key — behaviour: honest degradation when any component is unknowable
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CacheKey:
    """Either a fully computable key, or a named reason it could not be."""

    repo: str
    spec: str
    tree_sha: str = ""
    feature_set: str = ""
    toolchain_fingerprint: str = ""
    target_triple: str = ""
    # RMDD-10 extends the historical key without invalidating old manifests.
    # ``generation_id`` is retained as an opaque correlation while the digest
    # is what participates in the content address.
    config_digest: str = ""
    spec_digest: str = ""
    generation_id: str = ""
    generation_digest: str = ""
    key_version: str = "v2"
    degraded_reason: str = ""

    @property
    def computable(self) -> bool:
        return not self.degraded_reason

    @property
    def digest(self) -> str:
        if not self.computable:
            raise BuildQueueError("a degraded CacheKey has no digest — never cache it")
        payload = json.dumps(
            {
                "key_version": self.key_version,
                "repo": self.repo,
                "spec": self.spec,
                "tree_sha": self.tree_sha,
                "feature_set": self.feature_set,
                "toolchain_fingerprint": self.toolchain_fingerprint,
                "target_triple": self.target_triple,
                "config_digest": self.config_digest,
                "spec_digest": self.spec_digest,
                "generation_digest": self.generation_digest,
            },
            sort_keys=True,
        )
        return f"{self.key_version}:{hashlib.sha256(payload.encode()).hexdigest()[:32]}"

    @property
    def legacy_digest(self) -> str:
        """Return the pre-RMDD-10 key used by existing cache manifests."""

        if not self.computable:
            raise BuildQueueError("a degraded CacheKey has no legacy digest")
        payload = json.dumps(
            {
                "repo": self.repo,
                "spec": self.spec,
                "tree_sha": self.tree_sha,
                "feature_set": self.feature_set,
                "toolchain_fingerprint": self.toolchain_fingerprint,
                "target_triple": self.target_triple,
            },
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def components(self) -> dict[str, str]:
        return {
            "key_version": self.key_version,
            "repo": self.repo,
            "spec": self.spec,
            "tree_sha": self.tree_sha,
            "feature_set": self.feature_set,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "target_triple": self.target_triple,
            "config_digest": self.config_digest,
            "spec_digest": self.spec_digest,
            "generation_id": self.generation_id,
            "generation_digest": self.generation_digest,
        }

    def legacy_components(self) -> dict[str, str]:
        """Project only the v1 fields for migration/explain compatibility."""

        return {
            "repo": self.repo,
            "spec": self.spec,
            "tree_sha": self.tree_sha,
            "feature_set": self.feature_set,
            "toolchain_fingerprint": self.toolchain_fingerprint,
            "target_triple": self.target_triple,
        }


def _tree_is_dirty(tree: Path) -> bool:
    res = _run_git(["status", "--porcelain"], tree)
    return not res.ok or bool(res.out.strip())


def _paths_tree_sha(tree: Path, paths: tuple[str, ...]) -> str:
    """A hash of ``git ls-tree`` for the declared cache-key paths (or the whole
    tree at HEAD when none are declared) — narrower than a full tree-sha when a
    repo says so, so an unrelated doc change does not bust every build's cache.
    """
    head = _require_git(["rev-parse", "HEAD"], tree)
    if not paths:
        return _require_git(["rev-parse", f"{head}^{{tree}}"], tree)
    listing = _require_git(["ls-tree", "-r", head, "--", *paths], tree)
    return hashlib.sha256(listing.encode()).hexdigest()[:32]


def _toolchain_fingerprint(tree: Path, spec: BuildSpec) -> str | None:
    if not spec.toolchain_fingerprint:
        return None
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        list(spec.toolchain_fingerprint),
        cwd=str(tree),
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout.encode()).hexdigest()[:32]


def _target_triple(spec: BuildSpec) -> str:
    if spec.target_triple:
        return spec.target_triple
    import platform

    return f"{platform.system()}-{platform.machine()}".lower()


def _resolve_git_identity_root(resolved: Path) -> Path:
    """The repo's git-common-dir parent (stable across worktree lanes) when
    ``resolved`` is inside a git checkout, else ``resolved`` itself.
    """
    try:
        top = subprocess.run(
            [_TRUSTED_GIT, "rev-parse", "--show-toplevel"],
            cwd=str(resolved),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        common = subprocess.run(
            [_TRUSTED_GIT, "rev-parse", "--path-format=absolute", "--git-common-dir"],
            cwd=str(resolved),
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
        if top.returncode == 0 and common.returncode == 0:
            common_dir = Path(common.stdout.strip()).resolve(strict=False)
            return (
                common_dir.parent
                if common_dir.name == ".git"
                else Path(top.stdout.strip()).resolve(strict=False)
            )
    except (OSError, subprocess.SubprocessError):
        pass
    return resolved


def _find_workspace_root(identity_root: Path) -> Path | None:
    configured = os.environ.get("AGENT_UTILITIES_WORKSPACE_ROOT")
    candidates = [Path(configured).expanduser().resolve()] if configured else []
    candidates.extend(
        parent
        for parent in (identity_root, *identity_root.parents)
        if (parent / "workspace.yml").is_file()
    )
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "workspace.yml").is_file():
            return candidate
    return None


def stable_repository_id(path: Path | str) -> str:
    """Return the C-01 repository identity, never a display basename.

    Workspace-relative paths are readable and stable across worktree lanes;
    repositories outside the configured workspace use a deterministic path
    digest.  Either form keeps two repositories with the same basename
    distinct without introducing a second identity registry.
    """

    resolved = Path(path).expanduser().resolve(strict=False)
    identity_root = _resolve_git_identity_root(resolved)
    workspace_root = _find_workspace_root(identity_root)
    if workspace_root is not None:
        try:
            relative = identity_root.relative_to(workspace_root)
        except ValueError:
            relative = None
        if relative is not None:
            return f"repository:{relative.as_posix()}"
    digest = hashlib.sha256(str(identity_root).encode("utf-8")).hexdigest()[:24]
    return f"repository:path:{digest}"


def _spec_digest(spec: BuildSpec) -> str:
    """Hash the full runtime spec, excluding its machine-specific source path."""

    payload = {
        "name": spec.name,
        "command": list(spec.command),
        "workdir": spec.workdir,
        "toolchain_fingerprint": list(spec.toolchain_fingerprint),
        "cache_key_paths": list(spec.cache_key_paths),
        "artifacts": list(spec.artifacts),
        "timeout": spec.timeout,
        "target_triple": spec.target_triple,
        "resource_class": spec.resource_class,
        "resources": {
            "cpu_weight": spec.resources.cpu_weight,
            "memory_mb": spec.resources.memory_mb,
            "disk_mb": spec.resources.disk_mb,
            "process_slots": spec.resources.process_slots,
        },
        "disk_estimate_mb": spec.disk_estimate_mb,
        "placement": {
            "required_labels": list(spec.placement.required_labels),
            "preferred_host": spec.placement.preferred_host,
            "required_host": spec.placement.required_host,
            "anti_affinity": list(spec.placement.anti_affinity),
        },
        "artifact_contract": {
            "patterns": list(spec.artifact_contract.patterns),
            "required": spec.artifact_contract.required,
            "publish": spec.artifact_contract.publish,
            "retention": spec.artifact_contract.retention,
        },
        "stage": spec.stage,
        "generation_compatible": spec.generation_compatible,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _config_digest(repo: Path) -> str:
    config_path = repo / CONFIG_FILENAME
    try:
        data = _read_bounded_regular_file(config_path, _MAX_CONFIG_BYTES)
    except BuildQueueError as exc:
        raise BuildQueueError(f"could not read {config_path}: {exc}") from exc
    return hashlib.sha256(data).hexdigest()


def _generation_digest(generation_id: str | None) -> str:
    if not generation_id:
        return ""
    return hashlib.sha256(generation_id.encode("utf-8")).hexdigest()


def compute_cache_key(
    repo: Path,
    spec: BuildSpec,
    *,
    repo_name: str,
    generation_id: str | None = None,
    config_digest: str | None = None,
) -> CacheKey:
    """Compute (or explain why it cannot compute) this request's content-address.

    Every failure mode names itself in ``degraded_reason`` rather than
    returning a key built from a guess — the caller (:func:`request`) reads
    that field to decide whether the cache may be trusted at all.
    """
    if _tree_is_dirty(repo):
        return CacheKey(repo=repo_name, spec=spec.name, degraded_reason="dirty-tree")
    try:
        tree_sha = _paths_tree_sha(repo, spec.cache_key_paths)
    except LaneArbitrationError:
        return CacheKey(
            repo=repo_name, spec=spec.name, degraded_reason="tree-sha-unresolvable"
        )
    fingerprint = _toolchain_fingerprint(repo, spec)
    if spec.toolchain_fingerprint and fingerprint is None:
        return CacheKey(
            repo=repo_name,
            spec=spec.name,
            degraded_reason="toolchain-unfingerprintable",
        )
    return CacheKey(
        repo=repo_name,
        spec=spec.name,
        tree_sha=tree_sha,
        feature_set=" ".join(spec.command),
        toolchain_fingerprint=fingerprint or "unpinned",
        target_triple=_target_triple(spec),
        config_digest=config_digest or _config_digest(repo),
        spec_digest=_spec_digest(spec),
        generation_id=generation_id or "",
        generation_digest=_generation_digest(generation_id),
    )


# ---------------------------------------------------------------------------
# Artifact store — content-addressed, keyed by CacheKey.digest
# ---------------------------------------------------------------------------
def _artifact_root(path: Path | str | None = None) -> Path:
    scope = lane_scope(path)
    root = scope.arbitration_dir / ARTIFACT_STORE_DIRNAME
    root.mkdir(parents=True, exist_ok=True)
    return root


def _manifest_path(key_digest: str, path: Path | str | None = None) -> Path:
    return _artifact_root(path) / key_digest / "manifest.json"


def _read_manifest(
    key_digest: str, path: Path | str | None = None
) -> dict[str, Any] | None:
    manifest_path = _manifest_path(key_digest, path)
    try:
        raw = _read_bounded_regular_file(manifest_path, _MAX_MANIFEST_BYTES)
        value = json.loads(raw.decode("utf-8"))
    except (BuildQueueError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _artifact_path_is_safe(root: Path, candidate: Path) -> bool:
    """Require a stored artifact below root with no symlink components."""

    try:
        relative = candidate.relative_to(root)
    except ValueError:
        return False
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            return False
    return True


def _manifest_schema_state_ok(manifest: dict[str, Any], expected_key: Any) -> bool:
    """RMDD-10 manifests are published before terminal WorkItem commit.  A
    restart must not serve that intermediate state as a cache hit.  Legacy
    manifests have no schema/state marker and remain readable during the
    explicit migration window.
    """
    if manifest.get("schema") == "build-artifact:v2" and (
        manifest.get("publication_state") != "committed"
        or not isinstance(expected_key, str)
        or manifest.get("key") != expected_key
    ):
        return False
    return True


def _resolve_expected_artifact_root(
    path: Path | str | None, expected_key: Any
) -> tuple[bool, Path | None]:
    """``(True, root)`` on success, ``(True, None)`` when no path-scoped check
    applies, ``(False, None)`` when the manifest must be rejected outright.
    """
    if path is None or not isinstance(expected_key, str) or not expected_key:
        return True, None
    try:
        key_dir = _artifact_root(path) / expected_key
        artifacts_dir = key_dir / "artifacts"
        if key_dir.is_symlink() or artifacts_dir.is_symlink():
            return False, None
        return True, artifacts_dir.resolve(strict=True)
    except (OSError, ValueError):
        return False, None


@dataclass
class _ArtifactEntryInfo:
    path: Path
    checksum: str
    declared_bytes: int


def _artifact_entry_path(entry: Any, expected_root: Path | None) -> Path | None:
    """Resolve and confine ``entry["stored_at"]``; ``None`` means reject."""
    try:
        artifact_path = Path(entry["stored_at"])
    except (KeyError, TypeError):
        return None
    if expected_root is not None:
        try:
            if not _artifact_path_is_safe(expected_root, artifact_path):
                return None
            artifact_path.resolve(strict=True).relative_to(expected_root)
        except (OSError, ValueError):
            return None
    return artifact_path


def _artifact_entry_stat_info(
    entry: Any, artifact_path: Path
) -> tuple[str, int] | None:
    """``(checksum, declared_bytes)`` on a well-formed regular-file entry,
    else ``None`` — reject.
    """
    try:
        artifact_stat = os.stat(artifact_path, follow_symlinks=False)
        if not stat.S_ISREG(artifact_stat.st_mode):
            return None
        checksum = entry["sha256"]
        if not isinstance(checksum, str) or len(checksum) > 256:
            return None
        declared_bytes = int(entry.get("bytes", artifact_stat.st_size))
        if (
            isinstance(entry.get("bytes"), bool)
            or declared_bytes < 0
            or declared_bytes > _MAX_LEGACY_ARTIFACT_BYTES
            or artifact_stat.st_size != declared_bytes
        ):
            return None
    except (KeyError, OSError, TypeError, ValueError):
        return None
    return checksum, declared_bytes


def _artifact_entry_shape(
    entry: Any, expected_root: Path | None
) -> _ArtifactEntryInfo | None:
    """Structural/stat validation of one manifest artifact entry.  Returns
    ``None`` for every condition the original inline loop treated as an
    immediate ``return False``, short of the running byte-budget and checksum
    checks (which need cross-entry state and stay in the caller).
    """
    if not isinstance(entry, dict):
        return None
    artifact_path = _artifact_entry_path(entry, expected_root)
    if artifact_path is None:
        return None
    stat_info = _artifact_entry_stat_info(entry, artifact_path)
    if stat_info is None:
        return None
    checksum, declared_bytes = stat_info
    return _ArtifactEntryInfo(
        path=artifact_path, checksum=checksum, declared_bytes=declared_bytes
    )


def _manifest_artifacts_are_valid(
    artifacts: list[Any], expected_root: Path | None
) -> bool:
    total_bytes = 0
    for entry in artifacts:
        info = _artifact_entry_shape(entry, expected_root)
        if info is None:
            return False
        total_bytes += info.declared_bytes
        if total_bytes > _MAX_LEGACY_ARTIFACT_BYTES:
            return False
        try:
            if _sha256_file(info.path, max_bytes=info.declared_bytes) != info.checksum:
                return False
        except (BuildQueueError, OSError):
            return False
    return True


def _manifest_is_valid(
    manifest: dict[str, Any],
    path: Path | str | None = None,
    key_digest: str | None = None,
) -> bool:
    """A manifest is only a hit if every artifact it lists is STILL on disk with
    the recorded checksum — gc or an out-of-band deletion must degrade to a
    miss, never to serving a dangling reference.
    """
    expected_key = key_digest or manifest.get("key")
    if not _manifest_schema_state_ok(manifest, expected_key):
        return False
    root_ok, expected_root = _resolve_expected_artifact_root(path, expected_key)
    if not root_ok:
        return False
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts or len(artifacts) > 4096:
        return False
    return _manifest_artifacts_are_valid(artifacts, expected_root)


def _migrate_legacy_manifest(
    key: CacheKey, manifest: dict[str, Any], path: Path | str | None = None
) -> dict[str, Any]:
    """Return a read-only v2 compatibility projection of a v1 manifest.

    The projection deliberately is *not* written under the v2 key: a v2 alias
    pointing at v1 bytes would become a dangling cache hit when legacy GC runs.
    New durable publications own their bytes under the immutable v2 directory;
    old bytes remain readable only through this explicit migration window.
    """

    migrated = dict(manifest)
    migrated.update(
        {
            "key": key.digest,
            "schema": "build-artifact:v2",
            "publication_state": "committed",
            "migration": "legacy-v1-key-compatible",
            "migrated_from": key.legacy_digest,
            "components": key.components(),
        }
    )
    return migrated


def _cache_manifest(
    key: CacheKey, path: Path | str | None = None
) -> tuple[dict[str, Any] | None, str | None]:
    """Read a v2 hit, or return a non-owning v1 compatibility projection."""

    current = _read_manifest(key.digest, path)
    if current is not None and _manifest_is_valid(current, path, key.digest):
        return current, None
    # Read paths never mutate the cache.  In particular, a v2 ``published``
    # manifest is the live producer's crash-recovery evidence until the
    # durable WorkItem commits it; only the authority-aware artifact store may
    # quarantine it after proving the exact owner/fence is stale.
    legacy = _read_manifest(key.legacy_digest, path)
    if legacy is not None and _manifest_is_valid(legacy, path, key.legacy_digest):
        return _migrate_legacy_manifest(key, legacy, path), key.legacy_digest
    return None, None


def _hash_bounded_stream(handle: Any, digest: Any, max_bytes: int | None) -> int:
    """Feed ``handle`` into ``digest`` 1MiB at a time; raise if the bound is
    exceeded mid-stream.  Returns the number of bytes actually copied.
    """
    copied = 0
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        copied += len(chunk)
        if max_bytes is not None and copied > max_bytes:
            raise BuildQueueError("artifact exceeds its bounded byte limit")
        digest.update(chunk)
    return copied


def _checksum_open_file(
    handle: Any, digest: Any, path: Path, max_bytes: int | None
) -> None:
    """Validate + hash an already-open file handle in place (mutates ``digest``)."""
    initial = os.fstat(handle.fileno())
    if not stat.S_ISREG(initial.st_mode):
        raise BuildQueueError(f"artifact is not a regular file: {path}")
    if max_bytes is not None and initial.st_size > max_bytes:
        raise BuildQueueError("artifact exceeds its bounded byte limit")
    copied = _hash_bounded_stream(handle, digest, max_bytes)
    final = os.fstat(handle.fileno())
    if (
        final.st_dev != initial.st_dev
        or final.st_ino != initial.st_ino
        or final.st_size != copied
    ):
        raise BuildQueueError("artifact changed while it was checksummed")


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            _checksum_open_file(handle, digest, path, max_bytes)
    except OSError as exc:
        raise BuildQueueError(f"could not checksum artifact {path}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return digest.hexdigest()


def _publish_artifacts(
    build_tree: Path, spec: BuildSpec, key_digest: str, path: Path | str | None = None
) -> list[dict[str, Any]]:
    raw_workdir = build_tree / spec.workdir
    _reject_symlink_path(raw_workdir)
    workdir = raw_workdir.resolve(strict=True)
    tree = Path(build_tree).resolve(strict=True)
    try:
        workdir.relative_to(tree)
    except ValueError as exc:
        raise BuildQueueError("legacy artifact workdir escapes the build tree") from exc
    dest_dir = _artifact_root(path) / key_digest / "artifacts"
    _reject_symlink_path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    published: list[dict[str, Any]] = []
    total_bytes = 0
    try:
        from repository_manager.build_artifacts import (
            ArtifactStoreError,
            _bounded_matching_files,
        )

        matches_by_pattern = _bounded_matching_files(
            workdir,
            tuple(spec.artifacts),
            max_entries=_MAX_LEGACY_SCAN_ENTRIES,
            max_files=_MAX_LEGACY_ARTIFACTS,
            max_bytes=_MAX_LEGACY_ARTIFACT_BYTES,
        )
    except ArtifactStoreError as exc:
        raise BuildQueueError(str(exc)) from exc
    for pattern in spec.artifacts:
        matches = matches_by_pattern[pattern]
        if not matches:
            raise BuildQueueError(
                f"build spec {spec.name!r} declared artifact pattern {pattern!r} "
                "but the build produced no file matching it — a build that "
                "silently produces nothing is a failure, not a success."
            )
        for src in matches:
            relative = src.relative_to(workdir)
            dest = dest_dir / relative
            copied = _copy_legacy_file_no_follow(
                src,
                dest,
                max_bytes=_MAX_LEGACY_ARTIFACT_BYTES - total_bytes,
            )
            total_bytes += copied
            if total_bytes > _MAX_LEGACY_ARTIFACT_BYTES:
                raise BuildQueueError("legacy artifact bytes exceed their bound")
            published.append(
                {
                    "pattern": pattern,
                    "relative_path": str(relative),
                    "stored_at": str(dest),
                    "sha256": _sha256_file(dest),
                    "bytes": copied,
                }
            )
            if len(published) > _MAX_LEGACY_ARTIFACTS:
                raise BuildQueueError("legacy artifact count exceeds its bound")
    return published


# ---------------------------------------------------------------------------
# request() — the one entrypoint: dedup-or-build
# ---------------------------------------------------------------------------
@dataclass
class _ServiceIdentity:
    tenant_id: str
    owner_id: str
    session_id: str
    owner: dict[str, Any] | None


def _request_via_service(
    build_service: Any | None,
    job_service: Any | None,
    repo_path: Path | str | None,
    spec_name: str,
    generation_id: str | None,
    identity: _ServiceIdentity,
) -> dict[str, Any]:
    from repository_manager.build_service import BuildService

    service = build_service
    if service is None:
        assert job_service is not None, "job_service or build_service must be supplied"
        service = BuildService(
            job_service,
            tenant_id=identity.tenant_id or "repository-manager",
            owner_id=identity.owner_id
            or (
                identity.owner.get("owner_id")
                if isinstance(identity.owner, dict)
                else ""
            )
            or "repository-manager",
            session_id=identity.session_id
            or (
                identity.owner.get("session")
                if isinstance(identity.owner, dict)
                else ""
            )
            or "build-request",
        )
    return service.submit(
        repo_path=repo_path,
        spec_name=spec_name,
        generation_id=generation_id,
    )


def _cache_hit_response(
    manifest: dict[str, Any],
    key: CacheKey,
    digest: str,
    *,
    waited: bool = False,
    migrated_from: str | None = None,
) -> dict[str, Any]:
    response: dict[str, Any] = {
        "ok": True,
        "cached": True,
        "degraded": False,
        "key": digest,
        "components": key.components(),
        "artifacts": manifest["artifacts"],
        "built_at": manifest.get("built_at"),
    }
    if waited:
        response["waited"] = True
    if migrated_from:
        response["migrated_from"] = migrated_from
    return response


def _wait_and_reread_manifest(
    digest: str, scope: Any, wait_timeout: int
) -> dict[str, Any] | None:
    """Wait for the in-flight build; return its manifest iff it produced a
    valid one before the wait bound expired.
    """
    waited = _wait_for_task(digest, scope.tree, timeout=wait_timeout)
    manifest = _read_manifest(digest, scope.tree) if waited else None
    if manifest and _manifest_is_valid(manifest, scope.tree, digest):
        return manifest
    return None


def _await_build_or_claim(
    digest: str, spec: BuildSpec, key: CacheKey, scope: Any, wait_timeout: int
) -> tuple[_BuildClaim, dict[str, Any] | None]:
    """D-CDX-13 bounded two-cycle wait-then-reclaim.

    Returns ``(claim, None)`` when THIS caller now owns the RUNNING task and
    must build, or ``(claim, cached_response)`` when a valid manifest showed
    up while waiting on someone else's in-flight build. Raises
    :class:`BuildQueueError` if neither happens within two wait cycles.
    """
    claim = _claim_build_task(digest, spec, key, path=scope.tree)
    if claim.is_new:
        return claim, None
    manifest = _wait_and_reread_manifest(digest, scope, wait_timeout)
    if manifest is not None:
        return claim, _cache_hit_response(manifest, key, digest, waited=True)
    # The in-flight build finished without producing a valid manifest (it
    # failed) or never finished within wait_timeout. Try to claim it
    # ourselves now (atomically, same as above) rather than getting stuck OR
    # blindly re-building without re-checking for a fresher claim.
    claim = _claim_build_task(digest, spec, key, path=scope.tree)
    if claim.is_new:
        return claim, None
    # Someone else claimed it again in the interim — one more wait, then
    # give up cleanly rather than looping forever.
    manifest = _wait_and_reread_manifest(digest, scope, wait_timeout)
    if manifest is not None:
        return claim, _cache_hit_response(manifest, key, digest, waited=True)
    raise BuildQueueError(
        f"build {digest!r} did not complete after two wait cycles "
        f"({wait_timeout}s each) and this caller could not claim it"
    )


def request(
    *,
    repo_path: Path | str | None = None,
    spec_name: str = "",
    colocated: bool = False,
    owner: dict[str, Any] | None = None,
    wait_timeout: int = 60,
    generation_id: str | None = None,
    job_service: Any | None = None,
    build_service: Any | None = None,
    tenant_id: str = "",
    owner_id: str = "",
    session_id: str = "",
) -> dict[str, Any]:
    """Submit a durable build, or use the legacy synchronous compatibility path.

    When ``job_service``/``build_service`` is supplied, a cache miss returns
    immediately with the existing durable RepositoryJobService WorkItem ID.
    The worker owns resource admission, materialization, execution, and
    publication.  The no-service path remains intentionally synchronous for
    pre-RMDD-20 CLI/MCP compatibility; it is not production authority and is
    kept only so old callers can migrate without invalidating existing cache
    keys or artifacts.

    Two requests for the SAME computable key: the second waits (bounded by
    ``wait_timeout``) on the first's in-flight :class:`~task_queue.Task` and
    then reuses its artifacts rather than rebuilding — this is where the
    dedup value comes from. Two requests for DIFFERENT keys run in parallel up
    to the ``"build"`` execution class's pool cap; identical keys serialise
    through the Task check below, not through the pool.
    """
    if build_service is not None or job_service is not None:
        return _request_via_service(
            build_service,
            job_service,
            repo_path,
            spec_name,
            generation_id,
            _ServiceIdentity(
                tenant_id=tenant_id,
                owner_id=owner_id,
                session_id=session_id,
                owner=owner,
            ),
        )

    scope = lane_scope(repo_path)
    config = load_config(scope.tree)
    spec = config.spec(spec_name)
    repo_name = stable_repository_id(scope.main_tree)
    key = compute_cache_key(
        scope.tree,
        spec,
        repo_name=repo_name,
        generation_id=generation_id,
    )

    if not key.computable:
        result = _build(
            scope.tree, spec, key, cache_digest=None, colocated=colocated, owner=owner
        )
        result["degraded"] = True
        result["degraded_reason"] = key.degraded_reason
        result["cached"] = False
        return result

    digest = key.digest
    manifest, migrated_from = _cache_manifest(key, scope.tree)
    if manifest and (
        migrated_from is not None or _manifest_is_valid(manifest, scope.tree, digest)
    ):
        return _cache_hit_response(manifest, key, digest, migrated_from=migrated_from)

    # D-CDX-13 — the check ("is a build for this key already running?") and
    # the enqueue ("mark one running") used to be two separate, unlocked
    # steps: read `find_task` here, then (only inside `_build()`, further
    # down the call) `enqueue_task` + `record_state(RUNNING)`. Two same-key
    # callers could both observe "nothing running" before either one
    # appended its RUNNING record, both fall through, and both build and
    # publish — the exact duplicate-build/disk-waste event this cache exists
    # to prevent, with no test proving the barrier held under concurrency.
    # `_claim_build_task` makes the check-then-claim ATOMIC under a short,
    # key-scoped exclusive lease — held only for that instant, never for the
    # build's duration, so it never contends with the separate POOL-capacity
    # "build" execution class that still serializes the actual heavy compute
    # step below.
    claim, cached = _await_build_or_claim(digest, spec, key, scope, wait_timeout)
    if cached is not None:
        return cached

    result = _build(
        scope.tree,
        spec,
        key,
        cache_digest=digest,
        colocated=colocated,
        owner=owner,
        task=claim.task,
    )
    result["degraded"] = False
    return result


@dataclass(frozen=True)
class _BuildClaim:
    """Outcome of :func:`_claim_build_task`."""

    #: True iff THIS caller now owns the RUNNING task record and must build.
    is_new: bool
    task: tq.Task | None


def _claim_build_task(
    digest: str, spec: BuildSpec, key: CacheKey, *, path: Path
) -> _BuildClaim:
    """Atomically decide who builds key *digest* — the fix for D-CDX-13.

    Holds a short-lived, KEY-SCOPED exclusive lease (``build-claim-<digest>``,
    distinct from the POOL-capacity ``"build"`` execution class) around the
    read-then-append transition: re-check for an already-RUNNING task for
    this exact digest, and if none exists, enqueue + mark RUNNING before
    releasing the lease — so a second caller for the SAME key can only ever
    observe one of "already running" (never appends its own) or "I'm the
    claimant" (appended exactly once), never the gap between them.

    Honesty note (the hard constraint on this fix): the lease is an
    ``fcntl.flock`` in this workspace's shared arbitration directory. It
    arbitrates callers ON THIS HOST that go through :func:`request` — same
    scope as every other lease in this module and in :mod:`task_queue`. It
    gives no guarantee against a build-broker request issued from a
    different host, or any process that appends a ``"build"`` task record
    directly without going through this function.
    """
    lease_name = f"build-claim-{digest}"
    try:
        with hold_lease(
            lease_name,
            operation=f"claim build {key.repo}:{spec.name}",
            ttl_seconds=30,
            path=path,
        ):
            existing = tq.find_task("build", digest, path=path)
            if existing and existing.state == tq.RUNNING:
                return _BuildClaim(is_new=False, task=None)
            task = tq.enqueue_task(
                digest,
                "build",
                repo=key.repo,
                payload={"spec": spec.name, "components": key.components()},
                path=path,
            )
            task = tq.record_state(task, tq.RUNNING, "", path=path)
            return _BuildClaim(is_new=True, task=task)
    except LeaseUnavailable:
        # Another request is mid-claim for this EXACT key right now. It will
        # have appended its RUNNING record within a few filesystem ops (not
        # the build itself), so wait to actually OBSERVE it rather than
        # assuming it is already there: `_wait_for_task` treats "no task
        # record found" as "done waiting" (a legitimate fast-exit for a task
        # that genuinely finished), so returning immediately here — before
        # the winner has actually written anything — made the very next
        # `find_task` see nothing, exit the wait instantly, and give up
        # without ever observing the real build.
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline:
            existing = tq.find_task("build", digest, path=path)
            if existing is not None:
                return _BuildClaim(is_new=False, task=None)
            time.sleep(0.05)
        # The winner should have written its record in well under 2s (a
        # lease + two FragmentStore appends); if it somehow has not, treat
        # this the same as any other "nothing running" observation and let
        # the caller's own claim attempt decide.
        return _BuildClaim(is_new=False, task=None)


def _wait_for_task(task_id: str, path: Path | str, *, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = tq.find_task("build", task_id, path=path)
        if task is None or task.state != tq.RUNNING:
            return True
        time.sleep(2)
    return False


def _ensure_build_task(
    task: tq.Task | None,
    spec: BuildSpec,
    cache_digest: str | None,
    key: CacheKey,
    tree: Path,
) -> tq.Task:
    """Return ``task`` unchanged when the caller already claimed it (the
    cacheable path, D-CDX-13); otherwise enqueue+claim a fresh one for the
    degraded (uncacheable) path, exactly as the inline version did.
    """
    if task is not None:
        return task
    task_id = cache_digest or f"degraded-{spec.name}-{_now()}"
    task = tq.enqueue_task(
        task_id,
        "build",
        repo=key.repo,
        payload={"spec": spec.name, "components": key.components()},
        path=tree,
    )
    return tq.record_state(task, tq.RUNNING, "", path=tree)


def _run_degraded_build(tree: Path, spec: BuildSpec) -> list[dict[str, Any]]:
    """Degraded (dirty tree): build IN the caller's own tree — it is already
    isolated from the canonical checkout by lane_scope, and a dirty tree
    cannot be represented as a commit to materialize. Nothing is cached (see
    request()).
    """
    _run_build_command(tree, spec)
    artifacts: list[dict[str, Any]] = []
    for pattern in spec.artifacts:
        matches = sorted(
            str(p) for p in (tree / spec.workdir).glob(pattern) if p.is_file()
        )
        artifacts.extend(
            {
                "pattern": pattern,
                "stored_at": m,
                "sha256": _sha256_file(Path(m)),
            }
            for m in matches
        )
    return artifacts


def _write_build_manifest(
    cache_digest: str,
    key: CacheKey,
    spec: BuildSpec,
    seconds: float,
    artifacts: list[dict[str, Any]],
    tree: Path,
) -> None:
    manifest = {
        "key": cache_digest,
        "components": key.components(),
        "spec": spec.name,
        "command": list(spec.command),
        "built_at": _now(),
        "seconds": seconds,
        "artifacts": artifacts,
    }
    _manifest_path(cache_digest, tree).parent.mkdir(parents=True, exist_ok=True)
    _manifest_path(cache_digest, tree).write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def _build(
    tree: Path,
    spec: BuildSpec,
    key: CacheKey,
    *,
    cache_digest: str | None,
    colocated: bool,
    owner: dict[str, Any] | None,
    task: tq.Task | None = None,
) -> dict[str, Any]:
    """Run the build and publish artifacts. ``task`` is the RUNNING record —
    when the caller already claimed it via :func:`_claim_build_task` (the
    cacheable path), it is passed in and reused as-is so this function never
    appends a second, redundant task record for the same key (D-CDX-13).
    The degraded (uncacheable) path has no key to claim against and still
    enqueues its own task here, exactly as before.
    """
    task = _ensure_build_task(task, spec, cache_digest, key, tree)
    started = time.monotonic()
    try:
        with tq.acquire(
            "build",
            operation=f"build:{key.repo}:{spec.name}",
            owner=owner,
            path=tree,
            colocated=colocated,
        ):
            disk_reservation_id = cache_digest or f"degraded-{spec.name}"
            disk_decision = _admit_disk(tree, spec, reservation_id=disk_reservation_id)
            if not disk_decision.admitted:
                raise BuildQueueError(
                    f"build spec {spec.name!r} refused by disk admission after a "
                    f"bounded gc() self-heal attempt: {disk_decision.reason} "
                    f"(code={disk_decision.code.value}, "
                    f"free={disk_decision.observed.free_mib}MiB, "
                    f"predicted_free={disk_decision.predicted_free_mib}MiB) — "
                    "reclaim space or raise the repo's declared "
                    "disk_high_watermark_mib in .buildcache.yaml"
                )
            if cache_digest is not None:
                with materialized(tree, "HEAD", scope=lane_scope(tree)) as build_tree:
                    _run_build_command(build_tree, spec)
                    artifacts = _publish_artifacts(build_tree, spec, cache_digest, tree)
            else:
                artifacts = _run_degraded_build(tree, spec)
    except Exception as exc:  # noqa: BLE001 - re-recorded as a FAILED task, then re-raised
        tq.record_state(task, tq.FAILED, str(exc), path=tree)
        raise
    seconds = time.monotonic() - started
    if cache_digest is not None:
        _write_build_manifest(cache_digest, key, spec, seconds, artifacts, tree)
    tq.record_state(task, tq.DONE, "", path=tree)
    return {
        "ok": True,
        "cached": False,
        "key": cache_digest,
        "components": key.components(),
        "artifacts": artifacts,
        "seconds": seconds,
    }


# ---------------------------------------------------------------------------
# Disk admission (P0.6) — wires the EXISTING disk_policy.py hysteresis; never
# reimplemented here. "check `df -h` before a big build" was pure prose: the
# broker built into whatever space happened to be free and only failed loudly
# (mid-build, wasting the compute) when it ran out. `_admit_disk` is the
# structural replacement -- a refusal BEFORE `_run_build_command` is ever
# reached, with one bounded, synchronous self-heal attempt (this module's own
# `gc()`) before it gives up.
# ---------------------------------------------------------------------------
#: Fallback watermarks (fraction of total disk used) when a repo's
#: `.buildcache.yaml` declares no `disk_high_watermark_mib`/
#: `disk_low_watermark_mib` of its own. `DiskPolicy` already refuses a
#: request whose predicted usage would exceed observed free space with NO
#: watermarks declared at all (`INSUFFICIENT_FREE`) -- these give a build a
#: chance to be refused, and to trigger the GC self-heal, well before that
#: hard floor, which is the entire value of the high/low hysteresis over a
#: bare free-space check.
_DEFAULT_DISK_HIGH_FRACTION = 0.90
_DEFAULT_DISK_LOW_FRACTION = 0.80

_DISK_POLICY_LOCK = threading.Lock()
_DISK_POLICY: DiskPolicy | None = None


def _disk_policy() -> DiskPolicy:
    """The one process-local disk-hysteresis state machine every build shares.

    Mirrors :func:`repository_manager.remote_worker_actions.capacity_inventory`'s
    singleton convention: constructed once, never a second instance, so the
    high/low hysteresis (open->blocked->open) is coherent across requests
    instead of resetting on every call.
    """

    global _DISK_POLICY
    with _DISK_POLICY_LOCK:
        if _DISK_POLICY is None:
            _DISK_POLICY = DiskPolicy()
        return _DISK_POLICY


def _disk_watermarks(*, total_mib: int, policy_key: str) -> DiskWatermarks:
    """Fraction-of-total watermarks, keyed per build spec.

    ``BuildSpec.resources`` is `config_schema.ResourceRequest` — a plain
    ``(cpu_weight, memory_mb, disk_mb, process_slots)`` request with no
    watermark fields of its own (those live on the DIFFERENT, native
    ``development.models.ResourceRequest`` the not-yet-reachable
    ``ResourceScheduler`` uses — see GOC-60's "two `ResourceRequest` classes"
    finding). Rather than growing a second copy of that field here, this
    always derives the watermark from measured total disk; ``policy_key``
    (already scoped per spec name by the caller) keeps two specs in the same
    repo from sharing hysteresis state.
    """

    return DiskWatermarks(
        low_mib=int(total_mib * _DEFAULT_DISK_LOW_FRACTION),
        high_mib=int(total_mib * _DEFAULT_DISK_HIGH_FRACTION),
        policy_key=policy_key,
    )


def _admit_disk(tree: Path, spec: BuildSpec, *, reservation_id: str) -> DiskDecision:
    """Refuse (never silently proceed) when disk admission does not permit
    this build, attempting one bounded GC self-heal first.

    ``requested_mib`` comes from the repo's own declared
    ``disk_estimate_mb`` (0 when undeclared -- the request still gets the
    hard free-space floor and the high/low hysteresis, just with no
    predicted headroom subtracted).
    """

    root = _artifact_root(tree)
    usage = shutil.disk_usage(root)
    total_mib = usage.total // (1024 * 1024)
    free_mib = usage.free // (1024 * 1024)
    policy_key = f"build:{spec.name}"
    watermarks = _disk_watermarks(total_mib=total_mib, policy_key=policy_key)
    policy = _disk_policy()
    decision = policy.evaluate(
        "local",
        total_mib=total_mib,
        free_mib=free_mib,
        requested_mib=max(0, spec.disk_estimate_mb),
        watermarks=watermarks,
        reservation_id=reservation_id,
        request_gc=False,
        mutate=True,
        policy_key=policy_key,
    )
    if decision.admitted:
        return decision

    # Self-heal: bounded, safe reclamation (never touches a RUNNING task's
    # publication) — see `gc()`'s own docstring for the invariants it holds.
    gc(repo_path=tree, keep_recent=10, max_age_days=14)
    usage = shutil.disk_usage(root)
    free_mib = usage.free // (1024 * 1024)
    return policy.evaluate(
        "local",
        total_mib=total_mib,
        free_mib=free_mib,
        requested_mib=max(0, spec.disk_estimate_mb),
        watermarks=watermarks,
        reservation_id=reservation_id,
        request_gc=False,
        mutate=True,
        policy_key=policy_key,
    )


def _run_build_command(tree: Path, spec: BuildSpec) -> None:
    workdir = tree / spec.workdir
    # P0.6 invariant: the runner allocates CARGO_TARGET_DIR/TMPDIR itself —
    # never whatever the calling process happened to inherit — and appends
    # `--no-fail-fast` to a cargo-test-shaped command a caller's declared
    # spec omitted it from. Neither is representable in the argv/env that
    # actually reaches `subprocess.run` below.
    parts = partitioned_paths(tree)
    env = dict(os.environ)
    env["CARGO_TARGET_DIR"] = str(parts.cargo_target_dir)
    env["TMPDIR"] = str(parts.scratch_dir)
    argv = ensure_no_fail_fast(list(spec.command))
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        argv,
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
        timeout=spec.timeout,
        env=env,
    )
    if proc.returncode != 0:
        raise BuildQueueError(
            f"build spec {spec.name!r} failed (exit {proc.returncode}): "
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}"
        )


# ---------------------------------------------------------------------------
# status / artifacts / explain / gc — DELIVERABLE 3's verbs
# ---------------------------------------------------------------------------
def status(
    *, repo_path: Path | str | None = None, key: str = "", spec_name: str = ""
) -> dict[str, Any]:
    scope = lane_scope(repo_path)
    if key:
        manifest = _read_manifest(key, scope.tree)
        task = tq.find_task("build", key, path=scope.tree)
        return {
            "repo": stable_repository_id(scope.main_tree),
            "key": key,
            "cached": bool(manifest and _manifest_is_valid(manifest, scope.tree, key)),
            "manifest": manifest,
            "task": task.to_record() if task else None,
        }
    config = load_config(scope.tree)
    spec = config.spec(spec_name)
    computed = compute_cache_key(
        scope.tree, spec, repo_name=stable_repository_id(scope.main_tree)
    )
    return {
        "repo": stable_repository_id(scope.main_tree),
        "spec": spec.name,
        "computable": computed.computable,
        "degraded_reason": computed.degraded_reason,
        "key": computed.digest if computed.computable else None,
        "components": computed.components(),
        "execution_class": tq.class_status("build", scope.tree),
    }


def artifact_paths(*, repo_path: Path | str | None = None, key: str) -> dict[str, Any]:
    scope = lane_scope(repo_path)
    manifest = _read_manifest(key, scope.tree)
    if manifest is None or not _manifest_is_valid(manifest, scope.tree, key):
        raise BuildQueueError(
            f"no committed cached build for key {key!r} in {scope.main_tree.name}"
        )
    return {"key": key, "artifacts": manifest.get("artifacts", [])}


def explain(
    *, repo_path: Path | str | None = None, key: str, spec_name: str = ""
) -> dict[str, Any]:
    """*Why did this not hit cache?* Diff the CURRENT key's components against
    the most recently recorded ones for this (repo, spec) — without this, an
    operator sees a miss and has to guess whether the tree, the toolchain, or
    the feature set moved. This is what makes the cache legible enough that a
    miss does not get shrugged off and bypassed.
    """
    scope = lane_scope(repo_path)
    config = load_config(scope.tree)
    spec = config.spec(spec_name)
    current = compute_cache_key(
        scope.tree, spec, repo_name=stable_repository_id(scope.main_tree)
    )
    manifest = _read_manifest(key, scope.tree)
    if manifest is None:
        return {
            "key": key,
            "reason": "no manifest recorded for this key at all — first request, "
            "or it was gc'd",
            "current": current.components()
            if current.computable
            else {"degraded_reason": current.degraded_reason},
        }
    previous = manifest.get("components", {})
    if not current.computable:
        return {
            "key": key,
            "reason": f"current request is DEGRADED ({current.degraded_reason}) — "
            "the cache is never consulted for a degraded request",
            "previous": previous,
        }
    differing = {
        field: {"previous": previous.get(field), "current": value}
        for field, value in current.components().items()
        if previous.get(field) != value
    }
    return {
        "key": key,
        "current_key": current.digest,
        "differing_components": differing or None,
        "reason": (
            "keys match — this WOULD be a hit"
            if not differing
            else f"{', '.join(differing)} changed since this key was cached"
        ),
    }


def _scan_cache_manifests(
    root: Path, scope: Any
) -> tuple[list[tuple[str, dict[str, Any]]], str | None]:
    """Every readable manifest under ``root``, and an error string if the
    scan had to stop early (OS error, or the entry-count bound).  Entries
    accumulated before an early stop are still returned — the caller reports
    them as "kept" rather than discarding them.
    """
    entries: list[tuple[str, dict[str, Any]]] = []
    try:
        iterator = os.scandir(root)
    except OSError as exc:
        return entries, str(exc)
    with iterator:
        scanned = 0
        for entry in iterator:
            scanned += 1
            if scanned > _MAX_LEGACY_SCAN_ENTRIES:
                return entries, "legacy cache scan exceeds its entry bound"
            if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
                continue
            manifest = _read_manifest(entry.name, scope.tree)
            if manifest:
                entries.append((entry.name, manifest))
    return entries, None


def _gc_entry_is_protected(
    digest: str, manifest: dict[str, Any], keep_ids: set[str], cutoff: float, scope: Any
) -> bool:
    """True if this cache entry must be kept without even attempting reclamation."""
    if manifest.get("schema") == "build-artifact:v2":
        # Durable WorkItems are not represented in the compatibility task
        # queue.  Without an authority probe this surface must never
        # reclaim their live or waiting publications.
        return True
    task = tq.find_task("build", digest, path=scope.tree)
    if task is not None and task.state == tq.RUNNING:
        return True
    if digest in keep_ids:
        return True
    built_at = manifest.get("built_at", "")
    try:
        import datetime as _dt

        age_ok = _dt.datetime.fromisoformat(built_at).timestamp() < cutoff
    except (TypeError, ValueError):
        age_ok = True  # unparsable timestamp: treat as old enough to reclaim
    return not age_ok


def _gc_reclaim_entry(root: Path, digest: str) -> int | None:
    """Remove one cache entry's tree; ``None`` means keep (size probe failed)."""
    entry_dir = root / digest
    try:
        total_bytes = _bounded_legacy_tree_size(entry_dir)
    except BuildQueueError:
        return None
    shutil.rmtree(entry_dir, ignore_errors=True)
    return total_bytes


def gc(
    *,
    repo_path: Path | str | None = None,
    keep_recent: int = 10,
    max_age_days: int = 14,
) -> dict[str, Any]:
    """Bounded reclamation of cached artifacts. An unbounded cache is a future
    disk outage (measured: ``/home`` at ~77% used) — this is deliberately
    conservative: it NEVER removes a manifest with a still-``RUNNING`` task,
    and always keeps the ``keep_recent`` most-recently-built entries even if
    they are older than ``max_age_days``, so a rarely-rebuilt-but-still-valid
    artifact does not get evicted purely on age.
    """
    scope = lane_scope(repo_path)
    root = _artifact_root(scope.tree)
    entries, scan_error = _scan_cache_manifests(root, scope)
    if scan_error is not None:
        return {
            "repo": stable_repository_id(scope.main_tree),
            "removed": [],
            "kept": [digest for digest, _ in entries],
            "reclaimed_bytes": 0,
            "errors": [scan_error],
        }
    entries.sort(key=lambda pair: pair[1].get("built_at", ""), reverse=True)
    keep_ids = {digest for digest, _ in entries[:keep_recent]}
    cutoff = time.time() - max_age_days * 86400
    removed: list[str] = []
    kept: list[str] = []
    reclaimed_bytes = 0
    for digest, manifest in entries:
        if _gc_entry_is_protected(digest, manifest, keep_ids, cutoff, scope):
            kept.append(digest)
            continue
        reclaimed = _gc_reclaim_entry(root, digest)
        if reclaimed is None:
            kept.append(digest)
            continue
        removed.append(digest)
        reclaimed_bytes += reclaimed
    return {
        "repo": stable_repository_id(scope.main_tree),
        "removed": removed,
        "kept": kept,
        "reclaimed_bytes": reclaimed_bytes,
    }


# ---------------------------------------------------------------------------
# Host dispatch (P0.7) — "give rm_build request a host= that actually
# dispatches". `colocated=True` (the local path above) stays the honest
# default when no host is selected; this is a SEPARATE, additive path, not a
# rewrite of `request()`.
# ---------------------------------------------------------------------------
def request_on_host(
    host_id: str,
    *,
    repo_path: Path | str | None = None,
    spec_name: str = "",
) -> dict[str, Any]:
    """Dispatch this repo's declared build to a registered, authorized host.

    Resolves the ORIGIN and HEAD sha from the caller's own local checkout
    (refusing a dirty tree — a build dispatched from uncommitted state would
    silently build something the caller cannot reproduce) and hands them,
    with the declared ``.buildcache.yaml`` command, to
    :func:`repository_manager.remote_worker_actions.dispatch`'s
    ``dispatch_build`` action — which performs the actual staging, capacity
    admission, and remote execution over
    :class:`~repository_manager.remote_execution.ssh_executor.TunnelSSHExecutor`.
    Never caches (host dispatch is a distinct path from this module's own
    content-addressed cache) and does not yet retrieve build artifacts back
    to the caller — see ``dispatch_build``'s own returned ``note``.
    """

    from repository_manager import remote_worker_actions

    scope = lane_scope(repo_path)
    config = load_config(scope.tree)
    spec = config.spec(spec_name)
    repo_id = stable_repository_id(scope.main_tree)

    status_out = _require_git(["status", "--porcelain"], scope.tree)
    if status_out.strip():
        raise BuildQueueError(
            f"refusing to dispatch {repo_id!r} to host {host_id!r}: the local "
            "tree has uncommitted changes; commit (or use the local "
            "'request' path, which builds a dirty tree in place) before "
            "dispatching to a remote host"
        )
    tree_sha = _require_git(["rev-parse", "HEAD"], scope.tree).strip()
    origin = _require_git(["remote", "get-url", "origin"], scope.tree).strip()

    result = remote_worker_actions.dispatch(
        "dispatch_build",
        host_id=host_id,
        repository_id=repo_id,
        origin=origin,
        tree_sha=tree_sha,
        command=list(spec.command),
        workdir=spec.workdir,
        timeout_seconds=spec.timeout,
        cpu_weight=spec.resources.cpu_weight,
        memory_mib=spec.resources.memory_mb,
        disk_mib=max(spec.resources.disk_mb, spec.disk_estimate_mb),
        process_slots=spec.resources.process_slots,
    )
    return {"repo": repo_id, "spec": spec.name, **result}


# ---------------------------------------------------------------------------
# One action-routed entrypoint shared by the CLI, `python -m`, and the MCP tool
# ---------------------------------------------------------------------------
def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    handlers = {
        "request": (
            (
                lambda: request_on_host(
                    kwargs["host"],
                    repo_path=kwargs.get("path"),
                    spec_name=kwargs.get("spec", "") or "",
                )
            )
            if kwargs.get("host")
            else lambda: request(
                repo_path=kwargs.get("path"),
                spec_name=kwargs.get("spec", "") or "",
                colocated=bool(kwargs.get("colocated", False)),
                wait_timeout=int(kwargs.get("wait_timeout") or 60),
                generation_id=kwargs.get("generation_id"),
                job_service=kwargs.get("job_service"),
                build_service=kwargs.get("build_service"),
                tenant_id=kwargs.get("tenant_id", "") or "",
                owner_id=kwargs.get("owner_id", "") or "",
                session_id=kwargs.get("session_id", "") or "",
            )
        ),
        "status": lambda: status(
            repo_path=kwargs.get("path"),
            key=kwargs.get("key", "") or "",
            spec_name=kwargs.get("spec", "") or "",
        ),
        "artifacts": lambda: artifact_paths(
            repo_path=kwargs.get("path"), key=kwargs.get("key", "") or ""
        ),
        "explain": lambda: explain(
            repo_path=kwargs.get("path"),
            key=kwargs.get("key", "") or "",
            spec_name=kwargs.get("spec", "") or "",
        ),
        "gc": lambda: gc(
            repo_path=kwargs.get("path"),
            keep_recent=int(kwargs.get("keep_recent") or 10),
            max_age_days=int(kwargs.get("max_age_days") or 14),
        ),
    }
    handler = handlers.get(action)
    if handler is None:
        return {
            "ok": False,
            "error": f"unknown build-queue action: {action!r}",
            "actions": sorted(handlers),
        }
    return handler()


def main(argv: list[str] | None = None) -> int:
    """``python -m repository_manager.build_queue [request|status|artifacts|explain|gc]``"""
    import argparse

    p = argparse.ArgumentParser(prog="python -m repository_manager.build_queue")
    p.add_argument(
        "action",
        nargs="?",
        default="request",
        choices=["request", "status", "artifacts", "explain", "gc"],
    )
    p.add_argument("--path", default=None)
    p.add_argument("--spec", default="")
    p.add_argument("--key", default="")
    p.add_argument(
        "--same-node",
        action="store_true",
        help=(
            "Assert this invocation runs on the SAME node as the target repo's "
            "lease holder. Only pass this when that is actually true (e.g. this "
            "IS repository-manager-mcp, pinned there) — an unproven assertion "
            "reintroduces the exact false-safety this flag exists to prevent."
        ),
    )
    p.add_argument("--wait-timeout", type=int, default=60)
    p.add_argument("--keep-recent", type=int, default=10)
    p.add_argument("--max-age-days", type=int, default=14)
    args = p.parse_args(argv)
    try:
        out = dispatch(
            args.action,
            path=args.path,
            spec=args.spec,
            key=args.key,
            colocated=args.same_node,
            wait_timeout=args.wait_timeout,
            keep_recent=args.keep_recent,
            max_age_days=args.max_age_days,
        )
    except LaneArbitrationError as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    print(json.dumps(out, default=str, indent=2))
    return 1 if out.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
