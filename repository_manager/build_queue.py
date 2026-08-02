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
import shutil
import subprocess  # nosec B404 - fixed argv, never shell=True
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml
from agent_utilities.governance.lanes import (
    LaneArbitrationError,
    lane_scope,
)

from repository_manager import task_queue as tq
from repository_manager.merge_queue import (
    _now,  # reuse the same UTC-isoformat timestamp helper merge_queue uses
    _require_git,
    _run_git,
    materialized,
)

CONFIG_FILENAME = ".buildcache.yaml"
ARTIFACT_STORE_DIRNAME = "build-cache"
EXECUTION_CLASS = "build"


class BuildQueueError(LaneArbitrationError):
    """A build-broker operation refused, carrying the reason a caller must act on."""


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


@dataclass(frozen=True)
class BuildConfig:
    base: str = "main"
    specs: tuple[BuildSpec, ...] = ()
    source: str = ""

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
    specs: list[BuildSpec] = []
    for index, raw in enumerate(data.get("specs") or []):
        if not isinstance(raw, dict):
            raise BuildQueueError(
                f"{source or CONFIG_FILENAME}: specs[{index}] is not a mapping"
            )
        name = str(raw.get("name") or "").strip()
        if not name:
            raise BuildQueueError(
                f"{source or CONFIG_FILENAME}: specs[{index}] has no name"
            )
        specs.append(
            BuildSpec(
                name=name,
                command=_as_argv(raw.get("command"), where=f"spec {name!r}"),
                workdir=str(raw.get("workdir", ".")),
                toolchain_fingerprint=(
                    _as_argv(
                        raw["toolchain_fingerprint"],
                        where=f"spec {name!r} toolchain_fingerprint",
                    )
                    if raw.get("toolchain_fingerprint")
                    else ()
                ),
                cache_key_paths=tuple(str(p) for p in raw.get("cache_key_paths") or ()),
                artifacts=tuple(str(p) for p in raw.get("artifacts") or ()),
                timeout=int(raw.get("timeout", 3600)),
                target_triple=str(raw.get("target_triple", "")),
                source=source,
            )
        )
    names = [s.name for s in specs]
    duplicates = sorted({n for n in names if names.count(n) > 1})
    if duplicates:
        raise BuildQueueError(
            f"{source or CONFIG_FILENAME}: duplicate spec name(s) {duplicates}"
        )
    return BuildConfig(
        base=str(data.get("base", "main")), specs=tuple(specs), source=source
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
    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise BuildQueueError(f"{config_path}: must be a YAML mapping")
    return parse_config(data, source=str(config_path))


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


def compute_cache_key(repo: Path, spec: BuildSpec, *, repo_name: str) -> CacheKey:
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
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _manifest_is_valid(
    manifest: dict[str, Any], path: Path | str | None = None
) -> bool:
    """A manifest is only a hit if every artifact it lists is STILL on disk with
    the recorded checksum — gc or an out-of-band deletion must degrade to a
    miss, never to serving a dangling reference.
    """
    for entry in manifest.get("artifacts", []):
        artifact_path = Path(entry["stored_at"])
        if not artifact_path.is_file():
            return False
        if _sha256_file(artifact_path) != entry["sha256"]:
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _publish_artifacts(
    build_tree: Path, spec: BuildSpec, key_digest: str, path: Path | str | None = None
) -> list[dict[str, Any]]:
    workdir = build_tree / spec.workdir
    dest_dir = _artifact_root(path) / key_digest / "artifacts"
    dest_dir.mkdir(parents=True, exist_ok=True)
    published: list[dict[str, Any]] = []
    for pattern in spec.artifacts:
        matches = sorted(str(p) for p in workdir.glob(pattern) if p.is_file())
        if not matches:
            raise BuildQueueError(
                f"build spec {spec.name!r} declared artifact pattern {pattern!r} "
                "but the build produced no file matching it — a build that "
                "silently produces nothing is a failure, not a success."
            )
        for match in matches:
            src = Path(match)
            relative = src.relative_to(workdir)
            dest = dest_dir / relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest)
            published.append(
                {
                    "pattern": pattern,
                    "relative_path": str(relative),
                    "stored_at": str(dest),
                    "sha256": _sha256_file(dest),
                    "bytes": dest.stat().st_size,
                }
            )
    return published


# ---------------------------------------------------------------------------
# request() — the one entrypoint: dedup-or-build
# ---------------------------------------------------------------------------
def request(
    *,
    repo_path: Path | str | None = None,
    spec_name: str = "",
    colocated: bool = False,
    owner: dict[str, Any] | None = None,
    wait_timeout: int = 60,
) -> dict[str, Any]:
    """Serve a build from cache, or build it and publish the result.

    Two requests for the SAME computable key: the second waits (bounded by
    ``wait_timeout``) on the first's in-flight :class:`~task_queue.Task` and
    then reuses its artifacts rather than rebuilding — this is where the
    dedup value comes from. Two requests for DIFFERENT keys run in parallel up
    to the ``"build"`` execution class's pool cap; identical keys serialise
    through the Task check below, not through the pool.
    """
    scope = lane_scope(repo_path)
    config = load_config(scope.tree)
    spec = config.spec(spec_name)
    repo_name = scope.main_tree.name
    key = compute_cache_key(scope.tree, spec, repo_name=repo_name)

    if not key.computable:
        result = _build(
            scope.tree, spec, key, cache_digest=None, colocated=colocated, owner=owner
        )
        result["degraded"] = True
        result["degraded_reason"] = key.degraded_reason
        result["cached"] = False
        return result

    digest = key.digest
    manifest = _read_manifest(digest, scope.tree)
    if manifest and _manifest_is_valid(manifest, scope.tree):
        return {
            "ok": True,
            "cached": True,
            "degraded": False,
            "key": digest,
            "components": key.components(),
            "artifacts": manifest["artifacts"],
            "built_at": manifest.get("built_at"),
        }

    existing = tq.find_task("build", digest, path=scope.tree)
    if existing and existing.state == tq.RUNNING:
        waited = _wait_for_task(digest, scope.tree, timeout=wait_timeout)
        if waited:
            manifest = _read_manifest(digest, scope.tree)
            if manifest and _manifest_is_valid(manifest, scope.tree):
                return {
                    "ok": True,
                    "cached": True,
                    "degraded": False,
                    "waited": True,
                    "key": digest,
                    "components": key.components(),
                    "artifacts": manifest["artifacts"],
                    "built_at": manifest.get("built_at"),
                }
        # The in-flight build finished without producing a valid manifest
        # (it failed) or never finished within wait_timeout — fall through
        # and try building ourselves rather than getting stuck.

    result = _build(
        scope.tree, spec, key, cache_digest=digest, colocated=colocated, owner=owner
    )
    result["degraded"] = False
    return result


def _wait_for_task(task_id: str, path: Path | str, *, timeout: int) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        task = tq.find_task("build", task_id, path=path)
        if task is None or task.state != tq.RUNNING:
            return True
        time.sleep(2)
    return False


def _build(
    tree: Path,
    spec: BuildSpec,
    key: CacheKey,
    *,
    cache_digest: str | None,
    colocated: bool,
    owner: dict[str, Any] | None,
) -> dict[str, Any]:
    task_id = cache_digest or f"degraded-{spec.name}-{_now()}"
    task = tq.enqueue_task(
        task_id,
        "build",
        repo=key.repo,
        payload={"spec": spec.name, "components": key.components()},
        path=tree,
    )
    task = tq.record_state(task, tq.RUNNING, "", path=tree)
    started = time.monotonic()
    try:
        with tq.acquire(
            "build",
            operation=f"build:{key.repo}:{spec.name}",
            owner=owner,
            path=tree,
            colocated=colocated,
        ):
            if cache_digest is not None:
                with materialized(tree, "HEAD", scope=lane_scope(tree)) as build_tree:
                    _run_build_command(build_tree, spec)
                    artifacts = _publish_artifacts(build_tree, spec, cache_digest, tree)
            else:
                # Degraded (dirty tree): build IN the caller's own tree — it is
                # already isolated from the canonical checkout by lane_scope,
                # and a dirty tree cannot be represented as a commit to
                # materialize. Nothing is cached (see request()).
                _run_build_command(tree, spec)
                artifacts = []
                for pattern in spec.artifacts:
                    matches = sorted(
                        str(p)
                        for p in (tree / spec.workdir).glob(pattern)
                        if p.is_file()
                    )
                    artifacts.extend(
                        {
                            "pattern": pattern,
                            "stored_at": m,
                            "sha256": _sha256_file(Path(m)),
                        }
                        for m in matches
                    )
    except Exception as exc:  # noqa: BLE001 - re-recorded as a FAILED task, then re-raised
        tq.record_state(task, tq.FAILED, str(exc), path=tree)
        raise
    seconds = time.monotonic() - started
    if cache_digest is not None:
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
    tq.record_state(task, tq.DONE, "", path=tree)
    return {
        "ok": True,
        "cached": False,
        "key": cache_digest,
        "components": key.components(),
        "artifacts": artifacts,
        "seconds": seconds,
    }


def _run_build_command(tree: Path, spec: BuildSpec) -> None:
    workdir = tree / spec.workdir
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        list(spec.command),
        cwd=str(workdir),
        capture_output=True,
        text=True,
        check=False,
        timeout=spec.timeout,
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
            "repo": scope.main_tree.name,
            "key": key,
            "cached": bool(manifest and _manifest_is_valid(manifest, scope.tree)),
            "manifest": manifest,
            "task": task.to_record() if task else None,
        }
    config = load_config(scope.tree)
    spec = config.spec(spec_name)
    computed = compute_cache_key(scope.tree, spec, repo_name=scope.main_tree.name)
    return {
        "repo": scope.main_tree.name,
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
    if manifest is None:
        raise BuildQueueError(
            f"no cached build for key {key!r} in {scope.main_tree.name}"
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
    current = compute_cache_key(scope.tree, spec, repo_name=scope.main_tree.name)
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
    entries: list[tuple[str, dict[str, Any]]] = []
    for entry_dir in sorted(root.iterdir()) if root.is_dir() else []:
        manifest = _read_manifest(entry_dir.name, scope.tree)
        if manifest:
            entries.append((entry_dir.name, manifest))
    entries.sort(key=lambda pair: pair[1].get("built_at", ""), reverse=True)
    keep_ids = {digest for digest, _ in entries[:keep_recent]}
    cutoff = time.time() - max_age_days * 86400
    removed: list[str] = []
    kept: list[str] = []
    for digest, manifest in entries:
        task = tq.find_task("build", digest, path=scope.tree)
        if task is not None and task.state == tq.RUNNING:
            kept.append(digest)
            continue
        if digest in keep_ids:
            kept.append(digest)
            continue
        built_at = manifest.get("built_at", "")
        try:
            import datetime as _dt

            age_ok = _dt.datetime.fromisoformat(built_at).timestamp() < cutoff
        except ValueError:
            age_ok = True  # unparsable timestamp: treat as old enough to reclaim
        if not age_ok:
            kept.append(digest)
            continue
        entry_dir = root / digest
        total_bytes = sum(f.stat().st_size for f in entry_dir.rglob("*") if f.is_file())
        shutil.rmtree(entry_dir, ignore_errors=True)
        removed.append(digest)
        gc.reclaimed_bytes = getattr(gc, "reclaimed_bytes", 0) + total_bytes  # type: ignore[attr-defined]
    return {
        "repo": scope.main_tree.name,
        "removed": removed,
        "kept": kept,
        "reclaimed_bytes": getattr(gc, "reclaimed_bytes", 0),
    }


# ---------------------------------------------------------------------------
# One action-routed entrypoint shared by the CLI, `python -m`, and the MCP tool
# ---------------------------------------------------------------------------
def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    handlers = {
        "request": lambda: request(
            repo_path=kwargs.get("path"),
            spec_name=kwargs.get("spec", "") or "",
            colocated=bool(kwargs.get("colocated", False)),
            wait_timeout=int(kwargs.get("wait_timeout") or 60),
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
