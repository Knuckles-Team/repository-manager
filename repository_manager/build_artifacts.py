"""Fenced, checksummed publication for durable build artifacts.

The build WorkItem is the authority for lifecycle and fencing.  This module
only owns the byte store and its immutable manifest; it never creates a job,
reservation, or second task ledger.  Publication is deliberately split into
three operations so a worker can be recovered at every boundary::

    stage -> publish manifest -> terminal WorkItem commit

Only a committed manifest is a cache hit.  A process crash after publication
therefore cannot turn a partially completed WorkItem into a false cache hit;
the worker/reconciler may finalize it after observing the durable terminal
state.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import time
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path, PurePosixPath
from typing import Any

from agent_utilities.governance.lanes import lane_scope
from agent_utilities.knowledge_graph.core.file_lock import lock_exclusive, unlock

_SAFE_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")
MANIFEST_NAME = "manifest.json"
ARTIFACT_SCHEMA = "build-artifact:v2"
_MAX_MANIFEST_BYTES = 1 << 20
_MAX_MANIFEST_ARTIFACTS = 4096
_MAX_STAGE_ARTIFACTS = 4096
_MAX_STAGE_BYTES = 64 * 1024 * 1024 * 1024
_MAX_SCAN_ENTRIES = 10_000
_MAX_SCAN_FILES = 100_000
_MAX_SCAN_BYTES = 64 * 1024 * 1024 * 1024
_MAX_SCAN_DEPTH = 128
_DEFAULT_STALE_STAGE_SECONDS = 24 * 60 * 60


class ArtifactStoreError(RuntimeError):
    """A build artifact could not be staged, verified, or published."""


class ArtifactFenceLost(ArtifactStoreError):
    """The WorkItem fence changed before a filesystem publication."""


@dataclass(frozen=True)
class StagedArtifacts:
    """Immutable metadata for one attempt's staging directory."""

    key: str
    stage_dir: Path
    manifest: Mapping[str, Any]
    fence: str
    attempt: int
    job_id: str = ""
    work_item_id: str = ""


def _safe(value: str) -> str:
    return _SAFE_COMPONENT.sub("-", value)


def _read_bounded_json(path: Path) -> dict[str, Any] | None:
    """Read one bounded JSON file without following a replaced symlink."""

    descriptor: int | None = None
    try:
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            raw = handle.read(_MAX_MANIFEST_BYTES + 1)
        if len(raw) > _MAX_MANIFEST_BYTES:
            return None
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return dict(value) if isinstance(value, Mapping) else None


def _require_regular_file_stat(
    file_stat: os.stat_result, path: Path, max_bytes: int | None
) -> None:
    if not stat.S_ISREG(file_stat.st_mode):
        raise ArtifactStoreError(f"artifact is not a regular file: {path}")
    if max_bytes is not None and file_stat.st_size > max_bytes:
        raise ArtifactStoreError("artifact exceeds its bounded byte limit")


def _hash_file_stream(handle: Any, digest: Any, max_bytes: int | None) -> int:
    """Read ``handle`` in bounded chunks into ``digest``; return bytes copied."""
    copied = 0
    for chunk in iter(lambda: handle.read(1 << 20), b""):
        copied += len(chunk)
        if max_bytes is not None and copied > max_bytes:
            raise ArtifactStoreError("artifact exceeds its bounded byte limit")
        digest.update(chunk)
    return copied


def _require_unchanged_during_read(
    initial: os.stat_result, final: os.stat_result, copied: int
) -> None:
    if (
        final.st_dev != initial.st_dev
        or final.st_ino != initial.st_ino
        or final.st_size != copied
    ):
        raise ArtifactStoreError("artifact changed while it was checksummed")


def _sha256_file(path: Path, *, max_bytes: int | None = None) -> str:
    digest = hashlib.sha256()
    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(descriptor, "rb") as handle:
            descriptor = None
            initial = os.fstat(handle.fileno())
            _require_regular_file_stat(initial, path, max_bytes)
            copied = _hash_file_stream(handle, digest, max_bytes)
            final = os.fstat(handle.fileno())
            _require_unchanged_during_read(initial, final, copied)
    except OSError as exc:
        raise ArtifactStoreError(f"could not checksum artifact {path}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
    return digest.hexdigest()


def _require_regular_source_stat(
    source_stat: os.stat_result, source: Path, max_bytes: int | None
) -> None:
    if not stat.S_ISREG(source_stat.st_mode):
        raise ArtifactStoreError(f"artifact source is not a regular file: {source}")
    if max_bytes is not None and source_stat.st_size > max_bytes:
        raise ArtifactStoreError(
            "build artifacts exceed the admitted staging byte bound"
        )


def _copy_stream_bounded(
    source_handle: Any, destination_handle: Any, max_bytes: int | None
) -> int:
    """Copy ``source_handle`` into ``destination_handle`` in bounded chunks."""
    copied = 0
    while True:
        chunk = source_handle.read(1 << 20)
        if not chunk:
            break
        copied += len(chunk)
        if max_bytes is not None and copied > max_bytes:
            raise ArtifactStoreError(
                "build artifacts exceed the admitted staging byte bound"
            )
        destination_handle.write(chunk)
    destination_handle.flush()
    os.fsync(destination_handle.fileno())
    return copied


def _require_source_unchanged_after_copy(
    source: Path, source_stat: os.stat_result, copied: int
) -> None:
    final_source_stat = os.stat(source, follow_symlinks=False)
    if (
        final_source_stat.st_dev != source_stat.st_dev
        or final_source_stat.st_ino != source_stat.st_ino
        or final_source_stat.st_size != source_stat.st_size
        or final_source_stat.st_size != copied
    ):
        raise ArtifactStoreError("artifact source changed while it was copied")


def _copy_file_no_follow(
    source: Path,
    destination: Path,
    *,
    max_bytes: int | None = None,
) -> int:
    """Copy one stable regular file through no-follow descriptors."""

    _reject_symlink_path(source)
    source_descriptor: int | None = None
    destination_descriptor: int | None = None
    try:
        source_descriptor = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        source_stat = os.fstat(source_descriptor)
        _require_regular_source_stat(source_stat, source, max_bytes)
        _reject_symlink_path(destination.parent)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination_descriptor = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(source_descriptor, "rb") as source_handle:
            source_descriptor = None
            with os.fdopen(destination_descriptor, "wb") as destination_handle:
                destination_descriptor = None
                copied = _copy_stream_bounded(
                    source_handle, destination_handle, max_bytes
                )
        _require_source_unchanged_after_copy(source, source_stat, copied)
        return copied
    except OSError as exc:
        raise ArtifactStoreError(f"could not copy artifact {source}") from exc
    finally:
        for descriptor in (source_descriptor, destination_descriptor):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except OSError:
                    pass


class _CopyTreeCounters:
    """Mutable running totals for one `_copy_tree_no_follow` walk."""

    __slots__ = ("entries", "files", "total")

    def __init__(self) -> None:
        self.entries = 0
        self.files = 0
        self.total = 0


def _copy_tree_child(
    child: os.DirEntry[str],
    destination: Path,
    depth: int,
    counters: _CopyTreeCounters,
    pending: list[tuple[Path, Path, int]],
) -> None:
    """Handle one `os.scandir` entry during `_copy_tree_no_follow`'s bounded walk."""
    counters.entries += 1
    if counters.entries > _MAX_SCAN_ENTRIES:
        raise ArtifactStoreError("artifact stage exceeds its entry bound")
    source_path = Path(child.path)
    destination_path = destination / child.name
    if child.is_symlink():
        raise ArtifactStoreError("symlink artifact component is not allowed")
    if child.is_dir(follow_symlinks=False):
        _reject_symlink_path(destination_path)
        try:
            destination_path.mkdir(parents=False, exist_ok=False)
        except OSError as exc:
            raise ArtifactStoreError(
                "artifact publication directory could not be created"
            ) from exc
        pending.append((source_path, destination_path, depth + 1))
        return
    if not child.is_file(follow_symlinks=False):
        raise ArtifactStoreError("artifact stage contains a non-regular entry")
    counters.files += 1
    if counters.files > _MAX_STAGE_ARTIFACTS:
        raise ArtifactStoreError("artifact stage exceeds its file bound")
    counters.total += _copy_file_no_follow(
        source_path,
        destination_path,
        max_bytes=_MAX_STAGE_BYTES - counters.total,
    )
    if counters.total > _MAX_STAGE_BYTES:
        raise ArtifactStoreError("artifact stage exceeds its byte bound")


def _copy_tree_no_follow(source_root: Path, destination_root: Path) -> None:
    """Copy a bounded tree without ever following a source symlink."""

    _reject_symlink_path(source_root)
    if source_root.is_symlink() or not source_root.is_dir():
        raise ArtifactStoreError("artifact stage root is not a regular directory")
    try:
        destination_root.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        raise ArtifactStoreError(
            "artifact publication directory could not be created"
        ) from exc
    pending: list[tuple[Path, Path, int]] = [(source_root, destination_root, 0)]
    counters = _CopyTreeCounters()
    while pending:
        source, destination, depth = pending.pop()
        if depth > _MAX_SCAN_DEPTH:
            raise ArtifactStoreError("artifact stage exceeds its depth bound")
        try:
            iterator = os.scandir(source)
        except OSError as exc:
            raise ArtifactStoreError("artifact stage could not be read") from exc
        with iterator:
            for child in iterator:
                _copy_tree_child(child, destination, depth, counters, pending)


def _pattern_matches(relative: PurePosixPath, pattern: str) -> bool:
    """Match a relative path with bounded ``glob``-like semantics."""

    normalized = pattern.replace(os.sep, "/")
    value = relative.as_posix()
    if "**" not in normalized:
        if "/" not in normalized and len(relative.parts) != 1:
            return False
        return PurePosixPath(value).match(normalized)
    if normalized == "**":
        prefix, suffix = "", ""
    elif normalized.endswith("/**"):
        prefix, suffix = normalized[:-3], ""
    else:
        prefix, _, suffix = normalized.partition("**/")
    prefix = prefix.rstrip("/")
    if prefix:
        prefix_parts = tuple(PurePosixPath(prefix).parts)
        if tuple(relative.parts[: len(prefix_parts)]) != prefix_parts:
            return False
        remainder = PurePosixPath(*relative.parts[len(prefix_parts) :])
    else:
        remainder = relative
    if not suffix:
        return True
    remainder_value = remainder.as_posix()
    return fnmatchcase(remainder_value, suffix) or fnmatchcase(remainder.name, suffix)


def _bounded_matching_files(
    root: Path,
    patterns: tuple[str, ...],
    *,
    max_entries: int = _MAX_SCAN_ENTRIES,
    max_files: int = _MAX_SCAN_FILES,
    max_bytes: int = _MAX_SCAN_BYTES,
) -> dict[str, list[Path]]:
    """Traverse candidate pattern roots once, without following symlinks."""

    _reject_symlink_path(root)
    try:
        root_stat = os.lstat(root)
    except OSError as exc:
        raise ArtifactStoreError("artifact output root could not be read") from exc
    if not stat.S_ISDIR(root_stat.st_mode):
        raise ArtifactStoreError("artifact output root is not a regular directory")
    matched: dict[str, list[Path]] = {pattern: [] for pattern in patterns}
    # A literal prefix keeps a ``dist/**`` publication from walking an
    # unrelated node_modules tree.  Non-recursive patterns also carry a
    # maximum relative depth, so ``*.js`` only scans direct output files.
    specs: dict[Path, list[tuple[str, tuple[str, ...], bool, int | None]]] = {}
    for pattern in patterns:
        normalized = pattern.replace(os.sep, "/").strip("/")
        parts = tuple(PurePosixPath(normalized).parts)
        literal_count = 0
        for part in parts:
            if any(marker in part for marker in "*?["):
                break
            literal_count += 1
        literal_parts = parts[:literal_count]
        candidate_root = root.joinpath(*literal_parts)
        recursive = "**" in parts[literal_count:]
        max_depth = None if recursive else len(parts) - literal_count
        specs.setdefault(candidate_root, []).append(
            (pattern, literal_parts, recursive, max_depth)
        )
    pending: list[
        tuple[
            Path,
            tuple[str, ...],
            tuple[tuple[str, tuple[str, ...], bool, int | None], ...],
        ]
    ] = []
    entries = matching_files = total_bytes = 0
    for candidate_root, candidate_specs in specs.items():
        _reject_symlink_path(candidate_root)
        try:
            candidate_stat = os.lstat(candidate_root)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise ArtifactStoreError(
                "artifact output candidate root could not be read"
            ) from exc
        if stat.S_ISLNK(candidate_stat.st_mode):
            raise ArtifactStoreError("symlink artifact candidate root is not allowed")
        if stat.S_ISDIR(candidate_stat.st_mode):
            pending.append(
                (candidate_root, candidate_specs[0][1], tuple(candidate_specs))
            )
        elif stat.S_ISREG(candidate_stat.st_mode):
            entries += 1
            if entries > max_entries:
                raise ArtifactStoreError("artifact output tree exceeds its entry bound")
            relative = PurePosixPath(*candidate_specs[0][1])
            for pattern, _literal, _recursive, _max_depth in candidate_specs:
                if not _pattern_matches(relative, pattern):
                    continue
                matching_files += 1
                if matching_files > max_files:
                    raise ArtifactStoreError(
                        "artifact output exceeds its matching file bound"
                    )
                try:
                    size = candidate_stat.st_size
                except OSError as exc:
                    raise ArtifactStoreError(
                        "artifact output could not be statted"
                    ) from exc
                total_bytes += size
                if total_bytes > max_bytes:
                    raise ArtifactStoreError(
                        "artifact output exceeds its matching byte bound"
                    )
                matched[pattern].append(candidate_root)
    while pending:
        current, relative_prefix, current_specs = pending.pop()
        if len(relative_prefix) > _MAX_SCAN_DEPTH:
            raise ArtifactStoreError("artifact output tree exceeds its depth bound")
        try:
            iterator = os.scandir(current)
        except OSError as exc:
            raise ArtifactStoreError("artifact output tree could not be read") from exc
        with iterator:
            for child in iterator:
                entries += 1
                if entries > max_entries:
                    raise ArtifactStoreError(
                        "artifact output tree exceeds its entry bound"
                    )
                relative_parts = (*relative_prefix, child.name)
                relative = PurePosixPath(*relative_parts)
                matches = [
                    pattern
                    for pattern, _literal, _recursive, _max_depth in current_specs
                    if _pattern_matches(relative, pattern)
                ]
                if child.is_symlink():
                    if matches:
                        raise ArtifactStoreError(
                            "symlink artifact component is not allowed"
                        )
                    continue
                if child.is_dir(follow_symlinks=False):
                    can_descend = any(
                        recursive
                        or max_depth is None
                        or len(relative_parts) < len(literal_parts) + max_depth
                        for _pattern, literal_parts, recursive, max_depth in current_specs
                    )
                    if can_descend:
                        pending.append(
                            (Path(child.path), relative_parts, current_specs)
                        )
                    continue
                if not child.is_file(follow_symlinks=False):
                    if matches:
                        raise ArtifactStoreError(
                            "artifact output has a non-regular matching entry"
                        )
                    continue
                if not matches:
                    continue
                matching_files += 1
                if matching_files > max_files:
                    raise ArtifactStoreError(
                        "artifact output tree exceeds its file bound"
                    )
                try:
                    size = child.stat(follow_symlinks=False).st_size
                except OSError as exc:
                    raise ArtifactStoreError(
                        "artifact output could not be statted"
                    ) from exc
                total_bytes += size
                if total_bytes > max_bytes:
                    raise ArtifactStoreError(
                        "artifact output tree exceeds its byte bound"
                    )
                for pattern in matches:
                    matched[pattern].append(Path(child.path))
    for paths in matched.values():
        paths.sort()
    return matched


def _atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_file(path: Path) -> None:
    """Flush one regular file without following a symbolic link."""

    descriptor: int | None = None
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ArtifactStoreError(f"cannot fsync non-regular artifact {path}")
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
            descriptor = None
    except OSError as exc:
        raise ArtifactStoreError(f"could not fsync artifact {path}") from exc
    finally:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass


def _fsync_directory(path: Path) -> None:
    """Flush a directory entry so an atomic rename survives a crash."""

    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as exc:
        raise ArtifactStoreError(f"could not fsync artifact directory {path}") from exc


def _reject_symlink_components(root: Path, candidate: Path) -> None:
    """Reject symlink files or directories anywhere below ``root``."""

    root = root.resolve(strict=True)
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ArtifactStoreError("artifact path escapes its declared root") from exc
    current = root
    for component in relative.parts:
        current /= component
        if current.is_symlink():
            raise ArtifactStoreError(
                f"symlink artifact component is not allowed: {current}"
            )


def _reject_symlink_path(path: Path) -> None:
    """Reject symlinks in a path before resolving them away."""

    absolute = Path(os.path.abspath(path))
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ArtifactStoreError(
                f"symlink artifact component is not allowed: {current}"
            )


def _key_component(key: str) -> str:
    """Validate a cache key before using it as a directory component."""

    if not isinstance(key, str) or not key or key in {".", ".."}:
        raise ArtifactStoreError("artifact key must be a non-blank path component")
    if Path(key).name != key or "/" in key or "\\" in key or "\x00" in key:
        raise ArtifactStoreError("artifact key must not contain path separators")
    return key


def _manifest_attempt(manifest: Mapping[str, Any]) -> int | None:
    try:
        return int(manifest.get("attempt", 0))
    except (TypeError, ValueError):
        return None


def _require_finalize_preconditions(
    terminal_check: Callable[[], bool] | None,
    job_id: str,
    work_item_id: str,
    attempt: int | None,
) -> None:
    """Reject `finalize()` calls missing durable WorkItem identity/proof."""
    if terminal_check is None or not job_id.strip() or not work_item_id.strip():
        raise ArtifactFenceLost(
            "durable terminal WorkItem identity/proof is required before finalize"
        )
    if attempt is None or attempt < 1:
        raise ArtifactFenceLost("durable WorkItem attempt is required before finalize")


def _require_matching_manifest_identity(
    manifest: dict[str, Any] | None,
    key: str,
    fence: str,
    job_id: str,
    work_item_id: str,
    attempt: int | None,
) -> dict[str, Any]:
    """Return ``manifest`` only if it exactly matches this fenced identity."""
    if (
        manifest is None
        or manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("key") != key
        or manifest.get("fence") != fence
        or manifest.get("job_id") != job_id
        or manifest.get("work_item_id") != work_item_id
        or _manifest_attempt(manifest) != attempt
    ):
        raise ArtifactFenceLost(
            "artifact manifest is missing or belongs to another fence"
        )
    return manifest


def _require_finalize_terminal_state(terminal_check: Callable[[], bool]) -> None:
    """Verify the durable terminal WorkItem state authorizes commit."""
    try:
        terminal = terminal_check()
    except Exception as exc:  # pragma: no cover - defensive authority boundary
        raise ArtifactFenceLost(
            "durable terminal WorkItem state could not be verified"
        ) from exc
    if not terminal:
        raise ArtifactFenceLost(
            "durable terminal WorkItem state does not match this publication"
        )


def _exact_authority_proof(
    manifest: Mapping[str, Any],
    proof: object,
    *,
    marker: str,
) -> bool:
    """Accept destructive authority decisions only with exact ownership.

    A bare boolean from a liveness probe is not enough: another attempt may
    have taken the key while a stale controller is still holding its local
    view.  The durable probe must echo every immutable identity in the
    manifest and explicitly authorize the requested destructive transition.
    """

    if not isinstance(proof, Mapping) or proof.get(marker) is not True:
        return False
    for field in ("job_id", "work_item_id", "attempt", "fence"):
        expected = manifest.get(field)
        actual = proof.get(field)
        if expected is None or actual is None:
            return False
        if field == "attempt":
            try:
                if int(expected) != int(actual):
                    return False
            except (TypeError, ValueError):
                return False
        elif str(expected) != str(actual):
            return False
    return True


@dataclass(frozen=True)
class _ScanBounds:
    """Caller-supplied bounds for one `_bounded_tree_size` walk."""

    max_entries: int
    max_files: int
    max_bytes: int


class _TreeSizeCounters:
    """Mutable running totals for one `_bounded_tree_size` walk."""

    __slots__ = ("total", "entries", "files")

    def __init__(self) -> None:
        self.total = 0
        self.entries = 0
        self.files = 0


def _bounded_tree_size_child(
    child: os.DirEntry[str],
    depth: int,
    counters: _TreeSizeCounters,
    pending: list[tuple[Path, int]],
    bounds: _ScanBounds,
) -> None:
    """Handle one `os.scandir` entry during `_bounded_tree_size`'s bounded walk."""
    counters.entries += 1
    if counters.entries > bounds.max_entries:
        raise ArtifactStoreError("artifact scan exceeds its entry bound")
    if child.is_symlink():
        raise ArtifactStoreError("symlink artifact component is not allowed")
    if child.is_dir(follow_symlinks=False):
        pending.append((Path(child.path), depth + 1))
        return
    if not child.is_file(follow_symlinks=False):
        raise ArtifactStoreError("artifact tree contains a non-regular entry")
    counters.files += 1
    if counters.files > bounds.max_files:
        raise ArtifactStoreError("artifact scan exceeds its file bound")
    try:
        counters.total += child.stat(follow_symlinks=False).st_size
    except OSError as exc:
        raise ArtifactStoreError("artifact scan could not stat a file") from exc
    if counters.total > bounds.max_bytes:
        raise ArtifactStoreError("artifact scan exceeds its byte bound")


def _bounded_tree_size(
    root: Path,
    *,
    max_entries: int = _MAX_SCAN_ENTRIES,
    max_files: int = _MAX_SCAN_FILES,
    max_bytes: int = _MAX_SCAN_BYTES,
) -> int:
    """Bounded no-follow scan used before copying or deleting bytes."""

    _reject_symlink_path(root)
    if root.is_symlink() or not root.is_dir():
        raise ArtifactStoreError("artifact scan root is not a regular directory")
    counters = _TreeSizeCounters()
    bounds = _ScanBounds(
        max_entries=max_entries, max_files=max_files, max_bytes=max_bytes
    )
    pending: list[tuple[Path, int]] = [(root, 0)]
    while pending:
        current, depth = pending.pop()
        if depth > _MAX_SCAN_DEPTH:
            raise ArtifactStoreError("artifact scan exceeds its depth bound")
        try:
            iterator = os.scandir(current)
        except OSError as exc:
            raise ArtifactStoreError(
                "artifact scan could not read a directory"
            ) from exc
        with iterator:
            # NOTE: this per-directory bound intentionally uses the module
            # default `_MAX_SCAN_FILES`, not the caller-supplied `max_files`
            # -- preserved verbatim from the pre-refactor behaviour.
            children = 0
            for child in iterator:
                children += 1
                if children > _MAX_SCAN_FILES:
                    raise ArtifactStoreError("artifact scan directory is too large")
                _bounded_tree_size_child(child, depth, counters, pending, bounds)
    return counters.total


def _is_stale_entry(manifest: Mapping[str, Any], cutoff: float) -> bool:
    built_at = manifest.get("built_at")
    try:
        return built_at is None or float(built_at) < cutoff
    except (TypeError, ValueError):
        return True


@dataclass(frozen=True)
class _GCPolicy:
    """Immutable per-run GC decision inputs, bundled to keep helpers <=7 params."""

    recent: set[str]
    cutoff: float
    high_watermark_mib: int
    live_keys: Iterable[str]
    pinned_keys: Iterable[str]
    waited_keys: Iterable[str]
    running_keys: Iterable[str]
    authority_probe: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]


class _GCState:
    """Mutable running result/pressure state for one `garbage_collect` pass."""

    __slots__ = ("pressure", "removed", "kept", "reclaimed")

    def __init__(self, pressure: bool) -> None:
        self.pressure = pressure
        self.removed: list[str] = []
        self.kept: list[str] = []
        self.reclaimed = 0


def _require_matching_stage_identity(
    staged: StagedArtifacts, manifest: Mapping[str, Any]
) -> None:
    if (
        manifest.get("schema") != ARTIFACT_SCHEMA
        or manifest.get("key") != staged.key
        or manifest.get("fence") != staged.fence
        or _manifest_attempt(manifest) != staged.attempt
        or (staged.job_id and manifest.get("job_id") != staged.job_id)
        or (staged.work_item_id and manifest.get("work_item_id") != staged.work_item_id)
    ):
        raise ArtifactStoreError("artifact staging identity does not match the worker")


def _validate_stage_entry_path(entry: Mapping[str, Any], stage_root: Path) -> Path:
    """Validate one manifest artifact entry's declared path; return the resolved Path."""
    staged_at = entry.get("staged_at")
    relative_name = entry.get("relative_path")
    if (
        not isinstance(staged_at, str)
        or not isinstance(relative_name, str)
        or not relative_name
        or Path(relative_name).is_absolute()
        or ".." in Path(relative_name).parts
    ):
        raise ArtifactStoreError("artifact staging path is invalid")
    path = Path(staged_at)
    try:
        _reject_symlink_components(stage_root, path)
        path.resolve(strict=True).relative_to(stage_root)
        expected_path = stage_root / relative_name
        _reject_symlink_components(stage_root, expected_path)
        if path.resolve(strict=True) != expected_path.resolve(strict=True):
            raise ArtifactStoreError(
                "artifact staging path does not match its relative path"
            )
    except (ArtifactStoreError, OSError, ValueError):
        raise ArtifactStoreError("artifact staging path escapes its stage") from None
    return path


def _validate_stage_entry_checksum(path: Path, entry: Mapping[str, Any]) -> None:
    try:
        path_stat = os.stat(path, follow_symlinks=False)
        valid = (
            stat.S_ISREG(path_stat.st_mode)
            and _sha256_file(path, max_bytes=_MAX_STAGE_BYTES) == entry.get("sha256")
            and int(entry.get("bytes", -1)) == path_stat.st_size
        )
    except (ArtifactStoreError, OSError, TypeError, ValueError):
        valid = False
    if not valid:
        raise ArtifactStoreError("artifact staging checksum validation failed")


class _PublishState:
    """Mutable cross-step flag for one `publish()` call's stage cleanup decision."""

    __slots__ = ("cleanup_stage",)

    def __init__(self) -> None:
        self.cleanup_stage = False


def _copy_and_fsync_publish_tree(staged: StagedArtifacts, temporary_dir: Path) -> None:
    _copy_tree_no_follow(staged.stage_dir / "artifacts", temporary_dir / "artifacts")
    _bounded_tree_size(
        temporary_dir / "artifacts",
        max_files=_MAX_STAGE_ARTIFACTS,
        max_bytes=_MAX_STAGE_BYTES,
    )
    for artifact in (temporary_dir / "artifacts").rglob("*"):
        if artifact.is_file():
            _fsync_file(artifact)
    _fsync_directory(temporary_dir / "artifacts")


def _staged_artifacts_at(
    temporary_dir: Path, artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rewrite each artifact entry's ``staged_at`` to the temp publish dir."""
    return [
        {
            key: (
                str(temporary_dir / "artifacts" / entry["relative_path"])
                if key == "staged_at"
                else value
            )
            for key, value in entry.items()
            if key != "staged_at"
        }
        for entry in artifacts
    ]


def _final_artifacts_at(
    final_dir: Path, artifacts: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Rewrite each artifact entry's ``stored_at`` to the final publish dir."""
    return [
        {**entry, "stored_at": str(final_dir / "artifacts" / entry["relative_path"])}
        for entry in artifacts
    ]


def _validate_manifest_schema_key(
    manifest: Mapping[str, Any],
    schema: object,
    expected_key: str | None,
    require_committed: bool,
) -> bool:
    """`validate_manifest` stage 1: schema/expected-key/committed compatibility."""
    if schema == ARTIFACT_SCHEMA:
        if expected_key is not None and manifest.get("key") != expected_key:
            return False
        if require_committed and manifest.get("publication_state") != "committed":
            return False
        return True
    return not (
        expected_key is not None and manifest.get("key") not in {None, expected_key}
    )


def _validate_manifest_entry_fields(
    entry: object, schema: object
) -> tuple[str, str] | None:
    """Validate one entry's stored_at/sha256/bytes-type fields.

    Returns ``(stored_at, checksum)`` on success.
    """
    if not isinstance(entry, Mapping):
        return None
    stored_at = entry.get("stored_at")
    checksum = entry.get("sha256")
    if not isinstance(stored_at, str) or not isinstance(checksum, str):
        return None
    if len(stored_at) > 4096 or len(checksum) > 256:
        return None
    if schema == ARTIFACT_SCHEMA and (
        not isinstance(entry.get("bytes"), int) or isinstance(entry.get("bytes"), bool)
    ):
        return None
    return stored_at, checksum


def _validate_artifact_stat(
    artifact: Path, expected_root: Path | None, entry: Mapping[str, Any]
) -> int | None:
    """Validate that ``artifact`` stays under ``expected_root`` (if any) and
    stats as a regular file whose size matches its declared byte count.
    """
    if expected_root is not None:
        try:
            _reject_symlink_components(expected_root, artifact)
            artifact.resolve(strict=True).relative_to(expected_root)
        except (ArtifactStoreError, OSError, ValueError):
            return None
    try:
        artifact_stat = os.stat(artifact, follow_symlinks=False)
        if not stat.S_ISREG(artifact_stat.st_mode):
            return None
        declared_bytes = int(entry.get("bytes", artifact_stat.st_size))
        if (
            declared_bytes < 0
            or declared_bytes > _MAX_STAGE_BYTES
            or artifact_stat.st_size != declared_bytes
        ):
            return None
    except (OSError, TypeError, ValueError):
        return None
    return declared_bytes


def _validate_manifest_entry_metadata(
    entry: object, schema: object, expected_root: Path | None
) -> tuple[Path, str, int] | None:
    """`validate_manifest` stage 3, per entry: bounded metadata + stat check."""
    fields = _validate_manifest_entry_fields(entry, schema)
    if fields is None:
        return None
    stored_at, checksum = fields
    artifact = Path(stored_at)
    declared_bytes = _validate_artifact_stat(artifact, expected_root, entry)
    if declared_bytes is None:
        return None
    return artifact, checksum, declared_bytes


def _validate_manifest_entries(
    artifacts: list[object], schema: object, expected_root: Path | None
) -> list[tuple[Path, str, int]] | None:
    """`validate_manifest` stage 3: validate every entry, enforcing the
    aggregate byte bound. None means the manifest must fail validation.
    """
    total_bytes = 0
    verified: list[tuple[Path, str, int]] = []
    for entry in artifacts:
        metadata = _validate_manifest_entry_metadata(entry, schema, expected_root)
        if metadata is None:
            return None
        total_bytes += metadata[2]
        if total_bytes > _MAX_STAGE_BYTES:
            return None
        verified.append(metadata)
    return verified


def _verify_manifest_checksums(verified_metadata: list[tuple[Path, str, int]]) -> bool:
    """`validate_manifest` stage 4: hash every artifact already metadata-verified.

    Runs only after every manifest entry has passed the bounded
    metadata/aggregate pass -- a hostile first entry must not consume the
    full per-file budget before a later entry invalidates the manifest.
    """
    for artifact, checksum, declared_bytes in verified_metadata:
        try:
            if _sha256_file(artifact, max_bytes=declared_bytes) != checksum:
                return False
        except (OSError, ArtifactStoreError):
            return False
    return True


class BuildArtifactStore:
    """Content-addressed artifact storage with explicit recovery states.

    ``root`` may be supplied by tests or a host deployment.  The normal
    Repository Manager path is the lane arbitration directory, matching the
    legacy build cache so existing artifacts remain readable during migration.
    """

    def __init__(
        self,
        root: Path | str | None = None,
        *,
        repo_path: Path | str | None = None,
    ) -> None:
        if root is None:
            scope = lane_scope(repo_path)
            root = scope.arbitration_dir / "build-cache"
        self.root = Path(root).expanduser().resolve(strict=False)
        self.root.mkdir(parents=True, exist_ok=True)
        self.staging_root = self.root / ".staging"
        self.quarantine_root = self.root / "quarantine"
        self.staging_root.mkdir(parents=True, exist_ok=True)
        self.quarantine_root.mkdir(parents=True, exist_ok=True)

    def manifest_path(self, key: str) -> Path:
        return self.root / _key_component(key) / MANIFEST_NAME

    def _key_dir(self, key: str) -> Path:
        return self.root / _key_component(key)

    def read_manifest(self, key: str) -> dict[str, Any] | None:
        path = self.manifest_path(key)
        if self._key_dir(key).is_symlink():
            return None
        return _read_bounded_json(path)

    def _resolve_expected_artifacts_root(
        self, manifest: Mapping[str, Any], expected_key: str | None, schema: object
    ) -> tuple[bool, Path | None]:
        """`validate_manifest` stage 2: resolve the root manifest entries must
        stay under. Returns ``(ok, expected_root)``; ``ok=False`` means the
        manifest must fail validation.
        """
        if expected_key is None and schema != ARTIFACT_SCHEMA:
            return True, None
        key = expected_key or manifest.get("key")
        if not isinstance(key, str):
            return False, None
        try:
            key_dir = self._key_dir(key)
            artifacts_dir = key_dir / "artifacts"
            if key_dir.is_symlink() or artifacts_dir.is_symlink():
                return False, None
            return True, artifacts_dir.resolve(strict=True)
        except (ArtifactStoreError, OSError):
            return False, None

    def validate_manifest(
        self,
        manifest: Mapping[str, Any] | None,
        *,
        require_committed: bool = True,
        expected_key: str | None = None,
    ) -> bool:
        """Return true only when every listed byte matches its checksum."""

        if not isinstance(manifest, Mapping):
            return False
        schema = manifest.get("schema")
        if not _validate_manifest_schema_key(
            manifest, schema, expected_key, require_committed
        ):
            return False
        root_ok, expected_root = self._resolve_expected_artifacts_root(
            manifest, expected_key, schema
        )
        if not root_ok:
            return False
        artifacts = manifest.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(artifacts) > _MAX_MANIFEST_ARTIFACTS
        ):
            return False
        verified_metadata = _validate_manifest_entries(artifacts, schema, expected_root)
        if verified_metadata is None:
            return False
        return _verify_manifest_checksums(verified_metadata)

    def quarantine(self, key: str, *, reason: str = "invalid-manifest") -> Path | None:
        """Move a corrupt/incomplete entry out of the hit namespace."""

        source = self._key_dir(key)
        if not source.exists():
            return None
        target = self.quarantine_root / (
            f"{_safe(key)}-{int(time.time() * 1_000_000)}-{_safe(reason)}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(source, target)
        except OSError as exc:
            raise ArtifactStoreError(f"could not quarantine artifact {key!r}") from exc
        return target

    def quarantine_if(
        self,
        key: str,
        *,
        authority_check: Callable[[Mapping[str, Any]], Mapping[str, Any]],
        reason: str = "stale-publication",
    ) -> Path | None:
        """Quarantine only after an exact durable stale-owner proof."""

        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            lock_exclusive(lock.fileno())
            try:
                manifest = self.read_manifest(key)
                if manifest is None:
                    return None
                try:
                    proof = authority_check(manifest)
                except Exception:
                    return None
                if not _exact_authority_proof(manifest, proof, marker="stale"):
                    return None
                return self.quarantine(key, reason=reason)
            finally:
                unlock(lock.fileno())

    def stage(
        self,
        build_tree: Path | str,
        *,
        workdir: str,
        patterns: Iterable[str],
        key: str,
        attempt: int,
        fence: str,
        job_id: str = "",
        work_item_id: str = "",
        generation_id: str | None = None,
        max_artifacts: int | None = _MAX_STAGE_ARTIFACTS,
        max_bytes: int | None = _MAX_STAGE_BYTES,
    ) -> StagedArtifacts:
        """Copy and checksum declared outputs into a private stage directory."""

        if attempt < 1 or not fence.strip():
            raise ArtifactStoreError(
                "artifact staging requires a positive attempt and fence"
            )
        if not job_id.strip() or not work_item_id.strip():
            raise ArtifactStoreError(
                "artifact staging requires exact job and WorkItem identities"
            )
        raw_tree = Path(build_tree).expanduser()
        _reject_symlink_path(raw_tree)
        tree = raw_tree.resolve(strict=True)
        patterns = tuple(patterns)
        if not patterns:
            raise ArtifactStoreError("artifact staging requires at least one pattern")
        if Path(workdir).is_absolute() or ".." in Path(workdir).parts:
            raise ArtifactStoreError("artifact workdir must stay below the build tree")
        raw_output_root = raw_tree / workdir
        _reject_symlink_path(raw_output_root)
        output_root = raw_output_root.resolve(strict=True)
        if output_root != tree and tree not in output_root.parents:
            raise ArtifactStoreError("artifact workdir escapes the build tree")
        _reject_symlink_components(tree, output_root)
        for pattern in patterns:
            if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
                raise ArtifactStoreError(
                    "artifact patterns must be relative and stay below workdir"
                )
        key = _key_component(key)
        stage_dir = (
            self.staging_root
            / _safe(key)
            / (f"attempt-{attempt}-{_safe(fence)}-{uuid.uuid4().hex}")
        )
        artifact_dir = stage_dir / "artifacts"
        try:
            artifact_dir.mkdir(parents=True, exist_ok=False)
        except OSError:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        entries: list[dict[str, Any]] = []
        seen: set[str] = set()
        total_bytes = 0
        admitted_max_artifacts = (
            _MAX_STAGE_ARTIFACTS
            if max_artifacts is None
            else min(max_artifacts, _MAX_STAGE_ARTIFACTS)
        )
        admitted_max_bytes = (
            _MAX_STAGE_BYTES if max_bytes is None else min(max_bytes, _MAX_STAGE_BYTES)
        )
        try:
            matches_by_pattern = _bounded_matching_files(
                output_root,
                patterns,
                max_entries=_MAX_SCAN_ENTRIES,
                max_files=admitted_max_artifacts,
                max_bytes=admitted_max_bytes,
            )
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        for pattern in patterns:
            matches = matches_by_pattern[pattern]
            if not matches:
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise ArtifactStoreError(
                    f"build declared artifact pattern {pattern!r} but produced no file"
                )
            if len(seen) + len(matches) > admitted_max_artifacts:
                shutil.rmtree(stage_dir, ignore_errors=True)
                raise ArtifactStoreError(
                    "build produced more artifacts than the admitted staging bound"
                )
            for source in matches:
                try:
                    _reject_symlink_components(output_root, source)
                    if source.is_symlink():
                        raise ArtifactStoreError(
                            f"symlink artifact source is not allowed: {source}"
                        )
                    try:
                        source.resolve(strict=True).relative_to(output_root)
                    except (OSError, ValueError) as exc:
                        raise ArtifactStoreError(
                            "artifact source escapes its declared output root"
                        ) from exc
                    relative = source.relative_to(output_root)
                    relative_name = str(relative)
                    if relative_name in seen:
                        continue
                    seen.add(relative_name)
                    remaining_bytes = admitted_max_bytes - total_bytes
                    if remaining_bytes < 0:
                        raise ArtifactStoreError(
                            "build artifacts exceed the admitted staging byte bound"
                        )
                    destination = artifact_dir / relative
                    copied_bytes = _copy_file_no_follow(
                        source,
                        destination,
                        max_bytes=remaining_bytes,
                    )
                    _fsync_file(destination)
                    total_bytes += copied_bytes
                    entries.append(
                        {
                            "pattern": pattern,
                            "relative_path": relative_name,
                            "staged_at": str(destination),
                            "sha256": _sha256_file(
                                destination, max_bytes=_MAX_STAGE_BYTES
                            ),
                            "bytes": copied_bytes,
                        }
                    )
                except Exception:
                    shutil.rmtree(stage_dir, ignore_errors=True)
                    raise
        manifest: dict[str, Any] = {
            "schema": ARTIFACT_SCHEMA,
            "key": key,
            "job_id": job_id,
            "work_item_id": work_item_id,
            "fence": fence,
            "attempt": attempt,
            "generation_id": generation_id,
            "publication_state": "staged",
            "artifacts": entries,
        }
        try:
            _atomic_write_json(stage_dir / MANIFEST_NAME, manifest)
            _fsync_directory(artifact_dir)
            _fsync_directory(stage_dir)
        except Exception:
            shutil.rmtree(stage_dir, ignore_errors=True)
            raise
        return StagedArtifacts(
            key=key,
            stage_dir=stage_dir,
            manifest=manifest,
            fence=fence,
            attempt=attempt,
            job_id=job_id,
            work_item_id=work_item_id,
        )

    def _lock_path(self, key: str) -> Path:
        return self.root / f".{_safe(key)}.publish.lock"

    def discard_stage(self, staged: StagedArtifacts) -> bool:
        """Remove a stage after final publication; refuse arbitrary paths."""

        stage_dir = Path(staged.stage_dir)
        if not stage_dir.exists():
            return True
        try:
            _reject_symlink_path(stage_dir)
            stage_dir.resolve(strict=True).relative_to(
                self.staging_root.resolve(strict=True)
            )
        except (ArtifactStoreError, OSError, ValueError):
            return False
        if stage_dir == self.staging_root or stage_dir.parent == self.staging_root:
            return False
        try:
            shutil.rmtree(stage_dir)
            parent = stage_dir.parent
            if parent != self.staging_root and not any(parent.iterdir()):
                parent.rmdir()
        except OSError:
            return False
        return not stage_dir.exists()

    def reconcile_staging(
        self,
        *,
        max_age_seconds: int = _DEFAULT_STALE_STAGE_SECONDS,
        max_entries: int = _MAX_SCAN_ENTRIES,
        authority_probe: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Boundedly remove old stages only with exact authority proof.

        Without ``authority_probe`` this is deliberately report-only: mtime
        alone cannot distinguish a crashed producer from a long-running one.
        """

        if max_age_seconds < 0 or max_entries < 1:
            raise ValueError("staging reconciliation bounds must be non-negative")
        cutoff = time.time() - max_age_seconds
        removed: list[str] = []
        kept: list[str] = []
        errors: list[str] = []
        scanned = 0
        try:
            key_iterator = os.scandir(self.staging_root)
        except OSError as exc:
            return {"removed": [], "kept": [], "errors": [str(exc)]}
        with key_iterator:
            for key_entry in key_iterator:
                scanned += 1
                if scanned > max_entries:
                    errors.append("staging entry scan exceeds its bound")
                    break
                if key_entry.is_symlink() or not key_entry.is_dir(
                    follow_symlinks=False
                ):
                    kept.append(key_entry.name)
                    continue
                try:
                    stage_iterator = os.scandir(key_entry.path)
                except OSError as exc:
                    errors.append(f"{key_entry.name}: {exc}")
                    continue
                with stage_iterator:
                    for stage_entry in stage_iterator:
                        scanned += 1
                        if scanned > max_entries:
                            errors.append("staging entry scan exceeds its bound")
                            break
                        if stage_entry.is_symlink() or not stage_entry.is_dir(
                            follow_symlinks=False
                        ):
                            kept.append(f"{key_entry.name}/{stage_entry.name}")
                            continue
                        try:
                            old = (
                                stage_entry.stat(follow_symlinks=False).st_mtime
                                < cutoff
                            )
                        except OSError as exc:
                            errors.append(f"{stage_entry.name}: {exc}")
                            continue
                        stage_manifest = _read_bounded_json(
                            Path(stage_entry.path) / MANIFEST_NAME
                        )
                        if not old or authority_probe is None or stage_manifest is None:
                            kept.append(f"{key_entry.name}/{stage_entry.name}")
                            continue
                        try:
                            proof = authority_probe(stage_manifest)
                        except Exception as exc:
                            errors.append(f"{stage_entry.name}: {exc}")
                            kept.append(f"{key_entry.name}/{stage_entry.name}")
                            continue
                        if not _exact_authority_proof(
                            stage_manifest, proof, marker="stale"
                        ):
                            kept.append(f"{key_entry.name}/{stage_entry.name}")
                            continue
                        stage_key = str(stage_manifest.get("key") or "")
                        stage_fence = str(stage_manifest.get("fence") or "")
                        stage_job = str(stage_manifest.get("job_id") or "")
                        stage_work = str(stage_manifest.get("work_item_id") or "")
                        try:
                            stage_attempt = int(stage_manifest.get("attempt", 0))
                        except (TypeError, ValueError):
                            stage_attempt = 0
                        if (
                            not stage_key
                            or not stage_fence
                            or not stage_job
                            or not stage_work
                            or stage_attempt < 1
                            or _safe(stage_key) != key_entry.name
                        ):
                            kept.append(f"{key_entry.name}/{stage_entry.name}")
                            continue
                        stage = StagedArtifacts(
                            key=stage_key,
                            stage_dir=Path(stage_entry.path),
                            manifest=stage_manifest,
                            fence=stage_fence,
                            attempt=stage_attempt,
                            job_id=stage_job,
                            work_item_id=stage_work,
                        )
                        if self.discard_stage(stage):
                            removed.append(f"{key_entry.name}/{stage_entry.name}")
                        else:
                            kept.append(f"{key_entry.name}/{stage_entry.name}")
        return {"removed": removed, "kept": kept, "errors": errors}

    def _reconcile_existing_publication(
        self, staged: StagedArtifacts, stage_manifest: Mapping[str, Any]
    ) -> dict[str, Any] | None:
        """Handle a pre-existing manifest at this key.

        Returns it if this call is an exact fenced retry of an
        already-published/committed manifest; otherwise quarantines a
        corrupt same-attempt manifest (if any) and returns None so the
        caller proceeds with a fresh publish.
        """
        existing = self.read_manifest(staged.key)
        if not existing:
            return None
        if not self._same_publication_identity(existing, stage_manifest):
            raise ArtifactStoreError(
                "artifact key already belongs to another job, WorkItem, "
                "attempt, or fence"
            )
        if self.validate_manifest(
            existing, require_committed=False, expected_key=staged.key
        ) and existing.get("publication_state") in {"published", "committed"}:
            return existing
        self.quarantine(staged.key, reason="same-attempt-corrupt")
        return None

    def _quarantine_orphan_final_dir_if_present(
        self, key: str, final_dir: Path
    ) -> None:
        # A crash after the directory rename but before manifest write
        # leaves an orphan final directory.  Quarantine it before the next
        # atomic publish so os.replace cannot fail on a non-empty
        # destination.
        if not final_dir.exists():
            return
        if self.read_manifest(key) is None:
            self.quarantine(key, reason="orphan-final-directory")
        else:
            raise ArtifactStoreError(
                "artifact key appeared while publication was fenced"
            )

    def _commit_publish(
        self,
        staged: StagedArtifacts,
        stage_manifest: Mapping[str, Any],
        final_dir: Path,
        temporary_dir: Path,
        fence_check: Callable[[], bool] | None,
        state: _PublishState,
    ) -> dict[str, Any]:
        """Copy the stage tree into place and atomically publish its manifest."""
        _copy_and_fsync_publish_tree(staged, temporary_dir)
        published = dict(stage_manifest)
        published["publication_state"] = "published"
        published["published_at"] = time.time()
        published["artifacts"] = _staged_artifacts_at(
            temporary_dir, stage_manifest["artifacts"]
        )
        # The directory move and manifest write are separate filesystem
        # operations; a manifest is never written until all bytes are
        # present and checksummed.  A crash leaves an unreferenced temp
        # directory, not a false hit.
        os.replace(temporary_dir, final_dir)
        state.cleanup_stage = True
        _fsync_directory(self.root)
        published["artifacts"] = _final_artifacts_at(final_dir, published["artifacts"])
        _atomic_write_json(final_dir / MANIFEST_NAME, published)
        # If the lease expires after the manifest becomes durable, surface
        # the stale publication.  The worker may quarantine it only after
        # the durable WorkItem authority proves this exact owner/fence is
        # stale.
        self._require_fence(fence_check)
        return published

    def publish(
        self,
        staged: StagedArtifacts,
        *,
        fence_check: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        """Atomically publish a verified manifest under the current fence."""

        self._require_fence(fence_check)
        stage_manifest = self.read_stage_manifest(staged)
        self._validate_stage(staged, stage_manifest)
        # The lock below is intentionally limited to the artifact key.  It is
        # not a job/resource authority; WorkItem fence validation remains the
        # authorization boundary.
        lock_path = self._lock_path(staged.key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            lock_exclusive(lock.fileno())
            state = _PublishState()
            try:
                # The key lock may have waited behind another producer.  A
                # lease that was valid before flock is not authorization for
                # an exact retry after that wait.
                self._require_fence(fence_check)
                existing = self._reconcile_existing_publication(staged, stage_manifest)
                if existing is not None:
                    # This is an exact, fenced retry.  It is not a generic
                    # cache hit: the caller still has to prove terminal
                    # WorkItem state before finalize.
                    state.cleanup_stage = True
                    return existing
                self._require_fence(fence_check)
                final_dir = self.root / staged.key
                temporary_dir = self.root / (
                    f".{_safe(staged.key)}.publish-{uuid.uuid4().hex}"
                )
                self._quarantine_orphan_final_dir_if_present(staged.key, final_dir)
                try:
                    return self._commit_publish(
                        staged,
                        stage_manifest,
                        final_dir,
                        temporary_dir,
                        fence_check,
                        state,
                    )
                finally:
                    shutil.rmtree(temporary_dir, ignore_errors=True)
            finally:
                if state.cleanup_stage:
                    self.discard_stage(staged)
                unlock(lock.fileno())

    def finalize(
        self,
        key: str,
        *,
        fence: str,
        fence_check: Callable[[], bool] | None = None,
        terminal_check: Callable[[], bool] | None = None,
        job_id: str = "",
        work_item_id: str = "",
        attempt: int | None = None,
    ) -> dict[str, Any]:
        """Mark a published manifest committed after WorkItem terminal commit."""

        self._require_fence(fence_check)
        _require_finalize_preconditions(terminal_check, job_id, work_item_id, attempt)
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            lock_exclusive(lock.fileno())
            try:
                # A terminal proof and fence observation made before waiting
                # for the key lock cannot authorize a later replacement.
                self._require_fence(fence_check)
                manifest = _require_matching_manifest_identity(
                    self.read_manifest(key), key, fence, job_id, work_item_id, attempt
                )
                if not self.validate_manifest(
                    manifest, require_committed=False, expected_key=key
                ):
                    raise ArtifactStoreError(
                        f"published artifact {key!r} failed checksum validation"
                    )
                _require_finalize_terminal_state(terminal_check)
                if manifest.get("publication_state") == "committed":
                    return manifest
                manifest["publication_state"] = "committed"
                manifest["committed_at"] = time.time()
                _atomic_write_json(self.manifest_path(key), manifest)
                return manifest
            finally:
                unlock(lock.fileno())

    def read_stage_manifest(self, staged: StagedArtifacts) -> dict[str, Any]:
        manifest = self._read_path(staged.stage_dir / MANIFEST_NAME)
        if manifest is None:
            raise ArtifactStoreError("artifact staging manifest is missing")
        return manifest

    def _validate_stage(
        self, staged: StagedArtifacts, manifest: Mapping[str, Any]
    ) -> None:
        _require_matching_stage_identity(staged, manifest)
        _reject_symlink_path(staged.stage_dir)
        stage_root = (staged.stage_dir / "artifacts").resolve(strict=True)
        _bounded_tree_size(
            stage_root,
            max_files=_MAX_STAGE_ARTIFACTS,
            max_bytes=_MAX_STAGE_BYTES,
        )
        artifacts = manifest.get("artifacts")
        if (
            not isinstance(artifacts, list)
            or not artifacts
            or len(artifacts) > _MAX_MANIFEST_ARTIFACTS
        ):
            raise ArtifactStoreError("artifact staging manifest has no artifacts")
        for entry in artifacts:
            path = _validate_stage_entry_path(entry, stage_root)
            _validate_stage_entry_checksum(path, entry)

    @staticmethod
    def _same_publication_identity(
        existing: Mapping[str, Any], staged: Mapping[str, Any]
    ) -> bool:
        fields = ("schema", "key", "job_id", "work_item_id", "attempt", "fence")
        return all(existing.get(field) == staged.get(field) for field in fields)

    @staticmethod
    def _read_path(path: Path) -> dict[str, Any] | None:
        return _read_bounded_json(path)

    @staticmethod
    def _require_fence(check: Callable[[], bool] | None) -> None:
        if check is not None:
            try:
                current = check()
            except Exception as exc:  # pragma: no cover - defensive authority boundary
                raise ArtifactFenceLost("WorkItem fence could not be verified") from exc
            if not current:
                raise ArtifactFenceLost("WorkItem fence is no longer current")

    def iter_entries(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        """Return bounded, manifest-bearing entries for GC/status callers."""

        directories: list[Path] = []
        try:
            iterator = self.root.iterdir()
            scanned = 0
            for directory in iterator:
                scanned += 1
                if scanned > _MAX_SCAN_ENTRIES:
                    raise ArtifactStoreError("artifact entry scan exceeds its bound")
                if (
                    directory.is_symlink()
                    or not directory.is_dir()
                    or directory.name.startswith(".")
                ):
                    continue
                directories.append(directory)
        except OSError as exc:
            raise ArtifactStoreError(
                "artifact entry scan could not read the cache"
            ) from exc
        entries: list[tuple[str, dict[str, Any]]] = []
        for directory in sorted(directories):
            manifest = self.read_manifest(directory.name)
            if manifest is not None:
                entries.append((directory.name, manifest))
        return tuple(entries)

    def _authorize_entry_removal(
        self,
        key: str,
        directory: Path,
        authority_probe: Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None,
    ) -> int | None:
        """Return the entry's byte size if removal is durably authorized, else None."""
        if directory.is_symlink() or not directory.is_dir():
            return None
        manifest = self.read_manifest(key)
        if manifest is None or authority_probe is None:
            return None
        try:
            probe = authority_probe(key, manifest)
        except Exception:
            return None
        if not _exact_authority_proof(manifest, probe, marker="safe_to_remove"):
            return None
        if any(
            bool(probe.get(name)) for name in ("live", "pinned", "waited", "running")
        ):
            return None
        try:
            return _bounded_tree_size(directory)
        except ArtifactStoreError:
            return None

    def remove_entry(
        self,
        key: str,
        *,
        live_keys: Iterable[str] = (),
        pinned_keys: Iterable[str] = (),
        waited_keys: Iterable[str] = (),
        running_keys: Iterable[str] = (),
        authority_probe: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
    ) -> int:
        """Remove one entry only after a fail-closed durable liveness probe."""

        protected = (
            set(live_keys) | set(pinned_keys) | set(waited_keys) | set(running_keys)
        )
        if key in protected:
            return 0
        lock_path = self._lock_path(key)
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+") as lock:
            lock_exclusive(lock.fileno())
            try:
                directory = self._key_dir(key)
                total = self._authorize_entry_removal(key, directory, authority_probe)
                if total is None:
                    return 0
                try:
                    shutil.rmtree(directory)
                except OSError as exc:
                    raise ArtifactStoreError(
                        f"could not remove artifact entry {key!r}"
                    ) from exc
                return total
            finally:
                unlock(lock.fileno())

    def _gc_process_entry(
        self,
        key: str,
        manifest: Mapping[str, Any],
        policy: _GCPolicy,
        state: _GCState,
    ) -> None:
        """Apply the keep/remove decision for one GC candidate, mutating ``state``."""
        if key in policy.recent:
            state.kept.append(key)
            return
        if not state.pressure and not _is_stale_entry(manifest, policy.cutoff):
            state.kept.append(key)
            return
        gained = self.remove_entry(
            key,
            live_keys=policy.live_keys,
            pinned_keys=policy.pinned_keys,
            waited_keys=policy.waited_keys,
            running_keys=policy.running_keys,
            authority_probe=policy.authority_probe,
        )
        if not gained:
            state.kept.append(key)
            return
        state.removed.append(key)
        state.reclaimed += gained
        usage = shutil.disk_usage(self.root)
        if (
            state.pressure
            and policy.high_watermark_mib > 0
            and usage.free >= policy.high_watermark_mib * 1024 * 1024
        ):
            state.pressure = False

    def garbage_collect(
        self,
        *,
        keep_recent: int = 10,
        max_age_days: int = 14,
        low_watermark_mib: int = 0,
        high_watermark_mib: int = 0,
        authority_probe: Callable[[str, Mapping[str, Any]], Mapping[str, Any]]
        | None = None,
        live_keys: Iterable[str] = (),
        pinned_keys: Iterable[str] = (),
        waited_keys: Iterable[str] = (),
        running_keys: Iterable[str] = (),
    ) -> dict[str, Any]:
        """Reclaim old entries, or reclaim to the high watermark under pressure."""

        if keep_recent < 0 or max_age_days < 0:
            raise ValueError("GC bounds must be non-negative")
        entries = list(self.iter_entries())
        if not authority_probe:
            return {
                "removed": [],
                "kept": [key for key, _ in entries],
                "reclaimed_bytes": 0,
            }
        entries.sort(key=lambda pair: str(pair[1].get("built_at") or ""), reverse=True)
        recent = {key for key, _ in entries[:keep_recent]}
        cutoff = time.time() - max_age_days * 86400
        usage = shutil.disk_usage(self.root)
        pressure = (
            low_watermark_mib > 0 and usage.free < low_watermark_mib * 1024 * 1024
        )

        policy = _GCPolicy(
            recent=recent,
            cutoff=cutoff,
            high_watermark_mib=high_watermark_mib,
            live_keys=live_keys,
            pinned_keys=pinned_keys,
            waited_keys=waited_keys,
            running_keys=running_keys,
            authority_probe=authority_probe,
        )
        state = _GCState(pressure=pressure)
        for key, manifest in reversed(entries):
            self._gc_process_entry(key, manifest, policy, state)
        return {
            "removed": state.removed,
            "kept": state.kept,
            "reclaimed_bytes": state.reclaimed,
        }

    gc = garbage_collect


__all__ = [
    "ARTIFACT_SCHEMA",
    "ArtifactFenceLost",
    "ArtifactStoreError",
    "BuildArtifactStore",
    "MANIFEST_NAME",
    "StagedArtifacts",
]
