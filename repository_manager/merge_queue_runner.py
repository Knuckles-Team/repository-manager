#!/usr/bin/env python3
"""Drain every declared repository's merge queue.

The runner is deliberately a small process boundary around the existing
repo-scoped merge-queue implementation.  Repository selection belongs to the
canonical workspace manifest; filesystem walking is not an authority.  The
same rule keeps the installed timer useful when the workspace grows: adding a
repository to ``workspace.yml`` is enough to put it in the next drain, while a
clone that is merely present on disk cannot accidentally become eligible.
"""

from __future__ import annotations

import argparse
import copy
import os
import shutil
import signal
import subprocess  # nosec B404 - all commands are fixed argv
import sys
import tempfile
import threading
import time
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from repository_manager.resource_guard import default_guard_paths, process_start_time
from repository_manager.workspace_manifest import (
    WorkspaceManifestError,
    select_repositories,
)

# The queue timer owns only the fast, no-push drain.  Keep its per-repository
# wall-clock budget small enough that a slow or wedged fast gate cannot occupy
# the timer for hours. This is independent from gate worker/resource caps.
DEFAULT_DRAIN_DEADLINE_SECONDS = 180
# One invocation is bounded separately from every child. Twenty fast roots at
# the documented 180-second budget fit within this ceiling; a larger selection
# is admitted only until the global deadline and is reported as deferred.
DEFAULT_GLOBAL_DRAIN_DEADLINE_SECONDS = 3600
SYSTEMD_TIMEOUT_MARGIN_SECONDS = 60
DEFAULT_HEAVY_TIMEOUT_SECONDS = 18000
DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS = 14400
DEFAULT_LEASE_SAFETY_MARGIN_SECONDS = 600
MAX_LEASE_TTL_SECONDS = 86400
DEFAULT_MAX_QUEUE_AGE_SECONDS = 86400
QUEUE_DIRECTORY = Path("agent-lanes") / "merge-queue"
CANONICAL_FRAGMENT = "canonical.yaml"
QUEUED_STATE = "queued"
TERMINAL_STATES = frozenset(
    {"aborted", "cancelled", "done", "failed", "landed", "rejected", "withdrawn"}
)
KNOWN_STATES = TERMINAL_STATES | {QUEUED_STATE}

# This is a semantic exclusion, not a repository inventory.  The canonical
# manifest explicitly classifies open-source-libraries as reference inputs;
# those inputs must never become merge-queue targets even if a future manifest
# projection happens to describe one of them.
EXCLUDED_MANIFEST_COMPONENTS = frozenset({"open-source-libraries"})
_WORKSPACE_ROOT_REFERENCE = "${AGENT_UTILITIES_WORKSPACE_ROOT}"


class MergeQueueRunnerError(RuntimeError):
    """A manifest, queue store, or runner execution failed closed."""


@dataclass(frozen=True)
class DeclaredRepository:
    """One repository selected by the canonical workspace manifest."""

    identifier: str
    name: str
    path: Path


@dataclass(frozen=True)
class QueueRecord:
    """The minimal lifecycle view required for runner discovery."""

    candidate_id: str
    state: str
    enqueued_at: datetime
    recorded_at: datetime | None
    worktree: str
    source: Path


@dataclass(frozen=True)
class DrainResult:
    """One child invocation's observable result."""

    repository: DeclaredRepository
    returncode: int
    output: str

    @property
    def deferred(self) -> bool:
        """Whether the child reported the repo-scoped lease as busy."""

        return self.returncode == 75


def _positive_integer(value: str, *, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise MergeQueueRunnerError(f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise MergeQueueRunnerError(f"{name} must be a positive integer")
    return parsed


def _lease_ttl(deadline_seconds: int, value: int | None, *, env_name: str) -> int:
    """Require the repo lease to outlive the supervisor deadline."""

    if value is None:
        configured = os.environ.get(env_name)
        ttl = (
            _positive_integer(configured, name=env_name)
            if configured is not None
            else deadline_seconds + DEFAULT_LEASE_SAFETY_MARGIN_SECONDS
        )
    else:
        ttl = _positive_integer(str(value), name=env_name)
    if ttl <= deadline_seconds:
        raise MergeQueueRunnerError(
            f"{env_name} must exceed drain deadline ({deadline_seconds}s)"
        )
    if ttl > MAX_LEASE_TTL_SECONDS:
        raise MergeQueueRunnerError(f"{env_name} must be <= {MAX_LEASE_TTL_SECONDS}s")
    return ttl


def _timestamp(value: object, *, field: str, source: Path) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise MergeQueueRunnerError(
            f"{source}: queued record requires a non-blank {field} timestamp"
        )
    candidate = value.strip()
    if candidate.endswith("Z"):
        candidate = f"{candidate[:-1]}+00:00"
    try:
        parsed = datetime.fromisoformat(candidate)
    except ValueError as exc:
        raise MergeQueueRunnerError(
            f"{source}: invalid {field} timestamp {value!r}"
        ) from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise MergeQueueRunnerError(
            f"{source}: {field} timestamp must include a timezone"
        )
    return parsed.astimezone(UTC)


def _optional_timestamp(value: object, *, field: str, source: Path) -> datetime | None:
    """Parse a lifecycle timestamp while retaining pre-field legacy records."""

    if value is None or (isinstance(value, str) and not value.strip()):
        return None
    return _timestamp(value, field=field, source=source)


def _load_yaml(path: Path) -> Any:
    try:
        content = path.read_text(encoding="utf-8")
        return yaml.safe_load(content)
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        raise MergeQueueRunnerError(f"cannot read YAML from {path}: {exc}") from exc


def _manifest_workspace_root(
    data: dict[str, Any], *, configured_root: Path, source: Path
) -> Path:
    value = data.get("path")
    if not isinstance(value, str) or not value.strip():
        raise MergeQueueRunnerError(f"{source}: manifest path must be absolute")
    expanded = value.strip()
    if expanded in {_WORKSPACE_ROOT_REFERENCE, "$AGENT_UTILITIES_WORKSPACE_ROOT"}:
        expanded = str(configured_root)
    elif "${" in expanded or "$AGENT_UTILITIES_WORKSPACE_ROOT" in expanded:
        raise MergeQueueRunnerError(
            f"{source}: manifest path contains an unresolved environment reference"
        )
    manifest_root = Path(expanded).expanduser()
    if not manifest_root.is_absolute():
        raise MergeQueueRunnerError(f"{source}: manifest path must be absolute")
    try:
        manifest_root = manifest_root.resolve(strict=True)
        expected_root = configured_root.resolve(strict=True)
    except OSError as exc:
        raise MergeQueueRunnerError(
            f"{source}: cannot resolve the canonical workspace root: {exc}"
        ) from exc
    if manifest_root != expected_root:
        raise MergeQueueRunnerError(
            f"{source}: manifest path {manifest_root} disagrees with configured "
            f"workspace root {expected_root}; manifest drift is not ignored"
        )
    return expected_root


def _reject_symlink_components(root: Path, target: Path, *, identifier: str) -> None:
    current = root
    for component in target.relative_to(root).parts:
        current /= component
        if current.is_symlink():
            raise MergeQueueRunnerError(
                f"manifest repository {identifier} uses a symbolic-link path: {current}"
            )


def _configured_workspace_root(workspace_root: str | Path) -> Path:
    """Resolve the configured root without inventing a fallback inventory."""

    configured_root = Path(workspace_root).expanduser().resolve(strict=False)
    if not configured_root.is_dir():
        raise MergeQueueRunnerError(
            f"configured workspace root is missing or not a directory: {configured_root}"
        )
    return configured_root


def _manifest_path(configured_root: Path, manifest: str | Path | None) -> Path:
    """Resolve the canonical manifest and reject links or missing paths."""

    candidate = (
        Path(manifest).expanduser()
        if manifest is not None
        else configured_root / "workspace.yml"
    )
    if candidate.is_symlink() or not candidate.is_file():
        raise MergeQueueRunnerError(
            f"canonical workspace manifest is missing or invalid: {candidate}"
        )
    return candidate.resolve()


def _manifest_mapping(path: Path) -> dict[str, Any]:
    """Load one canonical manifest mapping."""

    data = _load_yaml(path)
    if not isinstance(data, dict):
        raise MergeQueueRunnerError(f"{path}: manifest must be a mapping")
    return data


def _manifest_identifiers(data: dict[str, Any], path: Path) -> tuple[str, ...]:
    """Use the shared manifest resolver and preserve its validation errors."""

    try:
        identifiers, _, _ = select_repositories(data)
    except WorkspaceManifestError as exc:
        raise MergeQueueRunnerError(
            f"{path}: invalid repository declaration: {exc}"
        ) from exc
    return tuple(identifiers)


def _validate_declared_repository(
    identifier: str, *, root: Path
) -> DeclaredRepository | None:
    """Validate one manifest entry, or return ``None`` for reference inputs."""

    if frozenset(identifier.split("/")) & EXCLUDED_MANIFEST_COMPONENTS:
        return None
    target = root.joinpath(*identifier.split("/"))
    _reject_symlink_components(root, target, identifier=identifier)
    if not target.is_dir():
        raise MergeQueueRunnerError(
            f"manifest repository {identifier!r} is missing: {target}"
        )
    if not (target / ".git").exists():
        raise MergeQueueRunnerError(
            f"manifest repository {identifier!r} is not a Git work tree: {target}"
        )
    top_level = Path(
        _git_output(
            ["-C", str(target), "rev-parse", "--show-toplevel"],
            cwd=root,
            label=f"manifest repository {identifier!r}",
        )
    ).resolve()
    if top_level != target.resolve():
        raise MergeQueueRunnerError(
            f"manifest repository {identifier!r} resolves to {top_level}, "
            f"not its declared path {target.resolve()}"
        )
    return DeclaredRepository(
        identifier, identifier.rsplit("/", 1)[-1], target.resolve()
    )


def _validated_repositories(
    identifiers: Iterable[str], *, root: Path, manifest_path: Path
) -> tuple[DeclaredRepository, ...]:
    """Validate all manifest entries while retaining duplicate detection."""

    seen: set[str] = set()
    result: list[DeclaredRepository] = []
    for identifier in identifiers:
        if identifier in seen:
            raise MergeQueueRunnerError(
                f"{manifest_path}: duplicate repository identifier {identifier!r}"
            )
        seen.add(identifier)
        repository = _validate_declared_repository(identifier, root=root)
        if repository is not None:
            result.append(repository)
    return tuple(result)


def _git_output(args: list[str], *, cwd: Path, label: str) -> str:
    git_executable = shutil.which("git")
    if git_executable is None:
        raise MergeQueueRunnerError(f"{label}: git is unavailable on PATH")
    try:
        result = subprocess.run(
            [git_executable, *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MergeQueueRunnerError(f"{label}: git could not run: {exc}") from exc
    if result.returncode != 0 or not result.stdout.strip():
        detail = (result.stderr or result.stdout).strip()[:500]
        raise MergeQueueRunnerError(f"{label}: git refused the repository: {detail}")
    return result.stdout.strip()


def declared_repositories(
    workspace_root: str | Path, *, manifest: str | Path | None = None
) -> tuple[DeclaredRepository, ...]:
    """Resolve and validate every repository declared by ``workspace.yml``.

    The manifest is the only selection authority.  Every non-excluded entry is
    required to exist as the exact declared path and to be a Git work tree;
    missing clones and path drift are errors, not silently skipped roots.
    """

    configured_root = _configured_workspace_root(workspace_root)
    manifest_path = _manifest_path(configured_root, manifest)
    data = _manifest_mapping(manifest_path)
    root = _manifest_workspace_root(
        data, configured_root=configured_root, source=manifest_path
    )
    identifiers = _manifest_identifiers(data, manifest_path)
    return _validated_repositories(identifiers, root=root, manifest_path=manifest_path)


def _git_common_directory_for_path(path: Path, *, label: str) -> Path:
    """Resolve Git's common directory for a registered worktree path."""

    raw = _git_output(
        ["-C", str(path), "rev-parse", "--git-common-dir"],
        cwd=path,
        label=label,
    )
    common = Path(raw)
    if not common.is_absolute():
        common = (path / common).resolve()
    else:
        common = common.resolve()
    if not common.is_dir():
        raise MergeQueueRunnerError(
            f"{label} has no readable git common directory: {common}"
        )
    return common


def _git_common_directory(repository: DeclaredRepository) -> Path:
    """Resolve the declared repository's Git common directory."""

    return _git_common_directory_for_path(
        repository.path, label=f"repository {repository.identifier!r}"
    )


def _fragment_records(path: Path) -> list[dict[str, Any]]:
    value = _load_yaml(path)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise MergeQueueRunnerError(f"queue fragment {path} is not a list of mappings")
    return value


def _latest_record(records: list[QueueRecord]) -> QueueRecord:
    """Fold one candidate's records without fabricating legacy timestamps."""

    timestamped = [record for record in records if record.recorded_at is not None]
    if not timestamped:
        return records[-1]
    return max(
        timestamped,
        key=lambda record: (
            record.recorded_at,
            record.source.name == CANONICAL_FRAGMENT,
        ),
    )


def _record_from_mapping(
    mapping: dict[str, Any], *, source: Path, index: int
) -> QueueRecord | None:
    # RMDD-12 generation/candidate snapshots share this directory but are not
    # legacy merge candidates.  The actual drain owns their separate fold.
    if mapping.get("kind") in {"candidate_snapshot", "generation"}:
        return None
    candidate_id = mapping.get("id")
    if not isinstance(candidate_id, str) or not candidate_id.strip():
        raise MergeQueueRunnerError(
            f"queue fragment {source} record {index} requires a non-blank id"
        )
    state_value = mapping.get("state", QUEUED_STATE)
    if not isinstance(state_value, str) or state_value not in KNOWN_STATES:
        raise MergeQueueRunnerError(
            f"queue fragment {source} record {index} has unknown state {state_value!r}"
        )
    enqueued_raw = mapping.get("enqueued_at")
    recorded_raw = mapping.get("recorded_at")
    if enqueued_raw is None and state_value != QUEUED_STATE:
        enqueued_raw = recorded_raw
    if recorded_raw is None and state_value != QUEUED_STATE:
        recorded_raw = enqueued_raw
    enqueued_at = _timestamp(enqueued_raw, field="enqueued_at", source=source)
    recorded_at = _optional_timestamp(recorded_raw, field="recorded_at", source=source)
    return QueueRecord(
        candidate_id=candidate_id,
        state=state_value,
        enqueued_at=enqueued_at,
        recorded_at=recorded_at,
        worktree=_record_worktree(mapping),
        source=source,
    )


def _record_worktree(mapping: dict[str, Any]) -> str:
    """Normalize a queue record's optional worktree without hiding fresh drift."""

    value = mapping.get("worktree", "")
    if isinstance(value, str):
        return value.strip()
    # Terminal and canonical projections are ignored by selection. Preserve
    # that lifecycle rule here; a fresh non-canonical queued record reaches
    # _validate_queued_worktree and fails closed with a useful reason.
    return ""


def _latest_queue_records(common: Path) -> tuple[QueueRecord, ...]:
    queue_dir = common / QUEUE_DIRECTORY
    if not queue_dir.exists():
        return ()
    if not queue_dir.is_dir() or queue_dir.is_symlink():
        raise MergeQueueRunnerError(f"merge queue directory is invalid: {queue_dir}")
    grouped: dict[str, list[QueueRecord]] = defaultdict(list)
    for fragment in sorted(queue_dir.glob("*.yaml")):
        for index, mapping in enumerate(_fragment_records(fragment)):
            record = _record_from_mapping(mapping, source=fragment, index=index)
            if record is not None:
                grouped[record.candidate_id].append(record)
    # Timestamps, not lane/filename order, are lifecycle authority.  The
    # canonical projection wins an exact timestamp tie so a copied terminal
    # state cannot be revived by an older lane fragment.  Before recorded_at
    # existed, preserve deterministic append order without fabricating a time;
    # a fresh queued record still fails closed below because it cannot establish
    # lifecycle ordering.
    latest: list[QueueRecord] = []
    for records in grouped.values():
        latest.append(_latest_record(records))
    return tuple(
        sorted(latest, key=lambda record: (record.enqueued_at, record.candidate_id))
    )


def _registered_worktrees(repository: DeclaredRepository) -> frozenset[Path]:
    """Read Git's registration authority for one declared repository."""

    listing = _git_output(
        ["-C", str(repository.path), "worktree", "list", "--porcelain"],
        cwd=repository.path,
        label=f"repository {repository.identifier!r} worktree registration",
    )
    paths: set[Path] = set()
    for line in listing.splitlines():
        if line.startswith("worktree "):
            paths.add(Path(line[len("worktree ") :].strip()).resolve(strict=False))
    return frozenset(paths)


def _validate_worktree_git_identity(
    repository: DeclaredRepository,
    record: QueueRecord,
    target: Path,
    normalized: Path,
) -> None:
    """Require a registered lane to belong to the declared Git repository."""

    top_level = Path(
        _git_output(
            ["-C", str(target), "rev-parse", "--show-toplevel"],
            cwd=repository.path,
            label=f"queue record {record.candidate_id!r} worktree",
        )
    ).resolve()
    if top_level != normalized:
        raise MergeQueueRunnerError(
            f"queue record {record.candidate_id!r} in {record.source} points to "
            f"worktree {normalized} from {top_level}, not its registered root"
        )
    repository_common = _git_common_directory(repository)
    worktree_common = _git_common_directory_for_path(
        target, label=f"queue record {record.candidate_id!r} worktree"
    )
    if worktree_common != repository_common:
        raise MergeQueueRunnerError(
            f"queue record {record.candidate_id!r} in {record.source} points to "
            f"Git common directory {worktree_common}, not the declared repository "
            f"{repository_common}"
        )


def _validate_queued_worktree(
    repository: DeclaredRepository, record: QueueRecord
) -> None:
    """Refuse a fresh queue record whose recorded lane is no longer live.

    Queue discovery is allowed to ignore old records, canonical projections,
    and terminal states. A fresh non-canonical queued record is different: if
    its lane disappeared or points at another checkout, selecting the root
    would hide a queue-authority failure. Fail closed before starting any gate.
    """

    if not record.worktree:
        raise MergeQueueRunnerError(
            f"queue record {record.candidate_id!r} in {record.source} has no "
            "recorded worktree; refusing to schedule an unverifiable lane"
        )
    target = Path(record.worktree).expanduser()
    if not target.is_absolute():
        raise MergeQueueRunnerError(
            f"queue record {record.candidate_id!r} in {record.source} records a "
            f"relative worktree {record.worktree!r}; refusing path drift"
        )
    if target.is_symlink() or not target.is_dir():
        raise MergeQueueRunnerError(
            f"queue record {record.candidate_id!r} in {record.source} records "
            f"stale or missing worktree {target}"
        )
    normalized = target.resolve(strict=False)
    if normalized not in _registered_worktrees(repository):
        raise MergeQueueRunnerError(
            f"queue record {record.candidate_id!r} in {record.source} records "
            f"unregistered worktree {normalized}; refusing stale lane"
        )
    _validate_worktree_git_identity(repository, record, target, normalized)


def queued_repositories(
    repositories: Iterable[DeclaredRepository],
    *,
    max_age_seconds: int = DEFAULT_MAX_QUEUE_AGE_SECONDS,
    now: datetime | None = None,
) -> tuple[DeclaredRepository, ...]:
    """Return declared repositories with a fresh, non-terminal queued record."""

    if max_age_seconds <= 0:
        raise MergeQueueRunnerError("max_age_seconds must be a positive integer")
    current = (now or datetime.now(UTC)).astimezone(UTC)
    cutoff = current - timedelta(seconds=max_age_seconds)
    selected: list[DeclaredRepository] = []
    for repository in repositories:
        if _has_fresh_queue_record(repository, cutoff):
            selected.append(repository)
    return tuple(selected)


def _is_fresh_queued_record(record: QueueRecord, cutoff: datetime) -> bool:
    """Accept only timestamped queued records inside the configured age window."""

    if record.state != QUEUED_STATE or record.enqueued_at < cutoff:
        return False
    if record.recorded_at is None:
        raise MergeQueueRunnerError(
            f"{record.source}: fresh queued record {record.candidate_id!r} "
            "is missing recorded_at; migrate the legacy queue record before "
            "scheduling it"
        )
    return True


def _has_fresh_queue_record(repository: DeclaredRepository, cutoff: datetime) -> bool:
    """Validate and report whether one repository has a fresh queued record."""

    common = _git_common_directory(repository)
    for record in _latest_queue_records(common):
        if record.source.name == CANONICAL_FRAGMENT:
            continue
        if not _is_fresh_queued_record(record, cutoff):
            continue
        _validate_queued_worktree(repository, record)
        return True
    return False


def discover_queued_repositories(
    workspace_root: str | Path,
    *,
    manifest: str | Path | None = None,
    max_age_seconds: int = DEFAULT_MAX_QUEUE_AGE_SECONDS,
    now: datetime | None = None,
) -> tuple[DeclaredRepository, ...]:
    """Validate the manifest and discover its currently queued repositories."""

    return queued_repositories(
        declared_repositories(workspace_root, manifest=manifest),
        max_age_seconds=max_age_seconds,
        now=now,
    )


def _resolve_executable(value: str) -> str:
    candidate = Path(value).expanduser()
    if candidate.is_absolute():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    else:
        resolved = shutil.which(value)
        if resolved:
            return resolved
    raise MergeQueueRunnerError(
        f"repository-manager executable is unavailable or not executable: {value!r}"
    )


def _process_parent(pid: int) -> int | None:
    """Read one process parent id from Linux' proc status line."""

    try:
        status = Path(f"/proc/{pid}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeError):
        return None
    closing = status.rfind(")")
    if closing < 0:
        return None
    fields = status[closing + 2 :].split()
    try:
        return int(fields[1])
    except (IndexError, ValueError):
        return None


def _process_tree(root_pid: int) -> set[int]:
    """Return root plus every currently observable descendant process."""

    parents: dict[int, set[int]] = defaultdict(set)
    proc_root = Path("/proc")
    try:
        entries = tuple(proc_root.iterdir())
    except OSError:
        return {root_pid}
    for entry in entries:
        if not entry.name.isdecimal():
            continue
        pid = int(entry.name)
        parent = _process_parent(pid)
        if parent is not None:
            parents[parent].add(pid)
    tree = {root_pid}
    pending = [root_pid]
    while pending:
        parent = pending.pop()
        for child in parents.get(parent, ()):
            if child not in tree:
                tree.add(child)
                pending.append(child)
    return tree


def _process_cgroup(pid: int) -> str:
    """Return a process' unified cgroup, refusing unverifiable execution."""

    try:
        content = Path(f"/proc/{pid}/cgroup").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise MergeQueueRunnerError(
            f"cannot verify cgroup for process {pid}: {exc}"
        ) from exc
    for line in content.splitlines():
        controller, separator, path = line.partition(":")
        if separator and controller == "0":
            value = path.partition(":")[2].strip()
            if value:
                return value
    raise MergeQueueRunnerError(
        f"cannot verify cgroup for process {pid}: unified cgroup membership is unavailable"
    )


def _cgroup_violation(
    root_pid: int, expected_cgroup: str, seen: set[int]
) -> str | None:
    """Check the full observed process tree against the runner's cgroup."""

    current = _process_tree(root_pid)
    seen.update(current)
    for pid in current | seen:
        try:
            actual = _process_cgroup(pid)
        except MergeQueueRunnerError as exc:
            # A process can disappear between the proc directory scan and the
            # read.  It is safe to ignore that one race; a live process must be
            # readable or the supervisor fails closed below.
            if not Path(f"/proc/{pid}").exists():
                continue
            return str(exc)
        if actual != expected_cgroup:
            return (
                f"process {pid} escaped runner cgroup {expected_cgroup!r} "
                f"into {actual!r}"
            )
    return None


def _process_exists(pid: int) -> bool:
    """Return whether a process is still observable in procfs."""

    return Path(f"/proc/{pid}").exists()


def _monitor_process_cgroup(
    process: subprocess.Popen[str],
    expected_cgroup: str,
    stop: threading.Event,
    observed: set[int],
    violations: list[str],
) -> None:
    """Continuously fail closed if a child or descendant leaves the unit."""

    while True:
        violation = _cgroup_violation(process.pid, expected_cgroup, observed)
        if violation is not None:
            violations.append(violation)
            if not _kill_process_group(process, observed):
                violations.append("escaped process tree could not be terminated")
            return
        if stop.wait(0.1):
            violation = _cgroup_violation(process.pid, expected_cgroup, observed)
            if violation is not None:
                violations.append(violation)
                if not _kill_process_group(process, observed):
                    violations.append("escaped process tree could not be terminated")
            return


def _signal_process_tree(
    process: subprocess.Popen[str], signum: int, observed: set[int]
) -> None:
    """Signal the process group and every descendant found in proc."""

    observed.update(_process_tree(process.pid))
    try:
        os.killpg(process.pid, signum)
    except (OSError, ProcessLookupError):
        pass
    for pid in observed:
        if pid == process.pid:
            continue
        try:
            os.kill(pid, signum)
        except (OSError, ProcessLookupError):
            pass


def _wait_for_process_tree(
    process: subprocess.Popen[str], observed: set[int], timeout: float
) -> bool:
    """Wait until the child and every previously observed descendant exit."""

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        process.poll()
        if not any(_process_exists(pid) for pid in observed):
            return True
        time.sleep(0.05)
    process.poll()
    return not any(_process_exists(pid) for pid in observed)


def _kill_process_group(
    process: subprocess.Popen[str], observed: set[int] | None = None
) -> bool:
    """Terminate the group and all observed descendants, then verify cleanup."""

    tracked = observed if observed is not None else set()
    _signal_process_tree(process, signal.SIGTERM, tracked)
    if _wait_for_process_tree(process, tracked, 10):
        return True
    _signal_process_tree(process, signal.SIGKILL, tracked)
    return _wait_for_process_tree(process, tracked, 10)


def _runner_command(
    repository: DeclaredRepository,
    executable: str | None,
    lease_ttl_seconds: int | None,
) -> list[str]:
    """Build a fixed-argv child command without a wrapper process."""

    command = (
        [sys.executable, "-m", "repository_manager"]
        if executable is None
        else [_resolve_executable(executable)]
    )
    command.extend(
        [
            "--merge-queue",
            "run",
            "--repo-path",
            str(repository.path),
            "--queue-no-prune",
            "--queue-no-push",
        ]
    )
    if lease_ttl_seconds is not None:
        command.extend(["--queue-lease-ttl-seconds", str(lease_ttl_seconds)])
    return command


def _launch_process(
    command: list[str],
    repository: DeclaredRepository,
    *,
    env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    """Start one direct child process in its own process group."""

    try:
        return subprocess.Popen(
            command,
            cwd=str(repository.path),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
            env=env,
        )
    except OSError as exc:
        raise MergeQueueRunnerError(
            f"repository {repository.identifier!r} drain could not start: {exc}"
        ) from exc


def _queue_child_environment(repository: DeclaredRepository) -> dict[str, str]:
    """Pass the disk-backed AU temp root and bounded pytest fanout to a child."""

    target_dir, tmp_dir = default_guard_paths(repository.path, "merge-queue")
    # The generic merge-queue entrypoint repeats the same values before it
    # admits work.  Supplying them here also keeps wrappers and early imports
    # from observing an operator's unrelated temporary or xdist settings.
    environment = os.environ.copy()
    environment.update(
        {
            "CARGO_TARGET_DIR": str(target_dir),
            "AU_LANE_TEMP_ROOT": str(tmp_dir.parents[2]),
            "TMPDIR": str(tmp_dir),
            "RUNNER_TEMP": str(tmp_dir),
            "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
            "RM_MERGE_QUEUE_SUPERVISOR_PID": str(os.getpid()),
            "RM_MERGE_QUEUE_SUPERVISOR_START_TIME": str(process_start_time()),
        }
    )
    return environment


def _timeout_output(exc: subprocess.TimeoutExpired) -> str:
    """Normalize partial output returned by a timed-out text subprocess."""

    raw_output = exc.output
    if isinstance(raw_output, bytes):
        return raw_output.decode("utf-8", errors="replace")
    return raw_output if isinstance(raw_output, str) else ""


def _communicate_with_deadline(
    process: subprocess.Popen[str],
    repository: DeclaredRepository,
    deadline_seconds: int,
    observed: set[int],
) -> tuple[str, int]:
    """Collect child output and kill its full tree when the deadline expires."""

    try:
        output, _ = process.communicate(timeout=deadline_seconds)
    except subprocess.TimeoutExpired as exc:
        terminated = _kill_process_group(process, observed)
        cleanup = "" if terminated else "\nprocess tree cleanup did not complete"
        raise MergeQueueRunnerError(
            f"repository {repository.identifier!r} drain timed out after "
            f"{deadline_seconds}s{cleanup}\n{_timeout_output(exc)}"
        ) from exc
    return output or "", process.returncode


def _run_monitored_process(
    process: subprocess.Popen[str],
    repository: DeclaredRepository,
    expected_cgroup: str,
    deadline_seconds: int,
) -> tuple[str, int, list[str]]:
    """Run one child while continuously checking its process tree cgroup."""

    stop = threading.Event()
    violations: list[str] = []
    observed: set[int] = set()
    monitor = threading.Thread(
        target=_monitor_process_cgroup,
        args=(process, expected_cgroup, stop, observed, violations),
        name="merge-queue-cgroup-monitor",
        daemon=True,
    )
    monitor.start()
    try:
        output, returncode = _communicate_with_deadline(
            process, repository, deadline_seconds, observed
        )
    finally:
        stop.set()
        monitor.join(timeout=22)
    return output, returncode, violations


def drain_repository(
    repository: DeclaredRepository,
    *,
    executable: str | None = None,
    deadline_seconds: int = DEFAULT_DRAIN_DEADLINE_SECONDS,
    lease_ttl_seconds: int | None = None,
) -> DrainResult:
    """Run one repo-scoped queue drain under a wall-clock deadline."""

    if deadline_seconds <= 0:
        raise MergeQueueRunnerError("deadline_seconds must be a positive integer")
    if lease_ttl_seconds is not None:
        lease_ttl_seconds = _lease_ttl(
            deadline_seconds,
            lease_ttl_seconds,
            env_name="MERGE_QUEUE_LEASE_TTL_SECONDS",
        )
    command = _runner_command(repository, executable, lease_ttl_seconds)
    expected_cgroup = _process_cgroup(os.getpid())
    process = _launch_process(
        command, repository, env=_queue_child_environment(repository)
    )
    try:
        output, returncode, violations = _run_monitored_process(
            process, repository, expected_cgroup, deadline_seconds
        )
    except subprocess.SubprocessError as exc:
        raise MergeQueueRunnerError(
            f"repository {repository.identifier!r} drain failed: {exc}"
        ) from exc
    if violations:
        raise MergeQueueRunnerError(
            f"repository {repository.identifier!r} drain escaped its systemd cgroup: "
            f"{violations[0]}\n{output or ''}"
        )
    return DrainResult(repository, returncode, output)


def _phased_push_command(workspace_root: Path, manifest: Path) -> list[str]:
    """Build the one-lane direct-module command for heavy publication."""

    return [
        sys.executable,
        "-m",
        "repository_manager",
        "--workspace",
        str(workspace_root),
        "--file",
        str(manifest),
        "--threads",
        "1",
        "--push",
    ]


def _project_manifest_repositories(
    raw_repositories: object, *, prefix: tuple[str, ...], source: Path
) -> list[object]:
    """Copy repository declarations except explicitly excluded path components."""

    if not isinstance(raw_repositories, list):
        raise MergeQueueRunnerError(
            f"{source}: repositories must be a list in generated projection"
        )
    repositories: list[object] = []
    for entry in raw_repositories:
        if not isinstance(entry, dict) or not isinstance(entry.get("url"), str):
            raise MergeQueueRunnerError(
                f"{source}: each repository must have a string url"
            )
        name = entry["url"].strip().rstrip("/").rsplit("/", 1)[-1]
        name = name.removesuffix(".git")
        identifier = "/".join((*prefix, name))
        if frozenset(identifier.split("/")) & EXCLUDED_MANIFEST_COMPONENTS:
            continue
        repositories.append(copy.deepcopy(entry))
    return repositories


def _project_manifest_subdirectories(
    raw_subdirectories: object, *, prefix: tuple[str, ...], source: Path
) -> dict[str, Any]:
    """Recursively copy non-reference workspace directory declarations."""

    if not isinstance(raw_subdirectories, dict):
        raise MergeQueueRunnerError(
            f"{source}: subdirectories must be a mapping in generated projection"
        )
    subdirectories: dict[str, Any] = {}
    for name, child in raw_subdirectories.items():
        if not isinstance(name, str) or not name:
            raise MergeQueueRunnerError(
                f"{source}: subdirectory names must be non-empty strings"
            )
        child_prefix = (*prefix, name)
        if frozenset(child_prefix) & EXCLUDED_MANIFEST_COMPONENTS:
            continue
        subdirectories[name] = _project_manifest_node(child, child_prefix, source)
    return subdirectories


def _project_manifest_node(
    node: object, prefix: tuple[str, ...], source: Path
) -> dict[str, Any]:
    """Copy one manifest directory and recursively remove excluded inputs."""

    if not isinstance(node, dict):
        raise MergeQueueRunnerError(f"{source}: manifest directory must be a mapping")
    projected = copy.deepcopy(node)
    projected["repositories"] = _project_manifest_repositories(
        node.get("repositories", []), prefix=prefix, source=source
    )
    projected["subdirectories"] = _project_manifest_subdirectories(
        node.get("subdirectories", {}), prefix=prefix, source=source
    )
    return projected


def _excluded_manifest_identifiers(
    identifiers: Iterable[str], *, source: Path
) -> tuple[str, ...]:
    """Find excluded identifiers that would indicate projection drift."""

    excluded = tuple(
        identifier
        for identifier in identifiers
        if frozenset(identifier.split("/")) & EXCLUDED_MANIFEST_COMPONENTS
    )
    if excluded:
        raise MergeQueueRunnerError(
            f"{source}: generated phased-push projection retained excluded "
            f"repositories: {', '.join(excluded)}"
        )
    return excluded


def _eligible_manifest_projection(
    data: dict[str, Any], *, source: Path
) -> dict[str, Any]:
    """Build the heavy lane's validated projection of the canonical manifest.

    The canonical manifest remains the authority.  This projection only removes
    entries classified as reference inputs (currently ``open-source-libraries``)
    using the same path-component rule as queue discovery; it never invents a
    second repository inventory.  Keeping the projection in a temporary
    directory prevents the scheduler from mutating the canonical manifest or
    leaving generated files in a workspace checkout.
    """

    projection = _project_manifest_node(data, (), source)
    identifiers = _manifest_identifiers(projection, source)
    _excluded_manifest_identifiers(identifiers, source=source)
    return projection


def run_phased_push(
    workspace_root: str | Path,
    *,
    manifest: str | Path | None = None,
    deadline_seconds: int = DEFAULT_HEAVY_TIMEOUT_SECONDS,
    gate_timeout_seconds: int = DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
) -> int:
    """Run the heavy phased-push child under the same cgroup supervisor.

    The service owns one dedicated lane, while this process boundary verifies
    every descendant remains in the service cgroup. This catches a gate that
    tries to launch a Snap/uv scope instead of relying on ``KillMode`` alone.
    """

    if deadline_seconds <= 0 or gate_timeout_seconds <= 0:
        raise MergeQueueRunnerError("heavy runner settings must be positive integers")
    root = _configured_workspace_root(workspace_root)
    manifest_path = _manifest_path(root, manifest)
    # Validate the same canonical inventory as the queue mode before starting a
    # heavy gate. A missing clone or manifest drift must not become a partial
    # publication run.
    declared_repositories(root, manifest=manifest_path)
    manifest_data = _manifest_mapping(manifest_path)
    projected_manifest = _eligible_manifest_projection(
        manifest_data, source=manifest_path
    )
    # The child is deliberately launched with an explicit workspace root. Make
    # the generated projection self-contained as well, so a canonical manifest
    # using the portable ``${AGENT_UTILITIES_WORKSPACE_ROOT}`` reference does
    # not depend on an ambient service environment variable.
    projected_manifest["path"] = str(root)
    repository = DeclaredRepository("phased-push", "phased-push", root)
    env = os.environ.copy()
    env.update(
        {
            "RM_GATE_TIMEOUT_SECONDS": str(gate_timeout_seconds),
            "RM_GATE_MAX_WORKERS": "2",
            "REPOSITORY_MANAGER_THREADS": "1",
        }
    )
    expected_cgroup = _process_cgroup(os.getpid())
    try:
        with tempfile.TemporaryDirectory(
            prefix="repository-manager-phased-push-"
        ) as directory:
            projected_path = Path(directory) / "workspace.yml"
            projected_path.write_bytes(
                yaml.safe_dump(projected_manifest, sort_keys=False).encode("utf-8")
            )
            process = _launch_process(
                _phased_push_command(root, projected_path), repository, env=env
            )
            try:
                output, returncode, violations = _run_monitored_process(
                    process, repository, expected_cgroup, deadline_seconds
                )
            except (MergeQueueRunnerError, subprocess.SubprocessError) as exc:
                _log(f"phased push FAILED: {exc}")
                return 1
    except (OSError, UnicodeError, yaml.YAMLError) as exc:
        _log(f"phased push FAILED: generated manifest projection unavailable: {exc}")
        return 1
    if output:
        print(output, end="" if output.endswith("\n") else "\n")
    if violations:
        _log(f"phased push FAILED: cgroup escape: {violations[0]}")
        return 1
    if returncode != 0:
        _log(f"phased push FAILED with status {returncode}")
        return 1
    _log("phased push completed successfully")
    return 0


def _log(message: str) -> None:
    print(f"{datetime.now().astimezone().isoformat()} merge-queue-runner: {message}")


def _environment_integer(name: str, default: int) -> int:
    value = os.environ.get(name, str(default))
    return _positive_integer(value, name=name)


@dataclass(frozen=True)
class RunnerSettings:
    """Validated runtime settings for one timer invocation."""

    deadline_seconds: int
    global_deadline_seconds: int
    lease_ttl_seconds: int
    max_age_seconds: int
    executable: str | None


@dataclass(frozen=True)
class RunDiscovery:
    """Validated manifest inventory and its fresh queue subset."""

    declared: tuple[DeclaredRepository, ...]
    selected: tuple[DeclaredRepository, ...]


def _optional_setting(value: int | None, env_name: str, default: int) -> int:
    """Resolve a positive CLI value or its environment-backed default."""

    if value is None:
        return _environment_integer(env_name, default)
    return _positive_integer(str(value), name=env_name)


def _runner_settings(args: argparse.Namespace) -> RunnerSettings:
    """Resolve and validate all settings before touching queue state."""

    deadline_seconds = _optional_setting(
        args.drain_deadline_seconds,
        "MERGE_QUEUE_DRAIN_DEADLINE_SECONDS",
        DEFAULT_DRAIN_DEADLINE_SECONDS,
    )
    global_deadline_seconds = _optional_setting(
        args.global_deadline_seconds,
        "MERGE_QUEUE_GLOBAL_DEADLINE_SECONDS",
        DEFAULT_GLOBAL_DRAIN_DEADLINE_SECONDS,
    )
    _validate_global_deadline(deadline_seconds, global_deadline_seconds)
    lease_ttl_seconds = _lease_ttl(
        deadline_seconds,
        args.lease_ttl_seconds,
        env_name="MERGE_QUEUE_LEASE_TTL_SECONDS",
    )
    max_age_seconds = _optional_setting(
        args.max_age_seconds,
        "MERGE_QUEUE_MAX_AGE_SECONDS",
        DEFAULT_MAX_QUEUE_AGE_SECONDS,
    )
    executable = (
        _resolve_executable(args.repository_manager_command)
        if args.repository_manager_command
        else None
    )
    return RunnerSettings(
        deadline_seconds,
        global_deadline_seconds,
        lease_ttl_seconds,
        max_age_seconds,
        executable,
    )


def _validate_global_deadline(per_root_seconds: int, global_seconds: int) -> None:
    """Require the fleet ceiling to admit at least one complete root budget."""

    if global_seconds < per_root_seconds:
        raise MergeQueueRunnerError(
            "global deadline must be at least the per-repository drain deadline "
            f"({per_root_seconds}s)"
        )


def _discover_for_run(
    args: argparse.Namespace, *, max_age_seconds: int
) -> RunDiscovery:
    """Validate the manifest and select fresh queues for this run."""

    declared = declared_repositories(args.workspace_root, manifest=args.manifest)
    selected = queued_repositories(declared, max_age_seconds=max_age_seconds)
    return RunDiscovery(declared, selected)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Drain merge queues for repositories declared by workspace.yml."
    )
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get("AGENT_UTILITIES_WORKSPACE_ROOT")
        or os.environ.get("REPOSITORY_MANAGER_WORKSPACE")
        or str(Path.cwd()),
        help="canonical workspace root containing workspace.yml",
    )
    parser.add_argument("--manifest", default=None, help="validated manifest path")
    parser.add_argument(
        "--drain-deadline-seconds",
        type=int,
        default=None,
        help=(
            "per-repository child wall-clock deadline; independent of gate "
            "worker/resource limits"
        ),
    )
    parser.add_argument(
        "--global-deadline-seconds",
        type=int,
        default=None,
        help=(
            "whole-invocation wall-clock deadline across all selected roots; "
            "independent from each root's deadline"
        ),
    )
    parser.add_argument(
        "--max-age-seconds",
        type=int,
        default=None,
        help="ignore queued records older than this age",
    )
    parser.add_argument(
        "--lease-ttl-seconds",
        type=int,
        default=None,
        help=(
            "repo lease TTL; must exceed the wall-clock deadline and stay within "
            "the bounded 24-hour lease maximum"
        ),
    )
    parser.add_argument(
        "--repository-manager-command",
        default=os.environ.get("REPOSITORY_MANAGER_COMMAND"),
        help="repository-manager executable used for each drain",
    )
    parser.add_argument(
        "--phased-push",
        action="store_true",
        help="run the dedicated heavy phased-push child instead of the queue drain",
    )
    parser.add_argument(
        "--heavy-deadline-seconds",
        type=int,
        default=None,
        help="heavy phased-push wall-clock deadline",
    )
    parser.add_argument(
        "--heavy-gate-timeout-seconds",
        type=int,
        default=None,
        help="RM_GATE_TIMEOUT_SECONDS for the heavy phased-push child",
    )
    return parser


def _drain_one(
    repository: DeclaredRepository,
    *,
    executable: str | None,
    deadline_seconds: int,
    lease_ttl_seconds: int,
) -> str:
    """Drain one selected repository and return its accounting outcome."""

    _log(f"draining {repository.identifier} ({repository.path})")
    try:
        result = drain_repository(
            repository,
            executable=executable,
            deadline_seconds=deadline_seconds,
            lease_ttl_seconds=lease_ttl_seconds,
        )
    except MergeQueueRunnerError as exc:
        _log(f"{repository.identifier}: drain FAILED: {exc}")
        return "failed"
    if result.output:
        print(result.output, end="" if result.output.endswith("\n") else "\n")
    if result.returncode == 0:
        return "drained"
    if result.deferred:
        _log(f"{repository.identifier}: lease held; deferring (expected)")
        return "deferred"
    _log(f"{repository.identifier}: drain FAILED with status {result.returncode}")
    return "failed"


def _root_deadline(
    per_root_seconds: int, global_seconds: int, elapsed_seconds: float
) -> int | None:
    """Return the integer child budget that fits within the global ceiling."""

    remaining = global_seconds - elapsed_seconds
    if remaining < 1:
        return None
    return min(per_root_seconds, int(remaining))


def _drain_selected_outcomes(
    selected: tuple[DeclaredRepository, ...],
    *,
    executable: str | None,
    deadline_seconds: int,
    global_deadline_seconds: int,
    lease_ttl_seconds: int,
) -> tuple[list[str], int, bool, float]:
    """Run selected roots until the global ceiling, returning accounting data."""

    _validate_global_deadline(deadline_seconds, global_deadline_seconds)
    started = time.monotonic()
    outcomes: list[str] = []
    skipped_for_deadline = False
    attempted = 0
    for repository in selected:
        child_deadline = _root_deadline(
            deadline_seconds, global_deadline_seconds, time.monotonic() - started
        )
        if child_deadline is None:
            _log(
                f"{repository.identifier}: global drain deadline reached; "
                "deferring without starting another child"
            )
            outcomes.append("deferred")
            skipped_for_deadline = True
            continue
        outcomes.append(
            _drain_one(
                repository,
                executable=executable,
                deadline_seconds=child_deadline,
                lease_ttl_seconds=lease_ttl_seconds,
            )
        )
        attempted += 1
    return outcomes, attempted, skipped_for_deadline, started


def _drain_exit(
    failed: int, skipped_for_deadline: bool, elapsed: float, global_seconds: int
) -> tuple[bool, int]:
    """Compute the supervisor result from child failures and deadline evidence."""

    deadline_exhausted = skipped_for_deadline or elapsed > global_seconds
    return deadline_exhausted, int(failed > 0 or deadline_exhausted)


def _drain_selected(
    selected: tuple[DeclaredRepository, ...],
    *,
    executable: str | None,
    deadline_seconds: int,
    global_deadline_seconds: int,
    lease_ttl_seconds: int,
) -> int:
    """Drain selected roots serially within per-root and global ceilings."""

    outcomes, attempted, skipped_for_deadline, started = _drain_selected_outcomes(
        selected,
        executable=executable,
        deadline_seconds=deadline_seconds,
        global_deadline_seconds=global_deadline_seconds,
        lease_ttl_seconds=lease_ttl_seconds,
    )
    drained = outcomes.count("drained")
    deferred = outcomes.count("deferred")
    failed = outcomes.count("failed")
    elapsed = time.monotonic() - started
    deadline_exhausted, exit_code = _drain_exit(
        failed, skipped_for_deadline, elapsed, global_deadline_seconds
    )
    _log(
        f"done: attempted {attempted}/{len(selected)} queued root(s), drained {drained}, "
        f"deferred {deferred}, failed {failed}, global_deadline_exhausted="
        f"{deadline_exhausted}, exit {exit_code}"
    )
    return exit_code


def _run_phased_push_cli(args: argparse.Namespace) -> int:
    """Resolve heavy-lane settings and run its direct child."""

    try:
        deadline_seconds = _optional_setting(
            args.heavy_deadline_seconds,
            "MERGE_QUEUE_HEAVY_DEADLINE_SECONDS",
            DEFAULT_HEAVY_TIMEOUT_SECONDS,
        )
        gate_timeout_seconds = _optional_setting(
            args.heavy_gate_timeout_seconds,
            "RM_GATE_TIMEOUT_SECONDS",
            DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
        )
        return run_phased_push(
            args.workspace_root,
            manifest=args.manifest,
            deadline_seconds=deadline_seconds,
            gate_timeout_seconds=gate_timeout_seconds,
        )
    except MergeQueueRunnerError as exc:
        _log(f"FATAL: {exc}")
        return 78


def _run_queue_cli(args: argparse.Namespace) -> int:
    """Resolve queue settings, discover eligible roots, and drain them."""

    try:
        settings = _runner_settings(args)
        discovery = _discover_for_run(args, max_age_seconds=settings.max_age_seconds)
    except MergeQueueRunnerError as exc:
        _log(f"FATAL: {exc}")
        return 78

    if not discovery.selected:
        _log(
            f"done: no fresh queued repositories among {len(discovery.declared)} "
            "declared roots, exit 0"
        )
        return 0
    return _drain_selected(
        discovery.selected,
        executable=settings.executable,
        deadline_seconds=settings.deadline_seconds,
        global_deadline_seconds=settings.global_deadline_seconds,
        lease_ttl_seconds=settings.lease_ttl_seconds,
    )


def _run_mode(args: argparse.Namespace) -> int:
    """Dispatch one parsed invocation to its dedicated scheduler lane."""

    if args.phased_push:
        return _run_phased_push_cli(args)
    return _run_queue_cli(args)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point used by the installed user service."""

    args = build_parser().parse_args(argv)
    return _run_mode(args)


if (
    __name__ == "__main__"
):  # pragma: no cover - exercised through the script entry point
    raise SystemExit(main(sys.argv[1:]))
