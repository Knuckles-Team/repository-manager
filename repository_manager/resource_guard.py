"""Host admission and systemd containment for repository build processes.

This is the in-process form of the fleet ``dispatch_build.sh`` contract.  The
dispatcher remains the remote transport; this module is the common local
boundary used by Repository Manager execution paths.  It admits one heavy job
per host user, refuses unhealthy memory or storage state, applies a transient
systemd scope, and returns the evidence used to make that decision.

CONCEPT:RM-RESOURCE-GUARD
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess  # nosec B404 - fixed argv, never a shell
import sys
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from contextvars import ContextVar, Token
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# This whole module is a POSIX/systemd-only boundary (systemd-run, cgroups,
# /proc, advisory ``flock``) -- it was never going to function on Windows.
# The guard below only keeps the module *importable* there (the same
# ``sys.platform == "win32"`` chokepoint pattern used by
# ``agent_utilities.knowledge_graph.core.file_lock``, R-07): a bare
# ``import fcntl`` fails at import time on Windows, before any lock is ever
# taken, breaking every caller on that module's import path -- not just the
# handful of functions that actually flock something.
if sys.platform != "win32":
    import fcntl

from repository_manager.execution.bounded_log import BoundedLogSink, StreamName
from repository_manager.resource_profiles import (
    DEFAULT_RESOURCE_PROFILES,
    ResourceProfile,
)

MIB = 1024**2
_STATE_SCHEMA = "repository-resource-guard-v1"
_MEMORY_PSI_HIGH = (10.0, 2.0)
_MEMORY_PSI_LOW = (2.0, 0.5)
_CARGO_NAMES = {"cargo", "cargo.exe"}
_DISK_BACKED_FS = {"tmpfs", "ramfs"}


class ResourceGuardError(RuntimeError):
    """Base error for an unenforceable or refused resource contract."""


class ResourceAdmissionError(ResourceGuardError):
    """The host cannot safely admit the requested process now."""

    def __init__(self, reason: str, evidence: Mapping[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.evidence = dict(evidence or {})


class _ExecutionCgroupAbsent(ResourceAdmissionError):
    """The already-bound cgroup directory was removed by systemd."""


@dataclass(frozen=True)
class FilesystemEvidence:
    purpose: str
    path: str
    filesystem: str
    free_bytes: int
    free_inodes: int
    required_free_bytes: int
    required_free_inodes: int


@dataclass(frozen=True)
class HostEvidence:
    captured_unix_ns: int
    memory_available_bytes: int
    swap_free_bytes: int
    memory_psi_some_avg10: float
    memory_psi_full_avg10: float
    cgroup_path: str
    cgroup_memory_current: int | None
    cgroup_memory_max: int | None
    cgroup_oom_events: int
    filesystems: tuple[FilesystemEvidence, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GuardedRunResult:
    completed: subprocess.CompletedProcess[str]
    unit: str
    profile: str
    heavy: bool
    properties: tuple[str, ...]
    before: HostEvidence
    after: HostEvidence
    output: Mapping[str, Any] = field(default_factory=dict)
    execution_cgroup: str | None = None

    def evidence(self) -> dict[str, Any]:
        return {
            "schema": _STATE_SCHEMA,
            "unit": self.unit,
            "profile": self.profile,
            "heavy": self.heavy,
            "execution_cgroup": self.execution_cgroup,
            "systemd_properties": list(self.properties),
            "before": self.before.as_dict(),
            "after": self.after.as_dict(),
            "oom_events_delta": max(
                0, self.after.cgroup_oom_events - self.before.cgroup_oom_events
            ),
            "output": dict(self.output),
            "returncode": self.completed.returncode,
        }


@dataclass(frozen=True)
class GuardedCommand:
    argv: tuple[str, ...]
    workdir: Path
    target_dir: Path
    tmp_dir: Path
    profile: ResourceProfile
    heavy: bool
    cargo: bool
    timeout: int
    environment: Mapping[str, str] = field(default_factory=dict)


@dataclass
class GuardAdmission:
    """One live host reservation shared by a guarded or inherited run.

    ``run_guarded`` uses a transient systemd scope after this context is
    entered.  Queue drains use the same admission context but launch directly
    into their already bounded service cgroup; creating a nested scope here
    would make the cgroup supervisor report a false escape.
    """

    command: GuardedCommand
    unit: str
    before: HostEvidence
    environment: dict[str, str]
    inherited_cgroup: str | None
    execution_cgroup: str | None = field(repr=False, default=None)
    _lock_handle: Any = field(repr=False, default=None)
    _dispatcher_reservation: Path | None = field(repr=False, default=None)
    _reservation_token: str | None = field(repr=False, default=None)
    _owner_pid: int = field(default_factory=os.getpid, init=False, repr=False)
    _inherited_context_token: Token[GuardAdmission | None] | None = field(
        repr=False, default=None
    )
    _released: bool = field(default=False, init=False, repr=False)
    _retain_reservation: bool = field(default=False, init=False, repr=False)
    _execution_closed_verified: bool = field(default=False, init=False, repr=False)

    def evidence(self) -> dict[str, Any]:
        """Return admission evidence before a child process is started."""

        return {
            "schema": _STATE_SCHEMA,
            "unit": self.unit,
            "profile": self.command.profile.name,
            "heavy": self.command.heavy,
            "before": self.before.as_dict(),
            "inherited_cgroup": self.inherited_cgroup,
            "execution_cgroup": self.execution_cgroup,
        }

    def release(self) -> None:
        """Release the dispatcher reservation and host-wide lock once."""

        if self._released:
            return
        self._released = True
        try:
            if not self._retain_reservation:
                if self.inherited_cgroup is None:
                    try:
                        self.verify_execution_closed()
                    except BaseException:
                        # A missing or unreadable containment proof must keep
                        # the dispatcher reservation as a visible blocker.
                        self._retain_reservation = True
                if not self._retain_reservation:
                    _release_dispatcher_heavy(
                        self._dispatcher_reservation,
                        self._reservation_token,
                        expected_unit=self.unit,
                    )
        finally:
            try:
                if self._lock_handle is not None:
                    fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_UN)
                    self._lock_handle.close()
            finally:
                if self._inherited_context_token is not None:
                    _INHERITED_ADMISSION.reset(self._inherited_context_token)
                    self._inherited_context_token = None

    def retain_reservation_on_release(self) -> None:
        """Leave capacity fenced when a child process could not be reaped."""

        self._retain_reservation = True

    def bind_execution_cgroup(
        self, cgroup_path: str, *, allow_missing: bool = False
    ) -> None:
        """Bind this admission to the actual standalone systemd cgroup."""

        if self._released:
            raise ResourceGuardError("cannot bind containment after admission release")
        execution_cgroup = _validated_execution_cgroup(
            cgroup_path, require_exists=not allow_missing
        )
        if self.inherited_cgroup is not None:
            if execution_cgroup != self.inherited_cgroup:
                raise ResourceGuardError(
                    "execution cgroup differs from the inherited admission cgroup"
                )
            self.execution_cgroup = execution_cgroup
            return
        _bind_dispatcher_execution_cgroup(
            self._dispatcher_reservation,
            self._reservation_token,
            self.unit,
            execution_cgroup,
        )
        self.execution_cgroup = execution_cgroup

    def verify_execution_closed(self) -> None:
        """Prove a standalone systemd scope is empty before release."""

        if self.inherited_cgroup is not None:
            return
        if self._execution_closed_verified:
            return
        if self.execution_cgroup is None:
            raise ResourceGuardError("standalone execution containment is unbound")
        if not _execution_cgroup_closed(self.unit, self.execution_cgroup):
            raise ResourceGuardError(
                "standalone execution cgroup closure could not be proven"
            )
        self._execution_closed_verified = True

    def mark_execution_closed(self) -> None:
        """Record a closure proof obtained while stopping the scope."""

        if self.inherited_cgroup is None and self.execution_cgroup is not None:
            self._execution_closed_verified = True

    def is_live(self) -> bool:
        """Verify the owner, reservation, and inherited cgroup still match."""

        if (
            self._released
            or self._owner_pid != os.getpid()
            or self.inherited_cgroup is None
            or self._lock_handle is None
            or self._dispatcher_reservation is None
        ):
            return False
        if not self._dispatcher_reservation.is_file():
            return False
        try:
            record = json.loads(
                self._dispatcher_reservation.read_text(encoding="utf-8")
            )
            if not isinstance(record, Mapping):
                return False
            record_pid = record.get("owner_pid")
            record_start = record.get("owner_start_time")
            record_token = record.get("owner_token")
            record_execution = record.get("execution_cgroup")
            if (
                record.get("unit") != self.unit
                or isinstance(record_pid, bool)
                or isinstance(record_start, bool)
                or not isinstance(record_pid, int)
                or not isinstance(record_start, int)
                or record_pid != self._owner_pid
                or record_start != process_start_time(self._owner_pid)
                or not isinstance(record_token, str)
                or record_token != self._reservation_token
                or record_execution != self.execution_cgroup
            ):
                return False
            return _self_cgroup_path() == self.inherited_cgroup
        except (OSError, UnicodeError, json.JSONDecodeError, ResourceGuardError):
            return False

    def __enter__(self) -> GuardAdmission:
        if self.inherited_cgroup is not None:
            self._inherited_context_token = _INHERITED_ADMISSION.set(self)
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.release()


_INHERITED_ADMISSION: ContextVar[GuardAdmission | None] = ContextVar(
    "repository_manager_inherited_admission", default=None
)


def inherited_admission_active() -> bool:
    """Return whether this call context holds a verified inherited admission."""

    admission = _INHERITED_ADMISSION.get()
    return admission is not None and admission.is_live()


def verified_inherited_admission() -> GuardAdmission | None:
    """Return the live inherited admission, or refuse a stale copied binding."""

    admission = _INHERITED_ADMISSION.get()
    if admission is None:
        return None
    if not admission.is_live():
        raise ResourceGuardError("inherited resource admission is no longer live")
    return admission


def is_cargo_shaped(argv: Sequence[str]) -> bool:
    """Detect Cargo even when a caller labels the operation as a fast gate."""

    if not argv:
        return False
    if Path(argv[0]).name.lower() in _CARGO_NAMES:
        return True
    return any(
        re.search(r"(?:^|[\s;&|])cargo(?:[\s;&|]|$)", part.lower())
        for part in argv[1:]
    )


def _runtime_root() -> Path:
    configured = os.environ.get("XDG_RUNTIME_DIR")
    root = Path(configured) if configured else Path(f"/run/user/{os.getuid()}")
    result = root / "repository-manager" / "resource-guard"
    try:
        result.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise ResourceGuardError(f"cannot create resource guard runtime: {exc}") from exc
    return result


def default_guard_paths(
    workdir: Path | str, namespace: str
) -> tuple[Path, Path]:
    """Return stable target/TMPDIR paths on the configured disk-backed root."""

    if not namespace or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,63}", namespace) is None:
        raise ResourceGuardError("resource guard namespace is invalid")
    root = Path(
        os.environ.get(
            "RM_RESOURCE_GUARD_ROOT", "/var/tmp/repository-manager-resource-guard"
        )
    ).resolve()
    identity = hashlib.sha256(str(Path(workdir).resolve()).encode()).hexdigest()[:16]
    lane = root / namespace / identity
    return lane / "target", lane / "tmp"


def _mount_type(path: Path) -> str:
    resolved = path.resolve()
    best: tuple[int, str] | None = None
    try:
        lines = Path("/proc/self/mountinfo").read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ResourceGuardError("cannot inspect mount types") from exc
    for line in lines:
        before, separator, after = line.partition(" - ")
        if not separator:
            continue
        left, right = before.split(), after.split()
        if len(left) < 5 or not right:
            continue
        mount = Path(left[4].replace("\\040", " "))
        try:
            resolved.relative_to(mount)
        except ValueError:
            continue
        candidate = (len(str(mount)), right[0].lower())
        if best is None or candidate[0] > best[0]:
            best = candidate
    if best is None:
        raise ResourceGuardError(f"cannot determine filesystem for {resolved}")
    return best[1]


def _memory_values() -> tuple[int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        name, separator, value = line.partition(":")
        if separator and name in {"MemAvailable", "SwapFree"}:
            values[name] = int(value.strip().split()[0]) * 1024
    return values.get("MemAvailable", 0), values.get("SwapFree", 0)


def _psi_values() -> tuple[float, float]:
    values = {"some": 0.0, "full": 0.0}
    try:
        lines = Path("/proc/pressure/memory").read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0.0, 0.0
    for line in lines:
        parts = line.split()
        if not parts or parts[0] not in values:
            continue
        for part in parts[1:]:
            if part.startswith("avg10="):
                values[parts[0]] = float(part.split("=", 1)[1])
    return values["some"], values["full"]


def _read_int(path: Path) -> int | None:
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if raw == "max":
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _cgroup_values() -> tuple[str, int | None, int | None, int]:
    preferred = Path(f"/sys/fs/cgroup/user.slice/user-{os.getuid()}.slice")
    current = preferred
    if not current.is_dir():
        current = Path("/sys/fs/cgroup")
        try:
            for line in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
                if line.startswith("0::"):
                    current /= line.split("::", 1)[1].lstrip("/")
                    break
        except OSError:
            pass
    oom_events = 0
    try:
        for line in (current / "memory.events").read_text(encoding="utf-8").splitlines():
            name, raw = line.split(maxsplit=1)
            if name in {"oom", "oom_kill", "oom_group_kill"}:
                oom_events += int(raw)
    except (OSError, ValueError):
        pass
    return (
        str(current),
        _read_int(current / "memory.current"),
        _read_int(current / "memory.max"),
        oom_events,
    )


def _self_cgroup_path() -> str:
    """Return this process' unified cgroup path, or refuse verification."""

    try:
        content = Path("/proc/self/cgroup").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ResourceGuardError(f"cannot inspect this process cgroup: {exc}") from exc
    for line in content.splitlines():
        controller, separator, value = line.partition(":")
        if controller != "0" or not separator:
            continue
        _ignored, _separator, path = value.partition(":")
        return path.strip() or "/"
    raise ResourceGuardError(
        "cannot verify inherited containment: unified cgroup membership is unavailable"
    )


def current_cgroup_path() -> str:
    """Return this process' cgroup path for an inherited admission caller."""

    return _self_cgroup_path()


def process_start_time(pid: int | None = None) -> int:
    """Return Linux ``/proc/<pid>/stat`` start ticks for an owner fence."""

    owner = os.getpid() if pid is None else pid
    if owner < 1:
        raise ResourceGuardError("process owner PID must be positive")
    try:
        raw = Path(f"/proc/{owner}/stat").read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ResourceGuardError(
            f"cannot inspect process owner {owner} start time"
        ) from exc
    _prefix, separator, fields = raw.rpartition(") ")
    if not separator:
        raise ResourceGuardError(f"process owner {owner} stat is malformed")
    values = fields.split()
    # After the comm field, starttime (procfs field 22) is index 19: state is
    # index 0 and ppid is index 1 in this suffix.
    if len(values) <= 19:
        raise ResourceGuardError(f"process owner {owner} stat is incomplete")
    try:
        start = int(values[19])
    except ValueError as exc:
        raise ResourceGuardError(f"process owner {owner} start time is invalid") from exc
    if start < 1:
        raise ResourceGuardError(f"process owner {owner} start time is not positive")
    return start


def _canonical_cgroup_path(cgroup_path: str) -> tuple[str, Path]:
    """Validate cgroup syntax and resolve it below the controller root."""

    if not isinstance(cgroup_path, str):
        raise ResourceGuardError("inherited cgroup path is not a string")
    raw = cgroup_path.strip()
    if not raw:
        raise ResourceGuardError("inherited cgroup path is blank")
    if "\n" in raw or "\x00" in raw:
        raise ResourceGuardError("inherited cgroup path is invalid")
    relative = raw.lstrip("/")
    if ".." in Path(relative).parts:
        raise ResourceGuardError("inherited cgroup path contains a parent component")
    root = Path("/sys/fs/cgroup").resolve()
    directory = (root / relative).resolve(strict=False)
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise ResourceGuardError("inherited cgroup path escapes the controller root") from exc
    return raw, directory


def _cgroup_directory(cgroup_path: str) -> Path:
    """Resolve an existing procfs cgroup path below the controller root."""

    _raw, directory = _canonical_cgroup_path(cgroup_path)
    if not directory.is_dir():
        raise ResourceGuardError(f"inherited cgroup does not exist: {directory}")
    return directory


def _validated_execution_cgroup(
    cgroup_path: str, *, require_exists: bool = True
) -> str:
    """Validate and return a real unified-controller cgroup path."""

    value, directory = _canonical_cgroup_path(cgroup_path)
    if require_exists:
        if not directory.is_dir():
            raise ResourceGuardError(f"inherited cgroup does not exist: {directory}")
    return value


def _execution_cgroup_path_absent(cgroup_path: str) -> bool:
    """Distinguish a removed cgroup path from an invalid/unreadable one."""

    try:
        _value, candidate = _canonical_cgroup_path(cgroup_path)
        candidate.stat()
    except FileNotFoundError:
        return True
    except (OSError, ResourceGuardError):
        return False
    return False


def _cgroup_control(directory: Path, name: str) -> str:
    try:
        value = (directory / name).read_text(encoding="ascii").strip()
    except (OSError, UnicodeError) as exc:
        raise ResourceGuardError(
            f"cannot inspect inherited cgroup control {name}: {exc}"
        ) from exc
    if not value:
        raise ResourceGuardError(f"inherited cgroup control {name} is empty")
    return value


def _cgroup_chain(directory: Path) -> tuple[Path, ...]:
    """Return a cgroup and its controller ancestors up to the mount root."""

    root = Path("/sys/fs/cgroup").resolve()
    chain: list[Path] = []
    current = directory
    while True:
        chain.append(current)
        if current == root:
            return tuple(chain)
        if current.parent == current:
            raise ResourceGuardError("inherited cgroup chain escaped controller root")
        current = current.parent


def _effective_cgroup_limit(
    chain: Sequence[Path], *, name: str, reasons: list[str]
) -> int | None:
    """Read the tightest finite value across a cgroup's ancestor chain."""

    values: list[int] = []
    for directory in chain:
        raw = _cgroup_control(directory, name)
        if raw == "max":
            continue
        try:
            value = int(raw)
        except ValueError:
            reasons.append(f"inherited cgroup {name} is invalid")
            continue
        if value < 1:
            reasons.append(f"inherited cgroup {name} is not positive")
            continue
        values.append(value)
    if not values:
        reasons.append(f"inherited cgroup {name} is unbounded")
        return None
    return min(values)


def _effective_cpu_quota(
    chain: Sequence[Path], reasons: list[str]
) -> int | None:
    values: list[int] = []
    for directory in chain:
        raw = _cgroup_control(directory, "cpu.max")
        parts = raw.split()
        if len(parts) != 2:
            reasons.append("inherited cgroup cpu.max is invalid")
            continue
        if parts[0] == "max":
            continue
        try:
            quota, period = int(parts[0]), int(parts[1])
        except ValueError:
            reasons.append("inherited cgroup cpu.max is invalid")
            continue
        if quota < 1 or period < 1:
            reasons.append("inherited cgroup cpu.max is not positive")
            continue
        # Round upward so a fractional quota above the profile cannot be
        # truncated into an apparently safe integer percentage.
        values.append((quota * 100 + period - 1) // period)
    if not values:
        reasons.append("inherited cgroup cpu.max is unbounded")
        return None
    return min(values)


def _inherited_memory_headroom_reasons(
    chain: Sequence[Path], reserve_bytes: int, reasons: list[str]
) -> None:
    """Require the queue reserve after current usage at every finite ancestor."""

    for directory in chain:
        maximum_raw = _cgroup_control(directory, "memory.max")
        if maximum_raw == "max":
            continue
        current_raw = _cgroup_control(directory, "memory.current")
        try:
            maximum = int(maximum_raw)
        except ValueError:
            reasons.append("inherited cgroup memory.max is invalid")
            continue
        try:
            current = int(current_raw)
        except ValueError:
            reasons.append("inherited cgroup memory.current is invalid")
            continue
        if maximum < 1:
            reasons.append("inherited cgroup memory.max is not positive")
            continue
        if current < 0:
            reasons.append("inherited cgroup memory.current is negative")
            continue
        if current > maximum:
            reasons.append("inherited cgroup memory.current exceeds memory.max")
        elif maximum - current < reserve_bytes:
            reasons.append("inherited cgroup lacks queue memory headroom")


def _inherited_cgroup_reasons(
    profile: ResourceProfile, expected_cgroup: str | None = None
) -> list[str]:
    """Verify finite inherited cgroup controls cover one queue profile.

    A queue child intentionally has no nested systemd scope.  It is therefore
    safe only when the current service cgroup itself exposes finite controls no
    looser than the profile contract.  A tighter effective limit is safe when
    host admission still has the profile's reserve.  This check is separate
    from host admission: ``memory_mib`` is the admission reserve, while the
    runtime ``memory_max_mib``/CPU/tasks values are containment limits.
    """

    current = _self_cgroup_path()
    if expected_cgroup is not None and expected_cgroup != current:
        return [
            "current process cgroup changed before queue admission "
            f"({current!r} != {expected_cgroup!r})"
        ]
    directory = _cgroup_directory(expected_cgroup or current)
    chain = _cgroup_chain(directory)
    reasons: list[str] = []
    memory_high = _effective_cgroup_limit(
        chain, name="memory.high", reasons=reasons
    )
    memory_max = _effective_cgroup_limit(chain, name="memory.max", reasons=reasons)
    swap_max = _effective_cgroup_limit(
        chain, name="memory.swap.max", reasons=reasons
    )
    tasks_max = _effective_cgroup_limit(chain, name="pids.max", reasons=reasons)
    cpu_quota = _effective_cpu_quota(chain, reasons)
    _inherited_memory_headroom_reasons(
        chain, profile.memory_mib * MIB, reasons
    )

    if memory_high is not None and memory_high > profile.memory_high_mib * MIB:
        reasons.append("inherited cgroup memory.high exceeds the queue profile")
    if memory_max is not None and memory_max > profile.memory_max_mib * MIB:
        reasons.append("inherited cgroup memory.max exceeds the queue profile")
    if swap_max is not None and swap_max > profile.memory_swap_max_mib * MIB:
        reasons.append("inherited cgroup memory.swap.max exceeds the queue profile")
    if cpu_quota is not None and cpu_quota > profile.cpu_quota_percent:
        reasons.append("inherited cgroup cpu.max exceeds the queue profile")
    if tasks_max is not None and tasks_max > profile.tasks_max:
        reasons.append("inherited cgroup pids.max exceeds the queue profile")
    return reasons


def _filesystem_evidence(
    purpose: str, path: Path, required_mib: int, required_inodes: int
) -> FilesystemEvidence:
    usage = os.statvfs(path)
    return FilesystemEvidence(
        purpose=purpose,
        path=str(path),
        filesystem=_mount_type(path),
        free_bytes=usage.f_bavail * usage.f_frsize,
        free_inodes=usage.f_favail,
        required_free_bytes=required_mib * MIB,
        required_free_inodes=required_inodes,
    )


def capture_host_evidence(command: GuardedCommand) -> HostEvidence:
    memory_available, swap_free = _memory_values()
    psi_some, psi_full = _psi_values()
    cgroup, current, maximum, oom_events = _cgroup_values()
    profile = command.profile
    probes = (
        ("target", command.target_dir, profile.minimum_target_free_mib),
        ("tmpdir", command.tmp_dir, profile.minimum_tmp_free_mib),
        ("home", Path("/home"), profile.minimum_home_free_mib),
        ("system-tmp", Path("/tmp"), profile.minimum_system_tmp_free_mib),
    )
    filesystems = tuple(
        _filesystem_evidence(name, path, required, profile.minimum_free_inodes)
        for name, path, required in probes
    )
    return HostEvidence(
        captured_unix_ns=time.time_ns(),
        memory_available_bytes=memory_available,
        swap_free_bytes=swap_free,
        memory_psi_some_avg10=psi_some,
        memory_psi_full_avg10=psi_full,
        cgroup_path=cgroup,
        cgroup_memory_current=current,
        cgroup_memory_max=maximum,
        cgroup_oom_events=oom_events,
        filesystems=filesystems,
    )


def _state_path() -> Path:
    return _runtime_root() / "admission-state.json"


def _write_state(value: Mapping[str, Any]) -> None:
    try:
        _atomic_json(_state_path(), value)
    except OSError as exc:
        raise ResourceAdmissionError(
            f"cannot persist resource admission state: {exc}", value
        ) from exc


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _dispatcher_reservation_dir() -> Path:
    directory = _runtime_root().parents[1] / "dispatch-build-reservations"
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    return directory


def _reserve_dispatcher_heavy(
    unit: str, command: GuardedCommand, *, inherited_cgroup: str | None = None
) -> tuple[Path, str]:
    """Join dispatch_build.sh's atomic heavy-capacity authority."""

    directory = _dispatcher_reservation_dir()
    lock_path = directory / ".lock"
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        for existing in directory.glob("*.json"):
            try:
                record = json.loads(existing.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ResourceAdmissionError(
                    f"malformed host capacity reservation blocks admission: {existing}"
                ) from exc
            if not isinstance(record, Mapping):
                raise ResourceAdmissionError(
                    f"malformed host capacity reservation blocks admission: {existing}"
                )
            if record.get("kind") == "heavy":
                owner_pid = record.get("owner_pid")
                owner_start = record.get("owner_start_time")
                if (
                    isinstance(owner_pid, bool)
                    or isinstance(owner_start, bool)
                    or not isinstance(owner_pid, int)
                    or not isinstance(owner_start, int)
                    or owner_pid < 1
                    or owner_start < 1
                ):
                    raise ResourceAdmissionError(
                        "host capacity reservation has no owner PID/start fence"
                    )
                if _owner_fence_is_live(owner_pid, owner_start):
                    raise ResourceAdmissionError(
                        "the fleet dispatcher already holds the host-wide heavy slot"
                    )
                _reclaim_stale_dispatcher_reservation(existing, record)
        destination = directory / f"{unit}.json"
        sentinel = _runtime_root() / f"{unit}.result.json"
        owner_start = process_start_time()
        owner_token = uuid.uuid4().hex
        base = {
            "schema": _STATE_SCHEMA,
            "unit": unit,
            "kind": "heavy",
            "target_dir": str(command.target_dir),
            "admitted_unix_ns": time.time_ns(),
            "owner_pid": os.getpid(),
            "owner_start_time": owner_start,
            "owner_token": owner_token,
            "inherited_cgroup": inherited_cgroup,
            "execution_cgroup": inherited_cgroup,
        }
        _atomic_json(sentinel, {**base, "status": "running"})
        _atomic_json(
            destination,
            {
                **base,
                "sentinel": str(sentinel),
                "sentinel_base": base,
                "reservation": str(destination),
            },
        )
        return destination, owner_token


def _bind_dispatcher_execution_cgroup(
    reservation: Path | None,
    expected_owner_token: str | None,
    unit: str,
    execution_cgroup: str,
) -> None:
    """Fence the reservation to the scope cgroup returned by systemd."""

    if reservation is None:
        return
    directory = reservation.parent
    try:
        with (directory / ".lock").open("a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            record = json.loads(reservation.read_text(encoding="utf-8"))
            if not isinstance(record, Mapping):
                raise ResourceGuardError(
                    "host capacity reservation is malformed before containment binding"
                )
            owner_pid = record.get("owner_pid")
            owner_start = record.get("owner_start_time")
            owner_token = record.get("owner_token")
            if not (
                isinstance(owner_pid, int)
                and not isinstance(owner_pid, bool)
                and isinstance(owner_start, int)
                and not isinstance(owner_start, bool)
                and owner_pid == os.getpid()
                and _owner_fence_is_live(owner_pid, owner_start)
                and isinstance(owner_token, str)
                and owner_token == expected_owner_token
                and record.get("unit") == unit
            ):
                raise ResourceGuardError(
                    "host capacity reservation owner changed before containment binding"
                )
            record["execution_cgroup"] = execution_cgroup
            sentinel_base = record.get("sentinel_base")
            if isinstance(sentinel_base, Mapping):
                updated_base = dict(sentinel_base)
            else:
                updated_base = dict(record)
            updated_base["execution_cgroup"] = execution_cgroup
            record["sentinel_base"] = updated_base
            _atomic_json(reservation, record)
            sentinel_value = record.get("sentinel")
            if isinstance(sentinel_value, str) and sentinel_value:
                _atomic_json(
                    Path(sentinel_value),
                    {**updated_base, "status": "running"},
                )
    except ResourceGuardError:
        raise
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ResourceGuardError(
            "could not bind the heavy reservation to the execution cgroup"
        ) from exc


def _owner_fence_is_live(pid: int, start_time: int) -> bool:
    """Return whether a reservation owner still names this exact process."""

    try:
        return process_start_time(pid) == start_time
    except ResourceGuardError as exc:
        if not Path(f"/proc/{pid}").exists():
            return False
        raise ResourceAdmissionError(
            f"cannot verify host capacity reservation owner {pid}"
        ) from exc


def _reclaim_stale_dispatcher_reservation(
    reservation: Path, record: Mapping[str, Any]
) -> None:
    """Close an owner-fenced dead reservation before admitting a new one."""

    if "execution_cgroup" not in record:
        raise ResourceAdmissionError(
            "stale host capacity reservation has no execution containment proof"
        )
    execution_cgroup = record.get("execution_cgroup")
    if not isinstance(execution_cgroup, str) or not execution_cgroup.strip():
        raise ResourceAdmissionError(
            "stale host capacity reservation has invalid execution cgroup"
        )
    _require_empty_execution_cgroup(execution_cgroup)

    sentinel_value = record.get("sentinel")
    if not isinstance(sentinel_value, str) or not sentinel_value:
        raise ResourceAdmissionError(
            f"stale host capacity reservation has no result sentinel: {reservation}"
        )
    sentinel = Path(sentinel_value)
    base = record.get("sentinel_base")
    if not isinstance(base, Mapping):
        base = dict(record)
    _atomic_json(
        sentinel,
        {
            **dict(base),
            "status": "indeterminate",
            "finished_unix_ns": time.time_ns(),
            "reason": "reservation_owner_not_live",
        },
    )
    try:
        reservation.unlink()
    except OSError as exc:
        raise ResourceAdmissionError(
            f"could not reclaim stale host capacity reservation: {reservation}"
        ) from exc


def _require_empty_execution_cgroup(cgroup_path: str) -> None:
    """Require cgroup events and every descendant process list to be empty."""

    try:
        directory = _cgroup_directory(cgroup_path)
    except ResourceGuardError as exc:
        if _execution_cgroup_path_absent(cgroup_path):
            raise _ExecutionCgroupAbsent(
                "execution cgroup was removed before closure could be checked"
            ) from exc
        raise ResourceAdmissionError(
            "cannot prove stale execution cgroup containment is closed"
        ) from exc
    try:
        events = (directory / "cgroup.events").read_text(encoding="ascii")
    except FileNotFoundError as exc:
        if not _directory_path_absent(directory):
            raise ResourceAdmissionError(
                "execution cgroup events are unavailable"
            ) from exc
        raise _ExecutionCgroupAbsent(
            "execution cgroup was removed before closure could be checked"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ResourceAdmissionError(
            "cannot prove stale execution cgroup containment is closed"
        ) from exc
    values: dict[str, str] = {}
    for line in events.splitlines():
        name, separator, value = line.partition(" ")
        if separator and name:
            values[name] = value.strip()
    if values.get("populated") != "0":
        raise ResourceAdmissionError(
            "stale execution cgroup remains populated; capacity stays reserved"
        )
    try:
        process_files = {directory / "cgroup.procs"}
        process_files.update(directory.rglob("cgroup.procs"))
        if not process_files:
            raise OSError("cgroup process lists are unavailable")
        for process_file in process_files:
            raw = process_file.read_text(encoding="ascii").strip()
            if raw:
                raise ResourceAdmissionError(
                    "stale execution cgroup still contains a process"
                )
    except FileNotFoundError as exc:
        if not _directory_path_absent(directory):
            raise ResourceAdmissionError(
                "execution cgroup process lists are unavailable"
            ) from exc
        raise _ExecutionCgroupAbsent(
            "execution cgroup was removed before closure could be checked"
        ) from exc
    except (OSError, UnicodeError) as exc:
        raise ResourceAdmissionError(
            "cannot prove stale execution cgroup process lists are empty"
        ) from exc


def _directory_path_absent(directory: Path) -> bool:
    """Return true only when a previously resolved cgroup directory is gone."""

    try:
        directory.stat()
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return False


def _scope_is_inactive(unit: str, execution_cgroup: str) -> bool:
    """Use systemd's terminal unit state when it removed an empty cgroup."""

    values: dict[str, str] = {}
    for property_name in ("ActiveState", "SubState", "ControlGroup"):
        try:
            result = subprocess.run(  # nosec B603 B607 - fixed systemctl argv
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    f"--property={property_name}",
                    "--value",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=5,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            return False
        if result.returncode != 0:
            return False
        values[property_name] = result.stdout.strip()
    if values.get("ActiveState") != "inactive" or values.get("SubState") != "dead":
        return False
    control_group = values.get("ControlGroup", "")
    return not control_group or control_group == execution_cgroup


def _execution_cgroup_closed(unit: str, execution_cgroup: str) -> bool:
    """Require an empty cgroup, or its systemd terminal-state proof."""

    try:
        _require_empty_execution_cgroup(execution_cgroup)
        return True
    except _ExecutionCgroupAbsent:
        return _scope_is_inactive(unit, execution_cgroup)
    except ResourceGuardError:
        return False


def _release_dispatcher_heavy(
    reservation: Path | None,
    expected_owner_token: str | None,
    *,
    expected_unit: str | None = None,
) -> None:
    if reservation is None:
        return
    directory = reservation.parent
    with (directory / ".lock").open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            record = json.loads(reservation.read_text(encoding="utf-8"))
            if not isinstance(record, Mapping):
                return
            sentinel = Path(str(record["sentinel"]))
            owner_pid = record.get("owner_pid")
            owner_start = record.get("owner_start_time")
            owner_token = record.get("owner_token")
            if not (
                isinstance(owner_pid, int)
                and isinstance(owner_start, int)
                and owner_pid == os.getpid()
                and _owner_fence_is_live(owner_pid, owner_start)
                and isinstance(owner_token, str)
                and owner_token == expected_owner_token
                and (expected_unit is None or record.get("unit") == expected_unit)
            ):
                return
        except (OSError, KeyError, TypeError, json.JSONDecodeError):
            # An ownerless or malformed record is not ours to delete.  It stays
            # as a visible capacity blocker until its authority repairs it.
            return
        reservation.unlink(missing_ok=True)
        if sentinel is not None:
            sentinel.unlink(missing_ok=True)


def _read_blocked() -> bool:
    try:
        value = json.loads(_state_path().read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return value.get("schema") == _STATE_SCHEMA and value.get("blocked") is True


def _admission_reasons(command: GuardedCommand, evidence: HostEvidence) -> list[str]:
    profile = command.profile
    required_memory = (profile.memory_max_mib + 8_192) * MIB
    required_swap = min(profile.memory_swap_max_mib, 4_096) * MIB
    reasons: list[str] = []
    if evidence.memory_available_bytes < required_memory:
        reasons.append("available memory is below the guarded headroom")
    if evidence.swap_free_bytes < required_swap:
        reasons.append("free swap is below the guarded reserve")
    if (
        evidence.memory_psi_some_avg10 >= _MEMORY_PSI_HIGH[0]
        or evidence.memory_psi_full_avg10 >= _MEMORY_PSI_HIGH[1]
    ):
        reasons.append("memory PSI is above the admission threshold")
    if (
        evidence.cgroup_memory_current is not None
        and evidence.cgroup_memory_max is not None
        and evidence.cgroup_memory_max - evidence.cgroup_memory_current
        < profile.memory_mib * MIB
    ):
        reasons.append("the user cgroup lacks MemoryMax headroom")
    reasons.extend(_storage_reasons(evidence.filesystems))
    return reasons


def _storage_reasons(filesystems: Sequence[FilesystemEvidence]) -> list[str]:
    """Report byte and inode exhaustion independently for every probe path."""

    reasons: list[str] = []
    for filesystem in filesystems:
        if filesystem.free_bytes < filesystem.required_free_bytes:
            reasons.append(f"{filesystem.purpose} lacks free bytes")
        if filesystem.free_inodes < filesystem.required_free_inodes:
            reasons.append(f"{filesystem.purpose} lacks free inodes")
    return reasons


def _admit(command: GuardedCommand) -> HostEvidence:
    evidence = capture_host_evidence(command)
    reasons = _admission_reasons(command, evidence)
    was_blocked = _read_blocked()
    recovered = (
        evidence.memory_psi_some_avg10 <= _MEMORY_PSI_LOW[0]
        and evidence.memory_psi_full_avg10 <= _MEMORY_PSI_LOW[1]
        and not reasons
    )
    if was_blocked and not recovered and not reasons:
        reasons.append("resource guard remains blocked until low PSI watermark")
    blocked = bool(reasons)
    state = {
        "schema": _STATE_SCHEMA,
        "blocked": blocked,
        "updated_unix_ns": time.time_ns(),
        "reasons": reasons,
        "evidence": evidence.as_dict(),
    }
    _write_state(state)
    if blocked:
        raise ResourceAdmissionError("; ".join(reasons), state)
    return evidence


def _profile_for(name: str, cargo: bool) -> ResourceProfile:
    # Command shape is authoritative: a Cargo command cannot escape into the
    # light pool by declaring ``light-check`` in a manifest.
    return DEFAULT_RESOURCE_PROFILES.resolve("rust-build" if cargo else name)


def _bounded_cargo_environment(
    env: dict[str, str], command: GuardedCommand, cargo: bool
) -> None:
    # Every guarded process gets an explicit disk-backed temporary directory.
    # Cargo adds its target and thread controls on top of this common boundary.
    env["TMPDIR"] = str(command.tmp_dir)
    env["RUNNER_TEMP"] = str(command.tmp_dir)
    if not cargo:
        return
    jobs = command.profile.cargo_jobs
    raw_jobs = env.get("CARGO_BUILD_JOBS", str(jobs))
    try:
        requested = int(raw_jobs)
    except ValueError as exc:
        raise ResourceGuardError("CARGO_BUILD_JOBS must be a positive integer") from exc
    if requested < 1:
        raise ResourceGuardError("CARGO_BUILD_JOBS must be a positive integer")
    env["CARGO_BUILD_JOBS"] = str(min(requested, jobs))
    env["CARGO_INCREMENTAL"] = "0"
    env["CARGO_TARGET_DIR"] = str(command.target_dir)
    for name in ("RUST_TEST_THREADS", "RAYON_NUM_THREADS", "TOKIO_WORKER_THREADS"):
        try:
            requested_threads = int(env.get(name, jobs))
        except ValueError as exc:
            raise ResourceGuardError(f"{name} must be a positive integer") from exc
        if requested_threads < 1:
            raise ResourceGuardError(f"{name} must be a positive integer")
        env[name] = str(min(requested_threads, jobs))
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        env[name] = "1"


def prepare_environment(command: GuardedCommand) -> dict[str, str]:
    """Return the command environment after guard-owned bounds are applied."""

    environment = dict(command.environment)
    _bounded_cargo_environment(environment, command, command.cargo)
    return environment


def _properties(profile: ResourceProfile, timeout: int) -> tuple[str, ...]:
    runtime = min(profile.runtime_max_seconds, timeout)
    properties: tuple[str, ...] = (
        "MemoryAccounting=yes",
        f"MemoryHigh={profile.memory_high_mib}M",
        f"MemoryMax={profile.memory_max_mib}M",
        f"MemorySwapMax={profile.memory_swap_max_mib}M",
        f"CPUQuota={profile.cpu_quota_percent}%",
        "CPUWeight=100",
        "TasksAccounting=yes",
        f"TasksMax={profile.tasks_max}",
        f"RuntimeMaxSec={runtime}",
        "OOMPolicy=stop",
        "Slice=app.slice",
    )
    if profile.managed_oom_preference:
        properties += (
            f"ManagedOOMPreference={profile.managed_oom_preference}",
        )
    return properties


def _drain_output(
    stream: Any,
    stream_name: StreamName,
    sink: BoundedLogSink,
    errors: list[BaseException],
) -> None:
    try:
        while True:
            chunk = stream.read(64 * 1024)
            if not chunk:
                return
            sink.write(stream_name, chunk)
    except BaseException as exc:  # reader failures invalidate the measurement
        errors.append(exc)


def _systemd_user_manager_cgroup() -> str:
    """Return the user manager cgroup used to derive a short-lived scope path."""

    result = subprocess.run(  # nosec B603 B607 - fixed systemctl argv
        [
            "systemctl",
            "--user",
            "show",
            "--property=ControlGroup",
            "--value",
        ],
        capture_output=True,
        check=False,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise ResourceGuardError("systemd user manager cgroup is unavailable")
    return _validated_execution_cgroup(result.stdout.strip())


def _expected_scope_cgroup(unit: str) -> str:
    """Derive the explicit app.slice path for a scope already known to systemd."""

    if not unit or "/" in unit or not unit.endswith(".scope"):
        raise ResourceGuardError("systemd scope unit name is invalid")
    manager = _systemd_user_manager_cgroup().rstrip("/")
    return _validated_execution_cgroup(
        f"{manager}/app.slice/{unit}", require_exists=False
    )


def _scope_execution_cgroup(unit: str) -> str:
    """Read the real cgroup path for a transient systemd scope."""

    last_error: BaseException | None = None
    for attempt in range(20):
        try:
            result = subprocess.run(  # nosec B603 B607 - fixed systemctl argv
                [
                    "systemctl",
                    "--user",
                    "show",
                    unit,
                    "--property=ControlGroup",
                    "--value",
                ],
                capture_output=True,
                check=False,
                text=True,
                timeout=1,
            )
            raw = ""
            if result.returncode == 0:
                raw = result.stdout.strip()
                if raw:
                    try:
                        return _validated_execution_cgroup(raw)
                    except ResourceGuardError as exc:
                        last_error = exc
                else:
                    last_error = ResourceGuardError(
                        "systemd returned an empty execution cgroup"
                    )
            else:
                last_error = ResourceGuardError(
                    f"systemd could not show scope {unit}"
                )
            if not raw:
                try:
                    expected = _expected_scope_cgroup(unit)
                    if _scope_is_inactive(unit, expected):
                        return expected
                except (
                    OSError,
                    subprocess.SubprocessError,
                    ResourceGuardError,
                    ValueError,
                ) as exc:
                    last_error = exc
        except (OSError, subprocess.SubprocessError, ValueError) as exc:
            last_error = exc
        if attempt < 19:
            time.sleep(0.05)
    raise ResourceGuardError(
        f"could not verify the execution cgroup for systemd scope {unit}"
    ) from last_error


def _stop_scope(
    unit: str,
    process: subprocess.Popen[bytes],
    execution_cgroup: str | None,
) -> bool:
    """Stop a scope and prove its cgroup is empty before capacity release."""

    stopped = True
    try:
        subprocess.run(  # nosec B603 B607 - fixed systemctl argv
            ["systemctl", "--user", "kill", "--signal=TERM", unit],
            capture_output=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError, ValueError):
        stopped = False
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(  # nosec B603 B607 - fixed systemctl argv
                ["systemctl", "--user", "kill", "--signal=KILL", unit],
                capture_output=True,
                check=False,
                timeout=30,
            )
        except (OSError, subprocess.SubprocessError, ValueError):
            stopped = False
        try:
            process.kill()
        except (OSError, ProcessLookupError):
            stopped = False
        try:
            process.wait(timeout=30)
        except (OSError, subprocess.SubprocessError):
            stopped = False
    except (OSError, subprocess.SubprocessError):
        stopped = False
    try:
        if process.poll() is None:
            stopped = False
    except (AttributeError, OSError):
        stopped = False
    if not execution_cgroup:
        return False
    return stopped and _execution_cgroup_closed(unit, execution_cgroup)


def _run_bounded_scope(
    argv: Sequence[str],
    *,
    unit: str,
    command: GuardedCommand,
    environment: Mapping[str, str],
    admission: GuardAdmission | None = None,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    sink = BoundedLogSink(
        max_stdout_bytes=2 * MIB,
        max_stderr_bytes=2 * MIB,
        terminal_tail_bytes=64 * 1024,
    )
    process = subprocess.Popen(  # nosec B603 - fixed argv, never shell=True
        argv,
        cwd=command.workdir,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    errors: list[BaseException] = []
    readers = tuple(
        threading.Thread(
            target=_drain_output,
            args=(stream, name, sink, errors),
            daemon=True,
            name=f"repository-resource-guard-{name}",
        )
        for name, stream in (("stdout", process.stdout), ("stderr", process.stderr))
        if stream is not None
    )
    for reader in readers:
        reader.start()
    execution_cgroup: str | None = None
    try:
        execution_cgroup = _scope_execution_cgroup(unit)
        if admission is not None:
            admission.bind_execution_cgroup(execution_cgroup, allow_missing=True)
    except BaseException as exc:
        if admission is not None:
            admission.retain_reservation_on_release()
        closed = _stop_scope(unit, process, execution_cgroup)
        if not closed and admission is not None:
            admission.retain_reservation_on_release()
        for reader in readers:
            reader.join(timeout=30)
        sink.close()
        raise ResourceGuardError(
            f"could not establish standalone execution containment for {unit}"
        ) from exc
    try:
        returncode = process.wait(timeout=command.timeout + 60)
    except subprocess.TimeoutExpired as exc:
        closed = _stop_scope(
            unit,
            process,
            admission.execution_cgroup if admission is not None else execution_cgroup,
        )
        if not closed:
            if admission is not None:
                admission.retain_reservation_on_release()
            raise ResourceGuardError(
                "guarded scope termination did not prove empty containment"
            ) from exc
        if admission is not None:
            admission.mark_execution_closed()
        raise ResourceGuardError(
            f"guarded scope exceeded {command.timeout}s plus stop grace"
        ) from exc
    finally:
        for reader in readers:
            reader.join(timeout=30)
        sink.close()
    if admission is not None:
        try:
            admission.verify_execution_closed()
        except BaseException as exc:
            admission.retain_reservation_on_release()
            raise ResourceGuardError(
                "guarded scope completion did not prove empty containment"
            ) from exc
    if errors:
        raise ResourceGuardError(f"guarded output reader failed: {errors[0]}")
    stream_names: tuple[StreamName, ...] = ("stdout", "stderr")
    snapshots = {name: sink.snapshot(name) for name in stream_names}
    output: dict[str, Any] = {
        name: {
            "total_bytes": snapshot.total_bytes,
            "retained_bytes": snapshot.retained_bytes,
            "discarded_bytes": snapshot.discarded_bytes,
            "truncated": snapshot.truncated,
            "sha256": snapshot.content_address,
        }
        for name, snapshot in snapshots.items()
    }
    return (
        subprocess.CompletedProcess(
            argv,
            returncode,
            sink.tail_text("stdout"),
            sink.tail_text("stderr"),
        ),
        output,
    )


def _validate_fixed_argv(argv: Sequence[str], timeout: int) -> None:
    if not argv or any(not isinstance(part, str) or not part for part in argv):
        raise ResourceGuardError("guarded command must be non-empty fixed argv")
    if timeout < 1:
        raise ResourceGuardError("guarded timeout must be positive")


def _cargo_job_values(argv: Sequence[str]) -> tuple[str, ...]:
    values: list[str] = []
    for index, part in enumerate(argv):
        if part in {"-j", "--jobs"}:
            if index + 1 >= len(argv):
                raise ResourceGuardError(f"missing value after Cargo {part}")
            values.append(argv[index + 1])
        elif part.startswith("--jobs="):
            values.append(part.split("=", 1)[1])
        elif re.fullmatch(r"-j\d+", part):
            values.append(part[2:])
    return tuple(values)


def _validate_cargo_jobs(argv: Sequence[str], profile: ResourceProfile) -> None:
    for raw_jobs in _cargo_job_values(argv):
        try:
            requested_jobs = int(raw_jobs)
        except ValueError as exc:
            raise ResourceGuardError("Cargo --jobs must be an integer") from exc
        if requested_jobs > profile.cargo_jobs or requested_jobs < 1:
            raise ResourceGuardError(
                f"Cargo jobs cannot exceed the profile cap {profile.cargo_jobs}"
            )


def _validate_cargo_target(argv: Sequence[str], cwd: Path, target: Path) -> None:
    for index, part in enumerate(argv):
        raw_target = None
        if part == "--target-dir":
            if index + 1 >= len(argv):
                raise ResourceGuardError("missing value after Cargo --target-dir")
            raw_target = argv[index + 1]
        elif part.startswith("--target-dir="):
            raw_target = part.split("=", 1)[1]
        if raw_target is None:
            continue
        declared = Path(raw_target)
        declared = declared if declared.is_absolute() else cwd / declared
        if declared.resolve() != target:
            raise ResourceGuardError(
                "Cargo --target-dir differs from the guarded target directory"
            )


def _prepare_paths(
    workdir: Path | str, target_dir: Path | str, tmp_dir: Path | str
) -> tuple[Path, Path, Path]:
    cwd = Path(workdir).resolve(strict=True)
    target = Path(target_dir).resolve()
    temporary = Path(tmp_dir).resolve()
    try:
        target.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary.mkdir(mode=0o700, parents=True, exist_ok=True)
    except OSError as exc:
        raise ResourceGuardError(f"cannot create guarded target/TMPDIR: {exc}") from exc
    if _mount_type(temporary) in _DISK_BACKED_FS:
        raise ResourceGuardError(f"TMPDIR must be disk-backed: {temporary}")
    return cwd, target, temporary


def prepare_command(
    argv: Sequence[str],
    *,
    workdir: Path | str,
    target_dir: Path | str,
    tmp_dir: Path | str,
    profile_name: str,
    timeout: int,
    env: Mapping[str, str] | None = None,
    force_heavy: bool = False,
    force_cargo: bool = False,
    profile_override: ResourceProfile | None = None,
) -> GuardedCommand:
    _validate_fixed_argv(argv, timeout)
    cwd, target, temporary = _prepare_paths(workdir, target_dir, tmp_dir)
    cargo = force_cargo or is_cargo_shaped(argv)
    profile = profile_override or _profile_for(profile_name, cargo)
    if cargo:
        _validate_cargo_jobs(argv, profile)
        _validate_cargo_target(argv, cwd, target)
    return GuardedCommand(
        argv=tuple(argv),
        workdir=cwd,
        target_dir=target,
        tmp_dir=temporary,
        profile=profile,
        heavy=force_heavy or cargo or profile.concurrency_limit == 1,
        cargo=cargo,
        timeout=timeout,
        environment=dict(env or os.environ),
    )


def admit_guarded(
    command: GuardedCommand, *, inherited_cgroup: str | None = None
) -> GuardAdmission:
    """Reserve host capacity and capture health evidence before launching.

    ``inherited_cgroup`` is used by callers such as the merge-queue runner
    whose service already supplies containment.  It verifies the current
    cgroup's finite controls without creating a nested scope; callers must keep
    the returned context alive until the child exits.
    """

    unit = f"rm-resource-guard-{uuid.uuid4().hex[:12]}.scope"
    lock_handle = None
    dispatcher_reservation: Path | None = None
    reservation_token: str | None = None
    try:
        if command.heavy:
            lock_path = _runtime_root() / "heavy.lock"
            try:
                lock_handle = lock_path.open("a+")
            except OSError as exc:
                raise ResourceGuardError(
                    f"cannot open host-wide heavy reservation: {exc}"
                ) from exc
            try:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                lock_handle.close()
                lock_handle = None
                raise ResourceAdmissionError(
                    "the host-wide heavy resource reservation is already held"
                ) from exc
            dispatcher_reservation, reservation_token = _reserve_dispatcher_heavy(
                unit, command, inherited_cgroup=inherited_cgroup
            )

        if inherited_cgroup is not None:
            reasons = _inherited_cgroup_reasons(command.profile, inherited_cgroup)
            if reasons:
                raise ResourceAdmissionError(
                    "; ".join(reasons),
                    {
                        "schema": _STATE_SCHEMA,
                        "inherited_cgroup": inherited_cgroup,
                        "reasons": reasons,
                    },
                )
        before = _admit(command)
        environment = prepare_environment(command)
        return GuardAdmission(
            command=command,
            unit=unit,
            before=before,
            environment=environment,
            inherited_cgroup=inherited_cgroup,
            execution_cgroup=inherited_cgroup,
            _lock_handle=lock_handle,
            _dispatcher_reservation=dispatcher_reservation,
            _reservation_token=reservation_token,
        )
    except BaseException:
        try:
            _release_dispatcher_heavy(dispatcher_reservation, reservation_token)
        finally:
            if lock_handle is not None:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
                lock_handle.close()
        raise


def admit_inherited(
    command: GuardedCommand, *, expected_cgroup: str
) -> GuardAdmission:
    """Admit a child that must remain in an existing bounded service cgroup."""

    return admit_guarded(command, inherited_cgroup=expected_cgroup)


def run_guarded(command: GuardedCommand) -> GuardedRunResult:
    """Admit and synchronously execute a fixed-argv command in a systemd scope."""

    try:
        with admit_guarded(command) as admission:
            unit = admission.unit
            properties = _properties(command.profile, command.timeout)
            systemd_argv = [
                "systemd-run",
                "--user",
                "--scope",
                "--quiet",
                f"--unit={unit.removesuffix('.scope')}",
                *[item for value in properties for item in ("--property", value)],
                "--",
                *command.argv,
            ]
            completed, output = _run_bounded_scope(
                systemd_argv,
                unit=unit,
                command=command,
                environment=admission.environment,
                admission=admission,
            )
            after = capture_host_evidence(command)
            return GuardedRunResult(
                completed=completed,
                unit=unit,
                profile=command.profile.name,
                heavy=command.heavy,
                properties=properties,
                before=admission.before,
                after=after,
                output=output,
                execution_cgroup=admission.execution_cgroup,
            )
    except OSError as exc:
        raise ResourceGuardError(f"could not start guarded systemd scope: {exc}") from exc


__all__ = [
    "FilesystemEvidence",
    "GuardAdmission",
    "GuardedCommand",
    "GuardedRunResult",
    "HostEvidence",
    "ResourceAdmissionError",
    "ResourceGuardError",
    "capture_host_evidence",
    "current_cgroup_path",
    "default_guard_paths",
    "admit_guarded",
    "admit_inherited",
    "is_cargo_shaped",
    "prepare_environment",
    "process_start_time",
    "inherited_admission_active",
    "verified_inherited_admission",
    "prepare_command",
    "run_guarded",
]
