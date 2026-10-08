"""Windows process containment for supervised runners (the cgroup equivalent).

On Linux a supervised child must stay in the runner's systemd cgroup.  The
Windows mechanism is a Job object: the runner joins one private job (nested
jobs are supported since Windows 8) and every descendant inherits membership.
The job grants no breakaway, so a child cannot leave it; membership is still
observed per process so a violation fails closed exactly like a cgroup escape.
"""

from __future__ import annotations

import ctypes as c
import sys
import threading
from collections.abc import Callable
from typing import Any

_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
_SYNCHRONIZE = 0x100000
_STILL_ACTIVE = 259
_SNAPSHOT_PROCESS = 0x2
_INVALID_HANDLE = c.c_void_p(-1).value


class _ProcessEntry(c.Structure):
    _fields_ = [
        ("dwSize", c.c_uint32),
        ("cntUsage", c.c_uint32),
        ("th32ProcessID", c.c_uint32),
        ("th32DefaultHeapID", c.c_size_t),
        ("th32ModuleID", c.c_uint32),
        ("cntThreads", c.c_uint32),
        ("th32ParentProcessID", c.c_uint32),
        ("pcPriClassBase", c.c_long),
        ("dwFlags", c.c_uint32),
        ("szExeFile", c.c_wchar * 260),
    ]


class _Kernel:
    """kernel32 binding for job membership and process enumeration."""

    api: Any
    last_error: Callable[[], int]

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows process containment requires Windows")
        api = c.WinDLL("kernel32", use_last_error=True)
        handle = c.c_void_p
        signatures: tuple[tuple[str, Any, list[Any]], ...] = (
            ("CreateJobObjectW", handle, [c.c_void_p, c.c_wchar_p]),
            ("AssignProcessToJobObject", c.c_int32, [handle, handle]),
            ("GetCurrentProcess", handle, []),
            ("OpenProcess", handle, [c.c_uint32, c.c_int32, c.c_uint32]),
            ("IsProcessInJob", c.c_int32, [handle, handle, c.POINTER(c.c_int32)]),
            ("GetExitCodeProcess", c.c_int32, [handle, c.POINTER(c.c_uint32)]),
            ("CloseHandle", c.c_int32, [handle]),
            ("CreateToolhelp32Snapshot", handle, [c.c_uint32, c.c_uint32]),
            ("Process32FirstW", c.c_int32, [handle, c.POINTER(_ProcessEntry)]),
            ("Process32NextW", c.c_int32, [handle, c.POINTER(_ProcessEntry)]),
        )
        _bind(
            *((api, name, result, arguments) for name, result, arguments in signatures)
        )
        self.api = api
        self.last_error = c.get_last_error

    def error(self, operation: str) -> OSError:
        code = self.last_error()
        return OSError(0, f"{operation} failed (Win32 {code})", None, code)


_LOCK = threading.Lock()
_STATE: dict[str, Any] = {}


def _kernel() -> _Kernel:
    with _LOCK:
        kernel = _STATE.get("kernel")
        if kernel is None:
            kernel = _STATE["kernel"] = _Kernel()
        return kernel


def _runner_job() -> int:
    """Create the runner's private job once and join it."""
    kernel = _kernel()
    with _LOCK:
        job = _STATE.get("job")
        if job is not None:
            return int(job)
        created = kernel.api.CreateJobObjectW(None, None)
        if not created:
            raise kernel.error("CreateJobObjectW")
        if not kernel.api.AssignProcessToJobObject(
            created, kernel.api.GetCurrentProcess()
        ):
            error = kernel.error("AssignProcessToJobObject")
            kernel.api.CloseHandle(created)
            raise error
        _STATE["job"] = int(created)
        return int(created)


def containment_identity(pid: int) -> str:
    """The containment unit ``pid`` runs in: the runner job, or outside it."""
    job = _runner_job()
    kernel = _kernel()
    process = kernel.api.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not process:
        raise kernel.error(f"OpenProcess({pid})")
    try:
        member = c.c_int32()
        if not kernel.api.IsProcessInJob(process, job, c.byref(member)):
            raise kernel.error(f"IsProcessInJob({pid})")
    finally:
        kernel.api.CloseHandle(process)
    return f"job:{job:x}" if member.value else f"outside-runner-job:{pid}"


def process_exists(pid: int) -> bool:
    """Whether ``pid`` names a process that has not exited."""
    kernel = _kernel()
    process = kernel.api.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if not process:
        return False
    try:
        code = c.c_uint32()
        if not kernel.api.GetExitCodeProcess(process, c.byref(code)):
            return True
        return code.value == _STILL_ACTIVE
    finally:
        kernel.api.CloseHandle(process)


def process_parents() -> dict[int, int]:
    """Map every live process id to its parent id."""
    kernel = _kernel()
    snapshot = kernel.api.CreateToolhelp32Snapshot(_SNAPSHOT_PROCESS, 0)
    if not snapshot or snapshot == _INVALID_HANDLE:
        raise kernel.error("CreateToolhelp32Snapshot")
    parents: dict[int, int] = {}
    try:
        entry = _ProcessEntry()
        entry.dwSize = c.sizeof(_ProcessEntry)
        more = kernel.api.Process32FirstW(snapshot, c.byref(entry))
        while more:
            parents[int(entry.th32ProcessID)] = int(entry.th32ParentProcessID)
            more = kernel.api.Process32NextW(snapshot, c.byref(entry))
    finally:
        kernel.api.CloseHandle(snapshot)
    return parents


_CREATE_SUSPENDED = 0x4


class _JobKernel:
    """Bindings for per-child Job objects (the process-group equivalent)."""

    api: Any
    nt: Any
    last_error: Callable[[], int]

    def __init__(self) -> None:
        if sys.platform != "win32":
            raise OSError("Windows process containment requires Windows")
        self.api = c.WinDLL("kernel32", use_last_error=True)
        self.nt = c.WinDLL("ntdll", use_last_error=True)
        handle = c.c_void_p
        _bind(
            (self.api, "CreateJobObjectW", handle, [c.c_void_p, c.c_wchar_p]),
            (self.api, "AssignProcessToJobObject", c.c_int32, [handle, handle]),
            (self.api, "TerminateJobObject", c.c_int32, [handle, c.c_uint32]),
            (self.api, "CloseHandle", c.c_int32, [handle]),
            (self.nt, "NtResumeProcess", c.c_int32, [handle]),
        )
        self.last_error = c.get_last_error


def _bind(*signatures: tuple[Any, str, Any, list[Any]]) -> None:
    for dll, name, result, arguments in signatures:
        function = getattr(dll, name)
        function.restype = result
        function.argtypes = arguments


def _job_kernel() -> _JobKernel:
    with _LOCK:
        kernel = _STATE.get("job_kernel")
        if kernel is None:
            kernel = _STATE["job_kernel"] = _JobKernel()
        return kernel


class ProcessJob:
    """A child and every descendant it starts, held in one Job object.

    The child is created suspended, assigned to a fresh job, and only then
    resumed, so no descendant can start outside the job.  Terminating the job
    ends the whole tree, which is what POSIX ``killpg`` gives a session.
    """

    def __init__(self, job: int) -> None:
        self._job: int | None = job

    @classmethod
    def contain(cls, process_handle: int) -> ProcessJob:
        """Assign a suspended process to a fresh job, then resume it."""
        kernel = _job_kernel()
        job = kernel.api.CreateJobObjectW(None, None)
        if not job:
            raise OSError(0, "CreateJobObjectW failed", None, kernel.last_error())
        if not kernel.api.AssignProcessToJobObject(job, process_handle):
            code = kernel.last_error()
            kernel.api.CloseHandle(job)
            raise OSError(0, "AssignProcessToJobObject failed", None, code)
        status = kernel.nt.NtResumeProcess(process_handle)
        if status != 0:
            kernel.api.TerminateJobObject(job, 1)
            kernel.api.CloseHandle(job)
            raise OSError(
                f"NtResumeProcess failed (NTSTATUS 0x{status & 0xFFFFFFFF:08x})"
            )
        return cls(int(job))

    def terminate(self) -> None:
        """End every process in the job (idempotent)."""
        if self._job is not None:
            _job_kernel().api.TerminateJobObject(self._job, 1)

    def close(self) -> None:
        job, self._job = self._job, None
        if job is not None:
            _job_kernel().api.CloseHandle(job)


def contain_suspended(process: Any) -> ProcessJob:
    """Contain a child started with ``CREATE_SUSPENDED`` and let it run."""
    return ProcessJob.contain(int(process._handle))


SUSPENDED_CREATION_FLAG = _CREATE_SUSPENDED
