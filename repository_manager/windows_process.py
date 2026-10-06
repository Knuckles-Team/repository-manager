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
        for name, result, arguments in signatures:
            function = getattr(api, name)
            function.restype = result
            function.argtypes = arguments
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
