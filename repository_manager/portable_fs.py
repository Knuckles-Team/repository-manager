"""Platform-portable filesystem primitives for descriptor-anchored operations.

POSIX can open a directory as a file descriptor and address its children with
``dir_fd=``.  Windows cannot: ``os.open`` on a directory fails and ``dir_fd``
is unsupported.  This module offers one API for both:

* when the platform supports ``dir_fd`` (:func:`native_dir_fd`) every call
  delegates straight to ``os``;
* otherwise a *directory handle* is a small opaque integer token mapped to its
  verified absolute path.  Each open re-checks with ``lstat`` that the entry is
  a real directory and not a symlink, junction or other reparse point, so link
  components are still refused (the TOCTOU window the descriptors close on
  POSIX is narrower to defend on Windows, where a held directory handle cannot
  be renamed away only if it was opened with delete sharing denied).

It also owns the portable resolution of user state/config/cache directories
(XDG on POSIX, ``LOCALAPPDATA``/``APPDATA`` on Windows) and process-group
termination, so callers never branch on the platform themselves.
"""

from __future__ import annotations

import itertools
import os
import signal
import stat
import subprocess  # nosec B404 - taskkill with a fixed argv
import threading
from typing import Any
from pathlib import Path

_TOKEN_BASE = 1 << 30
_FILE_ATTRIBUTE_REPARSE_POINT = 0x400

_lock = threading.Lock()
_tokens: dict[int, str] = {}
_counter = itertools.count(_TOKEN_BASE)


def native_dir_fd() -> bool:
    """Return whether directory descriptors and ``dir_fd=`` are available."""
    return os.name == "posix" and os.open in os.supports_dir_fd


def is_windows() -> bool:
    return os.name == "nt"


# --------------------------------------------------------------------------
# Handles
# --------------------------------------------------------------------------


def is_token(fd: int) -> bool:
    with _lock:
        return fd in _tokens


def _register(path: str) -> int:
    with _lock:
        token = next(_counter)
        _tokens[token] = path
        return token


def _path_of(fd: int) -> str:
    with _lock:
        return _tokens[fd]


def _child(parent_fd: int | None, name: str) -> str:
    if parent_fd is None:
        return name
    return os.path.join(_path_of(parent_fd), name)


def _is_link(result: os.stat_result) -> bool:
    if stat.S_ISLNK(result.st_mode):
        return True
    attributes = getattr(result, "st_file_attributes", 0)
    return bool(attributes & _FILE_ATTRIBUTE_REPARSE_POINT)


def directory_open_flags() -> int:
    """Flags for opening a directory that cannot follow a symlink."""
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def file_flags(flags: int) -> int:
    """Add the binary flag Windows needs to avoid newline translation."""
    return flags | getattr(os, "O_BINARY", 0)


def open_directory(name: str, parent_fd: int | None = None) -> int:
    """Open one directory (relative to ``parent_fd``) without following links."""
    if native_dir_fd():
        if parent_fd is None:
            return os.open(name, directory_open_flags())
        return os.open(name, directory_open_flags(), dir_fd=parent_fd)
    target = _child(parent_fd, name)
    result = os.lstat(target)
    if _is_link(result):
        raise OSError(f"{target!r} is a symlink or reparse point")
    if not stat.S_ISDIR(result.st_mode):
        raise NotADirectoryError(target)
    return _register(target)


def open_root() -> int:
    """Open the filesystem root (the current drive's anchor on Windows)."""
    if native_dir_fd():
        return os.open(os.sep, directory_open_flags())
    return _register(Path(os.getcwd()).anchor or os.sep)


def open_entry(name: str, flags: int, mode: int = 0o777, *, dir_fd: int | None) -> int:
    """Open a file entry relative to ``dir_fd``; a final link is refused."""
    if native_dir_fd() and dir_fd is not None:
        return os.open(name, flags, mode, dir_fd=dir_fd)
    target = _child(dir_fd, name) if dir_fd is not None else name
    try:
        existing = os.lstat(target)
    except FileNotFoundError:
        existing = None
    if existing is not None and _is_link(existing):
        raise OSError(f"{target!r} is a symlink or reparse point")
    return os.open(target, file_flags(flags), mode)


def stat_entry(name: str, *, dir_fd: int) -> os.stat_result:
    if native_dir_fd():
        return os.stat(name, dir_fd=dir_fd, follow_symlinks=False)
    return os.lstat(_child(dir_fd, name))


def fstat(fd: int) -> os.stat_result:
    if is_token(fd):
        return os.stat(_path_of(fd))
    return os.fstat(fd)


def close(fd: int) -> None:
    with _lock:
        if _tokens.pop(fd, None) is not None:
            return
    os.close(fd)


def dup(fd: int) -> int:
    if is_token(fd):
        return _register(_path_of(fd))
    return os.dup(fd)


def mkdir(name: str, mode: int = 0o755, *, dir_fd: int) -> None:
    if native_dir_fd():
        os.mkdir(name, mode=mode, dir_fd=dir_fd)
    else:
        os.mkdir(_child(dir_fd, name), mode=mode)


def unlink(name: str, *, dir_fd: int) -> None:
    if native_dir_fd():
        os.unlink(name, dir_fd=dir_fd)
    else:
        os.unlink(_child(dir_fd, name))


def rmdir(name: str, *, dir_fd: int) -> None:
    if native_dir_fd():
        os.rmdir(name, dir_fd=dir_fd)
    else:
        os.rmdir(_child(dir_fd, name))


def replace(src: str, dst: str, *, dir_fd: int) -> None:
    if native_dir_fd():
        os.replace(src, dst, src_dir_fd=dir_fd, dst_dir_fd=dir_fd)
    else:
        os.replace(_child(dir_fd, src), _child(dir_fd, dst))


def scandir_names(fd: int) -> list[str]:
    if is_token(fd):
        with os.scandir(_path_of(fd)) as iterator:
            return [entry.name for entry in iterator]
    with os.scandir(fd) as iterator:
        return [entry.name for entry in iterator]


def fsync_directory(fd: int) -> None:
    """Flush a directory; Windows has no directory fsync, so it is a no-op."""
    if is_token(fd):
        return
    os.fsync(fd)


def fsync_directory_path(path: str | os.PathLike[str]) -> None:
    """Durably flush the directory at ``path`` (no-op on Windows, which cannot)."""
    if is_windows():
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fd_path(fd: int) -> str:
    """A path a child process can use to reach the directory behind ``fd``."""
    if is_token(fd):
        return _path_of(fd)
    return f"/proc/self/fd/{fd}"


def child_pass_fds(fds: tuple[int, ...]) -> tuple[int, ...]:
    """Descriptors to inherit into a child (none for emulated handles)."""
    return tuple(fd for fd in fds if not is_token(fd))


def popen_fd_kwargs(fds: tuple[int, ...]) -> dict[str, Any]:
    """``Popen`` kwargs that inherit real descriptors; empty on Windows."""
    real = child_pass_fds(fds)
    if is_windows() or not real:
        return {}
    return {"pass_fds": real}


# --------------------------------------------------------------------------
# Directory links
# --------------------------------------------------------------------------

_ERROR_PRIVILEGE_NOT_HELD = 1314


def is_link(path: str | os.PathLike[str]) -> bool:
    """True for a symlink, or a directory junction on Windows."""
    return os.path.islink(path) or getattr(os.path, "isjunction", lambda _p: False)(
        path
    )


def read_link(path: str | os.PathLike[str]) -> str:
    """``os.readlink`` without the Windows ``\\\\?\\`` extended-length prefix."""
    value = os.readlink(path)
    if (
        is_windows()
        and value.startswith("\\\\?\\")
        and not value.startswith("\\\\?\\UNC")
    ):
        return value[4:]
    return value


def create_dir_link(
    link: str | os.PathLike[str], target: str | os.PathLike[str]
) -> None:
    """Create a directory link, falling back to a junction on Windows.

    Windows symlinks need the symlink privilege (or Developer Mode); a junction
    needs neither and is equivalent for local directory targets.
    """
    try:
        os.symlink(target, link, target_is_directory=True)
    except OSError as exc:
        if (
            not is_windows()
            or getattr(exc, "winerror", None) != _ERROR_PRIVILEGE_NOT_HELD
        ):
            raise
        import _winapi  # Windows-only stdlib module

        _winapi.CreateJunction(os.fspath(target), os.fspath(link))  # type: ignore[attr-defined]


def replace_link(staged: str | os.PathLike[str], link: str | os.PathLike[str]) -> None:
    """Move ``staged`` onto ``link``; atomic on POSIX, remove-then-move on Windows.

    ``os.replace`` cannot overwrite an existing directory link on Windows.
    """
    if is_windows() and is_link(link):
        os.rmdir(link)  # removes a directory symlink or junction, not its target
    os.replace(staged, link)


# --------------------------------------------------------------------------
# User directories
# --------------------------------------------------------------------------


def _env_dir(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value).expanduser() if value else None


def home_dir() -> Path:
    """User home: ``HOME`` first, then ``USERPROFILE`` (Windows), then pathlib."""
    return _env_dir("HOME") or _env_dir("USERPROFILE") or Path.home()


def state_home() -> Path:
    """``XDG_STATE_HOME`` or the platform equivalent."""
    explicit = _env_dir("XDG_STATE_HOME")
    if explicit:
        return explicit
    if is_windows():
        return _env_dir("LOCALAPPDATA") or home_dir() / "AppData" / "Local"
    return home_dir() / ".local" / "state"


def config_home() -> Path:
    """``XDG_CONFIG_HOME`` or the platform equivalent."""
    explicit = _env_dir("XDG_CONFIG_HOME")
    if explicit:
        return explicit
    if is_windows():
        return _env_dir("APPDATA") or home_dir() / "AppData" / "Roaming"
    return home_dir() / ".config"


def cache_home() -> Path:
    explicit = _env_dir("XDG_CACHE_HOME")
    if explicit:
        return explicit
    if is_windows():
        return _env_dir("LOCALAPPDATA") or home_dir() / "AppData" / "Local"
    return home_dir() / ".cache"


# --------------------------------------------------------------------------
# Processes
# --------------------------------------------------------------------------


#: ``Popen`` kwargs that put a child in its own group/session.
def new_group_kwargs() -> dict[str, Any]:
    if is_windows():
        return {"creationflags": getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)}
    return {"start_new_session": True}


def force_signal() -> int:
    """SIGKILL where it exists; Windows has none, so use its SIGBREAK."""
    return getattr(signal, "SIGKILL", getattr(signal, "SIGBREAK", signal.SIGABRT))


def terminate_process_group(process: subprocess.Popen, *, force: bool = False) -> None:
    """Terminate ``process``' group (POSIX) or whole tree (Windows).

    ``force`` selects SIGKILL / ``taskkill /F`` instead of SIGTERM.  Windows has
    no process groups in the POSIX sense, so the tree is killed with the stock
    ``taskkill`` tool; a graceful request maps to ``Popen.terminate``.
    """
    if is_windows() or not hasattr(os, "killpg"):
        if process.poll() is not None:
            return
        if not force:
            process.terminate()
            return
        result = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["taskkill", "/T", "/F", "/PID", str(process.pid)],
            capture_output=True,
            check=False,
        )
        if result.returncode != 0:
            process.kill()
        return
    sig = signal.SIGKILL if force else signal.SIGTERM
    os.killpg(os.getpgid(process.pid), sig)


def cpu_count() -> int:
    """CPUs usable by this process (affinity-aware where supported)."""
    getter = getattr(os, "sched_getaffinity", None)
    if getter is not None:
        return len(getter(0))
    return os.cpu_count() or 1


def hostname() -> str:
    uname = getattr(os, "uname", None)
    if uname is not None:
        return uname().nodename
    import platform

    return platform.node()


def temp_root() -> Path:
    """Platform temporary directory (never a hard-coded ``/tmp``)."""
    import tempfile

    return Path(tempfile.gettempdir())
