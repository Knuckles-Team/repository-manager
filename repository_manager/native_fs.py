"""Portable no-follow file primitives for publishers outside the boundary.

POSIX callers get ``O_NOFOLLOW`` descriptors and ``fsync``; Windows callers get
the handle-based equivalents from :mod:`repository_manager.windows_filesystem`
(reparse points refused on the opened handle, flushes through a write handle).
Both return ordinary integer file descriptors.
"""

from __future__ import annotations

import os
from pathlib import PurePosixPath, PureWindowsPath

WINDOWS = os.name == "nt"
# A Windows flush needs a write handle; POSIX flushes a read-only descriptor.
FSYNC_FILE_FLAGS = os.O_RDWR if WINDOWS else os.O_RDONLY


def open_no_follow(path: str | os.PathLike[str], flags: int, mode: int = 0o666) -> int:
    """Open ``path`` with ``flags`` without following a final link."""
    if WINDOWS:
        from repository_manager import windows_filesystem

        return windows_filesystem.path_open_no_follow(os.fspath(path), flags)
    return os.open(path, flags | os.O_NOFOLLOW | os.O_CLOEXEC, mode)


def fsync_path(path: str | os.PathLike[str], *, directory: bool = False) -> None:
    """Durably flush a regular file or directory without following a link."""
    if WINDOWS:
        from repository_manager import windows_filesystem

        windows_filesystem.path_fsync(os.fspath(path), directory=directory)
        return
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_CLOEXEC
    if directory:
        flags |= os.O_DIRECTORY
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def portable_name(component: str) -> str:
    """Spell one path component (a cache key) so this filesystem accepts it.

    Windows reserves ``:`` (it selects an NTFS data stream), so keys such as
    ``v2:<hex>`` are percent-escaped there; POSIX keeps the key verbatim.
    """
    if not WINDOWS:
        return component
    return component.replace("%", "%25").replace(":", "%3A")


def name_from_portable(name: str) -> str:
    """Invert :func:`portable_name` for a directory entry read back."""
    if not WINDOWS:
        return name
    return name.replace("%3A", ":").replace("%25", "%")


def is_rooted(value: str) -> bool:
    """Whether ``value`` is anchored in POSIX or Windows syntax.

    A path that must be relative is refused if either syntax would anchor it
    (``/x``, ``C:x``, ``\\\\host\\share``), so the check is platform-independent.
    """
    return bool(PurePosixPath(value).root or PureWindowsPath(value).anchor)


def read_link(path: str | os.PathLike[str]) -> str:
    """``os.readlink`` spelled the way the link was created.

    Windows returns the NT substitute name (``\\\\?\\C:\\...``); dropping that
    prefix gives back the target ``os.symlink`` was called with.
    """
    target = os.readlink(path)
    if WINDOWS and target.startswith("\\\\?\\UNC\\"):
        return "\\\\" + target[len("\\\\?\\UNC\\") :]
    if WINDOWS and target.startswith("\\\\?\\"):
        return target[len("\\\\?\\") :]
    return target


def replace_link(source: str | os.PathLike[str], link: str | os.PathLike[str]) -> None:
    """Move the symlink ``source`` onto ``link`` (POSIX: one atomic rename).

    Windows cannot rename onto an existing directory symlink, so the old link
    is removed first; callers keep the previous target to restore on failure.
    """
    if WINDOWS and os.path.islink(link):
        os.unlink(link)
    os.replace(source, link)
