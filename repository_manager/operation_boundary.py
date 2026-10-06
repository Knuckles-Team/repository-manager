"""Descriptor-pinned filesystem boundaries for repository operations.

The repository manager receives paths from a mutable manifest and from callers.
Checking a path and later passing its spelling to ``git`` leaves a time-of-check
to time-of-use window: a directory component can become a symlink after the
check, or a checkout can be replaced while a phased plan is in flight.  This
module keeps the operation boundary deliberately small:

* every existing component is opened with ``O_NOFOLLOW``;
* missing workspace components are created with ``mkdirat`` semantics through
  ``dir_fd`` and immediately reopened without following links;
* subprocesses use an inherited ``/proc/self/fd/<n>`` working directory; and
* clone destinations are reserved before ``git`` starts and checked again at
  the final handoff.

On Windows the same contract is provided by the handle-relative descriptor
layer in :mod:`repository_manager.windows_filesystem`: components are opened
relative to their pinned parent handle and never through a reparse point, and
pinned directories deny delete sharing, so the pinned spelling a child receives
cannot be rebound while the operation holds it.

The handles are intentionally independent of the repository-manager class so
the same primitive can protect sync, pull, push, and release-plan snapshots.
"""

from __future__ import annotations

import contextlib
import dataclasses
import fnmatch
import functools
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

_WINDOWS = os.name == "nt"
if _WINDOWS:
    from repository_manager import windows_filesystem as _native

_PINNED_GIT_ENVIRONMENT = frozenset(
    {
        "GIT_DIR",
        "GIT_WORK_TREE",
        "GIT_COMMON_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_CONFIG_SYSTEM",
        "GIT_CONFIG_GLOBAL",
        "GIT_CONFIG_NOSYSTEM",
        "GIT_CONFIG_PARAMETERS",
        "GIT_CONFIG_COUNT",
    }
)


class OperationBoundaryError(ValueError):
    """A path or descriptor cannot be admitted to a mutating operation."""


@dataclasses.dataclass(frozen=True)
class DirectoryIdentity:
    """Stable identity of a directory entry, including its object type."""

    device: int
    inode: int
    mode: int
    mount_id: int | None = None

    @classmethod
    def from_stat(
        cls, result: os.stat_result, mount_id: int | None = None
    ) -> DirectoryIdentity:
        """Capture the identity fields relevant to a directory boundary."""
        return cls(
            device=int(result.st_dev),
            inode=int(result.st_ino),
            mode=stat.S_IFMT(result.st_mode),
            mount_id=mount_id,
        )

    def as_dict(self) -> dict[str, int | None]:
        """Return a JSON-safe identity payload."""
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
            "mount_id": self.mount_id,
        }


@dataclasses.dataclass(frozen=True)
class FileIdentity:
    """Stable metadata for a regular file read through one pinned descriptor."""

    device: int
    inode: int
    mode: int
    links: int
    size: int
    modified_ns: int
    changed_ns: int
    mount_id: int | None = None

    @classmethod
    def from_stat(cls, result: os.stat_result, mount_id: int | None) -> FileIdentity:
        """Capture fields that expose replacement or in-place content mutation."""
        return cls(
            device=int(result.st_dev),
            inode=int(result.st_ino),
            mode=stat.S_IFMT(result.st_mode),
            links=int(result.st_nlink),
            size=int(result.st_size),
            modified_ns=int(result.st_mtime_ns),
            changed_ns=int(result.st_ctime_ns),
            mount_id=mount_id,
        )

    def as_dict(self) -> dict[str, int | None]:
        """Return a JSON-safe identity payload."""
        return dataclasses.asdict(self)


def _mount_id_for_path(path: Path) -> int | None:
    """Open one lexical directory and read its current kernel mount ID."""
    if os.name != "posix":
        return None
    try:
        descriptor = os.open(path, _directory_flags())
    except OSError:
        return None
    try:
        return _mount_id_for_fd(descriptor)
    finally:
        with contextlib.suppress(OSError):
            os.close(descriptor)


def _mount_id_for_fd(fd: int) -> int | None:
    """Read the mount ID attached to an open descriptor, without path inference."""
    try:
        lines = Path(f"/proc/self/fdinfo/{fd}").read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError):
        return None
    return _mount_id_from_fdinfo(lines)


def _mount_id_from_fdinfo(lines: list[str]) -> int | None:
    """Parse one kernel fdinfo payload without inferring from a lexical path."""
    for line in lines:
        key, separator, raw_value = line.partition(":")
        if separator and key == "mnt_id":
            try:
                return int(raw_value.strip())
            except ValueError:
                return None
    return None


def _identity_for_entry(result: os.stat_result, parent_fd: int) -> DirectoryIdentity:
    """Capture an entry identity plus the mount containing its parent."""
    return DirectoryIdentity.from_stat(result, _mount_id_for_fd(parent_fd))


def _directory_flags() -> int:
    """Return read-only directory flags that cannot follow a symlink."""
    return (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )


def _read_flags() -> int:
    """Return read-only no-follow flags for an anchored regular file."""
    return os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)


def _is_link(result: os.stat_result) -> bool:
    """A symlink, or on Windows any reparse point (junction, mount point)."""
    return stat.S_ISLNK(result.st_mode) or bool(
        getattr(result, "st_file_attributes", 0) & 0x400
    )


def _open_directory_entry(parent_fd: int, component: str) -> int:
    """``openat(parent, component, O_DIRECTORY | O_NOFOLLOW)``."""
    if _WINDOWS:
        return _native.descriptor_open_directory(parent_fd, component)
    return os.open(component, _directory_flags(), dir_fd=parent_fd)


def _stat_entry(parent_fd: int, component: str) -> os.stat_result:
    """``fstatat(parent, component, AT_SYMLINK_NOFOLLOW)``."""
    if _WINDOWS:
        return _native.descriptor_stat(parent_fd, component)
    return os.stat(component, dir_fd=parent_fd, follow_symlinks=False)


def _mkdir_entry(parent_fd: int, component: str) -> None:
    """``mkdirat(parent, component, 0o755)``."""
    if _WINDOWS:
        _native.descriptor_mkdir(parent_fd, component)
        return
    os.mkdir(component, mode=0o755, dir_fd=parent_fd)


def _open_file_entry(parent_fd: int, name: str, flags: int, mode: int = 0o644) -> int:
    """``openat(parent, name, flags | O_NOFOLLOW)`` for a regular file."""
    if _WINDOWS:
        return _native.descriptor_open_file(parent_fd, name, flags)
    return os.open(name, flags, mode, dir_fd=parent_fd)


def _remove_entry(parent_fd: int, name: str, *, directory: bool = False) -> None:
    """``unlinkat`` (``AT_REMOVEDIR`` for a directory)."""
    if _WINDOWS:
        _native.descriptor_remove(parent_fd, name, directory=directory)
    elif directory:
        os.rmdir(name, dir_fd=parent_fd)
    else:
        os.unlink(name, dir_fd=parent_fd)


def _replace_entry(directory_fd: int, source: str, target: str) -> None:
    """``renameat`` within one pinned directory, replacing ``target``."""
    if _WINDOWS:
        _native.descriptor_replace(directory_fd, source, target)
        return
    os.replace(source, target, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)


def _close_fd(descriptor: int) -> None:
    """Close one descriptor (a pinned Windows chain closes when unused)."""
    if _WINDOWS:
        _native.descriptor_close(descriptor)
    else:
        os.close(descriptor)


def _dup_fd(descriptor: int) -> int:
    """Duplicate a directory descriptor, keeping its pinned chain."""
    if _WINDOWS:
        return _native.descriptor_dup(descriptor)
    return os.dup(descriptor)


def _list_entries(directory_fd: int) -> list[str]:
    """Names in a pinned directory."""
    if _WINDOWS:
        return _native.descriptor_listdir(directory_fd)
    with os.scandir(directory_fd) as iterator:
        return [entry.name for entry in iterator]


def _fsync_directory_fd(directory_fd: int) -> None:
    """Flush a pinned directory's entries."""
    if _WINDOWS:
        _native.descriptor_fsync_directory(directory_fd)
    else:
        os.fsync(directory_fd)


def _open_root_fd(absolute: Path) -> int:
    """Pin the filesystem root that ``absolute`` hangs from."""
    if _WINDOWS:
        return _native.descriptor_open_root(absolute.anchor)
    return os.open(os.sep, _directory_flags())


def _absolute_parts(absolute: Path) -> tuple[str, ...]:
    """Components below the filesystem root (drive or share on Windows)."""
    if _WINDOWS:
        return tuple(absolute.parts[1:])
    return tuple(part for part in absolute.parts if part not in ("", os.sep))


_LINE_ENDING_KEYS = ("core.autocrlf", "core.eol")


@functools.lru_cache(maxsize=1)
def _inherited_line_endings() -> tuple[tuple[str, str], ...]:
    """Line-ending settings from the system and global Git config (Windows).

    Disabling system and user config must not change how an existing checkout
    is compared: Git for Windows enables ``core.autocrlf`` system-wide, so
    without it every CRLF working file of a racily-clean index reads as
    modified.  These keys only select line-ending conversion; they cannot name
    a path, remote or program, so carrying them forward keeps the isolation.
    """
    if not _WINDOWS:
        return ()
    values: dict[str, str] = {}
    for scope in ("--system", "--global"):
        for key in _LINE_ENDING_KEYS:
            value = _git_config_value(scope, key)
            if value is not None:
                values[key] = value
    return tuple(values.items())


def _git_config_value(scope: str, key: str) -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        result = subprocess.run(  # fixed argv, no shell
            [git, "config", scope, "--get", key],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    value = result.stdout.strip()
    return value if result.returncode == 0 and value else None


def _line_ending_environment() -> dict[str, str]:
    """``GIT_CONFIG_*`` entries re-applying the inherited line-ending keys."""
    pairs = _inherited_line_endings()
    environment = {"GIT_CONFIG_COUNT": str(len(pairs))} if pairs else {}
    for index, (key, value) in enumerate(pairs):
        environment[f"GIT_CONFIG_KEY_{index}"] = key
        environment[f"GIT_CONFIG_VALUE_{index}"] = value
    return environment


def descriptor_path(descriptor: int) -> str:
    """A child-visible spelling bound to an open directory descriptor."""
    if _WINDOWS:
        return _native.descriptor_path(descriptor)
    return f"/proc/self/fd/{descriptor}"


def subprocess_descriptor_options(descriptors: tuple[int, ...]) -> dict[str, Any]:
    """``Popen`` keywords that keep pinned descriptors valid for a child.

    POSIX children inherit the descriptors their ``/proc/self/fd`` paths name.
    Windows children receive pinned spellings instead, which the parent keeps
    valid by holding the chain open, so nothing is inherited.
    """
    if _WINDOWS:
        return {}
    return {"pass_fds": descriptors, "start_new_session": True}


def list_directory(directory_fd: int) -> list[str]:
    """Names in a pinned directory, without following links."""
    return _list_entries(directory_fd)


def _path_components(path: Path) -> tuple[str, ...]:
    """Return absolute path components without silently erasing traversal."""
    if ".." in path.parts:
        raise OperationBoundaryError("path contains a lexical parent segment")
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise OperationBoundaryError("operation path must be absolute")
    return _absolute_parts(absolute)


def _open_directory_at(parent_fd: int, component: str) -> int:
    """Open one directory component relative to an already-open directory."""
    try:
        return _open_directory_entry(parent_fd, component)
    except OSError as exc:
        entry = _stat_at(parent_fd, component)
        if entry is not None and _is_link(entry):
            raise OperationBoundaryError(
                f"directory component {component!r} is a symlink"
            ) from exc
        raise OperationBoundaryError(
            f"cannot open directory component {component!r} without following links"
        ) from exc


def _stat_at(parent_fd: int, component: str) -> os.stat_result | None:
    """Stat one directory entry without following a final symlink."""
    try:
        return _stat_entry(parent_fd, component)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OperationBoundaryError(
            f"cannot inspect directory entry {component!r} safely"
        ) from exc


@dataclasses.dataclass
class PinnedDirectory:
    """A directory reached through descriptor-relative no-follow opens.

    ``owned_fds`` excludes the workspace root fd when this object was opened
    relative to an existing root handle.  Closing this object therefore never
    invalidates the caller's root boundary.
    """

    path: Path
    fd: int
    identity: DirectoryIdentity
    owned_fds: tuple[int, ...]
    root_fd: int
    root_path: Path
    root_identity: DirectoryIdentity
    leaf: str | None = None
    target_fd: int | None = None
    target_identity: DirectoryIdentity | None = None
    git_fd: int | None = None
    git_path: Path | None = None
    git_identity: DirectoryIdentity | None = None
    git_entry_identity: DirectoryIdentity | None = None
    git_owned_fds: tuple[int, ...] = ()
    common_fd: int | None = None
    common_path: Path | None = None
    common_identity: DirectoryIdentity | None = None
    common_owned_fds: tuple[int, ...] = ()
    common_pointer_digest: str | None = None
    common_config_identity: FileIdentity | None = None
    common_config_digest: str | None = None
    configured_origin: str | None = None
    expected_origin: str | None = None
    boundary_assertion: Callable[[], None] | None = None
    _closed: bool = False

    def __enter__(self) -> PinnedDirectory:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @property
    def proc_path(self) -> str:
        """A child-visible path anchored to the open directory descriptor."""
        if self._closed:
            raise OperationBoundaryError("operation descriptor is already closed")
        return descriptor_path(self.fd)

    @property
    def destination_path(self) -> str:
        """A clone destination anchored to the reserved target descriptor."""
        if self.target_fd is not None:
            return descriptor_path(self.target_fd)
        if self.leaf is None:
            raise OperationBoundaryError("clone destination has no final component")
        return os.path.join(self.proc_path, self.leaf)

    @property
    def pass_fds(self) -> tuple[int, ...]:
        """Descriptors that must survive long enough for the child to chdir."""
        values = list(self.owned_fds)
        self._append_unique_descriptors(values, (self.root_fd, self.target_fd))
        self._append_unique_descriptors(
            values, (*self.git_owned_fds, *self.common_owned_fds)
        )
        return tuple(values)

    @staticmethod
    def _append_unique_descriptors(
        values: list[int], descriptors: tuple[int | None, ...]
    ) -> None:
        """Append open descriptors once, ignoring optional absent descriptors."""
        for descriptor in descriptors:
            if descriptor is not None and descriptor not in values:
                values.append(descriptor)

    def anchored_git_command(self, argv: list[str]) -> list[str]:
        """Anchor a direct Git command to the admitted descriptors."""
        _reject_git_path_controls(argv)
        if (
            self.git_fd is None
            or not argv
            or os.path.basename(argv[0]) != "git"
            or argv[1:2] in (["clone"], ["init"])
        ):
            return argv
        return [
            argv[0],
            f"--git-dir={descriptor_path(self.git_fd)}",
            f"--work-tree={self.proc_path}",
            *argv[1:],
        ]

    @staticmethod
    def anchored_git_environment(env: dict[str, str]) -> dict[str, str]:
        """Remove redirect controls and disable system and user Git config."""
        isolated = {
            key: value
            for key, value in env.items()
            if key not in _PINNED_GIT_ENVIRONMENT
            and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        }
        isolated["GIT_CONFIG_NOSYSTEM"] = "1"
        isolated["GIT_CONFIG_GLOBAL"] = os.devnull
        isolated.update(_line_ending_environment())
        return isolated

    def assert_root_identity(self) -> None:
        """Reject replacement of the lexical root before a child starts."""
        _assert_path_identity(self.root_path, self.root_identity, "workspace root")
        _assert_fd_identity(self.root_fd, self.root_identity, "workspace root")

    def assert_path_identity(self) -> None:
        """Reject replacement of the lexical directory before a child starts."""
        self.assert_root_identity()
        _assert_path_identity(self.path, self.identity, "operation target")
        _assert_fd_identity(self.fd, self.identity, "operation target")

    def assert_operation_identity(self) -> None:
        """Check the pinned path and any immutable plan boundary together."""
        self.assert_path_identity()
        self._ensure_git_metadata()
        if self.boundary_assertion is not None:
            self.boundary_assertion()

    def assert_git_identity(self) -> None:
        """Reject a Git metadata replacement after a child returns."""
        self._ensure_git_metadata()

    def _ensure_git_metadata(self) -> int | None:
        """Open and retain the checkout's Git metadata without links."""
        git_entry = _stat_at(self.fd, ".git")
        if git_entry is None:
            if self.git_fd is not None:
                raise OperationBoundaryError("checkout .git entry disappeared")
            return None
        self._validate_git_entry(git_entry)
        entry_identity = _identity_for_entry(git_entry, self.fd)
        self._assert_git_entry_identity(entry_identity)
        if self.git_fd is None:
            self._initialize_git_metadata(git_entry, entry_identity)
        else:
            self._assert_common_metadata()
        self._assert_git_metadata_identities()
        return self.git_fd

    @staticmethod
    def _validate_git_entry(git_entry: os.stat_result) -> None:
        """Reject linked or multiply-linked checkout metadata."""
        if _is_link(git_entry):
            raise OperationBoundaryError("checkout .git entry is a symlink")
        if stat.S_ISREG(git_entry.st_mode) and git_entry.st_nlink != 1:
            raise OperationBoundaryError("checkout .git entry is a hard link")

    def _assert_git_entry_identity(self, identity: DirectoryIdentity) -> None:
        """Reject replacement of the checkout's Git entry."""
        if self.git_entry_identity is not None and identity != self.git_entry_identity:
            raise OperationBoundaryError("checkout .git entry identity changed")

    def _initialize_git_metadata(
        self, git_entry: os.stat_result, entry_identity: DirectoryIdentity
    ) -> None:
        """Open and pin worktree, common directory, and common config identities."""
        git_fd, owned, metadata_path = _metadata_directory(
            self.fd,
            git_entry,
            str(self.path),
            root_fd=self.root_fd,
            root_path=self.root_path,
            checkout_path=self.path,
        )
        self.git_fd = git_fd
        self.git_path = metadata_path
        self.git_owned_fds = tuple(owned)
        self.git_identity = _identity_from_fd(git_fd, "git worktree")
        self.git_entry_identity = entry_identity
        (
            self.common_fd,
            common_owned,
            self.common_path,
            self.common_pointer_digest,
        ) = _common_metadata_directory(
            git_fd,
            metadata_path,
            self.root_fd,
            self.root_path,
            str(self.path),
        )
        self.common_owned_fds = tuple(common_owned)
        self.common_identity = _identity_from_fd(self.common_fd, "git common")
        (
            self.common_config_identity,
            self.common_config_digest,
            config,
        ) = _config_entry_snapshot(self.common_fd)
        self.configured_origin = _audit_local_git_config(config)
        _reject_worktree_config(self.git_fd)

    def _assert_git_metadata_identities(self) -> None:
        """Revalidate pinned Git worktree, common directory, and config."""
        if self.git_fd is None or self.git_identity is None:
            raise OperationBoundaryError("git worktree identity is missing")
        _assert_fd_identity(self.git_fd, self.git_identity, "git worktree")
        if self.common_fd is None or self.common_identity is None:
            raise OperationBoundaryError("git common identity is missing")
        _assert_fd_identity(self.common_fd, self.common_identity, "git common")
        current_identity, current_digest, config = _config_entry_snapshot(
            self.common_fd
        )
        self._assert_git_config_identity(current_identity, current_digest, config)
        _reject_worktree_config(self.git_fd)

    def _assert_git_config_identity(
        self,
        current_identity: FileIdentity | None,
        current_digest: str | None,
        config: bytes | None,
    ) -> None:
        """Compare both config content and its audited destination identity."""
        _assert_config_snapshot(
            current_identity,
            current_digest,
            self.common_config_identity,
            self.common_config_digest,
        )
        if _audit_local_git_config(config) != self.configured_origin:
            raise OperationBoundaryError("git configured origin identity changed")

    def _assert_common_metadata(self) -> None:
        """Re-open the declared common directory and compare its identity."""
        if self.git_fd is None or self.common_fd is None or self.common_path is None:
            raise OperationBoundaryError("git common metadata is missing")
        candidate_fd, candidate_owned, _candidate_path, pointer_digest = (
            _common_metadata_directory(
                self.git_fd,
                self.git_path or self.path,
                self.root_fd,
                self.root_path,
                str(self.path),
            )
        )
        try:
            candidate_identity = _identity_from_fd(candidate_fd, "git common")
            if candidate_identity != self.common_identity:
                raise OperationBoundaryError("git common directory identity changed")
            if pointer_digest != self.common_pointer_digest:
                raise OperationBoundaryError("git common pointer changed")
        finally:
            for descriptor in reversed(candidate_owned):
                with contextlib.suppress(OSError):
                    _close_fd(descriptor)

    def clear_boundary_assertion(self) -> None:
        """Stop plan checks after an operation intentionally changes metadata."""
        self.boundary_assertion = None

    def assert_handoff(self) -> None:
        """Prove a reserved clone target still names the opened directory."""
        if self.target_fd is None or self.leaf is None:
            raise OperationBoundaryError("clone destination was not reserved")
        self.assert_path_identity()
        _assert_fd_identity(self.target_fd, self.target_identity, "clone target")
        result = _stat_at(self.fd, self.leaf)
        if result is None:
            raise OperationBoundaryError("clone target disappeared before handoff")
        if _is_link(result):
            raise OperationBoundaryError("clone target became a symlink")
        actual = _identity_for_entry(result, self.fd)
        if actual != self.target_identity:
            raise OperationBoundaryError("clone target changed before handoff")

    def reserve_leaf(self) -> None:
        """Reserve a previously absent final component and pin its directory."""
        if self.leaf is None:
            raise OperationBoundaryError("clone destination has no final component")
        self.assert_root_identity()
        if _stat_at(self.fd, self.leaf) is not None:
            raise OperationBoundaryError("clone destination already exists")
        try:
            _mkdir_entry(self.fd, self.leaf)
        except FileExistsError as exc:
            raise OperationBoundaryError(
                "clone destination appeared during reservation"
            ) from exc
        except OSError as exc:
            raise OperationBoundaryError("cannot reserve clone destination") from exc

        try:
            target_fd = _open_directory_at(self.fd, self.leaf)
        except Exception:
            # The reservation is deliberately left in place on an unexpected
            # failure.  It is an empty, bounded directory and cannot redirect
            # a subsequent operation outside the root.
            raise
        self.target_fd = target_fd
        self.target_identity = _identity_from_fd(target_fd, "clone target")

    def close(self) -> None:
        """Close only descriptors owned by this handle, once."""
        if self._closed:
            return
        self._closed = True
        self.boundary_assertion = None
        self._close_descriptor_group(self.owned_fds)
        self._close_descriptor_group(self.git_owned_fds, self.owned_fds)
        self._close_descriptor_group(
            self.common_owned_fds, self.owned_fds, self.git_owned_fds
        )
        if self.target_fd is not None:
            self._close_descriptor_group(
                (self.target_fd,),
                self.owned_fds,
                self.git_owned_fds,
                self.common_owned_fds,
            )

    @staticmethod
    def _close_descriptor_group(
        descriptors: tuple[int, ...], *already_closed: tuple[int, ...]
    ) -> None:
        """Close descriptors unless an earlier ownership group already did."""
        excluded = {descriptor for group in already_closed for descriptor in group}
        for descriptor in reversed(descriptors):
            if descriptor in excluded:
                continue
            with contextlib.suppress(OSError):
                _close_fd(descriptor)


def _identity_from_fd(fd: int, label: str) -> DirectoryIdentity:
    """Capture a directory fd identity and require an actual directory."""
    try:
        result = os.fstat(fd)
    except OSError as exc:
        raise OperationBoundaryError(f"cannot inspect {label} descriptor") from exc
    if not stat.S_ISDIR(result.st_mode):
        raise OperationBoundaryError(f"{label} is not a directory")
    return DirectoryIdentity.from_stat(result, _mount_id_for_fd(fd))


def _assert_fd_identity(
    fd: int, expected: DirectoryIdentity | None, label: str
) -> None:
    """Compare a still-open descriptor with its immutable snapshot."""
    if expected is None:
        raise OperationBoundaryError(f"{label} identity is missing")
    actual = _identity_from_fd(fd, label)
    if actual != expected:
        raise OperationBoundaryError(f"{label} descriptor identity changed")


def _assert_path_identity(path: Path, expected: DirectoryIdentity, label: str) -> None:
    """Compare a lexical path using no-follow stats, never ``resolve``.

    Checking only the final entry is insufficient when a parent is replaced
    with a symlink that happens to point back at the same directory: the final
    ``st_dev``/``st_ino`` would still match even though the lexical path is no
    longer the approved path.  Walk every component with ``follow_symlinks``
    disabled as a diagnostic assertion.  The actual operation remains bound
    to the already-open descriptors, so this check never becomes the authority
    for the child process.
    """
    result = _stat_path_without_symlinks(path, label)
    actual = DirectoryIdentity.from_stat(result, _mount_id_for_path(path))
    if actual != expected:
        raise OperationBoundaryError(f"{label} identity changed")


def _stat_path_without_symlinks(path: Path, label: str) -> os.stat_result:
    """Stat every component of an absolute path without following links."""
    raw_path = Path(path).expanduser()
    if ".." in raw_path.parts:
        raise OperationBoundaryError(f"{label} contains a lexical parent segment")
    if not raw_path.is_absolute():
        raise OperationBoundaryError(f"{label} is not absolute")
    current = Path(raw_path.anchor)
    components = raw_path.parts[1:]
    if not components:
        try:
            return os.stat(raw_path, follow_symlinks=False)
        except OSError as exc:
            raise OperationBoundaryError(
                f"{label} disappeared or is inaccessible"
            ) from exc
    for index, component in enumerate(components):
        current /= component
        result = _stat_path_component(current, label)
        if _is_link(result):
            raise OperationBoundaryError(
                f"{label} contains symlink component {current}"
            )
        if index < len(components) - 1 and not stat.S_ISDIR(result.st_mode):
            raise OperationBoundaryError(
                f"{label} contains non-directory component {current}"
            )
    return result


def _stat_path_component(path: Path, label: str) -> os.stat_result:
    """Read one path component without following its final link."""
    try:
        return os.stat(path, follow_symlinks=False)
    except OSError as exc:
        raise OperationBoundaryError(f"{label} disappeared or is inaccessible") from exc


def _make_root_handle(path: Path, fd: int, owned: list[int]) -> PinnedDirectory:
    """Build a root handle from the final no-follow directory descriptor."""
    identity = _identity_from_fd(fd, "workspace root")
    return PinnedDirectory(
        path=path,
        fd=fd,
        identity=identity,
        owned_fds=tuple(owned),
        root_fd=fd,
        root_path=path,
        root_identity=identity,
    )


def open_directory(path: str | Path, *, create: bool = False) -> PinnedDirectory:
    """Open an absolute directory one component at a time without links.

    When ``create`` is true, only missing components are created relative to
    already-open parent descriptors.  An existing symlink is never accepted.
    """
    candidate = Path(path).expanduser()
    components = _path_components(candidate)
    absolute = Path(os.path.abspath(candidate))
    try:
        root_fd = _open_root_fd(absolute)
    except OSError as exc:
        raise OperationBoundaryError("cannot open filesystem root") from exc

    owned = [root_fd]
    current_fd = root_fd
    try:
        for component in components:
            child_fd = _open_or_create_directory_at(current_fd, component, create)
            owned.append(child_fd)
            current_fd = child_fd
        return _make_root_handle(absolute, current_fd, owned)
    except Exception:
        _close_descriptors(owned)
        raise


def _open_or_create_directory_at(parent_fd: int, component: str, create: bool) -> int:
    """Open one component, creating it relative to the pinned parent if needed."""
    try:
        return _open_directory_entry(parent_fd, component)
    except FileNotFoundError:
        if not create:
            raise OperationBoundaryError(
                f"directory component {component!r} does not exist"
            ) from None
        try:
            _mkdir_entry(parent_fd, component)
        except FileExistsError:
            # A concurrent creator won the race; reopening with O_NOFOLLOW
            # still rejects a symlink replacement.
            pass
        except OSError as exc:
            raise OperationBoundaryError(
                f"cannot create directory component {component!r}"
            ) from exc
        return _open_directory_at(parent_fd, component)
    except OSError as exc:
        entry = _stat_at(parent_fd, component)
        if entry is not None and _is_link(entry):
            raise OperationBoundaryError(
                f"directory component {component!r} is a symlink"
            ) from exc
        raise OperationBoundaryError(
            f"cannot open directory component {component!r} safely"
        ) from exc


def _close_descriptors(descriptors: list[int] | tuple[int, ...]) -> None:
    """Close a descriptor collection while preserving the original failure."""
    for descriptor in reversed(descriptors):
        with contextlib.suppress(OSError):
            _close_fd(descriptor)


def _relative_components(root: PinnedDirectory, target: Path) -> tuple[str, ...]:
    """Return target components relative to a pinned root."""
    raw_target = target.expanduser()
    if ".." in raw_target.parts:
        raise OperationBoundaryError(
            "operation target contains a lexical parent segment"
        )
    candidate = Path(os.path.abspath(raw_target))
    try:
        relative = candidate.relative_to(root.path)
    except ValueError as exc:
        raise OperationBoundaryError("operation target escapes workspace root") from exc
    if ".." in relative.parts:
        raise OperationBoundaryError(
            "operation target contains a lexical parent segment"
        )
    return tuple(relative.parts)


def pin_existing(root: PinnedDirectory, target: str | Path) -> PinnedDirectory:
    """Pin an existing target directory beneath ``root``."""
    components = _relative_components(root, Path(target))
    if not components:
        duplicate = _dup_fd(root.fd)
        return PinnedDirectory(
            path=root.path,
            fd=duplicate,
            identity=root.identity,
            owned_fds=(duplicate,),
            root_fd=root.fd,
            root_path=root.path,
            root_identity=root.identity,
        )

    current_fd = root.fd
    owned: list[int] = []
    try:
        for component in components:
            child_fd = _open_directory_at(current_fd, component)
            owned.append(child_fd)
            current_fd = child_fd
        identity = _identity_from_fd(current_fd, "operation target")
        return PinnedDirectory(
            path=Path(os.path.abspath(target)),
            fd=current_fd,
            identity=identity,
            owned_fds=tuple(owned),
            root_fd=root.fd,
            root_path=root.path,
            root_identity=root.identity,
        )
    except Exception:
        for descriptor in reversed(owned):
            with contextlib.suppress(OSError):
                _close_fd(descriptor)
        raise


def pin_existing_under(
    root: PinnedDirectory,
    target: str | Path,
    allowed_root: str | Path,
) -> PinnedDirectory:
    """Pin a registered external checkout while retaining the plan root.

    Linked worktrees are commonly placed under a sibling worktree root rather
    than beneath the manager workspace.  The target is first opened relative
    to that explicitly configured root (so the allowance cannot become a
    general external-path escape), then the resulting target descriptors are
    attached to the manager root boundary.  Git metadata pointers are still
    bounded by ``root`` and are therefore permitted only when they point back
    into the registered workspace.
    """
    with open_directory(allowed_root) as boundary:
        candidate = pin_existing(boundary, target)
        try:
            return dataclasses.replace(
                candidate,
                root_fd=root.fd,
                root_path=root.path,
                root_identity=root.identity,
            )
        except Exception:
            candidate.close()
            raise


def pin_creation(
    root: PinnedDirectory, target: str | Path, *, create_parents: bool = True
) -> PinnedDirectory:
    """Pin the parent of an absent target, safely creating missing parents."""
    raw_candidate = Path(target).expanduser()
    components = _relative_components(root, raw_candidate)
    candidate = Path(os.path.abspath(raw_candidate))
    if not components:
        raise OperationBoundaryError("operation target needs a final component")
    leaf = components[-1]
    parent_components = components[:-1]
    current_fd = root.fd
    owned: list[int] = []
    try:
        for component in parent_components:
            try:
                child_fd = _open_directory_entry(current_fd, component)
            except FileNotFoundError:
                if not create_parents:
                    raise OperationBoundaryError(
                        f"parent component {component!r} does not exist"
                    ) from None
                try:
                    _mkdir_entry(current_fd, component)
                except FileExistsError:
                    pass
                child_fd = _open_directory_at(current_fd, component)
            except OSError as exc:
                raise OperationBoundaryError(
                    f"cannot open parent component {component!r} safely"
                ) from exc
            owned.append(child_fd)
            current_fd = child_fd
        identity = _identity_from_fd(current_fd, "operation parent")
        return PinnedDirectory(
            path=candidate.parent,
            fd=current_fd,
            identity=identity,
            owned_fds=tuple(owned),
            root_fd=root.fd,
            root_path=root.path,
            root_identity=root.identity,
            leaf=leaf,
        )
    except Exception:
        for descriptor in reversed(owned):
            with contextlib.suppress(OSError):
                _close_fd(descriptor)
        raise


def path_exists(root: PinnedDirectory, target: str | Path) -> bool:
    """Return whether target is a non-symlink directory beneath root."""
    components = _relative_components(root, Path(target))
    if not components:
        return True
    current_fd = root.fd
    descriptors: list[int] = []
    try:
        for component in components:
            result = _stat_at(current_fd, component)
            if result is None:
                return False
            if _is_link(result) or not stat.S_ISDIR(result.st_mode):
                raise OperationBoundaryError(
                    "operation target contains a non-directory"
                )
            child_fd = _open_directory_at(current_fd, component)
            descriptors.append(child_fd)
            current_fd = child_fd
        return True
    finally:
        for descriptor in reversed(descriptors):
            with contextlib.suppress(OSError):
                _close_fd(descriptor)


def read_at(directory_fd: int, name: str) -> bytes | None:
    """Read a regular file relative to a pinned directory without links."""
    try:
        fd = _open_file_entry(directory_fd, name, _read_flags())
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OperationBoundaryError(f"cannot read {name!r} safely") from exc
    try:
        result = os.fstat(fd)
        if not stat.S_ISREG(result.st_mode):
            raise OperationBoundaryError(f"{name!r} is not a regular file")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        return b"".join(chunks)
    finally:
        os.close(fd)


def write_at(directory_fd: int, name: str, value: bytes, *, mode: int = 0o644) -> None:
    """Replace a regular file relative to a pinned directory, without links.

    ``name`` is deliberately a single entry rather than a path.  The parent
    directory is already descriptor-pinned by the caller and ``O_NOFOLLOW``
    protects the final entry from being redirected to an external file.
    """
    if not name or Path(name).name != name or name in {".", ".."}:
        raise OperationBoundaryError("anchored file name must be one path component")
    existing = _writable_entry(directory_fd, name)
    flags = os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    if existing is None:
        flags |= os.O_CREAT | os.O_EXCL
    fd = _open_write_entry(directory_fd, name, flags, mode)
    try:
        _assert_writable_fd(fd, existing, name)
        # Do not pass O_TRUNC to open(2): if the final entry is swapped between
        # the no-follow stat and open, O_TRUNC would damage the replacement
        # before its identity check could refuse the operation.  Once the file
        # descriptor is proven to be the admitted inode, truncation is pinned to
        # that inode and cannot follow a renamed path.
        if existing is not None:
            os.ftruncate(fd, 0)
        _write_all(fd, value, name)
        os.fsync(fd)
    finally:
        os.close(fd)


def _writable_entry(directory_fd: int, name: str) -> os.stat_result | None:
    """Validate the current final entry before opening it for replacement."""
    existing = _stat_at(directory_fd, name)
    if existing is not None and (
        _is_link(existing)
        or not stat.S_ISREG(existing.st_mode)
        or existing.st_nlink != 1
    ):
        raise OperationBoundaryError(f"{name!r} is not a regular file")
    return existing


def _open_write_entry(directory_fd: int, name: str, flags: int, mode: int) -> int:
    """Open one regular entry with no-follow and return its descriptor."""
    try:
        return _open_file_entry(directory_fd, name, flags, mode)
    except OSError as exc:
        raise OperationBoundaryError(f"cannot write {name!r} safely") from exc


def _assert_writable_fd(fd: int, existing: os.stat_result | None, name: str) -> None:
    """Prove the opened descriptor is the entry validated immediately before it."""
    actual = os.fstat(fd)
    if not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1:
        raise OperationBoundaryError(f"{name!r} changed before it was written")
    if existing is not None and (
        actual.st_dev,
        actual.st_ino,
        actual.st_ctime_ns,
    ) != (
        existing.st_dev,
        existing.st_ino,
        existing.st_ctime_ns,
    ):
        raise OperationBoundaryError(f"{name!r} changed before it was written")


def _write_all(fd: int, value: bytes, name: str) -> None:
    """Write every byte to an already-validated regular-file descriptor."""
    offset = 0
    while offset < len(value):
        written = os.write(fd, value[offset:])
        if written <= 0:
            raise OperationBoundaryError(f"short write for {name!r}")
        offset += written


def stat_at(directory_fd: int, name: str) -> DirectoryIdentity | None:
    """Capture a no-follow directory-entry identity relative to a directory."""
    result = _stat_at(directory_fd, name)
    if result is None:
        return None
    return _identity_for_entry(result, directory_fd)


def _hash_bytes(value: bytes | None) -> str | None:
    """Hash optional metadata without persisting its contents."""
    if value is None:
        return None
    return hashlib.sha256(value).hexdigest()


def _open_relative_directory(
    parent_fd: int, components: tuple[str, ...], label: str
) -> tuple[int, list[int]]:
    """Open metadata directories relative to a pinned descriptor.

    Git linked-worktree pointers legitimately contain ``..``.  Those segments
    are handled by descriptor-relative ``openat`` calls rather than normalized
    into a mutable lexical path.
    """
    current_fd = parent_fd
    owned: list[int] = []
    try:
        for component in components:
            if component in ("", "."):
                continue
            name = ".." if component == ".." else component
            try:
                child_fd = _open_directory_entry(current_fd, name)
            except OSError as exc:
                raise OperationBoundaryError(
                    f"cannot open {label} metadata directory safely"
                ) from exc
            owned.append(child_fd)
            current_fd = child_fd
        return current_fd, owned
    except Exception:
        for descriptor in reversed(owned):
            with contextlib.suppress(OSError):
                _close_fd(descriptor)
        raise


def _metadata_directory(
    checkout_fd: int,
    git_entry: os.stat_result,
    label: str,
    *,
    root_fd: int | None = None,
    root_path: Path | None = None,
    checkout_path: Path | None = None,
) -> tuple[int, list[int], Path]:
    """Resolve a worktree's ``.git`` entry through no-follow descriptors."""
    if root_fd is None or root_path is None or checkout_path is None:
        raise OperationBoundaryError("Git metadata is missing its workspace boundary")
    if stat.S_ISDIR(git_entry.st_mode):
        components, metadata_path = _bounded_metadata_target(
            root_path, checkout_path, Path(".git"), label
        )
        metadata_fd, owned = _open_relative_directory(root_fd, components, label)
        _assert_entry_identity(checkout_fd, ".git", git_entry, label)
        return metadata_fd, owned, metadata_path
    if not stat.S_ISREG(git_entry.st_mode):
        raise OperationBoundaryError(
            f"{label} .git entry is not a directory or pointer"
        )
    return _metadata_pointer_directory(
        checkout_fd,
        git_entry,
        label,
        root_fd,
        root_path,
        checkout_path,
    )


def _metadata_pointer_directory(
    checkout_fd: int,
    git_entry: os.stat_result,
    label: str,
    root_fd: int,
    root_path: Path,
    checkout_path: Path,
) -> tuple[int, list[int], Path]:
    """Open the bounded metadata named by a linked-worktree pointer."""
    pointer = read_at(checkout_fd, ".git")
    if pointer is None:
        raise OperationBoundaryError(f"{label} .git pointer disappeared")
    _assert_entry_identity(checkout_fd, ".git", git_entry, label)
    try:
        text = pointer.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise OperationBoundaryError(f"{label} .git pointer is not UTF-8") from exc
    if not text.startswith("gitdir:"):
        raise OperationBoundaryError(f"{label} .git pointer is malformed")
    raw_path = text[len("gitdir:") :].strip()
    if not raw_path:
        raise OperationBoundaryError(f"{label} .git pointer has no target")
    components, metadata_path = _metadata_pointer_target(
        root_path, checkout_path, raw_path, label
    )
    metadata_fd, owned = _open_relative_directory(root_fd, components, label)
    return metadata_fd, owned, metadata_path


def _metadata_pointer_target(
    root_path: Path, checkout_path: Path, raw_path: str, label: str
) -> tuple[tuple[str, ...], Path]:
    """Bound a Git pointer, allowing external checkouts to point inward."""
    pointer_path = Path(raw_path)
    # An external linked-worktree checkout may legitimately store an absolute
    # pointer back into the manager workspace.  Its checkout path is outside
    # ``root_path`` by design, so using it as the base would reject the pointer
    # before the absolute target can be bounded.  Relative pointers remain
    # rooted at the checkout and therefore require the checkout itself to be
    # inside the admitted workspace.
    metadata_base = root_path if pointer_path.is_absolute() else checkout_path
    return _bounded_metadata_target(root_path, metadata_base, pointer_path, label)


def _common_metadata_directory(
    git_fd: int,
    metadata_path: Path,
    root_fd: int,
    root_path: Path,
    label: str,
) -> tuple[int, list[int], Path, str | None]:
    """Open a linked worktree's common Git directory without links."""
    commondir = read_at(git_fd, "commondir")
    if commondir is None:
        return git_fd, [], metadata_path, None
    try:
        raw_common = commondir.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise OperationBoundaryError("git commondir is not UTF-8") from exc
    if not raw_common:
        raise OperationBoundaryError("git commondir is empty")
    common_components, common_path = _bounded_metadata_target(
        root_path,
        metadata_path,
        Path(raw_common),
        "git common",
    )
    common_fd, common_owned = _open_relative_directory(
        root_fd,
        common_components,
        label,
    )
    return common_fd, common_owned, common_path, _hash_bytes(commondir)


def _config_entry_snapshot(
    directory_fd: int,
) -> tuple[FileIdentity | None, str | None, bytes | None]:
    """Read common Git config once and bind its metadata plus content digest."""
    descriptor = _open_config_entry(directory_fd)
    if descriptor is None:
        return None, None, None
    try:
        identity, value = _read_stable_config(descriptor)
        _assert_config_entry_identity(directory_fd, identity)
        return identity, hashlib.sha256(value).hexdigest(), value
    finally:
        with contextlib.suppress(OSError):
            _close_fd(descriptor)


def _open_config_entry(directory_fd: int) -> int | None:
    """Open common Git config without following its final entry."""
    try:
        return _open_file_entry(directory_fd, "config", _read_flags())
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise OperationBoundaryError("cannot read git common config safely") from exc


def _read_stable_config(descriptor: int) -> tuple[FileIdentity, bytes]:
    """Read config from one descriptor and reject concurrent content mutation."""
    mount_id = _mount_id_for_fd(descriptor)
    before = _regular_file_identity(descriptor, mount_id)
    chunks: list[bytes] = []
    while chunk := os.read(descriptor, 1024 * 1024):
        chunks.append(chunk)
    after = _regular_file_identity(descriptor, mount_id)
    if before != after:
        raise OperationBoundaryError("git common config changed during read")
    return after, b"".join(chunks)


def _regular_file_identity(descriptor: int, mount_id: int | None) -> FileIdentity:
    """Capture validated regular-file metadata from one open descriptor."""
    result = os.fstat(descriptor)
    if not stat.S_ISREG(result.st_mode) or result.st_nlink != 1:
        raise OperationBoundaryError("git common config is not a regular file")
    return FileIdentity.from_stat(result, mount_id)


def _assert_config_entry_identity(directory_fd: int, expected: FileIdentity) -> None:
    """Prove the config path still names the descriptor that was read."""
    current = _stat_at(directory_fd, "config")
    if current is None:
        raise OperationBoundaryError("git common config disappeared during read")
    actual = FileIdentity.from_stat(current, expected.mount_id)
    if actual != expected:
        raise OperationBoundaryError("git common config changed during read")


def _assert_config_snapshot(
    actual_identity: FileIdentity | None,
    actual_digest: str | None,
    expected_identity: FileIdentity | None,
    expected_digest: str | None,
) -> None:
    """Reject metadata or content drift from the admitted common config."""
    if (actual_identity, actual_digest) != (expected_identity, expected_digest):
        raise OperationBoundaryError("git common config content identity changed")


def _audit_local_git_config(config: bytes | None) -> str | None:
    """Return the sole origin URL after refusing destination rewrite controls."""
    entries = _git_config_entries(config)
    origins: list[str] = []
    for section, subsection, key, value in entries:
        _reject_config_destination_control(section, subsection, key)
        if section == "remote" and key in {"url", "pushurl"}:
            if subsection != "origin" or key == "pushurl":
                label = f"remote.{subsection or '<unnamed>'}.{key}"
                raise OperationBoundaryError(
                    f"git common config declares forbidden {label}"
                )
            origins.append(value)
    if len(origins) > 1:
        raise OperationBoundaryError("git common config repeats remote.origin.url")
    return origins[0] if origins else None


def _reject_config_destination_control(
    section: str, subsection: str | None, key: str
) -> None:
    """Reject config constructs that can import or rewrite a push destination."""
    if _is_external_config(section):
        raise OperationBoundaryError("git common config includes external config")
    if _is_worktree_config(section, key):
        raise OperationBoundaryError("git common config enables worktreeConfig")
    if _is_url_rewrite(section, key):
        raise OperationBoundaryError("git common config declares URL rewrite")
    if _is_push_default(section, subsection, key):
        raise OperationBoundaryError("git common config declares remote.pushDefault")
    if _is_branch_push_remote(section, key):
        raise OperationBoundaryError("git common config declares branch.pushRemote")


def _is_external_config(section: str) -> bool:
    """Return whether a section imports another configuration file."""
    return section in {"include", "includeif"}


def _is_worktree_config(section: str, key: str) -> bool:
    """Return whether a key enables the separate worktree config file."""
    return (section, key) == ("extensions", "worktreeconfig")


def _is_url_rewrite(section: str, key: str) -> bool:
    """Return whether a key rewrites fetch or push URL prefixes."""
    return section == "url" and key in {"insteadof", "pushinsteadof"}


def _is_push_default(section: str, subsection: str | None, key: str) -> bool:
    """Return whether a key selects an implicit default push remote."""
    return (section, subsection, key) == ("remote", None, "pushdefault")


def _is_branch_push_remote(section: str, key: str) -> bool:
    """Return whether a branch key selects an alternate push remote."""
    return (section, key) == ("branch", "pushremote")


def _reject_worktree_config(git_fd: int) -> None:
    """Refuse per-worktree configuration instead of leaving it outside the pin."""
    if read_at(git_fd, "config.worktree") is not None:
        raise OperationBoundaryError("git checkout declares config.worktree")


def _git_config_entries(
    value: bytes | None,
) -> tuple[tuple[str, str | None, str, str], ...]:
    """Parse the bounded local config grammar needed for destination auditing."""
    if value is None:
        return ()
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperationBoundaryError("git config is not UTF-8") from exc
    section: tuple[str, str | None] | None = None
    entries: list[tuple[str, str | None, str, str]] = []
    for raw_line in text.splitlines():
        section, entry = _parse_git_config_line(raw_line, section)
        if entry is not None:
            entries.append(entry)
    return tuple(entries)


def _parse_git_config_line(
    raw_line: str, section: tuple[str, str | None] | None
) -> tuple[tuple[str, str | None] | None, tuple[str, str | None, str, str] | None]:
    """Parse one local-config line while retaining its current section."""
    line = raw_line.strip()
    if not line or line.startswith(("#", ";")):
        return section, None
    if line.startswith("["):
        return _git_config_section(line), None
    if section is None:
        raise OperationBoundaryError("git config contains an unscoped key")
    match = re.fullmatch(r"([A-Za-z][A-Za-z0-9-]*)\s*(?:=\s*(.*))?", line)
    if match is None or raw_line.rstrip().endswith("\\"):
        raise OperationBoundaryError("git config uses unsupported syntax")
    entry = (*section, match.group(1).lower(), (match.group(2) or "").strip())
    return section, entry


def _git_config_section(line: str) -> tuple[str, str | None]:
    """Normalize modern and legacy Git section spellings."""
    match = re.fullmatch(
        r'\[\s*([A-Za-z][A-Za-z0-9-]*)(?:\.([A-Za-z0-9._-]+)|\s+"([^"\\]*)")?\s*\]',
        line,
    )
    if match is None:
        raise OperationBoundaryError("git config uses unsupported section syntax")
    return match.group(1).lower(), (match.group(2) or match.group(3))


def _reject_git_path_controls(argv: list[str]) -> None:
    """Reject Git options that could override descriptor-pinned paths."""
    if not argv or os.path.basename(argv[0]) != "git":
        return
    for index, token in enumerate(argv[1:], start=1):
        if token == "-" * 2:
            break
        _reject_git_path_control(argv, index, token)


def _reject_git_path_control(argv: list[str], index: int, token: str) -> None:
    """Reject one Git argument that can escape the admitted boundary."""
    if token in {
        "-C",
        "--git-dir",
        "--work-tree",
        "--separate-git-dir",
        "--exec-path",
    } or token.startswith(
        ("-C", "--git-dir=", "--work-tree=", "--separate-git-dir=", "--exec-path=")
    ):
        raise OperationBoundaryError(
            f"Git path-control option {token!r} is not allowed in a pinned operation"
        )
    if token in {"-c", "--config-env"}:
        value = argv[index + 1] if index + 1 < len(argv) else ""
        _reject_git_config_control(value)
    elif token.startswith("-c="):
        _reject_git_config_control(token[3:])
    elif token.startswith("-c") and len(token) > 2:
        _reject_git_config_control(token[2:])
    elif token.startswith("--config-env="):
        _reject_git_config_control(token.split("=", 1)[1])


def _reject_git_config_control(value: str) -> None:
    """Reject one config assignment that can redirect a pinned operation."""
    if _unsafe_git_config_key(value):
        raise OperationBoundaryError(
            f"Git config path-control option {value!r} is not allowed"
        )


def _unsafe_git_config_key(value: str) -> bool:
    """Identify config keys that can redirect a pinned checkout or remote."""
    key = value.split("=", 1)[0].strip().lower()
    return key in {
        "remote.origin.url",
        "remote.origin.pushurl",
        "core.worktree",
        "core.gitdir",
        "core.bare",
    } or key.startswith("include.")


def _bounded_metadata_target(
    root_path: Path,
    base_path: Path,
    raw_target: Path,
    label: str,
) -> tuple[tuple[str, ...], Path]:
    """Normalize an internal Git metadata pointer without leaving ``root``.

    Git linked-worktree pointers commonly contain ``..`` or are absolute, but
    the metadata they name must still belong to the admitted workspace.  The
    lexical stack is reduced before opening any component; descriptor-relative
    opens then avoid the resulting path becoming a mutable authority.
    """
    root = Path(os.path.abspath(root_path))
    base = Path(os.path.abspath(base_path))
    try:
        stack = list(base.relative_to(root).parts)
    except ValueError as exc:
        raise OperationBoundaryError(
            f"{label} metadata base escapes workspace"
        ) from exc

    target = Path(raw_target)
    if target.is_absolute():
        if ".." in target.parts:
            raise OperationBoundaryError(f"{label} metadata contains a parent segment")
        try:
            relative = target.relative_to(root)
        except ValueError as exc:
            raise OperationBoundaryError(f"{label} metadata escapes workspace") from exc
        stack = list(relative.parts)
        metadata_path = target
        return tuple(stack), metadata_path

    for component in target.parts:
        if component in ("", "."):
            continue
        if component == "..":
            if not stack:
                raise OperationBoundaryError(f"{label} metadata escapes workspace")
            stack.pop()
            continue
        stack.append(component)

    metadata_path = root.joinpath(*stack)
    try:
        metadata_path.relative_to(root)
    except ValueError as exc:
        raise OperationBoundaryError(f"{label} metadata escapes workspace") from exc
    return tuple(stack), metadata_path


def _assert_entry_identity(
    parent_fd: int, name: str, expected: os.stat_result, label: str
) -> None:
    """Reject a final-entry replacement observed during metadata admission."""
    actual = _stat_at(parent_fd, name)
    if actual is None or _identity_for_entry(actual, parent_fd) != _identity_for_entry(
        expected, parent_fd
    ):
        raise OperationBoundaryError(f"{label} metadata entry changed during admission")


def _remote_origin_values(value: bytes | None, key: str) -> tuple[str, ...]:
    """Read every origin remote value for ``key`` across repeated sections."""
    return tuple(
        entry_value
        for section, subsection, entry_key, entry_value in _git_config_entries(value)
        if section == "remote" and subsection == "origin" and entry_key == key
    )


def _remote_origin_value(value: bytes | None, key: str) -> str | None:
    """Read the first origin remote value for ``key`` from Git config."""
    values = _remote_origin_values(value, key)
    return values[0] if values else None


def _origin_from_git_config(value: bytes | None) -> str | None:
    """Read the first ``remote \"origin\"`` fetch URL from Git config."""
    return _remote_origin_value(value, "url")


def _push_origin_from_git_config(value: bytes | None) -> str | None:
    """Read the first ``remote \"origin\"`` push URL from Git config."""
    return _remote_origin_value(value, "pushurl")


def _metadata_snapshot(
    checkout: PinnedDirectory,
    git_entry: os.stat_result | None,
) -> dict[str, Any] | None:
    """Capture exact checkout metadata while its directory fd is pinned."""
    if git_entry is None:
        return None
    entry_identity = _identity_for_entry(git_entry, checkout.fd)
    if _is_link(git_entry):
        raise OperationBoundaryError("checkout .git entry is a symlink")
    git_fd, owned, metadata_path = _metadata_directory(
        checkout.fd,
        git_entry,
        str(checkout.path),
        root_fd=checkout.root_fd,
        root_path=checkout.root_path,
        checkout_path=checkout.path,
    )
    try:
        git_identity = _identity_from_fd(git_fd, "git worktree")
        common_fd, common_owned, _common_path, _common_pointer_digest = (
            _common_metadata_directory(
                git_fd,
                metadata_path,
                checkout.root_fd,
                checkout.root_path,
                str(checkout.path),
            )
        )
        try:
            common_identity = _identity_from_fd(common_fd, "git common")
            config_identity, config_digest, config = _config_entry_snapshot(common_fd)
            _audit_local_git_config(config)
            _reject_worktree_config(git_fd)
            head = read_at(git_fd, "HEAD")
            return {
                "entry": entry_identity.as_dict(),
                "worktree": git_identity.as_dict(),
                "common": common_identity.as_dict(),
                "config_digest": config_digest,
                "config_identity": (
                    config_identity.as_dict() if config_identity is not None else None
                ),
                "head_digest": _hash_bytes(head),
                "origin": _origin_from_git_config(config),
                "push_origin": _push_origin_from_git_config(config),
            }
        finally:
            for descriptor in reversed(common_owned):
                with contextlib.suppress(OSError):
                    _close_fd(descriptor)
    finally:
        for descriptor in reversed(owned):
            with contextlib.suppress(OSError):
                _close_fd(descriptor)


def snapshot_pinned_checkout(checkout: PinnedDirectory) -> dict[str, Any]:
    """Capture identity and Git metadata from one pinned checkout."""
    checkout.assert_path_identity()
    checkout._ensure_git_metadata()
    git_entry = _stat_at(checkout.fd, ".git")
    return {
        "path": str(checkout.path),
        "checkout": checkout.identity.as_dict(),
        "git": _metadata_snapshot(checkout, git_entry),
    }


def snapshot_workspace(
    root_path: str | Path,
    projects: list[tuple[str, str]],
) -> dict[str, Any]:
    """Snapshot root and exact project paths for an immutable release plan."""
    with open_directory(root_path) as root:
        root.assert_root_identity()
        snapshots: list[dict[str, Any]] = []
        for url, raw_path in projects:
            raw_target = Path(raw_path).expanduser()
            target = Path(os.path.abspath(raw_target))
            if not path_exists(root, raw_target):
                snapshots.append(
                    {
                        "url": str(url),
                        "path": str(target),
                        "exists": False,
                        "checkout": None,
                        "git": None,
                    }
                )
                continue
            with pin_existing(root, raw_target) as checkout:
                snapshot = snapshot_pinned_checkout(checkout)
                snapshots.append(
                    {
                        "url": str(url),
                        "path": snapshot["path"],
                        "exists": True,
                        "checkout": snapshot["checkout"],
                        "git": snapshot["git"],
                    }
                )
        return {
            "root": root.identity.as_dict(),
            "projects": snapshots,
        }


def _receipt_name() -> str:
    """Filename for the one durable release-plan consumption record."""
    return ".repository-manager-release-plan.json"


def read_release_plan_receipt(root: PinnedDirectory) -> dict[str, Any] | None:
    """Read the durable anti-replay record beneath a pinned workspace root."""
    raw = read_at(root.fd, _receipt_name())
    if raw is None:
        return None
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OperationBoundaryError("release-plan receipt is malformed") from exc
    if not isinstance(value, dict):
        raise OperationBoundaryError("release-plan receipt must be a mapping")
    return value


def _fsync_directory(directory_fd: int) -> None:
    """Durably flush a directory after an entry create, replace, or remove."""
    try:
        _fsync_directory_fd(directory_fd)
    except OSError as exc:
        raise OperationBoundaryError("cannot durably flush receipt directory") from exc


def _atomic_write_at(
    directory_fd: int, name: str, value: bytes, *, mode: int = 0o600
) -> None:
    """Write a regular entry through a durable same-directory replacement."""
    if not name or Path(name).name != name or name in {".", ".."}:
        raise OperationBoundaryError("anchored file name must be one path component")
    temporary = f".{name}.{os.getpid()}.{id(value):x}.tmp"
    temporary_fd: int | None = None
    try:
        temporary_fd = _open_write_entry(
            directory_fd,
            temporary,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0),
            mode,
        )
        _assert_writable_fd(temporary_fd, None, temporary)
        _write_all(temporary_fd, value, temporary)
        os.fsync(temporary_fd)
        os.close(temporary_fd)
        temporary_fd = None
        existing = _stat_at(directory_fd, name)
        if existing is not None and (
            _is_link(existing)
            or not stat.S_ISREG(existing.st_mode)
            or existing.st_nlink != 1
        ):
            raise OperationBoundaryError(f"{name!r} is not a regular file")
        try:
            _replace_entry(directory_fd, temporary, name)
        except OSError as exc:
            raise OperationBoundaryError(f"cannot replace {name!r} safely") from exc
        _fsync_directory(directory_fd)
    finally:
        if temporary_fd is not None:
            with contextlib.suppress(OSError):
                os.close(temporary_fd)
        with contextlib.suppress(FileNotFoundError):
            _remove_entry(directory_fd, temporary)


def _write_receipt_start_at(directory_fd: int, name: str, value: bytes) -> None:
    """Create the initial receipt marker and durably flush its parent."""
    write_at(directory_fd, name, value, mode=0o600)
    _fsync_directory(directory_fd)


def write_release_plan_receipt(
    root: PinnedDirectory, value: dict[str, Any], *, atomic: bool = False
) -> None:
    """Persist one bounded release-plan receipt using a no-follow root fd.

    The initial consumption marker is created and parent-directory flushed;
    completion records use an atomic same-directory replacement before the
    parent directory is flushed.
    """
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    writers = {False: _write_receipt_start_at, True: _atomic_write_at}
    writers[atomic](root.fd, _receipt_name(), encoded)


def _cleanup_entry_identity(
    directory_fd: int, name: str, result: os.stat_result, label: str
) -> DirectoryIdentity:
    """Capture an entry identity, including a mounted child directory's ID."""
    if not stat.S_ISDIR(result.st_mode):
        return _identity_for_entry(result, directory_fd)
    child_fd = _open_directory_at(directory_fd, name)
    try:
        identity = _identity_from_fd(child_fd, f"cleanup {label}/{name}")
    finally:
        with contextlib.suppress(OSError):
            _close_fd(child_fd)
    plain = DirectoryIdentity.from_stat(result)
    if (identity.device, identity.inode, identity.mode) != (
        plain.device,
        plain.inode,
        plain.mode,
    ):
        raise OperationBoundaryError(
            f"cleanup {label}/{name} changed during inspection"
        )
    return identity


def _cleanup_entries(
    directory_fd: int, label: str
) -> list[tuple[str, os.stat_result, DirectoryIdentity]]:
    """List stable no-follow entries, rejecting links and unsupported nodes."""
    try:
        names = sorted(_list_entries(directory_fd))
    except OSError as exc:
        raise OperationBoundaryError(
            f"cannot scan {label} without following links"
        ) from exc
    entries: list[tuple[str, os.stat_result, DirectoryIdentity]] = []
    for name in names:
        result = _stat_at(directory_fd, name)
        if result is None:
            continue
        if _is_link(result):
            raise OperationBoundaryError(f"cleanup refuses symlink entry {name!r}")
        if not (stat.S_ISDIR(result.st_mode) or stat.S_ISREG(result.st_mode)):
            raise OperationBoundaryError(f"cleanup refuses unsupported entry {name!r}")
        entries.append(
            (name, result, _cleanup_entry_identity(directory_fd, name, result, label))
        )
    return entries


def _assert_cleanup_entry(
    directory_fd: int,
    name: str,
    expected_identity: DirectoryIdentity,
    label: str,
) -> os.stat_result:
    """Recheck one entry immediately before a descriptor-relative mutation."""
    current = _stat_at(directory_fd, name)
    if (
        current is None
        or _cleanup_entry_identity(directory_fd, name, current, label)
        != expected_identity
    ):
        raise OperationBoundaryError(f"cleanup {label} was replaced before mutation")
    return current


def _assert_cleanup_child_fd(
    child_fd: int, expected_identity: DirectoryIdentity, label: str
) -> None:
    """Prove an opened child still names the entry admitted by its parent."""
    if _identity_from_fd(child_fd, f"cleanup {label}") != expected_identity:
        raise OperationBoundaryError(f"cleanup {label} was rebound before descent")


def _assert_cleanup_plan(
    entries: list[tuple[str, os.stat_result, DirectoryIdentity]],
    plan: dict[tuple[str, ...], DirectoryIdentity],
    relative_path: tuple[str, ...],
    label: str,
) -> None:
    """Reject additions or replacements since the cleanup preflight."""
    for name, _result, identity in entries:
        expected = plan.get((*relative_path, name))
        if expected is None or expected != identity:
            raise OperationBoundaryError(
                f"cleanup {label}/{name} changed after preflight"
            )


def _preflight_cleanup_tree(
    directory_fd: int,
    *,
    ignored_directory_names: frozenset[str],
    label: str,
    relative_path: tuple[str, ...] = (),
) -> dict[tuple[str, ...], DirectoryIdentity]:
    """Inspect the complete mutable cleanup tree before deleting anything."""
    plan: dict[tuple[str, ...], DirectoryIdentity] = {}
    for name, result, identity in _cleanup_entries(directory_fd, label):
        entry_path = (*relative_path, name)
        plan[entry_path] = identity
        if not stat.S_ISDIR(result.st_mode) or name in ignored_directory_names:
            continue
        child_fd = _open_directory_at(directory_fd, name)
        try:
            _assert_cleanup_child_fd(child_fd, identity, f"{label}/{name}")
            _assert_cleanup_entry(directory_fd, name, identity, label)
            plan.update(
                _preflight_cleanup_tree(
                    child_fd,
                    ignored_directory_names=ignored_directory_names,
                    label=f"{label}/{name}",
                    relative_path=entry_path,
                )
            )
        finally:
            with contextlib.suppress(OSError):
                _close_fd(child_fd)
    return plan


def _remove_cleanup_tree(
    directory_fd: int,
    name: str,
    expected_identity: DirectoryIdentity,
    *,
    label: str,
    plan: dict[tuple[str, ...], DirectoryIdentity],
    relative_path: tuple[str, ...],
) -> None:
    """Remove one already-preflighted subtree through parent dirfds only."""
    _assert_cleanup_entry(directory_fd, name, expected_identity, label)
    child_fd = _open_directory_at(directory_fd, name)
    try:
        _assert_cleanup_child_fd(child_fd, expected_identity, label)
        child_entries = _cleanup_entries(child_fd, label)
        _assert_cleanup_plan(child_entries, plan, relative_path, label)
        for child_name, child_result, child_identity in child_entries:
            if stat.S_ISDIR(child_result.st_mode):
                _remove_cleanup_tree(
                    child_fd,
                    child_name,
                    child_identity,
                    label=f"{label}/{child_name}",
                    plan=plan,
                    relative_path=(*relative_path, child_name),
                )
            else:
                _assert_cleanup_entry(child_fd, child_name, child_identity, label)
                try:
                    _remove_entry(child_fd, child_name)
                except OSError as exc:
                    raise OperationBoundaryError(
                        f"cannot remove cleanup file {child_name!r} safely"
                    ) from exc
        _assert_cleanup_entry(directory_fd, name, expected_identity, label)
    finally:
        with contextlib.suppress(OSError):
            _close_fd(child_fd)
    try:
        _remove_entry(directory_fd, name, directory=True)
    except OSError as exc:
        raise OperationBoundaryError(
            f"cannot remove cleanup directory {name!r} safely"
        ) from exc


def cleanup_pinned_directory(
    directory: PinnedDirectory,
    *,
    file_patterns: tuple[str, ...],
    directory_names: frozenset[str],
    ignored_directory_names: frozenset[str],
    root_script_patterns: tuple[str, ...],
) -> None:
    """Clean one pinned directory with a no-follow, descriptor-only walk.

    The complete tree is preflighted first.  Every later descent and removal
    rechecks the no-follow entry identity and uses ``dir_fd`` operations, so a
    nested symlink or directory rebind cannot redirect deletion outside the
    pinned tree.
    """
    directory.assert_operation_identity()
    cleanup_plan = _preflight_cleanup_tree(
        directory.fd,
        ignored_directory_names=ignored_directory_names,
        label=str(directory.path),
    )

    def matches(name: str, patterns: tuple[str, ...]) -> bool:
        return any(fnmatch.fnmatchcase(name, pattern) for pattern in patterns)

    def remove_files(
        parent_fd: int,
        entries: list[tuple[str, os.stat_result, DirectoryIdentity]],
        *,
        is_root: bool,
    ) -> None:
        for name, result, identity in entries:
            if not stat.S_ISREG(result.st_mode):
                continue
            root_match = is_root and (
                matches(name, root_script_patterns)
                or (
                    name.endswith(".txt")
                    and name
                    not in {
                        "requirements.txt",
                        "requirements-dev.txt",
                    }
                )
            )
            if root_match or matches(name, file_patterns):
                _assert_cleanup_entry(parent_fd, name, identity, "file")
                try:
                    _remove_entry(parent_fd, name)
                except OSError as exc:
                    raise OperationBoundaryError(
                        f"cannot remove cleanup file {name!r} safely"
                    ) from exc

    def remove_at(
        parent_fd: int,
        *,
        is_root: bool,
        label: str,
        relative_path: tuple[str, ...],
    ) -> None:
        entries = _cleanup_entries(parent_fd, label)
        _assert_cleanup_plan(entries, cleanup_plan, relative_path, label)
        remove_files(parent_fd, entries, is_root=is_root)
        for name, result, identity in entries:
            if not stat.S_ISDIR(result.st_mode):
                continue
            if name in ignored_directory_names:
                continue
            if name in directory_names:
                _remove_cleanup_tree(
                    parent_fd,
                    name,
                    identity,
                    label=f"{label}/{name}",
                    plan=cleanup_plan,
                    relative_path=(*relative_path, name),
                )
            else:
                _assert_cleanup_entry(parent_fd, name, identity, "directory")
                child_fd = _open_directory_at(parent_fd, name)
                try:
                    _assert_cleanup_child_fd(child_fd, identity, f"{label}/{name}")
                    remove_at(
                        child_fd,
                        is_root=False,
                        label=f"{label}/{name}",
                        relative_path=(*relative_path, name),
                    )
                finally:
                    with contextlib.suppress(OSError):
                        _close_fd(child_fd)

    remove_at(
        directory.fd,
        is_root=True,
        label=str(directory.path),
        relative_path=(),
    )
    directory.assert_operation_identity()


def receipt_result_payload(results: list[Any]) -> list[dict[str, Any]]:
    """Serialize model-like results without retaining command paths."""
    payload: list[dict[str, Any]] = []
    for result in results:
        if hasattr(result, "model_dump"):
            value = result.model_dump(mode="json")
        elif isinstance(result, dict):
            value = dict(result)
        else:
            raise OperationBoundaryError("release result cannot be persisted")
        if not isinstance(value, dict):
            raise OperationBoundaryError("release result payload must be a mapping")
        payload.append(value)
    return payload


def discard_empty_reservation(destination: PinnedDirectory) -> bool:
    """Remove a reserved clone target that is still empty after a handoff.

    The reservation descriptor is released first: a pinned directory refuses
    deletion while held (Windows denies delete sharing on pinned handles).
    Returns whether the reservation was removed.
    """
    if destination.target_fd is None or destination.leaf is None:
        return False
    if _list_entries(destination.target_fd):
        return False
    target_fd, destination.target_fd = destination.target_fd, None
    _close_fd(target_fd)
    _remove_entry(destination.fd, destination.leaf, directory=True)
    return True


def _close_quietly(handle: PinnedDirectory) -> None:
    """Close a temporary handle for use with ``contextlib.closing``."""
    handle.close()


@contextlib.contextmanager
def pinned_path(path: str | Path) -> Iterator[PinnedDirectory]:
    """Context manager for direct git actions outside a workspace root."""
    handle = open_directory(path)
    try:
        yield handle
    finally:
        _close_quietly(handle)
