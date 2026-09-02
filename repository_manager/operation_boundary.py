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

The handles are intentionally independent of the repository-manager class so
the same primitive can protect sync, pull, push, and release-plan snapshots.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import re
import stat
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

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

    @classmethod
    def from_stat(cls, result: os.stat_result) -> DirectoryIdentity:
        """Capture the identity fields relevant to a directory boundary."""
        return cls(
            device=int(result.st_dev),
            inode=int(result.st_ino),
            mode=stat.S_IFMT(result.st_mode),
        )

    def as_dict(self) -> dict[str, int]:
        """Return a JSON-safe identity payload."""
        return {
            "device": self.device,
            "inode": self.inode,
            "mode": self.mode,
        }


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


def _path_components(path: Path) -> tuple[str, ...]:
    """Return absolute path components without silently erasing traversal."""
    if ".." in path.parts:
        raise OperationBoundaryError("path contains a lexical parent segment")
    absolute = Path(os.path.abspath(path))
    if not absolute.is_absolute():
        raise OperationBoundaryError("operation path must be absolute")
    return tuple(part for part in absolute.parts if part not in ("", os.sep))


def _open_directory_at(parent_fd: int, component: str) -> int:
    """Open one directory component relative to an already-open directory."""
    try:
        return os.open(component, _directory_flags(), dir_fd=parent_fd)
    except OSError as exc:
        entry = _stat_at(parent_fd, component)
        if entry is not None and stat.S_ISLNK(entry.st_mode):
            raise OperationBoundaryError(
                f"directory component {component!r} is a symlink"
            ) from exc
        raise OperationBoundaryError(
            f"cannot open directory component {component!r} without following links"
        ) from exc


def _stat_at(parent_fd: int, component: str) -> os.stat_result | None:
    """Stat one directory entry without following a final symlink."""
    try:
        return os.stat(component, dir_fd=parent_fd, follow_symlinks=False)
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
        return f"/proc/self/fd/{self.fd}"

    @property
    def destination_path(self) -> str:
        """A clone destination anchored to the reserved target descriptor."""
        if self.target_fd is not None:
            return f"/proc/self/fd/{self.target_fd}"
        if self.leaf is None:
            raise OperationBoundaryError("clone destination has no final component")
        return f"{self.proc_path}/{self.leaf}"

    @property
    def pass_fds(self) -> tuple[int, ...]:
        """Descriptors that must survive long enough for the child to chdir."""
        values = list(self.owned_fds)
        if self.root_fd not in values:
            values.append(self.root_fd)
        if self.target_fd is not None and self.target_fd not in values:
            values.append(self.target_fd)
        for descriptor in self.git_owned_fds:
            if descriptor not in values:
                values.append(descriptor)
        return tuple(values)

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
        origin_override = (
            [
                "-c",
                f"remote.origin.url={self.expected_origin}",
                "-c",
                f"remote.origin.pushurl={self.expected_origin}",
            ]
            if self.expected_origin is not None
            else []
        )
        return [
            argv[0],
            *origin_override,
            f"--git-dir=/proc/self/fd/{self.git_fd}",
            f"--work-tree={self.proc_path}",
            *argv[1:],
        ]

    @staticmethod
    def anchored_git_environment(env: dict[str, str]) -> dict[str, str]:
        """Remove environment controls that can redirect Git's writes."""
        return {
            key: value
            for key, value in env.items()
            if key not in _PINNED_GIT_ENVIRONMENT
            and not key.startswith(("GIT_CONFIG_KEY_", "GIT_CONFIG_VALUE_"))
        }

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
        if stat.S_ISLNK(git_entry.st_mode):
            raise OperationBoundaryError("checkout .git entry is a symlink")
        if stat.S_ISREG(git_entry.st_mode) and git_entry.st_nlink != 1:
            raise OperationBoundaryError("checkout .git entry is a hard link")
        entry_identity = DirectoryIdentity.from_stat(git_entry)
        if (
            self.git_entry_identity is not None
            and entry_identity != self.git_entry_identity
        ):
            raise OperationBoundaryError("checkout .git entry identity changed")
        if self.git_fd is None:
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
        _assert_fd_identity(self.git_fd, self.git_identity, "git worktree")
        return self.git_fd

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
        if stat.S_ISLNK(result.st_mode):
            raise OperationBoundaryError("clone target became a symlink")
        actual = DirectoryIdentity.from_stat(result)
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
            os.mkdir(self.leaf, mode=0o755, dir_fd=self.fd)
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
        self.target_identity = DirectoryIdentity.from_stat(os.fstat(target_fd))

    def close(self) -> None:
        """Close only descriptors owned by this handle, once."""
        if self._closed:
            return
        self._closed = True
        self.boundary_assertion = None
        for descriptor in reversed(self.owned_fds):
            with contextlib.suppress(OSError):
                os.close(descriptor)
        for descriptor in reversed(self.git_owned_fds):
            if descriptor not in self.owned_fds:
                with contextlib.suppress(OSError):
                    os.close(descriptor)
        if self.target_fd is not None and self.target_fd not in self.owned_fds:
            if self.target_fd not in self.git_owned_fds:
                with contextlib.suppress(OSError):
                    os.close(self.target_fd)


def _identity_from_fd(fd: int, label: str) -> DirectoryIdentity:
    """Capture a directory fd identity and require an actual directory."""
    try:
        result = os.fstat(fd)
    except OSError as exc:
        raise OperationBoundaryError(f"cannot inspect {label} descriptor") from exc
    if not stat.S_ISDIR(result.st_mode):
        raise OperationBoundaryError(f"{label} is not a directory")
    return DirectoryIdentity.from_stat(result)


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
    actual = DirectoryIdentity.from_stat(result)
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
        if stat.S_ISLNK(result.st_mode):
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
        root_fd = os.open(os.sep, _directory_flags())
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
        return os.open(component, _directory_flags(), dir_fd=parent_fd)
    except FileNotFoundError:
        if not create:
            raise OperationBoundaryError(
                f"directory component {component!r} does not exist"
            ) from None
        try:
            os.mkdir(component, mode=0o755, dir_fd=parent_fd)
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
        if entry is not None and stat.S_ISLNK(entry.st_mode):
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
            os.close(descriptor)


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
        duplicate = os.dup(root.fd)
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
                os.close(descriptor)
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
                child_fd = os.open(component, _directory_flags(), dir_fd=current_fd)
            except FileNotFoundError:
                if not create_parents:
                    raise OperationBoundaryError(
                        f"parent component {component!r} does not exist"
                    ) from None
                try:
                    os.mkdir(component, mode=0o755, dir_fd=current_fd)
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
                os.close(descriptor)
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
            if stat.S_ISLNK(result.st_mode) or not stat.S_ISDIR(result.st_mode):
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
                os.close(descriptor)


def read_at(directory_fd: int, name: str) -> bytes | None:
    """Read a regular file relative to a pinned directory without links."""
    try:
        fd = os.open(name, _read_flags(), dir_fd=directory_fd)
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
        stat.S_ISLNK(existing.st_mode)
        or not stat.S_ISREG(existing.st_mode)
        or existing.st_nlink != 1
    ):
        raise OperationBoundaryError(f"{name!r} is not a regular file")
    return existing


def _open_write_entry(directory_fd: int, name: str, flags: int, mode: int) -> int:
    """Open one regular entry with no-follow and return its descriptor."""
    try:
        return os.open(name, flags, mode, dir_fd=directory_fd)
    except OSError as exc:
        raise OperationBoundaryError(f"cannot write {name!r} safely") from exc


def _assert_writable_fd(fd: int, existing: os.stat_result | None, name: str) -> None:
    """Prove the opened descriptor is the entry validated immediately before it."""
    actual = os.fstat(fd)
    if not stat.S_ISREG(actual.st_mode) or actual.st_nlink != 1:
        raise OperationBoundaryError(f"{name!r} changed before it was written")
    if existing is not None and (actual.st_dev, actual.st_ino) != (
        existing.st_dev,
        existing.st_ino,
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
    return DirectoryIdentity.from_stat(result)


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
                child_fd = os.open(name, _directory_flags(), dir_fd=current_fd)
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
                os.close(descriptor)
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
    if actual is None or DirectoryIdentity.from_stat(
        actual
    ) != DirectoryIdentity.from_stat(expected):
        raise OperationBoundaryError(f"{label} metadata entry changed during admission")


def _origin_from_git_config(value: bytes | None) -> str | None:
    """Read the first ``remote \"origin\"`` URL from a git config blob."""
    if value is None:
        return None
    try:
        text = value.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise OperationBoundaryError("git config is not UTF-8") from exc
    section = re.search(
        r'(?ims)^[ \t]*\[remote[ \t]+"origin"[ \t]*\][ \t]*\n'
        r"(.*?)(?=^[ \t]*\[|\Z)",
        text,
    )
    if section is None:
        return None
    urls = re.findall(r"(?im)^[ \t]*url[ \t]*=[ \t]*(\S.*?)\s*$", section.group(1))
    return urls[0].strip() if urls else None


def _metadata_snapshot(
    checkout: PinnedDirectory,
    git_entry: os.stat_result | None,
) -> dict[str, Any] | None:
    """Capture exact checkout metadata while its directory fd is pinned."""
    if git_entry is None:
        return None
    entry_identity = DirectoryIdentity.from_stat(git_entry)
    if stat.S_ISLNK(git_entry.st_mode):
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
        common_fd = git_fd
        common_owned: list[int] = []
        commondir = read_at(git_fd, "commondir")
        if commondir is not None:
            try:
                raw_common = commondir.decode("utf-8").strip()
            except UnicodeDecodeError as exc:
                raise OperationBoundaryError("git commondir is not UTF-8") from exc
            if not raw_common:
                raise OperationBoundaryError("git commondir is empty")
            common_path = Path(raw_common)
            common_components, _common_target = _bounded_metadata_target(
                checkout.root_path,
                metadata_path,
                common_path,
                "git common",
            )
            common_fd, common_owned = _open_relative_directory(
                checkout.root_fd,
                common_components,
                "git common",
            )
        try:
            common_identity = _identity_from_fd(common_fd, "git common")
            config = read_at(git_fd, "config")
            head = read_at(git_fd, "HEAD")
            return {
                "entry": entry_identity.as_dict(),
                "worktree": git_identity.as_dict(),
                "common": common_identity.as_dict(),
                "config_digest": _hash_bytes(config),
                "head_digest": _hash_bytes(head),
                "origin": _origin_from_git_config(config),
            }
        finally:
            for descriptor in reversed(common_owned):
                with contextlib.suppress(OSError):
                    os.close(descriptor)
    finally:
        for descriptor in reversed(owned):
            with contextlib.suppress(OSError):
                os.close(descriptor)


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


def write_release_plan_receipt(root: PinnedDirectory, value: dict[str, Any]) -> None:
    """Persist one bounded release-plan receipt using a no-follow root fd."""
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    write_at(root.fd, _receipt_name(), encoded, mode=0o600)


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
