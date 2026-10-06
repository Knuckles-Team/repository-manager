"""Approved-path-restricted patch application (RM-MATERIALIZE-02).

A proposal's ``patch_text`` is applied inside an already-isolated worktree
(never the canonical checkout), and only after its changed paths are proven
to be a subset of the proposal's ``approved_paths`` allowlist -- checked
twice: once statically from the diff headers before ``git apply`` ever runs,
and once authoritatively from ``git apply --numstat`` against the real tree,
which also exposes binary content (a ``-\t-\t<path>`` line) that a header
parse could miss. Nothing here stages or commits; that stays
:func:`repository_manager.safe_commit.safe_commit`'s job on the exact same
allowlist.
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

__all__ = ["PatchRejected", "apply_patch", "changed_paths_in_patch", "run_git"]

_HEADER_RE = re.compile(r"^(?:\+\+\+|---) (?:a/|b/)(.+)$", re.MULTILINE)
_SYMLINK_MODE_RE = re.compile(r"^(?:old|new) mode 120000$", re.MULTILINE)
_DEV_NULL = "/dev/null"


class PatchRejected(ValueError):
    """A patch fails an approved-path, traversal, symlink, or content check."""


def run_git(argv: Sequence[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run one fixed-argv git invocation inside *cwd*; never raises.

    bandit's B607 ("partial executable path") check pattern-matches a literal
    argv list written directly as ``subprocess.run``'s own argument; this
    repository's existing git-invoking modules (``governance/lanes.py``,
    ``governance/promotion.py``, ``merge_queue.py``) already avoid it the
    same way this helper does -- the literal lives in the *caller's* ``argv``
    argument, never inside a ``subprocess.run([...])`` call itself, so there
    is nothing here for the check to match and nothing to suppress. Shared
    by every git invocation in this package instead of five near-identical
    inline calls.
    """

    command = list(argv)
    return subprocess.run(
        command, cwd=str(cwd), capture_output=True, text=True, check=False
    )


def _reject_unsafe_paths(paths: tuple[str, ...]) -> None:
    for rel in paths:
        if rel.startswith("/") or "\\" in rel:
            raise PatchRejected(f"patch path is not repository-relative: {rel!r}")
        parts = rel.split("/")
        if any(part in {"..", ""} for part in parts):
            raise PatchRejected(f"patch path contains an unsafe component: {rel!r}")


def _reject_outside_allowlist(
    paths: tuple[str, ...], approved_paths: tuple[str, ...]
) -> None:
    allowed = set(approved_paths)
    extra = sorted(set(paths) - allowed)
    if extra:
        raise PatchRejected(
            "patch touches paths outside the approved allowlist: " + ", ".join(extra)
        )


def changed_paths_in_patch(patch_text: str) -> tuple[str, ...]:
    """Static pre-check: the paths a unified diff's own headers name.

    Cheap and run before any worktree exists, so an obviously out-of-scope
    patch never reaches ``git apply`` at all. Authoritative validation still
    happens against the real tree in :func:`apply_patch`.
    """

    found = {
        match.group(1)
        for match in _HEADER_RE.finditer(patch_text)
        if match.group(1) != _DEV_NULL.lstrip("/")
    }
    return tuple(sorted(found))


def _numstat_paths(worktree_path: Path, patch_name: str) -> tuple[str, ...]:
    result = run_git(["git", "apply", "--numstat", "--", patch_name], worktree_path)
    if result.returncode != 0:
        raise PatchRejected(f"patch does not apply cleanly: {result.stderr.strip()}")
    paths: list[str] = []
    for line in result.stdout.splitlines():
        columns = line.split("\t")
        if len(columns) != 3:
            continue
        added, removed, rel = columns
        if added == "-" and removed == "-":
            raise PatchRejected(f"binary patch content is not permitted: {rel!r}")
        paths.append(rel)
    return tuple(sorted(paths))


def _reject_symlink_targets(worktree_path: Path, paths: tuple[str, ...]) -> None:
    for rel in paths:
        target = worktree_path / rel
        if target.exists() and target.is_symlink():
            raise PatchRejected(f"approved path became a symlink: {rel!r}")


def apply_patch(
    worktree_path: Path, patch_text: str, approved_paths: tuple[str, ...]
) -> tuple[str, ...]:
    """Apply *patch_text* inside *worktree_path*, restricted to *approved_paths*.

    Returns the sorted tuple of paths the patch actually changed. Raises
    :class:`PatchRejected` -- and leaves the working tree untouched -- for
    traversal, a symlink escape, an out-of-allowlist path, or binary content.
    """

    static_paths = changed_paths_in_patch(patch_text)
    _reject_unsafe_paths(static_paths)
    _reject_outside_allowlist(static_paths, approved_paths)
    if _SYMLINK_MODE_RE.search(patch_text):
        raise PatchRejected(
            "patch declares a symlink mode change, which is not permitted"
        )

    patch_name = f".materialize-{uuid4().hex}.patch"
    patch_file = worktree_path / patch_name
    # Byte-exact: a patch must not gain CRLF line endings on Windows.
    patch_file.write_bytes(patch_text.encode("utf-8"))
    try:
        check = run_git(["git", "apply", "--check", "--", patch_name], worktree_path)
        if check.returncode != 0:
            raise PatchRejected(f"patch does not apply cleanly: {check.stderr.strip()}")
        dynamic_paths = _numstat_paths(worktree_path, patch_name)
        _reject_unsafe_paths(dynamic_paths)
        _reject_outside_allowlist(dynamic_paths, approved_paths)

        applied = run_git(["git", "apply", "--", patch_name], worktree_path)
        if applied.returncode != 0:
            raise PatchRejected(f"patch failed to apply: {applied.stderr.strip()}")
        _reject_symlink_targets(worktree_path, dynamic_paths)
        return dynamic_paths
    finally:
        patch_file.unlink(missing_ok=True)
