"""Move a canonical checkout's WIP off the shared ``refs/stash`` stack
(CONCEPT:RM-STASH-GUARD, registry ``D-CP-1``).

Background — the hazard this closes: ``WorktreeManager.add(..., adopt=True)``
used to do::

    git stash push -u -m "rm_worktree adopt"   # canonical
    ...                                         # create the new worktree
    git stash pop                               # canonical or the new worktree

Every worktree under ``WORKTREE_ROOT`` shares one ``.git`` object/ref store with
the canonical checkout it was created from, and ``refs/stash`` is a **single
global stack** in that store — not per-worktree. ``git stash push`` always
writes to ``stash@{0}`` and ``git stash pop`` always reads it back, regardless
of which worktree or process issued the command. So the window between our
``push`` and our ``pop`` is a real race: any other actor sharing this ``.git``
(another repository-manager operation, a human or tool running a raw
``git stash`` directly in *any* worktree of this repo) that pushes its own
stash in that window lands on top of ours, and our ``pop`` then takes *its*
WIP instead of ours — crossing two lanes' uncommitted work, or burying one of
them until someone notices ``git stash list`` is non-empty. This is not
hypothetical: it is why every lane in this workspace's current session was
told **never to run ``git stash`` directly**.

The fix: never touch the shared stack at all. :func:`capture_wip` and
:func:`park` create a temporary index, stage tracked and untracked WIP into it,
write a tree and a private commit with ``git write-tree``/``git commit-tree``,
and point ``refs/lane/...`` at that commit. Only after the ref is durable do
they restore ``HEAD`` into the real index and remove the captured untracked
paths. The canonical capture holds
(:func:`repository_manager.canonical_guard.hold_canonical_lease`) while this
sequence runs, and lane capture uses a repository-local lease; neither relies
on a human or another process respecting stack order.

From that point on our WIP lives only on the private ref, which behaves like
any other git ref — nothing but code that knows its name can touch it.
:func:`apply_and_clear` restores the private commit into a clean target with
``git read-tree`` and a mixed reset, then deletes the private ref only after
the restore succeeds. A dirty or conflicted target leaves the ref in place and
reports its name, so the caller can surface it for manual recovery instead of
losing the WIP.

**Why this is a sibling of, not a call into,**
``agent_utilities.governance.lanes`` **(the ecosystem's PARTITION-class
pattern for exactly this hazard, registry note on ``D-CP-1``).**
``lanes.partitioned_paths(path).stash_ref`` (``refs/lane/<lane>/stash``) plus
``lanes.park_worktree``/``unpark_worktree`` is the sanctioned "give a lane a
private ref instead of the shared stack" primitive, and this module follows
the same namespace convention (``refs/lane/...``) deliberately. It is not
reused directly for two structural reasons, not preference:

* ``park_worktree`` calls ``lanes.require_mutable_tree``, which **raises**
  ``CanonicalCheckoutError`` for a canonical checkout (``scope.is_canonical``)
  outside a merge in progress — by design (au's own "never edit the canonical
  checkout" rule). Every call this module makes is *against the canonical
  checkout*, because rescuing WIP a human or process left there is exactly
  repository-manager's job — the same role au's own docs describe as "a
  global actor" (background sync, fleet cleanup) rather than a lane.
* The AU primitive intentionally refuses canonical checkouts and only restores
  the same tree, while ``add(adopt=True)`` captures canonical WIP and restores
  it into a **different** newly-created worktree. This module therefore keeps a
  local implementation of the same private-ref contract, with untracked files
  included and no shared-stack fallback.
"""

from __future__ import annotations

import contextlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import Any, Protocol

from repository_manager.canonical_guard import BlockedByLease, hold_canonical_lease

logger = logging.getLogger(__name__)
_TRUSTED_GIT = shutil.which("git") or "git"

__all__ = [
    "BlockedByLease",
    "LANE_STASH_NAMESPACE",
    "PRIVATE_STASH_NAMESPACE",
    "apply_and_clear",
    "capture_wip",
    "hold_tree_mutation_lease",
    "park",
    "private_stash_ref",
    "unpark",
]

#: Nested under the same ``refs/lane/`` prefix ``agent_utilities.governance.lanes``
#: uses for its own per-lane stash ref (``refs/lane/<lane>/stash``), so private,
#: off-the-shared-stack refs are legible under one convention across the
#: workspace. A distinct sub-path (not ``refs/lane/<lane>/stash`` itself) keeps
#: this from ever colliding with that ref: this module captures from the
#: *canonical* checkout (which is never a "lane" in that module's model — see
#: the module docstring) and keys its refs by the requested branch label, not
#: by lane identity.
PRIVATE_STASH_NAMESPACE = "refs/lane/rm-adopt-stash"
LANE_STASH_NAMESPACE = "refs/lane"


def _git_executable() -> str:
    """Resolve git without handing ``Popen`` a partial executable path."""
    return _TRUSTED_GIT


class _GitLike(Protocol):
    def git_action(
        self,
        command: str,
        path: str | None = ...,
        quiet: bool = ...,
        env: dict[str, str] | None = ...,
        timeout: int = ...,
        raw_output: bool = ...,
    ) -> Any: ...


def _ok(res: Any) -> bool:
    return getattr(res, "status", "") == "success"


def _err(res: Any) -> str:
    error = getattr(res, "error", None)
    if error is not None and getattr(error, "message", None):
        return str(error.message)
    return str(getattr(res, "data", "") or "")


def private_stash_ref(label: str) -> str:
    """A collision-proof private ref name for capturing one lane's WIP."""
    slug = "".join(c if c.isalnum() or c in "-_." else "-" for c in label) or "wip"
    return f"{PRIVATE_STASH_NAMESPACE}/{slug}-{uuid.uuid4().hex[:12]}"


def capture_wip(
    git: _GitLike,
    canonical: str,
    label: str,
    message: str = "repository-manager WIP capture",
) -> dict[str, Any]:
    """Move ``canonical``'s uncommitted WIP onto a private ref, off ``refs/stash``.

    Returns ``{"ok": True, "ref": None}`` when the tree was already clean
    (nothing to capture — not an error). Returns ``{"ok": True, "ref": <name>}``
    once the WIP is safely parked on the private ref and ``canonical``'s
    working tree is clean. Returns ``{"ok": False, "ref": None, "error": ...}``
    if capture could not be completed; a partial failure never leaves WIP
    stranded only on the shared stack without being reported (see inline
    comments below for exactly what each failure mode preserves).
    """
    try:
        with hold_canonical_lease(canonical, note=f"capture WIP: {label}"):
            with hold_tree_mutation_lease(canonical, note=f"capture WIP: {label}"):
                result = park(
                    git,
                    canonical,
                    lane=label,
                    message=message,
                    _ref=private_stash_ref(label),
                    _lease=False,
                )
            if result.get("ok") and not result.get("parked"):
                return {"ok": True, "ref": None, "error": None}
            if not result.get("ok"):
                return {"ok": False, "ref": None, "error": result.get("error")}
            return {"ok": True, "ref": result.get("ref"), "error": None}
    except (BlockedByLease, OSError, RuntimeError) as exc:
        return {"ok": False, "ref": None, "error": str(exc)}


def apply_and_clear(git: _GitLike, target_path: str, ref: str) -> dict[str, Any]:
    """Restore a private WIP commit into a clean target and then clear its ref."""
    restored = unpark(git, target_path, ref=ref)
    return {
        "ok": bool(restored.get("ok")),
        "ref": ref,
        "error": restored.get("error"),
    }


def _lane_ref(tree: str | Path, lane: str | None = None) -> str:
    """Return the PARTITION ref for ``tree`` without making it a requirement.

    Agent Utilities owns the canonical lane naming function.  This package is
    also used in small, dependency-light repair processes, though, so a
    basename fallback keeps the safety primitive usable while that optional
    dependency is unavailable.  The fallback is deliberately deterministic;
    callers get a refusal when a previous park already occupies the ref rather
    than silently overwriting another operation's recovery point.
    """
    if lane is None:
        try:
            from agent_utilities.governance.lanes import lane_name

            lane = lane_name(tree)
        except Exception:  # pragma: no cover - optional dependency/fake trees
            lane = Path(tree).resolve().name
    slug = "".join(c if c.isalnum() or c in "-_." else "-" for c in lane)
    return f"{LANE_STASH_NAMESPACE}/{slug or 'local'}/stash"


def _worktree_git_dir(tree: str | Path) -> Path | None:
    """Resolve the administrative directory belonging to this worktree."""
    try:
        result = subprocess.run(
            [_git_executable(), "-C", str(tree), "rev-parse", "--git-dir"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):  # pragma: no cover
        return None
    if result.returncode:
        return None
    git_dir = Path(result.stdout.strip())
    if not git_dir.is_absolute():
        git_dir = (Path(tree).resolve() / git_dir).resolve()
    return git_dir


@contextlib.contextmanager
def hold_tree_mutation_lease(tree: str | Path, note: str = "") -> Iterator[int]:
    """Hold a non-blocking lease scoped to one worktree's git directory.

    The lock is deliberately per-worktree rather than in the common git dir:
    two lanes may safely stage/commit in parallel, while repository-manager
    operations targeting the same worktree refuse a check/snapshot/execute
    interleave.  Ungoverned processes that ignore this cooperative lease remain
    an explicitly documented residual race.
    """
    git_dir = _worktree_git_dir(tree)
    if git_dir is None:
        raise RuntimeError(f"cannot resolve a worktree git directory: {tree}")
    lease_path = git_dir / "repository-manager-mutation.lease"
    try:
        fd = os.open(lease_path, os.O_CREAT | os.O_RDWR, 0o644)
    except OSError as exc:
        raise RuntimeError(f"cannot create worktree mutation lease: {exc}") from exc
    from agent_utilities.knowledge_graph.core.file_lock import lock_exclusive_nb, unlock

    try:
        try:
            lock_exclusive_nb(fd)
        except OSError as exc:
            raise BlockedByLease(
                f"worktree {str(tree)!r} mutation lease is already held"
            ) from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {note}".encode())
        except OSError:  # nosec B110 - the lease body is diagnostic only
            pass
        yield fd
    finally:
        try:
            unlock(fd)
        finally:
            os.close(fd)


@contextlib.contextmanager
def _hold_stash_lease(tree: str | Path) -> Iterator[int]:
    """Serialize private-ref capture against same-worktree RM actors."""
    with hold_tree_mutation_lease(tree, note="private WIP park") as lease:
        yield lease


def park(
    git: _GitLike,
    tree: str,
    *,
    lane: str | None = None,
    message: str = "repository-manager lane park",
    _ref: str | None = None,
    _lease: bool = True,
) -> dict[str, Any]:
    """Park tracked and untracked WIP in this lane's private ref.

    Unlike the historical canonical-adoption flow, this general lane primitive
    never reads or writes ``refs/stash``.  A temporary index stages the complete
    working tree, ``git write-tree``/``git commit-tree`` creates a private
    recovery commit, and only after the ref is durable are tracked changes and
    captured untracked files removed from the working tree.  A pre-existing
    lane ref is a hard refusal because replacing it would erase a prior recovery
    point.
    """
    ref = _ref or _lane_ref(tree, lane)
    lease = _hold_stash_lease(tree) if _lease else contextlib.nullcontext()
    with lease:
        status = git.git_action(command="git status --porcelain", path=tree, quiet=True)
        if not _ok(status):
            return {"ok": False, "parked": False, "ref": ref, "error": _err(status)}

        existing = git.git_action(
            command=f"git rev-parse --verify --quiet {shlex.quote(ref)}",
            path=tree,
            quiet=True,
        )
        if _ok(existing) and str(existing.data or "").strip():
            return {
                "ok": False,
                "parked": False,
                "ref": ref,
                "error": f"private lane stash ref already exists: {ref}",
            }

        untracked = git.git_action(
            command="git ls-files --others --exclude-standard -z",
            path=tree,
            quiet=True,
        )
        if not _ok(untracked):
            return {
                "ok": False,
                "parked": False,
                "ref": ref,
                "error": _err(untracked),
            }
        untracked_paths = [
            item for item in str(untracked.data or "").split("\0") if item
        ]
        temp_fd, temp_name = tempfile.mkstemp(prefix="rmdd26-index-")
        os.close(temp_fd)
        os.unlink(temp_name)
        temp_env = os.environ.copy()
        temp_env["GIT_INDEX_FILE"] = temp_name
        try:
            for command in ("git read-tree HEAD",):
                staged = git.git_action(
                    command=command, path=tree, env=temp_env, quiet=True
                )
                if not _ok(staged):
                    return {
                        "ok": False,
                        "parked": False,
                        "ref": ref,
                        "error": f"{command} failed: {_err(staged)}",
                    }
            indexed = git.git_action(
                command="git ls-files -z", path=tree, env=temp_env, quiet=True
            )
            if not _ok(indexed):
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": f"git ls-files failed: {_err(indexed)}",
                }
            tracked_paths = [
                item for item in str(indexed.data or "").split("\0") if item
            ]
            if tracked_paths:
                quoted_paths = " ".join(shlex.quote(item) for item in tracked_paths)
                for clear_flags in (
                    f"git update-index --no-assume-unchanged {quoted_paths}",
                    f"git update-index --no-skip-worktree {quoted_paths}",
                ):
                    cleared = git.git_action(
                        command=clear_flags, path=tree, env=temp_env, quiet=True
                    )
                    if not _ok(cleared):
                        return {
                            "ok": False,
                            "parked": False,
                            "ref": ref,
                            "error": f"{clear_flags} failed: {_err(cleared)}",
                        }
            staged = git.git_action(
                command="git add -A", path=tree, env=temp_env, quiet=True
            )
            if not _ok(staged):
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": f"git add -A failed: {_err(staged)}",
                }
            tree_obj = git.git_action(
                command="git write-tree", path=tree, env=temp_env, quiet=True
            )
            if not _ok(tree_obj) or not str(tree_obj.data or "").strip():
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": f"git write-tree failed: {_err(tree_obj)}",
                }
            head = git.git_action(command="git rev-parse HEAD", path=tree, quiet=True)
            if not _ok(head) or not str(head.data or "").strip():
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": "cannot create a private WIP commit without HEAD",
                }
            head_tree = git.git_action(
                command="git rev-parse HEAD^{tree}", path=tree, quiet=True
            )
            if not _ok(head_tree) or not str(head_tree.data or "").strip():
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": "cannot compare the working tree with HEAD",
                }
            if (
                str(tree_obj.data).strip() == str(head_tree.data).strip()
                and not untracked_paths
            ):
                return {
                    "ok": True,
                    "parked": False,
                    "ref": ref,
                    "commit": None,
                    "error": None,
                }
            commit = git.git_action(
                command=(
                    f"git commit-tree {shlex.quote(str(tree_obj.data).strip())} "
                    f"-p {shlex.quote(str(head.data).strip())} -m {shlex.quote(message)}"
                ),
                path=tree,
                quiet=True,
            )
            sha = str(commit.data or "").strip() if _ok(commit) else ""
            if not sha:
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": f"git commit-tree failed: {_err(commit)}",
                }
            stored = git.git_action(
                command=f"git update-ref {shlex.quote(ref)} {shlex.quote(sha)}",
                path=tree,
                quiet=True,
            )
            if not _ok(stored):
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "error": f"could not create private lane stash ref {ref}: {_err(stored)}",
                }
        finally:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass

        # Clear hidden-index bits in the real index before resetting the
        # working tree; read-tree otherwise honors skip-worktree and leaves
        # the captured bytes in place.
        if tracked_paths:
            quoted_paths = " ".join(shlex.quote(item) for item in tracked_paths)
            for clear_real in (
                f"git update-index --no-assume-unchanged {quoted_paths}",
                f"git update-index --no-skip-worktree {quoted_paths}",
            ):
                cleared_real = git.git_action(command=clear_real, path=tree, quiet=True)
                if not _ok(cleared_real):
                    return {
                        "ok": False,
                        "parked": False,
                        "ref": ref,
                        "commit": sha,
                        "error": f"{clear_real} failed: {_err(cleared_real)}",
                    }
        reset = git.git_action(
            command="git read-tree -u --reset HEAD", path=tree, quiet=True
        )
        if not _ok(reset):
            return {
                "ok": False,
                "parked": False,
                "ref": ref,
                "commit": sha,
                "error": f"private ref is safe but tree cleanup failed: {_err(reset)}",
            }
        for relative in untracked_paths:
            root = Path(tree).resolve()
            candidate = root / relative
            lexical = candidate.absolute()
            if lexical != root and root not in lexical.parents:
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "commit": sha,
                    "error": f"refused to remove untracked path outside tree: {relative}",
                }
            try:
                if candidate.is_symlink() or candidate.is_file():
                    candidate.unlink()
                elif candidate.is_dir():
                    candidate.rmdir()
            except OSError as exc:
                return {
                    "ok": False,
                    "parked": False,
                    "ref": ref,
                    "commit": sha,
                    "error": f"private ref is safe but untracked cleanup failed: {exc}",
                }
        return {"ok": True, "parked": True, "ref": ref, "commit": sha, "error": None}


def unpark(
    git: _GitLike,
    tree: str,
    *,
    lane: str | None = None,
    ref: str | None = None,
    _lease: bool = True,
) -> dict[str, Any]:
    """Restore a temporary-index WIP ref and clear it only after success."""
    target_ref = ref or _lane_ref(tree, lane)
    if _lease:
        try:
            with hold_tree_mutation_lease(tree, note="private WIP unpark"):
                return unpark(
                    git,
                    tree,
                    lane=lane,
                    ref=target_ref,
                    _lease=False,
                )
        except BlockedByLease as exc:
            return {"ok": False, "ref": target_ref, "error": str(exc)}
        except (OSError, RuntimeError) as exc:
            return {"ok": False, "ref": target_ref, "error": str(exc)}
    status = git.git_action(command="git status --porcelain", path=tree, quiet=True)
    if not _ok(status):
        return {"ok": False, "ref": target_ref, "error": _err(status)}
    if str(status.data or "").strip():
        return {
            "ok": False,
            "ref": target_ref,
            "error": "refusing to unpark into a dirty working tree",
        }
    commit = git.git_action(
        command=f"git rev-parse --verify --quiet {shlex.quote(target_ref)}",
        path=tree,
        quiet=True,
    )
    if not _ok(commit) or not str(commit.data or "").strip():
        return {
            "ok": False,
            "ref": target_ref,
            "error": "nothing parked at private ref",
        }
    restored = git.git_action(
        command=f"git read-tree -u --reset {shlex.quote(target_ref)}",
        path=tree,
        quiet=True,
    )
    if not _ok(restored):
        return {"ok": False, "ref": target_ref, "error": _err(restored)}
    unstage = git.git_action(command="git reset", path=tree, quiet=True)
    if not _ok(unstage):
        return {"ok": False, "ref": target_ref, "error": _err(unstage)}
    deleted = git.git_action(
        command=f"git update-ref -d {shlex.quote(target_ref)}",
        path=tree,
        quiet=True,
    )
    if not _ok(deleted):
        logger.warning("restored private WIP but could not delete ref %s", target_ref)
    return {
        "ok": True,
        "ref": target_ref,
        "commit": str(commit.data).strip(),
        "error": None,
    }
