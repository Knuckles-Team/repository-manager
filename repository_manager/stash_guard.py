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

The fix: never leave our capture sitting on the shared stack unattended.
:func:`capture_wip` does the capture and the same instant it is safely off the
stack, under the canonical checkout's cross-process lease
(:func:`repository_manager.canonical_guard.hold_canonical_lease`, so no other
repository-manager-initiated stash operation can interleave with our own):

1. ``git status --porcelain`` under the lease — decide there is even anything
   to capture, atomically with the steps that follow.
2. ``git stash push -u -m <message>`` — the only git plumbing that atomically
   captures tracked *and* untracked changes as one commit and cleans the
   working tree in a single step (``git stash create`` does the commit half
   but never resets the tree or includes untracked content; there is no
   ``git stash push --ref <ref>`` to target a private ref directly instead of
   ``refs/stash``).
3. Immediately capture the resulting commit (``git rev-parse refs/stash``) and
   point a **private ref** at it: ``refs/rm-stash/<label>-<uuid>`` — a
   namespace nothing else in this workspace writes to, so it can never
   collide with another lane's own stash entries.
4. ``git stash drop`` — vacate ``stash@{0}`` immediately, so the shared stack
   carries our WIP for as short a window as git's own plumbing allows (the
   push itself), not for the whole worktree-creation sequence that follows.

From that point on our WIP lives only on the private ref, which behaves like
any other git ref — nothing but code that knows its name can touch it.
:func:`apply_and_clear` applies it into the target tree
(``git stash apply <ref>`` — this accepts any commit shaped like a stash, not
just a ``stash@{n}`` reflog entry) and deletes the private ref only once the
apply reports success; a conflicted apply leaves the ref in place and reports
its name, so the caller can surface it for manual recovery instead of losing
the WIP.

**Residual gap, stated plainly** — matching the same honesty note in
:mod:`repository_manager.canonical_guard`: the lease closes the race between
repository-manager-initiated stash operations. It cannot make step 2-4 atomic
against a raw ``git stash`` a human or an unrelated tool runs directly in the
canonical checkout (or a linked worktree) at that exact instant, since that
actor never takes this lease. That window is now the width of one
``push``+``rev-parse``+``update-ref``+``drop`` sequence rather than an entire
worktree-creation call, but it is not zero. The workspace-wide "never
``git stash`` directly" rule is the mitigation for the part this cannot close.
"""

from __future__ import annotations

import logging
import shlex
import uuid
from typing import Any, Protocol

from repository_manager.canonical_guard import BlockedByLease, hold_canonical_lease

logger = logging.getLogger(__name__)

#: Nothing else in this workspace writes into this namespace, so a private ref
#: minted here can never collide with another lane's own stash entries or with
#: the shared ``refs/stash`` stack itself.
PRIVATE_STASH_NAMESPACE = "refs/rm-stash"


class _GitLike(Protocol):
    def git_action(
        self,
        command: str,
        path: str | None = ...,
        quiet: bool = ...,
        env: dict | None = ...,
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
            status = git.git_action(
                command="git status --porcelain", path=canonical, quiet=True
            )
            if not _ok(status):
                return {
                    "ok": False,
                    "ref": None,
                    "error": f"git status failed: {_err(status)}",
                }
            if not status.data.strip():
                return {"ok": True, "ref": None, "error": None}

            push = git.git_action(
                command=f"git stash push -u -m {shlex.quote(message)}",
                path=canonical,
            )
            if not _ok(push):
                # Nothing was moved: the working tree is exactly as it was.
                return {"ok": False, "ref": None, "error": _err(push)}

            sha_res = git.git_action(
                command="git rev-parse refs/stash", path=canonical, quiet=True
            )
            sha = sha_res.data.strip() if _ok(sha_res) else ""
            if not sha:
                # The push reported success but we cannot name what it made.
                # Leave it on the shared stack (do NOT drop) rather than lose
                # track of it, and say so loudly.
                return {
                    "ok": False,
                    "ref": None,
                    "error": (
                        "git stash push succeeded but refs/stash could not be "
                        "resolved; WIP remains on the shared stash stack "
                        "(stash@{0}) - restore it with 'git stash pop' by hand"
                    ),
                }

            ref = private_stash_ref(label)
            store = git.git_action(
                command=f"git update-ref {shlex.quote(ref)} {sha}",
                path=canonical,
                quiet=True,
            )
            if not _ok(store):
                # Could not mint the private ref: leave the WIP on the shared
                # stack (still recoverable) instead of dropping it into a ref
                # that does not exist.
                return {
                    "ok": False,
                    "ref": None,
                    "error": (
                        "could not create private stash ref "
                        f"{ref!r} ({_err(store)}); WIP remains on the shared "
                        "stash stack (stash@{0})"
                    ),
                }

            drop = git.git_action(command="git stash drop", path=canonical, quiet=True)
            if not _ok(drop):
                # Non-fatal: our copy is already safe on the private ref. A
                # redundant copy lingers on the shared stack until someone
                # drops it by hand; that is a nuisance, not a data-loss risk,
                # so we still report success.
                logger.warning(
                    "repository-manager: WIP for %s captured to %s but could "
                    "not drop it from the shared stash stack (%s) - a "
                    "redundant copy remains at stash@{0}",
                    label,
                    ref,
                    _err(drop),
                )
            return {"ok": True, "ref": ref, "error": None}
    except BlockedByLease as exc:
        return {"ok": False, "ref": None, "error": str(exc)}


def apply_and_clear(git: _GitLike, target_path: str, ref: str) -> dict[str, Any]:
    """Apply the WIP on private ``ref`` into ``target_path`` and delete ``ref``.

    On a clean apply the private ref is deleted (its job is done). On a
    conflicted or failed apply the ref is left in place and named in the
    result, so the WIP is never silently lost - the caller surfaces
    ``ref``/``error`` for manual recovery (``git stash apply <ref>`` from
    whichever tree needs it).
    """
    apply_res = git.git_action(
        command=f"git stash apply {shlex.quote(ref)}", path=target_path
    )
    if not _ok(apply_res):
        return {"ok": False, "ref": ref, "error": _err(apply_res)}
    delete_res = git.git_action(
        command=f"git update-ref -d {shlex.quote(ref)}", path=target_path, quiet=True
    )
    if not _ok(delete_res):
        # Applied fine; the leftover ref is inert clutter, not a hazard, but
        # say so rather than silently swallowing the failure.
        logger.warning(
            "repository-manager: applied WIP from %s to %s but could not "
            "delete the private ref afterward (%s)",
            ref,
            target_path,
            _err(delete_res),
        )
    return {"ok": True, "ref": ref, "error": None}
