"""Refuse to prune a worktree, or delete its branch, that is not provably spent.

CONCEPT:RM-PRUNE-GUARD

Background — the hazard this closes. ``WorktreeManager.audit(prune_merged=True)``
classifies a worktree ``merged`` from ``ahead == 0`` against a base branch and
then removes the directory *and* runs ``git branch -D``. Three things are wrong
with that, and on 2026-07-31 they combined to take a live lane's worktree and
branch ref out from under it (registry ``D-FE-9``):

1. **``merged`` is not ``finished``.** A lane that merges an intermediate chunk
   back to ``main`` and keeps working is, from the outside, indistinguishable
   from a lane that is done — its tree is clean and its branch is an ancestor of
   ``main``. Occupancy is a separate fact from mergedness and has to be asked
   separately.
2. **A brand-new worktree is also at ``ahead == 0``.** Sitting exactly at base is
   the *start* of a lane, not the end of one.
3. **The classification is stale by the time it is used.** ``audit()`` scans
   every worktree, then prunes; anything a lane commits inside that window is
   orphaned by ``git branch -D``, which never re-checks reachability.

The two rules this module enforces, and why they are structural rather than
flag-shaped:

**A ref is gated harder than a directory.** Removing a clean worktree directory
is recoverable — ``git worktree add`` puts it back from the branch. Deleting the
branch ref is what turns a lane's commits into garbage. So ref deletion never
uses ``git branch -D``. It re-asks ``git merge-base --is-ancestor`` at the moment
of deletion and then defers to ``git branch -d``, which re-evaluates reachability
itself, under git's own ref lock, atomically with the delete. Correctness comes
from git's refusal, not from our earlier scan agreeing with itself.

**Occupancy comes from the lane protocol, not a new mechanism.**
:func:`guarded_worktree_prune` routes through
``agent_utilities.governance.lanes.guarded_tree_mutation`` — the same repo-scoped
lease in the shared ``--git-common-dir`` and the same ``require_resettable_tree``
refusal every other global actor in this workspace must consult — and reads
occupancy from facts git and that protocol already record: a merge/rebase in
progress, uncommitted work, a live lease held by that lane. Nothing here asks a
lane to set a flag it can forget to set.

**Residual gap, stated plainly.** A lease only binds actors that take it, and a
lane that is merely *thinking* holds nothing and looks identical to an abandoned
one. This module does not close that; what it closes is the consequence. Even
when occupancy detection is wrong, the branch ref survives (rule A), so the
worst case degrades from "commits become garbage" to "a directory has to be
re-added". Do not record the occupancy question as solved.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
from collections.abc import Iterator

logger = logging.getLogger(__name__)

try:
    from agent_utilities.governance import lanes
except ImportError as exc:  # pragma: no cover - exercised only on an older au
    lanes = None  # type: ignore[assignment]
    _LANES_UNAVAILABLE = (
        "the lane arbitration protocol "
        "(agent_utilities.governance.lanes) is unavailable, so an active lane "
        f"cannot be distinguished from an abandoned one: {exc}"
    )
else:
    _LANES_UNAVAILABLE = ""


def _live_lease_hold(scope) -> str | None:
    """Names of live leases this worktree's own lane holds, as a refusal reason.

    Reads the lease files the LEASE protocol writes into the shared arbitration
    directory and confirms each candidate through the protocol's own
    :func:`lanes.lease_status`, so liveness (expiry, dead holder reclamation) is
    decided by the protocol rather than re-implemented here.
    """
    lease_dir = scope.arbitration_dir / "leases"
    if not lease_dir.is_dir():
        return None
    held: list[str] = []
    for lease_file in sorted(lease_dir.glob("*.lease")):
        try:
            holder = json.loads(lease_file.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            # An unreadable lease is not evidence of absence. Err toward keeping.
            return f"lease file {lease_file.name} could not be read ({exc})"
        if not isinstance(holder, dict) or holder.get("lane") != scope.lane:
            continue
        name = str(holder.get("name", ""))
        if name and lanes.lease_status(name, scope.tree) is not None:
            held.append(name)
    if held:
        return f"lane {scope.lane!r} holds the live lease(s) {', '.join(held)}"
    return None


def lane_hold(worktree_path: str) -> str | None:
    """Why a live lane still holds ``worktree_path``, or ``None`` if none does.

    Every signal is one git or lane-protocol already records; none of them is a
    marker a lane has to remember to place.
    """
    if lanes is None:  # pragma: no cover - exercised only on an older au
        return _LANES_UNAVAILABLE
    if lanes.current_tree(worktree_path) is None:
        # Not a working tree at all (a stale admin pointer): no lane can be in it.
        return None
    scope = lanes.lane_scope(worktree_path)
    if scope.merge_in_progress:
        return "a merge/rebase/cherry-pick is in progress in this worktree"
    if lanes.tree_has_uncommitted_work(scope.tree):
        return "the worktree holds uncommitted work"
    return _live_lease_hold(scope)


@contextlib.contextmanager
def guarded_worktree_prune(
    worktree_path: str, *, operation: str
) -> Iterator[str | None]:
    """Yield ``None`` while it is safe to destroy ``worktree_path``, else a reason.

    Usage::

        with guarded_worktree_prune(path, operation="prune merged worktree") as held:
            if held is not None:
                return keep(held)
            ...  # remove the worktree while the lease is still held

    The lane lease is held for the whole ``with`` body, so the caller's re-check
    and its removal happen inside the same arbitration window rather than either
    side of it.
    """
    if lanes is None:  # pragma: no cover - exercised only on an older au
        yield _LANES_UNAVAILABLE
        return
    held = lane_hold(worktree_path)
    if held is not None:
        logger.warning(
            "repository-manager: keeping worktree %s - %s", worktree_path, held
        )
        yield held
        return
    # ``owner`` is the identity of the actor doing the pruning (this process's
    # own lane), never the tree being pruned - passing the target's own lane
    # would make the protocol's ownership refusal a no-op.
    try:
        with lanes.guarded_tree_mutation(
            worktree_path, operation=operation, owner=lanes.lane_name()
        ):
            yield None
    except lanes.LaneArbitrationError as exc:
        logger.warning(
            "repository-manager: keeping worktree %s - %s", worktree_path, exc
        )
        yield str(exc)


def worktree_is_locked(worktree_path: str) -> bool:
    """Whether git's own ``worktree lock`` marker is set for this worktree.

    ``git worktree remove`` refuses a locked worktree, so honouring the lock up
    front turns a failed removal into an explained skip.
    """
    admin = os.path.join(worktree_path, ".git")
    if not os.path.isfile(admin):
        return False
    try:
        with open(admin, encoding="utf-8") as handle:
            gitdir = handle.read().strip()
    except OSError:
        return False
    prefix = "gitdir:"
    if not gitdir.startswith(prefix):
        return False
    return os.path.exists(os.path.join(gitdir[len(prefix) :].strip(), "locked"))
