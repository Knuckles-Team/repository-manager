"""Guard against mutating a *canonical* checkout's working tree while it holds
uncommitted work (CONCEPT:RM-CANON-GUARD).

Background — the hazard this closes: every place repository-manager runs a
tree-mutating git command directly against a **canonical** checkout (as
opposed to a session's own worktree under ``WORKTREE_ROOT``) —
``Git.pull_project``'s default-branch checkout and
``WorktreeManager.add``/``merge``'s park-on-base / switch-to-target
checkouts — can silently discard a concurrent session's uncommitted edits if
it runs while the tree is dirty. :func:`guarded_canonical_mutation` is the one
choke point all three go through.

What it does:

1. Takes a short-lived, cross-process advisory lease on the canonical
   checkout (an ``flock`` on ``<canonical>/.git/repository-manager.lease`` —
   inside ``.git/`` so it is never reported by ``git status`` and never
   committed) so two repository-manager-initiated mutations against the
   *same* canonical never interleave: the second one skips instead of racing
   the first.
2. With the lease held, checks ``git status --porcelain`` — which reports
   BOTH tracked modifications and untracked files in one pass — immediately
   before the caller's mutating command. A dirty tree means "skip, don't
   mutate": the caller gets back a ready-to-return, loud, actionable result
   instead of proceeding.

Honesty note — the in-flight race (a tree clean at check time, dirtied a
moment later by an unrelated process, e.g. a human running ``pre-commit``
directly in the canonical checkout): this closes the race *between
repository-manager operations* on one canonical checkout (they now
serialize through the same lease). It does **not** make the check-then-mutate
sequence atomic against an external process that never takes this lease —
that would require the external process to opt in. :func:`hold_canonical_lease`
is exposed for exactly that: an external long-running operation can wrap
itself with it (see the ``python -m repository_manager.canonical_guard``
CLI below) so repository-manager's guard sees it held and skips rather than
collides. Nothing currently makes an unwrapped external ``pre-commit`` run
take this lease automatically — that gap is real and stays open here; the
lease file/convention is offered as the concrete primitive for the workspace's
general LEASE arbitration protocol to adopt or generalize.
"""

from __future__ import annotations

import contextlib
import fcntl
import logging
import os
from typing import Any, Protocol

logger = logging.getLogger(__name__)

#: Lives inside ``.git/`` on purpose: never surfaced by ``git status``, never
#: committed, and always co-located with the checkout it guards.
LEASE_FILENAME = "repository-manager.lease"


class BlockedByLease(RuntimeError):
    """Raised when another holder already has the canonical checkout's lease."""


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


def _lease_path(canonical: str) -> str:
    return os.path.join(canonical, ".git", LEASE_FILENAME)


@contextlib.contextmanager
def hold_canonical_lease(canonical: str, note: str = ""):
    """Hold the cross-process advisory lease on ``canonical`` for the ``with`` body.

    Exposed for external long-running operations (e.g. a wrapped manual
    ``pre-commit run --all-files`` in a canonical checkout) that want
    repository-manager's guard to see them and skip rather than collide.
    Raises :class:`BlockedByLease` if another holder already has it.
    """
    path = _lease_path(canonical)
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o644)
    try:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise BlockedByLease(
                f"canonical checkout {canonical!r} lease is already held"
            ) from exc
        try:
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()} {note}".encode())
        except OSError:  # nosec B110 - the lease body is diagnostic only
            pass
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        except OSError:  # nosec B110 - best-effort unlock during teardown
            pass
        os.close(fd)


@contextlib.contextmanager
def guarded_canonical_mutation(
    git: _GitLike, canonical: str, repo_label: str, action: str
):
    """Yield ``None`` if it is safe to mutate ``canonical`` right now, or a
    ready-to-return skip dict if it is not.

    Usage::

        with guarded_canonical_mutation(git, canonical, repo, "checkout base") as blocked:
            if blocked is not None:
                return blocked
            git.git_action(f"git checkout {shlex.quote(base)}", path=canonical)

    The lease is held for the entire ``with`` body, so the caller's mutating
    command runs while still holding it.
    """
    try:
        with hold_canonical_lease(canonical, note=action):
            status = git.git_action(
                command="git status --porcelain", path=canonical, quiet=True
            )
            dirty_detail = (
                status.data.strip()
                if getattr(status, "status", "") == "success"
                else ""
            )
            if dirty_detail:
                logger.warning(
                    "repository-manager: REFUSING to %s in canonical checkout "
                    "of %s - uncommitted changes present (tracked "
                    "modifications and/or untracked files would be "
                    "clobbered): %s",
                    action,
                    repo_label,
                    dirty_detail.replace("\n", " | "),
                )
                yield {
                    "ok": False,
                    "repo": repo_label,
                    "skipped": True,
                    "reason": "dirty-canonical-checkout",
                    "error": (
                        f"refused to {action}: canonical checkout of "
                        f"{repo_label!r} has uncommitted changes that would "
                        "be clobbered; commit or move the work to a "
                        "worktree (rm_worktree add) and retry"
                    ),
                    "detail": dirty_detail,
                }
                return
            yield None
    except BlockedByLease:
        logger.warning(
            "repository-manager: REFUSING to %s in canonical checkout of "
            "%s - another repository-manager operation already holds this "
            "canonical checkout's lease",
            action,
            repo_label,
        )
        yield {
            "ok": False,
            "repo": repo_label,
            "skipped": True,
            "reason": "canonical-checkout-busy",
            "error": (
                f"refused to {action}: canonical checkout of {repo_label!r} "
                "is busy (another repository-manager operation holds its "
                "lease); retry shortly"
            ),
        }


def _main(argv: list[str] | None = None) -> int:
    """``python -m repository_manager.canonical_guard <repo> -- <command...>``

    Runs ``command`` inside ``repo`` while holding its canonical lease, so a
    long-running foreground operation (a manual ``pre-commit`` run, most
    notably) becomes visible to repository-manager's background sync, which
    will skip that checkout instead of racing it. Exits 2 without running
    ``command`` if the lease is already held by another operation.
    """
    import argparse
    import subprocess
    import sys

    parser = argparse.ArgumentParser(
        prog="python -m repository_manager.canonical_guard",
        description=(
            "Hold the repository-manager canonical-checkout lease for the "
            "duration of COMMAND, so background sync skips this checkout "
            "instead of racing it."
        ),
    )
    parser.add_argument("canonical", help="path to the canonical checkout")
    parser.add_argument(
        "command", nargs=argparse.REMAINDER, help="command to run under the lease"
    )
    args = parser.parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        parser.error(
            "a command to run is required, e.g.: <repo> -- pre-commit run --all-files"
        )

    canonical = os.path.abspath(args.canonical)
    try:
        with hold_canonical_lease(canonical, note=" ".join(command)):
            return subprocess.call(command, cwd=canonical)  # nosec B603
    except BlockedByLease as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(_main())
