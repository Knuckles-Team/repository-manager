"""Merge is not deploy — CONCEPT:AU-OS.governance.merge-deploy-decoupling.

Moved from agent-utilities' retired merge queue, where it was the
``merge-queue promotion`` verb; it is ``repository-manager-governance
promotion`` now. It needs nothing from a queue: it compares two refs of the
repository the caller stands in.

**The hazard this measures.** graph-os runs source-over-site-packages: the pod
NFS-mounts the canonical checkout read-only with the checkout on
``PYTHONPATH``, so the bytes a pod imports on its next start are whatever is in
the canonical working tree at that instant. Nothing hot-reloads — but nothing
pins, either, so *any* restart the operator did not choose (a node drain, an
eviction, an OOM kill, a reschedule) deploys whatever happens to be on
``main``. Merge is not deploy; merge is an **armed** deploy that fires at a
time nobody picked.

**The decoupling.** Point the mount at a checkout of :data:`PROMOTION_REF`
instead of the canonical ``main`` tree, and the two facts separate cleanly:
**merge** fast-forwards ``main`` and the fleet does not see it; **promote** is
an explicit, gated fast-forward of ``deployed`` to a ``main`` SHA the slow tier
has since gone green on, followed by a rollout restart. ``deployed`` can also
be moved back to a known-good SHA without touching ``main``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

from repository_manager.governance.lanes import lane_scope

#: The ref the fleet's source mount is expected to follow. Merging moves
#: ``main``; only an explicit promotion moves this.
PROMOTION_REF = "refs/heads/deployed"


def _git(args: list[str], repo: Path) -> subprocess.CompletedProcess[str]:
    argv = ["git", *args]
    return subprocess.run(  # fixed argv, no shell
        argv, cwd=str(repo), capture_output=True, text=True, check=False
    )


def _rev(ref: str, repo: Path) -> str | None:
    proc = _git(["rev-parse", "--verify", "--quiet", ref], repo)
    if proc.returncode != 0:
        return None
    return proc.stdout.strip() or None


def promotion_state(path: Path | str | None = None) -> dict[str, Any]:
    """How far the deployed ref lags ``main``, and whether merge is decoupled."""
    repo = Path(lane_scope(path).main_tree)
    main_tip = _rev("main", repo)
    if main_tip is None:
        raise RuntimeError(f"{repo} has no `main` branch to compare against")
    deployed = _rev(PROMOTION_REF, repo)
    if deployed is None:
        return {
            "decoupled": False,
            "promotion_ref": PROMOTION_REF,
            "main": main_tip,
            "deployed": None,
            "reason": (
                f"{PROMOTION_REF} does not exist, so the fleet's source mount can "
                "only be following the canonical `main` tree: every merge is an "
                "armed deploy that fires on the next unplanned pod restart. Create "
                "it and repoint the mount."
            ),
        }
    behind = _git(["rev-list", "--count", f"{deployed}..{main_tip}"], repo)
    return {
        "decoupled": True,
        "promotion_ref": PROMOTION_REF,
        "main": main_tip,
        "deployed": deployed,
        "unpromoted_commits": int(behind.stdout.strip() or 0),
    }
