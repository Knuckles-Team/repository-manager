"""Git worktree management for concurrent multi-session development.

CONCEPT:RM-WORKTREE — lets N agent sessions work the same repo at once. Each
session takes its **own branch** in its **own worktree** under ``WORKTREE_ROOT``,
all sharing a single ``.git`` object store (no re-clone). The canonical checkout
stays on its default branch — what the validate/sync cascade expects — so a
working-tree reset on the canonical path never touches a session's worktree
files. Git's invariant (a branch lives in at most one worktree) is the hard lock
that keeps concurrent sessions from colliding.

Worktrees live at ``<WORKTREE_ROOT>/<repo>/<branch-slug>``. ``WORKTREE_ROOT``
defaults to ``/home/apps/worktrees`` (outside the workspace scan, so discovery
and the cascade ignore it) and is overridable via
``REPOSITORY_MANAGER_WORKTREE_ROOT``.
"""

from __future__ import annotations

import os
import shlex
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from repository_manager.repository_manager import Git

WORKTREE_ROOT = os.path.abspath(
    os.path.expanduser(
        os.getenv("REPOSITORY_MANAGER_WORKTREE_ROOT", "/home/apps/worktrees")
    )
)


def _slug(branch: str) -> str:
    """Filesystem-safe single path segment for a branch (``feat/x`` -> ``feat__x``)."""
    return branch.replace("/", "__")


class WorktreeManager:
    """Worktree operations layered on a :class:`Git` instance.

    Reuses ``git.git_action`` (the audited subprocess runner) and
    ``git.project_map`` (the workspace repo set), so worktree management inherits
    the same logging, timeouts, and workspace resolution as the rest of
    repository-manager.
    """

    def __init__(self, git: "Git") -> None:
        self.git = git

    # ── resolution ────────────────────────────────────────────────────────
    def resolve_repo(self, repo: str) -> str | None:
        """Resolve a repo *basename* or absolute path to its canonical checkout."""
        if repo and os.path.isdir(os.path.join(repo, ".git")):
            return os.path.abspath(repo)
        for p in self.git.project_map.values():
            if os.path.basename(p) == repo:
                return p
        # Fallback: common agent-packages layout under the workspace root.
        for base in (
            os.path.join(self.git.path, "agent-packages"),
            os.path.join(self.git.path, "agent-packages", "agents"),
        ):
            cand = os.path.join(base, repo)
            if os.path.isdir(os.path.join(cand, ".git")):
                return cand
        return None

    def worktree_path(self, repo: str, branch: str) -> str:
        return os.path.join(WORKTREE_ROOT, repo, _slug(branch))

    def _ok(self, res: Any) -> bool:
        return getattr(res, "status", "") == "success"

    def _run(self, cmd: str, path: str, quiet: bool = False) -> Any:
        return self.git.git_action(command=cmd, path=path, quiet=quiet)

    # ── actions ───────────────────────────────────────────────────────────
    def add(
        self, repo: str, branch: str, base: str = "main", adopt: bool = False
    ) -> dict[str, Any]:
        """Create (or reuse) a worktree for ``branch`` of ``repo``.

        Idempotent: returns the existing path if the worktree is already there.
        Parks the canonical checkout on ``base`` if it currently holds ``branch``
        (a branch can only be checked out once). With ``adopt=True``, any
        uncommitted changes in the canonical tree are stashed and replayed onto
        the new branch in the worktree (the "move my WIP onto a branch" flow).
        """
        if not repo or not branch:
            return {"ok": False, "error": "repo and branch are required"}
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        wt = self.worktree_path(repo, branch)
        if os.path.isdir(wt):
            return {
                "ok": True, "repo": repo, "branch": branch, "path": wt,
                "created": False, "status": "exists",
            }
        os.makedirs(os.path.dirname(wt), exist_ok=True)

        # Best-effort: make sure base is current (ignore failures, e.g. offline).
        self._run(f"git fetch origin {shlex.quote(base)}", canonical, quiet=True)

        adopted = False
        if adopt:
            st = self._run("git status --porcelain", canonical, quiet=True)
            if self._ok(st) and st.data.strip():
                self._run('git stash push -u -m "rm_worktree adopt"', canonical)
                adopted = True

        cur = self._run("git rev-parse --abbrev-ref HEAD", canonical, quiet=True)
        if self._ok(cur) and cur.data.strip() == branch:
            self._run(f"git checkout {shlex.quote(base)}", canonical)

        exists = self._run(
            f"git rev-parse --verify --quiet refs/heads/{shlex.quote(branch)}",
            canonical, quiet=True,
        )
        if self._ok(exists) and exists.data.strip():
            cmd = f"git worktree add {shlex.quote(wt)} {shlex.quote(branch)}"
        else:
            cmd = f"git worktree add {shlex.quote(wt)} -b {shlex.quote(branch)} {shlex.quote(base)}"
        res = self._run(cmd, canonical)
        if not self._ok(res):
            if adopted:  # restore the stash we took
                self._run("git stash pop", canonical)
            return {
                "ok": False, "path": wt,
                "error": res.error.message if res.error else res.data,
            }

        if adopted:
            pop = self._run("git stash pop", wt)
            adopted = self._ok(pop)
        return {
            "ok": True, "repo": repo, "branch": branch, "path": wt,
            "base": base, "created": True, "adopted": adopted,
        }

    def list(self, repo: str | None = None) -> dict[str, Any]:
        """List worktrees for one repo, or across every workspace repo."""
        repos = (
            [self.resolve_repo(repo)] if repo
            else list(self.git.project_map.values())
        )
        out: list[dict[str, Any]] = []
        for canonical in filter(None, repos):
            res = self._run("git worktree list --porcelain", canonical, quiet=True)
            if not self._ok(res):
                continue
            name = os.path.basename(canonical)
            cur: dict[str, Any] = {}
            for line in res.data.splitlines():
                if line.startswith("worktree "):
                    if cur:
                        out.append(cur)
                    cur = {"repo": name, "path": line[len("worktree "):]}
                elif line.startswith("branch "):
                    cur["branch"] = line[len("branch "):].replace("refs/heads/", "")
                elif line.startswith("HEAD "):
                    cur["head"] = line[len("HEAD "):][:10]
                elif line.startswith("detached"):
                    cur["branch"] = "(detached)"
            if cur:
                out.append(cur)
        for w in out:
            w["linked"] = str(w.get("path", "")).startswith(WORKTREE_ROOT)
        return {"ok": True, "worktrees": out, "count": len(out)}

    def remove(
        self, repo: str, branch: str, force: bool = False,
        delete_branch: bool = False,
    ) -> dict[str, Any]:
        """Remove a worktree (and prune); refuses a dirty tree unless ``force``."""
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        wt = self.worktree_path(repo, branch)
        flag = " --force" if force else ""
        res = self._run(f"git worktree remove{flag} {shlex.quote(wt)}", canonical)
        if not self._ok(res):
            return {"ok": False, "error": res.error.message if res.error else res.data}
        self._run("git worktree prune", canonical, quiet=True)
        deleted = False
        if delete_branch:
            d = self._run(f"git branch -D {shlex.quote(branch)}", canonical)
            deleted = self._ok(d)
        return {
            "ok": True, "repo": repo, "branch": branch,
            "removed": wt, "branch_deleted": deleted,
        }

    def merge(
        self, repo: str, branch: str, into: str = "main", no_ff: bool = True
    ) -> dict[str, Any]:
        """Merge a worktree ``branch`` back into ``into`` on the canonical checkout."""
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        cur = self._run("git rev-parse --abbrev-ref HEAD", canonical, quiet=True)
        if self._ok(cur) and cur.data.strip() != into:
            co = self._run(f"git checkout {shlex.quote(into)}", canonical)
            if not self._ok(co):
                return {"ok": False, "error": f"cannot checkout {into}: {co.data}"}
        ff = "--no-ff" if no_ff else ""
        res = self._run(f"git merge {ff} {shlex.quote(branch)}".replace("  ", " "), canonical)
        return {
            "ok": self._ok(res), "repo": repo, "branch": branch, "into": into,
            "output": res.data, "conflict": "conflict" in res.data.lower(),
        }

    def sync(
        self, repo: str, branch: str, base: str = "main", strategy: str = "rebase"
    ) -> dict[str, Any]:
        """Bring a worktree branch up to date with ``base`` (rebase or merge)."""
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        wt = self.worktree_path(repo, branch)
        if not os.path.isdir(wt):
            return {"ok": False, "error": f"no worktree at {wt}"}
        self._run(f"git fetch origin {shlex.quote(base)}", wt, quiet=True)
        op = "merge" if strategy == "merge" else "rebase"
        res = self._run(f"git {op} origin/{shlex.quote(base)}", wt)
        return {
            "ok": self._ok(res), "repo": repo, "branch": branch,
            "strategy": op, "output": res.data,
        }

    def prune(self, repo: str | None = None) -> dict[str, Any]:
        """Prune stale worktree administrative entries across the workspace."""
        repos = (
            [self.resolve_repo(repo)] if repo
            else list(self.git.project_map.values())
        )
        pruned: list[dict[str, str]] = []
        for canonical in filter(None, repos):
            r = self._run("git worktree prune -v", canonical, quiet=True)
            if self._ok(r) and r.data.strip():
                pruned.append({"repo": os.path.basename(canonical), "output": r.data.strip()})
        return {"ok": True, "pruned": pruned}

    def bulk_add(
        self, branch: str, repos: list[str] | None = None, base: str = "main"
    ) -> dict[str, Any]:
        """Create one worktree/branch per repo (a cross-repo session)."""
        targets = repos or [os.path.basename(p) for p in self.git.project_map.values()]
        results = [self.add(r, branch, base=base) for r in targets]
        return {"ok": all(x.get("ok") for x in results), "results": results}
