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
import time
from typing import Any, Protocol


class GitLike(Protocol):
    """Structural type for the ``Git`` surface :class:`WorktreeManager` depends on.

    Typing the dependency by shape (not the concrete ``Git`` class) lets the real
    ``Git`` and lightweight test doubles both satisfy it.
    """

    path: str
    project_map: dict[str, str]

    # WorktreeManager only ever calls git_action(command=, path=, quiet=); real Git
    # carries extra defaulted params (env/timeout) and FakeGit absorbs them via
    # **kwargs, so both satisfy this minimal contract.
    def git_action(
        self, command: str, path: str | None = ..., quiet: bool = ...
    ) -> Any: ...


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

    def __init__(self, git: GitLike) -> None:
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
                "ok": True,
                "repo": repo,
                "branch": branch,
                "path": wt,
                "created": False,
                "status": "exists",
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
            canonical,
            quiet=True,
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
                "ok": False,
                "path": wt,
                "error": res.error.message if res.error else res.data,
            }

        if adopted:
            pop = self._run("git stash pop", wt)
            adopted = self._ok(pop)
        return {
            "ok": True,
            "repo": repo,
            "branch": branch,
            "path": wt,
            "base": base,
            "created": True,
            "adopted": adopted,
        }

    def list_worktrees(self, repo: str | None = None) -> dict[str, Any]:
        """List worktrees for one repo, or across every workspace repo."""
        repos = (
            [self.resolve_repo(repo)] if repo else list(self.git.project_map.values())
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
                    cur = {"repo": name, "path": line[len("worktree ") :]}
                elif line.startswith("branch "):
                    cur["branch"] = line[len("branch ") :].replace("refs/heads/", "")
                elif line.startswith("HEAD "):
                    cur["head"] = line[len("HEAD ") :][:10]
                elif line.startswith("detached"):
                    cur["branch"] = "(detached)"
            if cur:
                out.append(cur)
        for w in out:
            w["linked"] = str(w.get("path", "")).startswith(WORKTREE_ROOT)
        return {"ok": True, "worktrees": out, "count": len(out)}

    def remove(
        self,
        repo: str,
        branch: str,
        force: bool = False,
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
            "ok": True,
            "repo": repo,
            "branch": branch,
            "removed": wt,
            "branch_deleted": deleted,
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
        res = self._run(
            f"git merge {ff} {shlex.quote(branch)}".replace("  ", " "), canonical
        )
        return {
            "ok": self._ok(res),
            "repo": repo,
            "branch": branch,
            "into": into,
            "output": res.data,
            "conflict": "conflict" in res.data.lower(),
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
            "ok": self._ok(res),
            "repo": repo,
            "branch": branch,
            "strategy": op,
            "output": res.data,
        }

    def prune(self, repo: str | None = None) -> dict[str, Any]:
        """Prune stale worktree administrative entries across the workspace."""
        repos = (
            [self.resolve_repo(repo)] if repo else list(self.git.project_map.values())
        )
        pruned: list[dict[str, str]] = []
        for canonical in filter(None, repos):
            r = self._run("git worktree prune -v", canonical, quiet=True)
            if self._ok(r) and r.data.strip():
                pruned.append(
                    {"repo": os.path.basename(canonical), "output": r.data.strip()}
                )
        return {"ok": True, "pruned": pruned}

    def bulk_add(
        self, branch: str, repos: list[str] | None = None, base: str = "main"
    ) -> dict[str, Any]:
        """Create one worktree/branch per repo (a cross-repo session)."""
        targets = repos or [os.path.basename(p) for p in self.git.project_map.values()]
        results = [self.add(r, branch, base=base) for r in targets]
        return {"ok": all(x.get("ok") for x in results), "results": results}

    # ── audit (CONCEPT:RM-WORKTREE-AUDIT) ─────────────────────────────────
    def _branch_state(self, wt_path: str, base: str) -> dict[str, Any]:
        """Git state of a worktree relative to ``base``, run from the worktree.

        Worktrees share the object store and refs with their canonical checkout,
        so ``base`` (e.g. ``main``) resolves from inside a linked worktree even
        though it is checked out elsewhere. Returns dirty/ahead/behind/merged plus
        the age in days of the worktree's last commit.
        """
        state: dict[str, Any] = {
            "dirty": False,
            "ahead": 0,
            "behind": 0,
            "merged": False,
            "last_commit_age_days": None,
        }
        porcelain = self._run("git status --porcelain", wt_path, quiet=True)
        state["dirty"] = bool(self._ok(porcelain) and porcelain.data.strip())
        # merged == this worktree's HEAD is reachable from base (an ancestor).
        # `--is-ancestor` exits 0 when true, 1 when false (-> _ok False).
        anc = self._run(
            f"git merge-base --is-ancestor HEAD {shlex.quote(base)}",
            wt_path,
            quiet=True,
        )
        state["merged"] = self._ok(anc)
        counts = self._run(
            f"git rev-list --left-right --count {shlex.quote(base)}...HEAD",
            wt_path,
            quiet=True,
        )
        if self._ok(counts) and counts.data.strip():
            parts = counts.data.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                state["behind"], state["ahead"] = int(parts[0]), int(parts[1])
        ts = self._run("git log -1 --format=%ct HEAD", wt_path, quiet=True)
        if self._ok(ts) and ts.data.strip().isdigit():
            state["last_commit_age_days"] = (
                time.time() - int(ts.data.strip())
            ) / 86400.0
        return state

    @staticmethod
    def _classify(
        state: dict[str, Any], branch: str | None, exists: bool, stale_days: int
    ) -> str:
        """One of ``merged``/``active``/``stale``/``dangling`` for a worktree.

        Precedence is deliberate: a detached/missing worktree is ``dangling``; a
        dirty tree is always ``active`` (live edits); a clean tree whose work is
        already captured in ``base`` is ``merged`` (prunable) even if just
        committed; otherwise an unmerged branch is ``active`` while it has recent
        commits and ``stale`` once it goes quiet.
        """
        if not exists or branch in (None, "", "(detached)"):
            return "dangling"
        if state["dirty"]:
            return "active"
        if state["merged"]:
            return "merged"
        age = state["last_commit_age_days"]
        recent = age is not None and age <= stale_days
        if state["ahead"] > 0 and recent:
            return "active"
        return "stale"

    def _repo_states(
        self, repo: str | None = None, base: str = "main"
    ) -> list[dict[str, Any]]:
        """Per-canonical-repo git state: dirty / unpushed-to-origin / clean.

        Answers "which projects have unmerged or unpushed changes". ``base_unpushed``
        flags repos whose local ``base`` is ahead of ``origin/base`` so a worktree
        that is prunable-because-merged can still warn that ``base`` owes a push.
        """
        canon = (
            [self.resolve_repo(repo)] if repo else list(self.git.project_map.values())
        )
        out: list[dict[str, Any]] = []
        for path in filter(None, canon):
            cur = self._run("git rev-parse --abbrev-ref HEAD", path, quiet=True)
            branch = cur.data.strip() if self._ok(cur) and cur.data else None
            porc = self._run("git status --porcelain", path, quiet=True)
            dirty = bool(self._ok(porc) and porc.data.strip())
            ahead = behind = None
            no_upstream = True
            if branch:
                up = self._run(
                    f"git rev-parse --verify --quiet origin/{shlex.quote(branch)}",
                    path,
                    quiet=True,
                )
                if self._ok(up) and up.data.strip():
                    no_upstream = False
                    counts = self._run(
                        "git rev-list --left-right --count "
                        f"origin/{shlex.quote(branch)}...HEAD",
                        path,
                        quiet=True,
                    )
                    if self._ok(counts) and counts.data.strip():
                        parts = counts.data.split()
                        if len(parts) == 2 and all(p.isdigit() for p in parts):
                            behind, ahead = int(parts[0]), int(parts[1])
            base_unpushed = False
            ub = self._run(
                f"git rev-parse --verify --quiet origin/{shlex.quote(base)}",
                path,
                quiet=True,
            )
            if self._ok(ub) and ub.data.strip():
                bc = self._run(
                    f"git rev-list --count origin/{shlex.quote(base)}.."
                    f"{shlex.quote(base)}",
                    path,
                    quiet=True,
                )
                if self._ok(bc) and bc.data.strip().isdigit():
                    base_unpushed = int(bc.data.strip()) > 0
            if dirty:
                cls = "dirty"
            elif (ahead and ahead > 0) or no_upstream:
                # ahead of origin, or no remote to compare against -> work is
                # not on a remote.
                cls = "unpushed"
            else:
                cls = "clean"
            out.append(
                {
                    "repo": os.path.basename(path),
                    "branch": branch,
                    "dirty": dirty,
                    "ahead_origin": ahead,
                    "behind_origin": behind,
                    "no_upstream": no_upstream,
                    "base_unpushed": base_unpushed,
                    "class": cls,
                }
            )
        return out

    def _orphan_dirs(self, known_paths: set[str]) -> list[dict[str, str]]:
        """Dirs under ``WORKTREE_ROOT`` that look like worktrees but no repo tracks.

        Report-only: an orphan may still hold uncommitted work, so the auto-prune
        path never removes one. Scans one and two levels deep to cover both the
        flat (``<ROOT>/<repo>``) and nested (``<ROOT>/<repo>/<branch>``) layouts.
        """
        if not os.path.isdir(WORKTREE_ROOT):
            return []
        orphans: list[dict[str, str]] = []
        for top in sorted(os.listdir(WORKTREE_ROOT)):
            top_path = os.path.join(WORKTREE_ROOT, top)
            if not os.path.isdir(top_path):
                continue
            candidates = [top_path]
            for sub in sorted(os.listdir(top_path)):
                sub_path = os.path.join(top_path, sub)
                if os.path.isdir(sub_path):
                    candidates.append(sub_path)
            for cand in candidates:
                if os.path.abspath(cand) in known_paths:
                    continue
                if os.path.exists(os.path.join(cand, ".git")):
                    orphans.append(
                        {"path": cand, "reason": "untracked worktree directory"}
                    )
        return orphans

    def _prune_merged(
        self, worktrees: list[dict[str, Any]]
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Remove only ``merged`` worktrees and ``dangling`` admin pointers.

        Removes by the worktree's *actual* path (these repos use custom worktree
        paths that ``worktree_path()`` would not reconstruct). Never forces a dirty
        tree (merged worktrees are clean by definition) and never touches
        ``active``/``stale`` work.
        """
        pruned: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        for w in worktrees:
            cls = w["class"]
            canonical = self.resolve_repo(w["repo"])
            path = w.get("path", "")
            if cls == "merged" and canonical and path:
                res = self._run(
                    f"git worktree remove {shlex.quote(path)}", canonical
                )
                ok = self._ok(res)
                if ok and w.get("branch"):
                    self._run(
                        f"git branch -D {shlex.quote(w['branch'])}",
                        canonical,
                        quiet=True,
                    )
                entry = {
                    "repo": w["repo"],
                    "branch": w["branch"],
                    "path": path,
                    "class": cls,
                    "ok": ok,
                }
                if not ok:
                    entry["error"] = res.error.message if res.error else res.data
                (pruned if ok else kept).append(entry)
            elif cls == "dangling" and canonical:
                self._run("git worktree prune", canonical, quiet=True)
                pruned.append(
                    {
                        "repo": w["repo"],
                        "branch": w["branch"],
                        "path": path,
                        "class": cls,
                        "ok": True,
                    }
                )
            else:
                kept.append(
                    {
                        "repo": w["repo"],
                        "branch": w["branch"],
                        "class": cls,
                        "reason": "not prunable (active/stale)",
                    }
                )
        return pruned, kept

    def audit(
        self,
        repo: str | None = None,
        base: str = "main",
        stale_days: int = 14,
        prune_merged: bool = False,
    ) -> dict[str, Any]:
        """Classify every linked worktree (and canonical repo) by git state.

        Read-only by default. Buckets worktrees into ``merged`` (safe to prune),
        ``active`` (in-flight — do not disturb), ``stale`` (review), and
        ``dangling`` (stale admin entry); reports canonical repos with
        unmerged/unpushed changes and orphaned directories. With
        ``prune_merged=True`` it then removes only the ``merged`` worktrees and
        ``dangling`` admin pointers (orphans stay untouched). (CONCEPT:RM-WORKTREE-AUDIT)
        """
        listing = self.list_worktrees(repo=repo).get("worktrees", [])
        linked = [w for w in listing if w.get("linked")]
        worktrees: list[dict[str, Any]] = []
        for w in linked:
            path = str(w.get("path", ""))
            branch = w.get("branch")
            exists = bool(path) and os.path.isdir(path)
            if exists and branch not in (None, "", "(detached)"):
                state = self._branch_state(path, base)
            else:
                state = {
                    "dirty": False,
                    "ahead": 0,
                    "behind": 0,
                    "merged": False,
                    "last_commit_age_days": None,
                }
            worktrees.append(
                {
                    "repo": w.get("repo"),
                    "branch": branch,
                    "path": path,
                    "head": w.get("head"),
                    **state,
                    "class": self._classify(state, branch, exists, stale_days),
                }
            )

        repos_report = self._repo_states(repo=repo, base=base)
        base_unpushed = {r["repo"]: r.get("base_unpushed", False) for r in repos_report}
        for w in worktrees:
            w["base_unpushed"] = base_unpushed.get(w["repo"], False)

        known_paths = {os.path.abspath(str(w.get("path", ""))) for w in linked}
        orphans = self._orphan_dirs(known_paths)

        do_not_disturb = [w for w in worktrees if w["class"] == "active"]
        review = [w for w in worktrees if w["class"] == "stale"]
        safe = [w for w in worktrees if w["class"] in ("merged", "dangling")]
        summary = {
            "worktrees": len(worktrees),
            "merged": sum(w["class"] == "merged" for w in worktrees),
            "active": len(do_not_disturb),
            "stale": len(review),
            "dangling": sum(w["class"] == "dangling" for w in worktrees),
            "orphans": len(orphans),
            "unpushed_repos": sum(r["class"] == "unpushed" for r in repos_report),
        }
        result: dict[str, Any] = {
            "ok": True,
            "base": base,
            "stale_days": stale_days,
            "summary": summary,
            "worktrees": worktrees,
            "repos": repos_report,
            "orphans": orphans,
            "safe_to_prune": [
                {"repo": w["repo"], "branch": w["branch"], "class": w["class"]}
                for w in safe
            ],
            "do_not_disturb": [
                {"repo": w["repo"], "branch": w["branch"]} for w in do_not_disturb
            ],
            "review": [{"repo": w["repo"], "branch": w["branch"]} for w in review],
        }
        if prune_merged:
            result["pruned"], result["kept"] = self._prune_merged(worktrees)
        return result
