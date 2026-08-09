"""Git worktree management for concurrent multi-session development.

CONCEPT:RM-WORKTREE — lets N agent sessions work the same repo at once. Each
session takes its **own branch** in its **own worktree** under ``WORKTREE_ROOT``,
all sharing a single ``.git`` object store (no re-clone). The canonical checkout
stays on its default branch — what the validate/sync cascade expects — so a
working-tree reset on the canonical path never touches a session's worktree
files. Git's invariant (a branch lives in at most one worktree) is the hard lock
that keeps concurrent sessions from colliding.

Worktrees live at ``<WORKTREE_ROOT>/<repo>/<branch-slug>``. ``WORKTREE_ROOT``
defaults beneath the platform's XDG state directory (outside the workspace scan,
so discovery and the cascade ignore it) and is overridable via
``REPOSITORY_MANAGER_WORKTREE_ROOT``.

Every checkout this module runs directly against a *canonical* checkout (as
opposed to a linked worktree) goes through
:func:`repository_manager.canonical_guard.guarded_canonical_mutation`
(CONCEPT:RM-CANON-GUARD) first, so a dirty canonical tree is skipped-and-
reported instead of silently clobbered. Symmetrically, every *destructive* step
against a linked worktree — removing the directory, deleting the branch ref —
goes through :mod:`repository_manager.prune_guard` (CONCEPT:RM-PRUNE-GUARD), so
a worktree an active lane still holds is skipped and a branch ref is only ever
deleted when git itself agrees its commits survive the deletion. ``add``'s
``adopt=True`` WIP hand-off never touches the shared ``refs/stash`` stack for
longer than one git plumbing call — it goes through
:mod:`repository_manager.stash_guard` (CONCEPT:RM-STASH-GUARD, registry
``D-CP-1``), which parks the WIP on a private ref instead, so a concurrent
stash push/pop elsewhere in this shared ``.git`` can never cross with it.
"""

from __future__ import annotations

import hashlib
import os
import shlex
import time
from typing import Any, Protocol

from repository_manager import prune_guard, stash_guard
from repository_manager.canonical_guard import guarded_canonical_mutation


class GitLike(Protocol):
    """Structural type for the ``Git`` surface :class:`WorktreeManager` depends on.

    Typing the dependency by shape (not the concrete ``Git`` class) lets the real
    ``Git`` and lightweight test doubles both satisfy it.
    """

    path: str
    project_map: dict[str, str]

    # Worktree parsing needs raw machine-readable Git output; real Git carries
    # additional defaulted parameters and FakeGit absorbs them via ``**kwargs``.
    def git_action(
        self,
        command: str,
        path: str | None = ...,
        quiet: bool = ...,
        env: dict | None = ...,
        timeout: int = ...,
        raw_output: bool = ...,
    ) -> Any: ...


_XDG_STATE_HOME = os.getenv(
    "XDG_STATE_HOME", os.path.join(os.path.expanduser("~"), ".local", "state")
)
WORKTREE_ROOT = os.path.abspath(
    os.path.expanduser(
        os.getenv(
            "REPOSITORY_MANAGER_WORKTREE_ROOT",
            os.path.join(_XDG_STATE_HOME, "repository-manager", "worktrees"),
        )
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

    def __init__(self, git: GitLike, registry: Any | None = None) -> None:
        self.git = git
        # The registry is optional so the historical worktree verbs remain
        # usable during rollback and migration.  Managed lifecycle callers use
        # ``allocate`` below, which reserves before invoking ``add``.
        self.registry = registry

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
        """Return a worktree path that cannot escape ``WORKTREE_ROOT``.

        ``resolve_repo`` accepts absolute canonical paths, but passing one
        directly to :func:`os.path.join` discards every preceding component.
        Use the repository's final path component as the stable worktree key so
        basename and absolute-path callers resolve to the same isolated lane.
        """
        repo_key = os.path.basename(os.path.normpath(repo))
        branch_key = _slug(branch)
        if repo_key in {"", ".", ".."} or branch_key in {"", ".", ".."}:
            raise ValueError("repo and branch must name safe path components")

        root = os.path.abspath(WORKTREE_ROOT)
        path = os.path.abspath(os.path.join(root, repo_key, branch_key))
        if os.path.commonpath((root, path)) != root:
            raise ValueError("worktree path escapes WORKTREE_ROOT")
        return path

    def _ok(self, res: Any) -> bool:
        return getattr(res, "status", "") == "success"

    def _run(self, cmd: str, path: str, quiet: bool = False) -> Any:
        return self.git.git_action(command=cmd, path=path, quiet=quiet, raw_output=True)

    # ── actions ───────────────────────────────────────────────────────────
    def add(
        self, repo: str, branch: str, base: str = "main", adopt: bool = False
    ) -> dict[str, Any]:
        """Create (or reuse) a worktree for ``branch`` of ``repo``.

        Idempotent: returns the existing path if the worktree is already there.
        Parks the canonical checkout on ``base`` if it currently holds ``branch``
        (a branch can only be checked out once). With ``adopt=True``, any
        uncommitted changes in the canonical tree are moved onto the new branch
        in the worktree (the "move my WIP onto a branch" flow) via
        :mod:`repository_manager.stash_guard` (CONCEPT:RM-STASH-GUARD) — a
        private ref, never the shared ``refs/stash`` stack, so a concurrent
        stash push/pop elsewhere in this ``.git`` can never cross with it.
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

        stash_ref: str | None = None
        if adopt:
            capture = stash_guard.capture_wip(self.git, canonical, label=_slug(branch))
            if not capture["ok"]:
                return {
                    "ok": False,
                    "error": f"could not adopt WIP: {capture['error']}",
                }
            stash_ref = capture["ref"]  # None when the canonical tree was clean

        def _restore_onto_canonical_and_return(out: dict[str, Any]) -> dict[str, Any]:
            """Bail out of ``add`` while a captured WIP still needs a home."""
            if stash_ref is None:
                return out
            restore = stash_guard.apply_and_clear(self.git, canonical, stash_ref)
            if not restore["ok"]:
                out = {
                    **out,
                    "stash_ref": stash_ref,
                    "stash_recovery_error": restore["error"],
                }
            return out

        cur = self._run("git rev-parse --abbrev-ref HEAD", canonical, quiet=True)
        if self._ok(cur) and cur.data.strip() == branch:
            with guarded_canonical_mutation(
                self.git,
                canonical,
                repo,
                "park canonical checkout off requested branch",
            ) as blocked:
                if blocked is not None:
                    return _restore_onto_canonical_and_return(
                        {**blocked, "branch": branch}
                    )
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
            return _restore_onto_canonical_and_return(
                {
                    "ok": False,
                    "path": wt,
                    "error": res.error.message if res.error else res.data,
                }
            )

        adopted = False
        out: dict[str, Any] = {}
        if stash_ref is not None:
            applied = stash_guard.apply_and_clear(self.git, wt, stash_ref)
            adopted = applied["ok"]
            if not applied["ok"]:
                out["stash_ref"] = stash_ref
                out["stash_recovery_error"] = applied["error"]
        return {
            "ok": True,
            "repo": repo,
            "branch": branch,
            "path": wt,
            "base": base,
            "created": True,
            "adopted": adopted,
            **out,
        }

    def allocate(
        self,
        repo: str,
        branch: str,
        *,
        base: str = "main",
        owner_id: str,
        session_id: str,
        host_id: str = "",
        request_id: str | None = None,
        idempotency_key: str | None = None,
        predicted_disk_bytes: int = 0,
        disk_budget_bytes: int | None = None,
        ttl_seconds: int = 3600,
        adopt: bool = False,
        operator_id: str = "",
        registry: Any | None = None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        """Reserve a durable lane before creating its linked worktree.

        This is a bounded adapter: all Git creation still goes through
        :meth:`add`, while identity, quota, and fencing are owned by the lane
        registry.  A quota refusal therefore happens before ``add`` can create
        a directory.
        """

        from repository_manager.lane_registry import LaneRegistry

        authority = registry or self.registry
        if authority is None:
            authority = LaneRegistry()
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        worktree = self.worktree_path(repo, branch)
        key = idempotency_key or request_id
        if key is None:
            # Keep the deterministic material out of the public key itself.
            # LaneRecord rejects control characters, so forwarding the NUL-
            # delimited digest input directly made the ordinary no-key path
            # fail before a worktree could be created.
            material = f"{canonical}\0{branch}\0{owner_id}\0{session_id}"
            key = "auto:" + hashlib.sha256(material.encode("utf-8")).hexdigest()
        try:
            if adopt:
                if not operator_id:
                    return {
                        "ok": False,
                        "stage": "registry",
                        "error": "legacy adoption requires explicit operator_id",
                    }
                candidates = [
                    item
                    for item in authority.list_records()
                    if item.state.value == "observed_legacy"
                    and item.repository_path == canonical
                    and item.branch == branch
                    and item.worktree_path == worktree
                ]
                if len(candidates) != 1:
                    return {
                        "ok": False,
                        "stage": "registry",
                        "error": "observed legacy lane could not be uniquely resolved",
                    }
                record = authority.adopt(
                    candidates[0].lane_id,
                    owner_id=owner_id,
                    session_id=session_id,
                    host_id=host_id,
                    operator_id=operator_id,
                    now=now,
                )
            else:
                record = authority.allocate(
                    canonical,
                    branch,
                    worktree,
                    owner_id=owner_id,
                    session_id=session_id,
                    host_id=host_id,
                    request_id=key,
                    base_ref=base,
                    ttl_seconds=ttl_seconds,
                    predicted_disk_bytes=predicted_disk_bytes,
                    disk_budget_bytes=disk_budget_bytes,
                    now=now,
                )
        except Exception as exc:
            return {
                "ok": False,
                "stage": "registry",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        added = self.add(repo, branch, base=base, adopt=adopt)
        if not added.get("ok"):
            try:
                authority.abort(
                    record.lane_id,
                    owner_id=owner_id,
                    fence=record.fence,
                    reason="worktree creation failed",
                )
            except Exception:
                # The original creation error is the actionable response; the
                # durable row remains for reconciliation if abort itself fails.
                pass
            return {
                "ok": False,
                "stage": "worktree",
                "result": added,
                "lane_id": record.lane_id,
            }
        try:
            active = record
            if not adopt:
                active = authority.activate(
                    record.lane_id,
                    owner_id=owner_id,
                    fence=record.fence,
                    worktree_path=added.get("path") or worktree,
                    now=now,
                )
        except Exception as exc:
            return {
                "ok": False,
                "stage": "registry-activate",
                "lane_id": record.lane_id,
                "fence": record.fence,
                "worktree": added,
                "error": str(exc),
            }
        return {
            **added,
            "lane_id": active.lane_id,
            "fence": active.fence,
            "record": active.model_dump(mode="json"),
            "registry": authority,
        }

    # Explicit names make the adapter seam discoverable to later consumers
    # while retaining ``allocate`` as the concise worktree API.
    allocate_lane = allocate

    def heartbeat(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        observed_disk_bytes: int | None = None,
        now: Any | None = None,
        registry: Any | None = None,
    ) -> dict[str, Any]:
        """Forward a lane heartbeat to the durable fenced registry."""

        authority = registry or self.registry
        if authority is None:
            return {"ok": False, "error": "lane registry is not configured"}
        try:
            record = authority.heartbeat(
                lane_id,
                owner_id=owner_id,
                fence=fence,
                observed_disk_bytes=observed_disk_bytes,
                now=now,
            )
        except Exception as exc:
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        return {
            "ok": True,
            "lane_id": lane_id,
            "fence": record.fence,
            "record": record.model_dump(mode="json"),
        }

    heartbeat_lane = heartbeat

    def finish(
        self,
        lane_id: str,
        *,
        owner_id: str,
        fence: str,
        registry: Any | None = None,
        now: Any | None = None,
    ) -> dict[str, Any]:
        """Forward fenced lane completion without mutating Git directly."""

        authority = registry or self.registry
        if authority is None:
            return {"ok": False, "error": "lane registry is not configured"}
        try:
            record = authority.finish(lane_id, owner_id=owner_id, fence=fence, now=now)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        return {
            "ok": True,
            "lane_id": lane_id,
            "record": record.model_dump(mode="json"),
        }

    finish_lane = finish

    def status(
        self,
        lane_id: str,
        *,
        registry: Any | None = None,
    ) -> dict[str, Any]:
        """Return one durable lane status projection."""

        authority = registry or self.registry
        if authority is None:
            return {"ok": False, "error": "lane registry is not configured"}
        try:
            record = authority.status(lane_id)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
        return {
            "ok": True,
            "lane_id": lane_id,
            "record": record.model_dump(mode="json"),
        }

    lane_status = status

    def list_worktrees(self, repo: str | None = None) -> dict[str, Any]:
        """List worktrees for one repo, or across every workspace repo.

        D-CDX-1 — ``linked`` used to be ``path.startswith(WORKTREE_ROOT)``, a
        heuristic tied to THIS process's configured worktree root. Two
        surfaces of this same package legitimately run with different roots
        (the local CLI defaults to ``~/.local/state/.../worktrees``; the
        deployed MCP server sets ``REPOSITORY_MANAGER_WORKTREE_ROOT=
        /home/apps/worktrees``), so the identical lane came back
        ``linked=false`` from one surface and ``linked=true`` from the other
        for the SAME real linked worktree — a fact about git, misreported as
        a fact about a config value neither surface is wrong to have set
        differently. ``git worktree list`` already draws this line for free:
        the first entry it prints for a repo is always that repo's own
        (canonical) working tree; every entry after it is unconditionally a
        linked worktree, regardless of which directory any of them happen to
        live under. Comparing against the canonical path directly is the
        deployment-agnostic ground truth ``WORKTREE_ROOT`` was only ever a
        proxy for.
        """
        repos = (
            [self.resolve_repo(repo)] if repo else list(self.git.project_map.values())
        )
        out: list[dict[str, Any]] = []
        for canonical in filter(None, repos):
            res = self._run("git worktree list --porcelain", canonical, quiet=True)
            if not self._ok(res):
                continue
            name = os.path.basename(canonical)
            canonical_norm = os.path.abspath(canonical)
            cur: dict[str, Any] = {}
            for line in res.data.splitlines():
                if line.startswith("worktree "):
                    if cur:
                        out.append(cur)
                    entry_path = line[len("worktree ") :]
                    cur = {
                        "repo": name,
                        "path": entry_path,
                        "linked": os.path.abspath(entry_path) != canonical_norm,
                    }
                elif line.startswith("branch "):
                    cur["branch"] = line[len("branch ") :].replace("refs/heads/", "")
                elif line.startswith("HEAD "):
                    cur["head"] = line[len("HEAD ") :][:10]
                elif line.startswith("detached"):
                    cur["branch"] = "(detached)"
            if cur:
                out.append(cur)
        return {"ok": True, "worktrees": out, "count": len(out)}

    def remove(
        self,
        repo: str,
        branch: str,
        force: bool = False,
        delete_branch: bool = False,
        base: str = "main",
    ) -> dict[str, Any]:
        """Remove a worktree (and prune); refuses an occupied lane unconditionally.

        D-CDX-15 — **this used to be the one destructive worktree path that
        never asked the lane protocol whether anyone was still working here.**
        :meth:`audit`'s ``prune_merged`` path always re-justified a removal
        through :func:`repository_manager.prune_guard.guarded_worktree_prune`
        (CONCEPT:RM-PRUNE-GUARD); this direct entry point (reachable from
        ``rm_worktree action=remove`` with caller-controlled ``force``) did
        not, so a caller passing ``force=True`` bypassed occupancy detection
        entirely — the exact gap named by "existing canonical-tree guards do
        not protect lane branches from this failure mode". It now runs the
        SAME guard first and refuses (rather than mutating) whenever a lane
        still holds this worktree: a merge/rebase in progress, uncommitted
        work, or a live lease. ``force`` still exists, but its scope is
        narrowed to what the docstring always claimed — it only overrides
        *git's own* dirty-tree refusal once the lane-occupancy guard has
        already cleared the tree; it can never bypass that guard.

        ``delete_branch`` goes through :meth:`_delete_merged_branch`, so an
        explicit removal request cannot orphan commits either, and reports
        ``branch_kept_reason`` when it declines.

        Honesty note (the hard constraint on this class of fix): the
        occupancy check is an ``flock``-backed lease in this workspace's
        shared arbitration directory — it arbitrates actors *on this host*
        that go through this code path or the lane protocol's own primitives.
        It gives no guarantee against a process on a different host, or any
        process that mutates the branch/worktree with raw git and never
        touches the lease file.
        """
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        wt = self.worktree_path(repo, branch)
        if prune_guard.worktree_is_locked(wt):
            return {
                "ok": False,
                "repo": repo,
                "branch": branch,
                "skipped": True,
                "reason": "worktree-locked",
                "error": (
                    f"refused to remove worktree {wt!r}: git's own worktree lock is set"
                ),
            }
        with prune_guard.guarded_worktree_prune(
            wt, operation="rm_worktree remove"
        ) as held:
            if held is not None:
                return {
                    "ok": False,
                    "repo": repo,
                    "branch": branch,
                    "skipped": True,
                    "reason": "lane-occupied",
                    "error": (
                        f"refused to remove worktree {wt!r}: {held}. "
                        "`force` only overrides git's own dirty-tree check, "
                        "never lane occupancy - let the owning lane finish "
                        "or withdraw first."
                    ),
                }
            flag = " --force" if force else ""
            res = self._run(f"git worktree remove{flag} {shlex.quote(wt)}", canonical)
            if not self._ok(res):
                return {
                    "ok": False,
                    "error": res.error.message if res.error else res.data,
                }
            self._run("git worktree prune", canonical, quiet=True)
            deleted = False
            reason = anchor = ""
            if delete_branch:
                deleted, reason, anchor = self._delete_merged_branch(
                    canonical, branch, base
                )
        out: dict[str, Any] = {
            "ok": True,
            "repo": repo,
            "branch": branch,
            "removed": wt,
            "branch_deleted": deleted,
        }
        if reason:
            out["branch_kept_reason"] = reason
        if anchor:
            out["branch_anchor"] = anchor
        return out

    def reset_branch(
        self, repo: str, branch: str, target: str = "main"
    ) -> dict[str, Any]:
        """The ONE sanctioned way to move a lane branch's tip — guarded, always.

        D-CDX-15 — no code path in this package ever did this before (the
        recorded incident's "reset: moving to main" reflog entry was produced
        by something outside this package), but the acceptance criteria is to
        *require* a lease/occupancy check "before any automated lane reset" —
        i.e. provide the missing guarded primitive so any future lane-recycle
        or reclaim feature has exactly one, protected, path to reach for
        instead of a raw ``git reset --hard``. Refuses unconditionally when:

        1. a lane still occupies the worktree (same check as :meth:`remove`), or
        2. the branch carries commits that are not yet reachable from
           ``target`` — i.e. resetting would make them unreachable from any
           ref this branch still names. This is the literal "occupied
           worktree with unmerged commits is refused rather than reset" case.

        There is deliberately no ``force`` override for either refusal — that
        would recreate the exact defect this closes. A branch that is already
        an ancestor of ``target`` (nothing to lose) resets without friction;
        anything else must be landed or explicitly deleted first.
        """
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        wt = self.worktree_path(repo, branch)
        if os.path.isdir(wt):
            with prune_guard.guarded_worktree_prune(
                wt, operation="rm_worktree reset_branch"
            ) as held:
                if held is not None:
                    return {
                        "ok": False,
                        "repo": repo,
                        "branch": branch,
                        "skipped": True,
                        "reason": "lane-occupied",
                        "error": (
                            f"refused to reset branch {branch!r}: {held}. "
                            "a lane reset never overrides occupancy."
                        ),
                    }
        ancestor = self._run(
            f"git merge-base --is-ancestor {shlex.quote(branch)} {shlex.quote(target)}",
            canonical,
            quiet=True,
        )
        if not self._ok(ancestor):
            return {
                "ok": False,
                "repo": repo,
                "branch": branch,
                "target": target,
                "skipped": True,
                "reason": "unmerged-commits",
                "error": (
                    f"refused to reset {branch!r} to {target!r}: {branch!r} "
                    f"has commits not reachable from {target!r} - resetting "
                    "would make them unreachable from any ref this branch "
                    "still names. Land or explicitly discard them first."
                ),
            }
        target_sha = self._run(
            f"git rev-parse --verify --quiet {shlex.quote(target)}",
            canonical,
            quiet=True,
        )
        if not self._ok(target_sha) or not target_sha.data.strip():
            return {"ok": False, "error": f"target {target!r} is not resolvable"}
        new_sha = target_sha.data.strip()
        res = self._run(
            f"git update-ref refs/heads/{shlex.quote(branch)} {shlex.quote(new_sha)}",
            canonical,
        )
        return {
            "ok": self._ok(res),
            "repo": repo,
            "branch": branch,
            "target": target,
            "new_sha": new_sha,
            "output": res.data,
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
            with guarded_canonical_mutation(
                self.git, canonical, repo, f"check out {into!r} for merge"
            ) as blocked:
                if blocked is not None:
                    return {**blocked, "branch": branch, "into": into}
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
        """Bring a worktree branch up to date with ``base`` (rebase or merge).

        D-CDX-29 — **the authoritative ref is the local ``base`` branch, never
        ``origin/<base>``.** A linked worktree shares its canonical repo's
        object store and refs (everything except ``HEAD``/index/per-worktree
        config), so ``refs/heads/<base>`` is visible from the worktree exactly
        as the merge queue left it — every landing in this workspace advances
        it with a local, fast-forward-only ``git update-ref``
        (:func:`repository_manager.merge_queue.land`) and never requires a
        push. ``origin/<base>`` is a remote-tracking ref that only moves on an
        explicit push; in this all-local workflow it routinely sits behind
        local ``base`` by however many landings have not been pushed (proven
        live: local ``main`` at a merge-queue commit, ``origin/main`` still
        many commits behind it). Rebasing a lane branch onto that stale ref
        silently drops every commit that only exists on local ``base`` from
        the lane's ancestry — the exact "reverts landed work" failure mode
        this item was opened against. The fetch below stays best-effort (it
        keeps ``origin/<base>`` itself from drifting further and is harmless
        if offline); it is never the rebase/merge target.
        """
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        wt = self.worktree_path(repo, branch)
        if not os.path.isdir(wt):
            return {"ok": False, "error": f"no worktree at {wt}"}
        # Best-effort only: refreshes the remote-tracking ref for anyone who
        # later wants it, but is never itself the sync target (see above).
        self._run(f"git fetch origin {shlex.quote(base)}", wt, quiet=True)
        base_ref = f"refs/heads/{base}"
        resolved = self._run(
            f"git rev-parse --verify --quiet {shlex.quote(base_ref)}", wt, quiet=True
        )
        if not self._ok(resolved) or not resolved.data.strip():
            return {
                "ok": False,
                "repo": repo,
                "branch": branch,
                "error": (
                    f"authoritative ref {base_ref!r} is not resolvable from "
                    f"worktree {wt!r} - it must be a linked worktree of "
                    f"{canonical!r} sharing its refs; refusing to fall back "
                    "to a possibly-stale origin/<base>"
                ),
            }
        op = "merge" if strategy == "merge" else "rebase"
        res = self._run(f"git {op} {shlex.quote(base_ref)}", wt)
        return {
            "ok": self._ok(res),
            "repo": repo,
            "branch": branch,
            "strategy": op,
            "base_ref": base_ref,
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
        though it is checked out elsewhere. Returns dirty/ahead/behind/merged/
        at_base plus the age in days of the worktree's last commit.

        ``merged`` deliberately means **more** than "base contains every commit
        on this branch". ``ahead == 0`` alone is also true of a worktree that has
        not committed anything yet — it is still sitting exactly on ``base`` —
        and that worktree is the *start* of a lane, not the end of one. The
        observable difference is that a branch whose work was merged back leaves
        ``base`` strictly ahead of it (at minimum the merge commit itself), so
        ``merged`` requires ``behind > 0`` as the proof that this branch actually
        contributed something ``base`` now carries. A branch fast-forwarded into
        a ``base`` that has not moved since reports ``at_base`` instead and is
        kept — erring toward keeping is the only safe direction here.
        """
        state: dict[str, Any] = {
            "dirty": False,
            "ahead": 0,
            "behind": 0,
            "merged": False,
            "at_base": False,
            "last_commit_age_days": None,
        }
        porcelain = self._run("git status --porcelain", wt_path, quiet=True)
        state["dirty"] = bool(self._ok(porcelain) and porcelain.data.strip())
        # ahead/behind vs base in one shot; this call always exits 0. It is a
        # *classification* input only — it is never what authorises a deletion.
        # `_delete_merged_branch` re-asks `git merge-base --is-ancestor` at the
        # moment of deletion and then lets `git branch -d` re-decide under git's
        # own ref lock, so a count that has gone stale cannot orphan a commit.
        counts = self._run(
            f"git rev-list --left-right --count {shlex.quote(base)}...HEAD",
            wt_path,
            quiet=True,
        )
        if self._ok(counts) and counts.data.strip():
            parts = counts.data.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                state["behind"], state["ahead"] = int(parts[0]), int(parts[1])
                state["at_base"] = state["ahead"] == 0 and state["behind"] == 0
                state["merged"] = state["ahead"] == 0 and state["behind"] > 0
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
        dirty tree is always ``active`` (live edits); a clean tree that
        contributed work ``base`` now carries is ``merged``; a clean tree sitting
        exactly *on* ``base`` has contributed nothing yet and is ``active`` (a
        lane that has just opened its worktree, not one that has finished);
        otherwise an unmerged branch is ``active`` while it has recent commits
        and ``stale`` once it goes quiet.

        ``merged`` says only that this branch's work is captured in ``base`` — it
        never says the worktree is unoccupied. A lane that merges an intermediate
        chunk back and keeps working is ``merged`` and still live, which is
        exactly how D-FE-9 happened, so occupancy is asked separately at the
        moment of deletion (CONCEPT:RM-PRUNE-GUARD).
        """
        if not exists or branch in (None, "", "(detached)"):
            return "dangling"
        if state["dirty"]:
            return "active"
        if state["merged"]:
            return "merged"
        if state.get("at_base"):
            return "active"
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

    def _read_ref(self, canonical: str, ref: str) -> str:
        """The object a ref points at right now, or ``""`` when it does not exist."""
        res = self._run(
            f"git rev-parse --verify --quiet {shlex.quote(ref)}", canonical, quiet=True
        )
        return res.data.strip() if self._ok(res) else ""

    def _delete_merged_branch(
        self, canonical: str, branch: str, base: str
    ) -> tuple[bool, str, str]:
        """Delete ``branch`` only if its commits survive the deletion.

        A branch ref is the only thing standing between a lane's commits and
        ``gc``, so deleting one is gated harder than removing a directory, and
        gated *at the moment of deletion* rather than from an earlier scan:

        1. Read the tip. Every later step names that exact object, so a ref that
           moves underneath us cannot make an earlier answer apply to a commit it
           was never asked about.
        2. ``git merge-base --is-ancestor <tip> <base>`` — the honest
           reachability question, asked now. A non-zero exit is this command's
           *answer*, not a malfunction; the audited runner logs it as a failure,
           which is accurate here because it accompanies a real refusal, and it
           is asked only of the handful of branches actually up for deletion
           rather than of every worktree in the audit.
        3. **Anchor before deleting.** ``refs/lane-backup/<branch>`` is pointed at
           the tip immediately before the delete — one ref write, taken at the
           moment of deletion rather than at lane start, so it cannot go stale
           the way an anchor laid down once and never refreshed does. Belt and
           braces on the one operation whose failure turns commits into garbage:
           even a wrong answer above now costs a ref to clean up rather than a
           lane's work. The namespace is the one the workspace already uses for
           this; a pre-existing anchor is restored, never discarded, if the
           delete then fails.
        4. ``git branch -d`` — never ``-D``. Git re-decides reachability itself,
           under its own ref lock, atomically with the delete. This is what makes
           the guarantee hold through a check-then-delete race: no window exists
           between git's decision and git's action. If the tip moved to an
           unmerged commit in the meantime, ``-d`` refuses; if it moved to
           another merged commit, that commit is reachable from ``base`` anyway
           and the anchor still covers the tip we vouched for.

        Returns ``(deleted, reason, anchor)``; ``reason`` explains a refusal and
        ``anchor`` names the backup ref left behind by a successful delete.
        """
        tip = self._read_ref(canonical, f"refs/heads/{branch}")
        if not tip:
            return False, f"branch {branch!r} does not exist", ""
        ancestor = self._run(
            f"git merge-base --is-ancestor {tip} {shlex.quote(base)}",
            canonical,
            quiet=True,
        )
        if not self._ok(ancestor):
            return (
                False,
                (
                    f"refused to delete branch {branch!r}: {tip[:12]} has commits "
                    f"that are not reachable from {base!r}, so deleting the ref "
                    "would turn them into unreferenced objects"
                ),
                "",
            )
        anchor = f"refs/lane-backup/{branch.replace('/', '-')}"
        previous = self._read_ref(canonical, anchor)
        self._run(f"git update-ref {shlex.quote(anchor)} {tip}", canonical, quiet=True)
        deleted = self._run(f"git branch -d {shlex.quote(branch)}", canonical)
        if not self._ok(deleted):
            # Leave the ref namespace exactly as we found it: restore a
            # pre-existing anchor, or remove only the one we just wrote.
            if previous:
                self._run(
                    f"git update-ref {shlex.quote(anchor)} {previous} {tip}",
                    canonical,
                    quiet=True,
                )
            else:
                self._run(
                    f"git update-ref -d {shlex.quote(anchor)} {tip}",
                    canonical,
                    quiet=True,
                )
            detail = deleted.error.message if deleted.error else deleted.data
            return (
                False,
                f"git refused to delete branch {branch!r} as merged: {detail}",
                "",
            )
        return True, "", anchor

    def delete_merged_branch(
        self, canonical: str, branch: str, base: str = "main"
    ) -> tuple[bool, str, str]:
        """Public entry to :meth:`_delete_merged_branch`'s guarded ref deletion.

        The merge queue (:mod:`repository_manager.merge_queue`) prunes a landed
        candidate's branch and must go through the SAME anchor + merge-base
        re-check + ``git branch -d`` sequence rather than reimplementing it —
        reimplementing this guard inline is exactly the duplication D-ORC-21
        recorded. It cannot use :meth:`remove` for that, because ``remove``
        reconstructs the worktree path from ``WORKTREE_ROOT`` while lanes in this
        workspace create worktrees at arbitrary paths (the same reason
        :meth:`_prune_merged` removes by actual path).
        """
        return self._delete_merged_branch(canonical, branch, base)

    def _prune_merged(
        self, worktrees: list[dict[str, Any]], base: str = "main"
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Remove only ``merged`` worktrees and ``dangling`` admin pointers.

        Removes by the worktree's *actual* path (these repos use custom worktree
        paths that ``worktree_path()`` would not reconstruct) and never touches
        ``active``/``stale`` work.

        Every removal is re-justified inside
        :func:`prune_guard.guarded_worktree_prune` (CONCEPT:RM-PRUNE-GUARD): the
        lane lease is held across the whole check-then-delete, occupancy is read
        from the lane protocol, and ``_branch_state`` is recomputed *under that
        lease* so a classification that went stale during the audit scan is
        caught rather than acted on. The branch ref is then deleted only through
        :meth:`_delete_merged_branch`.
        """
        pruned: list[dict[str, Any]] = []
        kept: list[dict[str, Any]] = []
        for w in worktrees:
            cls = w["class"]
            canonical = self.resolve_repo(w["repo"])
            path = w.get("path", "")
            if cls == "merged" and canonical and path:
                entry = {
                    "repo": w["repo"],
                    "branch": w["branch"],
                    "path": path,
                    "class": cls,
                }
                if prune_guard.worktree_is_locked(path):
                    kept.append({**entry, "reason": "worktree is locked (git)"})
                    continue
                with prune_guard.guarded_worktree_prune(
                    path, operation="prune merged worktree"
                ) as held:
                    if held is not None:
                        kept.append({**entry, "reason": held})
                        continue
                    fresh = self._branch_state(path, base)
                    if fresh["dirty"] or not fresh["merged"]:
                        kept.append(
                            {
                                **entry,
                                "reason": (
                                    "state changed since the audit scan "
                                    f"(dirty={fresh['dirty']}, "
                                    f"ahead={fresh['ahead']}, "
                                    f"behind={fresh['behind']}) - a lane is "
                                    "still working here"
                                ),
                            }
                        )
                        continue
                    res = self._run(
                        f"git worktree remove {shlex.quote(path)}", canonical
                    )
                    ok = self._ok(res)
                    entry["ok"] = ok
                    if ok and w.get("branch"):
                        (
                            entry["branch_deleted"],
                            reason,
                            anchor,
                        ) = self._delete_merged_branch(canonical, w["branch"], base)
                        if reason:
                            entry["branch_kept_reason"] = reason
                        if anchor:
                            entry["branch_anchor"] = anchor
                    if not ok:
                        entry["error"] = res.error.message if res.error else res.data
                (pruned if entry.get("ok") else kept).append(entry)
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
        ``dangling`` admin pointers (orphans stay untouched), and only after
        re-justifying each removal at the moment of deletion — see
        :meth:`_prune_merged` (CONCEPT:RM-PRUNE-GUARD). A ``merged``
        classification here is a *candidate*, never an authorisation.
        (CONCEPT:RM-WORKTREE-AUDIT)
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
                    "at_base": False,
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
            result["pruned"], result["kept"] = self._prune_merged(worktrees, base)
        return result
