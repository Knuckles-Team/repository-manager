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
import logging
import os
import shlex
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol

from repository_manager import prune_guard, stash_guard
from repository_manager.canonical_guard import guarded_canonical_mutation

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _AllocateRequest:
    """Bundled, already-resolved inputs shared by ``allocate``'s helpers.

    Bundled so the extracted helpers stay under the fleet's parameter cap;
    ``allocate`` itself keeps its existing keyword-only signature
    (pre-existing debt, not widened here).
    """

    repo: str
    canonical: str
    branch: str
    worktree: str
    base: str
    owner_id: str
    session_id: str
    host_id: str
    key: str
    predicted_disk_bytes: int
    disk_budget_bytes: int | None
    ttl_seconds: int
    adopt: bool
    operator_id: str
    now: Any | None


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

    def __init__(
        self,
        git: GitLike,
        registry: Any | None = None,
        *,
        registry_factory: Callable[[], Any] | None = None,
    ) -> None:
        self.git = git
        # The registry is optional so the historical worktree verbs remain
        # usable during rollback and migration.  Managed lifecycle callers use
        # ``allocate`` below, which reserves before invoking ``add``.
        self.registry = registry
        # RMDD-28 (LANE-4): the production no-fallback constructor
        # (``repository_manager.native_lane_authority.create_production_lane_registry``)
        # has exactly one production call site -- ``allocate`` below -- and
        # nothing ever supplied it, so it was unreachable even though it
        # existed. This is a zero-arg seam a deployment's composition root can
        # bind (e.g. via ``functools.partial(create_production_lane_registry,
        # graph_client, engine, tenant_ref=..., host_ref=...)``) once it has a
        # live graph client/engine; ``allocate`` calls it only when neither
        # ``registry`` nor ``self.registry`` was already supplied, and never
        # catches a factory failure to substitute a local approximation --
        # that would recreate the exact silent-degrade this lane exists to
        # close. Deliberately NOT resolved from environment variables here:
        # this module has no established convention for acquiring a live
        # engine/graph client (RMDD-27's own ``create_production_resource_
        # scheduler`` has the identical gap), and inventing one is outside
        # this lane's scope and outside RMDD-09-owned territory.
        self._registry_factory = registry_factory

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

        stash_ref, error = self._capture_add_wip(canonical, branch, adopt)
        if error is not None:
            return error

        blocked_result = self._park_canonical_checkout(
            canonical, repo, branch, base, stash_ref
        )
        if blocked_result is not None:
            return blocked_result

        res = self._run_worktree_add(canonical, wt, branch, base)
        if not self._ok(res):
            return self._restore_wip_and_return(
                canonical,
                stash_ref,
                {
                    "ok": False,
                    "path": wt,
                    "error": res.error.message if res.error else res.data,
                },
            )

        adopted, out = self._apply_wip_to_worktree(wt, stash_ref)
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

    def _capture_add_wip(
        self, canonical: str, branch: str, adopt: bool
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Best-effort WIP capture for ``add(..., adopt=True)``.

        Returns ``(stash_ref, error)``; ``stash_ref`` is ``None`` both when
        ``adopt`` is False and when the canonical tree was already clean.
        """
        if not adopt:
            return None, None
        capture = stash_guard.capture_wip(self.git, canonical, label=_slug(branch))
        if not capture["ok"]:
            return None, {
                "ok": False,
                "error": f"could not adopt WIP: {capture['error']}",
            }
        return capture["ref"], None  # None when the canonical tree was clean

    def _restore_wip_and_return(
        self, canonical: str, stash_ref: str | None, out: dict[str, Any]
    ) -> dict[str, Any]:
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

    def _park_canonical_checkout(
        self, canonical: str, repo: str, branch: str, base: str, stash_ref: str | None
    ) -> dict[str, Any] | None:
        """Move the canonical checkout off ``branch`` so it can be worktree'd.

        Returns an error dict if the canonical mutation is refused/blocked
        (with any captured WIP restored first), else ``None`` to continue.
        """
        cur = self._run("git rev-parse --abbrev-ref HEAD", canonical, quiet=True)
        if not (self._ok(cur) and cur.data.strip() == branch):
            return None
        with guarded_canonical_mutation(
            self.git,
            canonical,
            repo,
            "park canonical checkout off requested branch",
        ) as blocked:
            if blocked is not None:
                return self._restore_wip_and_return(
                    canonical, stash_ref, {**blocked, "branch": branch}
                )
            self._run(f"git checkout {shlex.quote(base)}", canonical)
        return None

    def _run_worktree_add(self, canonical: str, wt: str, branch: str, base: str) -> Any:
        exists = self._run(
            f"git rev-parse --verify --quiet refs/heads/{shlex.quote(branch)}",
            canonical,
            quiet=True,
        )
        if self._ok(exists) and exists.data.strip():
            cmd = f"git worktree add {shlex.quote(wt)} {shlex.quote(branch)}"
        else:
            cmd = (
                f"git worktree add {shlex.quote(wt)} -b {shlex.quote(branch)} "
                f"{shlex.quote(base)}"
            )
        return self._run(cmd, canonical)

    def _apply_wip_to_worktree(
        self, wt: str, stash_ref: str | None
    ) -> tuple[bool, dict[str, Any]]:
        if stash_ref is None:
            return False, {}
        applied = stash_guard.apply_and_clear(self.git, wt, stash_ref)
        adopted = applied["ok"]
        out: dict[str, Any] = {}
        if not applied["ok"]:
            out["stash_ref"] = stash_ref
            out["stash_recovery_error"] = applied["error"]
        return adopted, out

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

        authority, error = self._resolve_lane_authority(registry)
        if error is not None:
            return error
        canonical = self.resolve_repo(repo)
        if not canonical:
            return {"ok": False, "error": f"repo not found: {repo}"}
        worktree = self.worktree_path(repo, branch)
        key = self._allocate_idempotency_key(
            idempotency_key, request_id, canonical, branch, owner_id, session_id
        )
        req = _AllocateRequest(
            repo=repo,
            canonical=canonical,
            branch=branch,
            worktree=worktree,
            base=base,
            owner_id=owner_id,
            session_id=session_id,
            host_id=host_id,
            key=key,
            predicted_disk_bytes=predicted_disk_bytes,
            disk_budget_bytes=disk_budget_bytes,
            ttl_seconds=ttl_seconds,
            adopt=adopt,
            operator_id=operator_id,
            now=now,
        )
        record, error = self._allocate_lane_record(authority, req)
        if error is not None:
            return error
        assert record is not None
        added, error = self._create_allocated_worktree(authority, record, req)
        if error is not None:
            return error
        assert added is not None
        active, error = self._activate_allocated_lane(authority, record, added, req)
        if error is not None:
            return error
        assert active is not None
        return {
            **added,
            "lane_id": active.lane_id,
            "fence": active.fence,
            "record": active.model_dump(mode="json"),
            "registry": authority,
        }

    def _resolve_lane_authority(
        self, registry: Any | None
    ) -> tuple[Any, dict[str, Any] | None]:
        from repository_manager.lane_registry import LaneRegistry

        authority = registry or self.registry
        if authority is None and self._registry_factory is not None:
            # No local approximation on failure: a configured-but-broken
            # factory (e.g. the RMDD-28 native transport unavailable, or a
            # transport missing one of its eight verbs) must refuse here,
            # not be swallowed into the authority-less ``LaneRegistry()``
            # below. Only the ABSENCE of a factory falls through to that
            # historical local-only default.
            try:
                authority = self._registry_factory()
            except Exception as exc:
                return None, {
                    "ok": False,
                    "stage": "registry",
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                }
        if authority is None:
            authority = LaneRegistry()
        return authority, None

    @staticmethod
    def _allocate_idempotency_key(
        idempotency_key: str | None,
        request_id: str | None,
        canonical: str,
        branch: str,
        owner_id: str,
        session_id: str,
    ) -> str:
        key = idempotency_key or request_id
        if key is not None:
            return key
        # Keep the deterministic material out of the public key itself.
        # LaneRecord rejects control characters, so forwarding the NUL-
        # delimited digest input directly made the ordinary no-key path
        # fail before a worktree could be created.
        material = f"{canonical}\0{branch}\0{owner_id}\0{session_id}"
        return "auto:" + hashlib.sha256(material.encode("utf-8")).hexdigest()

    @staticmethod
    def _resolve_adoption_candidate(
        authority: Any, canonical: str, branch: str, worktree: str
    ) -> tuple[Any | None, dict[str, Any] | None]:
        candidates = [
            item
            for item in authority.list_records()
            if item.state.value == "observed_legacy"
            and item.repository_path == canonical
            and item.branch == branch
            and item.worktree_path == worktree
        ]
        if len(candidates) != 1:
            return None, {
                "ok": False,
                "stage": "registry",
                "error": "observed legacy lane could not be uniquely resolved",
            }
        return candidates[0], None

    def _allocate_lane_record(
        self, authority: Any, req: _AllocateRequest
    ) -> tuple[Any | None, dict[str, Any] | None]:
        try:
            if req.adopt:
                if not req.operator_id:
                    return None, {
                        "ok": False,
                        "stage": "registry",
                        "error": "legacy adoption requires explicit operator_id",
                    }
                candidate, error = self._resolve_adoption_candidate(
                    authority, req.canonical, req.branch, req.worktree
                )
                if error is not None:
                    return None, error
                assert candidate is not None
                record = authority.adopt(
                    candidate.lane_id,
                    owner_id=req.owner_id,
                    session_id=req.session_id,
                    host_id=req.host_id,
                    operator_id=req.operator_id,
                    now=req.now,
                )
            else:
                record = authority.allocate(
                    req.canonical,
                    req.branch,
                    req.worktree,
                    owner_id=req.owner_id,
                    session_id=req.session_id,
                    host_id=req.host_id,
                    request_id=req.key,
                    base_ref=req.base,
                    ttl_seconds=req.ttl_seconds,
                    predicted_disk_bytes=req.predicted_disk_bytes,
                    disk_budget_bytes=req.disk_budget_bytes,
                    now=req.now,
                )
        except Exception as exc:
            return None, {
                "ok": False,
                "stage": "registry",
                "error": str(exc),
                "error_type": type(exc).__name__,
            }
        return record, None

    def _create_allocated_worktree(
        self, authority: Any, record: Any, req: _AllocateRequest
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        added = self.add(req.repo, req.branch, base=req.base, adopt=req.adopt)
        if added.get("ok"):
            return added, None
        try:
            authority.abort(
                record.lane_id,
                owner_id=req.owner_id,
                fence=record.fence,
                reason="worktree creation failed",
            )
        except Exception:
            # The original creation error is the actionable response; the
            # durable row remains for reconciliation if abort itself fails.
            logger.debug(
                "lane abort after failed worktree creation also failed for %s",
                record.lane_id,
                exc_info=True,
            )
        return None, {
            "ok": False,
            "stage": "worktree",
            "result": added,
            "lane_id": record.lane_id,
        }

    @staticmethod
    def _activate_allocated_lane(
        authority: Any, record: Any, added: dict[str, Any], req: _AllocateRequest
    ) -> tuple[Any | None, dict[str, Any] | None]:
        try:
            active = record
            if not req.adopt:
                active = authority.activate(
                    record.lane_id,
                    owner_id=req.owner_id,
                    fence=record.fence,
                    worktree_path=added.get("path") or req.worktree,
                    now=req.now,
                )
        except Exception as exc:
            return None, {
                "ok": False,
                "stage": "registry-activate",
                "lane_id": record.lane_id,
                "fence": record.fence,
                "worktree": added,
                "error": str(exc),
            }
        return active, None

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
            out.extend(
                self._parse_worktree_porcelain(
                    res.data, os.path.basename(canonical), os.path.abspath(canonical)
                )
            )
        return {"ok": True, "worktrees": out, "count": len(out)}

    @staticmethod
    def _parse_worktree_porcelain(
        data: str, name: str, canonical_norm: str
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        cur: dict[str, Any] = {}
        for line in data.splitlines():
            if line.startswith("worktree "):
                if cur:
                    entries.append(cur)
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
            entries.append(cur)
        return entries

    def remove(
        self,
        repo: str,
        branch: str,
        force: bool = False,
        delete_branch: bool = False,
        base: str = "main",
    ) -> dict[str, Any]:
        """Remove one worktree without pruning unrelated registrations.

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
            # This operation owns one exact registration only. Broad stale
            # administrative cleanup belongs to the explicit prune paths.
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
        state.update(self._branch_ahead_behind(wt_path, base))
        state["last_commit_age_days"] = self._branch_last_commit_age_days(wt_path)
        return state

    def _branch_ahead_behind(self, wt_path: str, base: str) -> dict[str, Any]:
        # ahead/behind vs base in one shot; this call always exits 0. It is a
        # *classification* input only — it is never what authorises a deletion.
        # `_delete_merged_branch` re-asks `git merge-base --is-ancestor` at the
        # moment of deletion and then lets `git branch -d` re-decide under git's
        # own ref lock, so a count that has gone stale cannot orphan a commit.
        result: dict[str, Any] = {
            "ahead": 0,
            "behind": 0,
            "at_base": False,
            "merged": False,
        }
        counts = self._run(
            f"git rev-list --left-right --count {shlex.quote(base)}...HEAD",
            wt_path,
            quiet=True,
        )
        if self._ok(counts) and counts.data.strip():
            parts = counts.data.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                result["behind"], result["ahead"] = int(parts[0]), int(parts[1])
                result["at_base"] = result["ahead"] == 0 and result["behind"] == 0
                result["merged"] = result["ahead"] == 0 and result["behind"] > 0
        return result

    def _branch_last_commit_age_days(self, wt_path: str) -> float | None:
        ts = self._run("git log -1 --format=%ct HEAD", wt_path, quiet=True)
        if self._ok(ts) and ts.data.strip().isdigit():
            return (time.time() - int(ts.data.strip())) / 86400.0
        return None

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
        return [self._repo_state_row(path, base) for path in filter(None, canon)]

    def _repo_state_row(self, path: str, base: str) -> dict[str, Any]:
        branch, dirty = self._branch_and_dirty_state(path)
        ahead, behind, no_upstream = self._ahead_behind_state(path, branch)
        base_unpushed = self._base_unpushed_state(path, base)
        cls = self._classify_repo_state(dirty, ahead, no_upstream)
        return {
            "repo": os.path.basename(path),
            "branch": branch,
            "dirty": dirty,
            "ahead_origin": ahead,
            "behind_origin": behind,
            "no_upstream": no_upstream,
            "base_unpushed": base_unpushed,
            "class": cls,
        }

    def _branch_and_dirty_state(self, path: str) -> tuple[str | None, bool]:
        cur = self._run("git rev-parse --abbrev-ref HEAD", path, quiet=True)
        branch = cur.data.strip() if self._ok(cur) and cur.data else None
        porc = self._run("git status --porcelain", path, quiet=True)
        dirty = bool(self._ok(porc) and porc.data.strip())
        return branch, dirty

    def _ahead_behind_state(
        self, path: str, branch: str | None
    ) -> tuple[int | None, int | None, bool]:
        ahead: int | None = None
        behind: int | None = None
        no_upstream = True
        if not branch:
            return ahead, behind, no_upstream
        up = self._run(
            f"git rev-parse --verify --quiet origin/{shlex.quote(branch)}",
            path,
            quiet=True,
        )
        if not (self._ok(up) and up.data.strip()):
            return ahead, behind, no_upstream
        no_upstream = False
        counts = self._run(
            f"git rev-list --left-right --count origin/{shlex.quote(branch)}...HEAD",
            path,
            quiet=True,
        )
        if self._ok(counts) and counts.data.strip():
            parts = counts.data.split()
            if len(parts) == 2 and all(p.isdigit() for p in parts):
                behind, ahead = int(parts[0]), int(parts[1])
        return ahead, behind, no_upstream

    def _base_unpushed_state(self, path: str, base: str) -> bool:
        ub = self._run(
            f"git rev-parse --verify --quiet origin/{shlex.quote(base)}",
            path,
            quiet=True,
        )
        if not (self._ok(ub) and ub.data.strip()):
            return False
        bc = self._run(
            f"git rev-list --count origin/{shlex.quote(base)}..{shlex.quote(base)}",
            path,
            quiet=True,
        )
        if self._ok(bc) and bc.data.strip().isdigit():
            return int(bc.data.strip()) > 0
        return False

    @staticmethod
    def _classify_repo_state(dirty: bool, ahead: int | None, no_upstream: bool) -> str:
        if dirty:
            return "dirty"
        if (ahead and ahead > 0) or no_upstream:
            # ahead of origin, or no remote to compare against -> work is
            # not on a remote.
            return "unpushed"
        return "clean"

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
            candidates = self._worktree_dir_candidates(top_path)
            orphans.extend(self._orphan_entries(candidates, known_paths))
        return orphans

    @staticmethod
    def _worktree_dir_candidates(top_path: str) -> list[str]:
        candidates = [top_path]
        for sub in sorted(os.listdir(top_path)):
            sub_path = os.path.join(top_path, sub)
            if os.path.isdir(sub_path):
                candidates.append(sub_path)
        return candidates

    @staticmethod
    def _orphan_entries(
        candidates: list[str], known_paths: set[str]
    ) -> list[dict[str, str]]:
        orphans: list[dict[str, str]] = []
        for cand in candidates:
            if os.path.abspath(cand) in known_paths:
                continue
            if os.path.exists(os.path.join(cand, ".git")):
                orphans.append({"path": cand, "reason": "untracked worktree directory"})
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
                bucket, entry = self._prune_one_merged(w, canonical, path, base)
                (pruned if bucket == "pruned" else kept).append(entry)
            elif cls == "dangling" and canonical:
                pruned.append(self._prune_one_dangling(w, canonical, path))
            else:
                kept.append(self._not_prunable_entry(w))
        return pruned, kept

    def _prune_one_merged(
        self, w: dict[str, Any], canonical: str, path: str, base: str
    ) -> tuple[str, dict[str, Any]]:
        """Prune (or explain keeping) one ``merged``-classified worktree.

        Returns ``("pruned"|"kept", entry)``. Every early exit below is a
        refusal-with-reason, re-checked under the lane lease held by
        :func:`prune_guard.guarded_worktree_prune` so a classification that
        went stale since the audit scan is caught rather than acted on.
        """
        entry: dict[str, Any] = {
            "repo": w["repo"],
            "branch": w["branch"],
            "path": path,
            "class": w["class"],
        }
        if prune_guard.worktree_is_locked(path):
            return "kept", {**entry, "reason": "worktree is locked (git)"}
        with prune_guard.guarded_worktree_prune(
            path, operation="prune merged worktree"
        ) as held:
            if held is not None:
                return "kept", {**entry, "reason": held}
            fresh = self._branch_state(path, base)
            if fresh["dirty"] or not fresh["merged"]:
                return "kept", {
                    **entry,
                    "reason": (
                        "state changed since the audit scan "
                        f"(dirty={fresh['dirty']}, "
                        f"ahead={fresh['ahead']}, "
                        f"behind={fresh['behind']}) - a lane is "
                        "still working here"
                    ),
                }
            self._finish_merged_prune(entry, canonical, path, w, base)
        return ("pruned" if entry.get("ok") else "kept"), entry

    def _finish_merged_prune(
        self,
        entry: dict[str, Any],
        canonical: str,
        path: str,
        w: dict[str, Any],
        base: str,
    ) -> None:
        res = self._run(f"git worktree remove {shlex.quote(path)}", canonical)
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

    def _prune_one_dangling(
        self, w: dict[str, Any], canonical: str, path: str
    ) -> dict[str, Any]:
        self._run("git worktree prune", canonical, quiet=True)
        return {
            "repo": w["repo"],
            "branch": w["branch"],
            "path": path,
            "class": w["class"],
            "ok": True,
        }

    @staticmethod
    def _not_prunable_entry(w: dict[str, Any]) -> dict[str, Any]:
        return {
            "repo": w["repo"],
            "branch": w["branch"],
            "class": w["class"],
            "reason": "not prunable (active/stale)",
        }

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
        worktrees = [self._worktree_audit_row(w, base, stale_days) for w in linked]

        repos_report = self._repo_states(repo=repo, base=base)
        self._apply_base_unpushed(worktrees, repos_report)

        known_paths = {os.path.abspath(str(w.get("path", ""))) for w in linked}
        orphans = self._orphan_dirs(known_paths)

        do_not_disturb, review, safe = self._bucket_worktrees(worktrees)
        summary = self._audit_summary(
            worktrees, do_not_disturb, review, orphans, repos_report
        )
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

    def _worktree_audit_row(
        self, w: dict[str, Any], base: str, stale_days: int
    ) -> dict[str, Any]:
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
        return {
            "repo": w.get("repo"),
            "branch": branch,
            "path": path,
            "head": w.get("head"),
            **state,
            "class": self._classify(state, branch, exists, stale_days),
        }

    @staticmethod
    def _bucket_worktrees(
        worktrees: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        do_not_disturb = [w for w in worktrees if w["class"] == "active"]
        review = [w for w in worktrees if w["class"] == "stale"]
        safe = [w for w in worktrees if w["class"] in ("merged", "dangling")]
        return do_not_disturb, review, safe

    @staticmethod
    def _audit_summary(
        worktrees: list[dict[str, Any]],
        do_not_disturb: list[dict[str, Any]],
        review: list[dict[str, Any]],
        orphans: list[dict[str, Any]],
        repos_report: list[dict[str, Any]],
    ) -> dict[str, Any]:
        return {
            "worktrees": len(worktrees),
            "merged": sum(w["class"] == "merged" for w in worktrees),
            "active": len(do_not_disturb),
            "stale": len(review),
            "dangling": sum(w["class"] == "dangling" for w in worktrees),
            "orphans": len(orphans),
            "unpushed_repos": sum(r["class"] == "unpushed" for r in repos_report),
        }

    @staticmethod
    def _apply_base_unpushed(
        worktrees: list[dict[str, Any]], repos_report: list[dict[str, Any]]
    ) -> None:
        base_unpushed = {r["repo"]: r.get("base_unpushed", False) for r in repos_report}
        for w in worktrees:
            w["base_unpushed"] = base_unpushed.get(w["repo"], False)
