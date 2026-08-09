"""A **generic task/reservation ledger** — the mechanism behind every serialized,
deduplicated, or capacity-bounded operation repository-manager arbitrates
(CONCEPT:RM-TASK-LEDGER). :mod:`repository_manager.merge_queue` and
:mod:`repository_manager.build_queue` are two **consumers** of this one core,
not two reimplementations of it.

**Why this module exists (D-ORC-46).** Four different mechanisms in this
ecosystem each express the same idea — "only N callers of THIS kind of
operation, at THIS scope, at once" — and each was built independently, after a
real incident:

============================  ==================================================
Class                          Incident that forced a fix
============================  ==================================================
``uv sync``                    A concurrent sync deleted ``.venv/bin/pytest``
                                out from under a running suite (D-W2T-3).
``pre-commit --all-files``     ``staged_files_only.py`` writes unstaged work to
                                a patch, checks it away, restores it after — a
                                crash mid-window loses it (D-OB-12).
``cargo build``                A shared ``CARGO_TARGET_DIR`` both serialises
                                AND corrupts concurrent builds (phantom
                                ``E0599``).
full ``pytest`` suite          ~28 concurrent runs corrupted a shared basetemp,
                                nearly causing a false regression call.
============================  ==================================================

Each got its own bespoke fix: a hand-rolled flock pool
(:func:`agent_utilities...uv_workspace._dependency_sync_slot`), a new
``PRE_COMMIT_HOME`` partition, a ``PARTITION``-class cargo target dir, a
``PARTITION``-class pytest basetemp. None of those fixes is wrong — but they
are four mechanisms for one idea, and the next incident gets a fifth unless the
idea itself is named. This module names it: an :class:`ExecutionClass` is a
declared *(scope, policy)* pair, :func:`acquire` is the one place that turns a
class into an actual reservation, and :class:`Task`/:class:`TaskStore` are the
generic append-only, fold-by-``recorded_at`` record model every queue-shaped
consumer needs (dedup, honest state transitions, no invented defaults).

**Scope × policy, not "builds".** A resource is contended at one of three
scopes:

* ``GLOBAL``  — host-wide (the shared dependency-sync pool).
* ``REPO``    — one repository, every worktree of it (a cargo build, a merge
  drain).
* ``WORKTREE``— one lane's own tree (pre-commit's unstash window; nobody else
  could touch this tree anyway).

and arbitrated under one of three policies:

* ``EXCLUSIVE`` — one holder at a time. Wraps
  :func:`agent_utilities.governance.lanes.hold_lease` directly; this module
  does not reimplement leasing, it names when to reach for it.
* ``POOL``      — up to *N* concurrent holders. *N* identical requests should
  serialise (they would step on the same target dir); *N* **different**
  requests must run in parallel up to the cap — a single 40-minute build
  starving every other agent is worse than the disk waste it replaces.
* ``PARTITION`` — no shared exclusion needed at all; each caller gets its own
  instance (mirrors ``partitioned_paths()``). :func:`acquire` still returns a
  :class:`Reservation` for this policy so callers have one uniform API, but it
  never blocks and never touches a lease file.

**Single-writer safety, stated once so nobody "optimises" it away.**
:class:`ExecutionClass` with ``policy=POOL`` still serialises **within** a
slot — corruption in a shared cargo target dir comes from *concurrent
writers*, not from reuse over time. A pool of N slots is N single-writer
partitions, each one safe for exactly the reason a lone ``PARTITION`` dir is
safe. If a future change lets two POOL holders share one slot "to save
memory", it has silently reintroduced the exact concurrent-writer corruption
the cargo ``PARTITION`` class exists to prevent. **Do not do that.**

**Co-location is not optional — same rule as everything else in this module:
refuse rather than assume.** ``fcntl.flock`` (which :func:`hold_lease` uses
under the hood) arbitrates processes sharing one kernel and one mount; it does
**not** arbitrate across nodes, and two facts here make that concrete rather
than theoretical: ``repository-manager-mcp`` is pinned to one node via
``nodeSelector``+``hostPath`` (so *it* is a valid same-node arbiter for
everything routed through it), while ``/home/apps`` — where every lease file
in this ecosystem lives — is exported **over NFS** to several other nodes, and
a caller running there sees the *same path* but not the same locking
guarantee (NFS advisory-lock support is inconsistent, and this workspace's own
merge-queue design already rejected NFS for exactly this reason). A caller
that takes a filesystem lock from a non-co-located node gets **false safety**:
the call succeeds, the exclusion does not exist, and the failure is silent —
the one disposition this codebase keeps finding and paying for. So
:func:`acquire` requires an explicit ``colocated=True`` before it will ever
touch a lease file; the default (``colocated=None``, "unknown") is a refusal
naming exactly what to do instead — route the operation through the
repository-manager MCP server, which — because it is itself the pinned,
single, same-node process — can assert co-location honestly. The CLI defaults
to ``colocated=False`` (an operator must prove pinning with
``--same-node``/``RM_TASK_LEDGER_COLOCATED=1``); the MCP tool surface always
passes ``colocated=True`` because being inside that process **is** the proof.

**What this does not close (state it, do not pretend otherwise — D-CP-8).**
Arbitration here is advisory: it governs callers that go through
:func:`acquire`. A bare ``cargo build`` or ``uv sync`` run directly bypasses
every mechanism in this module, same as it always could. Nothing below claims
to be enforced.
"""

from __future__ import annotations

import fcntl
import inspect
import json
import os
import re
import socket
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from pathlib import Path
from typing import Any

from agent_utilities.governance.lanes import (
    DEFAULT_LEASE_TTL_SECONDS,
    FragmentStore,
    LaneArbitrationError,
    LaneScope,
    LeaseUnavailable,
    hold_lease,
    lane_scope,
    lease_status,
    partitioned_paths,
    workspace_arbitration_dir,
)

_LEASE_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


class TaskQueueError(LaneArbitrationError):
    """A task-ledger operation refused, carrying the reason a caller must act on."""


class ColocationRequired(TaskQueueError):
    """``acquire()`` was asked for a lease-backed reservation without proof of
    same-node execution. The fix is never "pass colocated=True and hope" —
    it is either genuine same-node pinning or routing through the MCP server.
    """


# ---------------------------------------------------------------------------
# Task — the generic record every queue-shaped consumer folds into a view
# ---------------------------------------------------------------------------
QUEUED = "queued"
RUNNING = "running"
DONE = "done"
FAILED = "failed"
REJECTED = "rejected"
WITHDRAWN = "withdrawn"

#: A task in one of these states will never transition again. Reused by every
#: consumer's own report/status verb so "still live" is one definition.
TERMINAL_STATES = frozenset({DONE, FAILED, REJECTED, WITHDRAWN})

TASK_QUEUE_DIRNAME = "task-queue"


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Task:
    """One unit of work offered to a per-*(repo, kind)* queue.

    Deliberately thin: ``kind`` namespaces the queue (``"merge-candidate"``,
    ``"build"``, ...) and ``payload`` carries whatever that kind needs.
    :mod:`merge_queue`'s ``Candidate`` and :mod:`build_queue`'s ``BuildTask``
    are both this record with a typed view over ``payload`` — see each
    module's ``from_record``/``to_record`` pair.
    """

    id: str
    kind: str
    lane: str = ""
    repo: str = ""
    state: str = QUEUED
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)
    #: When this task was first offered — fixed across every later transition.
    enqueued_at: str = ""
    #: When THIS RECORD was appended — distinct from ``enqueued_at``. The fold
    #: resolves same-``id`` duplicates on this field, never on fragment order
    #: (see :func:`resolve_latest_record` for why that distinction is load-
    #: bearing, not cosmetic).
    recorded_at: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Task:
        return cls(
            id=str(record.get("id", "")),
            kind=str(record.get("kind", "")),
            lane=str(record.get("lane", "")),
            repo=str(record.get("repo", "")),
            state=str(record.get("state", QUEUED)),
            reason=str(record.get("reason", "")),
            payload=dict(record.get("payload") or {}),
            enqueued_at=str(record.get("enqueued_at", "")),
            recorded_at=str(record.get("recorded_at", "")),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "lane": self.lane,
            "repo": self.repo,
            "state": self.state,
            "reason": self.reason,
            "payload": self.payload,
            "enqueued_at": self.enqueued_at,
            "recorded_at": self.recorded_at,
        }


def resolve_latest_record(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the chronologically LATEST record in *group* by ``recorded_at``.

    Ported verbatim from ``merge_queue._resolve_latest_candidate_record``
    (D-CVG-9/D-F6-1): :meth:`FragmentStore.fold` groups records by
    alphabetically-sorted lane name and would otherwise take ``group[-1]`` —
    which prefers whichever LANE NAME sorts last, a property unrelated to
    which record was written most recently. A candidate enqueued by
    ``lane-foo`` and landed by ``canonical`` folded to the OLDER queued
    record purely because ``"canonical" < "lane-foo"``, so a genuinely landed
    candidate reported ``queued`` forever and a stale record could revive a
    dead one. ISO-8601 timestamps sort correctly as plain strings. Falls back
    to ``group[-1]`` ONLY when every record in the group lacks
    ``recorded_at`` (a fragment written before the field existed), so old
    stores degrade rather than needing a migration.
    """
    with_timestamp = [r for r in group if str(r.get("recorded_at", "")).strip()]
    if not with_timestamp:
        return group[-1]
    return max(with_timestamp, key=lambda r: str(r.get("recorded_at", "")))


def task_store(kind: str, path: Path | str | None = None) -> FragmentStore:
    """The append-only record set backing one repository's ``kind`` queue.

    ``lane_scope(path).arbitration_dir`` is that repository's own shared
    ``--git-common-dir`` — per-repo AND per-kind by construction (``merge-
    candidate`` and ``build`` records never share a store), identical from
    every worktree, and untouched by any checkout/reset/merge.
    """
    scope = lane_scope(path)
    return FragmentStore(
        root=scope.arbitration_dir / TASK_QUEUE_DIRNAME / kind, key="id"
    )


def enqueue_task(
    task_id: str,
    kind: str,
    *,
    lane: str = "",
    repo: str = "",
    payload: dict[str, Any] | None = None,
    path: Path | str | None = None,
) -> Task:
    """Append a new ``queued`` record. Cheap and non-blocking — nothing here
    is verified; verification happens once, at dispatch time, by the consumer.
    """
    scope = lane_scope(path)
    now = _now()
    task = Task(
        id=task_id,
        kind=kind,
        lane=lane or scope.lane,
        repo=repo or scope.main_tree.name,
        state=QUEUED,
        payload=dict(payload or {}),
        enqueued_at=now,
        recorded_at=now,
    )
    task_store(kind, scope.tree).append(task.to_record(), lane=task.lane)
    return task


def record_state(
    task: Task, state: str, reason: str, *, path: Path | str | None = None
) -> Task:
    """Supersede *task*'s record with a new state — a NEW append, never an
    edit. The fold collapses same-``id`` records to the chronologically
    latest, so a terminal record correctly supersedes the queued one even
    when written from a different lane (e.g. a driver process recording
    ``done`` for a task another lane enqueued).
    """
    scope = lane_scope(path)
    updated = replace(task, state=state, reason=reason, recorded_at=_now())
    task_store(task.kind, scope.tree).append(updated.to_record(), lane=scope.lane)
    return updated


def all_tasks(kind: str, path: Path | str | None = None) -> list[Task]:
    return [
        Task.from_record(r)
        for r in task_store(kind, path).fold(resolve=resolve_latest_record)
    ]


def queued_tasks(kind: str, path: Path | str | None = None) -> list[Task]:
    """Every still-pending task of this kind, oldest first (FIFO, fair by
    construction).
    """
    pending = [t for t in all_tasks(kind, path) if t.state == QUEUED]
    return sorted(pending, key=lambda t: (t.enqueued_at, t.id))


def find_task(kind: str, task_id: str, path: Path | str | None = None) -> Task | None:
    for task in all_tasks(kind, path):
        if task.id == task_id:
            return task
    return None


def withdraw_task(
    kind: str, task_id: str, *, reason: str = "", path: Path | str | None = None
) -> Task:
    task = find_task(kind, task_id, path)
    if task is None:
        raise TaskQueueError(f"{task_id!r} is not in the {kind!r} queue")
    return record_state(task, WITHDRAWN, reason, path=path)


# ---------------------------------------------------------------------------
# ExecutionClass — the declared (scope, policy) pair; the whole "no
# opinionation" seam, mirroring GateSpec in merge_queue.py
# ---------------------------------------------------------------------------
class Scope(StrEnum):
    """Contention scope of an :class:`ExecutionClass`."""

    GLOBAL = "global"
    REPO = "repo"
    WORKTREE = "worktree"


class Policy(StrEnum):
    """How concurrent requests for one :class:`ExecutionClass` are arbitrated."""

    EXCLUSIVE = "exclusive"
    POOL = "pool"
    PARTITION = "partition"


@dataclass(frozen=True)
class ExecutionClass:
    """One declared kind of contended operation.

    The ledger never invents a class for an unrecognised name — see
    :func:`acquire` — the same "absence is a refusal, not a default" rule
    :mod:`merge_queue` applies to a missing ``.mergequeue.yaml``.
    """

    name: str
    scope: Scope = Scope.REPO
    policy: Policy = Policy.EXCLUSIVE
    #: Only meaningful for ``POOL``.
    pool_size: int = 1
    ttl_seconds: int = DEFAULT_LEASE_TTL_SECONDS
    #: Name in the weighted scheduler's versioned profile registry.  This is
    #: metadata for the transitional local adapter; it does not make this
    #: filesystem lease a distributed capacity authority.
    resource_profile: str = ""
    #: Optional compatibility metadata used when explaining a legacy acquire.
    concurrency_key: str = ""
    cpu_weight: int = 0
    memory_mib: int = 0
    disk_mib: int = 0
    process_slots: int = 0


#: The declared execution classes this ledger arbitrates. Adding a resource is
#: a row here, mirroring ``agent_utilities/governance/lane_resources.yaml``'s
#: own convention for file/tree-level PARTITION resources — this registry is
#: its sibling for orchestration-level *operations* (things you RUN, not
#: paths you own), so it lives in repository-manager rather than duplicating
#: that file.
EXECUTION_CLASSES: dict[str, ExecutionClass] = {
    # The build broker's own build step — DELIVERABLE 2. POOL, not EXCLUSIVE:
    # two DIFFERENT builds must run in parallel up to the heavy-lane cap, or
    # one 40-minute cargo build starves every other agent (worse than the
    # 21.7 GB of duplicate target dirs it replaces). Identical builds still
    # serialise because they contend for the same cache key at the build_queue
    # layer, not because this class forces it.
    "build": ExecutionClass(
        "build",
        scope=Scope.REPO,
        policy=Policy.POOL,
        pool_size=4,
        resource_profile="rust-build",
        concurrency_key="rust-build",
        cpu_weight=8,
        memory_mib=16_384,
        disk_mib=16_384,
        process_slots=2,
    ),
    # `uv sync` — D-W2T-3: a concurrent sync deleted `.venv/bin/pytest` out
    # from under a running suite. Host-wide (GLOBAL): the contended resource
    # is the shared dependency cache/lockfile, not any one repo.
    "uv-sync": ExecutionClass(
        "uv-sync",
        scope=Scope.GLOBAL,
        policy=Policy.POOL,
        pool_size=4,
        resource_profile="light-check",
        concurrency_key="uv-sync",
        cpu_weight=1,
        memory_mib=512,
        disk_mib=512,
        process_slots=1,
    ),
    # `pre-commit run --all-files` driven centrally — D-OB-12/D-ORC-37: the
    # staged-files-only patch/restore window is a data-loss hazard if two
    # drivers ever touch the same tree at once. WORKTREE-scoped and
    # EXCLUSIVE: nobody but this lane could touch this tree anyway, but a
    # *driver* (e.g. this ledger's own future pre-commit consumer) still
    # must not double-enter the window.
    "pre-commit": ExecutionClass(
        "pre-commit",
        scope=Scope.WORKTREE,
        policy=Policy.EXCLUSIVE,
        resource_profile="pre-commit",
        concurrency_key="pre-commit",
        cpu_weight=4,
        memory_mib=2_048,
        disk_mib=1_024,
        process_slots=2,
    ),
    # A repository's merge drain. EXCLUSIVE, REPO-scoped — this is the SAME
    # resource merge_queue.py's own MERGE_LEASE names; declared here too so
    # `rm_task status` can report it uniformly alongside build/uv-sync.
    "merge-drain": ExecutionClass(
        "merge-drain",
        scope=Scope.REPO,
        policy=Policy.EXCLUSIVE,
        resource_profile="merge-drain",
        concurrency_key="merge-drain",
        cpu_weight=2,
        memory_mib=1_024,
        disk_mib=512,
        process_slots=1,
    ),
    # These names are declared for consumers migrating to the common weighted
    # scheduler.  ``acquire`` remains only a same-node compatibility adapter;
    # distributed admission uses ResourceScheduler + WorkItem CAS instead.
    "light-check": ExecutionClass(
        "light-check",
        scope=Scope.GLOBAL,
        policy=Policy.POOL,
        pool_size=16,
        resource_profile="light-check",
        concurrency_key="light-check",
        cpu_weight=1,
        memory_mib=256,
        disk_mib=256,
        process_slots=1,
    ),
    "frontend-build": ExecutionClass(
        "frontend-build",
        scope=Scope.GLOBAL,
        policy=Policy.EXCLUSIVE,
        resource_profile="frontend-build",
        concurrency_key="frontend-build",
        cpu_weight=8,
        memory_mib=8_192,
        disk_mib=4_096,
        process_slots=1,
    ),
    "rust-build": ExecutionClass(
        "rust-build",
        scope=Scope.GLOBAL,
        policy=Policy.POOL,
        pool_size=2,
        resource_profile="rust-build",
        concurrency_key="rust-build",
        cpu_weight=8,
        memory_mib=16_384,
        disk_mib=16_384,
        process_slots=2,
    ),
    "workspace-release": ExecutionClass(
        "workspace-release",
        scope=Scope.GLOBAL,
        policy=Policy.EXCLUSIVE,
        resource_profile="workspace-release",
        concurrency_key="workspace-release",
        cpu_weight=4,
        memory_mib=4_096,
        disk_mib=2_048,
        process_slots=2,
    ),
}


def register_execution_class(execution_class: ExecutionClass) -> None:
    """Declare a new class (or replace one) — the extension point a consumer
    outside this module uses instead of hand-rolling its own lease.
    """
    EXECUTION_CLASSES[execution_class.name] = execution_class


def execution_class_profile(name: str) -> str:
    """Return the weighted profile for a legacy execution class.

    This helper is intentionally descriptive only.  Callers still need to
    submit a WorkItem-backed :class:`resource_scheduler.AdmissionRequest` to
    obtain distributed capacity; ``acquire`` cannot be promoted into that
    authority merely by looking up this value.
    """

    execution_class = _resolve_class(name)
    return execution_class.resource_profile or execution_class.name


def _resolve_class(name: str) -> ExecutionClass:
    execution_class = EXECUTION_CLASSES.get(name)
    if execution_class is None:
        raise TaskQueueError(
            f"unknown execution class {name!r}; declared: {sorted(EXECUTION_CLASSES)}. "
            "A class must be declared (register_execution_class) before it can be "
            "acquired — this ledger never invents a policy for an unrecognised name."
        )
    return execution_class


# ---------------------------------------------------------------------------
# Reservation — what acquire() hands back
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Reservation:
    """A held slot of an :class:`ExecutionClass`."""

    execution_class: str
    scope: Scope
    policy: Policy
    #: The lease name actually held (``None`` for PARTITION, which holds none).
    lease_name: str | None
    #: Which pool slot (``POOL`` only; ``None`` otherwise).
    slot: int | None
    #: PARTITION policy's private directory for this lane (``None`` otherwise).
    partition_dir: Path | None
    owner: dict[str, Any]
    acquired_at: str


def owner_identity(*, fleet: str = "", session: str = "") -> dict[str, Any]:
    """The identity recorded against a reservation — who to blame/wake if it
    wedges. ``fleet``/``session`` distinguish Claude/Codex/human/vLLM callers
    and their lane/session id; ``host``/``pid`` are always this process's own.
    """
    return {
        "fleet": fleet or os.environ.get("RM_TASK_LEDGER_FLEET", "unknown"),
        "session": session or os.environ.get("RM_TASK_LEDGER_SESSION", ""),
        "host": socket.gethostname(),
        "pid": os.getpid(),
    }


def _lease_name_for(
    execution_class: ExecutionClass, scope: LaneScope, slot: int | None
) -> str:
    base = (
        execution_class.name
        if execution_class.scope != Scope.GLOBAL
        else f"global-{execution_class.name}"
    )
    return base if slot is None else f"{base}-slot-{slot}"


#: D-CDX-14 lands as two independently merged changes across two
#: repositories: agent-utilities' ``hold_lease`` gains an ``owner`` kwarg,
#: and this module starts passing it. Detect support once rather than
#: hard-depending on merge order — an older ``agent-utilities`` still works,
#: it just does not persist ``owner`` into the lease record yet.
_HOLD_LEASE_ACCEPTS_OWNER = "owner" in inspect.signature(hold_lease).parameters


def _hold_repo_lease(
    lease_name: str,
    *,
    operation: str,
    ttl_seconds: int,
    path: Path,
    owner: dict[str, Any] | None,
):
    if _HOLD_LEASE_ACCEPTS_OWNER:
        return hold_lease(
            lease_name,
            operation=operation,
            ttl_seconds=ttl_seconds,
            path=path,
            owner=owner,
        )
    return hold_lease(
        lease_name, operation=operation, ttl_seconds=ttl_seconds, path=path
    )


def _global_lease_dir() -> Path:
    """Host-wide lease directory for GLOBAL-scope execution classes (D-CDX-12).

    ``ExecutionClass(scope=Scope.GLOBAL)`` declares a resource contended by
    every worktree of every repository on the host (the shared dependency
    cache, in the ``"uv-sync"`` case) — but
    ``agent_utilities.governance.lanes.hold_lease(path=...)`` only escapes its
    default PER-REPOSITORY lease directory (``lane_scope(path).arbitration_dir``,
    that repo's own ``--git-common-dir``) for a lease NAME registered
    ``scope: workspace`` in that package's ``lane_resources.yaml``. This
    ledger's GLOBAL lease names are generated dynamically
    (``global-<class>``, ``global-<class>-slot-<N>`` — see
    :func:`_lease_name_for`) and were never registered there, so every
    ``acquire()`` call for a GLOBAL class silently fell through to the
    repo-scoped branch: two ``uv-sync`` acquisitions in two DIFFERENT
    repositories got two DIFFERENT lease files under two different
    ``--git-common-dir``s and could both hold "slot 0" of the SAME declared
    host-wide pool at once — the exact concurrent-``uv-sync`` event (D-W2T-3)
    this class exists to prevent, with no cross-repository regression test
    proving the barrier actually held.

    Rather than hand-registering one YAML row per dynamically generated name
    (silently wrong again the next time ``pool_size`` changes or a new
    GLOBAL class is declared), GLOBAL leases are stored directly under
    agent-utilities' own host-wide arbitration root
    (:func:`agent_utilities.governance.lanes.workspace_arbitration_dir`) — the
    SAME directory the ``dependency-lock`` resource (a genuinely
    ``scope: workspace`` row) already uses — bypassing the name-classification
    lookup entirely for this always-host-wide case.
    """
    path = workspace_arbitration_dir() / "leases"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _global_holder_alive(holder: dict[str, Any]) -> bool:
    """Same liveness rule as ``agent_utilities.governance.lanes._holder_is_live``
    (unexpired, and — for a same-host holder — its process still running);
    duplicated locally because that check is private to a module whose
    directory-selection logic this function deliberately bypasses.
    """
    expires = holder.get("expires_at")
    if expires:
        try:
            if datetime.fromisoformat(str(expires)) < datetime.now(UTC):
                return False
        except ValueError:
            return False
    if holder.get("host") != socket.gethostname():
        # A different host: expiry is the only evidence available, and it
        # says live. (Same honesty limit as every lease in this ecosystem —
        # see the module docstring's "Co-location is not optional" section.)
        return True
    pid = holder.get("pid")
    if not isinstance(pid, int):
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _hold_global_lease(
    name: str, *, operation: str, ttl_seconds: int, owner: dict[str, Any] | None = None
) -> Iterator[dict[str, Any]]:
    """Same holder/expiry/reclaim contract as ``lanes.hold_lease`` — raise
    :class:`LeaseUnavailable` rather than block; a dead holder's lease is
    reclaimed automatically — but stored unconditionally under
    :func:`_global_lease_dir`, independent of the caller's own repository
    (D-CDX-12). Threads ``owner`` (D-CDX-14) into the record exactly like
    ``lanes.hold_lease`` does.
    """
    lease_file = _global_lease_dir() / f"{_LEASE_SAFE_RE.sub('-', name)}.lease"
    mutex_file = lease_file.with_suffix(".mutex")
    now = datetime.now(UTC)
    record: dict[str, Any] = {
        "name": name,
        "operation": operation,
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "acquired_at": now.isoformat(),
        "expires_at": (now + timedelta(seconds=ttl_seconds)).isoformat(),
    }
    if owner:
        record["owner"] = dict(owner)
    fd = os.open(str(mutex_file), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if lease_file.exists():
                holder = json.loads(lease_file.read_text(encoding="utf-8"))
                if _global_holder_alive(holder):
                    raise LeaseUnavailable(name, holder)
            lease_file.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)
    try:
        yield record
    finally:
        fd = os.open(str(mutex_file), os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            try:
                if lease_file.exists():
                    current = json.loads(lease_file.read_text(encoding="utf-8"))
                    if (
                        current.get("pid") == record["pid"]
                        and current.get("acquired_at") == record["acquired_at"]
                    ):
                        lease_file.unlink()
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def _global_lease_status(name: str) -> dict[str, Any] | None:
    """Read-side counterpart to :func:`_hold_global_lease` — the live holder
    of a GLOBAL-scope lease, or ``None`` when free. Mirrors
    ``lanes.lease_status`` but reads from :func:`_global_lease_dir`.
    """
    lease_file = _global_lease_dir() / f"{_LEASE_SAFE_RE.sub('-', name)}.lease"
    mutex_file = lease_file.with_suffix(".mutex")
    fd = os.open(str(mutex_file), os.O_CREAT | os.O_WRONLY, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        try:
            if not lease_file.exists():
                return None
            holder = json.loads(lease_file.read_text(encoding="utf-8"))
            if _global_holder_alive(holder):
                return holder
            lease_file.unlink()
            return None
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
    finally:
        os.close(fd)


@contextmanager
def acquire(
    name: str,
    *,
    operation: str,
    owner: dict[str, Any] | None = None,
    path: Path | str | None = None,
    colocated: bool | None = None,
    ttl_seconds: int | None = None,
) -> Iterator[Reservation]:
    """Acquire a :class:`Reservation` for the declared execution class *name*.

    Raises rather than blocks — same convention as
    :func:`agent_utilities.governance.lanes.hold_lease`: the caller is forced
    to make deferral explicit (retry, queue, or give up), never silently wait
    on a resource under contention.

    ``colocated`` is the co-location gate (see the module docstring's
    "Co-location is not optional" section):

    * ``True``  — caller has proof of same-node execution (it *is* the pinned
      repository-manager-mcp process, or an operator has pinned it and passed
      ``--same-node``). The lease-backed fast path runs.
    * ``False`` or ``None`` — no proof. For ``EXCLUSIVE``/``POOL`` policies
      this raises :class:`ColocationRequired` naming the MCP route instead of
      silently taking a lock whose exclusion may not actually hold over NFS.
      ``PARTITION`` policy is exempt — it holds no shared lease at all, so
      co-location is moot.
    """
    execution_class = _resolve_class(name)
    scope = lane_scope(path)
    owner = owner or owner_identity()
    ttl = ttl_seconds or execution_class.ttl_seconds

    if execution_class.policy == Policy.PARTITION:
        partition_dir = partitioned_paths(scope.tree).scratch_dir / "task-ledger" / name
        partition_dir.mkdir(parents=True, exist_ok=True)
        yield Reservation(
            execution_class=name,
            scope=execution_class.scope,
            policy=execution_class.policy,
            lease_name=None,
            slot=None,
            partition_dir=partition_dir,
            owner=owner,
            acquired_at=_now(),
        )
        return

    if colocated is not True:
        raise ColocationRequired(
            f"execution class {name!r} is lease-backed ({execution_class.policy.value}) "
            "and this caller has no proof of same-node execution. `fcntl.flock` "
            "arbitrates one kernel/mount, not the workspace, and lease files live "
            "under /home/apps which is NFS-exported to other nodes. Either pin this "
            "operation to the same node as the lease holder and pass colocated=True "
            "(operators: --same-node / RM_TASK_LEDGER_COLOCATED=1), or route the "
            "request through the repository-manager MCP server, which IS the "
            "same-node arbiter because it is pinned there."
        )

    # GLOBAL-scope leases route through this ledger's own host-wide storage
    # (D-CDX-12) instead of `lanes.hold_lease`'s per-repository default,
    # which only escapes to the host-wide root for a NAME registered
    # `scope: workspace` in agent-utilities' own lane_resources.yaml — a
    # registry this ledger's dynamically-generated GLOBAL names never
    # satisfy. See :func:`_global_lease_dir` for the full incident.
    is_global = execution_class.scope == Scope.GLOBAL

    if execution_class.policy == Policy.EXCLUSIVE:
        lease_name = _lease_name_for(execution_class, scope, None)
        cm = (
            _hold_global_lease(
                lease_name, operation=operation, ttl_seconds=ttl, owner=owner
            )
            if is_global
            else _hold_repo_lease(
                lease_name,
                operation=operation,
                ttl_seconds=ttl,
                path=scope.tree,
                owner=owner,
            )
        )
        with cm as record:
            yield Reservation(
                execution_class=name,
                scope=execution_class.scope,
                policy=execution_class.policy,
                lease_name=lease_name,
                slot=None,
                partition_dir=None,
                owner=owner,
                acquired_at=record["acquired_at"],
            )
        return

    # POOL: try each slot in turn; refuse (never block) when every slot is held.
    busy: list[dict[str, Any]] = []
    for slot in range(execution_class.pool_size):
        lease_name = _lease_name_for(execution_class, scope, slot)
        try:
            cm = (
                _hold_global_lease(
                    lease_name, operation=operation, ttl_seconds=ttl, owner=owner
                )
                if is_global
                else _hold_repo_lease(
                    lease_name,
                    operation=operation,
                    ttl_seconds=ttl,
                    path=scope.tree,
                    owner=owner,
                )
            )
            with cm as record:
                yield Reservation(
                    execution_class=name,
                    scope=execution_class.scope,
                    policy=execution_class.policy,
                    lease_name=lease_name,
                    slot=slot,
                    partition_dir=None,
                    owner=owner,
                    acquired_at=record["acquired_at"],
                )
            return
        except LeaseUnavailable as exc:
            busy.append({"slot": slot, "holder": exc.holder})
            continue
    raise TaskQueueError(
        f"execution class {name!r} has no free slot ({execution_class.pool_size} "
        f"busy): {json.dumps(busy, default=str)}"
    )


def class_status(name: str, path: Path | str | None = None) -> dict[str, Any]:
    """Every slot's live holder (or ``None``) for a declared execution class —
    the read side of :func:`acquire`, safe to call without holding anything.
    """
    execution_class = _resolve_class(name)
    scope = lane_scope(path)
    is_global = execution_class.scope == Scope.GLOBAL
    if execution_class.policy == Policy.PARTITION:
        return {
            "class": name,
            "policy": execution_class.policy.value,
            "partitioned": True,
        }
    if execution_class.policy == Policy.EXCLUSIVE:
        lease_name = _lease_name_for(execution_class, scope, None)
        holder = (
            _global_lease_status(lease_name)
            if is_global
            else lease_status(lease_name, scope.tree)
        )
        return {
            "class": name,
            "policy": execution_class.policy.value,
            "holder": holder,
        }
    slots = []
    for slot in range(execution_class.pool_size):
        lease_name = _lease_name_for(execution_class, scope, slot)
        holder = (
            _global_lease_status(lease_name)
            if is_global
            else lease_status(lease_name, scope.tree)
        )
        slots.append({"slot": slot, "holder": holder})
    return {"class": name, "policy": execution_class.policy.value, "slots": slots}
