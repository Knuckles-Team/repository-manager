"""A **repo-agnostic** serialized merge queue — the mechanism, with the gates
declared per repository (CONCEPT:RM-MERGE-QUEUE).

**Why this lives here and not in agent-utilities.** The queue's *mechanism* —
an append-only candidate store, differential gating against a base ref,
regenerate-on-land, a guarded prune, one serializing lease, optimistic batching
with bisection — has nothing to do with Python, pytest, ruff, or
``agent_utilities``. It is git discipline. The *gates* are the only
project-specific part, and they were the only part hard-coded. So
``agent_utilities.governance.merge_queue`` could serve exactly one repository,
and the proof is empirical: **epistemic-graph has no merge queue at all**, and
the engine lane had to hand-apply the discipline by memory (verify the *merged
tree* with ``cargo check --all-features``, fast-forward only, never merge in the
canonical checkout). repository-manager already owns
:class:`~repository_manager.worktree.WorktreeManager`, :mod:`canonical_guard`
and :mod:`prune_guard` — the machinery the au queue had to reimplement inline
when repository-manager turned out not to be importable (D-ORC-21). Only the
queue layer was mis-homed. This module is that layer, made generic.

**Mechanism vs gates.** The queue **must not know what a gate IS** — only how to
run one and compare its result against the base ref. A gate is a row in the
repository's own ``.mergequeue.yaml`` (see :class:`QueueConfig`): a command, a
tier, a timeout, and *how to compare* its output against the base
(:class:`GateSpec.compare`). ``agent-utilities`` declares pytest + ruff + its
contract scripts; ``epistemic-graph`` declares ``cargo check --all-features`` +
clippy; a docs repo could declare nothing but a link check. Adding a gate is a
config row, never a change to this module.

**The five hard-won behaviours ported verbatim from
``agent_utilities.governance.merge_queue``**, each because it was paid for once
already:

1. **Differential gating** (:func:`compute_gate_baseline`,
   :func:`run_gate`) — a gate result is judged against the SAME gate's result on
   the base ref, so only *new* failures block. ``main`` is legitimately red here;
   an absolute gate deadlocked the queue and stranded 19 branches. **A baseline
   that cannot be produced REFUSES the candidate** — it never degrades to
   allow-all, which is the one failure mode that silently turns a gate off.
2. **Fold by ``recorded_at``** (:func:`_resolve_latest_candidate_record`) —
   resolving cross-lane duplicate records by *fragment iteration order* prefers
   whichever lane name sorts last, which stranded fresh candidates and revived
   dead ones (D-CVG-9 / D-F6-1). Records carry their own write time and the
   chronologically latest wins.
3. **Regenerate-on-land** (:func:`_resolve_generated_file_conflict`) — a
   conflict confined to *declared generated files* is resolved by regenerating
   them from the already-merged tree, never by ``--theirs`` picking a side.
4. **Guarded prune** (:func:`prune_landed`) — merge-base re-checked at delete
   time, a ``refs/lane-backup/<branch>`` anchor written first, ``git branch -d``
   and **never** ``-D``, and any worktree holding uncommitted work is refused.
   Delegated to :class:`WorktreeManager` — which is where that guard already
   lives — rather than reimplemented.
5. **Honest degradation** — every refusal names its reason. There is no path
   here that reports success it did not verify.

**The canonical working tree is never a merge arena.** The whole merge is
computed as objects (``git merge-tree --write-tree`` → ``git commit-tree``),
verified in a throwaway detached worktree, and only then does the base move — by
**fast-forward only**, under BOTH ``lanes.guarded_tree_mutation`` (refuses a tree
holding another lane's uncommitted work) and
:func:`~repository_manager.canonical_guard.guarded_canonical_mutation` (refuses a
dirty canonical checkout and serializes against repository-manager's own
background sync). repository-manager has destroyed work on a canonical tree
before — a background ``git reset`` discarded ~20 minutes of a lane's work — so
making it the merge *driver* means holding both guards, not one.

**Pre-commit is a data-loss hazard when driven on someone else's behalf**
(D-ORC-37). ``pre_commit/staged_files_only.py`` writes your UNSTAGED changes to a
patch file, ``git checkout``s them away so hooks see only staged content, and
restores them in a ``finally:``. A crash inside that window leaves the work only
as an orphaned patch nobody replays. So :func:`refuse_precommit_on_dirty_tree`
is called before any gate whose command runs ``pre-commit``: **a central driver
never runs pre-commit against a tree holding uncommitted work.** A clean tree has
no patch-restore window at all, and every tree this queue gates is a throwaway
checkout of a committed object — clean by construction.

No new arbitration class is introduced. Serialization is the existing
``reconciliation-merge`` LEASE, the candidate store is a
:class:`~agent_utilities.governance.lanes.FragmentStore` in the repository's own
shared ``--git-common-dir``, and scratch/basetemp come from
``partitioned_paths()`` — all repo-scoped already, so two repositories' queues
are independent by construction rather than by convention.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shutil
import subprocess  # nosec B404 - fixed argv, never shell=True
import time
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from agent_utilities.governance.lanes import (
    FragmentStore,
    LaneArbitrationError,
    LaneScope,
    guarded_tree_mutation,
    hold_lease,
    lane_scope,
    partitioned_paths,
)

from repository_manager.candidate_generation import (
    CandidateGenerationError,
    CandidateSnapshot,
    GenerationRecord,
    candidate_identity,
    fold_candidate_records,
    fold_generation_records,
    snapshot_branch_candidate,
    timestamp_value,
)
from repository_manager.canonical_guard import guarded_canonical_mutation
from repository_manager.config_schema import (
    ConfigSchemaError,
    ResourceRequest,
    load_yaml_mapping,
    load_yaml_mapping_text,
    parse_merge_config,
    runtime_tier,
)
from repository_manager.development import RepositoryIdentity, TargetPolicy
from repository_manager.generation_coalescing import seal_generation, select_batches
from repository_manager.lane_record import repository_id_for
from repository_manager.test_commands import ensure_no_fail_fast

#: The per-repository gate declaration. Its ABSENCE is a refusal, not a default:
#: a queue that invents gates for a repository that declared none would be a gate
#: that checks nothing while reporting green — the exact shape this codebase keeps
#: producing (see the module docstring's point 1).
CONFIG_FILENAME = ".mergequeue.yaml"

#: The LEASE that serializes a drain. Deliberately the SAME resource name
#: ``agent_utilities.governance.merge_queue`` holds, so an au queue runner and a
#: repository-manager runner can never drain the same repository at once during
#: the migration. It is repo-scoped in ``lane_resources.yaml``, so each
#: repository gets its own instance for free.
MERGE_LEASE = "reconciliation-merge"

#: Where candidate fragments live inside the repository's shared, unversioned
#: arbitration dir (``<git-common-dir>/agent-lanes``). Identical from every
#: worktree of that repo, and untouched by any checkout/reset/merge.
QUEUE_DIRNAME = "merge-queue"
BASELINE_CACHE_DIRNAME = "gate-baseline-cache"

# RMDD-12 records share the existing queue store.  ``FragmentStore`` predates
# these records and groups by ``id``; the domain records keep their authoritative
# ``record_id`` and carry this one compatibility alias only at the queue seam.
CANDIDATE_SNAPSHOT_KIND = "candidate_snapshot"
GENERATION_RECORD_KIND = "generation"
_ADDITIVE_RECORD_KINDS = frozenset({CANDIDATE_SNAPSHOT_KIND, GENERATION_RECORD_KIND})
_MAX_SHADOW_EVIDENCE_PATHS = 128
_MAX_SHADOW_DETAIL_CHARS = 4096

QUEUED = "queued"
LANDED = "landed"
REJECTED = "rejected"
WITHDRAWN = "withdrawn"

#: Candidates merged into one trial commit and gated together. A failing batch
#: BISECTS, so one bad candidate costs ceil(log2(N)) extra gate runs rather than
#: rejecting N-1 innocent ones.
DEFAULT_BATCH_SIZE = 8

#: pytest exit codes under which the ``-rfE`` short summary can be trusted to
#: enumerate every failing id: 0 green, 1 tests failed, 5 nothing collected (an
#: honest empty). 2/3/4 (interrupted / internal / usage) enumerate nothing, so a
#: run exiting one of those is UNREADABLE — never "zero failures".
_PYTEST_READABLE_EXIT_CODES = frozenset({0, 1, 5})

_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


class MergeQueueError(LaneArbitrationError):
    """A merge-queue operation refused, carrying the reason a caller must act on."""


# ---------------------------------------------------------------------------
# git plumbing — an exit code is an ANSWER here, not a malfunction
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GitResult:
    code: int
    out: str
    err: str

    @property
    def ok(self) -> bool:
        return self.code == 0


def _run_git(
    args: list[str],
    cwd: Path,
    *,
    timeout: int = 300,
    env: dict[str, str] | None = None,
) -> GitResult:
    """Run git and hand back the exit code unmodified.

    Deliberately NOT routed through ``GitLike.git_action``: that surface reports
    ``status: success|failure`` and drops the code, and this module's whole
    object-level merge computation depends on distinguishing exit codes.
    ``git merge-tree`` answers "there are conflicts" with **exit 1**, and
    ``git branch -d`` answers "those commits would be orphaned" with **exit 1**.
    Collapsing either into "failure" would throw away the answer being asked for.
    ``GitLike``/:class:`WorktreeManager` is still the surface used for every
    worktree and prune operation, where the abstraction genuinely pays.
    """
    git_executable = shutil.which("git")
    if git_executable is None:
        raise MergeQueueError("git executable was not found on PATH")
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
        [git_executable, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=False,
        timeout=timeout,
        env=process_env,
    )
    return GitResult(proc.returncode, proc.stdout.strip(), proc.stderr.strip())


def _require_git(
    args: list[str], cwd: Path, *, env: dict[str, str] | None = None
) -> str:
    res = _run_git(args, cwd) if env is None else _run_git(args, cwd, env=env)
    if not res.ok:
        raise MergeQueueError(f"git {' '.join(args)} failed in {cwd}: {res.err}")
    return res.out


def _now() -> str:
    return datetime.now(UTC).isoformat()


# ---------------------------------------------------------------------------
# The per-repository gate declaration — this is the whole "no opinionation" seam
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class GateSpec:
    """One declared gate: a command, and how to compare its result to the base.

    The queue never inspects what the command *does*. ``cargo check
    --all-features``, ``pytest``, ``ruff check``, ``lychee --offline docs/`` and a
    hand-rolled shell script are all the same object to it.
    """

    #: Stable identifier — the baseline cache key and every report use it, so
    #: renaming a gate correctly invalidates its baseline.
    name: str
    #: argv, never a shell string. ``shell: true`` is deliberately not offered:
    #: a gate command is repo-declared data and must not be an injection surface.
    command: tuple[str, ...]
    #: ``fast`` runs inside the queue's lease; ``slow`` is declared for the
    #: post-merge suite and is NOT run here. Keeping slow gates declarable (and
    #: visibly skipped) is how the config stays the single description of what
    #: guards a repo — a gate too expensive for the queue is still a fact.
    tier: str = "fast"
    timeout: int = 300
    #: Ceiling for the SAME gate's run on the base ref. Separate and larger by
    #: default: the base run is no smaller than the merged run it is compared
    #: against, and it contends with other lanes differently (D-MW-10 — sharing
    #: one budget between two runs with different contention profiles starved
    #: one of them and started refusing landable branches at ~8 lanes).
    baseline_timeout: int = 0
    #: ``exit`` | ``lines`` | ``pytest-ids`` — see :func:`_compare_gate`.
    compare: str = "lines"
    #: Only run when a changed path matches one of these globs. Empty = always.
    when_changed: tuple[str, ...] = ()
    #: Only lines matching one of these regexes participate in a ``lines``
    #: comparison. Empty = every line. This is what makes ``lines`` usable for a
    #: chatty tool: ``cargo`` prints ``Compiling`` / ``Finished in 3.4s`` lines
    #: that differ on every run and would otherwise read as new violations on
    #: every candidate. Declaring ``keep_lines: ['^error', '^warning']`` compares
    #: the diagnostics and ignores the noise.
    keep_lines: tuple[str, ...] = ()
    #: Lines matching any of these are dropped AFTER ``keep_lines``.
    ignore_lines: tuple[str, ...] = ()
    #: What exceeding ``timeout`` on the MERGED tree means. ``fail`` (default) is
    #: the safe reading. ``defer`` reproduces the au queue's targeted-test
    #: contract: a gate that blows its ceiling is handed to the post-merge suite
    #: rather than held against the candidate, because a queue that hangs is a
    #: queue that gets bypassed. Exceeding the BASELINE budget is always a
    #: refusal regardless — see point 1 in the module docstring.
    on_timeout: str = "fail"
    #: Explicit v2 validation stage. ``tier`` remains the compatibility
    #: projection used by the current queue; staged consumers use this field.
    stage: str = "integration"
    baseline_mode: str = "differential"
    path_exclude: tuple[str, ...] = ()
    resource_class: str = "light-check"
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    artifact_dependencies: tuple[str, ...] = ()

    @property
    def effective_baseline_timeout(self) -> int:
        return self.baseline_timeout or self.timeout * 2

    @property
    def runs_precommit(self) -> bool:
        """Whether this gate shells out to ``pre-commit`` (D-ORC-37).

        Matched on the command, not declared by the config, so a repo cannot
        opt out of the dirty-tree refusal by forgetting to flag it.
        """
        return any(
            "pre-commit" in part or "pre_commit" in part for part in self.command
        )


@dataclass(frozen=True)
class QueueConfig:
    """A repository's whole merge-queue declaration, read from ``.mergequeue.yaml``."""

    base: str = "main"
    batch_size: int = DEFAULT_BATCH_SIZE
    gates: tuple[GateSpec, ...] = ()
    #: Purely-derived files whose conflicts are resolved by REGENERATING them
    #: from the merged tree rather than by picking a side.
    generated_files: frozenset[str] = frozenset()
    #: The commands that regenerate them, in order (order matters — a generator
    #: that reads another's output must run after it).
    regenerate: tuple[tuple[str, ...], ...] = ()
    #: Command whose stdout fingerprints the toolchain the gates run under. Folded
    #: into the baseline cache key so an IN-PLACE toolchain change (same paths,
    #: different versions, same base sha) busts the cache instead of silently
    #: serving a baseline computed against tools that no longer exist (D-MQD-1).
    #: When absent AND no ``.venv`` is discoverable, the environment is
    #: **unpinned** and the baseline cache is DISABLED rather than keyed on a
    #: fiction — see :func:`_environment_signature`.
    environment_signature: tuple[str, ...] = ()
    #: Where the config was read from — reported in every result so an operator
    #: can see WHICH declaration produced a verdict.
    source: str = ""
    schema_version: int = 2

    def fast_gates(self) -> tuple[GateSpec, ...]:
        return tuple(g for g in self.gates if g.tier == "fast")

    def slow_gates(self) -> tuple[GateSpec, ...]:
        return tuple(g for g in self.gates if g.tier != "fast")


def _as_argv(value: Any, *, where: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise MergeQueueError(
            f"{where}: a command must be a LIST of argv items, not a string "
            f"({value!r}). A shell string would make repo-declared config an "
            "injection surface; this queue never runs a gate through a shell."
        )
    if not isinstance(value, list | tuple) or not value:
        raise MergeQueueError(f"{where}: a command must be a non-empty list of strings")
    return tuple(str(v) for v in value)


def parse_config(data: dict[str, Any], *, source: str = "") -> QueueConfig:
    """Build a :class:`QueueConfig` from already-loaded YAML/JSON data.

    Separated from :func:`load_config` so the declaration can be validated
    without a filesystem — and so a bad config fails LOUDLY at parse time with
    the offending key named, rather than at gate time with a confusing traceback.
    """
    try:
        schema = parse_merge_config(data, source=source)
    except ConfigSchemaError as exc:
        raise MergeQueueError(str(exc)) from exc
    gates = tuple(
        GateSpec(
            name=gate.name,
            command=gate.command,
            tier=runtime_tier(gate.stage),
            timeout=gate.timeout,
            baseline_timeout=gate.baseline_timeout,
            compare=gate.compare,
            when_changed=gate.path_selection.include,
            keep_lines=gate.keep_lines,
            ignore_lines=gate.ignore_lines,
            on_timeout=gate.on_timeout,
            stage=gate.stage,
            baseline_mode=gate.baseline_mode,
            path_exclude=gate.path_selection.exclude,
            resource_class=gate.resource_class,
            resources=gate.resources,
            artifact_dependencies=gate.artifact_dependencies,
        )
        for gate in schema.gates
    )
    return QueueConfig(
        base=schema.base,
        batch_size=schema.batch_size,
        schema_version=schema.schema_version,
        gates=gates,
        generated_files=schema.generated.files,
        regenerate=schema.generated.regenerate,
        environment_signature=schema.environment_signature,
        source=source,
    )


def load_config(tree: Path | str) -> QueueConfig:
    """Read ``<tree>/.mergequeue.yaml``.

    Read from the **merged tree** at gate time, never from the base: a candidate
    that ADDS a gate has that gate enforced against itself in the same run, and a
    candidate that deletes one is visibly running one fewer check rather than
    silently passing (:func:`_dropped_gates` turns that into a refusal).

    A missing config is an error, not an empty default. "This repository declared
    no gates" and "this repository has no queue configured" must not be the same
    value, because the first is a deliberate choice and the second is a mistake
    that would land everything unchecked.
    """
    path = Path(tree) / CONFIG_FILENAME
    if not path.is_file():
        raise MergeQueueError(
            f"no {CONFIG_FILENAME} in {tree} — this repository has not declared its "
            "merge-queue gates. The queue refuses rather than inventing gates or "
            "defaulting to none; add the file (see "
            "repository_manager/mergequeue_presets/) and re-run."
        )
    try:
        data = load_yaml_mapping(str(path))
        return parse_config(data, source=str(path))
    except ConfigSchemaError as exc:
        raise MergeQueueError(str(exc)) from exc


def config_at_ref(repo: Path, ref: str) -> QueueConfig | None:
    """The config as it exists on *ref*, or ``None`` when the ref has none.

    Used only by :func:`_dropped_gates`. ``None`` genuinely means "the base
    declared no queue", which is different from "unreadable" — an unreadable
    config raises.
    """
    res = _run_git(["show", f"{ref}:{CONFIG_FILENAME}"], repo)
    if not res.ok:
        return None
    try:
        data = load_yaml_mapping_text(res.out, source=f"{ref}:{CONFIG_FILENAME}")
        return parse_config(data, source=f"{ref}:{CONFIG_FILENAME}")
    except ConfigSchemaError as exc:
        raise MergeQueueError(str(exc)) from exc


def _dropped_gates(merged: QueueConfig, base: QueueConfig | None) -> list[str]:
    """Fast-tier gate names the base declares that the merged tree does not.

    Deleting the check that guards an invariant is not a way to satisfy it. Ported
    from the au queue's ``contract_baseline`` dropped-script comparison, which
    exists because a candidate that removes a gate would otherwise land by having
    nothing left to fail.
    """
    if base is None:
        return []
    return sorted(
        {g.name for g in base.fast_gates()} - {g.name for g in merged.fast_gates()}
    )


# ---------------------------------------------------------------------------
# The queue — APPEND-ONLY fragments, one per lane, per repository
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    """One branch offered for landing, as its own lane recorded it."""

    branch: str
    lane: str
    repo: str = ""
    base: str = "main"
    worktree: str = ""
    enqueued_at: str = ""
    state: str = QUEUED
    reason: str = ""
    #: WHEN THIS RECORD was appended — distinct from ``enqueued_at``, which is
    #: the original enqueue time and stays fixed across every later transition.
    #: The fold resolves duplicates on this (see
    #: :func:`_resolve_latest_candidate_record`).
    recorded_at: str = ""

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> Candidate:
        return cls(
            branch=str(record.get("id", "")),
            lane=str(record.get("lane", "")),
            repo=str(record.get("repo", "")),
            base=str(record.get("base", "main")),
            worktree=str(record.get("worktree", "")),
            enqueued_at=str(record.get("enqueued_at", "")),
            state=str(record.get("state", QUEUED)),
            reason=str(record.get("reason", "")),
            recorded_at=str(record.get("recorded_at", "")),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "id": self.branch,
            "lane": self.lane,
            "repo": self.repo,
            "base": self.base,
            "worktree": self.worktree,
            "enqueued_at": self.enqueued_at,
            "state": self.state,
            "reason": self.reason,
            "recorded_at": self.recorded_at,
        }


def queue_store(path: Path | str | None = None) -> FragmentStore:
    """The append-only record set backing ONE repository's queue.

    ``lane_scope(path).arbitration_dir`` resolves to that repository's own shared
    ``--git-common-dir``, so the store is **per-repo by construction** — two
    repositories can never see each other's candidates, and no `repo` key or
    namespacing is needed to keep them apart. It is also identical from every
    worktree of that repo and never rewritten by a checkout/reset/merge, so two
    lanes enqueueing at the same instant write two different files and neither
    can clobber the other.
    """
    scope = lane_scope(path)
    return FragmentStore(root=scope.arbitration_dir / QUEUE_DIRNAME, key="id")


def _queue_raw_records(path: Path | str | None = None) -> tuple[dict[str, Any], ...]:
    """Read every queue fragment without replacing the store's legacy view.

    The generation folds need all append-only records, while the historical
    candidate view intentionally collapses one ``id`` at a time.  Reading the
    fragments directly keeps those two projections separate and still uses the
    one authoritative :func:`queue_store` root.
    """

    store = queue_store(path)
    records: list[dict[str, Any]] = []
    for lane in store.lanes():
        records.extend(store.read_fragment(lane))
    return tuple(records)


@dataclass
class _SnapshotQueueCandidate:
    """Writable-shape view required by the storage-neutral snapshot port."""

    branch: str
    lane: str
    base: str
    enqueued_at: datetime | str


def _queue_domain_record(record: dict[str, Any]) -> dict[str, Any]:
    """Add the fixed-store key to one RMDD-12 record, fail closed on identity."""

    if not isinstance(record, dict):
        raise MergeQueueError("queue domain record must be a mapping")
    kind = record.get("kind")
    if kind not in _ADDITIVE_RECORD_KINDS:
        raise MergeQueueError(f"unsupported queue domain record kind: {kind!r}")
    record_id = record.get("record_id")
    if not isinstance(record_id, str) or not record_id:
        raise MergeQueueError("queue domain record requires a non-blank record_id")
    existing_id = record.get("id")
    if existing_id is not None and existing_id != record_id:
        raise MergeQueueError(
            f"queue domain record id disagrees with record_id {record_id!r}"
        )
    result = dict(record)
    result["id"] = record_id
    return result


def _append_domain_record(
    record: CandidateSnapshot | GenerationRecord,
    *,
    path: Path | str | None,
    lane: str,
) -> bool:
    """Append one domain record, making restart replay idempotent.

    Candidate snapshots are immutable by record ID.  Generation records may
    restate a result, but their sealed membership remains immutable and their
    state transition is checked before the append so an illegal record never
    enters the append-only authority.
    """

    raw = _queue_domain_record(record.to_record())
    kind = raw["kind"]
    existing_raw = tuple(
        item
        for item in _queue_raw_records(path)
        if item.get("kind") == kind and item.get("record_id") == raw["record_id"]
    )
    try:
        if kind == CANDIDATE_SNAPSHOT_KIND:
            decoded = [CandidateSnapshot.from_record(item) for item in existing_raw]
            for previous_snapshot in decoded:
                if previous_snapshot.immutable_digest() != record.immutable_digest():
                    raise MergeQueueError(
                        f"candidate snapshot {raw['record_id']} changed immutable inputs"
                    )
            if decoded:
                return False
            try:
                fold_candidate_records((*existing_raw, raw))
            except CandidateGenerationError as exc:
                raise MergeQueueError(str(exc)) from exc
        else:
            if not isinstance(record, GenerationRecord):
                raise MergeQueueError("generation record kind has the wrong type")
            decoded_generations = [
                GenerationRecord.from_record(item) for item in existing_raw
            ]
            for previous_generation in decoded_generations:
                if previous_generation.immutable_digest() != record.immutable_digest():
                    raise MergeQueueError(
                        f"generation {raw['record_id']} changed immutable inputs"
                    )
            if decoded_generations:
                latest = fold_generation_records(existing_raw)[0]
                if (
                    latest.result == record.result
                    and latest.generation == record.generation
                ):
                    return False
                try:
                    latest.with_update(
                        record.generation,
                        result=record.result,
                        recorded_at=record.recorded_at,
                    )
                except CandidateGenerationError as exc:
                    raise MergeQueueError(str(exc)) from exc
            try:
                fold_generation_records((*existing_raw, raw))
            except CandidateGenerationError as exc:
                raise MergeQueueError(str(exc)) from exc
    except CandidateGenerationError as exc:
        raise MergeQueueError(str(exc)) from exc
    queue_store(path).append(raw, lane=lane)
    return True


def enqueue(
    branch: str = "",
    *,
    base: str = "",
    worktree: str | Path | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Offer *branch* (default: this lane's own branch) for landing on *base*.

    Cheap and non-blocking on purpose: it appends one record and returns. Nothing
    is verified here — verification happens once, at the head of the queue,
    against the base that actually exists then. Verifying at enqueue time
    re-creates the stale-premise problem the whole design exists to kill.
    """
    scope = lane_scope(path)
    branch = branch or _require_git(["rev-parse", "--abbrev-ref", "HEAD"], scope.tree)
    base = base or _repo_config(scope).base
    if branch in {"HEAD", base}:
        raise MergeQueueError(
            f"refusing to enqueue {branch!r}: a candidate must be a named branch "
            f"that is not {base!r} (a detached HEAD or the base itself is never one)"
        )
    now = _now()
    candidate = Candidate(
        branch=branch,
        lane=scope.lane,
        repo=scope.main_tree.name,
        base=base,
        worktree=str(Path(worktree).resolve()) if worktree else str(scope.tree),
        enqueued_at=now,
        state=QUEUED,
        recorded_at=now,
    )
    queue_store(scope.tree).append(candidate.to_record(), lane=scope.lane)
    return {
        "enqueued": True,
        **candidate.to_record(),
        "note": (
            "queued != landed. D-MQR-7: this note used to say nothing drives "
            "the queue -- that was true under D-ORC-20 and is false now. "
            "`merge-queue-runner.timer` drains every repo with a queue store "
            "automatically (~every 5 minutes, across agent-packages/, "
            "services/, images/, and the infra roots): it gates the candidate "
            "DIFFERENTIALLY against the base, lands it, and prunes the "
            "worktree/branch. You do not need to run anything yourself — "
            "watch it with `repository-manager --merge-queue status "
            "--repo-path .`. Do NOT hand-drain with `--merge-queue run`: "
            "concurrent lanes share one reconciliation-merge lease and a "
            f"manual drain races the scheduler for no benefit before {branch!r} "
            f"lands on {base!r}."
        ),
        "queue_depth": len(queued(scope.tree)),
    }


def _record_state(
    candidate: Candidate, state: str, reason: str, path: Path | str
) -> None:
    """Supersede a candidate's record with its terminal state — a NEW append.

    Never an edit: the fold collapses records sharing an ``id`` to the
    chronologically latest one written, so a landed/rejected record supersedes
    the queued one without any writer ever rewriting a file another lane also
    writes. That matters because the terminal state is normally recorded from a
    DIFFERENT lane than the one that enqueued it (enqueue from the candidate's
    own worktree, land from the driver).
    """
    scope = lane_scope(path)
    queue_store(scope.tree).append(
        replace(candidate, state=state, reason=reason, recorded_at=_now()).to_record(),
        lane=scope.lane,
    )


def withdraw(branch: str, *, reason: str = "", path: Path | str | None = None) -> dict:
    """Pull a candidate back out of the queue (its lane changed its mind)."""
    scope = lane_scope(path)
    for candidate in _all_candidates(scope.tree):
        if candidate.branch == branch:
            _record_state(candidate, WITHDRAWN, reason, scope.tree)
            return {"withdrawn": True, "branch": branch, "reason": reason}
    raise MergeQueueError(f"{branch!r} is not in this repository's queue")


def _resolve_latest_candidate_record(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick the chronologically LATEST record by ``recorded_at`` — behaviour 2.

    :meth:`FragmentStore.fold` groups every lane's record for one id in
    ``lanes()``'s **alphabetically sorted** order and would otherwise take
    ``group[-1]`` — which silently prefers whichever LANE NAME sorts last, a
    property entirely unrelated to which record was written most recently. The
    live consequence (D-F6-1/D-CVG-9): a candidate enqueued by ``lane-foo`` and
    landed by ``canonical`` folds to the OLDER queued record, because
    ``"canonical" < "lane-foo"`` — so a candidate that had genuinely landed
    reported ``queued`` forever, and a stale record could revive a dead one.

    Delegates to :func:`repository_manager.task_queue.resolve_latest_record` —
    the SAME fold-by-``recorded_at`` rule :mod:`build_queue` relies on for its
    own ``Task`` records, extracted so the two queues share one mechanism
    (CONCEPT:RM-TASK-LEDGER) rather than maintaining two copies of this exact
    ordering fix. Only this pure comparison moved; ``Candidate``'s on-disk
    layout (``QUEUE_DIRNAME``) is untouched deliberately — the live queue is
    draining real branches on a 5-minute timer and a storage-path change would
    silently orphan whatever is mid-flight.
    """
    from repository_manager.task_queue import resolve_latest_record

    # A generation record may share the fixed FragmentStore key with a legacy
    # branch by coincidence.  The legacy projection must remain authoritative
    # for that branch; domain records are folded separately from raw fragments.
    legacy = [
        record for record in group if record.get("kind") not in _ADDITIVE_RECORD_KINDS
    ]
    return resolve_latest_record(legacy or group)


def _all_candidates(path: Path | str | None = None) -> list[Candidate]:
    return [
        Candidate.from_record(r)
        for r in queue_store(path).fold(resolve=_resolve_latest_candidate_record)
        if r.get("kind") not in _ADDITIVE_RECORD_KINDS
    ]


def queued(path: Path | str | None = None) -> list[Candidate]:
    """Every still-pending candidate, oldest first (FIFO, fair by construction)."""
    pending = [c for c in _all_candidates(path) if c.state == QUEUED]
    return sorted(pending, key=lambda c: (c.enqueued_at, c.branch))


def queue_report(path: Path | str | None = None) -> dict[str, Any]:
    """Depth, order, terminal outcomes, and the declaration that governs them."""
    scope = lane_scope(path)
    everything = _all_candidates(scope.tree)
    pending = queued(scope.tree)
    try:
        config: QueueConfig | None = _repo_config(scope)
        config_error = ""
    except MergeQueueError as exc:
        config, config_error = None, str(exc)
    return {
        "repo": scope.main_tree.name,
        "depth": len(pending),
        "queued": [c.to_record() for c in pending],
        "recent": [c.to_record() for c in everything if c.state in {LANDED, REJECTED}][
            -20:
        ],
        "lease": MERGE_LEASE,
        "config": (
            {
                "source": config.source,
                "base": config.base,
                "batch_size": config.batch_size,
                "fast_gates": [g.name for g in config.fast_gates()],
                "slow_gates": [g.name for g in config.slow_gates()],
                "generated_files": sorted(config.generated_files),
            }
            if config
            else None
        ),
        "config_error": config_error,
        "checked_at": _now(),
    }


def _repo_config(scope: LaneScope) -> QueueConfig:
    """The config as it exists in the canonical checkout right now.

    Used by ``enqueue``/``status`` (which have no merged tree yet). The GATE path
    deliberately re-reads it from the merged tree instead — see
    :func:`load_config`.
    """
    return load_config(scope.main_tree)


# ---------------------------------------------------------------------------
# Merge cleanliness, computed entirely as objects — no working tree touched
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TrialMerge:
    ok: bool
    tree: str
    conflicts: list[str] = field(default_factory=list)
    detail: str = ""


def trial_merge(repo: Path, base_ref: str, branch: str) -> TrialMerge:
    """Merge *branch* into *base_ref* as objects only; report conflicts, move no ref.

    ``git merge-tree --write-tree`` does the real three-way merge, writes the
    result into the object database, and exits 1 with the conflicted paths when
    it cannot. Nothing is checked out, no ref moves, no index is touched — which
    is why this is safe to run against a canonical checkout while other lanes are
    mid-work, and why it costs milliseconds rather than a checkout.
    """
    res = _run_git(
        ["merge-tree", "--write-tree", "--name-only", base_ref, branch], repo
    )
    if res.ok:
        return TrialMerge(ok=True, tree=res.out.splitlines()[0].strip())
    if res.code != 1:
        raise MergeQueueError(
            f"git merge-tree failed against {base_ref}..{branch}: {res.err or res.out}"
        )
    lines = res.out.splitlines()
    tree = lines[0].strip() if lines else ""
    conflicts = [ln.strip() for ln in lines[1:] if ln.strip()]
    blank = next((i for i, ln in enumerate(lines[1:], 1) if not ln.strip()), None)
    if blank is not None:
        conflicts = [ln.strip() for ln in lines[1:blank] if ln.strip()]
    return TrialMerge(ok=False, tree=tree, conflicts=conflicts, detail=res.out)


def _commit_trial(repo: Path, tree: str, parents: list[str], message: str) -> str:
    args = ["commit-tree", tree]
    for parent in parents:
        args += ["-p", parent]
    return _require_git([*args, "-m", message], repo)


def changed_paths(repo: Path, base_ref: str, ref: str) -> list[str]:
    """Paths *ref* changes relative to its MERGE BASE with *base_ref*.

    Merge-base rather than a plain diff: a branch merely behind the base has not
    "changed" the files the base moved underneath it, and counting those as its
    own would balloon every path-scoped gate with work the branch never did.
    """
    merge_base = _require_git(["merge-base", base_ref, ref], repo)
    out = _require_git(["diff", "--name-only", f"{merge_base}..{ref}"], repo)
    return [line for line in out.splitlines() if line]


def _bounded_shadow_paths(paths: Iterable[str]) -> dict[str, Any]:
    ordered = sorted({str(path) for path in paths if str(path)})
    return {
        "paths": ordered[:_MAX_SHADOW_EVIDENCE_PATHS],
        "count": len(ordered),
        "truncated": len(ordered) > _MAX_SHADOW_EVIDENCE_PATHS,
    }


def _bounded_shadow_detail(detail: str) -> dict[str, Any]:
    text = str(detail)
    return {
        "detail": text[:_MAX_SHADOW_DETAIL_CHARS],
        "detail_chars": len(text),
        "detail_truncated": len(text) > _MAX_SHADOW_DETAIL_CHARS,
    }


def _commit_shadow_trial(
    repo: Path,
    tree: str,
    parents: list[str],
    message: str,
    *,
    sealed_at: datetime,
) -> str:
    """Create a replay-stable private trial commit with no ref update."""

    epoch = str(int(timestamp_value(sealed_at).timestamp()))
    env = {
        "GIT_AUTHOR_NAME": "repository-manager-shadow",
        "GIT_AUTHOR_EMAIL": "shadow@repository-manager.invalid",
        "GIT_COMMITTER_NAME": "repository-manager-shadow",
        "GIT_COMMITTER_EMAIL": "shadow@repository-manager.invalid",
        "GIT_AUTHOR_DATE": f"@{epoch}",
        "GIT_COMMITTER_DATE": f"@{epoch}",
    }
    args = ["commit-tree", tree]
    for parent in parents:
        args += ["-p", parent]
    return _require_git([*args, "-m", message], repo, env=env)


def _shadow_trial_merge(repo: Path, record: GenerationRecord) -> dict[str, Any]:
    """Classify one sealed generation with object-only Git operations.

    This deliberately calls the same :func:`trial_merge` and
    :func:`changed_paths` primitives as the live queue, but feeds them the
    immutable snapshot SHAs rather than moving branch names.  It never invokes
    gate execution, materialization, landing, or cleanup.
    """

    generation = record.generation
    target_ref = generation.target_branch
    target_before = _require_git(
        ["rev-parse", "--verify", f"{target_ref}^{{commit}}"], repo
    )
    if generation.sealed_at is None:
        raise MergeQueueError(
            f"generation {record.record_id} has no seal timestamp for shadow replay"
        )
    sealed_at = timestamp_value(generation.sealed_at)
    if target_before != generation.base_sha:
        target_after = _require_git(
            ["rev-parse", "--verify", f"{target_ref}^{{commit}}"], repo
        )
        return {
            "mode": "shadow",
            "objects_only": True,
            "execution": "not-run",
            "landing": "not-run",
            "certifiable": False,
            "status": "stale-base",
            "stale_base": True,
            "target_ref": target_ref,
            "target_ref_before": target_before,
            "target_ref_after": target_after,
            "target_ref_stable": target_before == target_after,
            "base_sha": generation.base_sha,
            "synthetic_commit_sha": "",
            "tree_sha": "",
            "accepted": [],
            "conflicts": [],
            "accepted_candidate_ids": [],
            "conflicted_candidate_ids": [],
            "differential_paths": [],
            "differential_path_count": 0,
            "differential_paths_truncated": False,
        }
    head = generation.base_sha
    tree = _require_git(["rev-parse", f"{head}^{{tree}}"], repo)
    differential: set[str] = set()
    accepted: list[dict[str, Any]] = []
    conflicts: list[dict[str, Any]] = []

    for member in record.members:
        member_changed = changed_paths(repo, generation.base_sha, member.candidate_sha)
        differential.update(member_changed)
        trial = trial_merge(repo, head, member.candidate_sha)
        member_identity = {
            "candidate_id": member.candidate_id,
            "version": member.version,
            "branch": member.branch,
            "candidate_sha": member.candidate_sha,
        }
        if not trial.ok:
            conflict_paths = _bounded_shadow_paths(trial.conflicts)
            member_paths = _bounded_shadow_paths(member_changed)
            detail = _bounded_shadow_detail(trial.detail)
            against = [
                {
                    "candidate_id": item["candidate_id"],
                    "version": item["version"],
                }
                for item in accepted
            ]
            conflicts.append(
                {
                    **member_identity,
                    "conflicts": conflict_paths["paths"],
                    "conflict_count": conflict_paths["count"],
                    "conflicts_truncated": conflict_paths["truncated"],
                    **detail,
                    "differential_paths": member_paths["paths"],
                    "differential_path_count": member_paths["count"],
                    "differential_paths_truncated": member_paths["truncated"],
                    "conflict_against": {
                        "kind": "rolling-synthetic-head",
                        "members": against,
                        "base_sha": generation.base_sha,
                    },
                }
            )
            continue

        already_contained = (
            member.candidate_sha == head
            or _run_git(
                ["merge-base", "--is-ancestor", member.candidate_sha, head], repo
            ).ok
        )
        if already_contained:
            tree = trial.tree or tree
            accepted.append({**member_identity, "already_contained": True})
            continue
        head = _commit_shadow_trial(
            repo,
            trial.tree,
            [head, member.candidate_sha],
            f"shadow merge({member.lane_id}): {member.branch}",
            sealed_at=sealed_at,
        )
        tree = trial.tree
        accepted.append({**member_identity, "already_contained": False})

    target_after = _require_git(
        ["rev-parse", "--verify", f"{target_ref}^{{commit}}"], repo
    )
    differential_paths = _bounded_shadow_paths(differential)
    target_stable = target_before == target_after
    return {
        "mode": "shadow",
        "objects_only": True,
        "execution": "not-run",
        "landing": "not-run",
        "certifiable": False,
        "target_ref": target_ref,
        "target_ref_before": target_before,
        "target_ref_after": target_after,
        "target_ref_stable": target_stable,
        "base_sha": generation.base_sha,
        "synthetic_commit_sha": head,
        "tree_sha": tree,
        "accepted": accepted,
        "conflicts": conflicts,
        "accepted_candidate_ids": [item["candidate_id"] for item in accepted],
        "conflicted_candidate_ids": [item["candidate_id"] for item in conflicts],
        "differential_paths": differential_paths["paths"],
        "differential_path_count": differential_paths["count"],
        "differential_paths_truncated": differential_paths["truncated"],
        "status": (
            "stale-base"
            if not target_stable
            else "conflicted"
            if conflicts
            else "trial-merged"
        ),
        "stale_base": not target_stable,
    }


def _shadow_generation_unlocked(
    *,
    config_digest: str,
    toolchain_digest: str,
    resource_digest: str,
    base: str = "",
    batch_size: int = 0,
    debounce: float | int | timedelta = 0,
    maximum_age: float | int | timedelta = 0,
    build_target: str = "default",
    concept_claims: Iterable[object] = (),
    incompatibility_labels: Iterable[object] = (),
    target: TargetPolicy | None = None,
    now: datetime | str | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Snapshot and trial a queue batch without execution or ref movement.

    This is an internal RMDD-12 adapter, not a public queue action.  The three
    digest arguments are deliberately required and validated by
    ``snapshot_branch_candidate``; callers must supply canonical immutable
    inputs rather than deriving an authoritative digest from a short or
    unpinned environment label.  RMDD-29 owns any execution/landing handoff.
    """

    scope = lane_scope(path)
    repo = scope.main_tree
    config = _repo_config(scope)
    target_branch = base or config.base
    limit = batch_size or config.batch_size
    current = timestamp_value(now or _now())
    repository = RepositoryIdentity(
        repository_id=repository_id_for(repo),
        canonical_path=str(repo.resolve()),
    )
    try:
        folded_snapshots = fold_candidate_records(_queue_raw_records(repo))
        folded_generations = fold_generation_records(_queue_raw_records(repo))
    except CandidateGenerationError as exc:
        raise MergeQueueError(str(exc)) from exc
    previous_by_candidate: dict[str, CandidateSnapshot] = {}
    for snapshot in folded_snapshots:
        previous = previous_by_candidate.get(snapshot.candidate_id)
        if previous is None or snapshot.version > previous.version:
            previous_by_candidate[snapshot.candidate_id] = snapshot
    existing_generations = {item.record_id: item for item in folded_generations}

    pending = [
        candidate for candidate in queued(repo) if candidate.base == target_branch
    ]
    snapshots: list[CandidateSnapshot] = []
    snapshot_appends = 0

    def resolve_ref(ref: str) -> str:
        return _require_git(["rev-parse", "--verify", f"{ref}^{{commit}}"], repo)

    for candidate in pending:
        logical_id = candidate_identity(
            repository.repository_id, candidate.branch, candidate.lane
        )
        try:
            snapshot = snapshot_branch_candidate(
                _SnapshotQueueCandidate(
                    branch=candidate.branch,
                    lane=candidate.lane,
                    base=candidate.base,
                    enqueued_at=candidate.enqueued_at,
                ),
                repository=repository,
                resolve_ref=resolve_ref,
                target_branch=target_branch,
                config_digest=config_digest,
                toolchain_digest=toolchain_digest,
                resource_digest=resource_digest,
                build_target=build_target,
                concept_claims=concept_claims,
                incompatibility_labels=incompatibility_labels,
                target=target,
                previous=previous_by_candidate.get(logical_id),
            )
        except (CandidateGenerationError, TypeError, ValueError) as exc:
            raise MergeQueueError(
                f"could not snapshot queued candidate {candidate.branch!r}: {exc}"
            ) from exc
        snapshots.append(snapshot)
        if _append_domain_record(snapshot, path=repo, lane=scope.lane):
            snapshot_appends += 1

    # A replay may construct the same immutable snapshot with a fresh
    # observation timestamp.  Continue with the canonical stored member so the
    # nested member record in a sealed generation is byte-stable as well.
    try:
        canonical_snapshots = {
            item.record_id: item
            for item in fold_candidate_records(_queue_raw_records(repo))
        }
    except CandidateGenerationError as exc:
        raise MergeQueueError(str(exc)) from exc
    snapshots = [canonical_snapshots[item.record_id] for item in snapshots]

    try:
        selection = select_batches(
            snapshots,
            now=current,
            debounce=debounce,
            maximum_age=maximum_age,
            batch_size=limit,
            sealed_at=current,
        )
    except (CandidateGenerationError, TypeError, ValueError) as exc:
        raise MergeQueueError(f"could not select a shadow generation: {exc}") from exc

    generation_results: list[dict[str, Any]] = []
    generation_appends = 0
    for batch in selection.batches:
        sealed = seal_generation(batch, sealed_at=selection.sealed_at, target=target)
        generation = existing_generations.get(sealed.record_id, sealed)
        if generation is sealed:
            if _append_domain_record(generation, path=repo, lane=scope.lane):
                generation_appends += 1
        result = _shadow_trial_merge(repo, generation)
        result_time = selection.sealed_at + timedelta(microseconds=1)
        updated = generation.with_update(
            generation.generation,
            result=result,
            recorded_at=result_time,
        )
        if _append_domain_record(updated, path=repo, lane=scope.lane):
            generation_appends += 1
        generation_results.append(
            {
                "generation_id": generation.record_id,
                "candidate_ids": [member.candidate_id for member in generation.members],
                "versioned_members": [
                    member.record_id for member in generation.members
                ],
                "result": result,
            }
        )

    return {
        "repo": repo.name,
        "mode": "shadow",
        "landed": False,
        "target_branch": target_branch,
        "candidate_snapshot_appends": snapshot_appends,
        "generation_record_appends": generation_appends,
        "generations": generation_results,
        "late": [item.record_id for item in selection.late],
        "waiting": [item.record_id for item in selection.waiting],
        "selected": [item.record_id for item in selection.selected],
        "execution_handoff": "RMDD-29-gated",
    }


def shadow_generation(
    *,
    config_digest: str,
    toolchain_digest: str,
    resource_digest: str,
    base: str = "",
    batch_size: int = 0,
    debounce: float | int | timedelta = 0,
    maximum_age: float | int | timedelta = 0,
    build_target: str = "default",
    concept_claims: Iterable[object] = (),
    incompatibility_labels: Iterable[object] = (),
    target: TargetPolicy | None = None,
    now: datetime | str | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Serialize shadow snapshot/seal/replay through the merge lease."""

    scope = lane_scope(path)
    with hold_lease(
        MERGE_LEASE,
        operation=f"shadow generation for {scope.main_tree.name}",
        path=scope.tree,
    ):
        return _shadow_generation_unlocked(
            config_digest=config_digest,
            toolchain_digest=toolchain_digest,
            resource_digest=resource_digest,
            base=base,
            batch_size=batch_size,
            debounce=debounce,
            maximum_age=maximum_age,
            build_target=build_target,
            concept_claims=concept_claims,
            incompatibility_labels=incompatibility_labels,
            target=target,
            now=now,
            path=path,
        )


@contextmanager
def materialized(repo: Path, commit: str, *, scope: LaneScope) -> Iterator[Path]:
    """Check *commit* out into a disposable detached worktree, then remove it.

    This is what makes "gate it AS MERGED" affordable enough to sit inside the
    lease — and it is the whole reason the gate catches the defect class where
    git merges two branches with zero conflicts into a tree that does not build
    (D-OB-17). It lives under this lane's **partitioned** scratch dir, so two
    runners can never materialize into the same path.
    """
    root = partitioned_paths(scope.tree).scratch_dir / "merge-queue-verify"
    root.mkdir(parents=True, exist_ok=True)
    target = root / commit[:12]
    if target.exists():
        shutil.rmtree(target, ignore_errors=True)
    _require_git(["worktree", "add", "--detach", str(target), commit], repo)
    try:
        yield target
    finally:
        _run_git(["worktree", "remove", "--force", str(target)], repo)
        shutil.rmtree(target, ignore_errors=True)
        _run_git(["worktree", "prune"], repo)


# ---------------------------------------------------------------------------
# Running a gate, and comparing it against the base ref — behaviour 1
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Check:
    """One gate's verdict, with the evidence a rejected lane needs to act on."""

    name: str
    ok: bool
    seconds: float
    detail: str = ""
    deferred: bool = False


@dataclass(frozen=True)
class GateResult:
    ok: bool
    checks: list[Check]
    environment: str
    seconds: float

    def failures(self) -> list[Check]:
        return [c for c in self.checks if not c.ok]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "seconds": round(self.seconds, 2),
            "environment": self.environment,
            "checks": [
                {
                    "name": c.name,
                    "ok": c.ok,
                    "seconds": round(c.seconds, 2),
                    "deferred": c.deferred,
                    "detail": c.detail[:4000],
                }
                for c in self.checks
            ],
        }


@dataclass(frozen=True)
class GateBaseline:
    """One gate's result on the base ref.

    ``readable=False`` is the fail-closed case: callers MUST refuse a candidate
    whose merged-tree run is non-clean rather than guess the failure is
    pre-existing. Treating an unproducible baseline as "no pre-existing failures"
    is exactly the allow-everything fallback this design forbids — it would turn
    every gate off at precisely the moment the base is unmeasurable.
    """

    readable: bool
    ok: bool = True
    signals: frozenset[str] = frozenset()
    base_sha: str = ""
    detail: str = ""


def _timed_run(argv: list[str], cwd: Path, *, timeout: int, env: dict) -> tuple:
    """Run one gate command. The RUNNER, not the caller, owns the final argv.

    ``ensure_no_fail_fast`` is applied here — the single process-launch
    chokepoint every gate/baseline run passes through (``run_gate`` and
    ``compute_gate_baseline`` both call only this) — so a repo's
    ``.mergequeue.yaml`` cannot omit ``--no-fail-fast`` from a declared
    ``cargo test``/``cargo nextest run`` gate: this is where it is added if
    missing, before the process is ever spawned (CONCEPT:RM-TEST-COMMANDS).
    """
    argv = ensure_no_fail_fast(argv)
    start = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603 - argv from repo-declared config, no shell
            argv,
            cwd=str(cwd),
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        return None, time.monotonic() - start
    except OSError as exc:
        return exc, time.monotonic() - start
    return proc, time.monotonic() - start


def _normalized_lines(text: str, tree: Path, gate: GateSpec) -> frozenset[str]:
    """Comparable output lines from a gate run.

    The tree path is substituted out FIRST and this is not cosmetic: the merged
    run and the base run happen in two DIFFERENT throwaway worktrees, so every
    absolute path a tool prints differs between them. Without this substitution a
    ``lines`` comparison would report every diagnostic as new and refuse every
    candidate — a gate that is always red is indistinguishable from a gate that
    is turned off.
    """
    body = _ANSI.sub("", text).replace(str(tree), "<tree>")
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]
    if gate.keep_lines:
        keep = [re.compile(p) for p in gate.keep_lines]
        lines = [ln for ln in lines if any(p.search(ln) for p in keep)]
    if gate.ignore_lines:
        drop = [re.compile(p) for p in gate.ignore_lines]
        lines = [ln for ln in lines if not any(p.search(ln) for p in drop)]
    return frozenset(lines)


def _pytest_ids(stdout: str) -> frozenset[str]:
    """Test ids pytest's ``-rfE`` short summary reports as failed or errored.

    Parsed from pytest's own stable ``FAILED <nodeid> - reason`` /
    ``ERROR <nodeid>`` contract rather than a machine report format, so this
    needs no extra plugin and no schema of our own to keep in sync.
    """
    ids: set[str] = set()
    for line in stdout.splitlines():
        for prefix in ("FAILED ", "ERROR "):
            if line.startswith(prefix):
                nodeid = line[len(prefix) :].split(" - ", 1)[0].strip()
                if nodeid:
                    ids.add(nodeid)
    return frozenset(ids)


def _signals(proc: Any, tree: Path, gate: GateSpec) -> frozenset[str]:
    """The comparable failure signals one gate run produced."""
    if gate.compare == "pytest-ids":
        return _pytest_ids(proc.stdout)
    if gate.compare == "lines":
        return _normalized_lines(proc.stdout + "\n" + proc.stderr, tree, gate)
    return frozenset()


def _environment_signature(tree: Path, config: QueueConfig) -> str:
    """A signature of the toolchain the gates run under, folded into the cache key.

    D-MQD-1: keying a baseline on ``(base_sha, gate)`` alone is not enough. If the
    toolchain is rebuilt IN PLACE with different versions while ``base_sha`` stays
    the same, the old key still resolves and the cache serves a failing-signal set
    computed against tools that no longer exist.

    Three sources, in order, and the LAST one is the honest refusal:
    1. the config's declared ``environment_signature`` command (``cargo --version``,
       ``rustc -V``, whatever fingerprints that repo's toolchain);
    2. a discoverable ``.venv``'s installed distribution set (name-version from
       every ``*.dist-info``/``*.egg-info``) — one directory listing, no subprocess;
    3. ``"unpinned"`` — and an unpinned environment **disables the baseline cache**
       (:func:`_baseline_cache_path` returns ``None``) rather than keying it on a
       fiction. Every baseline is then recomputed, which is slower and correct.
    """
    if config.environment_signature:
        try:
            proc = subprocess.run(  # noqa: S603 - argv from repo-declared config
                list(config.environment_signature),
                cwd=str(tree),
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (OSError, subprocess.TimeoutExpired):
            return "unpinned"
        if proc.returncode == 0 and proc.stdout.strip():
            return hashlib.sha256(proc.stdout.encode()).hexdigest()[:16]
        return "unpinned"
    for venv in (tree / ".venv", tree.parent / ".venv"):
        dists = [
            p.name
            for site in (
                *venv.glob("lib/python*/site-packages"),
                *venv.glob("Lib/site-packages"),
            )
            for p in (*site.glob("*.dist-info"), *site.glob("*.egg-info"))
        ]
        if dists:
            return hashlib.sha256("\n".join(sorted(dists)).encode()).hexdigest()[:16]
    return "unpinned"


def _baseline_cache_path(
    scope: LaneScope, base_sha: str, gate: GateSpec, *, environment: str
) -> Path | None:
    """Cache location for one ``(base_sha, gate, environment)`` — or ``None``.

    ``None`` when the environment is ``"unpinned"``: with no way to notice a
    toolchain change, a cache entry could outlive the toolchain that produced it
    and serve a stale verdict. Recomputing is the honest trade.
    """
    if environment == "unpinned":
        return None
    digest = hashlib.sha256(
        "\n".join(
            [base_sha, gate.name, " ".join(gate.command), gate.compare, environment]
        ).encode()
    ).hexdigest()
    return (
        scope.arbitration_dir
        / QUEUE_DIRNAME
        / BASELINE_CACHE_DIRNAME
        / f"{digest}.json"
    )


def _load_baseline_cache(path: Path | None) -> GateBaseline | None:
    """A cached baseline, or ``None`` on a miss — including a CORRUPT entry.

    A corrupt cache file means "recompute", not "unreadable baseline": the
    fail-closed contract is about each RUN's honesty, not this cache's storage
    layer, and refusing every future candidate because one file got truncated
    would be a self-inflicted permanent denial.
    """
    if path is None or not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return GateBaseline(
            readable=True,
            ok=bool(data["ok"]),
            signals=frozenset(str(s) for s in data["signals"]),
            base_sha=str(data["base_sha"]),
            detail="from cache",
        )
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _store_baseline_cache(path: Path | None, baseline: GateBaseline) -> None:
    if path is None or not baseline.readable:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp")
    tmp.write_text(
        json.dumps(
            {
                "base_sha": baseline.base_sha,
                "ok": baseline.ok,
                "signals": sorted(baseline.signals),
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    tmp.replace(path)  # atomic: a concurrent reader never sees a half-written file


def compute_gate_baseline(
    repo: Path,
    base_ref: str,
    gate: GateSpec,
    *,
    scope: LaneScope,
    config: QueueConfig,
    env: dict[str, str],
) -> GateBaseline:
    """Run *gate* on *base_ref* to learn what it ALREADY reports there.

    This is behaviour 1's engine. ``main`` is legitimately red in this workspace,
    and a gate that judges the merged tree in isolation refuses every candidate
    that so much as touches an already-red area — measured on the au queue: a
    branch that FIXED 21 of 30 failing tests was still rejected because 9
    remained. An absolute gate once deadlocked the queue and stranded 19
    branches. So a failure is permitted only when the *same signal* is already
    present on the base.

    Every failure mode here returns ``readable=False``, which REFUSES the
    candidate. There is deliberately no code path that turns an unproducible
    baseline into an empty one.
    """
    base_sha = _require_git(["rev-parse", base_ref], repo)
    environment = _environment_signature(scope.main_tree, config)
    cache_path = _baseline_cache_path(scope, base_sha, gate, environment=environment)
    cached = _load_baseline_cache(cache_path)
    if cached is not None:
        return cached

    with materialized(repo, base_sha, scope=scope) as base_tree:
        proc, secs = _timed_run(
            list(gate.command),
            base_tree,
            timeout=gate.effective_baseline_timeout,
            env=env,
        )
        if proc is None:
            return GateBaseline(
                readable=False,
                base_sha=base_sha,
                detail=(
                    f"gate {gate.name!r} on {base_ref} ({base_sha[:12]}) exceeded "
                    f"{gate.effective_baseline_timeout}s after {secs:.1f}s — an "
                    "unproducible baseline is REFUSED, never silently treated as "
                    "'no pre-existing failures'"
                ),
            )
        if isinstance(proc, OSError):
            return GateBaseline(
                readable=False,
                base_sha=base_sha,
                detail=(
                    f"gate {gate.name!r} could not be executed on {base_ref}: {proc} "
                    f"(command: {' '.join(gate.command)}) — REFUSED, not assumed clean"
                ),
            )
        if (
            gate.compare == "pytest-ids"
            and proc.returncode not in _PYTEST_READABLE_EXIT_CODES
        ):
            return GateBaseline(
                readable=False,
                base_sha=base_sha,
                detail=(
                    f"gate {gate.name!r} on {base_ref} ({base_sha[:12]}) exited "
                    f"{proc.returncode} (collection/usage/internal error, not a test "
                    "outcome) — an unreadable baseline is REFUSED\n"
                    + proc.stdout[-2000:]
                ),
            )
        baseline = GateBaseline(
            readable=True,
            ok=proc.returncode == 0,
            signals=_signals(proc, base_tree, gate),
            base_sha=base_sha,
            detail=f"gate {gate.name!r} evaluated on {base_ref} ({base_sha[:12]}) in {secs:.1f}s",
        )
    _store_baseline_cache(cache_path, baseline)
    return baseline


def refuse_precommit_on_dirty_tree(tree: Path, gate: GateSpec) -> str:
    """Why *gate* must not run against *tree*, or ``""`` when it may (D-ORC-37).

    A central driver must **never** run pre-commit against a tree holding
    someone else's uncommitted work. ``pre_commit/staged_files_only.py`` writes
    the unstaged changes to a patch file, ``git checkout``s them away so hooks
    see only staged content, and restores them in a ``finally:``. If the process
    is killed, OOMs, or the host loses power inside that window the work is GONE
    from the tree and survives only as an orphaned patch nobody replays (D-OB-12
    — a power outage and a swap-exhaustion ``Bus error`` both happened on this
    host in one day). **A clean tree has no patch-restore window at all**, which
    is why the refusal is the whole fix rather than a mitigation.

    Every tree this queue gates is a throwaway checkout of a committed object, so
    this is normally vacuous — which is exactly the point: it makes the invariant
    *checked* rather than assumed, and it fires immediately if a future caller
    ever points a gate at a live worktree.
    """
    if not gate.runs_precommit:
        return ""
    status = _run_git(["status", "--porcelain"], tree)
    if not status.ok:
        return (
            f"gate {gate.name!r} runs pre-commit and this queue could not verify that "
            f"{tree} is clean (`git status --porcelain` failed: {status.err}) — "
            "REFUSED rather than risking pre-commit's patch-restore window"
        )
    if status.out.strip():
        return (
            f"gate {gate.name!r} runs pre-commit against {tree}, which holds "
            "uncommitted work. pre-commit stashes unstaged changes to a patch file "
            "and checks them out of the tree while hooks run; a crash in that window "
            "loses them (D-OB-12/D-ORC-37). Commit first, or run the gate against a "
            f"committed state.\n{status.out.strip()[:1500]}"
        )
    return ""


def _compare_gate(
    gate: GateSpec, proc: Any, tree: Path, baseline: GateBaseline, seconds: float
) -> Check:
    """Turn one merged-tree run + its baseline into a verdict — behaviour 1.

    Three comparison shapes, because tools report failure in three shapes:

    * ``pytest-ids`` — an ID-level compare. A failing id is permitted ONLY when
      that EXACT id already fails on the base. Never by file, module, pattern or
      count: an id-level compare is the only shape that cannot be gamed into
      masking a real regression.
    * ``lines`` — a line-level compare of normalized output. Catches a genuinely
      new diagnostic even when the base was already red for something ELSE, which
      a coarse pass/fail compare would mask.
    * ``exit`` — script granularity, for a tool that prints one static message (or
      nothing) regardless of *why* it failed. The finest available signal is "was
      this red before", so that is what is used, and it is reported as such rather
      than dressed up as more precision than it has.
    """
    if not baseline.readable:
        return Check(
            gate.name,
            ok=False,
            seconds=seconds,
            detail=(
                f"REFUSED: {gate.name!r} did not pass on the merged tree and the "
                f"base-ref baseline needed to tell a regression from pre-existing "
                f"red could not be produced: {baseline.detail}"
            ),
        )
    base_label = f"base ({baseline.base_sha[:12]})"

    if gate.compare == "exit":
        if baseline.ok:
            return Check(
                gate.name,
                ok=False,
                seconds=seconds,
                detail=(
                    f"NEW failure not present on {base_label} (exit {proc.returncode}; "
                    f"compare=exit gives no itemized signal to diff):\n"
                    + (proc.stderr or proc.stdout).strip()[:1500]
                ),
            )
        return Check(
            gate.name,
            ok=True,
            seconds=seconds,
            detail=(
                f"non-clean on {base_label} too (exit {proc.returncode}); compare=exit "
                "allows it at script granularity — pre-existing, reported, NOT silent"
            ),
        )

    merged = _signals(proc, tree, gate)
    new = sorted(merged - baseline.signals)
    pre_existing = sorted(merged & baseline.signals)
    fixed = sorted(baseline.signals - merged)

    if not merged and proc.returncode != 0 and baseline.ok:
        # Non-zero exit with nothing itemized to compare, and the base was clean:
        # unambiguously new. Falling through to "no new signals -> pass" here
        # would let a gate fail silently whenever its output shape changed.
        return Check(
            gate.name,
            ok=False,
            seconds=seconds,
            detail=(
                f"exited {proc.returncode} on the merged tree with no comparable "
                f"output, and was clean on {base_label} — refused as a NEW failure. "
                f"(If this gate reports failure without itemized output, declare "
                f"compare: exit.)\n" + (proc.stderr or proc.stdout).strip()[:1500]
            ),
        )

    def _fmt(label: str, items: list[str], limit: int = 25) -> str:
        shown = "\n  ".join(items[:limit])
        more = f"\n  (+{len(items) - limit} more)" if len(items) > limit else ""
        return f"{len(items)} {label}:\n  {shown}{more}"

    if new:
        detail = _fmt(f"NEW signal(s) not present on {base_label}", new)
        if pre_existing:
            detail += "\n" + _fmt(
                f"pre-existing on {base_label} (not blocking)", pre_existing
            )
        if fixed:
            detail += "\n" + _fmt(f"FIXED relative to {base_label}", fixed)
        return Check(gate.name, ok=False, seconds=seconds, detail=detail)

    parts: list[str] = []
    if pre_existing:
        # Allowed, but reported explicitly with counts, so a pass over a red base
        # can never read as a silent success. This is not a masking mechanism.
        parts.append(
            _fmt(
                f"pre-existing on {base_label} (allowed — not caused here)",
                pre_existing,
            )
        )
    if fixed:
        parts.append(_fmt(f"FIXED relative to {base_label}", fixed))
    return Check(
        gate.name,
        ok=True,
        seconds=seconds,
        detail="\n".join(parts) or f"clean (exit {proc.returncode})",
    )


def run_gate(
    gate: GateSpec,
    tree: Path,
    *,
    repo: Path,
    base_ref: str,
    scope: LaneScope,
    config: QueueConfig,
    changed: list[str],
    env: dict[str, str],
) -> Check:
    """Run one declared gate against the MERGED tree and judge it against the base."""
    if gate.when_changed and not any(
        fnmatch.fnmatch(p, pattern) for p in changed for pattern in gate.when_changed
    ):
        return Check(
            gate.name,
            ok=True,
            seconds=0.0,
            detail=f"no changed path matches when_changed={list(gate.when_changed)}",
        )
    refusal = refuse_precommit_on_dirty_tree(tree, gate)
    if refusal:
        return Check(gate.name, ok=False, seconds=0.0, detail=refusal)

    proc, seconds = _timed_run(list(gate.command), tree, timeout=gate.timeout, env=env)
    if proc is None:
        deferred = gate.on_timeout == "defer"
        return Check(
            gate.name,
            ok=deferred,
            seconds=seconds,
            deferred=deferred,
            detail=(
                f"exceeded {gate.timeout}s — "
                + (
                    "deferred to the post-merge suite rather than holding the lease "
                    "(on_timeout: defer); the queue's value is that it always returns"
                    if deferred
                    else "a gate that cannot answer is not a passing gate (on_timeout: fail)"
                )
            ),
        )
    if isinstance(proc, OSError):
        return Check(
            gate.name,
            ok=False,
            seconds=seconds,
            detail=(
                f"could not execute {' '.join(gate.command)}: {proc} — a gate that "
                "cannot run is refused, never assumed clean"
            ),
        )
    if (
        gate.compare == "pytest-ids"
        and proc.returncode not in _PYTEST_READABLE_EXIT_CODES
    ):
        # A collection error is not a pass on EITHER tree, and an unreadable
        # merged run gives no signal set to diff — there is nothing to compare.
        return Check(
            gate.name,
            ok=False,
            seconds=seconds,
            detail=(
                f"the merged tree's run exited {proc.returncode} "
                "(collection/usage/internal error, not a test outcome) — refused; a "
                "run that cannot enumerate its failures cannot be diffed against the "
                "base\n" + proc.stdout[-2000:] + "\n" + proc.stderr[-1000:]
            ),
        )
    if proc.returncode == 0 and gate.compare != "lines":
        return Check(gate.name, ok=True, seconds=seconds, detail="clean")

    # The baseline is computed whenever there is anything to compare — including
    # a CLEAN ``lines`` run — so that a candidate which FIXES pre-existing red
    # gets that improvement reported rather than silently swallowed.
    baseline = compute_gate_baseline(
        repo, base_ref, gate, scope=scope, config=config, env=env
    )
    if proc.returncode == 0 and not baseline.readable:
        # Nothing failed, so no baseline answer could change the verdict.
        # Fail-closed governs cases where it COULD; here it cannot.
        return Check(
            gate.name,
            ok=True,
            seconds=seconds,
            detail=f"clean (improvement delta unavailable: {baseline.detail})",
        )
    return _compare_gate(gate, proc, tree, baseline, seconds)


def run_fast_gates(
    tree: Path,
    *,
    repo: Path,
    base_ref: str,
    scope: LaneScope,
    config: QueueConfig,
    base_config: QueueConfig | None,
    changed: list[str],
) -> GateResult:
    """Every fast-tier gate the MERGED tree declares, plus the dropped-gate check."""
    started = time.monotonic()
    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTEST_ADDOPTS"] = ""
    # CONCEPT:RM-TEST-COMMANDS / P0.6 invariant. `env = dict(os.environ)` above
    # copies whatever the QUEUE PROCESS'S own environment happens to hold —
    # including a `CARGO_TARGET_DIR`/`TMPDIR` set by an unrelated shell or a
    # misconfigured MCP server env. A shared CARGO_TARGET_DIR does not merely
    # serialize concurrent worktree builds, it CORRUPTS them (see
    # lane_doctor.py's `_check_cargo_partition`); leaving whatever the process
    # inherited in place made that a fact about the CALLER'S shell, i.e. prose,
    # not something this runner enforced. `partitioned_paths` is the same
    # lane-scoped allocator `lane_doctor.lane_exports()` uses, so every gate
    # command this queue runs gets an allocation identical in shape to what a
    # lane doctoring itself would compute — never a value the caller chose.
    parts = partitioned_paths(scope.tree)
    env["CARGO_TARGET_DIR"] = str(parts.cargo_target_dir)
    env["TMPDIR"] = str(parts.scratch_dir)
    # D-ORC-37: even though every tree gated here is a throwaway checkout, give
    # any pre-commit a store of its own so a crash can never orphan or lock
    # against another lane's. One line, both hazards.
    env["PRE_COMMIT_HOME"] = str(parts.scratch_dir / "merge-queue-precommit")
    checks: list[Check] = []
    dropped = _dropped_gates(config, base_config)
    if dropped:
        checks.append(
            Check(
                "declared-gates",
                ok=False,
                seconds=0.0,
                detail=(
                    "the merged tree DROPS gate(s) the base still declares: "
                    + ", ".join(dropped)
                    + " — deleting the check that guards an invariant is not a way "
                    "to satisfy it"
                ),
            )
        )
    for gate in config.fast_gates():
        checks.append(
            run_gate(
                gate,
                tree,
                repo=repo,
                base_ref=base_ref,
                scope=scope,
                config=config,
                changed=changed,
                env=env,
            )
        )
    return GateResult(
        ok=all(c.ok for c in checks),
        checks=checks,
        environment=_environment_signature(scope.main_tree, config),
        seconds=time.monotonic() - started,
    )


# ---------------------------------------------------------------------------
# Regenerate-on-land — behaviour 3
# ---------------------------------------------------------------------------
def _regenerate_generated_files(
    repo: Path, tree: str, *, scope: LaneScope, config: QueueConfig
) -> str:
    """Materialize *tree*, run the declared regenerators, return the new tree OID.

    The trial tree already holds BOTH sides' real content merged, so this is
    regeneration **from truth** — never a ``--theirs``/``--ours`` resolution,
    which silently drops whichever side lost. Fail-closed: a generator that exits
    non-zero raises rather than leaving a stale or half-regenerated tree.
    """
    throwaway = _commit_trial(repo, tree, [], "merge-queue: regenerate-on-land scratch")
    with materialized(repo, throwaway, scope=scope) as work_tree:
        for command in config.regenerate:
            try:
                proc = subprocess.run(  # noqa: S603 - argv from repo-declared config
                    list(command),
                    cwd=str(work_tree),
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=600,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise MergeQueueError(
                    f"regenerate-on-land: could not run {' '.join(command)}: {exc}"
                ) from exc
            if proc.returncode != 0:
                raise MergeQueueError(
                    f"regenerate-on-land: {' '.join(command)} exited {proc.returncode}: "
                    f"{proc.stderr.strip() or proc.stdout.strip()}"
                )
        _require_git(["add", "--", *sorted(config.generated_files)], work_tree)
        return _require_git(["write-tree"], work_tree)


def _resolve_generated_file_conflict(
    repo: Path, trial: TrialMerge, *, scope: LaneScope, config: QueueConfig
) -> TrialMerge:
    """Resolve a conflict confined to declared generated files by regenerating.

    Deliberately narrow: if even ONE conflicted path falls outside
    ``generated_files``, this is a real, human-resolvable conflict and the
    candidate is rejected exactly as before. Regeneration never touches a
    conflict that also involves hand-written content.
    """
    if trial.ok or not trial.conflicts or not config.generated_files:
        return trial
    if not set(trial.conflicts) <= config.generated_files:
        return trial
    return TrialMerge(
        ok=True,
        tree=_regenerate_generated_files(repo, trial.tree, scope=scope, config=config),
        detail=trial.detail,
    )


# ---------------------------------------------------------------------------
# Push-on-land (D-W3WPS-3) — wires the EXISTING gated push path, never a new one
# ---------------------------------------------------------------------------
def _push_landed_base(
    canonical: Path,
    *,
    base: str,
    canonical_on_base: bool,
    git: Any,
) -> dict[str, Any]:
    """Push the just-advanced base ref, honestly reporting whether it shipped.

    D-W3WPS-3: ``land`` used to report ``landed: true`` after only fast-forwarding
    the LOCAL canonical checkout, never touching the remote — 78 repos ended up
    with unpushed "landed" work. This closes that gap by reusing the EXISTING
    gated push path (:meth:`Git.push_project`, which already runs
    ``_gate_before_push`` — the repo's own pre-commit gates minus full pytest —
    plus its GH013/divergent-remote/tag-conflict handling) rather than
    re-implementing any push logic here.

    ``push_project`` pushes whichever branch the target working tree currently
    has checked out (plain ``git push --follow-tags``, no explicit refspec) —
    correct only when the canonical checkout genuinely IS on ``base`` (the
    ``merge --ff-only`` landing method, :func:`land`'s common/au-shaped case).
    When landing instead took the compare-and-swap path (canonical parked on
    some other branch — a bisect, an operator poking around, a merge in
    progress), pushing through that checkout would push the WRONG branch, so
    the push is deliberately DEFERRED rather than guessed at: ``pushed: False``
    with a reason, never silent, never a guess. A later ``phased_push`` (or a
    manual push once the canonical is back on ``base``) picks it up.

    Sync, not batched: this pushes the ONE repo :func:`land` just fast-forwarded,
    immediately, inline with the landing call. ``phased_push``'s phase ordering
    exists for the fleet-wide *release* workflow (coordinated version-bump
    pushes across interdependent packages, where a downstream repo's pin must
    not reach a remote before the upstream version it names does) — a plain
    merge-queue landing publishes a branch's own commits, not a version bump, so
    there is no cross-repo pin for immediate publication to violate. Should the
    queue ever land version-bump commits, that is the moment to route through
    ``phased_push`` instead; forcing every landing through fleet-wide phase
    bookkeeping today would be scope the defect never asked for.

    A push failure never fails the landing — the local ref genuinely did
    fast-forward, and reporting that as a failure too would misrepresent what
    happened. Landed-but-unpushed is a legitimate, visible intermediate state
    (network blip, GH013, a diverged remote needing a reviewed sync), not
    something to hide behind a single ambiguous flag.
    """
    if git is None or not hasattr(git, "push_project"):
        return {"pushed": False, "push_error": "no push-capable git handle provided"}
    if not canonical_on_base:
        return {
            "pushed": False,
            "push_error": (
                f"canonical checkout is not on {base} (landed via compare-and-swap "
                "update-ref); push deferred to avoid pushing whichever branch that "
                "checkout has checked out -- push_project/phased_push against the "
                f"canonical checkout once it is back on {base}"
            ),
        }
    try:
        push_result = git.push_project(path=str(canonical))
    except Exception as exc:  # pragma: no cover - defensive: push must never raise
        return {
            "pushed": False,
            "push_error": f"push_project raised {type(exc).__name__}: {exc}",
        }
    if getattr(push_result, "status", None) == "success":
        return {"pushed": True}
    err = getattr(push_result, "error", None)
    message = (
        getattr(err, "message", None)
        or (str(err) if err is not None else None)
        or getattr(push_result, "data", None)
        or "push_project failed"
    )
    return {"pushed": False, "push_error": str(message)}


# ---------------------------------------------------------------------------
# Landing — the DECLARED base ref only ever FAST-FORWARDS, under both guards
# ---------------------------------------------------------------------------
def _base_ref(base: str) -> str:
    """The candidate's declared base as a fully-qualified ref.

    Never the bare name. ``git rev-parse main`` resolves through the whole
    ``.git/refs`` search order, so a tag or a remote-tracking ref named ``main``
    can answer a question that was asked about the local branch — and every
    later step here (the CAS write, the post-condition) must name the SAME
    object the batch was built on.
    """
    return base if base.startswith("refs/") else f"refs/heads/{base}"


def _worktrees_holding(repo: Path, base: str) -> list[str]:
    """Every worktree that currently has *base* checked out.

    ``git update-ref`` will happily move a branch that a worktree has checked
    out, leaving that worktree's index and working tree silently inconsistent
    with its own HEAD — a lane working there would find its tree rewritten
    underneath it. Git refuses this for ``branch -f``/``checkout``; it does not
    for ``update-ref``, so the refusal has to be ours.
    """
    listing = _run_git(["worktree", "list", "--porcelain"], repo)
    if not listing.ok:
        return []
    holders: list[str] = []
    current = ""
    for line in listing.out.splitlines():
        if line.startswith("worktree "):
            current = line[len("worktree ") :].strip()
        elif line.strip() == f"branch {_base_ref(base)}":
            holders.append(current)
    return holders


def land(
    repo: Path, commit: str, *, base: str, scope: LaneScope, git: Any = None
) -> dict[str, Any]:
    """Advance the **declared base ref** to *commit*, fast-forward only.

    D-RMD-1 — **the bug this function is written against, and why it was
    invisible.** This used to run ``git merge --ff-only <commit>`` in the
    canonical checkout, which merges into whatever ``HEAD`` happens to be, not
    into ``base``. In agent-utilities the canonical checkout always sits on
    ``main``, so "merge into HEAD" *coincidentally* equalled "land on base" and
    seven consecutive landings were audited as genuinely correct. That is luck,
    not correctness, and the coincidence breaks precisely where this queue is
    now going: a canonical checkout parked on another branch (a merge in
    progress, a bisect, an operator poking around), a repo whose base is a
    release branch or a fork, or any repo whose checkout convention differs
    from au's.

    **The failure mode was the worst kind — silent and positive.** The queue
    reported ``landed``, the guarded prune then deleted the branch *as landed*,
    and the work sat on a ref nobody looks at. A rejection is loud and
    recoverable; this manufactured confidence and then destroyed the evidence.

    So both halves are implemented, and the second is the durable one:

    1. **Write the declared ref explicitly.** :func:`_base_ref` qualifies it, and
       the ref is moved either by ``git merge --ff-only`` (only when the
       canonical checkout genuinely has ``base`` checked out, so ref and working
       tree move atomically together) or by a compare-and-swap ``git update-ref
       <ref> <new> <expected-old>``. Fast-forward-only is enforced by asking
       ``merge-base --is-ancestor`` of the BASE REF — not of ``HEAD`` — because
       ``update-ref`` would otherwise happily rewind history.
    2. ★ **Assert the post-condition.** After the write, the base ref is re-read
       and must equal *commit*. This is what makes the fix durable rather than
       merely correct today: it would have caught D-RMD-1 **even with the wrong
       write target still in place**, and it catches the next variant of the
       same mistake for free. A queue that reports ``landed`` without proving
       the ref moved is the same defect family as a gate that reports green
       while enforcing nothing.

    **Two guards, not one.** ``lanes.guarded_tree_mutation`` refuses a tree
    holding uncommitted work another lane owns.
    :func:`~repository_manager.canonical_guard.guarded_canonical_mutation`
    additionally takes repository-manager's own canonical lease, so this can
    never interleave with repository-manager's background sync — which has
    destroyed ~20 minutes of a lane's work on a canonical tree before.

    Note the deploy consequence of the CAS path: when the canonical checkout is
    NOT on ``base``, the ref advances and the canonical *working tree* does not.
    The fleet hostPath-mounts that working tree, so landing in that state moves
    ``main`` without moving the deployed bytes — strictly safer than the
    alternative, and the same merge/deploy decoupling the promotion ref exists
    for.

    **D-W3WPS-3 — landing now also pushes, honestly.** This used to return
    ``landed: true`` after only fast-forwarding the LOCAL canonical checkout,
    never the remote — 78 repos ended up "landed" with nothing shipped. The
    returned dict now ALSO carries ``pushed`` (bool) and, when ``False``,
    ``push_error`` explaining why — see :func:`_push_landed_base`. A push
    failure never raises out of this function: the fast-forward genuinely
    happened, so it is reported as such; landed-but-unpushed is a legitimate,
    visible state, not a reason to pretend the landing itself failed.
    """
    canonical = scope.main_tree
    ref = _base_ref(base)
    before = _run_git(["rev-parse", "--verify", "--quiet", ref], repo)
    if not before.ok or not before.out:
        raise MergeQueueError(
            f"refusing to land: the declared base ref {ref!r} does not exist in "
            f"{repo}. The queue never falls back to HEAD — that fallback is "
            "exactly D-RMD-1, which reported `landed` while the base never moved."
        )
    current = before.out
    head = _run_git(["symbolic-ref", "--quiet", "HEAD"], canonical)
    canonical_on_base = head.ok and head.out.strip() == ref
    if current == commit:
        push = _push_landed_base(
            canonical, base=base, canonical_on_base=canonical_on_base, git=git
        )
        return {
            "base": base,
            "ref": ref,
            "from": current,
            "to": commit,
            "method": "already-current",
            **push,
        }
    if not _run_git(["merge-base", "--is-ancestor", current, commit], repo).ok:
        raise MergeQueueError(
            f"refusing to land: {commit[:12]} is not a descendant of {ref} "
            f"({current[:12]}), so advancing it would not be a fast-forward — "
            f"{base} moved after the batch was built; the candidates stay queued "
            "and the next run rebuilds against it"
        )

    holders = [h for h in _worktrees_holding(repo, base) if Path(h) != canonical]

    with guarded_tree_mutation(
        canonical, operation=f"land merge-queue batch onto {base}", owner=scope.lane
    ):
        if canonical_on_base:
            method = "merge --ff-only (canonical is on the base)"
            if git is not None:
                with guarded_canonical_mutation(
                    git, str(canonical), canonical.name, f"fast-forward {base}"
                ) as blocked:
                    if blocked is not None:
                        raise MergeQueueError(
                            f"canonical checkout refused the fast-forward: "
                            f"{blocked.get('error')} — the candidates stay queued"
                        )
                    res = _run_git(["merge", "--ff-only", commit], canonical)
            else:
                res = _run_git(["merge", "--ff-only", commit], canonical)
        elif holders:
            # Someone else's tree is ON this branch. Moving the ref out from
            # under it would desync their index and working tree — the same
            # class of harm as resetting a canonical checkout mid-work.
            raise MergeQueueError(
                f"refusing to land on {base}: it is checked out in "
                f"{', '.join(holders)}, and advancing the ref would leave that "
                "working tree inconsistent with its own HEAD. The candidates stay "
                "queued; free the branch or land from that tree."
            )
        else:
            # Compare-and-swap: the expected-old argument makes this atomic
            # against a concurrent writer — if anything moved the ref since it
            # was read, git refuses rather than clobbering.
            method = "update-ref CAS (canonical is not on the base)"
            res = _run_git(["update-ref", ref, commit, current], repo)
        if not res.ok:
            raise MergeQueueError(
                f"advancing {ref} to {commit[:12]} refused by git: "
                f"{res.err or res.out} — {base} moved after the batch was built; the "
                "candidates stay queued and the next run rebuilds against it"
            )

        # ★ THE POST-CONDITION (D-RMD-1's durable half). Re-read the ref and
        # prove it holds the object we computed, BEFORE anyone reports `landed`
        # and before the guarded prune deletes a branch on the strength of it.
        after = _run_git(["rev-parse", "--verify", "--quiet", ref], repo)
        if not after.ok or after.out != commit:
            raise MergeQueueError(
                f"POST-CONDITION FAILED: {ref} is {after.out[:12] or '<unreadable>'} "
                f"after the write, not the computed {commit[:12]} (it was "
                f"{current[:12]} before). NOTHING is reported as landed and no "
                "branch is pruned — a queue that says `landed` without proving the "
                "ref moved manufactures confidence and then deletes the evidence "
                "(D-RMD-1). The candidates stay queued."
            )
    # Push AFTER the guarded block has released its lease — push_project takes
    # its own (in-process, re-entrant-safe but unnecessary to nest) mutation
    # lock, and pushing a ref is not a tree mutation the canonical-checkout
    # guards above are protecting against.
    push = _push_landed_base(
        canonical, base=base, canonical_on_base=canonical_on_base, git=git
    )
    return {
        "base": base,
        "ref": ref,
        "from": current,
        "to": commit,
        "method": method,
        "verified": True,
        **push,
    }


# ---------------------------------------------------------------------------
# Guarded prune — behaviour 4, delegated to where the guard already lives
# ---------------------------------------------------------------------------
def prune_landed(
    candidate: Candidate, *, repo: Path, base: str, git: Any
) -> dict[str, Any]:
    """Remove a landed candidate's worktree and branch through the guarded path.

    Both halves are delegated to where the guard already lives — reimplementing
    either is exactly the D-ORC-21 duplication this module exists to end:

    * the **worktree** goes through
      :func:`~repository_manager.prune_guard.guarded_worktree_prune`, which
      refuses a tree holding uncommitted work, one with a merge/rebase in
      progress, or one a live lane still leases;
    * the **branch ref** goes through
      :meth:`~repository_manager.worktree.WorktreeManager.delete_merged_branch`,
      which re-reads the tip, re-asks ``git merge-base --is-ancestor`` **at
      delete time** rather than trusting an earlier scan, writes a
      ``refs/lane-backup/<branch>`` anchor first, and uses ``git branch -d`` —
      **never** ``-D``, so git re-decides reachability under its own ref lock,
      atomically with the delete.

    The worktree is removed **by the path the candidate recorded at enqueue
    time**, not by ``WorktreeManager.worktree_path()``. Lanes in this workspace
    create worktrees wherever they like (``/home/apps/worktrees/…``,
    ``$XDG_STATE_HOME/repository-worktrees/…``), and ``worktree_path()``
    reconstructs a location under ``WORKTREE_ROOT`` — for a lane that used
    another path it would either miss the real worktree or, when handed an
    absolute repo, silently resolve inside the repo itself. ``_prune_merged``
    already removes by actual path for the same reason.

    Fails closed at every step: any refusal returns ``pruned: False`` with a
    reason and leaves everything as found. An un-pruned branch is untidy; a
    wrongly-pruned one loses work, and the second is never an acceptable trade
    for the first.
    """
    from repository_manager import prune_guard
    from repository_manager.worktree import WorktreeManager

    result: dict[str, Any] = {"branch": candidate.branch, "pruned": False}
    worktree = Path(candidate.worktree).resolve() if candidate.worktree else None
    if worktree and worktree != repo.resolve() and worktree.is_dir():
        with prune_guard.guarded_worktree_prune(
            str(worktree), operation=f"prune landed candidate {candidate.branch}"
        ) as refusal:
            if refusal is not None:
                return {**result, "reason": f"worktree {worktree}: {refusal}"}
            removed = _run_git(["worktree", "remove", str(worktree)], repo)
            if not removed.ok:
                return {
                    **result,
                    "reason": f"git worktree remove {worktree} failed: {removed.err or removed.out}",
                }
            result["removed"] = str(worktree)
        _run_git(["worktree", "prune"], repo)

    deleted, reason, anchor = WorktreeManager(git).delete_merged_branch(
        str(repo), candidate.branch, base
    )
    result["branch_deleted"] = deleted
    result["pruned"] = deleted
    if reason:
        result["branch_kept_reason"] = reason
        result["reason"] = reason
    if anchor:
        result["branch_anchor"] = anchor
    return result


# ---------------------------------------------------------------------------
# The runner — optimistic batching with bisection on failure
# ---------------------------------------------------------------------------
def _build_chain(
    repo: Path,
    base: str,
    candidates: list[Candidate],
    *,
    scope: LaneScope,
    config: QueueConfig,
) -> tuple[str, list[Candidate], list[tuple[Candidate, TrialMerge]]]:
    """Merge each candidate onto a rolling trial commit; split off the conflicted.

    A conflicting candidate is identified precisely (it is the one whose
    ``merge-tree`` exited 1 against the rolling head) and dropped from the batch
    rather than poisoning it, so one lane's conflict never rejects seven innocent
    ones — unless the conflict is confined to declared generated files, in which
    case it is regenerated from the merged truth and the candidate proceeds.

    The chain is rooted at the FULLY-QUALIFIED base ref (:func:`_base_ref`), the
    same object :func:`land` compare-and-swaps from — a bare ``rev-parse main``
    could resolve a tag or a remote-tracking ref instead, and the batch would
    then be built on one object and landed against another (D-RMD-1's family).
    """
    head = _require_git(["rev-parse", _base_ref(base)], repo)
    accepted: list[Candidate] = []
    conflicted: list[tuple[Candidate, TrialMerge]] = []
    for candidate in candidates:
        trial = _resolve_generated_file_conflict(
            repo, trial_merge(repo, head, candidate.branch), scope=scope, config=config
        )
        if not trial.ok:
            conflicted.append((candidate, trial))
            continue
        tip = _require_git(["rev-parse", candidate.branch], repo)
        if tip == head or _run_git(["merge-base", "--is-ancestor", tip, head], repo).ok:
            accepted.append(candidate)  # already contained; nothing to land
            continue
        head = _commit_trial(
            repo,
            trial.tree,
            [head, tip],
            f"merge({candidate.lane}): {candidate.branch} via the merge queue",
        )
        accepted.append(candidate)
    return head, accepted, conflicted


def integrate_batch(
    candidates: list[Candidate],
    *,
    base: str,
    scope: LaneScope,
    config: QueueConfig,
    git: Any = None,
    depth: int = 0,
) -> list[dict[str, Any]]:
    """Gate *candidates* together; on failure, BISECT rather than serialize.

    Gating one candidate at a time makes throughput ``1 / gate_duration``, which
    a hundred concurrent lanes overrun immediately — and an overrun queue gets
    bypassed. Gating ``N`` together makes it ``N / gate_duration`` while a
    *failing* batch costs ``ceil(log2(N))`` extra runs to attribute. Most
    candidates pass, so the average cost is the batched one and the worst case is
    logarithmic.

    Returns one outcome record per candidate. Never raises for a candidate's own
    fault — a rejection is a *result*, carrying the evidence its lane needs.
    """
    repo = scope.main_tree
    if not candidates:
        return []
    head, accepted, conflicted = _build_chain(
        repo, base, candidates, scope=scope, config=config
    )
    outcomes: list[dict[str, Any]] = [
        {
            "branch": c.branch,
            "landed": False,
            "reason": (
                f"conflicts with the current {base} in: "
                f"{', '.join(t.conflicts) or 'unreported paths'} — sync {base} down "
                f"into {c.branch}, resolve there, then re-enqueue"
            ),
            "conflicts": t.conflicts,
        }
        for c, t in conflicted
    ]
    if not accepted:
        return outcomes

    changed = sorted({p for c in accepted for p in changed_paths(repo, base, c.branch)})
    base_config = config_at_ref(repo, base)
    with materialized(repo, head, scope=scope) as tree:
        merged_config = load_config(tree)
        gate = run_fast_gates(
            tree,
            repo=repo,
            base_ref=base,
            scope=scope,
            config=merged_config,
            base_config=base_config,
            changed=changed,
        )

    if gate.ok:
        landing = land(repo, head, base=base, scope=scope, git=git)
        return outcomes + [
            {
                "branch": c.branch,
                "landed": True,
                "batch_size": len(accepted),
                "gate": gate.as_dict(),
                **landing,
            }
            for c in accepted
        ]

    if len(accepted) == 1:
        outcomes.append(
            {
                "branch": accepted[0].branch,
                "landed": False,
                "reason": "; ".join(
                    f"{c.name}: {c.detail.splitlines()[0] if c.detail else 'failed'}"
                    for c in gate.failures()
                ),
                "gate": gate.as_dict(),
            }
        )
        return outcomes

    middle = len(accepted) // 2
    outcomes += integrate_batch(
        accepted[:middle],
        base=base,
        scope=scope,
        config=config,
        git=git,
        depth=depth + 1,
    )
    outcomes += integrate_batch(
        accepted[middle:],
        base=base,
        scope=scope,
        config=config,
        git=git,
        depth=depth + 1,
    )
    return outcomes


def run_queue(
    *,
    base: str = "",
    batch_size: int = 0,
    prune: bool = True,
    path: Path | str | None = None,
    git: Any = None,
) -> dict[str, Any]:
    """Drain up to *batch_size* candidates for ONE repository, under the lease.

    Raises :class:`~agent_utilities.governance.lanes.LeaseUnavailable` when
    another runner holds ``reconciliation-merge`` for this repository — deferring
    is the correct outcome and the caller must make it explicit, exactly as every
    other LEASE-class resource here. The lease is repo-scoped, so draining
    agent-utilities and epistemic-graph concurrently is safe by construction.
    """
    scope = lane_scope(path)
    repo = scope.main_tree
    config = _repo_config(scope)
    base = base or config.base
    batch_size = batch_size or config.batch_size
    if git is None:
        from repository_manager.repository_manager import Git

        git = Git(path=str(repo.parent))

    # Refuse a non-existent base ONCE, here, with the reason — rather than
    # letting it surface as a raw `git rev-parse ... ambiguous argument` from
    # deep inside the chain builder. Both refuse (there is no fallback to HEAD;
    # that fallback IS D-RMD-1), but only one of them tells the caller what to do.
    probe = _run_git(["rev-parse", "--verify", "--quiet", _base_ref(base)], repo)
    if not probe.ok or not probe.out:
        raise MergeQueueError(
            f"refusing to drain the {repo.name} queue: the declared base ref "
            f"{_base_ref(base)!r} does not exist. Check `base:` in "
            f"{config.source} (or --queue-base). The queue never falls back to "
            "whatever HEAD happens to be — that fallback reported `landed` while "
            "the base never moved (D-RMD-1)."
        )
    started = time.monotonic()
    with hold_lease(
        MERGE_LEASE, operation=f"drain the {repo.name} merge queue", path=scope.tree
    ):
        batch = queued(scope.tree)[:batch_size]
        if not batch:
            return {"repo": repo.name, "drained": 0, "outcomes": [], "seconds": 0.0}
        by_branch = {c.branch: c for c in batch}
        outcomes = integrate_batch(
            batch, base=base, scope=scope, config=config, git=git
        )
        for outcome in outcomes:
            candidate = by_branch.get(outcome["branch"])
            if candidate is None:
                continue
            if outcome["landed"]:
                # `landed` and `pushed` are deliberately distinct, honest fields
                # (D-W3WPS-3) -- record push status in `reason` too so it stays
                # visible durably via `status`/`queue_report`'s "recent" list,
                # not only in this one-time `run` response.
                landed_reason = (
                    "pushed"
                    if outcome.get("pushed")
                    else f"landed, NOT pushed: {outcome.get('push_error', 'unknown')}"
                )
                _record_state(candidate, LANDED, landed_reason, scope.tree)
                if prune:
                    outcome["prune"] = prune_landed(
                        candidate, repo=repo, base=base, git=git
                    )
            else:
                _record_state(candidate, REJECTED, outcome["reason"], scope.tree)
    return {
        "repo": repo.name,
        "config": config.source,
        "drained": len(outcomes),
        "landed": sum(1 for o in outcomes if o["landed"]),
        "rejected": sum(1 for o in outcomes if not o["landed"]),
        "pushed": sum(1 for o in outcomes if o.get("pushed")),
        "landed_unpushed": sum(
            1 for o in outcomes if o["landed"] and not o.get("pushed")
        ),
        "outcomes": outcomes,
        "seconds": round(time.monotonic() - started, 2),
    }


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action-routed entrypoint shared by the CLI and the MCP tool.

    Both surfaces marshal arguments and render results; neither holds logic. A
    new verb is one row here, not two implementations that can drift.
    """
    handlers = {
        "enqueue": lambda: enqueue(
            kwargs.get("branch", "") or "",
            base=kwargs.get("base", "") or "",
            worktree=kwargs.get("worktree"),
            path=kwargs.get("path"),
        ),
        "status": lambda: queue_report(kwargs.get("path")),
        "withdraw": lambda: withdraw(
            kwargs.get("branch", "") or "",
            reason=kwargs.get("reason", "") or "",
            path=kwargs.get("path"),
        ),
        "run": lambda: run_queue(
            base=kwargs.get("base", "") or "",
            batch_size=int(kwargs.get("batch_size") or 0),
            prune=bool(kwargs.get("prune", True)),
            path=kwargs.get("path"),
        ),
        "config": lambda: _config_report(kwargs.get("path")),
    }
    handler = handlers.get(action)
    if handler is None:
        return {
            "ok": False,
            "error": f"unknown merge-queue action: {action!r}",
            "actions": sorted(handlers),
        }
    return handler()


def _config_report(path: Path | str | None = None) -> dict[str, Any]:
    """Show (and validate) the gate declaration governing this repository.

    A first-class verb because "which gates actually guard this repo, and does
    the file parse" must be answerable WITHOUT enqueueing anything — a config
    that only fails at drain time fails while holding the lease.
    """
    scope = lane_scope(path)
    config = load_config(scope.main_tree)
    return {
        "repo": scope.main_tree.name,
        "source": config.source,
        "base": config.base,
        "batch_size": config.batch_size,
        "gates": [
            {
                "name": g.name,
                "command": list(g.command),
                "tier": g.tier,
                "timeout": g.timeout,
                "baseline_timeout": g.effective_baseline_timeout,
                "compare": g.compare,
                "when_changed": list(g.when_changed),
                "on_timeout": g.on_timeout,
                "runs_precommit": g.runs_precommit,
            }
            for g in config.gates
        ],
        "generated_files": sorted(config.generated_files),
        "regenerate": [list(c) for c in config.regenerate],
        "environment": _environment_signature(scope.main_tree, config),
    }


def main(argv: list[str] | None = None) -> int:
    """``python -m repository_manager.merge_queue [run|status|enqueue|withdraw|config]``

    A minimal, dependency-light driver suitable as a systemd/CronJob
    ``ExecStart``, so a scheduler can drive the queue without importing the full
    CLI surface. ``--path`` selects the repository (default: cwd), which is the
    whole cross-project story: one driver, N repositories, each with its own
    lease, store, and gates.
    """
    import argparse

    from agent_utilities.governance.lanes import LeaseUnavailable

    p = argparse.ArgumentParser(prog="python -m repository_manager.merge_queue")
    p.add_argument(
        "action",
        nargs="?",
        default="run",
        choices=["run", "status", "enqueue", "withdraw", "config"],
    )
    p.add_argument(
        "--path", default=None, help="a working tree of the target repository"
    )
    p.add_argument("--base", default="", help="branch to land onto (default: config)")
    p.add_argument("--branch", default="")
    p.add_argument("--reason", default="")
    p.add_argument("--batch-size", type=int, default=0)
    p.add_argument("--no-prune", action="store_true")
    args = p.parse_args(argv)
    try:
        out = dispatch(
            args.action,
            path=args.path,
            base=args.base,
            branch=args.branch,
            reason=args.reason,
            batch_size=args.batch_size,
            prune=not args.no_prune,
        )
    except LeaseUnavailable as exc:
        # Exit 75 (EX_TEMPFAIL), the same contract every other LEASE surface
        # uses, so a shell chained with `&&` stops instead of proceeding.
        print(json.dumps({"deferred": True, "holder": exc.holder}))
        return 75
    except LaneArbitrationError as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    print(json.dumps(out, default=str, indent=2))
    return 1 if out.get("rejected") or out.get("ok") is False else 0


if __name__ == "__main__":
    raise SystemExit(main())
