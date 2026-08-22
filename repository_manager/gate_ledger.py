"""Durable memory of what a gate actually found (CONCEPT:RM-GATE-LEDGER).

**The gap this closes.** ``rm_gates``' job store is a process-local ``dict`` in
:mod:`repository_manager.mcp_server`; it dies with the process. So the two
questions that make a fast release loop possible — *what failed last time?* and
*did the thing I just fixed actually get re-run?* — are not merely hard to
answer, they are unrepresentable. Measured 2026-08-21: one repository's push
consumed roughly six hours because each fix was validated by re-running a
90-minute wave rather than the six steps that had failed.

**What it is.** A local, best-effort SQLite projection, exactly the shape
:class:`repository_manager.lane_registry.LaneRegistry` and
:class:`repository_manager.capacity_store.CapacityStore` already established
(same XDG path root, WAL, ``synchronous=FULL``, monotonic ``version``). It is a
*record of observations*, never an authority: nothing may treat a ledger row as
permission to skip work it would otherwise do, and a ledger outage must never
look like a gate failure — every write is best-effort and swallowed.

**Append-only, because a crash mid-write is not hypothetical.** ``gate_runs``,
``hook_results`` and ``test_results`` are inserted once and never updated; each
insert is a single statement, so SQLite's own atomic commit means a row either
exists complete or does not exist. ``hook_latest``/``test_latest`` are the
monotonic projections callers actually read, guarded by ``version`` so two
concurrent writers cannot regress a newer verdict to an older one.

**Three rules that are easy to get wrong, stated because getting them wrong is
silent:**

*Clear-on-improve.* When a hook re-runs, ``test_latest`` rows for that
``(repo, stage, hook)`` whose test id is absent from the new failing set are
DELETED. Upsert-only would leave a fixed test marked failed forever.

*Staleness is a status, never a silent reuse.* A row recorded against a
different ``git_sha`` than the tree being asked about is stale. Callers get it
back labelled, and it is their job to refuse to call it evidence — this module
will not quietly pretend an old observation describes new code.

*This is a FAILURE ledger, not a pass matrix.* pytest in ``-q`` prints no line
per passing test, so ``test_results`` records only what failed. "Not present"
therefore means "not observed failing", which is emphatically not the same as
"observed passing". Every consumer must be written against that distinction.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from typing import Any

__all__ = [
    "GATE_LEDGER_SCHEMA_VERSION",
    "GateLedger",
    "LedgerHook",
    "default_gate_ledger",
]

#: Bumped on any incompatible schema change. A connection whose stored version
#: differs refuses rather than reading columns that may mean something else.
GATE_LEDGER_SCHEMA_VERSION = 1

#: The closed outcome vocabulary. ``passed``/``failed`` are ordinary verdicts;
#: every other member exists so that "we did not learn anything" can never be
#: written down as "it passed" — the failure that made a killed build report
#: ``Result=success`` and a 500-timeout storm report as 500 defects.
OUTCOMES = frozenset(
    {
        "passed",
        "failed",
        "did_not_run",
        "could_not_run",
        "crashed",
        "oom_killed",
        "timed_out",
        "environment_unproven",
    }
)

#: Outcomes that mean the hook produced a real verdict about the code.
_VERDICT_OUTCOMES = frozenset({"passed", "failed"})

_SCHEMA = """
CREATE TABLE IF NOT EXISTS ledger_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS gate_runs (
    run_id TEXT PRIMARY KEY,
    parent_run_id TEXT NOT NULL DEFAULT '',
    repo_id TEXT NOT NULL,
    repo_path TEXT NOT NULL,
    stage TEXT NOT NULL,
    scope TEXT NOT NULL,
    trigger TEXT NOT NULL,
    hook_ids_requested TEXT NOT NULL DEFAULT '',
    git_sha TEXT NOT NULL DEFAULT '',
    tree_dirty INTEGER NOT NULL DEFAULT 0,
    toolchain_fingerprint TEXT NOT NULL DEFAULT '',
    runner_kind TEXT NOT NULL DEFAULT '',
    hostname TEXT NOT NULL DEFAULT '',
    unit_name TEXT NOT NULL DEFAULT '',
    pid INTEGER NOT NULL DEFAULT 0,
    started_at TEXT NOT NULL,
    completed_at TEXT NOT NULL,
    duration_s REAL NOT NULL DEFAULT 0.0,
    success INTEGER NOT NULL,
    exit_code INTEGER NOT NULL DEFAULT 0,
    error TEXT NOT NULL DEFAULT '',
    written_at TEXT NOT NULL,
    version INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS gate_runs_repo_stage
    ON gate_runs(repo_id, stage, started_at);

CREATE TABLE IF NOT EXISTS hook_results (
    run_id TEXT NOT NULL,
    hook_id TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    outcome TEXT NOT NULL,
    unrunnable INTEGER NOT NULL DEFAULT 0,
    duration_s REAL,
    output_tail TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, hook_id)
);
CREATE INDEX IF NOT EXISTS hook_results_repo_stage_hook
    ON hook_results(repo_id, stage, hook_id, recorded_at);

CREATE TABLE IF NOT EXISTS hook_latest (
    repo_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    hook_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    unrunnable INTEGER NOT NULL DEFAULT 0,
    duration_s REAL,
    git_sha TEXT NOT NULL DEFAULT '',
    toolchain_fingerprint TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (repo_id, stage, hook_id)
);

CREATE TABLE IF NOT EXISTS test_results (
    run_id TEXT NOT NULL,
    repo_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    hook_id TEXT NOT NULL,
    test_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    recorded_at TEXT NOT NULL,
    PRIMARY KEY (run_id, hook_id, test_id)
);
CREATE INDEX IF NOT EXISTS test_results_repo_stage_test
    ON test_results(repo_id, stage, test_id, recorded_at);

CREATE TABLE IF NOT EXISTS test_latest (
    repo_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    hook_id TEXT NOT NULL,
    test_id TEXT NOT NULL,
    run_id TEXT NOT NULL,
    outcome TEXT NOT NULL,
    git_sha TEXT NOT NULL DEFAULT '',
    recorded_at TEXT NOT NULL,
    version INTEGER NOT NULL,
    PRIMARY KEY (repo_id, stage, hook_id, test_id)
);
"""


class LedgerHook:
    """One hook's recorded observation, with staleness resolved for a caller.

    ``stale`` is computed against the sha the caller asked about, not stored —
    the same row is fresh evidence for the commit it ran on and stale evidence
    for any other, and only the caller knows which it is asking about.
    """

    __slots__ = (
        "duration_s",
        "git_sha",
        "hook_id",
        "outcome",
        "recorded_at",
        "run_id",
        "stale",
        "toolchain_fingerprint",
        "unrunnable",
    )

    def __init__(self, row: sqlite3.Row, *, stale: bool) -> None:
        self.hook_id = row["hook_id"]
        self.outcome = row["outcome"]
        self.unrunnable = bool(row["unrunnable"])
        self.duration_s = row["duration_s"]
        self.git_sha = row["git_sha"]
        self.toolchain_fingerprint = row["toolchain_fingerprint"]
        self.recorded_at = row["recorded_at"]
        self.run_id = row["run_id"]
        self.stale = stale

    @property
    def failed(self) -> bool:
        """Did this hook produce a real negative verdict about the code?

        ``unrunnable`` is deliberately excluded. A hook whose executable was
        missing found nothing; re-running it in the same environment will find
        nothing again. Treating it as a retest candidate is how a missing
        toolchain masquerades as a code defect.
        """

        return self.outcome == "failed" and not self.unrunnable

    def as_dict(self) -> dict[str, Any]:
        return {
            "hook_id": self.hook_id,
            "outcome": self.outcome,
            "unrunnable": self.unrunnable,
            "duration_s": self.duration_s,
            "git_sha": self.git_sha,
            "recorded_at": self.recorded_at,
            "run_id": self.run_id,
            "stale": self.stale,
        }


class GateLedger:
    """SQLite-backed record of gate observations. A projection, not an authority."""

    def __init__(self, store_path: str | Path | None = None) -> None:
        self.store_path = self._resolve_store_path(store_path)
        self._lock = RLock()
        self._connection = self._open(self.store_path)
        with self._lock:
            self._connection.executescript(_SCHEMA)
            self._assert_schema_version()
            self._connection.commit()

    # -- storage plumbing: identical shape to CapacityStore/LaneRegistry ----

    @staticmethod
    def _resolve_store_path(store_path: str | Path | None) -> str:
        if store_path is None:
            state_root = Path(
                os.getenv("XDG_STATE_HOME", str(Path.home() / ".local" / "state"))
            )
            path = state_root / "repository-manager" / "gate_ledger.sqlite3"
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        if str(store_path) == ":memory:":
            return ":memory:"
        path = Path(store_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return str(path.resolve(strict=False))

    @staticmethod
    def _open(path: str) -> sqlite3.Connection:
        connection = sqlite3.connect(
            path, timeout=30, isolation_level=None, check_same_thread=False
        )
        connection.row_factory = sqlite3.Row
        if path != ":memory:":
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("PRAGMA synchronous = FULL")
        return connection

    def _assert_schema_version(self) -> None:
        """Refuse a store written by an incompatible build.

        Reading unknown columns as if they meant what this build expects is how
        a schema change becomes silent data corruption instead of a loud stop.
        """

        row = self._connection.execute(
            "SELECT value FROM ledger_meta WHERE key = 'schema_version'"
        ).fetchone()
        if row is None:
            self._connection.execute(
                "INSERT INTO ledger_meta(key, value) VALUES('schema_version', ?)",
                (str(GATE_LEDGER_SCHEMA_VERSION),),
            )
            return
        stored = int(row["value"])
        if stored != GATE_LEDGER_SCHEMA_VERSION:
            raise RuntimeError(
                f"gate ledger at {self.store_path} is schema v{stored}; this build "
                f"speaks v{GATE_LEDGER_SCHEMA_VERSION}. Refusing to read it rather "
                f"than risk interpreting its columns as something they are not."
            )

    # -- writes -------------------------------------------------------------

    def record_run(
        self,
        *,
        repo_id: str,
        repo_path: str,
        stage: str,
        scope: str,
        trigger: str,
        success: bool,
        exit_code: int,
        duration_s: float,
        hooks: Sequence[dict[str, Any]],
        failing_tests: Mapping[str, Sequence[str]] | None = None,
        hook_ids_requested: Sequence[str] | None = None,
        git_sha: str = "",
        tree_dirty: bool = False,
        toolchain_fingerprint: str = "",
        runner_kind: str = "",
        hostname: str = "",
        unit_name: str = "",
        pid: int = 0,
        error: str = "",
        parent_run_id: str = "",
        started_at: str | None = None,
    ) -> str:
        """Record one gate run and return its ``run_id``.

        ``hooks`` entries are ``{"hook_id", "outcome", "unrunnable",
        "duration_s", "output_tail"}``. ``failing_tests`` maps a hook id to the
        test ids it reported failing; a hook present as a key with an empty
        sequence means "this hook ran and reported no failing test ids", which
        is what drives clear-on-improve.

        **Storage failures are swallowed; contract violations are not.** A ledger
        that cannot write is a lost observation, and turning that into a failed
        gate would be worse than having no ledger — so sqlite errors are caught
        and the run id returned anyway. But a caller handing us an outcome
        outside :data:`OUTCOMES`, or a hook marked both ``passed`` and
        ``unrunnable``, is a bug in the caller, and swallowing it would mean the
        guard exists only decoratively. Those are validated BEFORE the
        transaction opens and raise. (Found the hard way: the first version of
        this method wrapped validation in the same ``except`` and the guard
        silently did nothing.)
        """

        self._validate_hooks(hooks)
        run_id = uuid.uuid4().hex
        now = datetime.now(UTC).isoformat()
        try:
            with self._lock:
                self._connection.execute("BEGIN IMMEDIATE")
                try:
                    version = self._next_version()
                    self._connection.execute(
                        "INSERT INTO gate_runs("
                        " run_id, parent_run_id, repo_id, repo_path, stage, scope,"
                        " trigger, hook_ids_requested, git_sha, tree_dirty,"
                        " toolchain_fingerprint, runner_kind, hostname, unit_name,"
                        " pid, started_at, completed_at, duration_s, success,"
                        " exit_code, error, written_at, version"
                        ") VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                        (
                            run_id,
                            parent_run_id,
                            repo_id,
                            repo_path,
                            stage,
                            scope,
                            trigger,
                            json.dumps(list(hook_ids_requested or []), sort_keys=True),
                            git_sha,
                            int(bool(tree_dirty)),
                            toolchain_fingerprint,
                            runner_kind,
                            hostname,
                            unit_name,
                            int(pid),
                            started_at or now,
                            now,
                            float(duration_s),
                            int(bool(success)),
                            int(exit_code),
                            error,
                            now,
                            version,
                        ),
                    )
                    for hook in hooks:
                        self._record_hook(
                            run_id,
                            repo_id,
                            stage,
                            hook,
                            now,
                            version,
                            git_sha,
                            toolchain_fingerprint,
                        )
                    for hook_id, test_ids in (failing_tests or {}).items():
                        self._record_tests(
                            run_id,
                            repo_id,
                            stage,
                            hook_id,
                            test_ids,
                            now,
                            version,
                            git_sha,
                        )
                    self._connection.execute("COMMIT")
                except Exception:
                    self._connection.execute("ROLLBACK")
                    raise
        except Exception:  # noqa: BLE001 - a projection outage is not a gate verdict
            return run_id
        return run_id

    def _next_version(self) -> int:
        row = self._connection.execute(
            "SELECT COALESCE(MAX(version), 0) AS v FROM gate_runs"
        ).fetchone()
        return int(row["v"]) + 1

    @staticmethod
    def _validate_hooks(hooks: Sequence[dict[str, Any]]) -> None:
        """Reject a malformed observation before anything is written.

        Runs outside the write transaction precisely so its errors are NOT
        caught by :meth:`record_run`'s storage-failure handler.
        """

        for hook in hooks:
            hook_id = hook.get("hook_id")
            if not str(hook_id or "").strip():
                raise ValueError("gate ledger refuses a hook with no hook_id")
            outcome = str(hook.get("outcome", "")).strip()
            if outcome not in OUTCOMES:
                raise ValueError(
                    f"gate ledger refuses outcome {outcome!r} for hook {hook_id!r}: "
                    f"not one of {sorted(OUTCOMES)}. An unrecognised outcome must "
                    f"stop the write, never be coerced into a neighbouring one."
                )
            if outcome == "passed" and bool(hook.get("unrunnable", False)):
                raise ValueError(
                    f"hook {hook_id!r} recorded as passed AND unrunnable: a hook "
                    f"whose executable was missing cannot have produced a passing "
                    f"verdict, and recording one is how a missing toolchain becomes "
                    f"a green gate."
                )

    def _record_hook(
        self,
        run_id: str,
        repo_id: str,
        stage: str,
        hook: dict[str, Any],
        now: str,
        version: int,
        git_sha: str,
        toolchain_fingerprint: str,
    ) -> None:
        outcome = str(hook.get("outcome", "")).strip()
        hook_id = str(hook["hook_id"])
        unrunnable = int(bool(hook.get("unrunnable", False)))
        self._connection.execute(
            "INSERT OR REPLACE INTO hook_results("
            " run_id, hook_id, repo_id, stage, outcome, unrunnable, duration_s,"
            " output_tail, recorded_at) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                run_id,
                hook_id,
                repo_id,
                stage,
                outcome,
                unrunnable,
                hook.get("duration_s"),
                str(hook.get("output_tail", ""))[:4000],
                now,
            ),
        )
        # Monotonic projection: a slower concurrent writer must not regress a
        # newer verdict, so the guard is on version, not on recency of arrival.
        self._connection.execute(
            "INSERT INTO hook_latest("
            " repo_id, stage, hook_id, run_id, outcome, unrunnable, duration_s,"
            " git_sha, toolchain_fingerprint, recorded_at, version)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)"
            " ON CONFLICT(repo_id, stage, hook_id) DO UPDATE SET"
            "  run_id=excluded.run_id, outcome=excluded.outcome,"
            "  unrunnable=excluded.unrunnable, duration_s=excluded.duration_s,"
            "  git_sha=excluded.git_sha,"
            "  toolchain_fingerprint=excluded.toolchain_fingerprint,"
            "  recorded_at=excluded.recorded_at, version=excluded.version"
            " WHERE excluded.version > hook_latest.version",
            (
                repo_id,
                stage,
                hook_id,
                run_id,
                outcome,
                unrunnable,
                hook.get("duration_s"),
                git_sha,
                toolchain_fingerprint,
                now,
                version,
            ),
        )

    def _record_tests(
        self,
        run_id: str,
        repo_id: str,
        stage: str,
        hook_id: str,
        test_ids: Sequence[str],
        now: str,
        version: int,
        git_sha: str,
    ) -> None:
        fresh = sorted({str(t) for t in test_ids if str(t).strip()})
        for test_id in fresh:
            self._connection.execute(
                "INSERT OR REPLACE INTO test_results("
                " run_id, repo_id, stage, hook_id, test_id, outcome, recorded_at)"
                " VALUES (?,?,?,?,?,?,?)",
                (run_id, repo_id, stage, hook_id, test_id, "failed", now),
            )
            self._connection.execute(
                "INSERT INTO test_latest("
                " repo_id, stage, hook_id, test_id, run_id, outcome, git_sha,"
                " recorded_at, version) VALUES (?,?,?,?,?,?,?,?,?)"
                " ON CONFLICT(repo_id, stage, hook_id, test_id) DO UPDATE SET"
                "  run_id=excluded.run_id, outcome=excluded.outcome,"
                "  git_sha=excluded.git_sha, recorded_at=excluded.recorded_at,"
                "  version=excluded.version"
                " WHERE excluded.version > test_latest.version",
                (
                    repo_id,
                    stage,
                    hook_id,
                    test_id,
                    run_id,
                    "failed",
                    git_sha,
                    now,
                    version,
                ),
            )
        # Clear-on-improve. Anything this hook previously reported failing that
        # it did NOT report this time has stopped failing (fixed, renamed, or
        # removed) and must not stay marked failed forever. Version-guarded so
        # a stale writer cannot erase a newer run's findings.
        placeholders = ",".join("?" for _ in fresh)
        query = (
            "DELETE FROM test_latest WHERE repo_id=? AND stage=? AND hook_id=?"
            " AND version < ?"
        )
        params: list[Any] = [repo_id, stage, hook_id, version]
        if fresh:
            query += f" AND test_id NOT IN ({placeholders})"
            params.extend(fresh)
        self._connection.execute(query, params)

    # -- reads --------------------------------------------------------------

    def has_any_run(self, repo_id: str, stage: str) -> bool:
        with self._lock:
            row = self._connection.execute(
                "SELECT 1 FROM gate_runs WHERE repo_id=? AND stage=? LIMIT 1",
                (repo_id, stage),
            ).fetchone()
        return row is not None

    def latest_hooks(
        self, repo_id: str, stage: str, *, git_sha: str = ""
    ) -> list[LedgerHook]:
        """Every hook's most recent observation, each labelled stale or not."""

        with self._lock:
            rows = self._connection.execute(
                "SELECT * FROM hook_latest WHERE repo_id=? AND stage=?"
                " ORDER BY hook_id",
                (repo_id, stage),
            ).fetchall()
        return [
            LedgerHook(row, stale=bool(git_sha) and row["git_sha"] != git_sha)
            for row in rows
        ]

    def latest_failing_hooks(
        self, repo_id: str, stage: str, *, git_sha: str = ""
    ) -> list[LedgerHook]:
        """The retest candidate set: hooks that produced a real failed verdict.

        Excludes ``unrunnable`` hooks — see :attr:`LedgerHook.failed`.
        """

        return [
            h for h in self.latest_hooks(repo_id, stage, git_sha=git_sha) if h.failed
        ]

    def latest_failing_tests(self, repo_id: str, stage: str, hook_id: str) -> list[str]:
        with self._lock:
            rows = self._connection.execute(
                "SELECT test_id FROM test_latest"
                " WHERE repo_id=? AND stage=? AND hook_id=? ORDER BY test_id",
                (repo_id, stage, hook_id),
            ).fetchall()
        return [row["test_id"] for row in rows]

    def run(self, run_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT * FROM gate_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        return dict(row) if row is not None else None

    def is_shippable(self, repo_id: str, stage: str, git_sha: str) -> tuple[bool, str]:
        """Is there full-wave evidence for THIS commit? Returns (verdict, why).

        A ``retest`` pass never satisfies this on its own. Retest re-runs a
        narrowed set, so by construction it cannot observe an interaction that
        only appears when the whole suite runs together — which is exactly the
        shape of the deadlock that survived 95 clean isolated runs on
        2026-08-21. Only a ``full_wave`` run at the current sha counts.
        """

        if not git_sha:
            return False, "no commit sha supplied; nothing can be proven about 'now'"
        with self._lock:
            row = self._connection.execute(
                "SELECT success, run_id FROM gate_runs"
                " WHERE repo_id=? AND stage=? AND scope='full_wave' AND git_sha=?"
                " ORDER BY version DESC LIMIT 1",
                (repo_id, stage, git_sha),
            ).fetchone()
        if row is None:
            return False, (
                f"no full-wave run recorded at {git_sha[:12]}; a retest-only pass "
                f"is not evidence the suite is green"
            )
        if not row["success"]:
            return False, f"full-wave run {row['run_id'][:12]} at {git_sha[:12]} failed"
        return True, f"full-wave run {row['run_id'][:12]} passed at {git_sha[:12]}"


_DEFAULT: GateLedger | None = None
_DEFAULT_LOCK = RLock()


def default_gate_ledger() -> GateLedger:
    """Process-wide ledger, lazily constructed."""

    global _DEFAULT
    with _DEFAULT_LOCK:
        if _DEFAULT is None:
            _DEFAULT = GateLedger()
        return _DEFAULT
