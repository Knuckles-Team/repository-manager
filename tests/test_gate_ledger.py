"""``GateLedger`` — proving the six rules its own docstring states, because
getting any of them wrong is SILENT: a lost row under concurrency looks like
a clean run, an un-cleared fixed test looks like a persistent failure, an
``unrunnable`` hook masquerading as a retest candidate looks like a code
defect, a stale row read without staleness applied looks like fresh
evidence, and a shippability check satisfied by a narrowed retest looks like
proof the whole suite is green when it never ran together at all. Each test
below is chosen because a wrong storage layer here reintroduces the exact
six-hour-per-push cost :mod:`repository_manager.gate_ledger` exists to
remove (see that module's own docstring for the incident).
"""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from repository_manager.gate_ledger import GateLedger

_REPO = "repo:fixture"
_STAGE = "pre_push"


def _passed_hook(hook_id: str = "pytest") -> dict[str, object]:
    return {"hook_id": hook_id, "outcome": "passed", "unrunnable": False}


def _failed_hook(hook_id: str = "pytest") -> dict[str, object]:
    return {"hook_id": hook_id, "outcome": "failed", "unrunnable": False}


def _unrunnable_hook(hook_id: str = "cargo-test") -> dict[str, object]:
    return {"hook_id": hook_id, "outcome": "could_not_run", "unrunnable": True}


def _record(
    ledger: GateLedger,
    *,
    scope: str = "full_wave",
    success: bool = True,
    hooks: list[dict[str, object]] | None = None,
    failing_tests: dict[str, list[str]] | None = None,
    git_sha: str = "",
) -> str:
    return ledger.record_run(
        repo_id=_REPO,
        repo_path="/repos/fixture",
        stage=_STAGE,
        scope=scope,
        trigger="test",
        success=success,
        exit_code=0 if success else 1,
        duration_s=0.1,
        hooks=hooks if hooks is not None else [_passed_hook()],
        failing_tests=failing_tests,
        git_sha=git_sha,
    )


# --------------------------------------------------------------------------- #
# Append-only under concurrency.
# --------------------------------------------------------------------------- #


def test_concurrent_writers_lose_no_rows_and_never_leak_database_is_locked(
    tmp_path: Path,
) -> None:
    """N independent connections (not N threads sharing one ``GateLedger`` —
    that would only prove Python's own ``RLock`` works) hammer the SAME file.
    This is where SQLite's file-level locking and the 30s busy timeout set in
    ``GateLedger._open`` actually matter: a connection that cannot acquire
    the write lock within the timeout raises ``sqlite3.OperationalError:
    database is locked`` — and because ``record_run`` swallows ALL storage
    exceptions and returns a ``run_id`` regardless (by design, so a ledger
    outage never looks like a gate failure), that exact failure mode would
    be invisible from the caller's side: a run_id handed back with no row
    behind it. The only way to catch it is to prove every returned run_id
    is actually present afterward.
    """

    path = tmp_path / "gate_ledger.sqlite3"
    n_workers = 8
    calls_per_worker = 6

    # Materialize the schema (and its one-time `ledger_meta` seed row) with a
    # single opener FIRST. `_assert_schema_version` reads-then-inserts that
    # row on a brand-new store; racing that read-then-insert across N
    # simultaneous FIRST opens is a real but SEPARATE initialization-race
    # concern this test is not about -- it would fail every worker's
    # `GateLedger(path)` call before a single `record_run` ever ran, which is
    # not "lost a write under contention", it is "never got to write at all".
    GateLedger(path)

    def worker(_: int) -> list[str]:
        ledger = GateLedger(path)
        return [_record(ledger, git_sha=f"sha-{_}-{i}") for i in range(calls_per_worker)]

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        results = list(pool.map(worker, range(n_workers)))

    run_ids = [rid for batch in results for rid in batch]
    assert len(run_ids) == n_workers * calls_per_worker
    assert len(set(run_ids)) == len(run_ids), "uuid4 run ids must never collide"

    verify = sqlite3.connect(str(path))
    try:
        (count,) = verify.execute("SELECT COUNT(*) FROM gate_runs").fetchone()
        present = {row[0] for row in verify.execute("SELECT run_id FROM gate_runs")}
    finally:
        verify.close()

    assert count == len(run_ids)
    assert present == set(run_ids), (
        "every run_id record_run returned must have a real row -- a mismatch "
        "here means a write was silently swallowed by lock contention"
    )


# --------------------------------------------------------------------------- #
# Clear-on-improve.
# --------------------------------------------------------------------------- #


def test_clear_on_improve_deletes_fixed_test_ids_from_test_latest(
    tmp_path: Path,
) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(
        ledger,
        success=False,
        hooks=[_failed_hook()],
        failing_tests={
            "pytest": ["tests/test_a.py::test_1", "tests/test_b.py::test_2"]
        },
        git_sha="sha1",
    )
    assert ledger.latest_failing_tests(_REPO, _STAGE, "pytest") == [
        "tests/test_a.py::test_1",
        "tests/test_b.py::test_2",
    ]

    # The hook re-runs (a narrower retest) and only test_2 still fails --
    # test_1 was fixed and must be cleared, not left marked failed forever.
    _record(
        ledger,
        scope="retest",
        success=False,
        hooks=[_failed_hook()],
        failing_tests={"pytest": ["tests/test_b.py::test_2"]},
        git_sha="sha2",
    )
    assert ledger.latest_failing_tests(_REPO, _STAGE, "pytest") == [
        "tests/test_b.py::test_2",
    ]


def test_clear_on_improve_an_all_clear_rerun_empties_the_set(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(
        ledger,
        success=False,
        hooks=[_failed_hook()],
        failing_tests={"pytest": ["tests/test_a.py::test_1"]},
        git_sha="sha1",
    )
    assert ledger.latest_failing_tests(_REPO, _STAGE, "pytest") == [
        "tests/test_a.py::test_1"
    ]

    # The hook re-runs, reports the hook itself as passed, and (per the
    # documented contract) supplies an EMPTY failing-tests sequence for it --
    # "this hook ran and found no failing test ids".
    _record(
        ledger,
        scope="retest",
        success=True,
        hooks=[_passed_hook()],
        failing_tests={"pytest": []},
        git_sha="sha2",
    )
    assert ledger.latest_failing_tests(_REPO, _STAGE, "pytest") == []


# --------------------------------------------------------------------------- #
# latest_failing_hooks excludes unrunnable.
# --------------------------------------------------------------------------- #


def test_latest_failing_hooks_excludes_unrunnable_hooks(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(
        ledger,
        success=False,
        hooks=[_failed_hook("pytest"), _unrunnable_hook("cargo-test")],
        git_sha="sha1",
    )
    failing = ledger.latest_failing_hooks(_REPO, _STAGE)
    assert [h.hook_id for h in failing] == ["pytest"]

    # Re-running the unrunnable hook's executable is still missing, so it
    # finds nothing again -- it must never join the retest candidate set
    # just because it ran a second time.
    _record(
        ledger,
        scope="retest",
        success=False,
        hooks=[_unrunnable_hook("cargo-test")],
        git_sha="sha1",
    )
    still_failing = ledger.latest_failing_hooks(_REPO, _STAGE)
    assert [h.hook_id for h in still_failing] == ["pytest"]

    # latest_hooks (unfiltered) does still see it, correctly labelled.
    all_hooks = {h.hook_id: h for h in ledger.latest_hooks(_REPO, _STAGE)}
    assert all_hooks["cargo-test"].unrunnable is True
    assert all_hooks["cargo-test"].failed is False


# --------------------------------------------------------------------------- #
# Staleness labelling.
# --------------------------------------------------------------------------- #


def test_staleness_is_labelled_against_the_callers_git_sha(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(ledger, hooks=[_passed_hook()], git_sha="sha1")

    fresh = ledger.latest_hooks(_REPO, _STAGE, git_sha="sha1")
    assert len(fresh) == 1
    assert fresh[0].stale is False

    stale = ledger.latest_hooks(_REPO, _STAGE, git_sha="sha2")
    assert len(stale) == 1
    assert stale[0].stale is True
    # The row itself is untouched -- staleness is computed, never stored.
    assert stale[0].git_sha == "sha1"

    # No sha supplied at all: nothing can be judged stale against "now".
    unlabelled = ledger.latest_hooks(_REPO, _STAGE)
    assert unlabelled[0].stale is False


def test_staleness_also_applies_to_latest_failing_hooks(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(ledger, success=False, hooks=[_failed_hook()], git_sha="sha1")

    [failing_fresh] = ledger.latest_failing_hooks(_REPO, _STAGE, git_sha="sha1")
    assert failing_fresh.stale is False

    [failing_stale] = ledger.latest_failing_hooks(_REPO, _STAGE, git_sha="sha-other")
    assert failing_stale.stale is True


# --------------------------------------------------------------------------- #
# is_shippable: only a passing full_wave row at the EXACT sha counts.
# --------------------------------------------------------------------------- #


def test_is_shippable_false_with_no_evidence_at_all(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    ok, why = ledger.is_shippable(_REPO, _STAGE, "sha-x")
    assert ok is False
    assert "sha-x"[:12] in why or "no full-wave run" in why


def test_is_shippable_false_with_only_retest_scope_rows(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(ledger, scope="retest", success=True, hooks=[_passed_hook()], git_sha="sha-x")
    ok, why = ledger.is_shippable(_REPO, _STAGE, "sha-x")
    assert ok is False
    assert "full-wave" in why


def test_is_shippable_false_when_full_wave_at_that_sha_failed(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(
        ledger,
        scope="full_wave",
        success=False,
        hooks=[_failed_hook()],
        git_sha="sha-x",
    )
    ok, why = ledger.is_shippable(_REPO, _STAGE, "sha-x")
    assert ok is False
    assert "failed" in why


def test_is_shippable_true_only_with_a_passing_full_wave_at_the_exact_sha(
    tmp_path: Path,
) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(ledger, scope="retest", success=True, hooks=[_passed_hook()], git_sha="sha-x")
    _record(
        ledger, scope="full_wave", success=False, hooks=[_failed_hook()], git_sha="sha-x"
    )
    _record(
        ledger, scope="full_wave", success=True, hooks=[_passed_hook()], git_sha="sha-x"
    )

    ok, why = ledger.is_shippable(_REPO, _STAGE, "sha-x")
    assert ok is True
    assert "passed" in why

    # A full-wave PASS recorded for a different commit must never satisfy
    # shippability for this one.
    other_ok, _ = ledger.is_shippable(_REPO, _STAGE, "sha-y")
    assert other_ok is False


def test_is_shippable_requires_a_sha_argument(tmp_path: Path) -> None:
    ledger = GateLedger(tmp_path / "gl.sqlite3")
    _record(ledger, scope="full_wave", success=True, hooks=[_passed_hook()], git_sha="sha-x")
    ok, why = ledger.is_shippable(_REPO, _STAGE, "")
    assert ok is False
    assert "sha" in why.lower()


# --------------------------------------------------------------------------- #
# Contract violations raise; storage failures never do.
# --------------------------------------------------------------------------- #


def test_unknown_outcome_raises_and_writes_nothing(tmp_path: Path) -> None:
    path = tmp_path / "gl.sqlite3"
    ledger = GateLedger(path)
    with pytest.raises(ValueError):
        _record(ledger, hooks=[{"hook_id": "pytest", "outcome": "bogus", "unrunnable": False}])

    verify = sqlite3.connect(str(path))
    try:
        (count,) = verify.execute("SELECT COUNT(*) FROM gate_runs").fetchone()
    finally:
        verify.close()
    assert count == 0, "validation must run BEFORE the transaction opens"


def test_passed_and_unrunnable_together_raises_and_writes_nothing(
    tmp_path: Path,
) -> None:
    path = tmp_path / "gl.sqlite3"
    ledger = GateLedger(path)
    with pytest.raises(ValueError):
        _record(
            ledger,
            hooks=[{"hook_id": "pytest", "outcome": "passed", "unrunnable": True}],
        )

    verify = sqlite3.connect(str(path))
    try:
        (count,) = verify.execute("SELECT COUNT(*) FROM gate_runs").fetchone()
    finally:
        verify.close()
    assert count == 0


def test_a_storage_failure_is_swallowed_never_raised(tmp_path: Path) -> None:
    """A closed connection is the cheapest reliable way to force every
    statement in ``record_run``'s transaction to fail -- standing in for any
    real storage outage (disk full, permissions, corruption). The contract
    is explicit in the module docstring: a ledger outage must never look
    like a gate failure, so this must return a run_id, not raise.
    """

    ledger = GateLedger(tmp_path / "gl.sqlite3")
    ledger._connection.close()
    run_id = _record(ledger)
    assert isinstance(run_id, str) and run_id


# --------------------------------------------------------------------------- #
# Schema-version mismatch refuses to open.
# --------------------------------------------------------------------------- #


def test_schema_version_mismatch_refuses_to_open(tmp_path: Path) -> None:
    path = tmp_path / "gl.sqlite3"
    GateLedger(path)  # creates a fresh store at the current schema version

    corrupt = sqlite3.connect(str(path))
    try:
        corrupt.execute(
            "UPDATE ledger_meta SET value = '999999' WHERE key = 'schema_version'"
        )
        corrupt.commit()
    finally:
        corrupt.close()

    with pytest.raises(RuntimeError):
        GateLedger(path)
