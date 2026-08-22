"""Tests for :mod:`repository_manager.gate_runner`'s ``retest`` action.

``run``/``status``/``explain``/``profile`` are ported VERBATIM from what used
to be inline in ``mcp_tools/gates.py`` -- their end-to-end behavior (real
``pre-commit`` subprocess, real MCP tool call path) is already proven by
``tests/test_mcp_gates_tool.py`` and ``tests/test_gates.py``, both of which
still pass unmodified against this refactor (the regression check this file's
author ran BEFORE writing anything below, to separate a port bug from a
new-feature bug).

This file is about ``retest``'s new ledger-driven decision logic: what
baseline it reports, which hook ids (if any) it narrows to, whether it
degrades a stale baseline to the full wave, and whether an all-pass narrowed
retest escalates to a second, full-wave job. ``repository_manager.gates.
run_gate_stage`` is stubbed via monkeypatch (module-global lookup, so it
still applies through ``gate_runner.escalating_run_gate_stage``'s nested
call) -- exercising the real decision/submission wiring in milliseconds
rather than shelling out to real pre-commit.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import pytest

from repository_manager import build_queue, gate_runner
from repository_manager.gate_ledger import GateLedger
from repository_manager.scan_models import HookResult, RepoScanResult
from tests.conftest import isolated_git_subprocess_env


def _repo_id(path: Path) -> str:
    """The SAME identity ``gate_runner._retest_plan`` computes for ``path``.

    ``run_gate_stage`` (see ``gates.py``) now records ledger rows keyed by
    ``build_queue.stable_repository_id``, not a display basename -- tests
    must record baselines under that same identity or every lookup below
    would silently see "no prior run" regardless of what was recorded.
    """
    return build_queue.stable_repository_id(str(path))


def _init_git_repo(path: Path) -> None:
    env = isolated_git_subprocess_env()
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.email", "a@b.c"], cwd=path, check=True, env=env
    )  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=path, check=True, env=env
    )  # nosec B603 B607
    (path / ".pre-commit-config.yaml").write_text("repos: []\n")
    (path / "file.txt").write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, env=env)  # nosec B603 B607
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True, env=env)  # nosec B603 B607


def _new_commit(path: Path) -> None:
    """Advance HEAD -- used to make a previously-recorded ledger row stale."""
    env = isolated_git_subprocess_env()
    (path / "file.txt").write_text("changed\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, env=env)  # nosec B603 B607
    subprocess.run(
        ["git", "commit", "-q", "-m", "second"], cwd=path, check=True, env=env
    )  # nosec B603 B607


def _head_sha(path: Path) -> str:
    env = isolated_git_subprocess_env()
    completed = subprocess.run(  # nosec B603 B607
        ["git", "rev-parse", "HEAD"],
        cwd=path,
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return completed.stdout.strip()


def _resolve_targets_for(targets: list[tuple[str, str]]):
    def _resolve(threads: int | None, repos: str | None) -> list[tuple[str, str]]:
        del threads, repos
        return targets

    return _resolve


class _FakeRunGateStage:
    """Stand-in for ``repository_manager.gates.run_gate_stage``.

    ``script`` maps ``tuple(sorted(hook_ids or ()))`` to a pass/fail bool;
    unlisted keys default to passing. Every call is recorded (repo_path,
    hook_ids tuple) so tests can assert exactly what got re-run.
    """

    def __init__(self, script: dict[tuple[str, ...], bool] | None = None) -> None:
        self.script = script or {}
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def __call__(
        self,
        repo_path: str,
        stage: str,
        *,
        hook_ids: list[str] | None = None,
        timeout: int | None = None,
        trigger: str = "run",
        scope: str = "full_wave",
        colocated: bool | None = None,
        record: bool = True,
    ) -> RepoScanResult:
        del timeout, trigger, scope, colocated, record
        key = tuple(sorted(hook_ids or ()))
        self.calls.append((repo_path, key))
        success = self.script.get(key, True)
        ran_ids = list(hook_ids or ["all"])
        hooks = [HookResult(hook_id=h, passed=success) for h in ran_ids]
        return RepoScanResult(
            repo_path=repo_path,
            success=success,
            exit_code=0 if success else 1,
            hooks=hooks,
            stage=stage,
        )


class _RecordingSubmit:
    """A real :class:`gate_runner.LocalJobStore`-backed ``submit_one``.

    Exercises the actual nested-escalation call path (``submit_one`` called
    again, from inside ``escalating_run_gate_stage``, when the narrowed
    retest passes) rather than a mock that can't reproduce that wiring.
    """

    def __init__(self) -> None:
        self.store = gate_runner.LocalJobStore()

    def __call__(
        self,
        repo_name: str,
        path: str,
        *,
        hook_ids: list[str] | None = None,
        trigger: str = "run",
        scope: str = "full_wave",
        _escalate_on_pass: bool = False,
        _same_node: bool = False,
    ) -> dict[str, Any]:
        extra_job_data = {
            "repo_name": repo_name,
            "stage": "fast",
            "trigger": trigger,
            "scope": scope,
            "hook_ids_requested": list(hook_ids or []),
            "same_node": _same_node,
        }
        if _escalate_on_pass:
            return self.store.submit_job(
                "gate",
                gate_runner.escalating_run_gate_stage,
                path,
                "fast",
                hook_ids,
                timeout=None,
                escalate_on_pass=True,
                repo_name=repo_name,
                submit_one=self,
                same_node=_same_node,
                trigger=trigger,
                scope=scope,
                colocated=_same_node,
                record=True,
                _extra_job_data=extra_job_data,
            )
        return self.store.submit_job(
            "gate",
            gate_runner.run_gate_stage,
            path,
            "fast",
            timeout=None,
            hook_ids=hook_ids,
            trigger=trigger,
            scope=scope,
            colocated=_same_node,
            record=True,
            _extra_job_data=extra_job_data,
        )


@pytest.fixture
def ledger(tmp_path: Path) -> GateLedger:
    return GateLedger(store_path=tmp_path / "gate_ledger.sqlite3")


def _record(
    ledger: GateLedger,
    *,
    repo_id: str,
    stage: str,
    git_sha: str,
    hooks: list[dict[str, Any]],
    success: bool,
) -> None:
    ledger.record_run(
        repo_id=repo_id,
        repo_path=f"/fake/{repo_id}",
        stage=stage,
        scope="full_wave",
        trigger="run",
        success=success,
        exit_code=0 if success else 1,
        duration_s=1.0,
        hooks=hooks,
        git_sha=git_sha,
    )


def test_retest_missing_baseline_runs_full_wave(tmp_path, ledger, monkeypatch):
    """No prior run at all -> full wave, and the response says 'missing'."""
    repo_path = tmp_path / "repo-a"
    _init_git_repo(repo_path)
    fake = _FakeRunGateStage()
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    result = gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for([("repo-a", str(repo_path))]),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
    )

    entry = result["targets"]["repo-a"]
    assert entry["baseline"] == "missing"
    assert entry["retest_hook_ids"] is None
    assert entry["retest_job_id"] is not None
    assert entry["stale"] is False
    assert entry["escalate"] is False  # nothing narrowed, nothing to escalate from
    assert fake.calls == [(str(repo_path), ())]  # full wave: hook_ids=None -> ()


def test_retest_clean_baseline_submits_nothing(tmp_path, ledger, monkeypatch):
    """A recorded run with nothing failing -> no job submitted at all."""
    repo_path = tmp_path / "repo-b"
    _init_git_repo(repo_path)
    sha = _head_sha(repo_path)
    _record(
        ledger,
        repo_id=_repo_id(repo_path),
        stage="fast",
        git_sha=sha,
        hooks=[{"hook_id": "lint", "outcome": "passed"}],
        success=True,
    )
    fake = _FakeRunGateStage()
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    result = gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for([("repo-b", str(repo_path))]),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
    )

    entry = result["targets"]["repo-b"]
    assert entry["baseline"] == "clean"
    assert entry["retest_job_id"] is None
    assert entry["stale"] is False
    assert fake.calls == []  # nothing submitted, nothing ran
    assert result["status"] == "clean"


def test_retest_failing_hooks_narrows_to_exactly_those(tmp_path, ledger, monkeypatch):
    """A recorded run with failures -> only the failing hook ids are re-run."""
    repo_path = tmp_path / "repo-c"
    _init_git_repo(repo_path)
    sha = _head_sha(repo_path)
    _record(
        ledger,
        repo_id=_repo_id(repo_path),
        stage="fast",
        git_sha=sha,
        hooks=[
            {"hook_id": "lint", "outcome": "passed"},
            {"hook_id": "mypy", "outcome": "failed"},
            {"hook_id": "pytest", "outcome": "failed"},
        ],
        success=False,
    )
    fake = _FakeRunGateStage()  # everything passes this time
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    result = gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for([("repo-c", str(repo_path))]),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
        escalate=False,  # isolate narrowing from the separate escalation test
    )

    entry = result["targets"]["repo-c"]
    assert entry["baseline"] == "failing"
    assert entry["retest_hook_ids"] == ["mypy", "pytest"]
    assert entry["retest_job_id"] is not None
    assert entry["escalate"] is False
    assert fake.calls == [(str(repo_path), ("mypy", "pytest"))]


def test_retest_escalates_full_wave_on_all_pass(tmp_path, ledger, monkeypatch):
    """An all-pass narrowed retest submits a SECOND full-wave job automatically."""
    repo_path = tmp_path / "repo-d"
    _init_git_repo(repo_path)
    sha = _head_sha(repo_path)
    _record(
        ledger,
        repo_id=_repo_id(repo_path),
        stage="fast",
        git_sha=sha,
        hooks=[{"hook_id": "mypy", "outcome": "failed"}],
        success=False,
    )
    # Both the narrowed call (hook_ids=("mypy",)) and the escalated full-wave
    # call (hook_ids=()) pass.
    fake = _FakeRunGateStage(script={("mypy",): True, (): True})
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    result = gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for([("repo-d", str(repo_path))]),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
        escalate=True,
    )

    entry = result["targets"]["repo-d"]
    assert entry["retest_hook_ids"] == ["mypy"]
    assert entry["escalate"] is True
    # The narrowed job ran synchronously (LocalJobStore), so by the time
    # dispatch() returns, the escalation it triggered has ALSO already run.
    assert fake.calls == [(str(repo_path), ("mypy",)), (str(repo_path), ())]

    with submit.store.jobs_lock:
        triggers = sorted(
            job["trigger"]
            for job in submit.store.jobs.values()
            if job.get("repo_name") == "repo-d"
        )
    assert triggers == ["retest", "retest-escalate"]


def test_retest_no_escalation_when_narrowed_retest_still_fails(
    tmp_path, ledger, monkeypatch
):
    """Escalation only fires on an ALL-PASS narrowed retest, never on a failure."""
    repo_path = tmp_path / "repo-e"
    _init_git_repo(repo_path)
    sha = _head_sha(repo_path)
    _record(
        ledger,
        repo_id=_repo_id(repo_path),
        stage="fast",
        git_sha=sha,
        hooks=[{"hook_id": "mypy", "outcome": "failed"}],
        success=False,
    )
    fake = _FakeRunGateStage(script={("mypy",): False})  # still fails
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for([("repo-e", str(repo_path))]),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
        escalate=True,
    )

    # Only the narrowed call happened -- no escalation call.
    assert fake.calls == [(str(repo_path), ("mypy",))]


def test_retest_stale_baseline_degrades_to_full_wave(tmp_path, ledger, monkeypatch):
    """A baseline recorded against an old commit is never trusted -- full wave."""
    repo_path = tmp_path / "repo-f"
    _init_git_repo(repo_path)
    old_sha = _head_sha(repo_path)
    _record(
        ledger,
        repo_id=_repo_id(repo_path),
        stage="fast",
        git_sha=old_sha,
        hooks=[{"hook_id": "mypy", "outcome": "failed"}],
        success=False,
    )
    _new_commit(repo_path)  # HEAD moves; the recorded row is now stale
    fake = _FakeRunGateStage()
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    result = gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for([("repo-f", str(repo_path))]),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
        escalate=True,
    )

    entry = result["targets"]["repo-f"]
    assert entry["stale"] is True
    assert entry["retest_hook_ids"] is None  # degraded to the full wave
    assert entry["escalate"] is False  # never escalate a full-wave-already run
    assert fake.calls == [(str(repo_path), ())]


def test_retest_multiple_targets_summarized_in_message(tmp_path, ledger, monkeypatch):
    """The top-level message accounts for missing/stale counts across targets."""
    repo_missing = tmp_path / "repo-g"
    repo_clean = tmp_path / "repo-h"
    _init_git_repo(repo_missing)
    _init_git_repo(repo_clean)
    sha_clean = _head_sha(repo_clean)
    _record(
        ledger,
        repo_id=_repo_id(repo_clean),
        stage="fast",
        git_sha=sha_clean,
        hooks=[{"hook_id": "lint", "outcome": "passed"}],
        success=True,
    )
    fake = _FakeRunGateStage()
    monkeypatch.setattr(gate_runner, "run_gate_stage", fake)
    submit = _RecordingSubmit()

    result = gate_runner.dispatch(
        "retest",
        resolve_targets=_resolve_targets_for(
            [("repo-g", str(repo_missing)), ("repo-h", str(repo_clean))]
        ),
        submit_one=submit,
        stage="fast",
        gate_ledger=ledger,
    )

    assert result["targets"]["repo-g"]["baseline"] == "missing"
    assert result["targets"]["repo-h"]["baseline"] == "clean"
    assert result["status"] == "submitted"  # repo-g still submitted a job
    assert "1 retest job(s) submitted across 2 target(s)" in result["message"]
