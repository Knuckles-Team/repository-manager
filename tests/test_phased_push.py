import os
from unittest.mock import MagicMock, patch

import pytest

from repository_manager import dependency_readiness as dep_ready
from repository_manager.repository_manager import Git, GitResult
from repository_manager.scan_models import HookResult, RepoScanResult


@pytest.fixture
def mock_repo_manager(tmp_path):
    manager = Git(path=str(tmp_path))
    manager.project_map = {
        "https://github.com/Knuckles-Team/repo1.git": str(tmp_path / "repo1"),
        "https://github.com/Knuckles-Team/repo2.git": str(tmp_path / "repo2"),
        "https://github.com/Knuckles-Team/repo3.git": str(tmp_path / "repo3"),
    }
    # The phased push/bump loops skip projects whose local clone is absent
    # (os.path.isdir guard); create the mapped dirs so the mocked git_action runs.
    for name in ("repo1", "repo2", "repo3"):
        (tmp_path / name).mkdir(exist_ok=True)

    def git_action_side_effect(*args, **kwargs):
        command = kwargs.get("command", "")
        if not command and args:
            command = args[0]
        if "status --porcelain" in command:
            return GitResult(status="success", data="", error=None, metadata=None)
        return GitResult(status="success", data="Pushed", error=None, metadata=None)

    manager.git_action = MagicMock(side_effect=git_action_side_effect)  # type: ignore[method-assign]
    return manager


@patch("time.sleep")
def test_phased_push(mock_sleep, mock_repo_manager):
    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 5},
            {
                "phase": 2,
                "name": "Phase 2",
                "projects": ["repo2", "repo3"],
                "wait_minutes": 10,
            },
        ]
    }

    # auto_start=False isolates the raw push loop (no change-detection git calls).
    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    assert len(results) == 3  # 3 pushes
    # 3 status checks + 3 pushes = 6 calls
    assert mock_repo_manager.git_action.call_count == 6

    # CONCEPT:RM-DEP-READY: the old blind `time.sleep(wait_minutes * 60)` is
    # gone. The mocked repos here have no `pyproject.toml`, so
    # `_phase_published_packages` finds nothing published and the
    # poll-until-satisfied-or-abort barrier returns immediately (nothing to
    # wait FOR) instead of always sleeping the full budget regardless of
    # whether anything downstream needed it.
    assert mock_sleep.call_count == 0


@patch("time.sleep")
def test_phased_push_single_project(mock_sleep, mock_repo_manager):
    config = {
        "phases": [
            {
                "phase": 1,
                "name": "Phase 1",
                "projects": ["repo1", "repo2"],
                "wait_minutes": 5,
            }
        ]
    }

    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, project_filter="repo1"
    )

    assert len(results) == 1
    # 1 status check + 1 push = 2 calls
    assert mock_repo_manager.git_action.call_count == 2

    # No `pyproject.toml` in the mocked repo -> nothing published -> the
    # dependency-readiness barrier has nothing to wait for (CONCEPT:RM-DEP-READY).
    assert mock_sleep.call_count == 0


def test_phased_push_aborts_wave_when_barrier_times_out_unsatisfied(
    mock_repo_manager, monkeypatch
):
    """CONCEPT:RM-DEP-READY Layer 2 — the wave must ABORT, never advance past
    an unmet precondition. Phase 2 must never start when phase 1's gate
    barrier times out with repo2's downstream gate still failing. (Unit-level:
    `await_gate_readiness` itself is scripted here; see
    ``test_phased_push_blocks_the_wave_when_the_downstream_gate_keeps_failing``
    below for the end-to-end proof that runs the real
    ``dependency_readiness.await_gate_readiness`` against a scripted
    ``gates.run_gate_stage``.)"""
    monkeypatch.setattr(
        Git,
        "_phase_published_packages",
        lambda self, projects_to_push: {"epistemic-graph": "irrelevant/pyproject.toml"},
    )
    monkeypatch.setattr(
        dep_ready,
        "declared_fleet_constraints",
        lambda *a, **k: [
            dep_ready.DeclaredConstraint(
                package="epistemic-graph",
                raw_requirement="epistemic-graph[full]>=2.23.2,<3.0.0",
                specifier="<3.0.0,>=2.23.2",
                extras=("full",),
                declared_by="agent-utilities/pyproject.toml",
            )
        ],
    )
    unresolved = dep_ready.GateCheckFailure(
        repo_name="repo2",
        repo_path="irrelevant/repo2",
        detail="epistemic-graph declares >=2.23.2 but only 2.23.0 is available",
    )
    monkeypatch.setattr(
        dep_ready,
        "await_gate_readiness",
        lambda *a, **k: dep_ready.GateReadinessOutcome(
            ok=False, waited_s=1800.0, attempts=4, failures=[unresolved]
        ),
    )

    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 30},
            {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 0},
        ]
    }
    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    # Phase 1 pushed (status-check + push = 2 calls); phase 2 must NEVER run.
    assert mock_repo_manager.git_action.call_count == 2
    assert any(
        r.status == "error"
        and r.error
        and "aborted" in r.error.message
        and "repo2" in r.error.message
        and "epistemic-graph" in r.error.message
        for r in results
    )


def test_phased_push_proceeds_immediately_when_barrier_satisfied(
    mock_repo_manager, monkeypatch
):
    """CONCEPT:RM-DEP-READY Layer 2 — a satisfied gate barrier must not fall
    back to any blind sleep, and phase 2 must run."""
    monkeypatch.setattr(
        Git,
        "_phase_published_packages",
        lambda self, projects_to_push: {"epistemic-graph": "irrelevant/pyproject.toml"},
    )
    monkeypatch.setattr(
        dep_ready,
        "declared_fleet_constraints",
        lambda *a, **k: [
            dep_ready.DeclaredConstraint(
                package="epistemic-graph",
                raw_requirement="epistemic-graph>=2.23.0",
                specifier=">=2.23.0",
                extras=(),
                declared_by="agent-utilities/pyproject.toml",
            )
        ],
    )
    monkeypatch.setattr(
        dep_ready,
        "await_gate_readiness",
        lambda *a, **k: dep_ready.GateReadinessOutcome(
            ok=True, waited_s=12.0, attempts=1, targets_checked=["repo2"]
        ),
    )

    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 30},
            {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 0},
        ]
    }

    with patch("time.sleep") as mock_sleep:
        results = mock_repo_manager.phased_push(
            start_phase=1, config=config, auto_start=False
        )

    # Both phases ran (2 status checks + 2 pushes = 4 calls); no blind sleep.
    assert mock_repo_manager.git_action.call_count == 4
    assert all(r.status == "success" for r in results)
    assert mock_sleep.call_count == 0


def _fake_run_gate_stage_factory(script):
    """Returns a ``gates.run_gate_stage``-shaped callable that pops one
    ``RepoScanResult`` off ``script`` per call (repeating the last entry once
    exhausted), and asserts every call is scoped to the HOOK_ID hook at the
    heavy tier — exactly what ``dependency_readiness._default_run_gate``
    should be calling."""
    calls: list[tuple[str, str]] = []

    def fake(repo_path, stage, *, files=None, hook_ids=None, timeout=600):
        calls.append((repo_path, stage))
        assert stage == "heavy"
        assert hook_ids == [dep_ready.HOOK_ID]
        result = script[len(calls) - 1] if len(calls) <= len(script) else script[-1]
        return result

    fake.calls = calls  # type: ignore[attr-defined]
    return fake


def _hook_result(success: bool, detail: str = "") -> RepoScanResult:
    output = f"  [UNSATISFIED] {detail}" if (detail and not success) else ""
    return RepoScanResult(
        repo_path="irrelevant",
        success=success,
        exit_code=0 if success else 1,
        hooks=[HookResult(hook_id=dep_ready.HOOK_ID, passed=success, output=output)],
        stage="heavy",
    )


def test_phased_push_advances_the_instant_the_downstream_gate_passes(
    mock_repo_manager, monkeypatch
):
    """End-to-end proof (nothing above the subprocess boundary is mocked
    away): ``phased_push`` -> ``_await_phase_dependency_readiness`` -> the
    REAL ``dependency_readiness.await_gate_readiness`` -> the REAL
    ``gates.run_gate_stage`` call signature. repo2's gate fails once (still
    propagating), then passes — phase 2 must push the instant it does, not
    wait out the ceiling."""
    monkeypatch.setattr(
        Git,
        "_phase_published_packages",
        lambda self, projects_to_push: {"epistemic-graph": "irrelevant/pyproject.toml"},
    )
    monkeypatch.setattr(
        dep_ready,
        "declared_fleet_constraints",
        lambda *a, **k: [
            dep_ready.DeclaredConstraint(
                package="epistemic-graph",
                raw_requirement="epistemic-graph>=2.23.2",
                specifier=">=2.23.2",
                extras=(),
                declared_by="repo2/pyproject.toml",
            )
        ],
    )
    monkeypatch.setattr(dep_ready, "hook_declared", lambda repo_path: True)
    fake_run_gate_stage = _fake_run_gate_stage_factory(
        [
            _hook_result(
                False, "epistemic-graph declares >=2.23.2 but only 2.23.0 is available"
            ),
            _hook_result(True),
        ]
    )
    monkeypatch.setattr("repository_manager.gates.run_gate_stage", fake_run_gate_stage)

    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 5},
            {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 5},
        ]
    }

    with patch("time.sleep"):
        results = mock_repo_manager.phased_push(
            start_phase=1, config=config, auto_start=False
        )

    assert len(fake_run_gate_stage.calls) == 2  # blocked once, then passed
    assert mock_repo_manager.git_action.call_count == 4  # 2 status + 2 pushes
    assert all(r.status == "success" for r in results)


def test_phased_push_blocks_the_wave_when_the_downstream_gate_keeps_failing(
    mock_repo_manager, monkeypatch
):
    """End-to-end proof of the other half: repo2's gate NEVER passes ->
    the wave aborts, and repo2's push must never even be attempted."""
    monkeypatch.setattr(
        Git,
        "_phase_published_packages",
        lambda self, projects_to_push: {"epistemic-graph": "irrelevant/pyproject.toml"},
    )
    monkeypatch.setattr(
        dep_ready,
        "declared_fleet_constraints",
        lambda *a, **k: [
            dep_ready.DeclaredConstraint(
                package="epistemic-graph",
                raw_requirement="epistemic-graph>=2.23.2",
                specifier=">=2.23.2",
                extras=(),
                declared_by="repo2/pyproject.toml",
            )
        ],
    )
    monkeypatch.setattr(dep_ready, "hook_declared", lambda repo_path: True)
    fake_run_gate_stage = _fake_run_gate_stage_factory(
        [
            _hook_result(
                False, "epistemic-graph declares >=2.23.2 but only 2.23.0 is available"
            )
        ]
    )
    monkeypatch.setattr("repository_manager.gates.run_gate_stage", fake_run_gate_stage)

    # A tiny ceiling (well under one poll_interval_s) keeps this test fast
    # while still genuinely exercising the deadline path.
    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 0.001},
            {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 0},
        ]
    }

    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    # Phase 1 pushed; phase 2 (repo2) must NEVER be attempted.
    assert mock_repo_manager.git_action.call_count == 2
    assert len(fake_run_gate_stage.calls) >= 1
    assert any(
        r.status == "error"
        and r.error
        and "repo2" in r.error.message
        and "epistemic-graph" in r.error.message
        for r in results
    )


def test_phased_push_bulk_push_includes_images_and_services(mock_repo_manager):
    """CONCEPT:RM-PUSH bulk-push-scope: a ``bulk_push: true`` phase resolves
    against the WHOLE ``project_map`` (built from the entire workspace
    manifest) and pushes every repo in it that an earlier phase did not
    already handle — ``images/`` and ``services/`` INCLUDED.

    This is deliberate: the phased push exists to move the whole workspace,
    not only the Python packages. An earlier revision carved the infra trees
    out via a ``_bulk_push_excluded`` guard; that narrowed the push below its
    designed scope and was removed. Use the declarative ``exclude`` field
    (see the next test) to carve out a specific repo."""
    root = mock_repo_manager.path
    mock_repo_manager.project_map = {
        "https://gitlab.arpa/agent-packages/agents/repo1.git": os.path.join(
            root, "agent-packages", "agents", "repo1"
        ),
        "https://gitlab.arpa/images/foo.git": os.path.join(root, "images", "foo"),
        "https://gitlab.arpa/services/bar.git": os.path.join(root, "services", "bar"),
    }
    for rel in ("agent-packages/agents/repo1", "images/foo", "services/bar"):
        os.makedirs(os.path.join(root, rel), exist_ok=True)

    config = {
        "phases": [
            {
                "phase": 1,
                "name": "Phase 5: Agents",
                "bulk_push": True,
                "wait_minutes": 0,
            }
        ]
    }
    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    # All three pushed: agent-packages/repo1, images/foo AND services/bar.
    assert len(results) == 3
    assert all(r.status == "success" for r in results)
    assert mock_repo_manager.git_action.call_count == 6  # 3 x (status + push)


def test_phased_push_honors_declarative_exclude_pattern(mock_repo_manager):
    """The previously-modeled-but-unused ``MaintenancePhase.exclude`` field is
    now live: an fnmatch pattern against the project name carves a repo out
    of an explicit phase, not just bulk_push."""
    config = {
        "phases": [
            {
                "phase": 1,
                "name": "Phase 1",
                "projects": ["repo1", "repo2"],
                "exclude": ["repo2"],
                "wait_minutes": 0,
            }
        ]
    }
    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )
    assert len(results) == 1
    assert mock_repo_manager.git_action.call_count == 2  # 1 status + 1 push


def test_push_projects(mock_repo_manager):
    results = mock_repo_manager.push_projects(["/fake/path/repo1", "/fake/path/repo2"])

    assert len(results) == 2
    # 2 status checks + 2 pushes = 4 calls
    assert mock_repo_manager.git_action.call_count == 4
    # Verify the push commands called were git push --follow-tags
    push_calls = [
        call
        for call in mock_repo_manager.git_action.call_args_list
        if "git push --follow-tags"
        in (call.kwargs.get("command") or (call.args[0] if call.args else ""))
    ]
    assert len(push_calls) == 2
