import subprocess
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest

from repository_manager import dependency_readiness as dep_ready
from repository_manager.repository_manager import Git, GitResult
from repository_manager.scan_models import HookResult, RepoScanResult


def _run_git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _make_real_repo(root: Path, name: str) -> Path:
    """Create a tiny worktree with a real bare upstream for phase target checks."""
    root.mkdir(parents=True, exist_ok=True)
    remote = root / f"{name}.git"
    local = root / name
    _run_git(root, "init", "--bare", "--initial-branch=main", str(remote))
    _run_git(root, "init", "--initial-branch=main", str(local))
    _run_git(local, "config", "user.email", "tests@example.invalid")
    _run_git(local, "config", "user.name", "Repository Manager Tests")
    (local / "README.md").write_text(f"{name}\n")
    _run_git(local, "add", "README.md")
    _run_git(local, "commit", "-m", "initial")
    _run_git(local, "remote", "add", "origin", str(remote))
    _run_git(local, "push", "-u", "origin", "main")
    return local


def _real_repo_map(root: Path) -> dict[str, Path]:
    return {name: _make_real_repo(root, name) for name in ("repo1", "repo2", "repo3")}


def _project_map(repo_paths: dict[str, Path]) -> dict[str, str]:
    return {
        f"https://github.com/Knuckles-Team/{name}.git": str(path)
        for name, path in repo_paths.items()
    }


def _called_path(call: Any) -> str:
    """Read a path from either phased_push's keyword or push_projects' arg."""
    path = call.kwargs.get("path") or (call.args[0] if call.args else None)
    assert path is not None
    return path


def _pushed_paths(manager: Git) -> list[str]:
    mock = cast(MagicMock, manager.push_project)
    return [_called_path(call) for call in mock.call_args_list]


@pytest.fixture
def mock_repo_manager(tmp_path):
    manager = Git(path=str(tmp_path))
    manager.project_map = _project_map(_real_repo_map(tmp_path))

    # The phased workflow is responsible for ordering and barriers.  Patch the
    # transaction boundary so these tests do not bypass the fail-closed
    # status/upstream/gate/push protocol by mocking individual git commands.
    manager.push_project = MagicMock(  # type: ignore[method-assign]
        return_value=GitResult(
            status="success", data="Pushed", error=None, metadata=None
        )
    )
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
    pushed_paths = _pushed_paths(mock_repo_manager)
    assert pushed_paths[0] == str(mock_repo_manager.path) + "/repo1"
    assert set(pushed_paths[1:]) == {
        str(mock_repo_manager.path) + "/repo2",
        str(mock_repo_manager.path) + "/repo3",
    }

    # CONCEPT:RM-DEP-READY: the old blind `time.sleep(wait_minutes * 60)` is
    # gone. The fixture repos have no `pyproject.toml`, so
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
    assert _pushed_paths(mock_repo_manager) == [str(mock_repo_manager.path) + "/repo1"]

    # No `pyproject.toml` in the fixture repo -> nothing published -> the
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

    # Phase 1 pushed; phase 2 must NEVER run.
    assert _pushed_paths(mock_repo_manager) == [str(mock_repo_manager.path) + "/repo1"]
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

    # Both phases ran; no blind sleep.
    pushed_paths = _pushed_paths(mock_repo_manager)
    assert pushed_paths[0] == str(mock_repo_manager.path) + "/repo1"
    assert pushed_paths[1] == str(mock_repo_manager.path) + "/repo2"
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
    pushed_paths = _pushed_paths(mock_repo_manager)
    assert pushed_paths == [
        str(mock_repo_manager.path) + "/repo1",
        str(mock_repo_manager.path) + "/repo2",
    ]
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
            {
                "phase": 1,
                "name": "Phase 1",
                "projects": ["repo1"],
                "wait_minutes": 0.001,
            },
            {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 0},
        ]
    }

    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    # Phase 1 pushed; phase 2 (repo2) must NEVER be attempted.
    assert _pushed_paths(mock_repo_manager) == [str(mock_repo_manager.path) + "/repo1"]
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
    root = Path(mock_repo_manager.path)
    repo1 = _make_real_repo(root / "agent-packages" / "agents", "repo1")
    image = _make_real_repo(root / "images", "foo")
    service = _make_real_repo(root / "services", "bar")
    mock_repo_manager.project_map = {
        "https://gitlab.arpa/agent-packages/agents/repo1.git": str(repo1),
        "https://gitlab.arpa/images/foo.git": str(image),
        "https://gitlab.arpa/services/bar.git": str(service),
    }

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
    assert set(_pushed_paths(mock_repo_manager)) == {
        str(repo1),
        str(image),
        str(service),
    }


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
    assert _pushed_paths(mock_repo_manager) == [str(mock_repo_manager.path) + "/repo1"]


def test_push_projects(mock_repo_manager):
    project_dirs = [
        str(mock_repo_manager.path) + "/repo1",
        str(mock_repo_manager.path) + "/repo2",
    ]
    results = mock_repo_manager.push_projects(project_dirs)

    assert len(results) == 2
    assert set(_pushed_paths(mock_repo_manager)) == set(project_dirs)
