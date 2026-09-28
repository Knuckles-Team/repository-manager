import os
import subprocess
from pathlib import Path

import pytest

from repository_manager import dependency_readiness as dep_ready
from repository_manager.repository_manager import Git
from repository_manager.scan_models import HookResult, RepoScanResult


def _run_git(path: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _initialize_push_project(path: Path, origin: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    _run_git(path, "init", "-q", "-b", "main")
    _run_git(path, "config", "user.name", "test")
    _run_git(path, "config", "user.email", "test@example.invalid")
    (path / "README.md").write_text(f"{path.name}\n")
    _run_git(path, "add", "README.md")
    _run_git(path, "commit", "-qm", "initial")
    _run_git(path, "remote", "add", "origin", origin)


def _remote_ref(manager: Git, project: str) -> str | None:
    remote = Path(manager.path) / "remotes" / f"{project}.git"
    result = subprocess.run(
        ["git", "--git-dir", str(remote), "rev-parse", "--verify", "refs/heads/main"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


@pytest.fixture
def mock_repo_manager(tmp_path):
    manager = Git(path=str(tmp_path))
    manager.project_map = {
        "https://github.com/Knuckles-Team/repo1.git": str(tmp_path / "repo1"),
        "https://github.com/Knuckles-Team/repo2.git": str(tmp_path / "repo2"),
        "https://github.com/Knuckles-Team/repo3.git": str(tmp_path / "repo3"),
    }
    remotes = tmp_path / "remotes"
    remotes.mkdir()
    for url, project_path in manager.project_map.items():
        project = Path(project_path)
        _initialize_push_project(project, url)
        remote = remotes / f"{project.name}.git"
        _run_git(remotes, "init", "-q", "--bare", str(remote))

    def local_destination(target_path, _pinned):
        remote = remotes / f"{Path(target_path).name}.git"
        if not remote.exists():
            _run_git(remotes, "init", "-q", "--bare", str(remote))
        return str(remote)

    manager._sealed_push_destination = local_destination  # type: ignore[method-assign]
    manager.gate_before_push = False
    return manager


def test_phased_push(mock_repo_manager):
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
    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is not None
    assert _remote_ref(mock_repo_manager, "repo3") is not None

    # CONCEPT:RM-DEP-READY: the old blind `time.sleep(wait_minutes * 60)` is
    # gone. The mocked repos here have no `pyproject.toml`, so
    # `_phase_published_packages` finds nothing published and the
    # poll-until-satisfied-or-abort barrier returns immediately (nothing to
    # wait FOR) instead of always sleeping the full budget regardless of
    # whether anything downstream needed it.


def test_phased_push_single_project(mock_repo_manager):
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
    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is None

    # No `pyproject.toml` in the mocked repo -> nothing published -> the
    # dependency-readiness barrier has nothing to wait for (CONCEPT:RM-DEP-READY).


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
        lambda *_args, **_kwargs: [
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
        lambda *_args, **_kwargs: dep_ready.GateReadinessOutcome(
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

    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is None
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
        lambda *_args, **_kwargs: [
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
        lambda *_args, **_kwargs: dep_ready.GateReadinessOutcome(
            ok=True, waited_s=12.0, attempts=1, targets_checked=["repo2"]
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

    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is not None
    assert all(r.status == "success" for r in results)


def test_phase_readiness_cross_checks_the_independent_later_phase_universe(
    mock_repo_manager, monkeypatch
):
    """A narrowed constraint target cannot bypass the independent cross-check."""
    repo2 = Path(mock_repo_manager.path) / "repo2"
    (repo2 / "pyproject.toml").write_text(
        """\
[project]
name = "repo2"
dependencies = ["epistemic-graph>=2.0.0"]
"""
    )
    monkeypatch.setattr(
        Git,
        "_phase_published_packages",
        lambda self, projects_to_push: {"epistemic-graph": "unused"},
    )
    calls = {"n": 0}

    def narrowed_then_independent(path, *, fleet_packages: object):
        del fleet_packages
        calls["n"] += 1
        if calls["n"] == 1:
            # Simulate the planner's narrowed scan missing the real dependent.
            return []
        return [
            dep_ready.DeclaredConstraint(
                package="epistemic-graph",
                raw_requirement="epistemic-graph>=2.0.0",
                specifier=">=2.0.0",
                extras=(),
                declared_by=str(Path(path) / "pyproject.toml"),
            )
        ]

    monkeypatch.setattr(
        dep_ready, "declared_fleet_constraints", narrowed_then_independent
    )
    outcome = mock_repo_manager._await_phase_dependency_readiness(
        phase_num=1,
        phase_name="Phase 1",
        projects_to_push=[("repo1", str(Path(mock_repo_manager.path) / "repo1"))],
        later_phases=[
            {
                "name": "Phase 2",
                "projects_to_push": [("repo2", str(repo2))],
            }
        ],
        wait_minutes=1,
    )

    assert outcome.ok is False
    assert outcome.attempts == 0
    assert outcome.waited_s == 0.0
    assert outcome.failures[0].reason == "TARGETS_INCOMPLETE"
    assert calls["n"] == 2


def test_phased_push_aborts_before_mutation_when_frozen_plan_drifts(
    mock_repo_manager, monkeypatch
):
    """Legacy phased push binds execution to the frozen input/target digest."""
    config = {"phases": [{"phase": 1, "name": "Phase 1", "projects": ["repo1"]}]}
    original_execute = mock_repo_manager._execute_push_phase
    url = "https://github.com/Knuckles-Team/repo1.git"

    def drift_before_execute(**kwargs):
        mock_repo_manager.project_map[url] = str(
            Path(mock_repo_manager.path) / "repo1-renamed"
        )
        return original_execute(**kwargs)

    monkeypatch.setattr(mock_repo_manager, "_execute_push_phase", drift_before_execute)
    results = mock_repo_manager.phased_push(
        config=config,
        start_phase=1,
        auto_start=False,
    )

    assert any(
        result.status == "error"
        and result.error
        and "release plan changed" in result.error.message
        for result in results
    )
    assert _remote_ref(mock_repo_manager, "repo1") is None
    assert len(mock_repo_manager.progress["release_plan_digest"]) == 64


def _fake_run_gate_stage_factory(script):
    """Returns a ``gates.run_gate_stage``-shaped callable that pops one
    ``RepoScanResult`` off ``script`` per call (repeating the last entry once
    exhausted), and asserts every call is scoped to the HOOK_ID hook at the
    manual release tier — exactly what ``dependency_readiness._default_run_gate``
    should be calling."""
    calls: list[tuple[str, str]] = []

    def fake(repo_path, stage, *, files=None, hook_ids=None, timeout=600):
        calls.append((repo_path, stage))
        assert stage == "manual"
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
        stage="manual",
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
        lambda *_args, **_kwargs: [
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
    original_readiness = dep_ready.await_gate_readiness

    def no_delay_readiness(*args, **kwargs):
        return original_readiness(*args, **kwargs, sleep=lambda _seconds: None)

    monkeypatch.setattr(dep_ready, "await_gate_readiness", no_delay_readiness)

    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 5},
            {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 5},
        ]
    }

    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    assert len(fake_run_gate_stage.calls) == 2  # blocked once, then passed
    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is not None
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
        lambda *_args, **_kwargs: [
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

    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is None
    assert len(fake_run_gate_stage.calls) >= 1
    assert any(
        r.status == "error"
        and r.error
        and "repo2" in r.error.message
        and "epistemic-graph" in r.error.message
        for r in results
    )


def _write_release_pyproject(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        f"""\
[project]
name = "{path.name}"
version = "1.0.0"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"
"""
    )


def test_phased_push_bulk_push_excludes_infrastructure_even_when_buildable(
    mock_repo_manager,
):
    """Phase 5 selects PyPI agents, not every repository in ``project_map``.

    Giving images and services valid Python build metadata proves category is a
    required independent condition, rather than an accidental pyproject-only
    allow-list.
    """
    root = mock_repo_manager.path
    mock_repo_manager.project_map = {
        "https://gitlab.arpa/agent-packages/agents/repo1.git": os.path.join(
            root, "agent-packages", "agents", "repo1"
        ),
        "https://gitlab.arpa/images/foo.git": os.path.join(root, "images", "foo"),
        "https://gitlab.arpa/services/bar.git": os.path.join(root, "services", "bar"),
    }
    mock_repo_manager._project_categories = {
        "https://gitlab.arpa/agent-packages/agents/repo1.git": (
            "agent-packages",
            "agents",
        ),
        "https://gitlab.arpa/images/foo.git": ("images",),
        "https://gitlab.arpa/services/bar.git": ("services",),
    }
    for url, project_path in mock_repo_manager.project_map.items():
        path = Path(project_path)
        _initialize_push_project(path, url)
        _write_release_pyproject(path)
        _run_git(path, "add", "pyproject.toml")
        _run_git(path, "commit", "-qm", "add package metadata")

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

    assert len(results) == 1
    assert all(r.status == "success" for r in results)
    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "foo") is None
    assert _remote_ref(mock_repo_manager, "bar") is None


@pytest.mark.parametrize(
    ("category", "metadata"),
    [
        (None, "valid"),
        (("agent-packages", "unknown"), "valid"),
        (("agent-packages", "agents"), "missing"),
        (("agent-packages", "agents"), "empty"),
        (("agent-packages", "agents"), "malformed"),
    ],
)
def test_phased_push_bulk_metadata_and_category_fail_closed(
    mock_repo_manager, category, metadata
):
    url = "https://github.com/Knuckles-Team/candidate.git"
    path = Path(mock_repo_manager.path) / "agent-packages" / "agents" / "candidate"
    path.mkdir(parents=True)
    mock_repo_manager.project_map = {url: str(path)}
    mock_repo_manager._project_categories = {} if category is None else {url: category}
    if metadata == "valid":
        _write_release_pyproject(path)
    elif metadata == "empty":
        (path / "pyproject.toml").write_text("[project]\n")
    elif metadata == "malformed":
        (path / "pyproject.toml").write_text("[project\n")

    config = {"phases": [{"phase": 5, "name": "Phase 5: Agents", "bulk_push": True}]}
    results = mock_repo_manager.phased_push(
        start_phase=5, config=config, auto_start=False
    )

    assert results == []
    assert _remote_ref(mock_repo_manager, "candidate") is None


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
    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is None


def test_push_projects(mock_repo_manager):
    results = mock_repo_manager.push_projects(
        [
            str(Path(mock_repo_manager.path) / "repo1"),
            str(Path(mock_repo_manager.path) / "repo2"),
        ]
    )

    assert len(results) == 2
    assert all(result.status == "success" for result in results)
    assert _remote_ref(mock_repo_manager, "repo1") is not None
    assert _remote_ref(mock_repo_manager, "repo2") is not None
