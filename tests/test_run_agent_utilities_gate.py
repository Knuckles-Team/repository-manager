"""Regression tests for repository-manager's hermetic framework gate launcher."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "run_agent_utilities_gate.py"
_SPEC = importlib.util.spec_from_file_location("rm_gate_launcher", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_LAUNCHER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_LAUNCHER)


def test_gate_source_link_is_refused_when_it_points_elsewhere(tmp_path: Path) -> None:
    """A stale sibling link must never redirect a locked RM gate."""

    framework = tmp_path / "agent-utilities"
    framework.mkdir()
    repository = tmp_path / "repository-manager"
    sibling_root = repository / ".uv-workspace-siblings"
    sibling_root.mkdir(parents=True)
    (sibling_root / "agent-utilities").symlink_to(tmp_path / "wrong-source")

    try:
        _LAUNCHER._materialize_agent_utilities_source(repository, framework)
    except RuntimeError as exc:
        assert "expected symlink" in str(exc)
    else:  # pragma: no cover - makes a silent redirect an explicit failure
        raise AssertionError("stale sibling source was accepted")


def test_gate_runs_rm_tests_from_rm_project_without_path_leakage(
    tmp_path: Path, monkeypatch
) -> None:
    """The locked command cannot collect tests from the AU sibling checkout."""

    framework = tmp_path / "agent-utilities"
    (framework / "agent_utilities").mkdir(parents=True)
    (framework / "scripts").mkdir()
    (framework / "pyproject.toml").write_text("[project]\nname = 'agent-utilities'\n")
    calls: list[dict[str, object]] = []

    def fake_run(command, *, cwd, env):
        calls.append({"command": command, "cwd": cwd, "env": env})
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("AGENT_UTILITIES_ROOT", str(framework))
    monkeypatch.setenv("PYTHONPATH", "/foreign/checkout")
    monkeypatch.setattr(_LAUNCHER.shutil, "which", lambda _: "/usr/bin/uv")
    monkeypatch.setattr(
        _LAUNCHER,
        "_materialize_agent_utilities_source",
        lambda _repository, _framework: None,
    )
    monkeypatch.setattr(_LAUNCHER.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_agent_utilities_gate.py",
            "--extra",
            "test",
            "--module",
            "pytest",
            "--",
            "tests",
            "-q",
        ],
    )

    assert _LAUNCHER.main() == 0
    [call] = calls
    command = cast(list[str], call["command"])
    cwd = cast(Path, call["cwd"])
    environment = cast(dict[str, str], call["env"])
    assert command[:7] == [
        "/usr/bin/uv",
        "run",
        "--project",
        str(_ROOT),
        "--locked",
        "--extra",
        "test",
    ]
    assert command[7:] == ["python", "-m", "pytest", "tests", "-q"]
    assert cwd == _ROOT
    assert environment.get("PYTHONPATH") is None
    assert str(framework / "tests") not in command


def test_rm_pytest_hook_selects_local_tests_and_declared_test_extra() -> None:
    """The hook's command names RM's test extra and no AU test directory."""

    config = (_ROOT / ".pre-commit-config.yaml").read_text(encoding="utf-8")
    hook_start = config.index("  - id: pytest\n")
    hook_end = config.index("  - id: dependency-readiness\n", hook_start)
    entry = config[hook_start:hook_end]

    assert "run_agent_utilities_gate.py --extra test --module pytest" in entry
    assert 'test_target="tests"' in entry
    assert "agent-utilities/tests" not in entry
    assert "AGENT_UTILITIES_ROOT" not in entry
