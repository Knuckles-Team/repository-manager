"""Regression tests for repository-manager's gate launcher and skip contract."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, _ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_LAUNCHER = _load("rm_gate_launcher", "run_agent_utilities_gate.py")
_GATE_ENV = _load("rm_gate_env", "gate_env.py")


def _framework(tmp_path: Path) -> Path:
    framework = tmp_path / "agent-utilities"
    (framework / "agent_utilities").mkdir(parents=True)
    (framework / "scripts").mkdir()
    (framework / "pyproject.toml").write_text("[project]\nname = 'agent-utilities'\n")
    return framework


def _run_main(monkeypatch, *argv: str) -> tuple[int, list[dict[str, object]]]:
    calls: list[dict[str, object]] = []

    def fake_run(command, *, cwd, env):
        calls.append({"command": command, "cwd": cwd, "env": env})
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(_LAUNCHER.subprocess, "run", fake_run)
    monkeypatch.setattr(sys, "argv", ["run_agent_utilities_gate.py", *argv])
    return _LAUNCHER.main(), calls


@pytest.fixture
def located(tmp_path: Path, monkeypatch) -> Path:
    """An agent-utilities checkout and a project interpreter are available."""

    framework = _framework(tmp_path)
    python = tmp_path / "venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("")
    monkeypatch.setenv("AGENT_UTILITIES_ROOT", str(framework))
    monkeypatch.setenv("PYTHONPATH", "/foreign/checkout")
    monkeypatch.setattr(_LAUNCHER, "_project_python", lambda _root: python)
    monkeypatch.setattr(
        _LAUNCHER, "_materialize_agent_utilities_source", lambda *_: None
    )
    return framework


def test_stale_sibling_source_is_refused(tmp_path: Path) -> None:
    """A sibling link resolving elsewhere must never redirect a gate."""

    framework = _framework(tmp_path)
    sibling_root = tmp_path / "repository-manager" / ".uv-workspace-siblings"
    sibling_root.mkdir(parents=True)
    (sibling_root / "agent-utilities").symlink_to(tmp_path / "wrong-source")

    with pytest.raises(_LAUNCHER.SourceMismatch):
        _LAUNCHER._materialize_agent_utilities_source(
            tmp_path / "repository-manager", framework
        )


def test_bootstrapped_clone_is_accepted_as_the_source(
    tmp_path: Path, monkeypatch
) -> None:
    """The checkout scripts/bootstrap.sh clones in place is the framework root."""

    monkeypatch.delenv("AGENT_UTILITIES_ROOT", raising=False)
    repository = tmp_path / "repository-manager"
    clone = _framework(repository / ".uv-workspace-siblings")

    assert _LAUNCHER._agent_utilities_root(repository) == clone.resolve()
    _LAUNCHER._materialize_agent_utilities_source(repository, clone)


def test_module_gate_runs_in_the_project_environment(located, monkeypatch) -> None:
    """Tests run from RM's own root and interpreter, never from the sibling."""

    code, [call] = _run_main(monkeypatch, "--module", "pytest", "--", "tests", "-q")

    assert code == 0
    command = cast(list[str], call["command"])
    environment = cast(dict[str, str], call["env"])
    python = Path(command[0])
    assert command[1:] == ["-m", "pytest", "tests", "-q"]
    assert call["cwd"] == _ROOT
    assert environment.get("PYTHONPATH") is None
    assert environment["PATH"].split(os.pathsep)[0] == str(python.parent)
    assert str(located / "tests") not in command


def test_framework_script_must_be_allowlisted(located, monkeypatch) -> None:
    with pytest.raises(SystemExit):
        _run_main(monkeypatch, "--script", "scripts/anything.py")


def test_framework_script_runs_from_the_framework_checkout(
    located, monkeypatch
) -> None:
    script = located / "scripts" / "check_no_legacy_markers.py"
    script.write_text("pass\n")

    code, [call] = _run_main(
        monkeypatch,
        "--system-script",
        "--script",
        "scripts/check_no_legacy_markers.py",
        "--",
        ".",
    )

    assert code == 0
    assert cast(list[str], call["command"])[1:] == [str(script), "."]


@pytest.mark.parametrize("ci, expected", [(None, 0), ("true", 2)])
def test_missing_sibling_skips_locally_and_cannot_run_in_ci(
    tmp_path: Path, monkeypatch, capsys, ci: str | None, expected: int
) -> None:
    monkeypatch.delenv("CI", raising=False)
    if ci:
        monkeypatch.setenv("CI", ci)
    monkeypatch.setenv("AGENT_UTILITIES_ROOT", str(tmp_path / "missing"))

    code, calls = _run_main(monkeypatch, "--module", "pytest")

    assert (code, calls) == (expected, [])
    out = capsys.readouterr()
    if ci:
        assert "CANNOT RUN" in out.err
    else:
        assert out.out.startswith("SKIPPED (pytest): ")


def test_missing_project_environment_is_reported_not_passed(
    located, tmp_path: Path, monkeypatch, capsys
) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.setattr(_LAUNCHER, "_project_python", lambda _r: tmp_path / "none")

    code, calls = _run_main(monkeypatch, "--module", "pytest")

    assert (code, calls) == (0, [])
    assert "scripts/bootstrap.sh" in capsys.readouterr().out


@pytest.mark.parametrize("ci, expected", [(None, 0), ("1", 2)])
def test_missing_tool_contract(monkeypatch, capsys, ci, expected) -> None:
    monkeypatch.delenv("CI", raising=False)
    if ci:
        monkeypatch.setenv("CI", ci)

    code = _GATE_ENV.main(
        ["--gate", "demo", "--need", "no-such-tool-xyz", "--", "true"]
    )

    assert code == expected
    out = capsys.readouterr()
    assert ("CANNOT RUN" in out.err) if ci else ("SKIPPED (demo): " in out.out)


def test_present_tool_runs_the_command(monkeypatch) -> None:
    monkeypatch.setenv("CI", "1")
    assert _GATE_ENV.main(["--gate", "demo", "--need", "sh", "--", "false"]) == 1
    assert _GATE_ENV.main(["--gate", "demo", "--need", "sh", "--", "true"]) == 0


def test_lane_guard_is_exempt_only_in_cloud_sessions(located, monkeypatch) -> None:
    script = located / "scripts" / "check_lane_guard.py"
    script.write_text("pass\n")
    argv = ("--system-script", "--script", "scripts/check_lane_guard.py")

    monkeypatch.setenv("CLAUDE_CODE_REMOTE", "true")
    assert _run_main(monkeypatch, *argv) == (0, [])

    monkeypatch.delenv("CLAUDE_CODE_REMOTE")
    code, [call] = _run_main(monkeypatch, *argv)
    assert code == 0
    assert cast(list[str], call["command"])[1:] == [str(script)]
