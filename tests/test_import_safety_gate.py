"""Focused regressions for the repository import-safety gate."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GATE = ROOT / "scripts" / "check_import_safety.py"
SECURITY_CONTRACT = ROOT / "scripts" / "security_contract.py"


def _run_gate(
    *arguments: str,
    cwd: Path,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    process_env = os.environ.copy()
    if env:
        process_env.update(env)
    return subprocess.run(
        [sys.executable, str(GATE), *arguments],
        cwd=cwd,
        env=process_env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_native_gate_checks_standalone_script() -> None:
    result = _run_gate(
        "--package",
        "scripts.security_contract",
        "--script",
        str(SECURITY_CONTRACT),
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "standalone script(s)" in result.stdout


def test_exact_pre_commit_gate_command_is_green() -> None:
    result = _run_gate(
        "--package",
        "repository_manager",
        "--script",
        "scripts/security_contract.py",
        "--simulate-windows",
        cwd=ROOT,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "all " in result.stdout
    assert "standalone script(s)" in result.stdout


def test_checkout_source_wins_over_conflicting_installed_package(
    tmp_path: Path,
) -> None:
    fake_root = tmp_path / "fake-install"
    fake_package = fake_root / "repository_manager"
    fake_package.mkdir(parents=True)
    marker = tmp_path / "stale-install-was-loaded"
    fake_package.joinpath("__init__.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('loaded')\n",
        encoding="utf-8",
    )

    result = _run_gate(
        "--package",
        "repository_manager",
        "--script",
        str(SECURITY_CONTRACT),
        "--simulate-windows",
        cwd=tmp_path,
        env={"PYTHONPATH": str(fake_root)},
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists(), result.stdout + result.stderr


def test_simulated_windows_accepts_guarded_posix_import(tmp_path: Path) -> None:
    guarded = tmp_path / "guarded.py"
    guarded.write_text(
        "import sys\nif sys.platform != 'win32':\n    import fcntl\n",
        encoding="utf-8",
    )

    result = _run_gate(
        "--package",
        "scripts.security_contract",
        "--script",
        str(guarded),
        "--simulate-windows",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr


def test_simulated_windows_rejects_unguarded_resource_fixture(tmp_path: Path) -> None:
    unguarded = tmp_path / "unguarded.py"
    unguarded.write_text("import resource\n", encoding="utf-8")

    result = _run_gate(
        "--package",
        "scripts.security_contract",
        "--script",
        str(unguarded),
        "--simulate-windows",
        cwd=tmp_path,
    )

    assert result.returncode != 0
    assert "unguarded POSIX-only import 'resource'" in result.stdout


def test_source_checkout_gate_works_outside_repository(tmp_path: Path) -> None:
    result = _run_gate(
        "--package",
        "repository_manager",
        "--script",
        str(SECURITY_CONTRACT),
        "--simulate-windows",
        cwd=tmp_path,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "No module named 'repository_manager'" not in result.stdout
    assert "standalone script(s)" in result.stdout


def test_security_contract_fails_closed_without_unix_resource_limits() -> None:
    spec = importlib.util.spec_from_file_location(
        "security_contract_without_resource", SECURITY_CONTRACT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load security contract")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._resource = None

    with pytest.raises(module.SecurityContractError, match="Unix resource support"):
        module._limit_hook_output()

    with pytest.raises(module.SecurityContractError, match="Unix resource"):
        module.run_hook(ROOT, {}, "fuzz", "results")
