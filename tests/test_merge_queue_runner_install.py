"""Adversarial coverage for the atomic user-unit installer."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from repository_manager import merge_queue_runner
from repository_manager.merge_queue_runner_install import (
    DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
    DEFAULT_HEAVY_TIMEOUT_SECONDS,
    HEAVY_SERVICE_NAME,
    HEAVY_TIMER_NAME,
    MergeQueueRunnerInstallError,
    install,
)
from tests.portable_executables import write_python_program


# Windows keeps only a read-only flag: a writable file reports 0o666 and has
# no execute bits, whatever mode was requested.
_EXECUTABLE_MODE = 0o666 if os.name == "nt" else 0o755

def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def _systemctl(tmp_path: Path, *, status: int = 0) -> tuple[Path, Path]:
    marker = tmp_path / "daemon-reload"
    executable = write_python_program(
        tmp_path / "systemctl",
        "import pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]))\n"
        f"raise SystemExit({status})\n",
    )
    return executable, marker


def test_install_is_hash_verified_and_daemon_reloaded(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    bin_path = tmp_path / "bin" / "runner"
    units = tmp_path / "units"
    systemctl, marker = _systemctl(tmp_path)
    python_link = tmp_path / "python"
    python_link.symlink_to(sys.executable)
    report = install(
        root,
        bin_path=bin_path,
        unit_directory=units,
        systemctl=str(systemctl),
        python_executable=python_link,
        drain_deadline_seconds=42,
        global_deadline_seconds=300,
    )
    assert report.verified
    assert marker.read_text(encoding="utf-8") == "--user daemon-reload"
    assert bin_path.stat().st_mode & 0o777 == _EXECUTABLE_MODE
    runner_artifact = report.artifacts[0]
    assert runner_artifact.name == "runner"
    assert runner_artifact.source == Path(merge_queue_runner.__file__).resolve()
    assert bin_path.read_bytes() == runner_artifact.source.read_bytes()
    assert (units / "merge-queue-runner.service").read_text().find(str(bin_path)) >= 0
    service_text = (units / "merge-queue-runner.service").read_text()
    assert f"ExecStart={python_link}" in service_text
    assert service_text.find(str(root)) >= 0
    assert "--drain-deadline-seconds 42" in service_text
    assert "--global-deadline-seconds 300" in service_text
    assert "TimeoutStartSec=360s" in service_text
    assert "KillMode=control-group" in service_text
    assert "MemoryHigh=4G" in service_text
    assert "MemoryMax=6G" in service_text
    assert "MemorySwapMax=1G" in service_text
    assert "CPUQuota=200%" in service_text
    assert "TasksMax=256" in service_text
    assert "--queue-no-push" not in service_text
    heavy_text = (units / HEAVY_SERVICE_NAME).read_text()
    assert "--phased-push" in heavy_text
    assert "/snap/bin/uv" not in heavy_text
    assert "--heavy-deadline-seconds 18000" in heavy_text
    assert f"RM_GATE_TIMEOUT_SECONDS={DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS}" in heavy_text
    assert f"TimeoutStartSec={DEFAULT_HEAVY_TIMEOUT_SECONDS}s" in heavy_text
    assert "MemoryHigh=15G" in heavy_text
    assert "MemoryMax=16G" in heavy_text
    assert "MemorySwapMax=1G" in heavy_text
    assert "CPUQuota=400%" in heavy_text
    assert "TasksMax=512" in heavy_text
    assert (units / HEAVY_TIMER_NAME).is_file()
    heavy_timer_text = (units / HEAVY_TIMER_NAME).read_text()
    assert f"Unit={HEAVY_SERVICE_NAME}" in heavy_timer_text
    assert all(item.source_sha256 == item.installed_sha256 for item in report.artifacts)


def test_daemon_reload_failure_rolls_back_the_entire_bundle(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    bin_path = tmp_path / "bin" / "runner"
    units = tmp_path / "units"
    bin_path.parent.mkdir()
    units.mkdir()
    old = {
        bin_path: b"old-runner\n",
        units / "merge-queue-runner.service": b"old-service\n",
        units / "merge-queue-runner.timer": b"old-timer\n",
    }
    for path, content in old.items():
        path.write_bytes(content)
    systemctl, _ = _systemctl(tmp_path, status=1)
    with pytest.raises(MergeQueueRunnerInstallError, match="daemon-reload failed"):
        install(root, bin_path=bin_path, unit_directory=units, systemctl=str(systemctl))
    assert {path: path.read_bytes() for path in old} == old
    assert not (units / HEAVY_SERVICE_NAME).exists()
    assert not (units / HEAVY_TIMER_NAME).exists()


def test_symlink_destination_is_refused_before_any_write(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    target = tmp_path / "bin" / "runner"
    target.parent.mkdir()
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(MergeQueueRunnerInstallError, match="symbolic link"):
        install(root, bin_path=target, unit_directory=tmp_path / "units", reload=False)
