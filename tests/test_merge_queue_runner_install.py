"""Adversarial coverage for the atomic user-unit installer."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from repository_manager.merge_queue_runner_install import (
    MergeQueueRunnerInstallError,
    install,
)


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    root.mkdir()
    return root


def _systemctl(tmp_path: Path, *, status: int = 0) -> tuple[Path, Path]:
    marker = tmp_path / "daemon-reload"
    executable = tmp_path / "systemctl"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(marker)!r}).write_text(' '.join(sys.argv[1:]))\n"
        f"raise SystemExit({status})\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
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
    )
    assert report.verified
    assert marker.read_text(encoding="utf-8") == "--user daemon-reload"
    assert bin_path.stat().st_mode & 0o777 == 0o755
    assert (units / "merge-queue-runner.service").read_text().find(str(bin_path)) >= 0
    service_text = (units / "merge-queue-runner.service").read_text()
    assert f"ExecStart={Path(sys.executable).resolve()}" in service_text
    assert service_text.find(str(root)) >= 0
    assert "--drain-deadline-seconds 42" in service_text
    assert "TimeoutStartSec=42" in service_text
    assert "KillMode=control-group" in service_text
    assert "MemoryMax=2G" in service_text
    assert "CPUQuota=400%" in service_text
    assert "TasksMax=256" in service_text
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


def test_symlink_destination_is_refused_before_any_write(tmp_path: Path) -> None:
    root = _workspace(tmp_path)
    target = tmp_path / "bin" / "runner"
    target.parent.mkdir()
    target.symlink_to(tmp_path / "outside")
    with pytest.raises(MergeQueueRunnerInstallError, match="symbolic link"):
        install(root, bin_path=target, unit_directory=tmp_path / "units", reload=False)
