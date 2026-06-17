"""Tests for repository scanning and pre-commit failure reporting."""

import subprocess
from unittest import mock

from repository_manager import scanner


def _completed(returncode: int, stdout: str = "", stderr: str = ""):
    return subprocess.CompletedProcess(
        args=["pre-commit", "run", "--all-files", "--verbose"],
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
    )


def test_init_failure_surfaces_raw_output(tmp_path):
    """A non-zero pre-commit with no parseable hooks must report the real error.

    Reproduces the "failed with empty failures" gap: when pre-commit aborts at
    environment install / invalid config (before any hook line is emitted), the
    diagnostic lives only in raw_output. scan_repository must surface it as
    ``error`` so the failure is never silently empty.
    """
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    boom = "An error has occurred: InvalidConfigError\nfailed to install hook env"

    with mock.patch.object(scanner, "run_pre_commit", return_value=_completed(1, stderr=boom)):
        result = scanner.scan_repository(str(tmp_path))

    assert result.success is False
    assert result.exit_code == 1
    assert result.hooks == []  # nothing parseable
    assert result.error is not None
    assert "InvalidConfigError" in result.error


def test_real_hook_failure_not_overridden(tmp_path):
    """When a hook genuinely fails, error stays None and the hook is captured."""
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    output = "ruff.....................................................................Failed\n- hook id: ruff\nE501 line too long\n"

    with mock.patch.object(scanner, "run_pre_commit", return_value=_completed(1, stdout=output)):
        result = scanner.scan_repository(str(tmp_path))

    assert result.success is False
    assert any(not h.passed for h in result.hooks)
    assert result.error is None


def test_success_passes_cleanly(tmp_path):
    """All hooks passing yields success with no synthesized error."""
    (tmp_path / ".pre-commit-config.yaml").write_text("repos: []\n")
    output = "ruff.....................................................................Passed\n"

    with mock.patch.object(scanner, "run_pre_commit", return_value=_completed(0, stdout=output)):
        result = scanner.scan_repository(str(tmp_path))

    assert result.success is True
    assert result.error is None
