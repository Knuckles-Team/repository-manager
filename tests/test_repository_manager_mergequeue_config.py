"""Repository-manager's own merge-queue interpreter contract."""

from __future__ import annotations

import subprocess
from pathlib import Path

from agent_utilities.governance.lanes import lane_scope

from repository_manager import merge_queue

ROOT = Path(__file__).resolve().parents[1]
REPO_PYTHON = ".venv/bin/python"
QUEUE_TEST_COMMAND = (
    REPO_PYTHON,
    "-m",
    "pytest",
    "tests/test_merge_queue.py",
    "tests/test_config_schema.py",
    "tests/test_repository_manager_mergequeue_config.py",
    "-q",
)


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def test_repository_queue_config_never_invokes_ambient_python() -> None:
    config = merge_queue.load_config(ROOT)
    commands = (
        config.environment_signature,
        *(gate.command for gate in config.gates),
        *config.regenerate,
    )

    assert commands
    assert all(command[0] == REPO_PYTHON for command in commands)


def test_repository_queue_config_uses_exact_queue_owned_test_census() -> None:
    config = merge_queue.load_config(ROOT)
    queue_gate = next(gate for gate in config.gates if gate.name == "queue-tests")

    assert queue_gate.command == QUEUE_TEST_COMMAND
    assert "tests/" not in queue_gate.command


def test_materialized_snapshot_attaches_only_an_existing_ignored_venv(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "config", "user.name", "Queue Test")
    _git(repo, "config", "user.email", "queue@example.invalid")
    (repo / ".gitignore").write_text(".venv\n", encoding="utf-8")
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _git(repo, "add", "--", ".gitignore", "tracked.txt")
    _git(repo, "commit", "-q", "-m", "fixture")
    head = _git(repo, "rev-parse", "HEAD")
    scope = lane_scope(repo)

    with merge_queue.materialized(repo, head, scope=scope) as snapshot:
        assert not (snapshot / ".venv").exists()

    interpreter = repo / REPO_PYTHON
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)

    with merge_queue.materialized(repo, head, scope=scope) as snapshot:
        attached = snapshot / ".venv"
        assert attached.is_symlink()
        assert attached.resolve() == (repo / ".venv").resolve()
        assert (snapshot / REPO_PYTHON).is_file()
