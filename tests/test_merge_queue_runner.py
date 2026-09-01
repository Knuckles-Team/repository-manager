"""Adversarial coverage for manifest-owned merge-queue discovery."""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from repository_manager.merge_queue_runner import (
    CANONICAL_FRAGMENT,
    DeclaredRepository,
    MergeQueueRunnerError,
    _runner_command,
    _runner_settings,
    build_parser,
    declared_repositories,
    discover_queued_repositories,
    drain_repository,
)

NOW = datetime(2026, 8, 31, 12, 0, tzinfo=UTC)


def _git_repo(path: Path) -> None:
    path.mkdir(parents=True)
    subprocess.run(["git", "init", "-q", "-b", "main", str(path)], check=True)
    subprocess.run(
        ["git", "-C", str(path), "config", "user.email", "t@example.invalid"],
        check=True,
    )
    subprocess.run(["git", "-C", str(path), "config", "user.name", "test"], check=True)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    subprocess.run(["git", "-C", str(path), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(path), "commit", "-qm", "init"], check=True)


def _fragment(repo: Path, name: str, records: list[dict[str, object]]) -> None:
    target = repo / ".git" / "agent-lanes" / "merge-queue" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(records, sort_keys=False), encoding="utf-8")


def _record(
    candidate_id: str,
    *,
    state: str = "queued",
    enqueued: str = "2026-08-31T11:00:00+00:00",
    recorded: str | None = None,
) -> dict[str, object]:
    return {
        "id": candidate_id,
        "lane": candidate_id,
        "base": "main",
        "worktree": "/tmp/lane",
        "enqueued_at": enqueued,
        "recorded_at": recorded or enqueued,
        "state": state,
    }


def _workspace(root: Path) -> None:
    paths = (
        "plans",
        "pipelines",
        "gitlab-pipelines",
        "agent-packages/agent-utilities",
        "agent-packages/platform/epistemic-graph",
        "open-source-libraries/research-input",
    )
    for relative in paths:
        _git_repo(root / relative)
    manifest = {
        "name": "fixture",
        "path": str(root),
        "repositories": [
            {"url": "https://example.invalid/plans.git"},
            {"url": "https://example.invalid/pipelines.git"},
            {"url": "https://example.invalid/gitlab-pipelines.git"},
        ],
        "subdirectories": {
            "agent-packages": {
                "repositories": [
                    {"url": "https://example.invalid/agent-utilities.git"}
                ],
                "subdirectories": {
                    "platform": {
                        "repositories": [
                            {"url": "https://example.invalid/epistemic-graph.git"}
                        ]
                    }
                },
            },
            "open-source-libraries": {
                "repositories": [{"url": "https://example.invalid/research-input.git"}]
            },
        },
    }
    (root / "workspace.yml").write_text(
        yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8"
    )


def test_default_child_command_is_a_direct_python_module(
    tmp_path: Path,
) -> None:
    repository = DeclaredRepository("fixture", "fixture", tmp_path)
    command = _runner_command(repository, None, 1200)
    assert command[:3] == [sys.executable, "-m", "repository_manager"]
    assert "/snap/bin/uv" not in command
    assert command[-2:] == ["--queue-lease-ttl-seconds", "1200"]


def test_manifest_recurses_root_and_nested_repositories_but_excludes_research(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    repositories = declared_repositories(tmp_path)
    assert [item.identifier for item in repositories] == [
        "plans",
        "pipelines",
        "gitlab-pipelines",
        "agent-packages/agent-utilities",
        "agent-packages/platform/epistemic-graph",
    ]
    assert all("open-source-libraries" not in item.identifier for item in repositories)


def test_discovery_folds_latest_record_and_ignores_canonical_stale_terminal(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    _fragment(tmp_path / "plans", "lane.yaml", [_record("fresh")])
    _fragment(
        tmp_path / "pipelines",
        "lane.yaml",
        [_record("old", enqueued="2026-08-29T11:00:00+00:00")],
    )
    _fragment(
        tmp_path / "gitlab-pipelines",
        "lane.yaml",
        [_record("terminal", state="landed")],
    )
    _fragment(
        tmp_path / "agent-packages/agent-utilities",
        CANONICAL_FRAGMENT,
        [_record("canonical-queued")],
    )
    _fragment(
        tmp_path / "agent-packages/platform/epistemic-graph",
        "lane.yaml",
        [_record("same", enqueued="2026-08-31T11:00:00+00:00")],
    )
    _fragment(
        tmp_path / "agent-packages/platform/epistemic-graph",
        CANONICAL_FRAGMENT,
        [_record("same", state="landed", recorded="2026-08-31T11:30:00+00:00")],
    )

    selected = discover_queued_repositories(tmp_path, now=NOW, max_age_seconds=86400)
    assert [item.identifier for item in selected] == ["plans"]


def test_missing_declared_root_and_manifest_path_drift_fail_loudly(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    (tmp_path / "plans").rename(tmp_path / "plans-missing")
    with pytest.raises(MergeQueueRunnerError, match="plans.*missing"):
        declared_repositories(tmp_path)

    other = tmp_path / "other"
    other.mkdir()
    manifest = yaml.safe_load((tmp_path / "workspace.yml").read_text())
    manifest["path"] = str(other)
    (tmp_path / "workspace.yml").write_text(yaml.safe_dump(manifest))
    with pytest.raises(MergeQueueRunnerError, match="manifest path"):
        declared_repositories(tmp_path)


def test_malformed_queue_state_is_not_hidden(tmp_path: Path) -> None:
    _workspace(tmp_path)
    _fragment(tmp_path / "plans", "lane.yaml", [_record("bad", state="mystery")])
    with pytest.raises(MergeQueueRunnerError, match="unknown state"):
        discover_queued_repositories(tmp_path, now=NOW)


def test_drain_uses_fixed_argv_and_preserves_lease_defer_code(tmp_path: Path) -> None:
    _workspace(tmp_path)
    repository = declared_repositories(tmp_path)[0]
    args_file = tmp_path / "args"
    executable = tmp_path / "fake-runner"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(args_file)!r}).write_text('\\n'.join(sys.argv[1:]))\n"
        "print('deferred')\n"
        "raise SystemExit(75)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = drain_repository(
        repository, executable=str(executable), deadline_seconds=10
    )
    assert result.deferred
    assert args_file.read_text(encoding="utf-8").splitlines() == [
        "--merge-queue",
        "run",
        "--repo-path",
        str(repository.path),
        "--queue-no-prune",
    ]


def test_drain_passes_an_explicit_lease_ttl_without_using_a_wrapper(
    tmp_path: Path,
) -> None:
    _workspace(tmp_path)
    repository = declared_repositories(tmp_path)[0]
    args_file = tmp_path / "args"
    executable = tmp_path / "fake-runner"
    executable.write_text(
        "#!/usr/bin/env python3\n"
        "import pathlib, sys\n"
        f"pathlib.Path({str(args_file)!r}).write_text('\\n'.join(sys.argv[1:]))\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = drain_repository(
        repository,
        executable=str(executable),
        deadline_seconds=10,
        lease_ttl_seconds=20,
    )
    assert result.returncode == 0
    assert args_file.read_text(encoding="utf-8").splitlines()[-2:] == [
        "--queue-lease-ttl-seconds",
        "20",
    ]


def test_runner_lease_ttl_is_independent_but_must_outlive_deadline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MERGE_QUEUE_LEASE_TTL_SECONDS", raising=False)
    args = build_parser().parse_args(["--drain-deadline-seconds", "300"])
    settings = _runner_settings(args)
    assert settings.deadline_seconds == 300
    assert settings.lease_ttl_seconds > settings.deadline_seconds
    bad = build_parser().parse_args(
        ["--drain-deadline-seconds", "300", "--lease-ttl-seconds", "300"]
    )
    with pytest.raises(MergeQueueRunnerError, match="must exceed drain deadline"):
        _runner_settings(bad)


def test_cgroup_escape_fails_closed_and_kills_the_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _workspace(tmp_path)
    repository = declared_repositories(tmp_path)[0]
    executable = tmp_path / "escape-runner"
    executable.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(10)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    current_pid = os.getpid()

    def fake_cgroup(pid: int) -> str:
        return "/unit.scope" if pid == current_pid else "/snap.scope"

    monkeypatch.setattr(
        "repository_manager.merge_queue_runner._process_cgroup", fake_cgroup
    )
    with pytest.raises(MergeQueueRunnerError, match="escaped its systemd cgroup"):
        drain_repository(repository, executable=str(executable), deadline_seconds=10)


def test_progressing_child_is_not_killed_by_low_concurrency_budget(
    tmp_path: Path,
) -> None:
    """The supervisor deadline does not become a worker/resource limit."""

    _workspace(tmp_path)
    repository = declared_repositories(tmp_path)[0]
    executable = tmp_path / "progressing-runner"
    executable.write_text(
        "#!/usr/bin/env python3\nimport time\ntime.sleep(0.2)\nraise SystemExit(0)\n",
        encoding="utf-8",
    )
    executable.chmod(0o755)
    result = drain_repository(
        repository, executable=str(executable), deadline_seconds=1
    )
    assert result.returncode == 0
