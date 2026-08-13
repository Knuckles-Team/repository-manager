"""P0.6 — invariants, not knobs.

Proves each converted knob is now a structural fact the runner enforces, not a
convention a caller can omit or a leaked environment variable can defeat.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from agent_utilities.governance.lanes import LaneScope, partitioned_paths

from repository_manager import merge_queue as mq
from repository_manager.test_commands import ensure_no_fail_fast, is_test_suite_command


# ---------------------------------------------------------------------------
# --no-fail-fast — the runner constructs the command; callers cannot omit it
# ---------------------------------------------------------------------------
class TestEnsureNoFailFast:
    def test_appends_to_cargo_test(self) -> None:
        assert ensure_no_fail_fast(["cargo", "test", "--all-features"]) == [
            "cargo",
            "test",
            "--all-features",
            "--no-fail-fast",
        ]

    def test_appends_to_cargo_nextest_run(self) -> None:
        assert ensure_no_fail_fast(["cargo", "nextest", "run"]) == [
            "cargo",
            "nextest",
            "run",
            "--no-fail-fast",
        ]

    def test_idempotent_when_already_present(self) -> None:
        argv = ["cargo", "test", "--no-fail-fast", "--all-features"]
        assert ensure_no_fail_fast(argv) == argv

    def test_leaves_non_test_commands_untouched(self) -> None:
        for argv in (
            ["cargo", "check", "--all-features"],
            ["cargo", "clippy"],
            ["pytest", "-q"],
            ["./gate.sh"],
        ):
            assert ensure_no_fail_fast(argv) == argv

    def test_is_test_suite_command_classification(self) -> None:
        assert is_test_suite_command(["cargo", "test"])
        assert is_test_suite_command(["cargo", "nextest", "run", "--all-features"])
        assert not is_test_suite_command(["cargo", "check"])
        assert not is_test_suite_command(["cargo"])
        assert not is_test_suite_command([])


def test_timed_run_injects_no_fail_fast_before_spawning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The declared command omits the flag; the SPAWNED argv must not.

    This is the actual process-launch chokepoint every gate/baseline run goes
    through (``run_gate`` and ``compute_gate_baseline`` both call only this) —
    proving it here proves it for both callers without needing a full queue
    fixture.
    """

    captured: dict[str, object] = {}

    def _fake_run(argv, **kwargs):  # noqa: ANN001 - test double signature
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mq.subprocess, "run", _fake_run)
    mq._timed_run(
        ["cargo", "test", "--all-features"],
        tmp_path,
        timeout=5,
        env={},
    )
    assert captured["argv"] == [
        "cargo",
        "test",
        "--all-features",
        "--no-fail-fast",
    ]


def test_timed_run_does_not_touch_a_non_test_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv, **kwargs):  # noqa: ANN001 - test double signature
        captured["argv"] = list(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(mq.subprocess, "run", _fake_run)
    mq._timed_run(["cargo", "check", "--all-features"], tmp_path, timeout=5, env={})
    assert captured["argv"] == ["cargo", "check", "--all-features"]


# ---------------------------------------------------------------------------
# Per-lane CARGO_TARGET_DIR / TMPDIR — allocated by the tool, not the caller
# ---------------------------------------------------------------------------
def _run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(
        cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _run("git init -q -b main", root)
    _run("git config user.email t@t.io && git config user.name t", root)
    _run("git config commit.gpgsign false", root)
    return root


def _commit(repo: Path, message: str) -> str:
    _run("git add -A", repo)
    _run(f"git commit -q --allow-empty -m {json.dumps(message)}", repo)
    return _run("git rev-parse HEAD", repo)


def test_run_fast_gates_overrides_a_leaked_cargo_target_dir_and_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hostile inherited env must not reach the spawned gate process.

    ``env = dict(os.environ)`` in ``run_fast_gates`` used to hand every gate
    command whatever ``CARGO_TARGET_DIR``/``TMPDIR`` the QUEUE PROCESS happened
    to have — a shared value here does not merely serialize concurrent
    worktree builds, it CORRUPTS them. This proves the bad state (a leaked,
    non-lane-scoped value reaching the subprocess) is now impossible: the
    queue always allocates its own, from the same ``partitioned_paths``
    lane_doctor itself uses, regardless of what the ambient environment holds.
    """

    repo = _init_repo(tmp_path / "envrepo")
    out_file = repo / "envcheck.out"
    script = (
        "import os, pathlib; "
        f"pathlib.Path({str(out_file)!r}).write_text("
        "os.environ.get('CARGO_TARGET_DIR', '') + chr(10) + "
        "os.environ.get('TMPDIR', '') + chr(10))"
    )
    (repo / "gitignore_marker.txt").write_text("x\n")
    _commit(repo, "init")

    hazard_target = "/tmp/shared-hazard-target"
    hazard_tmp = "/tmp"
    monkeypatch.setenv("CARGO_TARGET_DIR", hazard_target)
    monkeypatch.setenv("TMPDIR", hazard_tmp)

    scope = LaneScope(tree=repo, common_dir=repo / ".git", main_tree=repo, lane="test")
    config = mq.QueueConfig(
        base="main",
        gates=(
            mq.GateSpec(
                name="env-check",
                command=(sys.executable, "-c", script),
                tier="fast",
                timeout=30,
                compare="exit",
            ),
        ),
    )
    result = mq.run_fast_gates(
        repo,
        repo=repo,
        base_ref="main",
        scope=scope,
        config=config,
        base_config=config,
        changed=[],
    )
    assert result.ok, result.checks

    written = out_file.read_text().splitlines()
    observed_target, observed_tmp = written[0], written[1]
    expected = partitioned_paths(scope.tree)

    assert observed_target == str(expected.cargo_target_dir)
    assert observed_target != hazard_target
    assert Path(observed_target).is_relative_to(repo)

    assert observed_tmp == str(expected.scratch_dir)
    assert observed_tmp != hazard_tmp
    # `scratch_dir` (unlike `cargo_target_dir`) is deliberately rooted at a
    # short `$HOME/.al/<token>` path, not the (potentially deep) worktree —
    # see `agent_utilities.governance.lanes.partitioned_paths`'s own
    # docstring on why: short paths for AF_UNIX sockets. "Short by
    # construction" is what is being proven here, not tree-relativity.
    assert len(observed_tmp) < 60
