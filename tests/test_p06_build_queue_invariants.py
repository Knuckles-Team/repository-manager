"""P0.6 — invariants, not knobs, for the build broker (`build_queue.py`).

Real git repos and real subprocesses (the module's own test convention) except
where a hostile disk state must be *simulated* — no real disk can be pushed to
95% used inside a CI sandbox on demand, so `shutil.disk_usage` is faked there
and nowhere else.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager import build_queue as bq
from repository_manager.disk_policy import DiskDecisionCode


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


def _commit(repo: Path, message: str = "commit") -> str:
    _run("git add -A", repo)
    _run(f"git commit -q -m {message!r}", repo)
    return _run("git rev-parse HEAD", repo)


# ---------------------------------------------------------------------------
# CARGO_TARGET_DIR / TMPDIR — allocated by the runner, never inherited
# ---------------------------------------------------------------------------
def test_run_build_command_overrides_a_leaked_cargo_target_dir_and_tmpdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`_run_build_command` used to pass NO `env=` at all -- a plain
    `subprocess.run(argv, cwd=..., ...)` with env=None inherits the CALLING
    PROCESS's entire environment unmodified. Any ambient CARGO_TARGET_DIR/
    TMPDIR (an operator's shell, a misconfigured MCP server) reached the
    build with no override at all. This proves it can no longer.
    """

    tree = tmp_path / "buildtree"
    tree.mkdir()
    out_file = tree / "envcheck.out"
    script = (
        "import os, pathlib; "
        f"pathlib.Path({str(out_file)!r}).write_text("
        "os.environ.get('CARGO_TARGET_DIR', '') + chr(10) + "
        "os.environ.get('TMPDIR', '') + chr(10))"
    )
    _init_repo(tree)
    (tree / "marker.txt").write_text("x\n")
    _commit(tree)

    hazard_target = "/tmp/shared-hazard-target"
    hazard_tmp = "/tmp"
    monkeypatch.setenv("CARGO_TARGET_DIR", hazard_target)
    monkeypatch.setenv("TMPDIR", hazard_tmp)

    spec = bq.BuildSpec(
        name="envcheck",
        command=(sys.executable, "-c", script),
        workdir=".",
        timeout=30,
    )
    bq._run_build_command(tree, spec)

    written = out_file.read_text().splitlines()
    observed_target, observed_tmp = written[0], written[1]
    assert observed_target != hazard_target
    assert Path(observed_target) == tree / "target-isolated"
    assert observed_tmp != hazard_tmp
    assert len(observed_tmp) < 60  # short by construction — see PartitionedPaths


def test_run_build_command_injects_no_fail_fast(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(argv, **kwargs):  # noqa: ANN001 - test double signature
        captured["argv"] = list(argv)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(bq.subprocess, "run", _fake_run)
    spec = bq.BuildSpec(
        name="cargo-test", command=("cargo", "test", "--all-features"), timeout=30
    )
    bq._run_build_command(tmp_path, spec)
    assert captured["argv"] == ["cargo", "test", "--all-features", "--no-fail-fast"]


# ---------------------------------------------------------------------------
# Disk admission (P0.6) — a structural refusal, with a bounded self-heal
# ---------------------------------------------------------------------------
def _buildcache_yaml(*, disk_estimate_mb: int = 0) -> str:
    return textwrap.dedent(
        f"""
        base: main
        specs:
          - name: widget
            command: ["bash", "-c", "echo payload > out.txt"]
            workdir: "."
            cache_key_paths: ["src.txt"]
            artifacts: ["out.txt"]
            timeout: 30
            disk_estimate_mb: {disk_estimate_mb}
        """
    )


@pytest.fixture
def disk_repo(tmp_path: Path) -> Path:
    root = _init_repo(tmp_path / "repo")
    (root / "src.txt").write_text("v1\n")
    (root / bq.CONFIG_FILENAME).write_text(_buildcache_yaml(disk_estimate_mb=5000))
    _commit(root)
    return root


def _usage(total_gb: float, free_gb: float) -> SimpleNamespace:
    gib = 1024**3
    return SimpleNamespace(
        total=int(total_gb * gib), free=int(free_gb * gib), used=0
    )


def test_request_refuses_when_disk_is_over_the_high_watermark_and_gc_cannot_help(
    disk_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The bad state — a build proceeding while disk is critically low — is
    now impossible: it is refused BEFORE `_run_build_command` ever runs,
    never merely logged.
    """

    # 100 GiB total, 2 GiB free: past the default 90% high watermark AND gc()
    # has nothing to reclaim (a fresh repo has no prior cache entries), so the
    # self-heal attempt cannot change the outcome — this must still refuse.
    calls = {"n": 0}

    def _fake_disk_usage(_path):
        calls["n"] += 1
        return _usage(100, 2)

    monkeypatch.setattr(bq.shutil, "disk_usage", _fake_disk_usage)
    with pytest.raises(bq.BuildQueueError, match="disk admission"):
        bq.request(repo_path=disk_repo, colocated=True)
    # Both the initial evaluation and the post-gc() re-evaluation happened —
    # this is a real self-heal attempt, not a report that skipped it.
    assert calls["n"] >= 2


def test_request_self_heals_via_gc_when_reclaiming_frees_enough_space(
    disk_repo: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the FIRST observation is blocked but disk usage improves by the
    time gc() has run (simulating gc() actually reclaiming enough), the
    second, real evaluation admits the build — `_admit_disk` retries rather
    than refusing on the first observation alone.
    """

    bq._disk_policy().reset("local")
    observations = iter([_usage(100, 2), _usage(100, 50)])

    def _fake_disk_usage(_path):
        return next(observations)

    monkeypatch.setattr(bq.shutil, "disk_usage", _fake_disk_usage)
    result = bq.request(repo_path=disk_repo, colocated=True)
    assert result["ok"] is True


def test_admit_disk_admits_a_healthy_host_without_touching_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _init_repo(tmp_path / "healthy")
    (root / "src.txt").write_text("v1\n")
    (root / bq.CONFIG_FILENAME).write_text(_buildcache_yaml(disk_estimate_mb=10))
    _commit(root)

    gc_calls = {"n": 0}
    real_gc = bq.gc

    def _counting_gc(**kwargs):
        gc_calls["n"] += 1
        return real_gc(**kwargs)

    monkeypatch.setattr(bq, "gc", _counting_gc)
    monkeypatch.setattr(bq.shutil, "disk_usage", lambda _p: _usage(100, 90))
    bq._disk_policy().reset("local")
    spec = bq.load_config(root).spec()
    decision = bq._admit_disk(root, spec, reservation_id="t1")
    assert decision.admitted
    assert decision.code == DiskDecisionCode.ADMIT
    assert gc_calls["n"] == 0
