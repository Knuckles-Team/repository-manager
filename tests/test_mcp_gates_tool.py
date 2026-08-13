"""End-to-end proof for the `rm_gates` MCP tool (GOC-60/P0.4).

Runs real ``pre-commit`` against real temp repos through the actual MCP tool
call path (``mcp.list_tools()`` -> ``tool.fn(...)``), not the internal
``gates.run_gate_stage`` unit tested in ``tests/test_gates.py``. Proves, at
this layer:

1. ``stage="heavy"`` fires a hook declared ONLY at ``pre-push``; ``stage="fast"``
   does not. This is the literal fix for GOC-60's blocking gap.
2. ``run`` across >=3 repos genuinely executes them in PARALLEL (wall time
   close to one repo's duration, not the sum) using the existing bounded job
   pool -- no second executor invented.
3. ``profile`` returns real measured per-hook timings from a real run.
4. ``status``/``explain`` read back real per-repo results.
"""

import subprocess
import time
from unittest.mock import MagicMock, patch

import pytest

from repository_manager.mcp_server import _jobs, _jobs_lock, get_mcp_instance
from tests.conftest import isolated_git_subprocess_env

#: Each repo's heavy-only hook sleeps this long -- long enough that serial
#: execution (3x) is unambiguously distinguishable from parallel (~1x) even
#: under CI scheduling noise, short enough not to make the suite slow.
_HEAVY_HOOK_SLEEP_S = 2.0


def _init_tiered_repo(path):
    env = isolated_git_subprocess_env()
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.email", "a@b.c"], cwd=path, check=True, env=env
    )  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=path, check=True, env=env
    )  # nosec B603 B607
    (path / ".pre-commit-config.yaml").write_text(
        "default_stages: [pre-commit]\n"
        "repos:\n"
        "- repo: local\n"
        "  hooks:\n"
        "  - id: fast-only\n"
        "    name: fast-only\n"
        "    entry: python3 -c \"print('FAST_ONLY_RAN')\"\n"
        "    language: system\n"
        "    always_run: true\n"
        "    pass_filenames: false\n"
        "  - id: heavy-only\n"
        "    name: heavy-only\n"
        f"    entry: python3 -c \"import time; time.sleep({_HEAVY_HOOK_SLEEP_S}); print('HEAVY_ONLY_RAN')\"\n"
        "    language: system\n"
        "    always_run: true\n"
        "    pass_filenames: false\n"
        "    stages: [pre-push, manual]\n"
    )
    (path / "file.txt").write_text("hello\n")
    subprocess.run(["git", "add", "-A"], cwd=path, check=True, env=env)  # nosec B603 B607
    subprocess.run(["git", "commit", "-q", "-m", "init"], cwd=path, check=True, env=env)  # nosec B603 B607


async def _get_rm_gates_tool():
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    return next(t for t in tools if t.name == "rm_gates")


async def _poll_until_done(rm_gates, job_ids, timeout_s=60.0):
    deadline = time.monotonic() + timeout_s
    done: dict[str, dict] = {}
    while time.monotonic() < deadline and len(done) < len(job_ids):
        for repo_name, jid in job_ids.items():
            if repo_name in done:
                continue
            status = await rm_gates.fn(
                action="status",
                job_id=jid,
                repos=None,
                stage="fast",
                threads=None,
                timeout=600,
                repo=None,
                summary=False,
                top_n=15,
                ctx=None,
            )
            if status.get("status") in ("completed", "failed"):
                done[repo_name] = status
        if len(done) < len(job_ids):
            import asyncio

            await asyncio.sleep(0.1)
    return done


@pytest.mark.anyio
async def test_rm_gates_run_is_parallel_and_stage_scoped(tmp_path):
    """The main proof: heavy fires the pre-push-only hook; N repos run concurrently."""
    repo_paths = {}
    for name in ("repo-a", "repo-b", "repo-c"):
        p = tmp_path / name
        _init_tiered_repo(p)
        repo_paths[f"https://example.invalid/{name}.git"] = str(p)

    mock_git = MagicMock()
    mock_git.project_map = repo_paths

    with _jobs_lock:
        _jobs.clear()

    with patch("repository_manager.mcp_server.get_git_instance", return_value=mock_git):
        rm_gates = await _get_rm_gates_tool()

        # --- 1. FAST stage: the pre-push-only hook must NOT fire. ---
        fast_submit = await rm_gates.fn(
            action="run",
            stage="fast",
            repos=None,
            threads=None,
            timeout=600,
            job_id=None,
            repo=None,
            summary=True,
            top_n=15,
            ctx=None,
        )
        assert fast_submit["status"] == "submitted"
        assert fast_submit["queued_count"] == 3
        fast_jobs = fast_submit["jobs"]
        fast_results = await _poll_until_done(rm_gates, fast_jobs)
        for repo_name, status in fast_results.items():
            assert status["status"] == "completed", (repo_name, status)
            assert status["outcome"] == "succeeded"
        # RepoScanResult has no to_markdown/model_dump summary rendering, so
        # inspect the raw job records directly for the parsed hooks instead.
        with _jobs_lock:
            for jid in fast_jobs.values():
                result = _jobs[jid]["result"]
                ran = {h.hook_id for h in result.hooks}
                assert "fast-only" in ran
                assert "heavy-only" not in ran
                assert "HEAVY_ONLY_RAN" not in result.raw_output
                assert result.stage == "fast"

        # --- 2. HEAVY stage: the pre-push-only hook MUST fire, and the 3 ---
        # ---    repos must run concurrently (~1x sleep, not ~3x). ---
        with _jobs_lock:
            _jobs.clear()
        started = time.monotonic()
        heavy_submit = await rm_gates.fn(
            action="run",
            stage="heavy",
            repos=None,
            threads=None,
            timeout=600,
            job_id=None,
            repo=None,
            summary=True,
            top_n=15,
            ctx=None,
        )
        heavy_jobs = heavy_submit["jobs"]
        heavy_results = await _poll_until_done(rm_gates, heavy_jobs)
        wall_s = time.monotonic() - started

        for repo_name, status in heavy_results.items():
            assert status["status"] == "completed", (repo_name, status)
            assert status["outcome"] == "succeeded"
        with _jobs_lock:
            for repo_name, jid in heavy_jobs.items():
                result = _jobs[jid]["result"]
                ran = {h.hook_id for h in result.hooks}
                assert "heavy-only" in ran, (
                    f"{repo_name}: pre-push-only hook never fired"
                )
                assert "HEAVY_ONLY_RAN" in result.raw_output
                assert "fast-only" not in ran  # default_stages excludes it at pre-push
                assert result.stage == "heavy"

        # Parallel proof: 3 repos x _HEAVY_HOOK_SLEEP_S serial would be >=3x;
        # generously bound at 2.5x to absorb scheduling/pre-commit startup
        # overhead while still clearly separating parallel from serial.
        assert wall_s < _HEAVY_HOOK_SLEEP_S * 2.5, (
            f"3 repos took {wall_s:.2f}s wall for a {_HEAVY_HOOK_SLEEP_S}s hook each "
            "-- looks serial, not parallel"
        )

        # --- 3. profile: real per-hook timings from the real heavy run. ---
        profile = await rm_gates.fn(
            action="profile",
            repos=None,
            stage="fast",
            threads=None,
            timeout=600,
            job_id=None,
            repo=None,
            summary=True,
            top_n=15,
            ctx=None,
        )
        assert profile["measured_gate_jobs"] == 3
        slow_hooks = {h["hook_id"] for h in profile["slowest_hooks"]}
        assert "heavy-only" in slow_hooks
        heavy_entry = next(
            h for h in profile["slowest_hooks"] if h["hook_id"] == "heavy-only"
        )
        assert heavy_entry["duration_s"] is not None
        assert heavy_entry["duration_s"] >= _HEAVY_HOOK_SLEEP_S * 0.5

        # --- 4. explain: condensed detail for one repo by name. ---
        one_repo = next(iter(heavy_jobs))
        explanation = await rm_gates.fn(
            action="explain",
            repo=one_repo,
            repos=None,
            stage="fast",
            threads=None,
            timeout=600,
            job_id=None,
            summary=True,
            top_n=15,
            ctx=None,
        )
        assert explanation["passed"] is True
        assert "passed" in explanation["explain"]

        # --- 5. status roll-up (no job_id): counts + failed set. ---
        rollup = await rm_gates.fn(
            action="status",
            repos=None,
            stage="fast",
            threads=None,
            timeout=600,
            job_id=None,
            repo=None,
            summary=True,
            top_n=15,
            ctx=None,
        )
        assert rollup["summary"]["total"] == 3
        assert rollup["summary"]["passed"] == 3
        assert rollup["failed_projects"] == []
