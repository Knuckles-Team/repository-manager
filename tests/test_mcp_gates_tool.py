"""End-to-end proof for the `rm_gates` MCP tool (GOC-60/P0.4).

Runs real ``pre-commit`` against real temp repos through the actual MCP tool
call path (``mcp.list_tools()`` -> ``tool.fn(...)``), not the internal
``gates.run_gate_stage`` unit tested in ``tests/test_gates.py``. Proves, at
this layer:

1. ``stage="heavy"`` fires a hook declared ONLY at ``pre-push``; ``stage="fast"``
   does not. This is the literal fix for GOC-60's blocking gap.
2. ``run`` across >=3 repos genuinely executes them in PARALLEL. Proven
   DETERMINISTICALLY via a file-based rendezvous barrier inside each
   heavy-only hook (``_BARRIER_HOOK_SCRIPT``): every hook blocks until it
   observes that all N repos have started before doing its fixed-duration
   work, so the test can assert the hooks' intervals actually overlapped
   instead of inferring parallelism from a wall-clock ceiling. A wall-clock
   ceiling is host-load dependent by construction -- this replaced an
   earlier version of this test that asserted ``wall_s < N * per_hook * 2.5``
   directly, which failed at 21.8s on a loaded host (>2.5x margin blown by
   scheduling/subprocess-startup delay alone, not by a code defect) and
   passed in ~8s at low load. See ``_assert_heavy_hooks_overlapped``.
3. ``profile`` returns real measured per-hook timings from a real run.
4. ``status``/``explain`` read back real per-repo results.
"""

import concurrent.futures
import shlex
import subprocess
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from repository_manager.mcp_server import _jobs, _jobs_lock, get_mcp_instance
from tests.conftest import isolated_git_subprocess_env

#: Fixed-duration work each heavy-only hook performs -- AFTER it has cleared
#: the rendezvous barrier below, so this sleep is provably shared by every
#: repo's hook. Kept short; it exists only to give ``profile`` a measurable,
#: non-trivial ``duration_s`` floor (see ``_verify_profile``), not to prove
#: parallelism -- the barrier proves that.
_HEAVY_HOOK_SLEEP_S = 2.0

#: How many repos' heavy-only hooks must rendezvous at the barrier.
_EXPECTED_HEAVY_REPOS = 3

#: Safety timeout for the barrier wait -- generous on purpose. Its only job
#: is to turn a genuinely serial invocation (repo B's hook process does not
#: even start until repo A's has finished) into a fast, clear
#: ``BARRIER_TIMEOUT`` diagnostic instead of a silent hang. It must never be
#: read as a parallelism assertion -- reaching the barrier at all, and the
#: overlap check in ``_assert_heavy_hooks_overlapped``, are what prove that.
_BARRIER_TIMEOUT_S = 45.0

#: How long the MCP job-status poll loop waits for the heavy stage to finish.
#: Must comfortably exceed ``_BARRIER_TIMEOUT_S + _HEAVY_HOOK_SLEEP_S`` so a
#: slow-but-successful barrier rendezvous under real host load is not itself
#: mistaken for a hang by the test's own polling, not the gate.
_HEAVY_POLL_TIMEOUT_S = 90.0

#: One `pre-commit` "system" hook per repo under test runs this script. It
#: implements the rendezvous barrier described in the module docstring:
#: write a start marker naming this repo, block until every other repo's
#: start marker is also present (proof that all were in flight at once),
#: only then do the fixed-duration work and write an end marker. Written
#: once per test to a shared file (see ``_write_barrier_script``) and
#: invoked with per-repo argv rather than embedded per-repo, so the barrier
#: logic itself is proven identical across repos and lives in one place.
_BARRIER_HOOK_SCRIPT = '''\
import pathlib
import sys
import time


def main() -> int:
    repo_name = sys.argv[1]
    barrier_dir = pathlib.Path(sys.argv[2])
    sleep_s = float(sys.argv[3])
    timeout_s = float(sys.argv[4])
    expected = int(sys.argv[5])

    barrier_dir.mkdir(parents=True, exist_ok=True)
    (barrier_dir / f"{repo_name}.start").write_text(repr(time.time()))

    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if len(list(barrier_dir.glob("*.start"))) >= expected:
            break
        time.sleep(0.05)
    else:
        started = len(list(barrier_dir.glob("*.start")))
        sys.stderr.write(
            f"BARRIER_TIMEOUT: {repo_name} waited {timeout_s}s but only "
            f"{started}/{expected} repos reached the barrier -- this proves "
            "serial (or stalled) execution, not parallel.\\n"
        )
        return 1

    # Every `expected` repo's hook was in flight (started, not yet finished)
    # at this instant. Do the fixed-duration work now, so its interval is
    # provably shared by all of them.
    time.sleep(sleep_s)
    (barrier_dir / f"{repo_name}.end").write_text(repr(time.time()))
    print("HEAVY_ONLY_RAN")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _write_barrier_script(tmp_path: Path) -> Path:
    """Write the shared barrier-hook script once for a test run."""
    script = tmp_path / "barrier_hook.py"
    script.write_text(_BARRIER_HOOK_SCRIPT)
    return script


def _heavy_hook_entry(
    *,
    barrier_script: Path,
    repo_name: str,
    barrier_dir: Path,
    heavy_timeout_s: float,
    expected_heavy_repos: int,
) -> str:
    """Shell-quoted ``pre-commit`` ``entry:`` invoking the barrier script.

    Split out of :func:`_init_tiered_repo` so that function's own complexity
    stays at its pre-barrier baseline (straight-line setup, no branching) --
    kept here instead, this helper's own generator expression counts against
    a function that never had a lower baseline to regress from.
    """
    argv = (
        "python3",
        str(barrier_script),
        repo_name,
        str(barrier_dir),
        str(_HEAVY_HOOK_SLEEP_S),
        str(heavy_timeout_s),
        str(expected_heavy_repos),
    )
    return " ".join(shlex.quote(part) for part in argv)


def _init_tiered_repo(
    path: Path,
    *,
    barrier_script: Path,
    barrier_dir: Path,
    heavy_timeout_s: float = _BARRIER_TIMEOUT_S,
    expected_heavy_repos: int = _EXPECTED_HEAVY_REPOS,
) -> None:
    env = isolated_git_subprocess_env()
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=path, check=True, env=env)  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.email", "a@b.c"], cwd=path, check=True, env=env
    )  # nosec B603 B607
    subprocess.run(
        ["git", "config", "user.name", "test"], cwd=path, check=True, env=env
    )  # nosec B603 B607
    # ``path.name`` (e.g. "repo-a") identifies this repo's marker files at the
    # shared barrier -- distinct per repo, filesystem-safe, and known to the
    # test up front (it chose the directory name), so no round-trip through
    # the MCP tool's own job/repo identifiers is needed to read the markers
    # back afterwards.
    heavy_entry = _heavy_hook_entry(
        barrier_script=barrier_script,
        repo_name=path.name,
        barrier_dir=barrier_dir,
        heavy_timeout_s=heavy_timeout_s,
        expected_heavy_repos=expected_heavy_repos,
    )
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
        f"    entry: {heavy_entry}\n"
        "    language: system\n"
        "    always_run: true\n"
        "    pass_filenames: false\n"
        "    stages: [pre-push, manual]\n"
    )
    # A minimal, valid `.mergequeue.yaml` -- `gates.py`'s best-effort ledger
    # metadata (`_gate_ledger_metadata`) calls `merge_queue.load_config` on
    # every gate run purely to fingerprint the toolchain, and degrades
    # gracefully (a WARNING + full traceback to stderr, never a raised
    # exception or a changed verdict) when the file is absent -- which it
    # always was here before this line. That noise is harmless but was
    # mistaken for this test's real failure cause by an earlier investigator
    # (see the module docstring); declaring an empty, valid gate list removes
    # the noise at its source instead of leaving a misleading trail.
    (path / ".mergequeue.yaml").write_text("schema_version: 2\nbase: main\ngates: []\n")
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


async def _run_and_verify_fast_stage(rm_gates) -> None:
    """FAST stage: the pre-push-only hook must NOT fire."""
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


def _read_barrier_timestamp(barrier_dir: Path, repo_name: str, marker: str) -> float:
    marker_path = barrier_dir / f"{repo_name}.{marker}"
    assert marker_path.exists(), (
        f"heavy-only hook for {repo_name!r} never wrote its {marker!r} barrier "
        f"marker at {marker_path} -- it did not run to completion (check the "
        "hook's captured output for a BARRIER_TIMEOUT diagnostic)."
    )
    return float(marker_path.read_text())


def _assert_heavy_hooks_overlapped(barrier_dir: Path, repo_names: list[str]) -> None:
    """Deterministic proof that every repo's heavy-only hook was in flight
    at the same instant -- not inferred from a wall-clock duration bound.

    Each hook (``_BARRIER_HOOK_SCRIPT``) only starts its fixed-duration work
    after observing every repo's barrier "start" marker, so the LATEST start
    necessarily precedes the EARLIEST end whenever the hooks genuinely
    overlapped. A serial run cannot produce this: the next repo's hook does
    not even begin until the previous one has finished, so either its start
    would not precede the earlier one's end, or (more likely, since the
    barrier blocks it) it would never clear the barrier at all and
    ``_read_barrier_timestamp`` above would already have failed loudly.
    """
    starts = {name: _read_barrier_timestamp(barrier_dir, name, "start") for name in repo_names}
    ends = {name: _read_barrier_timestamp(barrier_dir, name, "end") for name in repo_names}
    latest_start = max(starts.values())
    earliest_end = min(ends.values())
    assert latest_start < earliest_end, (
        f"heavy-only hooks did not overlap (latest start {latest_start} is not "
        f"before earliest end {earliest_end}); starts={starts} ends={ends} -- "
        "this means the repos ran serially, not in parallel."
    )


async def _run_and_verify_heavy_stage(
    rm_gates, barrier_dir: Path, repo_names: list[str]
) -> dict:
    """HEAVY stage: the pre-push-only hook MUST fire, N repos run concurrently.

    Parallelism is proven deterministically -- see
    ``_assert_heavy_hooks_overlapped`` and the module docstring -- rather than
    by asserting a wall-clock ceiling.
    """
    with _jobs_lock:
        _jobs.clear()
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
    heavy_results = await _poll_until_done(
        rm_gates, heavy_jobs, timeout_s=_HEAVY_POLL_TIMEOUT_S
    )

    for repo_name, status in heavy_results.items():
        assert status["status"] == "completed", (repo_name, status)
        assert status["outcome"] == "succeeded", (repo_name, status)
    with _jobs_lock:
        for repo_name, jid in heavy_jobs.items():
            result = _jobs[jid]["result"]
            ran = {h.hook_id for h in result.hooks}
            assert "heavy-only" in ran, f"{repo_name}: pre-push-only hook never fired"
            assert "HEAVY_ONLY_RAN" in result.raw_output
            assert "fast-only" not in ran  # default_stages excludes it at pre-push
            assert result.stage == "heavy"

    _assert_heavy_hooks_overlapped(barrier_dir, repo_names)
    return heavy_jobs


async def _verify_profile(rm_gates) -> None:
    """profile: real per-hook timings from the real heavy run."""
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


async def _verify_explain(rm_gates, one_repo: str) -> None:
    """explain: condensed detail for one repo by name."""
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


async def _verify_status_rollup(rm_gates) -> None:
    """status roll-up (no job_id): counts + failed set."""
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


@pytest.mark.anyio
async def test_rm_gates_run_is_parallel_and_stage_scoped(tmp_path, monkeypatch):
    """The main proof: heavy fires the pre-push-only hook; N repos run concurrently.

    Parallelism is proven deterministically via a rendezvous barrier in the
    heavy-only hook, not a wall-clock ceiling -- see the module docstring and
    ``_assert_heavy_hooks_overlapped``.
    """
    # Isolate pre-commit's own cache (``~/.cache/pre-commit/db.db`` by
    # default). Measured under artificial host load (50 CPU hogs, load
    # average ~70-90): with the AMBIENT, host-wide cache shared by every
    # concurrent `pre-commit` invocation on the box (this test's own 3
    # repos AND any other lane's), pre-commit's sqlite3 connection
    # intermittently raised ``OperationalError: database is locked``,
    # which surfaced as a genuine gate FAILURE (exit 3, "no parseable hook
    # results") -- a real flake, but caused by lock contention on a
    # host-shared file this fixture does not need to share, not by
    # anything under test. A dedicated, per-test cache directory removes
    # the shared writer entirely; confirmed 0 failures across repeated
    # runs at load ~76-90 after this change (was reproducible before it).
    monkeypatch.setenv("PRE_COMMIT_HOME", str(tmp_path / "pre-commit-home"))
    # The production pool is sized from a share of the host (one worker on a
    # 4-CPU runner); this test proves the tool fans repos out concurrently, so
    # it supplies a pool wide enough for every repo instead of depending on
    # the size of the machine it runs on.
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=_EXPECTED_HEAVY_REPOS)
    monkeypatch.setattr("repository_manager.mcp_server._executor", executor)

    repo_names = ["repo-a", "repo-b", "repo-c"]
    barrier_dir = tmp_path / "barrier"
    barrier_script = _write_barrier_script(tmp_path)

    repo_paths = {}
    for name in repo_names:
        p = tmp_path / name
        _init_tiered_repo(
            p,
            barrier_script=barrier_script,
            barrier_dir=barrier_dir,
            expected_heavy_repos=len(repo_names),
        )
        repo_paths[f"https://example.invalid/{name}.git"] = str(p)

    mock_git = MagicMock()
    mock_git.project_map = repo_paths

    with _jobs_lock:
        _jobs.clear()

    with patch("repository_manager.mcp_server.get_git_instance", return_value=mock_git):
        rm_gates = await _get_rm_gates_tool()

        await _run_and_verify_fast_stage(rm_gates)
        heavy_jobs = await _run_and_verify_heavy_stage(rm_gates, barrier_dir, repo_names)
        await _verify_profile(rm_gates)
        one_repo = next(iter(heavy_jobs))
        await _verify_explain(rm_gates, one_repo)
        await _verify_status_rollup(rm_gates)
