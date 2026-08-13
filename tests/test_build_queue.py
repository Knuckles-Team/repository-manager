"""Tests for the content-addressed build broker (CONCEPT:RM-TASK-LEDGER,
DELIVERABLE 2). Real git repos, real subprocesses — a mock build command would
prove nothing about whether dedup actually skips a rebuild.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager import build_queue as bq
from repository_manager import task_queue as tq


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


def _counter_spec(counter_path: Path) -> str:
    # `counter_path` is an ABSOLUTE path outside the repo tree on purpose: a
    # cache-hit-eligible (clean-tree) build runs in a THROWAWAY materialized
    # worktree elsewhere on disk (never in place), so a relative "../counter"
    # would land in a different directory each time and silently prove
    # nothing about whether the build command actually re-ran.
    return textwrap.dedent(
        f"""
        base: main
        specs:
          - name: widget
            command: ["bash", "-c", "echo built >> {counter_path}; echo payload > out.txt"]
            workdir: "."
            cache_key_paths: ["src.txt"]
            artifacts: ["out.txt"]
            timeout: 30
        """
    )


@pytest.fixture
def counter(tmp_path: Path) -> Path:
    return tmp_path / "counter.txt"


@pytest.fixture
def repo(tmp_path: Path, counter: Path) -> Path:
    root = _init_repo(tmp_path / "repo")
    (root / "src.txt").write_text("v1\n")
    (root / bq.CONFIG_FILENAME).write_text(_counter_spec(counter))
    _commit(root)
    return root


# ---------------------------------------------------------------------------
# config declaration — absence is a refusal, not a default
# ---------------------------------------------------------------------------
def test_missing_config_refuses(tmp_path: Path):
    root = _init_repo(tmp_path / "bare")
    (root / "README.md").write_text("hi\n")
    _commit(root)
    with pytest.raises(bq.BuildQueueError, match="has no .buildcache.yaml"):
        bq.load_config(root)


def test_command_must_be_argv_list_not_a_shell_string():
    with pytest.raises(bq.BuildQueueError, match="LIST of argv items"):
        bq.parse_config({"specs": [{"name": "x", "command": "echo hi"}]}, source="t")


def test_duplicate_spec_names_refused():
    data = {"specs": [{"name": "x", "command": ["a"]}, {"name": "x", "command": ["b"]}]}
    with pytest.raises(bq.BuildQueueError, match="duplicate spec name"):
        bq.parse_config(data, source="t")


# ---------------------------------------------------------------------------
# cache key — computable on a clean tree, honestly degraded on a dirty one
# ---------------------------------------------------------------------------
def test_cache_key_is_deterministic_for_the_same_clean_tree(repo: Path):
    config = bq.load_config(repo)
    spec = config.spec("widget")
    key1 = bq.compute_cache_key(repo, spec, repo_name="repo")
    key2 = bq.compute_cache_key(repo, spec, repo_name="repo")
    assert key1.computable and key2.computable
    assert key1.digest == key2.digest


def test_cache_key_changes_when_cache_key_path_content_changes(repo: Path):
    config = bq.load_config(repo)
    spec = config.spec("widget")
    before = bq.compute_cache_key(repo, spec, repo_name="repo")
    (repo / "src.txt").write_text("v2\n")
    _commit(repo)
    after = bq.compute_cache_key(repo, spec, repo_name="repo")
    assert before.digest != after.digest


def test_dirty_tree_is_honestly_degraded_not_silently_cached(repo: Path):
    (repo / "src.txt").write_text("uncommitted\n")
    config = bq.load_config(repo)
    spec = config.spec("widget")
    key = bq.compute_cache_key(repo, spec, repo_name="repo")
    assert not key.computable
    assert key.degraded_reason == "dirty-tree"
    with pytest.raises(bq.BuildQueueError, match="degraded CacheKey has no digest"):
        _ = key.digest


# ---------------------------------------------------------------------------
# request() — the dedup proof: a second identical request must NOT rebuild
# ---------------------------------------------------------------------------
def test_second_identical_request_is_served_from_cache_without_rebuilding(
    repo: Path, counter: Path
):
    first = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    assert first["ok"] is True
    assert first["cached"] is False
    assert len(counter.read_text().splitlines()) == 1, (
        "the build command must have run exactly once"
    )

    second = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    assert second["ok"] is True
    assert second["cached"] is True
    assert second["key"] == first["key"]
    assert len(counter.read_text().splitlines()) == 1, (
        "a cache hit must not run the build command again"
    )

    # The published artifact is real and checksummed, not a dangling claim.
    artifact = second["artifacts"][0]
    assert Path(artifact["stored_at"]).read_text().strip() == "payload"


def test_two_concurrent_same_key_requests_build_exactly_once(tmp_path: Path):
    """D-CDX-13: the check-then-enqueue race, reproduced with two REAL threads.

    Before the fix, `request()` read `find_task` unlocked, and only
    `_build()` (further down the SAME call) appended the RUNNING task record
    — so two same-key callers could both observe "nothing running" before
    either one appended, both build, and both publish. A slow build command
    widens that window to make the race deterministic rather than a coin
    flip; the counter file proves how many times the command actually ran.
    """
    import threading

    counter = tmp_path / "counter.txt"
    root = _init_repo(tmp_path / "repo")
    (root / "src.txt").write_text("v1\n")
    (root / bq.CONFIG_FILENAME).write_text(
        textwrap.dedent(
            f"""
            base: main
            specs:
              - name: widget
                command: ["bash", "-c", "sleep 0.5; echo built >> {counter}; echo payload > out.txt"]
                workdir: "."
                cache_key_paths: ["src.txt"]
                artifacts: ["out.txt"]
                timeout: 30
            """
        )
    )
    _commit(root)

    results: list[dict] = []
    barrier = threading.Barrier(2)

    def _caller() -> None:
        barrier.wait(timeout=5)  # maximize the odds both hit the check together
        results.append(bq.request(repo_path=root, spec_name="widget", colocated=True))

    threads = [threading.Thread(target=_caller) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
        assert not t.is_alive(), "a build request thread hung"

    assert len(results) == 2
    assert all(r["ok"] for r in results), results
    assert len({r["key"] for r in results}) == 1, "both requests computed the same key"
    lines = counter.read_text().splitlines() if counter.exists() else []
    assert len(lines) == 1, (
        f"the build command must have run exactly once, ran {len(lines)} times: {results}"
    )
    # exactly one caller actually built it; the other reused/waited for it.
    assert sum(1 for r in results if r["cached"] is False) == 1
    for r in results:
        assert r["artifacts"], r
        assert Path(r["artifacts"][0]["stored_at"]).read_text().strip() == "payload"


def test_a_changed_tree_busts_the_cache_and_rebuilds(repo: Path, counter: Path):
    first = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    (repo / "src.txt").write_text("v2\n")
    _commit(repo)
    second = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    assert second["cached"] is False
    assert second["key"] != first["key"]
    assert len(counter.read_text().splitlines()) == 2


def test_dirty_tree_request_builds_in_place_and_is_never_cached(repo: Path):
    (repo / "src.txt").write_text("dirty\n")
    result = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    assert result["ok"] is True
    assert result["degraded"] is True
    assert result["degraded_reason"] == "dirty-tree"
    assert result["cached"] is False
    assert result["key"] is None


def test_request_without_colocation_proof_is_refused(repo: Path):
    with pytest.raises(tq.ColocationRequired):
        bq.request(repo_path=repo, spec_name="widget", colocated=False)


def test_build_that_produces_no_declared_artifact_fails(tmp_path: Path):
    root = _init_repo(tmp_path / "empty-build")
    (root / "src.txt").write_text("v1\n")
    (root / bq.CONFIG_FILENAME).write_text(
        textwrap.dedent(
            """
            base: main
            specs:
              - name: widget
                command: ["true"]
                cache_key_paths: ["src.txt"]
                artifacts: ["nothing-*.bin"]
            """
        )
    )
    _commit(root)
    with pytest.raises(bq.BuildQueueError, match="produced no file matching"):
        bq.request(repo_path=root, spec_name="widget", colocated=True)


# ---------------------------------------------------------------------------
# status / artifacts / explain
# ---------------------------------------------------------------------------
def test_status_reports_computed_key_and_execution_class(repo: Path):
    report = bq.status(repo_path=repo, spec_name="widget")
    assert report["computable"] is True
    assert report["key"]
    assert report["execution_class"]["class"] == "build"


def test_artifacts_lookup_by_key(repo: Path):
    result = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    listing = bq.artifact_paths(repo_path=repo, key=result["key"])
    assert listing["artifacts"][0]["relative_path"] == "out.txt"


def test_explain_names_the_differing_component_after_a_source_change(repo: Path):
    first = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    (repo / "src.txt").write_text("v2\n")
    _commit(repo)
    explanation = bq.explain(repo_path=repo, key=first["key"], spec_name="widget")
    assert explanation["differing_components"]
    assert "tree_sha" in explanation["differing_components"]


def test_explain_on_a_key_that_would_still_hit_reports_no_diff(repo: Path):
    result = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    explanation = bq.explain(repo_path=repo, key=result["key"], spec_name="widget")
    assert explanation["differing_components"] is None


# ---------------------------------------------------------------------------
# gc — bounded reclamation that never touches a RUNNING task or the retained set
# ---------------------------------------------------------------------------
def _backdate_manifest(repo: Path, key: str, days_ago: int) -> None:
    manifest_path = bq._manifest_path(key, repo)
    manifest = json.loads(manifest_path.read_text())
    manifest["built_at"] = (
        dt.datetime.now(dt.UTC) - dt.timedelta(days=days_ago)
    ).isoformat()
    manifest_path.write_text(json.dumps(manifest))


def test_gc_reclaims_old_entries_but_keeps_recent_and_running(repo: Path):
    old = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    _backdate_manifest(repo, old["key"], days_ago=30)

    (repo / "src.txt").write_text("v2\n")
    _commit(repo)
    recent = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    assert recent["key"] != old["key"]

    result = bq.gc(repo_path=repo, keep_recent=1, max_age_days=14)
    assert old["key"] in result["removed"]
    assert recent["key"] in result["kept"]
    assert not bq._manifest_path(old["key"], repo).exists()
    assert bq._manifest_path(recent["key"], repo).exists()


def test_gc_never_removes_an_entry_with_a_running_task(repo: Path):
    result = bq.request(repo_path=repo, spec_name="widget", colocated=True)
    _backdate_manifest(repo, result["key"], days_ago=30)
    task = tq.find_task("build", result["key"], path=repo)
    assert task is not None
    tq.record_state(task, tq.RUNNING, "still going", path=repo)

    gc_result = bq.gc(repo_path=repo, keep_recent=0, max_age_days=0)
    assert result["key"] in gc_result["kept"]
    assert bq._manifest_path(result["key"], repo).exists()


# ---------------------------------------------------------------------------
# Three surfaces, one core — pinned so the CLI, the MCP tool, and `python -m`
# cannot drift (mirrors test_merge_queue.py's own surface contract tests).
# ---------------------------------------------------------------------------
def test_dispatch_routes_every_verb_and_names_the_unknown_ones(repo: Path):
    assert bq.dispatch("status", path=repo, spec="widget")["computable"] is True
    bad = bq.dispatch("nope", path=repo)
    assert bad["ok"] is False and "request" in bad["actions"]


def test_the_mcp_tool_is_registered_and_declares_every_action() -> None:
    """Registration alone is not the contract — the tool must ADVERTISE every
    action it routes, or a caller cannot discover the verb it needs.
    """
    from fastmcp import FastMCP

    from repository_manager.mcp_server import (
        RM_BUILD_ACTIONS,
        register_project_management_tools,
    )

    assert set(RM_BUILD_ACTIONS) == {"request", "status", "artifacts", "explain", "gc"}
    # Every advertised MCP action must exist in the shared dispatch core, and
    # vice versa — this is what stops the two surfaces drifting apart.
    assert set(bq.dispatch("__probe__")["actions"]) == set(RM_BUILD_ACTIONS)

    mcp = FastMCP("probe")
    register_project_management_tools(mcp)
    names = {t.name for t in asyncio.run(mcp.list_tools())}
    assert "rm_build" in names, sorted(names)


def test_the_cli_flag_routes_to_the_same_dispatch_core(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    from repository_manager.repository_manager import _run_build_queue_cli

    args = SimpleNamespace(
        build_broker="status",
        repo_path=str(repo),
        build_spec="widget",
        build_key="",
        same_node=False,
        build_wait_timeout=60,
        build_keep_recent=10,
        build_max_age_days=14,
        build_host=None,
    )
    assert _run_build_queue_cli(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["computable"] is True
    assert payload["spec"] == "widget"


def test_the_cli_defaults_to_not_colocated_and_is_refused_for_request(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    """A bare CLI invocation cannot prove it shares a node with the lease
    holder, so `request` (lease-backed) must be refused by default — silently
    trusting the lock would reintroduce the exact false-safety this gate
    exists to prevent.
    """
    from repository_manager.repository_manager import _run_build_queue_cli

    args = SimpleNamespace(
        build_broker="request",
        repo_path=str(repo),
        build_spec="widget",
        build_key="",
        same_node=False,
        build_wait_timeout=60,
        build_keep_recent=10,
        build_max_age_days=14,
        build_host=None,
    )
    assert _run_build_queue_cli(args) == 1
    payload = json.loads(capsys.readouterr().out)
    assert "refused" in payload
    assert (
        "colocated=True" in payload["refused"]
        or "co-location" in payload["refused"].lower()
    )


def test_the_python_dash_m_entrypoint_routes_to_the_same_dispatch_core(
    repo: Path, capsys: pytest.CaptureFixture
) -> None:
    from repository_manager.build_queue import main as build_queue_main

    exit_code = build_queue_main(["status", "--path", str(repo), "--spec", "widget"])
    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["computable"] is True
