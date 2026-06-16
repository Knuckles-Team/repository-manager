import asyncio
from unittest.mock import MagicMock, patch

import pytest

from repository_manager.mcp_server import (
    _get_job_status,
    _jobs,
    _jobs_lock,
    _submit_job,
    get_git_instance,
    get_mcp_instance,
    mcp_server,
)
from repository_manager.models import GitResult


@pytest.mark.anyio
async def test_background_job_lifecycle():
    """Verify background job submission, status polling, and result formatting."""

    # 1. Successful job
    def success_fn(x, y):
        return x + y

    res = _submit_job("test_success", success_fn, 5, y=10)
    assert res["status"] == "submitted"
    job_id = res["job_id"]
    assert job_id is not None

    # Poll for completion
    for _ in range(50):
        status_info = _get_job_status(job_id)
        if status_info["status"] == "completed":
            break
        await asyncio.sleep(0.05)

    assert status_info["status"] == "completed"
    assert status_info["result"] == "15"

    # 2. Failing job
    def fail_fn():
        raise ValueError("mock_failure")

    res_fail = _submit_job("test_fail", fail_fn)
    job_id_fail = res_fail["job_id"]

    for _ in range(50):
        status_info = _get_job_status(job_id_fail)
        if status_info["status"] == "failed":
            break
        await asyncio.sleep(0.05)

    assert status_info["status"] == "failed"
    assert "mock_failure" in status_info["error"]


def test_get_job_status_variations():
    """Test edge cases in _get_job_status such as listing all, empty jobs, and progress parsing."""
    # Clean up jobs
    with _jobs_lock:
        _jobs.clear()

    # Polling empty
    empty_res = _get_job_status()
    assert empty_res["status"] == "empty"

    # Polling non-existent job
    non_existent = _get_job_status("non_existent_id")
    assert non_existent["status"] == "error"

    # Inject job with complex progress details
    mock_job = {
        "status": "running",
        "action": "validate",
        "started_at": "2026-05-22T00:00:00Z",
        "completed_at": None,
        "result": None,
        "error": None,
        "progress_detail": {
            "current_phase": "Phase 2",
            "progress": 50,
            "phases": {
                "Phase 1": {
                    "repos": {
                        "repo_a": "success",
                        "repo_b": "failed",
                    }
                },
                "Phase 2": {
                    "repos": {
                        "repo_c": "running",
                        "repo_d": "pending",
                    }
                },
            },
        },
    }
    with _jobs_lock:
        _jobs["job-abc"] = mock_job

    # Terse default: counts + failed set + active names (no full per-repo dump).
    status = _get_job_status("job-abc")
    assert status["current_phase"] == "Phase 2"
    assert status["progress"] == 50
    assert status["counts"]["completed"] == 2  # repo_a (success) + repo_b (failed)
    assert status["counts"]["active"] == 1
    assert status["counts"]["remaining"] == 1
    assert status["failed_projects"] == ["repo_b"]
    assert "repo_c" in status["active_projects"]
    assert "completed_projects" not in status  # omitted in terse mode

    # Full per-repo detail is opt-in via summary=False.
    full = _get_job_status("job-abc", summary=False)
    assert "repo_a" in full["completed_projects"]
    assert "repo_b" in full["completed_projects"]
    assert "repo_c" in full["active_projects"]
    assert "repo_d" in full["remaining_projects"]
    assert full["failed_projects"] == ["repo_b"]

    # List all jobs — terse roll-up by default (omits the per-job dump for
    # scale); the full dump is opt-in via summary=False.
    terse_jobs = _get_job_status()
    assert "summary" in terse_jobs and "jobs" not in terse_jobs
    all_jobs = _get_job_status(summary=False)
    assert "jobs" in all_jobs
    assert "job-abc" in all_jobs["jobs"]

    # Test completed job with Pydantic model result
    mock_model_result = MagicMock()
    mock_model_result._format_timestamp_for_path.return_value = "20260522_120000"

    mock_job_comp = {
        "status": "completed",
        "action": "validate",
        "started_at": "2026-05-22T00:00:00Z",
        "completed_at": "2026-05-22T00:01:00Z",
        "result": mock_model_result,
        "error": None,
    }

    with _jobs_lock:
        _jobs["job-comp"] = mock_job_comp

    status_comp = _get_job_status("job-comp")
    assert "Validation completed" in status_comp["result"]


def test_get_git_instance_fallbacks(tmp_path):
    """Verify loading path and fallbacks when workspace YAML file is absent."""
    workspace_dir = str(tmp_path / "workspace")
    git = get_git_instance(path=workspace_dir)
    assert git is not None


@pytest.mark.anyio
async def test_mcp_rm_git_tool(tmp_path):
    """Test all actions and parameter checks of rm_git tool."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    rm_git = next(t for t in tools if t.name == "rm_git")

    # Mock Git Instance
    mock_git = MagicMock()
    mock_git.git_action.return_value = GitResult(
        status="success", data="mock_git_output"
    )
    mock_git.clone_projects.return_value = "cloned"
    mock_git.pull_projects.return_value = "pulled"
    mock_git.push_projects.return_value = "pushed"
    mock_git.phased_push.return_value = "phased_pushed"
    mock_git.path = str(tmp_path)
    # _resolve_repo_dir() resolves a bare name to a real on-disk dir via
    # os.path.exists() under the workspace path. Create the relative repo dir so
    # resolution is deterministic — the pull/push assertions below previously
    # depended on a stray ``/tmp/repo-a`` happening to exist (flaky).
    (tmp_path / "repo-a").mkdir()
    repo_a_dir = str(tmp_path / "repo-a")

    with patch("repository_manager.mcp_server.get_git_instance", return_value=mock_git):
        # 1. Action: raw (missing command)
        res = await rm_git.fn(
            action="raw",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            ctx=None,
        )
        assert res.status == "error"
        assert "command is required" in res.error.message

        # 2. Action: raw (with command)
        res = await rm_git.fn(
            action="raw",
            command="git status",
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            ctx=None,
        )
        assert res.status == "success"
        assert res.data == "mock_git_output"
        mock_git.git_action.assert_called_with(command="git status", path=None)

        # 3. Action: clone
        res = await rm_git.fn(
            action="clone",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            projects=None,
            ctx=None,
        )
        assert res["status"] == "submitted"

        # 3b. Action: clone with custom projects
        res = await rm_git.fn(
            action="clone",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            projects="https://github.com/org/repo-a.git, https://github.com/org/repo-b.git",
            ctx=None,
        )
        assert res["status"] == "submitted"
        # Steps 3 and 3b both submit clone jobs on background executor threads that
        # share this mock; completion order is nondeterministic. Wait until the
        # SPECIFIC step-3b call (projects=[...]) has actually run — not merely until
        # any clone fired — then assert it happened (assert_any_call).
        expected = [
            "https://github.com/org/repo-a.git",
            "https://github.com/org/repo-b.git",
        ]
        for _ in range(300):
            if any(
                c.kwargs.get("projects") == expected
                for c in mock_git.clone_projects.call_args_list
            ):
                break
            await asyncio.sleep(0.01)
        mock_git.clone_projects.assert_any_call(projects=expected)

        # 4. Action: pull
        res = await rm_git.fn(
            action="pull",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            projects=None,
            ctx=None,
        )
        assert res["status"] == "submitted"

        # 4b. Action: pull with custom projects
        res = await rm_git.fn(
            action="pull",
            command=None,
            path="/tmp",
            threads=None,
            phase=1,
            target_project=None,
            projects="repo-a, /absolute/path/repo-b",
            ctx=None,
        )
        assert res["status"] == "submitted"
        # Both pull jobs (4 + 4b) run on background threads that can finish in any
        # order; wait until both have invoked the mock, then assert the custom call
        # was made (order-independent — assert_called_with checks only the LAST
        # call, which flaked when 4a's thread landed after 4b's).
        for _ in range(300):
            if mock_git.pull_projects.call_count >= 2:
                break
            await asyncio.sleep(0.01)
        mock_git.pull_projects.assert_any_call(
            project_dirs=[repo_a_dir, "/absolute/path/repo-b"]
        )

        # 5. Action: push
        res = await rm_git.fn(
            action="push",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            projects=None,
            ctx=None,
        )
        assert res["status"] == "submitted"

        # 5b. Action: push with custom projects
        res = await rm_git.fn(
            action="push",
            command=None,
            path="/tmp",
            threads=None,
            phase=1,
            target_project=None,
            projects="repo-a, /absolute/path/repo-b",
            ctx=None,
        )
        assert res["status"] == "submitted"
        # Both push jobs (5 + 5b) run on background threads; wait for both, then
        # assert the custom call landed (order-independent — see pull above).
        for _ in range(300):
            if mock_git.push_projects.call_count >= 2:
                break
            await asyncio.sleep(0.01)
        mock_git.push_projects.assert_any_call(
            project_dirs=[repo_a_dir, "/absolute/path/repo-b"]
        )

        # 6. Action: phased_push
        res = await rm_git.fn(
            action="phased_push",
            command=None,
            path=None,
            threads=None,
            phase=2,
            target_project="proj-a",
            ctx=None,
        )
        assert res["status"] == "submitted"

        # 7. Action: invalid — the shared resolver raises a rich did-you-mean
        # error pointing callers at 'list_actions'.
        with pytest.raises(ValueError) as exc:
            await rm_git.fn(
                action="invalid_action",
                command=None,
                path=None,
                threads=None,
                phase=1,
                target_project=None,
                ctx=None,
            )
        assert "Unknown action" in str(exc.value)
        assert "list_actions" in str(exc.value)

        # 8. Action: discovery — 'list_actions' returns the available actions.
        disc = await rm_git.fn(
            action="list_actions",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            ctx=None,
        )
        assert isinstance(disc, dict)
        assert "raw" in disc["actions"]


@pytest.mark.anyio
async def test_mcp_rm_workspace_tool():
    """Test all actions and parameter checks of rm_workspace tool."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    rm_workspace = next(t for t in tools if t.name == "rm_workspace")

    mock_git = MagicMock()
    mock_git.get_workspace_projects.return_value = ["proj1", "proj2"]
    mock_git.list_branches.return_value = ["main", "dev"]
    mock_git.setup_from_yaml.return_value = "setup_ok"
    mock_git.generate_workspace_template.return_value = "template_ok"
    mock_git.save_workspace_config.return_value = "save_ok"
    mock_git.remediate_projects.return_value = {"success": 2, "errors": 0}

    with patch("repository_manager.mcp_server.get_git_instance", return_value=mock_git):
        # 1. Action: list
        res = await rm_workspace.fn(
            action="list",
            yml_path=None,
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res == ["proj1", "proj2"]

        # 2. Action: list_branches
        res = await rm_workspace.fn(
            action="list_branches",
            yml_path=None,
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res == ["main", "dev"]

        # 3. Action: setup (missing yml_path)
        res = await rm_workspace.fn(
            action="setup",
            yml_path=None,
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res.status == "error"

        # 4. Action: setup (with yml_path)
        res = await rm_workspace.fn(
            action="setup",
            yml_path="workspace.yml",
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res == "setup_ok"

        # 5. Action: template (missing yml_path)
        res = await rm_workspace.fn(
            action="template",
            yml_path=None,
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res.status == "error"

        # 6. Action: template (with yml_path)
        res = await rm_workspace.fn(
            action="template",
            yml_path="workspace.yml",
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res == "template_ok"

        # 7. Action: save (missing config_dict)
        res = await rm_workspace.fn(
            action="save",
            yml_path="workspace.yml",
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res.status == "error"

        # 8. Action: save (invalid config_dict)
        res = await rm_workspace.fn(
            action="save",
            yml_path="workspace.yml",
            config_dict={"invalid": "key"},
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res.status == "error"

        # 9. Action: save (valid config_dict)
        res = await rm_workspace.fn(
            action="save",
            yml_path="workspace.yml",
            config_dict={"name": "test_workspace", "path": "/path", "repositories": []},
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res == "save_ok"

        # 10. Action: maintain
        res = await rm_workspace.fn(
            action="maintain",
            yml_path=None,
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert res["status"] == "submitted"

        # 12. Action: invalid — rich did-you-mean error from the shared resolver.
        with pytest.raises(ValueError) as exc:
            await rm_workspace.fn(
                action="invalid_action",
                yml_path=None,
                config_dict=None,
                part="patch",
                phase=1,
                dry_run=False,
                use_default=True,
                ctx=None,
            )
        assert "Unknown action" in str(exc.value)
        assert "list_actions" in str(exc.value)


@pytest.mark.anyio
async def test_mcp_rm_projects_tool():
    """Test all actions and parameter checks of rm_projects tool."""
    mcp, _, _, _ = get_mcp_instance()
    tools = await mcp.list_tools()
    rm_projects = next(t for t in tools if t.name == "rm_projects")

    mock_git = MagicMock()
    mock_git.project_map = {
        "https://github.com/org/repo-a.git": "/path/repo-a",
        "https://github.com/org/repo-b.git": "/path/repo-b",
    }
    mock_git.install_projects.return_value = "install_ok"
    mock_git.build_projects.return_value = "build_ok"
    mock_git.validate_projects.return_value = "validate_ok"

    with patch("repository_manager.mcp_server.get_git_instance", return_value=mock_git):
        # 1. Repository filtering logic
        res = await rm_projects.fn(
            action="install",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            force_revalidate=False,
            auto_bump=False,
            auto_push=False,
            bump_part="minor",
            repositories="repo-a",
            job_id=None,
            ctx=None,
        )
        assert res["status"] == "submitted"
        assert len(mock_git.project_map) == 1
        assert "https://github.com/org/repo-a.git" in mock_git.project_map

        # Reset mock project_map
        mock_git.project_map = {
            "https://github.com/org/repo-a.git": "/path/repo-a",
            "https://github.com/org/repo-b.git": "/path/repo-b",
        }

        # 2. Action: build
        res = await rm_projects.fn(
            action="build",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            force_revalidate=False,
            auto_bump=False,
            auto_push=False,
            bump_part="minor",
            repositories=None,
            job_id=None,
            ctx=None,
        )
        assert res["status"] == "submitted"

        # 3. Action: validate
        res = await rm_projects.fn(
            action="validate",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            force_revalidate=False,
            auto_bump=False,
            auto_push=False,
            bump_part="minor",
            repositories=None,
            job_id=None,
            ctx=None,
        )
        # Terse submission by default (counts, not the full id↔name maps).
        assert res["status"] == "submitted"
        assert "queued_count" in res
        assert "queued_projects" in res

        # 4. Action: validate_status
        # Submit a status request
        with _jobs_lock:
            _jobs["job-status-test"] = {
                "status": "completed",
                "action": "validate",
                "result": "passed",
                "started_at": None,
                "completed_at": None,
            }
        res = await rm_projects.fn(
            action="validate_status",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            force_revalidate=False,
            auto_bump=False,
            auto_push=False,
            bump_part="minor",
            repositories=None,
            job_id="job-status-test",
            ctx=None,
        )
        assert res["result"] == "passed"

        # 5. Action: invalid — rich did-you-mean error from the shared resolver.
        with pytest.raises(ValueError) as exc:
            await rm_projects.fn(
                action="invalid_action",
                threads=None,
                extra="all",
                output_dir=None,
                generate_report=True,
                force_revalidate=False,
                auto_bump=False,
                auto_push=False,
                bump_part="minor",
                repositories=None,
                job_id=None,
                ctx=None,
            )
        assert "Unknown action" in str(exc.value)
        assert "list_actions" in str(exc.value)


def test_mcp_server_cli_execution():
    """Verify MCP Server startup command-line arguments routing."""
    # Test mcp_server call for stdio transport
    mock_mcp = MagicMock()
    mock_args = MagicMock()
    mock_args.transport = "stdio"

    with patch(
        "repository_manager.mcp_server.get_mcp_instance",
        return_value=(mock_mcp, mock_args, [], []),
    ):
        mcp_server()
        mock_mcp.run.assert_called_once_with(transport="stdio")

    # Test mcp_server call for streamable-http transport
    mock_mcp = MagicMock()
    mock_args = MagicMock()
    mock_args.transport = "streamable-http"
    mock_args.host = "127.0.0.1"
    mock_args.port = 8000

    with patch(
        "repository_manager.mcp_server.get_mcp_instance",
        return_value=(mock_mcp, mock_args, [], []),
    ):
        mcp_server()
        mock_mcp.run.assert_called_once_with(
            transport="streamable-http", host="127.0.0.1", port=8000
        )

    # Test mcp_server call for sse transport
    mock_mcp = MagicMock()
    mock_args = MagicMock()
    mock_args.transport = "sse"
    mock_args.host = "127.0.0.1"
    mock_args.port = 8000

    with patch(
        "repository_manager.mcp_server.get_mcp_instance",
        return_value=(mock_mcp, mock_args, [], []),
    ):
        mcp_server()
        mock_mcp.run.assert_called_once_with(
            transport="sse", host="127.0.0.1", port=8000
        )

    # Test mcp_server call with invalid transport
    mock_mcp = MagicMock()
    mock_args = MagicMock()
    mock_args.transport = "invalid"

    with patch(
        "repository_manager.mcp_server.get_mcp_instance",
        return_value=(mock_mcp, mock_args, [], []),
    ):
        with pytest.raises(SystemExit):
            mcp_server()


# ---------------------------------------------------------------------------
# Regression tests for the cascade-deadlock root causes (RM-TOPOLOGY / RM-BUMP)
# ---------------------------------------------------------------------------


def _mk_job(status, repo, started_at, *, passed=None, error=None):
    """Build a synthetic _jobs entry for roll-up tests."""
    job = {
        "status": status,
        "action": "validate",
        "repo_name": repo,
        "started_at": started_at,
        "completed_at": None,
        "result": None,
        "error": error,
    }
    if passed is not None:
        res = MagicMock()
        res.success = passed
        res.hooks = []
        res.error = None
        job["result"] = res
        if status == "completed":
            job["completed_at"] = started_at
    return job


def test_rollup_dedupes_stale_failure_to_latest_per_repo():
    """A repo that FAILED early then PASSED on a re-run must not linger as failed.

    Root cause: _get_job_status iterated every historical job_id, so the old
    failed job stayed in failed_projects forever. The fix collapses to the
    latest job per repo.
    """
    from repository_manager.mcp_server import _get_job_status

    with _jobs_lock:
        _jobs.clear()
        # earlier: failed; later: passed (same repo)
        _jobs["old1"] = _mk_job(
            "completed", "repo-a", "2026-06-06T01:00:00+00:00Z", passed=False
        )
        _jobs["new1"] = _mk_job(
            "completed", "repo-a", "2026-06-06T02:00:00+00:00Z", passed=True
        )
        # a genuinely still-failing repo
        _jobs["f2"] = _mk_job(
            "failed", "repo-b", "2026-06-06T01:30:00+00:00Z", error="boom"
        )

    rollup = _get_job_status()
    assert "repo-a" not in rollup["failed_projects"], rollup["failed_projects"]
    assert "repo-b" in rollup["failed_projects"]
    assert rollup["summary"]["total"] == 2  # deduped: repo-a (latest) + repo-b
    assert rollup["summary"]["passed"] == 1
    assert rollup["summary"]["failed"] == 1

    with _jobs_lock:
        _jobs.clear()


def test_reap_stale_jobs_flips_overdue_running_to_failed():
    """A wedged 'running' job past the ceiling is reaped so status never freezes."""
    from repository_manager.mcp_server import _reap_stale_jobs

    with _jobs_lock:
        _jobs.clear()
        _jobs["wedged"] = _mk_job("running", "repo-z", "2020-01-01T00:00:00+00:00Z")
        _jobs["fresh"] = _mk_job("running", "repo-y", "2099-01-01T00:00:00+00:00Z")

    _reap_stale_jobs(max_age_seconds=60)

    with _jobs_lock:
        assert _jobs["wedged"]["status"] == "failed"
        assert "reaped" in (_jobs["wedged"]["error"] or "")
        assert _jobs["fresh"]["status"] == "running"  # future ts, not reaped
        _jobs.clear()


def test_bump_skip_reason_avoids_double_bump_on_unpushed_repo():
    """Clean tree + ahead-of-origin + HEAD is a bump commit => skip (no re-bump)."""
    from repository_manager.repository_manager import Git

    git = Git.__new__(Git)  # bypass __init__; we only exercise _bump_skip_reason

    def fake_git_action(command, path=None, **kw):
        r = MagicMock()
        if command == "git status":
            r.data = (
                "On branch main\nYour branch is ahead of 'origin/main' by 1 commit.\n"
                "nothing to commit, working tree clean"
            )
        elif command.startswith("git log"):
            r.data = "Bump version: 0.38.0 -> 0.39.0"
        else:
            r.data = ""
        return r

    git.git_action = fake_git_action  # type: ignore[method-assign,assignment]
    reason = git._bump_skip_reason("/fake/repo")
    assert reason and "awaiting push" in reason


def test_bump_skip_reason_allows_bump_for_unbumped_feature_commit():
    """Clean tree + ahead, but HEAD is a FEATURE commit => must NOT skip (needs bump)."""
    from repository_manager.repository_manager import Git

    git = Git.__new__(Git)

    def fake_git_action(command, path=None, **kw):
        r = MagicMock()
        if command == "git status":
            r.data = (
                "On branch main\nYour branch is ahead of 'origin/main' by 2 commits.\n"
                "nothing to commit, working tree clean"
            )
        elif command.startswith("git log"):
            r.data = "feat: add new endpoint"
        else:
            r.data = ""
        return r

    git.git_action = fake_git_action  # type: ignore[method-assign,assignment]
    assert git._bump_skip_reason("/fake/repo") is None


def test_bump_skip_reason_skips_when_clean_and_in_sync():
    """Clean tree + in sync with origin => skip (no changes to bump)."""
    from repository_manager.repository_manager import Git

    git = Git.__new__(Git)

    def fake_git_action(command, path=None, **kw):
        r = MagicMock()
        r.data = (
            "On branch main\nYour branch is up to date with 'origin/main'.\n"
            "nothing to commit, working tree clean"
        )
        return r

    git.git_action = fake_git_action  # type: ignore[method-assign,assignment]
    reason = git._bump_skip_reason("/fake/repo")
    assert reason and "no code changes" in reason


def test_resolve_repo_dir_honors_nested_workspace_layout(tmp_path):
    """A bare repo name must resolve to its nested project_map path, not a flat join.

    Regression: rm_git pull/push/add/commit flat-joined ``git.path + name`` and
    so failed with ENOENT on every nested repo (e.g. push agent-utilities ->
    <ws>/agent-utilities) while validate (which uses project_map) succeeded.
    """
    from repository_manager.mcp_server import _resolve_repo_dir
    from repository_manager.repository_manager import Git

    ws = tmp_path
    nested = ws / "agent-packages" / "agent-utilities"
    nested.mkdir(parents=True)
    deep = ws / "agent-packages" / "agents" / "data-science-mcp"
    deep.mkdir(parents=True)
    git = Git.__new__(Git)  # bypass __init__; _resolve_repo_dir only reads two attrs
    git.path = str(ws)
    git.project_map = {
        "https://x/agent-utilities.git": str(nested),
        "https://x/data-science-mcp.git": str(deep),
    }

    # bare names -> nested paths (the fix)
    assert _resolve_repo_dir(git, "agent-utilities") == str(nested)
    assert _resolve_repo_dir(git, "data-science-mcp") == str(deep)
    # absolute spec -> verbatim
    assert _resolve_repo_dir(git, "/tmp/abs") == "/tmp/abs"
    # an existing flat relative dir -> used as-is (back-compat)
    (ws / "flatrepo").mkdir()
    assert _resolve_repo_dir(git, "flatrepo") == str(ws / "flatrepo")
    # unknown name -> flat-join fallback (prior error surface preserved)
    assert _resolve_repo_dir(git, "ghost") == str(ws / "ghost")
