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
async def test_mcp_rm_git_tool():
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
    mock_git.path = "/tmp"

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
        # Wait for thread to process
        await asyncio.sleep(0.1)
        mock_git.clone_projects.assert_called_with(
            projects=[
                "https://github.com/org/repo-a.git",
                "https://github.com/org/repo-b.git",
            ]
        )

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
        # Wait for thread to process
        await asyncio.sleep(0.1)
        mock_git.pull_projects.assert_called_with(
            project_dirs=["/tmp/repo-a", "/absolute/path/repo-b"]
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
        # Wait for thread to process
        await asyncio.sleep(0.1)
        mock_git.push_projects.assert_called_with(
            project_dirs=["/tmp/repo-a", "/absolute/path/repo-b"]
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

        # 7. Action: invalid
        res = await rm_git.fn(
            action="invalid_action",
            command=None,
            path=None,
            threads=None,
            phase=1,
            target_project=None,
            ctx=None,
        )
        assert "Unknown action" in res


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

        # 12. Action: invalid
        res = await rm_workspace.fn(
            action="invalid_action",
            yml_path=None,
            config_dict=None,
            part="patch",
            phase=1,
            dry_run=False,
            use_default=True,
            ctx=None,
        )
        assert "Unknown action" in res


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

        # 5. Action: invalid
        res = await rm_projects.fn(
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
        assert "Unknown action" in res


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
