"""Characterization tests for CXA-FL-REPOSITORYMANAGER-02 (Phase A).

Pins current behavior of the six worst-CCN functions in this lane's
partition, BEFORE any extract-method refactor, so the refactor commit can be
verified byte-identical in test content and identical in pass/fail outcome.

Covers branches not already exercised by the existing suite
(``tests/test_directory_report.py``, ``tests/test_mcp_handlers.py``,
``tests/test_workspace_versions.py``, ``tests/test_workspace_release_plan.py``,
``tests/test_landing_policy.py``), which remain the primary characterization
baseline per the lane brief's "prefer existing tests" guidance.
"""

from __future__ import annotations

import importlib
from unittest.mock import MagicMock, patch

import pytest

from repository_manager.models import GitMetadata, GitResult, ValidationReport

mcp_server_module = importlib.import_module("repository_manager.mcp_server")


# ---------------------------------------------------------------------------
# ValidationReport.from_results — uncategorized fallback + metadata=None
# ---------------------------------------------------------------------------


class TestFromResultsUncategorized:
    def _ts(self) -> str:
        return "2026-08-27T00:00:00Z"

    def test_uncategorized_result_lands_in_additional_operational_checks(self):
        """A result whose command matches none of the known category
        substrings is swept into the id()-deduped 'Additional Operational
        Checks' fallback category — not silently dropped."""
        result = GitResult(
            status="success",
            data="did a thing",
            metadata=GitMetadata(
                command="some_unrelated_tool --flag",
                workspace="/workspace/agents/zzz-agent",
                return_code=0,
                timestamp=self._ts(),
            ),
        )
        report = ValidationReport.from_results([result])
        names = [c.name for c in report.categories]
        assert names == ["Additional Operational Checks"]
        assert report.categories[0].total == 1
        assert report.categories[0].successes[0].project == "zzz-agent"

    def test_metadata_none_result_is_uncategorized_with_unknown_project(self):
        """A GitResult with metadata=None can never match any category filter
        (all filters require ``r.metadata and r.metadata.command``), so it
        also falls into the uncategorized bucket, with project label
        'unknown' (the ``if r.metadata else "unknown"`` fallback)."""
        result = GitResult(status="error", data="boom", metadata=None)
        report = ValidationReport.from_results([result])
        names = [c.name for c in report.categories]
        assert names == ["Additional Operational Checks"]
        assert report.categories[0].failures[0].project == "unknown"
        # error message falls back to r.data since r.error is None
        assert report.categories[0].failures[0].message == "boom"

    def test_no_uncategorized_results_means_no_fallback_category(self):
        """When every result matches a known category, 'Additional
        Operational Checks' must not appear at all (categories_map only
        gains that key when ``uncategorized`` is truthy)."""
        result = GitResult(
            status="success",
            data="ok",
            metadata=GitMetadata(
                command="pip install -e '.[all]'",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=self._ts(),
            ),
        )
        report = ValidationReport.from_results([result])
        names = [c.name for c in report.categories]
        assert "Additional Operational Checks" not in names
        assert names == ["Ecosystem Installation"]

    def test_category_ordering_follows_categories_map_declaration_order(self):
        """Categories in the report follow the fixed declaration order of the
        internal categories_map (Installation, ..., Pytest Suite as declared,
        with 'Additional Operational Checks' appended LAST since it is only
        inserted into the dict after the declared literal is built) — not
        result-arrival order."""
        ts = self._ts()
        pytest_result = GitResult(
            status="success",
            data="ok",
            metadata=GitMetadata(
                command="pytest tests/",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=ts,
            ),
        )
        install_result = GitResult(
            status="success",
            data="ok",
            metadata=GitMetadata(
                command="pip install -e '.[all]'",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=ts,
            ),
        )
        uncategorized_result = GitResult(
            status="success",
            data="ok",
            metadata=GitMetadata(
                command="totally_unknown_tool",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=ts,
            ),
        )
        # Deliberately supplied out of category-declaration order.
        report = ValidationReport.from_results(
            [pytest_result, uncategorized_result, install_result]
        )
        names = [c.name for c in report.categories]
        assert names == [
            "Ecosystem Installation",
            "Pytest Suite",
            "Additional Operational Checks",
        ]


# ---------------------------------------------------------------------------
# register_project_tools.rm_projects — chaining/wiring branches not covered
# by tests/test_mcp_handlers.py::test_mcp_rm_projects_tool
# ---------------------------------------------------------------------------


async def _get_rm_projects_tool():
    mcp, _, _, _ = mcp_server_module.get_mcp_instance()
    tools = await mcp.list_tools()
    return next(t for t in tools if t.name == "rm_projects")


def _mock_git(project_map=None):
    git = MagicMock()
    git.project_map = project_map or {
        "https://github.com/org/repo-a.git": "/path/repo-a",
        "https://github.com/org/repo-b.git": "/path/repo-b",
    }
    return git


@pytest.mark.anyio
async def test_rm_projects_validate_wires_auto_bump_and_auto_push_and_hygiene():
    """auto_bump=True, auto_push=True: the bump job must depend on the
    validation job ids; the push job must depend on the bump job (NOT on the
    raw validation ids); worktree_hygiene must depend on the push job (the
    last stage), and prune defaults to False."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git()

    calls = []

    def fake_submit_job(action, func, *args, _extra_job_data=None, **kwargs):
        calls.append({"action": action, "args": args, "kwargs": kwargs})
        return {"status": "submitted", "job_id": f"job-{len(calls)}"}

    with (
        patch.object(mcp_server_module, "get_git_instance", return_value=git),
        patch.object(mcp_server_module, "_submit_job", side_effect=fake_submit_job),
    ):
        res = await rm_projects.fn(
            action="validate",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            repositories=None,
            auto_bump=True,
            auto_push=True,
            bump_part="minor",
            prune_worktrees=False,
            commit_code=False,
            commit_message=None,
            force_revalidate=False,
            failed_only=False,
            summary=True,
            job_id=None,
            ctx=None,
        )

    actions = [c["action"] for c in calls]
    # 2 validate jobs (repo-a, repo-b) + maintain + phased_push + worktree_hygiene
    assert actions.count("validate") == 2
    assert actions.count("maintain") == 1
    assert actions.count("phased_push") == 1
    assert actions.count("worktree_hygiene") == 1

    validate_ids = [
        f"job-{i + 1}" for i, c in enumerate(calls) if c["action"] == "validate"
    ]
    maintain_call = next(c for c in calls if c["action"] == "maintain")
    push_call = next(c for c in calls if c["action"] == "phased_push")
    hygiene_call = next(c for c in calls if c["action"] == "worktree_hygiene")

    # wait_for_jobs_and_run(dependency_job_ids, require_success, func, ...)
    assert maintain_call["args"][0] == validate_ids
    assert maintain_call["args"][1] is True
    assert maintain_call["kwargs"]["part"] == "minor"
    assert maintain_call["kwargs"]["dry_run"] is False

    bump_job_id = res["bump_job_id"]
    assert push_call["args"][0] == [bump_job_id]

    push_job_id = res["push_job_id"]
    assert hygiene_call["args"][0] == [push_job_id]
    assert hygiene_call["args"][1] is False
    assert hygiene_call["kwargs"]["prune"] is False

    assert res["status"] == "submitted"


@pytest.mark.anyio
async def test_rm_projects_validate_commit_code_reroutes_bump_dependency():
    """commit_code=True: the commit job depends on the validation ids; when
    auto_bump is also set, the bump job depends ONLY on the commit job's id
    (not on the raw validation ids) — bump_dependencies is reassigned."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git()

    calls = []

    def fake_submit_job(action, func, *args, _extra_job_data=None, **kwargs):
        calls.append({"action": action, "args": args, "kwargs": kwargs})
        return {"status": "submitted", "job_id": f"job-{len(calls)}"}

    with (
        patch.object(mcp_server_module, "get_git_instance", return_value=git),
        patch.object(mcp_server_module, "_submit_job", side_effect=fake_submit_job),
    ):
        res = await rm_projects.fn(
            action="validate",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            repositories=None,
            auto_bump=True,
            auto_push=False,
            bump_part="patch",
            prune_worktrees=False,
            commit_code=True,
            commit_message="chore: test commit",
            force_revalidate=False,
            failed_only=False,
            summary=True,
            job_id=None,
            ctx=None,
        )

    validate_ids = [
        f"job-{i + 1}" for i, c in enumerate(calls) if c["action"] == "validate"
    ]
    commit_call = next(c for c in calls if c["action"] == "commit_code")
    maintain_call = next(c for c in calls if c["action"] == "maintain")

    assert commit_call["args"][0] == validate_ids
    assert commit_call["kwargs"]["message"] == "chore: test commit"

    commit_job_id = res["commit_job_id"]
    assert maintain_call["args"][0] == [commit_job_id]
    assert res["bump_job_id"]
    assert "push_job_id" not in res
    # worktree_hygiene not triggered unless auto_bump or auto_push -> it IS
    # triggered here since auto_bump=True, and depends on the bump job.
    hygiene_call = next(c for c in calls if c["action"] == "worktree_hygiene")
    assert hygiene_call["args"][0] == [res["bump_job_id"]]


@pytest.mark.anyio
async def test_rm_projects_validate_running_job_is_not_resubmitted():
    """A repo with an in-flight validate job (status running/queued/pending)
    is reported under 'running' and NOT resubmitted as a new job."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git({"https://github.com/org/repo-a.git": "/path/repo-a"})

    with mcp_server_module._jobs_lock:
        mcp_server_module._jobs["existing-running"] = {
            "status": "running",
            "action": "validate",
            "repo_name": "repo-a",
            "result": None,
        }
    try:
        with patch.object(mcp_server_module, "get_git_instance", return_value=git):
            res = await rm_projects.fn(
                action="validate",
                threads=None,
                extra="all",
                output_dir=None,
                generate_report=True,
                repositories=None,
                auto_bump=False,
                auto_push=False,
                bump_part="minor",
                prune_worktrees=False,
                commit_code=False,
                commit_message=None,
                force_revalidate=False,
                failed_only=False,
                summary=False,
                job_id=None,
                ctx=None,
            )
        assert res["running"] == {"repo-a": "existing-running"}
        assert res["queued"] == {}
    finally:
        with mcp_server_module._jobs_lock:
            mcp_server_module._jobs.pop("existing-running", None)


@pytest.mark.anyio
async def test_rm_projects_validate_completed_cache_hit_without_force_revalidate():
    """A repo with a completed validate job is served from cache (not
    resubmitted) when force_revalidate=False, with a derived cache_summary
    from the mocked result's .success/.hooks attributes."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git({"https://github.com/org/repo-a.git": "/path/repo-a"})

    fake_hook = MagicMock()
    fake_hook.passed = False
    fake_hook.hook_id = "ruff"
    fake_hook.output = "line too long"
    fake_result = MagicMock()
    fake_result.success = False
    fake_result.hooks = [fake_hook]
    fake_result.error = None

    with mcp_server_module._jobs_lock:
        mcp_server_module._jobs["existing-completed"] = {
            "status": "completed",
            "action": "validate",
            "repo_name": "repo-a",
            "result": fake_result,
        }
    try:
        with patch.object(mcp_server_module, "get_git_instance", return_value=git):
            res = await rm_projects.fn(
                action="validate",
                threads=None,
                extra="all",
                output_dir=None,
                generate_report=True,
                repositories=None,
                auto_bump=False,
                auto_push=False,
                bump_part="minor",
                prune_worktrees=False,
                commit_code=False,
                commit_message=None,
                force_revalidate=False,
                failed_only=False,
                summary=False,
                job_id=None,
                ctx=None,
            )
        assert res["completed"]["repo-a"]["job_id"] == "existing-completed"
        assert res["completed"]["repo-a"]["summary"]["passed"] is False
        assert res["completed"]["repo-a"]["summary"]["failures"] == [
            "Hook 'ruff' failed: line too long"
        ]
        assert res["queued"] == {}
    finally:
        with mcp_server_module._jobs_lock:
            mcp_server_module._jobs.pop("existing-completed", None)


@pytest.mark.anyio
async def test_rm_projects_validate_force_revalidate_bypasses_completed_cache():
    """force_revalidate=True re-submits a repo even though its most recent
    validate job already completed."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git({"https://github.com/org/repo-a.git": "/path/repo-a"})

    with mcp_server_module._jobs_lock:
        mcp_server_module._jobs["existing-completed-2"] = {
            "status": "completed",
            "action": "validate",
            "repo_name": "repo-a",
            "result": None,
        }
    try:
        with patch.object(mcp_server_module, "get_git_instance", return_value=git):
            res = await rm_projects.fn(
                action="validate",
                threads=None,
                extra="all",
                output_dir=None,
                generate_report=True,
                repositories=None,
                auto_bump=False,
                auto_push=False,
                bump_part="minor",
                prune_worktrees=False,
                commit_code=False,
                commit_message=None,
                force_revalidate=True,
                failed_only=False,
                summary=False,
                job_id=None,
                ctx=None,
            )
        assert "repo-a" in res["queued"]
        assert res["completed"] == {}
    finally:
        with mcp_server_module._jobs_lock:
            mcp_server_module._jobs.pop("existing-completed-2", None)


@pytest.mark.anyio
async def test_rm_projects_validate_failed_only_with_no_failures_is_clean_noop():
    """failed_only=True with nothing in the remediation set short-circuits to
    a 'clean' status and submits nothing, without even reaching the job scan."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git()

    with (
        patch.object(mcp_server_module, "get_git_instance", return_value=git),
        patch.object(mcp_server_module, "_last_failed_repos", return_value=[]),
    ):
        res = await rm_projects.fn(
            action="validate",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            repositories=None,
            auto_bump=False,
            auto_push=False,
            bump_part="minor",
            prune_worktrees=False,
            commit_code=False,
            commit_message=None,
            force_revalidate=False,
            failed_only=True,
            summary=True,
            job_id=None,
            ctx=None,
        )
    assert res == {
        "status": "clean",
        "message": "No previously-failed projects to re-validate.",
        "queued_count": 0,
    }


@pytest.mark.anyio
async def test_rm_projects_validate_failed_only_targets_failed_repos_and_forces():
    """failed_only=True with a non-empty failed set narrows 'repositories' to
    exactly those repos and forces force_revalidate=True, even though the
    caller passed force_revalidate=False."""
    rm_projects = await _get_rm_projects_tool()
    git = _mock_git()

    calls = []

    def fake_submit_job(action, func, *args, _extra_job_data=None, **kwargs):
        calls.append({"action": action, "extra_job_data": _extra_job_data})
        return {"status": "submitted", "job_id": f"job-{len(calls)}"}

    with (
        patch.object(mcp_server_module, "get_git_instance", return_value=git),
        patch.object(
            mcp_server_module, "_last_failed_repos", return_value=["repo-b"]
        ),
        patch.object(mcp_server_module, "_submit_job", side_effect=fake_submit_job),
    ):
        res = await rm_projects.fn(
            action="validate",
            threads=None,
            extra="all",
            output_dir=None,
            generate_report=True,
            repositories=None,
            auto_bump=False,
            auto_push=False,
            bump_part="minor",
            prune_worktrees=False,
            commit_code=False,
            commit_message=None,
            force_revalidate=False,
            failed_only=True,
            summary=True,
            job_id=None,
            ctx=None,
        )
    assert res["queued_projects"] == ["repo-b"]
    validate_calls = [c for c in calls if c["action"] == "validate"]
    assert len(validate_calls) == 1
    assert validate_calls[0]["extra_job_data"] == {"repo_name": "repo-b"}
