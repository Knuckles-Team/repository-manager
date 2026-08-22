"""Tests for repository_manager.forge_status (CONCEPT:RM-DEP-READY ci-run-barrier).

No network: every backend is exercised via an injected fake client. Coverage:
backend selection by remote host, both real backends (GitHub Actions / GitLab
pipelines) mapping their forge's own status vocabulary onto RunStatus, and --
importantly -- the degrade-to-'unknown' path when the optional forge client
package is not importable, proving that path never raises.
"""

from __future__ import annotations

from repository_manager import forge_status as fs

# --------------------------------------------------------------------------- #
# backend_for_remote -- selection by host
# --------------------------------------------------------------------------- #


def test_backend_for_remote_selects_github_for_github_com():
    backend = fs.backend_for_remote("https://github.com/knuckles-team/agent-utilities.git")
    assert isinstance(backend, fs.GitHubActionsBackend)


def test_backend_for_remote_selects_github_for_ssh_shorthand():
    backend = fs.backend_for_remote("git@github.com:knuckles-team/agent-utilities.git")
    assert isinstance(backend, fs.GitHubActionsBackend)


def test_backend_for_remote_selects_gitlab_for_internal_host():
    backend = fs.backend_for_remote("https://gitlab.example.internal/org/repository-manager.git")
    assert isinstance(backend, fs.GitLabPipelineBackend)


def test_backend_for_remote_unparseable_url_degrades_to_unknown_backend():
    backend = fs.backend_for_remote("")
    assert isinstance(backend, fs.UnknownForgeBackend)
    status = backend.latest_run_for_ref("owner", "repo", "v1.0.0")
    assert status.state == "unknown"


def test_owner_repo_from_remote_https():
    assert fs.owner_repo_from_remote(
        "https://github.com/knuckles-team/agent-utilities.git"
    ) == ("knuckles-team", "agent-utilities")


def test_owner_repo_from_remote_ssh_shorthand():
    assert fs.owner_repo_from_remote(
        "git@github.com:knuckles-team/agent-utilities.git"
    ) == ("knuckles-team", "agent-utilities")


def test_owner_repo_from_remote_nested_gitlab_group():
    assert fs.owner_repo_from_remote(
        "https://gitlab.example.internal/group/subgroup/repository-manager.git"
    ) == ("group/subgroup", "repository-manager")


def test_owner_repo_from_remote_too_few_segments_is_none():
    assert fs.owner_repo_from_remote("https://github.com/") is None


# --------------------------------------------------------------------------- #
# GitHubActionsBackend -- over a faked github_agent client
# --------------------------------------------------------------------------- #


class _FakeGitHubResponse:
    def __init__(self, data):
        self.data = data


class _FakeGitHubRun:
    def __init__(self, *, status, conclusion=None, html_url=None, run_started_at=None):
        self.status = status
        self.conclusion = conclusion
        self.html_url = html_url
        self.run_started_at = run_started_at


class _FakeGitHubClient:
    def __init__(self, runs=None, *, raise_exc=None):
        self._runs = runs or []
        self._raise_exc = raise_exc
        self.calls = []

    def get_workflow_runs(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeGitHubResponse(self._runs)


def test_github_backend_completed_success():
    client = _FakeGitHubClient(
        [
            _FakeGitHubRun(
                status="completed",
                conclusion="success",
                html_url="https://github.com/o/r/actions/runs/1",
                run_started_at="2026-08-21T00:00:00Z",
            )
        ]
    )
    backend = fs.GitHubActionsBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "completed"
    assert status.conclusion == "success"
    assert status.url == "https://github.com/o/r/actions/runs/1"
    assert status.started_at == "2026-08-21T00:00:00Z"
    assert client.calls == [{"owner": "o", "repo": "r", "branch": "v1.0.0"}]


def test_github_backend_completed_failure():
    client = _FakeGitHubClient([_FakeGitHubRun(status="completed", conclusion="failure")])
    backend = fs.GitHubActionsBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "completed"
    assert status.conclusion == "failure"


def test_github_backend_in_progress():
    client = _FakeGitHubClient([_FakeGitHubRun(status="in_progress")])
    backend = fs.GitHubActionsBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "in_progress"
    assert status.conclusion is None


def test_github_backend_no_runs_is_unknown():
    client = _FakeGitHubClient([])
    backend = fs.GitHubActionsBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"


def test_github_backend_client_exception_degrades_to_unknown_never_raises():
    client = _FakeGitHubClient(raise_exc=RuntimeError("network blew up"))
    backend = fs.GitHubActionsBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"
    assert status.conclusion is None


def test_github_backend_unrecognized_status_is_unknown():
    client = _FakeGitHubClient([_FakeGitHubRun(status="some-future-github-status")])
    backend = fs.GitHubActionsBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"


# --------------------------------------------------------------------------- #
# GitLabPipelineBackend -- over a faked gitlab_api client
# --------------------------------------------------------------------------- #


class _FakeGitLabResponse:
    def __init__(self, data):
        self.data = data


class _FakeGitLabPipeline:
    def __init__(self, *, status, web_url=None, created_at=None, started_at=None):
        self.status = status
        self.web_url = web_url
        self.created_at = created_at
        self.started_at = started_at


class _FakeGitLabClient:
    def __init__(self, pipelines=None, *, raise_exc=None):
        self._pipelines = pipelines or []
        self._raise_exc = raise_exc
        self.calls = []

    def get_pipelines(self, **kwargs):
        self.calls.append(kwargs)
        if self._raise_exc is not None:
            raise self._raise_exc
        return _FakeGitLabResponse(self._pipelines)


def test_gitlab_backend_success_pipeline_is_completed():
    client = _FakeGitLabClient(
        [
            _FakeGitLabPipeline(
                status="success",
                web_url="https://gitlab.example.internal/o/r/-/pipelines/1",
                created_at="2026-08-21T00:00:00Z",
            )
        ]
    )
    backend = fs.GitLabPipelineBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "completed"
    assert status.conclusion == "success"
    assert status.url == "https://gitlab.example.internal/o/r/-/pipelines/1"
    assert client.calls == [{"project_id": "o%2Fr", "ref": "v1.0.0"}]


def test_gitlab_backend_failed_pipeline_is_completed_with_failed_conclusion():
    client = _FakeGitLabClient([_FakeGitLabPipeline(status="failed")])
    backend = fs.GitLabPipelineBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "completed"
    assert status.conclusion == "failed"


def test_gitlab_backend_running_pipeline_is_in_progress():
    client = _FakeGitLabClient([_FakeGitLabPipeline(status="running")])
    backend = fs.GitLabPipelineBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "in_progress"
    assert status.conclusion is None


def test_gitlab_backend_pending_pipeline_is_queued():
    client = _FakeGitLabClient([_FakeGitLabPipeline(status="pending")])
    backend = fs.GitLabPipelineBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "queued"


def test_gitlab_backend_nested_group_encodes_project_id():
    client = _FakeGitLabClient([_FakeGitLabPipeline(status="success")])
    backend = fs.GitLabPipelineBackend(client=client)
    backend.latest_run_for_ref("group/subgroup", "repo", "v1.0.0")
    assert client.calls == [{"project_id": "group%2Fsubgroup%2Frepo", "ref": "v1.0.0"}]


def test_gitlab_backend_no_pipelines_is_unknown():
    client = _FakeGitLabClient([])
    backend = fs.GitLabPipelineBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"


def test_gitlab_backend_client_exception_degrades_to_unknown_never_raises():
    client = _FakeGitLabClient(raise_exc=RuntimeError("network blew up"))
    backend = fs.GitLabPipelineBackend(client=client)
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"


def test_gitlab_backend_no_client_and_no_url_degrades_to_unknown():
    """No injected client, and the optional package IS importable (this
    process has it), but with no `url` configured GitLabPipelineBackend must
    still degrade rather than raise MissingParameterError constructing the
    real client."""
    backend = fs.GitLabPipelineBackend()
    status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"


# --------------------------------------------------------------------------- #
# Degradation when the optional forge client package is not importable --
# never a silent skip (a warning is logged) and never a hard failure.
# --------------------------------------------------------------------------- #


def test_github_backend_degrades_when_client_import_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(fs, "_GitHubWorkflowsApi", None)
    monkeypatch.setattr(fs, "_GITHUB_AGENT_UNAVAILABLE", "No module named 'github_agent'")
    with caplog.at_level("WARNING", logger="repository_manager.forge_status"):
        backend = fs.GitHubActionsBackend()
        status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"
    assert status.conclusion is None
    assert status.url is None
    assert any("FORGE_STATUS_UNAVAILABLE" in rec.message for rec in caplog.records)


def test_gitlab_backend_degrades_when_client_import_unavailable(monkeypatch, caplog):
    monkeypatch.setattr(fs, "_GitLabPipelinesApi", None)
    monkeypatch.setattr(fs, "_GITLAB_API_UNAVAILABLE", "No module named 'gitlab_api'")
    with caplog.at_level("WARNING", logger="repository_manager.forge_status"):
        backend = fs.GitLabPipelineBackend(url="https://gitlab.example.internal")
        status = backend.latest_run_for_ref("o", "r", "v1.0.0")
    assert status.state == "unknown"
    assert any("FORGE_STATUS_UNAVAILABLE" in rec.message for rec in caplog.records)


def test_backend_for_remote_degrades_end_to_end_when_github_agent_unavailable(
    monkeypatch, caplog
):
    """The whole selection-to-degradation path, exactly as
    dependency_readiness.await_gate_readiness would exercise it when the
    optional client package is missing: never raises, always 'unknown'."""
    monkeypatch.setattr(fs, "_GitHubWorkflowsApi", None)
    monkeypatch.setattr(fs, "_GITHUB_AGENT_UNAVAILABLE", "No module named 'github_agent'")
    backend = fs.backend_for_remote("https://github.com/knuckles-team/agent-utilities.git")
    with caplog.at_level("WARNING", logger="repository_manager.forge_status"):
        status = backend.latest_run_for_ref("knuckles-team", "agent-utilities", "v1.0.0")
    assert status.state == "unknown"
    assert any("FORGE_STATUS_UNAVAILABLE" in rec.message for rec in caplog.records)
