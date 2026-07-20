import datetime
import os
from unittest.mock import patch

import pytest
import yaml  # type: ignore[import-untyped]

from repository_manager.models import (
    GitMetadata,
    GitResult,
    WorkspaceConfig,
)
from repository_manager.repository_manager import Git


def get_mock_metadata(command="test"):
    return GitMetadata(
        command=command,
        workspace="/tmp",
        return_code=0,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
    )


ACTUAL_WORKSPACE_YML = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "repository_manager", "workspace.yml"
)


@pytest.fixture
def real_workspace_data():
    """Returns the parsed content of the actual workspace.yml."""
    with open(ACTUAL_WORKSPACE_YML) as f:
        return yaml.safe_load(f)


@pytest.fixture
def sample_workspace_yml(tmp_path, real_workspace_data):
    workspace_dir = tmp_path / "my_workspace"
    workspace_dir.mkdir()

    config = real_workspace_data.copy()
    config["path"] = str(workspace_dir)

    yml_path = tmp_path / "workspace.yml"
    with open(yml_path, "w") as f:
        yaml.dump(config, f)

    return yml_path, workspace_dir


def test_actual_workspace_config_parsing(real_workspace_data):
    """Validates that the actual production workspace.yml is valid."""
    config = WorkspaceConfig(**real_workspace_data)
    assert config.name == "Agent Packages Workspace"
    assert "agent-packages" in config.subdirectories
    assert config.maintenance is not None
    assert len(config.maintenance.phases) > 0

    phase3 = next((p for p in config.maintenance.phases if p.phase == 3), None)
    assert phase3 is not None
    assert phase3.bulk_bump is False
    assert phase3.bulk_push is False


def test_workspace_config_parsing(sample_workspace_yml):
    yml_path, _ = sample_workspace_yml
    with open(yml_path) as f:
        data = yaml.safe_load(f)

    config = WorkspaceConfig(**data)
    assert config.name == "Agent Packages Workspace"

    assert "agent-packages" in config.subdirectories
    assert config.maintenance is not None
    assert config.maintenance.phases[0].name == "Phase 1: GitHub Pipelines"


@patch("repository_manager.repository_manager.Git.git_action")
@patch("repository_manager.repository_manager.Git.pull_project")
def test_setup_from_yaml(mock_pull, mock_git, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml

    mock_git.return_value = GitResult(
        status="success", data="Cloned", metadata=get_mock_metadata("clone")
    )
    mock_pull.return_value = GitResult(
        status="success", data="Pulled", metadata=get_mock_metadata("pull")
    )

    wm = Git(path=str(workspace_dir))
    result = wm.setup_from_yaml(str(yml_path))

    assert result.status == "success"

    assert mock_git.call_count > 0

    (workspace_dir / "agent-packages" / "agent-utilities").mkdir(parents=True)

    result = wm.setup_from_yaml(str(yml_path))

    assert mock_pull.call_count > 0


def test_maintenance_config_loading(sample_workspace_yml):
    yml_path, _ = sample_workspace_yml
    git = Git()

    os.environ["WORKSPACE_YML"] = str(yml_path)

    with patch.object(Git, "pre_commit_projects") as mock_pre:
        with patch.object(Git, "bump_version") as mock_bump:
            mock_pre.return_value = [
                GitResult(status="success", data="OK", metadata=get_mock_metadata())
            ]
            mock_bump.return_value = get_mock_metadata("bump")

            # auto_start=False keeps the high start_phase a clean no-op; this
            # test exercises config loading, not change-aware start detection.
            git.maintain_projects(dry_run=True, start_phase=100, auto_start=False)

            assert hasattr(git, "config")
            assert git.config is not None
            assert git.config.name == "Agent Packages Workspace"
            assert git.config.maintenance is not None
            assert git.config.maintenance.phases[0].phase == 1


def test_git_init_with_default_workspace():

    git = Git()
    assert git.path is not None

    from repository_manager.repository_manager import (
        DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
    )

    assert git.path == DEFAULT_REPOSITORY_MANAGER_WORKSPACE


@patch("repository_manager.repository_manager.Git.git_action")
def test_list_branches(mock_git_action, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))

    mock_git_action.return_value = GitResult(
        status="success", data="main\n", metadata=get_mock_metadata("rev-parse")
    )

    # Without project map
    assert git.list_branches() == {}

    # With project map
    git.load_projects_from_yaml(str(yml_path))

    with patch("os.path.exists", return_value=True):
        branches = git.list_branches()

    # Check that known projects are in the branch output
    assert "pipelines" in branches
    assert branches["pipelines"] == "main"


def test_discover_projects(tmp_path):
    # Set up some mock git repos
    repo1 = tmp_path / "repo1"
    repo1.mkdir()
    (repo1 / ".git").mkdir()

    repo2 = tmp_path / "repo2"
    repo2.mkdir()
    (repo2 / ".git").mkdir()

    # repo3 does not have .git, should not be discovered
    repo3 = tmp_path / "repo3"
    repo3.mkdir()

    git = Git(path=str(tmp_path))

    # Mock subprocess.run to return a mock remote URL
    with patch("subprocess.run") as mock_run:
        import subprocess

        def mock_git_config(args, cwd, **kwargs):
            if "repo1" in cwd:
                url = "https://github.com/user/repo1.git"
            elif "repo2" in cwd:
                url = "https://github.com/user/repo2.git"
            else:
                url = ""
            return subprocess.CompletedProcess(
                args=args, returncode=0, stdout=f"{url}\n"
            )

        mock_run.side_effect = mock_git_config

        projects = git.discover_projects()

        assert "https://github.com/user/repo1.git" in projects
        assert "https://github.com/user/repo2.git" in projects
        assert projects["https://github.com/user/repo1.git"] == str(repo1.resolve())
        assert projects["https://github.com/user/repo2.git"] == str(repo2.resolve())
        assert len(projects) == 2


@patch("repository_manager.repository_manager.Git.git_action")
def test_git_add_operations(mock_git_action, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))

    mock_git_action.return_value = GitResult(
        status="success", data="Staged", metadata=get_mock_metadata("git add -A")
    )

    # Test single add
    res = git.add_project(str(workspace_dir / "pipelines"))
    assert res.status == "success"
    mock_git_action.assert_called_with(
        command="git add -A", path=str(workspace_dir / "pipelines")
    )

    # Test bulk add
    results = git.add_projects([str(workspace_dir / "pipelines")])
    assert len(results) == 1
    assert results[0].status == "success"


@patch("repository_manager.repository_manager.Git.git_action")
def test_git_commit_operations(mock_git_action, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))

    # Mock status --porcelain to show no staged changes first
    mock_git_action.return_value = GitResult(
        status="success", data="", metadata=get_mock_metadata("git status --porcelain")
    )

    # Test commit with no changes (should skip)
    res = git.commit_project(
        message="Updating services", path=str(workspace_dir / "pipelines")
    )
    assert res.status == "success"
    assert "skipped" in res.data

    # Now mock status --porcelain to show staged changes
    def mock_status_call(command, path, **kwargs):
        if "status --porcelain" in command:
            return GitResult(
                status="success",
                data="M  somefile.py\n",
                metadata=get_mock_metadata("git status --porcelain"),
            )
        elif "commit" in command:
            return GitResult(
                status="success",
                data="Committed",
                metadata=get_mock_metadata("git commit"),
            )
        return GitResult(status="success", data="")

    mock_git_action.side_effect = mock_status_call

    # Test commit with staged changes
    res2 = git.commit_project(
        message="Updating services", path=str(workspace_dir / "pipelines")
    )
    assert res2.status == "success"
    assert "Committed" in res2.data

    # Test bulk commit
    results = git.commit_projects(
        message="Updating services", project_dirs=[str(workspace_dir / "pipelines")]
    )
    assert len(results) == 1
    assert results[0].status == "success"


@patch("repository_manager.repository_manager.Git.pre_commit")
@patch("repository_manager.repository_manager.Git.git_action")
def test_commit_code_stages_untracked_and_gates_on_precommit(
    mock_git_action, mock_pre_commit, sample_workspace_yml
):
    """commit_code stages ALL changes (incl. untracked), runs pre-commit, commits."""
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))
    target = str(workspace_dir / "pipelines")
    (workspace_dir / "pipelines" / ".git").mkdir(parents=True, exist_ok=True)
    # Make the repo look like it has an untracked file + a staged change.
    (workspace_dir / "pipelines" / ".pre-commit-config.yaml").write_text("repos: []\n")

    calls: list[str] = []

    def mock_call(command, path=None, **kwargs):
        calls.append(command)
        # After `git add -A`, the previously-untracked file is staged ("A ").
        data = "A  new_feature.py\n" if "status --porcelain" in command else "ok"
        return GitResult(
            status="success", data=data, metadata=get_mock_metadata(command)
        )

    mock_git_action.side_effect = mock_call
    mock_pre_commit.return_value = GitResult(
        status="success", data="hooks passed", metadata=get_mock_metadata("pre_commit")
    )

    res = git.commit_code_project("feat: x", run_precommit=True, path=target)
    assert res.status == "success"
    mock_pre_commit.assert_called_once()  # hooks gated the commit
    assert any(c == "git add -A" for c in calls)  # untracked files staged
    assert any("commit" in c for c in calls)


@patch("repository_manager.repository_manager.Git.pre_commit")
@patch("repository_manager.repository_manager.Git.git_action")
def test_commit_code_aborts_when_precommit_fails(
    mock_git_action, mock_pre_commit, sample_workspace_yml
):
    """A real pre-commit failure surfaces and nothing is committed."""
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))
    target = str(workspace_dir / "pipelines")
    (workspace_dir / "pipelines" / ".git").mkdir(parents=True, exist_ok=True)
    (workspace_dir / "pipelines" / ".pre-commit-config.yaml").write_text("repos: []\n")

    committed: list[str] = []

    def mock_call(command, path=None, **kwargs):
        if "commit" in command and "status" not in command:
            committed.append(command)
        data = "M  f.py\n" if "status --porcelain" in command else "ok"
        return GitResult(
            status="success", data=data, metadata=get_mock_metadata(command)
        )

    mock_git_action.side_effect = mock_call
    mock_pre_commit.return_value = GitResult(
        status="error", data="hook failed", metadata=get_mock_metadata("pre_commit")
    )

    res = git.commit_code_project("feat: x", run_precommit=True, path=target)
    assert res.status == "error"
    assert not committed  # never committed on hook failure


@patch("repository_manager.repository_manager.Git.git_action")
def test_bump_version_fallback_no_changes(mock_git_action, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))
    (workspace_dir / "pipelines").mkdir(parents=True, exist_ok=True)

    # Mock status --porcelain to show no changes
    mock_git_action.return_value = GitResult(
        status="success", data="", metadata=get_mock_metadata("git status --porcelain")
    )

    res = git.bump_version(part="patch", path=str(workspace_dir / "pipelines"))
    assert res.status == "skipped"
    assert "No changes to stage or commit" in res.data


@patch("repository_manager.repository_manager.Git.git_action")
def test_bump_version_fallback_dry_run(mock_git_action, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))
    (workspace_dir / "pipelines").mkdir(parents=True, exist_ok=True)

    # Mock status --porcelain to show dirty status
    mock_git_action.return_value = GitResult(
        status="success", data="M  file.py\n", metadata=get_mock_metadata("git status")
    )

    res = git.bump_version(
        part="patch", path=str(workspace_dir / "pipelines"), dry_run=True
    )
    assert res.status == "success"
    assert "current_version=unknown" in res.data
    assert "new_version=unknown" in res.data


@patch("repository_manager.repository_manager.Git.git_action")
def test_bump_version_fallback_execution(mock_git_action, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml
    git = Git(path=str(workspace_dir))
    git.load_projects_from_yaml(str(yml_path))
    (workspace_dir / "pipelines").mkdir(parents=True, exist_ok=True)

    def mock_git_calls(command, path, **kwargs):
        if "status --porcelain" in command:
            return GitResult(
                status="success",
                data="M  somefile.py\n",
                metadata=get_mock_metadata("git status"),
            )
        elif "add -A" in command:
            return GitResult(
                status="success",
                data="added",
                metadata=get_mock_metadata("git add"),
            )
        elif "commit" in command:
            assert "--no-verify" not in command
            assert "phased bump" in command
            return GitResult(
                status="success",
                data="Committed fallback",
                metadata=get_mock_metadata("git commit"),
            )
        return GitResult(status="success", data="")

    mock_git_action.side_effect = mock_git_calls

    res = git.bump_version(
        part="patch", path=str(workspace_dir / "pipelines"), dry_run=False
    )
    assert res.status == "success"
    assert "current_version=unknown" in res.data
    assert "new_version=unknown" in res.data


class TestUpdateDependencyPropagation:
    """update_dependency must propagate across all pin shapes + file types.

    Regression: previously only quoted ``>=`` pyproject entries were matched,
    silently leaving ``==`` pins and ALL requirements.txt refs stale.
    (CONCEPT:RM-BUMP cross-dependency propagation)
    """

    def _run(self, tmp_path, content):
        f = tmp_path / "deps.txt"
        f.write_text(content)
        git = Git(path=str(tmp_path))
        updated = git.update_dependency(str(f), "agent-utilities", "0.39.0")
        return updated, f.read_text()

    def test_pyproject_gte(self, tmp_path):
        upd, out = self._run(tmp_path, 'deps = ["agent-utilities>=0.38.0"]')
        assert upd and "agent-utilities>=0.39.0" in out

    def test_requirements_exact_pin(self, tmp_path):
        upd, out = self._run(tmp_path, "agent-utilities==0.16.0\n")
        assert upd and "agent-utilities==0.39.0" in out  # == preserved, not stale

    def test_extras_and_compatible(self, tmp_path):
        upd, out = self._run(tmp_path, "agent-utilities[all]~=0.30.0\n")
        assert upd and "agent-utilities[all]~=0.39.0" in out

    def test_transitive_comment_untouched(self, tmp_path):
        upd, out = self._run(tmp_path, "    # via agent-utilities\n")
        assert upd is False and out.strip() == "# via agent-utilities"

    def test_unpinned_untouched(self, tmp_path):
        upd, out = self._run(tmp_path, "agent-utilities\n")
        assert upd is False
