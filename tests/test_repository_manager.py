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

            git.maintain_projects(dry_run=True, start_phase=100)

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

    res = git.bump_version(part="patch", path=str(workspace_dir / "pipelines"), dry_run=True)
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
            assert "--no-verify" in command
            assert "phased push" in command
            return GitResult(
                status="success",
                data="Committed fallback",
                metadata=get_mock_metadata("git commit"),
            )
        return GitResult(status="success", data="")

    mock_git_action.side_effect = mock_git_calls

    res = git.bump_version(part="patch", path=str(workspace_dir / "pipelines"), dry_run=False)
    assert res.status == "success"
    assert "current_version=unknown" in res.data
    assert "new_version=unknown" in res.data
