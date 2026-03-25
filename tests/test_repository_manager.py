import pytest
import os
import yaml
import datetime
from unittest.mock import MagicMock, patch
from repository_manager.models import (
    WorkspaceConfig,
    GitResult,
    GitMetadata,
    RepositoryConfig,
    SubdirectoryConfig,
)
from repository_manager.repository_manager import WorkspaceManager, Git

def get_mock_metadata(command="test"):
    return GitMetadata(
        command=command,
        workspace="/tmp",
        return_code=0,
        timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"
    )

# Path to the actual workspace file for integration testing
ACTUAL_WORKSPACE_YML = os.path.join(
    os.path.dirname(os.path.dirname(__file__)),
    "repository_manager",
    "workspace.yml"
)

@pytest.fixture
def real_workspace_data():
    """Returns the parsed content of the actual workspace.yml."""
    with open(ACTUAL_WORKSPACE_YML, "r") as f:
        return yaml.safe_load(f)

@pytest.fixture
def sample_workspace_yml(tmp_path, real_workspace_data):
    workspace_dir = tmp_path / "my_workspace"
    workspace_dir.mkdir()

    # We can use the real structure but point to a tmp directory for safety
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
    assert len(config.maintenance.phases) > 0
    # Specific check for the complex Phase 4 with target_pattern
    phase4 = next((p for p in config.maintenance.phases if p.phase == 4), None)
    assert phase4 is not None
    assert phase4.updates[0].target_pattern == "*"

def test_workspace_config_parsing(sample_workspace_yml):
    yml_path, _ = sample_workspace_yml
    with open(yml_path, "r") as f:
        data = yaml.safe_load(f)

    config = WorkspaceConfig(**data)
    assert config.name == "Agent Packages Workspace"
    # In real file, repositories are under agent-packages
    assert "agent-packages" in config.subdirectories
    assert config.maintenance.phases[0].name == "Phase 1: universal-skills"

@patch("repository_manager.repository_manager.Git.git_action")
@patch("repository_manager.repository_manager.Git.pull_project")
def test_setup_from_yaml(mock_pull, mock_git, sample_workspace_yml):
    yml_path, workspace_dir = sample_workspace_yml

    # Mock success for git operations
    mock_git.return_value = GitResult(
        status="success",
        data="Cloned",
        metadata=get_mock_metadata("clone")
    )
    mock_pull.return_value = GitResult(
        status="success",
        data="Pulled",
        metadata=get_mock_metadata("pull")
    )

    wm = WorkspaceManager(git_instance=Git(path=str(workspace_dir)))
    result = wm.setup_from_yaml(str(yml_path))

    assert result.status == "success"
    # Should have called git_action for all repos in the real file
    # real file has 4 repos in agent-packages and ~30 in agent-packages/agents
    assert mock_git.call_count > 0

    # Simulate directory existence for the second run
    # For simplicity, we just check that it calls pull_project when dirs exist
    # We'll just create one directory from the real config to verify logic
    (workspace_dir / "agent-packages" / "agent-utilities").mkdir(parents=True)

    result = wm.setup_from_yaml(str(yml_path))
    # mock_pull should be called at least for the one we created
    assert mock_pull.call_count > 0

def test_maintenance_config_loading(sample_workspace_yml):
    yml_path, _ = sample_workspace_yml
    git = Git()

    # Set the environment variable for the test
    os.environ["WORKSPACE_YML"] = str(yml_path)

    # Mock the internal methods called by maintain_projects to avoid actual execution
    with patch.object(Git, "pre_commit_projects") as mock_pre:
        with patch.object(Git, "bump_version") as mock_bump:
            # mock_pre should return a list of GitResult objects
            mock_pre.return_value = [
                GitResult(status="success", data="OK", metadata=get_mock_metadata())
            ]
            mock_bump.return_value = get_mock_metadata("bump")

            # Use a high start_phase to skip execution but trigger config loading
            git.maintain_projects(dry_run=True, start_phase=100)

            assert hasattr(git, "config")
            assert git.config.name == "Agent Packages Workspace"
            assert git.config.maintenance.phases[0].phase == 1

def test_git_init_with_default_workspace():
    # Ensure Git initialized with default workspace if none provided
    git = Git()
    assert git.path is not None
    # Check if it picked up the default
    from repository_manager.repository_manager import DEFAULT_REPOSITORY_MANAGER_WORKSPACE
    assert git.path == DEFAULT_REPOSITORY_MANAGER_WORKSPACE
