import pytest
from unittest.mock import patch, MagicMock
from repository_manager.repository_manager import Git, GitResult

@pytest.fixture
def mock_repo_manager(tmp_path):
    manager = Git(path=str(tmp_path))
    manager.project_map = {
        "https://github.com/Knuckles-Team/repo1.git": str(tmp_path / "repo1"),
        "https://github.com/Knuckles-Team/repo2.git": str(tmp_path / "repo2"),
        "https://github.com/Knuckles-Team/repo3.git": str(tmp_path / "repo3")
    }
    manager.git_action = MagicMock(return_value=GitResult(status="success", data="Pushed", error=None, metadata=None))  # type: ignore[method-assign]
    return manager

@patch("time.sleep")
def test_phased_push(mock_sleep, mock_repo_manager):
    config = {
        "phases": [
            {
                "phase": 1,
                "name": "Phase 1",
                "projects": ["repo1"],
                "wait_minutes": 5
            },
            {
                "phase": 2,
                "name": "Phase 2",
                "projects": ["repo2", "repo3"],
                "wait_minutes": 10
            }
        ]
    }

    results = mock_repo_manager.phased_push(start_phase=1, config=config)

    assert len(results) == 3  # 3 pushes
    assert mock_repo_manager.git_action.call_count == 3

    # Should sleep for 5 mins after phase 1, and 10 mins after phase 2
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(5 * 60)
    mock_sleep.assert_any_call(10 * 60)

@patch("time.sleep")
def test_phased_push_single_project(mock_sleep, mock_repo_manager):
    config = {
        "phases": [
            {
                "phase": 1,
                "name": "Phase 1",
                "projects": ["repo1", "repo2"],
                "wait_minutes": 5
            }
        ]
    }

    results = mock_repo_manager.phased_push(start_phase=1, config=config, project_filter="repo1")

    assert len(results) == 1
    assert mock_repo_manager.git_action.call_count == 1

    # Still sleep after phase if project filter matches
    assert mock_sleep.call_count == 1

def test_push_projects(mock_repo_manager):
    results = mock_repo_manager.push_projects(["/fake/path/repo1", "/fake/path/repo2"])

    assert len(results) == 2
    assert mock_repo_manager.git_action.call_count == 2
    # Verify the git_action was called with the correct command
    calls = mock_repo_manager.git_action.call_args_list
    assert all(call.kwargs.get('command') == 'git push --follow-tags' for call in calls)
