from unittest.mock import MagicMock, patch

import pytest

from repository_manager.repository_manager import Git, GitResult


@pytest.fixture
def mock_repo_manager(tmp_path):
    manager = Git(path=str(tmp_path))
    manager.project_map = {
        "https://github.com/Knuckles-Team/repo1.git": str(tmp_path / "repo1"),
        "https://github.com/Knuckles-Team/repo2.git": str(tmp_path / "repo2"),
        "https://github.com/Knuckles-Team/repo3.git": str(tmp_path / "repo3"),
    }
    # The phased push/bump loops skip projects whose local clone is absent
    # (os.path.isdir guard); create the mapped dirs so the mocked git_action runs.
    for name in ("repo1", "repo2", "repo3"):
        (tmp_path / name).mkdir(exist_ok=True)

    def git_action_side_effect(*args, **kwargs):
        command = kwargs.get("command", "")
        if not command and args:
            command = args[0]
        if "status --porcelain" in command:
            return GitResult(status="success", data="", error=None, metadata=None)
        return GitResult(status="success", data="Pushed", error=None, metadata=None)

    manager.git_action = MagicMock(side_effect=git_action_side_effect)  # type: ignore[method-assign]
    return manager


@patch("time.sleep")
def test_phased_push(mock_sleep, mock_repo_manager):
    config = {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 5},
            {
                "phase": 2,
                "name": "Phase 2",
                "projects": ["repo2", "repo3"],
                "wait_minutes": 10,
            },
        ]
    }

    # auto_start=False isolates the raw push loop (no change-detection git calls).
    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, auto_start=False
    )

    assert len(results) == 3  # 3 pushes
    # 3 status checks + 3 pushes = 6 calls
    assert mock_repo_manager.git_action.call_count == 6

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
                "wait_minutes": 5,
            }
        ]
    }

    results = mock_repo_manager.phased_push(
        start_phase=1, config=config, project_filter="repo1"
    )

    assert len(results) == 1
    # 1 status check + 1 push = 2 calls
    assert mock_repo_manager.git_action.call_count == 2

    # Still sleep after phase if project filter matches
    assert mock_sleep.call_count == 1


def test_push_projects(mock_repo_manager):
    results = mock_repo_manager.push_projects(["/fake/path/repo1", "/fake/path/repo2"])

    assert len(results) == 2
    # 2 status checks + 2 pushes = 4 calls
    assert mock_repo_manager.git_action.call_count == 4
    # Verify the push commands called were git push --follow-tags
    push_calls = [
        call
        for call in mock_repo_manager.git_action.call_args_list
        if "git push --follow-tags"
        in (call.kwargs.get("command") or (call.args[0] if call.args else ""))
    ]
    assert len(push_calls) == 2
