from unittest.mock import MagicMock, patch

import pytest

from repository_manager.repository_manager import Git, GitResult

CLEAN = (
    "On branch main\n"
    "Your branch is up to date with 'origin/main'.\n"
    "nothing to commit, working tree clean"
)
DIRTY = "On branch main\nChanges not staged for commit:\n  modified:   foo.py"
AHEAD = (
    "On branch main\n"
    "Your branch is ahead of 'origin/main' by 1 commit.\n"
    "nothing to commit, working tree clean"
)

CONFIG = {
    "phases": [
        {"phase": 1, "name": "Phase 1", "projects": ["repo1"], "wait_minutes": 5},
        {"phase": 2, "name": "Phase 2", "projects": ["repo2"], "wait_minutes": 5},
        {"phase": 3, "name": "Phase 3", "projects": ["repo3"], "wait_minutes": 5},
    ]
}


def _make_manager(tmp_path, pending_status):
    """Git manager whose ``git status`` reports each repo per ``pending_status``.

    ``pending_status`` maps a repo name to the status text returned for it
    (CLEAN / DIRTY / AHEAD). Other git commands return a generic success.
    """
    manager = Git(path=str(tmp_path))
    manager.project_map = {
        "https://github.com/Knuckles-Team/repo1.git": str(tmp_path / "repo1"),
        "https://github.com/Knuckles-Team/repo2.git": str(tmp_path / "repo2"),
        "https://github.com/Knuckles-Team/repo3.git": str(tmp_path / "repo3"),
    }
    path_to_status = {
        str(tmp_path / name): status for name, status in pending_status.items()
    }

    def git_action_side_effect(*args, **kwargs):
        command = kwargs.get("command", "") or (args[0] if args else "")
        path = kwargs.get("path", "")
        if command == "git status":
            return GitResult(
                status="success", data=path_to_status[path], error=None, metadata=None
            )
        return GitResult(status="success", data="ok", error=None, metadata=None)

    manager.git_action = MagicMock(side_effect=git_action_side_effect)  # type: ignore[method-assign]
    return manager


def test_auto_start_phase_detects_lowest_changed(tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": DIRTY, "repo3": DIRTY}
    )
    assert manager._auto_start_phase(CONFIG) == 2


def test_auto_start_phase_counts_unpushed_commits(tmp_path):
    # repo2 has no working-tree changes but is ahead of origin (awaiting push).
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": AHEAD, "repo3": CLEAN}
    )
    assert manager._auto_start_phase(CONFIG) == 2


def test_auto_start_phase_none_when_all_clean(tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": CLEAN, "repo3": CLEAN}
    )
    assert manager._auto_start_phase(CONFIG) is None


def test_auto_start_phase_lowest_wins(tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": DIRTY, "repo2": CLEAN, "repo3": DIRTY}
    )
    assert manager._auto_start_phase(CONFIG) == 1


@patch("time.sleep")
def test_phased_push_auto_start_skips_early_phases(mock_sleep, tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": DIRTY, "repo3": DIRTY}
    )
    manager.push_project = MagicMock(
        return_value=GitResult(status="success", data="Pushed", error=None, metadata=None)
    )

    progress: dict = {"current_phase": "", "progress": 0, "phases": {}}
    results = manager.phased_push(config=CONFIG, auto_start=True, progress=progress)

    pushed_paths = {c.kwargs.get("path") for c in manager.push_project.call_args_list}
    assert pushed_paths == {str(tmp_path / "repo2"), str(tmp_path / "repo3")}
    assert len(results) == 2
    # Phase 1 was skipped entirely, so it never appears in the progress tree.
    assert set(progress["phases"]) == {"Phase 2", "Phase 3"}


@patch("time.sleep")
def test_phased_push_auto_start_no_changes_is_noop(mock_sleep, tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": CLEAN, "repo3": CLEAN}
    )
    manager.push_project = MagicMock()

    results = manager.phased_push(config=CONFIG, auto_start=True)

    manager.push_project.assert_not_called()
    assert results == []


def test_phased_bumpversion_auto_start_skips_early_phases(tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": DIRTY, "repo3": CLEAN}
    )
    manager.bump_version = MagicMock(
        return_value=GitResult(
            status="success", data="new_version=1.0.1", error=None, metadata=None
        )
    )
    manager.update_dependency = MagicMock(return_value=False)

    progress: dict = {"current_phase": "", "progress": 0, "phases": {}}
    manager.phased_bumpversion(config=CONFIG, auto_start=True, progress=progress)

    bumped_paths = {c.kwargs.get("path") for c in manager.bump_version.call_args_list}
    # repo1 (Phase 1) is skipped; repo2 changed and repo3 is downstream of it.
    assert str(tmp_path / "repo1") not in bumped_paths
    assert str(tmp_path / "repo2") in bumped_paths
    assert set(progress["phases"]) == {"Phase 2", "Phase 3"}


def test_phased_bumpversion_auto_start_no_changes_is_noop(tmp_path):
    manager = _make_manager(
        tmp_path, {"repo1": CLEAN, "repo2": CLEAN, "repo3": CLEAN}
    )
    manager.bump_version = MagicMock()

    results = manager.phased_bumpversion(config=CONFIG, auto_start=True)

    manager.bump_version.assert_not_called()
    assert results == []
