import sys
from unittest.mock import patch

import pytest

from repository_manager.repository_manager import main


@pytest.fixture
def mock_git():
    with patch("repository_manager.repository_manager.Git") as MockGit:
        mock_instance = MockGit.return_value
        yield mock_instance


@pytest.fixture
def mock_sys_argv():
    def _mock_argv(args):
        return patch.object(sys, "argv", ["repository-manager"] + args)

    return _mock_argv


def test_cli_graph_status(mock_git, mock_sys_argv):
    mock_git.graph_status.return_value = {"nodes": 0}
    with mock_sys_argv(["--graph-status"]):
        main()
        mock_git.graph_status.assert_called_once()


def test_cli_graph_reset(mock_git, mock_sys_argv):
    mock_git.graph_reset.return_value = "Reset"
    with mock_sys_argv(["--graph-reset"]):
        main()
        mock_git.graph_reset.assert_called_once()


def test_cli_graph_query(mock_git, mock_sys_argv):
    with mock_sys_argv(["--graph-query", "GitResult", "--graph-mode", "semantic"]):
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = [{"id": "test"}]
            main()
            mock_run.assert_called_once()


def test_cli_graph_impact(mock_git, mock_sys_argv):
    with mock_sys_argv(["--graph-impact", "WorkspaceConfig"]):
        with patch("asyncio.run") as mock_run:
            mock_run.return_value = [{"id": "test"}]
            main()
            mock_run.assert_called_once()


def test_cli_graph_path(mock_git, mock_sys_argv):
    mock_git.graph_path.return_value = []
    with mock_sys_argv(["--graph-path", "Source", "Target"]):
        main()
        mock_git.graph_path.assert_called_once_with("Source", "Target")


def test_cli_maintain_calls_ensure_graph(mock_git, mock_sys_argv):
    with mock_sys_argv(["--maintain", "--dry-run"]):
        main()
        mock_git.phased_bumpversion.assert_called_once()
        mock_git.ensure_graph.assert_called_once()


def test_cli_clone(mock_git, mock_sys_argv):
    with mock_sys_argv(["--clone"]):
        main()
        mock_git.clone_projects.assert_called_once()


def test_cli_install(mock_git, mock_sys_argv):
    with mock_sys_argv(["--install"]):
        main()
        mock_git.install_projects.assert_called_once()
