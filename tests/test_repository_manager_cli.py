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


def test_cli_maintain_calls_bumpversion(mock_git, mock_sys_argv):
    with mock_sys_argv(["--maintain", "--dry-run"]):
        main()
        mock_git.phased_bumpversion.assert_called_once()


def test_cli_clone(mock_git, mock_sys_argv):
    with mock_sys_argv(["--clone"]):
        main()
        mock_git.clone_projects.assert_called_once()


def test_cli_install(mock_git, mock_sys_argv):
    with mock_sys_argv(["--install"]):
        main()
        mock_git.install_projects.assert_called_once()
