"""Characterization tests for ``Git.install_projects`` (WC1-REPOSITORY-01).

``install_projects`` had NO real branch coverage before this lane: the only
pre-existing references (``tests/test_repository_manager_cli.py::test_cli_install``,
``tests/test_mcp_handlers.py``) mock the whole method and only assert it was
*called*. These tests pin the method's actual observable behavior --
``git_action``/``_materialize_uv_siblings`` call sequence and the returned
``GitResult`` list -- across every branch this lane's extract-method
refactor touches. Run once against the unmodified function (record green),
then once after the refactor (require identical), per the WC1-REPOSITORY-01
brief's characterization discipline.
"""

import datetime
from unittest.mock import patch

import pytest

from repository_manager.models import GitMetadata, GitResult
from repository_manager.repository_manager import Git


def _meta(command="test"):
    return GitMetadata(
        command=command,
        workspace="/tmp",
        return_code=0,
        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
    )


@pytest.fixture
def git_with_projects(tmp_path):
    """A Git instance with an agent-utilities project plus one downstream project."""
    git = Git(path=str(tmp_path))
    au_dir = tmp_path / "agent-utilities"
    au_dir.mkdir()
    proj_dir = tmp_path / "proj-a"
    proj_dir.mkdir()
    git.project_map = {
        "https://example.invalid/agent-utilities.git": str(au_dir),
        "https://example.invalid/proj-a.git": str(proj_dir),
    }
    return git, au_dir, proj_dir


def test_install_projects_no_project_map(tmp_path):
    git = Git(path=str(tmp_path))
    git.project_map = {}
    assert git.install_projects(report=False) == []


@patch("repository_manager.repository_manager.shutil.which", return_value=None)
def test_install_projects_uv_missing_short_circuits_au_step(
    mock_which, git_with_projects
):
    git, au_dir, proj_dir = git_with_projects
    with patch.object(Git, "git_action") as mock_git_action:
        results = git.install_projects(report=False)
    # uv missing => step 1 never calls git_action at all; step 2 finds no
    # marker files in either project dir, so it produces two "skipped"
    # results and never calls git_action either.
    mock_git_action.assert_not_called()
    assert [r.status for r in results] == ["skipped", "skipped"]


@patch("repository_manager.repository_manager.shutil.which", return_value="/usr/bin/uv")
def test_install_projects_au_path_missing(mock_which, tmp_path):
    git = Git(path=str(tmp_path))
    proj_dir = tmp_path / "proj-a"
    proj_dir.mkdir()
    git.project_map = {"https://example.invalid/proj-a.git": str(proj_dir)}
    with patch.object(Git, "git_action") as mock_git_action:
        results = git.install_projects(report=False)
    mock_git_action.assert_not_called()
    assert [r.status for r in results] == ["skipped"]


@patch("repository_manager.repository_manager.shutil.which", return_value="/usr/bin/uv")
def test_install_projects_launcher_missing(mock_which, git_with_projects):
    git, au_dir, proj_dir = git_with_projects
    with patch.object(Git, "git_action") as mock_git_action:
        results = git.install_projects(report=False)
    mock_git_action.assert_not_called()
    # step 1: error result (missing scripts/uv_workspace.py); step 2: both
    # projects have no marker files -> two skipped results.
    assert [r.status for r in results] == ["error", "skipped", "skipped"]
    assert "uv_workspace.py" in results[0].error.message


@patch("repository_manager.repository_manager.shutil.which", return_value="/usr/bin/uv")
def test_install_projects_au_success_syncs_declared_sibling(
    mock_which, git_with_projects
):
    git, au_dir, proj_dir = git_with_projects
    (au_dir / "scripts").mkdir()
    (au_dir / "scripts" / "uv_workspace.py").write_text("# launcher\n")

    calls = []

    def fake_git_action(command, path, **kwargs):
        calls.append((command, path))
        return GitResult(status="success", data="ok", metadata=_meta(command))

    with (
        patch.object(Git, "git_action", side_effect=fake_git_action),
        patch.object(
            Git, "_materialize_uv_siblings", return_value=("agent-utilities",)
        ),
    ):
        results = git.install_projects(extra="all", report=False)

    # First call must be the au sync via the launcher, in au_dir.
    assert "uv_workspace.py" in calls[0][0]
    assert calls[0][1] == str(au_dir)
    # Second call must be the sibling's own `uv sync --all-extras`, in proj_dir
    # (the only non-au, existing project_map path).
    assert calls[1] == ("uv sync --all-extras", str(proj_dir))
    # Both results from step 1 report success; step 2 finds no marker files
    # in either directory and appends two more "skipped" results.
    assert [r.status for r in results] == ["success", "success", "skipped", "skipped"]


@patch("repository_manager.repository_manager.shutil.which", return_value="/usr/bin/uv")
def test_install_projects_sibling_materialize_error_is_captured(
    mock_which, git_with_projects
):
    git, au_dir, proj_dir = git_with_projects
    (au_dir / "scripts").mkdir()
    (au_dir / "scripts" / "uv_workspace.py").write_text("# launcher\n")

    with (
        patch.object(
            Git,
            "git_action",
            return_value=GitResult(status="success", data="ok", metadata=_meta()),
        ),
        patch.object(
            Git, "_materialize_uv_siblings", side_effect=ValueError("bad sibling map")
        ),
    ):
        results = git.install_projects(report=False)

    statuses = [r.status for r in results]
    # step1 au sync succeeds, then the sibling materialize raises -> error
    # result captured (loop continues, does not raise); step2 appends two
    # more skipped results for the marker-file-less directories.
    assert statuses == ["success", "error", "skipped", "skipped"]
    assert "bad sibling map" in results[1].error.message


@patch("repository_manager.repository_manager.shutil.which", return_value=None)
def test_install_projects_step2_node_pnpm_ignored_build_scripts_becomes_error(
    mock_which, tmp_path
):
    git = Git(path=str(tmp_path))
    proj_dir = tmp_path / "proj-a"
    proj_dir.mkdir()
    (proj_dir / ".pre-commit-config.yaml").write_text("repos: []\n")
    (proj_dir / "package.json").write_text("{}")
    (proj_dir / "pnpm-lock.yaml").write_text("")
    git.project_map = {"https://example.invalid/proj-a.git": str(proj_dir)}

    with patch.object(
        Git,
        "git_action",
        return_value=GitResult(
            status="success", data="Ignored build scripts: foo", metadata=_meta()
        ),
    ) as mock_git_action:
        results = git.install_projects(report=False)

    mock_git_action.assert_called_once_with("pnpm install", path=str(proj_dir))
    assert len(results) == 1
    assert results[0].status == "error"
    assert "Ignored build scripts" in results[0].data


@patch("repository_manager.repository_manager.shutil.which", return_value=None)
def test_install_projects_step2_python_only_project_produces_no_result(
    mock_which, tmp_path
):
    """A pyproject-only (non-Node) project is handled by the uv-sync step,
    not step 2 -- step 2 must neither call git_action nor append a result
    for it."""
    git = Git(path=str(tmp_path))
    proj_dir = tmp_path / "proj-a"
    proj_dir.mkdir()
    (proj_dir / "pyproject.toml").write_text("[project]\nname='x'\n")
    git.project_map = {"https://example.invalid/proj-a.git": str(proj_dir)}

    with patch.object(Git, "git_action") as mock_git_action:
        results = git.install_projects(report=False)

    mock_git_action.assert_not_called()
    assert results == []


@patch("repository_manager.repository_manager.shutil.which", return_value=None)
def test_install_projects_report_exported_when_enabled(mock_which, tmp_path):
    git = Git(path=str(tmp_path), report_path=str(tmp_path / "report.md"))
    proj_dir = tmp_path / "proj-a"
    proj_dir.mkdir()
    git.project_map = {"https://example.invalid/proj-a.git": str(proj_dir)}

    with patch.object(Git, "_export_report") as mock_export:
        git.install_projects(report=True)
    mock_export.assert_called_once()
    args, _ = mock_export.call_args
    assert args[1] == "install_report.md"
    assert "INSTALLATION SUMMARY" in args[0]


@patch("repository_manager.repository_manager.shutil.which", return_value=None)
def test_install_projects_report_not_exported_when_disabled(mock_which, tmp_path):
    git = Git(path=str(tmp_path), report_path=str(tmp_path / "report.md"))
    proj_dir = tmp_path / "proj-a"
    proj_dir.mkdir()
    git.project_map = {"https://example.invalid/proj-a.git": str(proj_dir)}

    with patch.object(Git, "_export_report") as mock_export:
        git.install_projects(report=False)
    mock_export.assert_not_called()
