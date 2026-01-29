import os
import pytest
from unittest.mock import MagicMock, patch, mock_open
from repository_manager.repository_manager import Git


# Helper to normalize paths for assertions
def normalize_path(path):
    return os.path.normpath(os.path.abspath(path))


@pytest.fixture
def git_instance():
    # Setup
    workspace = normalize_path("/tmp/workspace")
    git = Git(workspace=workspace)
    return git


class TestDirectoryFunctions:

    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_create_directory_success(self, mock_exists, mock_makedirs, git_instance):
        mock_exists.return_value = (
            False  # Directory does not exist, project exists (handled in logic)
        )

        # We need to handle side_effect for exists to verify project existence if checked
        # The code checks:
        # 1. Project path exists (if project provided) -> logic: if not os.path.exists(project_path)
        # 2. Start of function check: if os.path.exists(target_path)

        # Let's mock a scenario where project exists but target doesn't
        def exists_side_effect(path):
            if path.endswith("myproject"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        result = git_instance.create_directory(
            workspace=git_instance.workspace, project="myproject", path="newdir"
        )

        assert result.status == "success"
        assert result.error is None
        mock_makedirs.assert_called_once()

    @patch("os.path.exists")
    def test_create_directory_already_exists(self, mock_exists, git_instance):
        mock_exists.return_value = True
        result = git_instance.create_directory(
            workspace=git_instance.workspace, project="myproject", path="existingdir"
        )

        assert result.status == "error"
        assert "Directory already exists" in result.error.message

    @patch("os.path.exists")
    def test_create_directory_project_not_found(self, mock_exists, git_instance):
        mock_exists.return_value = False
        result = git_instance.create_directory(
            workspace=git_instance.workspace, project="missingproject", path="newdir"
        )

        # The first check in create_directory when project is provided is valid project path
        assert result.status == "error"
        assert "Project directory does not exist" in result.error.message

    def test_create_directory_outside_workspace(self, git_instance):
        # No mocks needed for path manipulation mostly, but existence check might hit.
        # However, logic fails before existence check if path is outside.
        # path="../outside" relative to workspace

        result = git_instance.create_directory(
            workspace=git_instance.workspace, project=None, path="../outside"
        )

        assert result.status == "error"
        assert "outside of workspace" in result.error.message

    @patch("shutil.rmtree")
    @patch("os.remove")
    @patch("os.path.isfile")
    @patch("os.path.exists")
    def test_delete_directory_success_dir(
        self, mock_exists, mock_isfile, mock_remove, mock_rmtree, git_instance
    ):
        mock_exists.return_value = True
        mock_isfile.return_value = False

        result = git_instance.delete_directory(
            workspace=git_instance.workspace, project="myproject", path="todelete"
        )

        assert result.status == "success"
        mock_rmtree.assert_called_once()
        mock_remove.assert_not_called()

    @patch("os.path.exists")
    def test_delete_directory_not_found(self, mock_exists, git_instance):
        def exists_side_effect(path):
            if path.endswith("myproject"):
                return True
            return False

        mock_exists.side_effect = exists_side_effect

        result = git_instance.delete_directory(
            workspace=git_instance.workspace, project="myproject", path="missing"
        )

        assert result.status == "error"
        assert "not found" in result.error.message

    def test_delete_directory_workspace_root(self, git_instance):
        result = git_instance.delete_directory(
            workspace=git_instance.workspace, project=None, path="."
        )  # Relative to workspace, this is workspace root
        assert result.status == "error"
        assert "Cannot delete the workspace root" in result.error.message

        result = git_instance.delete_directory(
            workspace=git_instance.workspace, project=None, path=git_instance.workspace
        )
        assert result.status == "error"
        assert "Cannot delete the workspace root" in result.error.message

    @patch("os.renames")
    @patch("os.path.exists")
    def test_rename_directory_success(self, mock_exists, mock_renames, git_instance):
        # We need exists(old) -> True, exists(new) -> False
        def exists_side_effect(path):
            if "old" in path:
                return True
            if "new" in path:
                return False
            if "myproject" in path:
                return True  # Project exists
            return False

        mock_exists.side_effect = exists_side_effect

        result = git_instance.rename_directory(
            workspace=git_instance.workspace,
            project="myproject",
            old_path="old",
            new_path="new",
        )

        assert result.status == "success"
        mock_renames.assert_called_once()

    @patch("os.path.exists")
    def test_rename_directory_source_missing(self, mock_exists, git_instance):
        def exists_side_effect(path):
            if "missing" in path:
                return False
            if "myproject" in path:
                return True  # Project exists
            return False

        mock_exists.side_effect = exists_side_effect

        result = git_instance.rename_directory(
            workspace=git_instance.workspace,
            project="myproject",
            old_path="missing",
            new_path="new",
        )

        assert result.status == "error"
        assert "Source path not found" in result.error.message


class TestTextEditor:

    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\nline3")
    @patch("os.path.exists")
    def test_view_file(self, mock_exists, mock_file, git_instance):
        mock_exists.return_value = True

        result = git_instance.text_editor(command="view", path="test.txt")
        assert result.status == "success"
        assert result.data == "line1\nline2\nline3"

    @patch("builtins.open", new_callable=mock_open, read_data="line1\nline2\nline3")
    @patch("os.path.exists")
    def test_view_file_range(self, mock_exists, mock_file, git_instance):
        mock_exists.return_value = True

        # View lines 1-2
        result = git_instance.text_editor(
            command="view", path="test.txt", view_range=[1, 2]
        )
        assert result.status == "success"
        # Since lines are 0-indexed in list but 1-indexed in range logic of text_editor?
        # Checking implementation: "start = view_range[0] - 1", "end = view_range[1]"
        # So [1, 2] -> slice [0:2] -> line1, line2
        assert result.data == "line1\nline2"

    @patch("builtins.open", new_callable=mock_open)
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_create_file(self, mock_exists, mock_makedirs, mock_file, git_instance):
        mock_exists.return_value = False

        result = git_instance.text_editor(
            command="create", path="new.txt", file_text="hello"
        )

        assert result.status == "success"
        mock_file.assert_called_with(
            os.path.join(git_instance.workspace, "new.txt"), "w"
        )
        mock_file().write.assert_called_with("hello")

    @patch("builtins.open", new_callable=mock_open, read_data="hello world")
    @patch("os.path.exists")
    def test_str_replace(self, mock_exists, mock_file, git_instance):
        mock_exists.return_value = True

        # We need to simulate the file read and then write
        # mock_open handles read well. For write, we check what was written.

        result = git_instance.text_editor(
            command="str_replace", path="file.txt", old_str="world", new_str="universe"
        )

        assert result.status == "success"
        mock_file().write.assert_called_with("hello universe")

    @patch("builtins.open", new_callable=mock_open, read_data="hello world")
    @patch("os.path.exists")
    def test_str_replace_not_found(self, mock_exists, mock_file, git_instance):
        mock_exists.return_value = True
        result = git_instance.text_editor(
            command="str_replace", path="file.txt", old_str="universe", new_str="world"
        )
        assert result.status == "error"
        assert "not found in file" in result.error.message


class TestProjectFunctions:

    @patch("subprocess.Popen")
    @patch("os.makedirs")
    @patch("os.path.exists")
    def test_create_project_success(
        self, mock_exists, mock_makedirs, mock_popen, git_instance
    ):
        mock_exists.return_value = False  # Directory doesn't exist

        # Mock connection to git init
        process_mock = MagicMock()
        process_mock.communicate.return_value = ("Initialized empty Git repository", "")
        process_mock.wait.return_value = 0
        mock_popen.return_value = process_mock

        result = git_instance.create_project(project="newproj")

        assert result.status == "success"
        mock_makedirs.assert_called_once()
        mock_popen.assert_called()

    @patch("subprocess.Popen")
    @patch("os.path.exists")
    def test_pre_commit_success(self, mock_exists, mock_popen, git_instance):
        mock_exists.return_value = True  # Config exists

        process_mock = MagicMock()
        process_mock.communicate.return_value = ("Passed", "")
        process_mock.wait.return_value = 0
        mock_popen.return_value = process_mock

        result = git_instance.pre_commit(workspace=git_instance.workspace)

        assert result.status == "success"
        assert "pre-commit run --all-files" in mock_popen.call_args[0][0]

    @patch("builtins.open", new_callable=mock_open, read_data="# Readme Content")
    @patch("os.listdir")
    @patch("os.path.exists")
    def test_get_readme(self, mock_exists, mock_listdir, mock_file, git_instance):
        mock_exists.return_value = True
        mock_listdir.return_value = ["README.md", "other.txt"]

        result = git_instance.get_readme()

        assert result.content == "# Readme Content"
        assert result.path.endswith("README.md")

    @patch("os.path.exists")
    def test_bump_version_success(self, mock_exists, git_instance):
        mock_exists.return_value = True
        # Mock git_action
        git_instance.git_action = MagicMock()
        git_instance.git_action.return_value.status = "success"

        result = git_instance.bump_version(
            part="minor", workspace=git_instance.workspace, project="myproj"
        )

        assert result.status == "success"
        # Check if command was constructed correctly
        call_args = git_instance.git_action.call_args[1]
        assert call_args["command"] == "bump2version minor"
        assert "myproj" in call_args["workspace"]

    @patch("os.path.exists")
    def test_bump_version_allow_dirty(self, mock_exists, git_instance):
        mock_exists.return_value = True
        git_instance.git_action = MagicMock()
        git_instance.git_action.return_value.status = "success"

        result = git_instance.bump_version(
            part="patch",
            allow_dirty=True,
            workspace=git_instance.workspace,
            project="myproj",
        )

        call_args = git_instance.git_action.call_args[1]
        assert "--allow-dirty" in call_args["command"]

    @patch("os.path.exists")
    def test_bump_version_invalid_part(self, mock_exists, git_instance):
        mock_exists.return_value = True  # Ensure directory exists check passes
        result = git_instance.bump_version(
            part="supermajor", workspace=git_instance.workspace
        )
        assert result.status == "error"
        # The message includes "Invalid part 'supermajor'. Must be one of..."
        assert "Invalid part" in result.error.message

    @patch("os.path.exists")
    def test_bump_version_not_found(self, mock_exists, git_instance):
        mock_exists.return_value = False
        result = git_instance.bump_version(
            part="major", workspace=git_instance.workspace, project="missing"
        )
        assert result.status == "error"
        assert "Directory not found" in result.error.message
