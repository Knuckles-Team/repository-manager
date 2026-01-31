#!/usr/bin/env python
# coding: utf-8

"""
A command-line tool for managing Git repositories, supporting cloning and pulling
multiple repositories in parallel using Python's multiprocessing capabilities.
"""

import subprocess
import os
import re
import sys
import argparse
import logging

__version__ = "1.2.17"
import concurrent.futures
import datetime
from typing import List
from repository_manager.utils import (
    to_boolean,
    get_projects_file_path,
)
from repository_manager.models import GitResult, GitError, GitMetadata, ReadmeResult

DEFAULT_PROJECTS_FILE = os.getenv("PROJECTS_FILE", get_projects_file_path())
DEFAULT_REPOSITORY_MANAGER_THREADS = os.getenv("REPOSITORY_MANAGER_THREADS", 12)
DEFAULT_REPOSITORY_MANAGER_DEFAULT_BRANCH = to_boolean(
    os.getenv("REPOSITORY_MANAGER_DEFAULT_BRANCH", "False")
)
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.getenv(
    "REPOSITORY_MANAGER_WORKSPACE", os.path.normpath("/workspace")
)


# Configure logging
def setup_logging(is_mcp_server=False, log_file="repository_manager_mcp.log"):
    logger = logging.getLogger("RepositoryManager")
    logger.setLevel(logging.DEBUG)  # Logger processes all levels

    # Clear any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    if is_mcp_server:
        # Log to a file in MCP server mode, only ERROR and above
        handler = logging.FileHandler(log_file, mode="a")
        handler.setLevel(logging.ERROR)  # Only log ERROR and CRITICAL
    else:
        # Log to console (stdout) in CLI mode, all levels
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)  # Log INFO and above

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


class Git:
    """A class to handle Git operations such as cloning and pulling repositories."""

    def __init__(
        self,
        workspace: str = None,
        projects: list = None,
        threads: int = None,
        set_to_default_branch: bool = False,
        capture_output: bool = False,
        is_mcp_server: bool = False,  # Added parameter
    ):
        """Initialize the Git class with default settings."""
        self.logger = setup_logging(is_mcp_server=is_mcp_server)  # Pass is_mcp_server
        self.is_mcp_server = is_mcp_server
        if workspace:
            self.workspace = workspace
        else:
            self.workspace = f"{os.getcwd()}"
        if projects:
            self.projects = projects
        else:
            self.projects = []
        self.set_to_default_branch = set_to_default_branch
        self.threads = 12
        self.capture_output = capture_output
        self.maximum_threads = 36
        if threads:
            self.set_threads(threads=threads)
        self.deduplicate_projects()

    def deduplicate_projects(self):
        """
        Remove duplicate projects from the project list while preserving order.
        """
        if self.projects:
            self.projects = list(dict.fromkeys(self.projects))

    def git_action(
        self, command: str, workspace: str = None, project: str = None
    ) -> GitResult:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            directory (str, optional): The directory to execute the command in.
                Defaults to the repository directory.
            project (str, optional): Specify a single project

        Returns:
            GitResult: The combined stdout and stderr output of the command in structured format.
        """
        if workspace is None:
            workspace = self.workspace
        if project:
            workspace = os.path.join(workspace, project)
        pipe = subprocess.Popen(
            command,
            shell=True,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        out, err = pipe.communicate()
        return_code = pipe.wait()

        # Prepare Metadata
        metadata = GitMetadata(
            command=command,
            workspace=workspace,
            return_code=return_code,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        error_obj = None
        if return_code != 0:
            error_obj = GitError(
                message=err.strip() if err else "Unknown error", code=return_code
            )

        result = GitResult(
            status="success" if return_code == 0 else "error",
            data=out.strip() if out else "",
            error=error_obj,
            metadata=metadata,
        )

        # Logging
        if return_code != 0:
            self.logger.error(f"Command failed: {command}\nError: {err}")
        elif not self.is_mcp_server:
            self.logger.info(f"Command: {command}\nOutput: {out}")

        return result

    def clone_projects_in_parallel(self) -> List[GitResult]:
        """
        Clone all specified Git projects in parallel using multiple threads.

        Returns:
            List[GitResult]: A list of GitResult objects, one for each clone operation.
        """
        if not self.projects:
            return []

        if not self.threads:
            self.set_threads(len(self.projects))

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            future_to_project = {
                executor.submit(self.clone_project, project): project
                for project in self.projects
            }
            results = []
            for future in concurrent.futures.as_completed(future_to_project):
                project = future_to_project[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    self.logger.error(f"{project} generated an exception: {exc}")
                    # Create an error result for the exception
                    error_result = GitResult(
                        status="error",
                        data="",
                        error=GitError(message=str(exc), code=1),
                        metadata=GitMetadata(
                            command=f"clone {project}",
                            workspace=self.workspace,
                            return_code=1,
                            timestamp=datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat()
                            + "Z",
                        ),
                    )
                    results.append(error_result)

        return results

    def clone_project(self, git_project: str) -> GitResult:
        """
        Clone a single Git project.

        Args:
            git_project (str): The repository URL to clone.

        Returns:
            GitResult: The result of the Git clone command.
        """
        if not git_project:
            return GitResult(
                status="error",
                data="",
                error=GitError(message="No project URL provided", code=1),
                metadata=GitMetadata(
                    command="clone",
                    workspace=self.workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        command = f"git clone {git_project}"
        result = self.git_action(command, workspace=self.workspace)
        self.logger.info(f"Cloning {git_project}: {result}")
        return result

    def pull_projects_in_parallel(self) -> List[GitResult]:
        """
        Pull updates for all projects in the repository directory in parallel.

        Returns:
            List[GitResult]: A list of GitResult objects.
        """
        try:
            # Filter solely for directories that are git repositories
            project_dirs = [
                d
                for d in os.listdir(self.workspace)
                if os.path.isdir(os.path.join(self.workspace, d))
                and os.path.exists(os.path.join(self.workspace, d, ".git"))
            ]

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads
            ) as executor:
                results = list(executor.map(self.pull_project, project_dirs))

            return results

        except Exception as e:
            self.logger.error(f"Parallel pulling failed: {str(e)}")
            return [
                GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Parallel pulling failed: {str(e)}", code=-1
                    ),
                    metadata=GitMetadata(
                        command="pull_projects_in_parallel",
                        workspace=self.workspace,
                        return_code=-1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            ]

    def pull_project(self, git_project: str) -> GitResult:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            git_project (str): The name of the project directory to pull.

        Returns:
            GitResult: The result of the pull operation.
        """
        project_path = os.path.normpath(os.path.join(self.workspace, git_project))
        results = []

        # Execute git pull
        pull_result = self.git_action(command="git pull", workspace=project_path)
        results.append(pull_result)

        self.logger.info(
            f"Scanning: {self.workspace}/{git_project}\n"
            f"Pulling latest changes for {git_project}\n"
            f"{pull_result}"
        )

        # Handle default branch checkout if enabled
        if self.set_to_default_branch:
            # Get default branch
            default_branch_result = self.git_action(
                "git symbolic-ref refs/remotes/origin/HEAD",
                workspace=project_path,
            )
            if default_branch_result.status == "success":
                default_branch = re.sub(
                    "refs/remotes/origin/", "", default_branch_result.data
                ).strip()
                # Execute checkout
                checkout_result = self.git_action(
                    f'git checkout "{default_branch}"',
                    workspace=project_path,
                )
                results.append(checkout_result)
                self.logger.info(f"Checking out default branch: {checkout_result}")
            else:
                # If we fail to find default branch, just log and append result
                results.append(default_branch_result)
                self.logger.error(
                    f"Failed to get default branch for {git_project}: {default_branch_result.error}"
                )

        # Combine results into a single GitResult
        combined_status = (
            "success" if all(r.status == "success" for r in results) else "error"
        )

        # Combine data
        combined_data = "\n".join(
            [f"[{r.metadata.command}]: {r.data}" for r in results]
        )

        # Find first error
        combined_error = next((r.error for r in results if r.error), None)

        metadata = GitMetadata(
            command="pull_project",
            workspace=project_path,
            return_code=0 if combined_status == "success" else 1,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        return GitResult(
            status=combined_status,
            data=combined_data,
            error=combined_error,
            metadata=metadata,
        )

    def set_threads(self, threads: int) -> None:
        """
        Set the number of threads for parallel processing.

        Args:
            threads (int): The number of threads.

        Notes:
            If the input is invalid, defaults 12
        """
        try:
            if 0 < threads <= self.maximum_threads:
                self.threads = threads
            else:
                self.logger.warning(
                    f"Did not recognize {threads} as a valid value, defaulting to: {self.maximum_threads}"
                )
                self.threads = self.maximum_threads
        except Exception as e:
            self.logger.error(
                f"Did not recognize {threads} as a valid value, defaulting to: {self.maximum_threads}. Error: {e}"
            )
            self.threads = self.maximum_threads

    def pre_commit(
        self,
        run: bool = True,
        autoupdate: bool = False,
        workspace: str = None,
    ) -> GitResult:
        """
        Execute pre-commit commands in the specified workspace.

        Args:
            run (bool): Whether to run 'pre-commit run --all-files'. Default True.
            autoupdate (bool): Whether to run 'pre-commit autoupdate'. Default False.
            workspace (str, optional): Workspace to run in. Defaults to repository_manager root.
                                       Usually you want to pass a specific project path here.

        Returns:
            GitResult: Result of the execution.
        """
        if workspace is None:
            workspace = self.workspace

        workspace = os.path.abspath(workspace)
        root_workspace = os.path.abspath(self.workspace)

        # Security check: Ensure checking within main workspace
        if not workspace.startswith(root_workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Cannot run pre-commit outside of workspace: {workspace}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="pre_commit_check",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        # Check for config file
        if not os.path.exists(os.path.join(workspace, ".pre-commit-config.yaml")):
            return GitResult(
                status="skipped",
                data="No .pre-commit-config.yaml found.",
                error=None,
                metadata=GitMetadata(
                    command="pre_commit_check",
                    workspace=workspace,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        # Build command
        commands = []
        if autoupdate:
            commands.append("pre-commit autoupdate")
        if run:
            commands.append("pre-commit run --all-files")

        if not commands:
            return GitResult(
                status="skipped",
                data="No action selected (run=False, autoupdate=False).",
                error=None,
                metadata=GitMetadata(
                    command="pre_commit_check",
                    workspace=workspace,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        full_command = " && ".join(commands)

        # Execute
        result = self.git_action(command=full_command, workspace=workspace)
        return result

    def read_project_list_file(self, file: str = None):
        if file and not os.path.exists(file):
            self.logger.error(f"File not found: {file}")
            raise FileNotFoundError(f"File not found: {file}")
        with open(file, "r") as file_repositories:
            for repository in file_repositories:
                self.projects.append(repository.strip())
        self.deduplicate_projects()

    def get_readme(self, project: str = None) -> ReadmeResult:
        """
        Get the content and path of the README.md file for a project or the workspace root.

        Args:
            project (str, optional): The project directory name. If None, checks the workspace root.

        Returns:
            ReadmeResult: Object containing 'content' and 'path' of the README.md file.
        """
        target_dir = os.path.abspath(self.workspace)
        if project:
            target_dir = os.path.abspath(
                os.path.normpath(os.path.join(self.workspace, project))
            )

        # Security check
        if not target_dir.startswith(os.path.abspath(self.workspace)):
            return ReadmeResult(content="", path="")

        if not os.path.exists(target_dir):
            return ReadmeResult(content="", path="")

        # Case-insensitive search for README.md
        readme_path = None
        for filename in os.listdir(target_dir):
            if filename.lower() == "readme.md":
                readme_path = os.path.join(target_dir, filename)
                break

        if not readme_path:
            return ReadmeResult(content="", path="")

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ReadmeResult(content=content, path=readme_path)
        except Exception as e:
            self.logger.error(f"Error reading README: {e}")
            return ReadmeResult(content="", path=readme_path)

    def text_editor(
        self,
        command: str,
        path: str,
        file_text: str = None,
        view_range: list = None,
        old_str: str = None,
        new_str: str = None,
        insert_line: int = None,
    ) -> GitResult:
        """
        FileSystem Editor Tool
        """
        if os.path.isabs(path):
            path = os.path.abspath(os.path.normpath(path))
        else:
            path = os.path.abspath(os.path.normpath(os.path.join(self.workspace, path)))

        workspace_path = os.path.abspath(self.workspace)

        # Meta creation needs path, but if path is invalid usually we error.
        # But let's create meta with the resolved path.
        meta = GitMetadata(
            command=command,
            workspace=path,
            return_code=0,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        if not path.startswith(workspace_path):
            meta.return_code = 1
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Path is outside the workspace: {path}", code=1
                ),
                metadata=meta,
            )

        if command == "view":
            if not os.path.exists(path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File not found: {path}", code=1),
                    metadata=meta,
                )
            try:
                with open(path, "r") as f:
                    content = f.read()
                    if view_range:
                        lines = content.splitlines()
                        start = view_range[0] - 1
                        end = view_range[1]
                        # Check bounds
                        if start < 0:
                            start = 0
                        if end > len(lines):
                            end = len(lines)
                        content = "\n".join(lines[start:end])
                return GitResult(
                    status="success", data=content, error=None, metadata=meta
                )
            except Exception as e:
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=str(e), code=1),
                    metadata=meta,
                )

        elif command == "create":
            if os.path.exists(path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File already exists: {path}", code=1),
                    metadata=meta,
                )
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w") as f:
                    f.write(file_text if file_text else "")
                return GitResult(
                    status="success",
                    data=f"File created: {path}",
                    error=None,
                    metadata=meta,
                )
            except Exception as e:
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=str(e), code=1),
                    metadata=meta,
                )

        elif command == "str_replace":
            if not os.path.exists(path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File not found: {path}", code=1),
                    metadata=meta,
                )
            try:
                with open(path, "r") as f:
                    content = f.read()

                # Check for uniqueness if required, or just replace
                if content.count(old_str) > 1:
                    meta.return_code = 1
                    return GitResult(
                        status="error",
                        data="",
                        error=GitError(
                            message=f"Multiple occurrences of '{old_str}' found.",
                            code=1,
                        ),
                        metadata=meta,
                    )
                if old_str not in content:
                    meta.return_code = 1
                    return GitResult(
                        status="error",
                        data="",
                        error=GitError(
                            message=f"'{old_str}' not found in file.", code=1
                        ),
                        metadata=meta,
                    )

                new_content = content.replace(old_str, new_str)
                with open(path, "w") as f:
                    f.write(new_content)
                return GitResult(
                    status="success",
                    data=f"File updated: {path}",
                    error=None,
                    metadata=meta,
                )
            except Exception as e:
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=str(e), code=1),
                    metadata=meta,
                )

        elif command == "insert":
            if not os.path.exists(path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File not found: {path}", code=1),
                    metadata=meta,
                )
            try:
                with open(path, "r") as f:
                    lines = f.readlines()

                if insert_line < 0 or insert_line > len(lines) + 1:
                    meta.return_code = 1
                    return GitResult(
                        status="error",
                        data="",
                        error=GitError(
                            message=f"Invalid line number: {insert_line}", code=1
                        ),
                        metadata=meta,
                    )

                if new_str:
                    lines.insert(insert_line - 1, new_str + "\n")

                with open(path, "w") as f:
                    f.writelines(lines)
                return GitResult(
                    status="success",
                    data=f"File updated: {path}",
                    error=None,
                    metadata=meta,
                )
            except Exception as e:
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=str(e), code=1),
                    metadata=meta,
                )

        meta.return_code = 1
        return GitResult(
            status="error",
            data="",
            error=GitError(message=f"Unknown command: {command}", code=1),
            metadata=meta,
        )

    def create_project(self, project: str, workspace: str = None) -> GitResult:
        """
        Create a new project directory and initialize it as a git repository.

        Args:
            project (str): The name of the project directory to create.

        Returns:
            GitResult: Result of the operation.
        """
        if not workspace:
            workspace = self.workspace

        workspace = os.path.abspath(workspace)
        project_path = os.path.abspath(
            os.path.normpath(os.path.join(workspace, project))
        )

        # Security check
        if not project_path.startswith(workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Cannot create project outside of workspace: {project_path}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if os.path.exists(project_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory already exists: {project_path}", code=1
                ),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            os.makedirs(project_path, exist_ok=True)
            # Run git init
            init_result = self.git_action("git init", workspace=project_path)

            if init_result.status == "success":
                self.logger.info(f"Created project: {project_path}")
                return init_result
            else:
                return init_result

        except Exception as e:
            self.logger.error(f"Failed to create project {project}: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def create_directory(self, workspace: str, project: str, path: str) -> GitResult:
        """
        Create a new directory at the specified path within a project in the workspace.

        Args:
            path (str): The path where the directory should be created within the project
            project (str): The name of the project
            workspace (str): The path to the workspace

        Returns:
            GitResult: Result of the operation.
        """
        if not workspace:
            workspace = self.workspace

        # Ensure workspace is absolute
        workspace = os.path.abspath(workspace)

        # Construct target path safely
        parts = [workspace]
        if project:
            project_path = os.path.join(workspace, project)
            if not os.path.exists(project_path):
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Project directory does not exist: {project_path}",
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="create_directory",
                        workspace=workspace,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            parts.append(project)

        parts.append(path)
        target_path = os.path.normpath(os.path.join(*parts))

        # Security check: Ensure target_path is within workspace
        if not target_path.startswith(workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Cannot create directory outside of workspace: {target_path}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="create_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if os.path.exists(target_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory already exists: {target_path}", code=1
                ),
                metadata=GitMetadata(
                    command="create_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            os.makedirs(target_path, exist_ok=True)
            self.logger.info(f"Created directory: {target_path}")
            return GitResult(
                status="success",
                data=f"Created directory: {target_path}",
                error=None,
                metadata=GitMetadata(
                    command="create_directory",
                    workspace=workspace,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to create directory {target_path}: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="create_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def delete_directory(self, workspace: str, project: str, path: str) -> GitResult:
        """
        Delete a directory at the specified path.

        Args:
            workspace (str): The workspace path.
            project (str): The name of the project.
            path (str): The path of the directory to delete (relative to project or workspace).

        Returns:
            GitResult: Result of the operation.
        """
        import shutil

        if not workspace:
            workspace = self.workspace

        # Ensure workspace is absolute
        workspace = os.path.abspath(workspace)

        # Construct target path safely
        parts = [workspace]
        if project:
            project_path = os.path.join(workspace, project)
            # We don't strictly *need* to check if the project dir exists before checking the target,
            # but for consistency with create_directory we can, or just rely on target_path check.
            # create_directory checks it. Let's generally try to be consistent but lighter here implies
            # valid paths. However, if project doesn't exist, we can't delete a file inside it anyway.
            if not os.path.exists(project_path):
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Project directory does not exist: {project_path}",
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="delete_directory",
                        workspace=workspace,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            parts.append(project)

        parts.append(path)
        target_path = os.path.normpath(os.path.join(*parts))

        # Safety Check: Do not allow deletion of workspace root or anything outside it if strict
        if target_path == workspace:
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message="Cannot delete the workspace root directory.", code=1
                ),
                metadata=GitMetadata(
                    command="delete_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        # Ensure we are operating within the workspace
        if not target_path.startswith(workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message="Cannot delete directories outside of the workspace.",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="delete_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if not os.path.exists(target_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory/File not found: {target_path}", code=1
                ),
                metadata=GitMetadata(
                    command="delete_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            if os.path.isfile(target_path) or os.path.islink(target_path):
                os.remove(target_path)
            else:
                shutil.rmtree(target_path)

            self.logger.info(f"Deleted directory/file: {target_path}")
            return GitResult(
                status="success",
                data=f"Deleted: {target_path}",
                error=None,
                metadata=GitMetadata(
                    command="delete_directory",
                    workspace=workspace,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to delete {target_path}: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="delete_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def rename_directory(
        self, workspace: str, project: str, old_path: str, new_path: str
    ) -> GitResult:
        """
        Rename/Move a directory or file.

        Args:
            workspace (str): The workspace path.
            project (str): The name of the project.
            old_path (str): The current path (relative to project or workspace).
            new_path (str): The new path (relative to project or workspace).

        Returns:
            GitResult: Result of the operation.
        """
        if not workspace:
            workspace = self.workspace

        workspace = os.path.abspath(workspace)

        # Construct Base Path
        base_parts = [workspace]
        if project:
            project_path = os.path.join(workspace, project)
            if not os.path.exists(project_path):
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Project directory does not exist: {project_path}",
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="rename_directory",
                        workspace=workspace,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            base_parts.append(project)

        # Construct Full Paths
        abs_old_path = os.path.normpath(os.path.join(*base_parts, old_path))
        abs_new_path = os.path.normpath(os.path.join(*base_parts, new_path))

        # Security validation
        if not abs_old_path.startswith(workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Source path is outside the workspace: {abs_old_path}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if not abs_new_path.startswith(workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Destination path is outside the workspace: {abs_new_path}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if not os.path.exists(abs_old_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Source path not found: {abs_old_path}", code=1
                ),
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if os.path.exists(abs_new_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Destination path already exists: {abs_new_path}", code=1
                ),
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            os.renames(abs_old_path, abs_new_path)
            self.logger.info(f"Renamed {abs_old_path} to {abs_new_path}")
            return GitResult(
                status="success",
                data=f"Renamed {abs_old_path} to {abs_new_path}",
                error=None,
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=workspace,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            self.logger.error(f"Failed to rename {abs_old_path} to {abs_new_path}: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def bump_version(
        self,
        part: str,
        allow_dirty: bool = False,
        workspace: str = None,
        project: str = None,
    ) -> GitResult:
        """
        Bump the version of the project using bump2version.

        Args:
            part (str): The part of the version to bump (major, minor, patch).
            allow_dirty (bool): Whether to allow dirty working directory.
            workspace (str): The workspace path.
            project (str): The name of the project.

        Returns:
            GitResult: Result of the operation.
        """
        if not workspace:
            workspace = self.workspace

        workspace = os.path.abspath(workspace)

        # Determine target directory
        target_dir = workspace
        if project:
            target_dir = os.path.join(workspace, project)

        target_dir = os.path.abspath(os.path.normpath(target_dir))

        # Security check
        if not target_dir.startswith(workspace):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Cannot operate outside of workspace: {target_dir}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        if not os.path.exists(target_dir):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory not found: {target_dir}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        # Validate part
        valid_parts = ["major", "minor", "patch"]
        if part not in valid_parts:
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Invalid part '{part}'. Must be one of {valid_parts}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        # Build command
        command = f"bump2version {part}"
        if allow_dirty:
            command += " --allow-dirty"

        # Execute
        try:
            result = self.git_action(command=command, workspace=target_dir)

            # Log successful bump
            if result.status == "success":
                self.logger.info(f"Bumped version ({part}) in {target_dir}")
            else:
                self.logger.error(
                    f"Failed to bump version in {target_dir}: {result.error}"
                )

            return result
        except Exception as e:
            self.logger.error(f"Error in bump_version: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=workspace,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )


def usage() -> None:
    """Log the usage instructions for the command-line tool."""
    logger = setup_logging()
    logger.info(
        "Usage: \n"
        "-h | --help           [ See usage for script ]\n"
        "-b | --default-branch [ Checkout default branch ]\n"
        "-c | --clone          [ Clone projects specified  ]\n"
        "-w | --workspace      [ Workspace to clone/pull projects ]\n"
        "-f | --file           [ File with repository links   ]\n"
        "-p | --pull           [ Pull projects in parent directory ]\n"
        "-r | --repositories   [ Comma separated Git URLs ]\n"
        "-t | --threads        [ Number of parallel threads - Default 4 ]\n"
        "\n"
        "repository-manager \n\t"
        "--clone \n\t"
        "--pull \n\t"
        "--workspace '/home/user/Downloads'\n\t"
        "--file '/home/user/Downloads/repositories.txt' \n\t"
        "--repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot'\n\t"
        "--threads 8"
    )


def repository_manager() -> None:
    """
    Process command-line arguments and manage Git repository operations.
    """
    print(f"Repository Manager v{__version__}")
    parser = argparse.ArgumentParser(description="Git Repository Manager Utility")
    parser.add_argument(
        "-b",
        "--default-branch",
        action="store_true",
        help="Set repository to default branch",
        default=DEFAULT_REPOSITORY_MANAGER_DEFAULT_BRANCH,
    )
    parser.add_argument("-c", "--clone", action="store_true", help="Clone repositories")
    parser.add_argument("-p", "--pull", action="store_true", help="Pull repositories")
    parser.add_argument(
        "-w",
        "--workspace",
        type=str,
        help="Specify repository workspace",
        default=DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
    )
    parser.add_argument(
        "-f",
        "--file",
        type=str,
        help="Specify file with repository list",
        default=DEFAULT_PROJECTS_FILE,
    )
    parser.add_argument(
        "-r", "--repositories", type=str, help="Comma-separated list of repositories"
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Number of threads for parallel operations",
        default=DEFAULT_REPOSITORY_MANAGER_THREADS,
    )

    args = parser.parse_args()

    logger = setup_logging()
    git = Git()
    clone_flag = args.clone
    pull_flag = args.pull

    if args.default_branch:
        git.set_to_default_branch = True
    if args.workspace:
        if os.path.exists(args.workspace):
            git.workspace = args.workspace
        else:
            logger.error(f"Workspace not found: {args.workspace}")
            parser.print_help()
            sys.exit(2)
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            parser.print_help()
            sys.exit(2)
        git.read_project_list_file(file=args.file)
    if args.repositories:
        repositories = args.repositories.replace(" ", "").split(",")
        for repository in repositories:
            git.projects.append(repository)
    if args.threads:
        git.set_threads(threads=args.threads)

    git.deduplicate_projects()

    if clone_flag:

        git.clone_projects_in_parallel()
    if pull_flag:
        git.pull_projects_in_parallel()


if __name__ == "__main__":
    """
    Execute the main function when the script is run directly.
    """
    repository_manager()
