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

__version__ = "1.3.14"
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


def setup_logging(is_mcp_server=False, log_file="repository_manager_mcp.log"):
    logger = logging.getLogger("RepositoryManager")
    logger.setLevel(logging.DEBUG)

    logger.handlers.clear()

    if is_mcp_server:
        handler = logging.FileHandler(log_file, mode="a")
        handler.setLevel(logging.ERROR)
    else:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(logging.INFO)

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
        path: str = None,
        projects: list = None,
        threads: int = None,
        set_to_default_branch: bool = False,
        capture_output: bool = False,
        is_mcp_server: bool = False,
    ):
        """Initialize the Git class with default settings."""
        self.logger = setup_logging(is_mcp_server=is_mcp_server)
        self.is_mcp_server = is_mcp_server
        if path:
            self.path = path
        else:
            self.path = DEFAULT_REPOSITORY_MANAGER_WORKSPACE

        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass

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

    def _resolve_path(self, path: str = None) -> str:
        """
        Resolve the path to an absolute path.
        If path is None, returns self.path.
        If path is absolute, returns it.
        If path is relative, joins it with self.path.
        """
        if path is None:
            return os.path.abspath(self.path)

        if os.path.isabs(path):
            return os.path.abspath(path)

        return os.path.abspath(os.path.join(self.path, path))

    def git_action(self, command: str, path: str = None) -> GitResult:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            path (str, optional): The directory to execute the command in.
                Defaults to the base path.

        Returns:
            GitResult: The combined stdout and stderr output of the command in structured format.
        """
        target_path = self._resolve_path(path)

        pipe = subprocess.Popen(
            command,
            shell=True,
            cwd=target_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        out, err = pipe.communicate()
        return_code = pipe.wait()

        metadata = GitMetadata(
            command=command,
            workspace=target_path,
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

        if return_code != 0:
            self.logger.error(f"Command failed: {command}\nError: {err}")
        elif not self.is_mcp_server:
            self.logger.info(f"Command: {command}\nOutput: {out}")

        return result

    def clone_projects(self) -> List[GitResult]:
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
                    error_result = GitResult(
                        status="error",
                        data="",
                        error=GitError(message=str(exc), code=1),
                        metadata=GitMetadata(
                            command=f"clone {project}",
                            workspace=self.path,
                            return_code=1,
                            timestamp=datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat()
                            + "Z",
                        ),
                    )
                    results.append(error_result)

        return results

    def clone_project(self, url: str) -> GitResult:
        """
        Clone a single Git project.

        Args:
            url (str): The repository URL to clone.

        Returns:
            GitResult: The result of the Git clone command.
        """
        if not url:
            return GitResult(
                status="error",
                data="",
                error=GitError(message="No project URL provided", code=1),
                metadata=GitMetadata(
                    command="clone",
                    workspace=self.path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        command = f"git clone {url}"
        result = self.git_action(command, path=self.path)
        self.logger.info(f"Cloning {url}: {result}")
        return result

    def pull_projects(self) -> List[GitResult]:
        """
        Pull updates for all projects in the repository directory in parallel.

        Returns:
            List[GitResult]: A list of GitResult objects.
        """
        try:
            expanded_path = os.path.expanduser(self.path)
            if not os.path.exists(expanded_path):
                return []

            project_dirs = [
                os.path.join(expanded_path, d)
                for d in os.listdir(expanded_path)
                if os.path.isdir(os.path.join(expanded_path, d))
                and os.path.exists(os.path.join(expanded_path, d, ".git"))
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
                        command="pull_projects",
                        workspace=self.path,
                        return_code=-1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            ]

    def pull_project(self, path: str = None) -> GitResult:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            path (str): The path to the project to pull. Defaults to self.path.

        Returns:
            GitResult: The result of the pull operation.
        """
        target_path = self._resolve_path(path)
        results = []

        pull_result = self.git_action(command="git pull", path=target_path)
        results.append(pull_result)

        self.logger.info(
            f"Scanning: {target_path}\n"
            f"Pulling latest changes for {target_path}\n"
            f"{pull_result}"
        )

        if self.set_to_default_branch:
            default_branch_result = self.git_action(
                "git symbolic-ref refs/remotes/origin/HEAD",
                path=target_path,
            )
            if default_branch_result.status == "success":
                default_branch = re.sub(
                    "refs/remotes/origin/", "", default_branch_result.data
                ).strip()
                checkout_result = self.git_action(
                    f'git checkout "{default_branch}"',
                    path=target_path,
                )
                results.append(checkout_result)
                self.logger.info(f"Checking out default branch: {checkout_result}")
            else:
                results.append(default_branch_result)
                self.logger.error(
                    f"Failed to get default branch for {target_path}: {default_branch_result.error}"
                )

        combined_status = (
            "success" if all(r.status == "success" for r in results) else "error"
        )

        combined_data = "\n".join(
            [f"[{r.metadata.command}]: {r.data}" for r in results]
        )

        combined_error = next((r.error for r in results if r.error), None)

        metadata = GitMetadata(
            command="pull_project",
            workspace=target_path,
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
        path: str = None,
    ) -> GitResult:
        """
        Execute pre-commit commands in the specified path.

        Args:
            run (bool): Whether to run 'pre-commit run --all-files'. Default True.
            autoupdate (bool): Whether to run 'pre-commit autoupdate'. Default False.
            path (str, optional): Path to run in. Defaults to self.path.
        """
        target_path = self._resolve_path(path)

        if not os.path.exists(os.path.join(target_path, ".pre-commit-config.yaml")):
            return GitResult(
                status="skipped",
                data="No .pre-commit-config.yaml found.",
                error=None,
                metadata=GitMetadata(
                    command="pre_commit_check",
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

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
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        full_command = " && ".join(commands)

        result = self.git_action(command=full_command, path=target_path)
        return result

    def read_project_list_file(self, file: str = None):
        if file and not os.path.exists(file):
            self.logger.error(f"File not found: {file}")
            raise FileNotFoundError(f"File not found: {file}")
        with open(file, "r") as file_repositories:
            for repository in file_repositories:
                self.projects.append(repository.strip())
        self.deduplicate_projects()

    def get_readme(self, path: str = None) -> ReadmeResult:
        """
        Get the content and path of the README.md file in the specified path.

        Args:
            path (str, optional): The directory path. Defaults to self.path.

        Returns:
            ReadmeResult: Object containing 'content' and 'path' of the README.md file.
        """
        target_dir = self._resolve_path(path)

        if not os.path.exists(target_dir):
            return ReadmeResult(content="", path="")

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
        target_path = self._resolve_path(path)

        meta = GitMetadata(
            command=command,
            workspace=target_path,
            return_code=0,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        if command == "view":
            if not os.path.exists(target_path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File not found: {target_path}", code=1),
                    metadata=meta,
                )
            try:
                with open(target_path, "r") as f:
                    content = f.read()
                    if view_range:
                        lines = content.splitlines()
                        start = view_range[0] - 1
                        end = view_range[1]
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
            if os.path.exists(target_path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"File already exists: {target_path}", code=1
                    ),
                    metadata=meta,
                )
            try:
                os.makedirs(os.path.dirname(target_path), exist_ok=True)
                with open(target_path, "w") as f:
                    f.write(file_text if file_text else "")
                return GitResult(
                    status="success",
                    data=f"File created: {target_path}",
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
            if not os.path.exists(target_path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File not found: {target_path}", code=1),
                    metadata=meta,
                )
            try:
                with open(target_path, "r") as f:
                    content = f.read()

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
                with open(target_path, "w") as f:
                    f.write(new_content)
                return GitResult(
                    status="success",
                    data=f"File updated: {target_path}",
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
            if not os.path.exists(target_path):
                meta.return_code = 1
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message=f"File not found: {target_path}", code=1),
                    metadata=meta,
                )
            try:
                with open(target_path, "r") as f:
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

                with open(target_path, "w") as f:
                    f.writelines(lines)
                return GitResult(
                    status="success",
                    data=f"File updated: {target_path}",
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

    def create_project(self, path: str) -> GitResult:
        """
        Create a new project directory and initialize it as a git repository.

        Args:
            path (str): The path of the project directory to create.

        Returns:
            GitResult: Result of the operation.
        """
        target_path = self._resolve_path(path)

        if os.path.exists(target_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory already exists: {target_path}", code=1
                ),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            os.makedirs(target_path, exist_ok=True)
            init_result = self.git_action("git init", path=target_path)

            if init_result.status == "success":
                self.logger.info(f"Created project: {target_path}")
                return init_result
            else:
                return init_result

        except Exception as e:
            self.logger.error(f"Failed to create project {target_path}: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def create_directory(self, path: str) -> GitResult:
        """
        Create a new directory at the specified path.

        Args:
            path (str): The path where the directory should be created.

        Returns:
            GitResult: Result of the operation.
        """
        target_path = self._resolve_path(path)

        if os.path.exists(target_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory already exists: {target_path}", code=1
                ),
                metadata=GitMetadata(
                    command="create_directory",
                    workspace=target_path,
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
                    workspace=target_path,
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
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def delete_directory(self, path: str) -> GitResult:
        """
        Delete a directory or file at the specified path.

        Args:
            path (str): The path of the directory or file to delete.

        Returns:
            GitResult: Result of the operation.
        """
        import shutil

        target_path = self._resolve_path(path)

        if target_path == self.path:
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message="Cannot delete the workspace root directory.", code=1
                ),
                metadata=GitMetadata(
                    command="delete_directory",
                    workspace=target_path,
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
                    workspace=target_path,
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
                    workspace=target_path,
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
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def rename_directory(self, old_path: str, new_path: str) -> GitResult:
        """
        Rename/Move a directory or file.

        Args:
            old_path (str): The current path.
            new_path (str): The new path.

        Returns:
            GitResult: Result of the operation.
        """
        abs_old_path = self._resolve_path(old_path)
        abs_new_path = self._resolve_path(new_path)

        if not os.path.exists(abs_old_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Source path not found: {abs_old_path}", code=1
                ),
                metadata=GitMetadata(
                    command="rename_directory",
                    workspace=abs_old_path,
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
                    workspace=abs_new_path,
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
                    workspace=abs_old_path,
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
                    workspace=abs_old_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def bump_version(
        self,
        part: str,
        allow_dirty: bool = False,
        path: str = None,
    ) -> GitResult:
        """
        Bump the version of the project using bump2version.

        Args:
            part (str): The part of the version to bump (major, minor, patch).
            allow_dirty (bool): Whether to allow dirty working directory.
            path (str): The path to the project directory.

        Returns:
            GitResult: Result of the operation.
        """
        target_dir = self._resolve_path(path)

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
                    workspace=target_dir,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

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
                    workspace=target_dir,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        command = f"bump2version {part}"
        if allow_dirty:
            command += " --allow-dirty"

        try:
            result = self.git_action(command=command, path=target_dir)

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
                    workspace=target_dir,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def search_codebase(
        self,
        query: str,
        path: str = None,
        glob_pattern: str = None,
        case_sensitive: bool = False,
    ) -> GitResult:
        """
        Search the codebase using ripgrep.

        Args:
            query (str): The regex pattern to search for.
            path (str, optional): The path to search in (absolute or relative to CWD).
                                  Defaults to self.workspace or CWD.
            glob_pattern (str, optional): Glob pattern to filter files (e.g., '*.py').
            case_sensitive (bool): Whether the search should be case sensitive.

        Returns:
            GitResult: The result of the search operation.
        """
        if not path:
            path = self.workspace if self.workspace else os.getcwd()

        path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message=f"Path not found: {path}", code=1),
                metadata=GitMetadata(
                    command="search_codebase",
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        # Try ripgrep first
        rg_cmd = ["rg", "--json", "-n", "--column"]
        if not case_sensitive:
            rg_cmd.append("-i")
        if glob_pattern:
            rg_cmd.append("-g")
            rg_cmd.append(glob_pattern)

        rg_cmd.append(query)
        rg_cmd.append(path)

        command_str = " ".join(rg_cmd)

        try:
            result = subprocess.run(rg_cmd, capture_output=True, text=True)
            # If rg is found but returns error code (e.g. no matches is usually 1, but internal error is >1)
            # rg exit code 1 means no matches found, which is not an "error" for us.
            if result.returncode > 1:
                # Check if it was because rg is not installed
                pass

            return GitResult(
                status="success",
                data=result.stdout,
                error=None,
                metadata=GitMetadata(
                    command=command_str,
                    workspace=path,
                    return_code=result.returncode,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except FileNotFoundError:
            # Fallback to grep
            self.logger.warning("ripgrep not found, falling back to grep")
            grep_cmd = ["grep", "-rnI"]
            if not case_sensitive:
                grep_cmd.append("-i")
            if glob_pattern:
                grep_cmd.append(f"--include={glob_pattern}")

            grep_cmd.append(query)
            grep_cmd.append(path)

            command_str = " ".join(grep_cmd)

            try:
                result = subprocess.run(grep_cmd, capture_output=True, text=True)
                return GitResult(
                    status="success",
                    data=result.stdout,
                    error=None,
                    metadata=GitMetadata(
                        command=command_str,
                        workspace=path,
                        return_code=result.returncode,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            except Exception as e:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Both ripgrep and grep failed: {str(e)}", code=1
                    ),
                    metadata=GitMetadata(
                        command=command_str,
                        workspace=path,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
        except Exception as e:
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command=command_str,
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def find_files(self, name_pattern: str, path: str = None) -> GitResult:
        """
        Find files using find.

        Args:
            name_pattern (str): The name pattern to search for (e.g., '*.py').
            path (str, optional): The path to search in (absolute or relative to CWD).
                                  Defaults to self.workspace or CWD.

        Returns:
            GitResult: The result of the find operation.
        """
        if not path:
            path = self.workspace if self.workspace else os.getcwd()

        path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message=f"Path not found: {path}", code=1),
                metadata=GitMetadata(
                    command="find_files",
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        cmd = ["find", path, "-name", name_pattern]
        command_str = " ".join(cmd)

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            return GitResult(
                status="success",
                data=result.stdout,
                error=None,
                metadata=GitMetadata(
                    command=command_str,
                    workspace=path,
                    return_code=result.returncode,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command=command_str,
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def read_file(
        self, path: str, start_line: int = None, end_line: int = None
    ) -> GitResult:
        """
        Read a file with optional line range.

        Args:
            path (str): The path to the file to read (absolute or relative to CWD).
            start_line (int, optional): The starting line number (1-indexed).
            end_line (int, optional): The ending line number (1-indexed).

        Returns:
            GitResult: The content of the file.
        """
        path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message=f"File not found: {path}", code=1),
                metadata=GitMetadata(
                    command="read_file",
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            with open(path, "r") as f:
                lines = f.readlines()

            total_lines = len(lines)

            if start_line is None:
                start_line = 1
            if end_line is None:
                end_line = total_lines

            if start_line < 1:
                start_line = 1
            if end_line > total_lines:
                end_line = total_lines

            selected_lines = lines[start_line - 1 : end_line]
            content = "".join(selected_lines)

            return GitResult(
                status="success",
                data=content,
                error=None,
                metadata=GitMetadata(
                    command=f"read_file {path} {start_line}-{end_line}",
                    workspace=path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="read_file",
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def replace_in_file(
        self, path: str, target_content: str, replacement_content: str
    ) -> GitResult:
        """
        Replace a block of text in a file.

        Args:
            path (str): The path to the file to modify (absolute or relative to CWD).
            target_content (str): The exact content to be replaced.
            replacement_content (str): The new content to replace with.

        Returns:
            GitResult: Result of the operation.
        """
        path = os.path.abspath(os.path.expanduser(path))

        if not os.path.exists(path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message=f"File not found: {path}", code=1),
                metadata=GitMetadata(
                    command="replace_in_file",
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            with open(path, "r") as f:
                content = f.read()

            if content.count(target_content) == 0:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="Target content not found in file.", code=1),
                    metadata=GitMetadata(
                        command="replace_in_file",
                        workspace=path,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )

            if content.count(target_content) > 1:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message="Multiple occurrences of target content found. Please be more specific.",
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="replace_in_file",
                        workspace=path,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )

            new_content = content.replace(target_content, replacement_content)

            with open(path, "w") as f:
                f.write(new_content)

            return GitResult(
                status="success",
                data=f"Successfully updated {path}",
                error=None,
                metadata=GitMetadata(
                    command="replace_in_file",
                    workspace=path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        except Exception as e:
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="replace_in_file",
                    workspace=path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )


def repository_manager() -> None:
    """
    Process command-line arguments and manage Git repository operations.
    """
    print(f"Repository Manager v{__version__}")
    parser = argparse.ArgumentParser(
        add_help=False, description="Git Repository Manager Utility"
    )
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

    parser.add_argument("--help", action="store_true", help="Show usage")

    args = parser.parse_args()

    if hasattr(args, "help") and args.help:

        parser.print_help()
        sys.exit(0)

    logger = setup_logging()
    git = Git()
    clone_flag = args.clone
    pull_flag = args.pull

    if args.default_branch:
        git.set_to_default_branch = True
    if args.workspace:
        if os.path.exists(args.workspace):
            git.path = args.workspace
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
        git.clone_projects()
    if pull_flag:
        git.pull_projects()


if __name__ == "__main__":
    """
    Execute the main function when the script is run directly.
    """
    repository_manager()
