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
import getopt
import logging
import concurrent.futures
import datetime
from typing import List


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
        repository_directory: str = None,
        projects: list = None,
        threads: int = None,
        set_to_default_branch: bool = False,
        capture_output: bool = False,
        is_mcp_server: bool = False,  # Added parameter
    ):
        """Initialize the Git class with default settings."""
        self.logger = setup_logging(is_mcp_server=is_mcp_server)  # Pass is_mcp_server
        self.is_mcp_server = is_mcp_server
        if repository_directory:
            self.repository_directory = repository_directory
        else:
            self.repository_directory = f"{os.getcwd()}"
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

    def git_action(self, command: str, directory: str = None) -> dict:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            directory (str, optional): The directory to execute the command in.
                Defaults to the repository directory.

        Returns:
            dict: The combined stdout and stderr output of the command in structured format.
        """
        if directory is None:
            directory = self.repository_directory
        pipe = subprocess.Popen(
            command,
            shell=True,
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
        )
        out, err = pipe.communicate()
        return_code = pipe.wait()

        # Prepare the result dictionary
        result = {
            "status": "success" if return_code == 0 else "error",
            "data": out.strip() if out else "",
            "error": None if return_code == 0 else {
                "message": err.strip() if err else "Unknown error",
                "code": return_code
            },
            "metadata": {
                "command": command,
                "directory": directory,
                "return_code": return_code,
                "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z"
            }
        }
        # Logging
        if return_code != 0:
            self.logger.error(f"Command failed: {command}\nError: {err}")
        elif not self.is_mcp_server:
            self.logger.info(f"Command: {command}\nOutput: {out}")

        return result

    def clone_projects_in_parallel(self) -> List[dict]:
        """
        Clone all specified Git projects in parallel using multiple threads.

        Returns:
            List[dict]: A list of dictionaries, each containing the result of a clone operation
                        with repository URL and git_action result.
        """
        try:
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.threads
            ) as executor:
                results = list(executor.map(self.clone_project, self.projects))
            # Combine results with repository URLs for clarity
            combined_results = []
            for project, result in zip(self.projects, results):
                combined_result = {
                    "status": result["status"],
                    "data": result["data"],
                    "error": result["error"],
                    "metadata": {
                        **result["metadata"],
                        "repository_url": project
                    }
                }
                combined_results.append(combined_result)
                self.logger.info(f"Cloning {project}: {result['data'] if result['status'] == 'success' else result['error']}")
            return combined_results
        except Exception as e:
            self.logger.error(f"Parallel cloning failed: {str(e)}")
            return [{
                "status": "error",
                "data": "",
                "error": {
                    "message": f"Parallel cloning failed: {str(e)}",
                    "code": -1
                },
                "metadata": {
                    "command": "clone_projects_in_parallel",
                    "directory": self.repository_directory,
                    "return_code": None,
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    "repository_url": None
                }
            }]

    def clone_project(self, git_project: str) -> dict:
        """
        Clone a single Git project.

        Args:
            git_project (str): The repository URL to clone.

        Returns:
            str: The output of the Git clone command.
        """
        result = self.git_action(f"git clone {git_project}")
        self.logger.info(f"Cloning {git_project}: {result}")
        return result

    def pull_projects_in_parallel(self) -> List[dict]:
        """
        Pull updates for all projects in the repository directory in parallel.

        Returns:
            List[dict]: A list of dictionaries, each containing the result of a pull operation
                        with project directory name and git_action result.
        """
        try:
            project_dirs = os.listdir(self.repository_directory)
            with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.threads
            ) as executor:
                results = list(executor.map(self.pull_project, project_dirs))
            # Combine results with project directory names for clarity
            combined_results = []
            for project, result in zip(project_dirs, results):
                combined_result = {
                    "status": result["status"],
                    "data": result["data"],
                    "error": result["error"],
                    "metadata": {
                        **result["metadata"],
                        "project_directory": project
                    }
                }
                combined_results.append(combined_result)
                self.logger.info(
                    f"Pulling {project}: {result['data'] if result['status'] == 'success' else result['error']}"
                )
            return combined_results
        except Exception as e:
            self.logger.error(f"Parallel pulling failed: {str(e)}")
            return [{
                "status": "error",
                "data": "",
                "error": {
                    "message": f"Parallel pulling failed: {str(e)}",
                    "code": -1
                },
                "metadata": {
                    "command": "pull_projects_in_parallel",
                    "directory": self.repository_directory,
                    "return_code": None,
                    "timestamp": datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    "project_directory": None
                }
            }]

    def pull_project(self, git_project: str) -> dict:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            git_project (str): The name of the project directory to pull.
        """
        result = self.git_action(
            command="git pull",
            directory=os.path.normpath(
                os.path.join(self.repository_directory, git_project)
            ),
        )
        self.logger.info(
            f"Scanning: {self.repository_directory}/{git_project}\n"
            f"Pulling latest changes for {git_project}\n"
            f"{result}"
        )
        if self.set_to_default_branch:
            default_branch = self.git_action(
                "git symbolic-ref refs/remotes/origin/HEAD",
                directory=f"{self.repository_directory}/{git_project}",
            )
            default_branch = re.sub("refs/remotes/origin/", "", default_branch['data']).strip()
            default_branch_result = self.git_action(
                f'git checkout "{default_branch}"',
                directory=f"{self.repository_directory}/{git_project}",
            )
            self.logger.info(f"Checking out default branch: {default_branch_result}")
            result = f"{result}\n{default_branch_result}"
        return result

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

    def read_project_list_file(self, file: str = None):
        if file and not os.path.exists(file):
            self.logger.error(f"File not found: {file}")
            raise FileNotFoundError(f"File not found: {file}")
        with open(file, "r") as file_repositories:
            for repository in file_repositories:
                self.projects.append(repository.strip())


def usage() -> None:
    """Log the usage instructions for the command-line tool."""
    logger = setup_logging()
    logger.info(
        "Usage: \n"
        "-h | --help           [ See usage for script ]\n"
        "-b | --default-branch [ Checkout default branch ]\n"
        "-c | --clone          [ Clone projects specified  ]\n"
        "-d | --directory      [ Directory to clone/pull projects ]\n"
        "-f | --file           [ File with repository links   ]\n"
        "-p | --pull           [ Pull projects in parent directory ]\n"
        "-r | --repositories   [ Comma separated Git URLs ]\n"
        "-t | --threads        [ Number of parallel threads - Default 4 ]\n"
        "\n"
        "repository-manager \n\t"
        "--clone \n\t"
        "--pull \n\t"
        "--directory '/home/user/Downloads'\n\t"
        "--file '/home/user/Downloads/repositories.txt' \n\t"
        "--repositories 'https://github.com/Knucklessg1/media-downloader,https://github.com/Knucklessg1/genius-bot'\n\t"
        "--threads 8"
    )


def repository_manager(argv: list) -> None:
    """
    Process command-line arguments and manage Git repository operations.

    Args:
        argv (list): List of command-line arguments.

    Exits:
        If invalid arguments or paths are provided, or if usage is requested.
    """
    logger = setup_logging()
    git = Git()
    clone_flag = False
    pull_flag = False
    try:
        opts, args = getopt.getopt(
            argv,
            "hbcpd:f:r:t:",
            [
                "help",
                "default-branch",
                "clone",
                "pull",
                "directory=",
                "file=",
                "repositories=",
                "threads=",
            ],
        )
    except getopt.GetoptError:
        usage()
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            sys.exit()
        elif opt in ("-b", "--default-branch"):
            git.set_to_default_branch = True
        elif opt in ("-c", "--clone"):
            clone_flag = True
        elif opt in ("-p", "--pull"):
            pull_flag = True
        elif opt in ("-d", "--directory"):
            if os.path.exists(arg):
                git.repository_directory = arg
            else:
                logger.error(f"Directory not found: {arg}")
                usage()
                sys.exit(2)
        elif opt in ("-f", "--file"):
            # Verify file with repositories exists
            if arg and not os.path.exists(arg):
                logger.error(f"File not found: {arg}")
                usage()
                sys.exit(2)
            git.read_project_list_file(file=arg)
        elif opt in ("-r", "--repositories"):
            repositories = arg.replace(" ", "")
            repositories = repositories.split(",")
            for repository in repositories:
                git.projects.append(repository)
        elif opt in ("-t", "--threads"):
            git.set_threads(threads=int(arg))

    git.projects = list(dict.fromkeys(git.projects))

    if clone_flag:
        git.clone_projects_in_parallel()
    if pull_flag:
        git.pull_projects_in_parallel()


def main():
    """
    Entry point for the command-line tool.

    Exits:
        If insufficient arguments are provided, displays usage and exits.
    """
    logger = setup_logging()
    if len(sys.argv) < 2:
        logger.error("Insufficient arguments provided")
        usage()
        sys.exit(2)
    repository_manager(sys.argv[1:])


if __name__ == "__main__":
    """
    Execute the main function when the script is run directly.
    """
    logger = setup_logging()
    if len(sys.argv) < 2:
        logger.error("Insufficient arguments provided")
        usage()
        sys.exit(2)
    repository_manager(sys.argv[1:])
