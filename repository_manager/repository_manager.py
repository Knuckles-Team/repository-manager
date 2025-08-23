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
from multiprocessing import Pool


# Configure logging
def setup_logging(is_mcp_server=False, log_file="repository_manager.log"):
    logger = logging.getLogger("RepositoryManager")
    logger.setLevel(logging.DEBUG)

    # Clear any existing handlers to avoid duplicate logs
    logger.handlers.clear()

    if is_mcp_server:
        # Log to a file when running as MCP server
        handler = logging.FileHandler(log_file)
    else:
        # Log to console (stdout) when running standalone
        handler = logging.StreamHandler(sys.stdout)

    handler.setLevel(logging.DEBUG)
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
    ):
        """Initialize the Git class with default settings."""
        self.logger = setup_logging()
        if repository_directory:
            self.repository_directory = repository_directory
        else:
            self.repository_directory = f"{os.getcwd()}"
        if projects:
            self.projects = projects
        else:
            self.projects = []
        self.set_to_default_branch = set_to_default_branch
        self.threads = 1
        self.capture_output = capture_output
        if threads:
            self.set_threads(threads=threads)

    def git_action(self, command: str, directory: str = None) -> str:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            directory (str, optional): The directory to execute the command in.
                Defaults to the repository directory.
            capture_output (bool, optional): If True, capture stdout and stderr without printing.
                If False, print output to stdout in real-time. Defaults to True.

        Returns:
            str: The combined stdout and stderr output of the command.
        """
        if directory is None:
            directory = self.repository_directory
        try:
            # Split command into arguments, assuming simple commands for now
            args = ["git"] + command.split()
            process = subprocess.Popen(
                args,
                cwd=directory,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                text=True,
            )
            output = []
            if self.capture_output:
                # Collect output without printing
                out, error = process.communicate(timeout=120)
                result = f"{out}{error}"
                return result
            else:
                # Stream output to stdout in real-time and collect it
                while True:
                    stdout_line = process.stdout.readline()
                    stderr_line = process.stderr.readline()
                    if (
                        not stdout_line
                        and not stderr_line
                        and process.poll() is not None
                    ):
                        break
                    if stdout_line:
                        print(stdout_line, end="", flush=True)
                        output.append(stdout_line)
                    if stderr_line:
                        print(stderr_line, end="", flush=True, file=sys.stderr)
                        output.append(stderr_line)
                process.wait(timeout=120)
                result = "".join(output)
                return result
        except subprocess.TimeoutExpired:
            self.logger.error(f"Command '{command}' timed out after 120 seconds")
            process.kill()
            return "Error: Command timed out"
        except OSError as e:
            self.logger.error(f"Command '{command}' failed: {e}")
            return f"Error: {e}"

    def set_threads(self, threads: int) -> None:
        """
        Set the number of threads for parallel processing.

        Args:
            threads (int): The number of threads.

        Notes:
            If the input is invalid, defaults to the number of CPU cores and logs a warning.
        """
        try:
            if threads > 0 and threads <= os.cpu_count():
                self.threads = threads
            else:
                self.logger.warning(
                    f"Did not recognize {threads} as a valid value, defaulting to CPU Count: {os.cpu_count()}"
                )
                self.threads = os.cpu_count()
        except Exception as e:
            self.logger.error(
                f"Did not recognize {threads} as a valid value, defaulting to CPU Count: {os.cpu_count()}. Error: {e}"
            )
            self.threads = os.cpu_count()

    def read_project_list_file(self, file: str = None):
        if file and not os.path.exists(file):
            logger.error(f"File not found: {file}")
        with open(file, "r") as file_repositories:
            for repository in file_repositories:
                self.projects.append(repository.strip())

    def clone_projects_in_parallel(self) -> str:
        """
        Clone all specified Git projects in parallel using multiple threads.

        Returns:
            str: Combined output of all clone operations, with each project's result
                 prefixed by its repository URL.
        """
        pool = Pool(processes=self.threads)
        try:
            # Collect results from parallel clone operations
            results = pool.map(self.clone_project, self.projects)
            # Close the pool and wait for all processes to complete
            pool.close()
            pool.join()
            # Combine results with repository URLs for clarity
            combined_results = []
            for project, result in zip(self.projects, results):
                combined_results.append(f"Repository: {project}\n{result}\n")
            return "".join(combined_results)
        except Exception as e:
            self.logger.error(f"Parallel cloning failed: {e}")
            return f"Error: Parallel cloning failed: {e}"
        finally:
            pool.terminate()  # Ensure pool is cleaned up

    def clone_project(self, git_project: str) -> str:
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

    def pull_projects_in_parallel(self) -> str:
        """
        Pull updates for all projects in the repository directory in parallel.

        Returns:
            str: Combined output of all pull operations, with each project's result
                 prefixed by its directory name.
        """
        pool = Pool(processes=self.threads)
        try:
            # Collect results from parallel pull operations
            project_dirs = os.listdir(self.repository_directory)
            results = pool.map(self.pull_project, project_dirs)
            # Close the pool and wait for all processes to complete
            pool.close()
            pool.join()
            # Combine results with project directory names for clarity
            combined_results = []
            for project, result in zip(project_dirs, results):
                combined_results.append(f"Project: {project}\n{result}\n")
            return "".join(combined_results)
        except Exception as e:
            self.logger.error(f"Parallel pulling failed: {e}")
            return f"Error: Parallel pulling failed: {e}"
        finally:
            pool.terminate()  # Ensure pool is cleaned up

    def pull_project(self, git_project: str) -> str:
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
            default_branch = re.sub("refs/remotes/origin/", "", default_branch).strip()
            default_branch_result = self.git_action(
                f'git checkout "{default_branch}"',
                directory=f"{self.repository_directory}/{git_project}",
            )
            self.logger.info(f"Checking out default branch: {default_branch_result}")
            result = f"{result}\n{default_branch_result}"
        return result


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
            git.set_default_branch = True
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

    projects = list(dict.fromkeys(git.projects))
    git.projects = projects

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
