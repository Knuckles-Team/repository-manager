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
from multiprocessing import Pool


class Git:
    """A class to handle Git operations such as cloning and pulling repositories."""

    def __init__(self):
        """Initialize the Git class with default settings."""
        self.repository_directory = f"{os.getcwd()}"
        self.git_projects = []
        self.set_to_default_branch = False
        self.threads = os.cpu_count()

    def git_action(self, command: str, directory: str = None) -> str:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            directory (str, optional): The directory to execute the command in.
                Defaults to the repository directory.

        Returns:
            str: The combined stdout and stderr output of the command.
        """
        if directory is None:
            directory = self.repository_directory
        pipe = subprocess.Popen(
            command,
            shell=True,
            cwd=directory,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        (out, error) = pipe.communicate()
        result = f"{str(out, 'utf-8')}{str(error, 'utf-8')}"
        pipe.wait()
        return result

    def set_repository_directory(self, repository_directory: str) -> None:
        """
        Set the repository directory for Git operations.

        Args:
            repository_directory (str): The path to the repository directory.

        Notes:
            If the specified directory does not exist, a message is printed.
        """
        if os.path.exists(repository_directory):
            self.repository_directory = repository_directory.replace(os.sep, "/")
        else:
            print(
                f'Path specified does not exist: {repository_directory.replace(os.sep, "/")}'
            )

    def set_git_projects(self, git_projects: list) -> None:
        """
        Set the list of Git projects (repository URLs).

        Args:
            git_projects (list): A list of repository URLs.
        """
        self.git_projects = git_projects

    def set_default_branch(self, set_to_default_branch: bool) -> None:
        """
        Set the flag to checkout the default branch after pulling.

        Args:
            set_to_default_branch (bool): Whether to checkout the default branch.
        """
        self.set_to_default_branch = set_to_default_branch

    def set_threads(self, threads: int) -> None:
        """
        Set the number of threads for parallel processing.

        Args:
            threads (int): The number of threads.

        Notes:
            If the input is invalid, defaults to the number of CPU cores.
        """
        try:
            if threads > 0 or threads < os.cpu_count():
                self.threads = threads
            else:
                print(
                    f"Did not recognize {threads} as a valid value, defaulting to CPU Count: {os.cpu_count()}"
                )
                self.threads = os.cpu_count()
        except Exception as e:
            print(
                f"Did not recognize {threads} as a valid value, defaulting to CPU Count: {os.cpu_count()}\nError: {e}"
            )
            self.threads = os.cpu_count()

    def append_git_project(self, git_project: str) -> None:
        """
        Append a single Git project URL to the list of projects.

        Args:
            git_project (str): The repository URL to append.
        """
        self.git_projects.append(git_project)

    def clone_projects_in_parallel(self) -> None:
        """Clone all specified Git projects in parallel using multiple threads."""
        pool = Pool(processes=self.threads)
        pool.map(self.clone_project, self.git_projects)

    def clone_project(self, git_project: str) -> None:
        """
        Clone a single Git project.

        Args:
            git_project (str): The repository URL to clone.
        """
        print(self.git_action(f"git clone {git_project}"))

    def pull_projects_in_parallel(self) -> None:
        """Pull updates for all projects in the repository directory in parallel."""
        pool = Pool(processes=self.threads)
        pool.map(self.pull_project, os.listdir(self.repository_directory))

    def pull_project(self, git_project: str) -> None:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            git_project (str): The name of the project directory to pull.
        """
        print(
            f"Scanning: {self.repository_directory}/{git_project}\n"
            f"Pulling latest changes for {git_project}\n"
            f'{self.git_action(command="git pull", directory=os.path.normpath(os.path.join(self.repository_directory, git_project)))}'
        )
        if self.set_to_default_branch:
            default_branch = self.git_action(
                "git symbolic-ref refs/remotes/origin/HEAD",
                directory=f"{self.repository_directory}/{git_project}",
            )
            default_branch = re.sub("refs/remotes/origin/", "", default_branch).strip()
            print(
                "Checking out default branch ",
                self.git_action(
                    f'git checkout "{default_branch}"',
                    directory=f"{self.repository_directory}/{git_project}",
                ),
            )


def usage() -> None:
    """Print the usage instructions for the command-line tool."""
    print(
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
    gitlab = Git()
    projects = []
    default_branch_flag = False
    clone_flag = False
    pull_flag = False
    directory = os.curdir
    file = None
    repositories = None
    threads = os.cpu_count()
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
        elif opt in ("-b", "--b"):
            default_branch_flag = True
        elif opt in ("-c", "--clone"):
            clone_flag = True
        elif opt in ("-p", "--pull"):
            pull_flag = True
        elif opt in ("-d", "--directory"):
            directory = arg
        elif opt in ("-f", "--file"):
            file = arg
        elif opt in ("-r", "--repositories"):
            repositories = arg.replace(" ", "")
            repositories = repositories.split(",")
        elif opt in ("-t", "--threads"):
            threads = arg

    # Verify directory to clone/pull exists
    if os.path.exists(directory):
        gitlab.set_repository_directory(directory)
    else:
        print(f"Directory not found: {directory}")
        usage()
        sys.exit(2)

    # Verify file with repositories exists
    if os.path.exists(file):
        file_repositories = open(file, "r")
        for repository in file_repositories:
            projects.append(repository)
    else:
        print(f"File not found: {file}")
        usage()
        sys.exit(2)

    if repositories:
        for repository in repositories:
            projects.append(repository)

    gitlab.set_threads(threads=int(threads))

    projects = list(dict.fromkeys(projects))

    gitlab.set_git_projects(projects)

    gitlab.set_default_branch(set_to_default_branch=default_branch_flag)

    if clone_flag:
        gitlab.clone_projects_in_parallel()
    if pull_flag:
        gitlab.pull_projects_in_parallel()


def main():
    """
    Entry point for the command-line tool.

    Exits:
        If insufficient arguments are provided, displays usage and exits.
    """
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    repository_manager(sys.argv[1:])


if __name__ == "__main__":
    """
    Execute the main function when the script is run directly.
    """
    if len(sys.argv) < 2:
        usage()
        sys.exit(2)
    repository_manager(sys.argv[1:])
