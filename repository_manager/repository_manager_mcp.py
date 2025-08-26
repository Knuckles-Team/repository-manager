#!/usr/bin/env python
# coding: utf-8
import os
import sys
import getopt
from typing import Optional, List, Dict
from fastmcp import FastMCP
from repository_manager import setup_logging, Git

logger = setup_logging(is_mcp_server=True, log_file="repository_manager_mcp.log")

mcp = FastMCP(name="GitRepositoryManager")

def to_boolean(string):
    # Normalize the string: strip whitespace and convert to lowercase
    normalized = str(string).strip().lower()

    # Define valid true/false values
    true_values = {'t', 'true', 'y', 'yes', '1'}
    false_values = {'f', 'false', 'n', 'no', '0'}

    if normalized in true_values:
        return True
    elif normalized in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{string}' to boolean")

def to_integer(arg):
    try:
        # Strip whitespace and convert to int
        return int(arg.strip())
    except ValueError:
        raise ValueError(f"Cannot convert '{arg}' to integer")

environment_repository_directory = os.environ.get("REPOSITORY_DIRECTORY", None)
environment_threads = os.environ.get("THREADS", None)
environment_set_to_default_branch = os.environ.get("DEFAULT_BRANCH", None)
environment_projects_file = os.environ.get("PROJECTS_FILE", None)

if environment_set_to_default_branch:
    environment_set_to_default_branch = to_boolean(environment_set_to_default_branch)
if environment_threads:
    environment_threads = to_integer(environment_threads)

@mcp.tool()
def git_action(
    command: str,
    repository_directory: str = environment_repository_directory,
    projects: Optional[list] = None,
    projects_file: Optional[str] = environment_projects_file,
    threads: Optional[int] = environment_threads,
    set_to_default_branch: Optional[bool] = environment_set_to_default_branch,
) -> Dict:
    """
    Execute a Git command in the specified directory using a configured Git instance.

    Args:
        command (str): The Git command to execute (e.g., 'git pull', 'git clone <repository_url>').
        repository_directory (Optional[str], optional): The directory to execute the command in.
            This will also look for the environment variable REPOSITORY_DIRECTORY
        projects (Optional[List[str]], optional): List of repository URLs for Git operations.
        projects_file (Optional[str], optional): Path to a file containing a list of repository URLs.
            This will also look for the environment variable PROJECTS_FILE
        threads (Optional[int], optional): Number of threads for parallel processing.
            This will also look for the environment variable THREADS
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.
            This will also look for the environment variable DEFAULT_BRANCH
        ctx (Context, optional): MCP context for logging.

    Returns:
        Dict: The combined stdout and stderr output of the executed Git command in structured format.

    Raises:
        FileNotFoundError: If the specified repository directory or projects_file does not exist.
    """
    try:
        git = Git(
            repository_directory=repository_directory,
            projects=projects,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,  # Pass is_mcp_server
        )
        if projects_file:
            git.read_project_list_file(file=projects_file)
        response = git.git_action(command=command)
        return response
    except Exception as e:
        logger.error(f"Error in git_action: {e}")
        raise


@mcp.tool()
def clone_project(
    git_project: str = None,
    repository_directory: str = environment_repository_directory,
    threads: Optional[int] = environment_threads,
    set_to_default_branch: Optional[bool] = environment_set_to_default_branch,
) -> str:
    """
    Clone a single Git project using a configured Git instance.

    Args:
        git_project (Optional[str], optional): The repository URL to clone.
        repository_directory (Optional[str], optional): The directory to clone the project into.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.

    Returns:
        str: The response message string

    Raises:
        FileNotFoundError: If the repository directory does not exist.
        ValueError: If git_project is not provided.
    """
    try:
        git = Git(
            repository_directory=repository_directory,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        response = git.clone_project(git_project=git_project)
        return f"Project {git_project} cloned to {repository_directory} successfully!"
    except Exception as e:
        logger.error(f"Error in clone_project: {e}")
        raise


@mcp.tool()
def clone_projects(
    projects: Optional[List[str]] = None,
    projects_file: Optional[str] = environment_projects_file,
    repository_directory: str = environment_repository_directory,
    threads: Optional[int] = environment_threads,
    set_to_default_branch: Optional[bool] = environment_set_to_default_branch,
) -> str:
    """
    Clone multiple Git projects in parallel using a configured Git instance. Successful and Failed pulls
    are to be expected from the response output. This function should only be run once. Just let the user know the
    action was performed once finished.

    Args:
        projects (Optional[List[str]], optional): List of repository URLs to clone.
        projects_file (Optional[str], optional): Path to a file containing a list of repository URLs.
        repository_directory (Optional[str], optional): The directory to clone projects into.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.
        ctx (Context, optional): MCP context for logging.

    Returns:
        str: The response message string

    Raises:
        FileNotFoundError: If the repository directory or projects_file does not exist.
        ValueError: If neither projects nor projects_file is provided, or if both are empty.
    """
    try:
        if not projects and not projects_file:
            raise ValueError("Either projects or projects_file must be provided")
        if projects_file and not os.path.exists(projects_file):
            raise FileNotFoundError(f"Projects file not found: {projects_file}")
        if repository_directory and not os.path.exists(repository_directory):
            raise FileNotFoundError(
                f"Repository directory not found: {repository_directory}"
            )
        git = Git(
            repository_directory=repository_directory,
            projects=projects,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        if projects_file:
            git.read_project_list_file(file=projects_file)
        response = git.clone_projects_in_parallel()
        return f"Project {git.projects} cloned to {repository_directory} successfully!"
    except Exception as e:
        logger.error(f"Error in clone_projects: {e}")
        raise


@mcp.tool()
def pull_project(
    git_project: str,
    repository_directory: str = environment_repository_directory,
    threads: Optional[int] = environment_threads,
    set_to_default_branch: Optional[bool] = environment_set_to_default_branch,
) -> str:
    """
    Pull updates for a single Git project using a configured Git instance.

    Args:
        git_project (str): The name of the project directory to pull.
        repository_directory (Optional[str], optional): The parent directory containing the project.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.

    Returns:
        str: The response message string

    Raises:
        FileNotFoundError: If the project directory does not exist.
        ValueError: If git_project is not provided.
    """
    try:
        git = Git(
            repository_directory=repository_directory,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        response = git.pull_project(git_project=git_project)
        return f"Project {git_project} pulled to {repository_directory} successfully!"
    except Exception as e:
        logger.error(f"Error in pull_project: {e}")
        raise


@mcp.tool()
def pull_projects(
    repository_directory: str = environment_repository_directory,
    threads: Optional[int] = environment_threads,
    set_to_default_branch: Optional[bool] = environment_set_to_default_branch,
) -> str:
    """
    Pull updates for multiple Git projects located in the repository_directory. Successful and Failed pulls
    are to be expected from the response output. This function should only be run once. Just let the user know the
    action was performed once finished.

    Args:
        repository_directory (Optional[str], optional): The directory containing the projects to pull.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.
        ctx (Context, optional): MCP context for logging.

    Returns:
        str: The response message string

    Raises:
        FileNotFoundError: If the repository directory does not exist.
    """
    try:
        if repository_directory and not os.path.exists(repository_directory):
            raise FileNotFoundError(
                f"Repository directory not found: {repository_directory}"
            )
        git = Git(
            repository_directory=repository_directory,
            threads=threads,
            set_to_default_branch=set_to_default_branch,
            capture_output=True,
            is_mcp_server=True,
        )
        response = git.pull_projects_in_parallel()
        return f"All projects in {repository_directory} pulled successfully!"
    except Exception as e:
        logger.error(f"Error in pull_projects: {e}")
        raise


def repository_manager_mcp(argv):
    transport = "stdio"
    host = "0.0.0.0"
    port = 8000
    try:
        opts, args = getopt.getopt(
            argv,
            "ht:h:p:",
            ["help", "transport=", "host=", "port="],
        )
    except getopt.GetoptError:
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            sys.exit()
        elif opt in ("-t", "--transport"):
            transport = arg
        elif opt in ("-h", "--host"):
            host = arg
        elif opt in ("-p", "--port"):
            try:
                port = int(arg)  # Attempt to convert port to integer
                if not (0 <= port <= 65535):  # Valid port range
                    print(f"Error: Port {arg} is out of valid range (0-65535).")
                    sys.exit(1)
            except ValueError:
                print(f"Error: Port {arg} is not a valid integer.")
                sys.exit(1)
    if transport == "stdio":
        mcp.run(transport="stdio")
    elif transport == "http":
        mcp.run(transport="http", host=host, port=port)
    else:
        logger.error("Transport not supported")
        sys.exit(1)


def main():
    repository_manager_mcp(sys.argv[1:])


if __name__ == "__main__":
    repository_manager_mcp(sys.argv[1:])
