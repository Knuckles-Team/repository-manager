#!/usr/bin/env python
# coding: utf-8
import os
import sys
import getopt
from typing import Optional, List
from fastmcp import FastMCP, Context
from repository_manager import setup_logging, Git

logger = setup_logging(is_mcp_server=True, log_file="repository_manager_mcp.log")

mcp = FastMCP(name="GitRepositoryManager")


@mcp.tool()
def git_action(
    command: str,
    repository_directory: str = None,
    projects: Optional[list] = None,
    projects_file: Optional[str] = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
    ctx: Context = None,
) -> str:
    """
    Execute a Git command in the specified directory using a configured Git instance.

    Args:
        command (str): The Git command to execute (e.g., 'git pull', 'git clone <repository_url>').
        repository_directory (Optional[str], optional): The directory to execute the command in.
        projects (Optional[List[str]], optional): List of repository URLs for Git operations.
        projects_file (Optional[str], optional): Path to a file containing a list of repository URLs.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.
        ctx (Context, optional): MCP context for logging.

    Returns:
        str: The combined stdout and stderr output of the executed Git command.

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
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
) -> str:
    """
    Clone a single Git project using a configured Git instance.

    Args:
        git_project (Optional[str], optional): The repository URL to clone.
        repository_directory (Optional[str], optional): The directory to clone the project into.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.

    Returns:
        str: The output of the Git clone command.

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
        return response
    except Exception as e:
        logger.error(f"Error in clone_project: {e}")
        raise


@mcp.tool()
def clone_projects(
    projects: Optional[List[str]] = None,
    projects_file: Optional[str] = None,
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
    ctx: Context = None,
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
        str: Combined output of all clone operations. This will likely show that a project was
        cloned successfully, or that the project attempting to clone already exists.

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
        return response
    except Exception as e:
        logger.error(f"Error in clone_projects: {e}")
        raise


@mcp.tool()
def pull_project(
    git_project: str,
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
) -> str:
    """
    Pull updates for a single Git project using a configured Git instance.

    Args:
        git_project (str): The name of the project directory to pull.
        repository_directory (Optional[str], optional): The parent directory containing the project.
        threads (Optional[int], optional): Number of threads for parallel processing.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch.

    Returns:
        str: The output of the Git pull command, including checkout output if applicable.

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
        return response
    except Exception as e:
        logger.error(f"Error in pull_project: {e}")
        raise


@mcp.tool()
def pull_projects(
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
    ctx: Context = None,
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
        str: Combined output of all pull operations. This will return a combined output of all the projects that were
        identified in the repository folder. It is expected that some will show pulling successful,
        while others show failures.

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
        return response
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
            port = arg
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
