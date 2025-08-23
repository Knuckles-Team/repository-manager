#!/usr/bin/env python
# coding: utf-8
import os
import sys
import getopt
import logging
from typing import Optional, List
from fastmcp import FastMCP, Context
from repository_manager import setup_logging, Git

setup_logging(is_mcp_server=True, log_file="repository_manager_mcp.log")

mcp = FastMCP(name="GitRepositoryManager")


@mcp.tool()
async def git_action(
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

    This tool executes a specified Git command within a given repository directory, leveraging a Git class instance
    configured with the provided parameters for repository management. It supports parallel processing, optional default
    branch checkout settings, and reading repository URLs from a file.

    Args:
        command (str): The Git command to execute (e.g., 'git pull', 'git clone <repository_url>').
        repository_directory (Optional[str], optional): The directory to execute the command in.
            Defaults to the current working directory.
        projects (Optional[List[str]], optional): List of repository URLs for Git operations.
            Defaults to None.
        projects_file (Optional[str], optional): Path to a file containing a list of repository URLs,
            one per line. Defaults to None.
        threads (Optional[int], optional): Number of threads for parallel processing.
            Defaults to the number of CPU cores.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch after certain operations.
            Defaults to False.
        ctx (Context, optional): MCP context for logging.

    Returns:
        str: The combined stdout and stderr output of the executed Git command.

    Raises:
        FileNotFoundError: If the specified repository directory or projects_file does not exist.
    """

    await ctx.debug("Starting analysis of numerical data")
    git = Git(
        repository_directory=repository_directory,
        projects=projects,
        threads=threads,
        set_to_default_branch=set_to_default_branch,
        capture_output=True,
    )
    await ctx.info("Analyzing data points")
    if projects_file:
        git.read_project_list_file(file=projects_file)
    await ctx.info(f"Projects File: {projects_file}")
    response = git.git_action(command=command)
    await ctx.info(f"Response: {response}")
    return response


@mcp.tool()
def clone_project(
    git_project: str = None,
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
) -> str:
    """
    Clone a single Git project using a configured Git instance.

    This tool clones a specified Git repository into the given directory, leveraging a Git class instance
    configured with the provided parameters for repository management. It supports parallel processing and
    optional default branch checkout settings.

    Args:
        git_project (Optional[str], optional): The repository URL to clone. Defaults to None.
        repository_directory (Optional[str], optional): The directory to clone the project into.
            Defaults to the current working directory.
        threads (Optional[int], optional): Number of threads for parallel processing.
            Defaults to the number of CPU cores.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch after operations.
            Defaults to False.

    Returns:
        str: The output of the Git clone command.

    Raises:
        FileNotFoundError: If the repository directory does not exist.
        ValueError: If git_project is not provided.
    """
    git = Git(
        repository_directory=repository_directory,
        threads=threads,
        set_to_default_branch=set_to_default_branch,
        capture_output=True,
    )
    response = git.clone_project(git_project=git_project)
    return response


@mcp.tool()
async def clone_projects(
    projects: Optional[List[str]] = None,
    projects_file: Optional[str] = None,
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
    ctx: Context = None,
) -> str:
    """
    Clone multiple Git projects in parallel using a configured Git instance.

    This tool clones a list of Git repositories into the specified directory in parallel, leveraging a Git class instance
    configured with the provided parameters for repository management. It supports reading repository URLs from a file
    and uses multiple threads for efficient processing.

    Args:
        projects (Optional[List[str]], optional): List of repository URLs to clone. Defaults to None.
        projects_file (Optional[str], optional): Path to a file containing a list of repository URLs,
            one per line. Defaults to None.
        repository_directory (Optional[str], optional): The directory to clone projects into.
            Defaults to the current working directory.
        threads (Optional[int], optional): Number of threads for parallel processing.
            Defaults to the number of CPU cores.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch after operations.
            Defaults to False.
        ctx (Context, optional): MCP context for logging.

    Returns:
        str: Combined output of all clone operations.

    Raises:
        FileNotFoundError: If the repository directory or projects_file does not exist.
        ValueError: If neither projects nor projects_file is provided, or if both are provided but empty.
    """
    await ctx.debug("Starting cloning of projects")
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
    )
    if projects_file:
        git.read_project_list_file(file=projects_file)
    await ctx.info(f"Projects File: {projects_file}")
    response = git.clone_projects_in_parallel()
    await ctx.debug(f"Response: {response}")
    return response


@mcp.tool()
def pull_project(
    git_project: str,
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
) -> str:
    """
    Pull updates for a single Git project using a configured Git instance.

    This tool pulls updates for a specified Git project directory and optionally checks out the default branch,
    leveraging a Git class instance configured with the provided parameters for repository management.

    Args:
        git_project (str): The name of the project directory to pull.
        repository_directory (Optional[str], optional): The parent directory containing the project.
            Defaults to the current working directory.
        threads (Optional[int], optional): Number of threads for parallel processing.
            Defaults to the number of CPU cores.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch after pulling.
            Defaults to False.

    Returns:
        str: The output of the Git pull command, including checkout output if applicable.

    Raises:
        FileNotFoundError: If the project directory does not exist.
        ValueError: If git_project is not provided.
    """
    git = Git(
        repository_directory=repository_directory,
        threads=threads,
        set_to_default_branch=set_to_default_branch,
        capture_output=True,
    )
    response = git.pull_project(git_project=git_project)
    return response


@mcp.tool()
async def pull_projects(
    repository_directory: str = None,
    threads: Optional[int] = None,
    set_to_default_branch: Optional[bool] = False,
    ctx: Context = None,
) -> str:
    """
    Pull updates for multiple Git projects located in the repository_directory,
    optionally checking out the default branch.

    Args:
        repository_directory (Optional[str], optional): The directory containing the projects to pull.
            Defaults to the current working directory.
        threads (Optional[int], optional): Number of threads for parallel processing.
            Defaults to the number of CPU cores.
        set_to_default_branch (Optional[bool], optional): Whether to checkout the default branch after pulling.
            Defaults to False.
        ctx (Context, optional): MCP context for logging.

    Returns:
        str: Combined output of all pull operations.

    Raises:
        FileNotFoundError: If the repository directory does not exist.
    """
    await ctx.debug("Starting pulling of projects")
    if repository_directory and not os.path.exists(repository_directory):
        raise FileNotFoundError(
            f"Repository directory not found: {repository_directory}"
        )
    git = Git(
        repository_directory=repository_directory,
        threads=threads,
        set_to_default_branch=set_to_default_branch,
        capture_output=True,
    )
    response = git.pull_projects_in_parallel()
    await ctx.debug(f"Response: {response}")
    return response


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
        logger = logging.getLogger("RepositoryManager")
        logger.error("Transport not supported")
        sys.exit(1)


def main():
    repository_manager_mcp(sys.argv[1:])


if __name__ == "__main__":
    repository_manager_mcp(sys.argv[1:])
