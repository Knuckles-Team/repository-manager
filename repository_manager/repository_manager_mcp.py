#!/usr/bin/env python
# coding: utf-8
import os
import sys
import argparse
import logging
from typing import Optional, List, Dict, Union
from fastmcp import FastMCP
from repository_manager import setup_logging, Git
from pydantic import Field

# Initialize logging for MCP server
logger = setup_logging(is_mcp_server=True, log_file="repository_manager_mcp.log")

mcp = FastMCP(name="GitRepositoryManager")


def to_boolean(string: str = None) -> bool:
    if not string:
        return False
    normalized = str(string).strip().lower()
    true_values = {"t", "true", "y", "yes", "1"}
    false_values = {"f", "false", "n", "no", "0"}
    if normalized in true_values:
        return True
    elif normalized in false_values:
        return False
    else:
        raise ValueError(f"Cannot convert '{string}' to boolean")


def to_integer(string: Union[str] = None) -> int:
    if not string:
        return 0
    try:
        return int(string.strip())
    except ValueError:
        raise ValueError(f"Cannot convert '{string}' to integer")


@mcp.tool(
    annotations={
        "title": "Execute Git Command",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def git_action(
    command: str = Field(
        description="The Git command to execute (e.g., 'git pull', 'git clone <repository_url>')"
    ),
    repository_directory: Optional[str] = Field(
        description="The directory to execute the command in. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    projects: Optional[List[str]] = Field(
        description="List of repository URLs for Git operations.", default=None
    ),
    projects_file: Optional[str] = Field(
        description="Path to a file containing a list of repository URLs. Defaults to PROJECTS_FILE env variable.",
        default=os.environ.get("PROJECTS_FILE", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> Dict:
    """
    Executes a Git command in the specified directory.
    Returns details from that git action run
    """
    logger.debug(
        f"Executing git_action with command: {command}, directory: {repository_directory}"
    )
    try:
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
        response = git.git_action(command=command)
        return response
    except Exception as e:
        logger.error(f"Error in git_action: {e}")
        raise


@mcp.tool(
    annotations={
        "title": "Clone Single Git Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def clone_project(
    git_project: Optional[str] = Field(
        description="The repository URL to clone.", default=None
    ),
    repository_directory: Optional[str] = Field(
        description="The directory to clone the project into. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Clones a single Git project to the specified directory.
    Returns details about the cloned project
    """
    logger.debug(f"Cloning project: {git_project}, directory: {repository_directory}")
    try:
        if not git_project:
            raise ValueError("git_project must not be empty")
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


@mcp.tool(
    annotations={
        "title": "Clone Multiple Git Projects",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def clone_projects(
    projects: Optional[List[str]] = Field(
        description="List of repository URLs to clone.", default=None
    ),
    projects_file: Optional[str] = Field(
        description="Path to a file containing a list of repository URLs. Defaults to PROJECTS_FILE env variable.",
        default=os.environ.get("PROJECTS_FILE", None),
    ),
    repository_directory: Optional[str] = Field(
        description="The directory to clone projects into. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Clones multiple Git projects in parallel to the specified directory.
    Returns a list of projects that were cloned
    """
    logger.debug(f"Cloning projects to directory: {repository_directory}")
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


@mcp.tool(
    annotations={
        "title": "Pull Single Git Project",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def pull_project(
    git_project: str = Field(description="The name of the project directory to pull."),
    repository_directory: Optional[str] = Field(
        description="The parent directory containing the project. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Pulls updates for a single Git project.
    Returns details about project pulled using git
    """
    logger.debug(f"Pulling project: {git_project}, directory: {repository_directory}")
    try:
        if not git_project:
            raise ValueError("git_project must not be empty")
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


@mcp.tool(
    annotations={
        "title": "Pull Multiple Git Projects",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    },
    tags={"git_operations"},
)
async def pull_projects(
    repository_directory: Optional[str] = Field(
        description="The directory containing the projects to pull. Defaults to REPOSITORY_MANAGER_DIRECTORY env variable.",
        default=os.environ.get("REPOSITORY_MANAGER_DIRECTORY", None),
    ),
    threads: Optional[int] = Field(
        description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
        default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
    ),
    set_to_default_branch: Optional[bool] = Field(
        description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
        default=to_boolean(os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)),
    ),
) -> str:
    """
    Pulls updates for multiple Git projects in parallel.
    Returns a list of projects that were pulled
    """
    logger.debug(f"Pulling projects from directory: {repository_directory}")
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


def repository_manager_mcp():
    parser = argparse.ArgumentParser(description="Repository Manager MCP Utility")

    parser.add_argument(
        "-t",
        "--transport",
        default="stdio",
        choices=["stdio", "http", "sse"],
        help="Transport method: 'stdio', 'http', or 'sse' [legacy] (default: stdio)",
    )
    parser.add_argument(
        "-s",
        "--host",
        default="0.0.0.0",
        help="Host address for HTTP transport (default: 0.0.0.0)",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8000,
        help="Port number for HTTP transport (default: 8000)",
    )

    args = parser.parse_args()

    if args.port < 0 or args.port > 65535:
        print(f"Error: Port {args.port} is out of valid range (0-65535).")
        sys.exit(1)

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger = logging.getLogger("RepositoryManager")
        logger.error("Transport not supported")
        sys.exit(1)


if __name__ == "__main__":
    repository_manager_mcp()
