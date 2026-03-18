#!/usr/bin/env python
# coding: utf-8
from dotenv import load_dotenv, find_dotenv
from agent_utilities.base_utilities import to_boolean
import os
import sys

__version__ = "1.3.42"

from typing import Optional, List
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastmcp import FastMCP
import logging

from fastmcp.utilities.logging import get_logger
from agent_utilities.base_utilities import to_integer
from repository_manager.repository_manager import Git
from repository_manager.models import GitResult
from agent_utilities.base_utilities import get_library_file_path
from agent_utilities.mcp_utilities import (
    create_mcp_server,
    config,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("RepositoryManagerServer")


def register_misc_tools(mcp: FastMCP):
    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})


def register_git_operations_tools(mcp: FastMCP):
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
        path: Optional[str] = Field(
            description="The path to execute the command in. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", "/workspace"),
        ),
        threads: Optional[int] = Field(
            description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
            default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
        ),
        set_to_default_branch: Optional[bool] = Field(
            description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", False)
            ),
        ),
    ) -> GitResult:
        """
        Executes an arbitrary Git command in the specified path.
        Use this tool for any git operation not covered by specialized tools.
        """
        logger.debug(f"Executing git_action with command: {command}, path: {path}")
        try:
            git = Git(
                path=path,
                threads=threads,
                set_to_default_branch=set_to_default_branch,
                capture_output=True,
                is_mcp_server=True,
            )
            response = git.git_action(command=command, path=path)
            return response
        except Exception as e:
            logger.error(f"Error in git_action: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "List Git Projects",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"git_operations"},
    )
    async def list_projects(
        projects_file: Optional[str] = Field(
            description="Path to a file containing a list of repository URLs. Defaults to PROJECTS_FILE env variable.",
            default=os.environ.get(
                "PROJECTS_FILE", get_library_file_path(file="repositories-list.txt")
            ),
        ),
        path: Optional[str] = Field(
            description="The parent workspace containing the projects. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
    ) -> List[str]:
        """
        Lists all Git repositories found in the specified workspace path or defined in the projects file.
        Use this to discover available projects before performing operations.
        """
        logger.debug(f"Listing projects from file: {projects_file} and path: {path}")
        try:
            git = Git(
                path=path,
                projects=None,
                is_mcp_server=True,
            )

            if projects_file and os.path.exists(projects_file):
                git.read_project_list_file(file=projects_file)

            if path and os.path.exists(path):
                try:
                    for item in os.listdir(path):
                        if os.path.isdir(os.path.join(path, item)) and os.path.exists(
                            os.path.join(path, item, ".git")
                        ):
                            git.projects.append(item)
                except Exception:
                    pass

            return list(dict.fromkeys(git.projects))

        except Exception as e:
            logger.error(f"Error in list_projects: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Run Pre-Commit",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": False,
        },
        tags={"git_operations"},
    )
    async def run_pre_commit(
        run: bool = Field(description="Run 'pre-commit run --all-files'", default=True),
        autoupdate: bool = Field(
            description="Run 'pre-commit autoupdate'", default=False
        ),
        path: Optional[str] = Field(
            description="The path to run pre-commit in (e.g., path/to/project). Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
    ) -> GitResult:
        """
        Runs pre-commit hooks and/or autoupdate on a repository.
        Enhances code quality by running configured hooks.
        """
        logger.debug(
            f"Running pre-commit: run={run}, autoupdate={autoupdate}, path={path}"
        )
        try:
            target_dir = path
            if not target_dir:
                target_dir = os.getcwd()

            git = Git(
                path=target_dir,
                is_mcp_server=True,
            )

            response = git.pre_commit(run=run, autoupdate=autoupdate, path=target_dir)
            return response

        except Exception as e:
            logger.error(f"Error in run_pre_commit: {e}")
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
        url: str = Field(description="The repository URL to clone."),
        path: Optional[str] = Field(
            description="The path to clone the project into. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
        threads: Optional[int] = Field(
            description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
            default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
        ),
        set_to_default_branch: Optional[bool] = Field(
            description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
        ),
    ) -> GitResult:
        """
        Clones a single Git repository from the provided URL to the specified path.
        """
        logger.debug(f"Cloning project: {url}, path: {path}")
        try:
            if not url:
                raise ValueError("url must not be empty")
            git = Git(
                path=path,
                threads=threads,
                set_to_default_branch=set_to_default_branch,
                capture_output=True,
                is_mcp_server=True,
            )
            response = git.clone_project(url=url)
            return response
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
            default=os.environ.get(
                "PROJECTS_FILE", get_library_file_path(file="repositories-list.txt")
            ),
        ),
        path: Optional[str] = Field(
            description="The path to clone projects into. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
        threads: Optional[int] = Field(
            description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
            default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
        ),
        set_to_default_branch: Optional[bool] = Field(
            description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
        ),
    ) -> List[GitResult]:
        """
        Clones multiple Git repository URLs in parallel to the specified workspace path.
        Can use a list of URLs or a file containing URLs.
        """
        logger.debug(f"Cloning projects to path: {path}")
        try:
            if not projects and not projects_file:
                raise ValueError("Either projects or projects_file must be provided")
            if projects_file and not os.path.exists(projects_file):
                raise FileNotFoundError(f"Projects file not found: {projects_file}")
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"Repository path not found: {path}")
            git = Git(
                path=path,
                projects=projects,
                threads=threads,
                set_to_default_branch=set_to_default_branch,
                capture_output=True,
                is_mcp_server=True,
            )
            if projects_file:
                git.read_project_list_file(file=projects_file)
            response = git.clone_projects()
            return response
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
        path: str = Field(description="The path of the project directory to pull."),
        threads: Optional[int] = Field(
            description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
            default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
        ),
        set_to_default_branch: Optional[bool] = Field(
            description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
        ),
    ) -> GitResult:
        """
        Performs a 'git pull' on the repository at the specified path.
        Useful for updating a specific project to the latest commit.
        """
        logger.debug(f"Pulling project: path: {path}")
        try:
            if not path:
                raise ValueError("path must not be empty")
            git = Git(
                path=path,  # Use path as base or let Git manage?
                threads=threads,
                set_to_default_branch=set_to_default_branch,
                capture_output=True,
                is_mcp_server=True,
            )
            # We want to pull specific project. Git initialized with path might assume path is the workspace.
            # But pull_project in Git takes 'path' (the project path).
            # If we call git.pull_project(path=path), it uses self._resolve_path(path).
            # So if we init Git(path=os.path.dirname(path)) it might be cleaner, but Git defaults to self.path.
            # If we init Git() (default), then pass absolute path, it should work.
            response = git.pull_project(path=path)
            return response
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
        path: Optional[str] = Field(
            description="The workspace containing the projects to pull. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
        threads: Optional[int] = Field(
            description="Number of threads for parallel processing. Defaults to REPOSITORY_MANAGER_THREADS env variable.",
            default=to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "6")),
        ),
        set_to_default_branch: Optional[bool] = Field(
            description="Whether to checkout the default branch. Defaults to REPOSITORY_MANAGER_DEFAULT_BRANCH env variable.",
            default=to_boolean(
                os.environ.get("REPOSITORY_MANAGER_DEFAULT_BRANCH", None)
            ),
        ),
    ) -> List[GitResult]:
        """
        Pulls updates for all Git projects within the specified workspace path in parallel.
        Useful for bulk updating multiple repositories at once.
        """
        logger.debug(f"Pulling projects from path: {path}")
        try:
            if path and not os.path.exists(path):
                raise FileNotFoundError(f"Repository path not found: {path}")
            git = Git(
                path=path,
                threads=threads,
                set_to_default_branch=set_to_default_branch,
                capture_output=True,
                is_mcp_server=True,
            )
            response = git.pull_projects()
            return response
        except Exception as e:
            logger.error(f"Error in pull_projects: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Create Project",
            "description": "Create a new project directory and initialize it as a git repository.",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
        },
        tags={"git_operations"},
    )
    async def create_project(
        path: str = Field(description="The path for the new project directory."),
    ) -> GitResult:
        """
        Create a new project directory at the specified path and initialize it as a git repository.
        Use this to start a new project managed by git.
        """
        try:
            # Init Git with base path or current directory, actual creation uses absolute path logic
            git = Git(
                path=os.getcwd(),
                is_mcp_server=True,
            )

            response = git.create_project(path=path)
            return response
        except Exception as e:
            logger.error(f"Error in create_project: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Bump Version",
            "description": "Bump the version of the project using bump2version.",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
        },
        tags={"git_operations"},
    )
    async def bump_version(
        part: str = Field(
            description="The part of the version to bump (major, minor, patch)."
        ),
        allow_dirty: bool = Field(
            description="Whether to allow dirty working directory.", default=True
        ),
        path: Optional[str] = Field(
            description="The path to the project directory. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
    ) -> GitResult:
        """
        Bumps the version of the project using bump2version.
        Automatically commits and tags the new version.
        """
        try:
            target_dir = path
            if not target_dir:
                target_dir = (
                    os.environ.get("REPOSITORY_MANAGER_WORKSPACE") or os.getcwd()
                )

            git = Git(
                path=target_dir,
                is_mcp_server=True,
            )

            response = git.bump_version(
                part=part,
                allow_dirty=allow_dirty,
                path=path,
            )
            return response
        except Exception as e:
            logger.error(f"Error in bump_version: {e}")
            raise


def register_file_operations_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Search Codebase",
            "description": "Search the codebase using ripgrep.",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        },
        tags={"file_operations"},
    )
    async def search_codebase(
        query: str = Field(description="The regex pattern to search for."),
        path: Optional[str] = Field(
            description="The path to search in (absolute or relative to CWD). Defaults to CWD or workspace.",
            default=None,
        ),
        glob_pattern: Optional[str] = Field(
            description="Glob pattern to filter files (e.g., '*.py').", default=None
        ),
        case_sensitive: bool = Field(
            description="Whether the search should be case sensitive.", default=False
        ),
    ) -> GitResult:
        """
        Search for a regex pattern in the codebase using ripgrep.
        Supports filtering by glob pattern and case sensitivity.
        """
        try:
            target_dir = path
            if not target_dir:
                target_dir = (
                    os.environ.get("REPOSITORY_MANAGER_WORKSPACE") or os.getcwd()
                )

            git = Git(
                path=target_dir,
                is_mcp_server=True,
            )

            response = git.search_codebase(
                query=query,
                path=path,
                glob_pattern=glob_pattern,
                case_sensitive=case_sensitive,
            )
            return response
        except Exception as e:
            logger.error(f"Error in search_codebase: {e}")
            raise


def mcp_server():
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="GitRepositoryManager",
        version=__version__,
        instructions="Git Repository Manager MCP Server — Create, clone, pull, search, and manage git repositories and files.",
    )

    DEFAULT_MISCTOOL = to_boolean(os.getenv("MISCTOOL", "True"))
    if DEFAULT_MISCTOOL:
        register_misc_tools(mcp)
    DEFAULT_GIT_OPERATIONSTOOL = to_boolean(os.getenv("GIT_OPERATIONSTOOL", "True"))
    if DEFAULT_GIT_OPERATIONSTOOL:
        register_git_operations_tools(mcp)
    DEFAULT_FILE_OPERATIONSTOOL = to_boolean(os.getenv("FILE_OPERATIONSTOOL", "True"))
    if DEFAULT_FILE_OPERATIONSTOOL:
        register_file_operations_tools(mcp)

    for mw in middlewares:
        mcp.add_middleware(mw)

    print(f"Repository Manager MCP v{__version__}")
    print("\nStarting Git MCP Server")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Auth: {args.auth_type}")
    print(f"  Delegation: {'ON' if config['enable_delegation'] else 'OFF'}")
    print(f"  Eunomia: {args.eunomia_type}")

    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "streamable-http":
        mcp.run(transport="streamable-http", host=args.host, port=args.port)
    elif args.transport == "sse":
        mcp.run(transport="sse", host=args.host, port=args.port)
    else:
        logger.error("Invalid transport", extra={"transport": args.transport})
        sys.exit(1)


if __name__ == "__main__":
    mcp_server()
