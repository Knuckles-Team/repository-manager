#!/usr/bin/env python
# coding: utf-8
from dotenv import load_dotenv, find_dotenv
from agent_utilities.base_utilities import to_boolean
import os
import sys

__version__ = "1.3.36"

from typing import Optional, Dict, List, Any
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastmcp import FastMCP, Context
import subprocess
import logging

from fastmcp.utilities.logging import get_logger
from agent_utilities.base_utilities import to_integer
from repository_manager.repository_manager import Git
from repository_manager.models import GitResult, ReadmeResult
from agent_utilities.base_utilities import get_library_file_path
from agent_utilities.mcp_utilities import (
    create_mcp_server,
    config,
)

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = get_logger("SearXNGMCPServer")


async def execute_bash_command(command: str) -> Dict[str, Any]:
    """
    Core logic for executing a bash command.
    Used by MCP tool and skills.
    """
    logger.debug(f"Executing bash command: {command}")
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, check=False
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return {
            "status": 200 if result.returncode == 0 else 500,
            "output": output,
            "return_code": result.returncode,
        }
    except Exception as e:
        return {"status": 500, "error": str(e)}


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
            "title": "Get Project README",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
            "openWorldHint": False,
        },
        tags={"file_operations"},
    )
    async def get_project_readme(
        path: Optional[str] = Field(
            description="The path to the project or directory. Defaults to REPOSITORY_MANAGER_WORKSPACE env variable.",
            default=os.environ.get("REPOSITORY_MANAGER_WORKSPACE", None),
        ),
    ) -> ReadmeResult:
        """
        Retrieves the content of the README.md file for a project or directory.
        Use this to quickly get an overview of a project.
        """
        logger.debug(f"Getting README for path: {path}")
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
            response = git.get_readme(path=path)
            return response
        except Exception as e:
            logger.error(f"Error in get_project_readme: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Text Editor",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"file_operations"},
    )
    async def text_editor(
        command: str = Field(
            description="The command to execute: view, create, str_replace, insert, undo_edit."
        ),
        path: str = Field(
            description="Standardized file path relative to the project."
        ),
        file_text: Optional[str] = Field(
            description="The content to write to the file (for create command).",
            default=None,
        ),
        view_range: Optional[List[int]] = Field(
            description="The range of lines to view (for view command).", default=None
        ),
        old_str: Optional[str] = Field(
            description="The string to replace (for str_replace command).", default=None
        ),
        new_str: Optional[str] = Field(
            description="The new string (for str_replace and insert commands).",
            default=None,
        ),
        insert_line: Optional[int] = Field(
            description="The line number to insert at (for insert command).",
            default=None,
        ),
    ) -> GitResult:
        """
        A versatile file system editor tool.
        Supports viewing, creating, replacing text, and inserting text in files.
        """
        logger.debug(f"Executing text_editor with command: {command}, path: {path}")

        try:
            # We don't necessarily need to init Git with a specific path if path is absolute
            # But the underlying method uses self._resolve_path.
            git = Git(
                path=os.getcwd(),  # Default to CWD, let resolve_path handle the provided path
                is_mcp_server=True,
            )

            response = git.text_editor(
                command=command,
                path=path,
                file_text=file_text,
                view_range=view_range,
                old_str=old_str,
                new_str=new_str,
                insert_line=insert_line,
            )
            return response

        except Exception as e:
            logger.error(f"Error in text_editor: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Create Directory",
            "description": "Create a new directory at the specified path.",
            "readOnlyHint": False,
            "destructiveHint": False,
            "idempotentHint": False,
        },
        tags={"file_operations"},
    )
    async def create_directory(
        path: str = Field(
            description="The path where the directory should be created."
        ),
    ) -> GitResult:
        """
        Create a new directory at the specified path.
        """
        try:
            git = Git(
                path=os.getcwd(),
                is_mcp_server=True,
            )

            response = git.create_directory(path=path)
            return response
        except Exception as e:
            logger.error(f"Error in create_directory: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Delete Directory",
            "description": "Delete a directory at the specified path.",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
        },
        tags={"file_operations"},
    )
    async def delete_directory(
        path: str = Field(description="The path of the directory to delete."),
    ) -> GitResult:
        """
        Recursively delete a directory at the specified path.
        Use with caution as this action is destructive.
        """
        try:
            git = Git(
                path=os.getcwd(),
                is_mcp_server=True,
            )

            response = git.delete_directory(path=path)
            return response
        except Exception as e:
            logger.error(f"Error in delete_directory: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Rename Directory",
            "description": "Rename/Move a directory or file.",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
        },
        tags={"file_operations"},
    )
    async def rename_directory(
        old_path: str = Field(description="The current path."),
        new_path: str = Field(description="The new path."),
    ) -> GitResult:
        """
        Rename or move a directory or file from old_path to new_path.
        """
        try:
            git = Git(
                path=os.getcwd(),
                is_mcp_server=True,
            )

            response = git.rename_directory(
                old_path=old_path,
                new_path=new_path,
            )
            return response
        except Exception as e:
            logger.error(f"Error in rename_directory: {e}")
            raise

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

    @mcp.tool(
        annotations={
            "title": "Find Files",
            "description": "Find files using find.",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        },
        tags={"file_operations"},
    )
    async def find_files(
        name_pattern: str = Field(
            description="The name pattern to search for (e.g., '*.py')."
        ),
        path: Optional[str] = Field(
            description="The path to search in (absolute or relative to CWD). Defaults to CWD or workspace.",
            default=None,
        ),
    ) -> GitResult:
        """
        Find files in the codebase matching a name pattern.
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

            response = git.find_files(
                name_pattern=name_pattern,
                path=path,
            )
            return response
        except Exception as e:
            logger.error(f"Error in find_files: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Read File",
            "description": "Read a file from the codebase.",
            "readOnlyHint": True,
            "destructiveHint": False,
            "idempotentHint": True,
        },
        tags={"file_operations"},
    )
    async def read_file(
        path: str = Field(
            description="The path to the file to read (absolute or relative to CWD)."
        ),
        start_line: Optional[int] = Field(
            description="The starting line number (1-indexed).", default=None
        ),
        end_line: Optional[int] = Field(
            description="The ending line number (1-indexed).", default=None
        ),
    ) -> GitResult:
        """
        Reads the content of a file, optionally within a specific line range.
        use this to inspect code or configuration files.
        """
        try:
            git = Git(
                path=os.getcwd(),
                is_mcp_server=True,
            )

            response = git.read_file(
                path=path,
                start_line=start_line,
                end_line=end_line,
            )
            return response
        except Exception as e:
            logger.error(f"Error in read_file: {e}")
            raise

    @mcp.tool(
        annotations={
            "title": "Replace In File",
            "description": "Replace a block of text in a file.",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
        },
        tags={"file_operations"},
    )
    async def replace_in_file(
        path: str = Field(
            description="The path to the file to modify (absolute or relative to CWD)."
        ),
        target_content: str = Field(description="The exact content to be replaced."),
        replacement_content: str = Field(
            description="The new content to replace with."
        ),
    ) -> GitResult:
        """
        Replaces a specific block of text in a file with new content.
        Ensure target_content matches exactly, including whitespace.
        """
        try:
            git = Git(
                path=os.getcwd(),
                is_mcp_server=True,
            )

            response = git.replace_in_file(
                path=path,
                target_content=target_content,
                replacement_content=replacement_content,
            )
            return response
        except Exception as e:
            logger.error(f"Error in replace_in_file: {e}")
            raise


def register_system_operations_tools(mcp: FastMCP):
    @mcp.tool(
        annotations={
            "title": "Run Command",
            "readOnlyHint": False,
            "destructiveHint": True,
            "idempotentHint": False,
            "openWorldHint": True,
        },
        tags={"system_operations"},
    )
    async def run_command(
        command: str = Field(description="The command to run"),
        ctx: Context = Field(
            description="MCP context for progress reporting.", default=None
        ),
    ) -> Dict[str, Any]:
        """
        Executes a bash command on the local system.
        Use with caution. Returns exit code, stdout, and stderr.
        """
        if ctx:
            await ctx.report_progress(progress=0, total=100)

        result = await execute_bash_command(command)

        if ctx:
            await ctx.report_progress(progress=100, total=100)
        return result


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
    DEFAULT_SYSTEM_OPERATIONSTOOL = to_boolean(
        os.getenv("SYSTEM_OPERATIONSTOOL", "True")
    )
    if DEFAULT_SYSTEM_OPERATIONSTOOL:
        register_system_operations_tools(mcp)

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
