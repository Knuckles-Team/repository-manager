#!/usr/bin/env python

from dotenv import load_dotenv, find_dotenv
import os
import sys
from typing import Any, Optional, List, Dict
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse
from fastmcp import FastMCP
from fastmcp.utilities.logging import get_logger

from repository_manager.repository_manager import Git
from repository_manager.models import (
    GitResult,
    WorkspaceConfig,
    TaskList,
    TaskStatus,
)
from agent_utilities.base_utilities import to_boolean, to_integer
from agent_utilities.mcp_utilities import create_mcp_server

__version__ = "1.3.51"


DEFAULT_WORKSPACE = os.environ.get("REPOSITORY_MANAGER_WORKSPACE", "/workspace")
DEFAULT_THREADS = to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "12"))
DEFAULT_WORKSPACE_YML = os.environ.get("WORKSPACE_YML", "workspace.yml")

logger = get_logger("RepositoryManagerServer")


def get_git_instance(path: Optional[str] = None, threads: Optional[int] = None) -> Git:
    """Helper to get a Git instance with workpace YAML loaded."""
    workspace_path = path or DEFAULT_WORKSPACE
    git = Git(path=workspace_path, threads=threads or DEFAULT_THREADS)

    yml_path = os.path.join(workspace_path, DEFAULT_WORKSPACE_YML)
    if os.path.exists(yml_path):
        git.load_projects_from_yaml(yml_path)

    return git


def register_misc_tools(mcp: FastMCP):
    """Register miscellaneous tools like health check."""

    async def health_check(request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})


def register_git_operations_tools(mcp: FastMCP):
    @mcp.tool(tags={"git_operations"})
    async def git_action(
        command: str = Field(
            description="The Git command to execute (e.g., 'git status')"
        ),
        path: Optional[str] = Field(description="Path to execute in.", default=None),
    ) -> GitResult:
        """Executes an arbitrary Git command."""
        git = get_git_instance(path=path)
        return git.git_action(command=command, path=path)

    @mcp.tool(tags={"git_operations"})
    async def get_workspace_projects() -> List[str]:
        """Lists all project URLs defined in the workspace configuration."""
        git = get_git_instance()
        return list(git.project_map.keys())

    @mcp.tool(tags={"git_operations"})
    async def clone_projects(
        projects: Optional[List[str]] = Field(
            description="Optional list of URLs to clone.", default=None
        ),
        threads: Optional[int] = Field(description="Parallel workers.", default=None),
    ) -> List[GitResult]:
        """Clones repositories. Defaults to all in workspace.yml if none provided."""
        git = get_git_instance(threads=threads)
        results = git.clone_projects()
        return git.generate_markdown_summary("Clone", results)

    @mcp.tool(tags={"git_operations"})
    async def pull_projects(
        threads: Optional[int] = Field(description="Parallel workers.", default=None),
    ) -> List[GitResult]:
        """Pulls updates for all projects in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.pull_projects()
        return git.generate_markdown_summary("Pull", results)


def register_workspace_management_tools(mcp: FastMCP):
    @mcp.tool(tags={"workspace_management"})
    async def setup_workspace(
        yml_path: str = Field(description="Path to the workspace.yml file."),
    ) -> GitResult:
        """Sets up the entire workspace, clones repos, and organizes subdirectories."""
        git = get_git_instance()
        return git.setup_from_yaml(yml_path)

    @mcp.tool(tags={"workspace_management"})
    async def install_projects(
        threads: Optional[int] = Field(description="Parallel workers.", default=None),
        extra: str = Field(description="Install group (e.g. 'all').", default="all"),
    ) -> List[GitResult]:
        """Runs install scripts for all cloned repositories."""
        git = get_git_instance(threads=threads)
        results = git.install_projects(extra=extra)
        return git.generate_markdown_summary("Install", results)


def register_project_management_tools(mcp: FastMCP):
    """Register tools for the autonomous project harness."""

    @mcp.tool(tags={"project_management"})
    async def get_project_status(
        path: Optional[str] = Field(description="Project root path.", default=None)
    ) -> Dict[str, Any]:
        """Reads the current project state from tasks.json and progress.json."""
        root = path or DEFAULT_WORKSPACE
        status = {}

        for filename, key in [("tasks.json", "tasks"), ("progress.json", "progress")]:
            fpath = os.path.join(root, filename)
            if os.path.exists(fpath):
                try:
                    with open(fpath, "r") as f:
                        status[key] = json.load(f)
                except Exception as e:
                    status[key] = f"Error reading {filename}: {e}"
            else:
                status[key] = "Not found"
        return status

    @mcp.tool(tags={"project_management"})
    async def update_task_status(
        task_id: str,
        status: str,
        result: Optional[str] = None,
        path: Optional[str] = Field(description="Project root path.", default=None),
    ) -> str:
        """Updates the status and result of a specific task in tasks.json."""
        root = path or DEFAULT_WORKSPACE
        fpath = os.path.join(root, "tasks.json")
        if not os.path.exists(fpath):
            return "Error: tasks.json not found."

        try:
            with open(fpath, "r") as f:
                task_list = TaskList.model_validate_json(f.read())

            found = False
            for phase in task_list.phases:
                for task in phase.tasks:
                    if task.id == task_id:
                        task.status = TaskStatus(status)
                        if result:
                            task.result = result
                        found = True
                        break
                if found:
                    break

            if not found:
                return f"Error: Task {task_id} not found."

            with open(fpath, "w") as f:
                f.write(task_list.model_dump_json(indent=2))
            return f"Task {task_id} updated to {status}."
        except Exception as e:
            return f"Error updating task: {e}"


def register_workspace_management_tools(mcp: FastMCP):
    @mcp.tool(tags={"workspace_management"})
    async def setup_workspace(
        yml_path: str = Field(description="Path to the workspace.yml file."),
    ) -> GitResult:
        """Sets up the entire workspace, clones repos, and organizes subdirectories."""
        git = get_git_instance()
        return git.setup_from_yaml(yml_path)

    @mcp.tool(tags={"workspace_management"})
    async def install_projects(
        threads: Optional[int] = Field(description="Parallel workers.", default=None),
        extra: str = Field(description="Install group (e.g. 'all').", default="all"),
    ) -> List[GitResult]:
        """Bulk installs Python projects defined in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.install_projects(extra=extra)
        return git.generate_markdown_summary("Installation", results)

    @mcp.tool(tags={"workspace_management"})
    async def build_projects(
        threads: Optional[int] = Field(description="Parallel workers.", default=None),
    ) -> List[GitResult]:
        """Bulk builds Python projects defined in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.build_projects()
        return git.generate_markdown_summary("Build", results)

    @mcp.tool(tags={"workspace_management"})
    async def validate_projects(
        type: str = Field(
            description="Validation type: 'agent', 'mcp', or 'all'.", default="all"
        ),
        threads: Optional[int] = Field(description="Parallel workers.", default=None),
    ) -> List[GitResult]:
        """Bulk validates agent/MCP servers in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.validate_projects(type=type)
        return git.generate_markdown_summary("Validation", results)

    @mcp.tool(tags={"workspace_management"})
    async def generate_workspace_template(
        target_path: str = Field(description="Path where to save the template."),
        use_default: bool = Field(
            description="Use the pre-filled package template.", default=True
        ),
    ) -> GitResult:
        """Generates a new workspace.yml template."""
        git = Git()
        return git.generate_workspace_template(
            target_path=target_path, use_default=use_default
        )

    @mcp.tool(tags={"workspace_management"})
    async def save_workspace_config(
        yaml_path: str = Field(description="Target YAML path."),
        config_dict: Dict[str, Any] = Field(
            description="Dictionary representation of WorkspaceConfig."
        ),
    ) -> GitResult:
        """Saves a WorkspaceConfig to YAML. Useful for programmatically updating the workspace."""
        try:
            config = WorkspaceConfig(**config_dict)
            git = Git()
            return git.save_workspace_config(yaml_path=yaml_path, config=config)
        except Exception as e:
            return GitResult(
                status="error", data="", error={"message": str(e), "code": 1}
            )

    @mcp.tool(tags={"workspace_management"})
    async def maintain_workspace(
        part: str = Field(
            description="Version part to bump (major, minor, patch).", default="patch"
        ),
        phase: int = Field(description="Starting phase number.", default=1),
        dry_run: bool = Field(description="Perform a dry run.", default=False),
    ) -> List[GitResult]:
        """Runs the maintenance lifecycle across all projects in the workspace."""
        git = get_git_instance()
        results = git.maintain_projects(part=part, start_phase=phase, dry_run=dry_run)
        return git.generate_markdown_summary("Maintenance", results)


def register_visualization_tools(mcp: FastMCP):
    @mcp.tool(tags={"visualization"})
    async def get_workspace_tree(
        yml_path: Optional[str] = Field(
            description="Path to workspace.yml.", default=None
        )
    ) -> str:
        """Generates an ASCII tree of the workspace structure."""
        git = get_git_instance()
        path = yml_path or os.path.join(git.path, DEFAULT_WORKSPACE_YML)
        return git.generate_workspace_tree(path)

    @mcp.tool(tags={"visualization"})
    async def get_workspace_mermaid(
        yml_path: Optional[str] = Field(
            description="Path to workspace.yml.", default=None
        )
    ) -> str:
        """Generates a Mermaid diagram of the workspace structure."""
        git = get_git_instance()
        path = yml_path or os.path.join(git.path, DEFAULT_WORKSPACE_YML)
        return git.generate_workspace_mermaid(path)

    @mcp.tool(tags={"visualization"})
    async def generate_agents_documentation(
        target_path: Optional[str] = Field(
            description="Target path for AGENTS.md.", default=None
        )
    ) -> GitResult:
        """Generates an AGENTS.md catalog of discovered projects."""
        git = get_git_instance()
        return git.generate_agents_md(target_path=target_path)


def register_prompts(mcp: FastMCP):
    @mcp.prompt
    def validate_repositories() -> str:
        """
        Generates a prompt for validating projects and fixing errors
        """
        return (
            "I have several agents I have built in my agent-packages. "
            "Please validate and generate a report under Workspace/validation_report.md. "
            "Can we create a plan to resolve all those errors found for all projects? "
            "Once we resolve all errors, let's re-run and validate all issues were resolved"
        )


def get_mcp_instance() -> tuple[Any, Any, Any, Any]:
    """Initialize the MCP instance, args, and middlewares."""
    load_dotenv(find_dotenv())

    args, mcp, middlewares = create_mcp_server(
        name="RepositoryManager",
        version=__version__,
        instructions="Expert tool for managing hierarchical Git workspaces, engineering bulk operations, and documentation.",
    )

    registered_tags = []
    if to_boolean(os.getenv("MISCTOOL", "True")):
        register_misc_tools(mcp)
        registered_tags.append("misc")

    if to_boolean(os.getenv("GIT_OPERATIONSTOOL", "True")):
        register_git_operations_tools(mcp)
        registered_tags.append("git_operations")

    if to_boolean(os.getenv("WORKSPACE_MANAGEMENTTOOL", "True")):
        register_workspace_management_tools(mcp)
        register_project_management_tools(mcp)
        register_misc_tools(mcp)
        registered_tags.append("workspace_management")

    if to_boolean(os.getenv("VISUALIZATIONTOOL", "True")):
        register_visualization_tools(mcp)
        registered_tags.append("visualization")

    register_prompts(mcp)
    registered_tags.append("prompts")

    for mw in middlewares:
        mcp.add_middleware(mw)

    return mcp, args, middlewares, registered_tags


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags = get_mcp_instance()
    print(f"{args.name or 'repository-manager'} MCP v{__version__}", file=sys.stderr)
    print("\nStarting MCP Server", file=sys.stderr)
    print(f"  Transport: {args.transport.upper()}", file=sys.stderr)
    print(f"  Auth: {args.auth_type}", file=sys.stderr)
    print(f"  Dynamic Tags Loaded: {len(set(registered_tags))}", file=sys.stderr)

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
