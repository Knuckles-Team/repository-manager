#!/usr/bin/env python
import warnings

# Filter RequestsDependencyWarning early to prevent log spam
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    try:
        from requests.exceptions import RequestsDependencyWarning

        warnings.filterwarnings("ignore", category=RequestsDependencyWarning)
    except ImportError:
        pass

# General urllib3/chardet mismatch warnings
warnings.filterwarnings("ignore", message=".*urllib3.*or chardet.*")
warnings.filterwarnings("ignore", message=".*urllib3.*or charset_normalizer.*")

import os
import sys
from typing import Any

from agent_utilities.base_utilities import to_boolean, to_integer
from agent_utilities.mcp_utilities import (
    create_mcp_server,
    ctx_confirm_destructive,
    ctx_progress,
)
from dotenv import find_dotenv, load_dotenv
from fastmcp import Context, FastMCP
from fastmcp.utilities.logging import get_logger
from pydantic import Field
from starlette.requests import Request
from starlette.responses import JSONResponse

from repository_manager.models import (
    GitResult,
    ValidationReport,
    WorkspaceConfig,
)
from repository_manager.repository_manager import Git

__version__ = "1.6.0"


DEFAULT_WORKSPACE = os.environ.get("REPOSITORY_MANAGER_WORKSPACE", "/workspace")
DEFAULT_THREADS = to_integer(os.environ.get("REPOSITORY_MANAGER_THREADS", "12"))
DEFAULT_WORKSPACE_YML = os.environ.get("WORKSPACE_YML", "workspace.yml")

logger = get_logger("RepositoryManagerServer")


def get_git_instance(path: str | None = None, threads: int | None = None) -> Git:
    """Helper to get a Git instance with workpace YAML loaded."""
    workspace_path = path or DEFAULT_WORKSPACE
    git = Git(path=workspace_path, threads=threads or DEFAULT_THREADS)

    yml_path = os.path.join(workspace_path, DEFAULT_WORKSPACE_YML)
    if not os.path.exists(yml_path):
        # Fallback to the packaged version if the workspace-relative one isn't found
        from repository_manager.repository_manager import (
            DEFAULT_WORKSPACE_YML as PACKAGED_YML,
        )

        yml_path = PACKAGED_YML

    if os.path.exists(yml_path):
        git.load_projects_from_yaml(yml_path)
    else:
        logger.warning(f"No workspace.yml found at {yml_path}")

    return git


def register_misc_tools(mcp: FastMCP):
    """Register miscellaneous tools like health check."""

    async def health_check(_request: Request) -> JSONResponse:
        return JSONResponse({"status": "OK"})


def register_git_operations_tools(mcp: FastMCP):
    @mcp.tool(tags={"workspace_management", "project_manager", "devops_engineer"})
    async def git_action(
        command: str = Field(
            description="The Git command to execute (e.g., 'git status')"
        ),
        path: str | None = Field(description="Path to execute in.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult:
        """Executes an arbitrary Git command."""
        git = get_git_instance(path=path)
        return git.git_action(command=command, path=path)

    @mcp.tool(
        tags={
            "devops_engineer",
            "workspace_management",
            "project_management",
            "git_operations",
        }
    )
    async def get_workspace_projects(
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[str]:
        """Returns a list of project URLs defined in the workspace."""
        git = get_git_instance()
        return git.get_workspace_projects()

    @mcp.tool(tags={"git_operations", "project_manager", "devops_engineer"})
    async def clone_projects(
        threads: int | None = Field(description="Parallel workers.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Clones repositories. Defaults to all in workspace.yml."""
        git = get_git_instance(threads=threads)
        results = git.clone_projects()
        return git.generate_markdown_summary("Clone", results)

    @mcp.tool(tags={"git_operations", "project_manager", "devops_engineer"})
    async def pull_projects(
        threads: int | None = Field(description="Parallel workers.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        await ctx_progress(ctx, 0, 100)
        """Pulls updates for all projects in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.pull_projects()
        await ctx_progress(ctx, 100, 100)
        return git.generate_markdown_summary("Pull", results)

    @mcp.tool(tags={"git_operations", "project_manager", "devops_engineer"})
    async def push_projects(
        threads: int | None = Field(description="Parallel workers.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        await ctx_progress(ctx, 0, 100)
        """Pushes updates and tags for all projects in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.push_projects()
        await ctx_progress(ctx, 100, 100)
        return git.generate_markdown_summary("Push", results)

    @mcp.tool(tags={"git_operations", "project_manager", "devops_engineer"})
    async def phased_git_push(
        phase: int | None = Field(
            description="The starting phase. Default is 1.", default=1
        ),
        target_project: str | None = Field(
            description="Optional specific project to push.", default=None
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        await ctx_progress(ctx, 0, 100)
        """Executes a phased git push workflow based on workspace.yml."""
        git = get_git_instance()
        results = git.phased_push(start_phase=phase or 1, project_filter=target_project)
        await ctx_progress(ctx, 100, 100)
        return git.generate_markdown_summary("Phased Push", results)


def register_workspace_management_tools(mcp: FastMCP):
    """Register tools for core workspace setup and organization."""

    @mcp.tool(tags={"workspace_management"})
    async def setup_workspace(
        yml_path: str = Field(description="Path to the workspace.yml file."),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult:
        """Sets up the entire workspace, clones repos, and organizes subdirectories."""
        git = get_git_instance()
        return git.setup_from_yaml(yml_path)


def register_project_management_tools(mcp: FastMCP):
    """Register tools for the autonomous project harness."""

    @mcp.tool(tags={"workspace_management"})
    async def install_projects(
        threads: int | None = Field(description="Parallel workers.", default=None),
        extra: str = Field(description="Install group (e.g. 'all').", default="all"),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Bulk installs Python projects defined in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.install_projects(extra=extra)
        return git.generate_markdown_summary("Installation", results)

    @mcp.tool(tags={"workspace_management"})
    async def build_projects(
        threads: int | None = Field(description="Parallel workers.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Bulk builds Python projects defined in the workspace."""
        git = get_git_instance(threads=threads)
        results = git.build_projects()
        return git.generate_markdown_summary("Build", results)

    @mcp.tool(tags={"workspace_management"})
    async def validate_projects(
        type: str = Field(
            description="Validation type: 'agent', 'mcp', or 'all'.", default="all"
        ),
        threads: int | None = Field(description="Parallel workers.", default=None),
        output_dir: str | None = Field(
            description="Directory to write the validation-reports-<timestamp>/ output. Defaults to the workspace root.",
            default=None,
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> ValidationReport:
        """Bulk validates agent/MCP servers in the workspace.

        Results are written incrementally to a structured directory:
        ``validation-reports-<timestamp>/<repo-name>-results/<scan-type>.md``
        """
        git = get_git_instance(threads=threads)
        report = git.validate_projects(type=type, output_dir=output_dir)
        return report

    @mcp.tool(tags={"workspace_management"})
    async def generate_workspace_template(
        target_path: str = Field(description="Path where to save the template."),
        use_default: bool = Field(
            description="Use the pre-filled package template.", default=True
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult:
        """Generates a new workspace.yml template."""
        git = get_git_instance()
        return git.generate_workspace_template(
            target_path=target_path, use_default=use_default
        )

    @mcp.tool(tags={"workspace_management"})
    async def save_workspace_config(
        yaml_path: str = Field(description="Target YAML path."),
        config_dict: dict[str, Any] = Field(
            description="Dictionary representation of WorkspaceConfig."
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult:
        """Saves a WorkspaceConfig to YAML."""
        try:
            config = WorkspaceConfig(**config_dict)
            git = get_git_instance()
            return git.save_workspace_config(yaml_path=yaml_path, config=config)
        except Exception as e:
            from repository_manager.models import GitError

            return GitResult(
                status="error", data="", error=GitError(message=str(e), code=1)
            )

    @mcp.tool(tags={"workspace_management"})
    async def maintain_workspace(
        part: str = Field(
            description="Version part to bump (major, minor, patch).", default="patch"
        ),
        phase: int = Field(description="Starting phase number.", default=1),
        dry_run: bool = Field(description="Perform a dry run.", default=False),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Runs the maintenance lifecycle across all projects in the workspace."""
        git = get_git_instance()
        results = git.maintain_projects(part=part, start_phase=phase, dry_run=dry_run)
        return git.generate_markdown_summary("Maintenance", results)

    @mcp.tool(tags={"workspace_management", "git_operations"})
    async def push_projects_phased(
        phase: int = Field(description="Starting phase number.", default=1),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Executes the phased git push workflow across all projects in the workspace."""
        git = get_git_instance()
        results = git.phased_push(start_phase=phase)
        return git.generate_markdown_summary("Phased Push", results)


def register_visualization_tools(mcp: FastMCP):
    @mcp.tool(tags={"visualization"})
    async def get_workspace_tree(
        yml_path: str | None = Field(
            description="Path to workspace.yml.", default=None
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Generates an ASCII tree of the workspace structure."""
        git = get_git_instance()
        path = yml_path or os.path.join(git.path, DEFAULT_WORKSPACE_YML)
        return git.generate_workspace_tree(path)

    @mcp.tool(tags={"visualization"})
    async def get_workspace_mermaid(
        yml_path: str | None = Field(
            description="Path to workspace.yml.", default=None
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Generates a Mermaid diagram of the workspace structure."""
        git = get_git_instance()
        path = yml_path or os.path.join(git.path, DEFAULT_WORKSPACE_YML)
        return git.generate_workspace_mermaid(path)

    @mcp.tool(tags={"visualization"})
    async def generate_agents_documentation(
        target_path: str | None = Field(
            description="Target path for AGENTS.md.", default=None
        ),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> GitResult:
        """Generates an AGENTS.md catalog of discovered projects."""
        git = get_git_instance()
        return git.generate_agents_md(target_path=target_path)


def register_graph_tools(mcp: FastMCP):
    @mcp.tool(tags={"graph_intelligence"})
    async def graph_build(
        path: str | None = Field(description="Workspace path.", default=None),
        multimodal: bool = Field(
            description="Enable LLM multimodal rationale pass.", default=False
        ),
        incremental: bool = Field(description="Use incremental parsing.", default=True),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict[str, Any]:
        """Builds or synchronizes the Hybrid Workspace Graph (NetworkX + Ladybug)."""
        git = get_git_instance(path=path)
        if hasattr(git, "config") and git.config:
            if not git.config.graph:
                from repository_manager.models import GraphConfig

                git.config.graph = GraphConfig(
                    enabled=True, multimodal=multimodal, incremental=incremental
                )
            else:
                git.config.graph.multimodal = multimodal
                git.config.graph.incremental = incremental

        report = git.ensure_graph()
        if not report:
            return {
                "status": "error",
                "message": "Graph configuration disabled or failed.",
            }

        return report.model_dump()

    @mcp.tool(tags={"graph_intelligence"})
    async def graph_query(
        query: str = Field(
            description="Cypher query or semantic string to search the graph."
        ),
        mode: str = Field(
            description="Query mode: 'semantic' (vector), 'structural' (Cypher), or 'hybrid'.",
            default="hybrid",
        ),
        path: str | None = Field(description="Workspace path.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict[str, Any]]:
        """Queries the Hybrid Graph using vector similarity or Cypher structure."""
        git = get_git_instance(path=path)
        return await git.graph_query(query, mode=mode, path=path)

    @mcp.tool(tags={"graph_intelligence"})
    async def graph_path(
        source_id: str = Field(description="Source node ID (Symbol or File)."),
        target_id: str = Field(description="Target node ID."),
        path: str | None = Field(description="Workspace path.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[str]:
        """Finds the shortest path between two symbols across the workspace graph."""
        git = get_git_instance(path=path)
        return git.graph_path(source_id, target_id, path=path)

    @mcp.tool(tags={"graph_intelligence"})
    async def graph_status(
        path: str | None = Field(description="Workspace path.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> dict[str, Any]:
        """Returns the current status of the workspace graph."""
        git = get_git_instance(path=path)
        return git.graph_status(path=path)

    @mcp.tool(tags={"graph_intelligence"})
    async def graph_reset(
        path: str | None = Field(description="Workspace path.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> str:
        """Purges the graph database and forces a clean rebuild."""
        if not await ctx_confirm_destructive(ctx, "graph reset"):
            return "Operation cancelled by user"
        await ctx_progress(ctx, 0, 100)
        git = get_git_instance(path=path)
        return git.graph_reset(path=path)

    @mcp.tool(tags={"graph_intelligence"})
    async def graph_impact(
        symbol: str = Field(description="The code symbol to find impact for."),
        group_name: str | None = Field(description="Group filter.", default=None),
        path: str | None = Field(description="Workspace path.", default=None),
        ctx: Context = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[dict[str, Any]]:
        """Calculates multi-repo impact for a symbol using the GraphEngine."""
        git = get_git_instance(path=path)
        return await git.graph_impact(symbol, group_name=group_name, path=path)


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
        registered_tags.append("workspace_management")

    if to_boolean(os.getenv("GRAPH_INTELLIGENCETOOL", "True")):
        register_graph_tools(mcp)
        registered_tags.append("graph_intelligence")

    if to_boolean(os.getenv("VISUALIZATIONTOOL", "True")):
        register_visualization_tools(mcp)
        registered_tags.append("visualization")

    for mw in middlewares:
        mcp.add_middleware(mw)

    return mcp, args, middlewares, registered_tags


def mcp_server() -> None:
    mcp, args, middlewares, registered_tags = get_mcp_instance()
    print(f"{'repository-manager'} MCP v{__version__}", file=sys.stderr)
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
