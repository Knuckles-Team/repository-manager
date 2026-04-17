#!/usr/bin/env python


"""
A command-line tool for managing Git repositories, supporting cloning and pulling
multiple repositories in parallel using Python's multiprocessing capabilities.
"""

import subprocess
import os
import re
import sys
import argparse
import json

__version__ = "1.3.55"
import concurrent.futures
import datetime
import yaml
import shutil
import select
import signal
from pathlib import Path
from typing import List, Dict, Optional, Any
from agent_utilities.base_utilities import get_library_file_path
from agent_utilities.base_utilities import to_boolean

try:
    from universal_skills.skill_utilities import get_universal_skills_path
    from skill_graphs.skill_graph_utilities import get_skill_graphs_path
except ImportError:
    get_universal_skills_path = None
    get_skill_graphs_path = None

from repository_manager.models import (
    GitResult,
    GitError,
    GitMetadata,
    ReadmeResult,
    WorkspaceConfig,
    SubdirectoryConfig,
)
from repository_manager.graph.models import GraphReport

from importlib.resources import files
from agent_utilities.base_utilities import get_logger

logger = get_logger("RepositoryManager")


def get_packaged_file_path(package: str, file: str) -> str:
    """Robustly find a file in a package using importlib.resources."""
    try:

        path = files(package).joinpath(file)
        if path.exists():
            return str(path)
    except Exception:
        pass

    local_path = os.path.join(os.path.dirname(__file__), file)
    if os.path.exists(local_path):
        return local_path

    return get_library_file_path(file=file)


# Robust environment variable retrieval with empty string fallbacks
_raw_workspace = os.getenv("REPOSITORY_MANAGER_WORKSPACE", "")
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.path.abspath(
    os.path.expanduser(_raw_workspace if _raw_workspace else "~/Workspace")
)

_raw_yml = os.getenv("WORKSPACE_YML", "")
DEFAULT_WORKSPACE_YML = (
    _raw_yml
    if _raw_yml
    else get_packaged_file_path("repository_manager", "workspace.yml")
)

_raw_threads = os.getenv("REPOSITORY_MANAGER_THREADS", "")
DEFAULT_REPOSITORY_MANAGER_THREADS = int(
    _raw_threads if _raw_threads and _raw_threads.isdigit() else "12"
)

_raw_branch = os.getenv("REPOSITORY_MANAGER_DEFAULT_BRANCH", "")
DEFAULT_REPOSITORY_MANAGER_DEFAULT_BRANCH = to_boolean(
    _raw_branch if _raw_branch else "False"
)


class Git:
    """A class to handle Git operations such as cloning and pulling repositories."""

    def __init__(
        self,
        path: str = None,
        threads: int = None,
        set_to_default_branch: bool = False,
        capture_output: bool = False,
        report_path: str = None,
    ):
        """Initialize the Git class with default settings."""
        self.path = path or DEFAULT_REPOSITORY_MANAGER_WORKSPACE
        self.report_path = report_path
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:
                pass

        self.project_map = {}
        self.config = None
        self.threads = threads or DEFAULT_REPOSITORY_MANAGER_THREADS
        self.set_to_default_branch = set_to_default_branch
        self.capture_output = capture_output
        self.maximum_threads = 36
        if threads:
            self.set_threads(threads=threads)

    def _get_package_manager(self, path: str) -> str:
        """Determines the appropriate package manager for a given path."""
        if os.path.exists(os.path.join(path, "pnpm-lock.yaml")):
            return "pnpm"
        if os.path.exists(os.path.join(path, "yarn.lock")):
            return "yarn"
        return "npm"

    def setup_from_yaml(self, yaml_path: str) -> GitResult:
        """Sets up the workspace structure from a YAML file."""
        abs_yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
        if not os.path.exists(abs_yaml_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message=f"YAML not found: {abs_yaml_path}", code=1),
            )

        if not self.load_projects_from_yaml(abs_yaml_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message="Failed to load YAML", code=1),
            )

        logger.info(f"Creating workspace structure at {self.path}...")
        os.makedirs(self.path, exist_ok=True)

        for _, project_path in self.project_map.items():
            os.makedirs(os.path.dirname(project_path), exist_ok=True)

        logger.info("Syncing repositories (Clone/Pull)...")
        results = []
        for url, project_path in self.project_map.items():
            if os.path.exists(project_path):
                results.append(self.pull_project(project_path))
            else:
                results.append(self.clone_repository(url, project_path))

        return GitResult(
            status="success",
            data=f"Workspace setup completed at {self.path}",
            metadata=GitMetadata(
                command="setup_workspace",
                workspace=self.path,
                return_code=0,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                + "Z",
            ),
        )

    def generate_workspace_tree(self, yaml_path: str = None) -> str:
        """Generates an ASCII tree of the workspace defined in YAML."""
        if yaml_path:
            self.load_projects_from_yaml(yaml_path)

        if not hasattr(self, "config") or not self.config:
            return "No workspace configuration loaded."

        lines = [f"Workspace: {self.config.name}", f"Root: {self.path}", ""]

        def _build_tree(subdirs, indent=""):
            for name, data in subdirs.items():
                lines.append(f"{indent}├── {name}/")
                new_indent = indent + "│   "
                for repo in data.repositories:
                    repo_name = repo.url.split("/")[-1].replace(".git", "")
                    lines.append(f"{new_indent}└── {repo_name} ({repo.url})")
                if data.subdirectories:
                    _build_tree(data.subdirectories, new_indent)

        _build_tree(self.config.subdirectories)
        return "\n".join(lines)

    def generate_workspace_mermaid(self, yaml_path: str = None) -> str:
        """Generates a Mermaid diagram of the workspace defined in YAML."""
        if yaml_path:
            self.load_projects_from_yaml(yaml_path)

        if not hasattr(self, "config") or not self.config:
            return "No workspace configuration loaded."

        lines = ["graph TD", f'    Root["{self.config.name}"]']

        def _build_mermaid(subdirs, parent_node):
            for name, data in subdirs.items():
                node_id = name.replace("-", "_")
                lines.append(f'    {parent_node} --> {node_id}["{name}/"]')
                for repo in data.repositories:
                    repo_name = repo.url.split("/")[-1].replace(".git", "")
                    repo_id = repo_name.replace("-", "_")
                    lines.append(f'    {node_id} --> {repo_id}["{repo_name}"]')
                if data.subdirectories:
                    _build_mermaid(data.subdirectories, node_id)

        _build_mermaid(self.config.subdirectories, "Root")
        return "\n".join(lines)

    def generate_agents_md(self, target_path: str = None) -> GitResult:
        """
        Generates/Updates the AGENTS.md catalog in the workspace root.
        """
        return self.generate_agents_documentation(target_path)

    def get_project_map(self) -> Dict[str, str]:
        """
        Returns the mapping of repository URLs to their local project paths.
        Ensures paths are absolute and expanded.
        """
        return {
            url: os.path.abspath(os.path.expanduser(p))
            for url, p in self.project_map.items()
        }

    def get_workspace_projects(self) -> List[str]:
        """Returns a list of project basenames (e.g. 'genius-agent') defined in the workspace."""
        return [os.path.basename(p) for p in self.project_map.values()]

    async def graph_query(
        self, query: str, mode: str = "hybrid", path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Queries the Hybrid Graph using vector similarity or Cypher structure."""
        from repository_manager.graph.engine import GraphEngine

        root = self._resolve_path(path)
        engine = GraphEngine(root)
        # Note: We assume the graph was already synchronized via ensure_graph
        return await engine.query(query, mode=mode)

    def graph_path(
        self, source_id: str, target_id: str, path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Finds the shortest path between two symbols across the workspace graph."""
        from repository_manager.graph.engine import GraphEngine

        root = self._resolve_path(path)
        engine = GraphEngine(root)
        edges = engine.find_path(source_id, target_id)
        return [e.model_dump() for e in edges]

    def graph_status(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Returns the current status of the workspace graph."""
        from repository_manager.graph.engine import GraphEngine

        root = self._resolve_path(path)
        engine = GraphEngine(root)
        return engine.get_stats()

    def graph_reset(self, path: Optional[str] = None) -> str:
        """Purges the graph database and Forces a clean rebuild on next build."""
        from repository_manager.graph.engine import GraphEngine

        root = self._resolve_path(path)
        engine = GraphEngine(root)
        engine.reset_graph()
        return "Graph database purged successfully."

    async def graph_impact(
        self, symbol: str, group_name: Optional[str] = None, path: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Calculates multi-repo impact for a symbol using the GraphEngine."""
        from repository_manager.graph.engine import GraphEngine

        root = self._resolve_path(path)
        engine = GraphEngine(root)
        nodes = await engine.query_impact(symbol, group_name)
        return [n.model_dump() for n in nodes]

    def _resolve_path(self, path: str = None) -> str:
        """
        Resolve the path to an absolute path.
        If path is None, returns self.path.
        If path is absolute, returns it.
        If path is relative, joins it with self.path.
        """
        if path is None:
            return os.path.abspath(self.path)

        if os.path.isabs(path):
            return os.path.abspath(path)

        return os.path.abspath(os.path.join(self.path, path))

    def install_projects(
        self, extra: str = "all", threads: int = None, report: bool = True
    ) -> List[GitResult]:
        """Bulk installs Python projects in the workspace."""
        threads = threads or self.threads
        if not self.project_map:
            logger.warning("No projects to install.")
            return []

        CORE_PROJECTS = [
            "universal-skills",
            "skill-graphs",
            "agent-webui",
            "agent-utilities",
        ]

        logger.info(
            f"Installing {len(self.project_map)} projects in parallel ({threads} threads)..."
        )
        results = []
        futures = []

        # 1. Sequential Install for Core Projects (to avoid versioning race conditions)
        core_paths = []
        other_paths = []

        for url, path in self.project_map.items():
            pkg_name = path.split("/")[-1]
            if pkg_name in CORE_PROJECTS:
                core_paths.append(path)
            else:
                other_paths.append(path)

        # Sort core_paths based on CORE_PROJECTS order
        core_paths.sort(
            key=lambda p: (
                CORE_PROJECTS.index(p.split("/")[-1])
                if p.split("/")[-1] in CORE_PROJECTS
                else 999
            )
        )

        logger.info(f"Installing {len(core_paths)} core projects sequentially...")
        for path in core_paths:
            is_python = os.path.exists(
                os.path.join(path, "pyproject.toml")
            ) or os.path.exists(os.path.join(path, "setup.py"))
            is_node = os.path.exists(os.path.join(path, "package.json"))

            if is_node:
                pm = self._get_package_manager(path)
                results.append(self.git_action(f"{pm} install", path=path))
            if is_python:
                results.append(
                    self.git_action(f"pip install -e '.[{extra}]'", path=path)
                )

        # 2. Parallel Install for the rest
        logger.info(f"Installing {len(other_paths)} remaining projects in parallel...")
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            for path in other_paths:
                is_python = os.path.exists(
                    os.path.join(path, "pyproject.toml")
                ) or os.path.exists(os.path.join(path, "setup.py"))
                is_node = os.path.exists(os.path.join(path, "package.json"))

                if not is_python and not is_node:
                    results.append(
                        GitResult(
                            status="skipped",
                            data="Skipped (Not a Python or Node project)",
                            metadata=GitMetadata(
                                command="install",
                                workspace=path,
                                return_code=0,
                                timestamp=datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat()
                                + "Z",
                            ),
                        )
                    )
                    continue

                if is_node:
                    pm = self._get_package_manager(path)
                    futures.append(
                        executor.submit(self.git_action, f"{pm} install", path=path)
                    )
                if is_python:
                    futures.append(
                        executor.submit(
                            self.git_action, f"pip install -e '.[{extra}]'", path=path
                        )
                    )

            results.extend(
                [f.result() for f in concurrent.futures.as_completed(futures)]
            )

            successes = [r for r in results if r.status == "success"]
            failures = [r for r in results if r.status == "error"]

            print("\n" + "=" * 50)
            print("INSTALLATION SUMMARY")
            print(
                f"Total: {len(results)} | Success: {len(successes)} ✅ | Failure: {len(failures)} ❌"
            )
            if failures:
                print("\nFailures:")
                for r in failures:
                    pkg = r.metadata.workspace.split("/")[-1]
                    print(f"- {pkg}: {r.error.message if r.error else r.data}")
            print("=" * 50 + "\n")

            report_md = "# INSTALLATION SUMMARY\n"
            report_md += (
                f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
            )
            report_md += f"**Total:** {len(results)} | **Success:** {len(successes)} ✅ | **Failure:** {len(failures)} ❌\n\n"

            if successes:
                report_md += "## Successes ✅\n"
                for r in successes:
                    pkg = r.metadata.workspace.split("/")[-1]
                    report_md += f"- **{pkg}**: Installation success\n"

            if failures:
                report_md += "\n## Failures ❌\n"
                for r in failures:
                    pkg = r.metadata.workspace.split("/")[-1]
                    error_msg = r.error.message if r.error else r.data
                    report_md += f"- **{pkg}**: {error_msg}\n"

            if self.report_path and report:
                self._export_report(report_md, "install_report.md")

            return results

    def build_projects(self, threads: int = None) -> List[GitResult]:
        """Bulk builds Python and Node.js projects in the workspace."""
        threads = threads or self.threads
        if not self.project_map:
            logger.warning("No projects to build.")
            return []

        logger.info(
            f"Building {len(self.project_map)} projects in parallel ({threads} threads)..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for url, path in self.project_map.items():
                if os.path.exists(os.path.join(path, "package.json")):
                    pm = self._get_package_manager(path)
                    cmd = f"{pm} install && {pm} run build"
                else:
                    cmd = "python3 -m build"
                futures.append(executor.submit(self.git_action, cmd, path=path))
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    def validate_projects(
        self, type: str = "all", threads: int = None
    ) -> List[GitResult]:
        """Bulk validates agent/MCP servers using various modes (help, static, runtime)."""
        threads = threads or self.threads
        if not self.project_map:
            logger.warning("No projects to validate.")
            return []

        logger.info(
            f"Validating {len(self.project_map)} projects (mode={type}) in parallel ({threads} threads)..."
        )

        run_all = type in ["all", "flat", "graph"]
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:

            agent_targets = []
            for url, path in self.project_map.items():
                pkg_name = url.split("/")[-1].replace(".git", "")
                pkg_underscore = pkg_name.replace("-", "_")

                pkg_dir = os.path.join(path, pkg_underscore)
                if not os.path.exists(pkg_dir):
                    for sub in ["src", "agent"]:
                        candidate = os.path.join(path, sub, pkg_underscore)
                        if os.path.exists(candidate):
                            pkg_dir = candidate
                            break

                agent_file = None
                if os.path.exists(os.path.join(pkg_dir, "agent_server.py")):
                    agent_file = "agent_server"

                is_mcp = os.path.exists(os.path.join(pkg_dir, "mcp_server.py"))
                is_graph = os.path.exists(os.path.join(pkg_dir, "graph_config.py"))

                is_agent_suite = is_mcp or agent_file or is_graph

                if pkg_name == "pipelines":
                    agent_targets.append(
                        {
                            "name": pkg_name,
                            "path": path,
                            "skip_reason": "Skipped (Not a Python project)",
                        }
                    )
                    continue

                if not is_agent_suite:
                    agent_targets.append(
                        {
                            "name": pkg_name,
                            "path": path,
                            "skip_reason": "Skipped (Not an Agent project)",
                        }
                    )
                    continue

                if type == "flat" and (is_graph or not agent_file):
                    continue
                if type == "graph" and not is_graph:
                    continue

                agent_targets.append(
                    {
                        "name": pkg_name,
                        "pkg": pkg_underscore,
                        "path": path,
                        "pkg_dir": pkg_dir,
                        "file": agent_file,
                        "is_mcp": is_mcp,
                        "is_graph": is_graph,
                    }
                )

            if run_all or type == "installation" or type == "all":
                install_results = self.install_projects(report=False)
                results.extend(install_results)

            if run_all or type == "version-sync" or type == "all":
                bump_results = []

                for url, path in self.project_map.items():
                    repo_name = Path(path).name
                    if (Path(path) / ".bumpversion.cfg").exists():
                        bump_results.append(
                            self.bump_version(
                                "patch", allow_dirty=True, path=path, dry_run=True
                            )
                        )
                    else:
                        reason = (
                            "Skipped (Not a Python project)"
                            if repo_name == "pipelines"
                            else "Skipped (Missing .bumpversion.cfg)"
                        )
                        bump_results.append(
                            GitResult(
                                status="skipped",
                                data=reason,
                                metadata=GitMetadata(
                                    command="bump2version",
                                    workspace=path,
                                    return_code=0,
                                    timestamp=datetime.datetime.now(
                                        datetime.timezone.utc
                                    ).isoformat()
                                    + "Z",
                                ),
                            )
                        )
                results.extend(bump_results)

            fast_futures = []
            skip_ts = datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z"

            for target in agent_targets:
                if "skip_reason" in target:
                    reason = target["skip_reason"]

                    if run_all or type in ["mcp", "all"]:
                        results.append(
                            GitResult(
                                status="skipped",
                                data=reason,
                                metadata=GitMetadata(
                                    command="mcp_server --help",
                                    workspace=target["path"],
                                    return_code=0,
                                    timestamp=skip_ts,
                                ),
                            )
                        )
                    if run_all or type in ["agent", "all"]:
                        results.append(
                            GitResult(
                                status="skipped",
                                data=reason,
                                metadata=GitMetadata(
                                    command="agent_server --help",
                                    workspace=target["path"],
                                    return_code=0,
                                    timestamp=skip_ts,
                                ),
                            )
                        )
                    if run_all or type in ["static-analysis", "all"]:
                        results.append(
                            GitResult(
                                status="skipped",
                                data=reason,
                                metadata=GitMetadata(
                                    command="static_check",
                                    workspace=target["path"],
                                    return_code=0,
                                    timestamp=skip_ts,
                                ),
                            )
                        )
                    continue

                if (run_all or type == "mcp") and target.get("is_mcp"):
                    cmd = f"python3 -m {target['pkg']}.mcp_server --help"
                    fast_futures.append(
                        executor.submit(self._check_help, cmd, path=target["path"])
                    )
                elif run_all or type == "mcp":

                    results.append(
                        GitResult(
                            status="skipped",
                            data="Skipped (Not an MCP server)",
                            metadata=GitMetadata(
                                command="mcp_server --help",
                                workspace=target["path"],
                                return_code=0,
                                timestamp=skip_ts,
                            ),
                        )
                    )

                if (run_all or type == "agent") and target.get("file"):
                    cmd = f"python3 -m {target['pkg']}.{target['file']} --help"
                    fast_futures.append(
                        executor.submit(self._check_help, cmd, path=target["path"])
                    )
                elif run_all or type == "agent":

                    results.append(
                        GitResult(
                            status="skipped",
                            data="Skipped (MCP-only server)",
                            metadata=GitMetadata(
                                command="agent_server --help",
                                workspace=target["path"],
                                return_code=0,
                                timestamp=skip_ts,
                            ),
                        )
                    )

                if run_all or type == "static-analysis":
                    if target.get("file"):
                        fast_futures.append(
                            executor.submit(self._check_agent_static, target)
                        )
                    else:
                        results.append(
                            GitResult(
                                status="skipped",
                                data="Skipped (No server.py for static check)",
                                metadata=GitMetadata(
                                    command="static_check",
                                    workspace=target["path"],
                                    return_code=0,
                                    timestamp=skip_ts,
                                ),
                            )
                        )

            results.extend(
                [f.result() for f in concurrent.futures.as_completed(fast_futures)]
            )

            if run_all or type == "runtime-validation":
                heavy_futures = []
                for idx, target in enumerate(agent_targets):
                    if "skip_reason" in target:
                        results.append(
                            GitResult(
                                status="skipped",
                                data=target["skip_reason"],
                                metadata=GitMetadata(
                                    command="runtime_check",
                                    workspace=target["path"],
                                    return_code=0,
                                    timestamp=skip_ts,
                                ),
                            )
                        )
                        continue

                    if target.get("file"):
                        port = 9000 + (idx % 3000)
                        heavy_futures.append(
                            executor.submit(self._check_agent_runtime, target, port)
                        )
                    else:
                        results.append(
                            GitResult(
                                status="skipped",
                                data="Skipped (No server.py for runtime check)",
                                metadata=GitMetadata(
                                    command="runtime_check",
                                    workspace=target["path"],
                                    return_code=0,
                                    timestamp=skip_ts,
                                ),
                            )
                        )

                results.extend(
                    [f.result() for f in concurrent.futures.as_completed(heavy_futures)]
                )

            if run_all or type == "all":
                web_targets = [
                    t
                    for t in agent_targets
                    if os.path.exists(os.path.join(t["path"], "package.json"))
                ]
                if web_targets:
                    logger.info(
                        f"Validating Web UI builds for {len(web_targets)} projects..."
                    )
                    build_futures = []
                    for target in web_targets:
                        cmd = "npm run build"
                        build_futures.append(
                            executor.submit(self.git_action, cmd, path=target["path"])
                        )
                    results.extend(
                        [
                            f.result()
                            for f in concurrent.futures.as_completed(build_futures)
                        ]
                    )

            # 7. Pre-commit Standard Compliance (Dry Run)
            if run_all or type == "all" or type == "pre-commit":
                logger.info(
                    f"Checking pre-commit compliance for {len(self.project_map)} projects..."
                )
                pc_results = self.pre_commit_projects(run=True, autoupdate=False)
                # Filter to only include pre-commit results in this category
                results.extend(pc_results)

            successes = [r for r in results if r.status == "success"]
            failures = [r for r in results if r.status == "error"]
            skipped = [r for r in results if r.status == "skipped"]

            print("\n" + "=" * 50)
            print("VALIDATION SUMMARY")
            print(
                f"Total: {len(results)} | Success: {len(successes)} ✅ | Failure: {len(failures)} ❌ | Skipped: {len(skipped)} ⏭️"
            )
            if failures:
                print("\nFailures:")
                for r in failures:
                    pkg = r.metadata.workspace.split("/")[-1]
                    print(f"- {pkg}: {r.error.message if r.error else r.data}")
            print("=" * 50 + "\n")

            categories = {
                "Ecosystem Installation": [
                    r
                    for r in results
                    if r.metadata.command and "pip install" in r.metadata.command
                ],
                "Version Metadata Sync (Dry Run)": [
                    r
                    for r in results
                    if r.metadata.command and "bump2version" in r.metadata.command
                ],
                "Agent Standards Compliance": [
                    r
                    for r in results
                    if r.metadata.command and "static_check" in r.metadata.command
                ],
                "MCP Help Check": [
                    r
                    for r in results
                    if r.metadata.command
                    and "mcp_server" in r.metadata.command
                    and "--help" in r.metadata.command
                ],
                "Agent Help Check": [
                    r
                    for r in results
                    if r.metadata.command
                    and (
                        "agent_server" in r.metadata.command
                        or "server" in r.metadata.command
                    )
                    and "--help" in r.metadata.command
                    and "mcp_server" not in r.metadata.command
                ],
                "Agent Runtime & Web UI": [
                    r
                    for r in results
                    if r.metadata.command
                    and (
                        "runtime_check" in r.metadata.command
                        or "--web" in r.metadata.command
                    )
                ],
                "Pre-commit Standard Compliance": [
                    r
                    for r in results
                    if r.metadata.command and "pre-commit run" in r.metadata.command
                ],
            }

            known_ids = set()
            for cat_list in categories.values():
                for r in cat_list:
                    known_ids.add(id(r))

            other = [r for r in results if id(r) not in known_ids]
            if other:
                for r in other:
                    logger.debug(
                        f"Uncategorized result: {r.metadata.workspace} - cmd: {r.metadata.command}"
                    )
                categories["Additional Operational Checks"] = other

            report_md = "# VALIDATION SUMMARY\n"
            report_md += (
                f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
            )
            report_md += f"**Total:** {len(results)} | **Success:** {len(successes)} ✅ | **Failure:** {len(failures)} ❌ | **Skipped:** {len(skipped)} ⏭️\n\n"

            for cat_name, cat_results in categories.items():
                if not cat_results:
                    continue

                cat_successes = [r for r in cat_results if r.status == "success"]
                cat_failures = [r for r in cat_results if r.status == "error"]
                cat_skipped = [r for r in cat_results if r.status == "skipped"]

                report_md += f"## {cat_name}\n"
                report_md += f"**Success:** {len(cat_successes)} ✅ | **Failure:** {len(cat_failures)} ❌ | **Skipped:** {len(cat_skipped)} ⏭️\n\n"

                if cat_successes:
                    report_md += "#### Successes ✅\n"
                    for r in cat_successes:
                        pkg = r.metadata.workspace.split("/")[-1]
                        report_md += f"- **{pkg}**: Success\n"
                    report_md += "\n"

                if cat_failures:
                    report_md += "#### Failures ❌\n"
                    for r in cat_failures:
                        pkg = r.metadata.workspace.split("/")[-1]
                        error_msg = r.error.message if r.error else r.data
                        report_md += f"- **{pkg}**: {error_msg}\n"
                        if r.data and r.data != error_msg:
                            report_md += f"```text\n{r.data}\n```\n"
                    report_md += "\n"

                if cat_skipped:
                    report_md += "#### Skipped ⏭️\n"
                    for r in cat_skipped:
                        pkg = r.metadata.workspace.split("/")[-1]
                        report_md += f"- **{pkg}**: {r.data or 'Skipped'}\n"
                    report_md += "\n"

            if self.report_path:
                self._export_report(report_md, "validation_report.md")

            return results

    def _export_report(self, markdown_content: str, default_name: str) -> None:
        """Exports markdown content to a file if reporting is enabled."""
        if not self.report_path:
            return

        report_file = self.report_path
        if report_file is True:
            report_file = os.path.join(self.path, default_name)
        elif not os.path.isabs(report_file):
            report_file = os.path.join(self.path, report_file)

        try:
            with open(report_file, "w") as f:
                f.write(markdown_content)
            logger.info(f"Report exported to: {report_file}")
        except Exception as e:
            logger.error(f"Failed to export report to {report_file}: {e}")

    def _check_help(self, command: str, path: str) -> GitResult:
        """Helper to run a --help command and return a standardized result."""
        result = self.git_action(command=command, path=path, quiet=True)
        if result.status == "success":

            result.data = "--help loaded successfully"
        return result

    def _check_agent_static(self, target: Dict[str, str]) -> GitResult:
        """Internal helper to perform static analysis for agents checking for standards compliance."""
        pkg_underscore = target["pkg"]
        pkg_dir = target["pkg_dir"]
        path = target["path"]

        agent_file_name = target["file"]
        agent_file = os.path.join(pkg_dir, f"{agent_file_name}.py")

        try:
            with open(agent_file, "r") as f:
                content = f.read()

            missing = []
            if "warnings.filterwarnings" not in content:
                missing.append("warning_filter")
            if "file=sys.stderr" not in content:
                missing.append("stderr_print")

            metadata = GitMetadata(
                command=f"static_check {agent_file}",
                workspace=path,
                return_code=0 if not missing else 1,
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                + "Z",
            )

            if not missing:
                return GitResult(
                    status="success", data="Static patterns verified", metadata=metadata
                )
            else:
                return GitResult(
                    status="error",
                    data=f"Missing patterns: {', '.join(missing)}",
                    error=GitError(message="Static validation failed", code=1),
                    metadata=metadata,
                )
        except Exception as e:
            logger.error(f"Failed static check for {pkg_underscore}: {e}")
            return GitResult(
                status="error",
                data=str(e),
                error=GitError(message=f"Crash during static check: {e}", code=1),
                metadata=GitMetadata(
                    command="static_check", workspace=path, return_code=1, timestamp=""
                ),
            )

    def _check_agent_runtime(self, target: Dict[str, str], port: int) -> GitResult:
        """Internal helper to test a single agent's runtime startup with Web UI."""
        pkg_underscore = target["pkg"]
        agent_file = target["file"]
        path = target["path"]

        if not agent_file:
            return GitResult(
                status="skipped",
                data=f"No agent server found for {target['name']}",
                metadata=GitMetadata(
                    command="runtime_check",
                    workspace=path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        cmd = [
            sys.executable,
            "-m",
            f"{pkg_underscore}.{agent_file}",
            "--web",
            "--port",
            str(port),
        ]

        env = os.environ.copy()
        for key in [
            "LLM_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GROQ_API_KEY",
        ]:
            if key not in env:
                env[key] = "dummy"
        if "LLM_MODEL_ID" not in env:
            env["LLM_MODEL_ID"] = "dummy"

        env["VALIDATION_MODE"] = "True"

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=path,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if os.name != "nt" else None,
            )

            start_time = datetime.datetime.now()
            startup_time = start_time
            success = False
            error_msg = ""
            full_logs = ""

            while (datetime.datetime.now() - start_time).total_seconds() < 60:
                rlist, _, _ = select.select([proc.stdout, proc.stderr], [], [], 0.5)
                for r in rlist:
                    line = r.readline()
                    if line:
                        full_logs += line
                        if (
                            "Uvicorn running on" in line
                            or "Application startup complete" in line
                            or "Starting server on" in line
                        ) and not success:
                            success = True
                            logger.debug(
                                f"Startup signal detected for {target['name']}, waiting for settle..."
                            )

                            startup_time = datetime.datetime.now()

                        is_error = (
                            "ERROR -" in line.upper()
                            or "ERROR:" in line.upper()
                            or "CRITICAL -" in line.upper()
                            or "CRITICAL:" in line.upper()
                            or "TRACEBACK" in line.upper()
                            or "IMPORTERROR" in line.upper()
                            or "NAMEERROR" in line.upper()
                            or "APPLICATION STARTUP FAILED" in line.upper()
                        )
                        if is_error:
                            error_msg = line.strip()
                            success = False
                            logger.error(
                                f"Startup error detected for {target['name']}: {error_msg}"
                            )
                            break

                if proc.poll() is not None:

                    success = False
                    if not error_msg:
                        error_msg = f"Process exited with code {proc.returncode}"
                    break

                if (
                    success
                    and (datetime.datetime.now() - startup_time).total_seconds() >= 3
                ):
                    break

            try:
                if os.name != "nt":
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
                else:
                    proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                pass

            metadata = GitMetadata(
                command=" ".join(cmd),
                workspace=path,
                return_code=0 if success else (proc.returncode or 1),
                timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                + "Z",
            )

            if success:
                return GitResult(
                    status="success",
                    data=f"Web UI started successfully on port {port}",
                    metadata=metadata,
                )
            else:
                if not error_msg:
                    error_msg = "Timeout (60s): No startup signal detected"
                return GitResult(
                    status="error",
                    data=full_logs[-1000:],
                    error=GitError(message=error_msg, code=metadata.return_code),
                    metadata=metadata,
                )

        except Exception as e:
            return GitResult(
                status="error",
                data=str(e),
                error=GitError(message=f"Crash during runtime check: {e}", code=1),
                metadata=GitMetadata(
                    command=" ".join(cmd), workspace=path, return_code=1, timestamp=""
                ),
            )

    @staticmethod
    def generate_markdown_summary(action: str, results: List[GitResult]) -> str:
        """Generates a beautiful markdown summary of bulk operation results."""
        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "error"]
        skips = [r for r in results if r.status == "skipped"]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = [
            f"# {action.upper()} Summary",
            f"**Time:** {timestamp}  ",
            f"**Total:** {len(results)} | **Success:** {len(successes)} ✅ | **Failure:** {len(failures)} ❌ | **Skipped:** {len(skips)} ⏭️",
            "",
        ]

        if successes:
            md.append("## Successes ✅")
            for r in successes:
                name = os.path.basename(r.metadata.workspace)

                msg = r.data or "Success"
                if action.lower() in ["installation", "build", "validation"]:
                    msg = "Success"
                elif msg.count("\n") > 2:

                    if "new_version=" not in msg and "current_version=" not in msg:
                        msg = "Success"

                md.append(f"- **{name}**: {msg}")
            md.append("")

        if failures:
            md.append("## Failures ❌")
            for r in failures:
                name = os.path.basename(r.metadata.workspace)
                err_msg = r.error.message if r.error else "Unknown error"
                md.append(f"### ⚠️ {name}")
                md.append(f"**Command:** `{r.metadata.command}`")
                md.append("**Error:**")
                md.append(f"```text\n{err_msg}\n```")
                if r.data:
                    md.append("**Output:**")
                    md.append(f"```text\n{r.data}\n```")
                md.append("---")
            md.append("")

        if skips:
            md.append("## Skipped ⏭️")
            reasons = {}
            for r in skips:
                reason = r.data or "No reason provided"
                if reason not in reasons:
                    reasons[reason] = []
                reasons[reason].append(os.path.basename(r.metadata.workspace))

            for reason, projects in sorted(reasons.items()):
                project_list = ", ".join(sorted(list(set(projects))))
                md.append(f"- **{reason}**: {project_list}")
            md.append("")

        return "\n".join(md)

    def git_action(
        self, command: str, path: str = None, quiet: bool = False, env: dict = None
    ) -> GitResult:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            path (str, optional): The directory to execute the command in.
                Defaults to the base path.

        Returns:
            GitResult: The combined stdout and stderr output of the command in structured format.
        """
        target_path = self._resolve_path(path)

        pipe = subprocess.Popen(
            command,
            shell=True,
            cwd=target_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            env=env if env else os.environ,
        )
        out, err = pipe.communicate()
        return_code = pipe.wait()

        metadata = GitMetadata(
            command=command,
            workspace=target_path,
            return_code=return_code,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        error_obj = None
        if return_code != 0:
            error_obj = GitError(
                message=(
                    err.strip() if err else (out.strip() if out else "Unknown error")
                ),
                code=return_code,
            )

        result = GitResult(
            status="success" if return_code == 0 else "error",
            data=out.strip() if out else "",
            error=error_obj,
            metadata=metadata,
        )

        if result.status == "error":
            logger.error(f"Command failed: {command}\nError: {result.error.message}")
        elif not quiet:
            logger.info(f"Command: {command}\nOutput: {result.data}")

        return result

    def clone_projects(self, projects: List[str] = None) -> List[GitResult]:
        """
        Clone all specified Git projects in parallel using multiple threads.

        Returns:
            List[GitResult]: A list of GitResult objects, one for each clone operation.
        """
        try:
            expanded_path = os.path.expanduser(self.path)
            if not os.path.exists(expanded_path):
                os.makedirs(expanded_path, exist_ok=True)

            targets = []
            if self.project_map:
                for url, path in self.project_map.items():
                    targets.append((url, path))
            elif projects:
                for url in projects:
                    name = url.split("/")[-1].replace(".git", "")
                    targets.append((url, os.path.join(expanded_path, name)))

            if not targets:
                logger.warning("No projects to clone.")
                return []

            logger.info(
                f"Cloning {len(targets)} projects in parallel using {self.threads} threads..."
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads
            ) as executor:
                futures = {
                    executor.submit(self.clone_repository, url, path): (url, path)
                    for url, path in targets
                }
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return results

        except Exception as e:
            logger.error(f"Parallel project cloning failed: {str(e)}")
            return [
                GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Parallel project cloning failed: {str(e)}", code=-1
                    ),
                    metadata=GitMetadata(
                        command="clone_projects",
                        workspace=self.path,
                        return_code=-1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            ]

    def clone_repository(self, url: str, target_path: str) -> GitResult:
        """
        Clone a single Git repository to a specific target path.

        Args:
            url (str): The repository URL to clone.
            target_path (str): The absolute path where the repository should be cloned.

        Returns:
            GitResult: The result of the Git clone command.
        """
        if not url:
            return GitResult(
                status="error",
                data="",
                error=GitError(message="No repository URL provided", code=1),
                metadata=GitMetadata(
                    command="clone",
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        command = f"git clone {url} {target_path}"
        result = self.git_action(command, path=os.path.dirname(target_path))
        logger.info(f"Cloning {url} to {target_path}: {result.status}")
        return result

    def pull_projects(self, project_dirs: List[str] = None) -> List[GitResult]:
        """
        Pull updates for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to pull.")
                return []

        if not project_dirs:
            logger.warning("No projects found to pull.")
            return []

        logger.info(
            f"Pulling {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(self.pull_project, project_dirs))

    def pull_project(self, path: str = None) -> GitResult:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            path (str): The path to the project to pull. Defaults to self.path.

        Returns:
            GitResult: The result of the pull operation.
        """
        target_path = self._resolve_path(path)
        results = []

        pull_result = self.git_action(command="git pull", path=target_path)
        results.append(pull_result)

        logger.info(
            f"Scanning: {target_path}\n"
            f"Pulling latest changes for {target_path}\n"
            f"{pull_result}"
        )

        if self.set_to_default_branch:
            default_branch_result = self.git_action(
                "git symbolic-ref refs/remotes/origin/HEAD",
                path=target_path,
            )
            if default_branch_result.status == "success":
                default_branch = re.sub(
                    "refs/remotes/origin/", "", default_branch_result.data
                ).strip()
                checkout_result = self.git_action(
                    f'git checkout "{default_branch}"',
                    path=target_path,
                )
                results.append(checkout_result)
                logger.info(f"Checking out default branch: {checkout_result}")
            else:
                results.append(default_branch_result)
                logger.error(
                    f"Failed to get default branch for {target_path}: {default_branch_result.error}"
                )

        combined_status = (
            "success" if all(r.status == "success" for r in results) else "error"
        )

        combined_data = "\n".join(
            [f"[{r.metadata.command}]: {r.data}" for r in results]
        )

        combined_error = next((r.error for r in results if r.error), None)

        metadata = GitMetadata(
            command="pull_project",
            workspace=target_path,
            return_code=0 if combined_status == "success" else 1,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        return GitResult(
            status=combined_status,
            data=combined_data,
            error=combined_error,
            metadata=metadata,
        )

    def set_threads(self, threads: int) -> None:
        """
        Set the number of threads for parallel processing.

        Args:
            threads (int): The number of threads.

        Notes:
            If the input is invalid, defaults 12
        """
        try:
            if 0 < threads <= self.maximum_threads:
                self.threads = threads
            else:
                logger.warning(
                    f"Did not recognize {threads} as a valid value, defaulting to: {self.maximum_threads}"
                )
                self.threads = self.maximum_threads
        except Exception as e:
            logger.error(
                f"Did not recognize {threads} as a valid value, defaulting to: {self.maximum_threads}. Error: {e}"
            )
            self.threads = self.maximum_threads

    def pre_commit(
        self,
        run: bool = True,
        autoupdate: bool = False,
        path: str = None,
    ) -> GitResult:
        """
        Execute pre-commit commands in the specified path.

        Args:
            run (bool): Whether to run 'pre-commit run --all-files'. Default True.
            autoupdate (bool): Whether to run 'pre-commit autoupdate'. Default False.
            path (str, optional): Path to run in. Defaults to self.path.
        """
        target_path = self._resolve_path(path)

        if not os.path.exists(os.path.join(target_path, ".pre-commit-config.yaml")):
            return GitResult(
                status="skipped",
                data="No .pre-commit-config.yaml found.",
                error=None,
                metadata=GitMetadata(
                    command="pre_commit_check",
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        commands = []
        if autoupdate:
            commands.append("pre-commit autoupdate")
        if run:
            commands.append("git add -A")
            commands.append("pre-commit run --all-files")

        if not commands:
            return GitResult(
                status="skipped",
                data="No action selected (run=False, autoupdate=False).",
                error=None,
                metadata=GitMetadata(
                    command="pre_commit_check",
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        full_command = " && ".join(commands)

        env = os.environ.copy()
        if "SKIP" in env:
            env["SKIP"] += ",no-commit-to-branch"
        else:
            env["SKIP"] = "no-commit-to-branch"

        result = self.git_action(command=full_command, path=target_path, env=env)

        if result.status == "error" and result.error:
            msg = result.error.message.lower()
            if "don't commit to branch" in msg or "no-commit-to-branch" in msg:

                other_failures = False
                lines = (result.error.message + "\n" + result.data).splitlines()
                for line in lines:
                    if (
                        "Failed" in line
                        and "don't commit to branch" not in line.lower()
                    ):
                        other_failures = True
                        break

                if not other_failures:
                    logger.info(
                        f"Ignoring safe pre-commit failure (branch lock) in {target_path}"
                    )
                    return GitResult(
                        status="success",
                        data=result.data or "Skipped branch lock check",
                        metadata=result.metadata,
                    )

        return result

    def pre_commit_projects(
        self,
        run: bool = True,
        autoupdate: bool = False,
        projects: List[str] = None,
    ) -> List[GitResult]:
        """
        Execute pre-commit commands for all projects in parallel.

        Returns:
            List[GitResult]: A list of GitResult objects.
        """
        try:
            expanded_path = os.path.expanduser(self.path)
            if not os.path.exists(expanded_path):
                return []

            if projects is None:
                if self.project_map:
                    project_dirs = [
                        p
                        for p in self.project_map.values()
                        if os.path.exists(os.path.join(p, ".pre-commit-config.yaml"))
                    ]
                else:
                    logger.warning("No projects found in project_map for pre-commit.")
                    return []
            else:
                project_dirs = [
                    os.path.abspath(os.path.expanduser(p))
                    for p in projects
                    if os.path.isdir(os.path.abspath(os.path.expanduser(p)))
                    and os.path.exists(
                        os.path.join(
                            os.path.abspath(os.path.expanduser(p)),
                            ".pre-commit-config.yaml",
                        )
                    )
                ]

            if not project_dirs:
                return []

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads
            ) as executor:
                futures = {
                    executor.submit(self.pre_commit, run, autoupdate, d): d
                    for d in project_dirs
                }
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())

            return results

        except Exception as e:
            logger.error(f"Parallel pre-commit failed: {str(e)}")
            return [
                GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Parallel pre-commit failed: {str(e)}", code=-1
                    ),
                    metadata=GitMetadata(
                        command="pre_commit_projects",
                        workspace=self.path,
                        return_code=-1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
            ]

    def install_project(self, path: str = None, extra: str = "all") -> GitResult:
        """
        Install a Python project using pip install -e .[extra].
        """
        target_path = self._resolve_path(path)

        has_all_extra = False
        pyproject = os.path.join(target_path, "pyproject.toml")
        if os.path.exists(pyproject):
            try:
                with open(pyproject, "r") as f:
                    content = f.read()
                if (
                    "[project.optional-dependencies]" in content
                    and f"{extra} =" in content.replace(" ", "")
                ):
                    has_all_extra = True
            except Exception:
                pass

        install_target = f".[{extra}]" if has_all_extra else "."
        command = f"pip install -e {install_target}"

        logger.info(f"Installing project at {target_path} with {command}")
        result = self.git_action(command=command, path=target_path)

        for d in ["build", "dist"]:
            shutil.rmtree(os.path.join(target_path, d), ignore_errors=True)
        for egg_info in Path(target_path).glob("*.egg-info"):
            shutil.rmtree(egg_info, ignore_errors=True)

        return result

    def get_readme(self, path: str = None) -> ReadmeResult:
        """
        Get the content and path of the README.md file in the specified path.

        Args:
            path (str, optional): The directory path. Defaults to self.path.

        Returns:
            ReadmeResult: Object containing 'content' and 'path' of the README.md file.
        """
        target_dir = self._resolve_path(path)

        if not os.path.exists(target_dir):
            return ReadmeResult(content="", path="")

        readme_path = None
        for filename in os.listdir(target_dir):
            if filename.lower() == "readme.md":
                readme_path = os.path.join(target_dir, filename)
                break

        if not readme_path:
            return ReadmeResult(content="", path="")

        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            return ReadmeResult(content=content, path=readme_path)
        except Exception as e:
            logger.error(f"Error reading README: {e}")
            return ReadmeResult(content="", path=readme_path)

    def create_project(self, path: str) -> GitResult:
        """
        Create a new project directory and initialize it as a git repository.

        Args:
            path (str): The path of the project directory to create.

        Returns:
            GitResult: Result of the operation.
        """
        target_path = self._resolve_path(path)

        if os.path.exists(target_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory already exists: {target_path}", code=1
                ),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        try:
            os.makedirs(target_path, exist_ok=True)
            init_result = self.git_action("git init", path=target_path)

            if init_result.status == "success":
                logger.info(f"Created project: {target_path}")
                return init_result
            else:
                return init_result

        except Exception as e:
            logger.error(f"Failed to create project {target_path}: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=target_path,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def bump_version(
        self,
        part: str,
        allow_dirty: bool = False,
        path: str = None,
        dry_run: bool = False,
        verbose: bool = False,
    ) -> GitResult:
        """
        Bump the version of the project using bump2version.

        Args:
            part (str): The part of the version to bump (major, minor, patch).
            allow_dirty (bool): Whether to allow dirty working directory.
            path (str): The path to the project directory.
            dry_run (bool): Whether to perform a dry run.
            verbose (bool): Whether to use verbose output (for dry-run visibility).

        Returns:
            GitResult: Result of the operation.
        """
        target_dir = self._resolve_path(path)

        if not os.path.exists(target_dir):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Directory not found: {target_dir}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=target_dir,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        valid_parts = ["major", "minor", "patch"]
        if part not in valid_parts:
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Invalid part '{part}'. Must be one of {valid_parts}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=target_dir,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        command = f"bump2version {part}"
        if allow_dirty:
            command += " --allow-dirty"
        if dry_run:
            command += " --dry-run"
        if verbose:
            command += " --verbose"
        if not dry_run:
            command += " --list"

        try:
            result = self.git_action(command=command, path=target_dir)

            if result.status == "success":
                logger.info(f"Bumped version ({part}) in {target_dir}")
            else:
                logger.error(f"Failed to bump version in {target_dir}: {result.error}")

            return result
        except Exception as e:
            logger.error(f"Error in bump_version: {e}")
            return GitResult(
                status="error",
                data="",
                error=GitError(message=str(e), code=1),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=target_dir,
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

    def bulk_bump(
        self,
        part: str,
        dry_run: bool = False,
        exclude: List[str] = None,
        verbose: bool = False,
    ) -> List[GitResult]:
        """Bumps the version for all projects in the workspace in parallel."""
        exclude = exclude or []
        results = []

        for url, path in self.project_map.items():
            name = url.split("/")[-1].replace(".git", "")
            if name in exclude:
                continue

            project_dir = Path(path)
            if (project_dir / ".bumpversion.cfg").exists():
                results.append(
                    self.bump_version(
                        part,
                        allow_dirty=True,
                        path=str(project_dir),
                        dry_run=dry_run,
                        verbose=verbose,
                    )
                )
            else:
                results.append(
                    GitResult(
                        status="skipped",
                        data="Missing .bumpversion.cfg",
                        metadata=GitMetadata(
                            command="bulk_bump",
                            workspace=str(project_dir),
                            return_code=0,
                            timestamp=datetime.datetime.now(
                                datetime.timezone.utc
                            ).isoformat()
                            + "Z",
                        ),
                    )
                )
        return results

    def update_dependency(
        self, file_path: str, package_name: str, new_version: str, dry_run: bool = False
    ) -> bool:
        """Updates the version of a package in a pyproject.toml's dependencies."""
        target_file = Path(self._resolve_path(file_path))
        if not target_file.exists() or not target_file.is_file():
            return False

        content = target_file.read_text()
        pattern = rf'("{package_name}(?:\[.*?\])?>=)\d+\.\d+\.\d+'
        replacement = rf"\g<1>{new_version}"

        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            if not dry_run:
                target_file.write_text(new_content)
            logger.info(
                f"{'[DRY RUN] Would update' if dry_run else 'Updated'} {package_name} to >={new_version} in {target_file}"
            )
            return True
        return False

    def phased_bumpversion(
        self,
        part: str = "patch",
        start_phase: int = 1,
        dry_run: bool = False,
        skip_pre_commit: bool = False,
        config: dict = None,
    ) -> List[GitResult]:
        """
        Execute the phased bumpversion workflow: pre-commits + phased bumping.
        """
        all_results = []
        if config is None:
            if hasattr(self, "config") and self.config and self.config.maintenance:
                config_model = self.config.maintenance
            else:
                yml_path = os.environ.get("WORKSPACE_YML") or "workspace.yml"
                if not os.path.isabs(yml_path):
                    yml_path = os.path.join(self.path, yml_path)

                if os.path.exists(yml_path):
                    if self.load_projects_from_yaml(yml_path):
                        config_model = self.config.maintenance
                    else:
                        config_model = None
                else:
                    config_model = None

            if config_model:
                config = config_model.model_dump()
            else:
                logger.error("No maintenance configuration found.")
                return []

        if not skip_pre_commit:
            projects_to_check = None
            if config:
                projects_to_check = []
                has_bulk = False
                for phase in config.get("phases", []):
                    if phase.get("bulk_bump"):
                        has_bulk = True
                        break
                    if phase.get("project"):
                        projects_to_check.append(phase.get("project"))
                if has_bulk:
                    projects_to_check = None

            pc_results = self.pre_commit_projects(
                run=True, autoupdate=True, projects=projects_to_check
            )
            all_results.extend(pc_results)

        def run_step_bump(project_name, phase_num):
            if start_phase <= phase_num:
                project_dir = None
                for url, p_path in self.project_map.items():
                    if url.endswith(f"/{project_name}.git") or url.endswith(
                        f"/{project_name}"
                    ):
                        project_dir = p_path
                        break

                if not project_dir:
                    return None

                result = self.bump_version(
                    part=part,
                    allow_dirty=True,
                    path=str(project_dir),
                    dry_run=dry_run,
                    verbose=dry_run or not dry_run,
                )
                all_results.append(result)

                if result.status == "success":

                    match = re.search(r"new_version=(.*)", result.data)
                    if match:
                        return match.group(1).strip()

                    match = re.search(r"current_version=(.*)", result.data)
                    return match.group(1).strip() if match else "success"
                return None
            return None

        for phase in config.get("phases", []):
            phase_num = phase.get("phase")
            if phase_num < start_phase:
                continue

            if phase.get("bulk_bump"):
                exclude = phase.get("exclude", [])
                bulk_results = self.bulk_bump(
                    part=part, dry_run=dry_run, exclude=exclude
                )
                all_results.extend(bulk_results)
                continue

            project_name = phase.get("project")
            new_version = run_step_bump(project_name, phase_num)

            if new_version:
                for update in phase.get("updates", []):
                    pkg = update.get("package")
                    if "target" in update:
                        res = self.update_dependency(
                            update["target"], pkg, new_version, dry_run
                        )
                        if isinstance(res, GitResult):
                            all_results.append(res)
                    elif "target_pattern" in update:
                        exclude = update.get("exclude", [])
                        for url, path in self.project_map.items():
                            name = url.split("/")[-1].replace(".git", "")
                            if name not in exclude:
                                pyproject = Path(path) / "pyproject.toml"
                                if pyproject.exists():
                                    res = self.update_dependency(
                                        str(pyproject), pkg, new_version, dry_run
                                    )
                                    if isinstance(res, GitResult):
                                        all_results.append(res)

        return all_results

    maintain_projects = phased_bumpversion

    def load_projects_from_yaml(self, yaml_path: str) -> bool:
        """
        Loads repository URLs from a YAML workspace file using Pydantic models.
        Strictly determines self.path relative to the configuration file.
        """
        abs_yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
        yaml_dir = os.path.dirname(abs_yaml_path)

        if not os.path.exists(abs_yaml_path):
            logger.error(f"YAML file not found: {abs_yaml_path}")
            return False

        try:
            with open(abs_yaml_path, "r") as f:
                data = yaml.safe_load(f)

            if not data:
                return False

            self.config = WorkspaceConfig(**data)

            yaml_config_path = os.path.expanduser(self.config.path)
            is_default_yaml = yaml_path == DEFAULT_WORKSPACE_YML

            if os.path.isabs(yaml_config_path):
                self.path = os.path.abspath(yaml_config_path)
            elif is_default_yaml:

                self.path = os.path.abspath(
                    os.path.expanduser(DEFAULT_REPOSITORY_MANAGER_WORKSPACE)
                )
                logger.info(
                    f"Using packaged workspace.yml, preserving default path: {self.path}"
                )
            else:
                self.path = os.path.abspath(os.path.join(yaml_dir, yaml_config_path))

            logger.info(f"Workspace root resolved to: {self.path}")

            self.project_map = self._parse_subdirectories(
                self.config.subdirectories, self.path
            )

            for repo in self.config.repositories:
                repo_name = repo.url.split("/")[-1].replace(".git", "")
                self.project_map[repo.url] = os.path.join(self.path, repo_name)
            return True

        except Exception as e:
            logger.error(f"Failed to load projects from YAML: {e}")
            return False

    def _parse_subdirectories(
        self, subdirs: Dict[str, SubdirectoryConfig], current_path: str
    ) -> Dict[str, str]:
        """Helper to recursively parse subdirectories and collect repository paths."""
        project_map = {}
        for name, data in subdirs.items():
            new_path = os.path.join(current_path, name)

            for repo in data.repositories:
                repo_name = repo.url.split("/")[-1].replace(".git", "")
                project_map[repo.url] = os.path.join(new_path, repo_name)

            if data.subdirectories:
                project_map.update(
                    self._parse_subdirectories(data.subdirectories, new_path)
                )

        return project_map

    def generate_workspace_template(
        self, target_path: str, use_default: bool = True
    ) -> GitResult:
        """
        Generates a workspace.yml template at the specified path.
        """
        try:
            target_path = os.path.abspath(os.path.expanduser(target_path))
            if os.path.isdir(target_path):
                target_path = os.path.join(target_path, "workspace.yml")

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            template_content = ""
            if use_default:
                try:
                    from importlib.resources import files

                    template_content = (
                        files("repository_manager") / "workspace.yml"
                    ).read_text()
                except Exception:

                    template_content = "name: My Workspace\npath: .\ndescription: New workspace\nsubdirectories: {}\n"
            else:
                template_content = "name: My Workspace\npath: .\ndescription: New workspace\nsubdirectories:\n  agents:\n    description: Agent repositories\n    repositories: []\n"

            with open(target_path, "w") as f:
                f.write(template_content)

            return GitResult(
                status="success",
                data=f"Template generated at {target_path}",
                metadata=GitMetadata(
                    command="generate_template",
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            logger.error(f"Failed to generate template: {e}")
            return GitResult(
                status="error", data="", error=GitError(message=str(e), code=1)
            )

    def save_workspace_config(
        self, yaml_path: str, config: WorkspaceConfig = None
    ) -> GitResult:
        """
        Saves the current or provided WorkspaceConfig to a YAML file.
        """
        try:
            cfg = config or self.config
            if not cfg:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="No configuration to save", code=1),
                )

            yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

            data = cfg.model_dump()
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)

            return GitResult(
                status="success",
                data=f"Workspace saved to {yaml_path}",
                metadata=GitMetadata(
                    command="save_workspace",
                    workspace=yaml_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )
        except Exception as e:
            logger.error(f"Failed to save workspace: {e}")
            return GitResult(
                status="error", data="", error=GitError(message=str(e), code=1)
            )

    def generate_agents_documentation(self, target_path: str = None) -> GitResult:
        """
        Generates an AGENTS.md catalog by discovering agent types and main files.
        Uses a robust multi-level path discovery to identify Graph vs Flat agents.
        """
        try:
            target_path = target_path or os.path.join(self.path, "AGENTS.md")
            target_path = os.path.abspath(os.path.expanduser(target_path))

            rows = []
            for url, project_path in self.get_project_map().items():
                pkg_name = url.split("/")[-1].replace(".git", "")
                pkg_name_underscore = pkg_name.replace("-", "_")

                if not os.path.exists(project_path):
                    rows.append(f"| `{pkg_name}` | (Not Cloned) |")
                    continue

                is_graph = False
                is_agent = False

                search_paths = [
                    Path(project_path),
                    Path(project_path) / pkg_name,
                    Path(project_path) / pkg_name_underscore,
                ]

                for sp in search_paths:
                    if (sp / "graph_config.py").exists():
                        is_graph = True
                        is_agent = True
                        break
                    if (sp / "agent_server.py").exists() or (
                        sp / "mcp_server.py"
                    ).exists():
                        is_agent = True

                if not is_agent:
                    agent_type = "Library"
                else:
                    agent_type = "Graph" if is_graph else "Flat"

                rows.append(f"| `{pkg_name}` | {agent_type} |")

            rows.sort()
            catalog_table = "\n".join(rows)

            template_content = ""
            try:
                from importlib.resources import files

                template_content = (
                    files("repository_manager") / "AGENTS.md"
                ).read_text()
            except Exception:
                template_content = "# Agent Catalog\n\n| Agent Package | Type |\n|:--------------|:-----|\n<!-- AGENT_CATALOG_PLACEHOLDER -->\n"

            if "<!-- AGENT_CATALOG_PLACEHOLDER -->" in template_content:
                final_content = template_content.replace(
                    "<!-- AGENT_CATALOG_PLACEHOLDER -->", catalog_table
                )
            else:
                logger.warning(
                    "Placeholder <!-- AGENT_CATALOG_PLACEHOLDER --> not found in template. Appending catalog."
                )
                final_content = (
                    template_content
                    + "\n\n## Dynamic Catalog\n\n| Agent Package | Type |\n|:--------------|:-----|\n"
                    + catalog_table
                )

            with open(target_path, "w") as f:
                f.write(final_content)

            return GitResult(
                status="success",
                data=f"Agents documentation generated at {target_path}",
                metadata=GitMetadata(
                    command="generate_agents_md",
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        except Exception as e:
            logger.error(f"Failed to generate AGENTS.md: {e}")
            return GitResult(
                status="error", data="", error=GitError(message=str(e), code=1)
            )

    def get_consolidated_skill_paths(self) -> List[str]:
        """
        Returns absolute paths to the 15 specific building and documentation skills.
        """
        required_universal = [
            "agent-package-builder",
            "mcp-builder",
            "agent-builder",
            "skill-builder",
            "skill-graph-builder",
            "api-wrapper-builder",
            "web-search",
            "web-crawler",
        ]
        required_graphs = [
            "docker-docs",
            "fastapi-docs",
            "fastmcp-docs",
            "nodejs-docs",
            "vercel-docs",
            "python-docs",
            "pydantic-ai-docs",
        ]

        paths = []

        if get_universal_skills_path:
            try:

                from importlib.resources import files

                base = files("universal_skills") / "skills"
                for skill in required_universal:
                    skill_path = base / skill
                    if skill_path.joinpath("SKILL.md").exists():
                        paths.append(str(skill_path))
            except Exception as e:
                logger.warning(f"Could not load universal skills via importlib: {e}")

                all_universal = get_universal_skills_path()
                paths.extend(
                    [
                        p
                        for p in all_universal
                        if os.path.basename(p) in required_universal
                    ]
                )

        if get_skill_graphs_path:
            try:
                from importlib.resources import files

                base = files("skill_graphs") / "skill_graphs"
                for graph in required_graphs:
                    graph_path = base / graph
                    if graph_path.joinpath("SKILL.md").exists():
                        paths.append(str(graph_path))
            except Exception as e:
                logger.warning(f"Could not load skill graphs via importlib: {e}")

                all_graphs = get_skill_graphs_path(default_enabled=True)
                paths.extend(
                    [p for p in all_graphs if os.path.basename(p) in required_graphs]
                )

        return list(set(paths))

    def ensure_graph(self) -> Optional["GraphReport"]:
        """
        Incrementally builds or updates the Hybrid Graph for the workspace.
        """
        import asyncio
        from repository_manager.graph.engine import GraphEngine

        if (
            not hasattr(self, "config")
            or not self.config
            or not self.config.graph
            or not getattr(self.config.graph, "enabled", False)
        ):
            logger.info("Hybrid Graph is disabled in workspace config.")
            return None

        logger.info("Initiating Hybrid Graph build/sync process...")
        engine = GraphEngine(self.path)

        # Build graph synchronously or asynchronously based on engine implementation
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        report = loop.run_until_complete(
            engine.build_from_workspace(
                self.config,
                multimodal=getattr(self.config.graph, "multimodal", False),
                incremental=getattr(self.config.graph, "incremental", True),
            )
        )
        logger.info("Hybrid Graph build process complete.")
        return report


def main() -> None:
    """
    Main entry point for the Repository Manager CLI.
    Supports workspace management, Git bulk operations, and maintenance.
    """
    parser = argparse.ArgumentParser(
        description="Repository Manager - 100% Model-Driven Pydantic Graph Agent",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Examples:
  # Standard setup (clones all missing repos in parallel)
  repository-manager --clone

  # Generate documentation catalog
  repository-manager --agents-md

  # Maintenance workflow (Bump patch version everywhere)
  repository-manager --maintain --bump patch

  # Selective operations
  repository-manager --repositories "genius-agent, gitlab-api" --pull
  """,
    )

    group_general = parser.add_argument_group("General Options")
    group_general.add_argument(
        "-v", "--version", action="version", version=f"%(prog)s {__version__}"
    )
    group_general.add_argument(
        "-f",
        "--file",
        type=str,
        help="Path to workspace.yml file (Standard Source).",
        default=DEFAULT_WORKSPACE_YML,
    )
    group_general.add_argument(
        "-w",
        "--workspace",
        type=str,
        help="Path to the workspace root directory (default: ~/Workspace).",
        default=DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
    )
    group_general.add_argument(
        "-t",
        "--threads",
        type=int,
        help="Parallel thread count (default: 12).",
        default=DEFAULT_REPOSITORY_MANAGER_THREADS,
    )
    group_general.add_argument(
        "-r",
        "--repositories",
        type=str,
        help="Comma-separated list of repository names to filter operations.",
    )

    group_workspace = parser.add_argument_group("Workspace Management")
    group_workspace.add_argument(
        "--setup",
        action="store_true",
        help="Initialize workspace: directory structure & clones missing repos.",
    )
    group_workspace.add_argument(
        "--tree",
        action="store_true",
        help="Visualize workspace hierarchy as ASCII tree.",
    )
    group_workspace.add_argument(
        "--mermaid", action="store_true", help="Generate Mermaid graph of workspace."
    )
    group_workspace.add_argument(
        "--agents-md",
        action="store_true",
        help="Generate AGENTS.md catalog of discovered agents.",
    )
    group_workspace.add_argument(
        "--save",
        action="store_true",
        help="Save current in-memory config back to YAML (Updates).",
    )

    group_git = parser.add_argument_group("Git Bulk Operations (Parallelized)")
    group_git.add_argument(
        "--clone",
        action="store_true",
        help="Clone all missing repositories in the workspace.",
    )
    group_git.add_argument(
        "--pull", action="store_true", help="Pull latest changes for all projects."
    )
    group_git.add_argument(
        "--default-branch",
        action="store_true",
        help="Switch all repos to their default branch (via origin/HEAD).",
    )

    group_maintenance = parser.add_argument_group("Maintenance Lifecycle")
    group_maintenance.add_argument(
        "--install",
        action="store_true",
        help="Run 'pip install -e .' for all projects.",
    )
    group_maintenance.add_argument(
        "--build", action="store_true", help="Run 'python -m build' for all projects."
    )
    group_maintenance.add_argument(
        "--validate",
        action="store_true",
        help="Run comprehensive pre-release validation.",
    )
    group_maintenance.add_argument(
        "--type",
        choices=[
            "all",
            "static-analysis",
            "runtime-validation",
            "mcp",
            "agent",
            "flat",
            "graph",
        ],
        default="all",
        help="Filter validation mode or target (default: all).",
    )
    group_maintenance.add_argument(
        "--pre-commit",
        action="store_true",
        help="Run pre-commit checks and autoupdate hooks.",
    )
    group_maintenance.add_argument(
        "--maintain",
        action="store_true",
        help="Execute phased maintenance (Bump -> Pre-commit -> Verify).",
    )
    group_maintenance.add_argument(
        "--bump",
        choices=["patch", "minor", "major"],
        help="Version bump part (major/minor/patch). Use with --maintain or standalone.",
    )
    group_maintenance.add_argument(
        "--report",
        nargs="?",
        const=True,
        help="Export markdown summary results to a file (default: Workspace root).",
    )
    group_maintenance.add_argument(
        "--phase",
        type=int,
        default=1,
        help="Starting phase for maintenance lifecycle (1-5).",
    )
    group_maintenance.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform maintenance operations without committing changes. Use with --maintain.",
    )
    group_maintenance.add_argument(
        "--skip-pre-commit",
        action="store_true",
        help="Bypass the pre-commit phase during maintenance workflow. Use with --maintain.",
    )
    group_maintenance.add_argument(
        "--config",
        type=str,
        help="Path to an overridden maintenance JSON/YAML configuration. Use with --maintain.",
    )

    group_graph = parser.add_argument_group("Graph Intelligence (Hybrid Search)")
    group_graph.add_argument(
        "--graph-query",
        type=str,
        help="Query the Hybrid Graph using vector similarity or Cypher structure.",
    )
    group_graph.add_argument(
        "--graph-mode",
        choices=["semantic", "structural", "hybrid"],
        default="hybrid",
        help="Search mode for graph-query (default: hybrid).",
    )
    group_graph.add_argument(
        "--graph-path",
        nargs=2,
        metavar=("SOURCE", "TARGET"),
        help="Find the shortest path between two symbols across the workspace graph.",
    )
    group_graph.add_argument(
        "--graph-status", action="store_true", help="Show current graph metrics."
    )
    group_graph.add_argument(
        "--graph-reset",
        action="store_true",
        help="Purge the graph database and force a clean rebuild.",
    )
    group_graph.add_argument(
        "--graph-impact",
        type=str,
        help="Calculate multi-repo impact for a symbol.",
    )

    args = parser.parse_args()

    git = Git(
        path=args.workspace,
        threads=args.threads,
        report_path=args.report,
    )
    clone_flag = args.clone
    pull_flag = args.pull

    if args.default_branch:
        git.set_to_default_branch = True

    if args.threads:
        git.set_threads(threads=args.threads)

    if args.file:
        if os.path.exists(args.file):

            if not git.load_projects_from_yaml(args.file):
                logger.warning(f"Could not load {args.file} as a valid Workspace YAML.")
        else:
            logger.error(f"Workspace file not found: {args.file}")
            parser.print_help()
            sys.exit(2)

    if not git.project_map and os.path.exists(DEFAULT_WORKSPACE_YML):
        git.load_projects_from_yaml(DEFAULT_WORKSPACE_YML)

    if args.repositories:
        repositories = args.repositories.replace(" ", "").split(",")
        names_to_keep = set(repositories)
        if git.project_map:
            filtered = {}
            for url, path in git.project_map.items():
                name = url.split("/")[-1].replace(".git", "")
                if name in names_to_keep:
                    filtered[url] = path
            git.project_map = filtered
        else:

            for r in repositories:
                if "/" in r:
                    name = r.split("/")[-1].replace(".git", "")
                    git.project_map[r] = os.path.join(git.path, name)
                else:
                    git.project_map[os.path.join("https://github.com/", r)] = (
                        os.path.join(git.path, r)
                    )

    if args.file and os.path.exists(args.file):
        if args.tree:
            print(git.generate_workspace_tree(args.file))
            sys.exit(0)
        if args.mermaid:
            print(git.generate_workspace_mermaid(args.file))
            sys.exit(0)
        if args.agents_md:
            logger.info(f"Generating AGENTS.md catalog in {git.path}...")
            git.generate_agents_md()
            sys.exit(0)
        if args.setup:
            logger.info(f"Setting up workspace from {args.file}...")
            git.load_projects_from_yaml(args.file)

    if clone_flag:
        git.clone_projects()
    if pull_flag:
        git.pull_projects()

    if args.pre_commit:
        git.pre_commit_projects(run=True, autoupdate=True)

    if args.bump:
        logger.info(f"Bumping version ({args.bump}) for all projects projects...")

        project_dirs = list(git.project_map.values())

        results = []
        for d in project_dirs:
            if (Path(d) / ".bumpversion.cfg").exists():
                results.append(
                    git.bump_version(
                        args.bump, allow_dirty=True, path=d, dry_run=args.dry_run
                    )
                )

        summary = git.generate_markdown_summary("Bulk Version Bump", results)
        print(summary)
        git._export_report(summary, "version_bump_report.md")

    if args.maintain:
        config = None
        if args.config:
            try:
                with open(args.config, "r") as f:
                    config = json.load(f)
            except Exception as e:
                logger.error(f"Failed to load config from {args.config}: {e}")
                sys.exit(1)

        results = git.phased_bumpversion(
            part=args.bump if args.bump else "patch",
            start_phase=args.phase,
            dry_run=args.dry_run,
            skip_pre_commit=args.skip_pre_commit,
            config=config,
        )
        summary = git.generate_markdown_summary("Phased Maintenance Bump", results)

        # Invoke Graph Indexing as part of the unified intelligence pipeline
        logger.info("Starting structural graph update phase...")
        graph_report = git.ensure_graph()
        if graph_report:
            summary += f"\n\n## Hybrid Graph Execution\n\nNodes Processed: {graph_report.nodes_processed}\nEdges Processed: {graph_report.edges_processed}\n"

        print(summary)
        git._export_report(summary, "maintenance_report.md")

    if args.install:
        results = git.install_projects()
        summary = git.generate_markdown_summary("Installation", results)
        print(summary)
        git._export_report(summary, "install_report.md")

    if args.build:
        results = git.build_projects()
        summary = git.generate_markdown_summary("Build", results)
        print(summary)
        git._export_report(summary, "build_report.md")

    if args.validate:
        git.validate_projects(type=args.type)

    # Graph Operations
    if args.graph_status:
        print(json.dumps(git.graph_status(), indent=2))
    if args.graph_reset:
        print(git.graph_reset())
    if args.graph_query:
        import asyncio

        res = asyncio.run(git.graph_query(args.graph_query, mode=args.graph_mode))
        print(json.dumps(res, indent=2))
    if args.graph_path:
        res = git.graph_path(args.graph_path[0], args.graph_path[1])
        print(json.dumps(res, indent=2))
    if args.graph_impact:
        import asyncio

        res = asyncio.run(git.graph_impact(args.graph_impact))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    """
    Execute the main function when the script is run directly.
    """
    main()
