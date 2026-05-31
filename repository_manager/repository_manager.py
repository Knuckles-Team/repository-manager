#!/usr/bin/env python


"""
A command-line tool for managing Git repositories, supporting cloning and pulling
multiple repositories in parallel using Python's multiprocessing capabilities.
"""

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

__version__ = "1.27.2"

import concurrent.futures
import multiprocessing
import shutil
import signal

import yaml  # type: ignore[import-untyped]
from agent_utilities.base_utilities import get_library_file_path, to_boolean

try:
    from skill_graphs.skill_graph_utilities import get_skill_graphs_path
    from universal_skills.skill_utilities import get_universal_skills_path
except ImportError:
    get_universal_skills_path = None
    get_skill_graphs_path = None

from importlib.resources import files

from agent_utilities.base_utilities import get_logger

from repository_manager.models import (
    GitError,
    GitMetadata,
    GitResult,
    MaintenanceConfig,
    ReadmeResult,
    SubdirectoryConfig,
    WorkspaceConfig,
)
from repository_manager.scan_models import RepoScanResult
from repository_manager.scanner import scan_repository

logger = get_logger("RepositoryManager")


def get_packaged_file_path(package: str, file: str) -> str:
    """Robustly find a file in a package using importlib.resources."""
    try:
        path = files(package).joinpath(file)
        if path.is_file():
            return str(path)
    except Exception:  # nosec B110
        pass

    local_path = os.path.join(os.path.dirname(__file__), file)
    if os.path.exists(local_path):
        return local_path

    return get_library_file_path(file=file)


# Robust environment variable retrieval with empty string fallbacks
_raw_workspace = os.getenv("REPOSITORY_MANAGER_WORKSPACE", "")
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.path.abspath(
    os.path.expanduser(_raw_workspace if _raw_workspace else "/home/apps/workspace")
)

_raw_yml = os.getenv("WORKSPACE_YML", "")
DEFAULT_WORKSPACE_YML = (
    _raw_yml
    if _raw_yml
    else get_packaged_file_path("repository_manager", "workspace.yml")
)

_raw_threads = os.getenv("REPOSITORY_MANAGER_THREADS", "")
DEFAULT_REPOSITORY_MANAGER_THREADS = int(
    _raw_threads if _raw_threads and _raw_threads.isdigit() else "6"
)

_raw_branch = os.getenv("REPOSITORY_MANAGER_DEFAULT_BRANCH", "")
DEFAULT_REPOSITORY_MANAGER_DEFAULT_BRANCH = to_boolean(
    _raw_branch if _raw_branch else "False"
)


class Git:
    """A class to handle Git operations such as cloning and pulling repositories."""

    def __init__(
        self,
        path: str | None = None,
        threads: int | None = None,
        set_to_default_branch: bool = False,
        capture_output: bool = False,
        report_path: str | None = None,
    ):
        """Initialize the Git class with default settings."""
        self._explicit_path = path is not None
        self.path = path or DEFAULT_REPOSITORY_MANAGER_WORKSPACE
        self.report_path = report_path
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:  # nosec B110
                pass

        self.project_map: dict[str, str] = {}
        self.config: WorkspaceConfig | None = None
        self.set_to_default_branch = set_to_default_branch
        self.capture_output = capture_output
        self.maximum_threads = self._cpu_aware_threads(20.0)
        self.threads = min(threads or self.maximum_threads, self.maximum_threads)
        if threads:
            self.set_threads(threads=threads)

        # Centralized debug logging under XDG logs directory of agent-utilities
        try:
            from agent_utilities.core.paths import log_dir

            logs_dir = log_dir()
        except ImportError:
            import platformdirs

            logs_dir = Path(
                platformdirs.user_log_path("agent-utilities", "knuckles-team")
            )

        logs_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_path = str(logs_dir / "repository_manager_debug.log")
        self.debug_lock = threading.Lock()
        self.python_exe = self._find_python()

        self.progress: dict[str, Any] = {
            "current_phase": "Idle",
            "progress": 0,
            "phases": {},
        }

        # Initialize log file
        with open(self.debug_log_path, "a") as f:
            f.write(f"\n\n--- NEW SESSION: {datetime.datetime.now().isoformat()} ---\n")

    def _find_python(self) -> str:
        """Finds the best Python executable to use for validation."""
        venv_path = os.path.join(self.path, ".venv", "bin", "python3")
        if os.path.exists(venv_path):
            return venv_path
        return sys.executable

    def _get_pip_command(self, extra: str = "all") -> str:
        """Get the appropriate pip install command, preferring uv if available."""
        import shutil

        pip_cmd = "pip"
        if shutil.which("uv"):
            pip_cmd = "uv pip"

        return f"{pip_cmd} install --break-system-packages -e '.[{extra}]'"

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

    def get_project_map(self) -> dict[str, str]:
        """
        Returns the mapping of repository URLs to their local project paths.
        Ensures paths are absolute and expanded.
        """
        return {
            url: os.path.abspath(os.path.expanduser(p))
            for url, p in self.project_map.items()
        }

    def get_workspace_projects(self) -> list[str]:
        """Returns a list of project basenames (e.g. 'genius-agent') defined in the workspace."""
        return [os.path.basename(p) for p in self.project_map.values()]

    def list_branches(self) -> dict[str, str]:
        """Returns a dictionary mapping project basenames to their current active git branch."""
        branches: dict[str, str] = {}
        if not self.project_map:
            return branches

        for _url, path in self.project_map.items():
            repo_name = os.path.basename(path)
            if not os.path.exists(os.path.join(path, ".git")):
                branches[repo_name] = "not-cloned"
                continue

            res = self.git_action(
                "git rev-parse --abbrev-ref HEAD", path=path, quiet=True
            )
            if res.status == "success" and res.data:
                branches[repo_name] = res.data.strip()
            else:
                branches[repo_name] = "unknown"

        return branches

    def _resolve_path(self, path: str | None = None) -> str:
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
        self, extra: str = "all", threads: int | None = None, report: bool = True
    ) -> list[GitResult]:
        """Bulk installs Python and Node projects in the workspace."""
        effective_threads = threads if threads is not None else self.threads
        threads = min(effective_threads, self._cpu_aware_threads(20.0))
        if not self.project_map:
            logger.warning("No projects to install.")
            return []

        logger.info("Installing ecosystem using native uv workspace sync...")
        results = []

        import datetime
        import shutil

        # 1. Sync Python Ecosystem natively using uv workspace
        if shutil.which("uv"):
            cmd = "uv sync --all-packages"
            # execute at base_path where it will find the workspace pyproject.toml
            res = self.git_action(cmd, path=self.path, timeout=300)

            # Generate parity reporting for each Python project
            for _url, path in self.project_map.items():
                has_precommit = os.path.exists(
                    os.path.join(path, ".pre-commit-config.yaml")
                )
                has_pyproject = os.path.exists(os.path.join(path, "pyproject.toml"))
                if not has_precommit and not has_pyproject:
                    continue

                is_python = os.path.exists(
                    os.path.join(path, "pyproject.toml")
                ) or os.path.exists(os.path.join(path, "setup.py"))
                if not is_python:
                    continue

                pkg_result = GitResult(
                    status=res.status,
                    data=res.data,
                    error=res.error,
                    metadata=GitMetadata(
                        command="install",
                        workspace=path,
                        return_code=res.metadata.return_code if res.metadata else 0,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )
                results.append(pkg_result)
        else:
            logger.warning("uv not found. Native workspace sync requires uv.")

        # 2. Install Node Ecosystem sequentially
        for _url, path in self.project_map.items():
            has_precommit = os.path.exists(
                os.path.join(path, ".pre-commit-config.yaml")
            )
            has_pyproject = os.path.exists(os.path.join(path, "pyproject.toml"))

            if not has_precommit and not has_pyproject:
                results.append(
                    GitResult(
                        status="skipped",
                        data="Skipped (No .pre-commit-config.yaml and no pyproject.toml)",
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

            is_node = os.path.exists(os.path.join(path, "package.json"))
            if is_node:
                pm = self._get_package_manager(path)
                res = self.git_action(f"{pm} install", path=path)
                if pm == "pnpm" and "Ignored build scripts:" in res.data:
                    res.status = "error"
                    res.data = f"pnpm install succeeded but ignored build scripts:\n{res.data}\nPlease add allowed dependencies to package.json."
                results.append(res)

            is_python = os.path.exists(
                os.path.join(path, "pyproject.toml")
            ) or os.path.exists(os.path.join(path, "setup.py"))
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

        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "error"]

        report_md = "# INSTALLATION SUMMARY\n"
        report_md += (
            f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
        )
        report_md += f"**Total:** {len(results)} | **Success:** {len(successes)} ✅ | **Failure:** {len(failures)} ❌\n\n"

        if successes:
            report_md += "## Successes ✅\n"
            for r in successes:
                pkg = "unknown"
                if r.metadata:
                    pkg = r.metadata.workspace.split("/")[-1]
                report_md += f"- **{pkg}**: Installation success\n"

        if failures:
            report_md += "\n## Failures ❌\n"
            for r in failures:
                pkg = "unknown"
                if r.metadata:
                    pkg = r.metadata.workspace.split("/")[-1]
                error_msg = r.error.message if r.error else r.data
                report_md += f"- **{pkg}**: {error_msg}\n"

        if self.report_path and report:
            self._export_report(report_md, "install_report.md")

        return results

    def build_projects(self, threads: int | None = None) -> list[GitResult]:
        """Bulk builds Python and Node.js projects in the workspace."""
        effective_threads = threads if threads is not None else self.threads
        threads = min(effective_threads, self._cpu_aware_threads(20.0))
        if not self.project_map:
            logger.warning("No projects to build.")
            return []

        logger.info(
            f"Building {len(self.project_map)} projects in parallel ({threads} threads)..."
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=threads) as executor:
            futures = []
            for _url, path in self.project_map.items():
                if os.path.exists(os.path.join(path, "package.json")):
                    pm = self._get_package_manager(path)
                    cmd = f"{pm} install && {pm} run build"
                else:
                    cmd = f"{sys.executable} -m build"
                futures.append(executor.submit(self.git_action, cmd, path=path))
            return [f.result() for f in concurrent.futures.as_completed(futures)]

    @staticmethod
    def _cpu_aware_threads(max_cpu_pct: float = 20.0) -> int:
        """Calculate thread count to stay under *max_cpu_pct* CPU utilisation.

        For subprocess-heavy workloads each thread drives an external process,
        so we approximate 1 thread ≈ 1 core of load.  Targeting 20% of
        available cores keeps background validation from starving the IDE and
        MCP server.
        """
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = multiprocessing.cpu_count() or 4
        target = max(1, int(cores * max_cpu_pct / 100.0))
        return target

    def validate_single_project(self, repo_path: str) -> RepoScanResult:
        """Validates a single repository by running the scanner logic."""
        logger.info(f"Validating single project: {repo_path}")

        # Optionally perform ecosystem installation before scanning if needed
        # In this implementation, we focus on pre-commit and pytest via the scanner

        return scan_repository(repo_path)

    def validate_and_release(
        self,
        threads: int | None = None,
        auto_bump: bool = False,
        auto_push: bool = False,
        bump_part: str = "minor",
    ) -> dict[str, Any]:
        """Validate projects in parallel, optionally triggering a release if successful."""
        if not self.project_map:
            logger.warning("No projects to validate.")
            return {"passed": False, "validation_results": {}, "release_results": {}}

        effective_threads = threads or self._cpu_aware_threads()
        logger.info(
            f"Validating {len(self.project_map)} projects in parallel ({effective_threads} threads)..."
        )

        validation_results: dict[str, Any] = {}
        passed = True

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_threads
        ) as executor:
            futures = {
                executor.submit(self.validate_single_project, path): url
                for url, path in self.project_map.items()
            }
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                repo_name = url.split("/")[-1].replace(".git", "")
                try:
                    result = future.result()
                    validation_results[repo_name] = (
                        result.model_dump() if hasattr(result, "model_dump") else result
                    )
                    if hasattr(result, "success"):
                        if not result.success:
                            passed = False
                    else:
                        passed = False
                except Exception as e:
                    logger.error(f"Validation exception for {repo_name}: {e}")
                    validation_results[repo_name] = {"success": False, "error": str(e)}
                    passed = False

        release_results = {}
        if passed:
            logger.info("All validations passed.")
            if auto_bump:
                logger.info(f"Triggering phased bumpversion ({bump_part})...")
                release_results["bump"] = self.phased_bumpversion(part=bump_part)
            if auto_push:
                logger.info("Triggering phased push...")
                release_results["push"] = self.phased_push()
        else:
            if auto_bump or auto_push:
                logger.warning("Validation failed. Skipping bump and push.")

        return {
            "passed": passed,
            "validation_results": validation_results,
            "release_results": release_results,
        }

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

    @staticmethod
    def generate_markdown_summary(action: str, results: list[GitResult]) -> str:
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
                name = (
                    os.path.basename(r.metadata.workspace) if r.metadata else "unknown"
                )

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
                name = (
                    os.path.basename(r.metadata.workspace) if r.metadata else "unknown"
                )
                err_msg = r.error.message if r.error else "Unknown error"
                md.append(f"### ⚠️ {name}")
                if r.metadata:
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
            reasons: dict[str, list[str]] = {}
            for r in skips:
                reason = r.data or "No reason provided"
                if reason not in reasons:
                    reasons[reason] = []
                reasons[reason].append(
                    os.path.basename(r.metadata.workspace) if r.metadata else "unknown"
                )

            for reason, projects in sorted(reasons.items()):
                project_list = ", ".join(sorted(list(set(projects))))
                md.append(f"- **{reason}**: {project_list}")
            md.append("")

        return "\n".join(md)

    def remediate_projects(self, repositories: list[str] | None = None) -> dict:
        """Automatically remediate common validation failures in target projects."""

        if not self.project_map:
            self.get_project_map()

        target_repos = repositories or []
        if not target_repos:
            # Gather all if none specified
            for url in self.project_map.keys():
                target_repos.append(url.split("/")[-1].replace(".git", ""))

        results: dict[str, list[str]] = {"success": [], "errors": []}

        for repo_name in target_repos:
            repo_path = None
            for url, path in self.project_map.items():
                if url.endswith(repo_name) or url.endswith(repo_name + ".git"):
                    repo_path = path
                    break

            if not repo_path or not os.path.exists(repo_path):
                results["errors"].append(f"{repo_name}: path not found")
                continue

            try:
                # 1. Sync bumpversion in Dockerfile
                cfg_path = os.path.join(repo_path, ".bumpversion.cfg")
                docker_path = os.path.join(repo_path, "docker", "Dockerfile")
                if os.path.exists(cfg_path) and os.path.exists(docker_path):
                    with open(cfg_path) as f:
                        cfg_content = f.read()
                    import re

                    m = re.search(r"search = ([a-zA-Z0-9_-]+)>=([0-9\.]+)", cfg_content)
                    if m:
                        pkg_name = m.group(1)
                        version = m.group(2)
                        with open(docker_path) as df:
                            d_content = df.read()

                        # Replace something like github-agent>=0.10.0 with github-agent>=0.11.0
                        d_content_new = re.sub(
                            rf"{pkg_name}>=[0-9\.]+",
                            f"{pkg_name}>={version}",
                            d_content,
                        )
                        if d_content_new != d_content:
                            with open(docker_path, "w") as df:
                                df.write(d_content_new)
                            results["success"].append(
                                f"{repo_name}: Fixed Dockerfile version to {version}"
                            )

                # 2. Run EOF fixer via pre-commit if possible
                import subprocess

                subprocess.run(  # nosec B607 — uv/pre-commit are workspace-managed tools
                    [
                        "uv",
                        "run",
                        "pre-commit",
                        "run",
                        "end-of-file-fixer",
                        "--all-files",
                    ],
                    cwd=repo_path,
                    capture_output=True,
                )
                subprocess.run(  # nosec B607 — uv/ruff are workspace-managed tools
                    ["uv", "run", "ruff", "format"], cwd=repo_path, capture_output=True
                )

            except Exception as e:
                results["errors"].append(f"{repo_name}: {str(e)}")

        return results

    def git_action(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict | None = None,
        timeout: int = 1800,
    ) -> GitResult:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            path (str, optional): The directory to execute the command in.
                Defaults to the base path.

        Returns:
            GitResult: The combined stdout and stderr output of the command in structured format.

        Concept:
            CONCEPT:RM-GIT-ACTION
        """
        target_path = self._resolve_path(path)

        # Ensure ~/.local/bin is in PATH for tools like bump2version
        current_env = env if env else os.environ.copy()
        local_bin = os.path.expanduser("~/.local/bin")
        if local_bin not in current_env.get("PATH", ""):
            current_env["PATH"] = f"{local_bin}:{current_env.get('PATH', '')}"

        # Ensure Python output is unbuffered so we get real-time logs
        current_env["PYTHONUNBUFFERED"] = "1"

        logger.info(f"Executing: {command} in {target_path}")

        process = subprocess.Popen(
            command,
            shell=True,  # nosec B602
            cwd=target_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=current_env,
            bufsize=1,  # Line buffered
            start_new_session=True,  # Isolate process group so killpg only kills the command
        )

        output_lines = []
        try:
            # Write start marker
            with self.debug_lock:
                with open(self.debug_log_path, "a") as log_file:
                    log_file.write(
                        f"\n[{datetime.datetime.now().isoformat()}] Starting: {command}\n"
                    )
                    log_file.write(
                        f"[{datetime.datetime.now().isoformat()}] CWD: {target_path}\n"
                    )
                    log_file.flush()

            # Read output line by line as it becomes available
            def _read_output():
                if process.stdout:
                    for line in process.stdout:
                        output_lines.append(line)
                        with self.debug_lock:
                            with open(self.debug_log_path, "a") as log_file:
                                log_file.write(
                                    f"[{datetime.datetime.now().isoformat()}] {line}"
                                )
                                log_file.flush()

            reader_thread = threading.Thread(target=_read_output, daemon=True)
            reader_thread.start()

            # Wait for process to complete, with a safety timeout
            process.wait(timeout=timeout)
            reader_thread.join(timeout=1.0)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning(f"Command timed out: {command}")
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait(timeout=5)
                except Exception:  # nosec B110
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

            with self.debug_lock:
                with open(self.debug_log_path, "a") as log_file:
                    log_file.write(
                        f"[{datetime.datetime.now().isoformat()}] ERROR: Command timed out after {timeout} seconds\n"
                    )
                    log_file.flush()

        out = "".join(output_lines)
        return_code = process.returncode

        metadata = GitMetadata(
            command=command,
            workspace=target_path,
            return_code=return_code,
            timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat() + "Z",
        )

        error_obj = None
        if return_code != 0:
            error_obj = GitError(
                message=out.strip() if out else "Unknown error",
                code=return_code,
            )

        result = GitResult(
            status="success" if return_code == 0 else "error",
            data=out.strip() if out else "",
            error=error_obj,
            metadata=metadata,
        )

        if result.status == "error":
            err_msg = result.error.message if result.error else "Unknown error"
            logger.error(f"Command failed: {command}\nError: {err_msg}")
        elif not quiet:
            logger.info(f"Command: {command}\nOutput: {result.data}")

        return result

    def cleanup_artifacts(self, target_dir: str) -> None:
        """Removes test artifacts and temporary files from the specified directory."""
        import shutil
        from pathlib import Path

        dir_path = Path(target_dir)
        if not dir_path.exists():
            return

        patterns_to_remove = [
            "knowledge_graph.db*",
            "*.db-wal",
            "*.db-shm",
            "*.wal",
            "*.log",
            "session<MagicMock*",
            "coverage.xml",
            ".coverage",
            "*.orig",
            "*.rej",
            "*.patch",
            "failed_tests.txt",
            "pytest_errors.txt",
            "pytest_output.txt",
            "mypy_errors.txt",
            "mypy_output.txt",
            "pre-commit-out.txt",
            "cargo_check.log",
            "check.log",
            "check_out.txt",
            "test_out.txt",
            "trace.txt",
        ]

        dir_patterns_to_remove = {
            ".pytest_cache",
            "htmlcov",
            "agent_data",
        }

        ignored_dirs = {".venv", "node_modules", ".git"}

        # Use os.walk with top-down pruning to avoid iterating massive directories
        for dirpath, dirnames, filenames in os.walk(target_dir, topdown=True):
            # Prune ignored directories in-place (prevents os.walk from descending)
            dirnames[:] = [d for d in dirnames if d not in ignored_dirs]

            # Check for directory-level cleanup targets
            for d in list(dirnames):
                if d in dir_patterns_to_remove:
                    full_path = os.path.join(dirpath, d)
                    try:
                        shutil.rmtree(full_path)
                        logger.debug(f"Cleaned up directory: {full_path}")
                    except Exception as e:
                        logger.debug(f"Failed to clean up directory {full_path}: {e}")
                    dirnames.remove(d)

            # Check for root-level transient scripts and non-standard text files (only at target_dir root)
            if dirpath == target_dir:
                root_patterns = [
                    "test_*.py",
                    "fix_*.py",
                    "debug_*.py",
                    "scratch_*.py",
                    "temp_*.py",
                ]
                for f in filenames:
                    file_path = Path(os.path.join(dirpath, f))
                    # 1. Clean up root-level python transient scripts
                    matched_script = False
                    for pat in root_patterns:
                        if file_path.match(pat):
                            try:
                                file_path.unlink()
                                logger.info(
                                    f"Cleaned up root transient script: {file_path}"
                                )
                            except Exception as e:
                                logger.debug(
                                    f"Failed to clean up root script {file_path}: {e}"
                                )
                            matched_script = True
                            break
                    if matched_script:
                        continue
                    # 2. Clean up root-level non-standard *.txt files (exclude requirements.txt and requirements-dev.txt)
                    if file_path.suffix == ".txt" and file_path.name not in (
                        "requirements.txt",
                        "requirements-dev.txt",
                    ):
                        try:
                            file_path.unlink()
                            logger.info(
                                f"Cleaned up root non-standard text file: {file_path}"
                            )
                        except Exception as e:
                            logger.debug(
                                f"Failed to clean up root text file {file_path}: {e}"
                            )

            # Check for file-level cleanup targets
            for f in filenames:
                file_path = Path(os.path.join(dirpath, f))
                for pat in patterns_to_remove:
                    if file_path.match(pat):
                        try:
                            file_path.unlink()
                            logger.debug(f"Cleaned up file: {file_path}")
                        except Exception as e:
                            logger.debug(f"Failed to clean up file {file_path}: {e}")
                        break

    def clone_projects(self, projects: list[str] | None = None) -> list[GitResult]:
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
            if projects:
                for url in projects:
                    name = url.split("/")[-1].replace(".git", "")
                    targets.append((url, os.path.join(expanded_path, name)))
            elif self.project_map:
                for url, path in self.project_map.items():
                    targets.append((url, path))

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

    def pull_projects(self, project_dirs: list[str] | None = None) -> list[GitResult]:
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

    def pull_project(self, path: str | None = None) -> GitResult:
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
            [
                f"[{r.metadata.command if r.metadata else 'unknown'}]: {r.data}"
                for r in results
            ]
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

    def push_projects(self, project_dirs: list[str] | None = None) -> list[GitResult]:
        """
        Push updates for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to push.")
                return []

        if not project_dirs:
            logger.warning("No projects found to push.")
            return []

        logger.info(
            f"Pushing {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(self.push_project, project_dirs))

    def push_project(self, path: str | None = None) -> GitResult:
        """
        Push updates and tags for a single Git project, ensuring all staged and unstaged changes are committed first.

        Handles common failure modes:
        - Non-fast-forward: auto-rebases and retries (up to 2 attempts)
        - GitHub secret scanning (GH013): returns actionable error with unblock URL
        - Tag conflicts: falls back to pushing without --follow-tags
        """
        target_path = self._resolve_path(path)
        logger.info(f"Checking for uncommitted changes in {target_path} before pushing")

        status_check = self.git_action(
            command="git status --porcelain", path=target_path, quiet=True
        )
        if status_check.status == "success" and status_check.data.strip():
            logger.info(
                f"Detected uncommitted changes in {target_path}. Staging and committing them first."
            )
            add_res = self.git_action(command="git add -u", path=target_path)
            if add_res.status != "success":
                logger.error(
                    f"Failed to stage changes in {target_path}: {add_res.error}"
                )
            else:
                commit_res = self.git_action(
                    command='git commit --no-verify -m "phased push uncommitted changes"',
                    path=target_path,
                )
                if commit_res.status != "success":
                    logger.error(
                        f"Failed to commit changes in {target_path}: {commit_res.error}"
                    )
                else:
                    logger.info(
                        f"Successfully committed uncommitted changes in {target_path}"
                    )

        logger.info(f"Pushing latest changes and tags for {target_path}")

        max_attempts = 2
        for attempt in range(1, max_attempts + 1):
            result = self.git_action(command="git push --follow-tags", path=target_path)

            if result.status == "success":
                return result

            error_text = ""
            if result.error:
                error_text = (
                    str(result.error.message)
                    if hasattr(result.error, "message")
                    else str(result.error)
                )
            if result.data:
                error_text += " " + result.data

            # GitHub secret scanning block (GH013) — unrecoverable without manual action
            if "GH013" in error_text or "GITHUB PUSH PROTECTION" in error_text:
                logger.error(
                    f"GitHub secret scanning blocked push for {target_path}. "
                    "Remove the secret from git history or allow it via the GitHub URL in the error output."
                )
                return GitResult(
                    status="error",
                    data=error_text,
                    error=GitError(
                        message=f"GitHub secret scanning (GH013) blocked push for {target_path}. "
                        "A file in the commit history contains a detected secret. "
                        "Use git-filter-repo to expunge it or allow the secret via GitHub settings.",
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="git push --follow-tags",
                        workspace=target_path,
                        return_code=1,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )

            # Non-fast-forward: rebase and retry
            if (
                "non-fast-forward" in error_text
                or "tip of your current branch is behind" in error_text
            ):
                if attempt < max_attempts:
                    logger.warning(
                        f"Non-fast-forward rejected for {target_path}. Attempting rebase (attempt {attempt}/{max_attempts})..."
                    )
                    rebase_res = self.git_action(
                        command="git pull --rebase origin main", path=target_path
                    )
                    if rebase_res.status != "success":
                        rebase_err = (
                            str(rebase_res.error.message)
                            if rebase_res.error and hasattr(rebase_res.error, "message")
                            else str(rebase_res.error or "")
                        )
                        if "CONFLICT" in rebase_err or "could not apply" in rebase_err:
                            # Abort the failed rebase and try force push instead
                            self.git_action(
                                command="git rebase --abort", path=target_path
                            )
                            logger.warning(
                                f"Rebase conflicts in {target_path}. Falling back to force push."
                            )
                            force_result = self.git_action(
                                command="git push --force origin main", path=target_path
                            )
                            if force_result.status == "success":
                                return force_result
                            return force_result
                        return rebase_res
                    continue  # Retry the push after successful rebase
                else:
                    logger.warning(
                        f"Non-fast-forward still failing after rebase for {target_path}. Force pushing."
                    )
                    return self.git_action(
                        command="git push --force origin main", path=target_path
                    )

            # Tag already exists on remote — retry without tags
            if "tag already exists" in error_text or "already exists" in error_text:
                logger.warning(
                    f"Tag conflict for {target_path}. Retrying push without --follow-tags."
                )
                return self.git_action(command="git push origin main", path=target_path)

            # Unknown error — return as-is
            return result

        return result

    def add_projects(self, project_dirs: list[str] | None = None) -> list[GitResult]:
        """
        Stage all changes for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to add.")
                return []

        if not project_dirs:
            logger.warning("No projects found to add.")
            return []

        logger.info(
            f"Staging changes in {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(self.add_project, project_dirs))

    def add_project(self, path: str | None = None) -> GitResult:
        """
        Stage all changes (git add -A) for a single Git project.
        """
        target_path = self._resolve_path(path)
        logger.info(f"Staging all changes for {target_path}")
        return self.git_action(command="git add -A", path=target_path)

    def commit_projects(
        self, message: str, project_dirs: list[str] | None = None
    ) -> list[GitResult]:
        """
        Commit staged changes for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to commit.")
                return []

        if not project_dirs:
            logger.warning("No projects found to commit.")
            return []

        logger.info(
            f"Committing changes in {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        from functools import partial

        commit_func = partial(self.commit_project, message)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(commit_func, project_dirs))

    def commit_project(self, message: str, path: str | None = None) -> GitResult:
        """
        Commit staged changes (git commit -m "{message}") for a single Git project.
        """
        target_path = self._resolve_path(path)

        # Check if there are staged changes to commit
        status_res = self.git_action(command="git status --porcelain", path=target_path)
        if status_res.status == "success":
            # Check porcelain output for staged changes
            has_staged = False
            for line in status_res.data.splitlines():
                if line and not line.startswith("?"):
                    # Staged changes are indicated when the first character is not a space/untracked status
                    if line[0] not in (" ", "?"):
                        has_staged = True
                        break

            if not has_staged:
                logger.info(f"No staged changes to commit for {target_path}")
                metadata = GitMetadata(
                    command=f'git commit -m "{message}"',
                    workspace=target_path,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                )
                return GitResult(
                    status="success",
                    data="No staged changes to commit (skipped)",
                    error=None,
                    metadata=metadata,
                )

        logger.info(f"Committing staged changes for {target_path}")
        from shlex import quote

        safe_msg = quote(message)
        return self.git_action(
            command=f"git commit --no-verify -m {safe_msg}", path=target_path
        )

    def set_threads(self, threads: int) -> None:
        """
        Set the number of threads for parallel processing.

        Args:
            threads (int): The number of threads.

        Notes:
            If the input is invalid, defaults 6
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
        path: str | None = None,
    ) -> GitResult:
        """
        Execute pre-commit commands in the specified path.

        Args:
            run (bool): Whether to run 'pre-commit run --all-files'. Default True.
            autoupdate (bool): Whether to run 'pre-commit autoupdate'. Default False.
            path (str, optional): Path to run in. Defaults to self.path.
        """
        target_path = self._resolve_path(path)

        # Clean artifacts before running pre-commit
        self.cleanup_artifacts(target_path)

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
            commands.append("pre-commit autoupdate && git add -A")
        if run:
            commands.append("git add -A")
            # Run pre-commit once. If it fails (likely due to auto-formatting files),
            # stage the newly formatted changes and run it again to verify.
            commands.append(
                "SKIP=no-commit-to-branch pre-commit run --all-files --verbose || (git add -A && SKIP=no-commit-to-branch pre-commit run --all-files --verbose && git add -A)"
            )

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
        env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = "4"
        env["PYTEST_ADDOPTS"] = '-q --tb=short -m "not slow" --timeout=60'

        result = self.git_action(
            command=full_command, path=target_path, env=env, timeout=600
        )

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

    def _run_project_test(
        self, cmd: str, path: str, env: dict, timeout: int
    ) -> list[GitResult]:
        results = []
        res = self.git_action(cmd, path=path, env=env, timeout=timeout)
        results.append(res)
        return results

    def test_projects(
        self,
        targets: list[dict[str, str]],
        progress_phase: str | None = None,
        progress_dict: dict | None = None,
    ) -> list[GitResult]:
        """
        Execute pytests for the specified projects in parallel.

        Args:
            progress_phase: Phase name for live progress updates.
            progress_dict: Shared mutable dict for live progress reporting.
        """
        results = []
        thread_count = self._cpu_aware_threads()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count
        ) as executor:
            future_to_repo: dict[concurrent.futures.Future, str] = {}
            for target in targets:
                if "skip_reason" in target:
                    continue

                path = target["path"]
                repo_name = target.get("name", os.path.basename(path))

                has_precommit = os.path.exists(
                    os.path.join(path, ".pre-commit-config.yaml")
                )
                has_pyproject = os.path.exists(os.path.join(path, "pyproject.toml"))
                if not has_precommit and not has_pyproject:
                    results.append(
                        GitResult(
                            status="skipped",
                            data="Skipped (No .pre-commit-config.yaml and no pyproject.toml)",
                            metadata=GitMetadata(
                                command="pytest",
                                workspace=path,
                                return_code=0,
                                timestamp=datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat()
                                + "Z",
                            ),
                        )
                    )
                    if progress_dict and progress_phase:
                        phases = progress_dict.get("phases", {})
                        if progress_phase in phases:
                            phases[progress_phase]["repos"][repo_name] = "skipped"
                            phases[progress_phase]["completed"] = len(
                                phases[progress_phase]["repos"]
                            )
                    continue

                # Check for the existence of unit test directories first, then general test directories
                test_target = None
                for candidate in ["tests/unit", "test/unit", "tests", "test"]:
                    if os.path.exists(os.path.join(path, candidate)):
                        test_target = candidate
                        break

                if test_target is not None:
                    # Detect if we should use uv or standard python
                    if os.path.exists(os.path.join(path, "uv.lock")):
                        cmd = f'uv run --extra test pytest {test_target} -q --tb=short -m "not slow" --timeout=60'
                    else:
                        cmd = f'{sys.executable} -m pytest {test_target} -q --tb=short -m "not slow" --timeout=60'

                    # Ensure memory safety for ladybug and set validation mode
                    test_env = os.environ.copy()
                    test_env["LADYBUG_MAX_DB_SIZE"] = "1073741824"
                    test_env["VALIDATION_MODE"] = "True"
                    test_env["KNOWLEDGE_GRAPH_SYNC_BACKGROUND"] = "False"
                    test_env["GRAPH_DB_PATH"] = ":memory:"

                    fut = executor.submit(
                        self._run_project_test,
                        cmd,
                        path,
                        test_env,
                        600,  # 10 minute timeout for tests
                    )
                    future_to_repo[fut] = repo_name
                else:
                    results.append(
                        GitResult(
                            status="skipped",
                            data="No tests directory found",
                            metadata=GitMetadata(
                                command="pytest",
                                workspace=path,
                                return_code=0,
                                timestamp=datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat()
                                + "Z",
                            ),
                        )
                    )
                    if progress_dict and progress_phase:
                        phases = progress_dict.get("phases", {})
                        if progress_phase in phases:
                            phases[progress_phase]["repos"][repo_name] = "skipped"
                            phases[progress_phase]["completed"] = len(
                                phases[progress_phase]["repos"]
                            )

            for future in concurrent.futures.as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                res_list = future.result()
                # Determine aggregate status for the repo
                status = "success"
                if isinstance(res_list, list):
                    results.extend(res_list)
                    if any(r.status == "error" for r in res_list):
                        status = "error"
                else:
                    results.append(res_list)
                    status = res_list.status
                # Update live progress
                if progress_dict and progress_phase:
                    phases = progress_dict.get("phases", {})
                    if progress_phase in phases:
                        phases[progress_phase]["repos"][repo_name] = status
                        phases[progress_phase]["completed"] = len(
                            phases[progress_phase]["repos"]
                        )
                        phases[progress_phase]["failed"] = sum(
                            1
                            for s in phases[progress_phase]["repos"].values()
                            if s == "error"
                        )
        return results

    def pre_commit_projects(
        self,
        run: bool = True,
        autoupdate: bool = False,
        projects: list[str] | None = None,
    ) -> list[GitResult]:
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
                project_dirs = []
                for p in projects:
                    p_path = None
                    if os.path.isabs(p) and os.path.exists(p):
                        p_path = p
                    else:
                        for url, path in self.project_map.items():
                            if url.endswith(f"/{p}.git") or url.endswith(f"/{p}"):
                                p_path = path
                                break
                    if (
                        p_path
                        and os.path.isdir(p_path)
                        and os.path.exists(
                            os.path.join(p_path, ".pre-commit-config.yaml")
                        )
                    ):
                        project_dirs.append(p_path)

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

    def install_project(self, path: str | None = None, extra: str = "all") -> GitResult:
        """
        Install a Python project using pip install -e .[extra].
        """
        target_path = self._resolve_path(path)

        command = self._get_pip_command(extra)

        logger.info(f"Installing project at {target_path} with {command}")
        result = self.git_action(command=command, path=target_path)

        for d in ["build", "dist"]:
            shutil.rmtree(os.path.join(target_path, d), ignore_errors=True)
        for egg_info in Path(target_path).glob("*.egg-info"):
            shutil.rmtree(egg_info, ignore_errors=True)

        return result

    def get_readme(self, path: str | None = None) -> ReadmeResult:
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
            with open(readme_path, encoding="utf-8") as f:
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
        path: str | None = None,
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

        # Check if the project has a bump2version configuration file.
        has_cfg = os.path.exists(os.path.join(target_dir, ".bumpversion.cfg"))
        if not has_cfg and os.path.exists(os.path.join(target_dir, "setup.cfg")):
            try:
                with open(os.path.join(target_dir, "setup.cfg"), encoding="utf-8") as f:
                    if "[bumpversion]" in f.read():
                        has_cfg = True
            except Exception as e:
                logger.debug(f"Could not read setup.cfg in {target_dir}: {e}")

        if not has_cfg:
            # Fallback behavior: stage all changes and commit them as "phased bump"
            status_check = self.git_action(
                command="git status --porcelain", path=target_dir, quiet=True
            )
            if status_check.status != "success":
                return status_check

            changed_files = status_check.data.strip()
            if not changed_files:
                logger.info(f"No changes to stage or commit in {target_dir}. Skipping.")
                return GitResult(
                    status="skipped",
                    data="No changes to stage or commit (fallback mode)",
                    metadata=GitMetadata(
                        command="bump_version",
                        workspace=target_dir,
                        return_code=0,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )

            if dry_run:
                logger.info(
                    f"[DRY RUN] Would fallback to git add -A && git commit -m 'phased bump' in {target_dir}"
                )
                return GitResult(
                    status="success",
                    data="current_version=unknown\nnew_version=unknown\n",
                    metadata=GitMetadata(
                        command="bump_version",
                        workspace=target_dir,
                        return_code=0,
                        timestamp=datetime.datetime.now(
                            datetime.timezone.utc
                        ).isoformat()
                        + "Z",
                    ),
                )

            add_res = self.git_action(command="git add -u", path=target_dir)
            if add_res.status != "success":
                logger.error(f"Failed to add changes in {target_dir}: {add_res.error}")
                return add_res

            commit_res = self.git_action(
                command='git commit --no-verify -m "phased bump"', path=target_dir
            )
            if commit_res.status != "success":
                logger.error(
                    f"Failed to commit fallback changes in {target_dir}: {commit_res.error}"
                )
                return commit_res

            logger.info(
                f"Successfully committed fallback changes with 'phased bump' in {target_dir}"
            )
            return GitResult(
                status="success",
                data="current_version=unknown\nnew_version=unknown\n",
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=target_dir,
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
                    + "Z",
                ),
            )

        command = (
            f"SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build bump2version {part}"
        )
        if allow_dirty:
            command += " --allow-dirty"
        if dry_run:
            command += " --dry-run"
        if verbose:
            command += " --verbose"

        # Pre-flight check for existing tags
        if not dry_run:
            pre_cmd = f"bump2version {part} --dry-run --list"
            if allow_dirty:
                pre_cmd += " --allow-dirty"
            pre_result = self.git_action(command=pre_cmd, path=target_dir, quiet=True)
            if pre_result.status == "success":
                match = re.search(r"new_version=(.*)", pre_result.data)
                if match:
                    new_version = match.group(1).strip()
                    tag_check = self.git_action(
                        command=f"git tag -l v{new_version}",
                        path=target_dir,
                        quiet=True,
                    )
                    if (
                        tag_check.status == "success"
                        and f"v{new_version}" in tag_check.data
                    ):
                        logger.warning(
                            f"Tag v{new_version} already exists in {target_dir}. Skipping bump."
                        )
                        return GitResult(
                            status="success",
                            data=f"current_version={new_version}\nnew_version={new_version}\n",
                            metadata=GitMetadata(
                                command="bump_version",
                                workspace=target_dir,
                                return_code=0,
                                timestamp=datetime.datetime.now(
                                    datetime.timezone.utc
                                ).isoformat()
                                + "Z",
                            ),
                        )
            command += " --list"

        try:
            result = self.git_action(command=command, path=target_dir)

            if result.status == "success":
                logger.info(f"Bumped version ({part}) in {target_dir}")

                if not dry_run:
                    # Synchronize uv.lock after pyproject.toml version bump
                    uv_lock_path = os.path.join(target_dir, "uv.lock")
                    if os.path.exists(uv_lock_path):
                        self.git_action(command="uv lock", path=target_dir, quiet=True)

                    # Stage all changes (staged and uncommitted/unstaged changes) in the workspace
                    self.git_action(command="git add -u", path=target_dir, quiet=True)
                    status_check = self.git_action(
                        command="git status --porcelain",
                        path=target_dir,
                        quiet=True,
                    )
                    if status_check.data.strip():
                        # Commit all staged changes (including version bump, uv.lock, and other files) into the bump commit
                        self.git_action(
                            command="SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build git commit --amend --no-edit --no-verify",
                            path=target_dir,
                            quiet=True,
                        )

                        # Move the tag to point to the newly amended commit
                        match = re.search(r"new_version=(.*)", result.data)
                        if match:
                            new_version = match.group(1).strip()
                            self.git_action(
                                command=f"git tag -f v{new_version}",
                                path=target_dir,
                                quiet=True,
                            )
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
        exclude: list[str] | None = None,
        verbose: bool = False,
    ) -> list[GitResult]:
        """Bumps the version for all projects in the workspace in parallel."""
        exclude = exclude or []
        results = []

        for url, path in self.project_map.items():
            name = url.split("/")[-1].replace(".git", "")
            if name in exclude:
                continue

            project_dir = Path(path)
            results.append(
                self.bump_version(
                    part,
                    allow_dirty=True,
                    path=str(project_dir),
                    dry_run=dry_run,
                    verbose=verbose,
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
        pattern = rf'(["\']{package_name}(?:\[.*?\])?\s*>=?\s*)\d+\.\d+\.\d+'
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
        allow_pre_commit: bool = False,
        config: dict | None = None,
        single_phase: bool = False,
        project_filter: str | None = None,
        progress: dict | None = None,
        force: bool = False,
    ) -> list[GitResult]:
        """
        Execute the phased bumpversion workflow: pre-commits + phased bumping.

        Concept:
            CONCEPT:RM-BUMP
        """
        if progress is None:
            progress = self.progress

        all_results = []
        if config is None:
            config_model: MaintenanceConfig | None = None
            if hasattr(self, "config") and self.config and self.config.maintenance:
                config_model = self.config.maintenance
            else:
                yml_path = os.environ.get("WORKSPACE_YML") or "workspace.yml"
                if not os.path.isabs(yml_path):
                    yml_path = os.path.join(self.path, yml_path)

                if os.path.exists(yml_path):
                    if self.load_projects_from_yaml(yml_path) and self.config:
                        config_model = self.config.maintenance
                    else:
                        config_model = None
                else:
                    config_model = None

            maintenance_cfg = None
            if config_model:
                maintenance_cfg = config_model.model_dump()
            else:
                logger.error("No maintenance configuration found.")
                return []

            config = maintenance_cfg

        # Build a map of project_name -> phase_num for topological dependency check
        project_phases: dict[str, int] = {}
        bulk_phase_num = 5
        if config:
            for phase in config.get("phases", []):
                p_num = phase.get("phase", 1)
                if phase.get("bulk_bump"):
                    bulk_phase_num = p_num
                projects_in_phase = phase.get("projects", [])[:]
                if phase.get("project"):
                    projects_in_phase.append(phase.get("project"))
                for p in projects_in_phase:
                    project_phases[p] = p_num

        def get_project_phase(proj_name: str) -> int:
            return project_phases.get(proj_name, bulk_phase_num)

        if allow_pre_commit:
            projects_to_check: list[Any] | None = None
            if config:
                projects_to_check = []
                has_bulk = False
                for phase in config.get("phases", []):
                    if phase.get("bulk_bump"):
                        has_bulk = True
                        break
                    projects_to_check.extend(phase.get("projects", []))
                    if phase.get("project"):
                        projects_to_check.append(phase.get("project"))
                if has_bulk:
                    projects_to_check = None

            pc_results = self.pre_commit_projects(
                run=True, autoupdate=True, projects=projects_to_check
            )
            all_results.extend(pc_results)

            dirs_to_commit = None
            if projects_to_check is not None:
                dirs_to_commit = []
                for p_name in projects_to_check:
                    for url, p_path in self.project_map.items():
                        if url.endswith(f"/{p_name}.git") or url.endswith(f"/{p_name}"):
                            dirs_to_commit.append(p_path)
                            break
            else:
                dirs_to_commit = list(self.project_map.values())

            commit_res = self.commit_projects(
                message="chore: pre-commit autoupdate and formatting",
                project_dirs=dirs_to_commit,
            )
            all_results.extend(commit_res)

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

                if not force:
                    status_check = self.git_action("git status", path=project_dir)
                    data_lower = status_check.data.lower() if status_check.data else ""
                    if (
                        "nothing to commit" in data_lower
                        and "your branch is up to date" in data_lower
                    ):
                        logger.info(
                            f"Skipping bump for {project_name}: no code changes detected (use force=True to override)"
                        )
                        return "skipped"

                result = self.bump_version(
                    part=part,
                    allow_dirty=True,
                    path=project_dir,
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

        # Pre-expand phases & projects for progress tracking
        processed_projects: set[str] = set()
        phase_list = []
        total_projects = 0

        for phase in config.get("phases", []):
            phase_num = phase.get("phase")
            if phase_num < start_phase:
                continue

            projects = phase.get("projects", [])[:]
            if phase.get("project"):
                projects.append(phase.get("project"))

            if project_filter:
                projects = [p for p in projects if p == project_filter]

            # Also apply project filter to bulk operations if applicable
            if not projects and phase.get("bulk_bump"):
                if project_filter not in processed_projects:
                    projects = [project_filter]

            if phase.get("bulk_bump") and not project_filter:
                bulk_projects = []
                for url, _ in self.project_map.items():
                    name = url.split("/")[-1].replace(".git", "")
                    if name not in processed_projects and name not in projects:
                        bulk_projects.append(name)
                projects.extend(bulk_projects)

            if not projects:
                continue

            phase_name = phase.get("name", f"Phase {phase_num}")
            phase_list.append(
                {
                    "phase_num": phase_num,
                    "name": phase_name,
                    "projects": projects,
                }
            )
            total_projects += len(projects)

        if progress is not None:
            progress["current_phase"] = "Initializing Bumps"
            progress["progress"] = 0
            progress["phases"] = {}
            for p_info in phase_list:
                progress["phases"][p_info["name"]] = {
                    "status": "pending",
                    "total": len(p_info["projects"]),
                    "processed": 0,
                    "completed": 0,
                    "success": 0,
                    "failed": 0,
                    "details": {proj: "pending" for proj in p_info["projects"]},
                    "repos": {proj: "pending" for proj in p_info["projects"]},
                }

        processed_count = 0
        for p_info in phase_list:
            phase_name = p_info["name"]
            phase_num = p_info["phase_num"]
            projects = p_info["projects"]

            if progress is not None:
                progress["current_phase"] = f"{phase_name} in progress"
                progress["phases"][phase_name]["status"] = "running"

            for project_name in projects:
                if progress is not None:
                    progress["phases"][phase_name]["details"][project_name] = "running"
                    progress["phases"][phase_name]["repos"][project_name] = "running"

                processed_projects.add(project_name)
                logger.info(
                    f"Bumping version for project: {project_name} in {phase_name}..."
                )
                new_version = run_step_bump(project_name, phase_num)

                status_str = "success" if new_version else "failed"

                if progress is not None:
                    progress["phases"][phase_name]["details"][project_name] = status_str
                    progress["phases"][phase_name]["repos"][project_name] = status_str
                    progress["phases"][phase_name]["processed"] += 1
                    progress["phases"][phase_name]["completed"] += 1
                    if status_str == "success":
                        progress["phases"][phase_name]["success"] += 1
                    else:
                        progress["phases"][phase_name]["failed"] += 1

                    processed_count += 1
                    progress["progress"] = int((processed_count / total_projects) * 100)
                    logger.info(
                        f"[{processed_count}/{total_projects}] ({progress['progress']}%) "
                        f"Completed bump for {project_name}: {status_str}"
                    )

                if new_version and re.match(r"^v?\d+\.\d+\.\d+", new_version):
                    for _, path in self.project_map.items():
                        other_project_name = os.path.basename(path)
                        other_phase = get_project_phase(other_project_name)
                        if other_phase < phase_num:
                            logger.info(
                                f"Skipping dependency update for {project_name} in {other_project_name} "
                                f"to avoid circular updates of earlier phase (Phase {other_phase} < Phase {phase_num})"
                            )
                            continue

                        pyproject = Path(path) / "pyproject.toml"
                        if pyproject.exists():
                            is_updated = self.update_dependency(
                                str(pyproject), project_name, new_version, dry_run
                            )
                            if is_updated:
                                all_results.append(
                                    GitResult(
                                        status="success",
                                        data=f"Updated {project_name} to {new_version} in pyproject.toml",
                                        metadata=GitMetadata(
                                            command="update_dependency",
                                            workspace=path,
                                            return_code=0,
                                            timestamp=datetime.datetime.now(
                                                datetime.timezone.utc
                                            ).isoformat()
                                            + "Z",
                                        ),
                                    )
                                )

            if progress is not None:
                progress["phases"][phase_name]["status"] = "completed"

        if progress is not None:
            progress["current_phase"] = "Bumps Completed"
            progress["progress"] = 100

        return all_results

    maintain_projects = phased_bumpversion

    def phased_push(
        self,
        start_phase: int = 1,
        config: dict | None = None,
        single_phase: bool = False,
        project_filter: str | None = None,
        progress: dict | None = None,
    ) -> list[GitResult]:
        """
        Execute the phased git push workflow.

        Concept:
            CONCEPT:RM-PUSH
        """
        import time

        if progress is None:
            progress = self.progress

        all_results = []
        if config is None:
            config_model: MaintenanceConfig | None = None
            if hasattr(self, "config") and self.config and self.config.maintenance:
                config_model = self.config.maintenance
            else:
                yml_path = os.environ.get("WORKSPACE_YML") or "workspace.yml"
                if not os.path.isabs(yml_path):
                    yml_path = os.path.join(self.path, yml_path)

                if os.path.exists(yml_path):
                    if self.load_projects_from_yaml(yml_path) and self.config:
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

        processed_projects = set()
        phase_list: list[dict[str, Any]] = []
        total_projects = 0

        for phase in config.get("phases", []):
            phase_num = phase.get("phase")
            if phase_num < start_phase:
                continue

            projects_to_push = []

            projects = phase.get("projects", [])[:]
            if phase.get("project"):
                projects.append(phase.get("project"))

            if project_filter:
                projects = [p for p in projects if p == project_filter]
                if not projects and phase.get("bulk_push"):
                    if project_filter not in processed_projects:
                        projects = [project_filter]

            if phase.get("bulk_push") and not project_filter:
                for url, path in self.project_map.items():
                    name = url.split("/")[-1].replace(".git", "")
                    if name not in processed_projects:
                        projects_to_push.append((name, path))
            else:
                for project_name in projects:
                    processed_projects.add(project_name)
                    for url, p_path in self.project_map.items():
                        if url.endswith(f"/{project_name}.git") or url.endswith(
                            f"/{project_name}"
                        ):
                            projects_to_push.append((project_name, p_path))
                            break

            if not projects_to_push:
                continue

            phase_name = phase.get("name", f"Phase {phase_num}")
            phase_list.append(
                {
                    "phase_num": phase_num,
                    "name": phase_name,
                    "projects_to_push": projects_to_push,
                    "wait_minutes": float(phase.get("wait_minutes", 0)),
                }
            )
            total_projects += len(projects_to_push)

        if progress is not None:
            progress["current_phase"] = "Initializing Pushes"
            progress["progress"] = 0
            progress["phases"] = {}
            for p_info in phase_list:
                progress["phases"][p_info["name"]] = {
                    "status": "pending",
                    "total": len(p_info["projects_to_push"]),
                    "processed": 0,
                    "completed": 0,
                    "success": 0,
                    "failed": 0,
                    "details": {
                        proj: "pending" for proj, _ in p_info["projects_to_push"]
                    },
                    "repos": {
                        proj: "pending" for proj, _ in p_info["projects_to_push"]
                    },
                }

        processed_count = 0

        for p_info in phase_list:
            phase_name = p_info["name"]
            phase_num = p_info["phase_num"]
            projects_to_push = p_info["projects_to_push"]
            wait_minutes = p_info["wait_minutes"]

            if progress is not None:
                progress["current_phase"] = f"{phase_name} in progress"
                progress["phases"][phase_name]["status"] = "running"

            logger.info(
                f"Starting {phase_name} push for {len(projects_to_push)} projects..."
            )

            phase_had_pushes = False

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads
            ) as executor:
                future_to_proj = {}
                for proj_name, p_path in projects_to_push:
                    if progress is not None:
                        progress["phases"][phase_name]["details"][proj_name] = "running"
                        progress["phases"][phase_name]["repos"][proj_name] = "running"
                    future = executor.submit(self.push_project, path=p_path)
                    future_to_proj[future] = proj_name

                for future in concurrent.futures.as_completed(future_to_proj):
                    proj_name = future_to_proj[future]
                    try:
                        res = future.result()
                        all_results.append(res)
                        status_str = "success" if res.status == "success" else "failed"
                        if (
                            res.status == "success"
                            and "Everything up-to-date" not in res.data
                        ):
                            phase_had_pushes = True
                    except Exception as e:
                        all_results.append(
                            GitResult(
                                status="error",
                                data="",
                                error=GitError(message=str(e), code=1),
                            )
                        )
                        status_str = "failed"

                    if progress is not None:
                        progress["phases"][phase_name]["details"][proj_name] = (
                            status_str
                        )
                        progress["phases"][phase_name]["repos"][proj_name] = status_str
                        progress["phases"][phase_name]["processed"] += 1
                        progress["phases"][phase_name]["completed"] += 1
                        if status_str == "success":
                            progress["phases"][phase_name]["success"] += 1
                        else:
                            progress["phases"][phase_name]["failed"] += 1

                        processed_count += 1
                        progress["progress"] = int(
                            (processed_count / total_projects) * 100
                        )
                        logger.info(
                            f"[{processed_count}/{total_projects}] ({progress['progress']}%) "
                            f"Completed push for {proj_name}: {status_str}"
                        )

            if progress is not None:
                progress["phases"][phase_name]["status"] = "completed"

            if wait_minutes > 0:
                if not phase_had_pushes:
                    logger.info(
                        f"Phase {phase_num} complete. Skipping {wait_minutes} minutes wait because 0 commits were pushed."
                    )
                else:
                    logger.info(
                        f"Phase {phase_num} complete. Waiting {wait_minutes} minutes before proceeding..."
                    )
                    if progress is not None:
                        progress["current_phase"] = (
                            f"Waiting {wait_minutes} min after {phase_name}"
                        )
                    time.sleep(wait_minutes * 60)

        if progress is not None:
            progress["current_phase"] = "Pushes Completed"
            progress["progress"] = 100

        return all_results

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
            with open(abs_yaml_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return False

            self.config = WorkspaceConfig(**data)

            yaml_config_path = os.path.expanduser(self.config.path)
            is_default_yaml = yaml_path == DEFAULT_WORKSPACE_YML

            if self._explicit_path:
                logger.info(f"Preserving explicit workspace path: {self.path}")
            elif os.path.isabs(yaml_config_path):
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

    def discover_projects(self) -> dict[str, str]:
        """
        Scan self.path for immediate subdirectories containing a .git folder.
        Populates and returns self.project_map.
        """
        self.project_map = {}
        expanded_path = os.path.abspath(os.path.expanduser(self.path))
        if not os.path.exists(expanded_path):
            return self.project_map

        try:
            for item in os.listdir(expanded_path):
                full_path = os.path.join(expanded_path, item)
                if os.path.isdir(full_path) and os.path.exists(
                    os.path.join(full_path, ".git")
                ):
                    # Get remote URL
                    remote_url = None
                    try:
                        import shutil

                        git_path = shutil.which("git") or "git"
                        res = subprocess.run(
                            [git_path, "config", "--get", "remote.origin.url"],
                            cwd=full_path,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if res.returncode == 0 and res.stdout.strip():
                            remote_url = res.stdout.strip()
                    except Exception as exc:
                        logger.debug(f"Failed to get remote URL for {item}: {exc}")

                    if not remote_url:
                        remote_url = f"local://{item}"

                    self.project_map[remote_url] = os.path.abspath(full_path)

            logger.info(
                f"Auto-discovered {len(self.project_map)} git repositories in {expanded_path}"
            )
        except Exception as e:
            logger.error(f"Failed to discover git projects in {expanded_path}: {e}")

        return self.project_map

    def _parse_subdirectories(
        self, subdirs: dict[str, SubdirectoryConfig], current_path: str
    ) -> dict[str, str]:
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
                except Exception:  # nosec B110
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
        self, yaml_path: str, config: WorkspaceConfig | None = None
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

    def get_consolidated_skill_paths(self) -> list[str]:
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
                    if skill_path.joinpath("SKILL.md").is_file():
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
                    if graph_path.joinpath("SKILL.md").is_file():
                        paths.append(str(graph_path))
            except Exception as e:
                logger.warning(f"Could not load skill graphs via importlib: {e}")

                all_graphs = get_skill_graphs_path(default_enabled=True)
                paths.extend(
                    [p for p in all_graphs if os.path.basename(p) in required_graphs]
                )

        return list(set(paths))


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
        help="Parallel thread count (default: 6).",
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
        "--save",
        action="store_true",
        help="Save current in-memory config back to YAML (Updates).",
    )

    group_workspace.add_argument(
        "--branches",
        action="store_true",
        help="List the active git branch for all projects.",
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
        "--add",
        action="store_true",
        help="Stage all changes in the specified repositories.",
    )
    group_git.add_argument(
        "--commit",
        action="store_true",
        help="Commit staged changes in the specified repositories.",
    )
    group_git.add_argument(
        "-m",
        "--message",
        type=str,
        help="Commit message for bulk commits. Required for --commit.",
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
        help="Run 'pip install --break-system-packages -e .' for all projects.",
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
        "--no-report",
        action="store_true",
        help="Do not automatically generate the validation report directory.",
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
            "test",
            "pre-commit",
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
        "--push",
        action="store_true",
        help="Execute phased push.",
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
        help="Starting phase for maintenance lifecycle (1-3).",
    )
    group_maintenance.add_argument(
        "--single-phase",
        action="store_true",
        help="Only execute the specified phase, do not proceed to subsequent phases.",
    )
    group_maintenance.add_argument(
        "--project",
        type=str,
        help="Only execute maintenance operations for a specific project.",
    )
    group_maintenance.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform maintenance operations without committing changes. Use with --maintain.",
    )
    group_maintenance.add_argument(
        "--allow-pre-commit",
        action="store_true",
        help="Execute the pre-commit phase during maintenance workflow. Pre-commits are skipped by default.",
    )
    group_maintenance.add_argument(
        "--config",
        type=str,
        help="Path to an overridden maintenance JSON/YAML configuration. Use with --maintain.",
    )
    group_maintenance.add_argument(
        "--break-system-packages",
        action="store_true",
        help="Include --break-system-packages in pip install commands.",
    )

    args = parser.parse_args()

    git = Git(
        path=args.workspace
        if args.workspace != DEFAULT_REPOSITORY_MANAGER_WORKSPACE
        else None,
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
        if args.setup:
            logger.info(f"Setting up workspace from {args.file}...")
            git.load_projects_from_yaml(args.file)

    if clone_flag:
        git.clone_projects()
    if pull_flag:
        git.pull_projects()

    if args.add:
        results = git.add_projects()
        summary = git.generate_markdown_summary("Bulk Git Add", results)
        logger.info(summary)
        git._export_report(summary, "git_add_report.md")

    if args.commit:
        if not args.message:
            logger.error(
                "Error: --message/-m is required for bulk commits when using --commit."
            )
            sys.exit(1)
        results = git.commit_projects(message=args.message)
        summary = git.generate_markdown_summary("Bulk Git Commit", results)
        logger.info(summary)
        git._export_report(summary, "git_commit_report.md")

    if args.branches:
        branches = git.list_branches()
        logger.info("\n--- Workspace Branches ---")
        for proj, branch in sorted(branches.items()):
            logger.info(f"{proj:<30} | {branch}")

    if args.pre_commit:
        git.pre_commit_projects(run=True, autoupdate=True)

    if args.install:
        results = git.install_projects()
        summary = git.generate_markdown_summary("Installation", results)
        logger.info(summary)
        git._export_report(summary, "install_report.md")

    if args.build:
        results = git.build_projects()
        summary = git.generate_markdown_summary("Build", results)
        logger.info(summary)
        git._export_report(summary, "build_report.md")

    has_errors = False

    if args.validate:
        val_results = git.validate_and_release(
            threads=args.threads,
            auto_bump=bool(args.bump) if not args.maintain else False,
            auto_push=args.push,
            bump_part=args.bump if args.bump else "minor",
        )
        if not val_results.get("passed"):
            has_errors = True
            logger.error("Validation failed with errors.")
        else:
            logger.info("Validation and subsequent operations completed successfully.")

        # Prevent these from executing again below
        args.bump = None
        args.push = False

    if args.bump and not args.maintain:
        if has_errors and (args.push or args.bump):
            logger.error("Skipping bump due to preceding validation errors.")
            has_errors = True
        else:
            logger.info(f"Bumping version ({args.bump}) for all projects...")
            project_dirs = list(git.project_map.values())
            results = []
            for d in project_dirs:
                res = git.bump_version(
                    args.bump, allow_dirty=True, path=d, dry_run=args.dry_run
                )
                results.append(res)
                if res.status == "error":
                    has_errors = True

            summary = git.generate_markdown_summary("Bulk Version Bump", results)
            logger.info(summary)
            git._export_report(summary, "version_bump_report.md")

    if args.maintain:
        if has_errors and (args.push or args.maintain):
            logger.error(
                "Skipping maintenance bump due to preceding validation errors."
            )
            has_errors = True
        else:
            config = None
            if args.config:
                try:
                    with open(args.config) as f:
                        config = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load config from {args.config}: {e}")
                    sys.exit(1)

            results = git.phased_bumpversion(
                part=args.bump if args.bump else "patch",
                start_phase=args.phase,
                dry_run=args.dry_run,
                allow_pre_commit=args.allow_pre_commit,
                config=config,
                single_phase=args.single_phase,
                project_filter=args.project,
            )

            for res in results:
                if res.status == "error":
                    has_errors = True

            summary = git.generate_markdown_summary("Phased Maintenance Bump", results)

            logger.info(summary)
            git._export_report(summary, "maintenance_report.md")

    if args.push:
        if has_errors:
            logger.error("Skipping push due to preceding validation or bump errors.")
        else:
            config = None
            if args.config:
                try:
                    with open(args.config) as f:
                        config = json.load(f)
                except Exception as e:
                    logger.error(f"Failed to load config from {args.config}: {e}")
                    sys.exit(1)
            push_results = git.phased_push(
                start_phase=args.phase,
                config=config,
                single_phase=args.single_phase,
                project_filter=args.project,
            )
            summary = git.generate_markdown_summary("Phased Push", push_results)
            logger.info(summary)
            git._export_report(summary, "push_report.md")


if __name__ == "__main__":
    """
    Execute the main function when the script is run directly.
    """
    main()
