"""Dependency context used by Repository Manager's MCP adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from threading import RLock
from typing import Any


@dataclass(frozen=True)
class McpToolContext:
    """Runtime dependencies shared by the thin MCP registration modules.

    The context is intentionally assembled by the server module at registration
    time.  Keeping these references late-bound preserves the existing test and
    deployment seams (for example, replacing ``get_git_instance`` or the job
    executor) while ensuring adapters contain no queue, Git, or policy state.
    """

    get_git_instance: Callable[..., Any]
    resolve_repo_dir: Callable[..., str]
    resolve_commit_code_target: Callable[..., tuple[str | None, str | None]]
    submit_job: Callable[..., dict[str, str]]
    cancel_job: Callable[[str], dict[str, Any]]
    get_job_status: Callable[..., dict[str, Any]]
    last_failed_repos: Callable[[], list[str]]
    wait_for_jobs_and_run: Callable[..., Any]
    jobs: dict[str, dict[str, Any]]
    jobs_lock: RLock
    default_workspace: str
    default_workspace_yml: str


def from_server() -> McpToolContext:
    """Build a context from the currently loaded MCP server module.

    Importing lazily is important: the server imports this package while it is
    loading, and the context must see monkey-patched helpers in tests as well as
    the live server's current settings.
    """

    server = import_module("repository_manager.mcp_server")

    def resolve(name: str) -> Callable[..., Any]:
        def call(*args: Any, **kwargs: Any) -> Any:
            return getattr(server, name)(*args, **kwargs)

        return call

    return McpToolContext(
        get_git_instance=resolve("get_git_instance"),
        resolve_repo_dir=resolve("_resolve_repo_dir"),
        resolve_commit_code_target=resolve("_resolve_commit_code_target"),
        submit_job=resolve("_submit_job"),
        cancel_job=resolve("_cancel_job"),
        get_job_status=resolve("_get_job_status"),
        last_failed_repos=resolve("_last_failed_repos"),
        wait_for_jobs_and_run=resolve("_wait_for_jobs_and_run"),
        jobs=server._jobs,
        jobs_lock=server._jobs_lock,
        default_workspace=server.DEFAULT_WORKSPACE,
        default_workspace_yml=server.DEFAULT_WORKSPACE_YML,
    )
