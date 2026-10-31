"""Workspace-management MCP adapter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastmcp import Context, FastMCP
from pydantic import Field
from scion_sdk.mcp.action_dispatch import resolve_action
from scion_sdk.mcp.concurrency import run_blocking

from repository_manager.development.fleet_release_gate import (
    FleetReleaseGateError,
    authorize_fleet_evidence_reference,
    parse_fleet_evidence_reference,
    require_legacy_release_route,
)
from repository_manager.mcp_tools.context import McpToolContext, from_server
from repository_manager.mcp_tools.contracts import RM_WORKSPACE_ACTIONS
from repository_manager.models import GitError, GitResult, WorkspaceConfig


@dataclass(frozen=True)
class _WorkspaceActionRequest:
    """Bundled, already-normalized ``rm_workspace`` kwargs.

    Module-level (not nested in ``register_workspace_management_tools``) and
    defined well away from any ``@mcp.tool`` decorator, per the extraction
    rule for this file: a new helper placed ABOVE a decorated function
    silently rebinds the decorator to the helper instead, unregistering the
    tool while the file still parses and lints clean. Both this dataclass and
    :func:`_run_workspace_action` live entirely outside
    ``register_workspace_management_tools``, so there is no decorator
    anywhere near them to rebind.
    """

    yml_path: str | None
    install: bool
    config_dict: dict[str, Any] | None
    use_default: bool
    fleet_evidence_json: str | None = None


async def _setup_workspace_action(
    git: Any, request: _WorkspaceActionRequest
) -> GitResult | Any:
    require_legacy_release_route("setup")
    if not request.yml_path:
        return GitResult(
            status="error",
            data="",
            error=GitError(message="yml_path required for 'setup'", code=1),
        )
    return await run_blocking(git.setup_from_yaml, request.yml_path, request.install)


async def _template_workspace_action(
    git: Any, request: _WorkspaceActionRequest
) -> GitResult | Any:
    if not request.yml_path:
        return GitResult(
            status="error",
            data="",
            error=GitError(message="yml_path required for 'template'", code=1),
        )
    return await run_blocking(
        git.generate_workspace_template,
        target_path=request.yml_path,
        use_default=request.use_default,
    )


async def _save_workspace_action(
    git: Any, request: _WorkspaceActionRequest
) -> GitResult | Any:
    if not request.yml_path or not request.config_dict:
        return GitResult(
            status="error",
            data="",
            error=GitError(
                message="yml_path and config_dict required for 'save'", code=1
            ),
        )
    try:
        config = WorkspaceConfig(**request.config_dict)
        return await run_blocking(
            git.save_workspace_config, yaml_path=request.yml_path, config=config
        )
    except Exception:
        return GitResult(
            status="error",
            data="",
            error=GitError(message="Repository operation failed", code=1),
        )


async def _run_workspace_action(
    action: str,
    git: Any,
    adapter_context: McpToolContext,
    request: _WorkspaceActionRequest,
) -> list[str] | str | GitResult | dict[str, Any]:
    """The ``rm_workspace`` action body, verbatim, moved out of the decorated
    tool function so only the thin signature/normalization wrapper stays
    decorated.
    """
    if action == "fleet_evidence_check":
        return _check_workspace_fleet_evidence(adapter_context, request)
    return await _run_non_fleet_workspace_action(action, git, adapter_context, request)


def _check_workspace_fleet_evidence(
    adapter_context: McpToolContext, request: _WorkspaceActionRequest
) -> GitResult | dict[str, Any]:
    try:
        if request.fleet_evidence_json is None:
            raise FleetReleaseGateError("fleet evidence JSON is required")
        reference = parse_fleet_evidence_reference(
            request.fleet_evidence_json.encode("utf-8")
        )
        provider = adapter_context.fleet_preflight_provider
        if provider is None:
            raise FleetReleaseGateError("fleet preflight provider is unavailable")
        permit = authorize_fleet_evidence_reference(provider(reference), reference)
        return {
            "status": "ok",
            "stage_id": permit.stage_id,
            "fleet_binding_digest": permit.fleet_binding_digest,
        }
    except FleetReleaseGateError:
        return GitResult(
            status="error",
            data="",
            error=GitError(message="Fleet evidence preflight refused", code=1),
        )


async def _run_non_fleet_workspace_action(
    action: str,
    git: Any,
    adapter_context: McpToolContext,
    request: _WorkspaceActionRequest,
) -> list[str] | str | GitResult | dict[str, Any]:
    if action == "list":
        return git.get_workspace_projects()
    if action == "list_branches":
        return await run_blocking(git.list_branches)
    if action == "setup":
        return await _setup_workspace_action(git, request)
    if action == "template":
        return await _template_workspace_action(git, request)
    if action == "save":
        return await _save_workspace_action(git, request)
    return f"Error: Unknown action '{action}'"


def register_workspace_management_tools(
    mcp: FastMCP, *, context: McpToolContext | None = None
) -> None:
    """Register tools for core workspace setup and organization."""

    adapter_context = context or from_server()

    @mcp.tool(tags={"workspace_management"})
    async def rm_workspace(
        action: str = Field(
            description=(
                "Action: 'list', 'list_branches', 'setup', 'template', 'save', "
                "'fleet_evidence_check'"
            )
        ),
        yml_path: str | None = Field(
            default=None,
            description="Path to workspace.yml (for 'setup', 'template', 'save').",
        ),
        install: bool = Field(
            default=False,
            description=(
                "For 'setup': after clone/pull, materialize each project's "
                ".uv-workspace-siblings/scion-sdk symlink and run `uv "
                "sync`, dependency-ordered (scion-sdk first, since "
                "every fleet member depends on it and it is not yet "
                "published to PyPI at the required floor). Best-effort per "
                "project; a partial failure is reported (status='error' "
                "with per-project detail in 'data'), never masked as a "
                "uniform success. See CONCEPT:RM-BOOTSTRAP."
            ),
        ),
        config_dict: dict[str, Any] | None = Field(
            default=None,
            description="Dictionary representation of WorkspaceConfig (for 'save').",
        ),
        use_default: bool = Field(
            default=True,
            description="Use the pre-filled package template for 'template'.",
        ),
        fleet_evidence_json: str | None = Field(
            default=None,
            description="Bounded immutable reference JSON for fleet_evidence_check.",
        ),
        ctx: Context | None = Field(
            description="MCP context for progress reporting", default=None
        ),
    ) -> list[str] | str | GitResult | dict[str, Any]:
        """Core workspace organization and configuration."""
        resolved = resolve_action(
            action, RM_WORKSPACE_ACTIONS, service="repository-manager"
        )
        if isinstance(resolved, dict):
            return resolved
        action = resolved
        git = _workspace_git_for_action(adapter_context, action)

        return await _run_workspace_action(
            action,
            git,
            adapter_context,
            _WorkspaceActionRequest(
                yml_path=yml_path,
                install=install,
                config_dict=config_dict,
                use_default=use_default,
                fleet_evidence_json=fleet_evidence_json,
            ),
        )


def _workspace_git_for_action(adapter_context: McpToolContext, action: str) -> Any:
    """The fleet evidence read has no Git workspace side effects."""
    return (
        None if action == "fleet_evidence_check" else adapter_context.get_git_instance()
    )
