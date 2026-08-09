"""Runtime dependencies for CLI adapters."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any


@dataclass(frozen=True)
class CliRuntime:
    """References to application cores used by CLI marshalling code."""

    git_factory: Callable[..., Any]
    version: str
    default_workspace: str
    default_workspace_yml: str
    default_threads: int
    logger: Any
    synchronize_workspace_manifest: Callable[..., Any]
    manifest_error: type[Exception]


def runtime_from_module() -> CliRuntime:
    """Resolve runtime references from the canonical entrypoint module."""

    module = import_module("repository_manager.repository_manager")
    return CliRuntime(
        git_factory=module.Git,
        version=module.__version__,
        default_workspace=module.DEFAULT_REPOSITORY_MANAGER_WORKSPACE,
        default_workspace_yml=module.DEFAULT_WORKSPACE_YML,
        default_threads=module.DEFAULT_REPOSITORY_MANAGER_THREADS,
        logger=module.logger,
        synchronize_workspace_manifest=module.synchronize_workspace_manifest,
        manifest_error=module.WorkspaceManifestError,
    )
