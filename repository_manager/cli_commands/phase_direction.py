"""CLI adapter for the maintenance-phase dependency-direction check."""

from __future__ import annotations

import argparse
import json

from repository_manager import dependency_readiness


def run_phase_direction_cli(args: argparse.Namespace) -> int:
    """Marshal CLI values to :func:`repository_manager.dependency_readiness.dispatch`.

    ``-f/--file`` is the manifest whose ``maintenance.phases`` define the order
    and ``-w/--workspace`` is where its repositories are checked out. Exit 1
    on any blocking later-phase edge, unknown-phase repository, or unreadable
    metadata.
    """

    result = dependency_readiness.dispatch(
        "phase_direction",
        manifest_path=args.file,
        workspace_root=args.workspace,
        repositories=args.phase_direction_repository,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("ok") is True else 1


def run_phase_direction_here_cli(args: argparse.Namespace) -> int:
    """Marshal CLI values to :func:`dependency_readiness.dispatch`'s
    ``phase_direction_here`` action.

    Infers which manifest repository the current checkout (or the linked git
    worktree it is running from) is, and checks only that repository's own
    on-disk edges -- the mode a repository's own pre-push hook uses, since it
    has no other way to name itself. ``-f/--file`` still selects the manifest
    (default: the packaged canonical copy); ``-w/--workspace`` is
    deliberately NOT used here (see
    :func:`repository_manager.dependency_readiness.check_phase_direction_here`'s
    docstring) -- only ``--phase-direction-start`` overrides where the
    repository is resolved from (default: the current working directory).
    Exit 1 on any blocking edge, an unresolvable repository, or unreadable
    metadata.
    """

    result = dependency_readiness.dispatch(
        "phase_direction_here",
        manifest_path=args.file,
        start=args.phase_direction_start,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("ok") is True else 1


__all__ = ["run_phase_direction_cli", "run_phase_direction_here_cli"]
