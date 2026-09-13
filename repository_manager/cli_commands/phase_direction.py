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


__all__ = ["run_phase_direction_cli"]
