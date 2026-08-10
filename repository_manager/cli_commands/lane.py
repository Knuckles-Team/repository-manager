"""Lane lifecycle CLI adapter."""

from __future__ import annotations

import argparse
import json


def run_lane_cli(args: argparse.Namespace) -> int:
    """Marshal ``--lane`` onto the shared lane-doctor action core."""
    from repository_manager import lane_doctor

    result = lane_doctor.dispatch(
        args.lane,
        path=args.lane_path,
        repo=args.lane_repo,
        branch=args.lane_branch,
        base=args.lane_base,
        workspace=args.workspace,
        force=args.lane_force,
    )
    if args.lane_shell and "shell" in result:
        print(result["shell"])
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", False) else 1
