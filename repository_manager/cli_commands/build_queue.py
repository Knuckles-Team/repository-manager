"""Build-broker CLI adapter."""

from __future__ import annotations

import json
from typing import Any


def run_build_queue_cli(args: Any) -> int:
    """Marshal build-broker flags onto the shared dispatch core."""
    from agent_utilities.governance.lanes import LaneArbitrationError

    from repository_manager import build_queue

    try:
        result = build_queue.dispatch(
            args.build_broker,
            path=args.repo_path,
            spec=args.build_spec,
            key=args.build_key,
            colocated=args.same_node,
            wait_timeout=args.build_wait_timeout,
            keep_recent=args.build_keep_recent,
            max_age_days=args.build_max_age_days,
            host=args.build_host,
        )
    except LaneArbitrationError as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    print(json.dumps(result, default=str, indent=2))
    return 1 if result.get("ok") is False else 0
