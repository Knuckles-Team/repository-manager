"""Merge-queue CLI adapter."""

from __future__ import annotations

import json
from typing import Any


def run_merge_queue_cli(args: Any) -> int:
    """Marshal merge-queue flags onto the shared dispatch core."""
    from agent_utilities.governance.lanes import LaneArbitrationError, LeaseUnavailable

    from repository_manager import merge_queue

    try:
        result = merge_queue.dispatch(
            args.merge_queue,
            path=args.repo_path,
            branch=args.queue_branch,
            base=args.queue_base,
            reason=args.queue_reason,
            batch_size=args.queue_batch_size,
            prune=not args.queue_no_prune,
        )
    except LeaseUnavailable as exc:
        print(json.dumps({"deferred": True, "holder": exc.holder}, default=str))
        return 75
    except LaneArbitrationError as exc:
        print(json.dumps({"refused": str(exc)}))
        return 1
    print(json.dumps(result, default=str, indent=2))
    if result.get("ok") is False:
        return 1
    return 1 if result.get("rejected") else 0
