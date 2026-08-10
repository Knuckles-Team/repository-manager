"""Remote-worker CLI adapter (RMDD-20, exposing RMDD-15)."""

from __future__ import annotations

import argparse
import json
from typing import Any


def run_remote_workers_cli(args: argparse.Namespace) -> int:
    """Marshal ``--remote-workers`` flags onto the shared remote-worker core.

    Calls the exact same :func:`repository_manager.remote_worker_actions.dispatch`
    the MCP ``rm_remote_workers`` tool calls -- this is the parity chokepoint.
    """

    from repository_manager import remote_worker_actions

    params: dict[str, Any] = {}
    if args.remote_workers_params_json:
        try:
            params = json.loads(args.remote_workers_params_json)
        except json.JSONDecodeError as exc:
            print(
                json.dumps(
                    {
                        "ok": False,
                        "refused": f"invalid --remote-workers-params-json: {exc}",
                    }
                )
            )
            return 1
        if not isinstance(params, dict):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "refused": "--remote-workers-params-json must decode to an object",
                    }
                )
            )
            return 1

    result = remote_worker_actions.dispatch(args.remote_workers, **params)
    print(json.dumps(result, default=str, indent=2))
    return 0 if result.get("ok", False) else 1
