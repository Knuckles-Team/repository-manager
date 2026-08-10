"""Concept-claim coordination CLI adapter (RMDD-20, exposing RMDD-17)."""

from __future__ import annotations

import argparse
import json
from typing import Any


def run_concepts_cli(args: argparse.Namespace) -> int:
    """Marshal ``--concepts`` flags onto the shared concept-action core.

    Calls the exact same :func:`repository_manager.concept_actions.dispatch`
    the MCP ``rm_concepts`` tool calls -- this is the parity chokepoint.
    """

    from repository_manager import concept_actions

    params: dict[str, Any] = {}
    if args.concepts_params_json:
        try:
            params = json.loads(args.concepts_params_json)
        except json.JSONDecodeError as exc:
            print(
                json.dumps(
                    {"ok": False, "refused": f"invalid --concepts-params-json: {exc}"}
                )
            )
            return 1
        if not isinstance(params, dict):
            print(
                json.dumps(
                    {
                        "ok": False,
                        "refused": "--concepts-params-json must decode to an object",
                    }
                )
            )
            return 1

    result = concept_actions.dispatch(
        args.concepts,
        repo_root=args.concepts_repo_root,
        tenant_ref=args.concepts_tenant_ref,
        lane_ref=args.concepts_lane_ref,
        **params,
    )
    print(json.dumps(result, default=str, indent=2))
    return 0 if result.get("ok", False) else 1
