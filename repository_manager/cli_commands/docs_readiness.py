"""CLI adapter for the shared documentation-readiness action core."""

from __future__ import annotations

import argparse
import json

from repository_manager import docs_readiness


def run_docs_readiness_cli(args: argparse.Namespace) -> int:
    """Marshal CLI values to :func:`repository_manager.docs_readiness.dispatch`."""

    result = docs_readiness.dispatch(
        args.docs_readiness,
        workspace_root=args.workspace,
        manifest_path=args.file,
        repository=args.docs_readiness_repository,
        confirm=args.docs_readiness_confirm,
    )
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0 if result.get("ok") is True else 1


__all__ = ["run_docs_readiness_cli"]
