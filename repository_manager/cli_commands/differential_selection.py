"""CLI adapter for the differential pre-push test selector (CONCEPT:RM-DIFF-SELECT).

Deliberately thin — a standalone way to run/inspect a selection while
``rm_gates(action=run, stage=heavy)`` (a sibling lane) is the real, wired
consumer. See :mod:`repository_manager.differential_selection` for the rules.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def run_differential_select_cli(args: Any) -> int:
    from repository_manager.differential_selection import select_differential_tests

    repo = Path(args.repo_path) if args.repo_path else Path.cwd()
    src_roots = tuple(
        s.strip() for s in args.diff_src_roots.split(",") if s.strip()
    ) or (".",)
    test_roots = tuple(
        s.strip() for s in args.diff_test_roots.split(",") if s.strip()
    ) or ("tests",)
    result = select_differential_tests(
        repo,
        base_ref=args.diff_base or "main",
        ref=args.diff_ref or "HEAD",
        src_roots=src_roots,
        test_roots=test_roots,
        fanin_fallback_threshold=args.diff_fanin_threshold,
    )
    print(json.dumps(result.as_dict(), indent=2))
    return 0
