"""Strict manual hook adapter for the authoritative fleet readiness checker.

This is a configured-index fleet verdict, not exact-wheel publication proof or
signed-manifest admission. Pipelines owns those independent release contracts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from repository_manager import dependency_readiness as readiness


def main(argv: list[str] | None = None) -> int:
    """Reject absent fleet scope, metadata errors and overridden verdicts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=".")
    args = parser.parse_args(argv)
    repo = Path(args.path).resolve()
    scope = readiness._resolve_fleet_packages(repo, None, None)
    if not isinstance(scope, set) or not scope:
        print("CANNOT RUN: authoritative fleet scope is missing or invalid")
        return 2
    report = readiness.check_tree(repo)
    readiness._print_human_report(report)
    return 0 if report.ok and not report.overridden else 1


if __name__ == "__main__":
    raise SystemExit(main())
