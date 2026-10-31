"""``repository-manager rollout`` CLI adapter (RM-RELEASE-R009)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from repository_manager.rollout import DEFAULT_GROUP, DEFAULT_JOBS, run_rollout_action


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="repository-manager rollout",
        description=(
            "Roll a dependency floor across a fleet through auto-merging PRs.\n"
            "  rollout --dependency D --floor X [--dry-run]   plan + apply\n"
            "  rollout plan|status|release --dependency D --floor X"
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    p.add_argument(
        "action",
        nargs="?",
        default="apply",
        choices=["plan", "apply", "status", "release"],
    )
    p.add_argument("--dependency", required=True)
    p.add_argument(
        "--floor", required=True, help="MAJOR.MINOR.PATCH; written as >=X,<next-minor"
    )
    p.add_argument("--extra", default=None, help="ensure this extra on the requirement")
    p.add_argument(
        "--group",
        default=DEFAULT_GROUP,
        help="workspace.yml selector or subdirectory prefix",
    )
    p.add_argument("--only", default="", help="comma-separated repository names")
    p.add_argument("--jobs", type=int, default=DEFAULT_JOBS)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--manifest", "-f", default=None, help="workspace.yml path")
    p.add_argument(
        "--workspace", "-w", default=None, help="workspace root (local checkouts)"
    )
    p.add_argument(
        "--spec", default="none (dependency rollout)", help="Spec: trailer value"
    )
    p.add_argument(
        "--trailer", action="append", default=[], help="extra commit trailer line"
    )
    p.add_argument("--json", action="store_true", help="print full JSON rows")
    return p


def run_rollout_cli(argv: Sequence[str]) -> int:
    a = build_parser().parse_args(list(argv))
    options = {
        "extra": a.extra,
        "group": a.group,
        "only": tuple(x.strip() for x in a.only.split(",") if x.strip()),
        "jobs": a.jobs,
        "dry_run": a.dry_run,
        "spec": a.spec,
        "trailers": tuple(a.trailer),
    }
    if a.workspace:
        options["workspace_root"] = Path(a.workspace)
    try:
        result = run_rollout_action(
            a.action,
            dependency=a.dependency,
            floor=a.floor,
            manifest_path=Path(a.manifest) if a.manifest else None,
            **options,
        )
    except (ValueError, RuntimeError) as exc:
        print(f"rollout: {exc}", file=sys.stderr)
        return 2
    if a.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print(
            f"{result['action']} {result['dependency']}{result['range']} "
            f"branch={result['branch']} dry_run={result['dry_run']} ledger={result['ledger']}"
        )
        for row in result["rows"]:
            state = row.get("status") or row.get("state")
            extra = (
                row.get("pr")
                or row.get("reason")
                or row.get("error")
                or row.get("tag")
                or ""
            )
            checks = ",".join(row.get("failing_checks", []))
            print(
                f"  {row['repo']:<32} {state:<10} floor={row.get('current_floor')} "
                f"v={row.get('version')} {extra} {checks}".rstrip()
            )
        print("summary:", json.dumps(result["counts"]))
    bad = {"failed", "failing", "tag-failed"}
    return (
        1
        if any((r.get("status") or r.get("state")) in bad for r in result["rows"])
        else 0
    )
