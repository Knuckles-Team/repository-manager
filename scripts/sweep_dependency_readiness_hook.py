#!/usr/bin/env python3
"""Sweep the ``dependency-readiness`` pre-push hook across the fleet.

CONCEPT:RM-DEP-READY (Layer 1 fleet rollout)

Mirrors the same mechanical, indentation-safe injection technique the fleet's
two-tier fast/pre-commit vs. heavy/pre-push model itself was swept with
(``inject_precommit_hooks.py`` in agent-utilities/scripts): find the first
``repo: local`` / ``hooks:`` block, measure the indentation an existing
``- id:`` entry there already uses, and insert the new hook at that same
depth — never a hardcoded 2/4/6-space guess, which is exactly what produced
an unparseable ``.pre-commit-config.yaml`` before that fix (INFRA-5). Falls
back to appending a brand-new ``repo: local`` block when a repo's config has
no local block yet.

Idempotent: a repo whose config already has the ``dependency-readiness`` hook
id is reported as ``already-present`` and left untouched — safe to re-run.

The injected entry runs the check via ``uv run --with repository-manager``
rather than a bare ``python -m repository_manager.dependency_readiness`` —
most fleet repos do not (and should not) declare ``repository-manager``
itself as a project dependency just to run this one gate, so the hook
resolves it ephemerally through uv instead of widening every repo's own
dependency surface. (repository-manager's OWN ``.pre-commit-config.yaml`` is
the one exception: it already has itself installed locally, so its hook
invokes the module directly — see that file.) This does mean the swept hook
only becomes LIVE in a dependent repo once this feature itself is released to
PyPI — the same publish-before-consume shape this whole gate exists to
detect elsewhere in the fleet.

Usage::

    python scripts/sweep_dependency_readiness_hook.py --root <agent-packages> --dry-run
    python scripts/sweep_dependency_readiness_hook.py --root <agent-packages> --apply

``--dry-run`` (the default) only reports what WOULD change — never writes.
``--apply`` is required to actually mutate files; each write is one repo's
``.pre-commit-config.yaml``, so it is safe to interrupt and re-run at any
point (idempotent, per-file atomic).
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HOOK_ID = "dependency-readiness"

#: Lines of the hook block, indentation-relative (first line at the block's
#: own `- id:` depth, the rest one level deeper) — mirrors
#: `inject_precommit_hooks.py`'s `HOOK_LINES` shape exactly.
HOOK_LINES = [
    "- id: dependency-readiness",
    "name: Dependency readiness — every declared intra-fleet constraint is installable",
    "entry: bash -c 'uv run --with repository-manager python -m repository_manager.dependency_readiness .'",
    "language: system",
    "pass_filenames: false",
    "always_run: true",
    "stages: [manual, pre-push]",
]

NEW_REPO_BLOCK = (
    "- repo: local\n  hooks:\n"
    + "\n".join(
        f"    {HOOK_LINES[0]}" if i == 0 else f"      {line}"
        for i, line in enumerate(HOOK_LINES)
    )
    + "\n"
)

_EXISTING_HOOK_RE = re.compile(r"^([ \t]*)-\s*id:\s*\S", re.MULTILINE)
_DEFAULT_HOOK_INDENT = "      "  # 6 spaces -- this fleet's established depth


def _hook_indent(content: str, hooks_idx: int) -> str:
    match = _EXISTING_HOOK_RE.search(content, hooks_idx)
    if match:
        return match.group(1)
    return _DEFAULT_HOOK_INDENT


def _render_hook_block(indent: str) -> str:
    inner = indent + "  "
    lines = [f"{indent}{HOOK_LINES[0]}"]
    lines.extend(f"{inner}{line}" for line in HOOK_LINES[1:])
    return "\n".join(lines) + "\n"


@dataclass
class SweepResult:
    path: str
    action: str  # "already-present" | "would-inject-into-local-block" |
    #  "would-append-new-block" | "injected-into-local-block" |
    #  "appended-new-block" | "no-config"


def plan_or_apply(filepath: Path, *, apply: bool) -> SweepResult:
    if not filepath.exists():
        return SweepResult(str(filepath), "no-config")

    content = filepath.read_text(encoding="utf-8", errors="ignore")
    if f"id: {HOOK_ID}" in content:
        return SweepResult(str(filepath), "already-present")

    repo_local_idx = content.find("repo: local")
    if repo_local_idx != -1:
        hooks_idx = content.find("hooks:", repo_local_idx)
        if hooks_idx != -1:
            newline_idx = content.find("\n", hooks_idx)
            if newline_idx != -1:
                indent = _hook_indent(content, newline_idx + 1)
                block = _render_hook_block(indent)
                if apply:
                    new_content = (
                        content[: newline_idx + 1] + block + content[newline_idx + 1 :]
                    )
                    filepath.write_text(new_content, encoding="utf-8")
                    return SweepResult(str(filepath), "injected-into-local-block")
                return SweepResult(str(filepath), "would-inject-into-local-block")

    if apply:
        prefix = "\n" if not content.endswith("\n") else ""
        filepath.write_text(content + prefix + NEW_REPO_BLOCK, encoding="utf-8")
        return SweepResult(str(filepath), "appended-new-block")
    return SweepResult(str(filepath), "would-append-new-block")


def sweep(root: Path, *, apply: bool) -> list[SweepResult]:
    results: list[SweepResult] = []
    for filepath in sorted(root.rglob(".pre-commit-config.yaml")):
        if ".git" in filepath.parts:
            continue
        results.append(plan_or_apply(filepath, apply=apply))
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        required=True,
        type=Path,
        help="fleet root to scan for .pre-commit-config.yaml",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="report only, write nothing (default)",
    )
    mode.add_argument("--apply", action="store_true", help="actually write the changes")
    args = parser.parse_args(argv)

    apply = args.apply
    results = sweep(args.root, apply=apply)

    counts: dict[str, int] = {}
    for r in results:
        counts[r.action] = counts.get(r.action, 0) + 1
        print(f"  [{r.action}] {r.path}")

    print()
    print(f"{'APPLIED' if apply else 'DRY-RUN'} — {len(results)} repo(s) scanned:")
    for action, n in sorted(counts.items()):
        print(f"  {action}: {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
