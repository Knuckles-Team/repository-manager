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
    python scripts/sweep_dependency_readiness_hook.py --root <agent-packages> --resync --apply

``--dry-run`` (the default) only reports what WOULD change — never writes.
``--apply`` is required to actually mutate files; each write is one repo's
``.pre-commit-config.yaml``, so it is safe to interrupt and re-run at any
point (idempotent, per-file atomic).

``--resync`` (EH-173) rewrites an ALREADY-PRESENT hook's ``entry:`` line to
the current canonical resolver when it has drifted — e.g. every copy hand-
patched with a ``$PWD``-anchored upward walk for the sibling
``repository-manager`` checkout, which cannot resolve it from a linked git
worktree (a worktree routinely lives outside the workspace tree entirely,
e.g. ``/var/tmp/repository-worktrees/<repo>/<branch>``) and silently falls
back to installing an unpublished package from PyPI. The canonical resolver
anchors on ``git rev-parse --git-common-dir`` instead — the repo's shared
git directory, identical from every one of its linked worktrees, and stable
under a real git hook's ambient ``GIT_DIR``/``GIT_INDEX_FILE`` too — so the
sibling checkout resolves the same way from any worktree of any repo, at
either depth the fleet nests repos at (``agent-packages/<repo>`` or
``agent-packages/agents/<repo>``). Handles both an inline ``entry: bash -c
'...'`` and a YAML folded block scalar (``entry: >-``) without corrupting
the surrounding YAML, is idempotent (a no-op once already canonical), and
never touches repository-manager's own self-hosting exception entry.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

HOOK_ID = "dependency-readiness"

#: The one resolver every adopting repo's hook entry embeds to find a local
#: `repository-manager` checkout (EH-173). Anchored at
#: ``git rev-parse --git-common-dir`` — the repo's SHARED git directory,
#: identical across every linked worktree of the SAME repo — never at
#: ``$PWD``. A linked worktree routinely lives outside the workspace tree
#: (e.g. ``/var/tmp/repository-worktrees/<repo>/<branch>``), so a ``$PWD``
#: upward walk for ``agent-packages/agents/repository-manager`` never finds
#: it there and silently falls back to installing an unpublished package
#: from PyPI (the exact failure this hook exists to catch, self-inflicted).
#: ``--git-common-dir`` resolves correctly under a real git hook's ambient
#: ``GIT_DIR``/``GIT_INDEX_FILE`` too (that ambient state points at THIS
#: repo, which is what we want here — unlike the ``git -C <subdir>``
#: pattern documented at "git-hook-ambient-env-breaks-ls-files", this
#: resolver never combines ``-C`` with a path outside the env's own repo).
#: The walk itself (never a hardcoded `../repository-manager` or
#: `../agents/repository-manager` sibling offset) is still needed because
#: the canonical root sits at two different depths across the fleet:
#: `agent-packages/<repo>` (graph-os, geniusbot, agent-terminal-ui) vs.
#: `agent-packages/agents/<repo>` (every MCP/API-client package) — walking
#: up from the CANONICAL root (not `$PWD`) until `agent-packages/agents/
#: repository-manager` is found under it handles both without guessing
#: which depth a given repo lives at.
_RESOLVE_REPOSITORY_MANAGER = (
    'CANON_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)"; '
    'P=""; d="$CANON_ROOT"; '
    'while [ "$d" != "/" ]; do '
    '[ -d "$d/agent-packages/agents/repository-manager" ] && '
    '{ P="$d/agent-packages/agents/repository-manager"; break; }; '
    'd=$(dirname "$d"); '
    "done; "
    '[ -n "$P" ] || P=repository-manager'
)

#: Lines of the hook block, indentation-relative (first line at the block's
#: own `- id:` depth, the rest one level deeper) — mirrors
#: `inject_precommit_hooks.py`'s `HOOK_LINES` shape exactly.
HOOK_LINES = [
    "- id: dependency-readiness",
    "name: Dependency readiness — every declared intra-fleet constraint is installable",
    "entry: bash -c '"
    + _RESOLVE_REPOSITORY_MANAGER
    + '; uv run --no-project --python 3.12 --prerelease=allow --with "$P" '
    "python -m repository_manager.dependency_readiness .'",
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


#: Matches the existing hook's OWN `entry:` line only — anchored to the exact
#: indentation of the `- id: dependency-readiness` line found just above it,
#: so a sibling hook's `entry:` (or this hook's own `name:`/`stages:` lines)
#: is never touched. Built per-match in :func:`_resync_entry_line` since the
#: indent is discovered at each call site, not fixed fleet-wide (mirrors
#: `_hook_indent`'s same reasoning).
_CANONICAL_ENTRY_LINE = HOOK_LINES[2]  # "entry: bash -c '...'"


_ID_LINE_RE = re.compile(rf"^[ \t]*-\s*id:\s*{re.escape(HOOK_ID)}\s*$", re.MULTILINE)
#: The FIRST `entry:` line following an `- id: ...` line is that hook's own
#: (pre-commit hook blocks list `id`, `name`, `entry`, ... in that order;
#: nothing legitimately places a second hook's `entry:` before this one's).
_ENTRY_LINE_RE = re.compile(r"^([ \t]+)entry:[ \t]*(.*)$", re.MULTILINE)
#: A YAML block-scalar indicator (`>-`, `>`, `>+`, `|`, `|-`, `|+`) — when
#: `entry:` carries one of these instead of an inline value (graph-os's own
#: hand-fixed copy uses `entry: >-`), the value is the FOLLOWING more-indented
#: lines, which must be consumed as one span, never left as orphaned text
#: after only the `entry:` line itself is replaced (that would corrupt the
#: YAML: the old folded body would parse as sibling keys of the hook map).
_BLOCK_SCALAR_INDICATOR_RE = re.compile(r"^[|>][+-]?$")


def _entry_span(content: str, start: int) -> re.Match[str] | None:
    return _ENTRY_LINE_RE.search(content, start)


def _folded_block_end(content: str, entry_indent: str, after: int) -> int:
    """End offset of a YAML block scalar's continuation lines — every line
    more indented than `entry:` itself, starting right after it."""
    line_re = re.compile(r"^([ \t]*)(.*)$", re.MULTILINE)
    pos = after
    while True:
        match = line_re.match(content, pos)
        if match is None:
            return pos
        indent, rest = match.group(1), match.group(2)
        if rest.strip() and len(indent) <= len(entry_indent):
            return pos
        pos = match.end() + 1  # past this line's newline
        if pos > len(content):
            return len(content)


def _resync_entry_line(content: str) -> tuple[str, bool]:
    """Replace an ALREADY-PRESENT hook's `entry:` (inline OR YAML block
    scalar) with the canonical resolver (EH-173), leaving every other line —
    including a repo's own customized `name:`/`stages:` — untouched. A no-op
    (``changed=False``) when the entry already matches, so re-running is
    always safe."""
    id_match = _ID_LINE_RE.search(content)
    if id_match is None:
        return content, False
    entry_match = _entry_span(content, id_match.end())
    if entry_match is None:
        return content, False
    entry_indent, inline_value = entry_match.group(1), entry_match.group(2).strip()
    span_end = entry_match.end()
    if _BLOCK_SCALAR_INDICATOR_RE.match(inline_value):
        span_end = _folded_block_end(content, entry_indent, entry_match.end() + 1) - 1
    replacement = f"{entry_indent}{_CANONICAL_ENTRY_LINE}"
    if content[entry_match.start() : span_end] == replacement:
        return content, False
    new_content = content[: entry_match.start()] + replacement + content[span_end:]
    return new_content, True


#: repository-manager's OWN config is the one documented exception (module
#: docstring): it already has itself installed locally, so its hook invokes
#: `repository_manager.dependency_readiness` directly rather than resolving
#: a sibling checkout through `uv run --with`. Resync must never overwrite
#: that self-hosting form with the fleet's generic resolver.
_SELF_HOSTING_EXCEPTION = Path("agents/repository-manager/.pre-commit-config.yaml")


def _is_self_hosting_exception(filepath: Path) -> bool:
    return (
        filepath.parts[-len(_SELF_HOSTING_EXCEPTION.parts) :]
        == _SELF_HOSTING_EXCEPTION.parts
    )


@dataclass
class SweepResult:
    path: str
    action: str  # "already-present" | "would-inject-into-local-block" |
    #  "would-append-new-block" | "injected-into-local-block" |
    #  "appended-new-block" | "no-config" | "would-resync-entry" |
    #  "resynced-entry" | "already-canonical" | "self-hosting-exception"


def plan_or_apply(filepath: Path, *, apply: bool, resync: bool = False) -> SweepResult:
    if not filepath.exists():
        return SweepResult(str(filepath), "no-config")

    content = filepath.read_text(encoding="utf-8", errors="ignore")
    if f"id: {HOOK_ID}" in content:
        if not resync:
            return SweepResult(str(filepath), "already-present")
        if _is_self_hosting_exception(filepath):
            return SweepResult(str(filepath), "self-hosting-exception")
        new_content, changed = _resync_entry_line(content)
        if not changed:
            return SweepResult(str(filepath), "already-canonical")
        if apply:
            filepath.write_text(new_content, encoding="utf-8")
            return SweepResult(str(filepath), "resynced-entry")
        return SweepResult(str(filepath), "would-resync-entry")

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


#: Config filenames a repo's pre-commit hooks can live under. Most of the
#: fleet uses the standard `.pre-commit-config.yaml`; a handful of core
#: repos (epistemic-graph, agent-utilities, agent-connector-sdk, agent-webui
#: -- each documenting an "explicit-path tool configuration" convention in
#: their own AGENTS.md) use `.config/pre-commit.yaml` instead, invoked via
#: `pre-commit run --config .config/pre-commit.yaml`. Discovered live
#: (EH-173 follow-up): the original single-filename glob silently never
#: scanned any of these four repos, so they never received the fleet
#: rollout OR any `--resync` fix at all despite genuinely carrying the hook.
_HOOK_CONFIG_FILENAMES = (".pre-commit-config.yaml", ".config/pre-commit.yaml")


def sweep(root: Path, *, apply: bool, resync: bool = False) -> list[SweepResult]:
    results: list[SweepResult] = []
    seen: set[Path] = set()
    for filename in _HOOK_CONFIG_FILENAMES:
        for filepath in root.rglob(filename):
            if ".git" in filepath.parts or filepath in seen:
                continue
            seen.add(filepath)
            results.append(plan_or_apply(filepath, apply=apply, resync=resync))
    results.sort(key=lambda result: result.path)
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
    parser.add_argument(
        "--resync",
        action="store_true",
        help=(
            "for a repo that already has the hook (normally left untouched), "
            "rewrite its `entry:` line to the current canonical resolver "
            "(EH-173) if it differs — never touches `name:`/`stages:`/other "
            "customization. No-ops when already canonical."
        ),
    )
    args = parser.parse_args(argv)

    apply = args.apply
    results = sweep(args.root, apply=apply, resync=args.resync)

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
