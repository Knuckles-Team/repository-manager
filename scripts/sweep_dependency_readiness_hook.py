#!/usr/bin/env python3
"""Reconcile recognized fleet RELEASE hooks to the pipelines catalogue.

Dry runs print reviewable diffs. Apply requires an explicit publisher audit:
all actual publication paths must already use immutable guarded pipelines refs.
This tool never edits publishers or custom hooks and never installs a checker.
Pipelines owns staging; repository-manager owns fleet readiness and this updater.
"""

from __future__ import annotations

import argparse
import difflib
import re
import os
import tempfile
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

HOOK_ID = "dependency-readiness"
PIPELINES = "https://github.com/Knuckles-Team/pipelines"
LEGACY_ENTRY = "bash -c 'uv run --with repository-manager python -m repository_manager.dependency_readiness .'"
LOCATOR_ENTRY = '''bash -c 'P=""; d="$PWD"; while [ "$d" != "/" ]; do [ -d "$d/agent-packages/agents/repository-manager" ] && { P="$d/agent-packages/agents/repository-manager"; break; }; d=$(dirname "$d"); done; [ -n "$P" ] || P=repository-manager; uv run --no-project --python 3.12 --prerelease=allow --with "$P" python -m repository_manager.dependency_readiness .' '''.rstrip()
LEGACY_NAMES = {
    "Dependency readiness — every declared intra-fleet constraint is installable",
    "RELEASE dependency readiness — every declared intra-fleet constraint is installable",
}


@dataclass
class SweepResult:
    """One configuration's action and exact proposed/applied diff."""

    path: str
    action: str
    diff: str = ""


def _fields(node: yaml.MappingNode) -> dict[str, yaml.Node]:
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("expected mapping")
    keys = [key.value for key, _ in node.value]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate YAML keys")
    return dict((key.value, value) for key, value in node.value)


def _validate_nodes(node: yaml.Node) -> None:
    if isinstance(node, yaml.MappingNode):
        for child in _fields(node).values():
            _validate_nodes(child)
    elif isinstance(node, yaml.SequenceNode):
        for child in node.value:
            _validate_nodes(child)


def _parse(content: str) -> tuple[dict, yaml.SequenceNode]:
    if any(isinstance(t, (yaml.AliasToken, yaml.AnchorToken)) for t in yaml.scan(content)):
        raise ValueError("aliases/anchors need manual review")
    node = yaml.compose(content)
    _validate_nodes(node)
    repos = _fields(node)["repos"]
    if not isinstance(repos, yaml.SequenceNode) or repos.flow_style:
        raise ValueError("repos must be a block sequence")
    data = yaml.safe_load(content)
    if not isinstance(data["repos"], list) or not data["repos"]:
        raise ValueError("repos must not be empty")
    _validate_repos(data["repos"])
    return data, repos


def _validate_repos(repos: list) -> None:
    for repo in repos:
        if not isinstance(repo, dict) or not isinstance(repo.get("hooks"), list):
            raise ValueError("invalid repository hooks")
        if any(not isinstance(hook, dict) for hook in repo["hooks"]):
            raise ValueError("invalid hook")


def _last_line(node: yaml.Node) -> int:
    if isinstance(node, yaml.MappingNode):
        return max(_last_line(v) for pair in node.value for v in pair)
    if isinstance(node, yaml.SequenceNode) and not node.flow_style:
        return max(_last_line(v) for v in node.value)
    return node.end_mark.line + bool(node.end_mark.column)


def _legacy(hook: dict) -> bool:
    expected = {"id", "name", "entry", "language", "pass_filenames", "always_run", "stages"}
    return (
        set(hook) == expected
        and hook["name"] in LEGACY_NAMES
        and hook["entry"] in {LEGACY_ENTRY, LOCATOR_ENTRY}
        and hook["language"] == "system"
        and hook["pass_filenames"] is False
        and hook["always_run"] is True
        and hook["stages"] in (["manual"], ["pre-push"], ["manual", "pre-push"], ["pre-push", "manual"])
    )


def _legacy_span(repo: dict, repo_node: yaml.MappingNode, j: int,
                 lines: list[str]) -> tuple[int, int]:
    hooks = _fields(repo_node)["hooks"]
    if not isinstance(hooks, yaml.SequenceNode):
        raise ValueError("hooks must be a sequence")
    node = repo_node if len(repo["hooks"]) == 1 else hooks.value[j]
    if not isinstance(node, yaml.MappingNode):
        raise ValueError("hook must be a mapping")
    if node.flow_style or hooks.flow_style:
        raise ValueError("flow mappings need review")
    if len(repo["hooks"]) == 1 and set(repo) != {"repo", "hooks"}:
        raise ValueError("custom repository options need review")
    span = (node.start_mark.line, _last_line(node))
    if any("#" in line for line in lines[slice(*span)]):
        raise ValueError("comments need review")
    return span


def _select_removal(data: dict, repos: yaml.SequenceNode, matches: list,
                    lines: list[str], ref: str) -> tuple[str, tuple[int, int]]:
    end = _last_line(repos)
    if not matches:
        return "added-central", (end, end)
    i, j = matches[0]
    repo = data["repos"][i]
    hook = repo["hooks"][j]
    if repo.get("repo") == PIPELINES and repo.get("rev") == ref and hook == {"id": HOOK_ID}:
        return "already-central", (end, end)
    if repo.get("repo") != "local" or not _legacy(hook):
        return "review-required-custom", (end, end)
    return "reconciled", _legacy_span(repo, repos.value[i], j, lines)


def _render(content: str, repos: yaml.SequenceNode, ref: str,
            remove: tuple[int, int]) -> str:
    lines = content.splitlines(keepends=True)
    insert = _last_line(repos)
    indent = " " * repos.start_mark.column
    block = [f"{indent}- repo: {PIPELINES}\n", f"{indent}  rev: {ref}\n",
             f"{indent}  hooks:\n", f"{indent}  - id: {HOOK_ID}\n"]
    if "\r\n" in content:
        block = [line.replace("\n", "\r\n") for line in block]
    before = lines[:remove[0]] + lines[remove[1]:insert]
    if before and not before[-1].endswith("\n"):
        before[-1] += "\n"
    updated = "".join(before + block + lines[insert:])
    _parse(updated)
    return updated


def _replacement(content: str, pipelines_ref: str) -> tuple[str, str]:
    data, repos = _parse(content)
    matches = [(i, j) for i, repo in enumerate(data["repos"])
               for j, hook in enumerate(repo["hooks"]) if hook.get("id") == HOOK_ID]
    if len(matches) > 1:
        return "review-required-duplicate", content
    action, remove = _select_removal(data, repos, matches, content.splitlines(True), pipelines_ref)
    if action not in {"added-central", "reconciled"}:
        return action, content
    return action, _render(content, repos, pipelines_ref, remove)


def _read_config(filepath: Path) -> str | None:
    if any(path.is_symlink() for path in (filepath, *filepath.parents)):
        raise ValueError("symlink configuration needs review")
    if not filepath.exists():
        return None
    with filepath.open(encoding="utf-8", newline="") as stream:
        return stream.read()


def plan_or_apply(filepath: Path, *, apply: bool, pipelines_ref: str,
                  publishers_reviewed: bool = False) -> SweepResult:
    """Preserve unrecognized entries; mutate only after explicit publisher audit."""
    if not re.fullmatch(r"[0-9a-f]{40}", pipelines_ref):
        raise ValueError("pipelines-ref must be a full immutable commit SHA")
    try:
        content = _read_config(filepath)
        if content is None:
            return SweepResult(str(filepath), "no-config")
        action, updated = _replacement(content, pipelines_ref)
    except (ValueError, KeyError, TypeError, AttributeError, yaml.YAMLError):
        return SweepResult(str(filepath), "review-required-invalid")
    diff = "".join(difflib.unified_diff(content.splitlines(True), updated.splitlines(True),
                                        fromfile=str(filepath), tofile=str(filepath)))
    if not diff:
        return SweepResult(str(filepath), action)
    if apply and not publishers_reviewed:
        return SweepResult(str(filepath), "blocked-publisher-audit", diff)
    if apply:
        _write_atomic(filepath, content, updated)
    return SweepResult(str(filepath), action if apply else f"would-{action}", diff)


def _write_atomic(path: Path, expected: str, updated: str) -> None:
    """Keep mode and refuse to overwrite a configuration edited since planning."""
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="",
                                     dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(updated)
    try:
        temporary.chmod(path.stat().st_mode)
        with path.open(encoding="utf-8", newline="") as stream:
            if stream.read() != expected:
                raise ValueError("configuration changed during reconciliation")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def sweep(root: Path, *, apply: bool, pipelines_ref: str,
          publishers_reviewed: bool = False) -> list[SweepResult]:
    """Plan or apply changes to existing fleet configs, never Git internals."""
    return [plan_or_apply(p, apply=apply, pipelines_ref=pipelines_ref,
                          publishers_reviewed=publishers_reviewed)
            for p in sorted(root.rglob(".pre-commit-config.yaml")) if ".git" not in p.parts]


def validate_contract(checkout: Path, ref: str) -> None:
    """Read the immutable catalogue from Git, never from a dirty working tree."""
    if not re.fullmatch(r"[0-9a-f]{40}", ref):
        raise ValueError("pipelines-ref must be a full immutable commit SHA")
    raw = subprocess.check_output(["git", "-C", str(checkout), "show", f"{ref}:.pre-commit-hooks.yaml"], text=True)
    hooks = [h for h in yaml.safe_load(raw) if h["id"] == HOOK_ID]
    if len(hooks) != 1 or hooks[0].get("stages") != ["manual"] or hooks[0].get("entry") != "pipelines-hook dependency-readiness":
        raise ValueError("pinned catalogue does not own the manual release hook")


def main(argv: list[str] | None = None) -> int:
    """Print exact diffs; require an audited publisher graph before applying."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--pipelines-ref", required=True)
    parser.add_argument("--pipelines-checkout", required=True, type=Path)
    parser.add_argument("--publishers-reviewed", action="store_true",
                        help="all publication paths audited at immutable guarded refs")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--apply", action="store_true")
    args = parser.parse_args(argv)
    validate_contract(args.pipelines_checkout, args.pipelines_ref)
    results = sweep(args.root, apply=args.apply, pipelines_ref=args.pipelines_ref,
                    publishers_reviewed=args.publishers_reviewed)
    for result in results:
        print(f"[{result.action}] {result.path}")
        print(result.diff, end="")
    return int(any(r.action.startswith(("blocked-", "review-required-")) for r in results))


if __name__ == "__main__":
    sys.exit(main())
