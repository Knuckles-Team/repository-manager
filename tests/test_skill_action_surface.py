"""RMDD-21: prove every repository-manager skill matches the LIVE `rm_*` schema.

Skills are instructions other agents follow without reading source, so a skill
that references a non-existent action, or a "Tools & actions" table that quietly
understates the live action set, is a defect with a blast radius rather than a
documentation nit (an agent will follow it).

This module extracts every action reference from every packaged `SKILL.md` /
`WORKFLOW.md` under `repository_manager/skills/` and checks it against action sets
imported DIRECTLY from the modules the condensed MCP tools and the CLI parser
themselves dispatch through -- never a hand-copied list that could itself drift
out from under the skills.

Three reference shapes are checked:

1. An MCP-style call, e.g. ``rm_git(action="pull")``.
2. A CLI flag + literal action, e.g. ``--lane doctor`` / ``--merge-queue status``.
3. A CLI flag + brace choice-list, e.g. ``--concepts {reserve,list,...}``.
4. A "Tools & actions" markdown table row, e.g. ``| `rm_worktree` | `add`, `list` |``.
   A row whose cell is composed ONLY of backtick-wrapped action names (no trailing
   prose) reads as a claim of completeness, so it is checked for an EXACT set match
   against the live action set -- this is the shape that would have caught this
   lane's own worktree-orchestration drift (a live `reset_branch` action the table
   silently omitted). A row with trailing prose (an annotated/partial list, e.g.
   "``verify_candidate``, ``verify_generation`` -- see ... for the full surface")
   is checked only as a subset, since it never claimed to be complete.

``test_known_bad_*`` prove this checker actually catches bad input (H-9), by
running the SAME `find_violations` function against deliberately wrong fixtures
that never touch the real skill files.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from agent_utilities.mcp.action_dispatch import DISCOVERY_ACTIONS

from repository_manager.concept_actions import CONCEPT_ACTIONS
from repository_manager.lane_doctor import ACTIONS as RM_LANE_ACTIONS
from repository_manager.mcp_server import (
    RM_BUILD_ACTIONS,
    RM_GIT_ACTIONS,
    RM_MERGE_QUEUE_ACTIONS,
    RM_PROJECTS_ACTIONS,
    RM_WORKSPACE_ACTIONS,
    RM_WORKTREE_ACTIONS,
)
from repository_manager.remote_worker_actions import REMOTE_WORKER_ACTIONS

SKILLS_ROOT = Path(__file__).resolve().parent.parent / "repository_manager" / "skills"

# Condensed MCP tool name -> its live action set, imported directly (never
# hand-copied) from the exact module each tool dispatches through.
LIVE_ACTIONS: dict[str, frozenset[str]] = {
    "rm_git": frozenset(RM_GIT_ACTIONS),
    "rm_workspace": frozenset(RM_WORKSPACE_ACTIONS),
    "rm_worktree": frozenset(RM_WORKTREE_ACTIONS),
    "rm_merge_queue": frozenset(RM_MERGE_QUEUE_ACTIONS),
    "rm_build": frozenset(RM_BUILD_ACTIONS),
    "rm_projects": frozenset(RM_PROJECTS_ACTIONS),
    "rm_lane": frozenset(RM_LANE_ACTIONS),
    "rm_concepts": frozenset(CONCEPT_ACTIONS),
    "rm_remote_workers": frozenset(REMOTE_WORKER_ACTIONS),
}

# Every action-routed tool also accepts the shared discovery keywords
# (`list_actions`/`help`/`actions`) via `resolve_action`'s `DISCOVERY_ACTIONS` --
# imported live, not hand-copied, so this stays correct if that set ever changes.
_DISCOVERY = frozenset(DISCOVERY_ACTIONS)

# The five tools reachable through a dedicated top-level `--<flag> <action>` CLI
# form (repository_manager/cli_commands/parser.py); `rm_git`/`rm_workspace`/
# `rm_projects`/`rm_build`-as-packaging have no such single-flag CLI form and are
# only checked via the MCP-call and table-row shapes.
CLI_FLAG_TOOL: dict[str, str] = {
    "lane": "rm_lane",
    "merge-queue": "rm_merge_queue",
    "build-broker": "rm_build",
    "concepts": "rm_concepts",
    "remote-workers": "rm_remote_workers",
}

_MCP_CALL = re.compile(r"\b(rm_[a-z_]+)\(action=[\"']([a-z_]+)[\"']")
_CLI_FLAG_ACTION = re.compile(
    r"--(lane|merge-queue|build-broker|concepts|remote-workers)\s+([a-z_]+)\b"
)
_CLI_FLAG_CHOICES = re.compile(
    r"--(lane|merge-queue|build-broker|concepts|remote-workers)\s*\{([^}]+)\}"
)
_TOOL_TABLE_ROW = re.compile(
    r"^\|\s*`(rm_[a-z_]+)`\s*\|\s*(.+?)\s*\|\s*$", re.MULTILINE
)
_BACKTICK_ACTION = re.compile(r"`([a-z_]+)`")
# A cell composed ONLY of backtick-wrapped tokens/commas/whitespace -- no trailing
# prose -- reads as a claim of completeness and is held to an exact-set match.
_CANONICAL_CELL = re.compile(r"^(?:`[a-z_]+`,?\s*)+$")


def find_violations(text: str) -> list[str]:
    """Return every action reference in ``text`` that disagrees with the live schema.

    Pure over its input -- no filesystem access -- so the exact same function
    backs both the real-skill sweep and the known-bad-input proof below.
    """

    violations: list[str] = []

    for tool, action in _MCP_CALL.findall(text):
        live = LIVE_ACTIONS.get(tool)
        if live is None:
            continue
        if action in _DISCOVERY:
            continue
        if action not in live:
            violations.append(
                f"{tool}(action={action!r}) is not in the live action set "
                f"{sorted(live)}"
            )

    for flag, action in _CLI_FLAG_ACTION.findall(text):
        tool = CLI_FLAG_TOOL[flag]
        live = LIVE_ACTIONS[tool]
        if action in _DISCOVERY:
            continue
        if action not in live:
            violations.append(
                f"--{flag} {action} is not in {tool}'s live action set {sorted(live)}"
            )

    for flag, choices_blob in _CLI_FLAG_CHOICES.findall(text):
        tool = CLI_FLAG_TOOL[flag]
        live = LIVE_ACTIONS[tool]
        choices = {c.strip() for c in choices_blob.split(",") if c.strip()}
        extra = choices - live
        missing = live - choices
        if extra:
            violations.append(
                f"--{flag} {{...}} choice list names unknown actions {sorted(extra)}"
            )
        if missing:
            violations.append(
                f"--{flag} {{...}} choice list is missing live actions {sorted(missing)}"
            )

    for tool, cell in _TOOL_TABLE_ROW.findall(text):
        live = LIVE_ACTIONS.get(tool)
        if live is None:
            continue
        listed = set(_BACKTICK_ACTION.findall(cell))
        extra = listed - live
        if extra:
            violations.append(
                f"'Tools & actions' row for {tool} lists unknown actions "
                f"{sorted(extra)}"
            )
        if _CANONICAL_CELL.match(cell):
            missing = live - listed
            if missing:
                violations.append(
                    f"'Tools & actions' row for {tool} looks like a complete "
                    f"action list but is missing live actions {sorted(missing)}"
                )

    return violations


def _skill_markdown_files() -> list[Path]:
    files = sorted(SKILLS_ROOT.glob("*/SKILL.md")) + sorted(
        SKILLS_ROOT.glob("*/WORKFLOW.md")
    )
    assert files, f"no skill markdown found under {SKILLS_ROOT}"
    return files


@pytest.mark.parametrize(
    "path",
    _skill_markdown_files(),
    ids=lambda p: str(p.relative_to(SKILLS_ROOT)),
)
def test_skill_action_references_match_live_schema(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    violations = find_violations(text)
    assert not violations, (
        f"{path.relative_to(SKILLS_ROOT)} references actions that disagree with "
        f"the live rm_* schema:\n" + "\n".join(violations)
    )


def test_known_good_fixture_produces_no_violations() -> None:
    """Sanity control: the checker must not always fail."""

    good = (
        "| `rm_build` | `request`, `status`, `artifacts`, `explain`, `gc` |\n"
        'rm_build(action="request")\n'
        'rm_build(action="list_actions")\n'
        "CLI: `repository-manager --concepts "
        "{reserve,list,get,release,materialize,verify_candidate,verify_generation,reconcile}`\n"
    )
    assert find_violations(good) == []


def test_known_bad_mcp_call_action_is_caught() -> None:
    """H-9 proof #1: a nonexistent action inside an `rm_git(action=...)` call."""

    bad = '```\nrm_git(action="raw_shell_exec")\n```\n'
    violations = find_violations(bad)
    assert violations, "the checker did not flag a nonexistent rm_git action"
    assert any("raw_shell_exec" in v for v in violations)


def test_known_bad_cli_flag_action_is_caught() -> None:
    """H-9 proof #2: a nonexistent action after a dedicated CLI flag."""

    bad = "repository-manager --lane teleport --lane-path .\n"
    violations = find_violations(bad)
    assert violations, "the checker did not flag a nonexistent --lane action"
    assert any("teleport" in v for v in violations)


def test_known_bad_cli_choice_list_is_caught() -> None:
    """H-9 proof #3: a brace choice-list naming an action that does not exist."""

    bad = "CLI: `repository-manager --concepts {reserve,list,teleport}`\n"
    violations = find_violations(bad)
    assert violations, "the checker did not flag an unknown --concepts choice"
    assert any("teleport" in v for v in violations)


def test_known_bad_table_undercoverage_is_caught() -> None:
    """H-9 proof #4 -- the exact historical defect this lane fixed: a canonical-
    looking 'Tools & actions' row silently missing a live action. RMDD-20 froze
    `reset_branch` as RM_WORKTREE_ACTIONS' 9th member; the pre-fix
    worktree-orchestration table still listed only 8."""

    bad = (
        "| `rm_worktree` | `add`, `list`, `remove`, `merge`, `sync`, `prune`, "
        "`bulk_add`, `audit` |\n"
    )
    violations = find_violations(bad)
    assert violations, "the checker did not flag a table row missing reset_branch"
    assert any("reset_branch" in v for v in violations)


def test_annotated_partial_table_row_is_not_flagged_for_undercoverage() -> None:
    """A row with trailing prose never claimed completeness, so it is checked as
    a subset only -- this must NOT raise, or every deliberately partial
    cross-reference table in these skills would be an unfixable false positive."""

    partial = (
        "| `rm_concepts` | `verify_candidate`, `verify_generation` — see "
        "`repository-manager-concept-coordination` for the full 8-action surface. |\n"
    )
    assert find_violations(partial) == []
