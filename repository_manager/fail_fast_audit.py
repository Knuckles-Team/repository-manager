"""Static fail-fast-flag audit over a repo's pre-commit hooks (CONCEPT:RM-FAIL-FAST-AUDIT).

**This is DETECTION, not prevention — say it plainly so it is never mistaken
for the other thing.**
:mod:`repository_manager.test_commands` makes the never-stop-early guarantee
STRUCTURAL for argv this package itself builds: ``ensure_no_fail_fast`` is
applied at the process-launch chokepoint, immediately before
``subprocess.run``, so a declared command cannot reach the shell without it.
That trick has no equivalent here. :func:`repository_manager.gates.run_gate_stage`
never constructs a pre-commit hook's argv at all — it shells to
``pre-commit run --hook-stage <stage>`` and pre-commit itself parses each
repo's own ``.pre-commit-config.yaml`` ``entry:`` string, which is opaque
shell text this package never touches, let alone rewrites. This module reads
that same opaque text and reports what it finds; it cannot fix it, install a
runtime guard around it, or stop a hook from running with the flag still in
place. Do not describe the never-stop-early guarantee as closed for
pre-commit-driven gates anywhere — closing it there would require either
teaching ``pre-commit`` itself to rewrite hook argv (not this package's to
change) or replacing every repo's ``entry:`` with an invocation this package
constructs (a rollout, not a scan — see :mod:`repository_manager.xdist_rollout`
for what that class of change actually looks like and how conservatively it
has to be gated).

**The incident this exists to catch early.** ``test_commands``' own docstring
records the cost of a fail-fast flag surviving into a real run: one push
losing 89 of 90 minutes of signal to a single ``-x``, and a prior measurement
session that silently dropped ``--no-fail-fast`` from all 8 of its own
report points. Both of those were flags this package's own generated argv
could be fixed to never carry. A fail-fast flag baked into a repo's
*hand-authored* ``.pre-commit-config.yaml`` is the same defect wearing a
different hat — ``pytest -x`` in a pre-push hook's ``entry:`` still means
"stop reporting failures after the first one" — and nothing upstream of a
human reading that YAML currently notices. This module is that reading, done
mechanically and across the whole fleet.

**What counts as a violation.**

* ``pytest``: ``-x`` / ``--exitfirst``, ``--maxfail=N`` or ``--maxfail N``
  with ``N`` a nonzero count (``--maxfail=0`` means "no limit" to pytest, so
  it is explicitly not a truncation), and any bundled short option that
  carries ``x`` — e.g. ``-xvs``.
* ``cargo test`` / ``cargo nextest run``: MISSING ``--no-fail-fast``. Unlike
  the other two, cargo's truncation is opt-out, not opt-in — the absence of
  the flag is itself the violation.
* ``go test``: ``-failfast`` (either ``-`` or ``--`` spelling).

Recognition and the "what would fix this" diff are not reimplemented here.
:func:`repository_manager.test_commands.ensure_no_fail_fast` already knows,
for real argv, exactly which tokens a structurally-correct invocation would
add or remove; this module runs the *same* function against every candidate
window of tokens found inside an ``entry:`` string and reports where its
output differs from its input. That keeps exactly one definition of
"what does a fail-fast flag look like for this tool" in the codebase — this
module and ``test_commands`` can never quietly disagree about it, which
would otherwise be exactly how a rewrite fixes an argv shape a scanner no
longer recognizes (or vice versa).

**Not a shell parser, deliberately.** ``entry:`` is often a whole
``bash -c '...'`` script — conditionals, variable expansion, multiple
statements chained with ``;``/``&&``/``||``/``|``. This module tokenizes with
``shlex`` in punctuation-aware mode (so quoting is respected) and splits on
those four separators to approximate "one statement" boundaries, then walks
each statement's own token stream looking for a recognized test-runner
invocation. It does not understand subshells, command substitution
boundaries, or multi-line scripts that rely on a bare newline instead of an
explicit separator between two statements. Because this is detection, not
rewriting, the failure direction is the safe one: a construct this module
cannot segment correctly can produce a false negative (a flag it misses) but
never a false rewrite, since nothing here is ever written back to disk.
"""

from __future__ import annotations

import shlex
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from repository_manager.test_commands import (
    ensure_no_fail_fast,
    is_go_test_command,
    is_pytest_command,
    is_test_suite_command,
)

__all__ = [
    "Violation",
    "check_repo",
    "check_fleet",
    "dispatch",
]

#: Shell statement separators this module splits on. Deliberately excludes
#: ``(`` / ``)`` — a subshell boundary still contains a single command on
#: either side of most real hook entries in this fleet, and mis-splitting a
#: statement mid-token risks a crash on the tokenizer rather than a clean
#: decline, which is the wrong failure direction for a read-only scanner.
_SEPARATORS = frozenset({";", "&&", "||", "&", "|"})

#: Launchers whose ``-c``/``-lc`` payload is itself a nested shell script that
#: needs re-tokenizing, not scanned as a literal argument list. Every
#: fleet ``pytest``/``check_lane_guard`` hook entry seen in this repo is
#: wrapped exactly this way.
_SHELL_WRAPPERS = frozenset({"bash", "sh", "dash", "zsh"})
_SHELL_C_FLAGS = frozenset({"-c", "-lc", "-cl"})

_MAX_SNIPPET_CHARS = 220


@dataclass
class Violation:
    """One fail-fast flag found in one hook's ``entry:``, with its evidence."""

    repo: str
    hook_id: str
    tool: str  # "pytest" | "cargo" | "go"
    flags: tuple[str, ...]
    command: str
    message: str

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["flags"] = list(self.flags)
        return payload


def _tokenize(text: str) -> list[str]:
    """Punctuation-aware ``shlex`` tokenize; declines (returns ``[]``) rather
    than guess on anything it cannot parse, e.g. an unbalanced quote."""

    lexer = shlex.shlex(text, posix=True, punctuation_chars=True)
    lexer.whitespace_split = True
    try:
        return list(lexer)
    except ValueError:
        return []


def _split_statements(tokens: Sequence[str]) -> list[list[str]]:
    statements: list[list[str]] = []
    current: list[str] = []
    for token in tokens:
        if token in _SEPARATORS:
            if current:
                statements.append(current)
            current = []
            continue
        current.append(token)
    if current:
        statements.append(current)
    return statements


def _entry_statements(entry: str) -> list[list[str]]:
    """Every shell statement in *entry*, unwrapping one level of ``bash -c``.

    A hook that is not wrapped in a shell launcher is treated as a single
    statement (the common case for a plain ``entry: pytest -x`` line).
    """

    tokens = _tokenize(entry)
    if not tokens:
        return []
    statements: list[list[str]] = []
    for statement in _split_statements(tokens):
        if (
            len(statement) >= 3
            and statement[0] in _SHELL_WRAPPERS
            and statement[1] in _SHELL_C_FLAGS
        ):
            inner = _tokenize(statement[2])
            statements.extend(_split_statements(inner))
            if len(statement) > 3:
                # Rare: trailing args after the -c script itself (e.g. $0).
                statements.append(list(statement[3:]))
        else:
            statements.append(statement)
    return statements


def _removed_tokens(before: Sequence[str], after: Sequence[str]) -> list[str]:
    """Tokens present in *before* that have no matching slot left in *after*.

    Walks *before* left to right, consuming one occurrence from a multiset of
    *after* for every token that survived; anything left unconsumed is a
    token the rewrite dropped. Order-preserving and duplicate-safe, which a
    plain set difference is not (a repeated flag would otherwise vanish from
    the report after its first occurrence).
    """

    remaining = Counter(after)
    removed: list[str] = []
    for token in before:
        if remaining[token] > 0:
            remaining[token] -= 1
        else:
            removed.append(token)
    return removed


def _merge_maxfail_pair(flags: Sequence[str]) -> list[str]:
    """Cosmetic only: joins a two-token ``--maxfail``, ``N`` pair reported by
    :func:`_removed_tokens` into one readable ``"--maxfail N"`` entry."""

    merged: list[str] = []
    skip_next = False
    for index, flag in enumerate(flags):
        if skip_next:
            skip_next = False
            continue
        if flag == "--maxfail" and index + 1 < len(flags):
            merged.append(f"--maxfail {flags[index + 1]}")
            skip_next = True
        else:
            merged.append(flag)
    return merged


def _snippet(tokens: Sequence[str]) -> str:
    text = " ".join(tokens)
    if len(text) > _MAX_SNIPPET_CHARS:
        return text[: _MAX_SNIPPET_CHARS - 3] + "..."
    return text


def _message(tool: str, flags: tuple[str, ...]) -> str:
    if tool == "cargo":
        return (
            "cargo test/nextest invocation is missing --no-fail-fast; without "
            "it the run stops at the first failing test binary and every "
            "later failure is never reported."
        )
    joined = ", ".join(flags)
    return (
        f"{tool} invocation carries {joined}, which stops the run at the "
        f"first failing test and truncates every later failure from the "
        f"report."
    )


def _scan_statement(
    statement: Sequence[str],
) -> tuple[str, tuple[str, ...], str] | None:
    """First recognized test-runner invocation inside *statement*, if any.

    Tries every starting index so a launcher-prefixed invocation (``uv run
    --all-extras pytest ...``) is still recognized starting at ``uv`` (the
    recognisers in :mod:`repository_manager.test_commands` walk past known
    launchers themselves) without this module needing a second launcher
    table. Returns on the FIRST match, so one statement never yields more
    than one violation regardless of how many later indices would also
    happen to look like a match.

    Returns ``(tool, flags, command_snippet)`` or ``None``.
    """

    for start in range(len(statement)):
        suffix = list(statement[start:])
        if is_test_suite_command(suffix):
            fixed = ensure_no_fail_fast(suffix)
            if fixed == suffix:
                return None
            return "cargo", ("--no-fail-fast",), _snippet(suffix)
        if is_pytest_command(suffix):
            fixed = ensure_no_fail_fast(suffix)
            if fixed == suffix:
                return None
            removed = tuple(_merge_maxfail_pair(_removed_tokens(suffix, fixed)))
            if not removed:
                return None
            return "pytest", removed, _snippet(suffix)
        if is_go_test_command(suffix):
            fixed = ensure_no_fail_fast(suffix)
            if fixed == suffix:
                return None
            removed = tuple(_removed_tokens(suffix, fixed))
            if not removed:
                return None
            return "go", removed, _snippet(suffix)
    return None


def check_repo(repo_path: str | Path) -> list[Violation]:
    """Every fail-fast flag findable in *repo_path*'s ``.pre-commit-config.yaml``.

    A repo with no such file, an unparseable one, or one whose hooks carry no
    string ``entry:`` reports zero violations — this is a scanner for what
    IS declared and readable, never proof a repo without a readable config is
    clean. Callers that need to distinguish "checked, clean" from "could not
    check" should stat the config file themselves before calling this.
    """

    root = Path(repo_path)
    config_path = root / ".pre-commit-config.yaml"
    if not config_path.is_file():
        return []
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return []
    if not isinstance(data, dict):
        return []

    violations: list[Violation] = []
    for repo_block in data.get("repos") or []:
        if not isinstance(repo_block, dict):
            continue
        for hook in repo_block.get("hooks") or []:
            if not isinstance(hook, dict):
                continue
            entry = hook.get("entry")
            if not isinstance(entry, str) or not entry.strip():
                continue
            hook_id = str(hook.get("id") or "")
            for statement in _entry_statements(entry):
                if not statement:
                    continue
                found = _scan_statement(statement)
                if found is None:
                    continue
                tool, flags, command = found
                violations.append(
                    Violation(
                        repo=str(root),
                        hook_id=hook_id,
                        tool=tool,
                        flags=flags,
                        command=command,
                        message=_message(tool, flags),
                    )
                )
    return violations


def check_fleet(
    repo_paths: Sequence[str | Path],
) -> dict[str, list[Violation]]:
    """:func:`check_repo` for every path in *repo_paths*, keyed by repo path."""

    return {str(Path(repo_path)): check_repo(repo_path) for repo_path in repo_paths}


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core for the ``check``/``check_fleet`` surface (MCP + CLI share this).

    Returns ``{"ok": True, ...}`` when nothing was found (never mistake "ok"
    for "checked" — see :func:`check_repo`'s own caveat) or ``{"ok": False,
    ...}`` when at least one violation was found OR the call itself was
    malformed; those two failure shapes are distinguished by the presence of
    an ``"error"`` key.
    """

    if action == "check":
        repo_path = kwargs.get("repo_path")
        if not repo_path:
            return {"ok": False, "error": "check requires repo_path"}
        violations = check_repo(repo_path)
        return {
            "ok": not violations,
            "repo_path": str(repo_path),
            "violations": [violation.as_dict() for violation in violations],
        }
    if action == "check_fleet":
        repo_paths = kwargs.get("repo_paths") or []
        if not repo_paths:
            return {"ok": False, "error": "check_fleet requires repo_paths"}
        by_repo = check_fleet(repo_paths)
        flattened = [
            violation.as_dict()
            for violations in by_repo.values()
            for violation in violations
        ]
        return {
            "ok": not flattened,
            "repos_checked": len(repo_paths),
            "repos_with_violations": sum(1 for v in by_repo.values() if v),
            "violations": flattened,
            "by_repo": {
                repo: [violation.as_dict() for violation in violations]
                for repo, violations in by_repo.items()
            },
        }
    return {"ok": False, "error": f"unknown action: {action}"}
