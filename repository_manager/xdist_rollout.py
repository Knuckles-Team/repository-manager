"""pytest-xdist fleet rollout planner (CONCEPT:RM-XDIST-ROLLOUT).

**The gap this closes.** ``pytest-xdist`` is a declared dev/test dependency in
67 of this fleet's 75 repositories, and as of 2026-08-21 only
``agent-utilities`` itself actually passes ``-n`` at collection time — every
other repo pays the dependency's cost (resolve time, wheel download, lock-file
size) for none of its payoff. ``agent-utilities`` own ``pytest.ini`` records
why that payoff matters: 17,974 tests ran in ONE process, which is why its
pre-push gate exceeded 90 minutes and never returned a verdict before ``-n``
was switched on there. This module is the plan-then-apply machinery for
giving the other 66 repos the same switch, without guessing at any repo whose
hook does not look exactly like the ones already proven safe to touch.

**Conservative by construction — the eligibility gate never rewrites text it
cannot prove it understands.** Three gates, ALL required (see
:func:`_evaluate`):

1. ``pytest-xdist`` is declared in the repo's own ``pyproject.toml`` — either
   ``[project.optional-dependencies]`` (any group) or PEP 735
   ``[dependency-groups]`` (any group, string entries only — an
   ``{include-group = ...}`` reference is not a requirement and is skipped).
   A repo that never declared the dependency is out of scope for a ROLLOUT;
   adding a new dependency is a different, riskier operation this module
   does not perform.
2. The repo's ``pytest`` pre-commit hook ``entry:`` is BYTE-IDENTICAL to
   :data:`_BOILERPLATE_ENTRY` — captured 2026-08-21 from three independently
   read repos under ``agent-packages/agents/`` (``ansible-tower-mcp``,
   ``archivebox-api``, ``arr-mcp``) and confirmed byte-for-byte identical
   across all three (md5 of the extracted ``entry:`` line matched for all
   three). A bespoke entry — a repo that customized its test command — is
   reported ``skipped: non-boilerplate entry, no blind patch``, never
   force-patched. Never guess at a rewrite of text you cannot prove you
   understand: a scanner that mis-parses can at worst under-report (see
   :mod:`repository_manager.fail_fast_audit`'s own version of this same
   rule); a ROLLOUT that mis-rewrites can silently disable a repo's own
   pre-push gate. This module chooses the failure direction that costs a
   manual follow-up, never the one that costs a working gate.
3. No ``.rm-gates-no-xdist`` marker file at the repo root — an explicit,
   discoverable, per-repo opt-out that needs no code change to exercise.

**The insertion itself, and an honest account of where it came from.** The
flags applied are ``-n auto --dist loadfile -p no:randomly -rfE``.
``agent-utilities/pytest.ini`` was read on 2026-08-21 to source this from a
proven setting rather than inventing one, and it DOES confirm two of the four
pieces verbatim: ``-n auto`` and (materially) ``--dist loadfile`` — au's own
comment there explains why loadfile and not bare ``-n auto``: the suite has
order-dependent module state, so per-test round-robin distribution would turn
latent coupling into flakes, and loadfile keeps every test in one file on a
single worker. That file does NOT currently set ``-p no:randomly`` or
``-rfE`` (it sets ``--maxprocesses=12`` and ``-p no:logfire -p
no:pytest_logfire`` instead, for reasons specific to au's own plugin set) —
recorded here rather than silently overclaiming "confirmed" for a string that
was only two-quarters confirmed. Both additions are still deliberate, not
decorative: ``-p no:randomly`` is loadfile's necessary complement, not a
duplicate of it — loadfile only pins which WORKER a file's tests land on, it
does not stop a randomization plugin from reordering tests within a worker's
own file, which is exactly the order-dependent-state failure mode au's own
comment warns about. ``-rfE`` is required for a reason specific to THIS
package: :mod:`repository_manager.gate_ledger`'s clear-on-improve logic
extracts failing test ids from ``-r``-style per-test report lines; ``-q``
alone prints no such line, so without ``-rfE`` the ledger's per-test
extraction silently degrades to an empty failing set on any repo this module
touches.

**``apply`` defaults to ``dry_run=True``.** A rollout across 66 repos'
pre-commit configs is not a read; the default must never write a repo's gate
config as a side effect of an exploratory call.
"""

from __future__ import annotations

import tomllib
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

__all__ = [
    "XDIST_INSERTION",
    "RolloutPlanEntry",
    "plan",
    "apply",
    "dispatch",
]

#: agent-utilities' own proven setting (see module docstring for exactly
#: which pieces its pytest.ini does and does not confirm as of 2026-08-21).
XDIST_INSERTION: tuple[str, ...] = (
    "-n",
    "auto",
    "--dist",
    "loadfile",
    "-p",
    "no:randomly",
    "-rfE",
)
_INSERTION_TEXT = " ".join(XDIST_INSERTION)

#: The literal pytest invocation fragment every fleet boilerplate entry
#: carries twice (the ``uv run`` branch and the bare-``pytest`` fallback
#: branch) — insertion point for :data:`XDIST_INSERTION`.
_PYTEST_INVOCATION = 'pytest "$test_target" -q --tb=short'
_PYTEST_INVOCATION_PATCHED = f'pytest "$test_target" {_INSERTION_TEXT} -q --tb=short'

#: Captured verbatim from the ``entry:`` line of the ``pytest`` hook in
#: ``agent-packages/agents/ansible-tower-mcp/.pre-commit-config.yaml``,
#: ``.../archivebox-api/.pre-commit-config.yaml`` and
#: ``.../arr-mcp/.pre-commit-config.yaml`` on 2026-08-21 — all three were
#: byte-identical (md5-compared). Anything that does not match this exactly
#: is a bespoke entry and gate 2 refuses to touch it.
_BOILERPLATE_ENTRY = (
    "bash -c 'repo=$(dirname \"$(git rev-parse --path-format=absolute "
    '--git-common-dir)"); if [ -n "$AGENT_UTILITIES_ROOT" ]; then '
    'root="$AGENT_UTILITIES_ROOT"; else au_d="$repo"; root=""; while '
    '[ "$au_d" != "/" ]; do if [ -d "$au_d/agent-utilities/scripts" ]; then '
    'root="$au_d/agent-utilities"; break; fi; au_d=$(dirname "$au_d"); done; '
    'fi; if [ -d "$root" ] && [ "$repo" != "$root" ]; then mkdir -p '
    '.uv-workspace-siblings && ln -sfn "$root" '
    '.uv-workspace-siblings/agent-utilities; fi; test_target="tests"; for d '
    'in tests/unit test/unit tests test; do if [ -d "$d" ]; then '
    'test_target="$d"; break; fi; done; if [ -f uv.lock ]; then uv run '
    f'--all-extras {_PYTEST_INVOCATION} -m "not slow" '
    f'--timeout=60; else {_PYTEST_INVOCATION} -m "not slow" '
    "--timeout=60; fi'"
)

#: What :data:`_BOILERPLATE_ENTRY` looks like after the insertion has already
#: been applied — used to report "already applied" as distinct from
#: "eligible, not yet applied" without re-deriving the patched text twice.
_PATCHED_ENTRY = _BOILERPLATE_ENTRY.replace(
    _PYTEST_INVOCATION, _PYTEST_INVOCATION_PATCHED
)

#: Explicit, discoverable per-repo opt-out. A repo owner drops this file to
#: leave the fleet default without touching this module or its callers.
OPT_OUT_MARKER = ".rm-gates-no-xdist"

_XDIST_PACKAGE = canonicalize_name("pytest-xdist")


@dataclass
class RolloutPlanEntry:
    """One repo's eligibility verdict, and (from :func:`apply`) what was done."""

    repo: str
    eligible: bool
    reason: str
    already_applied: bool = False
    action: str = ""
    error: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _string_entries(values: Any) -> Iterable[str]:
    for value in values or []:
        if isinstance(value, str):
            yield value


def _iter_declared_requirement_strings(data: dict[str, Any]) -> Iterable[str]:
    """Every PEP 508 requirement string a repo's own ``pyproject.toml`` declares.

    Covers ``[project.dependencies]``, every group under
    ``[project.optional-dependencies]``, and every group under PEP 735
    ``[dependency-groups]`` — but only the STRING entries of the latter; a
    ``dependency-groups`` entry may also be ``{"include-group": "<name>"}``
    (a reference to another group, not a requirement), and treating that dict
    as a requirement string would raise deep inside ``packaging``, not
    report a clean "not declared".
    """

    project = data.get("project") or {}
    yield from _string_entries(project.get("dependencies"))
    for group in (project.get("optional-dependencies") or {}).values():
        yield from _string_entries(group)
    for group in (data.get("dependency-groups") or {}).values():
        yield from _string_entries(group)


def _declares_pytest_xdist(pyproject_path: Path) -> bool:
    try:
        text = pyproject_path.read_text(encoding="utf-8")
    except OSError:
        return False
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return False
    if not isinstance(data, dict):
        return False
    for raw in _iter_declared_requirement_strings(data):
        try:
            requirement = Requirement(raw)
        except InvalidRequirement:
            continue
        if canonicalize_name(requirement.name) == _XDIST_PACKAGE:
            return True
    return False


def _find_pytest_hook_entry(repo_blocks: Any) -> tuple[str | None, bool]:
    for repo_block in repo_blocks or []:
        if not isinstance(repo_block, dict):
            continue
        for hook in repo_block.get("hooks") or []:
            if isinstance(hook, dict) and hook.get("id") == "pytest":
                entry = hook.get("entry")
                return (entry if isinstance(entry, str) else None), True
    return None, False


def _pytest_hook_entry(repo_path: Path) -> tuple[str | None, bool]:
    """``(entry, hook_found)`` for the repo's ``id: pytest`` pre-commit hook.

    ``hook_found`` is ``False`` when there is no readable config or no hook
    named ``pytest`` at all; ``entry`` is ``None`` when the hook exists but
    its ``entry:`` is not a plain string (e.g. omitted, or a non-string YAML
    node this module has never seen in practice) — both are refusals to
    guess, not evidence of anything about the hook's correctness.
    """

    config_path = repo_path / ".pre-commit-config.yaml"
    if not config_path.is_file():
        return None, False
    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except (yaml.YAMLError, OSError, UnicodeDecodeError):
        return None, False
    if not isinstance(data, dict):
        return None, False
    return _find_pytest_hook_entry(data.get("repos"))


def _evaluate(repo_path: Path) -> RolloutPlanEntry:
    repo = str(repo_path)
    pyproject_path = repo_path / "pyproject.toml"
    if not pyproject_path.is_file():
        return RolloutPlanEntry(repo, False, "no pyproject.toml at repo root")
    if not _declares_pytest_xdist(pyproject_path):
        return RolloutPlanEntry(
            repo, False, "pytest-xdist not declared in this repo's dev/test deps"
        )

    entry, hook_found = _pytest_hook_entry(repo_path)
    if not hook_found:
        return RolloutPlanEntry(repo, False, "no pytest pre-commit hook found")
    if entry is None:
        return RolloutPlanEntry(
            repo, False, "pytest hook has no plain-string entry to inspect"
        )

    already_applied = entry == _PATCHED_ENTRY
    if not already_applied and entry != _BOILERPLATE_ENTRY:
        return RolloutPlanEntry(
            repo, False, "skipped: non-boilerplate entry, no blind patch"
        )

    if (repo_path / OPT_OUT_MARKER).exists():
        return RolloutPlanEntry(repo, False, f"opted out via {OPT_OUT_MARKER}")

    reason = "already carries the xdist insertion" if already_applied else "eligible"
    return RolloutPlanEntry(repo, True, reason, already_applied=already_applied)


def plan(repo_paths: Sequence[str | Path]) -> list[dict[str, Any]]:
    """Eligibility verdict for every repo in *repo_paths*. Never writes anything."""

    return [_evaluate(Path(repo_path)).as_dict() for repo_path in repo_paths]


def apply(
    repo_paths: Sequence[str | Path], *, dry_run: bool = True
) -> list[dict[str, Any]]:
    """Apply the xdist insertion to every ELIGIBLE, not-yet-applied repo.

    ``dry_run`` defaults ``True`` — a rollout across dozens of repos' gate
    configs must never be a side effect of an exploratory call. With
    ``dry_run=False``, a repo is only actually rewritten when its
    ``.pre-commit-config.yaml`` text still contains :data:`_BOILERPLATE_ENTRY`
    verbatim at write time (re-checked immediately before writing, not
    assumed from the earlier :func:`plan`/eligibility read — the two can
    race if something else touched the file in between).
    """

    results: list[dict[str, Any]] = []
    for raw_path in repo_paths:
        repo_path = Path(raw_path)
        entry_result = _evaluate(repo_path)
        record = entry_result.as_dict()

        if not entry_result.eligible:
            record["action"] = "skipped"
            results.append(record)
            continue
        if entry_result.already_applied:
            record["action"] = "noop_already_applied"
            results.append(record)
            continue
        if dry_run:
            record["action"] = "would_patch"
            results.append(record)
            continue

        config_path = repo_path / ".pre-commit-config.yaml"
        try:
            text = config_path.read_text(encoding="utf-8")
        except OSError as exc:
            record["action"] = "error"
            record["error"] = f"could not read {config_path}: {exc}"
            results.append(record)
            continue
        if _BOILERPLATE_ENTRY not in text:
            record["action"] = "error"
            record["error"] = (
                "boilerplate entry no longer present verbatim in the file "
                "at write time (re-run plan) -- refusing to guess"
            )
            results.append(record)
            continue
        patched = text.replace(_BOILERPLATE_ENTRY, _PATCHED_ENTRY)
        try:
            config_path.write_text(patched, encoding="utf-8")
        except OSError as exc:
            record["action"] = "error"
            record["error"] = f"could not write {config_path}: {exc}"
            results.append(record)
            continue
        record["action"] = "patched"
        results.append(record)
    return results


def _dispatch_plan_or_status(action: str, kwargs: dict[str, Any]) -> dict[str, Any]:
    repo_paths = kwargs.get("repo_paths") or []
    if not repo_paths:
        return {"ok": False, "error": f"{action} requires repo_paths"}
    entries = plan(repo_paths)
    return {
        "ok": True,
        "repos_checked": len(entries),
        "eligible": sum(1 for e in entries if e["eligible"]),
        "entries": entries,
    }


def _dispatch_apply(kwargs: dict[str, Any]) -> dict[str, Any]:
    repo_paths = kwargs.get("repo_paths") or []
    if not repo_paths:
        return {"ok": False, "error": "apply requires repo_paths"}
    dry_run = bool(kwargs.get("dry_run", True))
    entries = apply(repo_paths, dry_run=dry_run)
    errors = [e for e in entries if e["action"] == "error"]
    return {
        "ok": not errors,
        "dry_run": dry_run,
        "repos_checked": len(entries),
        "patched": sum(1 for e in entries if e["action"] == "patched"),
        "would_patch": sum(1 for e in entries if e["action"] == "would_patch"),
        "entries": entries,
    }


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core for the ``plan``/``apply``/``status`` surface.

    ``status`` is a read-only alias for ``plan`` — the same eligibility
    computation, worded for "what state is the fleet in right now" rather
    than "what would a rollout do", since the two questions have the same
    answer (this module keeps no separate applied/pending ledger of its own;
    ``already_applied`` is derived fresh from each repo's own config every
    call, never cached).
    """

    if action in {"plan", "status"}:
        return _dispatch_plan_or_status(action, kwargs)
    if action == "apply":
        return _dispatch_apply(kwargs)
    return {"ok": False, "error": f"unknown action: {action}"}
