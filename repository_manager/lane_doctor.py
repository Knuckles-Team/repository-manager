"""Lane preflight — the concurrent-development rules, executable (CONCEPT:RM-LANE-DOCTOR).

Every check in this module exists because a *specific* collision destroyed real
work on this workspace while the rule that would have prevented it was written
down, in prose, in front of the actor who broke it. Prohibitions demonstrably do
not stick; a check that names the violation, the evidence, and the literal
remedy command does.

The three verbs form the whole lane lifecycle, and each one is a composition of
machinery that already exists rather than a new mechanism:

``start``
    :class:`repository_manager.worktree.WorktreeManager.add` + this module's
    isolation exports + :func:`diagnose` on the tree it just made. One call
    replaces the three-command prose block in the concurrent-development
    charter, and it *proves* the isolation instead of asserting it.

``doctor``
    :func:`diagnose` alone — safe to run at any moment, mutates nothing, and is
    the thing to run when something behaves impossibly (phantom test failures,
    a build that will not go green, a merge that keeps being refused).

``finish``
    :func:`diagnose` as a **blocking** gate, then hand the branch to the
    serialized merge queue. A lane that cannot pass its own preflight must not
    become a merge candidate: the queue would spend a full gate cycle
    discovering what a sub-second check already knew.

What this module deliberately does **not** do: decide whether the work is
*correct*. Gates do that, they are declared per repository in
``.mergequeue.yaml``, and they run inside the queue against the **merged** tree.
This is the isolation layer only.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess  # nosec B404 - git/CLI orchestration is this module's entire job
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "Check",
    "OK",
    "WARN",
    "FAIL",
    "SKIP",
    "diagnose",
    "lane_exports",
    "start",
    "finish",
    "dispatch",
    "main",
]

#: A check's verdict. ``FAIL`` blocks ``finish``; ``WARN`` never does — it names
#: a condition that is legitimate in some lanes and fatal in others, so the
#: decision stays with the lane rather than being guessed here.
OK = "ok"
WARN = "warn"
FAIL = "fail"
SKIP = "skip"

_BLOCKING = (FAIL,)

#: pre-commit's own default store. Sharing it across lanes is the D-OB-12
#: hazard: ``staged_files_only()`` writes a lane's UNSTAGED work into a patch
#: file here, ``git checkout``s it away so hooks see only staged content, and
#: restores it in a ``finally:``. A crash inside that window leaves the work in
#: an orphaned patch nobody replays, and the same directory holds pre-commit's
#: SQLite ``db.db`` (``OperationalError: database is locked`` under concurrency).
_SHARED_PRECOMMIT_HOME = Path.home() / ".cache" / "pre-commit"


@dataclass
class Check:
    """One executable rule, its verdict, its evidence, and its literal remedy."""

    name: str
    status: str
    finding: str
    remedy: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def blocking(self) -> bool:
        return self.status in _BLOCKING

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(args: list[str], cwd: Path, *, timeout: int = 60) -> tuple[int, str]:
    """Run git and return ``(returncode, stripped stdout)``; never raises."""
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell, git from PATH
            ["git", *args],
            cwd=str(cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover
        return 1, f"{type(exc).__name__}: {exc}"
    return proc.returncode, (proc.stdout or "").strip()


def _resolve_tree(path: Path | str | None) -> Path:
    return Path(path).resolve() if path else Path.cwd().resolve()


def _lane_scope(tree: Path) -> Any:
    """agent-utilities' lane scope, or ``None`` when it cannot be resolved.

    repository-manager depends on agent-utilities, so this import is normally
    available. It is still defensive because this module must be able to
    diagnose a tree in an environment where the governance package is broken —
    that is precisely a moment when a lane needs a doctor.
    """
    try:
        from agent_utilities.governance import lanes
    except Exception:  # pragma: no cover - environment-dependent
        return None
    try:
        return lanes.lane_scope(tree)
    except Exception:  # pragma: no cover - not a git tree
        return None


def _lane_temp_root(tree: Path) -> Path | None:
    """This lane's private, host-valid temp root, or ``None`` (D-CDX-40).

    Mirrors what :func:`lane_exports` actually creates
    (:func:`_host_valid_lane_temp_root`, anchored at the worktree — not
    ``agent_utilities.governance.lanes.partitioned_paths()``'s
    ``$HOME``-anchored root) so this hint and the real export never diverge.
    """
    scope = _lane_scope(tree)
    if scope is None:
        return None
    return _host_valid_lane_temp_root(tree, scope)


def _host_valid_lane_temp_root(tree: Path, scope: Any) -> Path:
    """A per-lane temp root that is valid for WHOEVER operates on ``tree``.

    D-CDX-40 — ``agent_utilities.governance.lanes.partitioned_paths()`` roots
    per-lane temp state (pytest basetemp / TMPDIR / pre-commit store) at
    ``Path.home() / ".al" / <token>`` — a deliberate choice for short AF_UNIX
    socket paths (see that function's docstring), correct when the process
    resolving the path and the process using it share one ``$HOME``. That
    assumption holds for the local CLI but not the MCP surface:
    repository-manager-mcp resolves it in the *provider's* identity/container
    (verified live: exports rooted at ``/home/rm/.al/<hash>`` from a container
    running as user ``rm``, HOME=``/home/rm``), which a host caller running as
    a different user does not have — every one of 30 focused fan-out tests
    ERRORed with ``FileNotFoundError`` applying that basetemp on the host; the
    same tests passed 30/30 against a host-valid ``/tmp``-rooted basetemp.

    Anchoring at the worktree's own PARENT directory instead uses a path both
    the provider and the caller already agree on: it is the literal
    filesystem location of the lane's worktree. Whichever identity can reach
    ``tree`` at all (the only identity for which these exports are relevant)
    can create and use a sibling directory next to it — unlike ``$HOME``,
    which is a property of the *process*, not the *worktree*.
    """
    token = hashlib.sha1(
        f"{getattr(scope, 'common_dir', tree)}:{getattr(scope, 'lane', tree.name)}".encode(),
        usedforsecurity=False,
    ).hexdigest()[:12]
    return tree.parent / f".al-{token}"


# ---------------------------------------------------------------------------
# Isolation exports
# ---------------------------------------------------------------------------
def lane_exports(path: Path | str | None = None) -> dict[str, str]:
    """The environment that gives this lane its own build/test/hook state.

    ``PRE_COMMIT_HOME`` is resolved here rather than being left to
    agent-utilities' ``lane env`` on purpose: it is the newest PARTITION-class
    resource and the one whose absence is *silent* — a shared store does not
    fail, it just occasionally eats a lane's unstaged work. Resolving it in the
    same call that creates the worktree means no lane can start without it.

    The pytest basetemp / TMPDIR / pre-commit store paths are deliberately
    resolved via :func:`_host_valid_lane_temp_root`, NOT
    ``agent_utilities.governance.lanes.partitioned_paths()`` — see D-CDX-40 in
    that function's docstring. ``CARGO_TARGET_DIR`` is unaffected: it was
    already tree-relative (``tree / "target-isolated"``) and therefore always
    host-valid.
    """
    tree = _resolve_tree(path)
    exports: dict[str, str] = {}
    scope = _lane_scope(tree)
    if scope is None:
        return exports

    temp_root = _host_valid_lane_temp_root(tree, scope)
    temp_root.mkdir(parents=True, exist_ok=True)
    scratch_dir = temp_root / "scratch"
    scratch_dir.mkdir(parents=True, exist_ok=True)
    precommit_home = temp_root / "pre-commit"
    precommit_home.mkdir(parents=True, exist_ok=True)

    exports["CARGO_TARGET_DIR"] = str(tree / "target-isolated")
    exports["TMPDIR"] = str(scratch_dir)
    exports["PYTEST_ADDOPTS"] = f"--basetemp={temp_root / 'pytest'}"
    exports["PRE_COMMIT_HOME"] = str(precommit_home)
    return exports


def _shell_block(exports: dict[str, str]) -> str:
    return "\n".join(
        f"export {key}={value!r}" for key, value in sorted(exports.items())
    )


# ---------------------------------------------------------------------------
# The checks
# ---------------------------------------------------------------------------
def _check_not_canonical(tree: Path, scope: Any) -> Check:
    """READ-ONLY: a background actor once reset a canonical tree mid-edit."""
    if scope is None:
        return Check(
            "not-canonical",
            SKIP,
            "lane scope unresolvable — cannot tell a canonical checkout from a worktree",
            remedy="install agent-utilities so agent_utilities.governance.lanes imports",
        )
    if scope.is_canonical:
        return Check(
            "not-canonical",
            FAIL,
            f"{tree} IS the canonical checkout. It is READ-ONLY for lanes: a "
            "background sync's `git reset` here has already destroyed ~20 minutes "
            "of a lane's work.",
            remedy=(
                "repository-manager --lane start --lane-repo <repo> "
                "--lane-branch <branch>"
            ),
            evidence={"tree": str(tree), "lane": scope.lane},
        )
    return Check(
        "not-canonical",
        OK,
        f"working in linked worktree {tree} (lane {scope.lane!r})",
        evidence={"canonical": str(scope.main_tree)},
    )


def _probe_interpreter(python: Path) -> tuple[str, str]:
    """Classify a venv interpreter BEHAVIOURALLY: ``(state, detail)``.

    D-CDX-62 — the structural test this replaces (``python.is_file()``) cannot
    distinguish "this environment is broken" from "this environment's
    interpreter is not visible to the process asking". A uv-managed venv's
    ``bin/python`` is a SYMLINK into the uv interpreter store
    (``~/.local/share/uv/python/...``); ``Path.is_file()`` resolves symlinks,
    so wherever that store is not mounted — exactly the repository-manager-mcp
    in-pod case — a perfectly healthy interpreter reports as ABSENT and the
    lane is handed a blocking failure it cannot act on. That happened to a
    genuinely healthy lane on 2026-08-11.

    So: ask the interpreter, don't inspect it. Executing it is the only test
    that answers the question this check exists to ask.

    States: ``ok`` (it ran), ``absent`` (nothing at that path), ``broken``
    (present and resolvable, but will not execute), ``unevaluable`` (a symlink
    whose target does not resolve here — indistinguishable from a mount
    namespace that cannot see the interpreter store).
    """
    if not python.is_symlink() and not python.exists():
        return "absent", f"no such path ({python})"
    try:
        proc = subprocess.run(  # nosec B603 - fixed argv, no shell
            [str(python), "-c", "import sys; print(sys.executable)"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except FileNotFoundError:
        if python.is_symlink():
            return (
                "unevaluable",
                f"interpreter is a symlink to {os.readlink(python)}, which does "
                "not resolve from this process — indistinguishable from a mount "
                "namespace that cannot see the interpreter store, so the "
                "environment is NOT judged from here",
            )
        return "broken", f"interpreter will not execute ({python})"
    except (OSError, subprocess.SubprocessError) as exc:
        return "unevaluable", f"could not probe interpreter ({python}): {exc}"
    if proc.returncode != 0:
        return "broken", f"interpreter exited {proc.returncode} ({python})"
    return "ok", proc.stdout.strip()


def _check_no_local_venv(tree: Path) -> Check:
    """A worktree-local ``.venv`` produced ~167 phantom test failures."""
    venv = tree / ".venv"
    if not venv.exists():
        return Check("no-worktree-venv", OK, "no worktree-local .venv")

    launcher = tree / "scripts" / "uv_workspace.py"
    marker = venv / ".uv-workspace-selection.json"
    python = venv / "bin" / "python"
    expected = {"label": "", "selection": ["--all-extras"]}
    reason = ""
    unevaluable = ""
    loaded: Any = None
    if venv.is_symlink():
        reason = "the environment directory is a symlink"
    elif not launcher.is_file():
        reason = f"the owning launcher is absent ({launcher})"
    elif marker.is_symlink() or not marker.is_file():
        reason = f"the ownership marker is absent or not a regular file ({marker})"
    else:
        try:
            loaded = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            reason = f"the ownership marker is unreadable or invalid JSON ({marker})"
        if not reason and loaded != expected:
            reason = "the ownership marker does not identify the canonical all-extras selection"
        elif not reason and not (venv / "pyvenv.cfg").is_file():
            reason = f"the environment metadata is absent ({venv / 'pyvenv.cfg'})"
        elif not reason:
            state, detail = _probe_interpreter(python)
            if state == "unevaluable":
                unevaluable = detail
            elif state != "ok":
                reason = f"the environment interpreter is unusable: {detail}"

    if unevaluable:
        return Check(
            "no-worktree-venv",
            SKIP,
            f"cannot evaluate {venv} from this process: {unevaluable}",
            remedy=(
                "re-run the doctor from inside the lane's own namespace (the "
                f"local CLI: `lane doctor --path {tree}`) — a check that cannot "
                "see the lane's filesystem must not rule on it"
            ),
            evidence={"venv": str(venv), "marker": str(marker)},
        )

    if not reason:
        return Check(
            "no-worktree-venv",
            OK,
            f"{venv} is owned by scripts/uv_workspace.py and partitioned for "
            "the canonical all-extras selection",
            evidence={
                "venv": str(venv),
                "marker": str(marker),
                "selection": loaded["selection"],
            },
        )

    return Check(
        "no-worktree-venv",
        FAIL,
        f"{venv} exists but is not a healthy uv-workspace-managed environment: "
        f"{reason}. An unmanaged worktree-local venv shadows the workspace "
        "environment and has produced ~167 phantom test failures that were read "
        "as real regressions.",
        remedy=(
            f"rm -rf {venv}; python3 scripts/uv_workspace.py sync --all-extras"
            if launcher.is_file()
            else f"rm -rf {venv}   # then use the workspace runner, never a local venv"
        ),
        evidence={"venv": str(venv), "marker": str(marker)},
    )


#: A real ``uv sync --all-extras`` of any repository in this workspace resolves
#: dozens to hundreds of transitive dependencies. D-CIP-19 found a live venv
#: silently collapsed to 3-6 packages (an unrelated "alpha" project's
#: environment had overwritten it) while ``no-worktree-venv`` above reported
#: OK, because that check only verifies OWNERSHIP markers (the selection
#: marker file, an executable interpreter) — none of which proves the
#: environment those markers describe still has real content. This floor is
#: deliberately low (never fails a legitimately small, dependency-light repo)
#: — it exists to catch "near-empty", not to police dependency budgets.
_MIN_PLAUSIBLE_PACKAGE_COUNT = 15


def _venv_site_packages(venv: Path) -> Path | None:
    lib = venv / "lib"
    if not lib.is_dir():
        return None
    for python_dir in sorted(lib.glob("python3.*")):
        sp = python_dir / "site-packages"
        if sp.is_dir():
            return sp
    return None


def _project_name(tree: Path) -> str | None:
    """This repo's own ``[project].name`` from ``pyproject.toml``, if any."""
    pyproject = tree / "pyproject.toml"
    if not pyproject.is_file():
        return None
    import tomllib

    try:
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        name = data.get("project", {}).get("name")
        if isinstance(name, str) and name:
            return name
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError, AttributeError):
        # Falling through to the regex fallback below is the INTENDED handling
        # for each of these: an unreadable or malformed pyproject (OSError /
        # UnicodeDecodeError / TOMLDecodeError), or a `[project]` key that is
        # not a table (AttributeError from `.get` on a non-dict). Narrowed from
        # a bare `except Exception: pass`, which also swallowed genuine bugs in
        # this function and tripped bandit B110.
        pass
    # Regex fallback: tolerate a pyproject.toml tomllib cannot parse rather than
    # silently skipping the ownership half of this check.
    m = re.search(
        r'(?m)^\s*name\s*=\s*"([^"]+)"',
        pyproject.read_text(encoding="utf-8", errors="replace"),
    )
    return m.group(1) if m else None


def _norm_pkg(s: str) -> str:
    return re.sub(r"[-_.]+", "-", s.strip().lower())


def _check_venv_package_count(tree: Path) -> Check:
    """D-CIP-19: make doctor FAIL on an implausibly low resolved-package count.

    Complements ``no-worktree-venv`` (provenance/markers) with a direct content
    check: how many distributions actually resolved, and whether any of them
    is this project's own. Both conditions independently reproduce the exact
    live incident (a 3-6 package venv, all belonging to an unrelated project
    named "alpha", reported OK by the marker-only check).
    """
    venv = tree / ".venv"
    python = venv / "bin" / "python"
    if not venv.is_dir() or not python.is_file():
        return Check(
            "venv-package-count",
            SKIP,
            "no usable venv to inspect (see no-worktree-venv)",
        )

    site_packages = _venv_site_packages(venv)
    if site_packages is None:
        return Check(
            "venv-package-count",
            FAIL,
            f"{venv} has an interpreter but no lib/python*/site-packages directory at all "
            "-- an empty or destroyed environment.",
            remedy=f"rm -rf {venv}; python3 scripts/uv_workspace.py sync --all-extras",
            evidence={"venv": str(venv)},
        )

    dist_names = sorted(
        p.name
        for p in site_packages.iterdir()
        if p.is_dir()
        and (p.name.endswith(".dist-info") or p.name.endswith(".egg-info"))
    )
    count = len(dist_names)

    if count < _MIN_PLAUSIBLE_PACKAGE_COUNT:
        return Check(
            "venv-package-count",
            FAIL,
            f"{site_packages} resolved only {count} package(s) -- implausibly low for a "
            "uv-managed workspace sync. D-CIP-19 found this exact shape live: 3-6 packages "
            "where ~726 were expected, because an unrelated project's environment had "
            "silently overwritten this worktree's own .venv. A healthy managed venv is "
            f"never this thin (floor: {_MIN_PLAUSIBLE_PACKAGE_COUNT}).",
            remedy=f"rm -rf {venv}; python3 scripts/uv_workspace.py sync --all-extras   "
            "# then print sys.executable and the resolved count before trusting any test run",
            evidence={
                "venv": str(venv),
                "resolved_count": count,
                "distributions": dist_names,
            },
        )

    project = _project_name(tree)
    if project:
        needle = _norm_pkg(project)
        belongs = any(needle in _norm_pkg(n) for n in dist_names)
        if not belongs:
            return Check(
                "venv-package-count",
                FAIL,
                f"{site_packages} resolved {count} package(s), but NONE of them is this "
                f"project's own ({project!r}) -- the D-CIP-19 shape where a DIFFERENT "
                "project's environment (there: one named 'alpha') silently overwrote this "
                f"worktree's .venv. Sample of what is actually installed: {dist_names[:5]}",
                remedy=f"rm -rf {venv}; python3 scripts/uv_workspace.py sync --all-extras",
                evidence={
                    "venv": str(venv),
                    "resolved_count": count,
                    "expected_project": project,
                    "sample_distributions": dist_names[:10],
                },
            )

    return Check(
        "venv-package-count",
        OK,
        f"{site_packages} resolved {count} package(s)",
        evidence={"venv": str(venv), "resolved_count": count},
    )


def _check_cargo_partition(tree: Path, env: dict[str, str]) -> Check:
    """A shared ``CARGO_TARGET_DIR`` both serializes AND corrupts concurrent builds."""
    exported = env.get("CARGO_TARGET_DIR", "")
    if not exported:
        return Check(
            "cargo-partition",
            OK,
            "CARGO_TARGET_DIR not exported — cargo uses this tree's own ./target",
            remedy="build with `cargo build --target-dir ./target-isolated` and prune it after",
        )
    try:
        inside = Path(exported).resolve().is_relative_to(tree)
    except (OSError, ValueError):  # pragma: no cover
        inside = False
    if inside:
        return Check(
            "cargo-partition",
            OK,
            f"CARGO_TARGET_DIR is partitioned to this tree ({exported})",
        )
    return Check(
        "cargo-partition",
        FAIL,
        f"CARGO_TARGET_DIR={exported} is OUTSIDE this worktree, so it is shared "
        "with every other lane. A shared cargo target dir does not merely "
        "serialize concurrent worktree builds — it CORRUPTS them.",
        remedy=(
            f"unset CARGO_TARGET_DIR; cargo build --target-dir {tree / 'target-isolated'}"
        ),
        evidence={"CARGO_TARGET_DIR": exported, "tree": str(tree)},
    )


def _check_precommit_home(tree: Path, env: dict[str, str]) -> Check:
    """PARTITION: the shared store is where unstaged work goes to die (D-OB-12)."""
    declared = env.get("PRE_COMMIT_HOME", "")
    temp_root = _lane_temp_root(tree)
    want = str(temp_root / "pre-commit") if temp_root else "<lane temp root>/pre-commit"
    orphan_hint = _orphaned_patch_paths(_SHARED_PRECOMMIT_HOME)
    if not declared:
        return Check(
            "precommit-home",
            FAIL,
            "PRE_COMMIT_HOME is unset, so pre-commit uses the single shared store "
            f"{_SHARED_PRECOMMIT_HOME}. pre-commit writes your UNSTAGED work to a "
            "patch file there, `git checkout`s it away, and restores it in a "
            "finally: — a crash inside that window loses it, and the shared "
            "SQLite db.db deadlocks under concurrent lanes.",
            remedy=f"export PRE_COMMIT_HOME={want}",
            evidence={"shared_store_patches": orphan_hint},
        )
    if Path(declared).resolve() == _SHARED_PRECOMMIT_HOME.resolve():
        return Check(
            "precommit-home",
            FAIL,
            f"PRE_COMMIT_HOME={declared} IS the shared default store — declaring it "
            "explicitly does not make it private.",
            remedy=f"export PRE_COMMIT_HOME={want}",
            evidence={"shared_store_patches": orphan_hint},
        )
    return Check(
        "precommit-home",
        OK,
        f"PRE_COMMIT_HOME is partitioned ({declared})",
        evidence={"shared_store_patches": orphan_hint},
    )


def _orphaned_patch_paths(store: Path) -> list[str]:
    """Patch files sitting in a pre-commit store.

    A patch file alone is **not** proof of a crash — pre-commit never deletes one
    even on success. It is reported as a path to `git apply` *if* work is missing,
    never as a verdict, because calling a successful run's leftover an incident
    is how a check earns the right to be ignored.
    """
    patch_dir = store / "patch1"
    if not patch_dir.is_dir():
        return []
    try:
        return sorted(str(p) for p in patch_dir.iterdir() if p.is_file())[:25]
    except OSError:  # pragma: no cover
        return []


def _check_stash_ref(tree: Path, scope: Any) -> Check:
    """``refs/stash`` is ONE ref shared by every worktree of the repository."""
    code, out = _git(["rev-parse", "--verify", "--quiet", "refs/stash"], tree)
    if code != 0 or not out:
        return Check(
            "shared-stash-ref",
            OK,
            "refs/stash is empty — no lane is holding shared stash state",
            remedy="read a pristine file with `git show HEAD:<path>`; park with a `wip:` commit",
        )
    lane = getattr(scope, "lane", "this lane")
    return Check(
        "shared-stash-ref",
        WARN,
        f"refs/stash EXISTS at {out}. It is a single ref shared by every worktree "
        "of this repository, so a `git stash pop` in any lane can consume another "
        "lane's entry. It may not be yours.",
        remedy=(
            "never `git stash`. Read pristine content with `git show HEAD:<path>`; "
            f"park work with `git commit -m 'wip: …'` or `agent-utilities lane park` "
            f"(writes refs/lane/{lane}/stash, no shared ref)"
        ),
        evidence={"refs/stash": out},
    )


def _check_pytest_basetemp(env: dict[str, str]) -> Check:
    """PARTITION: ~28 concurrent pytest runs on one basetemp skewed a baseline."""
    addopts = env.get("PYTEST_ADDOPTS", "")
    if "--basetemp" in addopts:
        return Check("pytest-basetemp", OK, f"pytest basetemp partitioned ({addopts})")
    return Check(
        "pytest-basetemp",
        WARN,
        "PYTEST_ADDOPTS declares no --basetemp, so this lane shares pytest's temp "
        "root. Concurrent lanes on one basetemp made a baseline measurably worse "
        "and nearly produced a false regression call.",
        remedy='eval "$(repository-manager --lane env --lane-path . --lane-shell)"',
    )


def _check_canonical_clean(scope: Any) -> Check:
    """A land is REFUSED against a dirty canonical tree — find out now, not at merge."""
    if scope is None:
        return Check("canonical-clean", SKIP, "lane scope unresolvable")
    canonical = Path(scope.main_tree)
    code, out = _git(["status", "--porcelain"], canonical)
    if code != 0:
        return Check(
            "canonical-clean", SKIP, f"cannot read {canonical}: {out}"
        )  # pragma: no cover
    if not out:
        return Check("canonical-clean", OK, f"canonical checkout {canonical} is clean")
    entries = out.splitlines()
    return Check(
        "canonical-clean",
        WARN,
        f"the canonical checkout {canonical} holds {len(entries)} uncommitted "
        "entries. Both the canonical guard and the merge queue's land step REFUSE "
        "a dirty canonical tree — including an UNTRACKED-only one — so this will "
        "block every lane's landing, not just yours.",
        remedy=(
            f"git -C {canonical} status --porcelain   # then commit, or move the "
            "files out; never `git checkout`/`clean` someone else's work away"
        ),
        evidence={"entries": entries[:25]},
    )


def _check_canonical_is_worktree(scope: Any) -> Check:
    """FAIL loudly if the canonical checkout has been corrupted into `core.bare`.

    D-MQR-5/D-MQR-6: ``core.bare=true`` was found set directly in the
    ``agent-utilities`` canonical checkout's ``.git/config`` -- files and every
    linked worktree stayed intact, but every git command run WITH CWD=canonical
    (``git status``, the merge queue's own driver, invoked via ``cd $REPO``)
    then failed with "fatal: this operation must be run in a work tree". That
    silently killed the repo's merge queue outright, and under the pre-D-MQR-2
    runner it collapsed into "queue depth 0" and was skipped forever with no log
    line at all.

    The reason nothing upstream of this check ever caught it: :func:`lane_scope`
    is normally called from a LANE'S OWN linked worktree, and `git worktree
    list`/`git rev-parse --git-common-dir` run just fine there even while the
    MAIN tree is bare -- so ``scope.main_tree`` resolves correctly and every
    other check in this module (which all operate on the calling lane's own
    tree) stays green. This check is the one that inspects the canonical
    checkout DIRECTLY, the only way the corruption is visible at all.

    Also note ``_check_canonical_clean``'s ``_git()`` helper only captures
    stdout, so the exact same "must be run in a work tree" fatal error there
    reads as ``SKIP "cannot read <canonical>: "`` (empty message) rather than a
    named failure -- this check exists precisely to not repeat that silent
    pattern, by capturing and surfacing stderr.
    """
    if scope is None:
        return Check(
            "canonical-is-worktree",
            SKIP,
            "lane scope unresolvable — cannot locate the canonical checkout",
            remedy="install agent-utilities so agent_utilities.governance.lanes imports",
        )
    canonical = Path(scope.main_tree)
    try:
        proc = subprocess.run(  # nosec B603 B607 - fixed argv, no shell
            ["git", "-C", str(canonical), "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return Check(
            "canonical-is-worktree",
            FAIL,
            f"could not run git against canonical checkout {canonical}: "
            f"{type(exc).__name__}: {exc}",
            remedy=f"inspect {canonical}/.git/config by hand",
            evidence={"canonical": str(canonical)},
        )
    stdout = proc.stdout.strip()
    if proc.returncode == 0 and stdout == "true":
        return Check(
            "canonical-is-worktree",
            OK,
            f"canonical checkout {canonical} is a real work tree",
        )
    return Check(
        "canonical-is-worktree",
        FAIL,
        f"canonical checkout {canonical} is NOT a work tree ("
        f"`git rev-parse --is-inside-work-tree` -> rc={proc.returncode} "
        f"stdout={stdout!r} stderr={proc.stderr.strip()!r}). Every git command "
        f'run with cwd={canonical} now fails with "this operation must be run '
        "in a work tree\", which silently kills this repo's merge queue "
        "(D-MQR-5). Most likely cause: core.bare=true got written into "
        "`.git/config`. RECOVERY: back up the config, flip the flag back, and "
        "verify with git status BEFORE touching any lane worktree — a `core."
        "bare` flip on the shared common dir has been observed to leave at "
        "least one lane worktree's INDEX (not working tree) looking wiped "
        "(files staged deleted while the same paths show untracked); a plain "
        "`git reset` (mixed) there is enough to restore it. Do NOT reach for "
        "`git reset --hard` or `git checkout .` in that worktree — those "
        "discard real uncommitted work instead of just re-reading the index.",
        remedy=(
            f"cp {canonical}/.git/config "
            f"{canonical}/.git/config.bak-$(date -u +%Y%m%dT%H%M%SZ) && "
            f"git -C {canonical} config core.bare false && "
            f"git -C {canonical} status --porcelain   # then, per worktree ONLY "
            "if its index looks wiped: git -C <worktree> status --porcelain && "
            "git -C <worktree> reset   # mixed reset, never --hard"
        ),
        evidence={
            "canonical": str(canonical),
            "returncode": proc.returncode,
            "stdout": stdout,
            "stderr": proc.stderr.strip(),
        },
    )


def _check_merge_queue_config(tree: Path, scope: Any) -> Check:
    """A repository that declares no gates is REFUSED by the queue, not defaulted."""
    roots = [tree]
    if scope is not None:
        roots.append(Path(scope.main_tree))
    for root in roots:
        declaration = root / ".mergequeue.yaml"
        if declaration.is_file():
            return Check(
                "merge-queue-config",
                OK,
                f"gates declared in {declaration}",
                evidence={"config": str(declaration)},
            )
    return Check(
        "merge-queue-config",
        WARN,
        "no .mergequeue.yaml in this repository. The queue REFUSES a repository "
        'that declares no gates rather than defaulting it — "declared no gates" '
        'and "has no queue configured" must not be the same value.',
        remedy=(
            "cp <repository-manager>/repository_manager/mergequeue_presets/"
            "<repo>.mergequeue.yaml ./.mergequeue.yaml && "
            "repository-manager --merge-queue config --repo-path ."
        ),
    )


def _check_base_drift(tree: Path, base: str) -> Check:
    """The branch tip is not the thing that lands — the MERGED tree is."""
    code, merge_base = _git(["merge-base", "HEAD", base], tree)
    if code != 0 or not merge_base:
        return Check(
            "base-drift",
            SKIP,
            f"cannot compute merge-base against {base}: {merge_base or 'no output'}",
        )
    code, base_sha = _git(["rev-parse", base], tree)
    if code != 0:  # pragma: no cover
        return Check("base-drift", SKIP, f"cannot resolve {base}")
    if base_sha == merge_base:
        return Check("base-drift", OK, f"this branch is current with {base}")
    code, count = _git(["rev-list", "--count", f"{merge_base}..{base_sha}"], tree)
    behind = count if code == 0 else "?"
    return Check(
        "base-drift",
        WARN,
        f"{base} has moved {behind} commits since this branch forked. Anything you "
        "measured on the branch TIP describes a tree that will never exist. Three "
        "people were misled by this in one day; one concluded a branch had deleted "
        "a guard that the merged tree in fact kept.",
        remedy=(
            f"git merge-tree --write-tree {base} HEAD   # measure THAT tree, "
            "or let the queue do it — it gates the merged tree by construction"
        ),
        evidence={"base": base, "base_sha": base_sha, "merge_base": merge_base},
    )


def _check_uncommitted(tree: Path) -> Check:
    """Commits are the only thing a working-tree reset cannot take."""
    code, out = _git(["status", "--porcelain"], tree)
    if code != 0:  # pragma: no cover
        return Check("committed-work", SKIP, f"cannot read status: {out}")
    if not out:
        return Check("committed-work", OK, "working tree clean — nothing at risk")
    entries = out.splitlines()
    return Check(
        "committed-work",
        WARN,
        f"{len(entries)} uncommitted entries. Only committed work survives a "
        "working-tree reset or a crash inside pre-commit's patch window.",
        remedy="git add -A && git commit   # never --no-verify",
        evidence={"entries": entries[:25]},
    )


def _check_test_runner(tree: Path, scope: Any) -> Check:
    """``uv run pytest`` silently runs the SYSTEM pytest. ~80 phantom verdicts."""
    roots = [tree]
    if scope is not None:
        roots.append(Path(scope.main_tree))
    for root in roots:
        runner = root / "scripts" / "uv_workspace.py"
        if runner.is_file():
            return Check(
                "test-runner",
                WARN,
                "this repository ships scripts/uv_workspace.py, which means plain "
                "`uv run pytest` is POISONED here: it silently resolves the SYSTEM "
                "interpreter and produced ~80 phantom failures that cited the "
                "project's own guards. Six lanes were burned before it was found.",
                remedy=(
                    "python3 scripts/uv_workspace.py run --all-extras -- pytest <args>"
                    "   # and print sys.executable: ~726 packages is correct, ~44 is stale"
                ),
                evidence={"runner": str(runner)},
            )
    return Check("test-runner", SKIP, "no scripts/uv_workspace.py in this repository")


def _check_precommit_hook_installed(tree: Path) -> Check:
    """D-ORC-45: THE ACTUAL DEFECT IS THE SILENCE, not the missing hook. Two
    lanes independently assumed ``git commit`` gated their work here and were
    wrong, because nothing anywhere told them otherwise -- gate coverage was
    only the merge queue's own gates plus whatever a lane happened to run by
    hand. This makes that absence (or presence) loud instead. Deliberately
    does NOT install a hook: ~20 lanes can be live at once, and a hook that
    starts stashing unstaged work into pre-commit's SHARED, unpartitioned
    store mid-wave is the D-OB-12 data-loss shape.
    """
    code, hooks_path_out = _git(["rev-parse", "--git-path", "hooks"], tree)
    hooks_dir = (
        (tree / hooks_path_out.strip()).resolve()
        if code == 0 and hooks_path_out.strip()
        else None
    )
    hook = (hooks_dir / "pre-commit") if hooks_dir else None
    installed = bool(hook and hook.is_file() and os.access(hook, os.X_OK))
    if installed:
        return Check(
            "precommit-hook",
            WARN,
            f"a pre-commit git hook IS installed at {hook} -- `git commit` in "
            "this tree runs it. Do not assume it is current: a hand-generated "
            "hook can hardcode a stale interpreter path (D-PCG-1) that crashes "
            "or silently no-ops rather than gating anything.",
            remedy=(
                "inspect the hook's INSTALL_PYTHON line against this lane's own "
                "interpreter; reinstall with `pre-commit install` if it looks stale"
            ),
            evidence={"hook": str(hook)},
        )
    return Check(
        "precommit-hook",
        WARN,
        "no pre-commit git hook is installed (core.hooksPath unset, "
        "<git-common-dir>/hooks/pre-commit absent) -- `git commit` in this "
        "tree runs ZERO hooks. Gate coverage is ONLY the merge queue's own "
        "gates plus whatever you ran by hand; do not assume a commit was "
        "gated.",
        remedy="run gates by hand before committing: `pre-commit run --files <changed>`",
        evidence={"hooks_dir": str(hooks_dir) if hooks_dir else ""},
    )


def _check_tree_repair(tree: Path) -> Check:
    """D-CDX-105/RMDD-26: detect structural index and ``core.bare`` drift."""
    try:
        from repository_manager import tree_repair

        report = tree_repair.diagnose(tree)
        finding = report.get("finding", "clean")
    except Exception as exc:  # pragma: no cover - unavailable optional runtime
        error = f"{type(exc).__name__}: {exc}"
        return Check(
            "tree-repair",
            FAIL,
            "tree-repair diagnosis unavailable; refusing to authorize a gate: " + error,
            remedy=(
                "rerun repository_manager.tree_repair.diagnose for path="
                f"{json.dumps(str(tree))} and repair only after a valid finding "
                "is returned"
            ),
            evidence={"path": str(tree), "error": error},
        )
    if finding == "clean":
        return Check(
            "tree-repair",
            OK,
            "working-tree index and core.bare structure are healthy",
            evidence=report.get("evidence", {}),
        )
    return Check(
        "tree-repair",
        FAIL,
        f"tree-repair detected {finding}; repair before running a gate",
        remedy=(
            "call repository_manager.tree_repair.repair with path="
            f"{json.dumps(str(tree))} and finding={json.dumps(str(finding))}"
        ),
        evidence=report.get("evidence", {}),
    )


def _unevaluable_env_check(name: str) -> Check:
    """An environment-derived check whose environment cannot be attributed.

    D-CDX-62, the companion to D-CDX-61. That fix let a remote caller PASS its
    own environment so the env-derived checks judge the right one — but left
    the default intact: omit ``env`` over the MCP surface and the checks still
    read the PROVIDER's ``os.environ`` and rule on it. On 2026-08-11 that gave
    a healthy lane a blocking ``precommit-home`` failure naming
    ``/home/rm/.cache/pre-commit``, a path belonging to the in-pod server that
    the lane neither set nor can write.

    An opt-in mitigation does not fix a failure mode whose trigger is
    *forgetting to opt in*. So when the environment cannot be attributed to the
    lane, these checks report SKIP. A gate that issues verdicts it cannot
    support teaches callers to ignore its verdicts, and then it protects
    nothing.
    """
    return Check(
        name,
        SKIP,
        "not evaluated: this process's environment is not the lane's. "
        "Environment-derived checks are only meaningful when the process "
        "asking is the process that will run the gate.",
        remedy=(
            "pass the lane's exported environment explicitly "
            "(`doctor(..., env=<exports>)`, obtainable from `lane env --path "
            "<tree>`), or run the doctor locally inside the lane"
        ),
    )


def diagnose(
    path: Path | str | None = None,
    *,
    base: str = "main",
    env: dict[str, str] | None = None,
    env_is_lane: bool = True,
) -> dict[str, Any]:
    """Run every lane-isolation check against a working tree. Mutates nothing.

    ``env`` defaults to the live process environment and exists so the checks are
    testable against a *deliberately bad* environment — a gate nobody has proven
    catches a known-bad input is not a gate.
    """
    tree = _resolve_tree(path)
    environ = dict(os.environ if env is None else env)
    scope = _lane_scope(tree)

    checks = [
        _check_not_canonical(tree, scope),
        _check_no_local_venv(tree),
        _check_venv_package_count(tree),
        *(
            [
                _check_cargo_partition(tree, environ),
                _check_precommit_home(tree, environ),
                _check_pytest_basetemp(environ),
            ]
            if env_is_lane
            else [
                _unevaluable_env_check("cargo-partition"),
                _unevaluable_env_check("precommit-home"),
                _unevaluable_env_check("pytest-basetemp"),
            ]
        ),
        _check_stash_ref(tree, scope),
        _check_test_runner(tree, scope),
        _check_precommit_hook_installed(tree),
        _check_canonical_clean(scope),
        _check_canonical_is_worktree(scope),
        _check_tree_repair(tree),
        _check_merge_queue_config(tree, scope),
        _check_base_drift(tree, base),
        _check_uncommitted(tree),
    ]
    blocking = [c for c in checks if c.blocking]
    return {
        "ok": not blocking,
        "tree": str(tree),
        "lane": getattr(scope, "lane", ""),
        "base": base,
        "blocking": [c.name for c in blocking],
        "warnings": [c.name for c in checks if c.status == WARN],
        "checks": [c.as_dict() for c in checks],
    }


# ---------------------------------------------------------------------------
# start / finish
# ---------------------------------------------------------------------------
def _load_merge_queue() -> Any:
    """Load the optional universal queue at the lane handoff boundary."""
    from repository_manager import merge_queue

    return merge_queue


def start(
    repo: str,
    branch: str,
    *,
    base: str = "main",
    path: str | None = None,
    git: Any = None,
    registry: Any = None,
    owner_id: str = "",
    session_id: str = "",
    host_id: str = "",
    request_id: str | None = None,
    idempotency_key: str | None = None,
    predicted_disk_bytes: int = 0,
    disk_budget_bytes: int | None = None,
    ttl_seconds: int = 3600,
    adopt: bool = False,
    operator_id: str = "",
) -> dict[str, Any]:
    """Open an isolated lane: worktree + partitioned environment + proof.

    The proof is the point. Creating the worktree is one line of git; what
    actually goes wrong is a lane that *believes* it is isolated while sharing a
    cargo target dir, a pre-commit store, or a pytest basetemp with thirteen
    siblings. This returns the checks alongside the exports so the lane can see
    its isolation rather than assume it.
    """
    from repository_manager.worktree import WorktreeManager

    if git is None:
        from repository_manager.mcp_server import get_git_instance

        git = get_git_instance(path=path)
    manager = WorktreeManager(git, registry=registry)
    if registry is not None:
        if not owner_id or not session_id:
            return {
                "ok": False,
                "stage": "registry",
                "error": "managed lane start requires owner_id and session_id",
            }
        added = manager.allocate(
            repo,
            branch,
            base=base,
            owner_id=owner_id,
            session_id=session_id,
            host_id=host_id,
            request_id=request_id,
            idempotency_key=idempotency_key,
            predicted_disk_bytes=predicted_disk_bytes,
            disk_budget_bytes=disk_budget_bytes,
            ttl_seconds=ttl_seconds,
            adopt=adopt,
            operator_id=operator_id,
            registry=registry,
        )
    else:
        added = manager.add(repo, branch, base=base, adopt=adopt)
    if not added.get("ok", False):
        return {"ok": False, "stage": "worktree", "result": added}

    tree = Path(added.get("path", ""))
    exports = lane_exports(tree)
    report = diagnose(tree, base=base, env={**os.environ, **exports})
    return {
        "ok": report["ok"],
        "stage": "start",
        "worktree": str(tree),
        "branch": branch,
        "base": base,
        "exports": exports,
        "shell": _shell_block(exports),
        "preflight": report,
        **(
            {
                "lane_id": added.get("lane_id"),
                "fence": added.get("fence"),
                "record": added.get("record"),
            }
            if registry is not None
            else {}
        ),
    }


def allocate(
    repo: str,
    branch: str,
    **kwargs: Any,
) -> dict[str, Any]:
    """Bounded lane-doctor adapter for durable allocation."""

    return start(repo, branch, registry=kwargs.pop("registry", None), **kwargs)


def _registry_lane_for_path(registry: Any, path: Path, branch: str = "") -> Any:
    for record in registry.list_records():
        if Path(record.worktree_path) == path and (
            not branch or record.branch == branch
        ):
            return record
    return None


def heartbeat(
    lane_id: str,
    *,
    owner_id: str,
    fence: str,
    registry: Any,
    observed_disk_bytes: int | None = None,
    now: Any | None = None,
) -> dict[str, Any]:
    """Refresh a managed lane heartbeat through its current fence."""

    try:
        record = registry.heartbeat(
            lane_id,
            owner_id=owner_id,
            fence=fence,
            observed_disk_bytes=observed_disk_bytes,
            now=now,
        )
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    return {
        "ok": True,
        "lane_id": lane_id,
        "fence": record.fence,
        "record": record.model_dump(mode="json"),
    }


def status(
    lane_id: str | None = None,
    *,
    path: Path | str | None = None,
    branch: str = "",
    registry: Any,
) -> dict[str, Any]:
    """Read durable lane status without touching Git or the worktree."""

    try:
        record = registry.require(lane_id) if lane_id else None
        if record is None and path is not None:
            record = _registry_lane_for_path(registry, _resolve_tree(path), branch)
        if record is None:
            return {"ok": False, "error": "lane could not be resolved"}
    except Exception as exc:
        return {"ok": False, "error": str(exc), "error_type": type(exc).__name__}
    return {
        "ok": True,
        "lane_id": record.lane_id,
        "record": record.model_dump(mode="json"),
    }


def finish(
    path: Path | str | None = None,
    *,
    branch: str = "",
    base: str = "",
    force: bool = False,
    registry: Any = None,
    lane_id: str | None = None,
    owner_id: str = "",
    fence: str = "",
) -> dict[str, Any]:
    """Preflight, then hand the branch to the serialized merge queue.

    Refusing to enqueue a lane that fails its own preflight is not pedantry: the
    queue gates a candidate against a freshly computed baseline, which is the
    most expensive thing in the system. Spending that to rediscover an unset
    ``PRE_COMMIT_HOME`` is the wrong order of operations.

    ``force`` exists for the one honest case — a blocking check the lane has
    read and consciously accepted — and records itself in the result so the
    decision is visible afterwards rather than invisible.
    """
    tree = _resolve_tree(path)
    report = diagnose(tree, base=base or "main")
    if not report["ok"] and not force:
        return {
            "ok": False,
            "stage": "preflight",
            "enqueued": False,
            "reason": "blocking preflight checks; fix them or pass force",
            "preflight": report,
        }

    managed_record = None
    if registry is not None:
        if not owner_id or not fence:
            return {
                "ok": False,
                "stage": "registry",
                "enqueued": False,
                "reason": "managed lane finish requires explicit owner_id and fence",
                "preflight": report,
            }
        managed_record = (
            registry.require(lane_id)
            if lane_id
            else _registry_lane_for_path(registry, tree, branch)
        )
        if managed_record is None:
            return {
                "ok": False,
                "stage": "registry",
                "enqueued": False,
                "reason": "managed lane record could not be resolved",
                "preflight": report,
            }
        try:
            submitted_lane = registry.submit(
                managed_record.lane_id,
                owner_id=owner_id,
                fence=fence,
            )
        except Exception as exc:
            return {
                "ok": False,
                "stage": "registry",
                "enqueued": False,
                "reason": str(exc),
                "preflight": report,
            }
    else:
        submitted_lane = None

    try:
        merge_queue = _load_merge_queue()
    except ImportError as exc:
        return {
            "ok": False,
            "stage": "enqueue",
            "enqueued": False,
            "reason": (
                "the universal merge queue is not available in this build: "
                f"{exc}. Enqueue through the owning repository's own queue "
                "instead — do not hand-merge into the shared base."
            ),
            "preflight": report,
            "forced": force,
            "lane_id": managed_record.lane_id if managed_record else None,
        }

    try:
        result = merge_queue.enqueue(branch or "", base=base or "", path=tree)
    except merge_queue.MergeQueueError as exc:
        return {
            "ok": False,
            "stage": "enqueue",
            "enqueued": False,
            "reason": str(exc),
            "preflight": report,
            "forced": force,
        }
    return {
        "ok": bool(result.get("ok", True)),
        "stage": "enqueue",
        "enqueued": True,
        "candidate": result,
        "preflight": report,
        "forced": force,
        "lane_id": submitted_lane.lane_id if submitted_lane else None,
        "lane_state": submitted_lane.state.value if submitted_lane else None,
        "note": (
            "enqueued != landed, but you do not need to do anything: a scheduler "
            "drains this queue automatically. Watch it with "
            "`repository-manager --merge-queue status --repo-path .`."
        ),
    }


ACTIONS = ("doctor", "start", "finish", "env")


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core, shared by the CLI and the MCP tool so they cannot drift.

    D-CDX-61 — ``doctor`` accepts an explicit ``env`` (the CALLER's own
    exported environment) precisely because a remote caller's environment is
    not the dispatching process's. Local CLI use needs nothing here: when
    ``env`` is omitted, :func:`diagnose` still defaults to the live
    ``os.environ`` of whichever process is running — correct exactly when the
    caller and the resolving process are the same one. It stops being correct
    the moment ``doctor`` is reached over the MCP surface: repository-manager-
    mcp's ``os.environ`` is the PROVIDER's process environment, not the lane
    process that actually exported ``PRE_COMMIT_HOME``/``CARGO_TARGET_DIR`` —
    reading server globals there manufactured a blocker for a healthy lane
    (PRE_COMMIT_HOME reported as the shared default because the provider
    process never had the caller's export at all) that the source-backed
    local doctor never saw for the identical tree. Passing ``env`` closes
    that: the MCP tool accepts the caller's declared environment and
    evaluates the checks against it, the same evidence the local doctor uses.
    """
    if action == "doctor":
        # D-CDX-62 — when the caller supplied no ``env``, this process's
        # environment cannot be attributed to the lane, so the env-derived
        # checks report SKIP instead of ruling on a foreign environment.
        # ``env_is_lane`` is overridable for the local CLI, where the caller
        # IS this process and ``os.environ`` is exactly right.
        env = kwargs.get("env")
        return diagnose(
            kwargs.get("path"),
            base=kwargs.get("base") or "main",
            env=env,
            env_is_lane=bool(kwargs.get("env_is_lane", env is not None)),
        )
    if action == "env":
        exports = lane_exports(kwargs.get("path"))
        return {"ok": True, "exports": exports, "shell": _shell_block(exports)}
    if action == "start":
        repo = kwargs.get("repo") or ""
        branch = kwargs.get("branch") or ""
        if not repo or not branch:
            return {"ok": False, "error": "start requires both repo and branch"}
        return start(
            repo,
            branch,
            base=kwargs.get("base") or "main",
            path=kwargs.get("workspace"),
            registry=kwargs.get("registry"),
            owner_id=kwargs.get("owner_id") or "",
            session_id=kwargs.get("session_id") or "",
            host_id=kwargs.get("host_id") or "",
            request_id=kwargs.get("request_id"),
            idempotency_key=kwargs.get("idempotency_key"),
            predicted_disk_bytes=int(kwargs.get("predicted_disk_bytes") or 0),
            disk_budget_bytes=kwargs.get("disk_budget_bytes"),
            ttl_seconds=int(kwargs.get("ttl_seconds") or 3600),
            adopt=bool(kwargs.get("adopt")),
            operator_id=kwargs.get("operator_id") or "",
        )
    if action == "finish":
        return finish(
            kwargs.get("path"),
            branch=kwargs.get("branch") or "",
            base=kwargs.get("base") or "",
            force=bool(kwargs.get("force")),
            registry=kwargs.get("registry"),
            lane_id=kwargs.get("lane_id"),
            owner_id=kwargs.get("owner_id") or "",
            fence=kwargs.get("fence") or "",
        )
    return {"ok": False, "error": f"unknown action: {action}"}


def main(argv: list[str] | None = None) -> int:
    """``python -m repository_manager.lane_doctor`` — no CLI package import.

    Deliberately standalone so a lane can run its preflight in an environment
    where the full repository-manager CLI (and its dependency graph) will not
    import — which is exactly the environment a doctor is for.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m repository_manager.lane_doctor")
    parser.add_argument("action", choices=ACTIONS)
    parser.add_argument("--path", default=None, help="the lane's working tree")
    parser.add_argument("--repo", default="", help="repo for `start`")
    parser.add_argument("--branch", default="", help="branch for `start`/`finish`")
    parser.add_argument("--base", default="", help="base branch (default main)")
    parser.add_argument("--workspace", default=None, help="workspace root override")
    parser.add_argument("--force", action="store_true", help="enqueue despite failures")
    parser.add_argument("--shell", action="store_true", help="print exports only")
    args = parser.parse_args(argv)

    result = dispatch(
        args.action,
        path=args.path,
        repo=args.repo,
        branch=args.branch,
        base=args.base,
        workspace=args.workspace,
        force=args.force,
        # D-CDX-62 — the local CLI IS the lane's process, so its own
        # os.environ is exactly the environment the gate will run under.
        # This is the one caller that can honestly assert that.
        env_is_lane=True,
    )
    if args.shell and "shell" in result:
        print(result["shell"])
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
