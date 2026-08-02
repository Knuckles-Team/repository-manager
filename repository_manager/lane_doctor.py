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

import json
import os
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
    """This lane's private temp root (``~/.al/<token>``), or ``None``."""
    try:
        from agent_utilities.governance import lanes

        return lanes.partitioned_paths(tree).scratch_dir.parent
    except Exception:  # pragma: no cover - environment-dependent
        return None


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
    """
    tree = _resolve_tree(path)
    exports: dict[str, str] = {}
    try:
        from agent_utilities.governance import lanes

        parts = lanes.partitioned_paths(tree)
    except Exception:  # pragma: no cover - environment-dependent
        return exports

    temp_root = parts.scratch_dir.parent
    precommit_home = temp_root / "pre-commit"
    precommit_home.mkdir(parents=True, exist_ok=True)

    exports["CARGO_TARGET_DIR"] = str(parts.cargo_target_dir)
    exports["TMPDIR"] = str(parts.scratch_dir)
    exports["PYTEST_ADDOPTS"] = f"--basetemp={parts.pytest_basetemp}"
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
        elif not reason and not python.is_file():
            reason = f"the environment interpreter is absent ({python})"
        elif not reason and not os.access(python, os.X_OK):
            reason = f"the environment interpreter is not executable ({python})"

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


def diagnose(
    path: Path | str | None = None,
    *,
    base: str = "main",
    env: dict[str, str] | None = None,
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
        _check_cargo_partition(tree, environ),
        _check_precommit_home(tree, environ),
        _check_pytest_basetemp(environ),
        _check_stash_ref(tree, scope),
        _check_test_runner(tree, scope),
        _check_canonical_clean(scope),
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
    manager = WorktreeManager(git)
    added = manager.add(repo, branch, base=base)
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
    }


def finish(
    path: Path | str | None = None,
    *,
    branch: str = "",
    base: str = "",
    force: bool = False,
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
        "note": (
            "enqueued != landed, but you do not need to do anything: a scheduler "
            "drains this queue automatically. Watch it with "
            "`repository-manager --merge-queue status --repo-path .`."
        ),
    }


ACTIONS = ("doctor", "start", "finish", "env")


def dispatch(action: str, **kwargs: Any) -> dict[str, Any]:
    """One action core, shared by the CLI and the MCP tool so they cannot drift."""
    if action == "doctor":
        return diagnose(
            kwargs.get("path"),
            base=kwargs.get("base") or "main",
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
        )
    if action == "finish":
        return finish(
            kwargs.get("path"),
            branch=kwargs.get("branch") or "",
            base=kwargs.get("base") or "",
            force=bool(kwargs.get("force")),
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
    )
    if args.shell and "shell" in result:
        print(result["shell"])
    else:
        print(json.dumps(result, indent=2, default=str))
    return 0 if result.get("ok", False) else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
