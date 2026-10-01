"""Commit an explicitly reviewed set of paths before running its gate.

``pre-commit`` has a ``staged_files_only`` context that temporarily removes
unstaged changes while hooks run.  That context is useful for an ordinary
staged-only commit, but it is a data-loss boundary when a process is killed
inside it.  :func:`safe_commit` makes the boundary unreachable without ever
staging the whole tree: the caller passes ``paths``, an explicit, reviewed
allowlist (``git add -A``/``git add .`` equivalents are refused by
construction -- there is no "stage everything" mode).  Given that allowlist,
it stages exactly those paths (including a deletion or a new untracked file
*at one of those paths*), proves no content remains unstaged *within that
allowlist*, runs the configured gate, re-stages the same allowlist in case
the gate's formatter touched it, proves the invariant again, and only then
commits.  A path outside the allowlist -- tracked, untracked, or deleted --
is never staged and is not inspected by the "nothing left unstaged" proof;
an unrelated untracked file in the tree stays uncommitted.  Calling without
``paths`` (or with an empty list) is a typed refusal, not an implicit
whole-tree stage.

Callers that must create a WIP snapshot before a heavy gate is admitted may
pass ``defer_gate=True``.  That mode stages and verifies the same explicit
``paths``, commits with ``--no-verify``, and returns ``gate_deferred=True``.
It confers no validation evidence; the caller must submit the real gate
through the common scheduler/executor against the returned immutable SHA.

CONCEPT:RM-SAFE-COMMIT (C-12)
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - fixed-argv git/gate execution is this module's job
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from repository_manager import tree_repair

__all__ = ["safe_commit"]


def _lane_name(path: Path) -> str:
    """Resolve a lane name without making Agent Utilities a hard import."""
    try:
        from repository_manager.governance.lanes import lane_name

        return str(lane_name(path))
    except Exception:  # pragma: no cover - optional dependency/fake trees
        return path.name or "local"


def _run(
    argv: Sequence[str],
    path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> subprocess.CompletedProcess[str]:
    """Run one fixed-argv command and retain bounded text output."""
    try:
        return subprocess.run(
            list(argv),
            cwd=str(path),
            capture_output=True,
            text=True,
            check=False,
            env=env,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return subprocess.CompletedProcess(
            list(argv), 1, "", f"{type(exc).__name__}: {exc}"
        )


def _detail(result: subprocess.CompletedProcess[str]) -> str:
    return (result.stderr or result.stdout or "").strip()


def _names(result: subprocess.CompletedProcess[str]) -> list[str]:
    """Decode ``git diff --name-only -z`` without losing odd filenames."""
    if result.returncode != 0 or not result.stdout:
        return []
    return [name for name in result.stdout.split("\0") if name]


def _addable_paths(
    tree: Path, paths: Sequence[str], *, env: dict[str, str], timeout: int
) -> list[str]:
    """Which of *paths* a ``git add`` call can act on right now.

    A path git add can match is one still present on disk (tracked or not) or
    still tracked in the index.  A path that is neither -- an already-staged
    deletion, re-checked after the gate ran -- has nothing left to add;
    passing it through anyway would be an explicit pathspec that matches
    nothing, and ``git add`` aborts the ENTIRE call (unlike ``-A`` with no
    pathspec) rather than skipping just that one path.
    """
    on_disk = [p for p in paths if (tree / p).exists()]
    missing = [p for p in paths if p not in on_disk]
    if not missing:
        return list(paths)
    tracked = _run(
        ["git", "ls-files", "-z", "--", *missing], tree, env=env, timeout=timeout
    )
    tracked_names = set(_names(tracked)) if tracked.returncode == 0 else set()
    return [p for p in paths if p in on_disk or p in tracked_names]


def _unstaged_paths_within(path: Path, paths: Sequence[str]) -> list[str]:
    """Tracked-but-unstaged or untracked paths, scoped to the reviewed *paths*.

    A path outside this allowlist is never inspected: it is not this call's
    business whether it is dirty, and it must never be swept into the proof.
    """
    tracked = _run(["git", "diff", "--name-only", "-z", "--", *paths], path)
    untracked = _run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *paths],
        path,
    )
    if tracked.returncode != 0 or untracked.returncode != 0:
        return ["<unable-to-inspect-unstaged-tree>"]
    return _names(tracked) + _names(untracked)


def _staged_paths(path: Path) -> list[str]:
    result = _run(["git", "diff", "--cached", "--name-only", "-z"], path)
    return _names(result)


def _result(
    *,
    path: Path,
    lane: str,
    status: str,
    staged_paths: list[str],
    gate_stage: str,
    gate_invoked: bool,
    gate_deferred: bool = False,
    commit_sha: str | None = None,
    nothing_left_unstaged: bool = False,
    error: str | None = None,
    baseline: dict[str, Any] | None = None,
    baseline_recorded: bool = False,
    baseline_error: str | None = None,
) -> dict[str, Any]:
    """Build the stable C-12 result shape."""
    return {
        "ok": status == "success",
        "status": status,
        "repository": str(path),
        "lane": lane,
        "staged_paths": staged_paths,
        "gate_stage": gate_stage,
        "gate_invoked": gate_invoked,
        "gate_deferred": gate_deferred,
        "commit_sha": commit_sha,
        "nothing_left_unstaged": nothing_left_unstaged,
        "error": error,
        "baseline": baseline,
        "baseline_recorded": baseline_recorded,
        "baseline_error": baseline_error,
    }


def _run_gate(
    gate: Sequence[str] | Callable[[Path], Any],
    path: Path,
    *,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> tuple[bool, str]:
    """Run a configured gate, accepting a small testable callable seam."""
    if callable(gate):
        try:
            outcome = gate(path)
        except Exception as exc:  # pragma: no cover - caller-provided gate
            return False, f"gate raised {type(exc).__name__}: {exc}"
        if isinstance(outcome, bool):
            return outcome, ""
        if isinstance(outcome, dict):
            return bool(outcome.get("ok", False)), str(outcome.get("error", ""))
        return bool(outcome), ""
    result = _run(gate, path, env=env, timeout=timeout)
    return result.returncode == 0, _detail(result)


@dataclass
class _AbortDetail:
    """Fields ``_result`` needs to render one ``_CommitAbort`` as a response.

    Most phases leave ``gate_deferred``/``nothing_left_unstaged`` at
    ``_result``'s own default of ``False`` rather than the caller's actual
    ``defer_gate``, mirroring the module's original inline returns verbatim.
    """

    status: str = "error"
    staged_paths: list[str] = field(default_factory=list)
    gate_stage: str = "none"
    gate_invoked: bool = False
    gate_deferred: bool = False
    nothing_left_unstaged: bool = False


class _CommitAbort(Exception):
    """Internal control-flow signal: one phase of ``_safe_commit_run`` failed."""

    def __init__(self, error: str, detail: _AbortDetail | None = None) -> None:
        super().__init__(error)
        self.error = error
        self.detail = detail if detail is not None else _AbortDetail()


@dataclass
class _CommitConfig:
    """Bundled per-call parameters threaded through the commit phases."""

    paths: tuple[str, ...] = ()
    allow_empty: bool = False
    gate: Sequence[str] | Callable[[Path], Any] | None = None
    defer_gate: bool = False
    command_env: dict[str, str] = field(default_factory=dict)
    timeout: int = 1800


def _early_refusal(
    tree: Path,
    lane: str,
    paths: Sequence[str],
    gate: Sequence[str] | Callable[[Path], Any] | None,
    defer_gate: bool,
) -> dict[str, Any] | None:
    """A typed refusal before any lease or git state is touched, or ``None``."""
    if not paths:
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            error=(
                "safe_commit requires an explicit, non-empty list of reviewed "
                "paths to stage; refusing to stage the whole working tree"
            ),
        )
    if defer_gate and gate is not None:
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            gate_deferred=True,
            error="defer_gate cannot be combined with an explicit gate",
        )
    return None


def _safe_commit_locked(
    path: Path | str,
    message: str,
    *,
    paths: Sequence[str] = (),
    allow_empty: bool = False,
    gate: Sequence[str] | Callable[[Path], Any] | None = None,
    defer_gate: bool = False,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Stage an explicit, reviewed path list, gate, and commit it.

    Args:
        path: A repository worktree.  It is never interpreted as a shell value.
        message: Commit message passed as one argv element.
        paths: The explicit, reviewed allowlist of repository-relative paths to
            stage (a tracked change, a deletion, or a new untracked file *at one
            of these paths*).  Required and non-empty: there is no "stage
            everything" mode, and an empty/omitted allowlist is refused rather
            than silently staging the whole tree.
        allow_empty: Permit an empty commit when the given paths have no changes.
        gate: Optional fixed-argv gate or callable.  By default a repository
            with ``.pre-commit-config.yaml`` runs ``pre-commit run --all-files``;
            repositories without that file have no gate to invoke.
        defer_gate: Stage and commit the same explicit ``paths`` without
            invoking any repository hook.  The commit is made with
            ``--no-verify`` and the result explicitly records
            ``gate_deferred=True``; a caller must run the real gate through its
            admitted executor afterwards.
        env: Optional environment for the gate and git commands.
        timeout: Per-command timeout in seconds.

    The returned ``nothing_left_unstaged`` is an assertion, scoped to ``paths``,
    made immediately before the gate and again before commit -- not an
    inference from a green gate, and never a claim about the rest of the tree.
    """
    tree = Path(path).expanduser().resolve()
    lane = _lane_name(tree)
    refusal = _early_refusal(tree, lane, paths, gate, defer_gate)
    if refusal is not None:
        return refusal
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    config = _CommitConfig(
        paths=tuple(paths),
        allow_empty=allow_empty,
        gate=gate,
        defer_gate=defer_gate,
        command_env=command_env,
        timeout=timeout,
    )
    try:
        return _safe_commit_run(tree, lane, message, config)
    except _CommitAbort as exc:
        d = exc.detail
        return _result(
            path=tree,
            lane=lane,
            status=d.status,
            staged_paths=d.staged_paths,
            gate_stage=d.gate_stage,
            gate_invoked=d.gate_invoked,
            gate_deferred=d.gate_deferred,
            nothing_left_unstaged=d.nothing_left_unstaged,
            error=exc.error,
        )


def _check_tree_exists(tree: Path) -> None:
    if not tree.is_dir():
        raise _CommitAbort(f"repository path does not exist: {tree}")


def _initial_status_or_skip(tree: Path, config: _CommitConfig) -> None:
    """Raise ``skipped`` when the given paths are clean, ``error`` on failure."""
    initial = _run(
        ["git", "status", "--porcelain", "-z", "--", *config.paths],
        tree,
        env=config.command_env,
        timeout=config.timeout,
    )
    if initial.returncode != 0:
        raise _CommitAbort(_detail(initial) or "git status failed")
    if not initial.stdout and not config.allow_empty:
        raise _CommitAbort(
            "no changes to commit",
            _AbortDetail(status="skipped", nothing_left_unstaged=True),
        )


def _stage_paths(tree: Path, config: _CommitConfig) -> None:
    """Stage exactly the reviewed ``config.paths`` allowlist -- never ``-A``."""
    addable = _addable_paths(
        tree, config.paths, env=config.command_env, timeout=config.timeout
    )
    if not addable:
        return
    staged = _run(
        ["git", "add", "--", *addable],
        tree,
        env=config.command_env,
        timeout=config.timeout,
    )
    if staged.returncode != 0:
        raise _CommitAbort(_detail(staged) or "git add failed")


def _verify_nothing_unstaged(
    tree: Path,
    staged_paths: list[str],
    *,
    gate_stage: str,
    gate_invoked: bool,
    prefix: str,
    scoped_paths: Sequence[str],
) -> None:
    unstaged = _unstaged_paths_within(tree, scoped_paths)
    if unstaged:
        raise _CommitAbort(
            f"{prefix}: " + ", ".join(unstaged[:20]),
            _AbortDetail(
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=gate_invoked,
            ),
        )


def _resolve_gate(
    tree: Path,
    gate: Sequence[str] | Callable[[Path], Any] | None,
    defer_gate: bool,
) -> tuple[str, Sequence[str] | Callable[[Path], Any] | None]:
    gate_stage = "deferred" if defer_gate else "none"
    configured_gate: Sequence[str] | Callable[[Path], Any] | None = (
        None if defer_gate else gate
    )
    if (
        not defer_gate
        and configured_gate is None
        and (tree / ".pre-commit-config.yaml").is_file()
    ):
        configured_gate = ["pre-commit", "run", "--all-files"]
        gate_stage = "pre-commit"
    elif configured_gate is not None:
        gate_stage = "configured"
    return gate_stage, configured_gate


def _run_configured_gate(
    configured_gate: Sequence[str] | Callable[[Path], Any],
    tree: Path,
    config: _CommitConfig,
    staged_paths: list[str],
    gate_stage: str,
) -> None:
    passed, detail = _run_gate(
        configured_gate, tree, env=config.command_env, timeout=config.timeout
    )
    if not passed:
        raise _CommitAbort(
            detail or "configured gate failed",
            _AbortDetail(
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=True,
                nothing_left_unstaged=True,
            ),
        )


def _stage_paths_after_gate(
    tree: Path,
    config: _CommitConfig,
    staged_paths: list[str],
    gate_stage: str,
) -> None:
    # A formatter may have changed the reviewed paths during the gate.  Fold
    # that output into the same snapshot (still scoped to the allowlist, never
    # "-A") and prove the invariant again.  A path already fully staged as a
    # deletion (gone from both disk and the index) has nothing left to add.
    addable = _addable_paths(
        tree, config.paths, env=config.command_env, timeout=config.timeout
    )
    if not addable:
        return
    restaged = _run(
        ["git", "add", "--", *addable],
        tree,
        env=config.command_env,
        timeout=config.timeout,
    )
    if restaged.returncode != 0:
        raise _CommitAbort(
            _detail(restaged) or "git add after gate failed",
            _AbortDetail(
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=True,
                nothing_left_unstaged=False,
            ),
        )


def _commit(
    tree: Path,
    message: str,
    config: _CommitConfig,
    staged_paths: list[str],
    gate_stage: str,
    gate_invoked: bool,
) -> str | None:
    commit_argv = ["git", "commit"]
    if config.defer_gate:
        commit_argv.append("--no-verify")
    if config.allow_empty:
        commit_argv.append("--allow-empty")
    commit_argv.extend(["-m", message])
    committed = _run(commit_argv, tree, env=config.command_env, timeout=config.timeout)
    if committed.returncode != 0:
        raise _CommitAbort(
            _detail(committed) or "git commit failed",
            _AbortDetail(
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=gate_invoked,
                gate_deferred=config.defer_gate,
                nothing_left_unstaged=True,
            ),
        )
    sha_result = _run(
        ["git", "rev-parse", "HEAD"],
        tree,
        env=config.command_env,
        timeout=config.timeout,
    )
    return sha_result.stdout.strip() if sha_result.returncode == 0 else None


def _record_baseline(tree: Path) -> tuple[dict[str, Any], bool, str | None]:
    try:
        baseline = tree_repair.record_baseline(tree)
    except Exception as exc:  # pragma: no cover - defensive persistence seam
        baseline = {
            "ok": False,
            "finding": "unavailable",
            "path": str(tree),
            "error": f"baseline recording raised {type(exc).__name__}: {exc}",
        }
    baseline_recorded = bool(baseline.get("ok") and baseline.get("persisted"))
    baseline_error = (
        None
        if baseline_recorded
        else str(
            baseline.get("error")
            or baseline.get("persistence_error")
            or "baseline persistence was not confirmed"
        )
    )
    return baseline, baseline_recorded, baseline_error


def _safe_commit_run(
    tree: Path, lane: str, message: str, config: _CommitConfig
) -> dict[str, Any]:
    """Run every ``_safe_commit_locked`` phase; raises ``_CommitAbort`` on error."""
    _check_tree_exists(tree)
    _initial_status_or_skip(tree, config)
    _stage_paths(tree, config)
    staged_paths = _staged_paths(tree)
    _verify_nothing_unstaged(
        tree,
        staged_paths,
        gate_stage="none",
        gate_invoked=False,
        prefix="git add left a reviewed path unstaged",
        scoped_paths=config.paths,
    )

    gate_stage, configured_gate = _resolve_gate(tree, config.gate, config.defer_gate)
    if configured_gate is not None:
        _run_configured_gate(configured_gate, tree, config, staged_paths, gate_stage)
        _stage_paths_after_gate(tree, config, staged_paths, gate_stage)
        staged_paths = _staged_paths(tree)
        _verify_nothing_unstaged(
            tree,
            staged_paths,
            gate_stage=gate_stage,
            gate_invoked=True,
            prefix="gate left a reviewed path unstaged",
            scoped_paths=config.paths,
        )

    gate_invoked = configured_gate is not None
    sha = _commit(tree, message, config, staged_paths, gate_stage, gate_invoked)
    baseline, baseline_recorded, baseline_error = _record_baseline(tree)
    return _result(
        path=tree,
        lane=lane,
        status="success",
        staged_paths=staged_paths,
        gate_stage=gate_stage,
        gate_invoked=gate_invoked,
        gate_deferred=config.defer_gate,
        commit_sha=sha,
        nothing_left_unstaged=True,
        baseline=baseline,
        baseline_recorded=baseline_recorded,
        baseline_error=baseline_error,
    )


def safe_commit(
    path: Path | str,
    message: str,
    *,
    paths: Sequence[str] = (),
    allow_empty: bool = False,
    gate: Sequence[str] | Callable[[Path], Any] | None = None,
    defer_gate: bool = False,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Commit an explicit, reviewed path list under the per-worktree mutation lease.

    The lease spans status, staging the ``paths`` allowlist, the configured
    gate (or an explicitly deferred snapshot), commit, and baseline refresh.
    It is deliberately per-worktree, so independent lanes continue to run
    concurrently while same-tree callers cannot interleave a check with
    another mutation.  ``paths`` is required and non-empty; omitting it is a
    typed refusal from :func:`_safe_commit_locked` once the lease is held,
    same as any other early-refusal condition.
    """
    tree = Path(path).expanduser().resolve()
    from repository_manager import stash_guard

    try:
        with stash_guard.hold_tree_mutation_lease(
            str(tree), note=f"safe commit: {message}"
        ):
            return _safe_commit_locked(
                tree,
                message,
                paths=paths,
                allow_empty=allow_empty,
                gate=gate,
                defer_gate=defer_gate,
                env=env,
                timeout=timeout,
            )
    except (OSError, RuntimeError) as exc:
        response = _result(
            path=tree,
            lane=_lane_name(tree),
            status="error",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            gate_deferred=defer_gate,
            error=str(exc),
        )
        response["reason"] = (
            "tree-mutation-busy"
            if isinstance(exc, stash_guard.BlockedByLease)
            else "tree-mutation-lease-unavailable"
        )
        return response
