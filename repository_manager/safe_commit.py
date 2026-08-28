"""Commit a complete working-tree snapshot before running its gate.

``pre-commit`` has a ``staged_files_only`` context that temporarily removes
unstaged changes while hooks run.  That context is useful for an ordinary
staged-only commit, but it is a data-loss boundary when a process is killed
inside it.  :func:`safe_commit` makes the boundary unreachable: it stages the
complete tree (including deletions and untracked files), proves that no
unstaged content remains, runs the configured gate, stages formatter output,
proves the invariant again, and only then commits.

Callers that must create a WIP snapshot before a heavy gate is admitted may
pass ``defer_gate=True``.  That mode stages and verifies the complete tree,
commits with ``--no-verify``, and returns ``gate_deferred=True``.  It confers
no validation evidence; the caller must submit the real gate through the
common scheduler/executor against the returned immutable SHA.

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
        from agent_utilities.governance.lanes import lane_name

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


def _unstaged_paths(path: Path) -> list[str]:
    """Return tracked or untracked paths not represented in the index."""
    tracked = _run(["git", "diff", "--name-only", "-z"], path)
    untracked = _run(["git", "ls-files", "--others", "--exclude-standard", "-z"], path)
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

    allow_empty: bool = False
    gate: Sequence[str] | Callable[[Path], Any] | None = None
    defer_gate: bool = False
    command_env: dict[str, str] = field(default_factory=dict)
    timeout: int = 1800


def _safe_commit_locked(
    path: Path | str,
    message: str,
    *,
    allow_empty: bool = False,
    gate: Sequence[str] | Callable[[Path], Any] | None = None,
    defer_gate: bool = False,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Stage, gate, and commit the complete working tree as one safe operation.

    Args:
        path: A repository worktree.  It is never interpreted as a shell value.
        message: Commit message passed as one argv element.
        allow_empty: Permit an empty commit when the tree has no changes.
        gate: Optional fixed-argv gate or callable.  By default a repository
            with ``.pre-commit-config.yaml`` runs ``pre-commit run --all-files``;
            repositories without that file have no gate to invoke.
        defer_gate: Stage and commit the complete snapshot without invoking any
            repository hook.  The commit is made with ``--no-verify`` and the
            result explicitly records ``gate_deferred=True``; a caller must run
            the real gate through its admitted executor afterwards.
        env: Optional environment for the gate and git commands.
        timeout: Per-command timeout in seconds.

    The returned ``nothing_left_unstaged`` is an assertion made immediately
    before the gate and again before commit, not an inference from a green gate.
    """
    tree = Path(path).expanduser().resolve()
    lane = _lane_name(tree)
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
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    config = _CommitConfig(
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
    """Raise ``skipped`` when the tree is clean, ``error`` when status fails."""
    initial = _run(
        ["git", "status", "--porcelain", "-z"],
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


def _stage_all(tree: Path, config: _CommitConfig) -> None:
    staged = _run(
        ["git", "add", "-A"], tree, env=config.command_env, timeout=config.timeout
    )
    if staged.returncode != 0:
        raise _CommitAbort(_detail(staged) or "git add -A failed")


def _verify_nothing_unstaged(
    tree: Path,
    staged_paths: list[str],
    *,
    gate_stage: str,
    gate_invoked: bool,
    prefix: str,
) -> None:
    unstaged = _unstaged_paths(tree)
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


def _stage_all_after_gate(
    tree: Path,
    config: _CommitConfig,
    staged_paths: list[str],
    gate_stage: str,
) -> None:
    # A formatter may have changed files during the gate.  Fold that output
    # into the same snapshot and prove the invariant again.
    restaged = _run(
        ["git", "add", "-A"], tree, env=config.command_env, timeout=config.timeout
    )
    if restaged.returncode != 0:
        raise _CommitAbort(
            _detail(restaged) or "git add -A after gate failed",
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
    _stage_all(tree, config)
    staged_paths = _staged_paths(tree)
    _verify_nothing_unstaged(
        tree,
        staged_paths,
        gate_stage="none",
        gate_invoked=False,
        prefix="git add -A left content unstaged",
    )

    gate_stage, configured_gate = _resolve_gate(tree, config.gate, config.defer_gate)
    if configured_gate is not None:
        _run_configured_gate(configured_gate, tree, config, staged_paths, gate_stage)
        _stage_all_after_gate(tree, config, staged_paths, gate_stage)
        staged_paths = _staged_paths(tree)
        _verify_nothing_unstaged(
            tree,
            staged_paths,
            gate_stage=gate_stage,
            gate_invoked=True,
            prefix="gate left content unstaged",
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
    allow_empty: bool = False,
    gate: Sequence[str] | Callable[[Path], Any] | None = None,
    defer_gate: bool = False,
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Commit under the per-worktree mutation lease.

    The lease spans status, complete staging, the configured gate (or an
    explicitly deferred snapshot), commit, and baseline refresh.  It is
    deliberately per-worktree, so independent lanes continue to run
    concurrently while same-tree callers cannot interleave a check with
    another mutation.
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
