"""Commit a complete working-tree snapshot before running its gate.

``pre-commit`` has a ``staged_files_only`` context that temporarily removes
unstaged changes while hooks run.  That context is useful for an ordinary
staged-only commit, but it is a data-loss boundary when a process is killed
inside it.  :func:`safe_commit` makes the boundary unreachable: it stages the
complete tree (including deletions and untracked files), proves that no
unstaged content remains, runs the configured gate, stages formatter output,
proves the invariant again, and only then commits.

CONCEPT:RM-SAFE-COMMIT (C-12)
"""

from __future__ import annotations

import os
import subprocess  # nosec B404 - fixed-argv git/gate execution is this module's job
from collections.abc import Callable, Sequence
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


def _safe_commit_locked(
    path: Path | str,
    message: str,
    *,
    allow_empty: bool = False,
    gate: Sequence[str] | Callable[[Path], Any] | None = None,
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
        env: Optional environment for the gate and git commands.
        timeout: Per-command timeout in seconds.

    The returned ``nothing_left_unstaged`` is an assertion made immediately
    before the gate and again before commit, not an inference from a green gate.
    """
    tree = Path(path).expanduser().resolve()
    lane = _lane_name(tree)
    command_env = os.environ.copy()
    if env is not None:
        command_env.update(env)
    if not tree.is_dir():
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            error=f"repository path does not exist: {tree}",
        )

    initial = _run(
        ["git", "status", "--porcelain", "-z"], tree, env=command_env, timeout=timeout
    )
    if initial.returncode != 0:
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            error=_detail(initial) or "git status failed",
        )
    if not initial.stdout and not allow_empty:
        return _result(
            path=tree,
            lane=lane,
            status="skipped",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            nothing_left_unstaged=True,
            error="no changes to commit",
        )

    staged = _run(["git", "add", "-A"], tree, env=command_env, timeout=timeout)
    if staged.returncode != 0:
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=[],
            gate_stage="none",
            gate_invoked=False,
            error=_detail(staged) or "git add -A failed",
        )

    staged_paths = _staged_paths(tree)
    unstaged = _unstaged_paths(tree)
    if unstaged:
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=staged_paths,
            gate_stage="none",
            gate_invoked=False,
            error="git add -A left content unstaged: " + ", ".join(unstaged[:20]),
        )

    gate_stage = "none"
    configured_gate: Sequence[str] | Callable[[Path], Any] | None = gate
    if configured_gate is None and (tree / ".pre-commit-config.yaml").is_file():
        configured_gate = ["pre-commit", "run", "--all-files"]
        gate_stage = "pre-commit"
    elif configured_gate is not None:
        gate_stage = "configured"

    if configured_gate is not None:
        passed, detail = _run_gate(
            configured_gate, tree, env=command_env, timeout=timeout
        )
        if not passed:
            return _result(
                path=tree,
                lane=lane,
                status="error",
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=True,
                nothing_left_unstaged=True,
                error=detail or "configured gate failed",
            )
        # A formatter may have changed files during the gate.  Fold that
        # output into the same snapshot and prove the invariant again.
        restaged = _run(["git", "add", "-A"], tree, env=command_env, timeout=timeout)
        if restaged.returncode != 0:
            return _result(
                path=tree,
                lane=lane,
                status="error",
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=True,
                nothing_left_unstaged=False,
                error=_detail(restaged) or "git add -A after gate failed",
            )
        staged_paths = _staged_paths(tree)
        unstaged = _unstaged_paths(tree)
        if unstaged:
            return _result(
                path=tree,
                lane=lane,
                status="error",
                staged_paths=staged_paths,
                gate_stage=gate_stage,
                gate_invoked=True,
                error="gate left content unstaged: " + ", ".join(unstaged[:20]),
            )

    commit_argv = ["git", "commit"]
    if allow_empty:
        commit_argv.append("--allow-empty")
    commit_argv.extend(["-m", message])
    committed = _run(commit_argv, tree, env=command_env, timeout=timeout)
    if committed.returncode != 0:
        return _result(
            path=tree,
            lane=lane,
            status="error",
            staged_paths=staged_paths,
            gate_stage=gate_stage,
            gate_invoked=configured_gate is not None,
            nothing_left_unstaged=True,
            error=_detail(committed) or "git commit failed",
        )
    sha_result = _run(
        ["git", "rev-parse", "HEAD"], tree, env=command_env, timeout=timeout
    )
    sha = sha_result.stdout.strip() if sha_result.returncode == 0 else None
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
    return _result(
        path=tree,
        lane=lane,
        status="success",
        staged_paths=staged_paths,
        gate_stage=gate_stage,
        gate_invoked=configured_gate is not None,
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
    env: dict[str, str] | None = None,
    timeout: int = 1800,
) -> dict[str, Any]:
    """Commit under the per-worktree mutation lease.

    The lease spans status, complete staging, the configured gate, commit, and
    baseline refresh.  It is deliberately per-worktree, so independent lanes
    continue to run concurrently while same-tree callers cannot interleave a
    check with another mutation.
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
            error=str(exc),
        )
        response["reason"] = (
            "tree-mutation-busy"
            if isinstance(exc, stash_guard.BlockedByLease)
            else "tree-mutation-lease-unavailable"
        )
        return response
