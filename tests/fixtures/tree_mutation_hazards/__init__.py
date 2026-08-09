"""Disposable, importable reproductions for RMDD-26.

Every builder accepts a caller-owned temporary directory and performs all
mutations only inside the returned fixture repository.  RMDD-23 imports these
builders instead of reconstructing a subtly different incident in its
concurrency harness.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

__all__ = [
    "bare_destructive_attempt",
    "canonical_dirty_race",
    "core_bare_drift",
    "make_all",
    "mutation_race",
    "shared_stash",
    "tracked_destructive_loss",
    "truncated_index",
    "unstaged_deletions_with_hook",
]


def _run(
    args: list[str], cwd: Path, *, check: bool = True
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=check,
    )


def _repo(root: Path, name: str) -> Path:
    path = root / name
    path.mkdir(parents=True, exist_ok=True)
    _run(["git", "init", "-b", "main"], path)
    _run(["git", "config", "user.email", "rmdd26@example.invalid"], path)
    _run(["git", "config", "user.name", "RMDD-26 fixture"], path)
    (path / "README.md").write_text("fixture\n", encoding="utf-8")
    _run(["git", "add", "-A"], path)
    _run(["git", "commit", "-m", "fixture base"], path)
    return path


def unstaged_deletions_with_hook(root: Path) -> Path:
    """Create the deletion-plus-hook shape from the staged-only incident."""
    path = _repo(root, "unstaged-deletions")
    (path / "must-delete.txt").write_text("delete me\n", encoding="utf-8")
    (path / "staged-change.txt").write_text("base\n", encoding="utf-8")
    _run(["git", "add", "-A"], path)
    _run(["git", "commit", "-m", "fixture deletion files"], path)
    (path / "must-delete.txt").unlink()
    (path / "staged-change.txt").write_text("changed\n", encoding="utf-8")
    hook = path / ".git" / "hooks" / "pre-commit"
    hook.write_text(
        "#!/bin/sh\n"
        "# Mimic staged_files_only restoring the index snapshot over an\n"
        "# unstaged deletion. A fully staged safe_commit is unaffected.\n"
        "git checkout -- .\n",
        encoding="utf-8",
    )
    hook.chmod(hook.stat().st_mode | 0o111)
    return path


def truncated_index(root: Path, *, files: int = 4634, retained: int = 5) -> Path:
    """Create the documented ~4634-entry HEAD with a collapsed index."""
    path = _repo(root, "truncated-index")
    for index in range(files):
        (path / f"tracked-{index:04d}.txt").write_text(
            f"content {index}\n", encoding="utf-8"
        )
    _run(["git", "add", "-A"], path)
    _run(["git", "commit", "-m", "fixture tracked baseline"], path)
    _run(["git", "read-tree", "--empty"], path)
    _run(
        [
            "git",
            "add",
            "--",
            *(f"tracked-{index:04d}.txt" for index in range(min(retained, files))),
        ],
        path,
    )
    return path


def core_bare_drift(root: Path) -> Path:
    """Create a known-non-bare checkout whose local config says it is bare."""
    path = _repo(root, "core-bare-drift")
    _run(["git", "config", "core.bare", "true"], path)
    return path


def shared_stash(root: Path) -> Path:
    """Create a dirty tree with an actual non-empty shared ``refs/stash``."""
    path = _repo(root, "shared-stash")
    (path / "README.md").write_text("shared stash WIP\n", encoding="utf-8")
    (path / "stash-only.txt").write_text("must survive\n", encoding="utf-8")
    _run(["git", "stash", "push", "-u", "-m", "fixture shared stash"], path)
    return path


def canonical_dirty_race(root: Path) -> tuple[Path, Path]:
    """Create a canonical checkout and a linked lane with canonical WIP."""
    canonical = _repo(root, "canonical-race")
    lane = root / "canonical-race-lane"
    _run(["git", "worktree", "add", str(lane), "-b", "lane/fixture", "main"], canonical)
    (canonical / "canonical-wip.txt").write_text(
        "must not be clobbered\n", encoding="utf-8"
    )
    return canonical, lane


def bare_destructive_attempt(root: Path) -> Path:
    """Create a dirty tree containing an untracked file that must survive."""
    path = _repo(root, "bare-destructive")
    (path / "must-survive.txt").write_text("operator WIP\n", encoding="utf-8")
    return path


def tracked_destructive_loss(root: Path) -> Path:
    """Create a tracked edit that a bare hard reset demonstrably loses."""
    path = _repo(root, "tracked-destructive-loss")
    (path / "README.md").write_text("tracked operator WIP\n", encoding="utf-8")
    return path


def mutation_race(root: Path) -> Path:
    """Create a disposable tree used by the cooperative snapshot/execute race."""
    path = _repo(root, "mutation-race")
    (path / "race-must-survive.txt").write_text(
        "cooperative race WIP\n", encoding="utf-8"
    )
    return path


HAZARD_BUILDERS: dict[str, Callable[[Path], object]] = {
    "unstaged-deletions-with-hook": unstaged_deletions_with_hook,
    "truncated-index": truncated_index,
    "core-bare-drift": core_bare_drift,
    "shared-stash": shared_stash,
    "canonical-dirty-race": canonical_dirty_race,
    "bare-destructive-attempt": bare_destructive_attempt,
    "tracked-destructive-loss": tracked_destructive_loss,
    "mutation-race": mutation_race,
}


def make_all(root: Path) -> dict[str, object]:
    """Build all documented mutation-hazard shapes under one disposable root."""
    return {name: builder(root) for name, builder in HAZARD_BUILDERS.items()}
