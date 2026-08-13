"""``lane_doctor.heal`` — P0.6: lane_doctor OWNS the repair, not just the report.

Each test introduces the exact known-bad input (the `core.bare` archetype this
whole invariants program is named after) and proves `heal` fixes it — never
merely names a remedy — while a healthy tree proves `heal` does not "repair"
what was never broken.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repository_manager import lane_doctor


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(  # noqa: S603
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


def _init_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    _git(["init", "-b", "main"], root)
    _git(["config", "user.email", "lane@test"], root)
    _git(["config", "user.name", "lane"], root)
    (root / "README.md").write_text("base\n")
    _git(["add", "-A"], root)
    _git(["commit", "-m", "base"], root)
    return root


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    return _init_repo(tmp_path / "canonical")


@pytest.fixture
def worktree(canonical: Path, tmp_path: Path) -> Path:
    tree = tmp_path / "lane"
    _git(["worktree", "add", str(tree), "-b", "lane/test", "main"], canonical)
    return tree


def test_heal_repairs_core_bare_drift_on_its_own_tree(
    canonical: Path, worktree: Path
) -> None:
    _git(["config", "core.bare", "true"], worktree)
    assert _git(["config", "--bool", "--get", "core.bare"], worktree) == "true"

    result = lane_doctor.heal(worktree)

    assert result["attempted"] == ["own-tree"]
    assert result["healed"] == ["own-tree"]
    assert _git(["config", "--bool", "--get", "core.bare"], worktree) == "false"
    # And the isolation diagnosis itself no longer sees a structural problem.
    assert "tree-repair" not in result["after"]["blocking"]


def test_heal_repairs_core_bare_drift_visible_from_the_canonical_checkout(
    canonical: Path, worktree: Path
) -> None:
    """The D-MQR-5/6 shape: `core.bare=true` was found set directly on
    `agent-utilities`'s canonical checkout. `git config --get core.bare` run
    from ANY linked worktree resolves the shared config too (worktree-local
    config is the exception, not the default), so `heal`'s own-tree pass
    already observes and repairs it here through the shared file — proven by
    checking the CANONICAL's config directly afterward, not by which "target"
    label happened to fire.
    """
    _git(["config", "core.bare", "true"], canonical)
    assert _git(["config", "--bool", "--get", "core.bare"], canonical) == "true"

    result = lane_doctor.heal(worktree)

    assert result["attempted"], "heal must have detected and attempted a repair"
    assert result["healed"], "the repair must have succeeded, not merely attempted"
    assert _git(["config", "--bool", "--get", "core.bare"], canonical) == "false"
    assert "canonical-is-worktree" not in result["after"]["blocking"]
    assert "tree-repair" not in result["after"]["blocking"]


def test_heal_is_a_noop_on_a_healthy_tree(canonical: Path, worktree: Path) -> None:
    result = lane_doctor.heal(worktree)
    assert result["attempted"] == []
    assert result["healed"] == []
    assert result["repairs"] == []


def test_heal_fixes_co_occurring_findings_in_one_call(
    canonical: Path, worktree: Path
) -> None:
    """Live-observed shape (2026-08-13, during this program's own work): a
    `core.bare` flip on the shared common config left a linked worktree's
    index simultaneously reporting `probable-index-wipe` — `tree_repair
    .diagnose` only ever names ONE primary finding, so a `heal` that called
    `repair` exactly once left the tree still broken. `heal` must clear BOTH
    in a single call, not require the caller to invoke it twice.
    """

    # `probable-index-wipe` only fires past a floor of 12 tracked files
    # (`tree_repair._INDEX_COLLAPSE_COUNT`) — this fixture's single
    # `README.md` is below it, so give it enough tracked files to make the
    # collapse actually detectable, same as
    # `tests/fixtures/tree_mutation_hazards.py::truncated_index` does.
    for index in range(15):
        (worktree / f"tracked-{index:02d}.txt").write_text(f"content {index}\n")
    _git(["add", "-A"], worktree)
    _git(["commit", "-m", "add tracked files"], worktree)

    _git(["config", "core.bare", "true"], worktree)
    # Simulate the co-occurring index collapse independently of the
    # core.bare flip (its usual live trigger is a concurrent process, not
    # reproducible deterministically in a test).
    _git(["read-tree", "--empty"], worktree)

    result = lane_doctor.heal(worktree)

    assert "tree-repair" not in result["after"]["blocking"]
    own_tree_repairs = [r for r in result["repairs"] if r["target"] == "own-tree"]
    assert {r["finding"] for r in own_tree_repairs} == {
        "core-bare-drift",
        "probable-index-wipe",
    }
    assert all(r["result"].get("ok") for r in own_tree_repairs)
    assert _git(["config", "--bool", "--get", "core.bare"], worktree) == "false"
    # The README.md content that was there before the simulated wipe is back.
    assert (worktree / "README.md").read_text() == "base\n"


def test_heal_action_is_dispatchable(canonical: Path, worktree: Path) -> None:
    _git(["config", "core.bare", "true"], worktree)
    result = lane_doctor.dispatch("heal", path=str(worktree))
    assert result["healed"] == ["own-tree"]
    assert "heal" in lane_doctor.ACTIONS
