"""Tests for the lane preflight (CONCEPT:RM-LANE-DOCTOR).

Written against real git fixture repositories and real filesystem state, not
mocks of the checks themselves.

Every check gets a **pair**: one test that deliberately introduces the exact
known-bad input the check exists to catch and asserts it is caught, and one that
asserts the healthy shape passes. The negative half alone is not evidence — a
check that returns FAIL unconditionally would pass it, which is precisely how
three gates on this workspace were found green while enforcing nothing. The
positive half is what makes the refusal meaningful.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from repository_manager import lane_doctor
from repository_manager.lane_doctor import FAIL, OK, SKIP, WARN


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(  # noqa: S603
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    """A real canonical checkout with one commit on ``main``."""
    root = tmp_path / "canonical"
    root.mkdir()
    _git(["init", "-b", "main"], root)
    _git(["config", "user.email", "lane@test"], root)
    _git(["config", "user.name", "lane"], root)
    (root / "README.md").write_text("base\n")
    _git(["add", "-A"], root)
    _git(["commit", "-m", "base"], root)
    return root


@pytest.fixture
def worktree(canonical: Path, tmp_path: Path) -> Path:
    """A linked worktree of ``canonical`` on its own branch — a healthy lane."""
    tree = tmp_path / "lane"
    _git(["worktree", "add", str(tree), "-b", "lane/test", "main"], canonical)
    return tree


def _named(report: dict, name: str) -> dict:
    return next(c for c in report["checks"] if c["name"] == name)


# ---------------------------------------------------------------------------
# not-canonical — a background `git reset` on a canonical tree destroyed ~20min
# ---------------------------------------------------------------------------
def test_a_canonical_checkout_is_refused(canonical: Path) -> None:
    check = _named(lane_doctor.diagnose(canonical, env={}), "not-canonical")
    assert check["status"] == FAIL
    assert "READ-ONLY" in check["finding"]


def test_a_linked_worktree_is_allowed(worktree: Path) -> None:
    """The positive half: without it, an unconditional FAIL would pass above."""
    check = _named(lane_doctor.diagnose(worktree, env={}), "not-canonical")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# no-worktree-venv — a worktree-local .venv produced ~167 phantom failures
# ---------------------------------------------------------------------------
def test_a_worktree_local_venv_is_refused(worktree: Path) -> None:
    (worktree / ".venv").mkdir()
    check = _named(lane_doctor.diagnose(worktree, env={}), "no-worktree-venv")
    assert check["status"] == FAIL
    assert str(worktree / ".venv") in check["evidence"]["venv"]


def test_a_uv_workspace_managed_worktree_venv_passes(worktree: Path) -> None:
    (worktree / "scripts").mkdir()
    (worktree / "scripts" / "uv_workspace.py").write_text("# managed launcher\n")
    venv = worktree / ".venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("#!/usr/bin/env python3\n")
    python.chmod(0o755)
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / ".uv-workspace-selection.json").write_text(
        '{"label": "", "selection": ["--all-extras"]}\n'
    )

    check = _named(lane_doctor.diagnose(worktree, env={}), "no-worktree-venv")

    assert check["status"] == OK
    assert check["evidence"]["selection"] == ["--all-extras"]


def test_a_stale_managed_worktree_venv_is_refused(worktree: Path) -> None:
    (worktree / "scripts").mkdir()
    (worktree / "scripts" / "uv_workspace.py").write_text("# managed launcher\n")
    venv = worktree / ".venv"
    venv.mkdir()
    (venv / "pyvenv.cfg").write_text("home = /usr/bin\n")
    (venv / ".uv-workspace-selection.json").write_text(
        '{"label": "", "selection": ["--all-extras"]}\n'
    )

    check = _named(lane_doctor.diagnose(worktree, env={}), "no-worktree-venv")

    assert check["status"] == FAIL
    assert "bin/python" in check["finding"]


def test_no_local_venv_passes(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "no-worktree-venv")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# venv-package-count — D-CIP-19: doctor reported healthy on a venv silently
# overwritten by an unrelated project (3-6 packages instead of ~726). This
# check inspects site-packages CONTENT directly, independent of the
# ownership-marker check above.
# ---------------------------------------------------------------------------
def _write_pyproject(tree: Path, project_name: str) -> None:
    (tree / "pyproject.toml").write_text(f'[project]\nname = "{project_name}"\n')


def _make_venv(tree: Path) -> Path:
    venv = tree / ".venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin" / "python"
    python.write_text("#!/usr/bin/env python3\n")
    python.chmod(0o755)
    site_packages = venv / "lib" / "python3.12" / "site-packages"
    site_packages.mkdir(parents=True)
    return site_packages


def test_a_near_empty_venv_is_refused(worktree: Path) -> None:
    """Known-bad input #1: D-CIP-19's exact shape, an implausibly thin venv."""
    _write_pyproject(worktree, "repository-manager")
    site_packages = _make_venv(worktree)
    for name in ("alpha-0.1.0.dist-info", "__editable__.alpha-0.1.0.dist-info"):
        (site_packages / name).mkdir()

    check = _named(lane_doctor.diagnose(worktree, env={}), "venv-package-count")

    assert check["status"] == FAIL
    assert check["evidence"]["resolved_count"] == 2
    assert "implausibly low" in check["finding"]


def test_a_venv_belonging_to_an_unrelated_project_is_refused(worktree: Path) -> None:
    """Known-bad input #2: package COUNT looks fine, but none of it is this repo.

    Reproduces the live D-CIP-19 incident precisely: the venv is not empty and
    not obviously thin, it just belongs to a completely different project
    ("alpha") that silently overwrote this worktree's own environment.
    """
    _write_pyproject(worktree, "repository-manager")
    site_packages = _make_venv(worktree)
    for i in range(20):
        (site_packages / f"alpha-plugin-{i}-1.0.0.dist-info").mkdir()

    check = _named(lane_doctor.diagnose(worktree, env={}), "venv-package-count")

    assert check["status"] == FAIL
    assert "NONE of them is this project's own" in check["finding"]
    assert check["evidence"]["expected_project"] == "repository-manager"


def test_a_healthy_venv_with_plausible_own_packages_passes(worktree: Path) -> None:
    """Positive half: without it, an unconditional FAIL above would also pass."""
    _write_pyproject(worktree, "repository-manager")
    site_packages = _make_venv(worktree)
    (site_packages / "repository_manager-0.1.0.dist-info").mkdir()
    for i in range(20):
        (site_packages / f"dep-{i}-1.0.0.dist-info").mkdir()

    check = _named(lane_doctor.diagnose(worktree, env={}), "venv-package-count")

    assert check["status"] == OK
    assert check["evidence"]["resolved_count"] == 21


def test_venv_package_count_skips_when_there_is_no_venv(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "venv-package-count")
    assert check["status"] == SKIP


# ---------------------------------------------------------------------------
# cargo-partition — a shared CARGO_TARGET_DIR CORRUPTS concurrent builds
# ---------------------------------------------------------------------------
def test_a_shared_cargo_target_dir_is_refused(worktree: Path, tmp_path: Path) -> None:
    shared = tmp_path / "shared-target"
    env = {"CARGO_TARGET_DIR": str(shared)}
    check = _named(lane_doctor.diagnose(worktree, env=env), "cargo-partition")
    assert check["status"] == FAIL
    assert "target-isolated" in check["remedy"]


def test_a_target_dir_inside_the_worktree_passes(worktree: Path) -> None:
    env = {"CARGO_TARGET_DIR": str(worktree / "target-isolated")}
    check = _named(lane_doctor.diagnose(worktree, env=env), "cargo-partition")
    assert check["status"] == OK


def test_no_exported_target_dir_passes(worktree: Path) -> None:
    """An unset var is fine — cargo then uses this tree's own ./target."""
    check = _named(lane_doctor.diagnose(worktree, env={}), "cargo-partition")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# precommit-home — the shared store is where unstaged work goes to die (D-OB-12)
# ---------------------------------------------------------------------------
def test_an_unset_precommit_home_is_refused(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "precommit-home")
    assert check["status"] == FAIL
    assert "export PRE_COMMIT_HOME=" in check["remedy"]


def test_declaring_the_shared_store_explicitly_is_still_refused(
    worktree: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Naming the shared default does not make it private."""
    shared = tmp_path / "shared-precommit"
    shared.mkdir()
    monkeypatch.setattr(lane_doctor, "_SHARED_PRECOMMIT_HOME", shared)
    env = {"PRE_COMMIT_HOME": str(shared)}
    check = _named(lane_doctor.diagnose(worktree, env=env), "precommit-home")
    assert check["status"] == FAIL


def test_a_partitioned_precommit_home_passes(worktree: Path, tmp_path: Path) -> None:
    env = {"PRE_COMMIT_HOME": str(tmp_path / "my-lane-precommit")}
    check = _named(lane_doctor.diagnose(worktree, env=env), "precommit-home")
    assert check["status"] == OK


def test_patch_files_in_the_shared_store_are_reported_as_paths_not_verdicts(
    worktree: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A leftover patch is not proof of a crash — pre-commit never deletes one.

    So it must surface as a path to `git apply` if work is missing, never as a
    failing verdict on its own.
    """
    shared = tmp_path / "shared-precommit"
    (shared / "patch1").mkdir(parents=True)
    (shared / "patch1" / "12345-1700000000").write_text("diff --git a/x b/x\n")
    monkeypatch.setattr(lane_doctor, "_SHARED_PRECOMMIT_HOME", shared)
    env = {"PRE_COMMIT_HOME": str(tmp_path / "mine")}
    check = _named(lane_doctor.diagnose(worktree, env=env), "precommit-home")
    assert check["status"] == OK
    assert len(check["evidence"]["shared_store_patches"]) == 1


# ---------------------------------------------------------------------------
# shared-stash-ref — refs/stash is ONE ref shared by every worktree
# ---------------------------------------------------------------------------
def test_an_existing_shared_stash_ref_is_flagged(worktree: Path) -> None:
    (worktree / "README.md").write_text("dirty\n")
    _git(["stash"], worktree)
    check = _named(lane_doctor.diagnose(worktree, env={}), "shared-stash-ref")
    assert check["status"] == WARN
    assert "git show HEAD:<path>" in check["remedy"]


def test_an_empty_stash_ref_passes(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "shared-stash-ref")
    assert check["status"] == OK


def test_a_sibling_lanes_stash_is_visible_from_this_worktree(
    canonical: Path, worktree: Path, tmp_path: Path
) -> None:
    """The whole reason the rule exists: the ref is not per-worktree.

    A stash pushed in a SIBLING worktree is reported here, which is exactly how
    one lane's `git stash pop` consumes another lane's entry.
    """
    sibling = tmp_path / "sibling"
    _git(["worktree", "add", str(sibling), "-b", "lane/sibling", "main"], canonical)
    (sibling / "README.md").write_text("sibling work\n")
    _git(["stash"], sibling)

    check = _named(lane_doctor.diagnose(worktree, env={}), "shared-stash-ref")
    assert check["status"] == WARN


# ---------------------------------------------------------------------------
# pytest-basetemp
# ---------------------------------------------------------------------------
def test_a_missing_basetemp_is_flagged(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "pytest-basetemp")
    assert check["status"] == WARN


def test_a_declared_basetemp_passes(worktree: Path, tmp_path: Path) -> None:
    env = {"PYTEST_ADDOPTS": f"--basetemp={tmp_path / 'bt'}"}
    check = _named(lane_doctor.diagnose(worktree, env=env), "pytest-basetemp")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# canonical-clean — a land is refused against a dirty canonical, INCLUDING
# an untracked-only one
# ---------------------------------------------------------------------------
def test_a_dirty_canonical_is_flagged_from_inside_the_worktree(
    canonical: Path, worktree: Path
) -> None:
    (canonical / "README.md").write_text("someone else is mid-edit\n")
    check = _named(lane_doctor.diagnose(worktree, env={}), "canonical-clean")
    assert check["status"] == WARN
    assert check["evidence"]["entries"]


def test_an_untracked_only_canonical_is_also_flagged(
    canonical: Path, worktree: Path
) -> None:
    """`git status --porcelain` reports both; an untracked-only tree is refused
    by the canonical guard just the same, so it must be reported here too."""
    (canonical / "scratch.txt").write_text("untracked\n")
    check = _named(lane_doctor.diagnose(worktree, env={}), "canonical-clean")
    assert check["status"] == WARN


def test_a_clean_canonical_passes(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "canonical-clean")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# canonical-is-worktree — D-MQR-5/D-MQR-6: `core.bare=true` corrupted the
# agent-utilities canonical checkout's .git/config directly. Every git command
# run WITH CWD=canonical then failed ("this operation must be run in a work
# tree"), which silently killed the repo's merge queue -- but `lane_scope()`
# resolves fine from a LINKED WORKTREE even while the main tree is bare, so
# every OTHER check in this module (which all inspect the calling lane's own
# tree) stayed green throughout. This is the one check that inspects the
# canonical checkout directly -- the known-bad input is `core.bare=true`
# written onto a THROWAWAY canonical fixture, never a real checkout.
# ---------------------------------------------------------------------------
def test_a_bare_corrupted_canonical_is_refused_from_inside_the_worktree(
    canonical: Path, worktree: Path
) -> None:
    """The known-bad input this check exists to catch, reproduced directly."""
    _git(["config", "core.bare", "true"], canonical)
    check = _named(lane_doctor.diagnose(worktree, env={}), "canonical-is-worktree")
    assert check["status"] == FAIL
    assert "core.bare" in check["finding"]
    assert "must be run in a work tree" not in check["remedy"]  # remedy, not echo
    assert str(canonical) in check["evidence"]["canonical"]


def test_a_healthy_canonical_passes_the_worktree_check(worktree: Path) -> None:
    """The positive half: without it, an unconditional FAIL would pass above."""
    check = _named(lane_doctor.diagnose(worktree, env={}), "canonical-is-worktree")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# merge-queue-config — a repo declaring no gates is REFUSED, not defaulted
# ---------------------------------------------------------------------------
def test_a_repo_with_no_gate_declaration_is_flagged(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "merge-queue-config")
    assert check["status"] == WARN


def test_a_declaration_in_the_canonical_checkout_counts(
    canonical: Path, worktree: Path
) -> None:
    (canonical / ".mergequeue.yaml").write_text("base: main\ngates: []\n")
    check = _named(lane_doctor.diagnose(worktree, env={}), "merge-queue-config")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# base-drift — the branch TIP is not the tree that lands
# ---------------------------------------------------------------------------
def test_a_moved_base_is_flagged(canonical: Path, worktree: Path) -> None:
    (canonical / "other.txt").write_text("landed while you worked\n")
    _git(["add", "-A"], canonical)
    _git(["commit", "-m", "another lane landed"], canonical)

    check = _named(lane_doctor.diagnose(worktree, base="main", env={}), "base-drift")
    assert check["status"] == WARN
    assert "merge-tree --write-tree" in check["remedy"]


def test_a_current_branch_passes(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, base="main", env={}), "base-drift")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# committed-work
# ---------------------------------------------------------------------------
def test_uncommitted_work_is_flagged(worktree: Path) -> None:
    (worktree / "README.md").write_text("in flight\n")
    check = _named(lane_doctor.diagnose(worktree, env={}), "committed-work")
    assert check["status"] == WARN


def test_a_clean_tree_passes(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "committed-work")
    assert check["status"] == OK


# ---------------------------------------------------------------------------
# test-runner — `uv run pytest` silently runs the SYSTEM pytest
# ---------------------------------------------------------------------------
def test_a_repo_shipping_uv_workspace_names_the_correct_runner(
    worktree: Path,
) -> None:
    scripts = worktree / "scripts"
    scripts.mkdir()
    (scripts / "uv_workspace.py").write_text("# runner\n")
    check = _named(lane_doctor.diagnose(worktree, env={}), "test-runner")
    assert check["status"] == WARN
    assert "uv_workspace.py run --all-extras" in check["remedy"]
    assert "sys.executable" in check["remedy"]


def test_a_repo_without_that_runner_is_skipped_not_warned(worktree: Path) -> None:
    check = _named(lane_doctor.diagnose(worktree, env={}), "test-runner")
    assert check["status"] == SKIP


# ---------------------------------------------------------------------------
# Aggregate contract
# ---------------------------------------------------------------------------
def test_only_fail_blocks_and_warn_never_does(worktree: Path) -> None:
    """WARN names a condition legitimate in some lanes and fatal in others, so
    the decision stays with the lane. Only FAIL blocks."""
    report = lane_doctor.diagnose(worktree, env={"PRE_COMMIT_HOME": "/tmp/mine"})
    warned = [c for c in report["checks"] if c["status"] == WARN]
    assert warned, "fixture should produce at least one warning"
    assert report["blocking"] == []
    assert report["ok"] is True


def test_a_blocking_check_makes_the_whole_report_not_ok(canonical: Path) -> None:
    report = lane_doctor.diagnose(canonical, env={"PRE_COMMIT_HOME": "/tmp/mine"})
    assert report["ok"] is False
    assert "not-canonical" in report["blocking"]


def test_every_failing_check_carries_a_literal_remedy_command(canonical: Path) -> None:
    """A prohibition without its replacement command demonstrably does not stick."""
    report = lane_doctor.diagnose(canonical, env={})
    failing = [c for c in report["checks"] if c["status"] == FAIL]
    assert failing
    for check in failing:
        assert check["remedy"].strip(), f"{check['name']} refuses without a remedy"


def test_diagnose_mutates_nothing(worktree: Path) -> None:
    before = _git(["status", "--porcelain"], worktree)
    before_head = _git(["rev-parse", "HEAD"], worktree)
    lane_doctor.diagnose(worktree, env={})
    assert _git(["status", "--porcelain"], worktree) == before
    assert _git(["rev-parse", "HEAD"], worktree) == before_head


# ---------------------------------------------------------------------------
# finish — preflight is a BLOCKING gate before the expensive queue cycle
# ---------------------------------------------------------------------------
def test_finish_refuses_a_lane_that_fails_its_own_preflight(canonical: Path) -> None:
    result = lane_doctor.finish(canonical, base="main")
    assert result["ok"] is False
    assert result["enqueued"] is False
    assert result["stage"] == "preflight"
    assert "not-canonical" in result["preflight"]["blocking"]


def test_finish_records_a_forced_run_rather_than_hiding_it(
    canonical: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`force` is for a blocking check consciously accepted — and the decision
    must stay visible in the result afterwards."""
    result = lane_doctor.finish(canonical, base="main", force=True)
    # It gets past preflight; whether it enqueues depends on the queue being
    # importable, which is not what this test pins.
    assert result["stage"] == "enqueue"
    assert result["forced"] is True
    assert result["ok"] is False
    assert result["enqueued"] is False
    assert "candidate must be a named branch" in result["reason"]


def test_finish_degrades_honestly_when_the_queue_is_unavailable(
    worktree: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No path may report a success it did not verify."""

    def _no_queue():
        raise ImportError("merge queue not in this build")

    monkeypatch.setattr(lane_doctor, "_load_merge_queue", _no_queue)
    result = lane_doctor.finish(worktree, base="main", force=True)
    assert result["ok"] is False
    assert result["enqueued"] is False
    assert "not available in this build" in result["reason"]


# ---------------------------------------------------------------------------
# exports + dispatch
# ---------------------------------------------------------------------------
def test_lane_exports_partition_every_shared_build_resource(worktree: Path) -> None:
    exports = lane_doctor.lane_exports(worktree)
    assert set(exports) == {
        "CARGO_TARGET_DIR",
        "TMPDIR",
        "PYTEST_ADDOPTS",
        "PRE_COMMIT_HOME",
    }
    assert Path(exports["CARGO_TARGET_DIR"]).is_relative_to(worktree)
    assert Path(exports["PRE_COMMIT_HOME"]).is_dir()


def test_exports_satisfy_the_checks_they_exist_to_satisfy(worktree: Path) -> None:
    """The loop closes: `start`'s environment makes `doctor` green.

    Without this, the exports and the checks could drift into disagreeing and
    every lane would begin life with a failing preflight.
    """
    exports = lane_doctor.lane_exports(worktree)
    report = lane_doctor.diagnose(worktree, env=exports)
    assert report["ok"] is True
    for name in ("cargo-partition", "precommit-home", "pytest-basetemp"):
        assert _named(report, name)["status"] == OK


def test_dispatch_rejects_an_unknown_action() -> None:
    assert lane_doctor.dispatch("teleport")["ok"] is False


def test_dispatch_start_requires_repo_and_branch() -> None:
    assert lane_doctor.dispatch("start", repo="x")["ok"] is False
    assert lane_doctor.dispatch("start", branch="y")["ok"] is False


def test_the_mcp_tool_is_registered_and_declares_every_action() -> None:
    """The CLI and the MCP tool are thin marshallers over one action core, so
    an action that exists in one and not the other is a defect, not a gap."""
    source = (
        Path(__file__).resolve().parents[1] / "repository_manager" / "mcp_server.py"
    ).read_text()
    assert "async def rm_lane(" in source
    assert "lane_doctor.ACTIONS" in source

    cli = (
        Path(__file__).resolve().parents[1]
        / "repository_manager"
        / "repository_manager.py"
    ).read_text()
    assert "_run_lane_cli" in cli
    for action in lane_doctor.ACTIONS:
        assert f'"{action}"' in cli, f"--lane does not offer {action}"
