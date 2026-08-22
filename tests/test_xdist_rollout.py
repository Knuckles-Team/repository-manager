"""``xdist_rollout`` — the plan/apply eligibility machinery, proven against
each of the three required gates independently (so a bug that accidentally
drops one gate's check is caught by the fixture that exercises ONLY that
gate) plus the write path itself under both ``dry_run=True`` (the default)
and ``dry_run=False``.
"""

from __future__ import annotations

from pathlib import Path

from repository_manager.xdist_rollout import (
    _BOILERPLATE_ENTRY,
    _PATCHED_ENTRY,
    OPT_OUT_MARKER,
    XDIST_INSERTION,
    apply,
    dispatch,
    plan,
)

_PYPROJECT_WITH_XDIST = """
[project]
name = "fixture"
version = "1.0.0"
dependencies = []

[project.optional-dependencies]
test = ["pytest-xdist>=3.8.0", "pytest>=9.1.1"]
"""

_PYPROJECT_WITHOUT_XDIST = """
[project]
name = "fixture"
version = "1.0.0"
dependencies = []

[project.optional-dependencies]
test = ["pytest>=9.1.1"]
"""

_PYPROJECT_XDIST_IN_DEPENDENCY_GROUPS = """
[project]
name = "fixture"
version = "1.0.0"
dependencies = []

[dependency-groups]
dev = ["pytest-xdist>=3.8.0", {include-group = "other"}]
other = ["pytest-timeout>=2.4.0"]
"""


def _pytest_hook_config(entry: str) -> str:
    return (
        "repos:\n"
        "- repo: local\n"
        "  hooks:\n"
        "  - id: pytest\n"
        "    name: pytest\n"
        "    entry: |-\n"
        + "\n".join(f"      {line}" for line in entry.splitlines())
        + "\n"
        "    language: system\n"
        "    pass_filenames: false\n"
    )


def _fixture_repo(
    tmp_path: Path,
    *,
    pyproject: str = _PYPROJECT_WITH_XDIST,
    entry: str | None = _BOILERPLATE_ENTRY,
    opt_out: bool = False,
    name: str = "repo",
) -> Path:
    repo = tmp_path / name
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "pyproject.toml").write_text(pyproject)
    if entry is not None:
        (repo / ".pre-commit-config.yaml").write_text(_pytest_hook_config(entry))
    if opt_out:
        (repo / OPT_OUT_MARKER).write_text("")
    return repo


# --------------------------------------------------------------------------- #
# Gate 1: pytest-xdist declared.
# --------------------------------------------------------------------------- #


def test_eligible_when_all_three_gates_pass(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    [entry] = plan([repo])
    assert entry["eligible"] is True
    assert entry["already_applied"] is False


def test_not_eligible_when_xdist_not_declared(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, pyproject=_PYPROJECT_WITHOUT_XDIST)
    [entry] = plan([repo])
    assert entry["eligible"] is False
    assert "pytest-xdist" in entry["reason"]


def test_not_eligible_when_no_pyproject_at_all(tmp_path: Path) -> None:
    repo = tmp_path / "no-pyproject"
    repo.mkdir()
    (repo / ".pre-commit-config.yaml").write_text(
        _pytest_hook_config(_BOILERPLATE_ENTRY)
    )
    [entry] = plan([repo])
    assert entry["eligible"] is False
    assert "pyproject" in entry["reason"]


def test_xdist_declared_via_dependency_groups_string_entry_counts(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, pyproject=_PYPROJECT_XDIST_IN_DEPENDENCY_GROUPS)
    [entry] = plan([repo])
    assert entry["eligible"] is True


def test_dependency_groups_include_group_reference_never_crashes_the_scan(
    tmp_path: Path,
) -> None:
    """A PEP 735 ``{"include-group": "..."}`` entry is not a requirement
    string; treating it as one would raise deep inside ``packaging`` instead
    of reporting a clean verdict."""

    pyproject = """
[project]
name = "fixture"
version = "1.0.0"
dependencies = []

[dependency-groups]
dev = [{include-group = "other"}]
other = ["pytest-timeout>=2.4.0"]
"""
    repo = _fixture_repo(tmp_path, pyproject=pyproject)
    [entry] = plan([repo])
    assert entry["eligible"] is False
    assert "pytest-xdist" in entry["reason"]


# --------------------------------------------------------------------------- #
# Gate 2: byte-identical boilerplate entry, no blind patch on a bespoke one.
# --------------------------------------------------------------------------- #


def test_bespoke_entry_is_skipped_never_blind_patched(tmp_path: Path) -> None:
    repo = _fixture_repo(
        tmp_path, entry='pytest "tests" -q --tb=short --custom-flag-nobody-else-has'
    )
    [entry] = plan([repo])
    assert entry["eligible"] is False
    assert entry["reason"] == "skipped: non-boilerplate entry, no blind patch"


def test_no_pytest_hook_at_all_is_not_eligible(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, entry=None)
    (repo / ".pre-commit-config.yaml").write_text(
        "repos:\n- repo: local\n  hooks:\n  - id: lint\n    entry: ruff check .\n"
        "    language: system\n"
    )
    [entry] = plan([repo])
    assert entry["eligible"] is False
    assert "pytest pre-commit hook" in entry["reason"]


def test_already_patched_entry_reports_eligible_and_already_applied(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, entry=_PATCHED_ENTRY)
    [entry] = plan([repo])
    assert entry["eligible"] is True
    assert entry["already_applied"] is True


# --------------------------------------------------------------------------- #
# Gate 3: opt-out marker.
# --------------------------------------------------------------------------- #


def test_opt_out_marker_makes_an_otherwise_eligible_repo_ineligible(
    tmp_path: Path,
) -> None:
    repo = _fixture_repo(tmp_path, opt_out=True)
    [entry] = plan([repo])
    assert entry["eligible"] is False
    assert OPT_OUT_MARKER in entry["reason"]


# --------------------------------------------------------------------------- #
# apply(): dry_run default, and the actual write path.
# --------------------------------------------------------------------------- #


def test_apply_default_dry_run_never_writes(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    before = (repo / ".pre-commit-config.yaml").read_text()

    [record] = apply([repo])
    assert record["action"] == "would_patch"
    assert (repo / ".pre-commit-config.yaml").read_text() == before


def test_apply_dry_run_false_patches_the_file(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)

    [record] = apply([repo], dry_run=False)
    assert record["action"] == "patched"

    patched_text = (repo / ".pre-commit-config.yaml").read_text()
    for token in XDIST_INSERTION:
        assert token in patched_text
    # The insertion appears once per pytest invocation branch (uv run + bare).
    assert patched_text.count(" ".join(XDIST_INSERTION)) == 2


def test_apply_is_idempotent_a_second_run_is_a_noop(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    apply([repo], dry_run=False)
    patched_once = (repo / ".pre-commit-config.yaml").read_text()

    [record] = apply([repo], dry_run=False)
    assert record["action"] == "noop_already_applied"
    assert (repo / ".pre-commit-config.yaml").read_text() == patched_once


def test_apply_skips_ineligible_repos_without_touching_them(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path, pyproject=_PYPROJECT_WITHOUT_XDIST)
    before = (repo / ".pre-commit-config.yaml").read_text()

    [record] = apply([repo], dry_run=False)
    assert record["action"] == "skipped"
    assert (repo / ".pre-commit-config.yaml").read_text() == before


def test_apply_over_multiple_repos_each_gets_its_own_verdict(tmp_path: Path) -> None:
    eligible = _fixture_repo(tmp_path, name="eligible")
    bespoke = _fixture_repo(
        tmp_path, name="bespoke", entry='pytest "tests" --custom-flag-xyz'
    )
    no_xdist = _fixture_repo(
        tmp_path, name="no-xdist", pyproject=_PYPROJECT_WITHOUT_XDIST
    )

    results = apply([eligible, bespoke, no_xdist], dry_run=False)
    by_repo = {Path(r["repo"]).name: r["action"] for r in results}
    assert by_repo["eligible"] == "patched"
    assert by_repo["bespoke"] == "skipped"
    assert by_repo["no-xdist"] == "skipped"


# --------------------------------------------------------------------------- #
# dispatch()
# --------------------------------------------------------------------------- #


def test_dispatch_plan_reports_eligible_count(tmp_path: Path) -> None:
    eligible = _fixture_repo(tmp_path, name="eligible")
    ineligible = _fixture_repo(
        tmp_path, name="ineligible", pyproject=_PYPROJECT_WITHOUT_XDIST
    )
    result = dispatch("plan", repo_paths=[str(eligible), str(ineligible)])
    assert result["ok"] is True
    assert result["repos_checked"] == 2
    assert result["eligible"] == 1


def test_dispatch_status_is_a_read_only_alias_for_plan(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    plan_result = dispatch("plan", repo_paths=[str(repo)])
    status_result = dispatch("status", repo_paths=[str(repo)])
    assert plan_result["entries"] == status_result["entries"]
    # status must never write anything either.
    before = (repo / ".pre-commit-config.yaml").read_text()
    dispatch("status", repo_paths=[str(repo)])
    assert (repo / ".pre-commit-config.yaml").read_text() == before


def test_dispatch_apply_defaults_to_dry_run(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    before = (repo / ".pre-commit-config.yaml").read_text()
    result = dispatch("apply", repo_paths=[str(repo)])
    assert result["ok"] is True
    assert result["dry_run"] is True
    assert result["would_patch"] == 1
    assert (repo / ".pre-commit-config.yaml").read_text() == before


def test_dispatch_apply_explicit_dry_run_false_writes(tmp_path: Path) -> None:
    repo = _fixture_repo(tmp_path)
    result = dispatch("apply", repo_paths=[str(repo)], dry_run=False)
    assert result["ok"] is True
    assert result["patched"] == 1
    assert " ".join(XDIST_INSERTION) in (repo / ".pre-commit-config.yaml").read_text()


def test_dispatch_requires_repo_paths_for_plan_and_apply() -> None:
    assert dispatch("plan") == {"ok": False, "error": "plan requires repo_paths"}
    assert dispatch("apply") == {"ok": False, "error": "apply requires repo_paths"}


def test_dispatch_unknown_action() -> None:
    assert dispatch("bogus") == {"ok": False, "error": "unknown action: bogus"}
