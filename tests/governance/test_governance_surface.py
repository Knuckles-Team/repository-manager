"""The repository-neutral seams added when governance moved here.

agent-utilities hard-coded itself into these modules (its package directory as
the only concept-marker root, its install location as the ledger fallback, its
repo-relative registry paths, its package identity as the generated-view
trigger). Each test pins the repository-neutral replacement.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from repository_manager.governance import cli as gov_cli
from repository_manager.governance import concept_allocator as ca
from repository_manager.governance import concept_hierarchy as ch
from repository_manager.governance import concept_lineage as cl
from repository_manager.governance import lane_guard, promotion


def _package(root: Path, name: str) -> Path:
    package = root / name
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    return package


def test_scan_roots_are_the_repositorys_own_packages(tmp_path: Path) -> None:
    _package(tmp_path, "repository_manager")
    _package(tmp_path, "tests")
    (tmp_path / "docs").mkdir()

    roots = ca._default_scan_roots(tmp_path)

    assert roots == [tmp_path / "repository_manager", tmp_path / "crates"]


def test_reconcile_lands_a_marker_in_a_non_au_package(tmp_path: Path) -> None:
    package = _package(tmp_path, "repository_manager")
    (tmp_path / "docs").mkdir()
    (tmp_path / "docs" / "concepts.yaml").write_text(
        yaml.safe_dump({"concepts": []}), encoding="utf-8"
    )
    concept_id = "RM-OS.governance.neutral-scan"
    ca.reserve_concept_id(concept_id, session_id="s", repo_root=tmp_path)
    (package / "feature.py").write_text(f"# CONCEPT:{concept_id}\n", encoding="utf-8")

    assert ca.reconcile(repo_root=tmp_path)["landed"] == [concept_id]


def test_registry_paths_are_supplied_by_the_governed_repository(
    tmp_path: Path,
) -> None:
    concepts = tmp_path / ch.CONCEPTS_YAML_RELPATH
    concepts.parent.mkdir(parents=True)
    concepts.write_text(
        yaml.safe_dump({"concepts": [{"id": "AU-KG.compute.a"}]}), encoding="utf-8"
    )
    assert ch.total_concept_count(concepts) == 1
    assert cl.load_lineage(tmp_path / cl.LINEAGE_RELPATH) == cl.Lineage(
        parents={}, retired={}
    )


def test_generated_view_check_follows_the_fragment_directory(tmp_path: Path) -> None:
    staged = [lane_guard.LEDGER_VIEW]
    (tmp_path / "docs").mkdir()

    assert lane_guard._should_check_generated_view(tmp_path, staged) is False
    (tmp_path / "docs" / ca.FRAGMENT_DIRNAME).mkdir()
    assert lane_guard._should_check_generated_view(tmp_path, staged) is True
    assert lane_guard._should_check_generated_view(tmp_path, []) is False


def test_cli_resolves_a_concept_id(capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = gov_cli.main(["concept", "resolve", "--id", "AU-KG.compute.a"])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert (out["slug"], out["pillar"], out["domain"]) == ("AU", "KG", "compute")


def test_cli_reports_a_missing_id_without_mutating(
    capsys: pytest.CaptureFixture[str],
) -> None:
    exit_code = gov_cli.main(["concept", "reserve"])

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {"error": "reserve requires --id"}


def test_console_script_is_declared() -> None:
    pyproject = Path(gov_cli.__file__).resolve().parents[2] / "pyproject.toml"
    text = pyproject.read_text(encoding="utf-8")
    assert (
        'repository-manager-governance = "repository_manager.governance.cli:main"'
        in text
    )


def _git(args: list[str], cwd: Path) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture
def canonical(tmp_path: Path) -> Path:
    root = tmp_path / "canonical"
    root.mkdir()
    _git(["init", "-q", "-b", "main"], root)
    _git(["config", "user.email", "promotion@test"], root)
    _git(["config", "user.name", "Promotion Test"], root)
    (root / "a.txt").write_text("a\n", encoding="utf-8")
    _git(["add", "a.txt"], root)
    _git(["commit", "-qm", "base"], root)
    return root


def test_promotion_reports_an_undecoupled_fleet_as_undecoupled(
    canonical: Path,
) -> None:
    state = promotion.promotion_state(canonical)
    assert state["decoupled"] is False
    assert "armed deploy" in state["reason"]


def test_promotion_counts_merges_not_yet_promoted(
    canonical: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _git(["update-ref", promotion.PROMOTION_REF, "main"], canonical)
    (canonical / "b.txt").write_text("b\n", encoding="utf-8")
    _git(["add", "b.txt"], canonical)
    _git(["commit", "-qm", "merged, not promoted"], canonical)

    assert gov_cli.main(["promotion", "--path", str(canonical)]) == 0
    state = json.loads(capsys.readouterr().out)
    assert state["decoupled"] is True
    assert state["unpromoted_commits"] == 1
    # The deployed ref did NOT move: merging is not deploying.
    assert state["deployed"] != state["main"]
