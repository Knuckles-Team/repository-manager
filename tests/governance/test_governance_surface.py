"""The repository-neutral seams OQ-3 added when governance moved here.

agent-utilities hard-coded itself into these modules (its package directory as
the only concept-marker root, its install location as the ledger fallback, its
repo-relative registry paths, its package identity as the generated-view
trigger). Each test pins the repository-neutral replacement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from repository_manager.governance import cli as gov_cli
from repository_manager.governance import concept_allocator as ca
from repository_manager.governance import concept_hierarchy as ch
from repository_manager.governance import concept_lineage as cl
from repository_manager.governance import lane_guard


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
