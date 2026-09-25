"""Bounded fleet dependency evidence preserves unresolved facts and source bytes."""

import json
from pathlib import Path

import pytest

from repository_manager.development.fleet_census import (
    CensusSource,
    capture_census,
    verify_census,
)

REVISION = "a" * 40


def _write(root: Path, name: str, content: str) -> None:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def test_bounded_census_captures_four_evidence_families(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(
        tmp_path,
        "service/pyproject.toml",
        '[project]\nname="service"\ndependencies=["agent-utilities>=1"]\n'
        '[tool.uv.sources]\nagent-utilities={path="../au"}\n'
        'missing-package={path="../missing"}\n',
    )
    _write(
        tmp_path,
        "service/runtime.py",
        'import importlib\nimportlib.import_module("known.plugin")\n'
        "importlib.import_module(variable_name)\n",
    )
    _write(
        tmp_path,
        "deploy/compose.yml",
        "services:\n  api:\n    depends_on: [database]\n  database:\n    image: postgres\n",
    )
    _write(
        tmp_path,
        "ui/package.json",
        json.dumps(
            {
                "name": "ui",
                "dependencies": {"react": "^19"},
                "devDependencies": {"vitest": "^3"},
            }
        ),
    )
    sources = (
        CensusSource("service/pyproject.toml", "python_project"),
        CensusSource("service/runtime.py", "python_source"),
        CensusSource("deploy/compose.yml", "compose"),
        CensusSource("ui/package.json", "frontend_package"),
    )
    receipt = capture_census(tmp_path, "workspace.yml", sources, source_commit=REVISION)
    assert len(receipt.observations) == 7
    assert receipt.unresolved_count == 2
    assert receipt.missing_kinds == ()
    assert receipt.complete is False
    assert any(edge.target_id == "service:database" for edge in receipt.observations)
    assert any(edge.target_id == "npm:react" for edge in receipt.observations)
    assert (
        receipt.to_json()
        == capture_census(
            tmp_path, "workspace.yml", tuple(reversed(sources)), source_commit=REVISION
        ).to_json()
    )
    assert verify_census(tmp_path, "workspace.yml", receipt)


def test_receipt_refuses_changed_artifact(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(tmp_path, "ui/package.json", '{"name":"ui","dependencies":{}}')
    receipt = capture_census(
        tmp_path,
        "workspace.yml",
        (CensusSource("ui/package.json", "frontend_package"),),
        source_commit=REVISION,
    )
    assert receipt.missing_kinds == ("python_project", "python_source", "compose")
    _write(tmp_path, "ui/package.json", '{"name":"ui","dependencies":{"react":"19"}}')
    assert not verify_census(tmp_path, "workspace.yml", receipt)
    (tmp_path / "ui/package.json").unlink()
    assert not verify_census(tmp_path, "workspace.yml", receipt)


def test_dynamic_imports_require_real_loader_bindings(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(
        tmp_path,
        "service/runtime.py",
        "import importlib as loader\nfrom importlib import import_module as load\n"
        'loader.import_module("known.module")\nload(variable)\n'
        'def import_module(value): return value\nimport_module("not.a.loader")\n',
    )
    receipt = capture_census(
        tmp_path,
        "workspace.yml",
        (CensusSource("service/runtime.py", "python_source"),),
        source_commit=REVISION,
    )
    assert len(receipt.observations) == 2
    assert receipt.unresolved_count == 1


def test_source_must_stay_inside_workspace_and_be_unique(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(tmp_path, "ui/package.json", '{"name":"ui"}')
    source = CensusSource("ui/package.json", "frontend_package")
    with pytest.raises(ValueError, match="unique"):
        capture_census(
            tmp_path, "workspace.yml", (source, source), source_commit=REVISION
        )
    with pytest.raises(ValueError, match="escapes workspace"):
        capture_census(
            tmp_path,
            "workspace.yml",
            (CensusSource("../elsewhere.json", "frontend_package"),),
            source_commit=REVISION,
        )
