"""Bounded fleet dependency evidence preserves unresolved facts and source bytes."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from repository_manager.development.fleet_census import (
    CensusSource,
    EdgeObservation,
    _canonical,
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


def test_rehashed_forged_observation_is_not_a_scanner_receipt(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(tmp_path, "ui/package.json", '{"name":"ui","dependencies":{}}')
    receipt = capture_census(
        tmp_path,
        "workspace.yml",
        (CensusSource("ui/package.json", "frontend_package"),),
        source_commit=REVISION,
    )
    forged = replace(
        receipt,
        observations=(
            EdgeObservation(
                "npm:ui",
                "npm:secret",
                "depends_on",
                "production_runtime_package",
                "manifest_declared",
                "declared",
                "ui/package.json",
                "dependencies.secret",
            ),
        ),
    )
    forged = replace(
        forged, digest=hashlib.sha256(_canonical(forged.payload())).hexdigest()
    )
    assert not verify_census(tmp_path, "workspace.yml", forged)


def test_census_refuses_source_over_bound(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(tmp_path, "ui/package.json", "x" * (2 * 1024 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded size"):
        capture_census(
            tmp_path,
            "workspace.yml",
            (CensusSource("ui/package.json", "frontend_package"),),
            source_commit=REVISION,
        )


def test_rust_and_frontend_source_edges_keep_dependency_class(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(
        tmp_path,
        "engine/Cargo.toml",
        '[package]\nname="engine"\nversion="1.0.0"\n'
        '[dependencies]\ncore_alias={package="eg-core",path="../eg-core"}\n'
        '[dev-dependencies]\neg-test="1"\n'
        '[build-dependencies]\neg-codegen="1"\n'
        "[target.'cfg(unix)'.dependencies]\neg-unix=\"1\"\n",
    )
    _write(tmp_path, "engine/src/lib.rs", "use eg_core::Graph;\nuse std::path::Path;\n")
    _write(
        tmp_path,
        "ui/src/app.ts",
        'import client from "@scope/client/view";\nimport "polyfill";\n',
    )
    receipt = capture_census(
        tmp_path,
        "workspace.yml",
        (
            CensusSource("engine/Cargo.toml", "rust_manifest"),
            CensusSource("engine/src/lib.rs", "rust_source"),
            CensusSource("ui/src/app.ts", "frontend_source"),
        ),
        source_commit=REVISION,
    )
    assert {(edge.target_id, edge.edge_class) for edge in receipt.observations} == {
        ("rust:eg-core", "production_runtime_package"),
        ("rust:eg-test", "development_test"),
        ("rust:eg-codegen", "build_generation"),
        ("rust:eg-unix", "production_runtime_package"),
        ("npm:@scope/client", "production_runtime_package"),
        ("npm:polyfill", "production_runtime_package"),
    }
    assert receipt.unresolved_count == 0


def test_frontend_dynamic_import_remains_unresolved(tmp_path: Path) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(tmp_path, "ui/src/app.ts", "const plugin = import(name);\n")
    receipt = capture_census(
        tmp_path,
        "workspace.yml",
        (CensusSource("ui/src/app.ts", "frontend_source"),),
        source_commit=REVISION,
    )
    assert receipt.unresolved_count == 1
    assert receipt.observations[0].target_id is None


def test_cargo_workspace_alias_without_package_resolution_stays_unresolved(
    tmp_path: Path,
) -> None:
    _write(tmp_path, "workspace.yml", "repositories: []\n")
    _write(
        tmp_path,
        "engine/Cargo.toml",
        '[package]\nname="engine"\nversion="1.0.0"\n'
        "[dependencies]\nbackend={workspace=true}\n",
    )
    receipt = capture_census(
        tmp_path,
        "workspace.yml",
        (CensusSource("engine/Cargo.toml", "rust_manifest"),),
        source_commit=REVISION,
    )
    assert receipt.unresolved_count == 1
