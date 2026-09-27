"""A manifest cannot attest to its own complete source coverage."""

from __future__ import annotations

import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from repository_manager.development.fleet_census import verify_census
from repository_manager.development.fleet_manifest import (
    capture_complete_manifest_census,
    capture_manifest_census,
    reconcile_fleet_census,
)
from repository_manager.development.fleet_source_universe import (
    capture_source_universe,
    verify_source_universe,
)
from repository_manager.development.workspace_release import (
    FleetCycleError,
    ProjectRecord,
    build_dependency_graph,
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


def _fixture(root: Path) -> Path:
    (root / "workspace.yml").write_text(
        "repositories:\n"
        "  - url: https://example.test/app.git\n"
        "    id: repo:app\n    path: app\n    plane: agent\n"
        "    layer: products\n    fleet_ordinal: L4_products\n"
        "    provides: []\n    consumes: []\n"
        "    dependency_classes: [package, deployment]\n"
        "    manifests: [pyproject.toml, package.json, compose.yml, Cargo.toml]\n"
        "    discovery_sources: [runtime.py, src/lib.rs, src/index.ts]\n"
    )
    repository = root / "app"
    repository.mkdir()
    (repository / "pyproject.toml").write_text('[project]\nname="app"\n')
    (repository / "package.json").write_text('{"name":"app"}')
    (repository / "Cargo.toml").write_text('[package]\nname="app"\nversion="1.0.0"\n')
    (repository / "compose.yml").write_text("services:\n  app:\n    image: app\n")
    (repository / "src").mkdir()
    (repository / "src/lib.rs").write_text("pub fn app() {}\n")
    (repository / "src/index.ts").write_text("export const app = 1;\n")
    (repository / "runtime.py").write_text(
        'import importlib\nimportlib.import_module("app")\n'
    )
    _git(repository, "init", "-q")
    _git(
        repository,
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Test",
        "add",
        "--",
        ".",
    )
    _git(
        repository,
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Test",
        "commit",
        "-qm",
        "fixture",
    )
    return repository


def test_clean_git_tree_can_independently_certify_source_universe(
    tmp_path: Path,
) -> None:
    repository = _fixture(tmp_path)
    closure, census, universe = capture_complete_manifest_census(
        tmp_path, source_commit="a" * 40
    )
    assert census.complete is True
    assert census.source_universe_digest == universe.digest
    assert len(universe.source_paths) == 7
    assert verify_census(
        tmp_path, "workspace.yml", census, closure=closure, universe=universe
    )
    assert not verify_census(tmp_path, "workspace.yml", census)
    assert not verify_source_universe(
        tmp_path, closure, replace(universe, digest="0" * 64)
    )
    (repository / "runtime.py").write_text(
        "import importlib\nimportlib.import_module(name)\n"
    )
    assert not verify_source_universe(tmp_path, closure, universe)
    assert not verify_census(
        tmp_path, "workspace.yml", census, closure=closure, universe=universe
    )


def test_extra_tracked_source_refuses_manifest_self_attestation(tmp_path: Path) -> None:
    repository = _fixture(tmp_path)
    (repository / "hidden.py").write_text("import importlib\n")
    _git(repository, "add", "--", "hidden.py")
    _git(
        repository,
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Test",
        "commit",
        "-qm",
        "extra",
    )
    closure, census = capture_manifest_census(tmp_path, source_commit="a" * 40)
    assert census.complete is False
    with pytest.raises(ValueError, match="declared source set differs"):
        capture_source_universe(tmp_path, closure)


def test_unsupported_tracked_language_refuses_complete_proof(tmp_path: Path) -> None:
    repository = _fixture(tmp_path)
    (repository / "engine.go").write_text("package main\n")
    _git(repository, "add", "--", "engine.go")
    _git(
        repository,
        "-c",
        "user.email=test@example.test",
        "-c",
        "user.name=Test",
        "commit",
        "-qm",
        "rust",
    )
    closure, _ = capture_manifest_census(tmp_path, source_commit="a" * 40)
    with pytest.raises(ValueError, match="does not support engine.go"):
        capture_source_universe(tmp_path, closure)


def test_verified_git_tree_still_refuses_rust_product_scc(tmp_path: Path) -> None:
    (tmp_path / "workspace.yml").write_text(
        "repositories:\n"
        "  - url: https://example.test/a.git\n"
        "    id: repo:a\n    path: a\n    plane: engine\n    layer: products\n"
        "    fleet_ordinal: L4_products\n    provides: []\n    consumes: []\n"
        "    dependency_classes: [package, deployment]\n"
        "    manifests: [Cargo.toml, pyproject.toml, package.json, compose.yml]\n"
        "    discovery_sources: [runtime.py, src/lib.rs, src/app.ts]\n"
        "    artifact_ids: [rust:a, python:a, npm:a, service:a]\n"
        "  - url: https://example.test/b.git\n"
        "    id: repo:b\n    path: b\n    plane: engine\n    layer: products\n"
        "    fleet_ordinal: L4_products\n    provides: []\n    consumes: []\n"
        "    dependency_classes: [package]\n"
        "    manifests: [Cargo.toml]\n"
        "    artifact_ids: [rust:b]\n"
    )
    for name, other in (("a", "b"), ("b", "a")):
        repository = tmp_path / name
        (repository / "src").mkdir(parents=True)
        (repository / "Cargo.toml").write_text(
            f'[package]\nname="{name}"\nversion="1.0.0"\n'
            f'[dependencies]\n{other}_alias={{package="{other}",path="../{other}"}}\n'
        )
        if name == "a":
            (repository / "src/lib.rs").write_text(
                "use b_alias::run;\npub fn run() {}\n"
            )
            (repository / "pyproject.toml").write_text('[project]\nname="a"\n')
            (repository / "package.json").write_text('{"name":"a"}')
            (repository / "compose.yml").write_text("services:\n  a:\n    image: a\n")
            (repository / "runtime.py").write_text("def run(): pass\n")
            (repository / "src/app.ts").write_text("export const run = 1;\n")
        _git(repository, "init", "-q")
        _git(repository, "add", "--", ".")
        _git(
            repository,
            "-c",
            "user.email=test@example.test",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "fixture",
        )
    closure, census, universe = capture_complete_manifest_census(
        tmp_path, source_commit="a" * 40
    )
    assert census.complete
    graph = build_dependency_graph((ProjectRecord("a"), ProjectRecord("b")))
    with pytest.raises(FleetCycleError) as caught:
        reconcile_fleet_census(tmp_path, closure, census, graph, universe=universe)
    assert caught.value.components[0].members == ("repo:a", "repo:b")
