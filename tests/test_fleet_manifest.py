"""Fleet manifest closure requires every nested repository and declared source."""

from dataclasses import replace
from pathlib import Path

import pytest

from repository_manager.development.fleet_manifest import (
    capture_manifest_census,
    load_fleet_manifest,
    parse_fleet_manifest,
    reconcile_fleet_census,
    validate_fleet_manifest_mirrors,
    validated_declared_source_receipts,
)
from repository_manager.development.workspace_release import (
    FleetCycleError,
    FleetEvidenceKind,
    ProjectRecord,
    build_dependency_graph,
)
from repository_manager.workspace_manifest import synchronize_workspace_manifest

REVISION = "b" * 40


def _workspace(root: Path, *, second_metadata: str) -> None:
    (root / "workspace.yml").write_text(
        "repositories:\n"
        "  - url: https://example.test/root.git\n"
        "    id: repo:root\n"
        "    path: root\n"
        "    plane: agent\n"
        "    layer: products\n"
        "    fleet_ordinal: L4_products\n"
        "    provides: [capability/root]\n"
        "    consumes: []\n"
        "    dependency_classes: [package]\n"
        "    manifests: [pyproject.toml]\n"
        "subdirectories:\n"
        "  ui:\n"
        "    repositories:\n"
        "      - url: https://example.test/app.git\n"
        f"{second_metadata}"
    )


def test_nested_manifest_captures_every_declared_repository(tmp_path: Path) -> None:
    _workspace(
        tmp_path,
        second_metadata=(
            "        id: repo:ui/app\n"
            "        path: ui/app\n"
            "        plane: frontend\n"
            "        layer: products\n"
            "        fleet_ordinal: L4_products\n"
            "        provides: [capability/ui]\n"
            "        consumes: []\n"
            "        dependency_classes: [package]\n"
            "        manifests: [package.json]\n"
            "        discovery_sources: [src/loader.py]\n"
        ),
    )
    (tmp_path / "root").mkdir()
    (tmp_path / "root/pyproject.toml").write_text('[project]\nname="root"\n')
    (tmp_path / "ui/app/src").mkdir(parents=True)
    (tmp_path / "ui/app/package.json").write_text('{"name":"app"}')
    (tmp_path / "ui/app/src/loader.py").write_text(
        "import importlib\nimportlib.import_module(name)\n"
    )
    closure, receipt = capture_manifest_census(tmp_path, source_commit=REVISION)
    assert closure.declared_repository_count == 2
    assert [row.repository_id for row in closure.repositories] == [
        "repo:root",
        "repo:ui/app",
    ]
    assert len(closure.sources) == 3
    assert receipt.unresolved_count == 1
    assert receipt.complete is False


def test_legacy_manifest_cannot_assert_fleet_closure(tmp_path: Path) -> None:
    _workspace(tmp_path, second_metadata="")
    with pytest.raises(ValueError, match="lacks fleet fields"):
        load_fleet_manifest(tmp_path)


def test_manifest_refuses_foreign_identity_and_source_escape(tmp_path: Path) -> None:
    _workspace(
        tmp_path,
        second_metadata=(
            "        id: repo:elsewhere\n"
            "        path: ui/app\n"
            "        plane: frontend\n"
            "        layer: products\n"
            "        fleet_ordinal: L4_products\n"
            "        provides: []\n"
            "        consumes: []\n"
            "        dependency_classes: [package]\n"
            "        manifests: [package.json]\n"
        ),
    )
    with pytest.raises(ValueError, match="conflicting fleet identity"):
        load_fleet_manifest(tmp_path)
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        manifest.read_text()
        .replace("repo:elsewhere", "repo:ui/app")
        .replace("[package.json]", "[../outside.json]")
    )
    with pytest.raises(ValueError, match="unsafe repository source path"):
        load_fleet_manifest(tmp_path)


def test_explicit_unknown_metadata_keeps_scope_unresolved(tmp_path: Path) -> None:
    _workspace(
        tmp_path,
        second_metadata=(
            "        id: repo:ui/app\n"
            "        path: ui/app\n"
            "        plane: frontend\n"
            "        layer: unknown\n"
            "        fleet_ordinal: unknown\n"
            "        provides: []\n"
            "        consumes: []\n"
            "        dependency_classes: [package]\n"
            "        manifests: [package.json]\n"
            "        metadata_state: unresolved\n"
            "        unresolved_fields: [layer, fleet_ordinal, dependency_classes]\n"
        ),
    )
    scope = parse_fleet_manifest((tmp_path / "workspace.yml").read_bytes())
    assert scope.unresolved_repository_ids == ("repo:ui/app",)
    assert scope.declared_repository_count == 2
    assert scope.repositories[-1].dependency_classes == ("package",)


def test_strict_fleet_mirror_preflight_binds_exact_projections(tmp_path: Path) -> None:
    _workspace(
        tmp_path,
        second_metadata=(
            "        id: repo:ui/app\n"
            "        path: ui/app\n"
            "        plane: frontend\n"
            "        layer: products\n"
            "        fleet_ordinal: L4_products\n"
            "        provides: []\n"
            "        consumes: []\n"
            "        dependency_classes: [package]\n"
            "        manifests: [package.json]\n"
        ),
    )
    source = tmp_path / "workspace.yml"
    source.write_text(f'path: "{tmp_path}"\n' + source.read_text())
    runtime = tmp_path / "runtime.yml"
    seed = tmp_path / "seed.yml"
    synchronize_workspace_manifest(
        source, runtime_destination=runtime, seed_destination=seed
    )
    receipt = validate_fleet_manifest_mirrors(source, runtime=runtime, seed=seed)
    assert receipt.source_digest == receipt.runtime_digest
    assert receipt.repository_ids == ("repo:root", "repo:ui/app")
    runtime.write_text(runtime.read_text() + "\n")
    with pytest.raises(ValueError, match="mirrors differ"):
        validate_fleet_manifest_mirrors(source, runtime=runtime, seed=seed)


def test_strict_mirror_refuses_unresolved_metadata(tmp_path: Path) -> None:
    _workspace(
        tmp_path,
        second_metadata=(
            "        id: repo:ui/app\n"
            "        path: ui/app\n"
            "        plane: frontend\n"
            "        layer: unknown\n"
            "        fleet_ordinal: unknown\n"
            "        provides: []\n"
            "        consumes: []\n"
            "        dependency_classes: [package]\n"
            "        manifests: [package.json]\n"
            "        metadata_state: unresolved\n"
            "        unresolved_fields: [layer, fleet_ordinal]\n"
        ),
    )
    source = tmp_path / "workspace.yml"
    source.write_text(f'path: "{tmp_path}"\n' + source.read_text())
    runtime = tmp_path / "runtime.yml"
    seed = tmp_path / "seed.yml"
    synchronize_workspace_manifest(
        source, runtime_destination=runtime, seed_destination=seed
    )
    with pytest.raises(ValueError, match="metadata remains unresolved"):
        validate_fleet_manifest_mirrors(source, runtime=runtime, seed=seed)


def test_fleet_manifest_refuses_unsupported_repository_fields(tmp_path: Path) -> None:
    _workspace(tmp_path, second_metadata="")
    source = tmp_path / "workspace.yml"
    source.write_text(
        source.read_text().replace(
            "    manifests: [pyproject.toml]",
            "    manifests: [pyproject.toml]\n    shortcut: true",
        )
    )
    with pytest.raises(ValueError, match="unsupported fleet fields"):
        load_fleet_manifest(tmp_path)


def test_declared_source_receipts_require_all_four_reextracted_classes(
    tmp_path: Path,
) -> None:
    (tmp_path / "workspace.yml").write_text(
        "repositories:\n"
        "  - url: https://example.test/root.git\n"
        "    id: repo:root\n"
        "    path: root\n"
        "    plane: composition\n"
        "    layer: products\n"
        "    fleet_ordinal: L4_products\n"
        "    provides: []\n"
        "    consumes: []\n"
        "    dependency_classes: [package, deployment]\n"
        "    manifests: [pyproject.toml, compose.yml, package.json]\n"
        "    discovery_sources: [runtime.py]\n"
    )
    root = tmp_path / "root"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="root"\n')
    (root / "compose.yml").write_text("services:\n  api:\n    image: api\n")
    (root / "package.json").write_text('{"name":"ui","dependencies":{}}')
    (root / "runtime.py").write_text(
        'import importlib\nimportlib.import_module("known")\n'
    )
    closure, receipt = capture_manifest_census(tmp_path, source_commit=REVISION)
    digests = validated_declared_source_receipts(tmp_path, closure, receipt)
    assert set(digests) == set(FleetEvidenceKind)
    assert all(len(digest) == 64 for digest in digests.values())
    with pytest.raises(ValueError, match="closure differs"):
        validated_declared_source_receipts(
            tmp_path,
            replace(closure, third_party_artifact_ids=("npm:forged",)),
            receipt,
        )
    (root / "package.json").write_text('{"name":"ui","dependencies":{"new":"1"}}')
    with pytest.raises(ValueError, match="re-extraction"):
        validated_declared_source_receipts(tmp_path, closure, receipt)


def test_reconciliation_refuses_complete_two_repo_scc(tmp_path: Path) -> None:
    (tmp_path / "workspace.yml").write_text(
        "repositories:\n"
        "  - url: https://example.test/a.git\n"
        "    id: repo:a\n    path: a\n    plane: agent\n    layer: products\n"
        "    fleet_ordinal: L4_products\n    provides: []\n    consumes: []\n"
        "    dependency_classes: [package, deployment]\n"
        "    manifests: [pyproject.toml, compose.yml, package.json]\n"
        "    discovery_sources: [runtime.py]\n"
        "    artifact_ids: [python:a, service:api, npm:ui, module:known]\n"
        "  - url: https://example.test/b.git\n"
        "    id: repo:b\n    path: b\n    plane: agent\n    layer: products\n"
        "    fleet_ordinal: L4_products\n    provides: []\n    consumes: []\n"
        "    dependency_classes: [package]\n    manifests: [pyproject.toml]\n"
        "    artifact_ids: [python:b]\n"
    )
    for name, other in (("a", "b"), ("b", "a")):
        repo = tmp_path / name
        repo.mkdir()
        (repo / "pyproject.toml").write_text(
            f'[project]\nname="{name}"\ndependencies=["{other}>=1"]\n'
            f'[tool.uv.sources]\n{other}={{path="../{other}"}}\n'
        )
    (tmp_path / "a/compose.yml").write_text("services:\n  api:\n    image: api\n")
    (tmp_path / "a/package.json").write_text('{"name":"ui","dependencies":{}}')
    (tmp_path / "a/runtime.py").write_text(
        'import importlib\nimportlib.import_module("known")\n'
    )
    closure, receipt = capture_manifest_census(tmp_path, source_commit=REVISION)
    graph = build_dependency_graph((ProjectRecord("a"), ProjectRecord("b")))
    with pytest.raises(FleetCycleError) as caught:
        reconcile_fleet_census(tmp_path, closure, receipt, graph)
    assert caught.value.components[0].members == ("repo:a", "repo:b")
    assert caught.value.components[0].edges == (
        ("repo:a", "repo:b"),
        ("repo:b", "repo:a"),
    )
    (tmp_path / "b/pyproject.toml").write_text('[project]\nname="b"\ndependencies=[]\n')
    closure, receipt = capture_manifest_census(tmp_path, source_commit=REVISION)
    order = reconcile_fleet_census(tmp_path, closure, receipt, graph)
    assert order.parallel_groups == (("repo:b",), ("repo:a",))
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        manifest.read_text().replace(
            "id: repo:b\n    path: b\n    plane: agent\n    layer: products\n"
            "    fleet_ordinal: L4_products",
            "id: repo:b\n    path: b\n    plane: agent\n    layer: products\n"
            "    fleet_ordinal: L5_composition",
        )
    )
    closure, receipt = capture_manifest_census(tmp_path, source_commit=REVISION)
    with pytest.raises(ValueError, match="higher layer"):
        reconcile_fleet_census(tmp_path, closure, receipt, graph)
