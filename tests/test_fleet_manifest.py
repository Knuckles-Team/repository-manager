"""Fleet manifest closure requires every nested repository and declared source."""

from pathlib import Path

import pytest

from repository_manager.development.fleet_manifest import (
    capture_manifest_census,
    load_fleet_manifest,
    parse_fleet_manifest,
)

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
