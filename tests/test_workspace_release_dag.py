"""Adversarial checkpoint-1 tests for the pure workspace release seam."""

from __future__ import annotations

import json
import random
import socket
import subprocess
from pathlib import Path

import pytest

from repository_manager.development.workspace_metadata import (
    MetadataError,
    MetadataLimits,
    OverlayInput,
    read_node_metadata,
    read_python_metadata,
    read_rust_metadata,
)
from repository_manager.development.workspace_release import (
    DependencyEdge,
    DependencySpec,
    Ecosystem,
    GraphDiagnosticCode,
    GraphValidationError,
    PackageKey,
    PackageRecord,
    PackageReference,
    PlanStage,
    ProjectRecord,
    ReleaseStage,
    Version,
    VersionFloor,
    VersionSource,
    WorkspaceReleaseError,
    WorkspaceReleasePlan,
    build_dependency_graph,
    phase_manifest_from_mapping,
    plan_digest,
)


def _package(
    repository: str,
    name: str,
    *,
    ecosystem: Ecosystem = Ecosystem.PYTHON,
    dependencies: tuple[DependencySpec, ...] = (),
    version: str = "1.0.0",
    sources: tuple[VersionSource, ...] | None = None,
) -> PackageRecord:
    key = PackageKey(repository, ecosystem, name)
    selected_version = Version(version)
    return PackageRecord(
        key=key,
        version=selected_version,
        version_sources=sources or (VersionSource("fixture", selected_version),),
        dependencies=dependencies,
    )


def _project(repository: str, *packages: PackageRecord) -> ProjectRecord:
    return ProjectRecord(repository_id=repository, packages=packages)


def _edge_spec(
    name: str,
    *,
    ecosystem: Ecosystem = Ecosystem.PYTHON,
    repository: str | None = None,
    floor: str | None = None,
) -> DependencySpec:
    return DependencySpec(
        target=PackageReference(ecosystem, name, repository),
        floor=VersionFloor.parse(floor) if floor else None,
        source="fixture",
    )


def _codes(error: GraphValidationError) -> set[GraphDiagnosticCode]:
    return {diagnostic.code for diagnostic in error.diagnostics}


def test_same_basename_is_not_an_identity_and_ambiguous_owner_refuses() -> None:
    first = _project("services/foo", _package("services/foo", "foo"))
    second = _project("agent-packages/foo", _package("agent-packages/foo", "foo"))
    consumer = _project(
        "apps/consumer",
        _package(
            "apps/consumer",
            "consumer",
            dependencies=(_edge_spec("foo"),),
        ),
    )

    with pytest.raises(GraphValidationError) as captured:
        build_dependency_graph([consumer, second, first])

    assert GraphDiagnosticCode.AMBIGUOUS_PACKAGE_OWNER in _codes(captured.value)
    assert first.project_id != second.project_id
    assert first.packages[0].key.value != second.packages[0].key.value


def test_explicit_repository_identity_resolves_same_basename_packages() -> None:
    first = _project("services/foo", _package("services/foo", "foo"))
    second = _project("agent-packages/foo", _package("agent-packages/foo", "foo"))
    consumer = _project(
        "apps/consumer",
        _package(
            "apps/consumer",
            "consumer",
            dependencies=(
                _edge_spec("foo", repository="services/foo", floor=">=1.0.0"),
            ),
        ),
    )

    graph = build_dependency_graph([consumer, second, first])

    assert len(graph.edges) == 1
    assert graph.edges[0].dependency.repository_id == "repo:services/foo"
    assert graph.parallel_groups == (
        ("repo:agent-packages/foo", "repo:services/foo"),
        ("repo:apps/consumer",),
    )


def test_explicit_overlay_resolves_ambiguous_metadata_owner() -> None:
    first = _project("services/foo", _package("services/foo", "foo"))
    second = _project("agent-packages/foo", _package("agent-packages/foo", "foo"))
    consumer = _project(
        "apps/consumer",
        _package(
            "apps/consumer",
            "consumer",
            dependencies=(_edge_spec("foo"),),
        ),
    )
    overlay = DependencyEdge(
        dependent=PackageKey("apps/consumer", Ecosystem.PYTHON, "consumer"),
        dependency=PackageKey("services/foo", Ecosystem.PYTHON, "foo"),
        floor=VersionFloor.parse(">=1.0.0"),
        source="fixture-overlay",
    )

    graph = build_dependency_graph([second, consumer, first], overlay_edges=(overlay,))

    assert graph.edges == (overlay,)


def test_overlay_missing_project_and_package_are_refusal_diagnostics() -> None:
    consumer = _project("apps/consumer", _package("apps/consumer", "consumer"))
    overlay = DependencyEdge(
        dependent=PackageKey("apps/consumer", Ecosystem.PYTHON, "consumer"),
        dependency=PackageKey("missing/foo", Ecosystem.PYTHON, "foo"),
        source="fixture-overlay",
    )

    with pytest.raises(GraphValidationError) as captured:
        build_dependency_graph([consumer], overlay_edges=(overlay,))

    assert _codes(captured.value) >= {
        GraphDiagnosticCode.MISSING_PROJECT,
        GraphDiagnosticCode.MISSING_PACKAGE,
    }


def test_duplicate_package_identity_and_missing_edge_are_diagnostics() -> None:
    duplicate = _project(
        "apps/duplicate",
        _package("apps/duplicate", "same"),
        _package("apps/duplicate", "same"),
    )
    missing = _project(
        "apps/missing",
        _package(
            "apps/missing",
            "missing",
            dependencies=(_edge_spec("does-not-exist"),),
        ),
    )

    with pytest.raises(GraphValidationError) as captured:
        build_dependency_graph([missing, duplicate])

    assert _codes(captured.value) >= {
        GraphDiagnosticCode.DUPLICATE_PACKAGE,
        GraphDiagnosticCode.MISSING_PACKAGE,
    }


def test_cycle_refuses_before_a_graph_is_returned() -> None:
    a = _project(
        "packages/a",
        _package(
            "packages/a",
            "a",
            dependencies=(_edge_spec("b", repository="packages/b"),),
        ),
    )
    b = _project(
        "packages/b",
        _package(
            "packages/b",
            "b",
            dependencies=(_edge_spec("a", repository="packages/a"),),
        ),
    )

    with pytest.raises(GraphValidationError) as captured:
        build_dependency_graph([a, b])

    assert _codes(captured.value) == {GraphDiagnosticCode.CYCLE}
    assert "repo:packages/a" in str(captured.value)
    assert "repo:packages/b" in str(captured.value)


def test_graph_digest_and_parallel_groups_ignore_input_iteration_order() -> None:
    projects = [
        _project(
            "packages/downstream",
            _package(
                "packages/downstream",
                "downstream",
                dependencies=(_edge_spec("middle", repository="packages/middle"),),
            ),
        ),
        _project("packages/middle", _package("packages/middle", "middle")),
        _project(
            "packages/independent", _package("packages/independent", "independent")
        ),
    ]
    expected = build_dependency_graph(projects)
    shuffled = list(projects)
    random.Random(42).shuffle(shuffled)
    actual = build_dependency_graph(reversed(shuffled))

    assert actual.digest == expected.digest
    assert actual.parallel_groups == expected.parallel_groups
    assert actual.canonical_payload() == expected.canonical_payload()


def test_frozen_c11_plan_has_stable_digest_and_no_push_without_consent() -> None:
    project = _project("packages/demo", _package("packages/demo", "demo"))
    plan = WorkspaceReleasePlan(
        workspace_id="workspace:test",
        source_sha="a" * 40,
        selected_projects=("packages/demo",),
        projects=(project,),
        stages=(
            PlanStage(
                stage_id="validate:repo:packages/demo",
                stage=ReleaseStage.VALIDATE,
                project_id="packages/demo",
            ),
        ),
        parallel_groups=(("packages/demo",),),
    )

    assert plan.plan_digest == plan_digest(plan)
    assert plan.canonical_payload()["allow_push"] is False
    with pytest.raises(AttributeError):
        plan.workspace_id = "workspace:changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="push stages require"):
        PlanStage(
            stage_id="push:repo:packages/demo",
            stage=ReleaseStage.PUSH,
            project_id="packages/demo",
        )


def test_conflicting_python_version_sources_refuse_with_locations(
    tmp_path: Path,
) -> None:
    metadata = tmp_path / "pyproject.toml"
    metadata.write_text(
        "[project]\nname = 'demo'\nversion = '1.0.0'\n"
        "[tool.bumpversion]\ncurrent_version = '1.1.0'\n",
        encoding="utf-8",
    )

    with pytest.raises(GraphValidationError) as captured:
        read_python_metadata("packages/demo", metadata)

    diagnostic = captured.value.diagnostics[0]
    assert diagnostic.code == GraphDiagnosticCode.CONFLICTING_VERSION_SOURCE
    assert "pyproject.toml:[project].version" in dict(diagnostic.details)
    assert "pyproject.toml:[tool.bumpversion].current_version" in dict(
        diagnostic.details
    )


def test_overlay_unknown_fields_and_duplicate_json_keys_refuse(tmp_path: Path) -> None:
    with pytest.raises(MetadataError, match="unsupported fields"):
        OverlayInput.from_mapping(
            {"schema_version": 1, "edges": [], "unexpected": True}
        )

    package_json = tmp_path / "package.json"
    package_json.write_text(
        '{"name":"demo","name":"other","version":"1.0.0"}',
        encoding="utf-8",
    )
    with pytest.raises(MetadataError, match="duplicate JSON field"):
        read_node_metadata("packages/demo", package_json)

    with pytest.raises(MetadataError, match="unsupported dependency floor"):
        OverlayInput.from_mapping(
            {
                "schema_version": 1,
                "edges": [
                    {
                        "dependent": "repo:apps/consumer::python:consumer",
                        "dependency": "repo:services/foo::python:foo",
                        "floor": ">=1.0.0,<2.0.0",
                    }
                ],
            }
        )
    with pytest.raises(MetadataError, match="version must use"):
        OverlayInput.from_mapping(
            {
                "schema_version": 1,
                "versions": [
                    {
                        "package": "repo:apps/consumer::python:consumer",
                        "version": "latest",
                    }
                ],
            }
        )


def test_repository_identity_rejects_noncanonical_path_forms() -> None:
    for value in (".", "./packages/demo", "packages//demo", "packages/../demo"):
        with pytest.raises(WorkspaceReleaseError):
            PackageKey(value, Ecosystem.PYTHON, "demo")


def test_oversized_and_deep_metadata_refuse_before_any_mutation(tmp_path: Path) -> None:
    metadata = tmp_path / "package.json"
    original = '{"name":"demo","version":"1.0.0"}'
    metadata.write_text(original, encoding="utf-8")
    with pytest.raises(MetadataError, match="byte bound"):
        read_node_metadata(
            "packages/demo",
            metadata,
            limits=MetadataLimits(max_bytes=8),
        )
    assert metadata.read_text(encoding="utf-8") == original

    deeply_nested: object = "x"
    for _ in range(8):
        deeply_nested = [deeply_nested]
    metadata.write_text(
        json.dumps({"name": "demo", "version": "1.0.0", "nested": deeply_nested}),
        encoding="utf-8",
    )
    with pytest.raises(MetadataError, match="nesting"):
        read_node_metadata(
            "packages/demo",
            metadata,
            limits=MetadataLimits(max_depth=3),
        )


def test_rust_and_node_readers_never_execute_commands_or_use_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cargo = tmp_path / "Cargo.toml"
    cargo.write_text(
        "[package]\nname = 'demo'\nversion = '1.0.0'\n"
        "[dependencies]\nother = { version = '>=2.0.0' }\n",
        encoding="utf-8",
    )
    package_json = tmp_path / "package.json"
    package_json.write_text(
        '{"name":"node-demo","version":"1.0.0","dependencies":{"other":"^2.0.0"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("subprocess")
    )
    monkeypatch.setattr(
        socket, "socket", lambda *args, **kwargs: pytest.fail("network")
    )

    read_rust_metadata("packages/rust-demo", cargo)
    read_node_metadata("packages/node-demo", package_json)
    assert cargo.read_text(encoding="utf-8").startswith("[package]")
    assert package_json.read_text(encoding="utf-8").startswith("{")


def test_ecosystem_fixtures_are_read_as_declarative_metadata() -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "workspace_release"

    python_project = read_python_metadata(
        "fixtures/python",
        fixture_root / "python" / "pyproject.toml",
    )
    rust_project = read_rust_metadata(
        "fixtures/rust",
        fixture_root / "rust" / "Cargo.toml",
    )
    node_project = read_node_metadata(
        "fixtures/node",
        fixture_root / "node" / "package.json",
    )
    overlay = OverlayInput.from_mapping(
        json.loads((fixture_root / "overlay.json").read_text(encoding="utf-8"))
    )

    assert python_project.packages[0].version.value == "1.0.0"
    assert rust_project.packages[0].dependencies[0].floor is not None
    assert rust_project.packages[0].dependencies[0].floor.value == "^3.1.0"
    assert node_project.packages[0].key.repository_id == "repo:fixtures/node"
    assert overlay.edges[0].dependency.repository_id == "repo:services/foo"


def test_phase_manifest_is_read_only_and_preserves_bare_refs_for_later_shadowing() -> (
    None
):
    raw = {
        "description": "legacy",
        "phases": [
            {"name": "downstream", "phase": 2, "projects": ["foo"]},
            {"name": "upstream", "phase": 1, "project": "foo"},
        ],
    }
    before = json.loads(json.dumps(raw))

    view = phase_manifest_from_mapping(raw)

    assert raw == before
    assert [phase.name for phase in view.phases] == ["upstream", "downstream"]
    assert view.phases[0].project_references == ("foo",)
