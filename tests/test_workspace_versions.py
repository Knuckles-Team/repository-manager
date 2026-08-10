"""Focused RMDD-18 checkpoint-3 version/floor planning tests."""

from __future__ import annotations

import itertools
import os
import subprocess
import sys
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from textwrap import dedent
from typing import cast

import pytest

from repository_manager.development.workspace_release import (
    DependencyGraph,
    DependencySpec,
    Ecosystem,
    PackageKey,
    PackageRecord,
    PackageReference,
    ProjectRecord,
    Version,
    VersionFloor,
    VersionSource,
    build_dependency_graph,
)
from repository_manager.development.workspace_selection import (
    InclusionMode,
    SelectedChangeClosure,
    SelectionPolicy,
    derive_selected_closure,
)
from repository_manager.development.workspace_versions import (
    FloorPolicy,
    FloorPreviewReason,
    FloorRewriteSite,
    MetadataRepresentation,
    VersionBump,
    VersionPlan,
    VersionPlanningCode,
    VersionPlanningError,
    VersionPlanningInput,
    VersionSourcePolicy,
    VersionSourceSite,
    plan_version_floors,
    plan_version_input,
)


def _tree_sha(number: str) -> str:
    return (number * 40)[:40]


def _project(
    repository: str,
    name: str,
    ecosystem: Ecosystem,
    version: str,
    *,
    source_location: str,
    file_name: str,
    tree_sha: str,
    dependency: tuple[Ecosystem, str, str, str] | None = None,
) -> ProjectRecord:
    dependencies: tuple[DependencySpec, ...] = ()
    if dependency is not None:
        dep_ecosystem, dep_name, dep_repository, floor = dependency
        dependencies = (
            DependencySpec(
                target=PackageReference(dep_ecosystem, dep_name, dep_repository),
                floor=VersionFloor.parse(floor),
                source=(
                    f"{file_name}:[dependencies].{dep_name}"
                    if file_name == "Cargo.toml"
                    else f"{file_name}:dependencies.{dep_name}"
                ),
            ),
        )
    key = PackageKey(repository, ecosystem, name)
    selected_version = Version(version)
    package = PackageRecord(
        key=key,
        version=selected_version,
        version_sources=(VersionSource(source_location, selected_version),),
        dependencies=dependencies,
        metadata_files=(file_name,),
    )
    return ProjectRecord(
        repository_id=repository,
        tree_sha=tree_sha,
        packages=(package,),
        metadata_files=(file_name,),
    )


def _chain() -> tuple[DependencyGraph, SelectedChangeClosure, dict[str, PackageKey]]:
    a = _project(
        "services/a",
        "a",
        Ecosystem.PYTHON,
        "1.0.0",
        source_location="pyproject.toml:[project].version",
        file_name="pyproject.toml",
        tree_sha=_tree_sha("a"),
    )
    b = _project(
        "services/b",
        "b",
        Ecosystem.RUST,
        "1.0.0",
        source_location="Cargo.toml:[package].version",
        file_name="Cargo.toml",
        tree_sha=_tree_sha("b"),
        dependency=(Ecosystem.PYTHON, "a", "services/a", "^1.0.0"),
    )
    c = _project(
        "services/c",
        "c",
        Ecosystem.NODE,
        "1.0.0",
        source_location="package.json:version",
        file_name="package.json",
        tree_sha=_tree_sha("c"),
        dependency=(Ecosystem.RUST, "b", "services/b", "~1.0.0"),
    )
    d = _project(
        "services/d",
        "d",
        Ecosystem.NODE,
        "1.0.0",
        source_location="package.json:version",
        file_name="package.json",
        tree_sha=_tree_sha("d"),
        dependency=(Ecosystem.PYTHON, "a", "services/a", "^3.0.0"),
    )
    projects = (a, b, c, d)
    graph = build_dependency_graph(projects)
    selection = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("services/a",),
            selected_projects=tuple(project.repository_id for project in projects),
            upstream_mode=InclusionMode.NONE,
            downstream_mode=InclusionMode.NONE,
        ),
    )
    keys = {
        package.key.name: package.key
        for project in projects
        for package in project.packages
    }
    return graph, selection, keys


def _sites(
    keys: dict[str, PackageKey],
) -> tuple[tuple[VersionSourceSite, ...], tuple[FloorRewriteSite, ...]]:
    versions = (
        VersionSourceSite(
            package=keys["a"],
            file_path="pyproject.toml",
            old_text="1.0.0",
            policy=VersionSourcePolicy(
                "pyproject.toml:[project].version",
                MetadataRepresentation.PYTHON,
                VersionBump.MAJOR,
            ),
        ),
        VersionSourceSite(
            package=keys["b"],
            file_path="Cargo.toml",
            old_text='"1.0.0"',
            policy=VersionSourcePolicy(
                "Cargo.toml:[package].version",
                MetadataRepresentation.RUST,
                VersionBump.PATCH,
            ),
        ),
    )
    floors = (
        FloorRewriteSite(
            dependent=keys["b"],
            dependency=keys["a"],
            file_path="Cargo.toml",
            source_location="Cargo.toml:[dependencies].a",
            representation=MetadataRepresentation.RUST,
            old_text='"^1.0.0"',
            policy=FloorPolicy.CARET,
        ),
        FloorRewriteSite(
            dependent=keys["c"],
            dependency=keys["b"],
            file_path="package.json",
            source_location="package.json:dependencies.b",
            representation=MetadataRepresentation.NODE,
            old_text='"~1.0.0"',
            policy=FloorPolicy.TILDE,
        ),
        FloorRewriteSite(
            dependent=keys["d"],
            dependency=keys["a"],
            file_path="package.json",
            source_location="package.json:dependencies.a",
            representation=MetadataRepresentation.NODE,
            old_text='"^3.0.0"',
            policy=FloorPolicy.CARET,
        ),
    )
    return versions, floors


def test_multilevel_bumps_are_topological_and_independent_packages_share_a_batch() -> (
    None
):
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)

    plan = plan_version_floors(
        graph,
        selection,
        version_sites=versions,
        floor_sites=floors,
    )

    assert plan.next_versions == (
        (keys["a"].value, Version("2.0.0")),
        (keys["b"].value, Version("1.0.1")),
        (keys["c"].value, Version("1.0.0")),
        (keys["d"].value, Version("1.0.0")),
    )
    assert plan.package_batches == (
        (keys["a"].value,),
        (keys["b"].value, keys["d"].value),
        (keys["c"].value,),
    )
    assert [item.new_normalized for item in plan.floor_previews] == [
        "^2.0.0",
        "^3.0.0",
        "~1.0.1",
    ]
    assert plan.floor_previews[1].reason is FloorPreviewReason.ALREADY_SATISFIED
    assert plan.floor_previews[1].is_noop is True
    assert plan.floor_previews[0].witness[2] == "topological-batch:0"


def test_python_range_policy_and_exact_node_rust_rendering_are_explicit() -> None:
    graph, selection, keys = _chain()
    python_site = FloorRewriteSite(
        dependent=keys["b"],
        dependency=keys["a"],
        file_path="Cargo.toml",
        source_location="Cargo.toml:[dependencies].a",
        representation=MetadataRepresentation.RUST,
        old_text='"^1.0.0"',
        policy=FloorPolicy.RANGE,
    )
    exact_node = FloorRewriteSite(
        dependent=keys["d"],
        dependency=keys["a"],
        file_path="package.json",
        source_location="package.json:dependencies.a",
        representation=MetadataRepresentation.NODE,
        old_text='"^3.0.0"',
        policy=FloorPolicy.EXACT,
    )
    plan = plan_version_floors(
        graph,
        selection,
        version_sites=(_sites(keys)[0][0],),
        floor_sites=(python_site, exact_node),
    )

    assert plan.floor_previews[0].new_text == '">=2.0.0"'
    assert plan.floor_previews[1].new_text == '"^3.0.0"'
    assert plan.floor_previews[0].policy is FloorPolicy.RANGE


def test_input_permutations_have_the_same_canonical_plan() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    expected = plan_version_floors(
        graph, selection, version_sites=versions, floor_sites=floors
    )
    for version_order in itertools.permutations(versions):
        for floor_order in itertools.permutations(floors):
            actual = plan_version_floors(
                graph,
                selection,
                version_sites=version_order,
                floor_sites=floor_order,
            )
            assert actual.plan_digest == expected.plan_digest
            assert actual.canonical_payload(
                include_digest=True
            ) == expected.canonical_payload(include_digest=True)


def test_closed_input_bundle_is_immutable_and_restart_safe() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    request = VersionPlanningInput(graph, selection, versions, floors)
    expected = plan_version_floors(
        graph, selection, version_sites=versions, floor_sites=floors
    )

    assert plan_version_input(request).canonical_payload(include_digest=True) == (
        expected.canonical_payload(include_digest=True)
    )
    with pytest.raises(FrozenInstanceError):
        request.version_sites = ()  # type: ignore[misc]


def test_plan_tamper_and_restart_reconstruction_refuse_or_preserve_digest() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    plan = plan_version_floors(
        graph, selection, version_sites=versions, floor_sites=floors
    )

    rebuilt = type(plan)(
        graph_digest=plan.graph_digest,
        selection_digest=plan.selection_digest,
        next_versions=plan.next_versions,
        package_batches=plan.package_batches,
        version_previews=plan.version_previews,
        floor_previews=plan.floor_previews,
        plan_digest=plan.plan_digest,
    )
    assert rebuilt.plan_digest == plan.plan_digest
    assert rebuilt.canonical_payload(include_digest=True) == plan.canonical_payload(
        include_digest=True
    )
    with pytest.raises(VersionPlanningError, match="plan digest"):
        replace(plan, plan_digest="0" * 64)
    changed_preview = replace(plan.floor_previews[0], new_text="tampered")
    with pytest.raises(VersionPlanningError, match="plan digest"):
        replace(plan, floor_previews=(changed_preview, *plan.floor_previews[1:]))


def test_hash_seed_does_not_change_digest() -> None:
    script = dedent(
        """
        from repository_manager.development.workspace_release import *
        from repository_manager.development.workspace_selection import *
        from repository_manager.development.workspace_versions import *
        key = PackageKey('packages/a', Ecosystem.PYTHON, 'a')
        version = Version('1.0.0')
        package = PackageRecord(key, version, (VersionSource('pyproject.toml:[project].version', version),), (), ('pyproject.toml',))
        project = ProjectRecord('packages/a', 'a' * 40, (package,), ('pyproject.toml',))
        graph = build_dependency_graph([project])
        selection = derive_selected_closure(graph, SelectionPolicy(('packages/a',)))
        site = VersionSourceSite(key, 'pyproject.toml', '1.0.0', VersionSourcePolicy('pyproject.toml:[project].version', MetadataRepresentation.PYTHON, VersionBump.PATCH))
        print(plan_version_floors(graph, selection, version_sites=(site,), floor_sites=()).plan_digest)
        """
    )
    values = []
    for seed in ("1", "17", "random"):
        environment = os.environ.copy()
        environment["PYTHONHASHSEED"] = seed
        values.append(
            subprocess.check_output(
                [sys.executable, "-c", script],
                cwd=Path.cwd(),
                env=environment,
                text=True,
            ).strip()
        )
    assert len(set(values)) == 1


def test_bounds_and_adversarial_iterables_fail_closed() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)

    class FailingIterable:
        def __iter__(self):
            raise RuntimeError("private detail")

    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(
            graph,
            selection,
            version_sites=FailingIterable(),
            floor_sites=floors,
        )
    assert captured.value.code is VersionPlanningCode.UNBOUNDED_INPUT
    assert "private detail" not in str(captured.value)

    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(
            graph,
            selection,
            version_sites=versions,
            floor_sites=(
                replace(floors[0], old_text='"^1.0.0 || ^2.0.0"'),
                *floors[1:],
            ),
        )
    assert captured.value.code is VersionPlanningCode.UNSUPPORTED_SPECIFIER


def test_version_overflow_is_normalized_to_privacy_safe_planning_error() -> None:
    key = PackageKey("packages/a", Ecosystem.PYTHON, "a")
    current = Version(f"{'9' * 124}.0.0")
    package = PackageRecord(
        key,
        current,
        (VersionSource("pyproject.toml:[project].version", current),),
        (),
        ("pyproject.toml",),
    )
    project = ProjectRecord("packages/a", "a" * 40, (package,), ("pyproject.toml",))
    graph = build_dependency_graph([project])
    selection = derive_selected_closure(graph, SelectionPolicy(("packages/a",)))
    site = VersionSourceSite(
        key,
        "pyproject.toml",
        current.value,
        VersionSourcePolicy(
            "pyproject.toml:[project].version",
            MetadataRepresentation.PYTHON,
            VersionBump.MAJOR,
        ),
    )

    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(graph, selection, version_sites=(site,), floor_sites=())
    assert captured.value.code is VersionPlanningCode.NON_SEMVER
    assert type(captured.value) is VersionPlanningError
    assert "WorkspaceReleaseError" not in str(captured.value)


def test_hostile_nested_builtin_subclasses_fail_before_introspection() -> None:
    class EvilList(list):
        def __len__(self) -> int:
            raise RuntimeError("private detail")

        def __iter__(self):
            raise RuntimeError("private detail")

    with pytest.raises(VersionPlanningError) as captured:
        VersionPlanningError(
            VersionPlanningCode.INVALID_INPUT,
            "bounded diagnostic",
            details=cast(list[tuple[str, str]], [EvilList(("secret", "value"))]),
        )
    assert "private detail" not in str(captured.value)

    graph, selection, keys = _chain()
    with pytest.raises(VersionPlanningError) as captured:
        VersionPlan(
            graph_digest=graph.digest,
            selection_digest=selection.digest,
            next_versions=cast(
                tuple[tuple[str, Version], ...],
                (EvilList((keys["a"].value, Version("1.0.0"))),),
            ),
            package_batches=((keys["a"].value,),),
        )
    assert "private detail" not in str(captured.value)

    with pytest.raises(VersionPlanningError) as captured:
        VersionPlan(
            graph_digest=graph.digest,
            selection_digest=selection.digest,
            next_versions=((keys["a"].value, Version("1.0.0")),),
            package_batches=cast(
                tuple[tuple[str, ...], ...], (EvilList((keys["a"].value,)),)
            ),
        )
    assert "private detail" not in str(captured.value)


def test_node_sites_require_canonical_json_string_literals() -> None:
    _, _, keys = _chain()
    with pytest.raises(VersionPlanningError) as captured:
        FloorRewriteSite(
            dependent=keys["c"],
            dependency=keys["b"],
            file_path="package.json",
            source_location="package.json:dependencies.b",
            representation=MetadataRepresentation.NODE,
            old_text="'~1.0.0'",
            policy=FloorPolicy.TILDE,
        )
    assert captured.value.code is VersionPlanningCode.UNSUPPORTED_SPECIFIER

    with pytest.raises(VersionPlanningError) as captured:
        FloorRewriteSite(
            dependent=keys["c"],
            dependency=keys["b"],
            file_path="package.json",
            source_location="package.json:dependencies.b",
            representation=MetadataRepresentation.NODE,
            old_text="~1.0.0",
            policy=FloorPolicy.TILDE,
        )
    assert captured.value.code is VersionPlanningCode.UNSUPPORTED_SPECIFIER


def test_plan_rejects_missing_digest_and_recomputed_forged_preview() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    plan = plan_version_floors(
        graph, selection, version_sites=versions, floor_sites=floors
    )
    missing_preview_digest = replace(plan.floor_previews[0], plan_digest="")
    with pytest.raises(VersionPlanningError, match="plan digest"):
        type(plan)(
            graph_digest=plan.graph_digest,
            selection_digest=plan.selection_digest,
            next_versions=plan.next_versions,
            package_batches=plan.package_batches,
            version_previews=plan.version_previews,
            floor_previews=(missing_preview_digest, *plan.floor_previews[1:]),
            plan_digest=plan.plan_digest,
        )

    forged_preview = replace(plan.floor_previews[0], new_text="not-semver")
    forged = object.__new__(type(plan))
    object.__setattr__(forged, "graph_digest", plan.graph_digest)
    object.__setattr__(forged, "selection_digest", plan.selection_digest)
    object.__setattr__(forged, "next_versions", plan.next_versions)
    object.__setattr__(forged, "package_batches", plan.package_batches)
    object.__setattr__(forged, "version_previews", plan.version_previews)
    object.__setattr__(
        forged, "floor_previews", (forged_preview, *plan.floor_previews[1:])
    )
    object.__setattr__(forged, "plan_digest", plan.plan_digest)
    with pytest.raises(VersionPlanningError, match="plan digest"):
        forged.validate_against(graph, selection)


def test_input_payload_canonicalizes_unordered_site_collections() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    first = VersionPlanningInput(graph, selection, versions, floors)
    reversed_input = VersionPlanningInput(
        graph, selection, tuple(reversed(versions)), tuple(reversed(floors))
    )
    assert first.canonical_payload() == reversed_input.canonical_payload()


def test_site_representation_is_bound_to_package_ecosystem() -> None:
    _, _, keys = _chain()
    with pytest.raises(VersionPlanningError) as captured:
        VersionSourceSite(
            package=keys["a"],
            file_path="Cargo.toml",
            old_text="1.0.0",
            policy=VersionSourcePolicy(
                "Cargo.toml:[package].version",
                MetadataRepresentation.RUST,
                VersionBump.PATCH,
            ),
        )
    assert captured.value.code is VersionPlanningCode.INVALID_INPUT

    with pytest.raises(VersionPlanningError) as captured:
        FloorRewriteSite(
            dependent=keys["b"],
            dependency=keys["a"],
            file_path="pyproject.toml",
            source_location="pyproject.toml:dependencies.a",
            representation=MetadataRepresentation.PYTHON,
            old_text="^1.0.0",
            policy=FloorPolicy.RANGE,
        )
    assert captured.value.code is VersionPlanningCode.INVALID_INPUT


@pytest.mark.parametrize(
    "path",
    [
        "../pyproject.toml",
        "/tmp/pyproject.toml",
        "pyproject.toml/../x",
        "pyproject.toml\\x",
    ],
)
def test_path_traversal_and_symlink_sites_refuse(path: str) -> None:
    key = PackageKey("packages/a", Ecosystem.PYTHON, "a")
    policy = VersionSourcePolicy(
        "pyproject.toml:[project].version",
        MetadataRepresentation.PYTHON,
        VersionBump.PATCH,
    )
    with pytest.raises(VersionPlanningError) as captured:
        VersionSourceSite(key, path, "1.0.0", policy)
    assert captured.value.code is VersionPlanningCode.PATH_TRAVERSAL
    with pytest.raises(VersionPlanningError) as captured:
        VersionSourceSite(key, "pyproject.toml", "1.0.0", policy, symlink=True)
    assert captured.value.code is VersionPlanningCode.SYMLINK


def test_duplicate_or_conflicting_sites_and_missing_changed_floor_refuse() -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    duplicate = (versions[0], versions[0])
    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(
            graph, selection, version_sites=duplicate, floor_sites=floors
        )
    assert captured.value.code is VersionPlanningCode.DUPLICATE_SOURCE_SITE

    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(
            graph, selection, version_sites=versions, floor_sites=floors[1:]
        )
    assert captured.value.code is VersionPlanningCode.MISSING_FLOOR_SITE


@pytest.mark.parametrize("version", ["1.0.0-alpha", "1.0.0+local"])
def test_prerelease_and_local_versions_refuse(version: str) -> None:
    key = PackageKey("packages/a", Ecosystem.PYTHON, "a")
    value = Version(version)
    package = PackageRecord(
        key,
        value,
        (VersionSource("pyproject.toml:[project].version", value),),
        (),
        ("pyproject.toml",),
    )
    project = ProjectRecord("packages/a", "a" * 40, (package,), ("pyproject.toml",))
    graph = build_dependency_graph([project])
    selection = derive_selected_closure(graph, SelectionPolicy(("packages/a",)))
    site = VersionSourceSite(
        key,
        "pyproject.toml",
        version,
        VersionSourcePolicy(
            "pyproject.toml:[project].version",
            MetadataRepresentation.PYTHON,
            VersionBump.PATCH,
        ),
    )
    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(graph, selection, version_sites=(site,), floor_sites=())
    assert captured.value.code in {
        VersionPlanningCode.PRERELEASE,
        VersionPlanningCode.LOCAL_VERSION,
    }


def test_same_project_package_cycle_is_refused_by_package_ordering() -> None:
    first_key = PackageKey("packages/a", Ecosystem.PYTHON, "first")
    second_key = PackageKey("packages/a", Ecosystem.PYTHON, "second")
    first_version = Version("1.0.0")
    second_version = Version("1.0.0")
    first = PackageRecord(
        first_key,
        first_version,
        (VersionSource("pyproject.toml:[project].version", first_version),),
        (
            DependencySpec(
                PackageReference(Ecosystem.PYTHON, "second", "packages/a"),
                VersionFloor.parse(">=1.0.0"),
                "fixture",
            ),
        ),
        ("pyproject.toml",),
    )
    second = PackageRecord(
        second_key,
        second_version,
        (VersionSource("pyproject.toml:[project].version", second_version),),
        (
            DependencySpec(
                PackageReference(Ecosystem.PYTHON, "first", "packages/a"),
                VersionFloor.parse(">=1.0.0"),
                "fixture",
            ),
        ),
        ("pyproject.toml",),
    )
    project = ProjectRecord(
        "packages/a", "a" * 40, (first, second), ("pyproject.toml",)
    )
    graph = build_dependency_graph([project])
    selection = derive_selected_closure(graph, SelectionPolicy(("packages/a",)))
    with pytest.raises(VersionPlanningError) as captured:
        plan_version_floors(graph, selection, version_sites=(), floor_sites=())
    assert captured.value.code is VersionPlanningCode.CYCLE


def test_no_project_code_execution_is_required(monkeypatch: pytest.MonkeyPatch) -> None:
    graph, selection, keys = _chain()
    versions, floors = _sites(keys)
    monkeypatch.setattr(
        subprocess, "run", lambda *args, **kwargs: pytest.fail("execution")
    )
    plan = plan_version_floors(
        graph, selection, version_sites=versions, floor_sites=floors
    )
    assert plan.plan_digest
