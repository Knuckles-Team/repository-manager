"""Focused RMDD-18 checkpoint-4 frozen-plan and stage-preview tests."""

from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

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
from repository_manager.development.workspace_release_plan import (
    BuildProfile,
    FailurePolicy,
    FrozenReleasePlan,
    PushConsentReference,
    ReleasePlanCode,
    ReleasePlanError,
    StageKind,
    freeze_release_plan,
    validate_frozen_release_plan,
)
from repository_manager.development.workspace_selection import (
    InclusionMode,
    SelectedChangeClosure,
    SelectionPolicy,
    derive_selected_closure,
)
from repository_manager.development.workspace_versions import (
    FloorPolicy,
    FloorRewriteSite,
    MetadataRepresentation,
    VersionBump,
    VersionPlan,
    VersionSourcePolicy,
    VersionSourceSite,
    plan_version_floors,
)

BASE_SHA = "f" * 40
SOURCE_SHA = "e" * 40


def _project(
    repository: str,
    name: str,
    *,
    tree: str,
    dependencies: tuple[str, ...] = (),
) -> ProjectRecord:
    key = PackageKey(repository, Ecosystem.PYTHON, name)
    version = Version("1.0.0")
    specs = tuple(
        DependencySpec(
            target=PackageReference(
                Ecosystem.PYTHON, dependency, f"packages/{dependency}"
            ),
            floor=VersionFloor.parse(">=1.0.0"),
            source=f"pyproject.toml:dependencies.{dependency}",
        )
        for dependency in dependencies
    )
    package = PackageRecord(
        key=key,
        version=version,
        version_sources=(VersionSource("pyproject.toml:[project].version", version),),
        dependencies=specs,
        metadata_files=("pyproject.toml",),
    )
    return ProjectRecord(
        repository_id=repository,
        tree_sha=tree,
        packages=(package,),
        metadata_files=("pyproject.toml",),
    )


def _fixture() -> tuple[DependencyGraph, SelectedChangeClosure, VersionPlan]:
    # a and e are independent; b/c form the parallel middle of a diamond and d
    # depends on both.  Package names intentionally equal their repository tail
    # so source and package identity assertions remain easy to read.
    projects = (
        _project("packages/a", "a", tree="a" * 40),
        _project("packages/b", "b", tree="b" * 40, dependencies=("a",)),
        _project("packages/c", "c", tree="c" * 40, dependencies=("a",)),
        _project("packages/d", "d", tree="d" * 40, dependencies=("b", "c")),
        _project("packages/e", "e", tree="e" * 40),
    )
    graph = build_dependency_graph(reversed(projects))
    selected_ids = tuple(project.repository_id for project in projects)
    selection = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/a",),
            selected_projects=selected_ids,
            upstream_mode=InclusionMode.NONE,
            downstream_mode=InclusionMode.NONE,
        ),
    )
    keys = {
        package.key.name: package.key
        for project in projects
        for package in project.packages
    }
    version_sites = tuple(
        VersionSourceSite(
            package=key,
            file_path="pyproject.toml",
            old_text="1.0.0",
            policy=VersionSourcePolicy(
                "pyproject.toml:[project].version",
                MetadataRepresentation.PYTHON,
                VersionBump.PATCH,
            ),
        )
        for key in keys.values()
    )
    floor_sites = tuple(
        FloorRewriteSite(
            dependent=keys[dependent],
            dependency=keys[dependency],
            file_path="pyproject.toml",
            source_location=f"pyproject.toml:dependencies.{dependency}",
            representation=MetadataRepresentation.PYTHON,
            old_text='">=1.0.0"',
            policy=FloorPolicy.RANGE,
        )
        for dependent, dependencies in {
            "b": ("a",),
            "c": ("a",),
            "d": ("b", "c"),
        }.items()
        for dependency in dependencies
    )
    version_plan = plan_version_floors(
        graph,
        selection,
        version_sites=version_sites,
        floor_sites=floor_sites,
    )
    return graph, selection, version_plan


def _plan(*, push: PushConsentReference | None = None) -> FrozenReleasePlan:
    graph, selection, version_plan = _fixture()
    profiles = {
        project_id: BuildProfile(f"build-{index}", (str(index + 1) * 64)[:64])
        for index, project_id in enumerate(selection.selected_project_ids)
    }
    # Profile digests must be hexadecimal; use a real digest-shaped fixture and
    # let the profile name carry the human-readable distinction.
    profiles = {
        project_id: BuildProfile(profile.name, "a" * 64)
        for project_id, profile in profiles.items()
    }
    return freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        validation_profile="validation-fixture",
        build_profiles=profiles,
        push_consent=push,
    )


def test_diamond_stage_dag_keeps_independent_projects_parallel() -> None:
    plan = _plan()
    assert plan.parallel_groups == (
        ("repo:packages/a", "repo:packages/e"),
        ("repo:packages/b", "repo:packages/c"),
        ("repo:packages/d",),
    )
    assert plan.push_stages == ()
    validate_stages = tuple(
        stage for stage in plan.stages if stage.kind is StageKind.VALIDATE
    )
    assert all(not stage.depends_on for stage in validate_stages)
    bumps = {
        stage.project_id: stage for stage in plan.stages if stage.kind is StageKind.BUMP
    }
    assert len(bumps["repo:packages/a"].depends_on) == 1
    assert len(bumps["repo:packages/b"].depends_on) == 2
    assert len(bumps["repo:packages/c"].depends_on) == 2
    assert len(bumps["repo:packages/d"].depends_on) == 3
    assert all(
        stage.failure_policy is FailurePolicy.BLOCK_DEPENDENTS for stage in plan.stages
    )
    assert all(stage.input_digest and stage.stage_id for stage in plan.stages)


def test_plan_digest_and_stages_are_independent_of_input_order() -> None:
    graph, selection, version_plan = _fixture()
    first = freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        validation_profile="validation-fixture",
        build_profile="build-fixture",
    )
    # The graph/selection constructors already freeze their own order.  Passing
    # the same evidence through a different source order must still replay.
    second = freeze_release_plan(
        build_dependency_graph(reversed(selection.source_graph.projects)),
        derive_selected_closure(
            build_dependency_graph(reversed(selection.source_graph.projects)),
            selection.policy,
        ),
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        validation_profile="validation-fixture",
        build_profile="build-fixture",
    )
    assert first.plan_digest == second.plan_digest
    assert first.canonical_payload(include_digest=True) == second.canonical_payload(
        include_digest=True
    )


def test_push_is_separate_and_requires_an_immutable_consent_reference() -> None:
    graph, selection, version_plan = _fixture()
    with pytest.raises(ReleasePlanError) as captured:
        freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
            include_push=True,
        )
    assert captured.value.code is ReleasePlanCode.PUSH_CONSENT

    consent = PushConsentReference("consent:fixture", "c" * 64)
    plan = freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        push_consent=consent,
    )
    assert len(plan.push_stages) == len(plan.selected_projects)
    assert all(stage.consent_reference == consent for stage in plan.push_stages)
    plan.validate_against(graph, selection)


def test_frozen_plan_rejects_digest_and_nested_stage_tampering() -> None:
    plan = _plan()
    with pytest.raises(FrozenInstanceError):
        plan.plan_digest = "0" * 64  # type: ignore[misc]
    with pytest.raises(ReleasePlanError, match="plan digest"):
        replace(plan, plan_digest="0" * 64)

    bump = next(
        stage
        for stage in plan.stages
        if stage.kind is StageKind.BUMP and stage.project_id == "repo:packages/d"
    )
    assert len(bump.depends_on) > 1
    with pytest.raises(ReleasePlanError):
        replace(bump, depends_on=tuple(reversed(bump.depends_on)))
    forged = object.__new__(FrozenReleasePlan)
    for field_name in plan.__dataclass_fields__:
        object.__setattr__(forged, field_name, getattr(plan, field_name))
    object.__setattr__(forged, "source_sha", "0" * 40)
    with pytest.raises(ReleasePlanError):
        validate_frozen_release_plan(forged)

    forged_stage = object.__new__(type(bump))
    for field_name in bump.__dataclass_fields__:
        object.__setattr__(forged_stage, field_name, getattr(bump, field_name))
    object.__setattr__(forged_stage, "depends_on", tuple(reversed(bump.depends_on)))
    forged_stages = tuple(
        forged_stage if stage is bump else stage for stage in plan.stages
    )
    object.__setattr__(forged, "stages", forged_stages)
    with pytest.raises(ReleasePlanError):
        validate_frozen_release_plan(forged)


def test_profile_source_and_consent_changes_change_exact_digest() -> None:
    first = _plan()
    graph, selection, version_plan = _fixture()
    changed_profile = freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        validation_profile="different-validation",
        build_profile="build-fixture",
    )
    assert changed_profile.plan_digest != first.plan_digest
    changed_consent = _plan(push=PushConsentReference("consent:other", "d" * 64))
    assert changed_consent.plan_digest != first.plan_digest


def test_hostile_containers_and_untrusted_boolean_push_fail_closed() -> None:
    graph, selection, version_plan = _fixture()

    class EvilList(list[object]):
        def __iter__(self):
            raise RuntimeError("private payload")

    with pytest.raises(ReleasePlanError) as captured:
        freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
            allow_push=True,
            validation_profiles=EvilList(),
        )
    assert captured.value.code in {
        ReleasePlanCode.INVALID_INPUT,
        ReleasePlanCode.PROFILE,
        ReleasePlanCode.PUSH_CONSENT,
    }
    assert "private payload" not in str(captured.value)


def test_no_push_stage_can_be_created_from_a_boolean_alone() -> None:
    graph, selection, version_plan = _fixture()
    with pytest.raises(ReleasePlanError) as captured:
        freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
            allow_push=True,
        )
    assert captured.value.code is ReleasePlanCode.PUSH_CONSENT
