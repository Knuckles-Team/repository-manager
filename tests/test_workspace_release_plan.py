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
    DecisionReference,
    FailurePolicy,
    FrozenReleasePlan,
    FrozenReleasePlanInput,
    PushConsentReference,
    ReleaseDecisionContext,
    ReleasePlanCode,
    ReleasePlanError,
    RetryPolicy,
    StageKind,
    TimeoutPolicy,
    _digest_payload,
    _plan_payload,
    _stage_identity,
    _stage_payload_without_digests,
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


@pytest.mark.parametrize(
    "source_value,base_value,expected",
    [
        (0, BASE_SHA, ReleasePlanCode.SOURCE_SHA),
        (False, BASE_SHA, ReleasePlanCode.SOURCE_SHA),
        (b"", BASE_SHA, ReleasePlanCode.SOURCE_SHA),
        (SOURCE_SHA, 0, ReleasePlanCode.BASE_SHA),
        (SOURCE_SHA, False, ReleasePlanCode.BASE_SHA),
        (SOURCE_SHA, b"", ReleasePlanCode.BASE_SHA),
        (None, BASE_SHA, ReleasePlanCode.SOURCE_SHA),
        (SOURCE_SHA, None, ReleasePlanCode.BASE_SHA),
    ],
)
def test_sha_aliases_require_exact_explicit_builtin_scalars(
    source_value: object, base_value: object, expected: ReleasePlanCode
) -> None:
    graph, selection, version_plan = _fixture()

    class Trap(str):
        def strip(self, chars: str | None = None) -> str:
            del chars
            raise RuntimeError("secret string trap")

    if source_value == 0:
        source_value = Trap("0" * 40)
    with pytest.raises(ReleasePlanError) as captured:
        freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=source_value,  # type: ignore[arg-type]
            base_sha=base_value,  # type: ignore[arg-type]
            generation_id="generation:fixture",
        )
    assert captured.value.code is expected
    assert "secret string trap" not in str(captured.value)


def test_input_bundle_does_not_cross_substitute_sha_aliases() -> None:
    graph, selection, version_plan = _fixture()
    for kwargs, expected in (
        ({"source_sha": None, "base_sha": BASE_SHA}, ReleasePlanCode.SOURCE_SHA),
        ({"source_sha": SOURCE_SHA, "base_sha": None}, ReleasePlanCode.BASE_SHA),
    ):
        with pytest.raises(ReleasePlanError) as captured:
            FrozenReleasePlanInput(
                graph,
                selection,
                version_plan,
                generation_id="generation:fixture",
                **kwargs,  # type: ignore[arg-type]
            )
        assert captured.value.code is expected


@pytest.mark.parametrize(
    "include_push,allow_push,with_consent,accepted",
    [
        (None, None, False, False),
        (False, None, False, False),
        (None, False, False, False),
        (False, False, False, False),
        (True, None, True, True),
        (None, True, True, True),
        (True, True, True, True),
        (None, None, True, True),
    ],
)
def test_consent_and_legacy_push_alias_contract(
    include_push: bool | None,
    allow_push: bool | None,
    with_consent: bool,
    accepted: bool,
) -> None:
    graph, selection, version_plan = _fixture()
    consent = PushConsentReference("consent:matrix", "c" * 64) if with_consent else None
    plan = freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        include_push=include_push,
        allow_push=allow_push,
        push_consent=consent,
    )
    assert plan.push_enabled is accepted
    assert _digest_payload(_plan_payload(plan)) == plan.plan_digest


@pytest.mark.parametrize(
    "kwargs",
    [
        {"include_push": False, "push_consent": PushConsentReference("c:1", "c" * 64)},
        {"allow_push": False, "push_consent": PushConsentReference("c:2", "c" * 64)},
        {"include_push": True, "allow_push": False},
        {"include_push": False, "allow_push": True},
        {"include_push": True},
        {"allow_push": True},
    ],
)
def test_contradictory_or_unauthorized_push_inputs_refuse_before_construction(
    kwargs: dict[str, object],
) -> None:
    graph, selection, version_plan = _fixture()
    with pytest.raises(ReleasePlanError):
        freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
            **kwargs,  # type: ignore[arg-type]
        )


def test_all_consent_aliases_are_evidence_not_authority() -> None:
    graph, selection, version_plan = _fixture()
    for alias in ("push_consent", "consent_reference", "consent_ref"):
        consent = PushConsentReference(f"consent:{alias}", "c" * 64)
        accepted = freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
            **{alias: consent},  # type: ignore[arg-type]
        )
        assert accepted.push_enabled
        with pytest.raises(ReleasePlanError):
            freeze_release_plan(
                graph,
                selection,
                version_plan,
                source_sha=SOURCE_SHA,
                base_sha=BASE_SHA,
                generation_id="generation:fixture",
                include_push=False,
                **{alias: consent},  # type: ignore[arg-type]
            )


def test_forged_graph_and_selection_containers_are_rejected_before_iteration() -> None:
    graph, selection, version_plan = _fixture()

    class InfiniteGenerator:
        def __iter__(self):
            while True:
                yield object()

    forged_graph = object.__new__(DependencyGraph)
    for field_name in graph.__dataclass_fields__:
        object.__setattr__(forged_graph, field_name, getattr(graph, field_name))
    object.__setattr__(forged_graph, "projects", InfiniteGenerator())
    with pytest.raises(ReleasePlanError) as captured:
        freeze_release_plan(
            forged_graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
        )
    assert captured.value.code in {
        ReleasePlanCode.INVALID_INPUT,
        ReleasePlanCode.GRAPH_DRIFT,
    }

    forged_selection = object.__new__(SelectedChangeClosure)
    for field_name in selection.__dataclass_fields__:
        object.__setattr__(forged_selection, field_name, getattr(selection, field_name))
    object.__setattr__(forged_selection, "selected_project_ids", InfiniteGenerator())
    with pytest.raises(ReleasePlanError) as captured:
        freeze_release_plan(
            graph,
            forged_selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
        )
    assert captured.value.code in {
        ReleasePlanCode.INVALID_INPUT,
        ReleasePlanCode.SELECTION_DRIFT,
    }


def test_rehashed_semantically_invalid_stage_dag_is_rejected() -> None:
    plan = _plan()
    target = next(
        stage
        for stage in plan.stages
        if stage.kind is StageKind.BUMP and stage.project_id.endswith("/d")
    )
    forged_stage = object.__new__(type(target))
    for field_name in target.__dataclass_fields__:
        object.__setattr__(forged_stage, field_name, getattr(target, field_name))
    replacement_dependencies = target.depends_on[:1]
    object.__setattr__(forged_stage, "depends_on", replacement_dependencies)
    payload = _stage_payload_without_digests(
        kind=forged_stage.kind,
        project_id=forged_stage.project_id,
        base_sha=forged_stage.base_sha,
        tree_sha=forged_stage.tree_sha,
        generation_id=forged_stage.generation_id,
        graph_digest=forged_stage.graph_digest,
        selection_digest=forged_stage.selection_digest,
        version_plan_digest=forged_stage.version_plan_digest,
        version_preview_digests=forged_stage.version_preview_digests,
        floor_preview_digests=forged_stage.floor_preview_digests,
        validation_profile_digest=forged_stage.validation_profile_digest,
        build_profile_digest=forged_stage.build_profile_digest,
        depends_on=replacement_dependencies,
        consent_reference=forged_stage.consent_reference,
        decision_digest=forged_stage.decision_digest,
        resource_profile=forged_stage.resource_profile,
        retry_policy=forged_stage.retry_policy,
        retry_count=forged_stage.retry_count,
        timeout_policy=forged_stage.timeout_policy,
        timeout_seconds=forged_stage.timeout_seconds,
    )
    stage_id, input_digest = _stage_identity(payload)
    object.__setattr__(forged_stage, "stage_id", stage_id)
    object.__setattr__(forged_stage, "input_digest", input_digest)
    forged_stages = tuple(
        forged_stage if stage is target else stage for stage in plan.stages
    )
    forged_plan = object.__new__(FrozenReleasePlan)
    for field_name in plan.__dataclass_fields__:
        object.__setattr__(forged_plan, field_name, getattr(plan, field_name))
    object.__setattr__(forged_plan, "stages", forged_stages)
    # Even if an attacker recomputes the outer digest, source-derived stage
    # composition remains authoritative.
    object.__setattr__(
        forged_plan, "plan_digest", _digest_payload(_plan_payload(forged_plan))
    )
    with pytest.raises(ReleasePlanError):
        validate_frozen_release_plan(forged_plan)


def test_decision_fields_bind_plan_and_stage_identity() -> None:
    graph, selection, version_plan = _fixture()
    digest = "a" * 64
    base_context = ReleaseDecisionContext(
        release_profile=DecisionReference("release:one", digest),
        target_branch="main",
        candidate=DecisionReference("candidate:one", digest),
        certificate=DecisionReference("certificate:one", digest),
        config=DecisionReference("config:one", digest),
        toolchain=DecisionReference("toolchain:one", digest),
        command=DecisionReference("command:preview", digest),
        artifact_contract=DecisionReference("artifact:one", digest),
        resource_profile=DecisionReference("resource:one", digest),
        retry_policy=RetryPolicy.FIXED,
        retry_count=1,
        timeout_policy=TimeoutPolicy.FAIL,
        timeout_seconds=60,
    )
    first = freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        decision_context=base_context,
    )
    changed = replace(base_context, target_branch="release")
    second = freeze_release_plan(
        graph,
        selection,
        version_plan,
        source_sha=SOURCE_SHA,
        base_sha=BASE_SHA,
        generation_id="generation:fixture",
        decision_context=changed,
    )
    assert first.plan_digest != second.plan_digest
    assert first.stages[0].stage_id != second.stages[0].stage_id
    assert first.stages[0].resource_profile == base_context.resource_profile


def test_trusted_runtime_errors_are_not_normalized(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    graph, selection, version_plan = _fixture()
    import repository_manager.development.workspace_release_plan as module

    def explode(_value: object) -> str:
        raise RuntimeError("trusted injected failure")

    monkeypatch.setattr(module, "_canonical_json", explode)
    with pytest.raises(RuntimeError, match="trusted injected failure"):
        freeze_release_plan(
            graph,
            selection,
            version_plan,
            source_sha=SOURCE_SHA,
            base_sha=BASE_SHA,
            generation_id="generation:fixture",
        )
