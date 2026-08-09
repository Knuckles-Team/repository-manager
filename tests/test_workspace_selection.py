"""Adversarial checkpoint-2 tests for selection closure and phase shadowing."""

from __future__ import annotations

import json
import random
from dataclasses import replace

import pytest

from repository_manager.development.workspace_release import (
    DependencyGraph,
    DependencySpec,
    Ecosystem,
    GraphDiagnosticCode,
    GraphValidationError,
    LegacyPhase,
    LegacyPhaseManifest,
    PackageKey,
    PackageRecord,
    PackageReference,
    ProjectRecord,
    Version,
    VersionSource,
    WorkspaceReleaseError,
    build_dependency_graph,
    phase_manifest_from_mapping,
)
from repository_manager.development.workspace_selection import (
    MAX_SHADOW_DIAGNOSTICS,
    InclusionMode,
    SelectionError,
    SelectionPolicy,
    SelectionReason,
    ShadowDiagnosticCode,
    compare_legacy_phases,
    derive_selected_closure,
)


def _package(
    repository: str,
    name: str,
    *,
    dependencies: tuple[DependencySpec, ...] = (),
) -> PackageRecord:
    key = PackageKey(repository, Ecosystem.PYTHON, name)
    version = Version("1.0.0")
    return PackageRecord(
        key=key,
        version=version,
        version_sources=(VersionSource("fixture", version),),
        dependencies=dependencies,
    )


def _project(repository: str, *packages: PackageRecord) -> ProjectRecord:
    return ProjectRecord(repository_id=repository, packages=packages)


def _dependency(name: str, repository: str) -> DependencySpec:
    return DependencySpec(
        target=PackageReference(Ecosystem.PYTHON, name, repository),
        source="fixture",
    )


def _chain_graph() -> DependencyGraph:
    leaf = _project("packages/leaf", _package("packages/leaf", "leaf"))
    middle = _project(
        "packages/middle",
        _package(
            "packages/middle",
            "middle",
            dependencies=(_dependency("leaf", "packages/leaf"),),
        ),
    )
    changed = _project(
        "packages/changed",
        _package(
            "packages/changed",
            "changed",
            dependencies=(_dependency("middle", "packages/middle"),),
        ),
    )
    independent = _project(
        "packages/independent",
        _package("packages/independent", "independent"),
    )
    return build_dependency_graph([changed, independent, middle, leaf])


def _codes(report: object) -> set[ShadowDiagnosticCode]:
    return {diagnostic.code for diagnostic in report.diagnostics}  # type: ignore[attr-defined]


def test_transitive_upstream_and_downstream_closure_explains_every_project() -> None:
    graph = _chain_graph()
    policy = SelectionPolicy(
        changed_projects=("packages/changed",),
        upstream_mode=InclusionMode.TRANSITIVE,
        downstream_mode=InclusionMode.NONE,
    )

    closure = derive_selected_closure(graph, policy)

    assert closure.selected_project_ids == (
        "repo:packages/changed",
        "repo:packages/leaf",
        "repo:packages/middle",
    )
    assert closure.parallel_groups == (
        ("repo:packages/leaf",),
        ("repo:packages/middle",),
        ("repo:packages/changed",),
    )
    explanations = {item.project_id: item for item in closure.explanations}
    assert explanations["repo:packages/changed"].reasons == (SelectionReason.CHANGED,)
    assert explanations["repo:packages/middle"].reasons == (SelectionReason.UPSTREAM,)
    assert explanations["repo:packages/middle"].via_projects == (
        "repo:packages/changed",
    )
    assert explanations["repo:packages/leaf"].via_projects == ("repo:packages/middle",)
    assert explanations["repo:packages/independent"].included is False
    assert explanations["repo:packages/independent"].reasons == (
        SelectionReason.EXCLUDED,
    )

    downstream = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/leaf",),
            downstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    assert downstream.selected_project_ids == (
        "repo:packages/changed",
        "repo:packages/leaf",
        "repo:packages/middle",
    )
    assert downstream.parallel_groups == closure.parallel_groups


def test_direct_direction_keeps_one_hop_boundary_and_independent_nodes_parallel() -> (
    None
):
    graph = _chain_graph()
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.DIRECT,
        ),
    )

    assert closure.selected_project_ids == (
        "repo:packages/changed",
        "repo:packages/middle",
    )
    assert closure.parallel_groups == (
        ("repo:packages/middle",),
        ("repo:packages/changed",),
    )

    with_independent = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            selected_projects=(
                "packages/changed",
                "packages/independent",
            ),
            upstream_mode=InclusionMode.DIRECT,
        ),
    )
    assert with_independent.parallel_groups[0] == (
        "repo:packages/independent",
        "repo:packages/middle",
    )


def test_same_project_package_edge_is_preserved_without_project_self_cycle() -> None:
    package_b = _package("packages/same", "b")
    package_a = _package(
        "packages/same",
        "a",
        dependencies=(_dependency("b", "packages/same"),),
    )
    graph = build_dependency_graph([_project("packages/same", package_a, package_b)])
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(changed_projects=("packages/same",)),
    )

    assert len(closure.edges) == 1
    assert (
        closure.edges[0].dependent_project_id == closure.edges[0].dependency_project_id
    )
    assert closure.project_edges == ()
    assert closure.parallel_groups == (("repo:packages/same",),)


def test_policy_is_frozen_strict_and_unknown_or_contradictory_roots_refuse() -> None:
    with pytest.raises(SelectionError):
        SelectionPolicy(changed_projects=())
    with pytest.raises(SelectionError):
        SelectionPolicy(
            changed_projects=("packages/changed",),
            selected_projects=("packages/other",),
        )
    with pytest.raises(SelectionError):
        SelectionPolicy(
            changed_projects=("packages/changed", "repo:packages/changed"),
        )
    with pytest.raises(SelectionError):
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode="transitive",  # type: ignore[arg-type]
        )

    with pytest.raises(GraphValidationError) as captured:
        derive_selected_closure(
            _chain_graph(),
            SelectionPolicy(changed_projects=("packages/unknown",)),
        )
    assert {item.code for item in captured.value.diagnostics} == {
        GraphDiagnosticCode.MISSING_PROJECT,
    }


def test_cycle_refuses_during_selection_even_if_a_malformed_graph_is_supplied() -> None:
    a = _project(
        "packages/a",
        _package(
            "packages/a",
            "a",
            dependencies=(_dependency("b", "packages/b"),),
        ),
    )
    b = _project(
        "packages/b",
        _package(
            "packages/b",
            "b",
            dependencies=(_dependency("a", "packages/a"),),
        ),
    )
    with pytest.raises(GraphValidationError) as captured:
        build_dependency_graph([a, b])
    assert {item.code for item in captured.value.diagnostics} == {
        GraphDiagnosticCode.CYCLE,
    }


def test_selection_digest_and_explanations_ignore_project_iteration_order() -> None:
    graph = _chain_graph()
    shuffled = list(graph.projects)
    random.Random(17).shuffle(shuffled)
    permuted = build_dependency_graph(reversed(shuffled))
    policy = SelectionPolicy(
        changed_projects=("packages/changed",),
        upstream_mode=InclusionMode.TRANSITIVE,
    )

    first = derive_selected_closure(graph, policy)
    second = derive_selected_closure(permuted, policy)

    assert first.digest == second.digest
    assert first.canonical_payload() == second.canonical_payload()
    assert first.explanations == second.explanations


def test_bare_selection_identity_does_not_collapse_same_basename_projects() -> None:
    first = _project("services/foo", _package("services/foo", "foo"))
    second = _project("agent-packages/foo", _package("agent-packages/foo", "foo"))
    graph = build_dependency_graph([second, first])
    with pytest.raises(GraphValidationError) as captured:
        derive_selected_closure(
            graph,
            SelectionPolicy(changed_projects=("foo",)),
        )
    assert {item.code for item in captured.value.diagnostics} == {
        GraphDiagnosticCode.MISSING_PROJECT,
    }


def test_exact_phase_shadow_is_read_only_and_digestable() -> None:
    graph = _chain_graph()
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    raw = {
        "description": "legacy",
        "phases": [
            {
                "name": "change",
                "phase": 3,
                "projects": ["changed"],
            },
            {
                "name": "middle",
                "phase": 2,
                "projects": ["middle"],
            },
            {"name": "leaf", "phase": 1, "project": "leaf"},
        ],
    }
    before = json.loads(json.dumps(raw))
    manifest = phase_manifest_from_mapping(raw)
    report = compare_legacy_phases(closure, manifest)

    assert report.exact_equal is True
    assert report.diagnostics == ()
    assert report.manual_phases == report.derived_phases
    assert len(report.report_digest) == 64
    assert raw == before


def test_phase_shadow_reports_membership_order_and_bulk_differences() -> None:
    first = _project("packages/first", _package("packages/first", "first"))
    second = _project("packages/second", _package("packages/second", "second"))
    graph = build_dependency_graph([second, first])
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/first",),
            selected_projects=("packages/first", "packages/second"),
        ),
    )
    reordered = phase_manifest_from_mapping(
        {
            "phases": [
                {
                    "name": "all",
                    "phase": 1,
                    "projects": ["second", "first"],
                    "bulk_bump": True,
                }
            ]
        }
    )
    report = compare_legacy_phases(closure, reordered)
    assert report.exact_equal is False
    assert ShadowDiagnosticCode.PHASE_ORDER_MISMATCH in _codes(report)
    assert ShadowDiagnosticCode.PHASE_BULK_FLAG_MISMATCH in _codes(report)

    missing = phase_manifest_from_mapping(
        {
            "phases": [
                {"name": "only-one", "phase": 1, "project": "first"},
                {"name": "extra", "phase": 2, "project": "second"},
            ]
        }
    )
    missing_report = compare_legacy_phases(closure, missing)
    assert ShadowDiagnosticCode.PHASE_COUNT_MISMATCH in _codes(missing_report)
    assert ShadowDiagnosticCode.PHASE_MEMBERSHIP_MISMATCH in _codes(missing_report)


def test_same_basename_legacy_reference_is_ambiguous_and_canonical_reference_works() -> (
    None
):
    first = _project("services/foo", _package("services/foo", "foo"))
    second = _project("agent-packages/foo", _package("agent-packages/foo", "foo"))
    graph = build_dependency_graph([second, first])
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(changed_projects=("services/foo",)),
    )
    ambiguous = phase_manifest_from_mapping(
        {"phases": [{"name": "foo", "phase": 1, "project": "foo"}]}
    )
    report = compare_legacy_phases(closure, ambiguous)
    assert report.exact_equal is False
    assert ShadowDiagnosticCode.AMBIGUOUS_PROJECT in _codes(report)

    canonical = phase_manifest_from_mapping(
        {
            "phases": [
                {
                    "name": "foo",
                    "phase": 1,
                    "project": "repo:services/foo",
                }
            ]
        }
    )
    canonical_report = compare_legacy_phases(closure, canonical)
    assert canonical_report.exact_equal is True


def test_phase_parser_rejects_duplicate_refs_instead_of_silently_deduplicating() -> (
    None
):
    with pytest.raises(WorkspaceReleaseError):
        phase_manifest_from_mapping(
            {
                "phases": [
                    {
                        "name": "duplicate",
                        "phase": 1,
                        "project": "first",
                        "projects": ["first"],
                    }
                ]
            }
        )


def test_shadow_report_is_deterministic_for_manifest_iteration_permutations() -> None:
    graph = _chain_graph()
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    phases = [
        {"name": "change", "phase": 3, "project": "changed"},
        {"name": "leaf", "phase": 1, "project": "leaf"},
        {"name": "middle", "phase": 2, "project": "middle"},
    ]
    first = compare_legacy_phases(
        closure,
        phase_manifest_from_mapping({"phases": phases}),
    )
    second = compare_legacy_phases(
        closure,
        phase_manifest_from_mapping({"phases": list(reversed(phases))}),
    )
    assert first.report_digest == second.report_digest
    assert first.canonical_payload() == second.canonical_payload()


def test_shadow_comparator_fails_closed_on_oversized_manual_reference() -> None:
    with pytest.raises(WorkspaceReleaseError):
        phase_manifest_from_mapping(
            {
                "phases": [
                    {
                        "name": "oversized",
                        "phase": 1,
                        "project": "x" * 4097,
                    }
                ]
            }
        )


def test_wait_minutes_are_semantic_but_phase_names_are_display_only() -> None:
    graph = _chain_graph()
    closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    manifest = phase_manifest_from_mapping(
        {
            "phases": [
                {"name": "renamed-display-label", "phase": 1, "project": "leaf"},
                {"name": "different-label", "phase": 2, "project": "middle"},
                {
                    "name": "another-label",
                    "phase": 3,
                    "project": "changed",
                    "wait_minutes": 999,
                },
            ]
        }
    )
    report = compare_legacy_phases(closure, manifest)

    assert report.exact_equal is False
    assert ShadowDiagnosticCode.PHASE_WAIT_MISMATCH in _codes(report)
    assert ShadowDiagnosticCode.PHASE_ORDER_MISMATCH not in _codes(report)
    assert report.manual_phases[-1].wait_minutes == 999

    renamed_only = phase_manifest_from_mapping(
        {
            "phases": [
                {"name": "label-a", "phase": 1, "project": "leaf"},
                {"name": "label-b", "phase": 2, "project": "middle"},
                {"name": "label-c", "phase": 3, "project": "changed"},
            ]
        }
    )
    assert compare_legacy_phases(closure, renamed_only).exact_equal is True


def test_closure_rejects_dishonest_explanations_and_missing_source_evidence() -> None:
    closure = derive_selected_closure(
        _chain_graph(),
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    changed_index = next(
        index
        for index, explanation in enumerate(closure.explanations)
        if explanation.project_id == "repo:packages/changed"
    )
    dishonest = list(closure.explanations)
    dishonest[changed_index] = replace(
        dishonest[changed_index],
        included=False,
        reasons=(SelectionReason.EXCLUDED,),
        via_projects=(),
    )
    with pytest.raises(SelectionError):
        replace(closure, explanations=tuple(dishonest), digest="")

    with pytest.raises(WorkspaceReleaseError):
        replace(
            closure,
            source_graph=replace(closure.source_graph, project_edges=()),
            digest="",
        )

    graph = _chain_graph()
    all_projects = tuple(project.project_id for project in graph.projects)
    all_closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            selected_projects=all_projects,
        ),
    )
    with pytest.raises(SelectionError):
        replace(
            all_closure,
            policy=SelectionPolicy(changed_projects=("packages/changed",)),
            digest="",
        )


def test_graph_inventory_rejects_substituted_package_records_and_digest() -> None:
    graph = _chain_graph()
    original = graph.packages[0]
    altered_version = Version("2.0.0")
    altered = replace(
        original,
        version=altered_version,
        version_sources=(VersionSource("fixture", altered_version),),
    )
    altered_packages = (altered, *graph.packages[1:])
    with pytest.raises(GraphValidationError) as package_capture:
        derive_selected_closure(
            replace(graph, packages=altered_packages),
            SelectionPolicy(changed_projects=("packages/changed",)),
        )
    assert GraphDiagnosticCode.INVALID_METADATA in {
        item.code for item in package_capture.value.diagnostics
    }

    with pytest.raises(GraphValidationError) as digest_capture:
        derive_selected_closure(
            replace(graph, digest="f" * 64),
            SelectionPolicy(changed_projects=("packages/changed",)),
        )
    assert GraphDiagnosticCode.INVALID_METADATA in {
        item.code for item in digest_capture.value.diagnostics
    }


def test_directly_reconstructed_closure_rejects_duplicate_package_records() -> None:
    closure = derive_selected_closure(
        _chain_graph(),
        SelectionPolicy(changed_projects=("packages/changed",)),
    )
    first_project = closure.projects[0]
    duplicate_project = replace(
        first_project,
        packages=(first_project.packages[0], first_project.packages[0]),
    )
    with pytest.raises(SelectionError):
        replace(
            closure,
            projects=(duplicate_project,),
            digest="",
        )


def test_shadow_reports_each_unmatched_trailing_phase() -> None:
    graph = _chain_graph()
    derived_closure = derive_selected_closure(
        graph,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    trailing_derived = compare_legacy_phases(
        derived_closure,
        phase_manifest_from_mapping(
            {"phases": [{"phase": 1, "name": "leaf", "project": "leaf"}]}
        ),
    )
    assert ShadowDiagnosticCode.PHASE_COUNT_MISMATCH in _codes(trailing_derived)
    assert [
        item.subject
        for item in trailing_derived.diagnostics
        if item.code == ShadowDiagnosticCode.PHASE_TRAILING_DERIVED
    ] == ["phase-2", "phase-3"]

    single_closure = derive_selected_closure(
        graph,
        SelectionPolicy(changed_projects=("packages/changed",)),
    )
    trailing_manual = compare_legacy_phases(
        single_closure,
        phase_manifest_from_mapping(
            {
                "phases": [
                    {"phase": 1, "name": "changed", "project": "changed"},
                    {"phase": 2, "name": "extra", "project": "middle"},
                ]
            }
        ),
    )
    assert ShadowDiagnosticCode.PHASE_TRAILING_MANUAL in _codes(trailing_manual)
    assert any(
        item.subject == "phase-2"
        and item.code == ShadowDiagnosticCode.PHASE_TRAILING_MANUAL
        for item in trailing_manual.diagnostics
    )


def test_maximum_bounded_phase_mismatch_returns_a_bounded_report() -> None:
    closure = derive_selected_closure(
        _chain_graph(),
        SelectionPolicy(changed_projects=("packages/changed",)),
    )
    max_phases = 16_384
    manifest = phase_manifest_from_mapping(
        {
            "phases": [
                {"phase": index, "name": f"phase-{index}"}
                for index in range(1, max_phases + 1)
            ]
        }
    )
    report = compare_legacy_phases(closure, manifest)

    assert report.exact_equal is False
    assert len(report.diagnostics) <= MAX_SHADOW_DIAGNOSTICS
    assert (
        sum(
            item.code == ShadowDiagnosticCode.PHASE_TRAILING_MANUAL
            for item in report.diagnostics
        )
        == max_phases - 1
    )
    assert all(
        len(value) <= 4096 for item in report.diagnostics for _, value in item.details
    )


def test_frozen_legacy_models_copy_lists_and_consume_generators_once() -> None:
    references = ["leaf"]
    phase = LegacyPhase(
        name="leaf",
        phase=1,
        project_references=(item for item in references),
    )
    references.append("changed")
    assert phase.project_references == ("leaf",)

    phases = [phase]
    manifest = LegacyPhaseManifest(phases=(item for item in phases))
    phases.clear()
    assert manifest.phases == (phase,)
    assert manifest.phases == (phase,)


def test_dependency_graph_copies_nested_generators_and_repeats_without_drift() -> None:
    graph = _chain_graph()
    projects = list(graph.projects)
    packages = list(graph.packages)
    edges = list(graph.edges)
    project_edges = [list(pair) for pair in graph.project_edges]
    parallel_groups = [list(group) for group in graph.parallel_groups]
    frozen = DependencyGraph(
        projects=(item for item in projects),
        packages=(item for item in packages),
        edges=(item for item in edges),
        project_edges=(pair for pair in project_edges),
        parallel_groups=(group for group in parallel_groups),
        digest=graph.digest,
    )

    projects.clear()
    packages.clear()
    edges.clear()
    project_edges[0][0] = "repo:packages/mutated"
    parallel_groups[0].clear()

    assert frozen.projects == graph.projects
    assert frozen.packages == graph.packages
    assert frozen.edges == graph.edges
    assert frozen.project_edges == graph.project_edges
    assert frozen.parallel_groups == graph.parallel_groups
    first = derive_selected_closure(
        frozen,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    second = derive_selected_closure(
        frozen,
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    assert first.digest == second.digest
    assert first.canonical_payload() == second.canonical_payload()


def test_closure_binds_complete_unselected_source_edges_and_changes_digest() -> None:
    graph = _chain_graph()
    policy = SelectionPolicy(changed_projects=("packages/changed",))
    closure = derive_selected_closure(graph, policy)
    independent = next(
        project
        for project in graph.projects
        if project.project_id == "repo:packages/independent"
    )
    independent_package = replace(
        independent.packages[0],
        dependencies=(_dependency("leaf", "packages/leaf"),),
    )
    enriched_projects = tuple(
        replace(
            project,
            packages=(independent_package,)
            if project.project_id == independent.project_id
            else project.packages,
        )
        for project in graph.projects
    )
    enriched_graph = build_dependency_graph(enriched_projects)
    enriched_closure = derive_selected_closure(enriched_graph, policy)

    assert enriched_closure.selected_project_ids == closure.selected_project_ids
    assert enriched_closure.source_project_edges != closure.source_project_edges
    assert enriched_closure.digest != closure.digest
    with pytest.raises(SelectionError):
        replace(
            closure,
            source_graph=enriched_graph,
            digest=closure.digest,
        )
    with pytest.raises(SelectionError):
        replace(
            enriched_closure,
            source_graph=graph,
            digest=enriched_closure.digest,
        )


def test_closure_revalidates_the_authoritative_source_graph_digest() -> None:
    closure = derive_selected_closure(
        _chain_graph(),
        SelectionPolicy(changed_projects=("packages/changed",)),
    )
    with pytest.raises(GraphValidationError):
        replace(
            closure,
            source_graph=replace(closure.source_graph, digest="f" * 64),
            digest="",
        )


def test_closure_selected_records_and_edges_must_match_source_graph() -> None:
    closure = derive_selected_closure(
        _chain_graph(),
        SelectionPolicy(
            changed_projects=("packages/changed",),
            upstream_mode=InclusionMode.TRANSITIVE,
        ),
    )
    tampered_project = replace(closure.projects[0], tree_sha="a" * 40)
    with pytest.raises(SelectionError):
        replace(closure, projects=(tampered_project, *closure.projects[1:]), digest="")

    tampered_edge = replace(closure.edges[0], source="tampered-source")
    with pytest.raises(SelectionError):
        replace(closure, edges=(tampered_edge, *closure.edges[1:]), digest="")
