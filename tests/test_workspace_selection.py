"""Adversarial checkpoint-2 tests for selection closure and phase shadowing."""

from __future__ import annotations

import json
import random

import pytest

from repository_manager.development.workspace_release import (
    DependencyGraph,
    DependencySpec,
    Ecosystem,
    GraphDiagnosticCode,
    GraphValidationError,
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
