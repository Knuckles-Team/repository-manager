"""RF016 source-bound fleet preflight behavior."""

from __future__ import annotations

from dataclasses import replace

import pytest

from repository_manager.development.workspace_release import (
    Ecosystem,
    FleetCycleError,
    FleetEdgeObservation,
    FleetEvidenceKind,
    PackageKey,
    PackageRecord,
    ProjectRecord,
    Version,
    VersionSource,
    WorkspaceReleaseError,
    build_dependency_graph,
    build_fleet_order_receipt,
    find_fleet_sccs,
)
from repository_manager.development.workspace_release_plan import (
    ReleasePlanError,
    _digest_payload,
    bind_fleet_release_plan,
)
from tests.test_workspace_release_plan import _plan


def _graph():
    projects = []
    for repository in ("services/api", "services/db", "apps/frontend"):
        name = repository.rsplit("/", 1)[1]
        key = PackageKey(repository, Ecosystem.PYTHON, name)
        version = Version("1.0.0")
        package = PackageRecord(key, version, (VersionSource("fixture", version),))
        projects.append(ProjectRecord(repository, packages=(package,)))
    return build_dependency_graph(projects)


def _coverage():
    return {kind: "a" * 64 for kind in FleetEvidenceKind}


def test_fleet_order_is_deterministic_and_binds_each_source_partition():
    graph = _graph()
    rows = (
        FleetEdgeObservation(
            FleetEvidenceKind.DEPLOYMENT, "services/api", "services/db", "b" * 64
        ),
        FleetEdgeObservation(
            FleetEvidenceKind.FRONTEND, "apps/frontend", "services/api", "c" * 64
        ),
    )
    first = build_fleet_order_receipt(
        graph,
        manifest_digest="d" * 64,
        observations=rows,
        coverage_receipts=_coverage(),
    )
    second = build_fleet_order_receipt(
        graph,
        manifest_digest="d" * 64,
        observations=reversed(rows),
        coverage_receipts=_coverage(),
    )
    assert first == second
    assert first.parallel_groups == (
        ("repo:services/db",),
        ("repo:services/api",),
        ("repo:apps/frontend",),
    )
    assert replace(first, digest="") != first


def test_fleet_order_refuses_missing_receipt_unknown_endpoint_and_cycle():
    graph = _graph()
    missing = _coverage()
    del missing[FleetEvidenceKind.DYNAMIC]
    with pytest.raises(WorkspaceReleaseError, match="all four"):
        build_fleet_order_receipt(
            graph, manifest_digest="d" * 64, observations=(), coverage_receipts=missing
        )
    with pytest.raises(WorkspaceReleaseError, match="unknown repository endpoint"):
        build_fleet_order_receipt(
            graph,
            manifest_digest="d" * 64,
            observations=(
                FleetEdgeObservation(
                    FleetEvidenceKind.RESOLVER,
                    "services/api",
                    "services/unknown",
                    "b" * 64,
                ),
            ),
            coverage_receipts=_coverage(),
        )
    with pytest.raises(FleetCycleError) as caught:
        build_fleet_order_receipt(
            graph,
            manifest_digest="d" * 64,
            observations=(
                FleetEdgeObservation(
                    FleetEvidenceKind.DYNAMIC, "services/api", "services/db", "b" * 64
                ),
                FleetEdgeObservation(
                    FleetEvidenceKind.DYNAMIC, "services/db", "services/api", "c" * 64
                ),
            ),
            coverage_receipts=_coverage(),
        )
    assert caught.value.components[0].members == (
        "repo:services/api",
        "repo:services/db",
    )
    assert caught.value.components[0].edges == (
        ("repo:services/api", "repo:services/db"),
        ("repo:services/db", "repo:services/api"),
    )


def test_fleet_scc_census_reports_each_exact_component():
    nodes = tuple(f"repo:service/{name}" for name in "abcd")
    components = find_fleet_sccs(
        nodes,
        (
            (nodes[0], nodes[1]),
            (nodes[1], nodes[0]),
            (nodes[2], nodes[3]),
            (nodes[3], nodes[2]),
        ),
    )
    assert tuple(item.members for item in components) == (
        (nodes[0], nodes[1]),
        (nodes[2], nodes[3]),
    )
    assert all(len(item.edges) == 2 and len(item.digest) == 64 for item in components)


def test_fleet_plan_binding_rejects_stale_or_tampered_order():
    plan = _plan()
    order = build_fleet_order_receipt(
        plan.graph,
        manifest_digest="d" * 64,
        observations=(),
        coverage_receipts=_coverage(),
    )
    binding = bind_fleet_release_plan(plan, order)
    assert binding.plan_digest == plan.digest
    assert binding.fleet_order_digest == order.digest
    with pytest.raises(ReleasePlanError, match="fleet order digest"):
        bind_fleet_release_plan(plan, replace(order, manifest_digest="e" * 64))
    other = build_fleet_order_receipt(
        _graph(),
        manifest_digest="d" * 64,
        observations=(),
        coverage_receipts=_coverage(),
    )
    with pytest.raises(ReleasePlanError, match="fleet order"):
        bind_fleet_release_plan(plan, other)
    forged = replace(order, parallel_groups=tuple(reversed(order.parallel_groups)))
    forged = replace(forged, digest=_digest_payload(forged.canonical_payload()))
    with pytest.raises(ReleasePlanError, match="fleet order receipt"):
        bind_fleet_release_plan(plan, forged)
