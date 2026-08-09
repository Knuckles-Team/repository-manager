"""Pure selected-change closure and legacy phase shadow comparison.

Checkpoint 2 keeps selection and compatibility comparison as immutable data
operations.  The module consumes the checkpoint-1 dependency graph and phase
manifest view only; it does not inspect a checkout, execute project code, or
mutate a manifest.

Project identities in this module are always canonical ``repo:<path>`` values.
Bare legacy phase references are resolved only when their basename has exactly
one owner in the known graph.  A basename shared by two repositories is an
ambiguity, never an implicit choice.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import TypeVar, cast

from .workspace_release import (
    MAX_EDGES,
    MAX_PACKAGES,
    MAX_PLAN_STAGES,
    MAX_PROJECTS,
    MAX_STRING_LENGTH,
    DependencyEdge,
    DependencyGraph,
    Diagnostic,
    GraphDiagnosticCode,
    GraphValidationError,
    LegacyPhase,
    LegacyPhaseManifest,
    PackageRecord,
    ProjectRecord,
    WorkspaceReleaseError,
    _canonical_json,
    _edge_payload,
    _package_payload,
    _project_payload,
    _topological_groups,
    canonical_repository_id,
)

_T = TypeVar("_T")
# One count diagnostic, one required diagnostic per unmatched phase, and one
# overflow summary keep the maximum valid manifest bounded without permitting
# six-figure amplification.
MAX_SHADOW_DIAGNOSTICS = MAX_PLAN_STAGES + 2


class SelectionError(WorkspaceReleaseError):
    """A selection policy or derived closure is not safe to use."""


class InclusionMode(StrEnum):
    """How far selection should travel in one dependency direction."""

    NONE = "none"
    DIRECT = "direct"
    TRANSITIVE = "transitive"


# The aliases make the contract discoverable to callers that call the two
# directions "dependency" and "dependent" rather than "upstream" and
# "downstream".  They are type aliases, not separate accepted wire values.
SelectionMode = InclusionMode
ProjectInclusionMode = InclusionMode


class SelectionReason(StrEnum):
    """Stable reason labels included in every closure explanation."""

    CHANGED = "changed"
    EXPLICIT = "explicit"
    UPSTREAM = "upstream"
    DOWNSTREAM = "downstream"
    EXCLUDED = "excluded"


def _bounded_text(
    value: object, field_name: str, *, max_length: int = MAX_STRING_LENGTH
) -> str:
    if not isinstance(value, str):
        raise SelectionError(f"{field_name} must be a string")
    if not value or value.strip() != value:
        raise SelectionError(f"{field_name} must be non-blank and trimmed")
    if len(value) > max_length:
        raise SelectionError(f"{field_name} exceeds the bounded length")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise SelectionError(f"{field_name} contains a control character")
    return value


def _bounded_sequence(
    value: object, field_name: str, *, max_items: int
) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value, Iterable
    ):
        raise SelectionError(f"{field_name} must be a sequence")
    result = tuple(value)
    if len(result) > max_items:
        raise SelectionError(f"{field_name} exceeds the bounded item count")
    return result


def _typed_sequence(
    value: object,
    field_name: str,
    item_type: type[_T],
    *,
    max_items: int,
) -> tuple[_T, ...]:
    values = _bounded_sequence(value, field_name, max_items=max_items)
    if any(not isinstance(item, item_type) for item in values):
        raise SelectionError(
            f"{field_name} entries must be {item_type.__name__} values"
        )
    return cast(tuple[_T, ...], values)


def _canonical_ids(
    value: object, field_name: str, *, allow_empty: bool
) -> tuple[str, ...]:
    values = _bounded_sequence(value, field_name, max_items=MAX_PROJECTS)
    if not values and not allow_empty:
        raise SelectionError(f"{field_name} must not be empty")
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise SelectionError(f"{field_name} entries must be strings")
        result.append(canonical_repository_id(item))
    if len(result) != len(set(result)):
        raise SelectionError(f"{field_name} must not contain duplicate projects")
    return tuple(sorted(result))


def _canonical_refs(value: object, field_name: str) -> tuple[str, ...]:
    """Canonicalize ordered project references without changing their order."""

    values = _bounded_sequence(value, field_name, max_items=MAX_PROJECTS)
    result: list[str] = []
    for item in values:
        if not isinstance(item, str):
            raise SelectionError(f"{field_name} entries must be strings")
        result.append(canonical_repository_id(item))
    return tuple(result)


@dataclass(frozen=True, slots=True)
class SelectionPolicy:
    """Frozen roots and directional closure rules for one selected change."""

    changed_projects: tuple[str, ...]
    selected_projects: tuple[str, ...] = ()
    upstream_mode: InclusionMode = InclusionMode.NONE
    downstream_mode: InclusionMode = InclusionMode.NONE

    def __post_init__(self) -> None:
        changed = _canonical_ids(
            self.changed_projects,
            "changed projects",
            allow_empty=False,
        )
        selected = _canonical_ids(
            self.selected_projects,
            "selected projects",
            allow_empty=True,
        )
        if not isinstance(self.upstream_mode, InclusionMode):
            raise SelectionError("upstream mode must use a supported inclusion mode")
        if not isinstance(self.downstream_mode, InclusionMode):
            raise SelectionError("downstream mode must use a supported inclusion mode")
        if selected and not set(changed).issubset(selected):
            missing = ", ".join(sorted(set(changed) - set(selected)))
            raise SelectionError(
                "selected projects must include every changed project: " + missing
            )
        object.__setattr__(self, "changed_projects", changed)
        object.__setattr__(self, "selected_projects", selected)

    @property
    def roots(self) -> tuple[str, ...]:
        """Return explicit roots, or changed projects when no explicit set exists."""

        return self.selected_projects or self.changed_projects

    @property
    def dependency_mode(self) -> InclusionMode:
        """Alias for the upstream/dependency inclusion mode."""

        return self.upstream_mode

    @property
    def dependent_mode(self) -> InclusionMode:
        """Alias for the downstream/dependent inclusion mode."""

        return self.downstream_mode

    def canonical_payload(self) -> dict[str, object]:
        return {
            "changed_projects": self.changed_projects,
            "selected_projects": self.selected_projects,
            "upstream_mode": self.upstream_mode.value,
            "downstream_mode": self.downstream_mode.value,
        }


@dataclass(frozen=True, slots=True)
class SelectionExplanation:
    """Why one known project was included or excluded from a closure."""

    project_id: str
    included: bool
    reasons: tuple[SelectionReason, ...]
    via_projects: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "project_id", canonical_repository_id(self.project_id))
        if not isinstance(self.included, bool):
            raise SelectionError("selection explanation included must be a boolean")
        reasons = _typed_sequence(
            self.reasons,
            "selection explanation reasons",
            SelectionReason,
            max_items=5,
        )
        if len(reasons) != len(set(reasons)):
            raise SelectionError("selection explanation reasons must be unique")
        if self.included:
            if not reasons or SelectionReason.EXCLUDED in reasons:
                raise SelectionError(
                    "included explanation must have non-excluded reasons"
                )
        elif reasons != (SelectionReason.EXCLUDED,):
            raise SelectionError(
                "excluded explanation must contain only the excluded reason"
            )
        via = _canonical_ids(
            self.via_projects,
            "selection explanation via projects",
            allow_empty=True,
        )
        object.__setattr__(
            self, "reasons", tuple(sorted(reasons, key=lambda item: item.value))
        )
        object.__setattr__(self, "via_projects", via)

    def canonical_payload(self) -> dict[str, object]:
        return {
            "project_id": self.project_id,
            "included": self.included,
            "reasons": tuple(reason.value for reason in self.reasons),
            "via_projects": self.via_projects,
        }


def _edge_sort_key(edge: DependencyEdge) -> tuple[str, str, str, str, str]:
    return (
        edge.dependent.value,
        edge.dependency.value,
        edge.floor.value if edge.floor else "",
        edge.source,
        edge.confidence.value,
    )


def _graph_inventory(
    graph: DependencyGraph,
) -> tuple[
    tuple[str, ...],
    dict[str, ProjectRecord],
    dict[str, PackageRecord],
    tuple[DependencyEdge, ...],
    tuple[tuple[str, str], ...],
]:
    """Validate and normalize a graph before applying a selection policy."""

    if not isinstance(graph, DependencyGraph):
        raise SelectionError("selection requires a DependencyGraph")
    projects = _typed_sequence(
        graph.projects,
        "graph projects",
        ProjectRecord,
        max_items=MAX_PROJECTS,
    )
    project_map: dict[str, ProjectRecord] = {}
    diagnostics: list[Diagnostic] = []
    for project in projects:
        project_id = project.project_id
        if project_id in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.DUPLICATE_PROJECT,
                    project_id,
                    "graph contains a duplicate canonical project identity",
                )
            )
        else:
            project_map[project_id] = project
    if len(project_map) > MAX_PROJECTS:
        raise SelectionError("graph project count exceeds the bound")

    package_map: dict[str, PackageRecord] = {}
    for project in project_map.values():
        for package in project.packages:
            package_id = package.key.value
            if package_id in package_map:
                diagnostics.append(
                    Diagnostic(
                        GraphDiagnosticCode.DUPLICATE_PACKAGE,
                        package_id,
                        "graph contains a duplicate package identity",
                    )
                )
            else:
                package_map[package_id] = package
    if len(package_map) > MAX_PACKAGES:
        raise SelectionError("graph package count exceeds the bound")
    graph_packages = _typed_sequence(
        graph.packages,
        "graph packages",
        PackageRecord,
        max_items=MAX_PACKAGES,
    )
    graph_package_map = {package.key.value: package for package in graph_packages}
    if (
        len(graph_package_map) != len(graph_packages)
        or graph_package_map != package_map
    ):
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.INVALID_METADATA,
                "workspace dependency graph",
                "graph package records do not exactly match frozen project packages",
            )
        )

    edges = _typed_sequence(
        graph.edges,
        "graph edges",
        DependencyEdge,
        max_items=MAX_EDGES,
    )
    edge_map: dict[tuple[str, str], DependencyEdge] = {}
    for edge in edges:
        key = (edge.dependent.value, edge.dependency.value)
        if key in edge_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.DUPLICATE_EDGE,
                    f"{key[0]}->{key[1]}",
                    "graph contains a duplicate package edge",
                )
            )
            continue
        if edge.dependent.value not in package_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PACKAGE,
                    edge.dependent.value,
                    "graph edge dependent package is not frozen",
                )
            )
        if edge.dependency.value not in package_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PACKAGE,
                    edge.dependency.value,
                    "graph edge dependency package is not frozen",
                )
            )
        if edge.dependent_project_id not in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PROJECT,
                    edge.dependent_project_id,
                    "graph edge dependent project is not frozen",
                )
            )
        if edge.dependency_project_id not in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PROJECT,
                    edge.dependency_project_id,
                    "graph edge dependency project is not frozen",
                )
            )
        edge_map[key] = edge

    project_edge_values = _bounded_sequence(
        graph.project_edges,
        "graph project edges",
        max_items=MAX_EDGES,
    )
    project_edges: list[tuple[str, str]] = []
    project_edge_set: set[tuple[str, str]] = set()
    for item in project_edge_values:
        if not isinstance(item, (tuple, list)) or len(item) != 2:
            raise SelectionError("graph project edges must contain pairs")
        dependent, dependency = item
        if not isinstance(dependent, str) or not isinstance(dependency, str):
            raise SelectionError("graph project edge endpoints must be strings")
        dependent_id = canonical_repository_id(dependent)
        dependency_id = canonical_repository_id(dependency)
        pair = (dependent_id, dependency_id)
        if pair in project_edge_set:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.DUPLICATE_EDGE,
                    f"{dependent_id}->{dependency_id}",
                    "graph contains a duplicate project edge",
                )
            )
            continue
        project_edge_set.add(pair)
        project_edges.append(pair)
        if dependent_id == dependency_id:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.INVALID_METADATA,
                    dependent_id,
                    "project-level self edges are not allowed",
                )
            )
        if dependent_id not in project_map or dependency_id not in project_map:
            diagnostics.append(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PROJECT,
                    f"{dependent_id}->{dependency_id}",
                    "graph project edge endpoint is not frozen",
                )
            )
    expected_project_edges = {
        (edge.dependent_project_id, edge.dependency_project_id)
        for edge in edge_map.values()
        if edge.dependent_project_id != edge.dependency_project_id
        and edge.dependent_project_id in project_map
        and edge.dependency_project_id in project_map
    }
    if set(project_edge_set) != expected_project_edges:
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.INVALID_METADATA,
                "workspace dependency graph",
                "project edges do not match frozen package edges",
            )
        )
    ordered_projects = tuple(sorted(project_map))
    expected_groups, cycle_diagnostics = _topological_groups(
        ordered_projects,
        tuple(sorted(project_edge_set)),
    )
    diagnostics.extend(cycle_diagnostics)
    graph_groups = _bounded_sequence(
        graph.parallel_groups,
        "graph parallel groups",
        max_items=MAX_PROJECTS,
    )
    if graph_groups != expected_groups:
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.INVALID_METADATA,
                "workspace dependency graph",
                "graph parallel groups do not match deterministic topology",
            )
        )
    ordered_edges = tuple(sorted(edge_map.values(), key=_edge_sort_key))
    ordered_project_edges = tuple(sorted(project_edge_set))
    ordered_packages = tuple(
        sorted(package_map.values(), key=lambda item: item.key.value)
    )
    digest_value = graph.digest
    try:
        digest_value = _bounded_text(digest_value, "graph digest", max_length=64)
    except WorkspaceReleaseError:
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.INVALID_METADATA,
                "workspace dependency graph",
                "graph digest is not a bounded SHA-256 value",
            )
        )
    expected_digest = hashlib.sha256(
        _canonical_json(
            {
                "projects": [
                    _project_payload(project_map[project_id])
                    for project_id in ordered_projects
                ],
                "packages": [_package_payload(package) for package in ordered_packages],
                "edges": [_edge_payload(edge) for edge in ordered_edges],
                "project_edges": ordered_project_edges,
                "parallel_groups": expected_groups,
            }
        ).encode("utf-8")
    ).hexdigest()
    if (
        not isinstance(digest_value, str)
        or len(digest_value) != 64
        or any(char not in "0123456789abcdefABCDEF" for char in digest_value)
        or digest_value.lower() != expected_digest
    ):
        diagnostics.append(
            Diagnostic(
                GraphDiagnosticCode.INVALID_METADATA,
                "workspace dependency graph",
                "graph digest does not match canonical graph contents",
            )
        )
    if diagnostics:
        raise GraphValidationError(diagnostics)
    return (
        ordered_projects,
        project_map,
        package_map,
        ordered_edges,
        ordered_project_edges,
    )


def _expand_direction(
    roots: tuple[str, ...],
    adjacency: dict[str, tuple[str, ...]],
    mode: InclusionMode,
    reason: SelectionReason,
    included: set[str],
    reasons: dict[str, set[SelectionReason]],
    via_projects: dict[str, set[str]],
) -> None:
    if mode == InclusionMode.NONE:
        return
    if mode == InclusionMode.DIRECT:
        # Keep this loop explicit to make the one-hop boundary auditable.
        for source in roots:
            for target in adjacency[source]:
                included.add(target)
                reasons[target].add(reason)
                via_projects[target].add(source)
        return
    frontier_set = set(roots)
    visited = set(roots)
    while frontier_set:
        next_nodes: set[str] = set()
        for source in sorted(frontier_set):
            for target in adjacency[source]:
                included.add(target)
                reasons[target].add(reason)
                via_projects[target].add(source)
                if target not in visited:
                    visited.add(target)
                    next_nodes.add(target)
        frontier_set = next_nodes


def _selection_evidence(
    policy: SelectionPolicy,
    known: tuple[str, ...],
    project_edges: tuple[tuple[str, str], ...],
) -> tuple[set[str], dict[str, set[SelectionReason]], dict[str, set[str]]]:
    """Recompute closure membership/reasons from frozen policy and edge evidence."""

    known_set = set(known)
    if not set(policy.roots).issubset(known_set):
        raise SelectionError("selection policy roots must be known projects")
    upstream_sets: dict[str, set[str]] = {project_id: set() for project_id in known}
    downstream_sets: dict[str, set[str]] = {project_id: set() for project_id in known}
    for dependent, dependency in project_edges:
        upstream_sets[dependent].add(dependency)
        downstream_sets[dependency].add(dependent)
    upstream = {
        project_id: tuple(sorted(values))
        for project_id, values in upstream_sets.items()
    }
    downstream = {
        project_id: tuple(sorted(values))
        for project_id, values in downstream_sets.items()
    }
    included = set(policy.roots)
    reasons: dict[str, set[SelectionReason]] = {
        project_id: set() for project_id in known
    }
    via_projects: dict[str, set[str]] = {project_id: set() for project_id in known}
    for project_id in policy.changed_projects:
        reasons[project_id].add(SelectionReason.CHANGED)
    if policy.selected_projects:
        for project_id in policy.selected_projects:
            reasons[project_id].add(SelectionReason.EXPLICIT)
    _expand_direction(
        policy.roots,
        upstream,
        policy.upstream_mode,
        SelectionReason.UPSTREAM,
        included,
        reasons,
        via_projects,
    )
    _expand_direction(
        policy.roots,
        downstream,
        policy.downstream_mode,
        SelectionReason.DOWNSTREAM,
        included,
        reasons,
        via_projects,
    )
    return included, reasons, via_projects


def _closure_digest(closure: SelectedChangeClosure) -> str:
    return hashlib.sha256(
        json.dumps(
            closure.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class SelectedChangeClosure:
    """Frozen selected subgraph, explanations, and exact parallel groups."""

    policy: SelectionPolicy
    known_project_ids: tuple[str, ...]
    selected_project_ids: tuple[str, ...]
    projects: tuple[ProjectRecord, ...]
    edges: tuple[DependencyEdge, ...]
    project_edges: tuple[tuple[str, str], ...]
    parallel_groups: tuple[tuple[str, ...], ...]
    explanations: tuple[SelectionExplanation, ...]
    source_graph: DependencyGraph
    digest: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.policy, SelectionPolicy):
            raise SelectionError("closure policy must be a SelectionPolicy")
        if not isinstance(self.source_graph, DependencyGraph):
            raise SelectionError("closure requires a frozen source DependencyGraph")
        (
            source_known,
            source_project_map,
            _source_package_map,
            source_edges,
            source_project_edges,
        ) = _graph_inventory(self.source_graph)
        known = _canonical_ids(
            self.known_project_ids,
            "known project IDs",
            allow_empty=False,
        )
        selected = _canonical_ids(
            self.selected_project_ids,
            "selected project IDs",
            allow_empty=False,
        )
        if not set(selected).issubset(known):
            raise SelectionError("selected project IDs must be known project IDs")
        if known != source_known:
            raise SelectionError(
                "closure known projects must match the frozen source graph"
            )
        if not (
            set(self.policy.changed_projects) | set(self.policy.selected_projects)
        ).issubset(set(known)):
            raise SelectionError("closure policy must use known project IDs")
        project_values = _typed_sequence(
            self.projects,
            "closure projects",
            ProjectRecord,
            max_items=MAX_PROJECTS,
        )
        project_map = {project.project_id: project for project in project_values}
        if (
            len(project_map) != len(project_values)
            or tuple(sorted(project_map)) != selected
        ):
            raise SelectionError(
                "closure projects must exactly match selected project IDs"
            )
        expected_projects = tuple(
            source_project_map[project_id] for project_id in selected
        )
        if (
            tuple(project_map[project_id] for project_id in selected)
            != expected_projects
        ):
            raise SelectionError(
                "closure projects must exactly match the frozen source graph"
            )
        edges = _typed_sequence(
            self.edges,
            "closure edges",
            DependencyEdge,
            max_items=MAX_EDGES,
        )
        package_map: dict[str, PackageRecord] = {}
        for project in project_values:
            for package in project.packages:
                package_id = package.key.value
                if package_id in package_map:
                    raise SelectionError(
                        "closure projects must not duplicate package identities"
                    )
                package_map[package_id] = package
        package_ids = set(package_map)
        edge_keys: set[tuple[str, str]] = set()
        for edge in edges:
            key = (edge.dependent.value, edge.dependency.value)
            if key in edge_keys:
                raise SelectionError("closure edges must not duplicate endpoints")
            edge_keys.add(key)
            if (
                edge.dependent.value not in package_ids
                or edge.dependency.value not in package_ids
            ):
                raise SelectionError("closure edges must use frozen package identities")
            if {
                edge.dependent_project_id,
                edge.dependency_project_id,
            } - set(selected):
                raise SelectionError("closure edges must use selected projects")
        ordered_edges = tuple(sorted(edges, key=_edge_sort_key))
        expected_selected_edges = tuple(
            edge
            for edge in source_edges
            if edge.dependent_project_id in selected
            and edge.dependency_project_id in selected
        )
        if ordered_edges != expected_selected_edges:
            raise SelectionError(
                "closure edges must exactly match the frozen source graph"
            )

        project_edge_values = _bounded_sequence(
            self.project_edges,
            "closure project edges",
            max_items=MAX_EDGES,
        )
        project_edge_set: set[tuple[str, str]] = set()
        for item in project_edge_values:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise SelectionError("closure project edges must contain pairs")
            dependent, dependency = item
            if not isinstance(dependent, str) or not isinstance(dependency, str):
                raise SelectionError("closure project edge endpoints must be strings")
            pair = (
                canonical_repository_id(dependent),
                canonical_repository_id(dependency),
            )
            if pair in project_edge_set:
                raise SelectionError("closure project edges must not duplicate")
            if pair[0] == pair[1]:
                raise SelectionError(
                    "closure project edges must not contain self edges"
                )
            if set(pair) - set(selected):
                raise SelectionError("closure project edges must use selected projects")
            project_edge_set.add(pair)
        expected_project_edges = {
            (edge.dependent_project_id, edge.dependency_project_id)
            for edge in ordered_edges
            if edge.dependent_project_id != edge.dependency_project_id
        }
        if project_edge_set != expected_project_edges:
            raise SelectionError(
                "closure project edges must match selected package edges"
            )
        ordered_project_edges = tuple(sorted(project_edge_set))

        ordered_source_edges = source_project_edges
        expected_selected_source_edges = {
            pair for pair in ordered_source_edges if set(pair).issubset(set(selected))
        }
        if expected_selected_source_edges != project_edge_set:
            raise SelectionError(
                "closure project edges must match selected source graph edges"
            )
        expected_included, expected_reasons, expected_via = _selection_evidence(
            self.policy, known, ordered_source_edges
        )
        if selected != tuple(sorted(expected_included)):
            raise SelectionError(
                "selected project IDs must exactly match policy closure evidence"
            )

        group_values = _bounded_sequence(
            self.parallel_groups,
            "closure parallel groups",
            max_items=MAX_PROJECTS,
        )
        normalized_groups: list[tuple[str, ...]] = []
        grouped: list[str] = []
        for group in group_values:
            members = _canonical_refs(
                group,
                "closure parallel group",
            )
            if not members:
                raise SelectionError("closure parallel groups must not contain empties")
            if len(members) != len(set(members)):
                raise SelectionError("closure parallel group members must be unique")
            if members != tuple(sorted(members)):
                raise SelectionError(
                    "closure parallel group members must be canonical and ordered"
                )
            normalized_groups.append(members)
            grouped.extend(members)
        if set(grouped) != set(selected) or len(grouped) != len(set(grouped)):
            raise SelectionError(
                "closure parallel groups must contain every selected project once"
            )
        expected_groups, cycle_diagnostics = _topological_groups(
            selected, ordered_project_edges
        )
        if cycle_diagnostics:
            raise GraphValidationError(cycle_diagnostics)
        if tuple(normalized_groups) != expected_groups:
            raise SelectionError(
                "closure parallel groups must match deterministic dependency order"
            )

        explanation_values = _typed_sequence(
            self.explanations,
            "closure explanations",
            SelectionExplanation,
            max_items=MAX_PROJECTS,
        )
        explanation_map = {item.project_id: item for item in explanation_values}
        if (
            len(explanation_map) != len(explanation_values)
            or tuple(sorted(explanation_map)) != known
        ):
            raise SelectionError(
                "closure explanations must cover every known project exactly once"
            )
        if any(
            set(explanation.via_projects) - set(known)
            for explanation in explanation_values
        ):
            raise SelectionError("closure explanation witnesses must be known projects")
        for project_id in known:
            explanation = explanation_map[project_id]
            expected_included_value = project_id in expected_included
            expected_reasons_value = (
                tuple(
                    sorted(
                        expected_reasons[project_id],
                        key=lambda item: item.value,
                    )
                )
                if expected_included_value
                else (SelectionReason.EXCLUDED,)
            )
            expected_via_value = (
                tuple(sorted(expected_via[project_id]))
                if expected_included_value
                else ()
            )
            if (
                explanation.included != expected_included_value
                or explanation.reasons != expected_reasons_value
                or explanation.via_projects != expected_via_value
            ):
                raise SelectionError(
                    "closure explanations do not match policy and source graph evidence"
                )
        object.__setattr__(self, "known_project_ids", known)
        object.__setattr__(self, "selected_project_ids", selected)
        object.__setattr__(
            self, "projects", tuple(project_map[item] for item in selected)
        )
        object.__setattr__(self, "edges", ordered_edges)
        object.__setattr__(self, "project_edges", ordered_project_edges)
        object.__setattr__(self, "parallel_groups", tuple(normalized_groups))
        object.__setattr__(
            self, "explanations", tuple(explanation_map[item] for item in known)
        )
        if not isinstance(self.digest, str):
            raise SelectionError("closure digest must be a string")
        if self.digest:
            digest = _bounded_text(self.digest, "closure digest", max_length=64)
            if len(digest) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in digest
            ):
                raise SelectionError("closure digest must be a SHA-256 digest")
            object.__setattr__(self, "digest", digest.lower())
            if self.digest != _closure_digest(self):
                raise SelectionError("closure digest does not match frozen contents")
        else:
            object.__setattr__(self, "digest", _closure_digest(self))

    @property
    def included_project_ids(self) -> tuple[str, ...]:
        """Alias for the selected canonical project set."""

        return self.selected_project_ids

    @property
    def groups(self) -> tuple[tuple[str, ...], ...]:
        """Alias for deterministic dependency-first parallel groups."""

        return self.parallel_groups

    @property
    def explain_records(self) -> tuple[SelectionExplanation, ...]:
        """Alias for the complete known-project explanation sequence."""

        return self.explanations

    @property
    def all_project_edges(self) -> tuple[tuple[str, str], ...]:
        """Return the frozen complete project-edge evidence."""

        return self.source_graph.project_edges

    @property
    def source_project_edges(self) -> tuple[tuple[str, str], ...]:
        """Return complete source project edges derived from the frozen graph."""

        return self.source_graph.project_edges

    @property
    def source_graph_digest(self) -> str:
        """Return the verified authoritative source graph digest."""

        return self.source_graph.digest

    def canonical_payload(self, *, include_digest: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "policy": self.policy.canonical_payload(),
            "known_project_ids": self.known_project_ids,
            "selected_project_ids": self.selected_project_ids,
            "projects": [_project_payload(project) for project in self.projects],
            "edges": [_edge_payload(edge) for edge in self.edges],
            "project_edges": self.project_edges,
            "parallel_groups": self.parallel_groups,
            "explanations": [
                explanation.canonical_payload() for explanation in self.explanations
            ],
            "source_graph": {
                **self.source_graph.canonical_payload(),
                "digest": self.source_graph.digest,
            },
        }
        if include_digest:
            payload["digest"] = self.digest
        return payload


def derive_phase_view(
    closure: SelectedChangeClosure,
) -> tuple[DerivedPhase, ...]:
    """Project closure groups into deterministic read-only phase records."""

    if not isinstance(closure, SelectedChangeClosure):
        raise SelectionError("phase view requires a SelectedChangeClosure")
    return tuple(
        DerivedPhase(phase=index + 1, project_ids=group)
        for index, group in enumerate(closure.parallel_groups)
    )


def derive_selected_closure(
    graph: DependencyGraph, policy: SelectionPolicy
) -> SelectedChangeClosure:
    """Derive a deterministic selected subgraph from a frozen dependency graph."""

    if not isinstance(policy, SelectionPolicy):
        raise SelectionError("closure policy must be a SelectionPolicy")
    known, project_map, _, edges, project_edges = _graph_inventory(graph)
    if not known:
        raise SelectionError("cannot derive a closure from an empty graph")
    unknown = sorted(
        (set(policy.changed_projects) | set(policy.selected_projects)) - set(known)
    )
    if unknown:
        raise GraphValidationError(
            tuple(
                Diagnostic(
                    GraphDiagnosticCode.MISSING_PROJECT,
                    project_id,
                    "selection policy names an unknown canonical project",
                )
                for project_id in unknown
            )
        )
    included, reasons, via_projects = _selection_evidence(policy, known, project_edges)

    explanations = tuple(
        SelectionExplanation(
            project_id=project_id,
            included=project_id in included,
            reasons=(
                tuple(sorted(reasons[project_id], key=lambda item: item.value))
                if project_id in included
                else (SelectionReason.EXCLUDED,)
            ),
            via_projects=tuple(sorted(via_projects[project_id])),
        )
        for project_id in known
    )
    selected = tuple(sorted(included))
    selected_projects = tuple(project_map[item] for item in selected)
    selected_edges = tuple(
        edge
        for edge in edges
        if edge.dependent_project_id in included
        and edge.dependency_project_id in included
    )
    selected_project_edges = tuple(
        sorted(
            {
                (edge.dependent_project_id, edge.dependency_project_id)
                for edge in selected_edges
                if edge.dependent_project_id != edge.dependency_project_id
            }
        )
    )
    groups, selected_cycle_diagnostics = _topological_groups(
        selected, selected_project_edges
    )
    if selected_cycle_diagnostics:
        raise GraphValidationError(selected_cycle_diagnostics)
    return SelectedChangeClosure(
        policy=policy,
        known_project_ids=known,
        selected_project_ids=selected,
        projects=selected_projects,
        edges=selected_edges,
        project_edges=selected_project_edges,
        parallel_groups=groups,
        explanations=explanations,
        source_graph=graph,
    )


class ShadowDiagnosticCode(StrEnum):
    """Stable read-only differences reported by the phase shadow comparator."""

    INVALID_REFERENCE = "invalid_reference"
    MISSING_PROJECT = "missing_project"
    AMBIGUOUS_PROJECT = "ambiguous_project"
    DUPLICATE_PROJECT = "duplicate_project"
    PHASE_COUNT_MISMATCH = "phase_count_mismatch"
    PHASE_MEMBERSHIP_MISMATCH = "phase_membership_mismatch"
    PHASE_ORDER_MISMATCH = "phase_order_mismatch"
    PHASE_BULK_FLAG_MISMATCH = "phase_bulk_flag_mismatch"
    PHASE_WAIT_MISMATCH = "phase_wait_mismatch"
    PHASE_TRAILING_DERIVED = "phase_trailing_derived"
    PHASE_TRAILING_MANUAL = "phase_trailing_manual"
    DIAGNOSTICS_TRUNCATED = "diagnostics_truncated"


PhaseDiagnosticCode = ShadowDiagnosticCode


@dataclass(frozen=True, slots=True)
class DerivedPhase:
    """A deterministic phase projection with no execution semantics."""

    phase: int
    project_ids: tuple[str, ...]
    bulk_bump: bool = False
    bulk_push: bool = False
    wait_minutes: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.phase, bool)
            or not isinstance(self.phase, int)
            or self.phase < 1
        ):
            raise SelectionError("derived phase number must be positive")
        object.__setattr__(
            self, "project_ids", _canonical_refs(self.project_ids, "phase project IDs")
        )
        if not isinstance(self.bulk_bump, bool) or not isinstance(self.bulk_push, bool):
            raise SelectionError("phase bulk flags must be booleans")
        if (
            isinstance(self.wait_minutes, bool)
            or not isinstance(self.wait_minutes, int)
            or self.wait_minutes < 0
        ):
            raise SelectionError("phase wait_minutes must be non-negative")

    def canonical_payload(self) -> dict[str, object]:
        return {
            "phase": self.phase,
            "project_ids": self.project_ids,
            "bulk_bump": self.bulk_bump,
            "bulk_push": self.bulk_push,
            "wait_minutes": self.wait_minutes,
        }


@dataclass(frozen=True, slots=True)
class ShadowDiagnostic:
    """One bounded deterministic legacy phase difference."""

    code: ShadowDiagnosticCode
    subject: str
    message: str
    details: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.code, ShadowDiagnosticCode):
            raise SelectionError("shadow diagnostic code must be supported")
        _bounded_text(self.subject, "shadow diagnostic subject")
        _bounded_text(self.message, "shadow diagnostic message")
        details = _bounded_sequence(
            self.details,
            "shadow diagnostic details",
            max_items=MAX_PLAN_STAGES,
        )
        normalized: list[tuple[str, str]] = []
        for item in details:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise SelectionError("shadow diagnostic details must contain pairs")
            key, value = item
            normalized.append(
                (
                    _bounded_text(key, "shadow diagnostic detail key", max_length=128),
                    _bounded_text(value, "shadow diagnostic detail value"),
                )
            )
        object.__setattr__(self, "subject", self.subject)
        object.__setattr__(self, "message", self.message)
        object.__setattr__(self, "details", tuple(sorted(normalized)))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "code": self.code.value,
            "subject": self.subject,
            "message": self.message,
            "details": self.details,
        }


class _DiagnosticAccumulator:
    """Keep shadow diagnostics bounded while preserving mandatory phase evidence."""

    def __init__(self) -> None:
        self._mandatory: list[ShadowDiagnostic] = []
        self._ordinary: list[ShadowDiagnostic] = []
        self._omitted = 0
        self._omitted_fingerprint = 0

    def _record_omitted(self, diagnostic: ShadowDiagnostic) -> None:
        fingerprint = hashlib.sha256(
            _canonical_json(diagnostic.canonical_payload()).encode("utf-8")
        ).digest()
        self._omitted_fingerprint ^= int.from_bytes(fingerprint, "big")

    def add(self, diagnostic: ShadowDiagnostic, *, mandatory: bool = False) -> None:
        target = self._mandatory if mandatory else self._ordinary
        if len(target) < MAX_SHADOW_DIAGNOSTICS:
            target.append(diagnostic)
        else:
            self._omitted += 1
            self._record_omitted(diagnostic)

    def finalize(self) -> tuple[ShadowDiagnostic, ...]:
        mandatory = sorted(self._mandatory, key=_shadow_diagnostic_sort_key)
        ordinary = sorted(self._ordinary, key=_shadow_diagnostic_sort_key)
        if len(mandatory) > MAX_SHADOW_DIAGNOSTICS:
            mandatory = mandatory[:MAX_SHADOW_DIAGNOSTICS]
            omitted = len(self._mandatory) - len(mandatory) + self._omitted
        else:
            omitted = self._omitted
        available = MAX_SHADOW_DIAGNOSTICS - len(mandatory)
        if len(ordinary) > available or self._omitted:
            keep_count = max(0, available - 1) if available else 0
            omitted += len(ordinary) - keep_count
            for diagnostic in ordinary[keep_count:]:
                self._record_omitted(diagnostic)
            if available:
                ordinary = ordinary[:keep_count]
                ordinary.append(
                    ShadowDiagnostic(
                        ShadowDiagnosticCode.DIAGNOSTICS_TRUNCATED,
                        "shadow report",
                        "additional shadow diagnostics were omitted at the bounded limit",
                        (
                            ("omitted", str(omitted)),
                            (
                                "digest",
                                f"{self._omitted_fingerprint:064x}",
                            ),
                        ),
                    )
                )
            else:
                ordinary = []
        result = tuple(sorted((*mandatory, *ordinary), key=_shadow_diagnostic_sort_key))
        if len(result) > MAX_SHADOW_DIAGNOSTICS:
            return result[:MAX_SHADOW_DIAGNOSTICS]
        return result


def _shadow_diagnostic_sort_key(
    item: ShadowDiagnostic,
) -> tuple[str, str, str, tuple[tuple[str, str], ...]]:
    return (item.code.value, item.subject, item.message, item.details)


def _phase_sort_key(
    phase: LegacyPhase, references: tuple[str, ...]
) -> tuple[object, ...]:
    return (
        phase.phase,
        phase.name,
        references,
        phase.bulk_bump,
        phase.bulk_push,
        phase.wait_minutes,
    )


def _shadow_report_digest(report: PhaseShadowReport) -> str:
    return hashlib.sha256(
        json.dumps(
            report.canonical_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PhaseShadowReport:
    """Read-only derived/manual phase comparison and its stable digest."""

    derived_phases: tuple[DerivedPhase, ...]
    manual_phases: tuple[DerivedPhase, ...]
    diagnostics: tuple[ShadowDiagnostic, ...]
    exact_equal: bool
    report_digest: str = ""

    def __post_init__(self) -> None:
        derived = _typed_sequence(
            self.derived_phases,
            "derived phases",
            DerivedPhase,
            max_items=MAX_PLAN_STAGES,
        )
        manual = _typed_sequence(
            self.manual_phases,
            "manual phases",
            DerivedPhase,
            max_items=MAX_PLAN_STAGES,
        )
        diagnostics = _typed_sequence(
            self.diagnostics,
            "shadow diagnostics",
            ShadowDiagnostic,
            max_items=MAX_SHADOW_DIAGNOSTICS,
        )
        if not isinstance(self.exact_equal, bool):
            raise SelectionError("shadow exact equality must be a boolean")
        ordered_diagnostics = tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.code.value,
                    item.subject,
                    item.message,
                    item.details,
                ),
            )
        )
        object.__setattr__(self, "derived_phases", derived)
        object.__setattr__(self, "manual_phases", manual)
        object.__setattr__(self, "diagnostics", ordered_diagnostics)
        if self.exact_equal != (not ordered_diagnostics and derived == manual):
            raise SelectionError("shadow exact equality does not match report contents")
        if not isinstance(self.report_digest, str):
            raise SelectionError("shadow report digest must be a string")
        if self.report_digest:
            digest = _bounded_text(
                self.report_digest, "shadow report digest", max_length=64
            )
            if len(digest) != 64 or any(
                char not in "0123456789abcdefABCDEF" for char in digest
            ):
                raise SelectionError("shadow report digest must be a SHA-256 digest")
            object.__setattr__(self, "report_digest", digest.lower())
            if self.report_digest != _shadow_report_digest(self):
                raise SelectionError("shadow report digest does not match contents")
        else:
            object.__setattr__(self, "report_digest", _shadow_report_digest(self))

    def canonical_payload(self, *, include_digest: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "derived_phases": [
                phase.canonical_payload() for phase in self.derived_phases
            ],
            "manual_phases": [
                phase.canonical_payload() for phase in self.manual_phases
            ],
            "diagnostics": [
                diagnostic.canonical_payload() for diagnostic in self.diagnostics
            ],
            "exact_equal": self.exact_equal,
        }
        if include_digest:
            payload["report_digest"] = self.report_digest
        return payload


def _manual_reference(
    raw: str,
    known: tuple[str, ...],
) -> tuple[str | None, ShadowDiagnostic | None]:
    value = _bounded_text(raw, "legacy project reference")
    known_set = set(known)
    if value.startswith("repo:") or "/" in value:
        try:
            canonical = canonical_repository_id(value)
        except WorkspaceReleaseError:
            return (
                None,
                ShadowDiagnostic(
                    ShadowDiagnosticCode.INVALID_REFERENCE,
                    value,
                    "legacy project reference is not a canonical repository identity",
                ),
            )
        if canonical not in known_set:
            return (
                None,
                ShadowDiagnostic(
                    ShadowDiagnosticCode.MISSING_PROJECT,
                    value,
                    "legacy project reference is not a known canonical project",
                ),
            )
        return canonical, None
    if "\\" in value or ":" in value or value in {".", ".."}:
        return (
            None,
            ShadowDiagnostic(
                ShadowDiagnosticCode.INVALID_REFERENCE,
                value,
                "legacy bare project reference is not a valid basename",
            ),
        )
    candidates = tuple(
        project_id
        for project_id in known
        if PurePosixPath(project_id[5:]).name == value
    )
    if not candidates:
        return (
            None,
            ShadowDiagnostic(
                ShadowDiagnosticCode.MISSING_PROJECT,
                value,
                "legacy project basename has no known owner",
            ),
        )
    if len(candidates) > 1:
        return (
            None,
            ShadowDiagnostic(
                ShadowDiagnosticCode.AMBIGUOUS_PROJECT,
                value,
                "legacy project basename has multiple canonical owners",
                tuple(
                    (f"owner_{index}", candidate)
                    for index, candidate in enumerate(candidates)
                ),
            ),
        )
    return candidates[0], None


def _phase_members_display(project_ids: tuple[str, ...]) -> str:
    """Return bounded count/digest/prefix evidence for phase membership."""

    digest = hashlib.sha256(_canonical_json(project_ids).encode("utf-8")).hexdigest()
    prefix_parts: list[str] = []
    prefix_length = 0
    for project_id in project_ids:
        separator_length = 1 if prefix_parts else 0
        if prefix_length + separator_length + len(project_id) > 512:
            break
        prefix_parts.append(project_id)
        prefix_length += separator_length + len(project_id)
    prefix = ",".join(prefix_parts) or "<none>"
    if len(prefix_parts) < len(project_ids):
        prefix += ",..."
    return f"count={len(project_ids)};digest={digest};prefix={prefix}"


def _validate_legacy_phase(phase: LegacyPhase, index: int) -> tuple[str, ...]:
    if not isinstance(phase, LegacyPhase):
        raise SelectionError(f"legacy phase {index} must be a LegacyPhase")
    _bounded_text(phase.name, f"legacy phase {index} name")
    if (
        isinstance(phase.phase, bool)
        or not isinstance(phase.phase, int)
        or phase.phase < 1
    ):
        raise SelectionError(f"legacy phase {index} number must be positive")
    if not isinstance(phase.bulk_bump, bool) or not isinstance(phase.bulk_push, bool):
        raise SelectionError(f"legacy phase {index} bulk flags must be booleans")
    if (
        isinstance(phase.wait_minutes, bool)
        or not isinstance(phase.wait_minutes, int)
        or phase.wait_minutes < 0
    ):
        raise SelectionError(f"legacy phase {index} wait_minutes must be non-negative")
    references = _bounded_sequence(
        phase.project_references,
        f"legacy phase {index} project references",
        max_items=MAX_PROJECTS,
    )
    if any(not isinstance(reference, str) for reference in references):
        raise SelectionError(f"legacy phase {index} project references must be strings")
    return tuple(
        _bounded_text(reference, f"legacy phase {index} project reference")
        for reference in references
    )


def compare_legacy_phases(
    closure: SelectedChangeClosure,
    manifest: LegacyPhaseManifest,
) -> PhaseShadowReport:
    """Compare deterministic graph phases to a copied legacy phase manifest."""

    if not isinstance(closure, SelectedChangeClosure):
        raise SelectionError("phase comparison requires a SelectedChangeClosure")
    if not isinstance(manifest, LegacyPhaseManifest):
        raise SelectionError("phase comparison requires a LegacyPhaseManifest")
    phase_values = _typed_sequence(
        manifest.phases,
        "legacy manifest phases",
        LegacyPhase,
        max_items=MAX_PLAN_STAGES,
    )
    normalized_phases: list[tuple[LegacyPhase, tuple[str, ...]]] = []
    total_project_references = 0
    for index, phase in enumerate(phase_values):
        references = _validate_legacy_phase(phase, index)
        total_project_references += len(references)
        if total_project_references > MAX_EDGES:
            raise SelectionError(
                "legacy phase project references exceed the bounded total"
            )
        normalized_phases.append((phase, references))
    known = closure.known_project_ids
    derived = derive_phase_view(closure)
    diagnostics = _DiagnosticAccumulator()
    manual: list[DerivedPhase] = []
    seen_projects: dict[str, str] = {}
    for legacy, references in sorted(
        normalized_phases, key=lambda item: _phase_sort_key(*item)
    ):
        resolved: list[str] = []
        for raw in references:
            reference, diagnostic = _manual_reference(raw, known)
            if diagnostic is not None:
                diagnostics.add(diagnostic)
                continue
            assert reference is not None
            if reference in seen_projects:
                diagnostics.add(
                    ShadowDiagnostic(
                        ShadowDiagnosticCode.DUPLICATE_PROJECT,
                        reference,
                        "legacy phase manifest assigns a project more than once",
                        (
                            ("first_phase", seen_projects[reference]),
                            ("phase", legacy.name),
                        ),
                    )
                )
            else:
                seen_projects[reference] = legacy.name
            resolved.append(reference)
        manual.append(
            DerivedPhase(
                phase=legacy.phase,
                project_ids=tuple(resolved),
                bulk_bump=legacy.bulk_bump,
                bulk_push=legacy.bulk_push,
                wait_minutes=legacy.wait_minutes,
            )
        )
    manual_phases = tuple(manual)
    if len(derived) != len(manual_phases):
        diagnostics.add(
            ShadowDiagnostic(
                ShadowDiagnosticCode.PHASE_COUNT_MISMATCH,
                "workspace phases",
                "derived and legacy phase counts differ",
                (("derived", str(len(derived))), ("manual", str(len(manual_phases)))),
            ),
            mandatory=True,
        )
    common_count = min(len(derived), len(manual_phases))
    for index, (new_phase, old_phase) in enumerate(
        zip(derived[:common_count], manual_phases[:common_count], strict=True)
    ):
        subject = f"phase-{index + 1}"
        if new_phase.phase != old_phase.phase:
            diagnostics.add(
                ShadowDiagnostic(
                    ShadowDiagnosticCode.PHASE_ORDER_MISMATCH,
                    subject,
                    "derived and legacy phase numbers differ",
                    (
                        ("derived", str(new_phase.phase)),
                        ("manual", str(old_phase.phase)),
                    ),
                )
            )
        if new_phase.project_ids != old_phase.project_ids:
            if set(new_phase.project_ids) != set(old_phase.project_ids):
                code = ShadowDiagnosticCode.PHASE_MEMBERSHIP_MISMATCH
                message = "derived and legacy phase membership differs"
            else:
                code = ShadowDiagnosticCode.PHASE_ORDER_MISMATCH
                message = "derived and legacy project order differs"
            diagnostics.add(
                ShadowDiagnostic(
                    code,
                    subject,
                    message,
                    (
                        ("derived", _phase_members_display(new_phase.project_ids)),
                        ("manual", _phase_members_display(old_phase.project_ids)),
                    ),
                )
            )
        if (
            new_phase.bulk_bump != old_phase.bulk_bump
            or new_phase.bulk_push != old_phase.bulk_push
        ):
            diagnostics.add(
                ShadowDiagnostic(
                    ShadowDiagnosticCode.PHASE_BULK_FLAG_MISMATCH,
                    subject,
                    "derived and legacy phase bulk flags differ",
                    (
                        (
                            "derived",
                            f"bump={new_phase.bulk_bump},push={new_phase.bulk_push}",
                        ),
                        (
                            "manual",
                            f"bump={old_phase.bulk_bump},push={old_phase.bulk_push}",
                        ),
                    ),
                )
            )
        if new_phase.wait_minutes != old_phase.wait_minutes:
            diagnostics.add(
                ShadowDiagnostic(
                    ShadowDiagnosticCode.PHASE_WAIT_MISMATCH,
                    subject,
                    "derived and legacy phase wait times differ",
                    (
                        ("derived", str(new_phase.wait_minutes)),
                        ("manual", str(old_phase.wait_minutes)),
                    ),
                )
            )
    for index in range(common_count, len(derived)):
        trailing_phase = derived[index]
        diagnostics.add(
            ShadowDiagnostic(
                ShadowDiagnosticCode.PHASE_TRAILING_DERIVED,
                f"phase-{index + 1}",
                "derived phase has no matching legacy phase",
                (
                    ("phase", str(trailing_phase.phase)),
                    (
                        "membership",
                        _phase_members_display(trailing_phase.project_ids),
                    ),
                ),
            ),
            mandatory=True,
        )
    for index in range(common_count, len(manual_phases)):
        trailing_phase = manual_phases[index]
        diagnostics.add(
            ShadowDiagnostic(
                ShadowDiagnosticCode.PHASE_TRAILING_MANUAL,
                f"phase-{index + 1}",
                "legacy phase has no matching derived phase",
                (
                    ("phase", str(trailing_phase.phase)),
                    (
                        "membership",
                        _phase_members_display(trailing_phase.project_ids),
                    ),
                ),
            ),
            mandatory=True,
        )
    finalized_diagnostics = diagnostics.finalize()
    exact_equal = not finalized_diagnostics and derived == manual_phases
    return PhaseShadowReport(
        derived_phases=derived,
        manual_phases=manual_phases,
        diagnostics=finalized_diagnostics,
        exact_equal=exact_equal,
    )


# Short aliases keep the pure boundary easy to discover without adding public
# command or MCP wiring.
derive_closure = derive_selected_closure
derive_selection = derive_selected_closure
compare_phase_manifest = compare_legacy_phases
shadow_compare = compare_legacy_phases
derived_phases = derive_phase_view


__all__ = [
    "DerivedPhase",
    "InclusionMode",
    "MAX_SHADOW_DIAGNOSTICS",
    "PhaseDiagnosticCode",
    "PhaseShadowReport",
    "ProjectInclusionMode",
    "SelectedChangeClosure",
    "SelectionError",
    "SelectionExplanation",
    "SelectionMode",
    "SelectionPolicy",
    "SelectionReason",
    "ShadowDiagnostic",
    "ShadowDiagnosticCode",
    "compare_legacy_phases",
    "compare_phase_manifest",
    "derive_phase_view",
    "derive_closure",
    "derive_selected_closure",
    "derive_selection",
    "derived_phases",
    "shadow_compare",
]
