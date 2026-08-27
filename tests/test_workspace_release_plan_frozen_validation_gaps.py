"""Characterization tests for branches of ``_validate_frozen_plan_fields``
(CCN 81, repository_manager/development/workspace_release_plan.py) that
tests/test_workspace_release_plan.py does not reach, ahead of this lane's
pure extract-method decomposition (CXA-FL-REPOSITORYMANAGER-01).

Mutation probing against the UNMODIFIED function found these specific
raise-branches were not killed by the existing suite (see the lane report
for the exact probe commands): the graph-digest-drift check and the
edge-canonical-order check. Both are exercised here using the same
forge-with-object.__new__ pattern already used by
``test_frozen_plan_rejects_digest_and_nested_stage_tampering`` in the
sibling module, reusing its ``_plan`` builder so the fixture stays a
single source of truth.
"""

from __future__ import annotations

import pytest

from repository_manager.development.workspace_release_plan import (
    FrozenReleasePlan,
    ReleasePlanCode,
    ReleasePlanError,
    validate_frozen_release_plan,
)
from tests.test_workspace_release_plan import _plan


def _forge(plan: FrozenReleasePlan, **overrides: object) -> FrozenReleasePlan:
    forged = object.__new__(FrozenReleasePlan)
    for field_name in plan.__dataclass_fields__:
        object.__setattr__(forged, field_name, getattr(plan, field_name))
    for field_name, value in overrides.items():
        object.__setattr__(forged, field_name, value)
    return forged


def test_graph_digest_drift_is_rejected() -> None:
    plan = _plan()
    # A syntactically valid SHA-256 digest that does not match plan.graph's
    # actual digest -- must trip "frozen graph evidence is not bound".
    forged = _forge(plan, graph_digest="1" * 64)
    with pytest.raises(ReleasePlanError) as captured:
        validate_frozen_release_plan(forged)
    assert captured.value.code is ReleasePlanCode.GRAPH_DRIFT


def test_edges_out_of_canonical_order_are_rejected() -> None:
    plan = _plan()
    assert len(plan.edges) > 1, "fixture must carry more than one edge to reorder"
    forged = _forge(plan, edges=tuple(reversed(plan.edges)))
    with pytest.raises(ReleasePlanError) as captured:
        validate_frozen_release_plan(forged)
    assert captured.value.code is ReleasePlanCode.DIGEST


def test_stage_graph_or_selection_evidence_not_bound_is_rejected() -> None:
    plan = _plan()
    bump = next(
        stage
        for stage in plan.stages
        if stage.kind.value == "bump" and stage.project_id == "repo:packages/d"
    )
    forged_stage = object.__new__(type(bump))
    for field_name in bump.__dataclass_fields__:
        object.__setattr__(forged_stage, field_name, getattr(bump, field_name))
    object.__setattr__(forged_stage, "graph_digest", "2" * 64)
    forged_stages = tuple(
        forged_stage if stage is bump else stage for stage in plan.stages
    )
    forged = _forge(plan, stages=forged_stages)
    with pytest.raises(ReleasePlanError) as captured:
        validate_frozen_release_plan(forged)
    assert captured.value.code is ReleasePlanCode.DIGEST
