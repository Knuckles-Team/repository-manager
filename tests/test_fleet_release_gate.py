"""The fleet release boundary remains fail closed until full source proof."""

from __future__ import annotations

import json
from dataclasses import replace
from typing import cast

import pytest

import repository_manager.development.fleet_release_gate as fleet_gate_module
from repository_manager.development.fleet_census import capture_census
from repository_manager.development.fleet_manifest import FleetMirrorReceipt
from repository_manager.development.fleet_release_gate import (
    FLEET_RELEASE_ACTIVATION,
    FleetReleaseGateError,
    FleetReleasePreflight,
    _reference_digest,
    authorize_fleet_stage,
    parse_fleet_evidence_reference,
    prepare_fleet_release,
)
from repository_manager.development.workspace_release import (
    FleetEvidenceKind,
    build_fleet_order_receipt,
)
from repository_manager.development.workspace_release_plan import (
    bind_fleet_release_plan,
)
from repository_manager.repository_manager import Git
from tests.test_workspace_release_plan import _plan


def test_incomplete_census_refuses_before_any_live_release_authority(tmp_path):
    assert FLEET_RELEASE_ACTIVATION == "closed_pending_source_universe_proof"
    (tmp_path / "workspace.yml").write_text("repositories: []\n")
    census = capture_census(tmp_path, "workspace.yml", (), source_commit="a" * 40)
    plan = _plan()
    order = build_fleet_order_receipt(
        plan.graph,
        manifest_digest=census.manifest_sha256,
        observations=(),
        coverage_receipts={kind: "b" * 64 for kind in FleetEvidenceKind},
    )
    missing_runtime = tmp_path / "no-runtime-mirror"
    missing_seed = tmp_path / "no-seed-mirror"
    with pytest.raises(FleetReleaseGateError, match="activation is closed"):
        prepare_fleet_release(
            tmp_path,
            runtime=missing_runtime,
            seed=missing_seed,
            plan=plan,
            order=order,
            census=census,
        )
    assert not missing_runtime.exists()
    assert not missing_seed.exists()
    forged = FleetReleasePreflight(
        tmp_path,
        missing_runtime,
        missing_seed,
        plan,
        order,
        census,
        FleetMirrorReceipt(
            census.manifest_sha256, census.manifest_sha256, "c" * 64, (), "d" * 64
        ),
        bind_fleet_release_plan(plan, order),
    )
    stage = plan.stages[0]
    with pytest.raises(FleetReleaseGateError, match="activation is closed"):
        authorize_fleet_stage(
            forged,
            stage_id=stage.stage_id,
            kind=stage.kind,
            project_id=stage.project_id,
        )


def test_open_switch_alone_cannot_supply_source_universe_proof(tmp_path, monkeypatch):
    monkeypatch.setattr(fleet_gate_module, "FLEET_RELEASE_ACTIVATION", "open")
    (tmp_path / "workspace.yml").write_text("repositories: []\n")
    census = capture_census(tmp_path, "workspace.yml", (), source_commit="a" * 40)
    plan = _plan()
    order = build_fleet_order_receipt(
        plan.graph,
        manifest_digest=census.manifest_sha256,
        observations=(),
        coverage_receipts={kind: "b" * 64 for kind in FleetEvidenceKind},
    )
    with pytest.raises(FleetReleaseGateError, match="source universe is not complete"):
        prepare_fleet_release(
            tmp_path,
            runtime=tmp_path / "runtime.yml",
            seed=tmp_path / "seed.yml",
            plan=plan,
            order=order,
            census=census,
        )
    with pytest.raises(
        FleetReleaseGateError, match="independent fleet source universe proof"
    ):
        prepare_fleet_release(
            tmp_path,
            runtime=tmp_path / "runtime.yml",
            seed=tmp_path / "seed.yml",
            plan=plan,
            order=order,
            census=replace(census, complete=True),
        )


@pytest.mark.parametrize(
    "invoke",
    [
        lambda git: git.setup_from_yaml("missing.yml"),
        lambda git: git.install_projects(),
        lambda git: git.install_project("missing"),
        lambda git: git.build_projects(),
        lambda git: git.push_projects([]),
        lambda git: git.push_project("missing"),
        lambda git: git.bump_version("patch", path="missing"),
        lambda git: git.bulk_bump("patch"),
        lambda git: git.update_dependency("missing", "a", "1.2.3"),
        lambda git: git.phased_bumpversion(),
        lambda git: git.phased_push(),
        lambda git: git.validate_and_release(auto_push=True),
    ],
)
def test_open_cutover_refuses_legacy_git_routes_before_effect(
    tmp_path, monkeypatch, invoke
):
    monkeypatch.setattr(fleet_gate_module, "FLEET_RELEASE_ACTIVATION", "open")
    git = object.__new__(Git)
    git.path = str(tmp_path)
    with pytest.raises(FleetReleaseGateError, match="verified fleet stage required"):
        invoke(git)
    assert list(tmp_path.iterdir()) == []


def test_closed_cutover_preserves_legacy_route_selection():
    fleet_gate_module.require_legacy_release_route("push")


def test_transport_reference_is_bounded_and_never_self_authorizing():
    payload = {
        "schema": "fleet_evidence_reference/v1",
        "plan_digest": "a" * 64,
        "order_digest": "b" * 64,
        "census_digest": "c" * 64,
        "universe_digest": "d" * 64,
        "stage_id": "stage:fixture",
        "kind": "setup",
        "project_id": "repo:fixture",
    }
    raw = json.dumps({**payload, "digest": _reference_digest(payload)}).encode()
    reference = parse_fleet_evidence_reference(raw)
    assert reference.kind.value == "setup"
    assert parse_fleet_evidence_reference(reference.to_json()) == reference
    with pytest.raises(FleetReleaseGateError, match="invalid"):
        parse_fleet_evidence_reference(raw.replace(b'"setup"', b'"push"'))
    with pytest.raises(FleetReleaseGateError, match="invalid"):
        parse_fleet_evidence_reference(
            raw.replace(b'"stage_id":', b'"stage_id":"x","stage_id":')
        )
    with pytest.raises(FleetReleaseGateError, match="exceeds bound"):
        parse_fleet_evidence_reference(b"x" * 4097)
    with pytest.raises(FleetReleaseGateError, match="preflight is required"):
        fleet_gate_module.authorize_fleet_evidence_reference(
            cast(FleetReleasePreflight, None), reference
        )
