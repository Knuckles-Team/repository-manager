"""A caller selector never supplies source authority or bypasses release proof."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from repository_manager.development import fleet_evidence_provider as provider_module
from repository_manager.development.fleet_evidence_provider import (
    ReadOnlyFleetEvidenceProvider,
)
from repository_manager.development.fleet_release_gate import (
    FleetReleaseGateError,
    _reference_digest,
    parse_fleet_evidence_reference,
)
from repository_manager.development.workspace_release import FleetCycleError, FleetSCC
from repository_manager.workspace_manifest import synchronize_workspace_manifest
from tests.test_workspace_release_plan import _plan


def _reference(plan_digest: str, **digest_overrides):
    payload = {
        "schema": "fleet_evidence_reference/v1",
        "plan_digest": plan_digest,
        "order_digest": "b" * 64,
        "census_digest": "c" * 64,
        "universe_digest": "d" * 64,
        "stage_id": "stage:fixture",
        "kind": "setup",
        "project_id": "repo:fixture",
    }
    payload.update(digest_overrides)
    return parse_fleet_evidence_reference(
        json.dumps({**payload, "digest": _reference_digest(payload)}).encode()
    )


def _provider(tmp_path, supplier):
    return ReadOnlyFleetEvidenceProvider(
        tmp_path,
        tmp_path / "runtime.yml",
        tmp_path / "seed.yml",
        "a" * 40,
        supplier,
    )


def test_selector_cannot_choose_plan_or_workspace(tmp_path):
    plan = _plan()
    calls = []

    def supplier():
        calls.append("trusted")
        return plan

    provider = _provider(tmp_path, supplier)
    with pytest.raises(FleetReleaseGateError, match="selector differs"):
        provider(_reference("0" * 64))
    assert calls == ["trusted"]
    assert list(tmp_path.iterdir()) == []


def test_current_manifest_without_verified_metadata_refuses(tmp_path):
    plan = _plan()
    (tmp_path / "workspace.yml").write_text(
        f'path: "{tmp_path}"\nrepositories:\n  - url: https://example.test/app.git\n'
    )
    synchronize_workspace_manifest(
        tmp_path / "workspace.yml",
        runtime_destination=tmp_path / "runtime.yml",
        seed_destination=tmp_path / "seed.yml",
    )
    provider = _provider(tmp_path, lambda: plan)
    with pytest.raises(FleetReleaseGateError, match="trusted fleet evidence"):
        provider(_reference(plan.digest))
    assert (tmp_path / "runtime.yml").exists()
    assert (tmp_path / "seed.yml").exists()


def test_provider_refuses_untrusted_inputs_before_read(tmp_path):
    plan = _plan()
    provider = _provider(tmp_path, lambda: plan)
    with pytest.raises(FleetReleaseGateError, match="reference is required"):
        provider(None)
    with pytest.raises(FleetReleaseGateError, match="revision is unavailable"):
        replace(provider, source_commit="a" * 39)(_reference(plan.digest))
    with pytest.raises(FleetReleaseGateError, match="paths must be absolute"):
        replace(provider, runtime=tmp_path.relative_to(tmp_path))(
            _reference(plan.digest)
        )
    with pytest.raises(FleetReleaseGateError, match="plan supplier is unavailable"):
        replace(provider, plan_supplier=None)(_reference(plan.digest))
    assert list(tmp_path.iterdir()) == []


def test_malformed_trusted_plan_refuses_before_source_scan(tmp_path):
    provider = _provider(tmp_path, lambda: object())
    with pytest.raises(FleetReleaseGateError, match="trusted fleet evidence"):
        provider(_reference("a" * 64))
    assert list(tmp_path.iterdir()) == []


def _trusted_fixture(monkeypatch, *, census_digest="c" * 64):
    closure = SimpleNamespace(manifest_sha256="a" * 64)
    census = SimpleNamespace(digest=census_digest)
    universe = SimpleNamespace(digest="d" * 64)
    order = SimpleNamespace(digest="b" * 64)
    monkeypatch.setattr(
        provider_module,
        "validate_fleet_manifest_mirrors",
        lambda *_args, **_kwargs: SimpleNamespace(
            source_digest=closure.manifest_sha256
        ),
    )
    monkeypatch.setattr(
        provider_module,
        "capture_complete_manifest_census",
        lambda *_args, **_kwargs: (closure, census, universe),
    )
    monkeypatch.setattr(
        provider_module, "reconcile_fleet_census", lambda *_args, **_kwargs: order
    )
    return closure, census, universe, order


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("order_digest", "0" * 64),
        ("census_digest", "0" * 64),
        ("universe_digest", "0" * 64),
    ],
)
def test_trusted_provider_rejects_rehashed_reference_mismatch(
    tmp_path, monkeypatch, field, value
):
    plan = _plan()
    _trusted_fixture(monkeypatch)
    monkeypatch.setattr(
        provider_module,
        "prepare_fleet_release",
        lambda *_args, **_kwargs: pytest.fail("released"),
    )
    with pytest.raises(FleetReleaseGateError, match="selector differs from sources"):
        _provider(tmp_path, lambda: plan)(_reference(plan.digest, **{field: value}))


def test_trusted_provider_rejects_missing_census_digest(tmp_path, monkeypatch):
    plan = _plan()
    _trusted_fixture(monkeypatch, census_digest="")
    monkeypatch.setattr(
        provider_module,
        "prepare_fleet_release",
        lambda *_args, **_kwargs: pytest.fail("released"),
    )
    with pytest.raises(FleetReleaseGateError, match="selector differs from sources"):
        _provider(tmp_path, lambda: plan)(_reference(plan.digest))


def test_trusted_provider_preserves_scc_refusal_cause(tmp_path, monkeypatch):
    plan = _plan()
    _trusted_fixture(monkeypatch)
    component = FleetSCC(
        ("repo:a", "repo:b"),
        (("repo:a", "repo:b"), ("repo:b", "repo:a")),
        "e" * 64,
    )

    def refuse_scc(*_args, **_kwargs):
        raise FleetCycleError((component,))

    monkeypatch.setattr(provider_module, "reconcile_fleet_census", refuse_scc)
    monkeypatch.setattr(
        provider_module,
        "prepare_fleet_release",
        lambda *_args, **_kwargs: pytest.fail("released"),
    )
    with pytest.raises(FleetReleaseGateError, match="trusted fleet evidence") as caught:
        _provider(tmp_path, lambda: plan)(_reference(plan.digest))
    assert isinstance(caught.value.__cause__, FleetCycleError)
    assert caught.value.__cause__.components == (component,)
