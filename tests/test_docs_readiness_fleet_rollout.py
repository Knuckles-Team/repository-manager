"""Focused contract fixtures for the NE-146 rollout authority.

These tests exercise selection, source evidence, and journal idempotence only. They
never clone, branch, generate, push, or touch a fleet checkout.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from repository_manager import docs_readiness_rollout as rollout


def _project(identity: str = "agent-packages/example") -> rollout.ProjectSpec:
    return rollout.ProjectSpec(identity=identity, wave=1)


def _manifest(project: rollout.ProjectSpec | None = None) -> rollout.FleetManifest:
    project = project or _project()
    findings = rollout.SourceFindings((), ())
    provisional = rollout.FleetManifest(
        schema_version=rollout.SCHEMA_VERSION,
        manifest_name="fixture",
        expected_publishable_count=1,
        excluded_identities=("agent-packages/agents/tests",),
        waves=(rollout.WaveSpec(1, "fixture", 1),),
        projects=(project,),
        source_findings=findings,
        digest="0" * 64,
    )
    return provisional


def _dependencies() -> rollout.RolloutDependencies:
    return rollout.RolloutDependencies("a" * 40, "b" * 40)


class _Adapter:
    generator_revision = "a" * 40
    generator_version = "1.0.0"
    generator_schema = "agent-readiness/v1"
    tck_revision = "b" * 40
    tck_schema = "pages-readiness-tck/v1"

    def __init__(self, *, fail_apply: bool = False) -> None:
        self.fail_apply = fail_apply
        self.apply_calls = 0
        self.preview_calls = 0
        self.verify_calls = 0
        self.rollback_calls = 0

    @staticmethod
    def _result() -> dict[str, Any]:
        return {
            "ok": True,
            "generator_version": "1.0.0",
            "schema_version": "agent-readiness/v1",
            "provenance_digest": "c" * 64,
            "generated_outputs": ["agent-readiness-manifest.json", "llms.txt"],
        }

    def preview(self, repository_root: Path, project: rollout.ProjectSpec) -> dict[str, Any]:
        del repository_root, project
        self.preview_calls += 1
        return self._result()

    def apply(self, repository_root: Path, project: rollout.ProjectSpec) -> dict[str, Any]:
        del repository_root, project
        self.apply_calls += 1
        if self.fail_apply:
            raise RuntimeError("fixture failure is not durable evidence")
        return self._result()

    def verify(self, repository_root: Path, project: rollout.ProjectSpec) -> dict[str, Any]:
        del repository_root, project
        self.verify_calls += 1
        return self._result()

    def rollback(
        self,
        repository_root: Path,
        project: rollout.ProjectSpec,
        evidence: dict[str, Any],
    ) -> dict[str, Any]:
        del repository_root, project, evidence
        self.rollback_calls += 1
        return self._result()


def _evidence(project: rollout.ProjectSpec, *, clean: bool = True) -> rollout.RepositoryEvidence:
    return rollout.RepositoryEvidence(
        identity=project.identity,
        head_revision="d" * 40,
        branch="main",
        clean=clean,
        worktree_count=1,
        source_digest="e" * 64,
    )


def test_packaged_manifest_is_exact_75_and_wave_counts() -> None:
    manifest = rollout.load_fleet_manifest()

    assert manifest.expected_publishable_count == 75
    assert len(manifest.projects) == 75
    assert sum(wave.expected_count for wave in manifest.waves) == 75
    assert [plan.count for plan in rollout.plan_waves(manifest)] == [7, 14, 19, 17, 18]
    assert manifest.source_findings.missing_pages_workflows == (
        "agent-packages/agents/ciso-assistant-api",
        "agent-packages/agents/onetrust-api",
    )
    assert manifest.source_findings.missing_site_urls == (
        "agent-packages/agent-utilities",
        "agent-packages/agents/mealie-mcp",
        "agent-packages/agents/microsoft-agent",
        "agent-packages/agents/vector-mcp",
    )


def test_manifest_rejects_cardinality_drift(tmp_path: Path) -> None:
    source = rollout.default_manifest_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    data["expected_publishable_count"] = 74
    candidate = tmp_path / "manifest.json"
    candidate.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(rollout.RolloutError, match="manifest-cardinality-invalid"):
        rollout.load_fleet_manifest(candidate)


def test_dependencies_require_full_immutable_revisions() -> None:
    with pytest.raises(rollout.RolloutError, match="generator-revision-invalid"):
        rollout.RolloutDependencies("main", "b" * 40).validate()
    with pytest.raises(rollout.RolloutError, match="tck-revision-invalid"):
        rollout.RolloutDependencies("a" * 40, "main").validate()


def test_adapter_must_bind_both_generator_and_tck_authorities() -> None:
    adapter = _Adapter()
    adapter.tck_revision = "c" * 40

    with pytest.raises(rollout.RolloutError, match="tck-revision-mismatch"):
        rollout._adapter_revision_guard(adapter, _dependencies())


def test_normalize_generator_result_rejects_absolute_or_duplicate_outputs() -> None:
    base = _Adapter._result()
    base["generated_outputs"] = ["/tmp/leak"]
    with pytest.raises(rollout.RolloutError, match="generator-output-invalid"):
        rollout.normalize_generator_result(base)

    base["generated_outputs"] = ["llms.txt", "llms.txt"]
    with pytest.raises(rollout.RolloutError, match="generator-output-duplicate"):
        rollout.normalize_generator_result(base)


def test_apply_replay_is_idempotent_without_a_second_generator_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project()
    manifest = _manifest(project)
    adapter = _Adapter()
    monkeypatch.setattr(rollout, "collect_repository_evidence", lambda *_: _evidence(project))
    monkeypatch.setattr(rollout, "_safe_repository_root", lambda *_: tmp_path)
    journal = rollout.TransactionJournal(tmp_path / "journal.json")

    first = rollout.apply_project(
        tmp_path, manifest, project, _dependencies(), adapter, journal, confirm=True
    )
    second = rollout.apply_project(
        tmp_path, manifest, project, _dependencies(), adapter, journal, confirm=True
    )

    assert first["status"] == "applied"
    assert second == {
        "ok": True,
        "status": "applied",
        "identity": project.identity,
        "replayed": True,
        "transaction_id": first["transaction_id"],
        "artifact_digest": "c" * 64,
        "generator_version": "1.0.0",
        "generated_outputs": ["agent-readiness-manifest.json", "llms.txt"],
    }
    assert adapter.apply_calls == 1
    assert adapter.verify_calls == 1


def test_apply_failure_rolls_back_and_records_bounded_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project()
    manifest = _manifest(project)
    adapter = _Adapter(fail_apply=True)
    monkeypatch.setattr(rollout, "collect_repository_evidence", lambda *_: _evidence(project))
    monkeypatch.setattr(rollout, "_safe_repository_root", lambda *_: tmp_path)
    journal = rollout.TransactionJournal(tmp_path / "journal.json")

    result = rollout.apply_project(
        tmp_path, manifest, project, _dependencies(), adapter, journal, confirm=True
    )

    assert result["status"] == "rolled_back"
    assert result["error_code"] == "generator-failed"
    assert adapter.rollback_calls == 1
    assert journal.records()[-1]["state"] == "rolled_back"
    assert "fixture failure" not in json.dumps(journal.records())


def test_prepared_transaction_fails_closed_for_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project()
    manifest = _manifest(project)
    adapter = _Adapter()
    evidence = _evidence(project)
    monkeypatch.setattr(rollout, "collect_repository_evidence", lambda *_: evidence)
    monkeypatch.setattr(rollout, "_safe_repository_root", lambda *_: tmp_path)
    journal = rollout.TransactionJournal(tmp_path / "journal.json")
    key = rollout._operation_key(manifest, project, evidence, _dependencies())
    journal.append(
        rollout._base_record(
            operation_key=key,
            transaction_id="fixture-tx",
            mode="apply",
            state="prepared",
            manifest=manifest,
            evidence=evidence,
            dependencies=_dependencies(),
        )
    )

    with pytest.raises(rollout.RolloutError, match="rollout-recovery-required"):
        rollout.apply_project(
            tmp_path, manifest, project, _dependencies(), adapter, journal, confirm=True
        )
    assert adapter.apply_calls == 0


def test_dirty_target_is_refused_before_preview_generator(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = _project()
    manifest = _manifest(project)
    adapter = _Adapter()
    monkeypatch.setattr(rollout, "collect_repository_evidence", lambda *_: _evidence(project, clean=False))

    with pytest.raises(rollout.RolloutError, match="repository-dirty"):
        rollout.preview_project(tmp_path, manifest, project, _dependencies(), adapter)
    assert adapter.preview_calls == 0
