"""Focused RMDD-10 durability and authority-boundary tests."""

from __future__ import annotations

import os
import subprocess
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager import build_artifacts as artifact_module
from repository_manager import build_queue as bq
from repository_manager.build_artifacts import (
    ArtifactFenceLost,
    ArtifactStoreError,
    BuildArtifactStore,
)
from repository_manager.build_service import (
    BuildExecutionDescriptor,
    BuildService,
    BuildServiceError,
    descriptor_input_digest,
    dirty_snapshot_digest,
)
from repository_manager.build_worker import BuildWorker, BuildWorkerError
from repository_manager.capacity import CapacityInventory, HostCapacity, ResourceVector
from repository_manager.development import JobState, ResourceRequest
from repository_manager.development.jobs import (
    DurableJobView,
    FakeRepositoryJobPort,
    RepositoryJobService,
)
from repository_manager.reservations import InMemoryWorkItemReservationPort
from repository_manager.resource_scheduler import (
    AdmissionReason,
    AdmissionRequest,
    AdmissionStatus,
    ResourceScheduler,
)


def _repo(
    tmp_path: Path,
    *,
    command: str = "print(1)",
    toolchain: str | None = None,
) -> Path:
    repo = tmp_path / "same-basename"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "user.email", "tests@example.invalid"], cwd=repo, check=True
    )
    subprocess.run(["git", "config", "user.name", "RMDD tests"], cwd=repo, check=True)
    toolchain_line = (
        f"    toolchain_fingerprint: [python3, -c, {toolchain!r}]\n"
        if toolchain is not None
        else ""
    )
    (repo / ".buildcache.yaml").write_text(
        f"""schema_version: 2
base: main
specs:
  - name: test-build
    command: [python3, -c, {command!r}]
    artifacts: [out.txt]
{toolchain_line}
    resource_class: light-check
""",
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "."], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repo, check=True)
    return repo


class _TypedJobService:
    """Small typed-extension fake; no descriptor is stored in correlation_id."""

    def __init__(self) -> None:
        self.port = FakeRepositoryJobPort()
        self.service = RepositoryJobService(self.port)
        self.descriptors: dict[str, object] = {}

    def submit(self, *args: object, **kwargs: object) -> object:
        return self.service.submit(*args, **kwargs)  # type: ignore[arg-type]

    def get(self, job_id: str, *, auth: object) -> object:
        del auth
        return self.port.rows.get(job_id)

    def cancel(self, job_id: str, *, auth: object, reason: str) -> object:
        return self.service.cancel(job_id, auth=auth, reason=reason)  # type: ignore[arg-type]

    def submit_build(
        self, request: object, *, descriptor: object, **kwargs: object
    ) -> object:
        raw = dict(request)  # type: ignore[arg-type]
        raw.pop("build_descriptor", None)
        result = self.service.submit(raw, **kwargs)  # type: ignore[arg-type]
        self.descriptors[result.job.job_id] = descriptor  # type: ignore[attr-defined]
        return result


def test_build_service_fails_closed_without_typed_descriptor_authority(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    service = BuildService(RepositoryJobService(FakeRepositoryJobPort()))
    with pytest.raises(BuildServiceError, match="typed build descriptor extension"):
        service.submit(repo_path=repo, spec_name="test-build")


def test_v2_key_busts_each_declared_input_dimension(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    config = bq.load_config(repo)
    spec = config.spec("test-build")
    repository_id = bq.stable_repository_id(repo)
    base = bq.compute_cache_key(repo, spec, repo_name=repository_id)
    assert (
        base.digest
        != bq.compute_cache_key(
            repo, spec, repo_name=repository_id, generation_id="g1"
        ).digest
    )
    assert (
        base.digest
        != bq.compute_cache_key(
            repo, spec, repo_name=repository_id, config_digest="changed"
        ).digest
    )
    assert (
        base.digest
        != bq.compute_cache_key(
            repo,
            replace(spec, target_triple="test-target"),
            repo_name=repository_id,
        ).digest
    )
    assert (
        base.digest
        != bq.compute_cache_key(
            repo,
            replace(spec, command=("python3", "-c", "print('different')")),
            repo_name=repository_id,
        ).digest
    )
    toolchain_one = replace(
        spec, toolchain_fingerprint=("python3", "-c", "print('toolchain-one')")
    )
    toolchain_two = replace(
        spec, toolchain_fingerprint=("python3", "-c", "print('toolchain-two')")
    )
    assert bq.compute_cache_key(
        repo, toolchain_one, repo_name=repository_id
    ).digest != (
        bq.compute_cache_key(repo, toolchain_two, repo_name=repository_id).digest
    )
    (repo / "tree-input.txt").write_text("changed", encoding="utf-8")
    subprocess.run(["git", "add", "tree-input.txt"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "tree input"], cwd=repo, check=True)
    changed_tree = bq.compute_cache_key(repo, spec, repo_name=repository_id)
    assert changed_tree.digest != base.digest


def test_valid_v2_hit_does_not_submit_or_admit(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    typed = _TypedJobService()
    store = BuildArtifactStore(repo_path=repo)
    service = BuildService(typed, artifact_store=store)
    key, _spec, _config, _tree = service.key(repo_path=repo, spec_name="test-build")
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "out.txt").write_text("hit", encoding="utf-8")
    staged = store.stage(
        producer,
        workdir=".",
        patterns=["out.txt"],
        key=key.digest,
        attempt=1,
        fence="f-hit",
        job_id="job-hit",
        work_item_id="work-hit",
    )
    store.publish(staged)
    store.finalize(
        key.digest,
        fence="f-hit",
        terminal_check=lambda: True,
        job_id="job-hit",
        work_item_id="work-hit",
        attempt=1,
    )
    result = service.submit(repo_path=repo, spec_name="test-build")
    assert result["cached"] is True
    assert typed.port.rows == {}


def test_uncacheable_toolchain_submits_durable_without_cache_hit(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    config_path = repo / ".buildcache.yaml"
    config_path.write_text(
        config_path.read_text(encoding="utf-8").replace(
            "    artifacts: [out.txt]\n",
            "    artifacts: [out.txt]\n"
            '    toolchain_fingerprint: [python3, -c, "import sys; sys.exit(1)"]\n',
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", ".buildcache.yaml"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-qm", "unfingerprintable toolchain"], cwd=repo, check=True
    )
    typed = _TypedJobService()
    result = BuildService(typed).submit(repo_path=repo, spec_name="test-build")
    assert result["uncacheable"] is True
    assert result["cached"] is False
    assert result["key"] is None
    assert len(typed.port.rows) == 1


def test_worker_recomputes_enabled_toolchain_fingerprint_on_materialized_sha(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(
        tmp_path, store, toolchain="print('stable-toolchain')"
    )
    original = bq._toolchain_fingerprint  # noqa: SLF001
    calls: list[tuple[Path, tuple[str, ...]]] = []

    def observe(tree: Path, spec: bq.BuildSpec) -> str | None:
        calls.append((tree, spec.toolchain_fingerprint))
        return original(tree, spec)

    monkeypatch.setattr(bq, "_toolchain_fingerprint", observe)
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is True
    assert calls and calls[0][1] == (
        "python3",
        "-c",
        "print('stable-toolchain')",
    )


def test_worker_refuses_changed_cacheable_toolchain_before_executor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(
        tmp_path, store, toolchain="print('stable-toolchain')"
    )
    monkeypatch.setattr(
        bq,
        "_toolchain_fingerprint",
        lambda _tree, _spec: "changed-toolchain",
    )

    class NeverExecutor:
        def run(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("toolchain refusal must precede the compiler")

    result = BuildWorker(
        authority,
        scheduler,
        artifact_store=store,
        executor=NeverExecutor(),  # type: ignore[arg-type]
    ).run_job(authority.row.job_id, repo_path=repo, spec_name="test-build")
    assert result["state"] == JobState.FAILED.value
    assert result["refusal_code"] == "worker_environment_failure"
    assert authority.commits[-1]["outcome"] == "failed"


def test_worker_rechecks_uncacheable_toolchain_without_publishing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(
        tmp_path, store, toolchain="import sys; sys.exit(1)"
    )
    calls: list[tuple[str, ...]] = []

    def unavailable(_tree: Path, spec: bq.BuildSpec) -> None:
        calls.append(spec.toolchain_fingerprint)
        return None

    monkeypatch.setattr(bq, "_toolchain_fingerprint", unavailable)
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is True
    assert result["degraded"] is True
    assert calls == [("python3", "-c", "import sys; sys.exit(1)")]
    assert list(store.iter_entries()) == []


def test_dirty_canonical_build_is_refused_before_durable_submission(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / "dirty-input.txt").write_text("dirty", encoding="utf-8")
    typed = _TypedJobService()
    with pytest.raises(BuildServiceError, match="immutable typed snapshot"):
        BuildService(typed).submit(repo_path=repo, spec_name="test-build")
    assert typed.port.rows == {}


def test_two_services_share_one_job_and_fresh_worker_reads_frozen_descriptor(
    tmp_path: Path,
) -> None:
    repo = _repo(
        tmp_path, command="from pathlib import Path; Path('out.txt').write_text('old')"
    )
    typed = _TypedJobService()
    first = BuildService(typed)
    second = BuildService(typed)
    third = BuildService(typed)
    submitted = first.submit(
        repo_path=repo, spec_name="test-build", generation_id="g-1"
    )
    duplicate = second.submit(
        repo_path=repo, spec_name="test-build", generation_id="g-1"
    )
    third_duplicate = third.submit(
        repo_path=repo, spec_name="test-build", generation_id="g-1"
    )
    assert submitted["job_id"] == duplicate["job_id"]
    assert submitted["job_id"] == third_duplicate["job_id"]
    assert duplicate["deduplicated"] is True
    assert third_duplicate["deduplicated"] is True
    assert len(typed.port.rows) == 1

    view = typed.port.rows[submitted["job_id"]]

    class Authority:
        def get(self, job_id: str) -> object:
            return typed.port.rows[job_id]

        def get_build_descriptor(self, job_id: str) -> object:
            return typed.descriptors[job_id]

        def claim(self, job_id: str, *, token: str) -> None:
            del job_id, token

    worker = BuildWorker(Authority(), None)
    # Mutating the caller config after submission cannot change the frozen
    # command recovered by a fresh worker instance.
    (repo / ".buildcache.yaml").write_text(
        (repo / ".buildcache.yaml").read_text(encoding="utf-8").replace("old", "new"),
        encoding="utf-8",
    )
    _scope, spec, key, descriptor = worker._execution_plan(  # noqa: SLF001
        view, repo_path=repo, spec_name="test-build"
    )
    assert spec.command[-1].endswith("old')")
    assert key is not None and descriptor["base_sha"] == view.base_sha


def test_three_frontend_jobs_obey_native_profile_concurrency_limit() -> None:
    now = datetime.now(UTC)
    port = InMemoryWorkItemReservationPort()
    scheduler = ResourceScheduler(
        capacity=CapacityInventory(
            [
                HostCapacity(
                    "local",
                    ResourceVector(64, 100_000, 100_000, 32),
                    heartbeat_at=now,
                )
            ]
        ),
        work_item_port=port,
    )
    decisions = []
    for index in range(3):
        work_item = f"frontend-work-{index}"
        fence = f"frontend-fence-{index}"
        port.claim(work_item, fence=fence)
        decisions.append(
            scheduler.admit(
                AdmissionRequest(
                    work_item_id=work_item,
                    attempt=1,
                    fence=fence,
                    resources=ResourceRequest(resource_class="frontend-build"),
                    repository_id=f"repository-{index}",
                ),
                now=now,
            )
        )
    assert decisions[0].admitted
    assert all(
        decision.reason_code is AdmissionReason.CONCURRENCY
        for decision in decisions[1:]
    )
    assert sum(record.active for record in scheduler.reservations.all()) == 1


def test_submission_rechecks_head_after_key_before_typed_submit(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    typed = _TypedJobService()
    service = BuildService(typed)
    original_mapping = service._request_mapping  # noqa: SLF001

    def mutate_after_descriptor(**kwargs: object) -> dict[str, object]:
        request = original_mapping(**kwargs)  # type: ignore[arg-type]
        (repo / "between-key-and-submit.txt").write_text("changed", encoding="utf-8")
        subprocess.run(
            ["git", "add", "between-key-and-submit.txt"], cwd=repo, check=True
        )
        subprocess.run(
            ["git", "commit", "-qm", "mutation between key and submit"],
            cwd=repo,
            check=True,
        )
        return request

    service._request_mapping = mutate_after_descriptor  # type: ignore[method-assign]
    with pytest.raises(BuildServiceError, match="HEAD changed|key inputs changed"):
        service.submit(repo_path=repo, spec_name="test-build")
    assert typed.port.rows == {}


def test_waiter_cancel_only_stops_the_local_waiter(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    typed = _TypedJobService()
    producer = BuildService(typed)
    waiter = BuildService(typed)
    first = producer.submit(repo_path=repo, spec_name="test-build")
    second = waiter.submit(repo_path=repo, spec_name="test-build")
    assert first["job_id"] == second["job_id"]
    assert second["waiter"] is True
    assert waiter.status(job_id=second["job_id"])["job"]["session_id"]
    cancelled = waiter.cancel(second["job_id"])
    assert cancelled["wait_cancelled"] is True
    assert cancelled["producer_cancelled"] is False
    assert typed.port.rows[first["job_id"]].state is JobState.READY
    assert typed.port.cancel_calls == 0


def test_restart_status_and_cancel_are_durable(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    typed = _TypedJobService()
    submitted = BuildService(typed).submit(repo_path=repo, spec_name="test-build")
    fresh = BuildService(typed)
    status = fresh.status(job_id=submitted["job_id"])
    assert status["job"]["state"] == JobState.READY.value
    cancelled = fresh.cancel_producer(submitted["job_id"])
    assert cancelled["job"]["state"] == JobState.CANCELLED.value
    assert BuildService(typed).status(job_id=submitted["job_id"])["job"]["state"] == (
        JobState.CANCELLED.value
    )


def test_artifacts_reject_symlinks_exact_identity_and_require_terminal_proof(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    output = tree / "out"
    output.mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    os.symlink(outside, output / "link.txt")
    store = BuildArtifactStore(tmp_path / "cache")
    with pytest.raises(ArtifactStoreError, match="symlink"):
        store.stage(
            tree,
            workdir="out",
            patterns=["*.txt"],
            key="v2:symlink",
            attempt=1,
            fence="f1",
            job_id="job-1",
            work_item_id="work-1",
        )
    staging_key = store.staging_root / "v2-symlink"
    assert not staging_key.exists() or not any(staging_key.iterdir())

    (output / "link.txt").unlink()
    (output / "artifact.txt").write_text("ok", encoding="utf-8")
    staged = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:identity",
        attempt=1,
        fence="f1",
        job_id="job-1",
        work_item_id="work-1",
    )
    published = store.publish(staged)
    with pytest.raises(ArtifactFenceLost, match="terminal"):
        store.finalize(
            staged.key,
            fence="f1",
            job_id="job-1",
            work_item_id="work-1",
            attempt=1,
        )
    store.finalize(
        staged.key,
        fence="f1",
        terminal_check=lambda: True,
        job_id="job-1",
        work_item_id="work-1",
        attempt=1,
    )
    assert published["publication_state"] == "published"
    assert not staged.stage_dir.exists()
    other = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:identity",
        attempt=2,
        fence="f2",
        job_id="job-2",
        work_item_id="work-2",
    )
    with pytest.raises(ArtifactStoreError, match="another job"):
        store.publish(other)

    retry = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:identity",
        attempt=1,
        fence="f1",
        job_id="job-1",
        work_item_id="work-1",
    )
    assert store.publish(retry)["publication_state"] == "committed"
    assert not retry.stage_dir.exists()


def test_stage_scans_only_candidate_roots_and_bounds_sparse_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "frontend"
    dist = tree / "dist"
    dist.mkdir(parents=True)
    (dist / "bundle.js").write_text("ok", encoding="utf-8")
    assets = dist / "assets"
    assets.mkdir()
    (assets / "app.js").write_text("ok", encoding="utf-8")
    (tree / "unrelated-large.bin").write_bytes(b"x" * (1 << 20))
    node_modules = tree / "node_modules"
    external = tmp_path / "external-node-modules"
    external.mkdir()
    os.symlink(external, node_modules)
    store = BuildArtifactStore(tmp_path / "cache")
    staged = store.stage(
        tree,
        workdir=".",
        patterns=["dist/**"],
        key="v2:frontend-bounded",
        attempt=1,
        fence="f1",
        job_id="job-1",
        work_item_id="work-1",
        max_bytes=4,
    )
    assert {entry["relative_path"] for entry in staged.manifest["artifacts"]} == {
        "dist/bundle.js",
        "dist/assets/app.js",
    }
    store.discard_stage(staged)

    deep = dist / "deep"
    deep.mkdir()
    (deep / "bundle.js").write_text("ok", encoding="utf-8")
    monkeypatch.setattr(artifact_module, "_MAX_SCAN_ENTRIES", 1)
    with pytest.raises(ArtifactStoreError, match="entry bound"):
        store.stage(
            tree,
            workdir=".",
            patterns=["dist/**"],
            key="v2:frontend-bounded-2",
            attempt=1,
            fence="f1",
            job_id="job-1",
            work_item_id="work-1",
        )


def test_validate_manifest_caps_sparse_bytes_and_fsync_refuses_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    key = "v2:oversized-file"
    artifact_dir = store.root / key / "artifacts"
    artifact_dir.mkdir(parents=True)
    artifact = artifact_dir / "large.bin"
    with artifact.open("wb") as handle:
        handle.truncate(2 * 1024 * 1024)
    manifest = {
        "schema": "build-artifact:v2",
        "key": key,
        "publication_state": "committed",
        "artifacts": [
            {
                "stored_at": str(artifact),
                "sha256": "0" * 64,
                "bytes": 2 * 1024 * 1024,
            }
        ],
    }
    monkeypatch.setattr(artifact_module, "_MAX_STAGE_BYTES", 1 << 20)
    assert store.validate_manifest(manifest, expected_key=key) is False
    link = artifact_dir / "link.bin"
    os.symlink(artifact, link)
    with pytest.raises(ArtifactStoreError, match="could not fsync|non-regular"):
        artifact_module._fsync_file(link)  # noqa: SLF001


def test_publish_fence_loss_after_rename_consumes_stage_for_reconciliation(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    output = tree / "out"
    output.mkdir(parents=True)
    (output / "artifact.txt").write_text("ok", encoding="utf-8")
    store = BuildArtifactStore(tmp_path / "cache")
    staged = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:fence-loss",
        attempt=1,
        fence="f1",
        job_id="job-1",
        work_item_id="work-1",
    )
    checks = iter((True, True, True, False))
    with pytest.raises(ArtifactFenceLost):
        store.publish(staged, fence_check=lambda: next(checks))
    assert not staged.stage_dir.exists()


def test_staging_reconcile_requires_authority_and_removes_terminal_orphan(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    output = tree / "out"
    output.mkdir(parents=True)
    (output / "artifact.txt").write_text("ok", encoding="utf-8")
    store = BuildArtifactStore(tmp_path / "cache")
    staged = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:orphan-stage",
        attempt=1,
        fence="f1",
        job_id="job-1",
        work_item_id="work-1",
    )
    old = 1
    os.utime(staged.stage_dir, (old, old))
    report_only = store.reconcile_staging(max_age_seconds=0)
    assert report_only["removed"] == []
    assert staged.stage_dir.exists()
    removed = store.reconcile_staging(
        max_age_seconds=0,
        authority_probe=lambda manifest: {
            "job_id": manifest["job_id"],
            "work_item_id": manifest["work_item_id"],
            "attempt": manifest["attempt"],
            "fence": manifest["fence"],
            "stale": manifest["job_id"] == "job-1",
        },
    )
    assert removed["removed"]
    assert not staged.stage_dir.exists()

    staged_again = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:orphan-stage",
        attempt=2,
        fence="f2",
        job_id="job-2",
        work_item_id="work-2",
    )
    os.utime(staged_again.stage_dir, (old, old))
    wrong_proof = store.reconcile_staging(
        max_age_seconds=0,
        authority_probe=lambda _manifest: {
            "job_id": "job-other",
            "work_item_id": "work-2",
            "attempt": 2,
            "fence": "f2",
            "stale": True,
        },
    )
    assert wrong_proof["removed"] == []
    assert staged_again.stage_dir.exists()


def test_stage_cleans_private_bytes_when_admission_bound_is_exceeded(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    output = tree / "out"
    output.mkdir(parents=True)
    (output / "artifact.txt").write_text("too large", encoding="utf-8")
    store = BuildArtifactStore(tmp_path / "cache")
    with pytest.raises(ArtifactStoreError, match="byte bound"):
        store.stage(
            tree,
            workdir="out",
            patterns=["*.txt"],
            key="v2:bounded",
            attempt=1,
            fence="f1",
            job_id="job-1",
            work_item_id="work-1",
            max_bytes=1,
        )
    staging_key = store.staging_root / "v2-bounded"
    assert not staging_key.exists() or not any(staging_key.iterdir())


def test_publish_quarantines_orphan_final_directory_before_republish(
    tmp_path: Path,
) -> None:
    tree = tmp_path / "tree"
    output = tree / "out"
    output.mkdir(parents=True)
    (output / "artifact.txt").write_text("ok", encoding="utf-8")
    store = BuildArtifactStore(tmp_path / "cache")
    key = "v2:orphan"
    orphan = store.root / key / "artifacts"
    orphan.mkdir(parents=True)
    (orphan / "leftover").write_text("crash", encoding="utf-8")
    staged = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key=key,
        attempt=1,
        fence="f1",
        job_id="job-1",
        work_item_id="work-1",
    )
    published = store.publish(staged)
    assert published["publication_state"] == "published"
    assert store.read_manifest(key) is not None
    assert tuple(store.quarantine_root.iterdir())


def test_live_published_manifest_is_not_deleted_by_second_service(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    typed = _TypedJobService()
    store = BuildArtifactStore(repo_path=repo)
    first = BuildService(typed, artifact_store=store)
    second = BuildService(typed, artifact_store=store)
    submitted = first.submit(repo_path=repo, spec_name="test-build")
    view = typed.port.rows[submitted["job_id"]]
    typed.port.rows[submitted["job_id"]] = view.model_copy(
        update={"attempt": 1, "lease_fence": "f1"}
    )
    producer = tmp_path / "producer"
    producer.mkdir()
    (producer / "out.txt").write_text("producer", encoding="utf-8")
    staged = store.stage(
        producer,
        workdir=".",
        patterns=["out.txt"],
        key=submitted["key"],
        attempt=1,
        fence="f1",
        job_id=submitted["job_id"],
        work_item_id=submitted["work_item_id"],
    )
    store.publish(staged)
    duplicate = second.submit(repo_path=repo, spec_name="test-build")
    assert duplicate["job_id"] == submitted["job_id"]
    assert duplicate["deduplicated"] is True
    assert store.read_manifest(submitted["key"]) is not None
    assert bq._manifest_path(submitted["key"], repo).exists()  # noqa: SLF001
    with pytest.raises(bq.BuildQueueError, match="committed cached build"):
        bq.artifact_paths(repo_path=repo, key=submitted["key"])
    gc_result = bq.gc(repo_path=repo, keep_recent=0, max_age_days=0)
    assert submitted["key"] in gc_result["kept"]
    assert bq._manifest_path(submitted["key"], repo).exists()  # noqa: SLF001


def test_dirty_snapshot_changes_when_untracked_build_input_changes(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    untracked = repo / "dirty-input.txt"
    untracked.write_text("one", encoding="utf-8")
    first = dirty_snapshot_digest(repo)
    untracked.write_text("two", encoding="utf-8")
    second = dirty_snapshot_digest(repo)
    assert first != second


def test_dirty_snapshot_rejects_oversized_untracked_file_before_reading(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path)
    (repo / "oversized-input.bin").write_bytes(b"x" * ((1 << 20) + 1))
    with pytest.raises(BuildServiceError, match="durable bound"):
        dirty_snapshot_digest(repo)


def _worker_fixture(
    tmp_path: Path,
    store: BuildArtifactStore,
    *,
    toolchain: str | None = None,
):
    repo = _repo(
        tmp_path,
        command="from pathlib import Path; Path('out.txt').write_text('built')",
        toolchain=toolchain,
    )
    typed = _TypedJobService()
    submitted = BuildService(typed, artifact_store=store).submit(
        repo_path=repo, spec_name="test-build"
    )
    row = typed.port.rows[submitted["job_id"]]

    class Authority:
        def __init__(self) -> None:
            self.row = row
            self.commits: list[dict[str, object]] = []

        def get(self, job_id: str) -> DurableJobView:
            assert job_id == self.row.job_id
            return self.row

        def get_build_descriptor(self, job_id: str) -> object:
            return typed.descriptors[job_id]

        def claim(self, job_id: str, *, token: str) -> dict[str, object]:
            del token
            self.row = self.row.model_copy(
                update={
                    "state": JobState.LEASED,
                    "attempt": 1,
                    "lease_owner": "worker:test",
                    "lease_fence": "f1",
                }
            )
            return {
                "job_id": job_id,
                "work_item_id": self.row.work_item_id,
                "attempt": 1,
                "fence": "f1",
            }

        def is_current(self, job_id: str, claim: object) -> bool:
            del job_id
            return self.row.state is JobState.LEASED and claim["fence"] == "f1"

        def heartbeat(self, job_id: str, claim: object) -> bool:
            del job_id, claim
            return True

        def terminal_matches(
            self, job_id: str, claim: object, *, result_ref: str
        ) -> bool:
            return (
                job_id == self.row.job_id
                and self.row.state is JobState.SUCCEEDED
                and self.row.result_ref == result_ref
                and claim["fence"] == "f1"
            )

        def commit(self, job_id: str, claim: object, **kwargs: object) -> str:
            del claim
            self.commits.append(dict(kwargs))
            if kwargs.get("outcome") == "succeeded":
                self.row = self.row.model_copy(
                    update={
                        "state": JobState.SUCCEEDED,
                        "result_ref": kwargs.get("result_ref"),
                    }
                )
            else:
                self.row = self.row.model_copy(update={"state": JobState.FAILED})
            return "committed"

        def cancel(self, job_id: str, *, reason: str) -> bool:
            del job_id, reason
            return False

    class Scheduler:
        def __init__(self) -> None:
            self.releases: list[str] = []
            self.release_behavior: object = True

        def admit(self, request: object) -> object:
            del request
            return SimpleNamespace(admitted=True, reservation_id="reservation:test")

        def release(self, reservation_id: str, **kwargs: object) -> bool:
            del kwargs
            self.releases.append(reservation_id)
            if self.release_behavior == "raise":
                raise RuntimeError("injected scheduler release outage")
            return bool(self.release_behavior)

    return repo, Authority(), Scheduler()


def test_worker_executes_after_admission_and_recovers_terminal_manifest(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    worker = BuildWorker(authority, scheduler, artifact_store=store)
    result = worker.run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is True
    assert result["state"] == "succeeded"
    assert scheduler.releases == ["reservation:test"]
    manifest = store.read_manifest(result["key"])
    assert manifest and manifest["publication_state"] == "committed"
    fresh = BuildWorker(authority, scheduler, artifact_store=store)
    recovered = fresh.recover(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert recovered["ok"] is True
    assert recovered["recovered"] is True


def test_worker_recover_reconciles_old_stage_only_after_terminal_authority_proof(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    producer = tmp_path / "staged-output"
    producer.mkdir()
    (producer / "out.txt").write_text("staged", encoding="utf-8")
    descriptor = authority.get_build_descriptor(authority.row.job_id)
    key = descriptor.cache_key  # type: ignore[union-attr]
    staged = store.stage(
        producer,
        workdir=".",
        patterns=["out.txt"],
        key=key,
        attempt=1,
        fence="f1",
        job_id=authority.row.job_id,
        work_item_id=authority.row.work_item_id,
    )
    old = 1
    os.utime(staged.stage_dir, (old, old))
    authority.row = authority.row.model_copy(
        update={"state": JobState.FAILED, "attempt": 1, "lease_fence": "f1"}
    )
    recovered = BuildWorker(
        authority,
        scheduler,
        artifact_store=store,
        stale_stage_age_seconds=0,
    ).recover(authority.row.job_id, repo_path=repo, spec_name="test-build")
    reconciliation = recovered["staging_reconciliation"]
    assert reconciliation["removed"]
    assert not staged.stage_dir.exists()
    assert recovered["state"] == JobState.FAILED.value


def test_restart_quarantines_corrupt_terminal_manifest_with_exact_proof(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    manifest = store.read_manifest(result["key"])
    assert manifest is not None
    artifact = Path(manifest["artifacts"][0]["stored_at"])
    artifact.write_text("corrupt", encoding="utf-8")
    recovered = BuildWorker(authority, scheduler, artifact_store=store).recover(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert recovered["ok"] is False
    assert recovered["state"] == JobState.SUCCEEDED.value
    assert tuple(store.quarantine_root.iterdir())
    assert store.read_manifest(result["key"]) is None


def test_worker_reports_reconciliation_after_terminal_commit_finalize_failure(
    tmp_path: Path,
) -> None:
    class FinalizeFailsOnce(BuildArtifactStore):
        failed = False

        def finalize(self, *args: object, **kwargs: object) -> dict[str, object]:
            if not self.failed:
                self.failed = True
                raise ArtifactFenceLost("injected finalize crash")
            return super().finalize(*args, **kwargs)

    store = FinalizeFailsOnce(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    worker = BuildWorker(authority, scheduler, artifact_store=store)
    result = worker.run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is False
    assert result["state"] == "succeeded"
    assert result["reconciliation_pending"] is True
    assert [item["outcome"] for item in authority.commits] == ["succeeded"]
    producer = tmp_path / "post-terminal-stage"
    producer.mkdir()
    (producer / "out.txt").write_text("duplicate stage", encoding="utf-8")
    manifest = store.read_manifest(result["key"])
    assert manifest is not None
    staged = store.stage(
        producer,
        workdir=".",
        patterns=["out.txt"],
        key=result["key"],
        attempt=1,
        fence=manifest["fence"],
        job_id=authority.row.job_id,
        work_item_id=authority.row.work_item_id,
    )
    os.utime(staged.stage_dir, (1, 1))
    recovered = BuildWorker(authority, scheduler, artifact_store=store).recover(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert recovered["ok"] is True
    assert staged.stage_dir.exists()


def test_degraded_success_commit_response_must_be_accepted(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(
        tmp_path, store, toolchain="import sys; sys.exit(1)"
    )
    original_commit = authority.commit

    def unacknowledged_commit(job_id: str, claim: object, **kwargs: object) -> str:
        if kwargs.get("outcome") == "succeeded":
            return "not-committed"
        return original_commit(job_id, claim, **kwargs)

    authority.commit = unacknowledged_commit  # type: ignore[method-assign]
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is False
    assert result["state"] == JobState.FAILED.value
    assert authority.row.state is JobState.FAILED
    assert all(item["outcome"] != "succeeded" for item in authority.commits)


def test_degraded_success_commit_ack_loss_reports_reconciliation(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(
        tmp_path, store, toolchain="import sys; sys.exit(1)"
    )
    original_commit = authority.commit

    def commit_then_lose_response(job_id: str, claim: object, **kwargs: object) -> str:
        result = original_commit(job_id, claim, **kwargs)
        if kwargs.get("outcome") == "succeeded":
            raise RuntimeError("degraded success response lost after durable commit")
        return result

    authority.commit = commit_then_lose_response  # type: ignore[method-assign]
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is False
    assert result["state"] == JobState.SUCCEEDED.value
    assert result["reconciliation_pending"] is True
    assert authority.row.state is JobState.SUCCEEDED
    assert [item["outcome"] for item in authority.commits] == ["succeeded"]


def test_degraded_recover_requires_exact_terminal_result_ref(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(
        tmp_path, store, toolchain="import sys; sys.exit(1)"
    )
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is True
    assert result["result_ref"] == (f"build-degraded:{authority.row.job_id}:fence:f1")
    recovered = BuildWorker(authority, scheduler, artifact_store=store).recover(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert recovered["ok"] is True
    assert recovered["recovered"] is True

    authority.row = authority.row.model_copy(
        update={
            "result_ref": f"build-degraded:{authority.row.job_id}:fence:f2",
        }
    )
    forged = BuildWorker(authority, scheduler, artifact_store=store).recover(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert forged["ok"] is False
    assert "exact job/fence" in forged["error"]

    authority.row = authority.row.model_copy(update={"result_ref": None})
    missing = BuildWorker(authority, scheduler, artifact_store=store).recover(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert missing["ok"] is False
    assert "exact job/fence" in missing["error"]


@pytest.mark.parametrize("component", ["repo", "spec", "feature_set", "target_triple"])
def test_worker_rejects_forged_cache_key_identity_before_admission(
    tmp_path: Path, component: str
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, _scheduler = _worker_fixture(tmp_path, store)
    descriptor = authority.get_build_descriptor(authority.row.job_id)
    assert isinstance(descriptor, BuildExecutionDescriptor)
    components = dict(descriptor.key_components)
    components[component] = f"forged-{component}"
    forged_key = bq.CacheKey(**components)
    body = descriptor.model_dump(mode="python")
    body["key_components"] = components
    body["cache_key"] = forged_key.digest
    body["input_digest"] = descriptor_input_digest(body)
    forged_descriptor = BuildExecutionDescriptor.model_validate(body)
    authority.get_build_descriptor = lambda _job_id: forged_descriptor  # type: ignore[method-assign]

    class NeverScheduler:
        def admit(self, request: object) -> object:
            del request
            raise AssertionError("forged key must be rejected before admission")

    result = BuildWorker(
        authority,
        NeverScheduler(),  # type: ignore[arg-type]
        artifact_store=store,
    ).run_job(authority.row.job_id, repo_path=repo, spec_name="test-build")
    assert result["state"] == JobState.FAILED.value
    assert "persisted build key" in result["error"]
    assert component in result["error"]


def test_worker_does_not_fail_after_success_commit_response_is_lost(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    original_commit = authority.commit

    def commit_then_lose_response(job_id: str, claim: object, **kwargs: object) -> str:
        result = original_commit(job_id, claim, **kwargs)
        if kwargs.get("outcome") == "succeeded":
            raise RuntimeError("success response lost after durable commit")
        return result

    authority.commit = commit_then_lose_response  # type: ignore[method-assign]
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is False
    assert result["state"] == "succeeded"
    assert result["reconciliation_pending"] is True
    assert [item["outcome"] for item in authority.commits] == ["succeeded"]


def test_deferred_admission_stays_retryable_without_terminalizing_work_item(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, _scheduler = _worker_fixture(tmp_path, store)

    class DeferredScheduler:
        def admit(self, request: object) -> object:
            del request
            return SimpleNamespace(
                status=AdmissionStatus.DEFERRED,
                admitted=False,
                reservation_id="",
                reason_code=AdmissionReason.CAPACITY,
                reason="capacity is temporarily unavailable",
            )

        def release(self, reservation_id: str, **kwargs: object) -> bool:
            del reservation_id, kwargs
            raise AssertionError("deferred admission must not release a reservation")

    class NeverExecutor:
        def run(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("deferred admission must precede execution")

    result = BuildWorker(
        authority,
        DeferredScheduler(),  # type: ignore[arg-type]
        artifact_store=store,
        executor=NeverExecutor(),  # type: ignore[arg-type]
    ).run_job(authority.row.job_id, repo_path=repo, spec_name="test-build")
    assert result["deferred"] is True
    assert result["retryable"] is True
    assert result["admission_status"] == AdmissionStatus.DEFERRED.value
    assert authority.commits == []
    assert authority.row.state is JobState.LEASED


def test_stale_fence_admission_is_not_retryable_on_the_same_claim(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, _scheduler = _worker_fixture(tmp_path, store)

    class StaleFenceScheduler:
        def admit(self, request: object) -> object:
            del request
            return SimpleNamespace(
                status=AdmissionStatus.STALE_FENCE,
                admitted=False,
                reservation_id="",
                reason_code=AdmissionReason.STALE_FENCE,
                reason="claim fence is no longer current",
            )

        def release(self, reservation_id: str, **kwargs: object) -> bool:
            del reservation_id, kwargs
            raise AssertionError("stale-fence admission must not release a reservation")

    result = BuildWorker(
        authority,
        StaleFenceScheduler(),  # type: ignore[arg-type]
        artifact_store=store,
    ).run_job(authority.row.job_id, repo_path=repo, spec_name="test-build")
    assert result["stale_fence"] is True
    assert result["deferred"] is False
    assert result["retryable"] is False
    assert result["reconciliation_required"] is True
    assert authority.commits == []


def test_worker_rejects_claim_for_another_job_or_work_item(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, _scheduler = _worker_fixture(tmp_path, store)
    original_claim = authority.claim

    def wrong_claim(job_id: str, *, token: str) -> dict[str, object]:
        claimed = original_claim(job_id, token=token)
        claimed["job_id"] = "job-not-authorized"
        return claimed

    authority.claim = wrong_claim  # type: ignore[method-assign]
    with pytest.raises(BuildWorkerError, match="claim job identity"):
        BuildWorker(authority, None, artifact_store=store).run_job(
            authority.row.job_id, repo_path=repo, spec_name="test-build"
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_id", "generation-not-in-descriptor", "generation"),
        ("config_digest", "config-not-in-descriptor", "config digest"),
        ("resource_class", "frontend-build", "resource profile"),
    ],
)
def test_worker_rejects_descriptor_view_identity_mismatch(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    authority.row = authority.row.model_copy(update={field: value})
    result = BuildWorker(authority, scheduler, artifact_store=store).run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["state"] == JobState.FAILED.value
    assert message in result["error"]


@pytest.mark.parametrize("release_behavior", [False, "raise"])
def test_worker_surfaces_release_reconciliation_without_rerunning(
    tmp_path: Path, release_behavior: object
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, scheduler = _worker_fixture(tmp_path, store)
    scheduler.release_behavior = release_behavior
    worker = BuildWorker(authority, scheduler, artifact_store=store)
    result = worker.run_job(
        authority.row.job_id, repo_path=repo, spec_name="test-build"
    )
    assert result["ok"] is True
    assert result["release_pending"] is True
    assert result["reconciliation_pending"] is True
    assert worker.release_failures()
    scheduler.release_behavior = True
    reconciled = worker.reconcile_releases()
    assert reconciled["ok"] is True
    assert worker.release_failures() == {}
    assert [item["outcome"] for item in authority.commits] == ["succeeded"]


def test_disk_admission_refusal_happens_before_executor_process(
    tmp_path: Path,
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    repo, authority, _scheduler = _worker_fixture(tmp_path, store)

    class RefusingScheduler:
        def admit(self, request: object) -> object:
            del request
            return SimpleNamespace(
                admitted=False,
                reservation_id=None,
                reason_code=SimpleNamespace(value="disk_pressure"),
            )

        def release(self, reservation_id: str, **kwargs: object) -> bool:
            del reservation_id, kwargs
            raise AssertionError("a refused admission must not reserve or release")

    class NeverExecutor:
        def run(self, *args: object, **kwargs: object) -> object:
            del args, kwargs
            raise AssertionError("executor must not run after disk refusal")

    result = BuildWorker(
        authority,
        RefusingScheduler(),  # type: ignore[arg-type]
        artifact_store=store,
        executor=NeverExecutor(),  # type: ignore[arg-type]
    ).run_job(authority.row.job_id, repo_path=repo, spec_name="test-build")
    assert result["state"] == "failed"
    assert authority.commits[-1]["outcome"] == "failed"


def test_gc_protects_live_pinned_waited_and_running_keys(tmp_path: Path) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    key = "v2:protected"
    entry = store.root / key
    entry.mkdir()
    (entry / "manifest.json").write_text(
        '{"schema":"build-artifact:v2","key":"v2:protected",'
        '"publication_state":"committed","artifacts":[]}',
        encoding="utf-8",
    )
    for protected_name in ("live_keys", "pinned_keys", "waited_keys", "running_keys"):
        result = store.garbage_collect(
            keep_recent=0,
            max_age_days=0,
            authority_probe=lambda _key, _manifest: True,
            **{protected_name: {key}},
        )
        assert key in result["kept"]
        assert entry.exists()


def test_gc_requires_exact_authority_proof_before_reclaiming(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    output = tree / "out"
    output.mkdir(parents=True)
    (output / "artifact.txt").write_text("ok", encoding="utf-8")
    store = BuildArtifactStore(tmp_path / "cache")
    staged = store.stage(
        tree,
        workdir="out",
        patterns=["*.txt"],
        key="v2:gc-exact",
        attempt=1,
        fence="f1",
        job_id="job-gc",
        work_item_id="work-gc",
    )
    store.publish(staged)
    wrong = store.garbage_collect(
        keep_recent=0,
        max_age_days=0,
        authority_probe=lambda _key, _manifest: True,  # type: ignore[return-value]
    )
    assert "v2:gc-exact" in wrong["kept"]
    exact = store.garbage_collect(
        keep_recent=0,
        max_age_days=0,
        authority_probe=lambda _key, manifest: {
            "job_id": manifest["job_id"],
            "work_item_id": manifest["work_item_id"],
            "attempt": manifest["attempt"],
            "fence": manifest["fence"],
            "safe_to_remove": True,
        },
    )
    assert "v2:gc-exact" in exact["removed"]


def test_gc_probe_fails_closed_and_stable_identity_does_not_use_basename(
    tmp_path: Path,
) -> None:
    first = _repo(tmp_path)
    second_root = tmp_path / "other" / first.name
    second_root.mkdir(parents=True)
    assert bq.stable_repository_id(first) != bq.stable_repository_id(second_root)
    store = BuildArtifactStore(tmp_path / "cache")
    (store.root / "v2:gc" / "artifacts").mkdir(parents=True)
    (store.root / "v2:gc" / "artifacts" / "x").write_text("x", encoding="utf-8")
    (store.root / "v2:gc" / "manifest.json").write_text(
        f'{{"schema":"build-artifact:v2","key":"v2:gc","publication_state":"committed",'
        f'"artifacts":[{{"stored_at":"{store.root / "v2:gc" / "artifacts" / "x"}",'
        '"sha256":"2d711642b726b04401627ca9fbac32f5da7e1e1f9b9f0f3f8f2f7b9b1b9b9b9b9",'
        '"bytes":1}]}',
        encoding="utf-8",
    )
    assert store.remove_entry("v2:gc") == 0


def test_artifact_manifest_and_root_scan_bounds_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = BuildArtifactStore(tmp_path / "cache")
    oversized = store.root / "v2:oversized"
    oversized.mkdir()
    (oversized / "manifest.json").write_bytes(
        b"{" + b'"x":"' + b"x" * (1 << 20) + b'"}'
    )
    assert store.read_manifest("v2:oversized") is None

    (store.root / ".hidden-entry").mkdir()
    (store.root / "v2:visible-entry").mkdir()
    monkeypatch.setattr(artifact_module, "_MAX_SCAN_ENTRIES", 3)
    with pytest.raises(ArtifactStoreError, match="entry scan"):
        store.iter_entries()


def test_legacy_compatibility_artifacts_reject_symlinks_and_bound_gc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = _repo(tmp_path)
    output = tree / "out"
    output.mkdir(parents=True)
    real = output / "real.txt"
    real.write_text("ok", encoding="utf-8")
    os.symlink(real, output / "link.txt")
    spec = bq.BuildSpec(
        name="legacy", command=("true",), workdir="out", artifacts=("*.txt",)
    )
    with pytest.raises(bq.BuildQueueError, match="symlink"):
        bq._publish_artifacts(tree, spec, "v1:legacy", tree)  # noqa: SLF001

    root = bq._artifact_root(tree)  # noqa: SLF001
    (root / "v1:one").mkdir(parents=True, exist_ok=True)
    (root / "v1:two").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(bq, "_MAX_LEGACY_SCAN_ENTRIES", 1)
    result = bq.gc(repo_path=tree, keep_recent=0, max_age_days=0)
    assert result["removed"] == []
    assert result["errors"]
