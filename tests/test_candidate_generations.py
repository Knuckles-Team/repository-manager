"""Pure RMDD-12 candidate-generation domain tests."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from repository_manager.candidate_generation import (
    CandidateGenerationError,
    CandidateLedger,
    CandidateSnapshot,
    GenerationLedger,
    snapshot_branch_candidate,
    snapshot_candidate,
)
from repository_manager.development import (
    GenerationState,
    RepositoryIdentity,
    canonical_digest,
)
from repository_manager.generation_bisection import (
    AttemptResult,
    DecisionAction,
    FailureKind,
    child_lineage,
    decide,
    reusable_evidence,
)
from repository_manager.generation_coalescing import (
    generation_id_for,
    next_generation_candidates,
    seal_generation,
    select_batches,
)

BASE_SHA = "0" * 40
REPO_ROOT = Path("/tmp/rmdd-12-test-repository").resolve()


@dataclass
class MemoryStore:
    records: list[dict[str, Any]] = field(default_factory=list)

    def append(self, record: dict[str, Any], *, lane: str) -> Path:
        del lane
        self.records.append(record)
        return Path("memory")

    def fold(
        self, resolve: Callable[[list[dict[str, Any]]], dict[str, Any]]
    ) -> list[dict[str, Any]]:
        groups: dict[str, list[dict[str, Any]]] = {}
        for record in self.records:
            groups.setdefault(str(record["record_id"]), []).append(record)
        return [resolve(group) for group in groups.values()]


def _repository() -> RepositoryIdentity:
    return RepositoryIdentity(repository_id="repo", canonical_path=str(REPO_ROOT))


def _candidate(
    index: int,
    *,
    enqueued_at: str = "2026-08-09T12:00:00Z",
    candidate_sha: str | None = None,
    base_sha: str = BASE_SHA,
    config: str = "config",
    toolchain: str = "toolchain",
    resource: str = "resource",
    target: str = "default",
    concepts: tuple[str, ...] = (),
    labels: tuple[str, ...] = (),
) -> CandidateSnapshot:
    legacy = SimpleNamespace(
        branch=f"feature/{index}",
        lane=f"lane-{index}",
        base="main",
        enqueued_at=enqueued_at,
    )
    return snapshot_candidate(
        legacy,
        repository=_repository(),
        candidate_sha=candidate_sha or f"{index:040x}",
        base_sha=base_sha,
        config_digest=config,
        toolchain_digest=toolchain,
        resource_digest=resource,
        build_target=target,
        concept_claims=concepts,
        incompatibility_labels=labels,
        target_branch="main",
    )


def test_three_compatible_candidates_coalesce_in_stable_order() -> None:
    candidates = [_candidate(i) for i in (3, 1, 2)]
    now = datetime(2026, 8, 9, 12, 1, tzinfo=UTC)

    first = select_batches(candidates, now=now, batch_size=8)
    second = select_batches(reversed(candidates), now=now, batch_size=8)

    assert [[item.branch for item in batch] for batch in first.batches] == [
        ["feature/1", "feature/2", "feature/3"]
    ]
    assert first.batches == second.batches
    assert generation_id_for(first.selected) == generation_id_for(second.selected)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config", "other-config"),
        ("toolchain", "other-toolchain"),
        ("resource", "other-resource"),
        ("target", "other-target"),
        ("concepts", ("different-concept",)),
        ("labels", ("incompatible",)),
    ],
)
def test_incompatible_generation_inputs_form_separate_batches(
    field: str, value: object
) -> None:
    left = _candidate(1)
    if field == "config":
        right = _candidate(2, config=str(value))
    elif field == "toolchain":
        right = _candidate(2, toolchain=str(value))
    elif field == "resource":
        right = _candidate(2, resource=str(value))
    elif field == "target":
        right = _candidate(2, target=str(value))
    elif field == "concepts":
        assert isinstance(value, tuple)
        right = _candidate(2, concepts=tuple(str(item) for item in value))
    else:
        assert isinstance(value, tuple)
        right = _candidate(2, labels=tuple(str(item) for item in value))
    result = select_batches([left, right], now=datetime(2026, 8, 9, 12, 1, tzinfo=UTC))
    assert len(result.batches) == 2


def test_late_arrival_is_reserved_for_the_next_generation() -> None:
    seal = datetime(2026, 8, 9, 12, 1, tzinfo=UTC)
    late = _candidate(2, enqueued_at="2026-08-09T12:01:01Z")
    result = select_batches([_candidate(1), late], now=seal, sealed_at=seal)

    assert [item.branch for item in result.selected] == ["feature/1"]
    assert result.late == (late,)
    assert next_generation_candidates([late], sealed_at=seal) == (late,)


def test_debounce_waits_but_maximum_age_forces_selection() -> None:
    now = datetime(2026, 8, 9, 12, 10, tzinfo=UTC)
    fresh = _candidate(1, enqueued_at="2026-08-09T12:09:30Z")
    old = _candidate(2, enqueued_at="2026-08-09T12:00:00Z")
    result = select_batches(
        [fresh, old], now=now, debounce=60, maximum_age=300, batch_size=8
    )

    assert result.selected == (old,)
    assert result.waiting == (fresh,)


def test_branch_or_base_movement_creates_a_new_immutable_version() -> None:
    store = MemoryStore()
    ledger = CandidateLedger(store=store)
    legacy = SimpleNamespace(
        branch="feature/moving",
        lane="lane-moving",
        base="main",
        enqueued_at="2026-08-09T12:00:00Z",
    )
    refs = {"feature/moving": "1" * 40, "main": BASE_SHA}
    first = snapshot_branch_candidate(
        legacy,
        repository=_repository(),
        resolve_ref=refs.__getitem__,
        config_digest="config",
        ledger=ledger,
    )
    same = snapshot_branch_candidate(
        legacy,
        repository=_repository(),
        resolve_ref=refs.__getitem__,
        config_digest="config",
        ledger=ledger,
    )
    refs["feature/moving"] = "2" * 40
    refs["main"] = "3" * 40
    moved = snapshot_branch_candidate(
        legacy,
        repository=_repository(),
        resolve_ref=refs.__getitem__,
        config_digest="config",
        ledger=ledger,
    )

    assert first.version == same.version == 1
    assert moved.version == 2
    assert len(ledger.all()) == 2
    assert ledger.get(first.record_id) == first


def test_restart_fold_is_idempotent_and_generation_membership_is_immutable() -> None:
    candidate_store = MemoryStore()
    candidate_ledger = CandidateLedger(store=candidate_store)
    members = tuple(_candidate(index) for index in (1, 2, 3))
    for member in members:
        candidate_ledger.append(member)
    restarted_candidates = CandidateLedger(store=candidate_store)
    assert restarted_candidates.all() == members

    generation_store = MemoryStore()
    generation_ledger = GenerationLedger(store=generation_store)
    sealed = seal_generation(members, sealed_at=datetime(2026, 8, 9, 12, 1, tzinfo=UTC))
    generation_ledger.append(sealed)
    updated_generation = sealed.generation.model_copy(
        update={"state": GenerationState.INTEGRATING}
    )
    generation_ledger.append(
        sealed.with_update(updated_generation, result={"status": "integrating"})
    )
    restarted_generations = GenerationLedger(store=generation_store)
    current = restarted_generations.reconcile()[0]
    assert current.generation.state == GenerationState.INTEGRATING
    assert current.members == members

    changed_versions = (
        current.generation.candidate_versions[0].model_copy(
            update={"candidate_sha": "f" * 40}
        ),
        *current.generation.candidate_versions[1:],
    )
    changed_generation = current.generation.model_copy(
        update={"candidate_versions": changed_versions}
    )
    with pytest.raises(CandidateGenerationError):
        generation_ledger.append(
            sealed.with_update(changed_generation, result={"status": "mutated"})
        )


def test_bisection_isolates_one_bad_candidate_and_reuses_exact_good_evidence() -> None:
    attempt = AttemptResult(
        generation_id="generation:parent",
        member_ids=("a", "b", "c", "d"),
        passed=False,
        failure_class="validation_candidate_failure",
        detail="candidate gate failed",
    )
    split = decide(attempt)
    assert split.action == DecisionAction.SPLIT
    assert split.left_member_ids == ("a", "b")
    assert split.right_member_ids == ("c", "d")

    good = AttemptResult(
        generation_id="generation:good",
        member_ids=split.left_member_ids,
        passed=True,
        evidence_ids=("evidence:good",),
    )
    bad = AttemptResult(
        generation_id="generation:bad",
        member_ids=("c",),
        passed=False,
        failure_class="validation_candidate_failure",
    )
    assert decide(good).action == DecisionAction.ACCEPT
    assert decide(bad).action == DecisionAction.REJECT
    assert reusable_evidence(good, ("a", "b")) == ("evidence:good",)
    assert reusable_evidence(good, ("a",)) == ()
    assert child_lineage(
        "generation:parent", ("generation:bad", "generation:good")
    ) == (
        "generation:bad",
        "generation:good",
    )


def test_environment_failure_retries_unchanged_generation_without_rejection() -> None:
    result = decide(
        AttemptResult(
            generation_id="generation:env",
            member_ids=("a", "b"),
            passed=False,
            failure_class="worker_environment_failure",
            detail="toolchain unavailable",
        )
    )
    assert result.action == DecisionAction.RETRY
    assert result.kind == FailureKind.ENVIRONMENT
    assert result.retry_member_ids == ("a", "b")
    assert result.rejected_member_ids == ()


def test_candidate_snapshot_round_trip_preserves_digest_and_contract() -> None:
    snapshot = _candidate(7, concepts=("concept:a",))
    restored = CandidateSnapshot.from_record(snapshot.to_record())
    assert restored == snapshot
    assert restored.contract.canonical_json() == snapshot.contract.canonical_json()
    assert len(canonical_digest(snapshot.immutable_payload())) == 64
