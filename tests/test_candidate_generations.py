"""Pure RMDD-12 candidate-generation domain tests."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager.candidate_generation import (
    CandidateGenerationError,
    CandidateSnapshot,
    fold_candidate_records,
    fold_generation_records,
    snapshot_branch_candidate,
    snapshot_candidate,
)
from repository_manager.development import (
    FailureClass,
    GenerationState,
    RepositoryIdentity,
    canonical_digest,
)
from repository_manager.generation_bisection import (
    AttemptResult,
    DecisionAction,
    FailureKind,
    LineageEdge,
    child_lineage,
    decide,
    reusable_evidence,
)
from repository_manager.generation_coalescing import (
    candidates_compatible,
    generation_id_for,
    next_generation_candidates,
    seal_generation,
    select_batches,
)

BASE_SHA = "0" * 40
CONFIG_DIGEST = "1" * 64
TOOLCHAIN_DIGEST = "2" * 64
RESOURCE_DIGEST = "3" * 64
REPO_ROOT = Path("/tmp/rmdd-12-test-repository").resolve()


def _repository() -> RepositoryIdentity:
    return RepositoryIdentity(repository_id="repo", canonical_path=str(REPO_ROOT))


def _candidate(
    index: int,
    *,
    version: int = 1,
    enqueued_at: str = "2026-08-09T12:00:00Z",
    candidate_sha: str | None = None,
    base_sha: str = BASE_SHA,
    config: str = CONFIG_DIGEST,
    toolchain: str = TOOLCHAIN_DIGEST,
    resource: str = RESOURCE_DIGEST,
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
        version=version,
    )


def test_three_compatible_candidates_coalesce_in_stable_order() -> None:
    candidates = [_candidate(i, version=i) for i in (3, 1, 2)]
    now = datetime(2026, 8, 9, 12, 1, tzinfo=UTC)

    first = select_batches(candidates, now=now, batch_size=8)
    second = select_batches(reversed(candidates), now=now, batch_size=8)

    assert [[item.branch for item in batch] for batch in first.batches] == [
        ["feature/1", "feature/2", "feature/3"]
    ]
    assert first.batches == second.batches
    assert generation_id_for(first.selected) == generation_id_for(second.selected)


def test_generation_id_and_members_preserve_actual_v3_v7_versions() -> None:
    members = (_candidate(3, version=3), _candidate(7, version=7))
    first = seal_generation(members, sealed_at=datetime(2026, 8, 9, 12, 1, tzinfo=UTC))
    second = seal_generation(
        reversed(members), sealed_at=datetime(2026, 8, 9, 12, 1, tzinfo=UTC)
    )

    assert [item.version for item in first.generation.candidate_versions] == [3, 7]
    assert first.generation.generation_id == second.generation.generation_id
    assert first.generation.candidate_versions == second.generation.candidate_versions
    ordinal_like = seal_generation(
        (_candidate(3, version=1), _candidate(7, version=2)),
        sealed_at=datetime(2026, 8, 9, 12, 1, tzinfo=UTC),
    )
    assert first.generation.generation_id != ordinal_like.generation.generation_id


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("config", "4" * 64),
        ("toolchain", "5" * 64),
        ("resource", "6" * 64),
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


def test_different_base_shas_are_not_compatible_without_ancestry_lookup() -> None:
    left = _candidate(1, version=1, base_sha="0" * 40)
    right = _candidate(2, version=2, base_sha="f" * 40)

    assert not candidates_compatible(left, right)
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
        config_digest=CONFIG_DIGEST,
        toolchain_digest=TOOLCHAIN_DIGEST,
        resource_digest=RESOURCE_DIGEST,
    )
    same = snapshot_branch_candidate(
        legacy,
        repository=_repository(),
        resolve_ref=refs.__getitem__,
        config_digest=CONFIG_DIGEST,
        toolchain_digest=TOOLCHAIN_DIGEST,
        resource_digest=RESOURCE_DIGEST,
        previous=first,
    )
    refs["feature/moving"] = "2" * 40
    refs["main"] = "3" * 40
    moved = snapshot_branch_candidate(
        legacy,
        repository=_repository(),
        resolve_ref=refs.__getitem__,
        config_digest=CONFIG_DIGEST,
        toolchain_digest=TOOLCHAIN_DIGEST,
        resource_digest=RESOURCE_DIGEST,
        previous=first,
    )

    assert first.version == same.version == 1
    assert moved.version == 2
    assert moved.candidate_version.version == 2
    assert first.candidate_sha != moved.candidate_sha


def test_restart_fold_is_idempotent_and_generation_membership_is_immutable() -> None:
    members = tuple(_candidate(index, version=index) for index in (1, 2, 3))
    candidate_records = [member.to_record() for member in members]
    restarted_candidates = fold_candidate_records(candidate_records)
    assert restarted_candidates == tuple(
        sorted(members, key=lambda item: item.record_id)
    )

    sealed = seal_generation(members, sealed_at=datetime(2026, 8, 9, 12, 1, tzinfo=UTC))
    updated_generation = sealed.generation.model_copy(
        update={"state": GenerationState.INTEGRATING}
    )
    generation_records = [
        sealed.to_record(),
        sealed.with_update(
            updated_generation, result={"status": "integrating"}
        ).to_record(),
    ]
    current = fold_generation_records(generation_records)[0]
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
        sealed.with_update(changed_generation, result={"status": "mutated"})


def test_bisection_isolates_one_bad_candidate_and_reuses_exact_good_evidence() -> None:
    attempt = AttemptResult(
        generation_id="generation:parent",
        member_ids=("a", "b", "c", "d"),
        passed=False,
        failure_class=FailureClass.VALIDATION_CANDIDATE_FAILURE,
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
        failure_class=FailureClass.VALIDATION_CANDIDATE_FAILURE,
    )
    assert decide(good).action == DecisionAction.ACCEPT
    assert decide(bad).action == DecisionAction.REJECT
    assert reusable_evidence(good, ("a", "b")) == ("evidence:good",)
    assert reusable_evidence(good, ("a",)) == ()
    assert child_lineage(
        "generation:parent", ("generation:bad", "generation:good")
    ) == (
        LineageEdge("generation:parent", "generation:bad"),
        LineageEdge("generation:parent", "generation:good"),
    )
    assert (
        child_lineage("generation:parent", ("generation:good", "generation:bad"))[
            0
        ].parent_generation_id
        == "generation:parent"
    )


def test_environment_failure_retries_unchanged_generation_without_rejection() -> None:
    result = decide(
        AttemptResult(
            generation_id="generation:env",
            member_ids=("a", "b"),
            passed=False,
            failure_class=FailureClass.WORKER_ENVIRONMENT_FAILURE,
            detail="toolchain unavailable",
        )
    )
    assert result.action == DecisionAction.RETRY
    assert result.kind == FailureKind.ENVIRONMENT
    assert result.retry_member_ids == ("a", "b")
    assert result.rejected_member_ids == ()
    assert result.quarantined is False

    exhausted = decide(
        AttemptResult(
            generation_id="generation:env",
            member_ids=("a", "b"),
            passed=False,
            failure_class=FailureClass.WORKER_ENVIRONMENT_FAILURE,
            attempt=3,
            attempt_budget=3,
        )
    )
    assert exhausted.action == DecisionAction.RETRY
    assert exhausted.quarantined is True


def test_unknown_failure_code_is_opaque_and_detail_cannot_trigger_bisection() -> None:
    result = decide(
        AttemptResult(
            generation_id="generation:opaque",
            member_ids=("a", "b"),
            passed=False,
            failure_class="untrusted-worker-code",
            detail="candidate compile test environment failure",
        )
    )

    assert result.action == DecisionAction.RETRY
    assert result.kind == FailureKind.OPAQUE
    assert result.retry_member_ids == ("a", "b")
    assert result.left_member_ids == ()
    assert result.rejected_member_ids == ()
    assert result.quarantined is False

    exhausted = decide(
        AttemptResult(
            generation_id="generation:opaque",
            member_ids=("a", "b"),
            passed=False,
            failure_class="untrusted-worker-code",
            attempt=2,
            attempt_budget=2,
        )
    )
    assert exhausted.action == DecisionAction.RETRY
    assert exhausted.quarantined is True

    contradictory = decide(
        AttemptResult(
            generation_id="generation:opaque",
            member_ids=("a", "b"),
            passed=True,
            failure_class="untrusted-worker-code",
        )
    )
    assert contradictory.action == DecisionAction.RETRY
    assert contradictory.kind == FailureKind.OPAQUE


def test_lineage_requires_a_parent_even_for_empty_child_sets() -> None:
    with pytest.raises(ValueError):
        child_lineage("", ())


def test_candidate_snapshot_round_trip_preserves_digest_and_contract() -> None:
    snapshot = _candidate(7, concepts=("concept:a",))
    restored = CandidateSnapshot.from_record(snapshot.to_record())
    assert restored == snapshot
    assert restored.contract.canonical_json() == snapshot.contract.canonical_json()
    assert len(canonical_digest(snapshot.immutable_payload())) == 64


def test_digest_constructor_rejects_shorthand_and_missing_inputs() -> None:
    legacy = SimpleNamespace(
        branch="feature/digest",
        lane="lane-digest",
        base="main",
        enqueued_at="2026-08-09T12:00:00Z",
    )
    with pytest.raises(CandidateGenerationError, match="config_digest"):
        snapshot_candidate(
            legacy,
            repository=_repository(),
            candidate_sha="a" * 40,
            base_sha=BASE_SHA,
            config_digest="config",
            toolchain_digest=TOOLCHAIN_DIGEST,
            resource_digest=RESOURCE_DIGEST,
        )
    with pytest.raises(CandidateGenerationError, match="toolchain_digest"):
        snapshot_candidate(
            legacy,
            repository=_repository(),
            candidate_sha="a" * 40,
            base_sha=BASE_SHA,
            config_digest=CONFIG_DIGEST,
            toolchain_digest="",
            resource_digest=RESOURCE_DIGEST,
        )


def test_decode_fails_closed_on_missing_or_mismatched_record_identity() -> None:
    snapshot_record = _candidate(8).to_record()
    for field in ("record_id", "immutable_digest"):
        malformed = dict(snapshot_record)
        malformed.pop(field)
        with pytest.raises(CandidateGenerationError):
            CandidateSnapshot.from_record(malformed)

    mismatched = dict(snapshot_record)
    mismatched["record_id"] = "candidate:wrong:v1"
    with pytest.raises(CandidateGenerationError):
        CandidateSnapshot.from_record(mismatched)

    bad_digest = dict(snapshot_record)
    bad_digest["immutable_digest"] = "0" * 64
    with pytest.raises(CandidateGenerationError):
        CandidateSnapshot.from_record(bad_digest)


def test_generation_decode_validates_record_key_digest_and_membership() -> None:
    sealed = seal_generation(
        (_candidate(1, version=1), _candidate(2, version=2)),
        sealed_at=datetime(2026, 8, 9, 12, 1, tzinfo=UTC),
    )
    record = sealed.to_record()
    for field in ("record_id", "immutable_digest"):
        malformed = dict(record)
        malformed.pop(field)
        with pytest.raises(CandidateGenerationError):
            type(sealed).from_record(malformed)

    wrong_key = dict(record)
    wrong_key["record_id"] = "generation:wrong"
    with pytest.raises(CandidateGenerationError):
        type(sealed).from_record(wrong_key)

    wrong_digest = dict(record)
    wrong_digest["immutable_digest"] = "0" * 64
    with pytest.raises(CandidateGenerationError):
        type(sealed).from_record(wrong_digest)

    wrong_members = dict(record)
    wrong_members["members"] = [record["members"][0]]
    with pytest.raises(CandidateGenerationError):
        type(sealed).from_record(wrong_members)
