"""Focused pure RMDD-13 certified landing-policy coverage."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime

import pytest

from repository_manager.development import (
    CandidateVersion,
    Generation,
    GenerationState,
    RepositoryIdentity,
    TargetKind,
    TargetPolicy,
    ValidationStage,
)
from repository_manager.landing_policy import (
    CanonicalCheckoutState,
    LandingPolicyError,
    LandingRefusalCode,
    LandingVerificationRequest,
    LandingVerificationResult,
    TargetOccupancyState,
    verify_landability,
    verify_landing,
)
from repository_manager.validation import (
    EvidenceOutcome,
    GateEvidence,
    ValidationCertificate,
)

SHA0 = "0" * 40
SHA1 = "1" * 40
SHA2 = "2" * 40
SHA3 = "3" * 40
DIGEST0 = "0" * 64
DIGEST1 = "1" * 64
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _repository(repository_id: str = "repository:test") -> RepositoryIdentity:
    return RepositoryIdentity(
        repository_id=repository_id,
        canonical_path="/home/apps/workspace/agent-packages/agents/repository-manager",
    )


def _evidence(
    *,
    generation_id: str = "generation:test",
    tree_sha: str = SHA2,
) -> GateEvidence:
    return GateEvidence(
        evidence_id="evidence:certification",
        gate_name="certification",
        stage=ValidationStage.CERTIFICATION,
        tree_sha=tree_sha,
        generation_id=generation_id,
        gate_config_digest=DIGEST0,
        command_digest=DIGEST1,
        target_host="host:certification",
        toolchain_digest=DIGEST1,
        resource_digest=DIGEST0,
        profile_digest=DIGEST0,
        started_at=NOW,
        finished_at=NOW,
        outcome=EvidenceOutcome.PASSED,
        exit_code=0,
    )


def _generation(
    *,
    repository: RepositoryIdentity | None = None,
    state: GenerationState = GenerationState.CERTIFIED,
) -> Generation:
    return Generation(
        generation_id="generation:test",
        repository=repository or _repository(),
        target_branch="main",
        base_sha=SHA0,
        expected_landing_base_sha=SHA0,
        candidate_versions=(
            CandidateVersion(
                candidate_id="candidate:test",
                version=1,
                candidate_sha=SHA1,
            ),
        ),
        config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
        state=state,
        sealed_at=None if state is GenerationState.OPEN else NOW,
        synthetic_commit_sha=SHA1,
        tree_sha=SHA2,
        validation_evidence_ids=("evidence:certification",),
    )


def _certificate(
    *,
    generation_id: str = "generation:test",
    tree_sha: str = SHA2,
    evidence: GateEvidence | None = None,
) -> tuple[ValidationCertificate, tuple[GateEvidence, ...]]:
    item = evidence or _evidence(generation_id=generation_id, tree_sha=tree_sha)
    certificate = ValidationCertificate.issue(
        certificate_id="certificate:test",
        generation_id=generation_id,
        tree_sha=tree_sha,
        gate_config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
        target_host="host:certification",
        resource_digest=DIGEST0,
        blocking_gate_names=("certification",),
        evidence=(item,),
        issued_at=NOW,
        profile_digest=DIGEST0,
    )
    return certificate, (item,)


def _request(
    *,
    generation: Generation | None = None,
    certificate: ValidationCertificate | None = None,
    evidence: tuple[GateEvidence, ...] = (),
    repository: RepositoryIdentity | None = None,
    target_branch: str = "main",
    target: TargetPolicy | None = None,
    expected_base_sha: str = SHA0,
    observed_target_sha: str = SHA0,
    expected_landing_fence: str | None = "fence:landing-1",
    observed_landing_fence: str | None = "fence:landing-1",
    canonical: CanonicalCheckoutState | None = None,
    target_occupancy: TargetOccupancyState | None = None,
) -> LandingVerificationRequest:
    return LandingVerificationRequest(
        repository=repository or _repository(),
        target_branch=target_branch,
        target=target or TargetPolicy(),
        expected_base_sha=expected_base_sha,
        observed_target_sha=observed_target_sha,
        expected_landing_fence=expected_landing_fence,
        observed_landing_fence=observed_landing_fence,
        generation=generation,
        certificate=certificate,
        canonical=canonical or CanonicalCheckoutState(True, True),
        target_occupancy=target_occupancy or TargetOccupancyState(0),
        evidence=evidence,
    )


@pytest.fixture
def valid_request() -> LandingVerificationRequest:
    generation = _generation()
    certificate, evidence = _certificate()
    return _request(
        generation=generation,
        certificate=certificate,
        evidence=evidence,
    )


def test_certified_landing_is_accepted_without_git_or_filesystem_effects(
    valid_request: LandingVerificationRequest,
) -> None:
    result = verify_landing(valid_request)

    assert result.accepted
    assert result.refusal_code is None
    assert result.generation_id == "generation:test"
    assert result.synthetic_commit_sha == SHA1
    assert result.tree_sha == SHA2
    certificate = valid_request.certificate
    assert certificate is not None
    assert result.certificate_digest == certificate.digest
    assert result.landing_fence == "fence:landing-1"
    assert verify_landability(valid_request) == result


@pytest.mark.parametrize(
    ("landing_request", "code"),
    [
        (_request(), LandingRefusalCode.GENERATION_REQUIRED),
    ],
)
def test_missing_generation_is_refused(
    landing_request: LandingVerificationRequest,
    code: LandingRefusalCode,
) -> None:
    result = verify_landing(landing_request)
    assert result.refused
    assert result.code is code


def test_missing_certificate_is_refused() -> None:
    result = verify_landing(_request(generation=_generation()))
    assert result.code is LandingRefusalCode.CERTIFICATE_REQUIRED


def test_certificate_evidence_is_verified_not_trusted_as_a_status_bit() -> None:
    certificate, _ = _certificate()
    mismatched_evidence = _evidence(tree_sha=SHA3)
    request = _request(
        generation=_generation(),
        certificate=certificate,
        evidence=(mismatched_evidence,),
    )

    result = verify_landing(request)
    assert result.code is LandingRefusalCode.CERTIFICATE_INVALID
    assert "tree SHA" in result.detail


def test_certificate_generation_and_tree_identity_are_exact() -> None:
    certificate, evidence = _certificate()
    generation = _generation()

    generation_mismatch = verify_landing(
        _request(
            generation=generation,
            certificate=replace(certificate, generation_id="generation:other"),
            evidence=evidence,
        )
    )
    assert (
        generation_mismatch.code is LandingRefusalCode.CERTIFICATE_GENERATION_MISMATCH
    )

    tree_mismatch = verify_landing(
        _request(
            generation=generation,
            certificate=replace(certificate, tree_sha=SHA3),
            evidence=evidence,
        )
    )
    assert tree_mismatch.code is LandingRefusalCode.CERTIFICATE_TREE_MISMATCH


def test_certificate_configuration_toolchain_and_evidence_identity_are_exact() -> None:
    certificate, evidence = _certificate()
    generation = _generation()

    config_mismatch = verify_landing(
        _request(
            generation=generation,
            certificate=replace(certificate, gate_config_digest=DIGEST1),
            evidence=evidence,
        )
    )
    assert config_mismatch.code is LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH

    toolchain_mismatch = verify_landing(
        _request(
            generation=generation,
            certificate=replace(certificate, toolchain_digest=DIGEST0),
            evidence=evidence,
        )
    )
    assert toolchain_mismatch.code is LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH

    evidence_identity_mismatch = verify_landing(
        _request(
            generation=generation.model_copy(
                update={"validation_evidence_ids": ("evidence:other",)}
            ),
            certificate=certificate,
            evidence=evidence,
        )
    )
    assert (
        evidence_identity_mismatch.code is LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH
    )


@pytest.mark.parametrize(
    "generation",
    [
        _generation(state=GenerationState.OPEN),
        _generation(state=GenerationState.INTEGRATING),
    ],
)
def test_only_certified_generations_can_land(generation: Generation) -> None:
    certificate, evidence = _certificate()
    result = verify_landing(
        _request(generation=generation, certificate=certificate, evidence=evidence)
    )
    assert result.code is LandingRefusalCode.GENERATION_NOT_CERTIFIED


def test_certified_generation_requires_commit_and_tree() -> None:
    certificate, evidence = _certificate()
    incomplete = _generation().model_copy(update={"synthetic_commit_sha": None})
    result = verify_landing(
        _request(generation=incomplete, certificate=certificate, evidence=evidence)
    )
    assert result.code is LandingRefusalCode.GENERATION_INCOMPLETE


def test_repository_and_target_identity_are_exact() -> None:
    certificate, evidence = _certificate()
    generation = _generation()
    wrong_repository = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            repository=_repository("repository:other"),
        )
    )
    assert wrong_repository.code is LandingRefusalCode.REPOSITORY_MISMATCH

    wrong_branch = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            target_branch="release/1",
        )
    )
    assert wrong_branch.code is LandingRefusalCode.TARGET_MISMATCH

    wrong_target = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            target=TargetPolicy(kind=TargetKind.INVENTORY_ALIAS, alias="worker-one"),
        )
    )
    assert wrong_target.code is LandingRefusalCode.TARGET_MISMATCH


def test_expected_base_and_target_cas_observation_are_distinct_checks() -> None:
    certificate, evidence = _certificate()
    generation = _generation()
    wrong_generation_base = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            expected_base_sha=SHA3,
        )
    )
    assert wrong_generation_base.code is LandingRefusalCode.EXPECTED_BASE_MISMATCH

    moved_target = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            observed_target_sha=SHA3,
        )
    )
    assert moved_target.code is LandingRefusalCode.TARGET_MOVED


def test_fence_is_required_current_and_generation_bound() -> None:
    certificate, evidence = _certificate()
    generation = _generation()
    missing = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            expected_landing_fence=None,
            observed_landing_fence=None,
        )
    )
    assert missing.code is LandingRefusalCode.FENCE_REQUIRED

    stale = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            observed_landing_fence="fence:landing-2",
        )
    )
    assert stale.code is LandingRefusalCode.STALE_FENCE

    generation_fence_mismatch = generation.model_copy(
        update={"landing_fence": "fence:other"}
    )
    stale_generation = verify_landing(
        _request(
            generation=generation_fence_mismatch,
            certificate=certificate,
            evidence=evidence,
        )
    )
    assert stale_generation.code is LandingRefusalCode.STALE_FENCE


def test_canonical_lease_and_cleanliness_are_fail_closed() -> None:
    certificate, evidence = _certificate()
    generation = _generation()
    no_lease = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            canonical=CanonicalCheckoutState(False, True),
        )
    )
    assert no_lease.code is LandingRefusalCode.CANONICAL_LEASE_REQUIRED

    dirty = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            canonical=CanonicalCheckoutState(True, False),
        )
    )
    assert dirty.code is LandingRefusalCode.CANONICAL_DIRTY


def test_target_occupancy_refuses_other_worktrees() -> None:
    certificate, evidence = _certificate()
    result = verify_landing(
        _request(
            generation=_generation(),
            certificate=certificate,
            evidence=evidence,
            target_occupancy=TargetOccupancyState(1),
        )
    )
    assert result.code is LandingRefusalCode.TARGET_OCCUPIED


def test_inputs_and_results_are_closed_and_bounded() -> None:
    with pytest.raises(LandingPolicyError, match="immutable tuple"):
        LandingVerificationRequest(
            repository=_repository(),
            target_branch="main",
            target=TargetPolicy(),
            expected_base_sha=SHA0,
            observed_target_sha=SHA0,
            expected_landing_fence="fence:landing-1",
            observed_landing_fence="fence:landing-1",
            generation=None,
            certificate=None,
            canonical=CanonicalCheckoutState(True, True),
            target_occupancy=TargetOccupancyState(0),
            evidence=[],  # type: ignore[arg-type]
        )
    with pytest.raises(LandingPolicyError, match="boolean"):
        CanonicalCheckoutState(1, True)  # type: ignore[arg-type]
    with pytest.raises(LandingPolicyError, match="between"):
        TargetOccupancyState(1025)
    with pytest.raises(LandingPolicyError, match="accepted result requires"):
        LandingVerificationResult(accepted=True)

    certificate, evidence = _certificate()
    with pytest.raises(LandingPolicyError, match="bounded count"):
        _request(
            generation=_generation(),
            certificate=certificate,
            evidence=evidence * 257,
        )
    with pytest.raises(LandingPolicyError, match="bounded size"):
        _request(
            generation=_generation(),
            certificate=certificate,
            evidence=(replace(evidence[0], log_refs=("x" * 1025,)),),
        )
    with pytest.raises(LandingPolicyError, match="bounded size"):
        _request(
            generation=_generation(),
            certificate=certificate,
            evidence=evidence,
            expected_landing_fence="f" * 257,
            observed_landing_fence="f" * 257,
        )

    refusal = LandingVerificationResult(
        accepted=False,
        refusal_code=LandingRefusalCode.CERTIFICATE_INVALID,
        detail="candidate-controlled detail" * 1000,
    )
    assert refusal.refused
    assert len(refusal.detail.encode("utf-8")) <= 4096
