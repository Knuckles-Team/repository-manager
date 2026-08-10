"""Focused pure RMDD-13 certified landing-policy coverage."""

from __future__ import annotations

import warnings
from collections.abc import Iterator, Mapping
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import pytest

import repository_manager.landing_policy as landing_policy
from repository_manager.development import (
    CONTRACT_VERSION,
    CandidateVersion,
    Generation,
    GenerationState,
    LandingOutcome,
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
    _plain_payload_mapping,
    _require_sequence,
    verify_landability,
    verify_landing,
)
from repository_manager.validation import (
    EvidenceOutcome,
    GateEvidence,
    ValidationCertificate,
    ValidationFailureClass,
)

_UNSET = object()


class _NeverIteratedMapping(Mapping[str, object]):
    def __getitem__(self, key: str) -> object:
        raise AssertionError(f"hostile mapping was indexed: {key}")

    def __iter__(self) -> Iterator[str]:
        raise AssertionError("hostile mapping was iterated")

    def __len__(self) -> int:
        raise AssertionError("hostile mapping length was requested")


class _NeverIteratedList(list[object]):
    def __iter__(self) -> Iterator[object]:
        raise AssertionError("hostile list was iterated")

    def __len__(self) -> int:
        raise AssertionError("hostile list length was requested")


class _ExplodingText(str):
    def strip(self, *_args: object, **_kwargs: object) -> str:
        raise RuntimeError("private text method must not run")


SHA0 = "0" * 40
SHA1 = "1" * 40
SHA2 = "2" * 40
SHA3 = "3" * 40
DIGEST0 = "0" * 64
DIGEST1 = "1" * 64
NOW = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)


def _repository(repository_id: str = "repository:test") -> RepositoryIdentity:
    return RepositoryIdentity(
        contract_version=CONTRACT_VERSION,
        repository_id=repository_id,
        canonical_path="/home/apps/workspace/agent-packages/agents/repository-manager",
        configured_roots=(),
        origin=None,
    )


def _evidence(
    *,
    generation_id: str | None = None,
    tree_sha: str = SHA2,
) -> GateEvidence:
    if generation_id is None:
        generation_id = _generation().generation_id
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
    candidates = (
        CandidateVersion(
            contract_version=CONTRACT_VERSION,
            candidate_id="candidate:test",
            version=1,
            candidate_sha=SHA1,
        ),
    )
    generation_id = Generation.derive_id(
        repository_id=(repository or _repository()).repository_id,
        target_branch="main",
        base_sha=SHA0,
        candidate_versions=candidates,
        config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
    )
    return Generation(
        contract_version=CONTRACT_VERSION,
        generation_id=generation_id,
        repository=repository or _repository(),
        target_branch="main",
        target=TargetPolicy(
            contract_version=CONTRACT_VERSION,
            kind=TargetKind.LOCAL,
            alias=None,
            capability_labels=(),
        ),
        base_sha=SHA0,
        expected_landing_base_sha=SHA0,
        candidate_versions=candidates,
        config_digest=DIGEST0,
        toolchain_digest=DIGEST1,
        state=state,
        sealed_at=None if state is GenerationState.OPEN else NOW,
        synthetic_commit_sha=SHA1,
        tree_sha=SHA2,
        validation_evidence_ids=("evidence:certification",),
        build_artifact_refs=(),
        bisection_lineage=(),
        landing_fence="fence:landing-1",
        landing_result=None,
        reason="",
    )


def _certificate(
    *,
    generation_id: str | None = None,
    tree_sha: str = SHA2,
    evidence: GateEvidence | None = None,
) -> tuple[ValidationCertificate, tuple[GateEvidence, ...]]:
    if generation_id is None:
        generation_id = _generation().generation_id
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
    expected_certificate_digest: str | None | object = _UNSET,
    expected_generation_id: str | None | object = _UNSET,
    expected_synthetic_commit_sha: str | None | object = _UNSET,
) -> LandingVerificationRequest:
    if expected_certificate_digest is _UNSET:
        expected_certificate_digest = (
            certificate.digest if certificate is not None else None
        )
    if expected_generation_id is _UNSET:
        expected_generation_id = (
            generation.generation_id if generation is not None else None
        )
    if expected_synthetic_commit_sha is _UNSET:
        expected_synthetic_commit_sha = (
            generation.synthetic_commit_sha if generation is not None else None
        )
    return LandingVerificationRequest(
        repository=repository or _repository(),
        target_branch=target_branch,
        target=target
        or TargetPolicy(
            contract_version=CONTRACT_VERSION,
            kind=TargetKind.LOCAL,
            alias=None,
            capability_labels=(),
        ),
        expected_base_sha=expected_base_sha,
        observed_target_sha=observed_target_sha,
        expected_landing_fence=expected_landing_fence,
        observed_landing_fence=observed_landing_fence,
        generation=generation,
        certificate=certificate,
        canonical=canonical or CanonicalCheckoutState(True, True),
        target_occupancy=target_occupancy or TargetOccupancyState(0),
        evidence=evidence,
        expected_certificate_digest=cast(str | None, expected_certificate_digest),
        expected_generation_id=cast(str | None, expected_generation_id),
        expected_synthetic_commit_sha=cast(str | None, expected_synthetic_commit_sha),
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
    assert valid_request.generation is not None
    assert result.generation_id == valid_request.generation.generation_id
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
    assert "tree SHA" not in result.detail


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
            target=TargetPolicy(
                contract_version=CONTRACT_VERSION,
                kind=TargetKind.INVENTORY_ALIAS,
                alias="worker-one",
                capability_labels=(),
            ),
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
    certificate, evidence = _certificate()
    mutable_request = _request(
        generation=_generation(), certificate=certificate, evidence=evidence
    )
    object.__setattr__(mutable_request, "evidence", [])
    with pytest.raises(LandingPolicyError, match="immutable tuple"):
        mutable_request.__post_init__()

    malformed_canonical = CanonicalCheckoutState(True, True)
    object.__setattr__(malformed_canonical, "mutation_lease_held", 1)
    with pytest.raises(LandingPolicyError, match="boolean"):
        malformed_canonical.__post_init__()
    with pytest.raises(LandingPolicyError, match="between"):
        TargetOccupancyState(1025)
    with pytest.raises(LandingPolicyError, match="accepted result requires"):
        LandingVerificationResult(accepted=True)

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


def test_hostile_text_subclasses_are_refused_before_text_methods_run() -> None:
    with pytest.raises(LandingPolicyError) as construction_error:
        _request(target_branch=_ExplodingText("main"))
    assert "private text method" not in str(construction_error.value)

    generation = _generation()
    certificate, evidence = _certificate(generation_id=generation.generation_id)
    request = _request(
        generation=generation,
        certificate=certificate,
        evidence=evidence,
    )
    object.__setattr__(request, "target_branch", _ExplodingText("main"))
    result = verify_landing(request)
    assert result.code is LandingRefusalCode.REQUEST_INVALID
    assert "private text method" not in result.detail


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("contract_version", 1, id="contract-version-int"),
        pytest.param("generation_id", True, id="generation-id-bool"),
        pytest.param("target_branch", 1, id="target-branch-int"),
        pytest.param("base_sha", SHA0.encode(), id="base-sha-bytes"),
        pytest.param(
            "expected_landing_base_sha", SHA0.encode(), id="expected-base-bytes"
        ),
        pytest.param("sealed_at", 0, id="sealed-at-int"),
        pytest.param("synthetic_commit_sha", SHA1.encode(), id="commit-sha-bytes"),
        pytest.param("tree_sha", SHA2.encode(), id="tree-sha-bytes"),
        pytest.param("state", GenerationState.CERTIFIED.value, id="state-string"),
        pytest.param("landing_result", LandingOutcome.LANDED.value, id="result-string"),
        pytest.param("reason", b"bytes", id="reason-bytes"),
        pytest.param("bisection_lineage", (True,), id="bisection-lineage-bool"),
    ],
)
def test_generation_scalar_swaps_are_typed_refusals(
    field: str, replacement: object
) -> None:
    original = _generation()
    forged = original.model_copy(update={field: replacement})
    certificate, evidence = _certificate(generation_id=original.generation_id)
    result = verify_landing(
        _request(
            generation=forged,
            certificate=certificate,
            evidence=evidence,
            expected_certificate_digest=certificate.digest,
            expected_generation_id=original.generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID
    assert "private" not in result.detail


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("contract_version", b"1", id="contract-version-bytes"),
        pytest.param("candidate_id", 1, id="candidate-id-int"),
        pytest.param("version", True, id="version-bool"),
        pytest.param("version", "1", id="version-string"),
        pytest.param("candidate_sha", SHA1.encode(), id="candidate-sha-bytes"),
    ],
)
def test_candidate_scalar_swaps_are_typed_refusals(
    field: str, replacement: object
) -> None:
    original = _generation()
    candidate = original.candidate_versions[0].model_copy(update={field: replacement})
    forged = original.model_copy(update={"candidate_versions": (candidate,)})
    certificate, evidence = _certificate(generation_id=original.generation_id)
    result = verify_landing(
        _request(
            generation=forged,
            certificate=certificate,
            evidence=evidence,
            expected_certificate_digest=certificate.digest,
            expected_generation_id=original.generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("kind", TargetKind.LOCAL.value, id="kind-string"),
        pytest.param("alias", b"worker-one", id="alias-bytes"),
        pytest.param("capability_labels", (1,), id="capability-int"),
    ],
)
def test_target_scalar_and_nested_swaps_are_typed_refusals(
    field: str, replacement: object
) -> None:
    original = _generation()
    forged_target = original.target.model_copy(update={field: replacement})
    forged = original.model_copy(update={"target": forged_target})
    certificate, evidence = _certificate(generation_id=original.generation_id)
    result = verify_landing(
        _request(
            generation=forged,
            certificate=certificate,
            evidence=evidence,
            expected_certificate_digest=certificate.digest,
            expected_generation_id=original.generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("certificate_id", b"certificate:test", id="id-bytes"),
        pytest.param("tree_sha", SHA2.encode(), id="tree-bytes"),
        pytest.param("issued_at", 0, id="issued-at-int"),
        pytest.param("evidence_digests", b"digest", id="digests-bytes"),
        pytest.param("blocking_gate_names", (1,), id="gate-int"),
    ],
)
def test_certificate_scalar_and_nested_swaps_are_typed_refusals(
    field: str, replacement: object
) -> None:
    generation = _generation()
    certificate, evidence = _certificate(generation_id=generation.generation_id)
    forged = replace(certificate)
    object.__setattr__(forged, field, replacement)
    result = verify_landing(
        _request(
            generation=generation,
            certificate=forged,
            evidence=evidence,
            expected_certificate_digest=certificate.digest,
            expected_generation_id=generation.generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.CERTIFICATE_INVALID


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param("evidence_id", b"evidence:certification", id="id-bytes"),
        pytest.param("stage", ValidationStage.CERTIFICATION.value, id="stage-string"),
        pytest.param("started_at", 0, id="started-at-int"),
        pytest.param("outcome", True, id="outcome-bool"),
        pytest.param("differential", 1, id="differential-int"),
        pytest.param("exit_code", True, id="exit-code-bool"),
        pytest.param("dependency_job_ids", (1,), id="dependency-int"),
    ],
)
def test_evidence_scalar_and_nested_swaps_are_typed_refusals(
    field: str, replacement: object
) -> None:
    generation = _generation()
    certificate, evidence = _certificate(generation_id=generation.generation_id)
    forged = replace(evidence[0])
    object.__setattr__(forged, field, replacement)
    result = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=(forged,),
            expected_certificate_digest=certificate.digest,
            expected_generation_id=generation.generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.EVIDENCE_INVALID


def test_request_and_decision_scalars_are_strict() -> None:
    generation = _generation()
    certificate, evidence = _certificate(generation_id=generation.generation_id)
    request = _request(
        generation=generation,
        certificate=certificate,
        evidence=evidence,
    )
    object.__setattr__(request, "expected_base_sha", True)
    result = verify_landing(request)
    assert result.code is LandingRefusalCode.REQUEST_INVALID

    with pytest.raises(LandingPolicyError):
        LandingVerificationResult(accepted=cast(bool, 1))
    with pytest.raises(LandingPolicyError):
        LandingVerificationResult(
            accepted=False,
            refusal_code=LandingRefusalCode.CERTIFICATE_INVALID,
            detail=_ExplodingText("detail"),
        )


def test_model_copy_authority_values_are_rebuilt_and_anchored() -> None:
    original_generation = _generation()
    certificate, evidence = _certificate()
    candidate = original_generation.candidate_versions[0]
    forged_values = (
        (
            original_generation.model_copy(
                update={
                    "repository": original_generation.repository.model_copy(
                        update={"repository_id": "repository:forged"}
                    )
                }
            ),
            LandingRefusalCode.GENERATION_ID_MISMATCH,
        ),
        (
            original_generation.model_copy(
                update={
                    "target": TargetPolicy(
                        contract_version=CONTRACT_VERSION,
                        kind=TargetKind.INVENTORY_ALIAS,
                        alias="worker-one",
                        capability_labels=(),
                    )
                }
            ),
            LandingRefusalCode.TARGET_MISMATCH,
        ),
        (
            original_generation.model_copy(update={"base_sha": SHA3}),
            LandingRefusalCode.GENERATION_ID_MISMATCH,
        ),
        (
            original_generation.model_copy(
                update={
                    "candidate_versions": (
                        candidate.model_copy(update={"candidate_sha": SHA3}),
                    )
                }
            ),
            LandingRefusalCode.GENERATION_ID_MISMATCH,
        ),
        (
            original_generation.model_copy(update={"synthetic_commit_sha": SHA3}),
            LandingRefusalCode.GENERATION_ANCHOR_MISMATCH,
        ),
        (
            original_generation.model_copy(update={"tree_sha": SHA3}),
            LandingRefusalCode.CERTIFICATE_TREE_MISMATCH,
        ),
        (
            original_generation.model_copy(update={"validation_evidence_ids": ()}),
            LandingRefusalCode.GENERATION_INVALID,
        ),
        (
            original_generation.model_copy(update={"landing_fence": "fence:forged"}),
            LandingRefusalCode.STALE_FENCE,
        ),
    )
    for forged_generation, expected_code in forged_values:
        result = verify_landing(
            _request(
                generation=forged_generation,
                certificate=certificate,
                evidence=evidence,
                expected_certificate_digest=certificate.digest,
                expected_generation_id=original_generation.generation_id,
                expected_synthetic_commit_sha=original_generation.synthetic_commit_sha,
            )
        )
        assert result.code is expected_code


_GENERATION_FIELDS = (
    "contract_version",
    "generation_id",
    "repository",
    "target_branch",
    "target",
    "base_sha",
    "expected_landing_base_sha",
    "candidate_versions",
    "config_digest",
    "toolchain_digest",
    "state",
    "sealed_at",
    "synthetic_commit_sha",
    "tree_sha",
    "validation_evidence_ids",
    "build_artifact_refs",
    "bisection_lineage",
    "landing_fence",
    "landing_result",
    "reason",
)


@pytest.mark.parametrize(
    "omitted_fields",
    [pytest.param((field,), id=f"missing-{field}") for field in _GENERATION_FIELDS]
    + [pytest.param(None, id="missing-all-fields")],
)
def test_model_construct_missing_generation_fields_is_typed_refusal(
    omitted_fields: tuple[str, ...] | None,
) -> None:
    original = _generation()
    state = object.__getattribute__(original, "__dict__")
    if omitted_fields is None:
        forged = Generation.model_construct()
    else:
        forged = Generation.model_construct(
            **{
                field: state[field]
                for field in _GENERATION_FIELDS
                if field not in omitted_fields
            }
        )
    certificate, evidence = _certificate(generation_id=original.generation_id)
    request = _request(
        generation=forged,
        certificate=certificate,
        evidence=evidence,
        expected_certificate_digest=certificate.digest,
        expected_generation_id=original.generation_id,
        expected_synthetic_commit_sha=SHA1,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = verify_landing(request)

    assert result.code is LandingRefusalCode.GENERATION_INVALID
    assert "AttributeError" not in result.detail
    assert "private" not in result.detail


_REPOSITORY_FIELDS = (
    "contract_version",
    "repository_id",
    "canonical_path",
    "configured_roots",
    "origin",
)
_TARGET_FIELDS = (
    "contract_version",
    "kind",
    "alias",
    "capability_labels",
)
_CANDIDATE_FIELDS = (
    "contract_version",
    "candidate_id",
    "version",
    "candidate_sha",
)


@pytest.mark.parametrize(
    "omitted_fields",
    [pytest.param((field,), id=f"missing-{field}") for field in _REPOSITORY_FIELDS]
    + [pytest.param(None, id="missing-all-fields")],
)
def test_model_construct_missing_repository_fields_is_typed_refusal(
    omitted_fields: tuple[str, ...] | None,
) -> None:
    original = _generation()
    state = object.__getattribute__(original.repository, "__dict__")
    if omitted_fields is None:
        forged_repository = RepositoryIdentity.model_construct()
    else:
        forged_repository = RepositoryIdentity.model_construct(
            **{
                field: state[field]
                for field in _REPOSITORY_FIELDS
                if field not in omitted_fields
            }
        )
    forged_generation = original.model_copy(update={"repository": forged_repository})
    certificate, evidence = _certificate(generation_id=original.generation_id)
    result = verify_landing(
        _request(
            generation=forged_generation,
            certificate=certificate,
            evidence=evidence,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    "omitted_fields",
    [pytest.param((field,), id=f"missing-{field}") for field in _TARGET_FIELDS]
    + [pytest.param(None, id="missing-all-fields")],
)
def test_model_construct_missing_target_fields_is_typed_refusal(
    omitted_fields: tuple[str, ...] | None,
) -> None:
    original = _generation()
    state = object.__getattribute__(original.target, "__dict__")
    if omitted_fields is None:
        forged_target = TargetPolicy.model_construct()
    else:
        forged_target = TargetPolicy.model_construct(
            **{
                field: state[field]
                for field in _TARGET_FIELDS
                if field not in omitted_fields
            }
        )
    forged_generation = original.model_copy(update={"target": forged_target})
    certificate, evidence = _certificate(generation_id=original.generation_id)
    result = verify_landing(
        _request(
            generation=forged_generation,
            certificate=certificate,
            evidence=evidence,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    "omitted_fields",
    [pytest.param((field,), id=f"missing-{field}") for field in _CANDIDATE_FIELDS]
    + [pytest.param(None, id="missing-all-fields")],
)
def test_model_construct_missing_candidate_fields_is_typed_refusal(
    omitted_fields: tuple[str, ...] | None,
) -> None:
    original = _generation()
    candidate = original.candidate_versions[0]
    state = object.__getattribute__(candidate, "__dict__")
    if omitted_fields is None:
        forged_candidate = CandidateVersion.model_construct()
    else:
        forged_candidate = CandidateVersion.model_construct(
            **{
                field: state[field]
                for field in _CANDIDATE_FIELDS
                if field not in omitted_fields
            }
        )
    forged_generation = original.model_copy(
        update={"candidate_versions": (forged_candidate,)}
    )
    certificate, evidence = _certificate(generation_id=original.generation_id)
    result = verify_landing(
        _request(
            generation=forged_generation,
            certificate=certificate,
            evidence=evidence,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


def test_model_construct_target_string_enum_is_not_coerced() -> None:
    original = _generation()
    target_state = object.__getattribute__(original.target, "__dict__")
    forged_target = TargetPolicy.model_construct(
        **{**target_state, "kind": TargetKind.LOCAL.value}
    )
    forged_generation = original.model_copy(update={"target": forged_target})
    certificate, evidence = _certificate(generation_id=original.generation_id)

    result = verify_landing(
        _request(
            generation=forged_generation,
            certificate=certificate,
            evidence=evidence,
        )
    )

    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("state", GenerationState.CERTIFIED.value, id="state"),
        pytest.param(
            "landing_result", LandingOutcome.LANDED.value, id="landing-result"
        ),
    ],
)
def test_model_construct_generation_string_enums_are_not_coerced(
    field: str, value: str
) -> None:
    original = _generation()
    state = dict(object.__getattribute__(original, "__dict__"))
    state[field] = value
    forged_generation = Generation.model_construct(**state)
    certificate, evidence = _certificate(generation_id=original.generation_id)

    result = verify_landing(
        _request(
            generation=forged_generation,
            certificate=certificate,
            evidence=evidence,
        )
    )

    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    ("field", "value"),
    [
        pytest.param("stage", ValidationStage.CERTIFICATION.value, id="stage"),
        pytest.param("outcome", EvidenceOutcome.PASSED.value, id="outcome"),
        pytest.param(
            "failure_class", ValidationFailureClass.CODE.value, id="failure-class"
        ),
    ],
)
def test_evidence_string_enums_are_not_coerced(field: str, value: str) -> None:
    generation = _generation()
    certificate, evidence = _certificate()
    forged_evidence = replace(evidence[0])
    object.__setattr__(forged_evidence, field, value)

    result = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=(forged_evidence,),
        )
    )

    assert result.code is LandingRefusalCode.EVIDENCE_INVALID


_EVIDENCE_FIELDS = (
    "evidence_id",
    "gate_name",
    "stage",
    "tree_sha",
    "generation_id",
    "gate_config_digest",
    "command_digest",
    "target_host",
    "toolchain_digest",
    "resource_digest",
    "profile_digest",
    "started_at",
    "finished_at",
    "outcome",
    "failure_class",
    "job_id",
    "dependency_job_ids",
    "baseline_tree_sha",
    "baseline_readable",
    "differential",
    "failure_ids",
    "pre_existing_failure_ids",
    "fixed_failure_ids",
    "log_refs",
    "artifact_refs",
    "stdout_tail",
    "stderr_tail",
    "exit_code",
    "snapshot_gate_deferred",
    "snapshot_gate_replayed",
    "detail",
)
_CERTIFICATE_FIELDS = (
    "certificate_id",
    "generation_id",
    "tree_sha",
    "gate_config_digest",
    "toolchain_digest",
    "target_host",
    "resource_digest",
    "evidence_digests",
    "blocking_gate_names",
    "issued_at",
    "profile_digest",
)


@pytest.mark.parametrize(
    "omitted_fields",
    [pytest.param((field,), id=f"missing-{field}") for field in _EVIDENCE_FIELDS]
    + [pytest.param(None, id="missing-all-fields")],
)
def test_missing_evidence_fields_are_typed_refusals(
    omitted_fields: tuple[str, ...] | None,
) -> None:
    generation = _generation()
    certificate, evidence = _certificate()
    forged_evidence = replace(evidence[0])
    fields_to_delete = _EVIDENCE_FIELDS if omitted_fields is None else omitted_fields
    for field in fields_to_delete:
        object.__delattr__(forged_evidence, field)

    result = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=(forged_evidence,),
        )
    )

    assert result.code is LandingRefusalCode.EVIDENCE_INVALID


@pytest.mark.parametrize(
    "omitted_fields",
    [pytest.param((field,), id=f"missing-{field}") for field in _CERTIFICATE_FIELDS]
    + [pytest.param(None, id="missing-all-fields")],
)
def test_missing_certificate_fields_are_typed_refusals(
    omitted_fields: tuple[str, ...] | None,
) -> None:
    generation = _generation()
    certificate, evidence = _certificate()
    forged_certificate = replace(certificate)
    fields_to_delete = _CERTIFICATE_FIELDS if omitted_fields is None else omitted_fields
    for field in fields_to_delete:
        object.__delattr__(forged_certificate, field)

    result = verify_landing(
        _request(
            generation=generation,
            certificate=forged_certificate,
            evidence=evidence,
            expected_certificate_digest=DIGEST0,
        )
    )

    assert result.code is LandingRefusalCode.CERTIFICATE_INVALID


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        pytest.param(
            "repository", RepositoryIdentity.model_construct(), id="repository"
        ),
        pytest.param(
            "candidate_versions", (CandidateVersion.model_construct(),), id="candidate"
        ),
        pytest.param(
            "target",
            TargetPolicy.model_construct(capability_labels=_NeverIteratedList()),
            id="target-container",
        ),
    ],
)
def test_model_construct_malformed_nested_values_are_typed_refusals(
    field: str, replacement: object
) -> None:
    original = _generation()
    forged = original.model_copy(update={field: replacement})
    certificate, evidence = _certificate(generation_id=original.generation_id)

    request = _request(
        generation=forged,
        certificate=certificate,
        evidence=evidence,
        expected_certificate_digest=certificate.digest,
        expected_generation_id=original.generation_id,
        expected_synthetic_commit_sha=SHA1,
    )
    result = verify_landing(request)

    assert result.code is LandingRefusalCode.GENERATION_INVALID
    assert "AttributeError" not in result.detail
    assert "private" not in result.detail


def test_model_construct_generation_subclass_property_is_typed_refusal() -> None:
    class ForgedGeneration(Generation):
        repository = property(
            lambda _self: (_ for _ in ()).throw(RuntimeError("private"))
        )

    original = _generation()
    forged = ForgedGeneration.model_construct()
    certificate, evidence = _certificate(generation_id=original.generation_id)
    request = _request(
        generation=forged,
        certificate=certificate,
        evidence=evidence,
        expected_certificate_digest=certificate.digest,
        expected_generation_id=original.generation_id,
        expected_synthetic_commit_sha=SHA1,
    )

    result = verify_landing(request)

    assert result.code is LandingRefusalCode.GENERATION_INVALID
    assert "private" not in result.detail


@pytest.mark.parametrize(
    "forged_target",
    [
        {},
        {"kind": "malformed"},
        {"kind": "local", "alias": "forged"},
    ],
)
def test_forged_nested_target_dict_is_never_defaulted(
    forged_target: dict[str, object],
) -> None:
    generation = _generation().model_copy(update={"target": forged_target})
    certificate, evidence = _certificate()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = verify_landing(
            _request(generation=generation, certificate=certificate, evidence=evidence)
        )

    assert result.code is LandingRefusalCode.GENERATION_INVALID


def test_forged_nested_target_model_copy_is_typed_refusal() -> None:
    malformed_target = TargetPolicy().model_copy(update={"kind": "malformed"})
    generation = _generation().model_copy(update={"target": malformed_target})
    certificate, evidence = _certificate()

    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        result = verify_landing(
            _request(generation=generation, certificate=certificate, evidence=evidence)
        )

    assert result.code is LandingRefusalCode.GENERATION_INVALID


def test_hostile_custom_containers_are_refused_without_iteration() -> None:
    with pytest.raises(LandingPolicyError):
        _plain_payload_mapping(_NeverIteratedMapping(), "hostile mapping")
    with pytest.raises(LandingPolicyError):
        _require_sequence(_NeverIteratedList(), "hostile list")

    generation = _generation().model_copy(update={"target": _NeverIteratedMapping()})
    certificate, evidence = _certificate()
    result = verify_landing(
        _request(generation=generation, certificate=certificate, evidence=evidence)
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID

    hostile_candidates = _NeverIteratedList()
    generation = _generation().model_copy(
        update={"candidate_versions": hostile_candidates}
    )
    result = verify_landing(
        _request(generation=generation, certificate=certificate, evidence=evidence)
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID

    hostile_labels = _NeverIteratedList()
    malformed_target = TargetPolicy().model_copy(
        update={"capability_labels": hostile_labels}
    )
    generation = _generation().model_copy(update={"target": malformed_target})
    result = verify_landing(
        _request(generation=generation, certificate=certificate, evidence=evidence)
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


def test_list_snapshot_disposition_is_invariant_under_warning_errors() -> None:
    generation = _generation()
    candidates = list(generation.candidate_versions)
    copied_generation = generation.model_copy(update={"candidate_versions": candidates})
    certificate, evidence = _certificate()
    request = _request(
        generation=copied_generation,
        certificate=certificate,
        evidence=evidence,
    )

    normal_result = verify_landing(request)
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        strict_result = verify_landing(request)

    assert strict_result == normal_result
    assert strict_result.accepted


@pytest.mark.parametrize("state", [None, True, object()])
def test_malformed_generation_state_is_a_typed_refusal(state: object) -> None:
    generation = _generation().model_copy(update={"state": state})
    certificate, evidence = _certificate()
    result = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            expected_certificate_digest=certificate.digest,
            expected_generation_id=_generation().generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


def test_malformed_gate_evidence_fields_never_escape_as_attribute_errors() -> None:
    generation = _generation()
    certificate, evidence = _certificate()
    malformed_differential = replace(
        evidence[0], differential=True, baseline_tree_sha=SHA0
    )
    object.__setattr__(malformed_differential, "differential", 1)
    malformed_stage = replace(evidence[0])
    object.__setattr__(malformed_stage, "stage", object())
    malformed_outcome = replace(evidence[0])
    object.__setattr__(malformed_outcome, "outcome", None)
    malformed = (
        malformed_stage,
        malformed_differential,
        malformed_outcome,
    )
    for item in malformed:
        result = verify_landing(
            _request(
                generation=generation,
                certificate=certificate,
                evidence=(item,),
            )
        )
        assert result.code is LandingRefusalCode.EVIDENCE_INVALID

    unsafe = replace(evidence[0], detail="detail\u202ehidden")
    result = verify_landing(
        _request(generation=generation, certificate=certificate, evidence=(unsafe,))
    )
    assert result.code is LandingRefusalCode.EVIDENCE_INVALID


def test_recomputed_certificate_cannot_replace_durable_content_anchor() -> None:
    generation = _generation()
    original_certificate, original_evidence = _certificate()
    altered_evidence = replace(original_evidence[0], detail="altered evidence")
    recomputed_certificate = ValidationCertificate.issue(
        certificate_id=original_certificate.certificate_id,
        generation_id=generation.generation_id,
        tree_sha=original_certificate.tree_sha,
        gate_config_digest=original_certificate.gate_config_digest,
        toolchain_digest=original_certificate.toolchain_digest,
        target_host=original_certificate.target_host,
        resource_digest=original_certificate.resource_digest,
        blocking_gate_names=original_certificate.blocking_gate_names,
        evidence=(altered_evidence,),
        issued_at=original_certificate.issued_at,
        profile_digest=original_certificate.profile_digest,
    )
    result = verify_landing(
        _request(
            generation=generation,
            certificate=recomputed_certificate,
            evidence=(altered_evidence,),
            expected_certificate_digest=original_certificate.digest,
            expected_generation_id=generation.generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.CERTIFICATE_ANCHOR_MISMATCH


def test_generation_list_is_snapshotted_before_caller_mutation() -> None:
    generation = _generation()
    candidates = list(generation.candidate_versions)
    copied_generation = generation.model_copy(update={"candidate_versions": candidates})
    certificate, evidence = _certificate()
    request = _request(
        generation=copied_generation,
        certificate=certificate,
        evidence=evidence,
        expected_certificate_digest=certificate.digest,
        expected_generation_id=generation.generation_id,
        expected_synthetic_commit_sha=SHA1,
    )
    candidates[0] = candidates[0].model_copy(update={"candidate_sha": SHA3})
    assert verify_landing(request).accepted


def test_generator_candidate_membership_is_rejected_without_consumption() -> None:
    generation = _generation().model_copy(
        update={
            "candidate_versions": (item for item in _generation().candidate_versions)
        }
    )
    certificate, evidence = _certificate()
    result = verify_landing(
        _request(
            generation=generation,
            certificate=certificate,
            evidence=evidence,
            expected_certificate_digest=certificate.digest,
            expected_generation_id=_generation().generation_id,
            expected_synthetic_commit_sha=SHA1,
        )
    )
    assert result.code is LandingRefusalCode.GENERATION_INVALID


@pytest.mark.parametrize(
    "unsafe", ["\x7f", "\u0085", "\u202e", "\u2066", "\u2028", "\ue000"]
)
def test_unsafe_unicode_is_rejected_or_safely_normalized(unsafe: str) -> None:
    with pytest.raises(LandingPolicyError):
        _request(target_branch=f"main{unsafe}")
    with pytest.raises(LandingPolicyError):
        _request(expected_landing_fence=f"fence:{unsafe}")
    refusal = LandingVerificationResult(
        accepted=False,
        refusal_code=LandingRefusalCode.CERTIFICATE_INVALID,
        detail=f"safe{unsafe}detail",
    )
    assert unsafe not in refusal.detail


def test_missing_trusted_anchors_is_fail_closed() -> None:
    generation = _generation()
    certificate, evidence = _certificate()
    missing_certificate_anchor = _request(
        generation=generation,
        certificate=certificate,
        evidence=evidence,
        expected_certificate_digest=None,
    )
    assert (
        verify_landing(missing_certificate_anchor).code
        is LandingRefusalCode.CERTIFICATE_ANCHOR_REQUIRED
    )
    missing_generation_anchor = _request(
        generation=generation,
        certificate=certificate,
        evidence=evidence,
        expected_generation_id=None,
    )
    assert (
        verify_landing(missing_generation_anchor).code
        is LandingRefusalCode.GENERATION_ANCHOR_REQUIRED
    )


def test_generation_identity_runtime_error_propagates(
    valid_request: LandingVerificationRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_runtime_error(_cls: type[Generation], **_kwargs: object) -> str:
        raise RuntimeError("injected generation identity failure")

    monkeypatch.setattr(Generation, "derive_id", classmethod(raise_runtime_error))

    with pytest.raises(RuntimeError, match="injected generation identity failure"):
        verify_landing(valid_request)


def test_certificate_verification_runtime_error_propagates(
    valid_request: LandingVerificationRequest,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_runtime_error(
        _certificate: ValidationCertificate,
        _evidence: tuple[GateEvidence, ...],
    ) -> object:
        raise RuntimeError("injected certificate verification failure")

    monkeypatch.setattr(landing_policy, "verify_certificate", raise_runtime_error)

    with pytest.raises(RuntimeError, match="injected certificate verification failure"):
        verify_landing(valid_request)
