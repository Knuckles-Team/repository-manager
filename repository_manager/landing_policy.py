"""Pure RMDD-13 certified landing policy.

This module is deliberately a value-only boundary.  It verifies the immutable
generation and certificate against one caller-supplied observation of the
repository, canonical checkout, and target branch.  It does not open Git,
acquire a lease, inspect a worktree, move a ref, or write an event.  A later
landing controller owns those effects and must take a fresh observation while
holding the repository-manager canonical lease before it performs the update.

The refusal values are wire-level strings.  They are intentionally more
specific than the broad repository-development failure classes because a
landing caller needs an actionable answer without parsing free-form text.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum

from repository_manager.development import (
    Generation,
    GenerationState,
    RepositoryIdentity,
    TargetPolicy,
)
from repository_manager.development.serialization import canonical_json
from repository_manager.validation import (
    GateEvidence,
    ValidationCertificate,
    verify_certificate,
)

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")
_REF_FORBIDDEN_RE = re.compile(r"[\x00-\x20~^:?*\\\[]")
_MAX_DETAIL_BYTES = 4096
_MAX_TARGET_HOLDERS = 1024
_MAX_EVIDENCE_ITEMS = 256
_MAX_EVIDENCE_SEQUENCE_ITEMS = 256
_MAX_EVIDENCE_FIELD_BYTES = 1024
_MAX_EVIDENCE_TOTAL_BYTES = 4 * 1024 * 1024
_MAX_FENCE_BYTES = 256


class LandingPolicyError(ValueError):
    """A landing-policy input is not one of the closed typed values."""


class LandingRefusalCode(StrEnum):
    """Stable, actionable refusal values for the certified landing seam."""

    GENERATION_REQUIRED = "generation_required"
    CERTIFICATE_REQUIRED = "certificate_required"
    CERTIFICATE_INVALID = "certificate_invalid"
    CERTIFICATE_GENERATION_MISMATCH = "certificate_generation_mismatch"
    CERTIFICATE_TREE_MISMATCH = "certificate_tree_mismatch"
    CERTIFICATE_INPUT_MISMATCH = "certificate_input_mismatch"
    GENERATION_NOT_CERTIFIED = "generation_not_certified"
    GENERATION_INCOMPLETE = "generation_incomplete"
    REPOSITORY_MISMATCH = "repository_mismatch"
    TARGET_MISMATCH = "target_mismatch"
    EXPECTED_BASE_MISMATCH = "expected_base_mismatch"
    TARGET_MOVED = "target_moved"
    FENCE_REQUIRED = "fence_required"
    STALE_FENCE = "stale_fence"
    CANONICAL_LEASE_REQUIRED = "canonical_lease_required"
    CANONICAL_DIRTY = "canonical_dirty"
    TARGET_OCCUPIED = "target_occupied"


def _require_bool(value: object, field_name: str) -> bool:
    if type(value) is not bool:  # bool is intentionally strict at this boundary.
        raise LandingPolicyError(f"{field_name} must be a boolean")
    return value


def _require_sha(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
        raise LandingPolicyError(
            f"{field_name} must be a 40-character lowercase Git SHA"
        )
    return value


def _require_digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise LandingPolicyError(
            f"{field_name} must be a 64-character lowercase digest"
        )
    return value


def _require_bounded_text(
    value: object,
    field_name: str,
    *,
    maximum_bytes: int = _MAX_EVIDENCE_FIELD_BYTES,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise LandingPolicyError(f"{field_name} must be a non-blank string")
    if value.strip() != value:
        raise LandingPolicyError(f"{field_name} must not have surrounding whitespace")
    if any(ord(char) < 0x20 for char in value):
        raise LandingPolicyError(f"{field_name} must not contain control characters")
    if len(value.encode("utf-8")) > maximum_bytes:
        raise LandingPolicyError(f"{field_name} exceeds its bounded size")
    return value


def _require_ref(value: object, field_name: str) -> str:
    _require_bounded_text(value, field_name)
    if not isinstance(value, str) or not value or value.strip() != value:
        raise LandingPolicyError(f"{field_name} must be a non-blank Git ref")
    if (
        _REF_FORBIDDEN_RE.search(value)
        or value.startswith("-")
        or value.endswith(".")
        or value.endswith(".lock")
        or ".." in value
        or "@{" in value
        or "//" in value
        or value in {".", "..", "HEAD"}
        or _SHA_RE.fullmatch(value) is not None
    ):
        raise LandingPolicyError(f"{field_name} is not a stable named Git ref")
    return value


def _require_optional_fence(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return _require_bounded_text(value, field_name, maximum_bytes=_MAX_FENCE_BYTES)


def _bounded_detail(value: str) -> str:
    if not isinstance(value, str):
        raise LandingPolicyError("landing detail must be a string")
    encoded = value.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_DETAIL_BYTES:
        return value
    # Refusal details may contain candidate-controlled gate names.  A detail
    # overflow must remain a typed refusal, never turn verification into an
    # exception.  Decode after truncation so a partial UTF-8 code point cannot
    # escape the bound.
    return encoded[:_MAX_DETAIL_BYTES].decode("utf-8", errors="ignore")


def _validate_certificate_bounds(certificate: ValidationCertificate) -> None:
    """Apply this seam's bounds to the older certificate dataclass."""

    for field_name in ("certificate_id", "generation_id", "target_host"):
        _require_bounded_text(getattr(certificate, field_name), field_name)
    if len(certificate.evidence_digests) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("certificate evidence exceeds the bounded count")
    if len(certificate.blocking_gate_names) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("certificate blocking gates exceed the bounded count")
    for digest in certificate.evidence_digests:
        _require_digest(digest, "certificate evidence digest")
    for gate_name in certificate.blocking_gate_names:
        _require_bounded_text(gate_name, "certificate blocking gate name")


def _validate_evidence_bounds(evidence: tuple[GateEvidence, ...]) -> None:
    """Bound legacy evidence fields before certificate verification consumes them."""

    if len(evidence) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("landing evidence exceeds the bounded count")
    total_bytes = 0
    scalar_fields = (
        "evidence_id",
        "gate_name",
        "target_host",
        "stdout_tail",
        "stderr_tail",
        "detail",
    )
    optional_scalar_fields = ("generation_id", "job_id", "baseline_tree_sha")
    sequence_fields = (
        "dependency_job_ids",
        "failure_ids",
        "pre_existing_failure_ids",
        "fixed_failure_ids",
        "log_refs",
        "artifact_refs",
    )
    for item in evidence:
        for field_name in scalar_fields:
            _require_bounded_text(
                getattr(item, field_name),
                field_name,
                allow_empty=field_name in {"stdout_tail", "stderr_tail", "detail"},
            )
        for field_name in optional_scalar_fields:
            value = getattr(item, field_name)
            if value is not None:
                _require_bounded_text(value, field_name)
        for field_name in sequence_fields:
            values = getattr(item, field_name)
            if len(values) > _MAX_EVIDENCE_SEQUENCE_ITEMS:
                raise LandingPolicyError(
                    f"{field_name} exceeds the bounded evidence count"
                )
            for value in values:
                _require_bounded_text(value, f"{field_name} entry")
        total_bytes += len(canonical_json(item.canonical_payload()).encode("utf-8"))
        if total_bytes > _MAX_EVIDENCE_TOTAL_BYTES:
            raise LandingPolicyError("landing evidence exceeds the bounded size")


def _validate_generation_bounds(generation: Generation) -> None:
    """Bound the generation fields this pure seam returns or compares."""

    _require_bounded_text(generation.generation_id, "generation_id")
    if len(generation.validation_evidence_ids) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError(
            "generation validation evidence exceeds the bounded count"
        )
    for evidence_id in generation.validation_evidence_ids:
        _require_bounded_text(evidence_id, "generation validation evidence ID")


@dataclass(frozen=True, slots=True)
class CanonicalCheckoutState:
    """The trusted observation made while the canonical lease is held.

    The verifier never acquires this lease itself.  ``mutation_lease_held`` is
    proof supplied by the eventual controller that the cleanliness observation
    was made inside the repository-manager canonical mutation guard.
    """

    mutation_lease_held: bool
    clean: bool

    def __post_init__(self) -> None:
        _require_bool(self.mutation_lease_held, "mutation_lease_held")
        _require_bool(self.clean, "clean")


@dataclass(frozen=True, slots=True)
class TargetOccupancyState:
    """Bounded occupancy proof for the requested target branch.

    Paths and worktree contents are intentionally not part of this policy
    input.  The controller may retain those private diagnostics, but policy
    needs only the count of *other* worktrees that hold the target branch.
    """

    other_worktree_count: int

    def __post_init__(self) -> None:
        if type(self.other_worktree_count) is not int:
            raise LandingPolicyError("other_worktree_count must be an integer")
        if not 0 <= self.other_worktree_count <= _MAX_TARGET_HOLDERS:
            raise LandingPolicyError(
                f"other_worktree_count must be between 0 and {_MAX_TARGET_HOLDERS}"
            )


@dataclass(frozen=True, slots=True)
class LandingVerificationRequest:
    """Closed value input to :func:`verify_landing`.

    ``observed_target_sha`` and the canonical/occupancy snapshots must be
    collected immediately before the eventual update.  This pure checkpoint
    does not claim that observation and update are atomic; the later controller
    supplies that guarantee with its canonical lease and target CAS.
    """

    repository: RepositoryIdentity
    target_branch: str
    target: TargetPolicy
    expected_base_sha: str
    observed_target_sha: str
    expected_landing_fence: str | None
    observed_landing_fence: str | None
    generation: Generation | None
    certificate: ValidationCertificate | None
    canonical: CanonicalCheckoutState
    target_occupancy: TargetOccupancyState
    evidence: tuple[GateEvidence, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.repository, RepositoryIdentity):
            raise LandingPolicyError("repository must be RepositoryIdentity")
        _require_ref(self.target_branch, "target_branch")
        if not isinstance(self.target, TargetPolicy):
            raise LandingPolicyError("target must be TargetPolicy")
        _require_sha(self.expected_base_sha, "expected_base_sha")
        _require_sha(self.observed_target_sha, "observed_target_sha")
        _require_optional_fence(self.expected_landing_fence, "expected_landing_fence")
        _require_optional_fence(self.observed_landing_fence, "observed_landing_fence")
        if self.generation is not None and not isinstance(self.generation, Generation):
            raise LandingPolicyError("generation must be Generation or None")
        if self.generation is not None:
            _validate_generation_bounds(self.generation)
        if self.certificate is not None and not isinstance(
            self.certificate, ValidationCertificate
        ):
            raise LandingPolicyError(
                "certificate must be ValidationCertificate or None"
            )
        if self.certificate is not None:
            _validate_certificate_bounds(self.certificate)
        if not isinstance(self.canonical, CanonicalCheckoutState):
            raise LandingPolicyError("canonical must be CanonicalCheckoutState")
        if not isinstance(self.target_occupancy, TargetOccupancyState):
            raise LandingPolicyError("target_occupancy must be TargetOccupancyState")
        if type(self.evidence) is not tuple:
            raise LandingPolicyError("evidence must be an immutable tuple")
        if not all(isinstance(item, GateEvidence) for item in self.evidence):
            raise LandingPolicyError("evidence entries must be GateEvidence")
        _validate_evidence_bounds(self.evidence)


@dataclass(frozen=True, slots=True)
class LandingVerificationResult:
    """Typed accepted/refused result with no ambiguous partial success."""

    accepted: bool
    refusal_code: LandingRefusalCode | None = None
    detail: str = ""
    generation_id: str | None = None
    synthetic_commit_sha: str | None = None
    tree_sha: str | None = None
    certificate_digest: str | None = None
    landing_fence: str | None = None

    def __post_init__(self) -> None:
        _require_bool(self.accepted, "accepted")
        if self.accepted:
            if self.refusal_code is not None:
                raise LandingPolicyError("accepted result cannot carry a refusal")
            if not self.generation_id:
                raise LandingPolicyError("accepted result requires generation_id")
            if not self.synthetic_commit_sha:
                raise LandingPolicyError(
                    "accepted result requires synthetic_commit_sha"
                )
            if not self.tree_sha:
                raise LandingPolicyError("accepted result requires tree_sha")
            if not self.certificate_digest:
                raise LandingPolicyError("accepted result requires certificate_digest")
            if not self.landing_fence:
                raise LandingPolicyError("accepted result requires landing_fence")
            _require_bounded_text(self.generation_id, "generation_id")
            _require_sha(self.synthetic_commit_sha, "synthetic_commit_sha")
            _require_sha(self.tree_sha, "tree_sha")
            _require_digest(self.certificate_digest, "certificate_digest")
            _require_optional_fence(self.landing_fence, "landing_fence")
        else:
            if not isinstance(self.refusal_code, LandingRefusalCode):
                raise LandingPolicyError(
                    "refused result requires one LandingRefusalCode"
                )
            if any(
                value is not None
                for value in (
                    self.generation_id,
                    self.synthetic_commit_sha,
                    self.tree_sha,
                    self.certificate_digest,
                    self.landing_fence,
                )
            ):
                raise LandingPolicyError(
                    "refused result cannot carry accepted landing identity"
                )
        object.__setattr__(self, "detail", _bounded_detail(self.detail))

    @property
    def refused(self) -> bool:
        """Whether the policy refused the landing."""

        return not self.accepted

    @property
    def code(self) -> LandingRefusalCode | None:
        """Short alias used by adapters that expose a ``code`` field."""

        return self.refusal_code


def _refuse(code: LandingRefusalCode, detail: str) -> LandingVerificationResult:
    return LandingVerificationResult(
        accepted=False,
        refusal_code=code,
        detail=_bounded_detail(detail),
    )


def verify_landing(request: LandingVerificationRequest) -> LandingVerificationResult:
    """Verify one certified, fenced landing without performing any effect.

    Checks run in a deterministic order so a retry receives the same stable
    refusal for the same snapshot.  The order starts with immutable authority
    (generation/certificate), then identity/base/fence, then the local safety
    observations.  No result from this function authorizes a caller to skip a
    fresh target CAS or canonical lease check.
    """

    if not isinstance(request, LandingVerificationRequest):
        raise LandingPolicyError("request must be LandingVerificationRequest")

    generation = request.generation
    if generation is None:
        return _refuse(
            LandingRefusalCode.GENERATION_REQUIRED,
            "a sealed landing generation is required",
        )
    certificate = request.certificate
    if certificate is None:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_REQUIRED,
            "a certification certificate is required",
        )
    if generation.state is not GenerationState.CERTIFIED:
        return _refuse(
            LandingRefusalCode.GENERATION_NOT_CERTIFIED,
            f"generation {generation.generation_id!r} is {generation.state.value!r}, "
            "not certified",
        )
    if generation.synthetic_commit_sha is None or generation.tree_sha is None:
        return _refuse(
            LandingRefusalCode.GENERATION_INCOMPLETE,
            "certified generation must carry a synthetic commit and tree",
        )
    if certificate.generation_id != generation.generation_id:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_GENERATION_MISMATCH,
            "certificate and generation identities differ",
        )
    if certificate.tree_sha != generation.tree_sha:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_TREE_MISMATCH,
            "certificate tree does not match the certified generation tree",
        )
    if (
        certificate.gate_config_digest != generation.config_digest
        or certificate.toolchain_digest != generation.toolchain_digest
    ):
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH,
            "certificate configuration or toolchain differs from the generation",
        )
    if generation.validation_evidence_ids:
        supplied_evidence_ids = tuple(item.evidence_id for item in request.evidence)
        if len(set(supplied_evidence_ids)) != len(supplied_evidence_ids) or set(
            supplied_evidence_ids
        ) != set(generation.validation_evidence_ids):
            return _refuse(
                LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH,
                "certificate evidence identities differ from the generation",
            )
    certificate_check = verify_certificate(certificate, request.evidence)
    if not certificate_check.valid:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INVALID,
            "certificate evidence is invalid: " + "; ".join(certificate_check.reasons),
        )
    if generation.repository != request.repository:
        return _refuse(
            LandingRefusalCode.REPOSITORY_MISMATCH,
            "generation repository does not match the requested repository",
        )
    if generation.target_branch != request.target_branch:
        return _refuse(
            LandingRefusalCode.TARGET_MISMATCH,
            "generation target branch does not match the requested target",
        )
    if generation.target != request.target:
        return _refuse(
            LandingRefusalCode.TARGET_MISMATCH,
            "generation execution target does not match the requested target",
        )
    if (
        generation.base_sha != generation.expected_landing_base_sha
        or generation.expected_landing_base_sha != request.expected_base_sha
    ):
        return _refuse(
            LandingRefusalCode.EXPECTED_BASE_MISMATCH,
            "generation expected landing base does not match the requested base",
        )
    if request.observed_target_sha != request.expected_base_sha:
        return _refuse(
            LandingRefusalCode.TARGET_MOVED,
            "target branch moved after the expected base was captured",
        )
    if request.expected_landing_fence is None or request.observed_landing_fence is None:
        return _refuse(
            LandingRefusalCode.FENCE_REQUIRED,
            "an expected and observed landing fence are required",
        )
    if (
        generation.landing_fence is not None
        and generation.landing_fence != request.expected_landing_fence
    ) or request.observed_landing_fence != request.expected_landing_fence:
        return _refuse(
            LandingRefusalCode.STALE_FENCE,
            "the observed landing fence is not the current expected fence",
        )
    if not request.canonical.mutation_lease_held:
        return _refuse(
            LandingRefusalCode.CANONICAL_LEASE_REQUIRED,
            "canonical cleanliness must be observed while holding its mutation lease",
        )
    if not request.canonical.clean:
        return _refuse(
            LandingRefusalCode.CANONICAL_DIRTY,
            "canonical checkout has uncommitted or untracked content",
        )
    if request.target_occupancy.other_worktree_count:
        return _refuse(
            LandingRefusalCode.TARGET_OCCUPIED,
            "target branch is checked out by another worktree",
        )
    return LandingVerificationResult(
        accepted=True,
        generation_id=generation.generation_id,
        synthetic_commit_sha=generation.synthetic_commit_sha,
        tree_sha=generation.tree_sha,
        certificate_digest=certificate.digest,
        landing_fence=request.expected_landing_fence,
    )


def verify_landability(
    request: LandingVerificationRequest,
) -> LandingVerificationResult:
    """Compatibility spelling for callers that name the policy landability."""

    return verify_landing(request)


__all__ = [
    "CanonicalCheckoutState",
    "LandingPolicyError",
    "LandingRefusalCode",
    "LandingVerificationRequest",
    "LandingVerificationResult",
    "TargetOccupancyState",
    "verify_landability",
    "verify_landing",
]
