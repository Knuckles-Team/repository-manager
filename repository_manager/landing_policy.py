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
import unicodedata
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, StrEnum
from typing import TypeVar, cast

from pydantic import BaseModel

from repository_manager.development import (
    CandidateVersion,
    Generation,
    GenerationState,
    RepositoryIdentity,
    TargetPolicy,
    ValidationStage,
)
from repository_manager.development.serialization import canonical_json
from repository_manager.validation import (
    EvidenceOutcome,
    GateEvidence,
    ValidationCertificate,
    ValidationFailureClass,
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
_MAX_MAPPING_KEYS = 64
_MAX_AUTHORITY_BYTES = 1024 * 1024
_UNSAFE_UNICODE_CATEGORIES = frozenset({"Cc", "Cf", "Co", "Cs", "Zl", "Zp"})
_BIDI_CONTROLS = frozenset(
    {
        "\u061c",
        "\u200e",
        "\u200f",
        "\u202a",
        "\u202b",
        "\u202c",
        "\u202d",
        "\u202e",
        "\u2066",
        "\u2067",
        "\u2068",
        "\u2069",
    }
)

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
_CANDIDATE_FIELDS = ("contract_version", "candidate_id", "version", "candidate_sha")
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

_EnumT = TypeVar("_EnumT", bound=StrEnum)


class LandingPolicyError(ValueError):
    """A landing-policy input is not one of the closed typed values."""


class LandingRefusalCode(StrEnum):
    """Stable, actionable refusal values for the certified landing seam."""

    REQUEST_INVALID = "request_invalid"
    GENERATION_REQUIRED = "generation_required"
    GENERATION_INVALID = "generation_invalid"
    GENERATION_ID_MISMATCH = "generation_id_mismatch"
    GENERATION_ANCHOR_REQUIRED = "generation_anchor_required"
    GENERATION_ANCHOR_MISMATCH = "generation_anchor_mismatch"
    CERTIFICATE_REQUIRED = "certificate_required"
    CERTIFICATE_ANCHOR_REQUIRED = "certificate_anchor_required"
    CERTIFICATE_ANCHOR_MISMATCH = "certificate_anchor_mismatch"
    CERTIFICATE_INVALID = "certificate_invalid"
    EVIDENCE_INVALID = "evidence_invalid"
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
    if any(
        unicodedata.category(char) in _UNSAFE_UNICODE_CATEGORIES
        or char in _BIDI_CONTROLS
        for char in value
    ):
        raise LandingPolicyError(f"{field_name} contains unsafe Unicode characters")
    try:
        encoded = value.encode("utf-8")
    except UnicodeError as exc:
        raise LandingPolicyError(f"{field_name} contains invalid Unicode") from exc
    if len(encoded) > maximum_bytes:
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


def _require_datetime(value: object, field_name: str) -> datetime:
    if not isinstance(value, datetime):
        raise LandingPolicyError(f"{field_name} is not a timestamp")
    return value


def _require_optional_int(value: object, field_name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int:
        raise LandingPolicyError(f"{field_name} must be an integer or None")
    return value


def _bounded_detail(value: str) -> str:
    if not isinstance(value, str):
        raise LandingPolicyError("landing detail must be a string")
    try:
        normalized = unicodedata.normalize("NFC", value)
    except (TypeError, ValueError):
        normalized = ""
    safe = "".join(
        "?"
        if unicodedata.category(char) in _UNSAFE_UNICODE_CATEGORIES
        or char in _BIDI_CONTROLS
        else char
        for char in normalized
    )
    encoded = safe.encode("utf-8", errors="replace")
    if len(encoded) <= _MAX_DETAIL_BYTES:
        return safe
    # Refusal details may contain candidate-controlled gate names.  A detail
    # overflow must remain a typed refusal, never turn verification into an
    # exception.  Decode after truncation so a partial UTF-8 code point cannot
    # escape the bound.
    return encoded[:_MAX_DETAIL_BYTES].decode("utf-8", errors="ignore")


def _validate_plain_tree(value: object, *, depth: int = 0) -> None:
    """Reject non-plain or unbounded values before any model validation."""

    if depth > 8:
        raise LandingPolicyError("authority nesting exceeds its bounded depth")
    if type(value) is dict:
        mapping = cast(dict[object, object], value)
        if len(mapping) > _MAX_MAPPING_KEYS:
            raise LandingPolicyError("authority mapping exceeds its bounded count")
        for key, item in mapping.items():
            if not isinstance(key, str):
                raise LandingPolicyError("authority mapping keys must be strings")
            _require_bounded_text(key, "authority mapping key")
            _validate_plain_tree(item, depth=depth + 1)
        return
    if type(value) is list:
        list_value = cast(list[object], value)
        if len(list_value) > _MAX_EVIDENCE_ITEMS:
            raise LandingPolicyError("authority sequence exceeds its bounded count")
        for item in list_value:
            _validate_plain_tree(item, depth=depth + 1)
        return
    if type(value) is tuple:
        tuple_value = cast(tuple[object, ...], value)
        if len(tuple_value) > _MAX_EVIDENCE_ITEMS:
            raise LandingPolicyError("authority sequence exceeds its bounded count")
        for item in tuple_value:
            _validate_plain_tree(item, depth=depth + 1)
        return
    if value is None or type(value) in {bool, int, float, str}:
        if isinstance(value, str):
            _require_bounded_text(
                value,
                "authority text",
                maximum_bytes=_MAX_AUTHORITY_BYTES,
                allow_empty=True,
            )
        return
    if isinstance(value, (datetime, Enum)):
        return
    raise LandingPolicyError("authority contains a non-plain value")


def _plain_model_mapping(value: object, field_name: str) -> dict[str, object]:
    """Take a bounded JSON-shaped snapshot of a Pydantic authority model."""

    if not isinstance(value, BaseModel):
        raise LandingPolicyError(f"{field_name} is not a typed authority model")
    try:
        # ``warnings=False`` is deliberate: callers must receive a typed
        # refusal from the shape checks below, never a disposition that depends
        # on Pydantic's serializer warning policy.
        raw = BaseModel.model_dump(
            value, mode="json", exclude_none=False, warnings=False
        )
        if type(raw) is not dict:
            raise LandingPolicyError(f"{field_name} did not produce a mapping")
        _validate_plain_tree(raw)
    except LandingPolicyError:
        raise
    except Exception as exc:
        raise LandingPolicyError(f"{field_name} could not be serialized") from exc
    return dict(raw)


def _plain_payload_mapping(value: object, field_name: str) -> dict[str, object]:
    """Take a bounded mapping from a legacy dataclass canonical payload."""

    # Do not call ``dict(value)`` on an arbitrary Mapping: a hostile Mapping
    # may be infinite or perform effects while iterating.  The exact builtin
    # type is bounded before its shallow snapshot is materialized.
    if type(value) is not dict:
        raise LandingPolicyError(f"{field_name} did not produce a mapping")
    try:
        plain_value = cast(dict[str, object], value)
        if len(plain_value) > _MAX_MAPPING_KEYS:
            raise LandingPolicyError("authority mapping exceeds its bounded count")
        raw = dict(plain_value)
        _validate_plain_tree(raw)
    except LandingPolicyError:
        raise
    except Exception as exc:
        raise LandingPolicyError(f"{field_name} is not a plain mapping") from exc
    return raw


def _require_sequence(
    value: object, field_name: str, *, maximum: int = _MAX_EVIDENCE_ITEMS
) -> tuple[object, ...]:
    if type(value) is list:
        list_value = cast(list[object], value)
        if len(list_value) > maximum:
            raise LandingPolicyError(f"{field_name} exceeds the bounded count")
        return tuple(list_value)
    if type(value) is tuple:
        tuple_value = cast(tuple[object, ...], value)
        if len(tuple_value) > maximum:
            raise LandingPolicyError(f"{field_name} exceeds the bounded count")
        return tuple(tuple_value)
    raise LandingPolicyError(f"{field_name} must be a bounded list or tuple")


def _enum_value(value: object, enum_type: type[_EnumT], field_name: str) -> _EnumT:
    if isinstance(value, enum_type):
        return value
    if type(value) is str:
        try:
            return enum_type(value)
        except ValueError as exc:
            raise LandingPolicyError(f"{field_name} is not a valid enum value") from exc
    raise LandingPolicyError(f"{field_name} is not a valid enum value")


def _model_field_values(
    value: BaseModel, field_names: tuple[str, ...], field_name: str
) -> dict[str, object]:
    """Read model state without invoking forged field descriptors or properties."""

    try:
        state = object.__getattribute__(value, "__dict__")
    except AttributeError as exc:
        raise LandingPolicyError(f"{field_name} fields are unavailable") from exc
    if type(state) is not dict:
        raise LandingPolicyError(f"{field_name} fields are unavailable")
    if any(name not in state for name in field_names):
        raise LandingPolicyError(f"{field_name} fields are incomplete")
    return {name: state[name] for name in field_names}


def _rebuild_repository(value: object) -> RepositoryIdentity:
    if type(value) is not RepositoryIdentity:
        raise LandingPolicyError("repository is not a typed authority model")
    fields = _model_field_values(value, _REPOSITORY_FIELDS, "repository")
    roots = _require_sequence(fields["configured_roots"], "configured roots")
    for root in roots:
        _require_bounded_text(root, "configured root", maximum_bytes=4096)
    raw = _plain_model_mapping(value, "repository")
    try:
        repository = RepositoryIdentity.model_validate(raw)
    except Exception as exc:
        raise LandingPolicyError("repository authority is invalid") from exc
    repository_fields = _model_field_values(
        repository, _REPOSITORY_FIELDS, "repository"
    )
    _require_bounded_text(repository_fields["repository_id"], "repository_id")
    _require_bounded_text(
        repository_fields["canonical_path"], "canonical_path", maximum_bytes=4096
    )
    configured_roots = _require_sequence(
        repository_fields["configured_roots"], "configured roots"
    )
    if len(configured_roots) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("configured roots exceed the bounded count")
    for root in configured_roots:
        _require_bounded_text(root, "configured root", maximum_bytes=4096)
    if repository_fields["origin"] is not None:
        _require_bounded_text(
            repository_fields["origin"], "repository origin", maximum_bytes=4096
        )
    return repository


def _rebuild_target(value: object) -> TargetPolicy:
    if type(value) is not TargetPolicy:
        raise LandingPolicyError("target is not a typed authority model")
    fields = _model_field_values(value, _TARGET_FIELDS, "target")
    labels = _require_sequence(fields["capability_labels"], "target capabilities")
    for label in labels:
        _require_bounded_text(label, "target capability")
    raw = _plain_model_mapping(value, "target")
    try:
        target = TargetPolicy.model_validate(raw)
    except Exception as exc:
        raise LandingPolicyError("target authority is invalid") from exc
    target_fields = _model_field_values(target, _TARGET_FIELDS, "target")
    if target_fields["alias"] is not None:
        _require_bounded_text(target_fields["alias"], "target alias")
    target_labels = _require_sequence(
        target_fields["capability_labels"], "target capabilities"
    )
    if len(target_labels) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("target capabilities exceed the bounded count")
    for label in target_labels:
        _require_bounded_text(label, "target capability")
    return target


def _rebuild_candidate(value: object) -> CandidateVersion:
    if isinstance(value, BaseModel) and type(value) is not CandidateVersion:
        raise LandingPolicyError("candidate version is not a typed authority model")
    raw = (
        _plain_model_mapping(value, "candidate version")
        if isinstance(value, BaseModel)
        else _plain_payload_mapping(value, "candidate version")
    )
    try:
        candidate = CandidateVersion.model_validate(raw)
    except Exception as exc:
        raise LandingPolicyError("candidate version authority is invalid") from exc
    candidate_fields = _model_field_values(
        candidate, _CANDIDATE_FIELDS, "candidate version"
    )
    _require_bounded_text(candidate_fields["candidate_id"], "candidate_id")
    if type(candidate_fields["version"]) is not int:
        raise LandingPolicyError("candidate version number is invalid")
    _require_sha(candidate_fields["candidate_sha"], "candidate_sha")
    return candidate


def _rebuild_generation(value: object) -> Generation:
    if type(value) is not Generation:
        raise LandingPolicyError("generation is not a typed authority model")
    fields = _model_field_values(value, _GENERATION_FIELDS, "generation")
    repository = fields["repository"]
    target = fields["target"]
    if type(repository) is not RepositoryIdentity:
        raise LandingPolicyError("generation repository is not a typed authority model")
    if type(target) is not TargetPolicy:
        raise LandingPolicyError("generation target is not a typed authority model")
    sequences = (
        "candidate_versions",
        "validation_evidence_ids",
        "build_artifact_refs",
        "bisection_lineage",
    )
    for field_name in sequences:
        _require_sequence(fields[field_name], field_name)
    candidates = _require_sequence(fields["candidate_versions"], "candidate_versions")
    _rebuild_repository(repository)
    _rebuild_target(target)
    for candidate in candidates:
        _rebuild_candidate(candidate)
    raw = _plain_model_mapping(value, "generation")
    raw_candidates = _require_sequence(
        raw.get("candidate_versions"), "candidate_versions"
    )
    raw["repository"] = _plain_payload_mapping(raw.get("repository"), "repository")
    raw["target"] = _plain_payload_mapping(raw.get("target"), "target")
    raw["candidate_versions"] = [
        _plain_payload_mapping(item, "candidate version") for item in raw_candidates
    ]
    try:
        generation = Generation.model_validate(raw)
    except Exception as exc:
        raise LandingPolicyError("generation authority is invalid") from exc
    _validate_generation_content(generation)
    return generation


def _validate_generation_content(generation: Generation) -> None:
    fields = _model_field_values(generation, _GENERATION_FIELDS, "generation")
    _require_bounded_text(fields["generation_id"], "generation_id")
    _rebuild_repository(fields["repository"])
    _require_ref(fields["target_branch"], "target_branch")
    _rebuild_target(fields["target"])
    _require_sha(fields["base_sha"], "generation base_sha")
    _require_sha(fields["expected_landing_base_sha"], "expected_landing_base_sha")
    _require_digest(fields["config_digest"], "generation config_digest")
    _require_digest(fields["toolchain_digest"], "generation toolchain_digest")
    if not isinstance(fields["state"], GenerationState):
        raise LandingPolicyError("generation state is invalid")
    if fields["sealed_at"] is not None and not isinstance(
        fields["sealed_at"], datetime
    ):
        raise LandingPolicyError("generation sealed_at is invalid")
    if fields["synthetic_commit_sha"] is not None:
        _require_sha(fields["synthetic_commit_sha"], "synthetic_commit_sha")
    if fields["tree_sha"] is not None:
        _require_sha(fields["tree_sha"], "tree_sha")
    candidates = _require_sequence(fields["candidate_versions"], "candidate_versions")
    for candidate in candidates:
        _rebuild_candidate(candidate)
    evidence_ids = _require_sequence(
        fields["validation_evidence_ids"], "generation validation evidence"
    )
    if len(evidence_ids) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError(
            "generation validation evidence exceeds the bounded count"
        )
    for evidence_id in evidence_ids:
        _require_bounded_text(evidence_id, "generation validation evidence ID")
    if fields["landing_fence"] is not None:
        _require_optional_fence(fields["landing_fence"], "generation landing_fence")
    _require_bounded_text(fields["reason"], "generation reason", allow_empty=True)


_CERTIFICATE_FIELDS = frozenset(
    {
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
    }
)


def _rebuild_certificate(value: object) -> ValidationCertificate:
    if not isinstance(value, ValidationCertificate):
        raise LandingPolicyError("certificate is not a typed authority value")
    try:
        raw = _plain_payload_mapping(value.canonical_payload(), "certificate")
    except LandingPolicyError:
        raise
    except Exception as exc:
        raise LandingPolicyError(
            "certificate authority could not be serialized"
        ) from exc
    if set(raw) != _CERTIFICATE_FIELDS:
        raise LandingPolicyError("certificate authority fields are not exact")
    evidence_digests = _require_sequence(
        raw.get("evidence_digests"), "certificate evidence digests"
    )
    blocking_names = _require_sequence(
        raw.get("blocking_gate_names"), "certificate blocking gates"
    )
    certificate_id = _require_bounded_text(raw.get("certificate_id"), "certificate_id")
    generation_id = _require_bounded_text(raw.get("generation_id"), "generation_id")
    tree_sha = _require_sha(raw.get("tree_sha"), "certificate tree_sha")
    gate_config_digest = _require_digest(
        raw.get("gate_config_digest"), "gate_config_digest"
    )
    toolchain_digest = _require_digest(raw.get("toolchain_digest"), "toolchain_digest")
    target_host = _require_bounded_text(raw.get("target_host"), "target_host")
    resource_digest = _require_digest(raw.get("resource_digest"), "resource_digest")
    evidence_digest_values = tuple(
        _require_digest(item, "certificate evidence digest")
        for item in evidence_digests
    )
    blocking_gate_values = tuple(
        _require_bounded_text(item, "certificate blocking gate name")
        for item in blocking_names
    )
    issued_at = _require_datetime(raw.get("issued_at"), "issued_at")
    profile_digest = _require_digest(raw.get("profile_digest"), "profile_digest")
    try:
        certificate = ValidationCertificate(
            certificate_id=certificate_id,
            generation_id=generation_id,
            tree_sha=tree_sha,
            gate_config_digest=gate_config_digest,
            toolchain_digest=toolchain_digest,
            target_host=target_host,
            resource_digest=resource_digest,
            evidence_digests=evidence_digest_values,
            blocking_gate_names=blocking_gate_values,
            issued_at=issued_at,
            profile_digest=profile_digest,
        )
    except Exception as exc:
        raise LandingPolicyError("certificate authority is invalid") from exc
    _validate_certificate_content(certificate)
    return certificate


def _validate_certificate_content(certificate: ValidationCertificate) -> None:
    for field_name in ("certificate_id", "generation_id", "target_host"):
        _require_bounded_text(getattr(certificate, field_name), field_name)
    _require_sha(certificate.tree_sha, "certificate tree_sha")
    for field_name in (
        "gate_config_digest",
        "toolchain_digest",
        "resource_digest",
        "profile_digest",
    ):
        _require_digest(getattr(certificate, field_name), field_name)
    if not isinstance(certificate.issued_at, datetime):
        raise LandingPolicyError("certificate issued_at is invalid")
    if len(certificate.evidence_digests) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("certificate evidence exceeds the bounded count")
    if len(certificate.blocking_gate_names) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("certificate blocking gates exceed the bounded count")
    for digest in certificate.evidence_digests:
        _require_digest(digest, "certificate evidence digest")
    for gate_name in certificate.blocking_gate_names:
        _require_bounded_text(gate_name, "certificate blocking gate name")


_EVIDENCE_FIELDS = frozenset(
    {
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
    }
)
_EVIDENCE_SEQUENCE_FIELDS = (
    "dependency_job_ids",
    "failure_ids",
    "pre_existing_failure_ids",
    "fixed_failure_ids",
    "log_refs",
    "artifact_refs",
)


def _rebuild_evidence(value: object) -> GateEvidence:
    if not isinstance(value, GateEvidence):
        raise LandingPolicyError("evidence is not a typed value")
    try:
        raw = _plain_payload_mapping(value.canonical_payload(), "evidence")
    except LandingPolicyError:
        raise
    except Exception as exc:
        raise LandingPolicyError("evidence could not be serialized") from exc
    if set(raw) != _EVIDENCE_FIELDS:
        raise LandingPolicyError("evidence fields are not exact")
    evidence_id = _require_bounded_text(raw.get("evidence_id"), "evidence_id")
    gate_name = _require_bounded_text(raw.get("gate_name"), "gate_name")
    stage = _enum_value(raw.get("stage"), ValidationStage, "evidence stage")
    tree_sha = _require_sha(raw.get("tree_sha"), "evidence tree_sha")
    generation_id_value = raw.get("generation_id")
    generation_id = (
        None
        if generation_id_value is None
        else _require_bounded_text(generation_id_value, "generation_id")
    )
    gate_config_digest = _require_digest(
        raw.get("gate_config_digest"), "gate_config_digest"
    )
    command_digest = _require_digest(raw.get("command_digest"), "command_digest")
    target_host = _require_bounded_text(raw.get("target_host"), "target_host")
    toolchain_digest = _require_digest(raw.get("toolchain_digest"), "toolchain_digest")
    resource_digest = _require_digest(raw.get("resource_digest"), "resource_digest")
    profile_digest = _require_digest(raw.get("profile_digest"), "profile_digest")
    started_at = _require_datetime(raw.get("started_at"), "started_at")
    finished_at = _require_datetime(raw.get("finished_at"), "finished_at")
    outcome = _enum_value(raw.get("outcome"), EvidenceOutcome, "evidence outcome")
    failure_class_value = raw.get("failure_class")
    failure_class = (
        None
        if failure_class_value is None
        else _enum_value(
            failure_class_value,
            ValidationFailureClass,
            "evidence failure class",
        )
    )
    sequence_values: dict[str, tuple[str, ...]] = {}
    for field_name in _EVIDENCE_SEQUENCE_FIELDS:
        values = _require_sequence(
            raw.get(field_name),
            field_name,
            maximum=_MAX_EVIDENCE_SEQUENCE_ITEMS,
        )
        sequence_values[field_name] = tuple(
            _require_bounded_text(item, f"{field_name} entry") for item in values
        )
    stdout_tail = _require_bounded_text(
        raw.get("stdout_tail"),
        "stdout_tail",
        maximum_bytes=_MAX_EVIDENCE_FIELD_BYTES,
        allow_empty=True,
    )
    stderr_tail = _require_bounded_text(
        raw.get("stderr_tail"),
        "stderr_tail",
        maximum_bytes=_MAX_EVIDENCE_FIELD_BYTES,
        allow_empty=True,
    )
    detail = _require_bounded_text(
        raw.get("detail"),
        "detail",
        maximum_bytes=_MAX_EVIDENCE_FIELD_BYTES,
        allow_empty=True,
    )
    job_id_value = raw.get("job_id")
    job_id = (
        None if job_id_value is None else _require_bounded_text(job_id_value, "job_id")
    )
    baseline_tree_value = raw.get("baseline_tree_sha")
    baseline_tree_sha = (
        None
        if baseline_tree_value is None
        else _require_sha(baseline_tree_value, "baseline_tree_sha")
    )
    baseline_readable_value = raw.get("baseline_readable")
    baseline_readable = (
        None
        if baseline_readable_value is None
        else _require_bool(baseline_readable_value, "baseline_readable")
    )
    differential = _require_bool(raw.get("differential"), "differential")
    snapshot_gate_deferred = _require_bool(
        raw.get("snapshot_gate_deferred"), "snapshot_gate_deferred"
    )
    snapshot_gate_replayed = _require_bool(
        raw.get("snapshot_gate_replayed"), "snapshot_gate_replayed"
    )
    exit_code = _require_optional_int(raw.get("exit_code"), "exit_code")
    try:
        evidence = GateEvidence(
            evidence_id=evidence_id,
            gate_name=gate_name,
            stage=stage,
            tree_sha=tree_sha,
            generation_id=generation_id,
            gate_config_digest=gate_config_digest,
            command_digest=command_digest,
            target_host=target_host,
            toolchain_digest=toolchain_digest,
            resource_digest=resource_digest,
            profile_digest=profile_digest,
            started_at=started_at,
            finished_at=finished_at,
            outcome=outcome,
            failure_class=failure_class,
            job_id=job_id,
            dependency_job_ids=sequence_values["dependency_job_ids"],
            baseline_tree_sha=baseline_tree_sha,
            baseline_readable=baseline_readable,
            differential=differential,
            failure_ids=sequence_values["failure_ids"],
            pre_existing_failure_ids=sequence_values["pre_existing_failure_ids"],
            fixed_failure_ids=sequence_values["fixed_failure_ids"],
            log_refs=sequence_values["log_refs"],
            artifact_refs=sequence_values["artifact_refs"],
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            exit_code=exit_code,
            snapshot_gate_deferred=snapshot_gate_deferred,
            snapshot_gate_replayed=snapshot_gate_replayed,
            detail=detail,
        )
    except Exception as exc:
        raise LandingPolicyError("evidence authority is invalid") from exc
    try:
        total_bytes = len(canonical_json(evidence.canonical_payload()).encode("utf-8"))
    except Exception as exc:
        raise LandingPolicyError("evidence canonical form is invalid") from exc
    if total_bytes > _MAX_EVIDENCE_TOTAL_BYTES:
        raise LandingPolicyError("landing evidence exceeds the bounded size")
    return evidence


def _validate_evidence_bounds(evidence: tuple[GateEvidence, ...]) -> None:
    if len(evidence) > _MAX_EVIDENCE_ITEMS:
        raise LandingPolicyError("landing evidence exceeds the bounded count")
    for item in evidence:
        if not isinstance(item, GateEvidence):
            raise LandingPolicyError("evidence entries must be GateEvidence")
        # Preserve the constructor's useful count/size errors while allowing a
        # malformed model_copy to reach verify_landing and become a typed refusal.
        try:
            _rebuild_evidence(item)
        except LandingPolicyError as exc:
            message = str(exc)
            if "bounded count" in message or "bounded size" in message:
                raise


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
    # These are trusted anchors read from the durable certified-generation
    # authority by the later controller.  They must never be recomputed from
    # the request-local generation, certificate, or evidence.
    expected_certificate_digest: str | None = None
    expected_generation_id: str | None = None
    expected_synthetic_commit_sha: str | None = None

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
        if self.expected_certificate_digest is not None:
            _require_digest(
                self.expected_certificate_digest, "expected_certificate_digest"
            )
        if self.expected_generation_id is not None:
            _require_bounded_text(self.expected_generation_id, "expected_generation_id")
        if self.expected_synthetic_commit_sha is not None:
            _require_sha(
                self.expected_synthetic_commit_sha,
                "expected_synthetic_commit_sha",
            )
        if self.generation is not None and not isinstance(self.generation, Generation):
            raise LandingPolicyError("generation must be Generation or None")
        if self.certificate is not None and not isinstance(
            self.certificate, ValidationCertificate
        ):
            raise LandingPolicyError(
                "certificate must be ValidationCertificate or None"
            )
        if not isinstance(self.canonical, CanonicalCheckoutState):
            raise LandingPolicyError("canonical must be CanonicalCheckoutState")
        if not isinstance(self.target_occupancy, TargetOccupancyState):
            raise LandingPolicyError("target_occupancy must be TargetOccupancyState")
        if type(self.evidence) is not tuple:
            raise LandingPolicyError("evidence must be an immutable tuple")
        if not all(isinstance(item, GateEvidence) for item in self.evidence):
            raise LandingPolicyError("evidence entries must be GateEvidence")
        _validate_evidence_bounds(self.evidence)
        # Snapshot every valid authority value at the boundary.  A caller can
        # otherwise retain the mutable list supplied through Pydantic's
        # model_copy(update=...) escape hatch and mutate it after construction.
        # Invalid values remain available to verify_landing, which turns them
        # into a stable typed refusal rather than leaking an AttributeError.
        for field_name, rebuild in (
            ("repository", _rebuild_repository),
            ("target", _rebuild_target),
            ("generation", _rebuild_generation),
            ("certificate", _rebuild_certificate),
        ):
            value = getattr(self, field_name)
            if value is None:
                continue
            try:
                object.__setattr__(self, field_name, rebuild(value))
            except LandingPolicyError:
                pass
        rebuilt_evidence: list[GateEvidence] = []
        for item in self.evidence:
            try:
                rebuilt_evidence.append(_rebuild_evidence(item))
            except LandingPolicyError:
                rebuilt_evidence = []
                break
        if rebuilt_evidence:
            object.__setattr__(self, "evidence", tuple(rebuilt_evidence))


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
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "landing request is not a closed typed value",
        )

    # Keep missing-authority priority stable before inspecting any mutable or
    # potentially malformed model_copy payload.
    try:
        request_generation = request.generation
        request_certificate = request.certificate
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "landing request authority fields are unavailable",
        )
    if request_generation is None:
        return _refuse(
            LandingRefusalCode.GENERATION_REQUIRED,
            "a sealed landing generation is required",
        )
    if request_certificate is None:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_REQUIRED,
            "a certification certificate is required",
        )

    try:
        repository = _rebuild_repository(request.repository)
        target = _rebuild_target(request.target)
        target_branch = _require_ref(request.target_branch, "target_branch")
        expected_base_sha = _require_sha(request.expected_base_sha, "expected_base_sha")
        observed_target_sha = _require_sha(
            request.observed_target_sha, "observed_target_sha"
        )
        expected_fence = _require_optional_fence(
            request.expected_landing_fence, "expected_landing_fence"
        )
        observed_fence = _require_optional_fence(
            request.observed_landing_fence, "observed_landing_fence"
        )
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "landing request identity or observation is invalid",
        )

    try:
        generation = _rebuild_generation(request_generation)
    except Exception:
        return _refuse(
            LandingRefusalCode.GENERATION_INVALID,
            "generation authority could not be rebuilt",
        )
    try:
        certificate = _rebuild_certificate(request_certificate)
    except Exception:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INVALID,
            "certificate authority could not be rebuilt",
        )
    rebuilt_evidence: list[GateEvidence] = []
    try:
        if type(request.evidence) is not tuple:
            raise LandingPolicyError("evidence must be an immutable tuple")
        for item in request.evidence:
            rebuilt_evidence.append(_rebuild_evidence(item))
        evidence = tuple(rebuilt_evidence)
        _validate_evidence_bounds(evidence)
    except Exception:
        return _refuse(
            LandingRefusalCode.EVIDENCE_INVALID,
            "validation evidence could not be rebuilt",
        )

    try:
        expected_generation_id = Generation.derive_id(
            repository_id=generation.repository.repository_id,
            target_branch=generation.target_branch,
            base_sha=generation.base_sha,
            candidate_versions=tuple(generation.candidate_versions),
            config_digest=generation.config_digest,
            toolchain_digest=generation.toolchain_digest,
        )
    except Exception:
        return _refuse(
            LandingRefusalCode.GENERATION_INVALID,
            "generation identity could not be derived",
        )
    if generation.generation_id != expected_generation_id:
        return _refuse(
            LandingRefusalCode.GENERATION_ID_MISMATCH,
            "generation identity does not match immutable membership",
        )
    if request.expected_generation_id is None:
        return _refuse(
            LandingRefusalCode.GENERATION_ANCHOR_REQUIRED,
            "durable generation identity anchor is required",
        )
    try:
        expected_generation_anchor = _require_bounded_text(
            request.expected_generation_id, "expected_generation_id"
        )
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "generation identity anchor is invalid",
        )
    if expected_generation_anchor != generation.generation_id:
        return _refuse(
            LandingRefusalCode.GENERATION_ANCHOR_MISMATCH,
            "generation identity anchor does not match durable authority",
        )

    if generation.state is not GenerationState.CERTIFIED:
        return _refuse(
            LandingRefusalCode.GENERATION_NOT_CERTIFIED,
            "generation is not certified for landing",
        )
    if generation.synthetic_commit_sha is None or generation.tree_sha is None:
        return _refuse(
            LandingRefusalCode.GENERATION_INCOMPLETE,
            "certified generation is missing its immutable commit or tree",
        )
    if request.expected_synthetic_commit_sha is None:
        return _refuse(
            LandingRefusalCode.GENERATION_ANCHOR_REQUIRED,
            "durable synthetic commit anchor is required",
        )
    try:
        expected_synthetic_commit_sha = _require_sha(
            request.expected_synthetic_commit_sha,
            "expected_synthetic_commit_sha",
        )
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "synthetic commit anchor is invalid",
        )
    if expected_synthetic_commit_sha != generation.synthetic_commit_sha:
        return _refuse(
            LandingRefusalCode.GENERATION_ANCHOR_MISMATCH,
            "synthetic commit anchor does not match durable authority",
        )
    if generation.landing_fence is None:
        return _refuse(
            LandingRefusalCode.FENCE_REQUIRED,
            "certified generation is missing its landing fence",
        )
    if certificate.generation_id != generation.generation_id:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_GENERATION_MISMATCH,
            "certificate and generation identities differ",
        )
    if certificate.tree_sha != generation.tree_sha:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_TREE_MISMATCH,
            "certificate and generation trees differ",
        )
    if (
        certificate.gate_config_digest != generation.config_digest
        or certificate.toolchain_digest != generation.toolchain_digest
    ):
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH,
            "certificate inputs differ from generation authority",
        )
    supplied_evidence_ids = tuple(item.evidence_id for item in evidence)
    expected_evidence_ids = tuple(generation.validation_evidence_ids)
    if (
        len(supplied_evidence_ids) != len(set(supplied_evidence_ids))
        or tuple(sorted(supplied_evidence_ids)) != expected_evidence_ids
    ):
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INPUT_MISMATCH,
            "generation evidence identities differ from rebuilt evidence",
        )
    try:
        evidence_digests = tuple(sorted(item.digest for item in evidence))
    except Exception:
        return _refuse(
            LandingRefusalCode.EVIDENCE_INVALID,
            "validation evidence digest could not be derived",
        )
    if tuple(certificate.evidence_digests) != evidence_digests:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INVALID,
            "certificate evidence content differs from rebuilt evidence",
        )
    try:
        certificate_check = verify_certificate(certificate, evidence)
        certificate_digest = certificate.digest
        if not isinstance(certificate_check.valid, bool):
            raise LandingPolicyError("certificate verification result is invalid")
    except Exception:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INVALID,
            "certificate verification failed safely",
        )
    if not certificate_check.valid:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_INVALID,
            "certificate evidence is not valid for landing",
        )
    if request.expected_certificate_digest is None:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_ANCHOR_REQUIRED,
            "durable certificate content anchor is required",
        )
    try:
        expected_certificate_digest = _require_digest(
            request.expected_certificate_digest, "expected_certificate_digest"
        )
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "certificate content anchor is invalid",
        )
    if expected_certificate_digest != certificate_digest:
        return _refuse(
            LandingRefusalCode.CERTIFICATE_ANCHOR_MISMATCH,
            "certificate content anchor does not match durable authority",
        )
    if generation.repository != repository:
        return _refuse(
            LandingRefusalCode.REPOSITORY_MISMATCH,
            "generation repository does not match the requested repository",
        )
    if generation.target_branch != target_branch or generation.target != target:
        return _refuse(
            LandingRefusalCode.TARGET_MISMATCH,
            "generation target does not match the requested target",
        )
    if (
        generation.base_sha != generation.expected_landing_base_sha
        or generation.expected_landing_base_sha != expected_base_sha
    ):
        return _refuse(
            LandingRefusalCode.EXPECTED_BASE_MISMATCH,
            "generation expected landing base does not match the request",
        )
    if observed_target_sha != expected_base_sha:
        return _refuse(
            LandingRefusalCode.TARGET_MOVED,
            "target branch moved after the expected base was captured",
        )
    if expected_fence is None or observed_fence is None:
        return _refuse(
            LandingRefusalCode.FENCE_REQUIRED,
            "an expected and observed landing fence are required",
        )
    if generation.landing_fence != expected_fence or observed_fence != expected_fence:
        return _refuse(
            LandingRefusalCode.STALE_FENCE,
            "the observed landing fence is not the current expected fence",
        )
    try:
        lease_held = _require_bool(
            request.canonical.mutation_lease_held, "mutation_lease_held"
        )
        clean = _require_bool(request.canonical.clean, "clean")
        occupancy = request.target_occupancy.other_worktree_count
        if type(occupancy) is not int or not 0 <= occupancy <= _MAX_TARGET_HOLDERS:
            raise LandingPolicyError("occupancy is invalid")
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "canonical or occupancy observation is invalid",
        )
    if not lease_held:
        return _refuse(
            LandingRefusalCode.CANONICAL_LEASE_REQUIRED,
            "canonical cleanliness requires its mutation lease",
        )
    if not clean:
        return _refuse(
            LandingRefusalCode.CANONICAL_DIRTY,
            "canonical checkout is not clean",
        )
    if occupancy:
        return _refuse(
            LandingRefusalCode.TARGET_OCCUPIED,
            "target branch is occupied by another worktree",
        )
    try:
        return LandingVerificationResult(
            accepted=True,
            generation_id=generation.generation_id,
            synthetic_commit_sha=generation.synthetic_commit_sha,
            tree_sha=generation.tree_sha,
            certificate_digest=certificate_digest,
            landing_fence=expected_fence,
        )
    except Exception:
        return _refuse(
            LandingRefusalCode.REQUEST_INVALID,
            "landing result could not be constructed",
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
