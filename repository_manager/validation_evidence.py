"""Immutable validation evidence, differential verdicts, and certificates.

The merge queue's historical ``Check`` object is intentionally a short-lived
diagnostic.  RMDD-11 adds a durable-shaped evidence record that names every
input needed to decide whether a result is reusable: immutable tree and
generation, gate/config/command digests, toolchain, host/resource identity,
baseline, bounded output, and artifact references.  A certification
certificate is only a verified aggregation of those records; stage-0 feedback
and stage-1 integration results can never be promoted by copying a status bit.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from threading import RLock

from repository_manager.development import ValidationStage
from repository_manager.development.serialization import canonical_digest
from repository_manager.validation_policy import BaselineMode


class EvidenceError(ValueError):
    """Malformed, inconsistent, or unverifiable validation evidence."""


class EvidenceOutcome(StrEnum):
    """Terminal result of one validation gate attempt."""

    PASSED = "passed"
    FAILED = "failed"
    DEFERRED = "deferred"
    REFUSED = "refused"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    SKIPPED = "skipped"


class ValidationFailureClass(StrEnum):
    """Failure attribution used by policy and downstream bisection."""

    CODE = "code_failure"
    ENVIRONMENT = "environment_failure"
    RESOURCE = "resource_refusal"
    CANCELLATION = "cancelled_deadline"
    TIMEOUT = "timeout"
    INVALID_REQUEST = "invalid_request"
    STALE_TREE = "stale_tree"
    STALE_FENCE = "stale_fence"
    BASELINE_UNPRODUCIBLE = "baseline_unproducible"
    DEPENDENCY = "dependency_blocked"
    RECONCILIATION = "reconciliation_required"


_SHA = re.compile(r"^[0-9a-f]{40}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_OPAQUE = re.compile(r"^[^\x00\r\n]+$")
_MAX_TAIL = 64 * 1024
_MAX_FAILURE_IDS = 256
_MAX_FAILURE_ID_BYTES = 1024


def _opaque(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise EvidenceError(f"{field_name} must be a non-blank string")
    if not _OPAQUE.fullmatch(value):
        raise EvidenceError(f"{field_name} contains control characters")
    return value


def _sha(value: object, field_name: str) -> str:
    value = _opaque(value, field_name)
    if not _SHA.fullmatch(value):
        raise EvidenceError(f"{field_name} must be a 40-character lowercase Git SHA")
    return value


def _digest(value: object, field_name: str) -> str:
    value = _opaque(value, field_name)
    if not _DIGEST.fullmatch(value):
        raise EvidenceError(f"{field_name} must be a 64-character lowercase digest")
    return value


def _timestamp(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise EvidenceError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _bounded(value: str, field_name: str) -> str:
    if len(value.encode("utf-8")) > _MAX_TAIL:
        raise EvidenceError(f"{field_name} exceeds the bounded output tail")
    return value


def _failure_id(value: object, field_name: str) -> str:
    value = _opaque(value, field_name)
    if len(value.encode("utf-8")) > _MAX_FAILURE_ID_BYTES:
        raise EvidenceError(f"{field_name} exceeds the bounded failure ID size")
    return value


@dataclass(frozen=True, slots=True)
class BaselineObservation:
    """The only baseline result a differential verdict may consume."""

    readable: bool
    tree_sha: str
    exit_code: int | None = None
    failure_ids: tuple[str, ...] = ()
    detail: str = ""
    toolchain_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "tree_sha", _sha(self.tree_sha, "baseline tree_sha"))
        if self.exit_code is not None and not isinstance(self.exit_code, int):
            raise EvidenceError("baseline exit_code must be an integer or None")
        if len(self.failure_ids) > _MAX_FAILURE_IDS:
            raise EvidenceError("baseline failure IDs exceed the bounded count")
        ids = tuple(
            _failure_id(item, "baseline failure id") for item in self.failure_ids
        )
        object.__setattr__(self, "failure_ids", tuple(sorted(set(ids))))
        if self.toolchain_digest:
            object.__setattr__(
                self,
                "toolchain_digest",
                _digest(self.toolchain_digest, "toolchain_digest"),
            )

    @property
    def passed(self) -> bool:
        return self.readable and self.exit_code == 0

    def canonical_payload(self) -> dict[str, object]:
        return {
            "readable": self.readable,
            "tree_sha": self.tree_sha,
            "exit_code": self.exit_code,
            "failure_ids": self.failure_ids,
            "detail": self.detail,
            "toolchain_digest": self.toolchain_digest,
        }


@dataclass(frozen=True, slots=True)
class DifferentialVerdict:
    """A deterministic comparison of candidate and base failure signals."""

    ok: bool
    baseline_readable: bool
    new_failure_ids: tuple[str, ...] = ()
    pre_existing_failure_ids: tuple[str, ...] = ()
    fixed_failure_ids: tuple[str, ...] = ()
    detail: str = ""

    def __post_init__(self) -> None:
        for field_name in (
            "new_failure_ids",
            "pre_existing_failure_ids",
            "fixed_failure_ids",
        ):
            values = tuple(
                _opaque(
                    item, field_name[:-1] if field_name.endswith("s") else field_name
                )
                for item in getattr(self, field_name)
            )
            object.__setattr__(self, field_name, tuple(sorted(set(values))))


def compare_failure_signals(
    *,
    mode: BaselineMode,
    baseline: BaselineObservation | None,
    candidate_exit_code: int | None,
    candidate_failure_ids: Sequence[str] = (),
) -> DifferentialVerdict:
    """Compare a gate result without ever treating an unreadable base as green."""

    if len(candidate_failure_ids) > _MAX_FAILURE_IDS:
        raise EvidenceError("candidate failure IDs exceed the bounded count")
    candidate = frozenset(
        _failure_id(item, "candidate failure id") for item in candidate_failure_ids
    )
    if mode is BaselineMode.DISABLED:
        ok = candidate_exit_code == 0
        return DifferentialVerdict(
            ok=ok,
            baseline_readable=True,
            new_failure_ids=tuple(sorted(candidate)) if not ok else (),
            detail="absolute result (baseline comparison disabled)",
        )
    if mode is BaselineMode.ABSOLUTE:
        ok = candidate_exit_code == 0
        return DifferentialVerdict(
            ok=ok,
            baseline_readable=True,
            new_failure_ids=tuple(sorted(candidate)) if not ok else (),
            detail="absolute gate result",
        )
    if baseline is None or not baseline.readable:
        return DifferentialVerdict(
            ok=False,
            baseline_readable=False,
            new_failure_ids=tuple(sorted(candidate)),
            detail=(
                "REFUSED: differential baseline could not be produced; "
                "an unreadable baseline is not an empty failure set"
            ),
        )
    base = frozenset(baseline.failure_ids)
    new = candidate - base
    pre_existing = candidate & base
    fixed = base - candidate
    # A non-zero result with no itemized signals is still a new failure when
    # the base was clean.  This prevents a tool changing output shape from
    # silently turning a failure into a differential pass.
    if candidate_exit_code != 0 and not candidate:
        new = frozenset({"<unitemized-failure>"})
    return DifferentialVerdict(
        ok=not new,
        baseline_readable=True,
        new_failure_ids=tuple(sorted(new)),
        pre_existing_failure_ids=tuple(sorted(pre_existing)),
        fixed_failure_ids=tuple(sorted(fixed)),
        detail=(
            "no new failure signals"
            if not new
            else "new failure signal(s) are not present on the immutable base"
        ),
    )


@dataclass(frozen=True, slots=True)
class GateEvidence:
    """Immutable evidence for one gate attempt."""

    evidence_id: str
    gate_name: str
    stage: ValidationStage
    tree_sha: str
    generation_id: str | None
    gate_config_digest: str
    command_digest: str
    target_host: str
    toolchain_digest: str
    resource_digest: str
    profile_digest: str
    started_at: datetime
    finished_at: datetime
    outcome: EvidenceOutcome
    failure_class: ValidationFailureClass | None = None
    job_id: str | None = None
    dependency_job_ids: tuple[str, ...] = ()
    baseline_tree_sha: str | None = None
    baseline_readable: bool | None = None
    differential: bool = False
    failure_ids: tuple[str, ...] = ()
    pre_existing_failure_ids: tuple[str, ...] = ()
    fixed_failure_ids: tuple[str, ...] = ()
    log_refs: tuple[str, ...] = ()
    artifact_refs: tuple[str, ...] = ()
    stdout_tail: str = ""
    stderr_tail: str = ""
    exit_code: int | None = None
    snapshot_gate_deferred: bool = False
    snapshot_gate_replayed: bool = False
    detail: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "evidence_id", _opaque(self.evidence_id, "evidence_id")
        )
        object.__setattr__(self, "gate_name", _opaque(self.gate_name, "gate_name"))
        object.__setattr__(self, "tree_sha", _sha(self.tree_sha, "tree_sha"))
        if self.generation_id is not None:
            object.__setattr__(
                self, "generation_id", _opaque(self.generation_id, "generation_id")
            )
        object.__setattr__(
            self,
            "gate_config_digest",
            _digest(self.gate_config_digest, "gate_config_digest"),
        )
        object.__setattr__(
            self, "command_digest", _digest(self.command_digest, "command_digest")
        )
        object.__setattr__(
            self, "target_host", _opaque(self.target_host, "target_host")
        )
        object.__setattr__(
            self, "toolchain_digest", _digest(self.toolchain_digest, "toolchain_digest")
        )
        object.__setattr__(
            self, "resource_digest", _digest(self.resource_digest, "resource_digest")
        )
        object.__setattr__(
            self, "profile_digest", _digest(self.profile_digest, "profile_digest")
        )
        object.__setattr__(
            self, "started_at", _timestamp(self.started_at, "started_at")
        )
        object.__setattr__(
            self, "finished_at", _timestamp(self.finished_at, "finished_at")
        )
        if self.finished_at < self.started_at:
            raise EvidenceError("finished_at cannot precede started_at")
        if not isinstance(self.snapshot_gate_deferred, bool):
            raise EvidenceError("snapshot_gate_deferred must be a boolean")
        if not isinstance(self.snapshot_gate_replayed, bool):
            raise EvidenceError("snapshot_gate_replayed must be a boolean")
        if self.snapshot_gate_replayed and not self.snapshot_gate_deferred:
            raise EvidenceError(
                "snapshot_gate_replayed requires a deferred snapshot marker"
            )
        if self.baseline_tree_sha is not None:
            object.__setattr__(
                self,
                "baseline_tree_sha",
                _sha(self.baseline_tree_sha, "baseline_tree_sha"),
            )
        if self.differential and self.baseline_tree_sha is None:
            raise EvidenceError("differential evidence must identify baseline_tree_sha")
        if self.outcome is EvidenceOutcome.PASSED and self.failure_ids:
            raise EvidenceError("passed evidence cannot carry failure IDs")
        if (
            self.stage is ValidationStage.CERTIFICATION
            and self.outcome is EvidenceOutcome.PASSED
            and not self.generation_id
        ):
            raise EvidenceError("passed certification evidence requires generation_id")
        for field_name in (
            "dependency_job_ids",
            "failure_ids",
            "pre_existing_failure_ids",
            "fixed_failure_ids",
            "log_refs",
            "artifact_refs",
        ):
            raw_values = getattr(self, field_name)
            if (
                field_name
                in {
                    "failure_ids",
                    "pre_existing_failure_ids",
                    "fixed_failure_ids",
                }
                and len(raw_values) > _MAX_FAILURE_IDS
            ):
                raise EvidenceError(f"{field_name} exceed the bounded count")
            normalizer = (
                _failure_id
                if field_name
                in {"failure_ids", "pre_existing_failure_ids", "fixed_failure_ids"}
                else _opaque
            )
            values = tuple(normalizer(item, field_name[:-1]) for item in raw_values)
            object.__setattr__(self, field_name, tuple(sorted(set(values))))
        object.__setattr__(
            self, "stdout_tail", _bounded(self.stdout_tail, "stdout_tail")
        )
        object.__setattr__(
            self, "stderr_tail", _bounded(self.stderr_tail, "stderr_tail")
        )

    def canonical_payload(self) -> dict[str, object]:
        return {
            "evidence_id": self.evidence_id,
            "gate_name": self.gate_name,
            "stage": self.stage,
            "tree_sha": self.tree_sha,
            "generation_id": self.generation_id,
            "gate_config_digest": self.gate_config_digest,
            "command_digest": self.command_digest,
            "target_host": self.target_host,
            "toolchain_digest": self.toolchain_digest,
            "resource_digest": self.resource_digest,
            "profile_digest": self.profile_digest,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "outcome": self.outcome,
            "failure_class": self.failure_class,
            "job_id": self.job_id,
            "dependency_job_ids": self.dependency_job_ids,
            "baseline_tree_sha": self.baseline_tree_sha,
            "baseline_readable": self.baseline_readable,
            "differential": self.differential,
            "failure_ids": self.failure_ids,
            "pre_existing_failure_ids": self.pre_existing_failure_ids,
            "fixed_failure_ids": self.fixed_failure_ids,
            "log_refs": self.log_refs,
            "artifact_refs": self.artifact_refs,
            "stdout_tail": self.stdout_tail,
            "stderr_tail": self.stderr_tail,
            "exit_code": self.exit_code,
            "snapshot_gate_deferred": self.snapshot_gate_deferred,
            "snapshot_gate_replayed": self.snapshot_gate_replayed,
            "detail": self.detail,
        }

    @property
    def digest(self) -> str:
        """Content identity used by certificates and downstream provenance."""

        return canonical_digest(self.canonical_payload())


@dataclass(frozen=True, slots=True)
class CertificateVerification:
    """Machine-readable verification result; ``valid=False`` is not a cert."""

    valid: bool
    reasons: tuple[str, ...] = ()
    certificate_digest: str | None = None


@dataclass(frozen=True, slots=True)
class ValidationCertificate:
    """Exact-generation certification aggregate."""

    certificate_id: str
    generation_id: str
    tree_sha: str
    gate_config_digest: str
    toolchain_digest: str
    target_host: str
    resource_digest: str
    evidence_digests: tuple[str, ...]
    blocking_gate_names: tuple[str, ...]
    issued_at: datetime
    profile_digest: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "certificate_id", _opaque(self.certificate_id, "certificate_id")
        )
        object.__setattr__(
            self, "generation_id", _opaque(self.generation_id, "generation_id")
        )
        object.__setattr__(self, "tree_sha", _sha(self.tree_sha, "tree_sha"))
        object.__setattr__(
            self,
            "gate_config_digest",
            _digest(self.gate_config_digest, "gate_config_digest"),
        )
        object.__setattr__(
            self, "toolchain_digest", _digest(self.toolchain_digest, "toolchain_digest")
        )
        object.__setattr__(
            self, "target_host", _opaque(self.target_host, "target_host")
        )
        object.__setattr__(
            self, "resource_digest", _digest(self.resource_digest, "resource_digest")
        )
        if not self.profile_digest:
            raise EvidenceError("certificate requires profile_digest")
        object.__setattr__(
            self, "profile_digest", _digest(self.profile_digest, "profile_digest")
        )
        object.__setattr__(self, "issued_at", _timestamp(self.issued_at, "issued_at"))
        digests = tuple(
            _digest(item, "evidence digest") for item in self.evidence_digests
        )
        if not digests:
            raise EvidenceError("certificate evidence set must not be empty")
        if len(digests) != len(set(digests)):
            raise EvidenceError("certificate evidence digests must be unique")
        object.__setattr__(self, "evidence_digests", tuple(sorted(digests)))
        names = tuple(
            _opaque(item, "blocking gate name") for item in self.blocking_gate_names
        )
        if not names:
            raise EvidenceError("certificate blocking gate set must not be empty")
        object.__setattr__(self, "blocking_gate_names", tuple(sorted(set(names))))

    def canonical_payload(self) -> dict[str, object]:
        return {
            "certificate_id": self.certificate_id,
            "generation_id": self.generation_id,
            "tree_sha": self.tree_sha,
            "gate_config_digest": self.gate_config_digest,
            "toolchain_digest": self.toolchain_digest,
            "target_host": self.target_host,
            "resource_digest": self.resource_digest,
            "evidence_digests": self.evidence_digests,
            "blocking_gate_names": self.blocking_gate_names,
            "issued_at": self.issued_at,
            "profile_digest": self.profile_digest,
        }

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    @classmethod
    def issue(
        cls,
        *,
        certificate_id: str,
        generation_id: str,
        tree_sha: str,
        gate_config_digest: str,
        toolchain_digest: str,
        target_host: str,
        resource_digest: str,
        blocking_gate_names: Sequence[str],
        evidence: Sequence[GateEvidence],
        issued_at: datetime,
        profile_digest: str = "",
    ) -> ValidationCertificate:
        """Issue only after exact evidence has passed verification."""

        certificate = cls(
            certificate_id=certificate_id,
            generation_id=generation_id,
            tree_sha=tree_sha,
            gate_config_digest=gate_config_digest,
            toolchain_digest=toolchain_digest,
            target_host=target_host,
            resource_digest=resource_digest,
            evidence_digests=tuple(item.digest for item in evidence),
            blocking_gate_names=tuple(blocking_gate_names),
            issued_at=issued_at,
            profile_digest=profile_digest,
        )
        verified = verify_certificate(certificate, evidence)
        if not verified.valid:
            raise EvidenceError(
                "cannot issue certificate: " + "; ".join(verified.reasons)
            )
        return certificate


def verify_certificate(
    certificate: ValidationCertificate,
    evidence: Sequence[GateEvidence] | Mapping[str, GateEvidence],
) -> CertificateVerification:
    """Verify exact identity and blocking gate semantics from scratch."""

    records = (
        tuple(evidence.values()) if isinstance(evidence, Mapping) else tuple(evidence)
    )
    reasons: list[str] = []
    if not records:
        reasons.append("certificate evidence set is empty")
    if not certificate.evidence_digests:
        reasons.append("certificate declares no evidence digests")
    if not certificate.blocking_gate_names:
        reasons.append("certificate has no blocking gates")
    if not certificate.profile_digest:
        reasons.append("certificate has no profile digest")
    by_name: dict[str, GateEvidence] = {}
    digests: set[str] = set()
    for item in records:
        if item.digest in digests:
            reasons.append(f"duplicate evidence digest for {item.gate_name}")
        digests.add(item.digest)
        if item.stage is not ValidationStage.CERTIFICATION:
            reasons.append(f"{item.gate_name} is not certification evidence")
        if item.tree_sha != certificate.tree_sha:
            reasons.append(f"{item.gate_name} tree SHA does not match certificate")
        if item.generation_id != certificate.generation_id:
            reasons.append(f"{item.gate_name} generation does not match certificate")
        if item.gate_config_digest != certificate.gate_config_digest:
            reasons.append(f"{item.gate_name} config digest does not match certificate")
        if item.toolchain_digest != certificate.toolchain_digest:
            reasons.append(
                f"{item.gate_name} toolchain digest does not match certificate"
            )
        if item.target_host != certificate.target_host:
            reasons.append(f"{item.gate_name} host does not match certificate")
        if item.resource_digest != certificate.resource_digest:
            reasons.append(
                f"{item.gate_name} resource digest does not match certificate"
            )
        if item.profile_digest != certificate.profile_digest:
            reasons.append(
                f"{item.gate_name} profile digest does not match certificate"
            )
        if item.gate_name in by_name:
            reasons.append(f"duplicate evidence gate {item.gate_name}")
        by_name[item.gate_name] = item
    actual = {item.digest for item in records}
    expected = set(certificate.evidence_digests)
    if actual != expected:
        reasons.append(
            "certificate evidence digest set does not match supplied evidence"
        )
    for name in certificate.blocking_gate_names:
        blocking_item = by_name.get(name)
        if blocking_item is None:
            reasons.append(f"missing blocking certification evidence: {name}")
        elif blocking_item.outcome is not EvidenceOutcome.PASSED:
            reasons.append(
                f"blocking gate {name} is {blocking_item.outcome.value}, not passed"
            )
    if any(item.snapshot_gate_deferred for item in records) and not any(
        item.snapshot_gate_replayed and item.outcome is EvidenceOutcome.PASSED
        for item in records
    ):
        reasons.append(
            "deferred snapshot has no passed selected pre-commit replay evidence"
        )
    return CertificateVerification(
        valid=not reasons,
        reasons=tuple(dict.fromkeys(reasons)),
        certificate_digest=certificate.digest,
    )


class BaselineCache:
    """Bounded process-local cache keyed by every baseline-affecting identity."""

    def __init__(self, *, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        self.max_entries = max_entries
        self._values: dict[str, BaselineObservation] = {}
        self._lock = RLock()

    @staticmethod
    def key(
        *,
        base_sha: str,
        gate_config_digest: str,
        command_digest: str,
        toolchain_digest: str,
        target_host: str,
    ) -> str:
        return canonical_digest(
            {
                "base_sha": _sha(base_sha, "base_sha"),
                "gate_config_digest": _digest(gate_config_digest, "gate_config_digest"),
                "command_digest": _digest(command_digest, "command_digest"),
                "toolchain_digest": _digest(toolchain_digest, "toolchain_digest"),
                "target_host": _opaque(target_host, "target_host"),
            }
        )

    def get(self, **identity: str) -> BaselineObservation | None:
        with self._lock:
            return self._values.get(self.key(**identity))

    def put(self, observation: BaselineObservation, **identity: str) -> str:
        key = self.key(**identity)
        with self._lock:
            if key not in self._values and len(self._values) >= self.max_entries:
                del self._values[next(iter(self._values))]
            self._values[key] = observation
        return key

    def invalidate(self, **identity: str) -> bool:
        key = self.key(**identity)
        with self._lock:
            return self._values.pop(key, None) is not None


__all__ = [
    "BaselineCache",
    "BaselineObservation",
    "CertificateVerification",
    "DifferentialVerdict",
    "EvidenceError",
    "EvidenceOutcome",
    "GateEvidence",
    "ValidationCertificate",
    "ValidationFailureClass",
    "compare_failure_signals",
    "verify_certificate",
]
