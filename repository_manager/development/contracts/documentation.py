"""Deterministic engineering-documentation authority for review consumers.

The versioned JSON beside this module is the owned standard.  This module is
only its typed evaluator and lifecycle projection: it does not allocate
Concept IDs, mutate documentation, or infer a second authority.  Consumers
may use the result from a skill, package builder, or pipeline without loading
the full fleet.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from enum import StrEnum
from importlib.resources import files
from pathlib import PurePosixPath
from typing import Final

CONTRACT_NAME: Final[str] = "engineering-documentation"
CONTRACT_VERSION: Final[str] = "1"
CONTRACT_OWNER: Final[str] = "repository-manager"
DOCUMENTATION_CONTRACT_NAME: Final[str] = CONTRACT_NAME
DOCUMENTATION_CONTRACT_VERSION: Final[str] = CONTRACT_VERSION
DOCUMENTATION_CONTRACT_OWNER: Final[str] = CONTRACT_OWNER
CONCEPT_AUTHORITY_OWNER: Final[str] = "agent-utilities"
CONCEPT_AUTHORITY_RULE: Final[str] = "RMDD-16"
CONCEPT_AUTHORITY_MODULE: Final[str] = "agent_utilities.governance.concept_reservation"
_CONTRACT_RESOURCE = "engineering_documentation_v1.json"


class DocumentationContractError(ValueError):
    """Raised when a review input violates the owned documentation contract."""


class DocumentationBoundary(StrEnum):
    """Ownership boundary of a documentation surface."""

    AUTHORED = "authored"
    GENERATED = "generated"
    DEPRECATED = "deprecated"


class Materiality(StrEnum):
    """Deterministic change-materiality levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ImpactOutcome(StrEnum):
    """Explicit impact decision recorded for a high-materiality change."""

    UPDATE = "update"
    NO_CHANGE = "no_change"
    NOT_APPLICABLE = "not_applicable"


class FindingLifecycle(StrEnum):
    """Lifecycle states preserved across independent review runs."""

    OPEN = "open"
    ACCEPTED = "accepted"
    RESOLVED = "resolved"
    SUPERSEDED = "superseded"


@dataclass(frozen=True)
class RuleReference:
    """Exact rule identity attached to every emitted finding."""

    rule_id: str
    rule_version: str
    owner: str

    def as_dict(self) -> dict[str, str]:
        return {
            "rule_id": self.rule_id,
            "rule_version": self.rule_version,
            "owner": self.owner,
        }


@dataclass(frozen=True)
class ImpactDecision:
    """A human/operator decision required only at high materiality."""

    outcome: ImpactOutcome
    rationale: str

    def __post_init__(self) -> None:
        if not isinstance(self.outcome, ImpactOutcome):
            raise DocumentationContractError("impact outcome is invalid")
        if not isinstance(self.rationale, str) or not self.rationale.strip():
            raise DocumentationContractError("impact rationale must be non-blank")


@dataclass(frozen=True)
class MaterialityAssessment:
    """Stable materiality result and the evidence that triggered it."""

    level: Materiality
    paths: tuple[str, ...]
    boundaries: tuple[tuple[str, DocumentationBoundary], ...]
    triggers: tuple[str, ...]
    requires_impact_decisions: bool

    def boundary_for(self, path: str) -> DocumentationBoundary:
        normalized = _normalize_path(path)
        for candidate, boundary in self.boundaries:
            if candidate == normalized:
                return boundary
        raise DocumentationContractError(f"path is not in this assessment: {path}")


@dataclass(frozen=True)
class FindingCandidate:
    """One review observation before lifecycle reconciliation."""

    rule: RuleReference
    path: str
    subject: str
    message: str

    @classmethod
    def create(
        cls,
        *,
        rule_id: str,
        path: str,
        subject: str,
        message: str,
    ) -> FindingCandidate:
        normalized = _normalize_path(path)
        if (
            not isinstance(subject, str)
            or not isinstance(message, str)
            or not subject.strip()
            or not message.strip()
        ):
            raise DocumentationContractError(
                "finding subject and message must be non-blank"
            )
        return cls(
            rule=rule_reference(rule_id),
            path=normalized,
            subject=subject.strip(),
            message=message.strip(),
        )


@dataclass(frozen=True)
class DocumentationFinding:
    """A stable, deduplicated finding with re-review lifecycle evidence."""

    finding_id: str
    rule: RuleReference
    path: str
    subject: str
    message: str
    lifecycle: FindingLifecycle
    first_seen_revision: str
    last_seen_revision: str
    occurrences: int

    def as_dict(self) -> dict[str, object]:
        return {
            "finding_id": self.finding_id,
            **self.rule.as_dict(),
            "path": self.path,
            "subject": self.subject,
            "message": self.message,
            "lifecycle": self.lifecycle.value,
            "first_seen_revision": self.first_seen_revision,
            "last_seen_revision": self.last_seen_revision,
            "occurrences": self.occurrences,
        }


@dataclass(frozen=True)
class DocumentationReview:
    """Materiality decision plus the reconciled finding projection."""

    assessment: MaterialityAssessment
    findings: tuple[DocumentationFinding, ...]
    docs_impact: ImpactDecision | None
    agents_impact: ImpactDecision | None

    @property
    def blocking_findings(self) -> tuple[DocumentationFinding, ...]:
        return tuple(
            finding
            for finding in self.findings
            if finding.lifecycle is FindingLifecycle.OPEN
        )

    @property
    def passed(self) -> bool:
        return not self.blocking_findings

    def as_dict(self) -> dict[str, object]:
        return {
            "contract_name": CONTRACT_NAME,
            "contract_version": CONTRACT_VERSION,
            "materiality": self.assessment.level.value,
            "paths": list(self.assessment.paths),
            "boundaries": {
                path: boundary.value for path, boundary in self.assessment.boundaries
            },
            "triggers": list(self.assessment.triggers),
            "requires_impact_decisions": self.assessment.requires_impact_decisions,
            "docs_impact": _impact_as_dict(self.docs_impact),
            "agents_impact": _impact_as_dict(self.agents_impact),
            "findings": [finding.as_dict() for finding in self.findings],
            "passed": self.passed,
        }


def documentation_standard_contract() -> dict[str, object]:
    """Return the immutable machine-readable standard as a fresh mapping."""

    resource = files(__package__).joinpath(_CONTRACT_RESOURCE)
    try:
        contract = json.loads(resource.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DocumentationContractError(
            f"cannot load {_CONTRACT_RESOURCE}: {exc}"
        ) from exc
    if not isinstance(contract, dict):
        raise DocumentationContractError("documentation standard must be an object")
    _validate_contract_identity(contract)
    return contract


def rule_reference(rule_id: str) -> RuleReference:
    """Resolve one rule from the owned versioned standard."""

    if not isinstance(rule_id, str) or not rule_id.strip():
        raise DocumentationContractError("rule id must be non-blank")
    contract = documentation_standard_contract()
    rules = contract["rules"]
    if not isinstance(rules, list):
        raise DocumentationContractError("standard contract carries no rule list")
    for raw_rule in rules:
        if isinstance(raw_rule, dict) and raw_rule.get("id") == rule_id:
            return RuleReference(
                rule_id=rule_id,
                rule_version=str(raw_rule["version"]),
                owner=str(raw_rule["owner"]),
            )
    raise DocumentationContractError(f"unknown documentation rule: {rule_id}")


def classify_boundary(
    path: str,
    *,
    override: DocumentationBoundary | str | None = None,
) -> DocumentationBoundary:
    """Classify a path without reading or mutating the document."""

    normalized = _normalize_path(path)
    if override is not None:
        try:
            return DocumentationBoundary(override)
        except ValueError as exc:
            raise DocumentationContractError(
                f"unknown documentation boundary: {override}"
            ) from exc
    parts = PurePosixPath(normalized).parts
    name = parts[-1]
    if (
        "deprecated" in parts
        or "archived" in parts
        or "archive" in parts
        or "legacy" in parts
    ):
        return DocumentationBoundary.DEPRECATED
    if (
        "generated" in parts
        or name.endswith(".generated.md")
        or name.endswith(".generated.json")
    ):
        return DocumentationBoundary.GENERATED
    return DocumentationBoundary.AUTHORED


def assess_materiality(
    paths: Iterable[str],
    *,
    public_surface: bool = False,
    security_or_governance: bool = False,
    runtime_contract: bool = False,
    boundary_overrides: Mapping[str, DocumentationBoundary | str] | None = None,
) -> MaterialityAssessment:
    """Classify a change using only normalized paths and explicit signals."""

    normalized_paths = tuple(sorted({_normalize_path(path) for path in paths}))
    overrides = {
        _normalize_path(path): value
        for path, value in (boundary_overrides or {}).items()
    }
    unknown_overrides = set(overrides) - set(normalized_paths)
    if unknown_overrides:
        raise DocumentationContractError(
            "boundary override is outside the changed paths: "
            + ", ".join(sorted(unknown_overrides))
        )
    boundaries = tuple(
        (
            path,
            classify_boundary(path, override=overrides.get(path)),
        )
        for path in normalized_paths
    )
    contract = documentation_standard_contract()
    materiality = contract["materiality"]
    if not isinstance(materiality, dict):
        raise DocumentationContractError("materiality contract must be an object")
    high_patterns = _string_list(materiality.get("high_path_patterns"))
    medium_patterns = _string_list(materiality.get("medium_path_patterns"))
    triggers: list[str] = []
    explicit_signals = (
        ("public_surface", public_surface),
        ("security_or_governance", security_or_governance),
        ("runtime_contract", runtime_contract),
    )
    triggers.extend(f"explicit:{name}" for name, enabled in explicit_signals if enabled)
    triggers.extend(
        f"path:{path}"
        for path in normalized_paths
        if any(_matches(path, pattern) for pattern in high_patterns)
    )
    if triggers:
        level = Materiality.HIGH
    elif any(
        _matches(path, pattern)
        for path in normalized_paths
        for pattern in medium_patterns
    ):
        level = Materiality.MEDIUM
        triggers.extend(
            f"path:{path}"
            for path in normalized_paths
            if any(_matches(path, pattern) for pattern in medium_patterns)
        )
    else:
        level = Materiality.LOW
    return MaterialityAssessment(
        level=level,
        paths=normalized_paths,
        boundaries=boundaries,
        triggers=tuple(sorted(set(triggers))),
        requires_impact_decisions=level is Materiality.HIGH,
    )


def reconcile_findings(
    previous: Iterable[DocumentationFinding],
    current: Iterable[FindingCandidate],
    *,
    review_revision: str,
) -> tuple[DocumentationFinding, ...]:
    """Merge one review into its prior projection without duplicate rows."""

    revision = _non_blank(review_revision, "review_revision")
    prior_by_id: dict[str, DocumentationFinding] = {}
    for finding in previous:
        _validate_finding(finding)
        existing = prior_by_id.get(finding.finding_id)
        if existing is None or _finding_sort_key(finding) < _finding_sort_key(existing):
            prior_by_id[finding.finding_id] = finding
    current_by_id: dict[str, FindingCandidate] = {}
    for incoming in current:
        _validate_candidate(incoming)
        finding_id = _finding_id(incoming)
        existing_candidate = current_by_id.get(finding_id)
        if existing_candidate is None or _candidate_sort_key(
            incoming
        ) < _candidate_sort_key(existing_candidate):
            current_by_id[finding_id] = incoming

    reconciled: list[DocumentationFinding] = []
    for finding_id in sorted(set(prior_by_id) | set(current_by_id)):
        candidate = current_by_id.get(finding_id)
        prior = prior_by_id.get(finding_id)
        if candidate is not None:
            lifecycle = (
                FindingLifecycle.OPEN
                if prior is not None and prior.lifecycle is FindingLifecycle.RESOLVED
                else prior.lifecycle
                if prior is not None
                else FindingLifecycle.OPEN
            )
            reconciled.append(
                DocumentationFinding(
                    finding_id=finding_id,
                    rule=candidate.rule,
                    path=candidate.path,
                    subject=candidate.subject,
                    message=candidate.message,
                    lifecycle=lifecycle,
                    first_seen_revision=(
                        prior.first_seen_revision if prior is not None else revision
                    ),
                    last_seen_revision=revision,
                    occurrences=(prior.occurrences + 1 if prior is not None else 1),
                )
            )
        elif prior is not None:
            lifecycle = (
                FindingLifecycle.RESOLVED
                if prior.lifecycle is FindingLifecycle.OPEN
                else prior.lifecycle
            )
            reconciled.append(
                replace(prior, lifecycle=lifecycle, last_seen_revision=revision)
            )
    return tuple(reconciled)


def review_change(
    paths: Iterable[str],
    *,
    review_revision: str,
    public_surface: bool = False,
    security_or_governance: bool = False,
    runtime_contract: bool = False,
    docs_impact: ImpactDecision | None = None,
    agents_impact: ImpactDecision | None = None,
    boundary_overrides: Mapping[str, DocumentationBoundary | str] | None = None,
    manually_edited_generated_paths: Iterable[str] = (),
    current_claims_in_deprecated_paths: Iterable[str] = (),
    findings: Iterable[FindingCandidate] = (),
    previous_findings: Iterable[DocumentationFinding] = (),
) -> DocumentationReview:
    """Evaluate materiality, boundaries, and the finding lifecycle together."""

    assessment = assess_materiality(
        paths,
        public_surface=public_surface,
        security_or_governance=security_or_governance,
        runtime_contract=runtime_contract,
        boundary_overrides=boundary_overrides,
    )
    candidates = list(findings)
    normalized_manual = {
        _normalize_path(path) for path in manually_edited_generated_paths
    }
    normalized_current = {
        _normalize_path(path) for path in current_claims_in_deprecated_paths
    }
    changed_paths = set(assessment.paths)
    if not normalized_manual <= changed_paths:
        raise DocumentationContractError(
            "manual generated paths must be part of the changed paths"
        )
    if not normalized_current <= changed_paths:
        raise DocumentationContractError(
            "deprecated current-claim paths must be part of the changed paths"
        )
    if docs_impact is not None:
        _validate_impact_decision(docs_impact, "docs_impact")
    if agents_impact is not None:
        _validate_impact_decision(agents_impact, "agents_impact")
    for path, boundary in assessment.boundaries:
        if boundary is DocumentationBoundary.GENERATED and path in normalized_manual:
            candidates.append(
                FindingCandidate.create(
                    rule_id="DOC-BOUND-002",
                    path=path,
                    subject="generated-source-boundary",
                    message="generated documentation must be regenerated from its source",
                )
            )
        if boundary is DocumentationBoundary.DEPRECATED and path in normalized_current:
            candidates.append(
                FindingCandidate.create(
                    rule_id="DOC-BOUND-003",
                    path=path,
                    subject="deprecated-current-claim",
                    message="deprecated documentation cannot supply a current claim",
                )
            )
    if assessment.requires_impact_decisions:
        if docs_impact is None:
            candidates.append(
                FindingCandidate.create(
                    rule_id="DOC-MAT-001",
                    path=assessment.paths[0] if assessment.paths else "change",
                    subject="docs-impact-decision",
                    message="high-materiality change requires an explicit docs impact decision",
                )
            )
        if agents_impact is None:
            candidates.append(
                FindingCandidate.create(
                    rule_id="DOC-MAT-001",
                    path=assessment.paths[0] if assessment.paths else "change",
                    subject="agents-impact-decision",
                    message="high-materiality change requires an explicit AGENTS impact decision",
                )
            )
    return DocumentationReview(
        assessment=assessment,
        findings=reconcile_findings(
            previous_findings,
            candidates,
            review_revision=review_revision,
        ),
        docs_impact=docs_impact,
        agents_impact=agents_impact,
    )


def _validate_contract_identity(contract: Mapping[str, object]) -> None:
    if contract.get("contract_name") != CONTRACT_NAME:
        raise DocumentationContractError("documentation contract name drift")
    if contract.get("contract_version") != CONTRACT_VERSION:
        raise DocumentationContractError("documentation contract version drift")
    if contract.get("owner") != CONTRACT_OWNER:
        raise DocumentationContractError("documentation contract owner drift")
    authority = contract.get("concept_authority")
    if not isinstance(authority, dict):
        raise DocumentationContractError("concept authority must be declared")
    if (
        authority.get("owner") != CONCEPT_AUTHORITY_OWNER
        or authority.get("rule_id") != CONCEPT_AUTHORITY_RULE
        or authority.get("module") != CONCEPT_AUTHORITY_MODULE
        or authority.get("allocates_ids") is not False
    ):
        raise DocumentationContractError("Concept-ID authority boundary drift")


def _normalize_path(path: str) -> str:
    if not isinstance(path, str) or not path.strip():
        raise DocumentationContractError("documentation path must be non-blank")
    value = path.replace("\\", "/")
    relative = PurePosixPath(value)
    if relative.is_absolute() or any(
        part in {"", ".", ".."} for part in relative.parts
    ):
        raise DocumentationContractError(
            f"documentation path is not repository-relative: {path}"
        )
    return relative.as_posix()


def _string_list(value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise DocumentationContractError("documentation pattern list is invalid")
    return tuple(value)


def _matches(path: str, pattern: str) -> bool:
    return fnmatch.fnmatchcase(path, pattern) or (
        pattern.endswith("/**") and path.startswith(pattern[:-3].rstrip("/") + "/")
    )


def _non_blank(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DocumentationContractError(f"{label} must be non-blank")
    return value.strip()


def _finding_id(candidate: FindingCandidate) -> str:
    identity = {
        **candidate.rule.as_dict(),
        "path": candidate.path,
        "subject": candidate.subject,
    }
    payload = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return "docfinding-" + hashlib.sha256(payload).hexdigest()


def _candidate_sort_key(candidate: FindingCandidate) -> tuple[str, str, str, str]:
    return (
        candidate.rule.rule_id,
        candidate.path,
        candidate.subject,
        candidate.message,
    )


def _finding_sort_key(
    finding: DocumentationFinding,
) -> tuple[str, str, str, str, str]:
    return (
        finding.rule.rule_id,
        finding.path,
        finding.subject,
        finding.lifecycle.value,
        finding.message,
    )


def _validate_candidate(candidate: FindingCandidate) -> None:
    expected_rule = rule_reference(candidate.rule.rule_id)
    if candidate.rule != expected_rule:
        raise DocumentationContractError(
            f"finding rule reference drift: {candidate.rule.rule_id}"
        )
    if _normalize_path(candidate.path) != candidate.path:
        raise DocumentationContractError("finding path is not canonical")
    if not candidate.subject.strip() or not candidate.message.strip():
        raise DocumentationContractError("finding text must be non-blank")


def _validate_finding(finding: DocumentationFinding) -> None:
    if not isinstance(finding.lifecycle, FindingLifecycle):
        raise DocumentationContractError("finding lifecycle is invalid")
    if finding.occurrences < 1:
        raise DocumentationContractError("finding occurrences must be positive")
    _validate_candidate(
        FindingCandidate(
            rule=finding.rule,
            path=finding.path,
            subject=finding.subject,
            message=finding.message,
        )
    )
    if (
        _finding_id(
            FindingCandidate(
                rule=finding.rule,
                path=finding.path,
                subject=finding.subject,
                message=finding.message,
            )
        )
        != finding.finding_id
    ):
        raise DocumentationContractError("finding id does not match its identity")


def _validate_impact_decision(decision: ImpactDecision, label: str) -> None:
    if not isinstance(decision, ImpactDecision):
        raise DocumentationContractError(f"{label} is not an impact decision")


def _impact_as_dict(decision: ImpactDecision | None) -> dict[str, str] | None:
    if decision is None:
        return None
    return {"outcome": decision.outcome.value, "rationale": decision.rationale}


__all__ = [
    "CONCEPT_AUTHORITY_MODULE",
    "CONCEPT_AUTHORITY_OWNER",
    "CONCEPT_AUTHORITY_RULE",
    "CONTRACT_NAME",
    "CONTRACT_OWNER",
    "CONTRACT_VERSION",
    "DOCUMENTATION_CONTRACT_NAME",
    "DOCUMENTATION_CONTRACT_OWNER",
    "DOCUMENTATION_CONTRACT_VERSION",
    "DocumentationBoundary",
    "DocumentationContractError",
    "DocumentationFinding",
    "DocumentationReview",
    "FindingCandidate",
    "FindingLifecycle",
    "ImpactDecision",
    "ImpactOutcome",
    "Materiality",
    "MaterialityAssessment",
    "RuleReference",
    "assess_materiality",
    "classify_boundary",
    "documentation_standard_contract",
    "reconcile_findings",
    "review_change",
    "rule_reference",
]
