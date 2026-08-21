"""Focused contract fixtures for the engineering-documentation authority."""

from __future__ import annotations

from repository_manager.development.contracts.documentation import (
    CONCEPT_AUTHORITY_MODULE,
    CONCEPT_AUTHORITY_OWNER,
    CONCEPT_AUTHORITY_RULE,
    DocumentationBoundary,
    FindingLifecycle,
    ImpactDecision,
    ImpactOutcome,
    Materiality,
    assess_materiality,
    classify_boundary,
    documentation_standard_contract,
    review_change,
)


def test_owned_standard_is_versioned_and_keeps_concept_authority_external() -> None:
    contract = documentation_standard_contract()

    assert contract["contract_name"] == "engineering-documentation"
    assert contract["contract_version"] == "1"
    assert contract["owner"] == "repository-manager"
    assert contract["concept_authority"] == {
        "owner": CONCEPT_AUTHORITY_OWNER,
        "rule_id": CONCEPT_AUTHORITY_RULE,
        "module": CONCEPT_AUTHORITY_MODULE,
        "mode": "reference-only",
        "allocates_ids": False,
    }
    rules = contract["rules"]
    assert isinstance(rules, list)
    assert {rule["id"] for rule in rules} == {
        "DOC-BOUND-001",
        "DOC-BOUND-002",
        "DOC-BOUND-003",
        "DOC-MAT-001",
        "DOC-LIFE-001",
        "DOC-CONCEPT-001",
    }


def test_boundary_classification_distinguishes_authored_generated_and_deprecated() -> None:
    assert classify_boundary("docs/guide.md") is DocumentationBoundary.AUTHORED
    assert classify_boundary("docs/generated/guide.md") is DocumentationBoundary.GENERATED
    assert classify_boundary("docs/deprecated/guide.md") is DocumentationBoundary.DEPRECATED


def test_only_high_materiality_requires_docs_and_agents_impact_decisions() -> None:
    low = review_change(["repository_manager/worker.py"], review_revision="r1")
    medium = review_change(["docs/guide.md"], review_revision="r1")
    high = review_change(["AGENTS.md"], review_revision="r1")

    assert low.assessment.level is Materiality.LOW
    assert medium.assessment.level is Materiality.MEDIUM
    assert low.passed and medium.passed
    assert low.docs_impact is None and medium.agents_impact is None
    assert not high.passed
    assert {finding.subject for finding in high.blocking_findings} == {
        "docs-impact-decision",
        "agents-impact-decision",
    }
    assert all(finding.rule.rule_version == "1" for finding in high.findings)
    assert all(finding.rule.owner == "repository-manager" for finding in high.findings)

    accepted = review_change(
        ["AGENTS.md"],
        review_revision="r2",
        docs_impact=ImpactDecision(ImpactOutcome.NO_CHANGE, "docs remain accurate"),
        agents_impact=ImpactDecision(ImpactOutcome.UPDATE, "AGENTS gains the new rule"),
    )
    assert accepted.passed
    assert accepted.findings == ()


def test_boundary_findings_are_explicit_and_re_review_deduplicates_lifecycle() -> None:
    first = review_change(
        ["docs/generated/guide.md", "docs/deprecated/old.md"],
        review_revision="r1",
        manually_edited_generated_paths=["docs/generated/guide.md"],
        current_claims_in_deprecated_paths=["docs/deprecated/old.md"],
    )
    second = review_change(
        ["docs/generated/guide.md", "docs/deprecated/old.md"],
        review_revision="r2",
        manually_edited_generated_paths=["docs/generated/guide.md"],
        current_claims_in_deprecated_paths=["docs/deprecated/old.md"],
        previous_findings=first.findings,
    )

    assert {finding.rule.rule_id for finding in first.findings} == {
        "DOC-BOUND-002",
        "DOC-BOUND-003",
    }
    assert len(second.findings) == 2
    assert len({finding.finding_id for finding in second.findings}) == 2
    assert all(finding.occurrences == 2 for finding in second.findings)
    assert all(finding.first_seen_revision == "r1" for finding in second.findings)
    assert all(finding.last_seen_revision == "r2" for finding in second.findings)
    assert all(finding.lifecycle is FindingLifecycle.OPEN for finding in second.findings)


def test_resolved_finding_is_retained_once_and_reopens_on_recurrence() -> None:
    first = review_change(
        ["docs/generated/guide.md"],
        review_revision="r1",
        manually_edited_generated_paths=["docs/generated/guide.md"],
    )
    resolved = review_change(
        ["docs/generated/guide.md"],
        review_revision="r2",
        previous_findings=first.findings,
    )
    reopened = review_change(
        ["docs/generated/guide.md"],
        review_revision="r3",
        manually_edited_generated_paths=["docs/generated/guide.md"],
        previous_findings=resolved.findings,
    )

    assert len(resolved.findings) == 1
    assert resolved.findings[0].lifecycle is FindingLifecycle.RESOLVED
    assert len(reopened.findings) == 1
    assert reopened.findings[0].lifecycle is FindingLifecycle.OPEN
    assert reopened.findings[0].occurrences == 2


def test_materiality_paths_and_boundaries_are_canonical_and_deterministic() -> None:
    assessment = assess_materiality(
        ["docs/guide.md", "src/a.py", "docs/guide.md"],
        boundary_overrides={"src/a.py": DocumentationBoundary.GENERATED},
    )

    assert assessment.paths == ("docs/guide.md", "src/a.py")
    assert assessment.boundary_for("src/a.py") is DocumentationBoundary.GENERATED
    assert assessment.triggers == ("path:docs/guide.md",)
