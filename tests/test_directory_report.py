"""Tests for the directory report generation system."""

import os

import pytest

from repository_manager.models import (
    CATEGORY_SLUG_MAP,
    GitError,
    GitMetadata,
    GitResult,
    IncrementalReportWriter,
    ProjectResult,
    ValidationCategory,
    ValidationReport,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def sample_results() -> list[GitResult]:
    """Create a representative set of GitResult objects spanning multiple categories."""
    ts = "2026-04-28T12:00:00Z"
    return [
        # Installation successes
        GitResult(
            status="success",
            data="Installed",
            metadata=GitMetadata(
                command="pip install -e '.[all]'",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=ts,
            ),
        ),
        GitResult(
            status="success",
            data="Installed",
            metadata=GitMetadata(
                command="pip install -e '.[all]'",
                workspace="/workspace/agents/beta-agent",
                return_code=0,
                timestamp=ts,
            ),
        ),
        # Installation failure
        GitResult(
            status="error",
            data="ModuleNotFoundError: No module named 'foo'",
            error=GitError(message="Install failed", code=1),
            metadata=GitMetadata(
                command="pip install -e '.[all]'",
                workspace="/workspace/agents/gamma-agent",
                return_code=1,
                timestamp=ts,
            ),
        ),
        # Bump2version skip
        GitResult(
            status="skipped",
            data="Skipped (Missing .bumpversion.cfg)",
            metadata=GitMetadata(
                command="bump2version",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=ts,
            ),
        ),
        # Bump2version success
        GitResult(
            status="success",
            data="OK",
            metadata=GitMetadata(
                command="bump2version",
                workspace="/workspace/agents/beta-agent",
                return_code=0,
                timestamp=ts,
            ),
        ),
        # Pre-commit results
        GitResult(
            status="success",
            data="All checks passed",
            metadata=GitMetadata(
                command="pre-commit run --all-files",
                workspace="/workspace/agents/alpha-agent",
                return_code=0,
                timestamp=ts,
            ),
        ),
        GitResult(
            status="error",
            data="ruff check failed",
            error=GitError(message="Pre-commit failed", code=1),
            metadata=GitMetadata(
                command="pre-commit run --all-files",
                workspace="/workspace/agents/beta-agent",
                return_code=1,
                timestamp=ts,
            ),
        ),
    ]


@pytest.fixture
def sample_report(sample_results: list[GitResult]) -> ValidationReport:
    return ValidationReport.from_results(sample_results)


# ---------------------------------------------------------------------------
# ValidationCategory tests
# ---------------------------------------------------------------------------


class TestValidationCategory:
    def test_to_markdown_renders_sections(self):
        cat = ValidationCategory(
            name="Test Category",
            total=3,
            success_count=1,
            failure_count=1,
            skipped_count=1,
            successes=[ProjectResult(project="alpha", message="Success")],
            failures=[
                ProjectResult(
                    project="beta", message="Error occurred", output="traceback..."
                )
            ],
            skipped=[ProjectResult(project="gamma", message="Skipped (reason)")],
        )
        md = cat.to_markdown()
        assert "# Test Category" in md
        assert "## Successes ✅" in md
        assert "## Failures ❌" in md
        assert "## Skipped ⏭️" in md
        assert "**alpha**" in md
        assert "traceback..." in md

    def test_for_project_filters_correctly(self):
        cat = ValidationCategory(
            name="Install",
            total=3,
            success_count=2,
            failure_count=1,
            successes=[
                ProjectResult(project="alpha", message="OK"),
                ProjectResult(project="beta", message="OK"),
            ],
            failures=[ProjectResult(project="alpha", message="Fail")],
        )
        filtered = cat.for_project("alpha")
        assert filtered.success_count == 1
        assert filtered.failure_count == 1
        assert filtered.total == 2
        assert all(r.project == "alpha" for r in filtered.successes)
        assert all(r.project == "alpha" for r in filtered.failures)

    def test_for_project_returns_empty_when_no_match(self):
        cat = ValidationCategory(
            name="Install",
            total=1,
            success_count=1,
            successes=[ProjectResult(project="alpha", message="OK")],
        )
        filtered = cat.for_project("nonexistent")
        assert filtered.total == 0
        assert filtered.success_count == 0


# ---------------------------------------------------------------------------
# ValidationReport.to_directory_report tests
# ---------------------------------------------------------------------------


class TestToDirectoryReport:
    def test_creates_directory_structure(self, sample_report, tmp_path):
        report_dir = sample_report.to_directory_report(str(tmp_path))

        assert os.path.isdir(report_dir)
        assert "validation-reports-" in os.path.basename(report_dir)
        assert os.path.isfile(os.path.join(report_dir, "index.md"))

    def test_creates_per_repo_directories(self, sample_report, tmp_path):
        report_dir = sample_report.to_directory_report(str(tmp_path))

        # We should have directories for alpha-agent, beta-agent, gamma-agent
        entries = os.listdir(report_dir)
        repo_dirs = [e for e in entries if e.endswith("-results")]
        assert len(repo_dirs) >= 2  # At minimum alpha and beta

    def test_per_scan_files_have_correct_format(self, sample_report, tmp_path):
        report_dir = sample_report.to_directory_report(str(tmp_path))

        # Check alpha-agent has files
        alpha_dir = os.path.join(report_dir, "alpha-agent-results")
        if os.path.exists(alpha_dir):
            files = os.listdir(alpha_dir)
            assert len(files) > 0
            for f in files:
                assert f.endswith(".md")
                assert "error(s)" in f
                assert "warning(s)" in f

    def test_index_contains_summary_table(self, sample_report, tmp_path):
        report_dir = sample_report.to_directory_report(str(tmp_path))
        index_path = os.path.join(report_dir, "index.md")

        with open(index_path) as f:
            content = f.read()

        assert "# 📋 Validation Report" in content
        assert "## Category Summary" in content
        assert "## Repository Results" in content
        assert "| Category |" in content
        assert "| Repository |" in content

    def test_index_contains_repo_links(self, sample_report, tmp_path):
        report_dir = sample_report.to_directory_report(str(tmp_path))
        index_path = os.path.join(report_dir, "index.md")

        with open(index_path) as f:
            content = f.read()

        # Should have links to repo result directories
        assert "alpha-agent-results" in content
        assert "beta-agent-results" in content

    def test_handles_empty_results(self, tmp_path):
        report = ValidationReport.from_results([])
        report_dir = report.to_directory_report(str(tmp_path))

        assert os.path.isdir(report_dir)
        index_path = os.path.join(report_dir, "index.md")
        assert os.path.isfile(index_path)

        with open(index_path) as f:
            content = f.read()
        assert "Total Checks:** 0" in content

    def test_timestamp_format_in_dirname(self, tmp_path):
        report = ValidationReport(
            timestamp="2026-04-28 17:13:56",
            total=0,
        )
        report_dir = report.to_directory_report(str(tmp_path))
        dirname = os.path.basename(report_dir)
        assert dirname == "validation-reports-2026-04-28_17-13-56"


# ---------------------------------------------------------------------------
# IncrementalReportWriter tests
# ---------------------------------------------------------------------------


class TestIncrementalReportWriter:
    def test_write_phase_creates_files(self, tmp_path):
        writer = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 12:00:00"
        )

        results = [
            GitResult(
                status="success",
                data="OK",
                metadata=GitMetadata(
                    command="pip install -e '.[all]'",
                    workspace="/workspace/agents/test-agent",
                    return_code=0,
                    timestamp="2026-04-28T12:00:00Z",
                ),
            ),
        ]

        cat = writer.write_phase("Ecosystem Installation", results)

        assert cat.name == "Ecosystem Installation"
        assert cat.success_count == 1

        # Check files were written
        repo_dir = os.path.join(writer.report_root, "test-agent-results")
        assert os.path.isdir(repo_dir)
        files = os.listdir(repo_dir)
        assert len(files) == 1
        assert files[0].startswith("installation-")

    def test_write_phase_incremental_accumulation(self, tmp_path):
        writer = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 12:00:00"
        )

        # Phase 1: installation
        writer.write_phase(
            "Ecosystem Installation",
            [
                GitResult(
                    status="success",
                    data="OK",
                    metadata=GitMetadata(
                        command="pip install",
                        workspace="/workspace/agents/my-agent",
                        return_code=0,
                        timestamp="",
                    ),
                ),
            ],
        )

        # Phase 2: pre-commit
        writer.write_phase(
            "Pre-commit Standard Compliance",
            [
                GitResult(
                    status="error",
                    data="ruff failed",
                    error=GitError(message="Pre-commit failed", code=1),
                    metadata=GitMetadata(
                        command="pre-commit run",
                        workspace="/workspace/agents/my-agent",
                        return_code=1,
                        timestamp="",
                    ),
                ),
            ],
        )

        # my-agent should now have 2 files
        repo_dir = os.path.join(writer.report_root, "my-agent-results")
        files = os.listdir(repo_dir)
        assert len(files) == 2

        # Stats should be accumulated
        assert writer.project_stats["my-agent"]["success"] == 1
        assert writer.project_stats["my-agent"]["failure"] == 1

    def test_finalize_writes_index(self, tmp_path):
        writer = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 12:00:00"
        )

        writer.write_phase(
            "Ecosystem Installation",
            [
                GitResult(
                    status="success",
                    data="OK",
                    metadata=GitMetadata(
                        command="pip install",
                        workspace="/workspace/agents/demo-agent",
                        return_code=0,
                        timestamp="",
                    ),
                ),
            ],
        )

        report_dir = writer.finalize()

        index_path = os.path.join(report_dir, "index.md")
        assert os.path.isfile(index_path)

        with open(index_path) as f:
            content = f.read()

        assert "📋 Validation Report" in content
        assert "demo-agent" in content
        assert "🟢" in content  # All passed, should be green

    def test_finalize_shows_red_for_failures(self, tmp_path):
        writer = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 12:00:00"
        )

        writer.write_phase(
            "Ecosystem Installation",
            [
                GitResult(
                    status="error",
                    data="Failed",
                    error=GitError(message="crash", code=1),
                    metadata=GitMetadata(
                        command="pip install",
                        workspace="/workspace/agents/broken-agent",
                        return_code=1,
                        timestamp="",
                    ),
                ),
            ],
        )

        report_dir = writer.finalize()

        with open(os.path.join(report_dir, "index.md")) as f:
            content = f.read()

        assert "🔴" in content  # Has failures, should be red
        assert "broken-agent" in content

    def test_unique_directory_per_timestamp(self, tmp_path):
        w1 = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 12:00:00"
        )
        w2 = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 12:00:01"
        )
        assert w1.report_root != w2.report_root

    def test_report_root_is_timestamped(self, tmp_path):
        writer = IncrementalReportWriter(
            output_dir=str(tmp_path), timestamp="2026-04-28 17:13:56"
        )
        dirname = os.path.basename(writer.report_root)
        assert dirname == "validation-reports-2026-04-28_17-13-56"


# ---------------------------------------------------------------------------
# CATEGORY_SLUG_MAP tests
# ---------------------------------------------------------------------------


class TestCategorySlugMap:
    def test_all_known_categories_have_slugs(self):
        expected = {
            "Ecosystem Installation",
            "Version Metadata Sync (Dry Run)",
            "Agent Standards Compliance",
            "MCP Help Check",
            "Agent Help Check",
            "Agent Runtime & Web UI",
            "Pre-commit Standard Compliance",
            "Additional Operational Checks",
        }
        assert expected == set(CATEGORY_SLUG_MAP.keys())

    def test_slugs_are_filesystem_safe(self):
        for slug in CATEGORY_SLUG_MAP.values():
            assert "/" not in slug
            assert "\\" not in slug
            assert " " not in slug
            assert slug == slug.lower()


# ---------------------------------------------------------------------------
# Backward compatibility: to_markdown still works
# ---------------------------------------------------------------------------


class TestToMarkdownBackwardCompat:
    def test_to_markdown_still_produces_output(self, sample_report):
        md = sample_report.to_markdown()
        assert "# VALIDATION SUMMARY" in md
        assert "✅" in md
        assert "❌" in md
