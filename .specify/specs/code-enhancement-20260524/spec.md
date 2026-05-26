# Code Enhancement: repository-manager

> Automated code enhancement review for repository-manager. Covers 17 analysis domains.

## User Stories

- As a **developer**, I want to **address Project Analysis findings (grade: C, score: 74)**, so that **improve project project analysis from C to at least B (80+)**.
- As a **developer**, I want to **address Codebase Optimization findings (grade: F, score: 42)**, so that **improve project codebase optimization from F to at least B (80+)**.
- As a **developer**, I want to **address Security Analysis findings (grade: F, score: 55)**, so that **improve project security analysis from F to at least B (80+)**.
- As a **developer**, I want to **address Test Coverage findings (grade: C, score: 75)**, so that **improve project test coverage from C to at least B (80+)**.
- As a **developer**, I want to **address Architecture & Design Patterns findings (grade: D, score: 65)**, so that **improve project architecture & design patterns from D to at least B (80+)**.
- As a **developer**, I want to **address Concept Traceability findings (grade: F, score: 25)**, so that **improve project concept traceability from F to at least B (80+)**.
- As a **developer**, I want to **address Test Execution findings (grade: F, score: 25)**, so that **improve project test execution from F to at least B (80+)**.
- As a **developer**, I want to **address Version Sync Analysis findings (grade: D, score: 60)**, so that **improve project version sync analysis from D to at least B (80+)**.
- As a **developer**, I want to **address Changelog Audit findings (grade: C, score: 75)**, so that **improve project changelog audit from C to at least B (80+)**.
- As a **developer**, I want to **address Pytest Quality findings (grade: C, score: 77)**, so that **improve project pytest quality from C to at least B (80+)**.
- As a **developer**, I want to **address Environment Variables findings (grade: C, score: 75)**, so that **improve project environment variables from C to at least B (80+)**.
- As a **developer**, I want to **address analyze_xdg_kg findings (grade: F, score: 0)**, so that **improve project analyze_xdg_kg from F to at least B (80+)**.

## Functional Requirements

- **FR-001**: Minor update: agent-utilities 0.2.40 (installed) -> 0.16.0
- **FR-002**: Minor update: pytest-xdist 3.6.0 (constraint — not installed) -> 3.8.0
- **FR-003**: Moderate avg cyclomatic complexity: 7.6
- **FR-004**: 5 functions exceed 200 lines (actionable refactoring targets): validate_projects (453L), main (353L), phased_bumpversion (243L), to_directory_report (209L), phased_push (207L)
- **FR-005**: Monolithic: mcp_server.py (585L) — 1 functions with high complexity (worst: _get_job_status at 78L, CC=20); Low cohesion: 11 distinct concepts in one file
- **FR-006**: Monolithic: repository_manager.py (3169L) — 11 functions with high complexity (worst: Git.validate_projects at 453L, CC=70); God class: Git (42 methods) — consider mixins/composition
- **FR-007**: Monolithic: models.py (1452L) — 7 functions with high complexity (worst: ValidationReport.to_directory_report at 209L, CC=23); Low cohesion: 30 distinct concepts in one file
- **FR-008**: 18 functions with nesting depth >4
- **FR-009**: 3 HIGH severity vulnerabilities found
- **FR-010**: Test suite lacks intent diversity (only one type)
- **FR-011**: 17 potential doc-test drift items
- **FR-012**: README.md missing sections: usage|quick start
- **FR-013**: 2 broken internal links in README.md
- **FR-014**: README missing: Has a Table of Contents
- **FR-015**: README missing: Has usage examples with code blocks
- **FR-016**: 39 broken file references in documentation
- **FR-017**: SRP: 5 modules exceed 500 lines (god modules)
- **FR-018**: SRP: 1 classes have >15 methods
- **FR-019**: No discernible layer architecture (no domain/service/adapter separation)
- **FR-020**: Low dependency injection ratio: 8%
- **FR-021**: Low traceability ratio: 0% concepts fully traced
- **FR-022**: 12 orphaned concepts (only in one source)
- **FR-023**: 54 test functions missing concept markers
- **FR-024**: 73 significant functions (>10 lines) missing concept markers in docstrings
- **FR-025**: Total lint findings: 0 (high/error: 0, medium/warning: 0, low: 0)
- **FR-026**: 1 hook(s) may be outdated: ruff-pre-commit
- **FR-027**: 4 rogue/throwaway scripts detected (fix_*, validate_*, patch_*, etc.): scripts/debug_map.py, scripts/validate_agent.py, scripts/validate_ecosystem.py, scripts/validate_a2a_agent.py
- **FR-028**: Found 2 file(s) with version '1.18.0' that are NOT tracked in .bumpversion.cfg:
- **FR-029**:   - .specify/results.json
- **FR-030**:   - .specify/reports/code_enhancement_report.md
- **FR-031**: CHANGELOG.md exists but could not be parsed — check format compliance
- **FR-032**: No changelog entries within the last 30 days
- **FR-033**: keepachangelog not installed — pip install 'universal-skills[code-enhancer]'
- **FR-034**: 2 test files exceed 500 lines — split into focused modules
- **FR-035**: Test directory lacks subdirectory organization (consider unit/, integration/, e2e/)
- **FR-036**: No @pytest.mark.parametrize usage — consider data-driven tests
- **FR-037**: 2 tests have no assertions
- **FR-038**: 3 tests exceed 100 lines — likely doing too much per test
- **FR-039**: Partial env var documentation: 50% coverage
- **FR-040**: Undocumented env vars: AUTH_TYPE, EUNOMIA_POLICY_FILE, EUNOMIA_TYPE, GIT_OPERATIONSTOOL, LLM_API_KEY, LLM_BASE_URL, MISCTOOL, OTEL_EXPORTER_OTLP_ENDPOINT, OTEL_EXPORTER_OTLP_PUBLIC_KEY, OTEL_EXPORTER_OTLP_SECRET_KEY
- **FR-041**: 8 Python env vars not in .env.example: LLM_API_KEY, LLM_BASE_URL, MCP_URL, MODEL_ID, REPOSITORY_MANAGER_DEFAULT_BRANCH
- **FR-042**: Analysis error: No module named 'agent_utilities.knowledge_graph'

## Success Criteria

- Overall GPA: 1.71 → 3.0
- Domains at B or above: 5 → 17
- Actionable findings: 42 → 0
