import datetime
import logging
import os
import re

from pydantic import BaseModel, Field

logger = logging.getLogger("RepositoryManager")

# Mapping from display category names to filesystem-safe slug names
CATEGORY_SLUG_MAP: dict[str, str] = {
    "Ecosystem Installation": "installation",
    "Version Metadata Sync (Dry Run)": "bump2version-dryrun",
    "Agent Standards Compliance": "agent-standards",
    "MCP Help Check": "mcp-help",
    "Agent Help Check": "agent-help",
    "Agent Runtime & Web UI": "agent-runtime",
    "Pre-commit Standard Compliance": "pre-commit-results",
    "Additional Operational Checks": "additional-checks",
    "Coverage Report": "coverage-report",
}


def _sanitize_filename(name: str) -> str:
    """Sanitize a string to be a valid Windows and cross-platform filename."""
    # Replace illegal Windows characters with a hyphen
    return re.sub(r'[<>:"/\\|?*]', "-", name)


def _build_next_command_block(
    failed_projects: list[str],
    validated_repositories: list[str] | None = None,
) -> list[str]:
    """Build the 'Next Validation Command' markdown block for index.md.

    Three states:
    1. Failures exist → generate command targeting only failed repos.
    2. No failures, targeted run → suggest full-sweep regression test.
    3. No failures, full run → all clear, no further action.

    Args:
        failed_projects: Sorted list of project names that had failures.
        validated_repositories: The list of repos explicitly targeted, or ``None``
            if all repos were validated.

    Returns:
        List of markdown lines to extend into the index.
    """
    lines: list[str] = []
    output_dir = "/home/apps/workspace/reports"

    if failed_projects:
        repos_csv = ",".join(failed_projects)
        lines.extend(
            [
                "---",
                "",
                "## 🔄 Next Validation Command",
                "> [!IMPORTANT]",
                "> **Execute this command to continue the validation loop after fixing the failures above.**",
                "",
                f"**Action:** Re-validate the {len(failed_projects)} {'repository' if len(failed_projects) == 1 else 'repositories'} that failed.",
                "",
                "Use the `rm_projects` MCP tool with these parameters:",
                '- `type`: `"all"`',
                f'- `repositories`: `"{repos_csv}"`',
                f'- `output_dir`: `"{output_dir}"`',
                "",
                "---",
                "",
            ]
        )
    elif validated_repositories is not None:
        # Targeted run with 0 failures → suggest full regression sweep
        lines.extend(
            [
                "---",
                "",
                "## ✅ Targeted Validation Passed — Run Full Regression Sweep",
                "",
                "> [!IMPORTANT]",
                "> **All targeted repositories passed. Execute this command to run a full regression test across ALL repositories.**",
                "",
                "**Action:** Run a full validation sweep to ensure no cross-dependency regressions.",
                "",
                "Use the `rm_projects` MCP tool with these parameters:",
                '- `action`: `"validate"`',
                '- `type`: `"all"`',
                f'- `output_dir`: `"{output_dir}"`',
                "",
                "*Do NOT pass the `repositories` parameter — this must validate all projects.*",
                "",
                "---",
                "",
            ]
        )
    else:
        # Full run with 0 failures → all clear
        lines.extend(
            [
                "---",
                "",
                "## ✅ All Repositories Passed — Validation Complete",
                "",
                "> [!TIP]",
                "> **No further validation is required.** All repositories passed the full validation sweep with 0 errors.",
                "",
                "---",
                "",
            ]
        )

    return lines


class GitError(BaseModel):
    message: str
    code: int


class GitMetadata(BaseModel):
    command: str
    workspace: str
    return_code: int
    timestamp: str


class GitResult(BaseModel):
    status: str
    data: str
    error: GitError | None = None
    metadata: GitMetadata | None = None


class ReadmeResult(BaseModel):
    content: str
    path: str


class RepositoryConfig(BaseModel):
    url: str
    description: str | None = None


class SubdirectoryConfig(BaseModel):
    description: str | None = None
    repositories: list[RepositoryConfig] = Field(default_factory=list)
    subdirectories: dict[str, "SubdirectoryConfig"] = Field(default_factory=dict)


class MaintenanceUpdate(BaseModel):
    target: str | None = None
    target_pattern: str | None = None
    package: str
    exclude: list[str] = Field(default_factory=list)


class MaintenancePhase(BaseModel):
    name: str
    phase: int
    project: str | None = None
    projects: list[str] = Field(default_factory=list)
    bulk_bump: bool = False
    bulk_push: bool = False
    wait_minutes: int = 0
    updates: list[MaintenanceUpdate] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)


class MaintenanceConfig(BaseModel):
    description: str | None = None
    phases: list[MaintenancePhase] = Field(default_factory=list)


class GraphConfig(BaseModel):
    enabled: bool = True
    multimodal: bool = False
    incremental: bool = True
    groups: list[dict] = Field(default_factory=list)


class WorkspaceConfig(BaseModel):
    name: str
    path: str
    description: str | None = None
    repositories: list[RepositoryConfig] = Field(default_factory=list)
    subdirectories: dict[str, SubdirectoryConfig] = Field(default_factory=dict)
    maintenance: MaintenanceConfig | None = None
    graph: GraphConfig | None = None


class ProjectResult(BaseModel):
    project: str
    message: str | None = None
    output: str | None = None
    command: str | None = None


class ValidationCategory(BaseModel):
    name: str
    total: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    successes: list[ProjectResult] = Field(default_factory=list)
    failures: list[ProjectResult] = Field(default_factory=list)
    skipped: list[ProjectResult] = Field(default_factory=list)

    def to_markdown(self) -> str:
        """Render this category as a standalone markdown document."""
        md = [
            f"# {self.name}",
            f"**Success:** {self.success_count} ✅ | **Failure:** {self.failure_count} ❌ | **Skipped:** {self.skipped_count} ⏭️",
            "",
        ]

        if self.successes:
            md.append("## Successes ✅")
            for r in self.successes:
                md.append(f"- **{r.project}**: {r.message}")
            md.append("")

        if self.failures:
            md.append("## Failures ❌")
            for r in self.failures:
                md.append(f"- **{r.project}**: {r.message}")
                if r.output:
                    md.append(f"```text\n{r.output}\n```")
            md.append("")

        if self.skipped:
            md.append("## Skipped ⏭️")
            for r in self.skipped:
                md.append(f"- **{r.project}**: {r.message}")
            md.append("")

        return "\n".join(md)

    def for_project(self, project_name: str) -> "ValidationCategory":
        """Return a filtered copy of this category containing only results for the given project."""
        filtered_successes = [r for r in self.successes if r.project == project_name]
        filtered_failures = [r for r in self.failures if r.project == project_name]
        filtered_skipped = [r for r in self.skipped if r.project == project_name]
        return ValidationCategory(
            name=self.name,
            total=len(filtered_successes)
            + len(filtered_failures)
            + len(filtered_skipped),
            success_count=len(filtered_successes),
            failure_count=len(filtered_failures),
            skipped_count=len(filtered_skipped),
            successes=filtered_successes,
            failures=filtered_failures,
            skipped=filtered_skipped,
        )


def _filter_pre_commit_output(output: str) -> str:
    """
    Parse the pre-commit stdout and only return output for hooks that 'Failed'.
    Strips out hooks that 'Passed' and their verbose outputs (like nosec warnings).
    """
    if not output:
        return output

    import re

    lines = output.split("\n")
    filtered_lines = []

    current_block: list[str] = []
    keep_block = True  # Keep everything before the first hook

    for line in lines:
        match = re.search(r"\.{5,}(Passed|Failed|Skipped)", line)
        if match:
            if keep_block and current_block:
                filtered_lines.extend(current_block)
            current_block = []

            status = match.group(1)
            keep_block = status == "Failed"
            current_block.append(line)
        else:
            current_block.append(line)

    if keep_block and current_block:
        filtered_lines.extend(current_block)

    return "\n".join(filtered_lines).strip()


# ---------------------------------------------------------------------------
# Patterns used by _extract_error_lines to classify output lines
# ---------------------------------------------------------------------------
_HOOK_STATUS_RE = re.compile(r"^(.+?)\.{5,}(Passed|Failed|Skipped)\s*$")
_EXIT_CODE_RE = re.compile(r"^-\s*exit code:\s*\d+")
_FILES_MODIFIED_RE = re.compile(r"^-\s*files were modified by this hook")
_FIXING_RE = re.compile(r"^Fixing\s+")
_ERROR_LINE_RE = re.compile(r"error[:|\[]", re.IGNORECASE)
_FOUND_ERRORS_RE = re.compile(r"^Found \d+ errors?", re.IGNORECASE)
_PYTEST_FAILED_RE = re.compile(r"^FAILED\s+")
_PYTEST_SHORT_SUMMARY_RE = re.compile(r"^=+\s*(FAILURES|short test summary)")
_PYTEST_RESULT_LINE_RE = re.compile(r"^=+\s*(\d+\s+failed.*|.*\d+\s+error.*)\s*=+\s*$")
_RUFF_DIAG_RE = re.compile(r"^\s*[A-Z]\d{3,4}\s+")
_BANDIT_ISSUE_RE = re.compile(r"^>{1,2}\s*Issue:\s*\[B\d+")
_BANDIT_LOCATION_RE = re.compile(r"^\s*Location:\s*\./")
_BANDIT_SEVERITY_RE = re.compile(r"^\s*Severity:\s+(Low|Medium|High)")
_VULTURE_UNUSED_RE = re.compile(
    r"unused (variable|function|import|class|method|attribute|property)"
)


# Lines to unconditionally drop from summaries
_NOISE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"^\[tester\]\s+WARNING\s+nosec"),
    re.compile(r"^\[main\]\s+INFO"),
    re.compile(r"^\[manager\]\s+WARNING"),
    re.compile(r"^Working\.\.\.\s+[━░▓█]"),
    re.compile(r"^Run started:"),
    re.compile(r"^Test results:\s*$"),
    re.compile(r"^\s*No issues identified"),
    re.compile(r"^Code scanned:"),
    re.compile(r"^\s*Total lines of code:"),
    re.compile(r"^\s*Total lines skipped"),
    re.compile(r"^\s*Total potential issues skipped"),
    re.compile(r"^Run metrics:"),
    re.compile(r"^\s*Total issues \(by"),
    re.compile(r"^\s*Undefined:\s*\d+"),
    re.compile(r"^\s*Low:\s*\d+"),
    re.compile(r"^\s*Medium:\s*\d+"),
    re.compile(r"^\s*High:\s*\d+"),
    re.compile(r"^Files skipped"),
    re.compile(r"^platform linux --"),
    re.compile(r"^cachedir:"),
    re.compile(r"^hypothesis profile"),
    re.compile(r"^rootdir:"),
    re.compile(r"^configfile:"),
    re.compile(r"^testpaths:"),
    re.compile(r"^plugins:"),
    re.compile(r"^timeout:"),
    re.compile(r"^timeout method:"),
    re.compile(r"^timeout func_only:"),
    re.compile(r"^asyncio:"),
    re.compile(r"^collecting \.\.\.\s"),
    re.compile(r"^collected \d+ items"),
    re.compile(r"PASSED\s*$"),
    re.compile(r"^\s*\^+\s*$"),  # ruff caret underlines
    re.compile(r"^-\s*hook id:"),
    re.compile(r"^-\s*duration:"),
    re.compile(r"^\s*\d+ files? left unchanged"),
    re.compile(r"^No notebooks found"),
    re.compile(r"^Resolved \d+ packages"),
    re.compile(r"^Summary:\s*.*\d+ valid"),  # mermaid-validate summary
    re.compile(r"^=+ test session starts =+"),
    re.compile(r"^=+ warnings summary =+"),
    re.compile(r"^-- Docs:"),
    re.compile(r"^\s*warnings\.warn\("),
    re.compile(r"PydanticDeprecatedSince"),
    re.compile(r"^\s*$"),  # blank lines
    re.compile(r"^Env:\s+http"),  # TestModel env lines
    re.compile(r"^Model:\s+TestModel"),
    re.compile(r"^All checks passed"),
    re.compile(r"^Agent v\d+"),
    re.compile(r"^\s*\|$"),  # ruff context pipe-only lines
    re.compile(r"^\s*\d+\s*\|"),  # ruff source context lines (e.g. " 82 | ...")
    re.compile(r"^\s*Success: no issues found"),
    re.compile(r"^SyntaxWarning:"),
    re.compile(r"^\s*s_\w+ = extract_section"),
]


def _extract_error_lines(output: str) -> list[str]:
    """Extract only actionable error lines from pre-commit / validation output.

    Parses the raw stdout and returns a compact list of lines containing only:
    - Failed hook headers (hook name + "Failed")
    - Exit codes and "files were modified" notices
    - Actual error messages (mypy ``error:``, ruff diagnostics, etc.)
    - ``Found N errors`` summary lines
    - ``FAILED`` pytest test names and result summaries
    - ``Fixing <file>`` lines (collapsed into a single summary)

    Everything else (passed hooks, bandit info, pytest PASSED lines, progress
    bars, session boilerplate, caret underlines, etc.) is dropped.

    Returns:
        A list of stripped, non-empty strings — one per actionable line.
    """
    if not output:
        return []

    raw_lines = output.split("\n")
    result: list[str] = []
    in_failed_block = False
    fixing_files: list[str] = []

    def _flush_fixing() -> None:
        """Collapse accumulated 'Fixing ...' lines into a compact summary."""
        if not fixing_files:
            return
        if len(fixing_files) <= 3:
            for ff in fixing_files:
                result.append(f"  Fixing {ff}")
        else:
            result.append(
                f"  Fixing {len(fixing_files)} files: "
                f"{fixing_files[0]}, {fixing_files[1]}, ... {fixing_files[-1]}"
            )
        fixing_files.clear()

    for line in raw_lines:
        stripped = line.strip()
        if not stripped:
            continue

        # --- Hook boundary detection ---
        hook_match = _HOOK_STATUS_RE.match(stripped)
        if hook_match:
            _flush_fixing()
            status = hook_match.group(2)
            if status == "Failed":
                in_failed_block = True
                hook_name = hook_match.group(1).strip().rstrip(".")
                result.append(f"- **{hook_name}** — Failed")
            else:
                in_failed_block = False
            continue

        # Only process lines inside a Failed block
        if not in_failed_block:
            continue

        # --- Drop noise lines ---
        if any(pat.search(stripped) for pat in _NOISE_PATTERNS):
            continue

        # --- Keep: exit code ---
        if _EXIT_CODE_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: files were modified ---
        if _FILES_MODIFIED_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: Fixing <file> (accumulate) ---
        fix_match = _FIXING_RE.match(stripped)
        if fix_match:
            # Extract just the filename
            fname = stripped.replace("Fixing ", "").strip()
            fixing_files.append(fname)
            continue

        # --- Keep: error lines (mypy, ruff, general) ---
        if _ERROR_LINE_RE.search(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: ruff diagnostic header (e.g. "B904 Within an...") ---
        if _RUFF_DIAG_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: "Found N errors" summary ---
        if _FOUND_ERRORS_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: FAILED test names ---
        if _PYTEST_FAILED_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: pytest result summary line (e.g. "= 6 failed, 2287 passed =") ---
        if _PYTEST_RESULT_LINE_RE.match(stripped):
            # Clean up the '=' borders
            clean = stripped.strip("= ").strip()
            result.append(f"  {clean}")
            continue

        # --- Keep: pytest short test summary header ---
        if _PYTEST_SHORT_SUMMARY_RE.match(stripped):
            continue  # Skip the header itself, FAILED lines follow

        # --- Keep: ruff arrow lines (e.g. "--> arr_mcp/mcp_server.py:84:9") ---
        if stripped.startswith("-->"):
            result.append(f"  {stripped}")
            continue

        # --- Keep: bandit issue lines (e.g. ">> Issue: [B311:blacklist]") ---
        if _BANDIT_ISSUE_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: bandit Location and Severity ---
        if _BANDIT_LOCATION_RE.match(stripped):
            result.append(f"  {stripped}")
            continue
        if _BANDIT_SEVERITY_RE.match(stripped):
            result.append(f"  {stripped}")
            continue

        # --- Keep: vulture unused variable/function/etc. ---
        if _VULTURE_UNUSED_RE.search(stripped):
            result.append(f"  {stripped}")
            continue

    _flush_fixing()
    return result


def _build_summary_md(
    timestamp: str,
    total: int,
    success_count: int,
    failure_count: int,
    categories: list["ValidationCategory"],
    failed_projects: list[str],
    validated_repositories: list[str] | None = None,
    skipped_count: int = 0,
) -> list[str]:
    """Build a condensed, LLM-actionable summary.md with only errors.

    The summary is designed to let an LLM one-shot understand all failures,
    plan fixes, apply them, and re-validate — all without reading the full
    per-repo detail files.

    Args:
        timestamp: Human-readable timestamp for the report header.
        total: Total number of checks executed.
        success_count: Number of checks that passed.
        failure_count: Number of checks that failed.
        categories: All ValidationCategory objects from the run.
        failed_projects: Sorted list of project names with failures.
        validated_repositories: List of explicitly targeted repos, or None for all.
        skipped_count: Number of checks that were skipped.

    Returns:
        List of markdown lines for summary.md.
    """
    MAX_LINES_PER_REPO = 60

    summary_md = [
        "# 📋 Validation Report",
        f"**Time:** {timestamp}  ",
        f"**Total Checks:** {total} | **Success:** {success_count} ✅ "
        f"| **Failure:** {failure_count} ❌ | **Skipped:** {skipped_count} ⏭️",
        "",
    ]

    # --- Next Validation Command (top of file for immediate agent action) ---
    summary_md.extend(
        _build_next_command_block(
            failed_projects=failed_projects,
            validated_repositories=validated_repositories,
        )
    )

    summary_md.append("## ❌ Failures by Repository")
    summary_md.append("")

    if failed_projects:
        for project in failed_projects:
            safe_project = _sanitize_filename(project)
            repo_dir = f"{safe_project}-results"
            summary_md.append(f"### ❌ {project}")
            summary_md.append(f"> 📂 Full logs: [{repo_dir}/]({repo_dir}/)")
            summary_md.append("")

            repo_line_count = 0
            truncated = False

            for cat in categories:
                if truncated:
                    break
                filtered = cat.for_project(project)
                if filtered.failure_count == 0:
                    continue

                summary_md.append(f"**{cat.name}**")
                repo_line_count += 1

                for failure in filtered.failures:
                    if truncated:
                        break

                    # Extract actionable error lines from output.
                    # When r.error is None, the entire raw stdout ends up in
                    # failure.message (not failure.output) because
                    # output = None when message == data.  Always try both —
                    # _extract_error_lines handles content without hooks
                    # gracefully (returns []).
                    raw = failure.output or failure.message or ""
                    error_lines = _extract_error_lines(raw)

                    if not error_lines:
                        # No structured output — just show the message
                        summary_md.append(f"- {failure.message}")
                        repo_line_count += 1
                    else:
                        # Add error lines with truncation
                        remaining = MAX_LINES_PER_REPO - repo_line_count
                        if len(error_lines) > remaining:
                            error_lines = error_lines[:remaining]
                            truncated = True

                        for eline in error_lines:
                            summary_md.append(eline)
                            repo_line_count += 1

                        if truncated:
                            _slug = CATEGORY_SLUG_MAP.get(
                                cat.name, cat.name.lower().replace(" ", "-")
                            )
                            summary_md.append(
                                f"> ℹ️ ...truncated at {MAX_LINES_PER_REPO} lines. "
                                f"See full output: [{repo_dir}/]({repo_dir}/)"
                            )

            summary_md.append("")
    else:
        summary_md.append("🎉 No failures found across any repository!")
        summary_md.append("")

    return summary_md


class ValidationReport(BaseModel):
    timestamp: str
    total: int = 0
    success_count: int = 0
    failure_count: int = 0
    skipped_count: int = 0
    categories: list[ValidationCategory] = Field(default_factory=list)

    @classmethod
    def from_results(cls, results: list[GitResult]) -> "ValidationReport":
        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "error"]
        skipped = [r for r in results if r.status == "skipped"]

        report = cls(
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            total=len(results),
            success_count=len(successes),
            failure_count=len(failures),
            skipped_count=len(skipped),
        )

        categories_map = {
            "Ecosystem Installation": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and "pip install" in r.metadata.command
            ],
            "Version Metadata Sync (Dry Run)": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and "bump2version" in r.metadata.command
            ],
            "Agent Standards Compliance": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and "static_check" in r.metadata.command
            ],
            "MCP Help Check": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and "mcp_server" in r.metadata.command
                and "--help" in r.metadata.command
            ],
            "Agent Help Check": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and (
                    "agent_server" in r.metadata.command
                    or "server" in r.metadata.command
                )
                and "--help" in r.metadata.command
                and "mcp_server" not in r.metadata.command
            ],
            "Agent Runtime & Web UI": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and (
                    "runtime_check" in r.metadata.command
                    or "--web" in r.metadata.command
                )
            ],
            "Pre-commit Standard Compliance": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and "pre-commit run" in r.metadata.command
            ],
            "Coverage Report": [
                r
                for r in results
                if r.metadata
                and r.metadata.command
                and "coverage report" in r.metadata.command
            ],
        }

        known_ids = set()
        for cat_list in categories_map.values():
            for r in cat_list:
                if r.metadata:
                    known_ids.add(id(r))

        uncategorized = [r for r in results if id(r) not in known_ids]
        if uncategorized:
            categories_map["Additional Operational Checks"] = uncategorized

        for cat_name, cat_results in categories_map.items():
            if not cat_results:
                continue

            cat = ValidationCategory(
                name=cat_name,
                total=len(cat_results),
                success_count=len([r for r in cat_results if r.status == "success"]),
                failure_count=len([r for r in cat_results if r.status == "error"]),
                skipped_count=len([r for r in cat_results if r.status == "skipped"]),
            )

            for r in cat_results:
                pkg = r.metadata.workspace.split("/")[-1] if r.metadata else "unknown"
                cmd = r.metadata.command if r.metadata else None

                if r.status == "success":
                    cat.successes.append(
                        ProjectResult(
                            project=pkg, message="Success", output=r.data, command=cmd
                        )
                    )
                elif r.status == "error":
                    error_msg = (
                        r.error.message if r.error else (r.data or "Unknown error")
                    )
                    output = r.data if r.data and r.data != error_msg else None
                    if output and cmd and ("pre_commit" in cmd or "pre-commit" in cmd):
                        output = _filter_pre_commit_output(output)
                    cat.failures.append(
                        ProjectResult(
                            project=pkg, message=error_msg, output=output, command=cmd
                        )
                    )
                elif r.status == "skipped":
                    cat.skipped.append(
                        ProjectResult(
                            project=pkg, message=r.data or "Skipped", command=cmd
                        )
                    )

            report.categories.append(cat)

        return report

    def to_markdown(self) -> str:
        md = [
            "# VALIDATION SUMMARY",
            f"**Time:** {self.timestamp}  ",
            f"**Total:** {self.total} | **Success:** {self.success_count} ✅ | **Failure:** {self.failure_count} ❌ | **Skipped:** {self.skipped_count} ⏭️",
            "",
        ]

        for cat in self.categories:
            md.append(f"## {cat.name}")
            md.append(
                f"**Success:** {cat.success_count} ✅ | **Failure:** {cat.failure_count} ❌ | **Skipped:** {cat.skipped_count} ⏭️\n"
            )

            if cat.successes:
                md.append("#### Successes ✅")
                for r in cat.successes:
                    md.append(f"- **{r.project}**: {r.message}")
                    if r.output and "coverage" in (r.command or "").lower():
                        md.append(f"\n{r.output}\n")
                md.append("")

            if cat.failures:
                md.append("#### Failures ❌")
                for r in cat.failures:
                    md.append(f"- **{r.project}**: {r.message}")
                    if r.output:
                        md.append(f"```text\n{r.output}\n```")
                md.append("")

            if cat.skipped:
                md.append("#### Skipped ⏭️")
                for r in cat.skipped:
                    md.append(f"- **{r.project}**: {r.message}")
                md.append("")

        return "\n".join(md)

    def _collect_all_projects(self) -> set[str]:
        """Collect all unique project names across all categories."""
        projects: set[str] = set()
        for cat in self.categories:
            for r in cat.successes:
                projects.add(r.project)
            for r in cat.failures:
                projects.add(r.project)
            for r in cat.skipped:
                projects.add(r.project)
        return projects

    def _format_timestamp_for_path(self) -> str:
        """Format the report timestamp into a filesystem-safe, human-readable string."""
        try:
            dt = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
            return dt.strftime("%Y-%m-%d_%H-%M-%S")
        except ValueError:
            # Fallback: use current time if parsing fails
            return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    def to_directory_report(
        self,
        output_dir: str,
        validated_repositories: list[str] | None = None,
    ) -> str:
        """Write the validation report as a directory structure with per-repo, per-scan files.

        Directory layout::

            validation-reports-<timestamp>/
            ├── index.md
            ├── <repo-name>-results/
            │   ├── <scan-slug>-<errors>-error(s)-<warnings>-warning(s)-<timestamp>.md
            │   └── ...
            └── ...

        Args:
            output_dir: The parent directory under which the ``validation-reports-<ts>/``
                folder will be created.
            validated_repositories: If provided, the list of repository names that were
                explicitly targeted for validation. ``None`` means all repositories
                were validated.

        Returns:
            The absolute path to the created report directory.
        """
        ts = self._format_timestamp_for_path()
        report_root = os.path.join(output_dir, f"validation-reports-{ts}")
        os.makedirs(report_root, exist_ok=True)

        all_projects = sorted(self._collect_all_projects())

        # --- Per-repo, per-scan files ---
        # Track per-project stats for the index
        project_stats: dict[
            str, dict[str, int]
        ] = {}  # project -> {success, failure, skipped}
        project_files: dict[
            str, list[str]
        ] = {}  # project -> list of relative file paths

        for project in all_projects:
            safe_project = _sanitize_filename(project)
            repo_dir_name = f"{safe_project}-results"
            repo_dir = os.path.join(report_root, repo_dir_name)
            os.makedirs(repo_dir, exist_ok=True)

            project_stats[project] = {"success": 0, "failure": 0, "skipped": 0}
            project_files[project] = []

            for cat in self.categories:
                filtered = cat.for_project(project)
                if filtered.total == 0:
                    continue

                # Accumulate per-project stats
                project_stats[project]["success"] += filtered.success_count
                project_stats[project]["failure"] += filtered.failure_count
                project_stats[project]["skipped"] += filtered.skipped_count

                # Build filename
                slug = CATEGORY_SLUG_MAP.get(
                    cat.name, cat.name.lower().replace(" ", "-")
                )
                safe_slug = _sanitize_filename(slug)
                errors = filtered.failure_count
                warnings = filtered.skipped_count
                filename = (
                    f"{safe_slug}-{errors}-error(s)-{warnings}-warning(s)-{ts}.md"
                )
                filepath = os.path.join(repo_dir, filename)

                # Write the per-scan file
                scan_md = [
                    f"# {project} — {cat.name}",
                    f"**Time:** {self.timestamp}  ",
                    f"**Success:** {filtered.success_count} ✅ | **Failure:** {filtered.failure_count} ❌ | **Skipped:** {filtered.skipped_count} ⏭️",
                    "",
                ]

                if filtered.successes:
                    scan_md.append("## Successes ✅")
                    for r in filtered.successes:
                        scan_md.append(f"- **{r.project}**: {r.message}")
                        if r.output and "coverage" in (r.command or "").lower():
                            scan_md.append(f"\n{r.output}\n")
                    scan_md.append("")

                if filtered.failures:
                    scan_md.append("## Failures ❌")
                    for r in filtered.failures:
                        scan_md.append(f"- **{r.project}**: {r.message}")
                        if r.output:
                            scan_md.append(f"```text\n{r.output}\n```")
                    scan_md.append("")

                if filtered.skipped:
                    scan_md.append("## Skipped ⏭️")
                    for r in filtered.skipped:
                        scan_md.append(f"- **{r.project}**: {r.message}")
                    scan_md.append("")

                try:
                    with open(filepath, "w") as f:
                        f.write("\n".join(scan_md))
                    rel_path = os.path.join(repo_dir_name, filename)
                    project_files[project].append(rel_path)
                except Exception as e:
                    logger.error(f"Failed to write report file {filepath}: {e}")

        # Identify failed projects for the next-command block
        failed_projects = sorted(
            p for p, s in project_stats.items() if s["failure"] > 0
        )

        # --- Index file ---
        index_md = [
            "# 📋 Validation Report",
            f"**Time:** {self.timestamp}  ",
            f"**Total Checks:** {self.total} | **Success:** {self.success_count} ✅ | **Failure:** {self.failure_count} ❌ | **Skipped:** {self.skipped_count} ⏭️",
            "",
        ]

        # --- Next Validation Command block ---
        index_md.extend(
            _build_next_command_block(
                failed_projects=failed_projects,
                validated_repositories=validated_repositories,
            )
        )

        index_md.append("## Category Summary")
        index_md.append("")
        index_md.append("| Category | ✅ Pass | ❌ Fail | ⏭️ Skip |")
        index_md.append("|---|---|---|---|")

        for cat in self.categories:
            index_md.append(
                f"| {cat.name} | {cat.success_count} | {cat.failure_count} | {cat.skipped_count} |"
            )

        index_md.append("")
        index_md.append("## Repository Results")
        index_md.append("")
        index_md.append("| Repository | ✅ Pass | ❌ Fail | ⏭️ Skip | Details |")
        index_md.append("|---|---|---|---|---|")

        for project in all_projects:
            stats = project_stats.get(
                project, {"success": 0, "failure": 0, "skipped": 0}
            )
            files_list = project_files.get(project, [])
            safe_project = _sanitize_filename(project)
            repo_dir_name = f"{safe_project}-results"
            details_link = f"[📂 {project}-results]({repo_dir_name}/)"
            status_icon = "🔴" if stats["failure"] > 0 else "🟢"
            index_md.append(
                f"| {status_icon} {project} | {stats['success']} | {stats['failure']} | {stats['skipped']} | {details_link} |"
            )

        index_md.append("")

        # Detailed per-repo sections with file links
        index_md.append("---")
        index_md.append("")
        index_md.append("## Detailed File Index")
        index_md.append("")

        for project in all_projects:
            files_list = project_files.get(project, [])
            if not files_list:
                continue
            stats = project_stats.get(
                project, {"success": 0, "failure": 0, "skipped": 0}
            )
            status_icon = "🔴" if stats["failure"] > 0 else "🟢"
            index_md.append(f"### {status_icon} {project}")
            for fpath in sorted(files_list):
                fname = os.path.basename(fpath)
                index_md.append(f"- [{fname}]({fpath})")
            index_md.append("")

        index_path = os.path.join(report_root, "index.md")
        try:
            with open(index_path, "w") as f:
                f.write("\n".join(index_md))
        except Exception as e:
            logger.error(f"Failed to write index file {index_path}: {e}")

        # --- Summary file ---
        summary_md = _build_summary_md(
            timestamp=self.timestamp,
            total=self.total,
            success_count=self.success_count,
            failure_count=self.failure_count,
            categories=self.categories,
            failed_projects=failed_projects,
            validated_repositories=validated_repositories,
            skipped_count=self.skipped_count,
        )
        summary_path = os.path.join(report_root, "summary.md")
        try:
            with open(summary_path, "w") as f:
                f.write("\n".join(summary_md))
        except Exception as e:
            logger.error(f"Failed to write summary file {summary_path}: {e}")

        logger.info(f"Validation report written to: {report_root}")
        return report_root


class IncrementalReportWriter:
    """Writes validation report files incrementally as each scan phase completes.

    Usage::

        writer = IncrementalReportWriter(output_dir="/workspace")
        # After installation phase completes:
        writer.write_phase("Ecosystem Installation", install_results)
        # After bump2version phase completes:
        writer.write_phase("Version Metadata Sync (Dry Run)", bump_results)
        # ... more phases ...
        # At the very end, write the index with aggregate stats:
        report_path = writer.finalize()
    """

    def __init__(
        self,
        output_dir: str,
        timestamp: str | None = None,
        validated_repositories: list[str] | None = None,
    ):
        self.timestamp = timestamp or datetime.datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S"
        )
        try:
            dt = datetime.datetime.strptime(self.timestamp, "%Y-%m-%d %H:%M:%S")
            self.ts_path = dt.strftime("%Y-%m-%d_%H-%M-%S")
        except ValueError:
            self.ts_path = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        self.report_root = os.path.join(
            output_dir, f"validation-reports-{self.ts_path}"
        )
        os.makedirs(self.report_root, exist_ok=True)

        # Track whether this run was targeted or full-sweep
        self.validated_repositories = validated_repositories

        # Accumulated state
        self.categories: list[ValidationCategory] = []
        self.project_stats: dict[str, dict[str, int]] = {}
        self.project_files: dict[str, list[str]] = {}

    def write_phase(
        self, category_name: str, results: list[GitResult]
    ) -> ValidationCategory:
        """Build a ValidationCategory from raw results and immediately write per-repo files.

        Args:
            category_name: The display name of the scan category (e.g. "Pre-commit Standard Compliance").
            results: The list of GitResult objects from this phase.

        Returns:
            The constructed ValidationCategory.
        """
        cat = ValidationCategory(
            name=category_name,
            total=len(results),
            success_count=len([r for r in results if r.status == "success"]),
            failure_count=len([r for r in results if r.status == "error"]),
            skipped_count=len([r for r in results if r.status == "skipped"]),
        )

        for r in results:
            pkg = r.metadata.workspace.split("/")[-1] if r.metadata else "unknown"
            cmd = r.metadata.command if r.metadata else None

            if r.status == "success":
                cat.successes.append(
                    ProjectResult(
                        project=pkg, message="Success", output=r.data, command=cmd
                    )
                )
            elif r.status == "error":
                error_msg = r.error.message if r.error else (r.data or "Unknown error")
                output = r.data if r.data and r.data != error_msg else None
                if output and cmd and ("pre_commit" in cmd or "pre-commit" in cmd):
                    output = _filter_pre_commit_output(output)
                cat.failures.append(
                    ProjectResult(
                        project=pkg, message=error_msg, output=output, command=cmd
                    )
                )
            elif r.status == "skipped":
                cat.skipped.append(
                    ProjectResult(project=pkg, message=r.data or "Skipped", command=cmd)
                )

        self.categories.append(cat)
        self._write_category_files(cat)
        return cat

    def _write_category_files(self, cat: ValidationCategory) -> None:
        """Write per-repo files for a single category immediately."""
        all_projects_in_cat: set[str] = set()
        for r in cat.successes:
            all_projects_in_cat.add(r.project)
        for r in cat.failures:
            all_projects_in_cat.add(r.project)
        for r in cat.skipped:
            all_projects_in_cat.add(r.project)

        for project in sorted(all_projects_in_cat):
            filtered = cat.for_project(project)
            if filtered.total == 0:
                continue

            # Ensure project tracking dicts exist
            if project not in self.project_stats:
                self.project_stats[project] = {
                    "success": 0,
                    "failure": 0,
                    "skipped": 0,
                }
            if project not in self.project_files:
                self.project_files[project] = []

            # Accumulate stats
            self.project_stats[project]["success"] += filtered.success_count
            self.project_stats[project]["failure"] += filtered.failure_count
            self.project_stats[project]["skipped"] += filtered.skipped_count

            # Build filename & directory
            safe_project = _sanitize_filename(project)
            repo_dir_name = f"{safe_project}-results"
            repo_dir = os.path.join(self.report_root, repo_dir_name)
            os.makedirs(repo_dir, exist_ok=True)

            slug = CATEGORY_SLUG_MAP.get(cat.name, cat.name.lower().replace(" ", "-"))
            safe_slug = _sanitize_filename(slug)
            errors = filtered.failure_count
            warnings = filtered.skipped_count
            filename = (
                f"{safe_slug}-{errors}-error(s)-{warnings}-warning(s)-{self.ts_path}.md"
            )
            filepath = os.path.join(repo_dir, filename)

            # Write the per-scan file
            scan_md = [
                f"# {project} — {cat.name}",
                f"**Time:** {self.timestamp}  ",
                f"**Success:** {filtered.success_count} ✅ | **Failure:** {filtered.failure_count} ❌ | **Skipped:** {filtered.skipped_count} ⏭️",
                "",
            ]

            if filtered.successes:
                scan_md.append("## Successes ✅")
                for r in filtered.successes:
                    scan_md.append(f"- **{r.project}**: {r.message}")
                    if r.output and "coverage" in (r.command or "").lower():
                        scan_md.append(f"\n{r.output}\n")
                scan_md.append("")

            if filtered.failures:
                scan_md.append("## Failures ❌")
                for r in filtered.failures:
                    scan_md.append(f"- **{r.project}**: {r.message}")
                    if r.output:
                        scan_md.append(f"```text\n{r.output}\n```")
                scan_md.append("")

            if filtered.skipped:
                scan_md.append("## Skipped ⏭️")
                for r in filtered.skipped:
                    scan_md.append(f"- **{r.project}**: {r.message}")
                scan_md.append("")

            try:
                with open(filepath, "w") as f:
                    f.write("\n".join(scan_md))
                rel_path = os.path.join(repo_dir_name, filename)
                self.project_files[project].append(rel_path)
                logger.info(f"Wrote scan report: {filepath}")
            except Exception as e:
                logger.error(f"Failed to write report file {filepath}: {e}")

    def finalize(self) -> str:
        """Write the index.md with aggregate stats and return the report directory path."""
        total = sum(c.total for c in self.categories)
        success_total = sum(c.success_count for c in self.categories)
        failure_total = sum(c.failure_count for c in self.categories)
        skipped_total = sum(c.skipped_count for c in self.categories)

        all_projects = sorted(self.project_stats.keys())

        # Identify failed projects for the next-command block
        failed_projects = sorted(
            p for p, s in self.project_stats.items() if s["failure"] > 0
        )

        index_md = [
            "# 📋 Validation Report",
            f"**Time:** {self.timestamp}  ",
            f"**Total Checks:** {total} | **Success:** {success_total} ✅ | **Failure:** {failure_total} ❌ | **Skipped:** {skipped_total} ⏭️",
            "",
        ]

        # --- Next Validation Command block ---
        index_md.extend(
            _build_next_command_block(
                failed_projects=failed_projects,
                validated_repositories=self.validated_repositories,
            )
        )

        index_md.append("## Category Summary")
        index_md.append("")
        index_md.append("| Category | ✅ Pass | ❌ Fail | ⏭️ Skip |")
        index_md.append("|---|---|---|---|")

        for cat in self.categories:
            index_md.append(
                f"| {cat.name} | {cat.success_count} | {cat.failure_count} | {cat.skipped_count} |"
            )

        index_md.append("")
        index_md.append("## Repository Results")
        index_md.append("")
        index_md.append("| Repository | ✅ Pass | ❌ Fail | ⏭️ Skip | Details |")
        index_md.append("|---|---|---|---|---|")

        for project in all_projects:
            stats = self.project_stats.get(
                project, {"success": 0, "failure": 0, "skipped": 0}
            )
            safe_project = _sanitize_filename(project)
            repo_dir_name = f"{safe_project}-results"
            details_link = f"[📂 {project}-results]({repo_dir_name}/)"
            status_icon = "🔴" if stats["failure"] > 0 else "🟢"
            index_md.append(
                f"| {status_icon} {project} | {stats['success']} | {stats['failure']} | {stats['skipped']} | {details_link} |"
            )

        index_md.append("")

        # Detailed per-repo sections with file links
        index_md.append("---")
        index_md.append("")
        index_md.append("## Detailed File Index")
        index_md.append("")

        for project in all_projects:
            files_list = self.project_files.get(project, [])
            if not files_list:
                continue
            stats = self.project_stats.get(
                project, {"success": 0, "failure": 0, "skipped": 0}
            )
            status_icon = "🔴" if stats["failure"] > 0 else "🟢"
            index_md.append(f"### {status_icon} {project}")
            for fpath in sorted(files_list):
                fname = os.path.basename(fpath)
                index_md.append(f"- [{fname}]({fpath})")
            index_md.append("")

        index_path = os.path.join(self.report_root, "index.md")
        try:
            with open(index_path, "w") as f:
                f.write("\n".join(index_md))
        except Exception as e:
            logger.error(f"Failed to write index file {index_path}: {e}")

        # --- Summary file ---
        total = sum(c.total for c in self.categories)
        success_total = sum(c.success_count for c in self.categories)
        failure_total = sum(c.failure_count for c in self.categories)
        skipped_total = sum(c.skipped_count for c in self.categories)

        summary_md = _build_summary_md(
            timestamp=self.timestamp,
            total=total,
            success_count=success_total,
            failure_count=failure_total,
            categories=self.categories,
            failed_projects=failed_projects,
            validated_repositories=self.validated_repositories,
            skipped_count=skipped_total,
        )
        summary_path = os.path.join(self.report_root, "summary.md")
        try:
            with open(summary_path, "w") as f:
                f.write("\n".join(summary_md))
        except Exception as e:
            logger.error(f"Failed to write summary file {summary_path}: {e}")

        logger.info(f"Validation report finalized at: {self.report_root}")
        return self.report_root
