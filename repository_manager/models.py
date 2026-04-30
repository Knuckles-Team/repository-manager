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
}


def _sanitize_filename(name: str) -> str:
    """Sanitize a string to be a valid Windows and cross-platform filename."""
    # Replace illegal Windows characters with a hyphen
    return re.sub(r'[<>:"/\\|?*]', "-", name)


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
                        ProjectResult(project=pkg, message="Success", command=cmd)
                    )
                elif r.status == "error":
                    error_msg = (
                        r.error.message if r.error else (r.data or "Unknown error")
                    )
                    output = r.data if r.data and r.data != error_msg else None
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

    def to_directory_report(self, output_dir: str) -> str:
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

        # --- Index file ---
        index_md = [
            "# 📋 Validation Report",
            f"**Time:** {self.timestamp}  ",
            f"**Total Checks:** {self.total} | **Success:** {self.success_count} ✅ | **Failure:** {self.failure_count} ❌ | **Skipped:** {self.skipped_count} ⏭️",
            "",
            "## Category Summary",
            "",
            "| Category | ✅ Pass | ❌ Fail | ⏭️ Skip |",
            "|---|---|---|---|",
        ]

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

    def __init__(self, output_dir: str, timestamp: str | None = None):
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
                    ProjectResult(project=pkg, message="Success", command=cmd)
                )
            elif r.status == "error":
                error_msg = r.error.message if r.error else (r.data or "Unknown error")
                output = r.data if r.data and r.data != error_msg else None
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

        index_md = [
            "# 📋 Validation Report",
            f"**Time:** {self.timestamp}  ",
            f"**Total Checks:** {total} | **Success:** {success_total} ✅ | **Failure:** {failure_total} ❌ | **Skipped:** {skipped_total} ⏭️",
            "",
            "## Category Summary",
            "",
            "| Category | ✅ Pass | ❌ Fail | ⏭️ Skip |",
            "|---|---|---|---|",
        ]

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

        logger.info(f"Validation report finalized at: {self.report_root}")
        return self.report_root
