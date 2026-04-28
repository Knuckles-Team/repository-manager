import datetime
from pydantic import BaseModel, Field


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
<<<<<<< HEAD
    error: GitError | None = None
    metadata: GitMetadata | None = None
=======
>>>>>>> 61af4a3 (Fixed several issues.)


class ReadmeResult(BaseModel):
    content: str
    path: str


class RepositoryConfig(BaseModel):
    url: str
    description: str | None = None
<<<<<<< HEAD
    description: str | None = None
=======
>>>>>>> 61af4a3 (Fixed several issues.)


class SubdirectoryConfig(BaseModel):
    description: str | None = None
    repositories: list[RepositoryConfig] = Field(default_factory=list)
    subdirectories: dict[str, "SubdirectoryConfig"] = Field(default_factory=dict)
<<<<<<< HEAD
    description: str | None = None
    repositories: list[RepositoryConfig] = Field(default_factory=list)
    subdirectories: dict[str, "SubdirectoryConfig"] = Field(default_factory=dict)
=======
>>>>>>> 61af4a3 (Fixed several issues.)


class MaintenanceUpdate(BaseModel):
    target: str | None = None
    target_pattern: str | None = None
    package: str
    exclude: list[str] = Field(default_factory=list)


class MaintenancePhase(BaseModel):
    name: str
    phase: int
    project: str | None = None
<<<<<<< HEAD
    project: str | None = None
    bulk_bump: bool = False
    updates: list[MaintenanceUpdate] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
    updates: list[MaintenanceUpdate] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
=======
    bulk_bump: bool = False
    updates: list[MaintenanceUpdate] = Field(default_factory=list)
    exclude: list[str] = Field(default_factory=list)
>>>>>>> 61af4a3 (Fixed several issues.)


class MaintenanceConfig(BaseModel):
    description: str | None = None
    phases: list[MaintenancePhase] = Field(default_factory=list)
<<<<<<< HEAD
    description: str | None = None
    phases: list[MaintenancePhase] = Field(default_factory=list)
=======
>>>>>>> 61af4a3 (Fixed several issues.)


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
