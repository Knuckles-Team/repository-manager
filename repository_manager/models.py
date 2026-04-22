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
    bulk_bump: bool = False
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
