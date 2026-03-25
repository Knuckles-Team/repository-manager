from pydantic import BaseModel, Field
from typing import Optional, List, Dict


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
    error: Optional[GitError] = None
    metadata: Optional[GitMetadata] = None


class ReadmeResult(BaseModel):
    content: str
    path: str


class RepositoryConfig(BaseModel):
    url: str
    description: Optional[str] = None


class SubdirectoryConfig(BaseModel):
    description: Optional[str] = None
    repositories: List[RepositoryConfig] = Field(default_factory=list)
    subdirectories: Dict[str, "SubdirectoryConfig"] = Field(default_factory=dict)


class MaintenanceUpdate(BaseModel):
    target: Optional[str] = None
    target_pattern: Optional[str] = None
    package: str
    exclude: List[str] = Field(default_factory=list)


class MaintenancePhase(BaseModel):
    name: str
    phase: int
    project: Optional[str] = None
    bulk_bump: bool = False
    updates: List[MaintenanceUpdate] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


class MaintenanceConfig(BaseModel):
    description: Optional[str] = None
    phases: List[MaintenancePhase] = Field(default_factory=list)


class WorkspaceConfig(BaseModel):
    name: str
    path: str
    description: Optional[str] = None
    repositories: List[RepositoryConfig] = Field(default_factory=list)
    subdirectories: Dict[str, SubdirectoryConfig] = Field(default_factory=dict)
    maintenance: Optional[MaintenanceConfig] = None
