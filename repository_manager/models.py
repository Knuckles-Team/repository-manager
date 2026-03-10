from pydantic import BaseModel
from typing import Optional


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
