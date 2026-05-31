from pydantic import BaseModel
from typing import List, Optional


class HookResult(BaseModel):
    hook_id: str
    passed: bool
    output: str = ""


class RepoScanResult(BaseModel):
    repo_path: str
    success: bool
    exit_code: int
    hooks: List[HookResult] = []
    raw_output: str = ""
    pytest_output: Optional[str] = None
    error: Optional[str] = None
