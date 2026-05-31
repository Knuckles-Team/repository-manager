from pydantic import BaseModel


class HookResult(BaseModel):
    hook_id: str
    passed: bool
    output: str = ""


class RepoScanResult(BaseModel):
    repo_path: str
    success: bool
    exit_code: int
    hooks: list[HookResult] = []
    raw_output: str = ""
    pytest_output: str | None = None
    error: str | None = None
