from pydantic import BaseModel


class HookResult(BaseModel):
    hook_id: str
    passed: bool
    output: str = ""
    #: Wall-clock seconds pre-commit reported for this hook (parsed from its own
    #: ``--verbose`` ``- duration: <n>s`` line). ``None`` when not measured/parsed.
    duration_s: float | None = None


class RepoScanResult(BaseModel):
    repo_path: str
    success: bool
    exit_code: int
    hooks: list[HookResult] = []
    raw_output: str = ""
    pytest_output: str | None = None
    error: str | None = None
    #: The gate tier this result was produced at ("fast"/"heavy"), or "" when
    #: the result predates tiering / doesn't apply (e.g. skipped, no config).
    stage: str = ""
    #: Total wall-clock seconds for the whole gate invocation.
    duration_s: float = 0.0
