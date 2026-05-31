import subprocess
import os
import re
from typing import List
from .scan_models import HookResult, RepoScanResult


def run_pre_commit(
    repo_path: str, timeout: int = 600, skip_pytest: bool = False
) -> subprocess.CompletedProcess:
    """Runs a single pass of pre-commit."""
    env = os.environ.copy()
    if "SKIP" in env:
        env["SKIP"] += ",no-commit-to-branch"
    else:
        env["SKIP"] = "no-commit-to-branch"

    if skip_pytest:
        env["SKIP"] += ",pytest"

    return subprocess.run(
        ["pre-commit", "run", "--all-files", "--verbose"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def parse_pre_commit_output(output: str) -> List[HookResult]:
    """Parse pre-commit verbose output into structured HookResults."""
    lines = output.split("\n")
    hooks = []

    current_hook_name = None
    current_status = None
    current_output: List[str] = []

    # Pre-commit output line structure: `Hook Name.................................................................Passed`
    hook_regex = re.compile(r"^(.+?)\.{5,}(Passed|Failed|Skipped)\s*$")

    for line in lines:
        match = hook_regex.match(line.strip())
        if match:
            # Save the previous hook
            if current_hook_name:
                hooks.append(
                    HookResult(
                        hook_id=current_hook_name.strip(),
                        passed=(
                            current_status == "Passed" or current_status == "Skipped"
                        ),
                        output="\n".join(current_output).strip()
                        if current_status == "Failed"
                        else "",
                    )
                )

            # Start new hook
            current_hook_name = match.group(1).strip().rstrip(".")
            current_status = match.group(2)
            current_output = []
        else:
            if current_hook_name:
                current_output.append(line)

    # Save the last hook
    if current_hook_name:
        hooks.append(
            HookResult(
                hook_id=current_hook_name.strip(),
                passed=(current_status == "Passed" or current_status == "Skipped"),
                output="\n".join(current_output).strip()
                if current_status == "Failed"
                else "",
            )
        )

    return hooks


def scan_repository(repo_path: str) -> RepoScanResult:
    """Scans a single repository by running pre-commit twice to avoid auto-fix false positives."""
    if not os.path.exists(os.path.join(repo_path, ".pre-commit-config.yaml")):
        return RepoScanResult(
            repo_path=repo_path,
            success=True,
            exit_code=0,
            hooks=[],
            error="No .pre-commit-config.yaml found.",
        )

    try:
        # First Pass (Allows formatters to apply fixes, skips slow tests)
        run_pre_commit(repo_path, skip_pytest=True)

        # Second Pass (Captures actual failures)
        result = run_pre_commit(repo_path)

        hooks = parse_pre_commit_output(result.stdout + "\n" + result.stderr)

        success = result.returncode == 0

        return RepoScanResult(
            repo_path=repo_path,
            success=success,
            exit_code=result.returncode,
            hooks=hooks,
            raw_output=result.stdout + "\n" + result.stderr,
        )
    except subprocess.TimeoutExpired as e:
        return RepoScanResult(
            repo_path=repo_path,
            success=False,
            exit_code=-1,
            error=f"Timeout expired during pre-commit run: {str(e)}",
        )
    except Exception as e:
        return RepoScanResult(
            repo_path=repo_path,
            success=False,
            exit_code=-1,
            error=f"Error executing scan: {str(e)}",
        )
