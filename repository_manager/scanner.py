import os
import re
import subprocess

from .scan_models import HookResult, RepoScanResult


def run_pre_commit(
    repo_path: str,
    timeout: int = 600,
    skip_pytest: bool = False,
    files: list[str] | None = None,
) -> subprocess.CompletedProcess:
    """Runs a single pass of pre-commit.

    ``files`` (when given and non-empty) scopes the per-file hooks to just those
    paths via ``--files`` instead of ``--all-files`` — much faster on large
    repos. ``always_run`` hooks (the guardrail gates) still run regardless, so
    nothing is skipped, only narrowed. Empty/None ``files`` runs ``--all-files``.
    """
    env = os.environ.copy()
    if "SKIP" in env:
        env["SKIP"] += ",no-commit-to-branch"
    else:
        env["SKIP"] = "no-commit-to-branch"

    if skip_pytest:
        env["SKIP"] += ",pytest"

    scope = ["--files", *files] if files else ["--all-files"]

    return subprocess.run(  # nosec B603 B607
        ["pre-commit", "run", *scope, "--verbose"],
        cwd=repo_path,
        capture_output=True,
        text=True,
        timeout=timeout,
        env=env,
    )


def parse_pre_commit_output(output: str) -> list[HookResult]:
    """Parse pre-commit verbose output into structured HookResults."""
    lines = output.split("\n")
    hooks = []

    current_hook_name = None
    current_status = None
    current_output: list[str] = []

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

        raw_output = result.stdout + "\n" + result.stderr
        hooks = parse_pre_commit_output(raw_output)

        success = result.returncode == 0

        # When pre-commit exits non-zero but produced NO parseable failing hook
        # lines (e.g. it errored during environment install, hit an invalid
        # config, or failed before any hook ran), the real diagnostic lives only
        # in ``raw_output``. Surface it as ``error`` so the failure is never
        # reported as "failed with empty failures" — the actual reason is shown.
        error: str | None = None
        if not success and not any(not h.passed for h in hooks):
            tail = "\n".join(raw_output.strip().splitlines()[-40:]).strip()
            error = (
                "pre-commit exited "
                f"{result.returncode} with no parseable hook results "
                f"(likely an init/config/environment error):\n{tail}"
                if tail
                else f"pre-commit exited {result.returncode} with no output."
            )

        return RepoScanResult(
            repo_path=repo_path,
            success=success,
            exit_code=result.returncode,
            hooks=hooks,
            raw_output=raw_output,
            error=error,
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
