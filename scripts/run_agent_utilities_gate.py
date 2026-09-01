"""Run a framework gate from this checkout's locked project environment.

Repository-manager depends on the live ``agent-utilities`` checkout, but its
worktrees live outside the workspace that uv normally discovers.  The helper
therefore resolves that sibling and materializes the ignored path source that
``pyproject.toml`` declares, then runs uv against *this* repository.  Keeping
the project root and working directory aligned is important: a gate must not
silently run repository-manager tests with an agent-utilities environment (or
collect a sibling checkout's tests).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


def _is_agent_utilities_root(path: Path) -> bool:
    """Return whether *path* contains the framework and its gate scripts."""

    return (
        (path / "pyproject.toml").is_file()
        and (path / "agent_utilities").is_dir()
        and (path / "scripts").is_dir()
    )


def _agent_utilities_root(repository_root: Path) -> Path:
    """Resolve the framework checkout without assuming this worktree's location."""

    configured = os.environ.get("AGENT_UTILITIES_ROOT")
    if configured:
        root = Path(configured).expanduser().resolve()
        if _is_agent_utilities_root(root):
            return root
        raise RuntimeError("AGENT_UTILITIES_ROOT is not an agent-utilities checkout")

    result = subprocess.run(
        ["git", "-C", str(repository_root), "worktree", "list", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        if not line.startswith("worktree "):
            continue
        worktree = Path(line.removeprefix("worktree ")).resolve()
        candidate = worktree.parent.parent / "agent-utilities"
        if _is_agent_utilities_root(candidate):
            return candidate

    raise RuntimeError(
        "Cannot locate an agent-utilities checkout. Set AGENT_UTILITIES_ROOT to the "
        "framework checkout before running pre-commit."
    )


def _materialize_agent_utilities_source(
    repository_root: Path, framework_root: Path
) -> None:
    """Materialize the declared editable sibling source for this worktree.

    ``.uv-workspace-siblings`` is ignored by design and is the only generated
    state this helper may create.  A pre-existing directory, wrong symlink, or
    broken link is refused rather than replaced, because silently selecting a
    different checkout would invalidate the lockfile evidence.
    """

    sibling_root = repository_root / ".uv-workspace-siblings"
    sibling = sibling_root / "agent-utilities"
    expected = framework_root.resolve()
    if sibling.exists() or sibling.is_symlink():
        if not sibling.is_symlink() or sibling.resolve() != expected:
            raise RuntimeError(
                f"refusing unexpected agent-utilities source at {sibling}; "
                f"expected symlink to {expected}"
            )
        return

    sibling_root.mkdir(parents=True, exist_ok=True)
    sibling.symlink_to(expected, target_is_directory=True)


def _build_command(
    uv: str,
    repository_root: Path,
    framework_root: Path,
    *,
    module: str | None,
    script: Path | None,
    extras: list[str],
    arguments: list[str],
) -> list[str]:
    """Build the locked command while keeping its project root explicit."""

    command = [uv, "run", "--project", str(repository_root), "--locked"]
    for extra in extras:
        command.extend(["--extra", extra])
    command.append("python")
    if module:
        command.extend(["-m", module])
    else:
        assert script is not None
        command.append(str(framework_root / script))
    if arguments[:1] == ["--"]:
        arguments = arguments[1:]
    command.extend(arguments)
    return command


def main() -> int:
    """Execute a framework module or script in this project's locked environment."""

    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module")
    target.add_argument("--script", type=Path)
    parser.add_argument(
        "--extra",
        action="append",
        default=[],
        help="select a repository extra for the locked invocation (repeatable)",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[1]
    try:
        framework_root = _agent_utilities_root(repository_root)
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))

    try:
        _materialize_agent_utilities_source(repository_root, framework_root)
    except OSError as exc:
        parser.error(f"cannot materialize agent-utilities source: {exc}")

    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required to run Agent Utilities pre-commit gates")

    script = None
    if options.script:
        script = framework_root / options.script
        if not script.is_file():
            parser.error(f"Agent Utilities script was not found: {options.script}")
    command = _build_command(
        uv,
        repository_root,
        framework_root,
        module=options.module,
        script=script,
        extras=options.extra,
        arguments=options.arguments,
    )

    # A caller's PYTHONPATH can point at a different checkout and make a
    # hermetic locked gate report evidence for the wrong source tree.  The
    # project cwd and editable dependency above are the complete import
    # authority; inherited path injection is deliberately not accepted.
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    return subprocess.run(command, cwd=repository_root, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())
