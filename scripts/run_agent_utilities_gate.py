"""Run a gate inside this checkout's locked project environment.

Repository-manager's runtime-dependent gates need two things a bare clone
does not have: the locked ``.venv`` and the ``agent-utilities`` checkout that
``pyproject.toml`` installs editable from ``.uv-workspace-siblings``.
``scripts/bootstrap.sh`` provides both (it clones the sibling at the commit
pinned in ``scripts/siblings.lock``). A developer may instead point
``AGENT_UTILITIES_ROOT`` at their own checkout, or keep one next to this
repository's main worktree; the launcher then links it into place.

The project root and working directory stay this repository: a gate must never
run repository-manager tests with an agent-utilities environment (or collect a
sibling checkout's tests). When the environment or the sibling is missing, the
gate reports SKIPPED locally and CANNOT RUN in CI (see ``gate_env.py``).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gate_env import unavailable  # noqa: E402

_SYSTEM_SCRIPTS = {
    Path("scripts/check_no_legacy_markers.py"),
    Path("scripts/check_stubs.py"),
    Path("scripts/mermaid_linter.py"),
    Path("scripts/check_lockfile_version_mirrors.py"),
    Path("scripts/check_lane_guard.py"),
}
_SIBLING = Path(".uv-workspace-siblings") / "agent-utilities"


# Gates that govern the owner's shared multi-worktree host. A Claude Code cloud
# container is a single-writer clone with no lanes, so they have nothing to
# protect there (and the lane guard would refuse every commit).
_CLOUD_EXEMPT = {Path("scripts/check_lane_guard.py")}


class SourceMismatch(Exception):
    """The installed agent-utilities source is not the selected checkout."""


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

    bootstrapped = repository_root / _SIBLING
    if _is_agent_utilities_root(bootstrapped):
        return bootstrapped.resolve()

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
        "no agent-utilities checkout; run scripts/bootstrap.sh or set "
        "AGENT_UTILITIES_ROOT"
    )


def _materialize_agent_utilities_source(
    repository_root: Path, framework_root: Path
) -> None:
    """Materialize the declared editable sibling source for this worktree.

    ``.uv-workspace-siblings`` is ignored by design and is the only generated
    state this helper may create.  A checkout or link that resolves elsewhere
    is refused rather than replaced, because silently selecting a different
    checkout would invalidate the lockfile evidence.
    """

    sibling_root = repository_root / ".uv-workspace-siblings"
    sibling = sibling_root / "agent-utilities"
    expected = framework_root.resolve()
    if sibling.exists() or sibling.is_symlink():
        if sibling.resolve() != expected:
            raise SourceMismatch(
                f"refusing unexpected agent-utilities source at {sibling}; "
                f"expected symlink to {expected}"
            )
        return

    sibling_root.mkdir(parents=True, exist_ok=True)
    sibling.symlink_to(expected, target_is_directory=True)


def _project_python(repository_root: Path) -> Path:
    """The interpreter of the locked environment ``scripts/bootstrap.sh`` syncs."""

    if os.name == "nt":
        return repository_root / ".venv" / "Scripts" / "python.exe"
    return repository_root / ".venv" / "bin" / "python"


def _parse_options() -> argparse.Namespace:
    """Validate a requested module, framework script, or local script gate."""
    parser = argparse.ArgumentParser()
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--module", help="run `python -m MODULE` in the project env")
    target.add_argument(
        "--script", type=Path, help="run an allowlisted agent-utilities script"
    )
    target.add_argument(
        "--local-script", type=Path, help="run a script of this repository"
    )
    parser.add_argument(
        "--system-script",
        action="store_true",
        help="compatibility flag: --script is always an allowlisted static gate",
    )
    parser.add_argument("arguments", nargs=argparse.REMAINDER)
    options = parser.parse_args()
    if options.script is not None and options.script not in _SYSTEM_SCRIPTS:
        parser.error("--script must name an allowlisted agent-utilities gate")
    if options.system_script and options.script is None:
        parser.error("--system-script requires --script")
    if options.arguments[:1] == ["--"]:
        options.arguments = options.arguments[1:]
    return options


def _gate_name(options: argparse.Namespace) -> str:
    if options.module:
        return str(options.module)
    return Path(options.script or options.local_script).stem


def _target(
    options: argparse.Namespace, repository_root: Path
) -> tuple[list[str] | None, str | None]:
    """Return the gate's argv (after the interpreter) or why it cannot run."""

    try:
        framework_root = _agent_utilities_root(repository_root)
        _materialize_agent_utilities_source(repository_root, framework_root)
    except (OSError, subprocess.CalledProcessError) as exc:
        return None, f"cannot locate agent-utilities: {exc}"
    except RuntimeError as exc:
        return None, str(exc)
    if options.module:
        return ["-m", options.module], None
    if options.local_script:
        return [str(repository_root / options.local_script)], None
    script = framework_root / options.script
    if not script.is_file():
        return None, f"agent-utilities has no {options.script} at this pin"
    return [str(script)], None


def main() -> int:
    """Execute the requested gate from this checkout's locked environment."""
    options = _parse_options()
    repository_root = Path(__file__).resolve().parents[1]
    gate = _gate_name(options)
    if options.script in _CLOUD_EXEMPT and os.environ.get("CLAUDE_CODE_REMOTE") == "true":
        print(f"SKIPPED ({gate}): not applicable in a single-writer cloud session")
        return 0

    try:
        target, reason = _target(options, repository_root)
    except SourceMismatch as exc:
        print(f"{gate}: {exc}", file=sys.stderr)
        return 1
    if target is None:
        assert reason is not None
        return unavailable(gate, reason)
    python = _project_python(repository_root)
    if not python.is_file():
        return unavailable(gate, "no project environment; run scripts/bootstrap.sh")

    # A caller's PYTHONPATH can point at a different checkout and make the
    # gate report evidence for the wrong source tree. The project environment
    # is the complete import authority; the venv's bin directory leads PATH so
    # console scripts (pre-commit, bump2version) resolve to the locked ones.
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["VIRTUAL_ENV"] = str(python.parent.parent)
    environment["PATH"] = os.pathsep.join(
        [str(python.parent), environment.get("PATH", "")]
    )
    command = [str(python), *target, *options.arguments]
    return subprocess.run(command, cwd=repository_root, env=environment).returncode


if __name__ == "__main__":
    sys.exit(main())
