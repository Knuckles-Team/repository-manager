#!/usr/bin/env python


"""
A command-line tool for managing Git repositories, supporting cloning and pulling
multiple repositories in parallel using Python's multiprocessing capabilities.
"""

import contextlib
import dataclasses
import datetime
import fnmatch
import functools
import inspect
import os
import re
import shlex
import subprocess
import sys
import threading
import tomllib
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TypeVar

__version__ = "3.4.0"

__all__ = [
    "Git",
    "WorkspaceManifestError",
    "main",
    "synchronize_workspace_manifest",
    "_run_build_queue_cli",
    "_run_lane_cli",
    "_run_merge_queue_cli",
]

import concurrent.futures
import multiprocessing
import shutil
import signal

import yaml  # type: ignore[import-untyped]
from agent_utilities.base_utilities import get_library_file_path, to_boolean

try:
    from skill_graphs.skill_graph_utilities import get_skill_graphs_path
    from universal_skills.skill_utilities import get_universal_skills_path
except ImportError:
    get_universal_skills_path = None
    get_skill_graphs_path = None

from importlib.resources import files

from agent_utilities.base_utilities import get_logger

from repository_manager import dependency_readiness
from repository_manager.canonical_guard import guarded_canonical_mutation
from repository_manager.gates import HOOK_STAGE_BY_GATE_STAGE, run_gate_stage
from repository_manager.models import (
    GitError,
    GitMetadata,
    GitResult,
    MaintenanceConfig,
    ReadmeResult,
    SubdirectoryConfig,
    WorkspaceConfig,
)
from repository_manager.scan_models import RepoScanResult
from repository_manager.workspace_manifest import (
    WorkspaceManifestError,
    synchronize_workspace_manifest,
)

logger = get_logger("RepositoryManager")

_UNRESOLVED_ENV_REFERENCE = re.compile(
    r"\$\{[A-Za-z_][A-Za-z0-9_]*\}|\$[A-Za-z_][A-Za-z0-9_]*"
)
_DIAGNOSTIC_ENDPOINT = re.compile(r"(?i)\b(?:https?|ssh)://[^\s]+|\bgit@[^\s:]+:[^\s]+")
_DIAGNOSTIC_SECRET = re.compile(
    r"(?i)\b(?:access[_-]?token|api[_-]?key|authorization|client[_-]?secret|"
    r"password|refresh[_-]?token|secret|token)\s*[:=]\s*[^\s,;]+"
)
_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=(.*)$", re.DOTALL)
_SHELL_CONTROL_TOKENS = {"&&", "||", ";", "|", "&", "(", ")"}
_MAX_CAPTURED_OUTPUT_BYTES = 1024 * 1024
_MutationResult = TypeVar("_MutationResult")
_UV_WORKSPACE_SIBLINGS_DIRNAME = ".uv-workspace-siblings"
_PEP503_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_PEP503_NAME_SEPARATORS = re.compile(r"[-_.]+")

# Keep this list in sync with uv's documented ``tool.uv.sources`` table
# fields.  Parsing source tables structurally, rather than looking only for
# ``path``, is important here: a malformed extra field or a path/remote
# combination must fail before the materializer changes the checkout.
_UV_SOURCE_KEYS = frozenset(
    {
        "branch",
        "editable",
        "extra",
        "git",
        "index",
        "lfs",
        "marker",
        "package",
        "path",
        "rev",
        "subdirectory",
        "tag",
        "url",
        "workspace",
    }
)
_UV_SOURCE_PRIMARY_KEYS = frozenset({"git", "index", "path", "url", "workspace"})
_UV_SOURCE_SELECTOR_KEYS = frozenset({"branch", "rev", "tag"})
_UV_SOURCE_STRING_KEYS = frozenset(
    {
        "branch",
        "extra",
        "git",
        "index",
        "marker",
        "package",
        "path",
        "rev",
        "subdirectory",
        "tag",
        "url",
    }
)


class _RepoMutationLock:
    """One re-entrant repository lock plus its holder/waiter reference count."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.users = 0


_REPO_MUTATION_LOCKS: dict[str, _RepoMutationLock] = {}
_REPO_MUTATION_LOCKS_GUARD = threading.Lock()


@contextlib.contextmanager
def _hold_repo_mutation(path: str) -> Iterator[None]:
    """Serialize mutation of one resolved repository without leaking lock keys."""
    key = os.path.realpath(path)
    with _REPO_MUTATION_LOCKS_GUARD:
        entry = _REPO_MUTATION_LOCKS.setdefault(key, _RepoMutationLock())
        entry.users += 1
    acquired = False
    try:
        entry.lock.acquire()
        acquired = True
        yield
    finally:
        if acquired:
            entry.lock.release()
        with _REPO_MUTATION_LOCKS_GUARD:
            entry.users -= 1
            if entry.users == 0 and _REPO_MUTATION_LOCKS.get(key) is entry:
                del _REPO_MUTATION_LOCKS[key]


def _exclusive_repo_mutation(
    method: Callable[..., _MutationResult],
) -> Callable[..., _MutationResult]:
    """Hold one repo lock across the complete decorated mutation method."""
    method_signature = inspect.signature(method)

    @functools.wraps(method)
    def wrapped(*args: Any, **kwargs: Any) -> _MutationResult:
        bound = method_signature.bind(*args, **kwargs)
        bound.apply_defaults()
        manager = args[0]
        if "target_path" in bound.arguments:
            # clone_repository's contract names this as the destination path,
            # already relative to the caller's cwd when it is not absolute. Do
            # not feed a workspace-prefixed target through ``_resolve_path`` a
            # second time (``workspace/workspace/repo``).
            target_path = os.path.abspath(
                os.path.expanduser(str(bound.arguments["target_path"]))
            )
        else:
            target_path = manager._resolve_path(bound.arguments.get("path"))
        with _hold_repo_mutation(target_path):
            return method(*args, **kwargs)

    return wrapped


def _privacy_safe_diagnostic(value: object) -> str:
    """Sanitize command output before returning or persisting it."""

    try:
        from agent_utilities.security.persistence_privacy import (
            sanitize_for_persistence,
        )

        clean, _ = sanitize_for_persistence(str(value or ""))
    except Exception:
        return "repository operation output withheld"
    clean = _DIAGNOSTIC_ENDPOINT.sub("[REDACTED_ENDPOINT]", str(clean))
    return _DIAGNOSTIC_SECRET.sub("[REDACTED_SECRET]", clean)


#: (executable-basename, subcommand) -> label. Both positions are STRUCTURAL
#: (argv[0] and argv[1]) — never a free-form argument value.
_STRUCTURAL_OPERATION_LABELS: dict[tuple[str, str], str] = {
    ("git", "clone"): "git clone",
    ("git", "pull"): "git pull",
    ("git", "push"): "git push",
    ("git", "status"): "git status",
    ("git", "commit"): "git commit",
    ("git", "checkout"): "git checkout",
    ("git", "diff"): "git diff",
    ("git", "rev-parse"): "git rev-parse",
    ("pip", "install"): "pip install",
    ("uv", "sync"): "uv sync",
    ("pre-commit", "run"): "pre-commit run",
}

#: Executable-basename alone is enough — no subcommand position exists.
_SINGLE_TOKEN_OPERATION_LABELS: dict[str, str] = {
    "bump2version": "bump2version",
    "pytest": "pytest",
}

#: `python -m <module>` invocations, keyed by the module path (argv[2]).
_MODULE_OPERATION_LABELS: dict[str, str] = {
    "repository_manager.mcp_server": "mcp_server --help",
    "repository_manager.agent_server": "agent_server --help",
}


def _operation_label(command_argv: list[str]) -> str:
    """Classify a command from its PARSED executable + structural subcommand
    position only — never by scanning free-form argument text (D-CDX-6).

    Confirmed live: a commit message reading 'fix(pre-commit): preserve lane
    pytest partition' made ``git commit -m '<that message>'`` classify as
    ``pytest`` — the old implementation lowercased the WHOLE command string
    and returned the first known label found anywhere in it, so any argument
    value (a commit message, a file path, a branch name) could spoof a
    different operation's label, corrupting provenance/metrics/policy keyed
    on the classification. Only ``command_argv[0]`` (the executable) and,
    where relevant, ``command_argv[1]`` (a git subcommand or ``-m`` module
    path) are ever consulted — never any later token, which is exactly where
    a commit message or other adversarial argument value lives.
    """
    if not command_argv:
        return "repository operation"
    exe = os.path.basename(command_argv[0]).lower()
    rest = command_argv[1:]
    sub = os.path.basename(rest[0]).lower() if rest else ""

    label = _STRUCTURAL_OPERATION_LABELS.get((exe, sub))
    if label:
        return label
    if exe in _SINGLE_TOKEN_OPERATION_LABELS:
        return _SINGLE_TOKEN_OPERATION_LABELS[exe]
    if exe in ("python", "python3") and len(rest) >= 2 and rest[0] == "-m":
        module_label = _MODULE_OPERATION_LABELS.get(rest[1].lower())
        if module_label:
            return module_label
    return "repository operation"


def _project_label(path: object) -> str:
    """Return a logical project label without retaining its filesystem path."""

    candidate = Path(str(path or "")).name
    clean = _privacy_safe_diagnostic(candidate).strip()
    if not clean or "[REDACTED_" in clean:
        return "configured-workspace"
    return re.sub(r"[^A-Za-z0-9._-]+", "-", clean).strip("-") or "configured-workspace"


def _uv_extra_flag(extra: str | None) -> str:
    """Render an ``extra`` selection (from `install_projects`) as a `uv
    sync`/`uv_workspace.py sync` CLI flag suffix."""
    if extra == "all":
        return " --all-extras"
    if extra:
        return f" --extra {shlex.quote(extra)}"
    return ""


def _build_install_report_markdown(
    results: list[GitResult],
    successes: list[GitResult],
    failures: list[GitResult],
) -> str:
    """Render `install_projects`'s human-readable summary report."""
    report_md = "# INSTALLATION SUMMARY\n"
    report_md += (
        f"**Time:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    )
    report_md += f"**Total:** {len(results)} | **Success:** {len(successes)} ✅ | **Failure:** {len(failures)} ❌\n\n"

    if successes:
        report_md += "## Successes ✅\n"
        for r in successes:
            pkg = "unknown"
            if r.metadata:
                pkg = r.metadata.workspace.split("/")[-1]
            report_md += f"- **{pkg}**: Installation success\n"

    if failures:
        report_md += "\n## Failures ❌\n"
        for r in failures:
            pkg = "unknown"
            if r.metadata:
                pkg = r.metadata.workspace.split("/")[-1]
            error_msg = r.error.message if r.error else r.data
            report_md += f"- **{pkg}**: {error_msg}\n"

    return report_md


def _expand_required_environment(value: str, *, label: str) -> str:
    """Expand a portable config value and fail before leaking an unresolved token."""

    expanded = os.path.expandvars(value)
    if _UNRESOLVED_ENV_REFERENCE.search(expanded):
        raise ValueError(f"{label} environment reference is unresolved")
    return expanded


def get_packaged_file_path(package: str, file: str) -> str:
    """Robustly find a file in a package using importlib.resources."""
    try:
        path = files(package).joinpath(file)
        if path.is_file():
            return str(path)
    except Exception:  # nosec B110
        pass

    local_path = os.path.join(os.path.dirname(__file__), file)
    if os.path.exists(local_path):
        return local_path

    return get_library_file_path(file=file)


# Robust environment variable retrieval with empty string fallbacks
_raw_workspace = os.getenv("REPOSITORY_MANAGER_WORKSPACE", "")
_portable_workspace = os.getenv("AGENT_UTILITIES_WORKSPACE_ROOT", "")
DEFAULT_REPOSITORY_MANAGER_WORKSPACE = os.path.abspath(
    os.path.expanduser(_raw_workspace or _portable_workspace or os.getcwd())
)

_raw_yml = os.getenv("WORKSPACE_YML", "")
DEFAULT_WORKSPACE_YML = (
    _raw_yml
    if _raw_yml
    else get_packaged_file_path("repository_manager", "workspace.yml")
)

_raw_threads = os.getenv("REPOSITORY_MANAGER_THREADS", "")
DEFAULT_REPOSITORY_MANAGER_THREADS = int(
    _raw_threads if _raw_threads and _raw_threads.isdigit() else "6"
)

_raw_branch = os.getenv("REPOSITORY_MANAGER_DEFAULT_BRANCH", "")
DEFAULT_REPOSITORY_MANAGER_DEFAULT_BRANCH = to_boolean(
    _raw_branch if _raw_branch else "False"
)

# D-EGK-2 / D-EGK-1 (see reports/deferred/eg-kernel-0802.md): a canonical-checkout
# refresh is the operation that can destroy epistemic_graph's compiled numeric
# kernel (a gitignored .so nothing else regenerates) and it is the moment right
# after which the entire *-mcp fleet's live hostPath mounts should be re-verified
# -- both trees are hostPath-mounted straight into every pod, so "refreshed the
# checkout" and "changed what every pod reads" are the same event here. Run both
# checks best-effort after every pull_projects() batch: never let a check failure
# break the pull itself, and never raise past this function.
_MOUNT_CHECK_TIMEOUT_S = 90


def _run_post_hydration_mount_checks(workspace_root: str) -> None:
    """Best-effort: log loudly (never raise) if a post-pull mount check fails."""

    checks = [
        (
            "D-EGK-2 mounted-kernel check",
            Path(workspace_root)
            / "agent-packages"
            / "epistemic-graph"
            / "scripts"
            / "check_mounted_kernel.py",
            [],
        ),
        (
            "D-EGK-1 python-mount-parity check",
            Path(workspace_root) / "scripts" / "check_python_mount_parity.py",
            ["--mode", "live"],
        ),
    ]
    for label, script, extra_args in checks:
        if not script.is_file():
            continue  # workspace doesn't carry this tree/tooling -- nothing to check
        try:
            result = subprocess.run(
                [sys.executable, str(script), *extra_args],
                capture_output=True,
                text=True,
                timeout=_MOUNT_CHECK_TIMEOUT_S,
                check=False,
            )
        except Exception as exc:  # noqa: BLE001 - a check must never break the pull
            logger.warning(f"{label} could not run after hydration: {exc}")
            continue
        if result.returncode == 0:
            logger.info(f"{label}: passed")
        else:
            logger.critical(
                f"{label} FAILED after this hydration -- {result.stdout.strip()[-2000:]}"
            )


#: Patterns removed anywhere under a `cleanup_artifacts` target dir.
_CLEANUP_FILE_PATTERNS = [
    "knowledge_graph.db*",
    "*.db-wal",
    "*.db-shm",
    "*.wal",
    "*.log",
    "session<MagicMock*",
    "coverage.xml",
    ".coverage",
    "*.orig",
    "*.rej",
    "*.patch",
    "failed_tests.txt",
    "pytest_errors.txt",
    "pytest_output.txt",
    "mypy_errors.txt",
    "mypy_output.txt",
    "pre-commit-out.txt",
    "cargo_check.log",
    "check.log",
    "check_out.txt",
    "test_out.txt",
    "trace.txt",
]

#: Directory names removed (whole subtree) anywhere under a `cleanup_artifacts`
#: target dir.
_CLEANUP_DIR_PATTERNS = {
    ".pytest_cache",
    "htmlcov",
    "agent_data",
}

#: Directories `cleanup_artifacts` never descends into.
_CLEANUP_IGNORED_DIRS = {".venv", "node_modules", ".git"}

#: Transient script filename patterns `cleanup_artifacts` removes, but ONLY
#: at the target dir's own root (never in subdirectories).
_CLEANUP_ROOT_SCRIPT_PATTERNS = [
    "test_*.py",
    "fix_*.py",
    "debug_*.py",
    "scratch_*.py",
    "temp_*.py",
]


def _cleanup_matched_dirs(dirpath: str, dirnames: list[str]) -> None:
    """Remove (and un-descend into) any directory in *dirnames* that matches
    `_CLEANUP_DIR_PATTERNS`."""
    for d in list(dirnames):
        if d in _CLEANUP_DIR_PATTERNS:
            full_path = os.path.join(dirpath, d)
            try:
                shutil.rmtree(full_path)
                logger.debug("Cleaned up managed directory")
            except Exception as e:
                logger.debug("Operation failed: error_type=%s", type(e).__name__)
            dirnames.remove(d)


def _cleanup_root_transient_script(file_path: Path) -> bool:
    """Remove *file_path* if it matches a root-level transient script pattern.

    Returns True if it matched (and cleanup was attempted), so the caller
    can skip the non-standard-``.txt`` check for the same file.
    """
    for pat in _CLEANUP_ROOT_SCRIPT_PATTERNS:
        if file_path.match(pat):
            try:
                file_path.unlink()
                logger.info(f"Cleaned up root transient script: {file_path}")
            except Exception as e:
                logger.debug(
                    "Root-script cleanup failed: error_type=%s", type(e).__name__
                )
            return True
    return False


def _cleanup_root_nonstandard_txt(file_path: Path) -> None:
    """Remove *file_path* if it's a root-level ``.txt`` file other than the
    two standard requirements files."""
    if file_path.suffix == ".txt" and file_path.name not in (
        "requirements.txt",
        "requirements-dev.txt",
    ):
        try:
            file_path.unlink()
            logger.info(f"Cleaned up root non-standard text file: {file_path}")
        except Exception as e:
            logger.debug("Root-text cleanup failed: error_type=%s", type(e).__name__)


def _cleanup_root_level_files(dirpath: str, filenames: list[str]) -> None:
    """Root-only cleanup pass: transient scripts, then non-standard ``.txt``."""
    for f in filenames:
        file_path = Path(os.path.join(dirpath, f))
        if _cleanup_root_transient_script(file_path):
            continue
        _cleanup_root_nonstandard_txt(file_path)


def _cleanup_matched_files(dirpath: str, filenames: list[str]) -> None:
    """Remove any file in *filenames* matching `_CLEANUP_FILE_PATTERNS`."""
    for f in filenames:
        file_path = Path(os.path.join(dirpath, f))
        for pat in _CLEANUP_FILE_PATTERNS:
            if file_path.match(pat):
                try:
                    file_path.unlink()
                    logger.debug("Cleaned up managed file")
                except Exception as e:
                    logger.debug("Operation failed: error_type=%s", type(e).__name__)
                break


@dataclasses.dataclass
class _PhaseProgress:
    """Progress bookkeeping for one phased (bump / push) run.

    Owns the ``progress is not None`` guard and the per-item counters that the
    phased bump and phased push workflows both maintain, so those workflows
    read as the phase topology they actually are.

    ``state`` is the caller-supplied progress mapping (``None`` disables every
    update); ``noun`` is the verb used in the per-item completion log line.
    """

    state: dict | None
    noun: str
    total: int = 0
    processed: int = 0

    def initialize(self, heading: str, phases: list[tuple[str, list[str]]]) -> None:
        """Seed the per-phase counters for every phase about to run."""
        if self.state is None:
            return
        self.state["current_phase"] = heading
        self.state["progress"] = 0
        self.state["phases"] = {}
        for name, items in phases:
            self.state["phases"][name] = {
                "status": "pending",
                "total": len(items),
                "processed": 0,
                "completed": 0,
                "success": 0,
                "failed": 0,
                "details": dict.fromkeys(items, "pending"),
                "repos": dict.fromkeys(items, "pending"),
            }

    def nothing_to_do(self, heading: str) -> None:
        """Mark the run complete without any phase having run."""
        if self.state is None:
            return
        self.state["current_phase"] = heading
        self.state["progress"] = 100
        self.state["phases"] = {}

    def note(self, heading: str) -> None:
        """Update only the human-readable current-phase banner."""
        if self.state is None:
            return
        self.state["current_phase"] = heading

    def begin_phase(self, phase_name: str) -> None:
        if self.state is None:
            return
        self.state["current_phase"] = f"{phase_name} in progress"
        self.state["phases"][phase_name]["status"] = "running"

    def end_phase(self, phase_name: str) -> None:
        if self.state is None:
            return
        self.state["phases"][phase_name]["status"] = "completed"

    def begin_item(self, phase_name: str, item: str) -> None:
        if self.state is None:
            return
        phase = self.state["phases"][phase_name]
        phase["details"][item] = "running"
        phase["repos"][item] = "running"

    def finish_item(self, phase_name: str, item: str, status_str: str) -> None:
        """Record one project's terminal status and advance the overall percentage."""
        if self.state is None:
            return
        phase = self.state["phases"][phase_name]
        phase["details"][item] = status_str
        phase["repos"][item] = status_str
        phase["processed"] += 1
        phase["completed"] += 1
        phase["success" if status_str == "success" else "failed"] += 1

        self.processed += 1
        percent = int((self.processed / self.total) * 100)
        self.state["progress"] = percent
        logger.info(
            f"[{self.processed}/{self.total}] ({percent}%) "
            f"Completed {self.noun} for {item}: {status_str}"
        )

    def finish(self, heading: str) -> None:
        if self.state is None:
            return
        self.state["current_phase"] = heading
        self.state["progress"] = 100


class Git:
    """A class to handle Git operations such as cloning and pulling repositories."""

    def __init__(
        self,
        path: str | None = None,
        threads: int | None = None,
        set_to_default_branch: bool = False,
        capture_output: bool = False,
        report_path: str | None = None,
    ):
        """Initialize the Git class with default settings."""
        self._explicit_path = path is not None
        self.path = path or DEFAULT_REPOSITORY_MANAGER_WORKSPACE
        self.report_path = report_path
        if not os.path.exists(self.path):
            try:
                os.makedirs(self.path, exist_ok=True)
            except Exception:  # nosec B110
                pass

        self.project_map: dict[str, str] = {}
        self.config: WorkspaceConfig | None = None
        self.set_to_default_branch = set_to_default_branch
        self.capture_output = capture_output
        self.maximum_threads = self._cpu_aware_threads(20.0)
        self.threads = min(threads or self.maximum_threads, self.maximum_threads)
        if threads:
            self.set_threads(threads=threads)

        # Centralized debug logging under XDG logs directory of agent-utilities
        try:
            from agent_utilities.core.paths import log_dir

            logs_dir = log_dir()
        except ImportError:
            import platformdirs

            logs_dir = Path(
                platformdirs.user_log_path("agent-utilities", "knuckles-team")
            )

        logs_dir.mkdir(parents=True, exist_ok=True)
        self.debug_log_path = str(logs_dir / "repository_manager_debug.log")
        self.debug_lock = threading.Lock()
        self.python_exe = self._find_python()

        self.progress: dict[str, Any] = {
            "current_phase": "Idle",
            "progress": 0,
            "phases": {},
        }

        # Run each repo's pre-commit gates (minus the slow full pytest suite)
        # before pushing, so a push can't ship a commit the repo's CI gate would
        # then reject. Skips the ``pytest`` hook for speed (the reason this was
        # previously disabled); the guardrail/lint gates still run. Disable with
        # RM_GATE_BEFORE_PUSH=false.
        self.gate_before_push = to_boolean(
            os.environ.get("RM_GATE_BEFORE_PUSH", "true")
        )

        # Initialize log file
        with open(self.debug_log_path, "a") as f:
            f.write(f"\n\n--- NEW SESSION: {datetime.datetime.now().isoformat()} ---\n")

    def _find_python(self) -> str:
        """Finds the best Python executable to use for validation."""
        venv_path = os.path.join(self.path, ".venv", "bin", "python3")
        if os.path.exists(venv_path):
            return venv_path
        return sys.executable

    def _get_pip_command(self, extra: str = "all") -> str:
        """Get the appropriate pip install command, preferring uv if available."""
        import shutil

        pip_cmd = "pip"
        if shutil.which("uv"):
            pip_cmd = "uv pip"

        return f"{pip_cmd} install --break-system-packages -e '.[{extra}]'"

    def _get_package_manager(self, path: str) -> str:
        """Determines the appropriate package manager for a given path."""
        if os.path.exists(os.path.join(path, "pnpm-lock.yaml")):
            return "pnpm"
        if os.path.exists(os.path.join(path, "yarn.lock")):
            return "yarn"
        return "npm"

    def setup_from_yaml(self, yaml_path: str, install: bool = False) -> GitResult:
        """Sets up the workspace structure from a YAML file.

        ``install=True`` extends clone/pull with the fresh-machine bootstrap
        gap this closes (CONCEPT:RM-BOOTSTRAP): after every repository is
        cloned or pulled, materialize the `.uv-workspace-siblings/` symlinks
        and run `uv sync` for agent-utilities and every cloned project that
        declares a path dependency on it, dependency-ordered (agent-utilities
        first). See :meth:`install_projects` for the mechanism and its
        documented limits. Install failures are reported, never masked --
        `setup_from_yaml` returns ``status="error"`` if any project failed to
        install, even though every repository was still cloned/pulled.
        """
        abs_yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
        if not os.path.exists(abs_yaml_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message="Configured workspace manifest was not found", code=1
                ),
            )

        if not self.load_projects_from_yaml(abs_yaml_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(message="Failed to load YAML", code=1),
            )

        logger.info("Creating configured workspace structure")
        os.makedirs(self.path, exist_ok=True)

        for _, project_path in self.project_map.items():
            os.makedirs(os.path.dirname(project_path), exist_ok=True)

        logger.info("Syncing repositories (Clone/Pull)...")
        results = []
        for url, project_path in self.project_map.items():
            if os.path.exists(project_path):
                results.append(self.pull_project(project_path))
            else:
                results.append(self.clone_repository(url, project_path))

        failed_clones = [r for r in results if r.status != "success"]

        if not install:
            return GitResult(
                status="success" if not failed_clones else "error",
                data="Workspace setup completed",
                error=(
                    GitError(
                        message=f"{len(failed_clones)} repository(ies) failed to clone/pull",
                        code=1,
                    )
                    if failed_clones
                    else None
                ),
                metadata=GitMetadata(
                    command="setup_workspace",
                    workspace=_project_label(self.path),
                    return_code=0 if not failed_clones else 1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        install_results = self.install_projects()
        failed_installs = [r for r in install_results if r.status != "success"]
        summary = [
            f"Cloned/pulled {len(results) - len(failed_clones)}/{len(results)} "
            "repository(ies).",
            f"Installed {len(install_results) - len(failed_installs)}/"
            f"{len(install_results)} project(s).",
        ]
        for r in install_results:
            label = r.metadata.workspace if r.metadata else "unknown"
            summary.append(f"- {label}: {r.status}")

        failures = failed_clones + failed_installs
        return GitResult(
            status="success" if not failures else "error",
            data="\n".join(summary),
            error=(
                GitError(
                    message=(
                        f"{len(failed_clones)} clone/pull failure(s), "
                        f"{len(failed_installs)} install failure(s)"
                    ),
                    code=1,
                )
                if failures
                else None
            ),
            metadata=GitMetadata(
                command="setup_workspace",
                workspace=_project_label(self.path),
                return_code=0 if not failures else 1,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _find_project_path(self, name: str) -> str | None:
        """Return the cloned path for project *name* (by directory basename)."""
        for path in self.project_map.values():
            if os.path.basename(path) == name:
                return path
        return None

    def get_project_map(self) -> dict[str, str]:
        """
        Returns the mapping of repository URLs to their local project paths.
        Ensures paths are absolute and expanded.
        """
        return {
            url: os.path.abspath(os.path.expanduser(p))
            for url, p in self.project_map.items()
        }

    def get_workspace_projects(self) -> list[str]:
        """Returns a list of project basenames (e.g. 'genius-agent') defined in the workspace."""
        return [os.path.basename(p) for p in self.project_map.values()]

    def list_branches(self) -> dict[str, str]:
        """Returns a dictionary mapping project basenames to their current active git branch."""
        branches: dict[str, str] = {}
        if not self.project_map:
            return branches

        for _url, path in self.project_map.items():
            repo_name = os.path.basename(path)
            if not os.path.exists(os.path.join(path, ".git")):
                branches[repo_name] = "not-cloned"
                continue

            res = self.git_action(
                "git rev-parse --abbrev-ref HEAD", path=path, quiet=True
            )
            if res.status == "success" and res.data:
                branches[repo_name] = res.data.strip()
            else:
                branches[repo_name] = "unknown"

        return branches

    def _resolve_path(self, path: str | None = None) -> str:
        """
        Resolve the path to an absolute path.
        If path is None, returns self.path.
        If path is absolute, returns it.
        If path is relative, joins it with self.path.
        """
        if path is None:
            return os.path.abspath(self.path)

        if os.path.isabs(path):
            return os.path.abspath(path)

        return os.path.abspath(os.path.join(self.path, path))

    def _current_release_tag(self, path: str | None = None) -> str | None:
        """Return ``v<current_version>`` from the repo's .bumpversion.cfg, if any.

        The tag the most recent bump created for this repo — pushed explicitly so
        lightweight tags reach the remote without dragging along stale historical
        tags. Returns None when there's no bumpversion config or it exists only
        locally (never created).
        """
        target_dir = self._resolve_path(path)
        cfg = os.path.join(target_dir, ".bumpversion.cfg")
        if not os.path.exists(cfg):
            return None
        try:
            with open(cfg) as fh:
                for line in fh:
                    if line.strip().startswith("current_version"):
                        ver = line.split("=", 1)[1].strip()
                        if ver:
                            tag = f"v{ver}"
                            # Only if the tag actually exists locally.
                            chk = self.git_action(
                                command=f"git tag -l {tag}",
                                path=target_dir,
                                quiet=True,
                            )
                            if chk.status == "success" and tag in (chk.data or ""):
                                return tag
                        return None
        except Exception as exc:  # noqa: BLE001
            logger.debug("Operation failed: error_type=%s", type(exc).__name__)
        return None

    def _tag_on_remote(self, tag: str, path: str | None = None) -> bool:
        """True if ``tag`` exists on the ``origin`` remote (so it's published).

        Used to guard force-deletion of an orphan local tag: we only ever delete
        a tag that is local-only (never one already pushed). Network failure is
        treated as "on remote" (conservative — don't delete).
        """
        target_dir = self._resolve_path(path)
        res = self.git_action(
            command=f"git ls-remote --tags origin {tag}", path=target_dir, quiet=True
        )
        if res.status != "success":
            return True  # can't verify -> assume present, do not delete
        return f"refs/tags/{tag}" in (res.data or "")

    def _workspace_root(self) -> Path:
        """Return the approved workspace root after rejecting symlink ancestry."""
        root = Path(os.path.abspath(os.path.expanduser(self.path)))
        current = Path(root.anchor)
        for component in root.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f"workspace root contains symlink component {current}")
            if current != root and current.exists() and not current.is_dir():
                raise ValueError(
                    f"workspace root contains non-directory component {current}"
                )
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace root is not a directory {root}")
        return root

    def _validate_workspace_path(
        self,
        candidate: str | Path,
        *,
        label: str,
        require_directory: bool = False,
        allow_leaf_symlink: bool = False,
    ) -> Path:
        """Validate one path lexically and by its real location under the root.

        Symlink registrations are allowed only for the final sibling link that
        this class owns and replaces. Project paths and canonical targets must
        have no symlink components at all, so a link cannot redirect source
        discovery or canonical target selection outside the approved root.
        """
        root = self._workspace_root()
        path = Path(os.path.expanduser(os.fspath(candidate)))
        if not path.is_absolute():
            path = root / path
        path = Path(os.path.abspath(path))
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes workspace root") from exc

        current = root
        components = relative.parts
        for index, component in enumerate(components):
            current /= component
            if current.is_symlink() and not (
                allow_leaf_symlink and index == len(components) - 1
            ):
                raise ValueError(f"{label} contains symlink component {current}")
            if (
                index < len(components) - 1
                and current.exists()
                and not current.is_dir()
            ):
                raise ValueError(f"{label} contains non-directory component {current}")

        if require_directory and (not path.exists() or not path.is_dir()):
            raise ValueError(f"{label} is not a directory {path}")

        # No symlink components were accepted above for project/target paths;
        # still verify the real path to close lexical-vs-real containment gaps
        # if the filesystem changed during validation.
        if not (allow_leaf_symlink and path.is_symlink()):
            try:
                real_path = path.resolve(strict=False)
                real_path.relative_to(root)
            except (OSError, ValueError) as exc:
                raise ValueError(f"{label} escapes workspace root") from exc
        return path

    @staticmethod
    def _normalize_uv_name(name: str, *, label: str) -> str:
        """Return the PEP 503 identity for one uv package/source name."""
        if not isinstance(name, str) or _PEP503_NAME.fullmatch(name) is None:
            raise ValueError(f"{label} must be an ASCII PEP 503 distribution name")
        return _PEP503_NAME_SEPARATORS.sub("-", name).lower()

    @staticmethod
    def _load_uv_source_manifest(
        project_path: Path,
    ) -> dict[str, Any]:
        """Load a project's uv source manifest without following its symlink."""
        manifest = project_path / "pyproject.toml"
        if manifest.is_symlink():
            raise ValueError(f"refusing symlink uv source manifest {manifest}")
        if not manifest.is_file():
            return {}
        try:
            with manifest.open("rb") as handle:
                document = tomllib.load(handle)
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"cannot parse uv source manifest {manifest}") from exc
        if not isinstance(document, dict):  # pragma: no cover - tomllib guarantee
            raise ValueError(f"uv source manifest {manifest} must be a table")
        return document

    @staticmethod
    def _project_name_from_manifest(
        document: dict[str, Any], *, label: str, required: bool = False
    ) -> str | None:
        project = document.get("project")
        if project is None:
            if required:
                raise ValueError(f"{label} is missing [project].name")
            return None
        if not isinstance(project, dict):
            raise ValueError(f"{label} [project] must be a table")
        name = project.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} [project].name must be a non-empty string")
        return name

    @staticmethod
    def _reject_unknown_uv_fields(source_label: str, entry: dict[str, Any]) -> None:
        """Refuse any field the uv source schema does not define."""
        unknown = set(entry) - _UV_SOURCE_KEYS
        if unknown:
            unknown_text = ", ".join(sorted(str(key) for key in unknown))
            raise ValueError(f"{source_label} has unknown field(s): {unknown_text}")

    @staticmethod
    def _validate_uv_field_types(source_label: str, entry: dict[str, Any]) -> None:
        """Type-check every declared field of one uv source alternative."""
        for key in _UV_SOURCE_STRING_KEYS & set(entry):
            value = entry[key]
            if not isinstance(value, str) or not value:
                if key == "path" and not isinstance(value, str):
                    raise ValueError(f"{source_label} has a non-string path")
                raise ValueError(
                    f"{source_label} field {key!r} must be a non-empty string"
                )
        for key in {"editable", "lfs", "workspace"} & set(entry):
            if not isinstance(entry[key], bool):
                raise ValueError(f"{source_label} field {key!r} must be boolean")

    @staticmethod
    def _uv_source_kind(source_label: str, entry: dict[str, Any]) -> str:
        """The single source kind (git / index / path / url / workspace) declared."""
        primary = _UV_SOURCE_PRIMARY_KEYS & set(entry)
        if len(primary) != 1:
            if not primary:
                raise ValueError(f"{source_label} must declare exactly one source kind")
            kinds = ", ".join(sorted(primary))
            raise ValueError(f"{source_label} has conflicting source kinds: {kinds}")
        return next(iter(primary))

    @staticmethod
    def _validate_uv_selectors(
        source_label: str, entry: dict[str, Any], source_kind: str
    ) -> None:
        """Branch/rev/tag selectors are git-only and mutually exclusive."""
        selectors = _UV_SOURCE_SELECTOR_KEYS & set(entry)
        if selectors and source_kind != "git":
            names = ", ".join(sorted(selectors))
            raise ValueError(f"{source_label} selector(s) {names} require a git source")
        if len(selectors) > 1:
            names = ", ".join(sorted(selectors))
            raise ValueError(
                f"{source_label} has mutually exclusive selectors: {names}"
            )

    @staticmethod
    def _validate_uv_source_kind_flags(
        source_label: str, entry: dict[str, Any], source_kind: str
    ) -> None:
        """The boolean flags each source kind is allowed to carry."""
        if (
            source_kind == "git"
            and "lfs" in entry
            and not isinstance(entry["lfs"], bool)
        ):
            raise ValueError(f"{source_label} field 'lfs' must be boolean")
        if "editable" in entry and source_kind != "path":
            raise ValueError(f"{source_label} editable requires a path source")
        if "lfs" in entry and source_kind != "git":
            raise ValueError(f"{source_label} lfs requires a git source")

    @staticmethod
    def _validate_uv_source_kind_fields(
        source_label: str, entry: dict[str, Any], source_kind: str
    ) -> None:
        """The addressing fields each source kind is allowed to carry."""
        if "subdirectory" in entry and source_kind not in {"git", "url"}:
            raise ValueError(f"{source_label} subdirectory requires git or url")
        if "package" in entry and source_kind not in {"git", "url"}:
            raise ValueError(f"{source_label} package requires git or url")
        if source_kind == "workspace" and entry["workspace"] is not True:
            raise ValueError(f"{source_label} workspace must be true")
        if source_kind == "path" and "subdirectory" in entry:
            raise ValueError(f"{source_label} path cannot select a subdirectory")

    @staticmethod
    def _validate_uv_own_project_source(
        source_label: str, normalized_source: str, project_name: str | None
    ) -> None:
        """A ``path = "."`` source may only name the project that declares it."""
        if project_name is None:
            raise ValueError(f"{source_label} path '.' requires an owning project name")
        if normalized_source != Git._normalize_uv_name(
            project_name, label="owning project name"
        ):
            raise ValueError(
                f"{source_label} path '.' may identify only its owning project"
            )

    @staticmethod
    def _uv_sibling_name_from_path(
        source_label: str, raw_path: str, normalized_source: str
    ) -> str:
        """The sibling repository name a local path source resolves to."""
        prefix = _UV_WORKSPACE_SIBLINGS_DIRNAME
        parts = raw_path.split("/")
        if (
            raw_path.startswith(("/", "\\"))
            or "\\" in raw_path
            or len(parts) != 2
            or parts[0] != prefix
            or not parts[1]
            or parts[1] in {".", ".."}
        ):
            raise ValueError(f"{source_label} must use a direct {prefix}/<name> path")
        sibling_name = parts[1]
        normalized_sibling = Git._normalize_uv_name(
            sibling_name, label=f"{source_label} path component"
        )
        if normalized_source != normalized_sibling:
            raise ValueError(
                f"{source_label} name does not match its direct workspace path"
            )
        return sibling_name

    @staticmethod
    def _validate_uv_source_entry(
        source_name: str,
        entry: dict[str, Any],
        *,
        project_name: str | None,
    ) -> str | None:
        """Validate one uv source alternative and return a local sibling name.

        ``None`` means that the alternative is remote (or that it is the
        owning project represented by ``path = "."``), not that it was
        skipped without validation.  Every alternative is checked before any
        sibling directory or link is created.

        The checks run in a fixed order -- unknown fields, field types, source
        kind, selectors, then kind-specific constraints -- so a doubly-invalid
        entry always reports the same error it reported before.
        """
        source_label = f"uv source {source_name!r}"
        Git._reject_unknown_uv_fields(source_label, entry)

        normalized_source = Git._normalize_uv_name(
            source_name, label=f"{source_label} name"
        )
        Git._validate_uv_field_types(source_label, entry)
        source_kind = Git._uv_source_kind(source_label, entry)
        Git._validate_uv_selectors(source_label, entry, source_kind)
        Git._validate_uv_source_kind_flags(source_label, entry, source_kind)
        Git._validate_uv_source_kind_fields(source_label, entry, source_kind)

        if source_kind != "path":
            return None

        raw_path = entry["path"]
        if raw_path == ".":
            Git._validate_uv_own_project_source(
                source_label, normalized_source, project_name
            )
            return None

        return Git._uv_sibling_name_from_path(source_label, raw_path, normalized_source)

    @staticmethod
    def _uv_sources_table(document: dict[str, Any]) -> dict[str, Any] | None:
        """The ``[tool.uv.sources]`` table of a manifest, or ``None`` if absent."""
        tool = document.get("tool")
        if tool is None:
            return None
        if not isinstance(tool, dict):
            raise ValueError("[tool] must be a table")
        uv = tool.get("uv")
        if uv is None:
            return None
        if not isinstance(uv, dict):
            raise ValueError("[tool.uv] must be a table")
        sources = uv.get("sources")
        if sources is None:
            return None
        if not isinstance(sources, dict):
            raise ValueError("[tool.uv.sources] must be a table")
        return sources

    @staticmethod
    def _uv_source_entries(source_name: str, configured: Any) -> list[Any]:
        """The list of alternatives one uv source declaration expands to.

        Element types are deliberately NOT checked here: the caller validates
        each alternative as it consumes it, so a malformed second alternative
        still reports the first one's error first.
        """
        if isinstance(configured, list):
            if not configured:
                raise ValueError(
                    f"uv source {source_name!r} must contain at least one table"
                )
            return configured
        if isinstance(configured, dict):
            return [configured]
        raise ValueError(f"uv source {source_name!r} must be a table or list of tables")

    @staticmethod
    def _record_uv_sibling_name(
        sibling_name: str | None, names: list[str], normalized_names: set[str]
    ) -> None:
        """Append a newly seen sibling name, de-duplicated by normalized form."""
        if sibling_name is None:
            return
        normalized_name = Git._normalize_uv_name(
            sibling_name, label="uv sibling path component"
        )
        if normalized_name in normalized_names:
            return
        normalized_names.add(normalized_name)
        names.append(sibling_name)

    @staticmethod
    def _declared_uv_sibling_names(project_path: str) -> tuple[str, ...]:
        """Return the sibling names declared by a project's uv sources.

        Local path sources are deliberately constrained to the one stable shape
        used by the fleet: ``.uv-workspace-siblings/<repository>``.  Parsing
        this declaration, rather than maintaining a repository allowlist, lets
        a consumer add another local package without changing repository-manager
        and prevents a manifest from smuggling a traversal path into the
        materializer.
        """
        manifest = Path(project_path) / "pyproject.toml"
        document = Git._load_uv_source_manifest(Path(project_path))
        if not document:
            return ()

        sources = Git._uv_sources_table(document)
        if sources is None:
            return ()

        project_name = Git._project_name_from_manifest(document, label=str(manifest))
        names: list[str] = []
        normalized_names: set[str] = set()
        for source_name, configured in sources.items():
            if not isinstance(source_name, str):  # pragma: no cover - TOML keys are str
                raise ValueError("uv source names must be strings")
            for entry in Git._uv_source_entries(source_name, configured):
                if not isinstance(entry, dict):
                    raise ValueError(
                        f"uv source {source_name!r} must contain only tables"
                    )
                sibling_name = Git._validate_uv_source_entry(
                    source_name,
                    entry,
                    project_name=project_name,
                )
                Git._record_uv_sibling_name(sibling_name, names, normalized_names)
        return tuple(names)

    def _canonical_uv_sibling_targets(self) -> dict[str, Path]:
        """Build a bounded, canonical repository-name-to-path map.

        The map is derived solely from the configured workspace projects.  A
        duplicate basename is refused instead of letting one registration
        silently shadow another, and every target must remain under the
        configured workspace root.
        """
        targets: dict[str, Path] = {}
        for configured in self.project_map.values():
            candidate = self._validate_workspace_path(
                configured,
                label=f"canonical sibling target {configured!r}",
            )
            name = self._normalize_uv_name(
                candidate.name, label="canonical sibling target name"
            )
            if not name:
                continue
            target_manifest = candidate / "pyproject.toml"
            if target_manifest.is_symlink():
                raise ValueError(
                    f"canonical sibling target manifest is a symlink {target_manifest}"
                )
            if target_manifest.is_file():
                target_document = self._load_uv_source_manifest(candidate)
                target_project_name = self._project_name_from_manifest(
                    target_document, label=str(target_manifest)
                )
                if (
                    target_project_name is not None
                    and self._normalize_uv_name(
                        target_project_name, label=f"{target_manifest} project name"
                    )
                    != name
                ):
                    raise ValueError(
                        f"canonical sibling target {candidate} has a project name "
                        "that does not match its direct workspace path"
                    )
            previous = targets.get(name)
            if previous is not None and previous != candidate:
                raise ValueError(f"ambiguous canonical sibling target {name!r}")
            targets[name] = candidate
        return targets

    @staticmethod
    def _replace_uv_sibling_link(link: Path, target: Path) -> None:
        """Create or correct one sibling symlink without replacing real files."""
        if link.is_symlink():
            try:
                if link.resolve(strict=False) == target:
                    return
            except OSError:
                # A broken link is still safe to replace: it is handled by the
                # symlink-only branch below and never followed as a directory.
                pass
        elif link.exists():
            raise ValueError(f"refusing to replace non-symlink path {link}")

        staged = Git._uv_sibling_temp_path(link)
        staged_created = False
        try:
            staged.symlink_to(target, target_is_directory=True)
            staged_created = True
            if link.exists() and not link.is_symlink():
                raise ValueError(f"refusing to replace non-symlink path {link}")
            os.replace(staged, link)
        finally:
            if staged_created:
                cleanup_errors = Git._cleanup_uv_sibling_temp(staged, os.fspath(target))
                if cleanup_errors:
                    raise RuntimeError("; ".join(cleanup_errors))

    @staticmethod
    def _uv_sibling_temp_path(link: Path) -> Path:
        return link.with_name(
            f".{link.name}.{os.getpid()}.{threading.get_ident()}.{uuid.uuid4().hex}.tmp"
        )

    @staticmethod
    def _cleanup_uv_sibling_temp(staged: Path, expected_target: str) -> list[str]:
        """Remove one task-owned staging symlink without deleting real files."""
        if not staged.is_symlink():
            if staged.exists():
                return [f"staging path is no longer a symlink: {staged}"]
            return []
        try:
            actual_target = os.readlink(staged)
        except OSError as exc:
            return [f"cannot inspect staging path {staged}: {exc}"]
        if actual_target != expected_target:
            return [f"staging path target changed unexpectedly: {staged}"]
        try:
            staged.unlink()
        except OSError as exc:
            return [f"cannot remove staging path {staged}: {exc}"]
        return []

    @staticmethod
    def _rollback_new_uv_link(link: Path, new_target: str, errors: list[str]) -> None:
        """Remove a link this transaction created, if it is still ours to remove."""
        if not link.is_symlink():
            if link.exists():
                errors.append(
                    f"refusing to remove non-symlink path during rollback: {link}"
                )
            return
        if os.readlink(link) != new_target:
            errors.append(f"refusing to remove changed symlink during rollback: {link}")
            return
        link.unlink()

    @staticmethod
    def _restore_uv_link(link: Path, previous_target: str, errors: list[str]) -> None:
        """Point a pre-existing link back at the target it had before."""
        if link.is_symlink() and os.readlink(link) == previous_target:
            return
        if link.exists() and not link.is_symlink():
            errors.append(
                f"refusing to replace non-symlink path during rollback: {link}"
            )
            return

        restore = Git._uv_sibling_temp_path(link)
        restore_created = False
        try:
            restore.symlink_to(previous_target)
            restore_created = True
            os.replace(restore, link)
        finally:
            if restore_created:
                errors.extend(Git._cleanup_uv_sibling_temp(restore, previous_target))

    @staticmethod
    def _rollback_uv_sibling_links(
        updates: list[tuple[Path, Path, str | None, Path]],
    ) -> list[str]:
        """Restore every link in a failed multi-link publication.

        Existing registrations are symlinks by construction.  A real file or
        directory that appears during the transaction is never replaced or
        removed; instead it is reported as an incomplete rollback.
        """
        errors: list[str] = []
        for link, target, previous_target, _staged in updates:
            try:
                if previous_target is None:
                    Git._rollback_new_uv_link(link, os.fspath(target), errors)
                else:
                    Git._restore_uv_link(link, previous_target, errors)
            except BaseException as exc:
                errors.append(f"cannot restore {link}: {exc}")
        return errors

    def _resolve_uv_sibling_targets(self, names: tuple[str, ...]) -> dict[str, Path]:
        """Map every declared sibling name to its canonical workspace directory."""
        targets = self._canonical_uv_sibling_targets()
        resolved_targets: dict[str, Path] = {}
        for name in names:
            normalized_name = self._normalize_uv_name(
                name, label="uv sibling path component"
            )
            target = targets.get(normalized_name)
            if target is None or not target.is_dir():
                raise ValueError(
                    f"canonical sibling target {name!r} is missing from the workspace map"
                )
            resolved_targets[normalized_name] = target
        return resolved_targets

    def _validated_uv_sibling_links(
        self,
        sibling_dir: Path,
        names: tuple[str, ...],
        resolved_targets: dict[str, Path],
    ) -> list[tuple[str, Path]]:
        """Validate every owned link before the sibling directory is created.

        A malformed second declaration must not leave the first link behind.
        """
        validated_links: list[tuple[str, Path]] = []
        for name in names:
            link = sibling_dir / name
            self._validate_workspace_path(
                link,
                label=f"uv sibling link {name!r}",
                allow_leaf_symlink=True,
            )
            if link.exists() and not link.is_symlink():
                raise ValueError(f"refusing to replace non-symlink path {link}")
            normalized_name = self._normalize_uv_name(
                name, label="uv sibling path component"
            )
            validated_links.append((name, resolved_targets[normalized_name]))
        return validated_links

    @staticmethod
    def _uv_link_already_points_at(link: Path, target: Path) -> bool:
        """True when *link* already resolves to *target*.

        A broken or looping registration answers ``False`` and is replaced.
        """
        try:
            return link.resolve(strict=False) == target
        except (OSError, RuntimeError):
            return False

    def _pending_uv_sibling_updates(
        self, sibling_dir: Path, validated_links: list[tuple[str, Path]]
    ) -> list[tuple[Path, Path, str | None, Path]]:
        """The links that actually need re-pointing, with their staging paths."""
        updates: list[tuple[Path, Path, str | None, Path]] = []
        for name, target in validated_links:
            link = sibling_dir / name
            previous_target = os.readlink(link) if link.is_symlink() else None
            if previous_target is not None and self._uv_link_already_points_at(
                link, target
            ):
                continue
            updates.append(
                (link, target, previous_target, self._uv_sibling_temp_path(link))
            )
        return updates

    @staticmethod
    def _stage_uv_sibling_links(
        updates: list[tuple[Path, Path, str | None, Path]],
        staged: list[tuple[Path, str]],
    ) -> None:
        """Create every replacement symlink under a task-owned staging name."""
        for _link, target, _previous_target, staged_path in updates:
            staged_path.symlink_to(target, target_is_directory=True)
            staged.append((staged_path, os.fspath(target)))

    @staticmethod
    def _swap_uv_sibling_links(
        updates: list[tuple[Path, Path, str | None, Path]],
    ) -> None:
        """Move every staged symlink onto its final name."""
        for link, _target, _previous_target, staged_path in updates:
            if link.exists() and not link.is_symlink():
                raise ValueError(f"refusing to replace non-symlink path {link}")
            os.replace(staged_path, link)

    @staticmethod
    def _remove_created_uv_sibling_dir(sibling_dir: Path) -> list[str]:
        """Remove a sibling directory this transaction created, if still safe."""
        if sibling_dir.is_symlink() or not sibling_dir.is_dir():
            return [f"refusing to remove changed sibling directory {sibling_dir}"]
        try:
            sibling_dir.rmdir()
        except OSError as cleanup_exc:
            return [
                f"cannot remove empty sibling directory {sibling_dir}: {cleanup_exc}"
            ]
        return []

    def _recover_failed_uv_publication(
        self,
        *,
        sibling_dir: Path,
        updates: list[tuple[Path, Path, str | None, Path]],
        staged: list[tuple[Path, str]],
        created_sibling_dir: bool,
    ) -> list[str]:
        """Undo a failed publication; return whatever could not be undone."""
        rollback_errors = self._rollback_uv_sibling_links(updates)
        cleanup_errors: list[str] = []
        for staged_path, expected_target in staged:
            cleanup_errors.extend(
                self._cleanup_uv_sibling_temp(staged_path, expected_target)
            )
        if created_sibling_dir:
            cleanup_errors.extend(self._remove_created_uv_sibling_dir(sibling_dir))
        return rollback_errors + cleanup_errors

    def _publish_uv_sibling_links(
        self,
        sibling_dir: Path,
        updates: list[tuple[Path, Path, str | None, Path]],
    ) -> None:
        """Stage and swap every pending sibling link as one transaction.

        Any failure rolls the whole set back; an incomplete rollback is raised
        as a RuntimeError chained from the original exception.
        """
        created_sibling_dir = False
        staged: list[tuple[Path, str]] = []
        try:
            if not sibling_dir.exists():
                try:
                    sibling_dir.mkdir()
                    created_sibling_dir = True
                except FileExistsError:
                    pass
            if sibling_dir.is_symlink() or not sibling_dir.is_dir():
                raise ValueError(f"sibling path is not a directory {sibling_dir}")

            self._stage_uv_sibling_links(updates, staged)
            self._swap_uv_sibling_links(updates)
        except BaseException as exc:
            errors = self._recover_failed_uv_publication(
                sibling_dir=sibling_dir,
                updates=updates,
                staged=staged,
                created_sibling_dir=created_sibling_dir,
            )
            if errors:
                detail = "; ".join(errors)
                raise RuntimeError(
                    f"uv sibling publication failed and rollback was incomplete: {detail}"
                ) from exc
            raise

    def _materialize_uv_siblings(self, project_path: str) -> tuple[str, ...]:
        """Materialize every declared uv sibling from the canonical map.

        This is intentionally only the source-view step.  Dependency ordering
        and the epistemic-graph wheel fast path remain owned by their existing
        install/launcher flows; this helper only makes their declared paths
        resolve to canonical sibling repositories.
        """
        project = self._validate_workspace_path(
            project_path,
            label="project path",
            require_directory=True,
        )
        names = self._declared_uv_sibling_names(str(project))
        if not names:
            return ()

        resolved_targets = self._resolve_uv_sibling_targets(names)

        sibling_dir = project / _UV_WORKSPACE_SIBLINGS_DIRNAME
        if sibling_dir.is_symlink():
            raise ValueError(f"refusing symlink sibling directory {sibling_dir}")
        if sibling_dir.exists() and not sibling_dir.is_dir():
            raise ValueError(f"sibling path is not a directory {sibling_dir}")

        validated_links = self._validated_uv_sibling_links(
            sibling_dir, names, resolved_targets
        )
        updates = self._pending_uv_sibling_updates(sibling_dir, validated_links)
        if not updates:
            return names

        self._publish_uv_sibling_links(sibling_dir, updates)
        return names

    def install_projects(
        self, extra: str = "all", threads: int | None = None, report: bool = True
    ) -> list[GitResult]:
        """Bulk installs Python and Node projects in the workspace."""
        effective_threads = threads if threads is not None else self.threads
        threads = min(effective_threads, self._cpu_aware_threads(20.0))
        if not self.project_map:
            logger.warning("No projects to install.")
            return []

        logger.info("Installing ecosystem using native uv workspace sync...")
        results: list[GitResult] = []
        results.extend(self._install_agent_utilities_first(extra))
        results.extend(self._install_remaining_ecosystem_projects())
        self._maybe_export_install_report(results, report)
        return results

    def _install_agent_utilities_first(self, extra: str) -> list[GitResult]:
        """Step 1: install agent-utilities first, then every other cloned
        project that depends on it -- CONCEPT:RM-BOOTSTRAP.

        This replaces a prior `uv sync --all-packages` run at the
        workspace root, which cannot succeed structurally: agent-utilities
        is its own uv workspace root (a dedicated, security-motivated
        boundary -- see its own AGENTS.md), and uv refuses a workspace
        member that is itself a workspace root ("Nested workspaces are not
        supported"). Verified empirically that even a project using the
        correct per-repo `.uv-workspace-siblings` path-source workaround
        still gets pulled into ecosystem-root resolution -- with its own
        local override silently ignored -- whenever its checkout also
        matches an ancestor workspace's `[tool.uv.workspace].members`
        glob; running each project's `uv sync` directly IN that project's
        own directory (never at `self.path`) avoids both failure modes.
        Every fleet member depends on agent-utilities, and it is not yet
        published to PyPI at the floor the fleet requires (only <=1.26.4
        is public; the fleet requires >=2.0.0), so agent-utilities must
        install successfully before any dependent is attempted -- this
        fails closed rather than reporting a partial "N/M installed" that
        would mask a downstream project never having had a chance.
        """
        if not shutil.which("uv"):
            logger.warning("uv not found. Native workspace sync requires uv.")
            return []

        au_path = self._find_project_path("agent-utilities")
        if au_path is None or not os.path.isdir(au_path):
            logger.warning(
                "agent-utilities not present in this workspace's project "
                "set; every fleet member depends on it, so no project "
                "can be installed."
            )
            return []

        launcher = os.path.join(au_path, "scripts", "uv_workspace.py")
        if not os.path.isfile(launcher):
            return [
                GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=(
                            "agent-utilities checkout is missing "
                            "scripts/uv_workspace.py"
                        ),
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="install",
                        workspace=_project_label(au_path),
                        return_code=1,
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    ),
                )
            ]

        au_sync_command = f"python3 {shlex.quote(launcher)} sync" + _uv_extra_flag(
            extra
        )
        au_result = self.git_action(au_sync_command, path=au_path, timeout=300)
        results = [au_result]
        if au_result.status == "success":
            results.extend(self._sync_uv_siblings(au_path, extra))
        return results

    def _sync_uv_siblings(self, au_path: str, extra: str) -> list[GitResult]:
        """Materialize + `uv sync` every declared uv sibling once
        agent-utilities itself has installed successfully."""
        results: list[GitResult] = []
        for path in list(self.project_map.values()):
            if path == au_path or not os.path.isdir(path):
                continue
            try:
                sibling_names = self._materialize_uv_siblings(path)
            except (OSError, ValueError) as exc:
                results.append(
                    GitResult(
                        status="error",
                        data="",
                        error=GitError(message=str(exc), code=1),
                        metadata=GitMetadata(
                            command="install",
                            workspace=_project_label(path),
                            return_code=1,
                            timestamp=datetime.datetime.now(datetime.UTC).isoformat()
                            + "Z",
                        ),
                    )
                )
                continue
            if not sibling_names:
                continue

            dep_sync_command = "uv sync" + _uv_extra_flag(extra)
            results.append(self.git_action(dep_sync_command, path=path, timeout=300))
        return results

    def _install_remaining_ecosystem_projects(self) -> list[GitResult]:
        """Step 2: install Node/Python projects sequentially."""
        results: list[GitResult] = []
        for _url, path in self.project_map.items():
            results.extend(self._install_one_ecosystem_project(path))
        return results

    def _install_one_ecosystem_project(self, path: str) -> list[GitResult]:
        has_precommit = os.path.exists(os.path.join(path, ".pre-commit-config.yaml"))
        has_pyproject = os.path.exists(os.path.join(path, "pyproject.toml"))

        if not has_precommit and not has_pyproject:
            return [
                GitResult(
                    status="skipped",
                    data="Skipped (No .pre-commit-config.yaml and no pyproject.toml)",
                    metadata=GitMetadata(
                        command="install",
                        workspace=_project_label(path),
                        return_code=0,
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    ),
                )
            ]

        results: list[GitResult] = []
        is_node = os.path.exists(os.path.join(path, "package.json"))
        if is_node:
            pm = self._get_package_manager(path)
            res = self.git_action(f"{pm} install", path=path)
            if pm == "pnpm" and "Ignored build scripts:" in res.data:
                res.status = "error"
                res.data = f"pnpm install succeeded but ignored build scripts:\n{res.data}\nPlease add allowed dependencies to package.json."
            results.append(res)

        is_python = os.path.exists(
            os.path.join(path, "pyproject.toml")
        ) or os.path.exists(os.path.join(path, "setup.py"))
        if not is_python and not is_node:
            results.append(
                GitResult(
                    status="skipped",
                    data="Skipped (Not a Python or Node project)",
                    metadata=GitMetadata(
                        command="install",
                        workspace=_project_label(path),
                        return_code=0,
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    ),
                )
            )
        return results

    def _maybe_export_install_report(
        self, results: list[GitResult], report: bool
    ) -> None:
        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "error"]
        report_md = _build_install_report_markdown(results, successes, failures)
        if self.report_path and report:
            self._export_report(report_md, "install_report.md")

    def build_projects(self, threads: int | None = None) -> list[GitResult]:
        """Build projects serially so compilation cannot exhaust the workstation."""
        del threads
        if not self.project_map:
            logger.warning("No projects to build.")
            return []

        logger.info("Building configured projects in the serialized build lane")
        results: list[GitResult] = []
        for _url, path in self.project_map.items():
            if os.path.exists(os.path.join(path, "package.json")):
                package_manager = self._get_package_manager(path)
                install = self.git_action(f"{package_manager} install", path=path)
                results.append(install)
                if install.status != "success":
                    continue
                results.append(
                    self.git_action(f"{package_manager} run build", path=path)
                )
            else:
                results.append(self.git_action(f"{sys.executable} -m build", path=path))
        return results

    @staticmethod
    def _cpu_aware_threads(max_cpu_pct: float = 20.0) -> int:
        """Calculate thread count to stay under *max_cpu_pct* CPU utilisation.

        For subprocess-heavy workloads each thread drives an external process,
        so we approximate 1 thread ≈ 1 core of load.  Targeting 20% of
        available cores keeps background validation from starving the IDE and
        MCP server.
        """
        try:
            cores = len(os.sched_getaffinity(0))
        except AttributeError:
            cores = multiprocessing.cpu_count() or 4
        target = max(1, int(cores * max_cpu_pct / 100.0))
        return target

    def validate_single_project(self, repo_path: str) -> RepoScanResult:
        """Validates a single repository by running its FAST-tier gates.

        Delegates to :func:`repository_manager.gates.run_gate_stage` (the same
        engine ``rm_gates`` uses) with ``stage="fast"`` (``--hook-stage
        pre-commit``). Under the two-tier gate model a repo's HEAVY hooks
        (pytest, cargo, ``uv lock --check``, ...) are declared ``stages:
        [pre-push, manual]`` and are therefore correctly excluded here -- use
        ``rm_gates action=run stage=heavy`` for those.
        """
        logger.info("Validating configured project")
        return run_gate_stage(repo_path, "fast", trigger="validate", colocated=True)

    def validate_and_release(
        self,
        threads: int | None = None,
        auto_bump: bool = False,
        auto_push: bool = False,
        bump_part: str = "minor",
    ) -> dict[str, Any]:
        """Validate projects in parallel, optionally triggering a release if successful."""
        if not self.project_map:
            logger.warning("No projects to validate.")
            return {"passed": False, "validation_results": {}, "release_results": {}}

        effective_threads = threads or self._cpu_aware_threads()
        logger.info(
            f"Validating {len(self.project_map)} projects in parallel ({effective_threads} threads)..."
        )

        validation_results: dict[str, Any] = {}
        passed = True

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=effective_threads
        ) as executor:
            futures = {
                executor.submit(self.validate_single_project, path): url
                for url, path in self.project_map.items()
            }
            for future in concurrent.futures.as_completed(futures):
                url = futures[future]
                repo_name = url.split("/")[-1].replace(".git", "")
                try:
                    result = future.result()
                    validation_results[repo_name] = (
                        result.model_dump() if hasattr(result, "model_dump") else result
                    )
                    if hasattr(result, "success"):
                        if not result.success:
                            passed = False
                    else:
                        passed = False
                except Exception as e:
                    logger.error("Operation failed: error_type=%s", type(e).__name__)
                    validation_results[repo_name] = {
                        "success": False,
                        "error": "Operation failed",
                    }
                    passed = False

        release_results = {}
        if passed:
            logger.info("All validations passed.")
            if auto_bump:
                logger.info(f"Triggering phased bumpversion ({bump_part})...")
                release_results["bump"] = self.phased_bumpversion(part=bump_part)
            if auto_push:
                logger.info("Triggering phased push...")
                release_results["push"] = self.phased_push()
        else:
            if auto_bump or auto_push:
                logger.warning("Validation failed. Skipping bump and push.")

        return {
            "passed": passed,
            "validation_results": validation_results,
            "release_results": release_results,
        }

    def _export_report(self, markdown_content: str, default_name: str) -> None:
        """Exports markdown content to a file if reporting is enabled."""
        if not self.report_path:
            return

        report_file = self.report_path
        if report_file is True:
            report_file = os.path.join(self.path, default_name)
        elif not os.path.isabs(report_file):
            report_file = os.path.join(self.path, report_file)

        try:
            with open(report_file, "w") as f:
                f.write(markdown_content)
            logger.info("Repository report exported")
        except Exception as e:
            logger.error(
                "Failed to export repository report: error_type=%s",
                type(e).__name__,
            )

    @staticmethod
    def _summary_result_name(result: GitResult) -> str:
        """The project label used for one result row in a markdown summary."""
        if result.metadata:
            return os.path.basename(result.metadata.workspace)
        return "unknown"

    @staticmethod
    def _summary_success_message(action: str, result: GitResult) -> str:
        """The one-line success blurb for *result*.

        Bulk actions with uninteresting stdout collapse to "Success", as does
        any multi-line payload that is not a version-bump report.
        """
        msg = result.data or "Success"
        if action.lower() in ["installation", "build", "validation"]:
            return "Success"
        if (
            msg.count("\n") > 2
            and "new_version=" not in msg
            and "current_version=" not in msg
        ):
            return "Success"
        return msg

    @staticmethod
    def _summary_success_section(action: str, successes: list[GitResult]) -> list[str]:
        """The "Successes" block of a markdown summary."""
        md = ["## Successes ✅"]
        for r in successes:
            name = Git._summary_result_name(r)
            msg = Git._summary_success_message(action, r)
            md.append(f"- **{name}**: {msg}")
        md.append("")
        return md

    @staticmethod
    def _summary_failure_entry(result: GitResult) -> list[str]:
        """The per-project detail block for one failed result."""
        md = [f"### ⚠️ {Git._summary_result_name(result)}"]
        if result.metadata:
            md.append(f"**Command:** `{result.metadata.command}`")
        err_msg = result.error.message if result.error else "Unknown error"
        md.append("**Error:**")
        md.append(f"```text\n{err_msg}\n```")
        if result.data:
            md.append("**Output:**")
            md.append(f"```text\n{result.data}\n```")
        md.append("---")
        return md

    @staticmethod
    def _summary_failure_section(failures: list[GitResult]) -> list[str]:
        """The "Failures" block of a markdown summary."""
        md = ["## Failures ❌"]
        for r in failures:
            md.extend(Git._summary_failure_entry(r))
        md.append("")
        return md

    @staticmethod
    def _summary_skip_section(skips: list[GitResult]) -> list[str]:
        """The "Skipped" block of a markdown summary, grouped by reason."""
        reasons: dict[str, list[str]] = {}
        for r in skips:
            reason = r.data or "No reason provided"
            reasons.setdefault(reason, []).append(Git._summary_result_name(r))

        md = ["## Skipped ⏭️"]
        for reason, projects in sorted(reasons.items()):
            project_list = ", ".join(sorted(set(projects)))
            md.append(f"- **{reason}**: {project_list}")
        md.append("")
        return md

    @staticmethod
    def generate_markdown_summary(action: str, results: list[GitResult]) -> str:
        """Generates a beautiful markdown summary of bulk operation results."""
        successes = [r for r in results if r.status == "success"]
        failures = [r for r in results if r.status == "error"]
        skips = [r for r in results if r.status == "skipped"]

        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        md = [
            f"# {action.upper()} Summary",
            f"**Time:** {timestamp}  ",
            f"**Total:** {len(results)} | **Success:** {len(successes)} ✅ | **Failure:** {len(failures)} ❌ | **Skipped:** {len(skips)} ⏭️",
            "",
        ]

        if successes:
            md.extend(Git._summary_success_section(action, successes))
        if failures:
            md.extend(Git._summary_failure_section(failures))
        if skips:
            md.extend(Git._summary_skip_section(skips))

        return "\n".join(md)

    def git_action(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict | None = None,
        timeout: int = 1800,
        raw_output: bool = False,
    ) -> GitResult:
        """
        Execute a Git command in the specified directory.

        Args:
            command (str): The Git command to execute.
            path (str, optional): The directory to execute the command in.
                Defaults to the base path.

        Returns:
            GitResult: The combined stdout and stderr output of the command in structured format.

        Concept:
            CONCEPT:RM-GIT-ACTION
        """
        target_path = self._resolve_path(path)

        try:
            command_argv = shlex.split(str(command), posix=True)
        except ValueError as exc:
            raise ValueError(
                "repository operation has invalid argument quoting"
            ) from exc
        if not command_argv:
            raise ValueError("repository operation is empty")

        command_env: dict[str, str] = {}
        while command_argv and _ENV_ASSIGNMENT.fullmatch(command_argv[0]):
            name, value = command_argv.pop(0).split("=", 1)
            command_env[name] = value
        if not command_argv:
            raise ValueError("repository operation has no executable")
        if any(
            token in _SHELL_CONTROL_TOKENS
            or token.startswith((">", "<"))
            or "\x00" in token
            for token in command_argv
        ):
            raise ValueError("shell control syntax is not permitted")

        # Ensure ~/.local/bin is in PATH for tools like bump2version
        current_env = env if env else os.environ.copy()
        current_env.update(command_env)
        local_bin = os.path.expanduser("~/.local/bin")
        if local_bin not in current_env.get("PATH", ""):
            current_env["PATH"] = f"{local_bin}:{current_env.get('PATH', '')}"

        # Ensure Python output is unbuffered so we get real-time logs
        current_env["PYTHONUNBUFFERED"] = "1"

        operation = _operation_label(command_argv)
        logger.info("Executing repository operation")

        process = subprocess.Popen(
            command_argv,
            shell=False,
            cwd=target_path,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            text=True,
            env=current_env,
            bufsize=1,  # Line buffered
            start_new_session=True,  # Isolate process group so killpg only kills the command
        )

        output_lines: list[str] = []
        output_bytes = 0
        output_truncated = False
        try:
            # Write start marker
            with self.debug_lock:
                with open(self.debug_log_path, "a") as log_file:
                    log_file.write(
                        f"\n[{datetime.datetime.now().isoformat()}] Starting repository operation\n"
                    )
                    log_file.flush()

            # Read output line by line as it becomes available
            def _read_output():
                nonlocal output_bytes, output_truncated
                if process.stdout:
                    for line in process.stdout:
                        encoded = line.encode("utf-8", "replace")
                        remaining = _MAX_CAPTURED_OUTPUT_BYTES - output_bytes
                        if remaining > 0:
                            clipped = encoded[:remaining].decode("utf-8", "ignore")
                            output_lines.append(clipped)
                            output_bytes += len(clipped.encode("utf-8"))
                        if len(encoded) > remaining:
                            output_truncated = True
                        with self.debug_lock:
                            with open(self.debug_log_path, "a") as log_file:
                                log_file.write(
                                    f"[{datetime.datetime.now().isoformat()}] "
                                    "[repository output line omitted]\n"
                                )
                                log_file.flush()

            reader_thread = threading.Thread(target=_read_output, daemon=True)
            reader_thread.start()

            # Wait for process to complete, with a safety timeout
            process.wait(timeout=timeout)
            reader_thread.join(timeout=1.0)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Repository operation timed out")
            if hasattr(os, "killpg"):
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                        process.wait(timeout=5)
                except Exception:  # nosec B110
                    process.kill()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
            else:
                process.kill()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass

            with self.debug_lock:
                with open(self.debug_log_path, "a") as log_file:
                    log_file.write(
                        f"[{datetime.datetime.now().isoformat()}] ERROR: Command timed out after {timeout} seconds\n"
                    )
                    log_file.flush()

        if output_truncated:
            output_lines.append("\n[repository output truncated]\n")
        captured = "".join(output_lines)
        out = captured if raw_output else _privacy_safe_diagnostic(captured)
        return_code = process.returncode

        metadata = GitMetadata(
            command=operation,
            workspace=_project_label(target_path),
            return_code=return_code,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        )

        error_obj = None
        if return_code != 0:
            error_obj = GitError(
                message=out.strip() if out else "Unknown error",
                code=return_code,
            )

        result = GitResult(
            status="success" if return_code == 0 else "error",
            data=out.strip() if out else "",
            error=error_obj,
            metadata=metadata,
        )

        if result.status == "error":
            logger.error("Repository operation failed")
        elif not quiet:
            logger.info("Repository operation completed")

        return result

    def cleanup_artifacts(self, target_dir: str) -> None:
        """Removes test artifacts and temporary files from the specified directory."""
        dir_path = Path(target_dir)
        if not dir_path.exists():
            return

        # Use os.walk with top-down pruning to avoid iterating massive directories
        for dirpath, dirnames, filenames in os.walk(target_dir, topdown=True):
            # Prune ignored directories in-place (prevents os.walk from descending)
            dirnames[:] = [d for d in dirnames if d not in _CLEANUP_IGNORED_DIRS]

            _cleanup_matched_dirs(dirpath, dirnames)

            if dirpath == target_dir:
                _cleanup_root_level_files(dirpath, filenames)

            _cleanup_matched_files(dirpath, filenames)

    def clone_projects(self, projects: list[str] | None = None) -> list[GitResult]:
        """
        Clone all specified Git projects in parallel using multiple threads.

        Returns:
            List[GitResult]: A list of GitResult objects, one for each clone operation.
        """
        try:
            expanded_path = os.path.expanduser(self.path)
            if not os.path.exists(expanded_path):
                os.makedirs(expanded_path, exist_ok=True)

            targets = []
            if projects:
                for url in projects:
                    name = url.split("/")[-1].replace(".git", "")
                    targets.append((url, os.path.join(expanded_path, name)))
            elif self.project_map:
                for url, path in self.project_map.items():
                    targets.append((url, path))

            if not targets:
                logger.warning("No projects to clone.")
                return []

            logger.info(
                f"Cloning {len(targets)} projects in parallel using {self.threads} threads..."
            )
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=self.threads
            ) as executor:
                futures = {
                    executor.submit(self.clone_repository, url, path): (url, path)
                    for url, path in targets
                }
                results = []
                for future in concurrent.futures.as_completed(futures):
                    results.append(future.result())
            return results

        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)
            return [
                GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"Parallel project cloning failed: {type(e).__name__}",
                        code=-1,
                    ),
                    metadata=GitMetadata(
                        command="clone_projects",
                        workspace=_project_label(self.path),
                        return_code=-1,
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    ),
                )
            ]

    @_exclusive_repo_mutation
    def clone_repository(self, url: str, target_path: str) -> GitResult:
        """
        Clone a single Git repository to a specific target path.

        Args:
            url (str): The repository URL to clone.
            target_path (str): The absolute path where the repository should be cloned.

        Returns:
            GitResult: The result of the Git clone command.
        """
        target_path = os.path.abspath(os.path.expanduser(target_path))
        if not url:
            return GitResult(
                status="error",
                data="",
                error=GitError(message="No repository URL provided", code=1),
                metadata=GitMetadata(
                    command="clone",
                    workspace=_project_label(target_path),
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        os.makedirs(os.path.dirname(target_path), exist_ok=True)

        clone_filter = os.environ.get("REPOSITORY_MANAGER_CLONE_FILTER", "").strip()
        filter_arg = (
            f" --filter={shlex.quote(clone_filter)}"
            if clone_filter in {"blob:none", "tree:0"}
            else ""
        )
        command = (
            f"git clone{filter_arg} -- {shlex.quote(url)} {shlex.quote(target_path)}"
        )
        result = self.git_action(command, path=os.path.dirname(target_path))
        logger.info("Repository clone completed with status %s", result.status)
        return result

    def pull_projects(self, project_dirs: list[str] | None = None) -> list[GitResult]:
        """
        Pull updates for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to pull.")
                return []

        if not project_dirs:
            logger.warning("No projects found to pull.")
            return []

        logger.info(
            f"Pulling {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            results = list(executor.map(self.pull_project, project_dirs))
        _run_post_hydration_mount_checks(self.path)
        return results

    @_exclusive_repo_mutation
    def pull_project(self, path: str | None = None) -> GitResult:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            path (str): The path to the project to pull. Defaults to self.path.

        Returns:
            GitResult: The result of the pull operation.
        """
        target_path = self._resolve_path(path)
        results = []

        pull_result = self.git_action(command="git pull", path=target_path)
        results.append(pull_result)

        logger.info("Repository pull completed")

        if self.set_to_default_branch:
            default_branch_result = self.git_action(
                "git symbolic-ref refs/remotes/origin/HEAD",
                path=target_path,
            )
            if default_branch_result.status == "success":
                default_branch = re.sub(
                    "refs/remotes/origin/", "", default_branch_result.data
                ).strip()
                # WT-3 (CONCEPT:RM-WORKTREE, CONCEPT:RM-CANON-GUARD) —
                # non-destructive / worktree-aware. Never switch branches on a
                # dirty canonical tree: a concurrent session may have
                # uncommitted work here, and a forced checkout would disrupt
                # it. guarded_canonical_mutation skips the checkout (loudly,
                # with the repo named) instead. Session work belongs in a
                # worktree under WORKTREE_ROOT anyway, which this never
                # touches.
                current_branch = self.git_action(
                    "git rev-parse --abbrev-ref HEAD", path=target_path, quiet=True
                ).data.strip()
                if current_branch == default_branch:
                    logger.info("Configured project is already on its default branch")
                else:
                    repo_label = _project_label(target_path)
                    with guarded_canonical_mutation(
                        self, target_path, repo_label, "check out default branch"
                    ) as blocked:
                        if blocked is not None:
                            results.append(
                                GitResult(
                                    status="skipped",
                                    data=blocked.get("detail", ""),
                                    error=GitError(message=blocked["error"], code=0),
                                    metadata=GitMetadata(
                                        command="checkout-guard",
                                        workspace=repo_label,
                                        return_code=0,
                                        timestamp=datetime.datetime.now(
                                            datetime.UTC
                                        ).isoformat()
                                        + "Z",
                                    ),
                                )
                            )
                        else:
                            checkout_result = self.git_action(
                                f'git checkout "{default_branch}"',
                                path=target_path,
                            )
                            results.append(checkout_result)
                            logger.info("Checked out configured default branch")
            else:
                results.append(default_branch_result)
                logger.error("Failed to resolve the configured default branch")

        combined_status = (
            "success" if all(r.status == "success" for r in results) else "error"
        )

        combined_data = "\n".join(
            [
                f"[{r.metadata.command if r.metadata else 'unknown'}]: {r.data}"
                for r in results
            ]
        )

        combined_error = next((r.error for r in results if r.error), None)

        metadata = GitMetadata(
            command="pull_project",
            workspace=_project_label(target_path),
            return_code=0 if combined_status == "success" else 1,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        )

        return GitResult(
            status=combined_status,
            data=combined_data,
            error=combined_error,
            metadata=metadata,
        )

    def push_projects(self, project_dirs: list[str] | None = None) -> list[GitResult]:
        """
        Push updates for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to push.")
                return []

        if not project_dirs:
            logger.warning("No projects found to push.")
            return []

        logger.info(
            f"Pushing {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(self.push_project, project_dirs))

    def _has_unpushed_commits(self, target_path: str) -> bool:
        """True when the local branch has commits the remote lacks.

        Used to skip the pre-push gate on no-op repos (nothing to validate).
        On any uncertainty (no upstream, error) returns True so the gate runs.
        """
        res = self.git_action(
            command="git rev-list --count @{u}..HEAD", path=target_path, quiet=True
        )
        if res.status != "success" or not res.data:
            return True
        try:
            return int(res.data.strip()) > 0
        except (ValueError, AttributeError):
            return True

    def _unpushed_changed_files(self, target_path: str) -> list[str]:
        """Files touched by the commits about to be pushed (``@{u}..HEAD``).

        Lets the pre-push gate scope per-file hooks to just the diff being
        pushed. Returns ``[]`` when the diff can't be computed (no upstream,
        error) — the caller then falls back to an ``--all-files`` run.
        """
        res = self.git_action(
            command="git diff --name-only @{u}..HEAD", path=target_path, quiet=True
        )
        if res.status != "success" or not res.data:
            return []
        return [line.strip() for line in res.data.splitlines() if line.strip()]

    def _gate_before_push(self, target_path: str) -> GitResult | None:
        """Run the repo's declared HEAVY (pre-push-stage) gates before pushing.

        Mirrors the repo's CI gates locally so a push can't ship a commit the CI
        would reject. Returns a failed ``GitResult`` (caller aborts the push) or
        ``None`` to proceed. No-op when disabled, when the repo has no
        ``.pre-commit-config.yaml``, or when there is nothing to push. The
        gate-harness failing (tooling/env) never blocks a push — only a real
        hook failure does.

        Runs ``stage="heavy"`` (``--hook-stage pre-push``) via
        :func:`repository_manager.gates.run_gate_stage` — the fix for the
        two-tier model's blocking gap (GOC-60): this method's name always
        promised "pre-push", but until this change it ran pre-commit's default
        (commit-stage) hooks scoped to the diff, so a repo's HEAVY hooks
        (pytest, cargo, ``uv lock --check``, ...) were never exercised by any
        push. FAST-stage hooks are intentionally NOT re-run here — they already
        ran at commit time; this method's job is exclusively the HEAVY tier
        that only a push can trigger.
        """
        if not self.gate_before_push:
            return None
        if not os.path.exists(os.path.join(target_path, ".pre-commit-config.yaml")):
            return None
        if not self._has_unpushed_commits(target_path):
            return None

        # Scope per-file hooks to the diff being pushed; always_run guardrail
        # gates still run fully. Falls back to --all-files if the diff is empty.
        changed = self._unpushed_changed_files(target_path)
        scope = f"{len(changed)} changed file(s)" if changed else "all files"
        logger.info("Running pre-push (HEAVY) gate over %s", scope)
        try:
            result = run_gate_stage(
                target_path,
                "heavy",
                files=changed or None,
                trigger="pre-push",
                colocated=True,
            )
        except Exception as e:  # pragma: no cover - tooling/env failure
            logger.warning("Operation failed: error_type=%s", type(e).__name__)
            return None
        if result.success:
            return None
        failed = [h.hook_id for h in result.hooks if not h.passed]
        if not failed and result.error:
            # No hook reported a verdict -- the gate did not complete (timeout,
            # tooling error). Reporting that as "Pre-push gate failed (pre-push
            # gate)" made a 600s HEAVY-tier timeout look identical to a real
            # hook failure, which is how agent-utilities appeared to be blocked
            # on merit when it had simply run out of clock. Surface the harness
            # error verbatim instead of inventing a hook name.
            logger.error("Pre-push gate did not complete: %s", result.error)
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=(
                        f"Pre-push gate did not complete; push aborted. {result.error}"
                    ),
                    code=1,
                ),
            )
        # A gate that could not RUN is not a gate that found a defect.
        #
        # If every failing hook failed because its executable is absent, this
        # environment cannot gate this repository at all, and saying "fix the
        # gate" sends the reader hunting for a defect that does not exist. The
        # push is still refused -- an ungated push is worse -- but the reason is
        # reported truthfully so it can be acted on.
        unrunnable = [h.hook_id for h in result.hooks if not h.passed and h.unrunnable]
        if unrunnable and len(unrunnable) == len(failed):
            missing = ", ".join(unrunnable)
            logger.error("Pre-push gate cannot run in this environment: %s", missing)
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=(
                        f"Pre-push gate CANNOT RUN here; push aborted. Every failing "
                        f"hook ({missing}) failed because its executable is missing "
                        f"from this environment, not because it found a defect. This "
                        f"is an environment gap, not a code verdict -- install the "
                        f"toolchain these hooks need, or run the gate and the push "
                        f"from a host that has it."
                    ),
                    code=1,
                ),
            )
        names = ", ".join(failed) or "pre-push gate"
        logger.error("Pre-push gate failed: %s", names)
        return GitResult(
            status="error",
            data="",
            error=GitError(
                message=(
                    f"Pre-push gate failed ({names}); push aborted. "
                    "Fix the gate, or set RM_GATE_BEFORE_PUSH=false to bypass."
                ),
                code=1,
            ),
        )

    @_exclusive_repo_mutation
    def push_project(self, path: str | None = None) -> GitResult:
        """
        Push committed updates and tags for a single clean Git project.

        Handles common failure modes:
        - Non-fast-forward: fails closed for an explicit reviewed sync
        - GitHub secret scanning (GH013): returns actionable error with unblock URL
        - Tag conflicts: falls back to pushing without --follow-tags
        """
        target_path = self._resolve_path(path)
        logger.info("Checking configured project for uncommitted changes")

        status_check = self.git_action(
            command="git status --porcelain", path=target_path, quiet=True
        )
        if status_check.status == "success" and status_check.data.strip():
            logger.warning("Push refused because the configured project is dirty")
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=(
                        "Push refused: the repository has uncommitted changes. "
                        "Review and commit them explicitly before pushing."
                    ),
                    code=409,
                ),
                metadata=GitMetadata(
                    command="git push",
                    workspace=_project_label(target_path),
                    return_code=409,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        # Fast pre-push gate: run the repo's own pre-commit gates (minus the
        # slow full pytest suite) so a push can't ship a commit the repo's CI
        # would reject. Aborts this repo's push on a real gate failure.
        gate = self._gate_before_push(target_path)
        if gate is not None:
            return gate

        logger.info("Pushing latest changes and tags for configured project")

        max_attempts = 1
        for _attempt in range(1, max_attempts + 1):
            result = self.git_action(command="git push --follow-tags", path=target_path)

            if result.status == "success":
                # --follow-tags only pushes ANNOTATED tags. bump2version can emit
                # LIGHTWEIGHT tags (objecttype=commit), which would silently never
                # reach the remote — so no tag-triggered CI / image build. Push the
                # CURRENT release tag explicitly (v<current_version> from
                # .bumpversion.cfg) to cover both annotated and lightweight, WITHOUT
                # also dumping stale never-pushed historical tags onto the remote
                # (which would trigger CI for old versions).
                # (CONCEPT:RM-BUMP tag-publish correctness)
                rel_tag = self._current_release_tag(target_path)
                if rel_tag:
                    tag_res = self.git_action(
                        command=f"git push origin {rel_tag}", path=target_path
                    )
                    if tag_res.status != "success":
                        logger.warning("Branch pushed but the release-tag push failed")
                return result

            error_text = ""
            if result.error:
                error_text = (
                    str(result.error.message)
                    if hasattr(result.error, "message")
                    else str(result.error)
                )
            if result.data:
                error_text += " " + result.data

            # GitHub secret scanning block (GH013) — unrecoverable without manual action
            if "GH013" in error_text or "GITHUB PUSH PROTECTION" in error_text:
                logger.error(
                    "GitHub secret scanning blocked the push; remove the secret from history"
                )
                return GitResult(
                    status="error",
                    data="GitHub push protection blocked the push",
                    error=GitError(
                        message="GitHub secret scanning (GH013) blocked the push. "
                        "A file in the commit history contains a detected secret. "
                        "Use git-filter-repo to expunge it or allow the secret via GitHub settings.",
                        code=1,
                    ),
                    metadata=GitMetadata(
                        command="git push --follow-tags",
                        workspace=_project_label(target_path),
                        return_code=1,
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    ),
                )

            # A divergent remote requires an explicit reviewed sync. Never
            # rewrite remote history or mutate the local branch as a fallback.
            if (
                "non-fast-forward" in error_text
                or "tip of your current branch is behind" in error_text
            ):
                logger.warning("Push refused because the remote branch has diverged")
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=(
                            "Push refused: the remote branch has diverged. "
                            "Fetch and perform an explicit reviewed merge or rebase; "
                            "automatic force-push is permanently disabled."
                        ),
                        code=409,
                    ),
                    metadata=result.metadata,
                )

            # Tag already exists on remote — retry without tags
            if "tag already exists" in error_text:
                logger.warning("Tag conflict detected; retrying without follow-tags")
                return self.git_action(command="git push origin main", path=target_path)

            # Unknown error — return as-is
            return result

        return result

    def add_projects(self, project_dirs: list[str] | None = None) -> list[GitResult]:
        """
        Stage all changes for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to add.")
                return []

        if not project_dirs:
            logger.warning("No projects found to add.")
            return []

        logger.info(
            f"Staging changes in {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(self.add_project, project_dirs))

    @_exclusive_repo_mutation
    def add_project(self, path: str | None = None) -> GitResult:
        """
        Stage all changes (git add -A) for a single Git project.
        """
        target_path = self._resolve_path(path)
        logger.info("Staging all changes for configured project")
        return self.git_action(command="git add -A", path=target_path)

    def commit_projects(
        self, message: str, project_dirs: list[str] | None = None
    ) -> list[GitResult]:
        """
        Commit staged changes for multiple projects in parallel.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to commit.")
                return []

        if not project_dirs:
            logger.warning("No projects found to commit.")
            return []

        logger.info(
            f"Committing changes in {len(project_dirs)} projects in parallel using {self.threads} threads..."
        )
        from functools import partial

        commit_func = partial(self.commit_project, message)
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(commit_func, project_dirs))

    @_exclusive_repo_mutation
    def commit_project(self, message: str, path: str | None = None) -> GitResult:
        """
        Commit staged changes (git commit -m "{message}") for a single Git project.
        """
        target_path = self._resolve_path(path)

        # Check if there are staged changes to commit
        status_res = self.git_action(command="git status --porcelain", path=target_path)
        if status_res.status == "success":
            # Check porcelain output for staged changes
            has_staged = False
            for line in status_res.data.splitlines():
                if line and not line.startswith("?"):
                    # Staged changes are indicated when the first character is not a space/untracked status
                    if line[0] not in (" ", "?"):
                        has_staged = True
                        break

            if not has_staged:
                logger.info("No staged changes to commit for configured project")
                metadata = GitMetadata(
                    command="git commit",
                    workspace=_project_label(target_path),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                )
                return GitResult(
                    status="success",
                    data="No staged changes to commit (skipped)",
                    error=None,
                    metadata=metadata,
                )

        logger.info("Committing staged changes for configured project")
        from shlex import quote

        safe_msg = quote(message)
        return self.git_action(command=f"git commit -m {safe_msg}", path=target_path)

    @_exclusive_repo_mutation
    def commit_code_project(
        self, message: str, run_precommit: bool = True, path: str | None = None
    ) -> GitResult:
        """Stage ALL changes (git add -A), optionally gate on pre-commit, then commit.

        This is the per-repo "commit our feature code" step the release pipeline
        runs BEFORE bumping versions. Unlike :meth:`commit_project` it stages
        untracked files too (``git add -A``) and runs the project's pre-commit
        hooks (auto-formatters land in the same commit), so feature code is never
        left behind for an implicit push-time commit.

        If pre-commit fails for real (not just auto-format), the failure is
        surfaced and nothing is committed.
        """
        target_path = self._resolve_path(path)

        # Un-cloned / missing repo (e.g. a workspace.yml entry not pulled): skip
        # gracefully — a missing dir must never abort the whole batch. D-CDX-60:
        # ``.git`` is a FILE (a gitdir pointer), not a directory, in a linked
        # worktree — an `isdir` check here wrongly reported a valid isolated
        # worktree as "not a cloned Git repository" and skipped it silently.
        # ``os.path.exists`` accepts either shape, matching the check at
        # `_resolve_path`'s sibling validation above.
        if not os.path.exists(os.path.join(target_path, ".git")):
            return GitResult(
                status="skipped",
                data="Configured project is not a cloned Git repository",
                error=None,
                metadata=GitMetadata(
                    command="commit_code",
                    workspace=_project_label(target_path),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        status_res = self.git_action(
            command="git status --porcelain", path=target_path, quiet=True
        )
        if status_res.status == "success" and not status_res.data.strip():
            return GitResult(
                status="skipped",
                data="No changes to commit.",
                error=None,
                metadata=GitMetadata(
                    command="commit_code",
                    workspace=_project_label(target_path),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        # The stage, optional gate, re-stage, and commit intentionally happen in
        # this one call.  Callers must use this ordered operation instead of
        # racing separately submitted ``add`` and ``commit`` background jobs.
        stage_res = self.add_project(target_path)
        if stage_res.status != "success":
            return stage_res

        if run_precommit and os.path.exists(
            os.path.join(target_path, ".pre-commit-config.yaml")
        ):
            pc_res = self.pre_commit(run=True, autoupdate=False, path=target_path)
            if pc_res.status == "error":
                return pc_res

        # Stage again (pre-commit may have reformatted files) and commit.
        stage_res = self.add_project(target_path)
        if stage_res.status != "success":
            return stage_res
        result = self.commit_project(message, path=target_path)
        if result.status == "success":
            # D-CDX-60 acceptance: a truthful commit_code result names BOTH the
            # resolved repository/worktree it acted on (already carried by
            # metadata.workspace) AND the resulting commit SHA, so a caller can
            # verify what actually happened rather than trust a bare "success".
            sha_res = self.git_action(
                command="git rev-parse HEAD", path=target_path, quiet=True
            )
            if sha_res.status == "success" and sha_res.data.strip():
                result = result.model_copy(
                    update={
                        "data": (f"{result.data}\ncommit_sha={sha_res.data.strip()}")
                    }
                )
        return result

    def commit_code_projects(
        self,
        message: str,
        run_precommit: bool = True,
        project_dirs: list[str] | None = None,
    ) -> list[GitResult]:
        """Concurrently stage + pre-commit + commit feature code across projects.

        The "add all our code, pre-commit, then commit" release-prep step,
        throttled by ``self.threads`` (the 20% CPU/RAM cap). Scales to thousands
        of repositories.
        """
        if project_dirs is None:
            if self.project_map:
                project_dirs = list(self.project_map.values())
            else:
                logger.warning("No projects found in project_map to commit_code.")
                return []
        if not project_dirs:
            logger.warning("No projects found to commit_code.")
            return []

        logger.info(
            f"Committing feature code in {len(project_dirs)} projects in parallel "
            f"(pre_commit={run_precommit}) using {self.threads} threads..."
        )

        def _safe(d: str) -> GitResult:
            # Isolate every repo: one raising item must never abort the batch
            # (which would cascade-skip the bump + push). Convert to an error
            # GitResult instead.
            try:
                return self.commit_code_project(message, run_precommit, d)
            except Exception as exc:  # noqa: BLE001
                logger.error("commit_code failed: error_type=%s", type(exc).__name__)
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(
                        message=f"commit_code {d}: {type(exc).__name__}", code=1
                    ),
                    metadata=GitMetadata(
                        command="commit_code",
                        workspace=_project_label(d),
                        return_code=1,
                        timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                    ),
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(_safe, project_dirs))

    def set_threads(self, threads: int) -> None:
        """
        Set the number of threads for parallel processing.

        Args:
            threads (int): The number of threads.

        Notes:
            If the input is invalid, defaults 6
        """
        try:
            if 0 < threads <= self.maximum_threads:
                self.threads = threads
            else:
                logger.warning(
                    f"Did not recognize {threads} as a valid value, defaulting to: {self.maximum_threads}"
                )
                self.threads = self.maximum_threads
        except Exception as e:
            logger.error(
                "Invalid worker-count configuration; using safe default: error_type=%s",
                type(e).__name__,
            )
            self.threads = self.maximum_threads

    @staticmethod
    def _precommit_skipped(target_path: str, reason: str) -> GitResult:
        """A no-op pre-commit outcome for a project there is nothing to run on."""
        return GitResult(
            status="skipped",
            data=reason,
            error=None,
            metadata=GitMetadata(
                command="pre_commit_check",
                workspace=_project_label(target_path),
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    @staticmethod
    def _precommit_env() -> dict[str, str]:
        """Environment for a pre-commit run.

        Skips the branch lock (this helper is used off-branch on purpose) and
        bounds any pytest hook so a repo's own ``-n auto`` cannot fan out.
        """
        env = os.environ.copy()
        if "SKIP" in env:
            env["SKIP"] += ",no-commit-to-branch"
        else:
            env["SKIP"] = "no-commit-to-branch"
        env["PYTEST_XDIST_AUTO_NUM_WORKERS"] = "4"
        lane_pytest_options = env.get("PYTEST_ADDOPTS", "").strip()
        bounded_pytest_options = '-q --tb=short -m "not slow" --timeout=60'
        env["PYTEST_ADDOPTS"] = " ".join(
            option for option in (lane_pytest_options, bounded_pytest_options) if option
        )
        return env

    def _run_precommit_autoupdate(self, target_path: str, env: dict) -> GitResult:
        """``pre-commit autoupdate``, then stage whatever it rewrote.

        Returns the first error encountered, or the autoupdate result itself.
        """
        result = self.git_action(
            command="pre-commit autoupdate",
            path=target_path,
            env=env,
            timeout=600,
        )
        if result.status == "error":
            return result
        staged = self.git_action(command="git add -A", path=target_path)
        if staged.status == "error":
            return staged
        return result

    def _run_precommit_hooks(
        self, target_path: str, env: dict
    ) -> tuple[GitResult, bool]:
        """Stage, run the FAST-tier hooks, and retry once after reformatting.

        Returns ``(result, is_final)``; ``is_final`` marks a *staging* failure,
        which the caller must surface verbatim without the branch-lock
        post-processing that applies to hook results.
        """
        staged = self.git_action(command="git add -A", path=target_path)
        if staged.status == "error":
            return staged, True

        # Explicit, not pre-commit's implicit default: this method is the
        # FAST tier's interactive stage-and-commit helper (CONCEPT
        # GOC-59/60 two-tier gate model) -- declare the stage it runs
        # rather than relying on `pre-commit run` defaulting to it.
        hook_stage = HOOK_STAGE_BY_GATE_STAGE["fast"]
        hook_command = f"pre-commit run --hook-stage {hook_stage} --all-files --verbose"
        result = self.git_action(
            command=hook_command, path=target_path, env=env, timeout=600
        )
        if result.status == "error":
            # Hooks may have reformatted files. Stage those bounded changes
            # and run once more, without a shell retry expression.
            restaged = self.git_action(command="git add -A", path=target_path)
            if restaged.status == "error":
                return restaged, True
            result = self.git_action(
                command=hook_command, path=target_path, env=env, timeout=600
            )

        self.git_action(command="git add -A", path=target_path)
        return result, False

    @staticmethod
    def _is_branch_lock_only_failure(result: GitResult) -> bool:
        """True when the ONLY pre-commit failure was the no-commit-to-branch lock."""
        if result.status != "error" or not result.error:
            return False
        msg = result.error.message.lower()
        if "don't commit to branch" not in msg and "no-commit-to-branch" not in msg:
            return False
        lines = (result.error.message + "\n" + result.data).splitlines()
        return not any(
            "Failed" in line and "don't commit to branch" not in line.lower()
            for line in lines
        )

    @staticmethod
    def _branch_lock_success(result: GitResult) -> GitResult:
        """Re-badge a branch-lock-only failure as the success it really was."""
        return GitResult(
            status="success",
            data=result.data or "Skipped branch lock check",
            metadata=result.metadata,
        )

    @_exclusive_repo_mutation
    def pre_commit(
        self,
        run: bool = True,
        autoupdate: bool = False,
        path: str | None = None,
    ) -> GitResult:
        """
        Execute pre-commit commands in the specified path.

        Args:
            run (bool): Whether to run 'pre-commit run --all-files'. Default True.
            autoupdate (bool): Whether to run 'pre-commit autoupdate'. Default False.
            path (str, optional): Path to run in. Defaults to self.path.
        """
        target_path = self._resolve_path(path)

        # Clean artifacts before running pre-commit
        self.cleanup_artifacts(target_path)

        if not os.path.exists(os.path.join(target_path, ".pre-commit-config.yaml")):
            return self._precommit_skipped(
                target_path, "No .pre-commit-config.yaml found."
            )

        if not autoupdate and not run:
            return self._precommit_skipped(
                target_path, "No action selected (run=False, autoupdate=False)."
            )

        env = self._precommit_env()

        result: GitResult | None = None
        if autoupdate:
            result = self._run_precommit_autoupdate(target_path, env)
            if result.status == "error":
                return result

        if run:
            result, is_final = self._run_precommit_hooks(target_path, env)
            if is_final:
                return result

        if result is None:
            raise RuntimeError("pre-commit operation produced no result")

        if self._is_branch_lock_only_failure(result):
            logger.info(
                f"Ignoring safe pre-commit failure (branch lock) in {target_path}"
            )
            return self._branch_lock_success(result)

        return result

    def _run_project_test(
        self, cmd: str, path: str, env: dict, timeout: int
    ) -> list[GitResult]:
        results = []
        res = self.git_action(cmd, path=path, env=env, timeout=timeout)
        results.append(res)
        return results

    @staticmethod
    def _find_test_target(path: str) -> str | None:
        """The pytest target dir for *path* -- unit test dirs preferred."""
        for candidate in ("tests/unit", "test/unit", "tests", "test"):
            if os.path.exists(os.path.join(path, candidate)):
                return candidate
        return None

    def _project_test_plan(self, path: str) -> tuple[str | None, str | None]:
        """``(pytest target dir, skip reason)`` for one project.

        Exactly one half is ever set. Order matters: a project with neither a
        pre-commit config nor a ``pyproject.toml`` is reported as unconfigured
        even when it happens to carry a tests directory.
        """
        has_precommit = os.path.exists(os.path.join(path, ".pre-commit-config.yaml"))
        has_pyproject = os.path.exists(os.path.join(path, "pyproject.toml"))
        if not has_precommit and not has_pyproject:
            return None, "Skipped (No .pre-commit-config.yaml and no pyproject.toml)"

        test_target = self._find_test_target(path)
        if test_target is None:
            return None, "No tests directory found"
        return test_target, None

    @staticmethod
    def _skipped_test_result(path: str, reason: str) -> GitResult:
        """The ``skipped`` record for a project that cannot be pytest'd."""
        return GitResult(
            status="skipped",
            data=reason,
            metadata=GitMetadata(
                command="pytest",
                workspace=_project_label(path),
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    @staticmethod
    def _pytest_command(path: str, test_target: str) -> str:
        """The pytest invocation for *path*, preferring uv when it is uv-locked."""
        bounded = '-q --tb=short -m "not slow" --timeout=60'
        if os.path.exists(os.path.join(path, "uv.lock")):
            return f"uv run --extra test pytest {test_target} {bounded}"
        return f"{sys.executable} -m pytest {test_target} {bounded}"

    @staticmethod
    def _pytest_environment() -> dict[str, str]:
        """Test env: memory-safe ladybug, validation mode, in-memory graph."""
        test_env = os.environ.copy()
        test_env["LADYBUG_MAX_DB_SIZE"] = "1073741824"
        test_env["VALIDATION_MODE"] = "True"
        test_env["KNOWLEDGE_GRAPH_SYNC_BACKGROUND"] = "False"
        test_env["GRAPH_DB_PATH"] = ":memory:"
        return test_env

    @staticmethod
    def _mark_repo_progress(
        progress_dict: dict | None,
        progress_phase: str | None,
        repo_name: str,
        status: str,
        *,
        recount_failures: bool = False,
    ) -> None:
        """Record one repo's status in the shared live-progress mapping."""
        if not progress_dict or not progress_phase:
            return
        phases = progress_dict.get("phases", {})
        if progress_phase not in phases:
            return
        phase = phases[progress_phase]
        phase["repos"][repo_name] = status
        phase["completed"] = len(phase["repos"])
        if recount_failures:
            phase["failed"] = sum(1 for s in phase["repos"].values() if s == "error")

    @staticmethod
    def _collect_project_test_results(
        future: "concurrent.futures.Future", results: list[GitResult]
    ) -> str:
        """Append one project's test results; return its aggregate status."""
        res_list = future.result()
        if not isinstance(res_list, list):
            results.append(res_list)
            return res_list.status
        results.extend(res_list)
        if any(r.status == "error" for r in res_list):
            return "error"
        return "success"

    def test_projects(
        self,
        targets: list[dict[str, str]],
        progress_phase: str | None = None,
        progress_dict: dict | None = None,
    ) -> list[GitResult]:
        """
        Execute pytests for the specified projects in parallel.

        Args:
            progress_phase: Phase name for live progress updates.
            progress_dict: Shared mutable dict for live progress reporting.
        """
        results: list[GitResult] = []
        thread_count = self._cpu_aware_threads()
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=thread_count
        ) as executor:
            future_to_repo: dict[concurrent.futures.Future, str] = {}
            for target in targets:
                if "skip_reason" in target:
                    continue

                path = target["path"]
                repo_name = target.get("name", os.path.basename(path))

                test_target, skip_reason = self._project_test_plan(path)
                if skip_reason is not None:
                    results.append(self._skipped_test_result(path, skip_reason))
                    self._mark_repo_progress(
                        progress_dict, progress_phase, repo_name, "skipped"
                    )
                    continue

                fut = executor.submit(
                    self._run_project_test,
                    self._pytest_command(path, test_target),
                    path,
                    self._pytest_environment(),
                    600,  # 10 minute timeout for tests
                )
                future_to_repo[fut] = repo_name

            for future in concurrent.futures.as_completed(future_to_repo):
                repo_name = future_to_repo[future]
                status = self._collect_project_test_results(future, results)
                self._mark_repo_progress(
                    progress_dict,
                    progress_phase,
                    repo_name,
                    status,
                    recount_failures=True,
                )
        return results

    def _named_precommit_dirs(self, projects: list[str]) -> list[str]:
        """Resolve an explicit project list to hook-carrying directories."""
        dirs: list[str] = []
        for p in projects:
            if os.path.isabs(p) and os.path.exists(p):
                p_path: str | None = p
            else:
                p_path = self._project_path_for(p)
            if (
                p_path
                and os.path.isdir(p_path)
                and os.path.exists(os.path.join(p_path, ".pre-commit-config.yaml"))
            ):
                dirs.append(p_path)
        return dirs

    def _precommit_project_dirs(self, projects: list[str] | None) -> list[str]:
        """Directories carrying a ``.pre-commit-config.yaml`` for the given scope.

        ``projects=None`` means "every mapped project"; an empty project map is
        warned about and yields nothing to run.
        """
        if projects is not None:
            return self._named_precommit_dirs(projects)
        if not self.project_map:
            logger.warning("No projects found in project_map for pre-commit.")
            return []
        return [
            p
            for p in self.project_map.values()
            if os.path.exists(os.path.join(p, ".pre-commit-config.yaml"))
        ]

    def _run_precommit_pool(
        self, project_dirs: list[str], run: bool, autoupdate: bool
    ) -> list[GitResult]:
        """Run pre-commit across *project_dirs* in parallel."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            futures = {
                executor.submit(self.pre_commit, run, autoupdate, d): d
                for d in project_dirs
            }
            return [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

    def _precommit_projects_error(self, exc: Exception) -> GitResult:
        """The failure record for a parallel pre-commit sweep that blew up."""
        return GitResult(
            status="error",
            data="",
            error=GitError(
                message=f"Parallel pre-commit failed: {type(exc).__name__}",
                code=-1,
            ),
            metadata=GitMetadata(
                command="pre_commit_projects",
                workspace=_project_label(self.path),
                return_code=-1,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def pre_commit_projects(
        self,
        run: bool = True,
        autoupdate: bool = False,
        projects: list[str] | None = None,
    ) -> list[GitResult]:
        """
        Execute pre-commit commands for all projects in parallel.

        Returns:
            List[GitResult]: A list of GitResult objects.
        """
        try:
            expanded_path = os.path.expanduser(self.path)
            if not os.path.exists(expanded_path):
                return []

            project_dirs = self._precommit_project_dirs(projects)
            if not project_dirs:
                return []

            return self._run_precommit_pool(project_dirs, run, autoupdate)

        except Exception as e:
            logger.error("Parallel pre-commit failed: error_type=%s", type(e).__name__)
            return [self._precommit_projects_error(e)]

    def install_project(self, path: str | None = None, extra: str = "all") -> GitResult:
        """
        Install a Python project using pip install -e .[extra].
        """
        target_path = self._resolve_path(path)

        command = self._get_pip_command(extra)

        logger.info("Installing configured project")
        result = self.git_action(command=command, path=target_path)

        for d in ["build", "dist"]:
            shutil.rmtree(os.path.join(target_path, d), ignore_errors=True)
        for egg_info in Path(target_path).glob("*.egg-info"):
            shutil.rmtree(egg_info, ignore_errors=True)

        return result

    def get_readme(self, path: str | None = None) -> ReadmeResult:
        """
        Get the content and path of the README.md file in the specified path.

        Args:
            path (str, optional): The directory path. Defaults to self.path.

        Returns:
            ReadmeResult: Object containing 'content' and 'path' of the README.md file.
        """
        target_dir = self._resolve_path(path)

        if not os.path.exists(target_dir):
            return ReadmeResult(content="", path="")

        readme_path = None
        for filename in os.listdir(target_dir):
            if filename.lower() == "readme.md":
                readme_path = os.path.join(target_dir, filename)
                break

        if not readme_path:
            return ReadmeResult(content="", path="")

        try:
            with open(readme_path, encoding="utf-8") as f:
                content = f.read()
            return ReadmeResult(content=content, path=readme_path)
        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)
            return ReadmeResult(content="", path=readme_path)

    def create_project(self, path: str) -> GitResult:
        """
        Create a new project directory and initialize it as a git repository.

        Args:
            path (str): The path of the project directory to create.

        Returns:
            GitResult: Result of the operation.
        """
        target_path = self._resolve_path(path)

        if os.path.exists(target_path):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message="Configured target directory already exists", code=1
                ),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=_project_label(target_path),
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        try:
            os.makedirs(target_path, exist_ok=True)
            init_result = self.git_action("git init", path=target_path)

            if init_result.status == "success":
                logger.info("Repository project created")
                return init_result
            else:
                return init_result

        except Exception as e:
            logger.error(
                "Failed to create repository project: error_type=%s",
                type(e).__name__,
            )
            return GitResult(
                status="error",
                data="",
                error=GitError(message=type(e).__name__, code=1),
                metadata=GitMetadata(
                    command="create_project",
                    workspace=_project_label(target_path),
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

    def _bump_skip_reason(self, project_dir: str) -> str | None:
        """Why a bump should be skipped for this repo, or ``None`` if it needs one.

        Returns a human-readable reason when no (further) bump is warranted:

        * **no code changes** — clean tree AND in sync with origin.
        * **already bumped, awaiting push** — clean tree but AHEAD of origin
          with a ``Bump version:`` commit at HEAD. Re-bumping here is the
          double-bump bug: while the push step is starved, every retry sees the
          repo as "not up to date" and bumps again (0.38→0.39→0.40 …). The push
          step will deliver the existing bump, so skip.

        A clean tree whose HEAD is a *feature* commit (committed but not yet
        version-bumped) returns ``None`` so it still gets its first bump — the
        fix narrows the skip to genuine no-ops, it does not suppress real bumps.
        (CONCEPT:RM-BUMP idempotency)
        """
        status_check = self.git_action("git status", path=project_dir)
        data_lower = status_check.data.lower() if status_check.data else ""
        clean = "nothing to commit" in data_lower
        up_to_date = "your branch is up to date" in data_lower
        if clean and up_to_date:
            return "no code changes detected (use force=True to override)"
        if clean and not up_to_date:
            head_subj = self.git_action(
                "git log -1 --pretty=%s", path=project_dir, quiet=True
            )
            subject = (head_subj.data or "").strip().lower()
            if subject.startswith("bump version:"):
                return "already bumped, awaiting push (avoids double-bump)"
        return None

    @staticmethod
    def _build_phase_map(config: dict) -> tuple[dict[str, int], int]:
        """Map each explicitly-named project to its phase number.

        Returns ``(project_phases, bulk_phase_num)`` where ``project_phases``
        maps a project name to the phase that names it, and ``bulk_phase_num``
        is the phase carrying ``bulk_bump``/``bulk_push`` (the catch-all every
        unnamed repo falls into). Mirrors the per-project phase resolution
        already used by :meth:`phased_bumpversion`.
        """
        project_phases: dict[str, int] = {}
        bulk_phase_num = 5
        for phase in config.get("phases", []):
            p_num = phase.get("phase", 1)
            if phase.get("bulk_bump") or phase.get("bulk_push"):
                bulk_phase_num = p_num
            projects_in_phase = phase.get("projects", [])[:]
            if phase.get("project"):
                projects_in_phase.append(phase.get("project"))
            for p in projects_in_phase:
                project_phases[p] = p_num
        return project_phases, bulk_phase_num

    def _repo_has_pending_work(self, project_dir: str) -> bool:
        """True when a repo has anything to bump or push.

        A repo that is both clean and in sync with origin has no uncommitted
        changes, no unpushed feature commits, and no unpushed version bump — so
        neither the bump nor the push step would act on it. This is the same
        no-op test :meth:`_bump_skip_reason` uses; anything else (dirty tree,
        ahead of origin) is treated as pending work.
        """
        status_check = self.git_action("git status", path=project_dir, quiet=True)
        data_lower = status_check.data.lower() if status_check.data else ""
        clean = "nothing to commit" in data_lower
        up_to_date = "your branch is up to date" in data_lower
        return not (clean and up_to_date)

    def _auto_start_phase(self, config: dict) -> int | None:
        """Lowest phase number that contains a repo with pending work.

        Phases are topologically ordered (lower phase = more upstream): a change
        in phase *N* can only cascade to phases ``>= N`` via dependency-pin
        propagation, never to an earlier phase. So the bump/push can safely begin
        at the lowest phase that actually has work and still capture every
        downstream effect, skipping purely-unaffected upstream phases (and their
        inter-phase waits). Returns ``None`` when no repo has pending work — the
        caller should then do nothing. (CONCEPT:RM-PHASE-START)
        """
        project_phases, bulk_phase_num = self._build_phase_map(config)
        lowest: int | None = None
        for url, path in self.project_map.items():
            name = url.split("/")[-1].replace(".git", "")
            phase_num = project_phases.get(name, bulk_phase_num)
            # Once a candidate is found, only earlier phases can lower it.
            if lowest is not None and phase_num >= lowest:
                continue
            if self._repo_has_pending_work(path):
                lowest = phase_num
        return lowest

    @_exclusive_repo_mutation
    def bump_version(
        self,
        part: str,
        allow_dirty: bool = False,
        path: str | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        force: bool = False,
    ) -> GitResult:
        """
        Bump the version of the project using bump2version.

        Args:
            part (str): The part of the version to bump (major, minor, patch).
            allow_dirty (bool): Whether to allow dirty working directory.
            path (str): The path to the project directory.
            dry_run (bool): Whether to perform a dry run.
            verbose (bool): Whether to use verbose output (for dry-run visibility).
            force (bool): If the target version's tag already exists locally (an
                orphan tag from a prior partial bump that left the version file
                un-updated), delete that local tag and re-bump instead of
                silently skipping. The orphan tag must NOT be on the remote.

        Returns:
            GitResult: Result of the operation.
        """
        target_dir = self._resolve_path(path)

        validation_error = self._bump_version_validate_target(target_dir, part)
        if validation_error is not None:
            return validation_error

        if not self._project_has_bumpversion_config(target_dir):
            return self._bump_version_fallback(target_dir, dry_run)

        command = self._build_bump2version_command(part, allow_dirty, dry_run, verbose)

        if not dry_run:
            preflight_result = self._bump_version_preflight_tag_check(
                target_dir, part, allow_dirty, force
            )
            if preflight_result is not None:
                return preflight_result
            command += " --list"

        return self._run_bump2version(command, target_dir, part, dry_run)

    def _bump_version_validate_target(
        self, target_dir: str, part: str
    ) -> GitResult | None:
        """Guard clauses for `bump_version`: missing directory / invalid
        ``part``. Returns a GitResult to short-circuit, or None to continue."""
        if not os.path.exists(target_dir):
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message="Configured project directory was not found",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=_project_label(target_dir),
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        valid_parts = ["major", "minor", "patch"]
        if part not in valid_parts:
            return GitResult(
                status="error",
                data="",
                error=GitError(
                    message=f"Invalid part '{part}'. Must be one of {valid_parts}",
                    code=1,
                ),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=_project_label(target_dir),
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )
        return None

    def _project_has_bumpversion_config(self, target_dir: str) -> bool:
        """Whether *target_dir* declares a bump2version configuration
        (``.bumpversion.cfg``, or a ``[bumpversion]`` section in
        ``setup.cfg``)."""
        has_cfg = os.path.exists(os.path.join(target_dir, ".bumpversion.cfg"))
        if not has_cfg and os.path.exists(os.path.join(target_dir, "setup.cfg")):
            try:
                with open(os.path.join(target_dir, "setup.cfg"), encoding="utf-8") as f:
                    if "[bumpversion]" in f.read():
                        has_cfg = True
            except Exception as e:
                logger.debug("Operation failed: error_type=%s", type(e).__name__)
        return has_cfg

    def _bump_version_fallback(self, target_dir: str, dry_run: bool) -> GitResult:
        """Fallback behavior for a project with no bump2version config: stage
        all changes and commit them as "phased bump"."""
        status_check = self.git_action(
            command="git status --porcelain", path=target_dir, quiet=True
        )
        if status_check.status != "success":
            return status_check

        changed_files = status_check.data.strip()
        if not changed_files:
            logger.info("No changes to stage or commit; skipping configured project")
            return GitResult(
                status="skipped",
                data="No changes to stage or commit (fallback mode)",
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=_project_label(target_dir),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        if dry_run:
            logger.info(
                f"[DRY RUN] Would fallback to git add -A && git commit -m 'phased bump' in {target_dir}"
            )
            return GitResult(
                status="success",
                data="current_version=unknown\nnew_version=unknown\n",
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=_project_label(target_dir),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

        add_res = self.git_action(command="git add -u", path=target_dir)
        if add_res.status != "success":
            logger.error("Failed to add changes for configured project")
            return add_res

        commit_res = self.git_action(
            command='git commit -m "phased bump"', path=target_dir
        )
        if commit_res.status != "success":
            logger.error("Failed to commit fallback changes")
            return commit_res

        logger.info("Successfully committed fallback changes with phased bump")
        return GitResult(
            status="success",
            data="current_version=unknown\nnew_version=unknown\n",
            metadata=GitMetadata(
                command="bump_version",
                workspace=_project_label(target_dir),
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _build_bump2version_command(
        self, part: str, allow_dirty: bool, dry_run: bool, verbose: bool
    ) -> str:
        command = (
            f"SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build bump2version {part}"
        )
        if allow_dirty:
            command += " --allow-dirty"
        if dry_run:
            command += " --dry-run"
        if verbose:
            command += " --verbose"
        return command

    def _bump_version_preflight_tag_check(
        self, target_dir: str, part: str, allow_dirty: bool, force: bool
    ) -> GitResult | None:
        """Pre-flight check for an existing tag on the version bump2version
        would produce. Returns a GitResult to short-circuit `bump_version`
        (tag exists and is not force-deletable), or None to proceed with the
        real bump2version invocation (including after deleting an orphan
        local tag under ``force``)."""
        pre_cmd = f"bump2version {part} --dry-run --list"
        if allow_dirty:
            pre_cmd += " --allow-dirty"
        pre_result = self.git_action(command=pre_cmd, path=target_dir, quiet=True)
        if pre_result.status != "success":
            return None

        match = re.search(r"new_version=(.*)", pre_result.data)
        if not match:
            return None

        new_version = match.group(1).strip()
        tag_check = self.git_action(
            command=f"git tag -l v{new_version}",
            path=target_dir,
            quiet=True,
        )
        if not (tag_check.status == "success" and f"v{new_version}" in tag_check.data):
            return None

        if force and not self._tag_on_remote(f"v{new_version}", target_dir):
            # Orphan local tag from a prior partial bump (version
            # file never updated). Delete it locally and re-bump so
            # the version actually advances. Never touch a remote
            # tag this way.
            logger.warning(
                "Tag v%s exists locally in %s but force=True and it "
                "is not on the remote — deleting orphan tag and "
                "re-bumping.",
                new_version,
                target_dir,
            )
            self.git_action(
                command=f"git tag -d v{new_version}",
                path=target_dir,
                quiet=True,
            )
            return None

        logger.warning(
            f"Tag v{new_version} already exists in {target_dir}. "
            "Skipping bump." + ("" if force else " (use force=True to override)")
        )
        return GitResult(
            status="skipped",
            data=f"current_version={new_version}\nnew_version={new_version}\ntag_exists=true\n",
            metadata=GitMetadata(
                command="bump_version",
                workspace=_project_label(target_dir),
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _run_bump2version(
        self, command: str, target_dir: str, part: str, dry_run: bool
    ) -> GitResult:
        try:
            result = self.git_action(command=command, path=target_dir)

            if result.status == "success":
                logger.info("Bumped configured project version: part=%s", part)

                if not dry_run:
                    self._finalize_successful_bump(target_dir, result)
            else:
                logger.error("Failed to bump configured project version")

            return result
        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)
            return GitResult(
                status="error",
                data="",
                error=GitError(message=type(e).__name__, code=1),
                metadata=GitMetadata(
                    command="bump_version",
                    workspace=_project_label(target_dir),
                    return_code=1,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )

    def _finalize_successful_bump(self, target_dir: str, result: GitResult) -> None:
        """Post-success sequence for a real bump2version run: sync uv.lock,
        stage everything, and -- IF there is anything staged -- fold it into
        the bump commit (``commit --amend``) and re-point the tag. Step order
        here is exactly the release-path stage/commit/tag sequence and must
        never be reordered (CX WC1-REPOSITORY-01: this fleet has a documented
        failure where bump2version stages everything and then fails to
        commit, leaving a half-applied bump with no tag)."""
        # Synchronize uv.lock after pyproject.toml version bump
        uv_lock_path = os.path.join(target_dir, "uv.lock")
        if os.path.exists(uv_lock_path):
            self.git_action(command="uv lock", path=target_dir, quiet=True)

        # Stage all changes (staged and uncommitted/unstaged changes) in the workspace
        self.git_action(command="git add -u", path=target_dir, quiet=True)
        status_check = self.git_action(
            command="git status --porcelain",
            path=target_dir,
            quiet=True,
        )
        if status_check.data.strip():
            # Commit all staged changes (including version bump, uv.lock, and other files) into the bump commit
            self.git_action(
                command="SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build git commit --amend --no-edit",
                path=target_dir,
                quiet=True,
            )

            # Move the tag to point to the newly amended commit
            match = re.search(r"new_version=(.*)", result.data)
            if match:
                new_version = match.group(1).strip()
                self.git_action(
                    command=f"git tag -f v{new_version}",
                    path=target_dir,
                    quiet=True,
                )

    def bulk_bump(
        self,
        part: str,
        dry_run: bool = False,
        exclude: list[str] | None = None,
        verbose: bool = False,
    ) -> list[GitResult]:
        """Bumps the version for all projects in the workspace in parallel."""
        exclude = exclude or []
        results = []

        for url, path in self.project_map.items():
            name = url.split("/")[-1].replace(".git", "")
            if name in exclude:
                continue

            project_dir = Path(path)
            results.append(
                self.bump_version(
                    part,
                    allow_dirty=True,
                    path=str(project_dir),
                    dry_run=dry_run,
                    verbose=verbose,
                )
            )
        return results

    def update_dependency(
        self, file_path: str, package_name: str, new_version: str, dry_run: bool = False
    ) -> bool:
        """Update a package's pinned version in a deps file (pyproject OR requirements).

        Handles every common pin shape so cross-dependency bumps propagate fully
        (previously only ``>=`` in quoted pyproject entries was matched, which
        silently left ``==`` pins and ALL ``requirements.txt`` references stale):

        * quoted (pyproject ``"pkg>=1.2.3"``) AND unquoted (requirements
          ``pkg==1.2.3``) — the optional surrounding quote is preserved.
        * optional extras: ``pkg[all]==1.2.3``.
        * operators: ``==`` ``>=`` ``<=`` ``~=`` ``!=`` ``>`` ``<`` (the captured
          operator is preserved — an ``==`` pin stays ``==`` at the new version).

        Skips transitive ``# via pkg`` comment lines (no operator+version → no match).
        (CONCEPT:RM-BUMP cross-dependency propagation)
        """
        target_file = Path(self._resolve_path(file_path))
        if not target_file.exists() or not target_file.is_file():
            return False

        content = target_file.read_text()
        pattern = (
            rf'(["\']?{re.escape(package_name)}(?:\[[^\]]*\])?\s*'
            r"(?:==|>=|<=|~=|!=|>|<)\s*)\d+\.\d+\.\d+"
        )
        replacement = rf"\g<1>{new_version}"

        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            if not dry_run:
                target_file.write_text(new_content)
            logger.info(
                f"{'[DRY RUN] Would update' if dry_run else 'Updated'} "
                f"{package_name} -> {new_version} ({count}x) in {target_file}"
            )
            return True
        return False

    @staticmethod
    def _project_phase_index(config: dict) -> tuple[dict[str, int], int]:
        """Map every declared project to its phase number, plus the bulk phase.

        Used for the topological dependency check: a bump must not be
        propagated backwards into a project owned by an earlier phase.
        """
        project_phases: dict[str, int] = {}
        bulk_phase_num = 5
        for phase in config.get("phases", []):
            p_num = phase.get("phase", 1)
            if phase.get("bulk_bump"):
                bulk_phase_num = p_num
            projects_in_phase = phase.get("projects", [])[:]
            if phase.get("project"):
                projects_in_phase.append(phase.get("project"))
            for p in projects_in_phase:
                project_phases[p] = p_num
        return project_phases, bulk_phase_num

    @staticmethod
    def _pre_commit_project_names(config: dict) -> list[Any] | None:
        """Projects the pre-commit stage should cover, or ``None`` for "all".

        A ``bulk_bump`` phase means the run sweeps everything, so scoping the
        pre-commit stage to a name list would be wrong -- ``None`` is returned
        the moment one is seen.
        """
        if not config:
            return None
        projects_to_check: list[Any] = []
        for phase in config.get("phases", []):
            if phase.get("bulk_bump"):
                return None
            projects_to_check.extend(phase.get("projects", []))
            if phase.get("project"):
                projects_to_check.append(phase.get("project"))
        return projects_to_check

    def _pre_commit_project_dirs(
        self, projects_to_check: list[Any] | None
    ) -> list[str]:
        """Local clone paths for the pre-commit stage's project scope."""
        if projects_to_check is None:
            return list(self.project_map.values())
        dirs: list[str] = []
        for p_name in projects_to_check:
            p_path = self._project_path_for(p_name)
            if p_path is not None:
                dirs.append(p_path)
        return dirs

    def _run_bump_pre_commit_stage(self, config: dict) -> list[GitResult]:
        """Run pre-commit (with autoupdate) and commit the resulting formatting."""
        projects_to_check = self._pre_commit_project_names(config)
        results = list(
            self.pre_commit_projects(
                run=True, autoupdate=True, projects=projects_to_check
            )
        )
        results.extend(
            self.commit_projects(
                message="chore: pre-commit autoupdate and formatting",
                project_dirs=self._pre_commit_project_dirs(projects_to_check),
            )
        )
        return results

    def _unassigned_project_names(
        self, assigned_projects: set[str], claimed: list[str]
    ) -> list[str]:
        """Mapped project names not claimed by an earlier phase or *claimed*."""
        names: list[str] = []
        for url in self.project_map:
            name = url.split("/")[-1].replace(".git", "")
            if name not in assigned_projects and name not in claimed:
                names.append(name)
        return names

    def _bump_phase_projects(
        self, phase: dict, filter_set: set[str] | None, assigned_projects: set[str]
    ) -> list[str]:
        """The project names one configured phase contributes to the bump plan.

        When a ``project_filter`` set is active, a bulk phase contributes exactly
        the not-yet-assigned filter members (so filtered agents/services land in
        the bulk phase). Without a filter, bulk sweeps everything unassigned.
        """
        projects = phase.get("projects", [])[:]
        if phase.get("project"):
            projects.append(phase.get("project"))
        if filter_set is not None:
            projects = [p for p in projects if p in filter_set]

        if not phase.get("bulk_bump"):
            return projects
        if filter_set is None:
            projects.extend(self._unassigned_project_names(assigned_projects, projects))
            return projects
        for name in filter_set:
            if name not in assigned_projects and name not in projects:
                projects.append(name)
        return projects

    @staticmethod
    def _parse_project_filter(project_filter: str | None) -> set[str] | None:
        """Parse ``project_filter`` into a set of project names, or ``None``.

        ``project_filter`` may be a single name or a comma-separated set, letting
        a caller re-bump exactly N specific repos (e.g. repos a prior partial
        run silently skipped) without re-bumping the whole ecosystem. When a
        filter set is active, the bulk phase is restricted to its members.
        """
        if not project_filter:
            return None
        return {p.strip() for p in project_filter.split(",") if p.strip()}

    def _build_bump_phase_list(
        self, *, config: dict, start_phase: int, filter_set: set[str] | None
    ) -> tuple[list[dict[str, Any]], int]:
        """Expand the configured phases into the concrete bump plan.

        A project claimed by an earlier phase is dropped from every later one --
        otherwise a project named in an explicit phase (e.g. agent-utilities in
        Phase 3) is ALSO swept into the Phase-5 bulk list and gets BUMPED TWICE.
        (CONCEPT:RM-BUMP single-bump-per-project)
        """
        assigned_projects: set[str] = set()
        phase_list: list[dict[str, Any]] = []
        total_projects = 0

        for phase in config.get("phases", []):
            phase_num = phase.get("phase")
            if phase_num < start_phase:
                continue

            projects = self._bump_phase_projects(phase, filter_set, assigned_projects)
            projects = [p for p in projects if p not in assigned_projects]
            assigned_projects.update(projects)
            if not projects:
                continue

            phase_list.append(
                {
                    "phase_num": phase_num,
                    "name": phase.get("name", f"Phase {phase_num}"),
                    "projects": projects,
                }
            )
            total_projects += len(projects)

        return phase_list, total_projects

    @staticmethod
    def _parse_bumped_version(data: str) -> str:
        """Pull the post-bump version out of a ``bump_version`` result payload."""
        match = re.search(r"new_version=(.*)", data)
        if match:
            return match.group(1).strip()
        match = re.search(r"current_version=(.*)", data)
        return match.group(1).strip() if match else "success"

    def _bump_one_project(
        self,
        *,
        project_name: str,
        part: str,
        dry_run: bool,
        force: bool,
        all_results: list[GitResult],
    ) -> str | None:
        """Bump one project's version.

        Returns the new version string, ``"skipped"`` when the project had no
        bump-worthy change, or ``None`` when it was unresolvable or the bump
        failed. A declared project whose local clone is absent (stale registry
        entry / never-cloned repo) must not crash the whole phased bump, so it
        is skipped with a warning and the rest of the topology proceeds.
        """
        project_dir = self._project_path_for(project_name)
        if not project_dir:
            return None

        if not os.path.isdir(project_dir):
            logger.warning(
                "Skipping bump for %s: project directory missing (%s)",
                project_name,
                project_dir,
            )
            return None

        if not force and self._bump_skip_reason(project_dir):
            logger.info("Skipping project version bump")
            return "skipped"

        result = self.bump_version(
            part=part,
            allow_dirty=True,
            path=project_dir,
            dry_run=dry_run,
            force=force,
            verbose=dry_run or not dry_run,
        )
        all_results.append(result)
        if result.status != "success":
            return None
        return self._parse_bumped_version(result.data)

    @staticmethod
    def _dependency_update_result(
        path: str, project_name: str, new_version: str, dep_file_name: str
    ) -> GitResult:
        """The success record for one propagated dependency-pin update."""
        return GitResult(
            status="success",
            data=f"Updated {project_name} to {new_version} in {dep_file_name}",
            metadata=GitMetadata(
                command="update_dependency",
                workspace=_project_label(path),
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _update_dependency_files(
        self, *, path: str, project_name: str, new_version: str, dry_run: bool
    ) -> list[GitResult]:
        """Repin *project_name* in every dependency-declaring file under *path*.

        Not just ``pyproject.toml``: ``requirements.txt`` commonly pins the same
        package (often ``==``) and would otherwise go stale.
        (CONCEPT:RM-BUMP cross-dependency propagation)
        """
        results: list[GitResult] = []
        for dep_file_name in ("pyproject.toml", "requirements.txt"):
            dep_file = Path(path) / dep_file_name
            if not dep_file.exists():
                continue
            if not self.update_dependency(
                str(dep_file), project_name, new_version, dry_run
            ):
                continue
            results.append(
                self._dependency_update_result(
                    path, project_name, new_version, dep_file_name
                )
            )
        return results

    def _propagate_bump_to_dependents(
        self,
        *,
        project_name: str,
        new_version: str,
        phase_num: int,
        phase_of: Callable[[str], int],
        dry_run: bool,
        all_results: list[GitResult],
    ) -> None:
        """Repin the just-bumped project across every same-or-later-phase repo.

        Earlier phases are skipped so a later bump cannot circle back and dirty
        a phase that has already been released.
        """
        for path in self.project_map.values():
            other_project_name = os.path.basename(path)
            other_phase = phase_of(other_project_name)
            if other_phase < phase_num:
                logger.info(
                    f"Skipping dependency update for {project_name} in {other_project_name} "
                    f"to avoid circular updates of earlier phase (Phase {other_phase} < Phase {phase_num})"
                )
                continue
            all_results.extend(
                self._update_dependency_files(
                    path=path,
                    project_name=project_name,
                    new_version=new_version,
                    dry_run=dry_run,
                )
            )

    @staticmethod
    def _run_bump_phase(
        *,
        p_info: dict[str, Any],
        tracker: "_PhaseProgress",
        processed_projects: set[str],
        bump_one: Callable[[str], str | None],
        propagate: Callable[[str, str, int], None],
    ) -> None:
        """Bump every project in one phase, propagating each new version onward."""
        phase_name = p_info["name"]
        phase_num = p_info["phase_num"]
        tracker.begin_phase(phase_name)

        for project_name in p_info["projects"]:
            # Defensive: never bump a project twice in one run (a later phase
            # must not re-bump one an earlier phase already handled).
            if project_name in processed_projects:
                continue

            tracker.begin_item(phase_name, project_name)
            processed_projects.add(project_name)
            logger.info(
                f"Bumping version for project: {project_name} in {phase_name}..."
            )
            new_version = bump_one(project_name)
            tracker.finish_item(
                phase_name, project_name, "success" if new_version else "failed"
            )

            if new_version and re.match(r"^v?\d+\.\d+\.\d+", new_version):
                propagate(project_name, new_version, phase_num)

        tracker.end_phase(phase_name)

    def phased_bumpversion(
        self,
        part: str = "patch",
        start_phase: int = 1,
        dry_run: bool = False,
        allow_pre_commit: bool = False,
        config: dict | None = None,
        single_phase: bool = False,
        project_filter: str | None = None,
        progress: dict | None = None,
        force: bool = False,
        auto_start: bool = True,
    ) -> list[GitResult]:
        """
        Execute the phased bumpversion workflow: pre-commits + phased bumping.

        ``auto_start`` (the default) begins the run at the lowest phase that
        actually contains a repo with pending work (advancing ``start_phase``
        forward, never backward) so unchanged upstream phases are skipped.
        A change in phase *N* still cascades to every phase ``>= N``. It stands
        down — running from the explicit ``start_phase`` — when ``project_filter``
        or ``force`` is set, since those are explicit-targeting requests that
        deliberately bypass change detection. Pass ``auto_start=False`` to opt
        out and always start at ``start_phase``.

        Concept:
            CONCEPT:RM-BUMP
        """
        if progress is None:
            progress = self.progress

        all_results: list[GitResult] = []
        resolved = self._resolve_maintenance_config(config)
        if resolved is None:
            return []
        config = resolved

        project_phases, bulk_phase_num = self._project_phase_index(config)

        def phase_of(proj_name: str) -> int:
            return project_phases.get(proj_name, bulk_phase_num)

        tracker = _PhaseProgress(state=progress, noun="bump")

        if auto_start and not project_filter and not force:
            detected = self._resolve_auto_start_phase(
                config=config,
                start_phase=start_phase,
                tracker=tracker,
                noun="bump",
                lowest_label="lowest changed phase",
            )
            if detected is None:
                return all_results
            start_phase = detected

        if allow_pre_commit:
            all_results.extend(self._run_bump_pre_commit_stage(config))

        def bump_one(project_name: str) -> str | None:
            return self._bump_one_project(
                project_name=project_name,
                part=part,
                dry_run=dry_run,
                force=force,
                all_results=all_results,
            )

        def propagate(project_name: str, new_version: str, phase_num: int) -> None:
            self._propagate_bump_to_dependents(
                project_name=project_name,
                new_version=new_version,
                phase_num=phase_num,
                phase_of=phase_of,
                dry_run=dry_run,
                all_results=all_results,
            )

        filter_set = self._parse_project_filter(project_filter)
        phase_list, tracker.total = self._build_bump_phase_list(
            config=config, start_phase=start_phase, filter_set=filter_set
        )
        tracker.initialize(
            "Initializing Bumps", [(p["name"], p["projects"]) for p in phase_list]
        )

        processed_projects: set[str] = set()
        for p_info in phase_list:
            self._run_bump_phase(
                p_info=p_info,
                tracker=tracker,
                processed_projects=processed_projects,
                bump_one=bump_one,
                propagate=propagate,
            )

        tracker.finish("Bumps Completed")
        return all_results

    maintain_projects = phased_bumpversion

    def worktree_hygiene(
        self,
        prune: bool = False,
        base: str = "main",
        stale_days: int = 14,
    ) -> dict[str, Any]:
        """Audit (and optionally prune) session worktrees as a release-flow step.

        Wraps :meth:`WorktreeManager.audit`. Read-only by default — it returns the
        ``safe_to_prune``/``do_not_disturb`` classification so a release run can
        report what *could* be cleaned without touching anything. With
        ``prune=True`` it removes only ``merged`` worktrees (and ``dangling`` admin
        pointers), never ``active``/``stale`` work or orphaned directories. This is
        the audit-aware cleanup the release pipeline runs instead of a blind reaper.
        (CONCEPT:RM-WORKTREE-AUDIT)
        """
        from repository_manager.worktree import WorktreeManager

        return WorktreeManager(self).audit(
            base=base, stale_days=stale_days, prune_merged=prune
        )

    def _maintenance_config_model(self) -> MaintenanceConfig | None:
        """The ``maintenance`` section of the active workspace config, if any.

        Prefers an already-loaded :attr:`config`; otherwise loads ``workspace.yml``
        (``WORKSPACE_YML`` overrides the name, relative paths resolve against
        :attr:`path`). Returns ``None`` when no maintenance config is reachable.
        """
        if hasattr(self, "config") and self.config and self.config.maintenance:
            return self.config.maintenance

        yml_path = os.environ.get("WORKSPACE_YML") or "workspace.yml"
        if not os.path.isabs(yml_path):
            yml_path = os.path.join(self.path, yml_path)
        if not os.path.exists(yml_path):
            return None
        if self.load_projects_from_yaml(yml_path) and self.config:
            return self.config.maintenance
        return None

    def _resolve_maintenance_config(self, config: dict | None) -> dict | None:
        """Return *config* unchanged, or load the maintenance config in its place.

        ``None`` means no maintenance configuration is reachable and the caller
        must abort; the error is logged here so every phased workflow reports it
        identically.
        """
        if config is not None:
            return config
        config_model = self._maintenance_config_model()
        if config_model is None:
            logger.error("No maintenance configuration found.")
            return None
        return config_model.model_dump()

    def _resolve_auto_start_phase(
        self,
        *,
        config: dict,
        start_phase: int,
        tracker: "_PhaseProgress",
        noun: str,
        lowest_label: str,
    ) -> int | None:
        """Advance *start_phase* to the lowest phase that still has pending work.

        Never moves the start phase backwards. Returns ``None`` when nothing at
        all is pending — in which case *tracker* has already been finalized and
        the caller should return immediately.
        """
        detected = self._auto_start_phase(config)
        if detected is None:
            logger.info(
                f"Phased {noun}: no repository changes detected; nothing to {noun}."
            )
            tracker.nothing_to_do(f"No changes — nothing to {noun}")
            return None
        if detected > start_phase:
            logger.info(
                f"Phased {noun}: {lowest_label} is {detected}; "
                f"starting there (skipping phases {start_phase}–{detected - 1})."
            )
        return max(start_phase, detected)

    def _project_path_for(self, project_name: str) -> str | None:
        """Local clone path of *project_name* from the URL->path project map."""
        for url, p_path in self.project_map.items():
            if url.endswith(f"/{project_name}.git") or url.endswith(f"/{project_name}"):
                return p_path
        return None

    def _bulk_push_targets(self, processed_projects: set[str]) -> list[tuple[str, str]]:
        """Every mapped project not already claimed by an earlier phase."""
        targets: list[tuple[str, str]] = []
        for url, path in self.project_map.items():
            name = url.split("/")[-1].replace(".git", "")
            if name in processed_projects:
                continue
            targets.append((name, path))
        return targets

    def _named_push_targets(
        self, projects: list[str], processed_projects: set[str]
    ) -> list[tuple[str, str]]:
        """Resolve an explicit ``projects:`` list to (name, path) pairs.

        Claims every named project in *processed_projects* (even one with no
        local clone) so a later bulk phase cannot re-push it.
        """
        targets: list[tuple[str, str]] = []
        for project_name in projects:
            processed_projects.add(project_name)
            p_path = self._project_path_for(project_name)
            if p_path is not None:
                targets.append((project_name, p_path))
        return targets

    def _phase_push_targets(
        self, phase: dict, project_filter: str | None, processed_projects: set[str]
    ) -> list[tuple[str, str]]:
        """The (name, path) pairs one configured phase would push."""
        projects = phase.get("projects", [])[:]
        if phase.get("project"):
            projects.append(phase.get("project"))

        if project_filter:
            projects = [p for p in projects if p == project_filter]
            if (
                not projects
                and phase.get("bulk_push")
                and project_filter not in processed_projects
            ):
                projects = [project_filter]

        if phase.get("bulk_push") and not project_filter:
            return self._bulk_push_targets(processed_projects)
        return self._named_push_targets(projects, processed_projects)

    @staticmethod
    def _apply_phase_excludes(
        phase: dict, phase_num: Any, targets: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Drop targets matching the phase's declarative ``exclude`` patterns.

        fnmatch patterns are checked against the project name for every phase
        (not only ``bulk_push`` ones) so an operator can carve a specific repo
        out of an explicit ``projects:`` list too.
        """
        exclude_patterns = phase.get("exclude") or []
        if not exclude_patterns:
            return targets
        kept: list[tuple[str, str]] = []
        for name, path in targets:
            if any(fnmatch.fnmatch(name, pat) for pat in exclude_patterns):
                logger.info(
                    "Phase %s: excluding %s (matches an 'exclude' pattern)",
                    phase_num,
                    name,
                )
                continue
            kept.append((name, path))
        return kept

    @staticmethod
    def _drop_missing_clones(
        targets: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Drop declared projects whose local clone is absent.

        A stale registry entry / never-cloned repo must not surface as a false
        push failure -- mirrors the same guard in the phased bump.
        """
        kept: list[tuple[str, str]] = []
        for name, path in targets:
            if not os.path.isdir(path):
                logger.warning(
                    "Skipping push for %s: project directory missing (%s)", name, path
                )
                continue
            kept.append((name, path))
        return kept

    def _build_push_phase_list(
        self, *, config: dict, start_phase: int, project_filter: str | None
    ) -> tuple[list[dict[str, Any]], int]:
        """Expand the configured phases into the concrete push plan.

        Returns the ordered phase records and the total number of projects
        across them (used for the overall progress percentage).
        """
        processed_projects: set[str] = set()
        phase_list: list[dict[str, Any]] = []
        total_projects = 0

        for phase in config.get("phases", []):
            phase_num = phase.get("phase")
            if phase_num < start_phase:
                continue

            targets = self._phase_push_targets(
                phase, project_filter, processed_projects
            )
            targets = self._apply_phase_excludes(phase, phase_num, targets)
            targets = self._drop_missing_clones(targets)
            if not targets:
                continue

            phase_list.append(
                {
                    "phase_num": phase_num,
                    "name": phase.get("name", f"Phase {phase_num}"),
                    "projects_to_push": targets,
                    "wait_minutes": float(phase.get("wait_minutes", 0)),
                }
            )
            total_projects += len(targets)

        return phase_list, total_projects

    @staticmethod
    def _collect_push_result(
        future: "concurrent.futures.Future", all_results: list[GitResult]
    ) -> tuple[str, bool]:
        """Append one push future's outcome; return (status, pushed-anything)."""
        try:
            res = future.result()
        except Exception as e:
            all_results.append(
                GitResult(
                    status="error",
                    data="",
                    error=GitError(message=type(e).__name__, code=1),
                )
            )
            return "failed", False

        all_results.append(res)
        if res.status != "success":
            return "failed", False
        return "success", "Everything up-to-date" not in res.data

    def _execute_push_phase(
        self,
        *,
        p_info: dict[str, Any],
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Push one phase's projects in parallel; return whether anything landed."""
        phase_name = p_info["name"]
        projects_to_push = p_info["projects_to_push"]

        tracker.begin_phase(phase_name)
        logger.info(
            f"Starting {phase_name} push for {len(projects_to_push)} projects..."
        )

        phase_had_pushes = False
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            future_to_proj = {}
            for proj_name, p_path in projects_to_push:
                tracker.begin_item(phase_name, proj_name)
                future = executor.submit(self.push_project, path=p_path)
                future_to_proj[future] = proj_name

            for future in concurrent.futures.as_completed(future_to_proj):
                proj_name = future_to_proj[future]
                status_str, pushed = self._collect_push_result(future, all_results)
                phase_had_pushes = phase_had_pushes or pushed
                tracker.finish_item(phase_name, proj_name, status_str)

        tracker.end_phase(phase_name)
        return phase_had_pushes

    @staticmethod
    def _report_barrier_timeout(
        *,
        outcome: Any,
        p_info: dict[str, Any],
        next_phase_name: str,
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> None:
        """Log, record, and surface a timed-out downstream gate-readiness barrier."""
        phase_name = p_info["name"]
        unresolved = "; ".join(f"{f.repo_name} ({f.detail})" for f in outcome.failures)
        logger.error(
            "Phase %s gate-readiness barrier TIMED OUT after %.1fs "
            "(%d attempt(s)) with %d downstream repo(s) still failing "
            "their pre-push gate -- ABORTING the wave before %s (or "
            "any later phase) starts: %s. Set %s=<reason> to override "
            "(loud + audit-logged), or re-run once the failing repo(s) "
            "pass their own gate.",
            p_info["phase_num"],
            outcome.waited_s,
            outcome.attempts,
            len(outcome.failures),
            next_phase_name,
            unresolved,
            dependency_readiness.OVERRIDE_ENV_VAR,
        )
        tracker.note(f"ABORTED — downstream gate(s) unmet after {phase_name}")
        all_results.append(
            GitResult(
                status="error",
                data="",
                error=GitError(
                    message=(
                        f"phased_push aborted after {phase_name}: "
                        f"downstream gate-readiness barrier timed out "
                        f"after {outcome.waited_s:.1f}s "
                        f"({outcome.attempts} attempt(s)) still "
                        f"failing: {unresolved}"
                    ),
                    code=1,
                ),
            )
        )

    def _settle_phase_barrier(
        self,
        *,
        p_info: dict[str, Any],
        phase_had_pushes: bool,
        later_phases: list[dict[str, Any]],
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Hold the wave until downstream repos pass their own pre-push gate.

        Returns ``False`` when the barrier timed out and ``phased_push`` must
        abort before any later phase starts.
        """
        wait_minutes = p_info["wait_minutes"]
        if wait_minutes <= 0:
            return True

        phase_name = p_info["name"]
        phase_num = p_info["phase_num"]
        if not phase_had_pushes:
            logger.info(
                f"Phase {phase_num} complete. Skipping the {wait_minutes}-minute "
                "gate-readiness ceiling because 0 commits were pushed."
            )
            return True

        tracker.note(
            f"Running downstream pre-push gates after {phase_name} "
            f"(retry ceiling {wait_minutes} min)"
        )
        outcome = self._await_phase_dependency_readiness(
            phase_num=phase_num,
            phase_name=phase_name,
            projects_to_push=p_info["projects_to_push"],
            later_phases=later_phases,
            wait_minutes=wait_minutes,
        )
        if not outcome.ok:
            self._report_barrier_timeout(
                outcome=outcome,
                p_info=p_info,
                next_phase_name=(
                    later_phases[0]["name"] if later_phases else "the next phase"
                ),
                tracker=tracker,
                all_results=all_results,
            )
            return False

        if outcome.targets_checked:
            logger.info(
                "Phase %s gate-readiness barrier satisfied after %.1fs "
                "(%d attempt(s)) for %d downstream repo(s)%s — proceeding "
                "immediately (retry ceiling was %s min).",
                phase_num,
                outcome.waited_s,
                outcome.attempts,
                len(outcome.targets_checked),
                " (override used)" if outcome.overridden else "",
                wait_minutes,
            )
        return True

    def phased_push(
        self,
        start_phase: int = 1,
        config: dict | None = None,
        single_phase: bool = False,
        project_filter: str | None = None,
        progress: dict | None = None,
        auto_start: bool = True,
    ) -> list[GitResult]:
        """
        Execute the phased git push workflow.

        ``auto_start`` (the default) begins the push at the lowest phase that has
        a repo with unpushed work (advancing ``start_phase`` forward, never
        backward), skipping the inter-phase waits of unchanged upstream phases.
        It stands down — pushing from the explicit ``start_phase`` — when
        ``project_filter`` is set, since that is an explicit-targeting request.
        Pass ``auto_start=False`` to opt out and always start at ``start_phase``.

        Concept:
            CONCEPT:RM-PUSH
        """
        if progress is None:
            progress = self.progress

        all_results: list[GitResult] = []
        resolved = self._resolve_maintenance_config(config)
        if resolved is None:
            return []
        config = resolved

        tracker = _PhaseProgress(state=progress, noun="push")

        if auto_start and not project_filter:
            detected = self._resolve_auto_start_phase(
                config=config,
                start_phase=start_phase,
                tracker=tracker,
                noun="push",
                lowest_label="lowest unpushed phase",
            )
            if detected is None:
                return all_results
            start_phase = detected

        phase_list, tracker.total = self._build_push_phase_list(
            config=config, start_phase=start_phase, project_filter=project_filter
        )
        tracker.initialize(
            "Initializing Pushes",
            [(p["name"], [n for n, _ in p["projects_to_push"]]) for p in phase_list],
        )

        for phase_idx, p_info in enumerate(phase_list):
            phase_had_pushes = self._execute_push_phase(
                p_info=p_info, tracker=tracker, all_results=all_results
            )
            if not self._settle_phase_barrier(
                p_info=p_info,
                phase_had_pushes=phase_had_pushes,
                later_phases=phase_list[phase_idx + 1 :],
                tracker=tracker,
                all_results=all_results,
            ):
                return all_results

        tracker.finish("Pushes Completed")
        return all_results

    def _phase_published_packages(
        self, projects_to_push: list[tuple[str, str]]
    ) -> dict[str, str]:
        """Canonicalized package name -> declaring ``pyproject.toml`` path, for
        every project actually pushed in one ``phased_push`` phase.

        Reads each pushed project's OWN ``[project].name`` rather than
        assuming the git repo name equals the published package name — a
        assumption that would be exactly the kind of hardcoded au/eg-shaped
        guess this gate is meant to avoid. A project with no
        ``pyproject.toml`` (or no ``[project].name``) publishes nothing this
        barrier can reason about and is silently skipped, not an error.
        """
        import tomllib

        from packaging.utils import canonicalize_name

        published: dict[str, str] = {}
        for _proj_name, p_path in projects_to_push:
            pyproject_path = os.path.join(p_path, "pyproject.toml")
            if not os.path.isfile(pyproject_path):
                continue
            try:
                with open(pyproject_path, "rb") as handle:
                    data = tomllib.load(handle)
            except (OSError, tomllib.TOMLDecodeError):
                continue
            name = (data.get("project") or {}).get("name")
            if name:
                published[canonicalize_name(name)] = pyproject_path
        return published

    def _await_phase_dependency_readiness(
        self,
        *,
        phase_num: int,
        phase_name: str,
        projects_to_push: list[tuple[str, str]],
        later_phases: list[dict[str, Any]],
        wait_minutes: float,
        poll_interval_s: float = 30.0,
    ) -> "dependency_readiness.GateReadinessOutcome":
        """Layer 2 of CONCEPT:RM-DEP-READY — gate-driven phase transitions.

        The owner's refinement over the original blind
        ``time.sleep(wait_minutes * 60)`` (slow when a publish took 4 minutes,
        silently wrong when it never landed) and over the poll-the-index
        barrier that briefly replaced it (a second implementation of exactly
        what the pre-push gate already checks): **a phase transition is
        decided by RUNNING the next phase's repos' own pre-push gates.**
        Those gates already include the ``dependency-readiness`` hook
        (Layer 1, ``[manual, pre-push]``), which fails closed when a declared
        intra-fleet constraint is unsatisfiable — that hook IS the oracle, so
        this method's only job is retry/backoff/deadline orchestration around
        calling it (:func:`repository_manager.dependency_readiness.await_gate_readiness`,
        which in turn calls :func:`repository_manager.gates.run_gate_stage` —
        the SAME function ``Git._gate_before_push`` calls before that repo's
        own real push). One mechanism decides both "is this phase transition
        ready" and "will this repo's own push succeed".

        Determines which package(s) THIS phase just published (each pushed
        project's own declared name, via :meth:`_phase_published_packages`),
        then narrows to the later-phase repos that actually declare a
        constraint on one of those packages (never every later-phase repo —
        a repo with no stake in what was just published has nothing to gate
        on), and gate-checks exactly those. Returns immediately (``waited_s``
        near zero) when nothing published or nothing downstream cares — the
        old blind-sleep code always waited the full budget regardless, even
        when nothing needed it.

        ``wait_minutes`` is preserved as exactly the retry-ceiling budget it
        always was (a per-phase ``workspace.yml`` field an operator already
        tunes in minutes) — now enforced as the deadline for the gate-check
        retry loop instead of a sleep duration or an index-poll deadline, so
        existing manifests keep working unmodified with the same meaning an
        operator would expect ("how long am I willing to wait for the next
        phase to become pushable").
        """
        published = self._phase_published_packages(projects_to_push)
        if not published:
            logger.info(
                "Phase %s: no pushed project declares a [project].name — "
                "nothing for the gate-readiness barrier to check.",
                phase_num,
            )
            return dependency_readiness.GateReadinessOutcome(ok=True, waited_s=0.0)

        targets: dict[str, tuple[str, str]] = {}
        for later in later_phases:
            for proj_name, p_path in later["projects_to_push"]:
                if p_path in targets:
                    continue
                constraints = dependency_readiness.declared_fleet_constraints(
                    p_path, fleet_packages=set(published)
                )
                if constraints:
                    targets[p_path] = (proj_name, p_path)

        if not targets:
            logger.info(
                "Phase %s published %s; no later-phase repo declares a "
                "constraint on it — proceeding immediately.",
                phase_num,
                sorted(published),
            )
            return dependency_readiness.GateReadinessOutcome(ok=True, waited_s=0.0)

        logger.info(
            "Phase %s published %s; running the pre-push gate for %d downstream "
            "repo(s) (%s), retrying every %.0fs up to a %.0f-minute ceiling, "
            "abort-and-never-silently-advance if still failing.",
            phase_num,
            sorted(published),
            len(targets),
            ", ".join(name for name, _ in targets.values()),
            poll_interval_s,
            wait_minutes,
        )
        return dependency_readiness.await_gate_readiness(
            list(targets.values()),
            wait_minutes=wait_minutes,
            poll_interval_s=poll_interval_s,
            audit_repo_path=self.path,
        )

    def load_projects_from_yaml(self, yaml_path: str) -> bool:
        """
        Loads repository URLs from a YAML workspace file using Pydantic models.
        Strictly determines self.path relative to the configuration file.
        """
        abs_yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
        yaml_dir = os.path.dirname(abs_yaml_path)

        if not os.path.exists(abs_yaml_path):
            logger.error("Workspace configuration file was not found")
            return False

        try:
            with open(abs_yaml_path) as f:
                data = yaml.safe_load(f)

            if not data:
                return False

            self.config = WorkspaceConfig(**data)

            yaml_config_path = os.path.expanduser(
                _expand_required_environment(
                    self.config.path,
                    label="workspace root",
                )
            )
            is_default_yaml = yaml_path == DEFAULT_WORKSPACE_YML

            if self._explicit_path:
                logger.info("Preserving the explicitly configured workspace root")
            elif os.path.isabs(yaml_config_path):
                self.path = os.path.abspath(yaml_config_path)
            elif is_default_yaml:
                self.path = os.path.abspath(
                    os.path.expanduser(DEFAULT_REPOSITORY_MANAGER_WORKSPACE)
                )
                logger.info("Using the packaged workspace configuration")
            else:
                self.path = os.path.abspath(os.path.join(yaml_dir, yaml_config_path))

            logger.info("Workspace root resolved")

            self.project_map = self._parse_subdirectories(
                self.config.subdirectories, self.path
            )

            for repo in self.config.repositories:
                repo_url = _expand_required_environment(
                    repo.url,
                    label="repository origin",
                )
                repo_name = repo_url.split("/")[-1].replace(".git", "")
                self.project_map[repo_url] = os.path.join(self.path, repo_name)
            return True

        except Exception as e:
            # Log the message, not just the type. Logging `error_type` alone made a
            # real failure invisible: a phased push resolved 0 repos and exited 0
            # ("nothing to push") because this loader had failed with a bare
            # `ValueError`, and the cause -- an unexpanded `${...}` placeholder in
            # the manifest -- was discarded here. A summary that says "Total: 0"
            # reads as success, so the swallowed cause is what makes it dangerous.
            logger.error(
                "Failed to load projects from YAML %s: %s: %s",
                yaml_path,
                type(e).__name__,
                e,
            )
            return False

    def discover_projects(self) -> dict[str, str]:
        """
        Scan self.path for immediate subdirectories containing a .git folder.
        Populates and returns self.project_map.
        """
        self.project_map = {}
        expanded_path = os.path.abspath(os.path.expanduser(self.path))
        if not os.path.exists(expanded_path):
            return self.project_map

        try:
            for item in os.listdir(expanded_path):
                full_path = os.path.join(expanded_path, item)
                if os.path.isdir(full_path) and os.path.exists(
                    os.path.join(full_path, ".git")
                ):
                    # Get remote URL
                    remote_url = None
                    try:
                        import shutil

                        git_path = shutil.which("git") or "git"
                        res = subprocess.run(
                            [git_path, "config", "--get", "remote.origin.url"],
                            cwd=full_path,
                            capture_output=True,
                            text=True,
                            check=False,
                        )
                        if res.returncode == 0 and res.stdout.strip():
                            remote_url = res.stdout.strip()
                    except Exception as exc:
                        logger.debug(
                            "Failed to get a remote URL: error_type=%s",
                            type(exc).__name__,
                        )

                    if not remote_url:
                        remote_url = f"local://{item}"

                    self.project_map[remote_url] = os.path.abspath(full_path)

            logger.info(
                f"Auto-discovered {len(self.project_map)} git repositories in {expanded_path}"
            )
        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)

        return self.project_map

    def _parse_subdirectories(
        self, subdirs: dict[str, SubdirectoryConfig], current_path: str
    ) -> dict[str, str]:
        """Helper to recursively parse subdirectories and collect repository paths."""
        project_map = {}
        for name, data in subdirs.items():
            new_path = os.path.join(current_path, name)

            for repo in data.repositories:
                repo_url = _expand_required_environment(
                    repo.url,
                    label="repository origin",
                )
                repo_name = repo_url.split("/")[-1].replace(".git", "")
                project_map[repo_url] = os.path.join(new_path, repo_name)

            if data.subdirectories:
                project_map.update(
                    self._parse_subdirectories(data.subdirectories, new_path)
                )

        return project_map

    def generate_workspace_template(
        self, target_path: str, use_default: bool = True
    ) -> GitResult:
        """
        Generates a workspace.yml template at the specified path.
        """
        try:
            target_path = os.path.abspath(os.path.expanduser(target_path))
            if os.path.isdir(target_path):
                target_path = os.path.join(target_path, "workspace.yml")

            os.makedirs(os.path.dirname(target_path), exist_ok=True)

            template_content = ""
            if use_default:
                try:
                    from importlib.resources import files

                    template_content = (
                        files("repository_manager") / "workspace.yml"
                    ).read_text()
                except Exception:  # nosec B110
                    template_content = "name: My Workspace\npath: .\ndescription: New workspace\nsubdirectories: {}\n"
            else:
                template_content = "name: My Workspace\npath: .\ndescription: New workspace\nsubdirectories:\n  agents:\n    description: Agent repositories\n    repositories: []\n"

            with open(target_path, "w") as f:
                f.write(template_content)

            return GitResult(
                status="success",
                data="Workspace template generated",
                metadata=GitMetadata(
                    command="generate_template",
                    workspace=_project_label(target_path),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )
        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)
            return GitResult(
                status="error",
                data="",
                error=GitError(message="Repository operation failed", code=1),
            )

    def save_workspace_config(
        self, yaml_path: str, config: WorkspaceConfig | None = None
    ) -> GitResult:
        """
        Saves the current or provided WorkspaceConfig to a YAML file.
        """
        try:
            cfg = config or self.config
            if not cfg:
                return GitResult(
                    status="error",
                    data="",
                    error=GitError(message="No configuration to save", code=1),
                )

            yaml_path = os.path.abspath(os.path.expanduser(yaml_path))
            os.makedirs(os.path.dirname(yaml_path), exist_ok=True)

            data = cfg.model_dump()
            with open(yaml_path, "w") as f:
                yaml.dump(data, f, sort_keys=False)

            return GitResult(
                status="success",
                data="Workspace manifest saved",
                metadata=GitMetadata(
                    command="save_workspace",
                    workspace=_project_label(yaml_path),
                    return_code=0,
                    timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
                ),
            )
        except Exception as e:
            logger.error(
                "Failed to save workspace configuration: error_type=%s",
                type(e).__name__,
            )
            return GitResult(
                status="error",
                data="",
                error=GitError(message="Failed to save workspace", code=1),
            )

    def get_consolidated_skill_paths(self) -> list[str]:
        """
        Returns absolute paths to the 15 specific building and documentation skills.
        """
        required_universal = [
            "agent-package-builder",
            "mcp-builder",
            "agent-builder",
            "skill-builder",
            "skill-graph-builder",
            "api-wrapper-builder",
            "web-search",
            "web-crawler",
        ]
        required_graphs = [
            "docker-docs",
            "fastapi-docs",
            "fastmcp-docs",
            "nodejs-docs",
            "vercel-docs",
            "python-docs",
            "pydantic-ai-docs",
        ]

        paths = []

        if get_universal_skills_path:
            try:
                from importlib.resources import files

                base = files("universal_skills") / "skills"
                for skill in required_universal:
                    skill_path = base / skill
                    if skill_path.joinpath("SKILL.md").is_file():
                        paths.append(str(skill_path))
            except Exception as e:
                logger.warning("Operation failed: error_type=%s", type(e).__name__)

                all_universal = get_universal_skills_path()
                paths.extend(
                    [
                        p
                        for p in all_universal
                        if os.path.basename(p) in required_universal
                    ]
                )

        if get_skill_graphs_path:
            try:
                from importlib.resources import files

                base = files("skill_graphs") / "skill_graphs"
                for graph in required_graphs:
                    graph_path = base / graph
                    if graph_path.joinpath("SKILL.md").is_file():
                        paths.append(str(graph_path))
            except Exception as e:
                logger.warning("Operation failed: error_type=%s", type(e).__name__)

                all_graphs = get_skill_graphs_path(default_enabled=True)
                paths.extend(
                    [p for p in all_graphs if os.path.basename(p) in required_graphs]
                )

        return list(set(paths))


from repository_manager.cli_commands import (
    run as _run_cli,
)
from repository_manager.cli_commands import (
    run_build_queue_cli as _run_build_queue_cli,
)
from repository_manager.cli_commands import (
    run_lane_cli as _run_lane_cli,
)
from repository_manager.cli_commands import (
    run_merge_queue_cli as _run_merge_queue_cli,
)
from repository_manager.cli_commands.context import runtime_from_module


def main() -> None:
    """Run the Repository Manager command-line adapter."""
    _run_cli(runtime_from_module())


if __name__ == "__main__":
    main()
