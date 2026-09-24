#!/usr/bin/env python


"""
A command-line tool for managing Git repositories, supporting cloning and pulling
multiple repositories in parallel using Python's multiprocessing capabilities.
"""

import contextlib
import copy
import dataclasses
import datetime
import fnmatch
import functools
import hashlib
import inspect
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import tomllib
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path, PureWindowsPath
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlsplit, urlunsplit

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
from agent_connector_sdk.utilities import to_boolean
from agent_utilities.base_utilities import (
    get_library_file_path,  # SDK-GAPS.md: no SDK equivalent
)
from pydantic import ValidationError

try:
    from skill_graphs.skill_graph_utilities import get_skill_graphs_path
    from universal_skills.skill_utilities import get_universal_skills_path
except ImportError:
    get_universal_skills_path = None
    get_skill_graphs_path = None

from importlib.resources import files

from agent_connector_sdk.utilities import get_logger

from repository_manager import dependency_readiness
from repository_manager.canonical_guard import guarded_canonical_mutation
from repository_manager.gates import (
    HOOK_STAGE_BY_GATE_STAGE,
    precommit_gate_environment,
    run_gate_stage,
)
from repository_manager.models import (
    GitError,
    GitMetadata,
    GitResult,
    MaintenanceConfig,
    ReadmeResult,
    RepositoryConfig,
    SubdirectoryConfig,
    WorkspaceConfig,
)
from repository_manager.operation_boundary import (
    OperationBoundaryError,
    PinnedDirectory,
    cleanup_pinned_directory,
    open_directory,
    path_exists,
    pin_creation,
    pin_existing,
    pin_existing_under,
    read_at,
    read_release_plan_receipt,
    receipt_result_payload,
    snapshot_pinned_checkout,
    snapshot_workspace,
    write_at,
    write_release_plan_receipt,
)
from repository_manager.release_validation import (
    canonical_repository_url,
    read_release_document,
    release_repository_name,
    repository_name,
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
_ReleaseTarget = tuple[str, str]


@dataclasses.dataclass(frozen=True)
class _SealedPushRefs:
    """Exact local refs admitted to one atomic remote publication."""

    branch: str
    head_oid: str
    tag: str | None = None
    tag_oid: str | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """Return the exact refs exported into the sealed admin repository."""
        return (self.branch,) if self.tag is None else (self.branch, self.tag)

    @property
    def expected(self) -> dict[str, str]:
        """Return expected remote object identities by full ref name."""
        values = {self.branch: self.head_oid}
        if self.tag is not None and self.tag_oid is not None:
            values[self.tag] = self.tag_oid
        return values


_CONSOLIDATED_UNIVERSAL_SKILLS = (
    "agent-package-builder",
    "mcp-builder",
    "agent-builder",
    "skill-builder",
    "skill-graph-builder",
    "api-wrapper-builder",
    "web-search",
    "web-crawler",
)

_CONSOLIDATED_SKILL_GRAPHS = (
    "docker-docs",
    "fastapi-docs",
    "fastmcp-docs",
    "nodejs-docs",
    "vercel-docs",
    "python-docs",
    "pydantic-ai-docs",
)

_UV_WORKSPACE_SIBLINGS_DIRNAME = ".uv-workspace-siblings"
_PEP503_NAME = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?\Z")
_PEP503_NAME_SEPARATORS = re.compile(r"[-_.]+")


class _UnsafeMutationTarget(ValueError):
    """Carry a typed refusal from target validation to the mutation wrapper."""

    def __init__(self, operation: str, candidate: object, cause: ValueError) -> None:
        super().__init__(str(cause))
        self.operation = operation
        self.candidate = candidate


class _ReleasePlanDrift(ValueError):
    """Signal that a frozen legacy release plan no longer matches its inputs."""

    def __init__(self, operation: str, expected_digest: str) -> None:
        super().__init__(operation)
        self.operation = operation
        self.expected_digest = expected_digest


@dataclasses.dataclass(frozen=True)
class _ReleasePlanProvenance:
    """Immutable provenance for one legacy phased bump/push invocation.

    The older phased APIs predate the typed workspace release-plan contracts.
    They still need the same safety property: the exact input registry and
    ordered ``(phase, name, path)`` membership used for planning must be the
    input used at every mutation boundary.  The payload itself stays local;
    only bounded SHA-256 digests are exposed to progress/results.
    """

    operation: Literal["bump", "push"]
    input_digest: str
    plan_digest: str
    root_identity: dict[str, int | None] | None = None
    scope_identity: dict[str, Any] | None = None
    registry_digest: str | None = None


_CASE_INSENSITIVE_ORIGIN_HOSTS = frozenset({"github.com"})


def _reject_lexical_parent(path: Path, *, label: str) -> Path:
    """Reject traversal syntax before any path normalization can erase it."""
    if ".." in path.parts:
        raise ValueError(f"{label} contains a lexical parent segment")
    return path


def _release_plan_digest(payload: object) -> str:
    """Return a deterministic digest for a bounded release-plan payload."""
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _release_phase_payload(phase_list: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Serialize only the ordered, exact release targets in a phase list."""
    return [
        {
            "phase_num": phase["phase_num"],
            "name": phase["name"],
            "targets": [
                [name, path]
                for name, path in phase.get(
                    "targets", phase.get("projects_to_push", [])
                )
            ],
            **(
                {"wait_minutes": phase["wait_minutes"]}
                if "wait_minutes" in phase
                else {}
            ),
        }
        for phase in phase_list
    ]


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


def _mutation_lock_path(manager: Any, bound: inspect.BoundArguments) -> str:
    """Resolve the lock key without normalizing a mutation target for use."""
    if "target_path" in bound.arguments:
        # clone_repository's contract names this as the destination path,
        # already relative to the caller's cwd when it is not absolute. Do
        # not feed a workspace-prefixed target through ``_resolve_path`` a
        # second time (``workspace/workspace/repo``).
        return os.path.abspath(os.path.expanduser(str(bound.arguments["target_path"])))
    return _operation_lock_path(manager, bound.arguments.get("path"))


def _operation_lock_path(manager: Any, path: object | None) -> str:
    """Keep invalid lexical inputs out of path normalization for lock keys."""
    try:
        return manager._resolve_path(path)
    except ValueError:
        # The mutation method performs the typed refusal.  This fallback only
        # supplies a harmless lock key and never becomes an operation target.
        return str(path or manager.path)


def _call_exclusive_mutation(
    manager: Any,
    method: Callable[..., _MutationResult],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    target_path: str,
) -> _MutationResult:
    """Run one mutation under its lock and type unsafe-target refusals."""
    try:
        with _hold_repo_mutation(target_path):
            return method(*args, **kwargs)
    except _UnsafeMutationTarget as exc:
        return manager._path_validation_result(exc.operation, exc.candidate, exc)


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
        target_path = _mutation_lock_path(manager, bound)
        return _call_exclusive_mutation(manager, method, args, kwargs, target_path)

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


@dataclasses.dataclass
class _CommandOutputCapture:
    """Bounded capture of one repository command's interleaved stdout/stderr.

    Output past ``_MAX_CAPTURED_OUTPUT_BYTES`` is dropped rather than held in
    memory, and the fact that it was dropped is reported in the text.
    """

    lines: list[str] = dataclasses.field(default_factory=list)
    byte_count: int = 0
    truncated: bool = False

    def add(self, line: str) -> None:
        """Append one output line, clipping at the capture ceiling."""
        encoded = line.encode("utf-8", "replace")
        remaining = _MAX_CAPTURED_OUTPUT_BYTES - self.byte_count
        if remaining > 0:
            clipped = encoded[:remaining].decode("utf-8", "ignore")
            self.lines.append(clipped)
            self.byte_count += len(clipped.encode("utf-8"))
        if len(encoded) > remaining:
            self.truncated = True

    def text(self) -> str:
        """The captured output, with a marker appended when it was clipped."""
        if self.truncated:
            return "".join([*self.lines, "\n[repository output truncated]\n"])
        return "".join(self.lines)


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

    _active_cleanup_handle: PinnedDirectory | None = None

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
        # Establish the workspace root through the same descriptor-relative,
        # no-follow boundary used by every later mutation.  The old
        # ``exists``/``makedirs`` pair accepted a symlink (and could follow a
        # parent swapped between those two calls), making the root itself an
        # unchecked authority before manifest loading or setup began.
        with open_directory(
            Path(os.path.abspath(os.path.expanduser(os.fspath(self.path)))),
            create=True,
        ) as workspace_root:
            workspace_root.assert_root_identity()

        self.project_map: dict[str, str] = {}
        # Manifest category path for each configured origin. Bulk release
        # selection consults this registry instead of inferring eligibility from
        # a clone's filesystem path. An absent entry therefore fails closed.
        self._project_categories: dict[str, tuple[str, ...]] = {}
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

    def _sync_workspace_repositories(self) -> list[GitResult]:
        """Create the workspace tree, then clone or pull every declared repo."""
        logger.info("Creating configured workspace structure")
        try:
            root_path = self._workspace_root_candidate()
            with open_directory(root_path, create=True) as root:
                root.assert_root_identity()
                sync_targets = self._validated_workspace_sync_targets()
                if isinstance(sync_targets, GitResult):
                    return [sync_targets]
                results = self._sync_workspace_targets(sync_targets, root=root)
                root.assert_root_identity()
                return results
        except (OperationBoundaryError, ValueError) as exc:
            logger.error("Workspace sync root refused: %s", exc)
            return [
                self._path_validation_result(
                    "workspace_sync", getattr(self, "path", ""), exc
                )
            ]

    def _validated_workspace_sync_targets(
        self,
    ) -> list[tuple[str, str]] | GitResult:
        """Validate every setup destination before any parent is created."""
        # The manifest is an input to a mutating operation. Revalidate every
        # exact destination after the workspace root exists and before creating
        # even a parent directory; a changed map or a newly-created symlink must
        # never redirect setup outside the approved root.
        sync_targets: list[tuple[str, str]] = []
        for url, project_path in self.project_map.items():
            try:
                validated_path = self._validate_workspace_path(
                    project_path,
                    label=f"workspace sync project {_project_label(project_path)!r}",
                )
            except ValueError as exc:
                logger.error("Workspace sync target refused: %s", exc)
                return self._path_validation_result("workspace_sync", project_path, exc)
            sync_targets.append((url, str(validated_path)))
        return sync_targets

    def _sync_workspace_targets(
        self,
        sync_targets: list[tuple[str, str]],
        *,
        root: PinnedDirectory | None = None,
    ) -> list[GitResult]:
        """Clone or pull one already-validated workspace target list."""
        logger.info("Syncing repositories (Clone/Pull)...")
        return [
            self._sync_workspace_target(url, project_path, root)
            for url, project_path in sync_targets
        ]

    def _sync_workspace_target(
        self, url: str, project_path: str, root: PinnedDirectory | None
    ) -> GitResult:
        """Synchronize one target and convert boundary refusal to a result."""
        try:
            if self._workspace_target_exists(project_path, root):
                return self.pull_project(project_path, _root=root)
            return self.clone_repository(url, project_path, _root=root)
        except OperationBoundaryError as exc:
            return self._path_validation_result("workspace_sync", project_path, exc)

    @staticmethod
    def _workspace_target_exists(
        project_path: str, root: PinnedDirectory | None
    ) -> bool:
        """Check a workspace target through the active root boundary."""
        if root is not None:
            return path_exists(root, project_path)
        return os.path.exists(project_path)

    def _setup_metadata(self, failed: bool) -> GitMetadata:
        """Metadata for a ``setup_workspace`` result."""
        return GitMetadata(
            command="setup_workspace",
            workspace=_project_label(self.path),
            return_code=1 if failed else 0,
            timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
        )

    def _clone_only_setup_result(self, failed_clones: list[GitResult]) -> GitResult:
        """The workspace-setup result when no install step was requested."""
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
            metadata=self._setup_metadata(bool(failed_clones)),
        )

    def _install_setup_result(
        self, results: list[GitResult], failed_clones: list[GitResult]
    ) -> GitResult:
        """Install every synced project and fold the outcome into one result."""
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
            metadata=self._setup_metadata(bool(failures)),
        )

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

        results = self._sync_workspace_repositories()
        failed_clones = [r for r in results if r.status != "success"]

        if not install:
            return self._clone_only_setup_result(failed_clones)
        return self._install_setup_result(results, failed_clones)

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

        _reject_lexical_parent(
            Path(os.path.expanduser(os.fspath(path))), label="operation path"
        )

        if os.path.isabs(path):
            return os.path.abspath(path)

        return os.path.abspath(os.path.join(self.path, path))

    def _current_release_tag(
        self,
        path: str | None = None,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> str | None:
        """Return ``v<current_version>`` from the repo's .bumpversion.cfg, if any.

        The tag the most recent bump created for this repo — pushed explicitly so
        lightweight tags reach the remote without dragging along stale historical
        tags. Returns None when there's no bumpversion config or it exists only
        locally (never created).
        """
        target_dir = self._resolve_path(path)
        try:
            cfg_lines = self._release_tag_config_lines(target_dir, pinned)
            for line in cfg_lines:
                if line.strip().startswith("current_version"):
                    ver = line.split("=", 1)[1].strip()
                    if ver:
                        tag = f"v{ver}"
                        # Only if the tag actually exists locally.
                        chk = self.git_action(
                            command=f"git tag -l {tag}",
                            path=target_dir,
                            quiet=True,
                            **self._pinned_git_kwargs(pinned),
                        )
                        if chk.status == "success" and tag in (chk.data or ""):
                            return tag
                    return None
        except OperationBoundaryError:
            raise
        except Exception as exc:  # noqa: BLE001
            logger.debug("Operation failed: error_type=%s", type(exc).__name__)
        return None

    @staticmethod
    def _release_tag_config_lines(
        target_dir: str, pinned: PinnedDirectory | None
    ) -> list[str]:
        """Read bumpversion configuration through the active operation boundary."""
        if pinned is not None:
            raw_cfg = read_at(pinned.fd, ".bumpversion.cfg")
            return raw_cfg.decode("utf-8").splitlines() if raw_cfg is not None else []
        cfg = os.path.join(target_dir, ".bumpversion.cfg")
        if not os.path.exists(cfg):
            return []
        with open(cfg, encoding="utf-8") as fh:
            return fh.read().splitlines()

    @staticmethod
    def _pinned_git_kwargs(pinned: PinnedDirectory | None) -> dict[str, Any]:
        """Return the private Git invocation kwargs for one pinned checkout."""
        if pinned is None:
            return {}
        return {
            "_cwd_fd": pinned.fd,
            "_pass_fds": pinned.pass_fds,
            "_pinned_handle": pinned,
        }

    def _tag_on_remote(
        self,
        tag: str,
        path: str | None = None,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> bool:
        """True if ``tag`` exists on the ``origin`` remote (so it's published).

        Used to guard force-deletion of an orphan local tag: we only ever delete
        a tag that is local-only (never one already pushed). Network failure is
        treated as "on remote" (conservative — don't delete).
        """
        target_dir = self._resolve_path(path)
        res = self.git_action(
            command=f"git ls-remote --tags origin {tag}",
            path=target_dir,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        if res.status != "success":
            return True  # can't verify -> assume present, do not delete
        return f"refs/tags/{tag}" in (res.data or "")

    def _workspace_root_candidate(self) -> Path:
        """Return the absolute root spelling after rejecting lexical parents."""
        raw_root = _reject_lexical_parent(
            Path(os.path.expanduser(os.fspath(self.path))),
            label="workspace root",
        )
        root = Path(os.path.abspath(raw_root))
        current = Path(root.anchor)
        for component in root.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f"workspace root contains symlink component {current}")
            if current != root and current.exists() and not current.is_dir():
                raise ValueError(
                    f"workspace root contains non-directory component {current}"
                )
        return root

    def _workspace_root(self) -> Path:
        """Return the approved workspace root after rejecting symlink ancestry."""
        root = self._workspace_root_candidate()
        if not root.exists() or not root.is_dir():
            raise ValueError(f"workspace root is not a directory {root}")
        return root

    @staticmethod
    def _check_path_components(
        root: Path,
        components: tuple[str, ...],
        *,
        label: str,
        allow_leaf_symlink: bool,
    ) -> None:
        """Refuse a symlink or non-directory component anywhere along a path."""
        current = root
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

    @staticmethod
    def _check_real_containment(path: Path, root: Path, *, label: str) -> None:
        """Verify the RESOLVED path is still inside the workspace root.

        Closes the lexical-vs-real containment gap if the filesystem changed
        during validation.
        """
        try:
            real_path = path.resolve(strict=False)
            real_path.relative_to(root)
        except (OSError, ValueError) as exc:
            raise ValueError(f"{label} escapes workspace root") from exc

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
        path = _reject_lexical_parent(
            Path(os.path.expanduser(os.fspath(candidate))),
            label=label,
        )
        if not path.is_absolute():
            path = root / path
        path = Path(os.path.abspath(path))
        try:
            relative = path.relative_to(root)
        except ValueError as exc:
            raise ValueError(f"{label} escapes workspace root") from exc

        self._check_path_components(
            root,
            relative.parts,
            label=label,
            allow_leaf_symlink=allow_leaf_symlink,
        )

        if require_directory and (not path.exists() or not path.is_dir()):
            raise ValueError(f"{label} is not a directory {path}")

        # No symlink components were accepted above for project/target paths;
        # still verify the real path to close lexical-vs-real containment gaps
        # if the filesystem changed during validation.
        if not (allow_leaf_symlink and path.is_symlink()):
            self._check_real_containment(path, root, label=label)
        return path

    def _path_validation_result(
        self, operation: str, candidate: object, error: ValueError
    ) -> GitResult:
        """Return a typed refusal when an operation target is unsafe."""
        return GitResult(
            status="error",
            data="",
            error=GitError(message=f"{operation} refused: {error}", code=1),
            metadata=GitMetadata(
                command=operation,
                workspace=_project_label(candidate),
                return_code=1,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _validated_operation_path(self, path: str | None, *, operation: str) -> str:
        """Resolve an operation target only after the full path safety check."""
        # Resolve the default workspace through the approved root itself. This
        # matters for callers that constructed ``Git(path="relative-root")``:
        # feeding that relative spelling back through the root join would
        # produce ``relative-root/relative-root``.
        candidate: object = self.path if path is None else path
        try:
            candidate = self._operation_target_candidate(path)
            return str(
                self._validate_workspace_path(
                    candidate,
                    label=f"{operation} target",
                )
            )
        except ValueError as exc:
            logger.error("Unsafe %s target refused: %s", operation, exc)
            raise _UnsafeMutationTarget(operation, candidate, exc) from exc

    def _operation_target_candidate(self, path: str | None) -> str | Path:
        """Resolve an optional mutation target without changing its contract."""
        return self._workspace_root() if path is None else path

    def _revalidate_release_target(
        self,
        name: str,
        path: str,
        *,
        operation: str,
        require_directory: bool = False,
    ) -> _ReleaseTarget:
        """Revalidate and return one canonical release ``(name, path)`` pair.

        Release plans are built from a mutable project registry.  Rechecking
        the exact pair at the operation boundary prevents a lexical alias or a
        filesystem swap between planning and the mutation from widening the
        operation's scope.
        """
        validated = self._validate_workspace_path(
            path,
            label=f"{operation} project {name!r}",
            require_directory=require_directory,
        )
        if validated.name != name:
            raise ValueError(
                f"{operation} project {name!r} does not match canonical path {validated}"
            )
        return name, str(validated)

    def _release_plan_input_payload(
        self, config: dict[str, Any], options: dict[str, Any]
    ) -> dict[str, Any]:
        """Capture the exact legacy-plan inputs without retaining raw secrets."""
        project_map = [
            [str(url), str(path)]
            for url, path in sorted(self.project_map.items(), key=lambda item: item[0])
        ]
        categories = [
            [str(url), list(category)]
            for url, category in sorted(
                self._project_categories.items(), key=lambda item: item[0]
            )
        ]
        return {
            "workspace": str(self.path),
            "project_map": project_map,
            "project_categories": categories,
            "config": config,
            "options": options,
        }

    def _release_plan_with_filesystem_payload(
        self, config: dict[str, Any], options: dict[str, Any]
    ) -> dict[str, Any]:
        """Add immutable checkout state to the shared legacy-plan payload."""
        payload = self._release_plan_input_payload(config, options)
        payload["filesystem"] = self._release_scope_identity_payload()
        return payload

    def _release_scope_identity_payload(self) -> dict[str, Any]:
        """Capture root, checkout, Git, and normalized-origin identities.

        The legacy release planner remains the sole source of ordered target
        membership.  This payload only adds the immutable filesystem and Git
        state that must still agree when that existing plan reaches a mutation.
        """
        projects = [
            (str(url), str(path))
            for url, path in sorted(self.project_map.items(), key=lambda item: item[0])
        ]
        try:
            snapshot = snapshot_workspace(self._workspace_root(), projects)
            for project in snapshot["projects"]:
                metadata = project.get("git")
                if not isinstance(metadata, dict):
                    continue
                origin = metadata.get("origin")
                if origin is None:
                    metadata["origin_identity"] = None
                    continue
                try:
                    metadata["origin_identity"] = self._canonical_checkout_origin(
                        str(project["url"]), str(origin)
                    )
                except (TypeError, ValueError):
                    metadata["origin_identity"] = "<invalid-origin>"
                    snapshot["valid"] = False
            snapshot.setdefault("valid", True)
            return snapshot
        except (OperationBoundaryError, OSError, ValueError):
            return {"valid": False, "error": "unsafe release scope"}

    def _freeze_release_plan(
        self,
        operation: Literal["bump", "push"],
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        *,
        options: dict[str, Any],
        auxiliary_targets: list[_ReleaseTarget] | None = None,
    ) -> _ReleasePlanProvenance:
        """Freeze input provenance and ordered target membership for a run."""
        auxiliary = [[name, path] for name, path in (auxiliary_targets or [])]
        input_payload = self._release_plan_with_filesystem_payload(config, options)
        registry_digest = _release_plan_digest(
            self._release_plan_input_payload(config, options)
        )
        input_digest = _release_plan_digest(input_payload)
        plan_digest = _release_plan_digest(
            {
                "operation": operation,
                "input_digest": input_digest,
                "phases": _release_phase_payload(phase_list),
                "auxiliary_targets": auxiliary,
            }
        )
        root_identity, scope_identity = self._release_plan_provenance_identities(
            input_payload
        )
        return _ReleasePlanProvenance(
            operation=operation,
            input_digest=input_digest,
            plan_digest=plan_digest,
            root_identity=root_identity,
            scope_identity=scope_identity,
            registry_digest=registry_digest,
        )

    @staticmethod
    def _release_plan_provenance_identities(
        input_payload: dict[str, Any],
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Extract immutable filesystem identities from a frozen input payload."""
        filesystem = input_payload.get("filesystem")
        if not isinstance(filesystem, dict):
            return None, None
        root = filesystem.get("root")
        root_identity = dict(root) if isinstance(root, dict) else None
        return root_identity, copy.deepcopy(filesystem)

    def _release_plan_target_snapshot(
        self,
        provenance: _ReleasePlanProvenance,
        project_name: str,
        project_path: str,
    ) -> dict[str, Any]:
        """Return the frozen identity for one exact release target."""
        scope = provenance.scope_identity
        if not isinstance(scope, dict) or scope.get("valid") is not True:
            raise OperationBoundaryError("release plan has no valid filesystem scope")
        entries = scope.get("projects")
        if not isinstance(entries, list):
            raise OperationBoundaryError("release plan filesystem scope is malformed")
        expected_path = str(
            Path(os.path.abspath(os.path.expanduser(os.fspath(project_path))))
        )
        matches = self._release_plan_target_matches(
            entries, expected_path, project_name
        )
        if len(matches) != 1:
            raise OperationBoundaryError(
                f"release plan target is not uniquely bound: {project_name}"
            )
        return copy.deepcopy(matches[0])

    def _release_plan_target_matches(
        self,
        entries: list[Any],
        expected_path: str,
        project_name: str,
    ) -> list[dict[str, Any]]:
        """Select exact frozen ``(name, path)`` entries for one target."""
        matches: list[dict[str, Any]] = []
        for entry in entries:
            if not isinstance(entry, dict) or entry.get("path") != expected_path:
                continue
            url = entry.get("url")
            if not isinstance(url, str):
                continue
            if self._release_project_name_or_empty(url) != project_name:
                continue
            if entry.get("exists") is True:
                matches.append(entry)
        return matches

    def _normalized_release_target_git(
        self,
        expected_git: dict[str, Any],
        actual_git: Any,
        expected: dict[str, Any],
    ) -> dict[str, Any]:
        """Normalize live origin identity before comparing a frozen Git snapshot."""
        if not isinstance(actual_git, dict):
            raise OperationBoundaryError("release plan Git identity disappeared")
        normalized = copy.deepcopy(actual_git)
        self._reject_release_push_origin(expected_git, normalized)
        origin = normalized.get("origin")
        manifest_url = expected.get("url")
        if origin is None:
            normalized["origin_identity"] = None
        elif isinstance(manifest_url, str):
            try:
                normalized["origin_identity"] = self._canonical_checkout_origin(
                    manifest_url, str(origin)
                )
            except ValueError:
                normalized["origin_identity"] = "<invalid-origin>"
        else:
            normalized["origin_identity"] = "<invalid-origin>"
        return normalized

    @staticmethod
    def _reject_release_push_origin(
        expected_git: dict[str, Any], actual_git: dict[str, Any]
    ) -> None:
        """Reject old or current snapshots that contain a configured push URL."""
        values = (expected_git.get("push_origin"), actual_git.get("push_origin"))
        if any(value is not None for value in values):
            raise OperationBoundaryError(
                "release plan refuses configured remote.origin.pushurl"
            )

    def _assert_pinned_release_target(
        self,
        provenance: _ReleasePlanProvenance,
        project_name: str,
        project_path: str,
        pinned: PinnedDirectory,
        *,
        expected: dict[str, Any] | None = None,
    ) -> None:
        """Compare a pinned target with the immutable release-plan snapshot."""
        expected = expected or self._release_plan_target_snapshot(
            provenance, project_name, project_path
        )
        scope = provenance.scope_identity
        expected_root = scope.get("root") if isinstance(scope, dict) else None
        if expected_root != pinned.root_identity.as_dict():
            raise OperationBoundaryError("release plan workspace root identity changed")
        pinned.assert_path_identity()
        actual = snapshot_pinned_checkout(pinned)
        expected_git = expected.get("git")
        actual_git = actual.get("git")
        if isinstance(expected_git, dict):
            actual_git = self._normalized_release_target_git(
                expected_git, actual_git, expected
            )
        if (
            actual.get("path") != expected.get("path")
            or actual.get("checkout") != expected.get("checkout")
            or actual_git != expected_git
        ):
            raise OperationBoundaryError(
                f"release plan target identity changed: {project_name}"
            )

    def _bind_release_plan_target(
        self,
        provenance: _ReleasePlanProvenance,
        project_name: str,
        project_path: str,
        pinned: PinnedDirectory,
        *,
        plan_assertion: Callable[[], None] | None = None,
    ) -> None:
        """Attach the frozen target assertion to its operation handle."""
        expected = self._release_plan_target_snapshot(
            provenance, project_name, project_path
        )

        def assert_target() -> None:
            self._assert_pinned_release_target(
                provenance,
                project_name,
                project_path,
                pinned,
                expected=expected,
            )
            if plan_assertion is not None:
                plan_assertion()

        expected_url = expected.get("url")
        if not isinstance(expected_url, str):
            raise OperationBoundaryError("release plan target has no repository URL")
        pinned.expected_origin = canonical_repository_url(expected_url)
        pinned.boundary_assertion = assert_target
        assert_target()

    def _assert_release_plan_structure(
        self,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        auxiliary_targets: list[_ReleaseTarget] | None = None,
    ) -> None:
        """Check mutable plan inputs without rescanning every checkout."""
        expected_registry = provenance.registry_digest
        current_registry = _release_plan_digest(
            self._release_plan_input_payload(config, plan_options)
        )
        auxiliary = [[name, path] for name, path in (auxiliary_targets or [])]
        current_plan = _release_plan_digest(
            {
                "operation": provenance.operation,
                "input_digest": provenance.input_digest,
                "phases": _release_phase_payload(phase_list),
                "auxiliary_targets": auxiliary,
            }
        )
        if (
            expected_registry != current_registry
            or current_plan != provenance.plan_digest
        ):
            raise _ReleasePlanDrift(provenance.operation, provenance.plan_digest)

    def _release_plan_matches(
        self,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        *,
        options: dict[str, Any],
        auxiliary_targets: list[_ReleaseTarget] | None = None,
    ) -> bool:
        """Check input registry, config, and ordered target membership for drift."""
        current_payload = self._release_plan_with_filesystem_payload(config, options)
        current_input = self._release_plan_input_digest(current_payload)
        if current_input != provenance.input_digest:
            return False
        auxiliary = [[name, path] for name, path in (auxiliary_targets or [])]
        current_plan = _release_plan_digest(
            {
                "operation": provenance.operation,
                "input_digest": current_input,
                "phases": _release_phase_payload(phase_list),
                "auxiliary_targets": auxiliary,
            }
        )
        return current_plan == provenance.plan_digest

    @staticmethod
    def _release_plan_scope_valid(payload: dict[str, Any]) -> bool:
        """Return whether a current plan payload still has a usable scope."""
        scope = payload.get("filesystem", {})
        return isinstance(scope, dict) and bool(scope.get("valid", False))

    @classmethod
    def _release_plan_input_digest(cls, payload: dict[str, Any]) -> str:
        """Digest only a payload whose filesystem scope is valid."""
        if not cls._release_plan_scope_valid(payload):
            return "<invalid-scope>"
        return _release_plan_digest(payload)

    def _assert_release_plan(
        self,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        *,
        options: dict[str, Any],
        auxiliary_targets: list[_ReleaseTarget] | None = None,
    ) -> None:
        """Raise a typed refusal when a legacy plan no longer matches inputs."""
        try:
            matches = self._release_plan_matches(
                provenance,
                config,
                phase_list,
                options=options,
                auxiliary_targets=auxiliary_targets,
            )
        except (KeyError, TypeError, ValueError, OverflowError):
            # A concurrently malformed phase/config input is drift, not a
            # reason to leak an exception past the legacy API or continue with
            # an unverified target set.
            matches = False
        if not matches:
            raise _ReleasePlanDrift(provenance.operation, provenance.plan_digest)

    def _release_plan_drift_result(self, drift: _ReleasePlanDrift) -> GitResult:
        """Return a privacy-safe, auditable result for release-plan drift."""
        return GitResult(
            status="error",
            data=f"release_plan_digest={drift.expected_digest}",
            error=GitError(
                message=(
                    f"phased_{drift.operation} aborted: release plan changed "
                    "before the next mutation"
                ),
                code=409,
            ),
            metadata=GitMetadata(
                command=f"phased_{drift.operation}",
                workspace=_project_label(self.path),
                return_code=409,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    @staticmethod
    def _record_release_plan(
        progress: dict | None, provenance: _ReleasePlanProvenance
    ) -> None:
        """Expose only digests, never raw plan paths, in progress state."""
        if progress is None:
            return
        progress["release_plan_digest"] = provenance.plan_digest
        progress["release_plan_input_digest"] = provenance.input_digest

    @staticmethod
    def _release_plan_receipt_result(message: str) -> GitResult:
        """Return a typed refusal for an unsafe or replayed push plan."""
        return GitResult(
            status="error",
            data="",
            error=GitError(message=message, code=409),
            metadata=GitMetadata(
                command="phased_push",
                workspace="configured-workspace",
                return_code=409,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _begin_push_plan_receipt(
        self, provenance: _ReleasePlanProvenance
    ) -> tuple[list[GitResult] | None, GitResult | None]:
        """Consume a push plan once, or return its recorded idempotent result."""
        try:
            receipt, created = self._record_push_plan_start(provenance)
        except (OperationBoundaryError, OSError) as exc:
            return None, self._release_plan_receipt_result(
                f"phased_push refused: cannot record release plan ({type(exc).__name__})"
            )
        if created:
            return None, None
        return self._replay_push_plan_receipt(receipt, provenance)

    def _record_push_plan_start(
        self, provenance: _ReleasePlanProvenance
    ) -> tuple[dict[str, Any] | None, bool]:
        """Atomically create the durable push-consumption marker when absent."""
        with _hold_repo_mutation(str(self._workspace_root())):
            with open_directory(self._workspace_root()) as root:
                if (
                    provenance.root_identity is not None
                    and root.identity.as_dict() != provenance.root_identity
                ):
                    raise OperationBoundaryError("workspace root identity changed")
                receipt = read_release_plan_receipt(root)
                if receipt is not None:
                    return receipt, False
                write_release_plan_receipt(
                    root,
                    {
                        "operation": "push",
                        "plan_digest": provenance.plan_digest,
                        "input_digest": provenance.input_digest,
                        "state": "started",
                        "results": [],
                    },
                )
                return None, True

    def _replay_push_plan_receipt(
        self,
        receipt: dict[str, Any] | None,
        provenance: _ReleasePlanProvenance,
    ) -> tuple[list[GitResult] | None, GitResult | None]:
        """Validate and replay a completed durable push outcome."""
        if receipt is None:
            return None, self._release_plan_receipt_result(
                "phased_push refused: release-plan receipt disappeared"
            )

        operation = receipt.get("operation")
        digest = receipt.get("plan_digest")
        input_digest = receipt.get("input_digest")
        if (
            operation != "push"
            or digest != provenance.plan_digest
            or input_digest != provenance.input_digest
        ):
            return None, self._release_plan_receipt_result(
                "phased_push refused: a different release plan has already been "
                "consumed"
            )
        if receipt.get("state") != "completed":
            return None, self._release_plan_receipt_result(
                "phased_push refused: release plan was consumed without a "
                "recorded outcome"
            )
        raw_results = receipt.get("results")
        if not isinstance(raw_results, list):
            return None, self._release_plan_receipt_result(
                "phased_push refused: release-plan outcome is malformed"
            )
        try:
            replayed = [GitResult.model_validate(item) for item in raw_results]
        except ValidationError:
            return None, self._release_plan_receipt_result(
                "phased_push refused: release-plan outcome is malformed"
            )
        return replayed, None

    def _complete_push_plan_receipt(
        self,
        provenance: _ReleasePlanProvenance,
        results: list[GitResult],
    ) -> None:
        """Persist the terminal push outcome for exact idempotent retries."""
        try:
            with _hold_repo_mutation(str(self._workspace_root())):
                with open_directory(self._workspace_root()) as root:
                    if (
                        provenance.root_identity is not None
                        and root.identity.as_dict() != provenance.root_identity
                    ):
                        raise OperationBoundaryError(
                            "workspace root identity changed before receipt completion"
                        )
                    receipt = read_release_plan_receipt(root)
                    if (
                        not isinstance(receipt, dict)
                        or receipt.get("operation") != "push"
                        or receipt.get("plan_digest") != provenance.plan_digest
                        or receipt.get("input_digest") != provenance.input_digest
                    ):
                        raise OperationBoundaryError(
                            "release-plan receipt changed before completion"
                        )
                    write_release_plan_receipt(
                        root,
                        {
                            "operation": "push",
                            "plan_digest": provenance.plan_digest,
                            "input_digest": provenance.input_digest,
                            "state": "completed",
                            "results": receipt_result_payload(results),
                        },
                        atomic=True,
                    )
        except (OperationBoundaryError, OSError, TypeError, ValueError) as exc:
            # The push may already have had an external side effect.  Do not
            # pretend a retry is safe when its terminal outcome could not be
            # made durable; the started receipt intentionally remains the
            # conservative anti-replay marker.
            logger.error(
                "Could not persist phased_push outcome: %s", type(exc).__name__
            )

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

    @staticmethod
    def _collect_validation_result(
        future: "concurrent.futures.Future",
    ) -> tuple[Any, bool]:
        """One project's validation payload plus whether it counts as a pass.

        A result object with no ``success`` attribute cannot be read as a pass,
        so it is recorded verbatim and treated as a failure.
        """
        try:
            result = future.result()
        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)
            return {"success": False, "error": "Operation failed"}, False

        entry = result.model_dump() if hasattr(result, "model_dump") else result
        if not hasattr(result, "success"):
            return entry, False
        return entry, bool(result.success)

    def _run_parallel_validation(
        self, effective_threads: int
    ) -> tuple[dict[str, Any], bool]:
        """Validate every mapped project; return per-repo results and the verdict."""
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
                repo_name = futures[future].split("/")[-1].replace(".git", "")
                entry, ok = self._collect_validation_result(future)
                validation_results[repo_name] = entry
                passed = passed and ok
        return validation_results, passed

    def _release_after_validation(
        self, *, passed: bool, auto_bump: bool, auto_push: bool, bump_part: str
    ) -> dict[str, Any]:
        """Run the bump/push release steps, but only when validation passed."""
        if not passed:
            if auto_bump or auto_push:
                logger.warning("Validation failed. Skipping bump and push.")
            return {}

        logger.info("All validations passed.")
        release_results: dict[str, Any] = {}
        if auto_bump:
            logger.info(f"Triggering phased bumpversion ({bump_part})...")
            release_results["bump"] = self.phased_bumpversion(part=bump_part)
        if auto_push:
            logger.info("Triggering phased push...")
            release_results["push"] = self.phased_push()
        return release_results

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

        validation_results, passed = self._run_parallel_validation(effective_threads)
        release_results = self._release_after_validation(
            passed=passed,
            auto_bump=auto_bump,
            auto_push=auto_push,
            bump_part=bump_part,
        )

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

    @staticmethod
    def _parse_repository_command(command: str) -> tuple[list[str], dict[str, str]]:
        """Split one repository command into argv plus leading VAR=value pairs.

        Shell control syntax is refused outright: this executor never runs a
        shell, so a pipeline or redirect would silently become a literal
        argument rather than doing what its author intended.
        """
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
        return command_argv, command_env

    @staticmethod
    def _repository_command_env(
        env: dict | None, command_env: dict[str, str]
    ) -> dict[str, str]:
        """The child environment for one repository command."""
        current_env = env if env else os.environ.copy()
        current_env.update(command_env)

        # Ensure ~/.local/bin is in PATH for tools like bump2version
        local_bin = os.path.expanduser("~/.local/bin")
        if local_bin not in current_env.get("PATH", ""):
            current_env["PATH"] = f"{local_bin}:{current_env.get('PATH', '')}"

        # Ensure Python output is unbuffered so we get real-time logs
        current_env["PYTHONUNBUFFERED"] = "1"
        return current_env

    def _append_debug_log(self, text: str) -> None:
        """Append one entry to the debug log under the shared lock."""
        with self.debug_lock, open(self.debug_log_path, "a") as log_file:
            log_file.write(text)
            log_file.flush()

    def _drain_process_output(
        self, process: subprocess.Popen, capture: "_CommandOutputCapture"
    ) -> None:
        """Read the child's output line by line as it becomes available."""
        if not process.stdout:
            return
        for line in process.stdout:
            capture.add(line)
            self._append_debug_log(
                f"[{datetime.datetime.now().isoformat()}] "
                "[repository output line omitted]\n"
            )

    @staticmethod
    def _kill_process(process: subprocess.Popen) -> None:
        """Last-resort kill of a command that would not terminate."""
        process.kill()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass

    @staticmethod
    def _terminate_process_group(process: subprocess.Popen) -> None:
        """SIGTERM then SIGKILL a timed-out command's whole process group."""
        if not hasattr(os, "killpg"):
            Git._kill_process(process)
            return
        try:
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                process.wait(timeout=5)
        except Exception:  # nosec B110
            Git._kill_process(process)

    @staticmethod
    def _repository_command_result(
        *,
        operation: str,
        target_path: str,
        return_code: int,
        out: str,
        quiet: bool,
    ) -> GitResult:
        """Build -- and log -- the result of one finished repository command."""
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

    def _await_repository_command(
        self,
        process: subprocess.Popen,
        capture: "_CommandOutputCapture",
        timeout: int,
    ) -> None:
        """Drain and wait for one command, killing it if it overruns *timeout*."""
        try:
            # Write start marker
            self._append_debug_log(
                f"\n[{datetime.datetime.now().isoformat()}] "
                "Starting repository operation\n"
            )

            reader_thread = threading.Thread(
                target=self._drain_process_output,
                args=(process, capture),
                daemon=True,
            )
            reader_thread.start()

            # Wait for process to complete, with a safety timeout
            process.wait(timeout=timeout)
            reader_thread.join(timeout=1.0)
            process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            logger.warning("Repository operation timed out")
            self._terminate_process_group(process)
            self._append_debug_log(
                f"[{datetime.datetime.now().isoformat()}] "
                f"ERROR: Command timed out after {timeout} seconds\n"
            )

    def _run_pinned_repository_command(
        self,
        command_argv: list[str],
        current_env: dict[str, str],
        *,
        target_path: str,
        operation: str,
        quiet: bool,
        timeout: int,
        raw_output: bool,
        cwd_fd: int,
        pass_fds: tuple[int, ...],
        path_anchor: tuple[str, int] | None,
    ) -> GitResult:
        """Run one command with a descriptor-pinned working directory."""
        inherited = tuple(dict.fromkeys((cwd_fd, *pass_fds)))
        if path_anchor is not None:
            lexical_path, destination_fd = path_anchor
            if lexical_path not in command_argv:
                raise OperationBoundaryError(
                    "descriptor-pinned path anchor is absent from the Git command"
                )
            command_argv = [
                f"/proc/self/fd/{destination_fd}" if token == lexical_path else token
                for token in command_argv
            ]
        try:
            process = subprocess.Popen(
                command_argv,
                shell=False,
                cwd=f"/proc/self/fd/{cwd_fd}",
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
                env=current_env,
                bufsize=1,
                start_new_session=True,
                pass_fds=inherited,
            )
        except OSError as exc:
            return self._repository_command_result(
                operation=operation,
                target_path=target_path,
                return_code=1,
                out=_privacy_safe_diagnostic(
                    f"descriptor-pinned repository operation failed: {type(exc).__name__}"
                ),
                quiet=quiet,
            )

        capture = _CommandOutputCapture()
        self._await_repository_command(process, capture, timeout)

        captured = capture.text()
        out = captured if raw_output else _privacy_safe_diagnostic(captured)
        return self._repository_command_result(
            operation=operation,
            target_path=target_path,
            return_code=process.returncode,
            out=out,
            quiet=quiet,
        )

    def git_action(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict | None = None,
        timeout: int = 1800,
        raw_output: bool = False,
        *,
        _cwd_fd: int | None = None,
        _pass_fds: tuple[int, ...] = (),
        _path_anchor: tuple[str, int] | None = None,
        _pinned_handle: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Execute a Git command in the specified directory.

        The operation is admitted through descriptor-pinned helpers before
        any subprocess is started.
        """
        return self._git_action_impl(
            command=command,
            path=path,
            quiet=quiet,
            env=env,
            timeout=timeout,
            raw_output=raw_output,
            _cwd_fd=_cwd_fd,
            _pass_fds=_pass_fds,
            _path_anchor=_path_anchor,
            _pinned_handle=_pinned_handle,
        )

    def _git_action_impl(
        self,
        command: str,
        path: str | None = None,
        quiet: bool = False,
        env: dict | None = None,
        timeout: int = 1800,
        raw_output: bool = False,
        *,
        _cwd_fd: int | None = None,
        _pass_fds: tuple[int, ...] = (),
        _path_anchor: tuple[str, int] | None = None,
        _pinned_handle: PinnedDirectory | None = None,
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
        try:
            target_path = self._resolve_path(path)
        except ValueError as exc:
            return self._path_validation_result(
                "git_action", self.path if path is None else path, exc
            )

        command_argv, command_env = self._parse_repository_command(command)
        current_env = self._repository_command_env(env, command_env)

        operation = _operation_label(command_argv)
        is_git_push = self._is_git_push(command_argv)
        logger.info("Executing repository operation")
        runner = {
            False: self._git_action_with_path,
            True: self._git_action_with_fd,
        }[_cwd_fd is not None]
        try:
            if _cwd_fd is None and (_pass_fds or _path_anchor or _pinned_handle):
                raise OperationBoundaryError(
                    "descriptor-pinned arguments require a pinned cwd handle"
                )
            return runner(
                command_argv=command_argv,
                current_env=current_env,
                target_path=target_path,
                operation=operation,
                quiet=quiet,
                timeout=timeout,
                raw_output=raw_output,
                cwd_fd=_cwd_fd,
                pass_fds=_pass_fds,
                path_anchor=_path_anchor,
                pinned_handle=_pinned_handle,
                is_git_push=is_git_push,
            )
        except (OperationBoundaryError, OSError) as exc:
            return self._repository_command_result(
                operation=operation,
                target_path=target_path,
                return_code=1,
                out=_privacy_safe_diagnostic(
                    "descriptor-pinned repository operation refused: "
                    f"{type(exc).__name__}: {exc}"
                ),
                quiet=quiet,
            )

    @staticmethod
    def _is_git_push(command_argv: list[str]) -> bool:
        """Recognize a direct Git push for post-command boundary validation."""
        return bool(
            len(command_argv) > 1
            and os.path.basename(command_argv[0]) == "git"
            and command_argv[1] == "push"
        )

    def _git_action_with_fd(
        self,
        *,
        command_argv: list[str],
        current_env: dict[str, str],
        target_path: str,
        operation: str,
        quiet: bool,
        timeout: int,
        raw_output: bool,
        cwd_fd: int | None,
        pass_fds: tuple[int, ...],
        path_anchor: tuple[str, int] | None,
        pinned_handle: PinnedDirectory | None,
        is_git_push: bool,
    ) -> GitResult:
        """Run a command using a caller-provided descriptor-pinned cwd."""
        if cwd_fd is None:
            raise OperationBoundaryError("descriptor-pinned cwd is missing")
        if pinned_handle is not None:
            if cwd_fd != pinned_handle.fd:
                raise OperationBoundaryError(
                    "descriptor-pinned cwd does not match the operation handle"
                )
            if path_anchor is not None:
                _anchor_path, destination_fd = path_anchor
                if pinned_handle.target_fd != destination_fd:
                    raise OperationBoundaryError(
                        "descriptor-pinned destination does not match the operation handle"
                    )
            pinned_handle.assert_operation_identity()
            command_argv = pinned_handle.anchored_git_command(command_argv)
            current_env = pinned_handle.anchored_git_environment(current_env)
            pass_fds = pinned_handle.pass_fds
        else:
            raise OperationBoundaryError(
                "descriptor-pinned operation handle is missing"
            )
        result = self._run_pinned_repository_command(
            command_argv,
            current_env,
            target_path=target_path,
            operation=operation,
            quiet=quiet,
            timeout=timeout,
            raw_output=raw_output,
            cwd_fd=cwd_fd,
            pass_fds=pass_fds,
            path_anchor=path_anchor,
        )
        self._assert_git_action_boundary(pinned_handle, is_git_push)
        return result

    def _pin_git_action_target(
        self, root: PinnedDirectory, target_path: str
    ) -> PinnedDirectory:
        """Pin a workspace target or an explicitly registered linked worktree."""
        target = Path(target_path)
        try:
            target.relative_to(root.path)
        except ValueError:
            # WorktreeManager deliberately places linked checkouts under its
            # own configured root, which may be a sibling of the workspace.
            # Admit only that registered root and keep the manager workspace
            # as the metadata/plan boundary; arbitrary external paths remain
            # refused.
            from repository_manager.worktree import WORKTREE_ROOT

            allowed_root = Path(
                os.path.abspath(os.path.expanduser(os.fspath(WORKTREE_ROOT)))
            )
            try:
                target.relative_to(allowed_root)
            except ValueError as exc:
                raise OperationBoundaryError(
                    "operation target escapes workspace root"
                ) from exc
            return pin_existing_under(root, target, allowed_root)
        return pin_existing(root, target)

    def _git_action_with_path(
        self,
        *,
        command_argv: list[str],
        current_env: dict[str, str],
        target_path: str,
        operation: str,
        quiet: bool,
        timeout: int,
        raw_output: bool,
        cwd_fd: int | None,
        pass_fds: tuple[int, ...],
        path_anchor: tuple[str, int] | None,
        pinned_handle: PinnedDirectory | None,
        is_git_push: bool,
    ) -> GitResult:
        """Pin a lexical path for the duration of one Git command."""
        del cwd_fd, pass_fds, path_anchor, pinned_handle
        with open_directory(self._workspace_root()) as root:
            with self._pin_git_action_target(root, target_path) as pinned:
                pinned.assert_operation_identity()
                anchored = pinned.anchored_git_command(command_argv)
                current_env = pinned.anchored_git_environment(current_env)
                result = self._run_pinned_repository_command(
                    anchored,
                    current_env,
                    target_path=target_path,
                    operation=operation,
                    quiet=quiet,
                    timeout=timeout,
                    raw_output=raw_output,
                    cwd_fd=pinned.fd,
                    pass_fds=pinned.pass_fds,
                    path_anchor=None,
                )
                self._assert_git_action_boundary(pinned, is_git_push)
                return result

    @staticmethod
    def _assert_git_action_boundary(
        pinned: PinnedDirectory | None, is_git_push: bool
    ) -> None:
        """Revalidate Git metadata and the planned boundary after execution."""
        if pinned is None:
            return
        pinned.assert_git_identity()
        if is_git_push and pinned.boundary_assertion is not None:
            pinned.boundary_assertion()
        pinned.assert_path_identity()

    def cleanup_artifacts(self, target_dir: str) -> None:
        """Removes test artifacts and temporary files from the specified directory."""
        active = getattr(self, "_active_cleanup_handle", None)
        if active is not None and target_dir == active.proc_path:
            cleanup_pinned_directory(
                active,
                file_patterns=tuple(_CLEANUP_FILE_PATTERNS),
                directory_names=frozenset(_CLEANUP_DIR_PATTERNS),
                ignored_directory_names=frozenset(_CLEANUP_IGNORED_DIRS),
                root_script_patterns=tuple(_CLEANUP_ROOT_SCRIPT_PATTERNS),
            )
            return

        try:
            target_path = self._validated_operation_path(
                target_dir, operation="cleanup_artifacts"
            )
            with open_directory(self._workspace_root()) as root:
                if not path_exists(root, target_path):
                    return
                with pin_existing(root, target_path) as pinned:
                    cleanup_pinned_directory(
                        pinned,
                        file_patterns=tuple(_CLEANUP_FILE_PATTERNS),
                        directory_names=frozenset(_CLEANUP_DIR_PATTERNS),
                        ignored_directory_names=frozenset(_CLEANUP_IGNORED_DIRS),
                        root_script_patterns=tuple(_CLEANUP_ROOT_SCRIPT_PATTERNS),
                    )
        except (
            _UnsafeMutationTarget,
            OperationBoundaryError,
            OSError,
            ValueError,
        ) as exc:
            logger.error("Artifact cleanup refused: error_type=%s", type(exc).__name__)

    def clone_projects(self, projects: list[str] | None = None) -> list[GitResult]:
        """
        Clone all specified Git projects in parallel using multiple threads.

        Returns:
            List[GitResult]: A list of GitResult objects, one for each clone operation.
        """
        try:
            root_path = self._workspace_root_candidate()
            with open_directory(root_path, create=True) as root:
                root.assert_root_identity()

                targets = []
                if projects:
                    for url in projects:
                        name = repository_name(url)
                        targets.append((url, str(root_path / name)))
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
                        executor.submit(
                            self.clone_repository,
                            url,
                            path,
                            _root=root,
                        ): (url, path)
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
    def clone_repository(
        self,
        url: str,
        target_path: str,
        *,
        _root: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Clone a single Git repository to a specific target path.

        Args:
            url (str): The repository URL to clone.
            target_path (str): The absolute path where the repository should be cloned.

        Returns:
            GitResult: The result of the Git clone command.
        """
        target_path = self._validated_clone_target(target_path)
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

        clone_filter = os.environ.get("REPOSITORY_MANAGER_CLONE_FILTER", "").strip()
        filter_arg = (
            f" --filter={shlex.quote(clone_filter)}"
            if clone_filter in {"blob:none", "tree:0"}
            else ""
        )
        return self._clone_repository_operation(url, target_path, filter_arg, _root)

    def _clone_repository_operation(
        self,
        url: str,
        target_path: str,
        filter_arg: str,
        supplied_root: PinnedDirectory | None,
    ) -> GitResult:
        """Create, hand off, and validate one descriptor-pinned clone target."""
        try:
            root_context = (
                contextlib.nullcontext(supplied_root)
                if supplied_root is not None
                else open_directory(self._workspace_root())
            )
            with root_context as root:
                if root is None:
                    raise OperationBoundaryError("workspace root handle is missing")
                root.assert_root_identity()
                destination = pin_creation(root, target_path, create_parents=True)
                with destination:
                    destination.assert_path_identity()
                    destination.reserve_leaf()
                    destination.assert_path_identity()
                    if destination.target_fd is None:
                        raise OperationBoundaryError(
                            "clone destination was not pinned after reservation"
                        )
                    command = (
                        f"git clone{filter_arg} -- {shlex.quote(url)} "
                        f"{shlex.quote(target_path)}"
                    )
                    result = self.git_action(
                        command,
                        path=str(destination.path),
                        _cwd_fd=destination.fd,
                        _pass_fds=destination.pass_fds,
                        _path_anchor=(target_path, destination.target_fd),
                        _pinned_handle=destination,
                    )
                    if result.status == "success":
                        destination.assert_handoff()
                        # A mocked/test transport may report success without
                        # materializing a checkout.  Release the empty
                        # reservation in that case so the historical sync
                        # contract (and a later retry) still sees it as
                        # absent.  A real clone always leaves at least .git.
                        if destination.target_fd is not None:
                            try:
                                if destination.leaf is not None and not os.listdir(
                                    destination.target_fd
                                ):
                                    os.rmdir(destination.leaf, dir_fd=destination.fd)
                            except OSError:
                                # A non-empty destination is the expected real
                                # clone handoff; any inability to inspect it is
                                # left for the descriptor close/final result.
                                pass
        except OperationBoundaryError as exc:
            return self._path_validation_result("clone_repository", target_path, exc)
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

    @staticmethod
    def _checkout_guard_skip(repo_label: str, blocked: dict) -> GitResult:
        """The ``skipped`` record for a checkout the canonical guard refused."""
        return GitResult(
            status="skipped",
            data=blocked.get("detail", ""),
            error=GitError(message=blocked["error"], code=0),
            metadata=GitMetadata(
                command="checkout-guard",
                workspace=repo_label,
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _validated_clone_target(self, target_path: str) -> str:
        """Validate clone destinations before resolving caller-relative paths."""
        # ``clone_repository`` historically accepted a destination relative to
        # the caller's cwd (unlike the other repository operations, which are
        # workspace-relative). Preserve that contract, but reject lexical
        # traversal before converting it to an absolute path and then route the
        # canonical destination through the same workspace validator as every
        # other mutation.
        raw_target = Path(os.path.expanduser(target_path))
        try:
            _reject_lexical_parent(raw_target, label="clone_repository target")
        except ValueError as exc:
            raise _UnsafeMutationTarget("clone_repository", target_path, exc) from exc
        if not raw_target.is_absolute():
            raw_target = Path.cwd() / raw_target
        return self._validated_operation_path(
            str(raw_target), operation="clone_repository"
        )

    def _guarded_default_branch_checkout(
        self,
        target_path: str,
        default_branch: str,
        results: list[GitResult],
        *,
        pinned: PinnedDirectory | None = None,
    ) -> None:
        """Check out *default_branch*, never on a dirty canonical tree.

        WT-3 (CONCEPT:RM-WORKTREE, CONCEPT:RM-CANON-GUARD) --
        non-destructive / worktree-aware. Never switch branches on a
        dirty canonical tree: a concurrent session may have
        uncommitted work here, and a forced checkout would disrupt
        it. guarded_canonical_mutation skips the checkout (loudly,
        with the repo named) instead. Session work belongs in a
        worktree under WORKTREE_ROOT anyway, which this never
        touches.
        """
        repo_label = _project_label(target_path)
        with guarded_canonical_mutation(
            self, target_path, repo_label, "check out default branch"
        ) as blocked:
            if blocked is not None:
                results.append(self._checkout_guard_skip(repo_label, blocked))
                return
            checkout_result = self.git_action(
                f'git checkout "{default_branch}"',
                **self._pinned_path_kwargs(target_path, pinned),
            )
            results.append(checkout_result)
            logger.info("Checked out configured default branch")

    def _checkout_default_branch(
        self,
        target_path: str,
        results: list[GitResult],
        *,
        pinned: PinnedDirectory | None = None,
    ) -> None:
        """Resolve the project's default branch and switch to it if needed."""
        default_branch_result = self.git_action(
            "git symbolic-ref refs/remotes/origin/HEAD",
            **self._pinned_path_kwargs(target_path, pinned),
        )
        if default_branch_result.status != "success":
            results.append(default_branch_result)
            logger.error("Failed to resolve the configured default branch")
            return

        default_branch = re.sub(
            "refs/remotes/origin/", "", default_branch_result.data
        ).strip()
        current_branch = self.git_action(
            "git rev-parse --abbrev-ref HEAD",
            quiet=True,
            **self._pinned_path_kwargs(target_path, pinned),
        ).data.strip()
        if current_branch == default_branch:
            logger.info("Configured project is already on its default branch")
            return

        self._guarded_default_branch_checkout(
            target_path, default_branch, results, pinned=pinned
        )

    @staticmethod
    def _pinned_path_kwargs(
        target_path: str, pinned: PinnedDirectory | None
    ) -> dict[str, Any]:
        """Build Git keyword arguments for a path, preserving its pin."""
        if pinned is None:
            return {"path": target_path}
        return {
            "path": target_path,
            "_cwd_fd": pinned.fd,
            "_pass_fds": pinned.pass_fds,
            "_pinned_handle": pinned,
        }

    @staticmethod
    def _combine_pull_results(target_path: str, results: list[GitResult]) -> GitResult:
        """Fold the pull (and optional checkout) results into one result."""
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

    def _pull_project_unpinned(self, target_path: str) -> GitResult:
        """Keep the legacy mocked-transport behavior for absent checkouts."""
        results = [self.git_action(command="git pull", path=target_path)]
        return self._combine_pull_results(target_path, results)

    @_exclusive_repo_mutation
    def pull_project(
        self,
        path: str | None = None,
        *,
        _root: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Pull updates for a single Git project and optionally checkout the default branch.

        Args:
            path (str): The path to the project to pull. Defaults to self.path.

        Returns:
            GitResult: The result of the pull operation.
        """
        target_path = self._validated_operation_path(path, operation="pull_project")
        return self._pull_project_operation(target_path, _root)

    def _pull_project_operation(
        self, target_path: str, supplied_root: PinnedDirectory | None
    ) -> GitResult:
        """Execute a pinned pull and preserve the legacy absent-checkout fallback."""
        try:
            results = self._pull_project_with_root(target_path, supplied_root)
        except OperationBoundaryError as exc:
            if not Path(target_path).is_dir():
                return self._pull_project_unpinned(target_path)
            return self._path_validation_result("pull_project", target_path, exc)

        logger.info("Repository pull completed")

        return self._combine_pull_results(target_path, results)

    def _pull_project_with_root(
        self, target_path: str, supplied_root: PinnedDirectory | None
    ) -> list[GitResult]:
        """Pull one checkout while retaining the caller's root boundary."""
        root_context = (
            contextlib.nullcontext(supplied_root)
            if supplied_root is not None
            else open_directory(self._workspace_root())
        )
        with root_context as root:
            if root is None:
                raise OperationBoundaryError("workspace root handle is missing")
            root.assert_root_identity()
            with pin_existing(root, target_path) as pinned:
                pinned.assert_path_identity()
                results = [
                    self.git_action(
                        command="git pull",
                        path=target_path,
                        _cwd_fd=pinned.fd,
                        _pass_fds=pinned.pass_fds,
                        _pinned_handle=pinned,
                    )
                ]
                if self.set_to_default_branch:
                    self._checkout_default_branch(target_path, results, pinned=pinned)
                return results

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

    def _has_unpushed_commits(
        self, target_path: str, *, pinned: PinnedDirectory | None = None
    ) -> bool:
        """True when the local branch has commits the remote lacks.

        Used to skip the pre-push gate on no-op repos (nothing to validate).
        On any uncertainty (no upstream, error) returns True so the gate runs.
        """
        res = self.git_action(
            command="git rev-list --count @{u}..HEAD",
            path=target_path,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        if res.status != "success" or not res.data:
            return True
        try:
            return int(res.data.strip()) > 0
        except (ValueError, AttributeError):
            return True

    def _unpushed_changed_files(
        self, target_path: str, *, pinned: PinnedDirectory | None = None
    ) -> list[str]:
        """Files touched by the commits about to be pushed (``@{u}..HEAD``).

        Lets the pre-push gate scope per-file hooks to just the diff being
        pushed. Returns ``[]`` when the diff can't be computed (no upstream,
        error) — the caller then falls back to an ``--all-files`` run.
        """
        res = self.git_action(
            command="git diff --name-only @{u}..HEAD",
            path=target_path,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        if res.status != "success" or not res.data:
            return []
        return [line.strip() for line in res.data.splitlines() if line.strip()]

    @staticmethod
    def _gate_incomplete_result(error: object) -> GitResult:
        """The refusal for a pre-push gate that never reached a verdict.

        No hook reported a verdict -- the gate did not complete (timeout,
        tooling error). Reporting that as "Pre-push gate failed (pre-push
        gate)" made a 600s HEAVY-tier timeout look identical to a real
        hook failure, which is how agent-utilities appeared to be blocked
        on merit when it had simply run out of clock. Surface the harness
        error verbatim instead of inventing a hook name.
        """
        logger.error("Pre-push gate did not complete: %s", error)
        return GitResult(
            status="error",
            data="",
            error=GitError(
                message=f"Pre-push gate did not complete; push aborted. {error}",
                code=1,
            ),
        )

    @staticmethod
    def _gate_unrunnable_result(unrunnable: list[str]) -> GitResult:
        """The refusal for a gate whose every failing hook is simply missing.

        A gate that could not RUN is not a gate that found a defect. Saying
        "fix the gate" would send the reader hunting for a defect that does
        not exist. The push is still refused -- an ungated push is worse --
        but the reason is reported truthfully so it can be acted on.
        """
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

    @staticmethod
    def _gate_failed_result(failed: list[str]) -> GitResult:
        """The refusal for a gate that genuinely found a defect."""
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

    @staticmethod
    def _pre_push_gate_refusal(result: Any) -> GitResult:
        """Explain WHY the pre-push gate refused this push."""
        failed = [h.hook_id for h in result.hooks if not h.passed]
        if not failed and result.error:
            return Git._gate_incomplete_result(result.error)
        unrunnable = [h.hook_id for h in result.hooks if not h.passed and h.unrunnable]
        if unrunnable and len(unrunnable) == len(failed):
            return Git._gate_unrunnable_result(unrunnable)
        return Git._gate_failed_result(failed)

    def _gate_before_push(
        self,
        target_path: str,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> GitResult | None:
        """Run the repo's declared HEAVY (pre-push-stage) gates before pushing.

        Mirrors the repo's CI gates locally so a push can't ship a commit the CI
        would reject. Returns a failed ``GitResult`` (caller aborts the push) or
        ``None`` to proceed. No-op when disabled, when the repo has no
        ``.pre-commit-config.yaml``, or when there is nothing to push. The
        A gate-harness failure (tooling/env) blocks the push because an
        unverified release must not reach a remote.

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
        try:
            gate_path, has_precommit_config = self._push_gate_target(
                target_path, pinned
            )
            if not has_precommit_config:
                return None
            if not self._has_unpushed_commits(target_path, pinned=pinned):
                return None

            # Scope per-file hooks to the diff being pushed; always_run guardrail
            # gates still run fully. Falls back to --all-files if the diff is empty.
            changed = self._unpushed_changed_files(target_path, pinned=pinned)
            scope = f"{len(changed)} changed file(s)" if changed else "all files"
            logger.info("Running pre-push (HEAVY) gate over %s", scope)
            result = self._run_push_gate(gate_path, changed, pinned)
        except OperationBoundaryError as exc:
            return self._path_validation_result("push_project", target_path, exc)
        except Exception as exc:  # pragma: no cover - tooling/env failure
            logger.error(
                "Pre-push gate did not complete: error_type=%s", type(exc).__name__
            )
            return self._gate_incomplete_result(type(exc).__name__)

        if result.success:
            return None
        return self._pre_push_gate_refusal(result)

    @staticmethod
    def _push_gate_target(
        target_path: str, pinned: PinnedDirectory | None
    ) -> tuple[str, bool]:
        """Return the gate path and config presence through the active boundary."""
        if pinned is not None:
            pinned.assert_operation_identity()
            return pinned.proc_path, read_at(
                pinned.fd, ".pre-commit-config.yaml"
            ) is not None
        return target_path, os.path.exists(
            os.path.join(target_path, ".pre-commit-config.yaml")
        )

    def _run_push_gate(
        self,
        gate_path: str,
        changed: list[str],
        pinned: PinnedDirectory | None,
    ) -> Any:
        """Run the heavy gate and retain its descriptor boundary."""
        result = run_gate_stage(
            gate_path,
            "heavy",
            files=changed or None,
            trigger="pre-push",
            colocated=True,
        )
        if pinned is not None:
            pinned.assert_operation_identity()
        return result

    @staticmethod
    def _dirty_push_refusal(target_path: str) -> GitResult:
        """Refuse to push a repository with uncommitted changes."""
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

    @staticmethod
    def _secret_scanning_refusal(target_path: str) -> GitResult:
        """GitHub push protection (GH013) -- unrecoverable without manual action."""
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
                command="git push --atomic <authorized-destination> <exact-refs>",
                workspace=_project_label(target_path),
                return_code=1,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    @staticmethod
    def _diverged_push_refusal(result: GitResult) -> GitResult:
        """A divergent remote requires an explicit reviewed sync.

        Never rewrite remote history or mutate the local branch as a fallback.
        """
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

    @staticmethod
    def _push_error_text(result: GitResult) -> str:
        """The combined error message and output of a failed push."""
        error_text = ""
        if result.error:
            error_text = (
                str(result.error.message)
                if hasattr(result.error, "message")
                else str(result.error)
            )
        if result.data:
            error_text += " " + result.data
        return error_text

    def _pinned_git_value(
        self, target_path: str, pinned: PinnedDirectory, command: str
    ) -> str:
        """Read one required Git value through the admitted source descriptor."""
        result = self.git_action(
            command=command,
            path=target_path,
            quiet=True,
            raw_output=True,
            **self._pinned_git_kwargs(pinned),
        )
        value = (result.data or "").strip()
        if result.status != "success" or not value:
            raise OperationBoundaryError("cannot resolve exact ref for sealed push")
        return value

    @staticmethod
    def _validate_sealed_ref(ref: str, prefix: str) -> str:
        """Refuse names whose spelling could alter a refspec or command shape."""
        forbidden = ("..", "@{", "\\", "~", "^", ":", "?", "*", "[")
        if (
            not ref.startswith(prefix)
            or ref.endswith(("/", "."))
            or "//" in ref
            or any(token in ref for token in forbidden)
            or any(ord(char) <= 0x20 or ord(char) == 0x7F for char in ref)
        ):
            raise OperationBoundaryError("sealed push ref is unsafe")
        return ref

    @staticmethod
    def _validate_object_id(value: str) -> str:
        """Require an unabbreviated SHA-1 or SHA-256 Git object identity."""
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", value) is None:
            raise OperationBoundaryError("sealed push object identity is invalid")
        return value

    def _sealed_push_refs(
        self, target_path: str, pinned: PinnedDirectory
    ) -> _SealedPushRefs:
        """Snapshot the exact current branch and optional current release tag."""
        branch = self._validate_sealed_ref(
            self._pinned_git_value(
                target_path, pinned, "git symbolic-ref --quiet HEAD"
            ),
            "refs/heads/",
        )
        head_oid = self._validate_object_id(
            self._pinned_git_value(
                target_path, pinned, "git rev-parse --verify HEAD^{commit}"
            )
        )
        release_tag = self._current_release_tag(target_path, pinned=pinned)
        if release_tag is None:
            return _SealedPushRefs(branch=branch, head_oid=head_oid)
        tag = self._validate_sealed_ref(f"refs/tags/{release_tag}", "refs/tags/")
        tag_oid = self._validate_object_id(
            self._pinned_git_value(
                target_path, pinned, f"git rev-parse --verify {shlex.quote(tag)}"
            )
        )
        tag_commit = self._validate_object_id(
            self._pinned_git_value(
                target_path,
                pinned,
                f"git rev-parse --verify {shlex.quote(tag)}^{{commit}}",
            )
        )
        if tag_commit != head_oid:
            raise OperationBoundaryError("current release tag does not name HEAD")
        return _SealedPushRefs(branch, head_oid, tag, tag_oid)

    def _sealed_push_destination(
        self, target_path: str, pinned: PinnedDirectory
    ) -> str:
        """Resolve one authorized endpoint without consulting a mutation child."""
        if pinned.expected_origin is not None:
            return self._authorized_push_destination(pinned)
        return self._configured_push_destination(target_path, pinned)

    @staticmethod
    def _authorized_push_destination(pinned: PinnedDirectory) -> str:
        """Require the local fetch origin to match a release-plan destination."""
        expected = canonical_repository_url(pinned.expected_origin)
        if pinned.configured_origin is None:
            return expected
        configured = canonical_repository_url(pinned.configured_origin)
        if configured != expected:
            raise OperationBoundaryError(
                "configured origin differs from the authorized destination"
            )
        return expected

    @staticmethod
    def _configured_push_destination(target_path: str, pinned: PinnedDirectory) -> str:
        """Normalize a generic push's sole admitted local origin."""
        configured = pinned.configured_origin
        if configured is None:
            raise OperationBoundaryError("push target has no admitted origin URL")
        if any(ord(char) < 0x20 or ord(char) == 0x7F for char in configured):
            raise OperationBoundaryError("push target contains control characters")
        parsed = urlsplit(configured)
        if parsed.scheme in {"http", "https"}:
            return canonical_repository_url(configured)
        if parsed.scheme == "file":
            return configured
        if parsed.scheme:
            raise OperationBoundaryError("push target uses an unsupported transport")
        return os.path.abspath(os.path.join(target_path, configured))

    @staticmethod
    def _sealed_git_environment() -> dict[str, str]:
        """Build an environment with no system, user, or injected Git config."""
        return PinnedDirectory.anchored_git_environment(os.environ.copy())

    def _run_sealed_git(
        self,
        admin: PinnedDirectory,
        argv: list[str],
        *,
        target_path: str,
        initialized: bool = True,
    ) -> GitResult:
        """Run Git only inside the private admin repository."""
        command = ["git"]
        if initialized:
            command.append(f"--git-dir=/proc/self/fd/{admin.fd}")
        command.extend(argv)
        return self._run_pinned_repository_command(
            command,
            self._sealed_git_environment(),
            target_path=target_path,
            operation=f"git {argv[0]}",
            quiet=True,
            timeout=1800,
            raw_output=True,
            cwd_fd=admin.fd,
            pass_fds=admin.pass_fds,
            path_anchor=None,
        )

    @staticmethod
    def _require_sealed_success(result: GitResult, label: str) -> None:
        """Translate an internal admin failure into a fail-closed boundary error."""
        if result.status != "success":
            raise OperationBoundaryError(f"sealed release admin {label} failed")

    def _export_sealed_bundle(
        self,
        target_path: str,
        pinned: PinnedDirectory,
        refs: _SealedPushRefs,
        bundle_path: Path,
    ) -> None:
        """Export only admitted refs; this source child performs no network I/O."""
        arguments = " ".join(shlex.quote(ref) for ref in refs.names)
        result = self.git_action(
            command=f"git bundle create {shlex.quote(str(bundle_path))} {arguments}",
            path=target_path,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        self._require_sealed_success(result, "bundle export")
        pinned.assert_operation_identity()

    def _populate_sealed_admin(
        self,
        admin: PinnedDirectory,
        bundle_path: Path,
        refs: _SealedPushRefs,
        target_path: str,
    ) -> None:
        """Initialize a bare admin repository and import only the admitted refs."""
        initialized = self._run_sealed_git(
            admin,
            ["init", "--bare", "--template=", "."],
            target_path=target_path,
            initialized=False,
        )
        self._require_sealed_success(initialized, "initialization")
        refspecs = [f"+{ref}:{ref}" for ref in refs.names]
        fetched = self._run_sealed_git(
            admin,
            ["fetch", "--no-tags", str(bundle_path), *refspecs],
            target_path=target_path,
        )
        self._require_sealed_success(fetched, "ref import")
        self._verify_sealed_admin_refs(admin, refs, target_path)

    def _verify_sealed_admin_refs(
        self, admin: PinnedDirectory, refs: _SealedPushRefs, target_path: str
    ) -> None:
        """Prove the private repository contains the exact expected object set."""
        listed = self._run_sealed_git(
            admin,
            ["for-each-ref", "--format=%(refname)"],
            target_path=target_path,
        )
        self._require_sealed_success(listed, "ref inventory")
        if set((listed.data or "").splitlines()) != set(refs.names):
            raise OperationBoundaryError("sealed release admin ref inventory changed")
        for ref, expected in refs.expected.items():
            result = self._run_sealed_git(
                admin,
                ["rev-parse", "--verify", ref],
                target_path=target_path,
            )
            self._require_sealed_success(result, "ref verification")
            actual = self._validate_object_id((result.data or "").strip())
            if actual != expected:
                raise OperationBoundaryError(
                    "sealed release admin ref identity changed"
                )

    def _verify_sealed_destination(
        self, admin: PinnedDirectory, destination: str, target_path: str
    ) -> None:
        """Require Git's own rewrite resolution to preserve the authorized URL."""
        result = self._run_sealed_git(
            admin,
            ["ls-remote", "--get-url", "--", destination],
            target_path=target_path,
        )
        self._require_sealed_success(result, "destination verification")
        if (result.data or "").strip() != destination:
            raise OperationBoundaryError("sealed release destination was rewritten")

    def _verify_remote_refs(
        self,
        admin: PinnedDirectory,
        destination: str,
        refs: _SealedPushRefs,
        target_path: str,
    ) -> bool:
        """Read back every published ref and compare its exact object identity."""
        result = self._run_sealed_git(
            admin,
            ["ls-remote", "--refs", "--", destination, *refs.names],
            target_path=target_path,
        )
        if result.status != "success":
            return False
        observed = {
            name: oid
            for line in (result.data or "").splitlines()
            if "\t" in line
            for oid, name in [line.split("\t", 1)]
        }
        return observed == refs.expected

    def _sealed_atomic_push(
        self,
        admin: PinnedDirectory,
        destination: str,
        refs: _SealedPushRefs,
        target_path: str,
    ) -> GitResult:
        """Atomically publish exact refs from the sealed private repository."""
        self._verify_sealed_destination(admin, destination, target_path)
        self._verify_sealed_admin_refs(admin, refs, target_path)
        refspecs = [f"{ref}:{ref}" for ref in refs.names]
        result = self._run_sealed_git(
            admin,
            ["push", "--porcelain", "--atomic", "--", destination, *refspecs],
            target_path=target_path,
        )
        self._verify_sealed_admin_refs(admin, refs, target_path)
        if result.status == "success" and not self._verify_remote_refs(
            admin, destination, refs, target_path
        ):
            return GitResult(
                status="error",
                data="Remote publication could not be verified",
                error=GitError(message="remote ref verification failed", code=1),
                metadata=result.metadata,
            )
        return result

    def _push_from_sealed_admin(
        self, target_path: str, pinned: PinnedDirectory
    ) -> GitResult:
        """Publish without exposing source repository config to the push child."""
        refs = self._sealed_push_refs(target_path, pinned)
        destination = self._sealed_push_destination(target_path, pinned)
        with tempfile.TemporaryDirectory(prefix="repository-manager-release-") as temp:
            temp_path = Path(temp)
            bundle_path = temp_path / "release.bundle"
            admin_path = temp_path / "admin.git"
            admin_path.mkdir(mode=0o700)
            self._export_sealed_bundle(target_path, pinned, refs, bundle_path)
            with open_directory(admin_path) as admin:
                self._populate_sealed_admin(admin, bundle_path, refs, target_path)
                return self._sealed_atomic_push(admin, destination, refs, target_path)

    def _handle_push_failure(
        self,
        target_path: str,
        result: GitResult,
    ) -> GitResult:
        """Translate a failed ``git push`` into an actionable result."""
        error_text = self._push_error_text(result)

        # GitHub secret scanning block (GH013) — unrecoverable without manual action
        if "GH013" in error_text or "GITHUB PUSH PROTECTION" in error_text:
            logger.error(
                "GitHub secret scanning blocked the push; remove the secret from history"
            )
            return self._secret_scanning_refusal(target_path)

        if (
            "non-fast-forward" in error_text
            or "tip of your current branch is behind" in error_text
            or "[rejected] (fetch first)" in error_text
        ):
            logger.warning("Push refused because the remote branch has diverged")
            return self._diverged_push_refusal(result)

        # Any other failure, including a tag collision, preserves atomic refusal.
        return result

    def _push_preconditions(
        self, target_path: str, pinned: PinnedDirectory, git_kwargs: dict[str, Any]
    ) -> GitResult | None:
        """Verify status and all gates before allowing a remote mutation."""
        status_check = self.git_action(
            command="git status --porcelain",
            path=target_path,
            quiet=True,
            **git_kwargs,
        )
        if status_check.status != "success":
            logger.error("Push refused because repository status could not be verified")
            return status_check
        if status_check.data.strip():
            logger.warning("Push refused because the configured project is dirty")
            return self._dirty_push_refusal(target_path)
        return self._gate_before_push(target_path, pinned=pinned)

    def _push_project_with_handle(
        self, target_path: str, pinned: PinnedDirectory
    ) -> GitResult:
        """Run one push while retaining its descriptor-pinned checkout."""
        pinned.assert_path_identity()
        git_kwargs = self._pinned_git_kwargs(pinned)
        logger.info("Checking configured project for uncommitted changes")
        precondition = self._push_preconditions(target_path, pinned, git_kwargs)
        if precondition is not None:
            return precondition

        logger.info("Pushing exact refs from a sealed release admin repository")
        pinned.assert_operation_identity()
        result = self._sealed_push_result(target_path, pinned)
        return (
            result
            if result.status == "success"
            else self._handle_push_failure(target_path, result)
        )

    def _sealed_push_result(
        self, target_path: str, pinned: PinnedDirectory
    ) -> GitResult:
        """Run a sealed push and translate boundary drift to a typed refusal."""
        try:
            result = self._push_from_sealed_admin(target_path, pinned)
            pinned.assert_operation_identity()
            return result
        except OperationBoundaryError as exc:
            return self._path_validation_result("push_project", target_path, exc)

    @_exclusive_repo_mutation
    def push_project(
        self,
        path: str | None = None,
        *,
        _pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Atomically publish a clean checkout's exact current branch and release tag.

        Publication runs from a sealed private admin repository, uses only the
        authorized direct destination, and reads every remote ref back before
        reporting success. Dirty state, gate failure, identity/config drift,
        non-fast-forward updates, tag collisions, and partial atomic updates all
        refuse publication; GitHub push protection returns an actionable error.
        """
        target_path = self._validated_operation_path(path, operation="push_project")
        try:
            if _pinned is not None:
                _pinned.assert_path_identity()
                if Path(target_path) != _pinned.path:
                    raise OperationBoundaryError(
                        "pinned push target does not match the requested path"
                    )
                return self._push_project_with_handle(target_path, _pinned)
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, target_path) as pinned:
                    return self._push_project_with_handle(target_path, pinned)
        except OperationBoundaryError as exc:
            return self._path_validation_result("push_project", target_path, exc)

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
    def add_project(
        self,
        path: str | None = None,
        *,
        _pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Stage all changes (git add -A) for a single Git project.
        """
        target_path = self._validated_operation_path(path, operation="add_project")
        return self._add_project_operation(target_path, _pinned)

    def _add_project_operation(
        self, target_path: str, pinned: PinnedDirectory | None
    ) -> GitResult:
        """Stage one project after validating an optional pinned handle."""
        try:
            self._assert_optional_pinned_target(
                target_path, pinned, operation="add_project"
            )
        except OperationBoundaryError as exc:
            return self._path_validation_result("add_project", target_path, exc)
        logger.info("Staging all changes for configured project")
        return self.git_action(
            command="git add -A",
            path=target_path,
            **self._pinned_git_kwargs(pinned),
        )

    @staticmethod
    def _assert_optional_pinned_target(
        target_path: str,
        pinned: PinnedDirectory | None,
        *,
        operation: str,
    ) -> None:
        """Ensure an optional handle still denotes the requested operation path."""
        if pinned is None:
            return
        pinned.assert_path_identity()
        if Path(target_path) != pinned.path:
            raise OperationBoundaryError(
                f"pinned {operation} target does not match the requested path"
            )

    def commit_projects(
        self,
        message: str,
        project_dirs: list[str] | None = None,
        *,
        _pinned_targets: dict[str, PinnedDirectory] | None = None,
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

        commit_func: Callable[[str], GitResult] = partial(
            self._commit_project_dispatch,
            message,
            pinned_targets=_pinned_targets,
        )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            return list(executor.map(commit_func, project_dirs))

    def _commit_project_dispatch(
        self,
        message: str,
        project_path: str,
        *,
        pinned_targets: dict[str, PinnedDirectory] | None,
    ) -> GitResult:
        """Commit one path using its frozen handle when a release owns it."""
        pinned = (
            pinned_targets.get(project_path) if pinned_targets is not None else None
        )
        return self.commit_project(message, project_path, _pinned=pinned)

    def _commit_project_with_handle(
        self,
        message: str,
        target_path: str,
        pinned: PinnedDirectory,
    ) -> GitResult:
        """Commit staged changes while retaining the pinned checkout."""
        pinned.assert_path_identity()

        # Check if there are staged changes to commit
        status_res = self.git_action(
            command="git status --porcelain",
            path=target_path,
            **self._pinned_git_kwargs(pinned),
        )
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
        pinned.assert_path_identity()
        return self.git_action(
            command=f"git commit -m {safe_msg}",
            path=target_path,
            **self._pinned_git_kwargs(pinned),
        )

    def _commit_project_unpinned(self, message: str, target_path: str) -> GitResult:
        """Preserve the legacy missing-directory seam for mocked transports.

        The real ``git_action`` still performs its own no-follow descriptor
        admission.  This fallback exists only when the target cannot be pinned
        (for example a unit-test transport that does not materialize clones),
        so it does not create a new filesystem mutation path.
        """
        status_res = self.git_action(
            command="git status --porcelain",
            path=target_path,
        )
        if status_res.status == "success":
            has_staged = any(
                line and not line.startswith("?") and line[0] not in (" ", "?")
                for line in status_res.data.splitlines()
            )
            if not has_staged:
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
        from shlex import quote

        return self.git_action(
            command=f"git commit -m {quote(message)}",
            path=target_path,
        )

    @_exclusive_repo_mutation
    def commit_project(
        self,
        message: str,
        path: str | None = None,
        *,
        _pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Commit staged changes (git commit -m "{message}") for a single Git project.
        """
        target_path = self._validated_operation_path(path, operation="commit_project")
        try:
            if _pinned is not None:
                _pinned.assert_path_identity()
                if Path(target_path) != _pinned.path:
                    raise OperationBoundaryError(
                        "pinned commit target does not match the requested path"
                    )
                return self._commit_project_with_handle(
                    message,
                    target_path,
                    _pinned,
                )
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, target_path) as pinned:
                    return self._commit_project_with_handle(
                        message,
                        target_path,
                        pinned,
                    )
        except OperationBoundaryError as exc:
            if not Path(target_path).is_dir():
                return self._commit_project_unpinned(message, target_path)
            return self._path_validation_result("commit_project", target_path, exc)

    @staticmethod
    def _commit_code_skip(target_path: str, reason: str) -> GitResult:
        """A no-op commit_code outcome (missing clone, or nothing to commit)."""
        return GitResult(
            status="skipped",
            data=reason,
            error=None,
            metadata=GitMetadata(
                command="commit_code",
                workspace=_project_label(target_path),
                return_code=0,
                timestamp=datetime.datetime.now(datetime.UTC).isoformat() + "Z",
            ),
        )

    def _stamp_commit_sha(self, result: GitResult, target_path: str) -> GitResult:
        """Append the resulting commit SHA to a successful commit result.

        D-CDX-60 acceptance: a truthful commit_code result names BOTH the
        resolved repository/worktree it acted on (already carried by
        metadata.workspace) AND the resulting commit SHA, so a caller can
        verify what actually happened rather than trust a bare "success".
        """
        sha_res = self.git_action(
            command="git rev-parse HEAD", path=target_path, quiet=True
        )
        if sha_res.status != "success" or not sha_res.data.strip():
            return result
        return result.model_copy(
            update={"data": f"{result.data}\ncommit_sha={sha_res.data.strip()}"}
        )

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
        target_path = self._validated_operation_path(
            path, operation="commit_code_project"
        )

        # Un-cloned / missing repo (e.g. a workspace.yml entry not pulled): skip
        # gracefully — a missing dir must never abort the whole batch. D-CDX-60:
        # ``.git`` is a FILE (a gitdir pointer), not a directory, in a linked
        # worktree — an `isdir` check here wrongly reported a valid isolated
        # worktree as "not a cloned Git repository" and skipped it silently.
        # ``os.path.exists`` accepts either shape, matching the check at
        # `_resolve_path`'s sibling validation above.
        if not os.path.exists(os.path.join(target_path, ".git")):
            return self._commit_code_skip(
                target_path, "Configured project is not a cloned Git repository"
            )

        status_res = self.git_action(
            command="git status --porcelain", path=target_path, quiet=True
        )
        if status_res.status == "success" and not status_res.data.strip():
            return self._commit_code_skip(target_path, "No changes to commit.")

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
            result = self._stamp_commit_sha(result, target_path)
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
        uses the gate engine's stage-aware environment without injecting the
        heavy pytest/Cargo/Tokio limits into this fast pre-commit workflow.
        """
        env = precommit_gate_environment("pre-commit")
        lane_pytest_options = env.get("PYTEST_ADDOPTS", "").strip()
        bounded_pytest_options = '-q --tb=short -m "not slow" --timeout=60'
        env["PYTEST_ADDOPTS"] = " ".join(
            option for option in (lane_pytest_options, bounded_pytest_options) if option
        )
        return env

    def _run_precommit_autoupdate(
        self,
        target_path: str,
        env: dict,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        """``pre-commit autoupdate``, then stage whatever it rewrote.

        Returns the first error encountered, or the autoupdate result itself.
        """
        result = self.git_action(
            command="pre-commit autoupdate",
            path=target_path,
            env=env,
            timeout=600,
            **self._pinned_git_kwargs(pinned),
        )
        if result.status == "error":
            return result
        staged = self.git_action(
            command="git add -A",
            path=target_path,
            **self._pinned_git_kwargs(pinned),
        )
        if staged.status == "error":
            return staged
        return result

    def _run_precommit_hooks(
        self,
        target_path: str,
        env: dict,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> tuple[GitResult, bool]:
        """Stage, run the FAST-tier hooks, and retry once after reformatting.

        Returns ``(result, is_final)``; ``is_final`` marks a *staging* failure,
        which the caller must surface verbatim without the branch-lock
        post-processing that applies to hook results.
        """
        staged = self.git_action(
            command="git add -A",
            path=target_path,
            **self._pinned_git_kwargs(pinned),
        )
        if staged.status == "error":
            return staged, True

        # Explicit, not pre-commit's implicit default: this method is the
        # FAST tier's interactive stage-and-commit helper (CONCEPT
        # GOC-59/60 two-tier gate model) -- declare the stage it runs
        # rather than relying on `pre-commit run` defaulting to it.
        hook_stage = HOOK_STAGE_BY_GATE_STAGE["fast"]
        hook_command = f"pre-commit run --hook-stage {hook_stage} --all-files --verbose"
        result = self.git_action(
            command=hook_command,
            path=target_path,
            env=env,
            timeout=600,
            **self._pinned_git_kwargs(pinned),
        )
        if result.status == "error":
            # Hooks may have reformatted files. Stage those bounded changes
            # and run once more, without a shell retry expression.
            restaged = self.git_action(
                command="git add -A",
                path=target_path,
                **self._pinned_git_kwargs(pinned),
            )
            if restaged.status == "error":
                return restaged, True
            result = self.git_action(
                command=hook_command,
                path=target_path,
                env=env,
                timeout=600,
                **self._pinned_git_kwargs(pinned),
            )

        self.git_action(
            command="git add -A",
            path=target_path,
            **self._pinned_git_kwargs(pinned),
        )
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

    def _pre_commit_with_handle(
        self,
        *,
        target_path: str,
        run: bool,
        autoupdate: bool,
        pinned: PinnedDirectory,
    ) -> GitResult:
        """Run pre-commit while retaining the descriptor-pinned checkout."""
        pinned.assert_operation_identity()

        # Cleanup is a mutation too.  Keep the compatibility callback surface
        # while making the active proc-fd path resolve only to this pinned
        # handle; untrusted proc-fd strings are rejected by cleanup_artifacts.
        previous_cleanup = getattr(self, "_active_cleanup_handle", None)
        self._active_cleanup_handle = pinned
        try:
            self.cleanup_artifacts(pinned.proc_path)
        finally:
            self._active_cleanup_handle = previous_cleanup

        # ``read_at`` refuses a symlinked pre-commit configuration; allowing
        # one would let the hook runner execute an external file after the
        # target was admitted.
        if read_at(pinned.fd, ".pre-commit-config.yaml") is None:
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
            pinned.assert_operation_identity()
            result = self._run_precommit_autoupdate(
                target_path,
                env,
                pinned=pinned,
            )
            if result.status == "error":
                return result

        if run:
            pinned.assert_operation_identity()
            result, is_final = self._run_precommit_hooks(
                target_path,
                env,
                pinned=pinned,
            )
            if is_final:
                return result

        if result is None:
            raise RuntimeError("pre-commit operation produced no result")

        pinned.assert_path_identity()
        if self._is_branch_lock_only_failure(result):
            logger.info(
                f"Ignoring safe pre-commit failure (branch lock) in {target_path}"
            )
            return self._branch_lock_success(result)

        return result

    @_exclusive_repo_mutation
    def pre_commit(
        self,
        run: bool = True,
        autoupdate: bool = False,
        path: str | None = None,
        *,
        _pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        """
        Execute pre-commit commands in the specified path.

        Args:
            run (bool): Whether to run 'pre-commit run --all-files'. Default True.
            autoupdate (bool): Whether to run 'pre-commit autoupdate'. Default False.
            path (str, optional): Path to run in. Defaults to self.path.
        """
        target_path = self._validated_operation_path(path, operation="pre_commit")
        try:
            if _pinned is not None:
                _pinned.assert_path_identity()
                if Path(target_path) != _pinned.path:
                    raise OperationBoundaryError(
                        "pinned pre-commit target does not match the requested path"
                    )
                return self._pre_commit_with_handle(
                    target_path=target_path,
                    run=run,
                    autoupdate=autoupdate,
                    pinned=_pinned,
                )

            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, target_path) as pinned:
                    return self._pre_commit_with_handle(
                        target_path=target_path,
                        run=run,
                        autoupdate=autoupdate,
                        pinned=pinned,
                    )
        except OperationBoundaryError as exc:
            return self._path_validation_result("pre_commit", target_path, exc)

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
    def _require_test_target(test_target: str | None) -> str:
        """Return a planned test target or reject an incomplete plan."""
        if test_target is None:
            raise RuntimeError(
                "project test plan returned no target without a skip reason"
            )
        return test_target

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
                test_target = self._require_test_target(test_target)

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
            p_path = self._validated_precommit_path(p, p_path)
            if (
                p_path
                and os.path.isdir(p_path)
                and os.path.exists(os.path.join(p_path, ".pre-commit-config.yaml"))
            ):
                dirs.append(p_path)
        return dirs

    def _validated_precommit_path(
        self, project_name: str, project_path: str | None
    ) -> str | None:
        """Return one validated pre-commit path, preserving missing-name skips."""
        if project_path is None:
            return None
        return str(
            self._validate_workspace_path(
                project_path,
                label=f"pre-commit project {project_name!r}",
            )
        )

    def _validated_precommit_dir(self, project_path: str) -> str | None:
        """Return a mapped directory only when it carries pre-commit config."""
        validated = self._validate_workspace_path(
            project_path,
            label="pre-commit project",
        )
        if (
            not validated.is_dir()
            or not (validated / ".pre-commit-config.yaml").exists()
        ):
            return None
        return str(validated)

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
            validated
            for p in self.project_map.values()
            if (validated := self._validated_precommit_dir(p)) is not None
        ]

    def _run_precommit_pool(
        self,
        project_dirs: list[str],
        run: bool,
        autoupdate: bool,
        *,
        pinned_targets: dict[str, PinnedDirectory] | None = None,
    ) -> list[GitResult]:
        """Run pre-commit across *project_dirs* in parallel."""
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.threads
        ) as executor:
            futures = {
                self._submit_precommit_future(
                    executor,
                    run,
                    autoupdate,
                    d,
                    pinned_targets,
                ): d
                for d in project_dirs
            }
            return [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

    def _submit_precommit_future(
        self,
        executor: concurrent.futures.ThreadPoolExecutor,
        run: bool,
        autoupdate: bool,
        project_dir: str,
        pinned_targets: dict[str, PinnedDirectory] | None,
    ) -> "concurrent.futures.Future":
        """Submit one pre-commit run with its optional pinned target."""
        if pinned_targets is None:
            return executor.submit(self.pre_commit, run, autoupdate, project_dir)
        return executor.submit(
            self.pre_commit,
            run,
            autoupdate,
            project_dir,
            _pinned=pinned_targets[project_dir],
        )

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
        *,
        _pinned_targets: dict[str, PinnedDirectory] | None = None,
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

            return self._run_precommit_pool(
                project_dirs,
                run,
                autoupdate,
                pinned_targets=_pinned_targets,
            )

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

    def _bump_skip_reason(
        self,
        project_dir: str,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> str | None:
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
        status_check = self.git_action(
            "git status",
            path=project_dir,
            **self._pinned_git_kwargs(pinned),
        )
        data_lower = status_check.data.lower() if status_check.data else ""
        clean = "nothing to commit" in data_lower
        up_to_date = "your branch is up to date" in data_lower
        if clean and up_to_date:
            return "no code changes detected (use force=True to override)"
        if clean and not up_to_date:
            head_subj = self.git_action(
                "git log -1 --pretty=%s",
                path=project_dir,
                quiet=True,
                **self._pinned_git_kwargs(pinned),
            )
            subject = (head_subj.data or "").strip().lower()
            if subject.startswith("bump version:"):
                return "already bumped, awaiting push (avoids double-bump)"
        return None

    def _repo_has_pending_work(
        self,
        project_dir: str,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> bool:
        """True when a repo has anything to bump or push.

        A repo that is both clean and in sync with origin has no uncommitted
        changes, no unpushed feature commits, and no unpushed version bump — so
        neither the bump nor the push step would act on it. This is the same
        no-op test :meth:`_bump_skip_reason` uses; anything else (dirty tree,
        ahead of origin) is treated as pending work.
        """
        status_check = self.git_action(
            "git status",
            path=project_dir,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        data_lower = status_check.data.lower() if status_check.data else ""
        clean = "nothing to commit" in data_lower
        up_to_date = "your branch is up to date" in data_lower
        return not (clean and up_to_date)

    def _auto_start_phase(
        self, config: dict, *, operation: Literal["bump", "push"]
    ) -> int | None:
        """Lowest phase number that contains a repo with pending work.

        Phases are topologically ordered (lower phase = more upstream): a change
        in phase *N* can only cascade to phases ``>= N`` via dependency-pin
        propagation, never to an earlier phase. So the bump/push can safely begin
        at the lowest phase that actually has work and still capture every
        downstream effect, skipping purely-unaffected upstream phases (and their
        inter-phase waits). Returns ``None`` when no repo has pending work — the
        caller should then do nothing. (CONCEPT:RM-PHASE-START)
        """
        claimed: set[str] = set()
        for phase in self._ordered_release_phases(config):
            targets = self._select_phase_targets(
                phase,
                filter_set=None,
                claimed=claimed,
                include_bulk=bool(phase.get(f"bulk_{operation}")),
            )
            if self._phase_has_pending_work(targets):
                return phase["phase"]
        return None

    def _phase_has_pending_work(self, targets: list[_ReleaseTarget]) -> bool:
        """Whether any selected project has a local clone with pending work."""
        return any(
            self._phase_target_has_pending_work(name, path) for name, path in targets
        )

    def _phase_target_has_pending_work(self, name: str, path: str) -> bool:
        """Check one canonical target for work without widening its scope."""
        return self._phase_target_pending_probe(name, path)

    def _phase_target_pending_probe(self, name: str, path: str) -> bool:
        """Run the descriptor-pinned pending-work probe for one target."""
        try:
            _, validated_path = self._revalidate_release_target(
                name,
                path,
                operation="auto-start",
            )
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, validated_path) as pinned:
                    pinned.assert_path_identity()
                    pending_method = self._repo_has_pending_work
                    pending = self._pending_work_probe(
                        pending_method, validated_path, pinned
                    )
                    pinned.assert_path_identity()
                    return pending
        except (OperationBoundaryError, ValueError) as exc:
            logger.warning(
                "Auto-start refused unsafe target %s: %s",
                name,
                type(exc).__name__,
            )
            return False

    @staticmethod
    def _pending_work_probe(
        pending_method: Callable[..., bool],
        validated_path: str,
        pinned: PinnedDirectory,
    ) -> bool:
        """Invoke an injected pending-work probe while preserving pin support."""
        if "pinned" in inspect.signature(pending_method).parameters:
            return pending_method(validated_path, pinned=pinned)
        # Keep the small private seam compatible with callers that inject a
        # read-only pending-work probe; the real implementation remains pinned.
        return pending_method(validated_path)

    def _bump_version_with_handle(
        self,
        *,
        target_dir: str,
        part: str,
        allow_dirty: bool,
        dry_run: bool,
        verbose: bool,
        force: bool,
        pinned: PinnedDirectory,
    ) -> GitResult:
        """Run one version bump while retaining its pinned checkout handle."""
        pinned.assert_path_identity()
        validation_error = self._bump_version_validate_target(target_dir, part)
        if validation_error is not None:
            return validation_error

        if not self._project_has_bumpversion_config(target_dir, pinned=pinned):
            return self._bump_version_fallback(
                target_dir,
                dry_run,
                pinned=pinned,
            )

        command = self._build_bump2version_command(part, allow_dirty, dry_run, verbose)

        if not dry_run:
            preflight_result = self._bump_version_preflight_tag_check(
                target_dir,
                part,
                allow_dirty,
                force,
                pinned=pinned,
            )
            if preflight_result is not None:
                return preflight_result
            command += " --list"

        pinned.assert_path_identity()
        return self._run_bump2version(
            command,
            target_dir,
            part,
            dry_run,
            pinned=pinned,
        )

    @_exclusive_repo_mutation
    def bump_version(
        self,
        part: str,
        allow_dirty: bool = False,
        path: str | None = None,
        dry_run: bool = False,
        verbose: bool = False,
        force: bool = False,
        *,
        _pinned: PinnedDirectory | None = None,
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
        target_dir = self._validated_operation_path(path, operation="bump_version")
        try:
            if _pinned is not None:
                _pinned.assert_path_identity()
                if Path(target_dir) != _pinned.path:
                    raise OperationBoundaryError(
                        "pinned bump target does not match the requested path"
                    )
                return self._bump_version_with_handle(
                    target_dir=target_dir,
                    part=part,
                    allow_dirty=allow_dirty,
                    dry_run=dry_run,
                    verbose=verbose,
                    force=force,
                    pinned=_pinned,
                )

            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, target_dir) as pinned:
                    return self._bump_version_with_handle(
                        target_dir=target_dir,
                        part=part,
                        allow_dirty=allow_dirty,
                        dry_run=dry_run,
                        verbose=verbose,
                        force=force,
                        pinned=pinned,
                    )
        except OperationBoundaryError as exc:
            return self._path_validation_result("bump_version", target_dir, exc)

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

    def _project_has_bumpversion_config(
        self,
        target_dir: str,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> bool:
        """Whether *target_dir* declares a bump2version configuration
        (``.bumpversion.cfg``, or a ``[bumpversion]`` section in
        ``setup.cfg``)."""
        reader = self._bumpversion_config_reader(target_dir, pinned)
        cfg = reader(".bumpversion.cfg")
        if cfg is not None:
            return True
        setup = reader("setup.cfg")
        if setup is None:
            return False
        try:
            return b"[bumpversion]" in setup
        except TypeError:
            return False

    @staticmethod
    def _bumpversion_config_reader(
        target_dir: str, pinned: PinnedDirectory | None
    ) -> Callable[[str], bytes | None]:
        """Return a descriptor-relative or lexical config reader."""
        if pinned is not None:
            return lambda name: read_at(pinned.fd, name)

        def read_config(name: str) -> bytes | None:
            config_path = Path(target_dir) / name
            if not config_path.exists() or not config_path.is_file():
                return None
            return config_path.read_bytes()

        return read_config

    def _bump_version_fallback(
        self,
        target_dir: str,
        dry_run: bool,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        """Fallback behavior for a project with no bump2version config: stage
        all changes and commit them as "phased bump"."""
        status_check = self.git_action(
            command="git status --porcelain",
            path=target_dir,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
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

        add_res = self.git_action(
            command="git add -u",
            path=target_dir,
            **self._pinned_git_kwargs(pinned),
        )
        if add_res.status != "success":
            logger.error("Failed to add changes for configured project")
            return add_res

        commit_res = self.git_action(
            command='git commit -m "phased bump"',
            path=target_dir,
            **self._pinned_git_kwargs(pinned),
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
        self,
        target_dir: str,
        part: str,
        allow_dirty: bool,
        force: bool,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> GitResult | None:
        """Pre-flight check for an existing tag on the version bump2version
        would produce. Returns a GitResult to short-circuit `bump_version`
        (tag exists and is not force-deletable), or None to proceed with the
        real bump2version invocation (including after deleting an orphan
        local tag under ``force``)."""
        pre_cmd = f"bump2version {part} --dry-run --list"
        if allow_dirty:
            pre_cmd += " --allow-dirty"
        pre_result = self.git_action(
            command=pre_cmd,
            path=target_dir,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
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
            **self._pinned_git_kwargs(pinned),
        )
        if not (tag_check.status == "success" and f"v{new_version}" in tag_check.data):
            return None

        if force and not self._tag_on_remote(
            f"v{new_version}",
            target_dir,
            pinned=pinned,
        ):
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
                **self._pinned_git_kwargs(pinned),
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
        self,
        command: str,
        target_dir: str,
        part: str,
        dry_run: bool,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> GitResult:
        return self._run_bump2version_operation(
            command, target_dir, part, dry_run, pinned
        )

    def _run_bump2version_operation(
        self,
        command: str,
        target_dir: str,
        part: str,
        dry_run: bool,
        pinned: PinnedDirectory | None,
    ) -> GitResult:
        """Run bump2version and translate ordinary failures to a result."""
        try:
            result = self._run_bump2version_command(command, target_dir, pinned)

            if result.status == "success":
                logger.info("Bumped configured project version: part=%s", part)

                if not dry_run:
                    # bump2version commits and advances HEAD as part of its
                    # successful command.  The immutable plan was checked
                    # immediately before that mutation; subsequent cleanup
                    # commands remain descriptor-pinned but must not compare
                    # their now-intended HEAD against the pre-bump snapshot.
                    self._clear_bump_boundary(pinned)
                    self._finalize_successful_bump(
                        target_dir,
                        result,
                        pinned=pinned,
                    )
            else:
                logger.error("Failed to bump configured project version")

            return result
        except OperationBoundaryError:
            raise
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

    def _run_bump2version_command(
        self,
        command: str,
        target_dir: str,
        pinned: PinnedDirectory | None,
    ) -> GitResult:
        """Run bump2version after checking the retained checkout boundary."""
        if pinned is not None:
            pinned.assert_path_identity()
        return self.git_action(
            command=command,
            path=target_dir,
            **self._pinned_git_kwargs(pinned),
        )

    @staticmethod
    def _clear_bump_boundary(pinned: PinnedDirectory | None) -> None:
        """Allow post-bump cleanup after bump2version intentionally advances HEAD."""
        if pinned is not None:
            pinned.clear_boundary_assertion()

    def _finalize_successful_bump(
        self,
        target_dir: str,
        result: GitResult,
        *,
        pinned: PinnedDirectory | None = None,
    ) -> None:
        """Post-success sequence for a real bump2version run: sync uv.lock,
        stage everything, and -- IF there is anything staged -- fold it into
        the bump commit (``commit --amend``) and re-point the tag. Step order
        here is exactly the release-path stage/commit/tag sequence and must
        never be reordered (CX WC1-REPOSITORY-01: this fleet has a documented
        failure where bump2version stages everything and then fails to
        commit, leaving a half-applied bump with no tag)."""
        # Synchronize uv.lock after pyproject.toml version bump
        if self._has_uv_lock(target_dir, pinned):
            self.git_action(
                command="uv lock",
                path=target_dir,
                quiet=True,
                **self._pinned_git_kwargs(pinned),
            )

        # Stage all changes (staged and uncommitted/unstaged changes) in the workspace
        self.git_action(
            command="git add -u",
            path=target_dir,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        status_check = self.git_action(
            command="git status --porcelain",
            path=target_dir,
            quiet=True,
            **self._pinned_git_kwargs(pinned),
        )
        if status_check.data.strip():
            # Commit all staged changes (including version bump, uv.lock, and other files) into the bump commit
            self.git_action(
                command="SKIP=no-commit-to-branch,uv-lock,pytest,pnpm-build git commit --amend --no-edit",
                path=target_dir,
                quiet=True,
                **self._pinned_git_kwargs(pinned),
            )

            # Move the tag to point to the newly amended commit
            match = re.search(r"new_version=(.*)", result.data)
            if match:
                new_version = match.group(1).strip()
                self.git_action(
                    command=f"git tag -f v{new_version}",
                    path=target_dir,
                    quiet=True,
                    **self._pinned_git_kwargs(pinned),
                )

    @staticmethod
    def _has_uv_lock(target_dir: str, pinned: PinnedDirectory | None) -> bool:
        """Check for uv.lock without following a mutable path when pinned."""
        if pinned is not None:
            return read_at(pinned.fd, "uv.lock") is not None
        return os.path.exists(os.path.join(target_dir, "uv.lock"))

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
        self,
        file_path: str,
        package_name: str,
        new_version: str,
        dry_run: bool = False,
        *,
        _directory_fd: int | None = None,
        _file_name: str | None = None,
        _boundary_assertion: Callable[[], None] | None = None,
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
        file_name = _file_name or target_file.name
        content = self._read_dependency_content(target_file, _directory_fd, file_name)
        if content is None:
            return False
        pattern = (
            rf'(["\']?{re.escape(package_name)}(?:\[[^\]]*\])?\s*'
            r"(?:==|>=|<=|~=|!=|>|<)\s*)\d+\.\d+\.\d+"
        )
        replacement = rf"\g<1>{new_version}"

        new_content, count = re.subn(pattern, replacement, content)
        if count > 0:
            self._apply_dependency_update(
                target_file,
                _directory_fd,
                file_name,
                new_content,
                dry_run=dry_run,
                boundary_assertion=_boundary_assertion,
            )
            logger.info(
                f"{'[DRY RUN] Would update' if dry_run else 'Updated'} "
                f"{package_name} -> {new_version} ({count}x) in {target_file}"
            )
            return True
        return False

    def _apply_dependency_update(
        self,
        target_file: Path,
        directory_fd: int | None,
        file_name: str,
        content: str,
        *,
        dry_run: bool,
        boundary_assertion: Callable[[], None] | None,
    ) -> None:
        """Commit one dependency rewrite only after its active boundary check."""
        if dry_run:
            return
        if boundary_assertion is not None:
            boundary_assertion()
        self._write_dependency_content(target_file, directory_fd, file_name, content)

    @staticmethod
    def _read_dependency_content(
        target_file: Path, directory_fd: int | None, file_name: str
    ) -> str | None:
        """Read a dependency file through its pinned directory when available."""
        if directory_fd is not None:
            raw_content = read_at(directory_fd, file_name)
            if raw_content is None:
                return None
            try:
                return raw_content.decode("utf-8")
            except UnicodeDecodeError as exc:
                raise OperationBoundaryError(
                    f"dependency file {file_name!r} is not UTF-8"
                ) from exc
        if not target_file.exists() or not target_file.is_file():
            return None
        return target_file.read_text()

    @staticmethod
    def _write_dependency_content(
        target_file: Path,
        directory_fd: int | None,
        file_name: str,
        content: str,
    ) -> None:
        """Write a dependency file through its pinned directory when available."""
        if directory_fd is not None:
            write_at(directory_fd, file_name, content.encode())
            return
        target_file.write_text(content)

    @classmethod
    def _project_phase_index(cls, config: dict) -> tuple[dict[str, int], int]:
        """Map every declared project to its phase number, plus the bulk phase.

        Used for the topological dependency check: a bump must not be
        propagated backwards into a project owned by an earlier phase.
        """
        project_phases: dict[str, int] = {}
        bulk_phase_num = 5
        for phase in cls._ordered_release_phases(config):
            p_num = phase["phase"]
            if phase.get("bulk_bump"):
                bulk_phase_num = p_num
            for p in cls._phase_named_projects(phase):
                project_phases[p] = p_num
        return project_phases, bulk_phase_num

    def _pre_commit_project_targets(
        self,
        config: dict,
        filter_set: set[str] | None = None,
        *,
        start_phase: int = 1,
        single_phase: bool = False,
    ) -> list[_ReleaseTarget] | None:
        """Exact targets the pre-commit stage should cover, or ``None`` for all.

        Explicit phase members are always included. A ``bulk_bump`` phase adds
        only manifest-classified, PyPI-buildable agent repositories; it never
        expands pre-commit to unrelated infrastructure repositories.
        """
        if not config:
            return None
        phases = self._selected_release_phases(
            config, start_phase=start_phase, single_phase=single_phase
        )
        return self._pre_commit_targets_for_phases(phases, filter_set)

    def _pre_commit_targets_for_phases(
        self,
        phases: list[dict[str, Any]],
        filter_set: set[str] | None,
    ) -> list[_ReleaseTarget]:
        """Flatten exact targets for already-selected maintenance phases."""
        claimed: set[str] = set()
        targets: list[_ReleaseTarget] = []
        for phase in phases:
            targets.extend(
                self._select_phase_targets(
                    phase,
                    filter_set=filter_set,
                    claimed=claimed,
                    include_bulk=bool(phase.get("bulk_bump")),
                )
            )
        return targets

    def _pre_commit_target_dirs(
        self, targets: list[_ReleaseTarget] | None
    ) -> list[str]:
        """Validated local paths for the pre-commit stage's exact scope."""
        if targets is None:
            return self._all_precommit_target_dirs()
        return self._release_precommit_target_dirs(targets)

    def _all_precommit_target_dirs(self) -> list[str]:
        """Validate every mapped project path for an unrestricted pre-commit run."""
        return [
            str(self._validate_workspace_path(path, label="pre-commit project"))
            for path in self.project_map.values()
        ]

    def _release_precommit_target_dirs(
        self, targets: list[_ReleaseTarget]
    ) -> list[str]:
        """Revalidate the exact release pairs selected for pre-commit."""
        return [
            self._revalidate_release_target(name, path, operation="pre-commit")[1]
            for name, path in targets
        ]

    def _run_bump_pre_commit_stage(
        self,
        config: dict,
        filter_set: set[str] | None,
        *,
        start_phase: int,
        single_phase: bool,
        targets: list[_ReleaseTarget] | None = None,
        _secure: bool = False,
        provenance: _ReleasePlanProvenance | None = None,
        plan_assertion: Callable[[], None] | None = None,
    ) -> list[GitResult]:
        """Run pre-commit (with autoupdate) and commit the resulting formatting."""
        targets = self._resolve_bump_pre_commit_targets(
            config,
            filter_set,
            start_phase=start_phase,
            single_phase=single_phase,
            targets=targets,
        )
        project_dirs = self._pre_commit_target_dirs(targets)
        runner = {
            False: self._run_legacy_bump_pre_commit_stage,
            True: self._run_secure_bump_pre_commit_stage,
        }[_secure]
        return cast(Callable[..., list[GitResult]], runner)(
            project_dirs=project_dirs,
            targets=targets,
            provenance=provenance,
            plan_assertion=plan_assertion,
        )

    def _run_legacy_bump_pre_commit_stage(
        self,
        *,
        project_dirs: list[str],
        targets: list[_ReleaseTarget] | None,
        provenance: _ReleasePlanProvenance | None,
        plan_assertion: Callable[[], None] | None,
    ) -> list[GitResult]:
        """Run the compatibility pre-commit path without pinned handles."""
        del targets, provenance, plan_assertion
        results = list(
            self.pre_commit_projects(
                run=True,
                autoupdate=True,
                projects=project_dirs,
            )
        )
        results.extend(
            self.commit_projects(
                message="chore: pre-commit autoupdate and formatting",
                project_dirs=project_dirs,
            )
        )
        return results

    def _run_secure_bump_pre_commit_stage(
        self,
        *,
        project_dirs: list[str],
        targets: list[_ReleaseTarget] | None,
        provenance: _ReleasePlanProvenance | None,
        plan_assertion: Callable[[], None] | None,
    ) -> list[GitResult]:
        """Run pre-commit and commits while retaining exact target handles."""
        secure_results: list[GitResult] = []
        handles: list[PinnedDirectory] = []
        try:
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                pinned_targets: dict[str, PinnedDirectory] = {}
                names_by_path = {path: name for name, path in (targets or [])}
                for project_dir in project_dirs:
                    pinned = pin_existing(root, project_dir)
                    pinned.assert_path_identity()
                    if provenance is not None:
                        project_name = names_by_path.get(project_dir)
                        if project_name is None:
                            raise OperationBoundaryError(
                                "pre-commit target is not in the release plan"
                            )
                        self._bind_release_plan_target(
                            provenance,
                            project_name,
                            project_dir,
                            pinned,
                            plan_assertion=plan_assertion,
                        )
                    handles.append(pinned)
                    pinned_targets[project_dir] = pinned
                secure_results.extend(
                    self.pre_commit_projects(
                        run=True,
                        autoupdate=True,
                        projects=project_dirs,
                        _pinned_targets=pinned_targets,
                    )
                )
                secure_results.extend(
                    self.commit_projects(
                        message="chore: pre-commit autoupdate and formatting",
                        project_dirs=project_dirs,
                        _pinned_targets=pinned_targets,
                    )
                )
        except OperationBoundaryError as exc:
            secure_results.append(
                self._path_validation_result("pre_commit", self.path, exc)
            )
        finally:
            for pinned in reversed(handles):
                pinned.close()
        return secure_results

    def _resolve_bump_pre_commit_targets(
        self,
        config: dict,
        filter_set: set[str] | None,
        *,
        start_phase: int,
        single_phase: bool,
        targets: list[_ReleaseTarget] | None,
    ) -> list[_ReleaseTarget] | None:
        """Reuse the frozen pre-commit scope when one was supplied."""
        if targets is not None:
            return targets
        return self._pre_commit_project_targets(
            config,
            filter_set,
            start_phase=start_phase,
            single_phase=single_phase,
        )

    def _bump_phase_targets(
        self, phase: dict, filter_set: set[str] | None, assigned_projects: set[str]
    ) -> list[_ReleaseTarget]:
        """The exact targets one configured phase contributes to the bump plan.

        A filter only narrows a phase's existing members. It can never manufacture
        a bulk target that failed the manifest/category/package eligibility gate.
        """
        return self._select_phase_targets(
            phase,
            filter_set=filter_set,
            claimed=assigned_projects,
            include_bulk=bool(phase.get("bulk_bump")),
        )

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
        self,
        *,
        config: dict,
        start_phase: int,
        filter_set: set[str] | None,
        single_phase: bool = False,
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

        for phase in self._selected_release_phases(
            config, start_phase=start_phase, single_phase=single_phase
        ):
            phase_num = phase["phase"]

            targets = self._bump_phase_targets(phase, filter_set, assigned_projects)
            if not targets:
                continue

            phase_list.append(
                {
                    "phase_num": phase_num,
                    "name": phase.get("name", f"Phase {phase_num}"),
                    "targets": targets,
                }
            )
            total_projects += len(targets)

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
        project_dir: str,
        part: str,
        dry_run: bool,
        force: bool,
        all_results: list[GitResult],
        _pinned: PinnedDirectory | None = None,
    ) -> str | None:
        """Bump one project's version.

        Returns the new version string, ``"skipped"`` when the project had no
        bump-worthy change, or ``None`` when it was unresolvable or the bump
        failed. A declared project whose local clone is absent (stale registry
        entry / never-cloned repo) must not crash the whole phased bump, so it
        is skipped with a warning and the rest of the topology proceeds.
        """
        runner = {
            False: self._bump_one_project_unpinned,
            True: self._bump_one_project_pinned,
        }[_pinned is not None]
        return cast(Callable[..., str | None], runner)(
            project_name=project_name,
            project_dir=project_dir,
            part=part,
            dry_run=dry_run,
            force=force,
            all_results=all_results,
            pinned=_pinned,
        )

    def _bump_one_project_unpinned(
        self,
        *,
        project_name: str,
        project_dir: str,
        part: str,
        dry_run: bool,
        force: bool,
        all_results: list[GitResult],
        pinned: PinnedDirectory | None,
    ) -> str | None:
        """Pin a standalone bump target before running the common operation."""
        del pinned
        try:
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, project_dir) as operation_pinned:
                    return self._bump_one_project_pinned(
                        project_name=project_name,
                        project_dir=project_dir,
                        part=part,
                        dry_run=dry_run,
                        force=force,
                        all_results=all_results,
                        pinned=operation_pinned,
                    )
        except OperationBoundaryError as exc:
            logger.warning(
                "Skipping bump for unsafe target %s: %s",
                project_name,
                type(exc).__name__,
            )
            return None

    def _bump_one_project_pinned(
        self,
        *,
        project_name: str,
        project_dir: str,
        part: str,
        dry_run: bool,
        force: bool,
        all_results: list[GitResult],
        pinned: PinnedDirectory,
    ) -> str | None:
        """Run the bump logic against one retained descriptor-pinned checkout."""
        pinned.assert_path_identity()
        _, validated_path = self._revalidate_release_target(
            project_name,
            project_dir,
            operation="bump",
        )
        if Path(validated_path) != pinned.path:
            raise OperationBoundaryError("pinned bump target changed during validation")
        project_dir = validated_path

        if not force and self._bump_skip_reason(project_dir, pinned=pinned):
            logger.info("Skipping project version bump")
            return "skipped"

        pinned.assert_path_identity()
        result = self.bump_version(
            part=part,
            allow_dirty=True,
            path=project_dir,
            dry_run=dry_run,
            force=force,
            verbose=dry_run or not dry_run,
            _pinned=pinned,
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
        self,
        *,
        path: str,
        project_name: str,
        new_version: str,
        dry_run: bool,
        _pinned: PinnedDirectory | None = None,
    ) -> list[GitResult]:
        """Repin *project_name* in every dependency-declaring file under *path*.

        Not just ``pyproject.toml``: ``requirements.txt`` commonly pins the same
        package (often ``==``) and would otherwise go stale.
        (CONCEPT:RM-BUMP cross-dependency propagation)
        """
        runner = {
            False: self._update_dependency_files_unpinned,
            True: self._update_dependency_files_pinned,
        }[_pinned is not None]
        return cast(Callable[..., list[GitResult]], runner)(
            path=path,
            project_name=project_name,
            new_version=new_version,
            dry_run=dry_run,
            pinned=_pinned,
        )

    def _update_dependency_files_unpinned(
        self,
        *,
        path: str,
        project_name: str,
        new_version: str,
        dry_run: bool,
        pinned: PinnedDirectory | None,
    ) -> list[GitResult]:
        """Pin an unbound dependency-update target before editing it."""
        del pinned
        try:
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, path) as operation_pinned:
                    return self._update_dependency_files_pinned(
                        path=path,
                        project_name=project_name,
                        new_version=new_version,
                        dry_run=dry_run,
                        pinned=operation_pinned,
                    )
        except OperationBoundaryError as exc:
            return [self._path_validation_result("update_dependency", path, exc)]

    def _update_dependency_files_pinned(
        self,
        *,
        path: str,
        project_name: str,
        new_version: str,
        dry_run: bool,
        pinned: PinnedDirectory,
    ) -> list[GitResult]:
        """Repin every dependency file through one retained descriptor."""
        pinned.assert_path_identity()
        validated_path = self._validate_workspace_path(
            path,
            label=f"dependency update project {project_name!r}",
        )
        path = str(validated_path)
        if Path(path) != pinned.path:
            raise OperationBoundaryError(
                "pinned dependency-update target changed during validation"
            )
        return self._update_dependency_files_for_pinned_path(
            path, project_name, new_version, dry_run, pinned
        )

    def _update_dependency_files_for_pinned_path(
        self,
        path: str,
        project_name: str,
        new_version: str,
        dry_run: bool,
        pinned: PinnedDirectory,
    ) -> list[GitResult]:
        """Update each supported dependency document under one pinned checkout."""
        results: list[GitResult] = []
        for dep_file_name in ("pyproject.toml", "requirements.txt"):
            result = self._update_one_dependency_file(
                path, project_name, new_version, dry_run, dep_file_name, pinned
            )
            if result is not None:
                results.append(result)
        return results

    def _update_one_dependency_file(
        self,
        path: str,
        project_name: str,
        new_version: str,
        dry_run: bool,
        dep_file_name: str,
        pinned: PinnedDirectory,
    ) -> GitResult | None:
        """Update one dependency document and return its audit record."""
        pinned.assert_operation_identity()
        if read_at(pinned.fd, dep_file_name) is None:
            return None
        updated = self.update_dependency(
            str(Path(path) / dep_file_name),
            project_name,
            new_version,
            dry_run,
            _directory_fd=pinned.fd,
            _file_name=dep_file_name,
            _boundary_assertion=pinned.assert_operation_identity,
        )
        if not updated:
            return None
        return self._dependency_update_result(
            path, project_name, new_version, dep_file_name
        )

    def _propagate_bump_to_dependents(
        self,
        *,
        project_name: str,
        new_version: str,
        phase_num: int,
        phase_of: Callable[[str], int],
        dry_run: bool,
        all_results: list[GitResult],
        provenance: _ReleasePlanProvenance | None = None,
        plan_assertion: Callable[[], None] | None = None,
        allowed_targets: set[tuple[str, str]] | None = None,
    ) -> None:
        """Repin the just-bumped project across every same-or-later-phase repo.

        Earlier phases are skipped so a later bump cannot circle back and dirty
        a phase that has already been released.
        """
        planned_names_by_path = self._planned_dependency_names(allowed_targets)
        with self._dependency_propagation_root(provenance) as root:
            for path in self.project_map.values():
                self._propagate_bump_for_path(
                    path=path,
                    project_name=project_name,
                    new_version=new_version,
                    phase_num=phase_num,
                    phase_of=phase_of,
                    dry_run=dry_run,
                    all_results=all_results,
                    provenance=provenance,
                    plan_assertion=plan_assertion,
                    root=root,
                    planned_names_by_path=planned_names_by_path,
                )

    @staticmethod
    def _planned_dependency_names(
        allowed_targets: set[tuple[str, str]] | None,
    ) -> dict[str, set[str]] | None:
        """Map each normalized planned path to its exact release names."""
        if allowed_targets is None:
            return None
        names_by_path: dict[str, set[str]] = {}
        for name, planned_path in allowed_targets:
            names_by_path.setdefault(
                str(Path(os.path.abspath(planned_path))), set()
            ).add(name)
        return names_by_path

    def _dependency_propagation_root(
        self, provenance: _ReleasePlanProvenance | None
    ) -> contextlib.AbstractContextManager[PinnedDirectory | None]:
        """Open a root pin only for release-plan-bound propagation."""
        if provenance is not None:
            return open_directory(self._workspace_root())
        return contextlib.nullcontext(None)

    def _propagate_bump_for_path(
        self,
        *,
        path: str,
        project_name: str,
        new_version: str,
        phase_num: int,
        phase_of: Callable[[str], int],
        dry_run: bool,
        all_results: list[GitResult],
        provenance: _ReleasePlanProvenance | None,
        plan_assertion: Callable[[], None] | None,
        root: PinnedDirectory | None,
        planned_names_by_path: dict[str, set[str]] | None,
    ) -> None:
        """Apply one dependency propagation to one exact registry path."""
        normalized_path = str(Path(os.path.abspath(path)))
        if provenance is not None and planned_names_by_path is not None:
            planned_names = planned_names_by_path.get(normalized_path, set())
            if len(planned_names) != 1:
                # A service or an otherwise unplanned checkout must never become
                # a dependency target merely because its basename collides.
                return
            other_project_name = next(iter(planned_names))
        else:
            other_project_name = os.path.basename(path)
        if other_project_name == project_name:
            return
        other_phase = phase_of(other_project_name)
        if other_phase < phase_num:
            logger.info(
                f"Skipping dependency update for {project_name} in {other_project_name} "
                f"to avoid circular updates of earlier phase (Phase {other_phase} < Phase {phase_num})"
            )
            return
        if provenance is None:
            all_results.extend(
                self._update_dependency_files(
                    path=path,
                    project_name=project_name,
                    new_version=new_version,
                    dry_run=dry_run,
                )
            )
            return
        if root is None:
            raise OperationBoundaryError("dependency update root handle is missing")
        root.assert_root_identity()
        with pin_existing(root, path) as pinned:
            self._bind_release_plan_target(
                provenance,
                other_project_name,
                path,
                pinned,
                plan_assertion=plan_assertion,
            )
            all_results.extend(
                self._update_dependency_files(
                    path=path,
                    project_name=project_name,
                    new_version=new_version,
                    dry_run=dry_run,
                    _pinned=pinned,
                )
            )

    @staticmethod
    def _run_bump_phase(
        *,
        p_info: dict[str, Any],
        tracker: "_PhaseProgress",
        processed_paths: set[str],
        bump_one: Callable[[str, str], str | None],
        propagate: Callable[[str, str, int], None],
    ) -> None:
        """Bump every project in one phase, propagating each new version onward."""
        phase_name = p_info["name"]
        phase_num = p_info["phase_num"]
        tracker.begin_phase(phase_name)

        for project_name, project_path in p_info["targets"]:
            # Defensive: never bump a project twice in one run (a later phase
            # must not re-bump one an earlier phase already handled).
            if project_path in processed_paths:
                continue

            tracker.begin_item(phase_name, project_name)
            processed_paths.add(project_path)
            logger.info(
                f"Bumping version for project: {project_name} in {phase_name}..."
            )
            new_version = bump_one(project_name, project_path)
            tracker.finish_item(
                phase_name, project_name, "success" if new_version else "failed"
            )

            if new_version and re.match(r"^v?\d+\.\d+\.\d+", new_version):
                propagate(project_name, new_version, phase_num)

        tracker.end_phase(phase_name)

    def _prepare_bump_release_plan(
        self,
        *,
        config: dict,
        part: str,
        start_phase: int,
        dry_run: bool,
        allow_pre_commit: bool,
        single_phase: bool,
        project_filter: str | None,
        force: bool,
        auto_start: bool,
    ) -> tuple[
        set[str] | None,
        list[dict[str, Any]],
        int,
        list[_ReleaseTarget] | None,
        dict[str, Any],
        _ReleasePlanProvenance,
    ]:
        """Build the immutable bump scope and its input/provenance digests."""
        filter_set = self._parse_project_filter(project_filter)
        phase_list, total_projects = self._build_bump_phase_list(
            config=config,
            start_phase=start_phase,
            filter_set=filter_set,
            single_phase=single_phase,
        )
        precommit_targets: list[_ReleaseTarget] | None = (
            self._pre_commit_project_targets(
                config,
                filter_set,
                start_phase=start_phase,
                single_phase=single_phase,
            )
            if allow_pre_commit
            else []
        )
        plan_options = {
            "part": part,
            "start_phase": start_phase,
            "dry_run": dry_run,
            "allow_pre_commit": allow_pre_commit,
            "single_phase": single_phase,
            "project_filter": project_filter,
            "force": force,
            "auto_start": auto_start,
        }
        provenance = self._freeze_release_plan(
            "bump",
            config,
            phase_list,
            options=plan_options,
            auxiliary_targets=precommit_targets,
        )
        return (
            filter_set,
            phase_list,
            total_projects,
            precommit_targets,
            plan_options,
            provenance,
        )

    def _assert_bump_plan(
        self,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        precommit_targets: list[_ReleaseTarget] | None,
    ) -> None:
        """Check bump plan structure at one descriptor-boundary admission.

        Each target's root/checkout/Git identity is checked through its pinned
        handle.  A full workspace rescan here would be a separate sequential
        check rather than part of the mutation boundary.
        """
        self._assert_release_plan_structure(
            provenance,
            config,
            phase_list,
            plan_options,
            precommit_targets,
        )

    def _bump_plan_one(
        self,
        project_name: str,
        project_path: str,
        *,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        precommit_targets: list[_ReleaseTarget] | None,
        part: str,
        dry_run: bool,
        force: bool,
        all_results: list[GitResult],
    ) -> str | None:
        """Guard and bump one exact member of the frozen release plan."""
        self._assert_bump_plan(
            provenance, config, phase_list, plan_options, precommit_targets
        )
        return self._bump_plan_one_operation(
            project_name,
            project_path,
            provenance=provenance,
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
            precommit_targets=precommit_targets,
            part=part,
            dry_run=dry_run,
            force=force,
            all_results=all_results,
        )

    def _bump_plan_one_operation(
        self,
        project_name: str,
        project_path: str,
        *,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        precommit_targets: list[_ReleaseTarget] | None,
        part: str,
        dry_run: bool,
        force: bool,
        all_results: list[GitResult],
    ) -> str | None:
        """Pin and execute one frozen bump, converting refusal to a result."""
        try:
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                with pin_existing(root, project_path) as pinned:
                    pinned.assert_path_identity()
                    self._bind_release_plan_target(
                        provenance,
                        project_name,
                        project_path,
                        pinned,
                        plan_assertion=functools.partial(
                            self._assert_release_plan_structure,
                            provenance,
                            config,
                            phase_list,
                            plan_options,
                            precommit_targets,
                        ),
                    )
                    return self._bump_one_project(
                        project_name=project_name,
                        project_dir=project_path,
                        part=part,
                        dry_run=dry_run,
                        force=force,
                        all_results=all_results,
                        _pinned=pinned,
                    )
        except (OperationBoundaryError, ValueError) as exc:
            all_results.append(self._path_validation_result("bump", project_path, exc))
            return None

    def _propagate_bump_plan(
        self,
        project_name: str,
        new_version: str,
        phase_num: int,
        *,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        precommit_targets: list[_ReleaseTarget] | None,
        phase_of: Callable[[str], int],
        dry_run: bool,
        all_results: list[GitResult],
    ) -> None:
        """Guard and propagate one exact version across later plan phases."""
        self._assert_bump_plan(
            provenance, config, phase_list, plan_options, precommit_targets
        )
        return self._propagate_bump_plan_operation(
            project_name,
            new_version,
            phase_num,
            provenance=provenance,
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
            precommit_targets=precommit_targets,
            phase_of=phase_of,
            dry_run=dry_run,
            all_results=all_results,
        )

    def _propagate_bump_plan_operation(
        self,
        project_name: str,
        new_version: str,
        phase_num: int,
        *,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        precommit_targets: list[_ReleaseTarget] | None,
        phase_of: Callable[[str], int],
        dry_run: bool,
        all_results: list[GitResult],
    ) -> None:
        """Propagate one bump across only exact frozen release targets."""
        try:
            allowed_targets = {
                (name, path)
                for phase in phase_list
                for name, path in phase.get("targets", [])
            }
            self._propagate_bump_to_dependents(
                project_name=project_name,
                new_version=new_version,
                phase_num=phase_num,
                phase_of=phase_of,
                dry_run=dry_run,
                all_results=all_results,
                provenance=provenance,
                plan_assertion=functools.partial(
                    self._assert_release_plan_structure,
                    provenance,
                    config,
                    phase_list,
                    plan_options,
                    precommit_targets,
                ),
                allowed_targets=allowed_targets,
            )
        except (OperationBoundaryError, ValueError) as exc:
            all_results.append(
                self._path_validation_result("dependency_update", project_name, exc)
            )

    def _record_release_plan_abort(
        self,
        drift: _ReleasePlanDrift,
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> None:
        """Record a typed abort when a frozen release plan changed."""
        all_results.append(self._release_plan_drift_result(drift))
        tracker.note(f"ABORTED — phased {drift.operation} release plan changed")

    def _run_bump_precommit_plan(
        self,
        *,
        allow_pre_commit: bool,
        config: dict[str, Any],
        filter_set: set[str] | None,
        start_phase: int,
        single_phase: bool,
        precommit_targets: list[_ReleaseTarget] | None,
        provenance: _ReleasePlanProvenance,
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Run the guarded pre-commit stage, returning false on plan drift."""
        return self._run_bump_precommit_plan_operation(
            allow_pre_commit=allow_pre_commit,
            config=config,
            filter_set=filter_set,
            start_phase=start_phase,
            single_phase=single_phase,
            precommit_targets=precommit_targets,
            provenance=provenance,
            phase_list=phase_list,
            plan_options=plan_options,
            tracker=tracker,
            all_results=all_results,
        )

    def _run_bump_precommit_plan_operation(
        self,
        *,
        allow_pre_commit: bool,
        config: dict[str, Any],
        filter_set: set[str] | None,
        start_phase: int,
        single_phase: bool,
        precommit_targets: list[_ReleaseTarget] | None,
        provenance: _ReleasePlanProvenance,
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Run the guarded pre-commit stage and convert boundary failures."""
        if not allow_pre_commit:
            return True
        try:
            self._assert_bump_plan(
                provenance, config, phase_list, plan_options, precommit_targets
            )
            all_results.extend(
                self._run_bump_pre_commit_stage(
                    config,
                    filter_set,
                    start_phase=start_phase,
                    single_phase=single_phase,
                    targets=precommit_targets,
                    _secure=True,
                    provenance=provenance,
                    plan_assertion=functools.partial(
                        self._assert_release_plan_structure,
                        provenance,
                        config,
                        phase_list,
                        plan_options,
                        precommit_targets,
                    ),
                )
            )
        except _ReleasePlanDrift as drift:
            self._record_release_plan_abort(drift, tracker, all_results)
            return False
        except (OperationBoundaryError, ValueError) as exc:
            all_results.append(self._path_validation_result("bump", self.path, exc))
            tracker.note("ABORTED — unsafe bump operation")
            return False
        return True

    def _execute_frozen_bump_plan(
        self,
        *,
        config: dict[str, Any],
        filter_set: set[str] | None,
        start_phase: int,
        part: str,
        dry_run: bool,
        allow_pre_commit: bool,
        single_phase: bool,
        force: bool,
        precommit_targets: list[_ReleaseTarget] | None,
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        provenance: _ReleasePlanProvenance,
        project_phases: dict[str, int],
        bulk_phase_num: int,
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Execute all guarded bump mutations for one frozen plan."""
        if not self._run_bump_precommit_plan(
            allow_pre_commit=allow_pre_commit,
            config=config,
            filter_set=filter_set,
            start_phase=start_phase,
            single_phase=single_phase,
            precommit_targets=precommit_targets,
            provenance=provenance,
            phase_list=phase_list,
            plan_options=plan_options,
            tracker=tracker,
            all_results=all_results,
        ):
            return False

        def phase_of(name: str) -> int:
            return project_phases.get(name, bulk_phase_num)

        bump_one = functools.partial(
            self._bump_plan_one,
            provenance=provenance,
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
            precommit_targets=precommit_targets,
            part=part,
            dry_run=dry_run,
            force=force,
            all_results=all_results,
        )
        propagate = functools.partial(
            self._propagate_bump_plan,
            provenance=provenance,
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
            precommit_targets=precommit_targets,
            phase_of=phase_of,
            dry_run=dry_run,
            all_results=all_results,
        )
        processed_paths: set[str] = set()
        try:
            self._run_guarded_bump_phases(
                phase_list=phase_list,
                provenance=provenance,
                config=config,
                plan_options=plan_options,
                precommit_targets=precommit_targets,
                tracker=tracker,
                processed_paths=processed_paths,
                bump_one=bump_one,
                propagate=propagate,
            )
        except _ReleasePlanDrift as drift:
            self._record_release_plan_abort(drift, tracker, all_results)
            return False
        except (OperationBoundaryError, ValueError) as exc:
            all_results.append(self._path_validation_result("bump", self.path, exc))
            tracker.note("ABORTED — unsafe bump operation")
            return False
        return True

    def _run_guarded_bump_phases(
        self,
        *,
        phase_list: list[dict[str, Any]],
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        plan_options: dict[str, Any],
        precommit_targets: list[_ReleaseTarget] | None,
        tracker: "_PhaseProgress",
        processed_paths: set[str],
        bump_one: Callable[[str, str], str | None],
        propagate: Callable[[str, str, int], None],
    ) -> None:
        """Run every phase with a structural admission immediately beforehand."""
        for p_info in phase_list:
            self._assert_bump_plan(
                provenance, config, phase_list, plan_options, precommit_targets
            )
            self._run_bump_phase(
                p_info=p_info,
                tracker=tracker,
                processed_paths=processed_paths,
                bump_one=bump_one,
                propagate=propagate,
            )

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

        tracker = _PhaseProgress(state=progress, noun="bump")

        if self._should_auto_start(
            auto_start=auto_start,
            project_filter=project_filter,
            single_phase=single_phase,
            force=force,
        ):
            detected = self._resolve_auto_start_phase(
                config=config,
                start_phase=start_phase,
                operation="bump",
                tracker=tracker,
                noun="bump",
                lowest_label="lowest changed phase",
            )
            if detected is None:
                return all_results
            start_phase = detected

        (
            filter_set,
            phase_list,
            total_projects,
            precommit_targets,
            plan_options,
            provenance,
        ) = self._prepare_bump_release_plan(
            config=config,
            part=part,
            start_phase=start_phase,
            dry_run=dry_run,
            allow_pre_commit=allow_pre_commit,
            single_phase=single_phase,
            project_filter=project_filter,
            force=force,
            auto_start=auto_start,
        )
        tracker.total = total_projects
        self._record_release_plan(progress, provenance)
        tracker.initialize(
            "Initializing Bumps",
            self._bump_progress_phases(phase_list),
        )

        if not self._execute_frozen_bump_plan(
            config=config,
            filter_set=filter_set,
            start_phase=start_phase,
            part=part,
            dry_run=dry_run,
            allow_pre_commit=allow_pre_commit,
            single_phase=single_phase,
            force=force,
            precommit_targets=precommit_targets,
            phase_list=phase_list,
            plan_options=plan_options,
            provenance=provenance,
            project_phases=project_phases,
            bulk_phase_num=bulk_phase_num,
            tracker=tracker,
            all_results=all_results,
        ):
            return all_results

        tracker.finish("Bumps Completed")
        return all_results

    @staticmethod
    def _bump_progress_phases(
        phase_list: list[dict[str, Any]],
    ) -> list[tuple[str, list[str]]]:
        """Convert exact targets to the progress display's project-name view."""
        return [
            (phase["name"], [name for name, _path in phase["targets"]])
            for phase in phase_list
        ]

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

    def _resolve_maintenance_config(self, config: object | None) -> dict | None:
        """Strictly validate *config*, or load the manifest config in its place.

        ``None`` means no maintenance configuration is reachable and the caller
        must abort; the error is logged here so every phased workflow reports it
        identically.
        """
        if config is not None:
            return self._validated_maintenance_mapping(config)
        config_model = self._maintenance_config_model()
        if config_model is None:
            logger.error("No maintenance configuration found.")
            return None
        return config_model.model_dump(exclude_none=True)

    @staticmethod
    def _validated_maintenance_mapping(config: object) -> dict[str, Any] | None:
        """Validate a caller-supplied maintenance mapping without coercion."""
        if type(config) is not dict:
            logger.error("Maintenance configuration must be a mapping.")
            return None
        try:
            model = MaintenanceConfig.model_validate(config, strict=True)
        except ValidationError as exc:
            logger.error("Invalid maintenance configuration: %s", exc)
            return None
        return model.model_dump(exclude_none=True)

    def _resolve_auto_start_phase(
        self,
        *,
        config: dict,
        start_phase: int,
        operation: Literal["bump", "push"],
        tracker: "_PhaseProgress",
        noun: str,
        lowest_label: str,
    ) -> int | None:
        """Advance *start_phase* to the lowest phase that still has pending work.

        Never moves the start phase backwards. Returns ``None`` when nothing at
        all is pending — in which case *tracker* has already been finalized and
        the caller should return immediately.
        """
        detected = self._auto_start_phase(config, operation=operation)
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

    @staticmethod
    def _should_auto_start(
        *,
        auto_start: bool,
        project_filter: str | None,
        single_phase: bool,
        force: bool = False,
    ) -> bool:
        """Apply the same explicit-scope exclusions to bump and push discovery."""
        return bool(
            auto_start and project_filter is None and not single_phase and not force
        )

    def _project_path_for(self, project_name: str) -> str | None:
        """Local clone path of *project_name* from the URL->path project map."""
        for url, p_path in sorted(self.project_map.items()):
            if self._release_project_name_or_empty(url) == project_name:
                return p_path
        return None

    def _validated_bulk_release_path(
        self, project_path: str, expected_name: str
    ) -> Path | None:
        """Canonical direct child of the manifest's PyPI-agent category."""
        try:
            validated = self._validate_workspace_path(
                project_path,
                label="bulk release project",
                require_directory=True,
            )
            category_root = self._validate_workspace_path(
                Path(self.path) / "agent-packages" / "agents",
                label="bulk release category",
                require_directory=True,
            )
        except ValueError:
            return None
        if validated.parent != category_root or validated.name != expected_name:
            return None
        return validated

    def _has_pypi_release_metadata(
        self, project_path: str, *, expected_name: str
    ) -> bool:
        """Whether *project_path* declares a buildable PyPI distribution.

        Phase 5 is a package-release wave, so merely having a git repository or
        a file named ``pyproject.toml`` is insufficient. The metadata must name
        the distribution, provide a static or dynamic version, and select a
        PEP 517 build backend with non-empty requirements. Malformed, partial,
        or unreadable metadata fails closed.
        """
        validated = self._validated_bulk_release_path(project_path, expected_name)
        if validated is None:
            return False
        return bool(
            read_release_document(
                validated / "pyproject.toml", expected_name=expected_name
            )
        )

    @classmethod
    def _release_project_name(cls, url: str) -> str:
        """Return a safe normalized repository basename or reject the URL."""
        return release_repository_name(url)

    @classmethod
    def _release_project_name_or_empty(cls, url: str) -> str:
        """Fail-closed URL identity for callers that use an empty sentinel."""
        try:
            return cls._release_project_name(url)
        except ValueError:
            return ""

    def _url_has_pypi_release_metadata(self, url: str, project_path: str) -> bool:
        """Validate package metadata against a URL's safe repository identity."""
        name = self._release_project_name_or_empty(url)
        return bool(
            name and self._has_pypi_release_metadata(project_path, expected_name=name)
        )

    def _is_bulk_release_target(self, url: str, project_path: str) -> bool:
        """Whether a manifest entry is eligible for the Phase-5 PyPI wave."""
        return bool(
            self._project_categories.get(url) == ("agent-packages", "agents")
            and self._url_has_pypi_release_metadata(url, project_path)
        )

    def _validated_bulk_release_target(
        self, url: str, project_path: str
    ) -> _ReleaseTarget | None:
        """Return one eligible target with its canonical path, or ``None``."""
        name = self._release_project_name_or_empty(url)
        if not name or self._project_categories.get(url) != (
            "agent-packages",
            "agents",
        ):
            return None
        validated = self._validated_bulk_release_path(project_path, name)
        if validated is None or not read_release_document(
            validated / "pyproject.toml", expected_name=name
        ):
            return None
        return name, str(validated)

    def _eligible_bulk_release_targets(self) -> list[tuple[str, str]]:
        """Validate and deduplicate the complete eligible Phase-5 universe."""
        targets: list[tuple[str, str]] = []
        eligible_names: set[str] = set()
        for url, path in sorted(self.project_map.items()):
            target = self._validated_bulk_release_target(url, path)
            if target is None:
                continue
            name, canonical_path = target
            if name in eligible_names:
                raise ValueError(f"duplicate Phase-5 repository basename: {name}")
            eligible_names.add(name)
            targets.append((name, canonical_path))
        return sorted(targets, key=lambda target: (target[0], target[1]))

    def _bulk_release_targets(
        self, processed_projects: set[str]
    ) -> list[tuple[str, str]]:
        """Eligible PyPI agent repos not already claimed by an earlier phase."""
        targets = [
            target
            for target in self._eligible_bulk_release_targets()
            if target[0] not in processed_projects
        ]
        processed_projects.update(name for name, _path in targets)
        return targets

    def _phase_push_targets(
        self,
        phase: dict,
        filter_set: set[str] | None,
        processed_projects: set[str],
    ) -> list[tuple[str, str]]:
        """The (name, path) pairs one configured phase would push."""
        return self._select_phase_targets(
            phase,
            filter_set=filter_set,
            claimed=processed_projects,
            include_bulk=bool(phase.get("bulk_push")),
        )

    def _filtered_bulk_push_targets(
        self,
        processed_projects: set[str],
        filter_set: set[str] | None,
        phase: dict[str, Any] | None = None,
    ) -> list[tuple[str, str]]:
        """Eligible bulk targets narrowed by the shared comma-filter grammar."""
        return self._select_phase_targets(
            phase or {},
            filter_set=filter_set,
            claimed=processed_projects,
            include_bulk=True,
        )

    def _select_phase_targets(
        self,
        phase: dict[str, Any],
        *,
        filter_set: set[str] | None,
        claimed: set[str],
        include_bulk: bool,
    ) -> list[_ReleaseTarget]:
        """Select exact canonical targets with one deterministic policy."""
        candidates = self._phase_target_candidates(
            phase,
            filter_set=filter_set,
            claimed=claimed,
            include_bulk=include_bulk,
        )
        selected = self._selected_candidate_targets(
            candidates, phase=phase, filter_set=filter_set, claimed=claimed
        )
        claimed.update(name for name, _path in selected)
        return selected

    def _phase_target_candidates(
        self,
        phase: dict[str, Any],
        *,
        filter_set: set[str] | None,
        claimed: set[str],
        include_bulk: bool,
    ) -> dict[str, str]:
        """Build exact candidate identities before common narrowing."""
        candidates = (
            {name: path for name, path in self._eligible_bulk_release_targets()}
            if include_bulk
            else {}
        )
        for name in self._phase_named_projects(phase):
            if self._explicit_candidate_is_selected(
                name,
                candidates=candidates,
                phase=phase,
                filter_set=filter_set,
                claimed=claimed,
            ):
                explicit_name, explicit_path = self._explicit_release_target(name)
                self._merge_release_candidate(candidates, explicit_name, explicit_path)
        return candidates

    @staticmethod
    def _merge_release_candidate(
        candidates: dict[str, str], name: str, path: str
    ) -> None:
        """Merge one explicit target only when its canonical path agrees."""
        existing_path = candidates.get(name)
        if existing_path is not None and existing_path != path:
            raise ValueError(f"explicit and bulk release targets disagree for {name!r}")
        candidates[name] = path

    def _explicit_candidate_is_selected(
        self,
        name: str,
        *,
        candidates: dict[str, str],
        phase: dict[str, Any],
        filter_set: set[str] | None,
        claimed: set[str],
    ) -> bool:
        """Whether an explicit name still needs an exact target resolution."""
        return bool(
            name not in claimed
            and (filter_set is None or name in filter_set)
            and not self._phase_excludes_name(phase, name)
        )

    def _selected_candidate_targets(
        self,
        candidates: dict[str, str],
        *,
        phase: dict[str, Any],
        filter_set: set[str] | None,
        claimed: set[str],
    ) -> list[_ReleaseTarget]:
        """Narrow exact candidates with the shared filter/exclude policy."""
        return [
            (name, candidates[name])
            for name in sorted(candidates)
            if name not in claimed
            and (filter_set is None or name in filter_set)
            and not self._phase_excludes_name(phase, name)
        ]

    def _validated_explicit_targets(
        self, url: str, path: str, expected_name: str
    ) -> tuple[_ReleaseTarget, ...]:
        """Return the exact validated match for one manifest entry."""
        if self._release_project_name_or_empty(url) != expected_name:
            return ()
        validated = self._validate_workspace_path(
            path, label="explicit release project"
        )
        if validated.name != expected_name:
            return ()
        return ((expected_name, str(validated)),)

    def _explicit_release_target(self, expected_name: str) -> _ReleaseTarget:
        """Resolve one unambiguous manifest name to its validated canonical path."""
        matches: list[_ReleaseTarget] = []
        for url, path in sorted(self.project_map.items()):
            matches.extend(self._validated_explicit_targets(url, path, expected_name))
        if len(matches) != 1:
            raise ValueError(
                f"maintenance project must resolve to one canonical path: {expected_name}"
            )
        return matches[0]

    @staticmethod
    def _phase_named_projects(phase: dict[str, Any]) -> list[str]:
        """Validated explicit project names declared by one phase."""
        projects = phase.get("projects", [])
        if not isinstance(projects, list):
            raise ValueError("maintenance phase projects must be a list")
        names = list(projects)
        singular = phase.get("project")
        if singular is not None:
            names.append(singular)
        if not all(isinstance(name, str) and name.strip() for name in names):
            raise ValueError("maintenance project names must be non-empty strings")
        return names

    @staticmethod
    def _phase_exclude_patterns(phase: dict[str, Any]) -> list[str]:
        """Validated fnmatch exclusions declared by one phase."""
        patterns = phase.get("exclude", []) or []
        if not isinstance(patterns, list) or not all(
            isinstance(pattern, str) and pattern for pattern in patterns
        ):
            raise ValueError("maintenance phase exclude must be a string list")
        return patterns

    @classmethod
    def _phase_excludes_name(cls, phase: dict[str, Any], name: str) -> bool:
        """Whether one project identity matches a phase exclusion."""
        return any(
            fnmatch.fnmatch(name, pattern)
            for pattern in cls._phase_exclude_patterns(phase)
        )

    @staticmethod
    def _claim_unique_phase_projects(names: list[str], claimed: set[str]) -> None:
        """Reject a repeated explicit project within or across phases."""
        for name in names:
            if name in claimed:
                raise ValueError(f"duplicate maintenance project name: {name}")
            claimed.add(name)

    @classmethod
    def _ordered_release_phases(cls, config: dict) -> list[dict[str, Any]]:
        """Validate phase identifiers and return a deterministic numeric order."""
        phases = config.get("phases", [])
        if not isinstance(phases, list):
            raise ValueError("maintenance phases must be a list")
        seen: set[int] = set()
        claimed_projects: set[str] = set()
        ordered: list[dict[str, Any]] = []
        for phase in phases:
            if not isinstance(phase, dict):
                raise ValueError("each maintenance phase must be a mapping")
            number = phase.get("phase")
            if isinstance(number, bool) or not isinstance(number, int) or number < 1:
                raise ValueError("maintenance phase numbers must be positive integers")
            if number in seen:
                raise ValueError(f"duplicate maintenance phase number: {number}")
            seen.add(number)
            cls._claim_unique_phase_projects(
                cls._phase_named_projects(phase), claimed_projects
            )
            cls._phase_exclude_patterns(phase)
            ordered.append(phase)
        return sorted(ordered, key=lambda phase: phase["phase"])

    @classmethod
    def _selected_release_phases(
        cls, config: dict, *, start_phase: int, single_phase: bool
    ) -> list[dict[str, Any]]:
        """Ordered phases at/after the start, or exactly the start phase."""
        phases = cls._ordered_release_phases(config)
        if single_phase:
            return [phase for phase in phases if phase["phase"] == start_phase]
        return [phase for phase in phases if phase["phase"] >= start_phase]

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
        self,
        *,
        config: dict,
        start_phase: int,
        project_filter: str | None,
        single_phase: bool = False,
    ) -> tuple[list[dict[str, Any]], int]:
        """Expand the configured phases into the concrete push plan.

        Returns the ordered phase records and the total number of projects
        across them (used for the overall progress percentage).
        """
        processed_projects: set[str] = set()
        phase_list: list[dict[str, Any]] = []
        total_projects = 0
        filter_set = self._parse_project_filter(project_filter)

        for phase in self._selected_release_phases(
            config, start_phase=start_phase, single_phase=single_phase
        ):
            phase_num = phase["phase"]

            targets = self._phase_push_targets(phase, filter_set, processed_projects)
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
        before_mutation: Callable[[str, str], None] | None = None,
        provenance: _ReleasePlanProvenance | None = None,
        plan_assertion: Callable[[], None] | None = None,
    ) -> bool:
        """Push one release phase through the descriptor-boundary operation."""
        return self._execute_push_phase_operation(
            p_info=p_info,
            tracker=tracker,
            all_results=all_results,
            before_mutation=before_mutation,
            provenance=provenance,
            plan_assertion=plan_assertion,
        )

    def _execute_push_phase_operation(
        self,
        *,
        p_info: dict[str, Any],
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
        before_mutation: Callable[[str, str], None] | None = None,
        provenance: _ReleasePlanProvenance | None = None,
        plan_assertion: Callable[[], None] | None = None,
    ) -> bool:
        """Push one phase's projects in parallel; return whether anything landed."""
        phase_name = p_info["name"]
        projects_to_push = p_info["projects_to_push"]

        tracker.begin_phase(phase_name)
        logger.info(
            f"Starting {phase_name} push for {len(projects_to_push)} projects..."
        )

        phase_had_pushes = False
        try:
            with open_directory(self._workspace_root()) as root:
                root.assert_root_identity()
                pinned_targets: list[PinnedDirectory] = []
                try:
                    with concurrent.futures.ThreadPoolExecutor(
                        max_workers=self.threads
                    ) as executor:
                        future_to_proj = {}
                        for proj_name, p_path in projects_to_push:
                            try:
                                validated_path = self._validated_push_phase_target(
                                    proj_name, p_path, before_mutation
                                )
                                pinned = pin_existing(root, validated_path)
                                pinned.assert_path_identity()
                                if provenance is not None:
                                    self._bind_release_plan_target(
                                        provenance,
                                        proj_name,
                                        validated_path,
                                        pinned,
                                        plan_assertion=plan_assertion,
                                    )
                            except _ReleasePlanDrift:
                                raise
                            except (OperationBoundaryError, ValueError) as exc:
                                all_results.append(
                                    self._path_validation_result("push", p_path, exc)
                                )
                                tracker.note(
                                    f"ABORTED — unsafe push target {proj_name}"
                                )
                                return False
                            pinned_targets.append(pinned)
                            tracker.begin_item(phase_name, proj_name)
                            future = executor.submit(
                                self.push_project,
                                path=validated_path,
                                _pinned=pinned,
                            )
                            future_to_proj[future] = proj_name

                        for future in concurrent.futures.as_completed(future_to_proj):
                            proj_name = future_to_proj[future]
                            status_str, pushed = self._collect_push_result(
                                future, all_results
                            )
                            phase_had_pushes = phase_had_pushes or pushed
                            tracker.finish_item(phase_name, proj_name, status_str)
                finally:
                    for pinned in reversed(pinned_targets):
                        pinned.close()
        except OperationBoundaryError as exc:
            all_results.append(self._path_validation_result("push", self.path, exc))
            tracker.note(f"ABORTED — unsafe push phase {phase_name}")
            return False

        tracker.end_phase(phase_name)
        return phase_had_pushes

    def _validated_push_phase_target(
        self,
        project_name: str,
        project_path: str,
        before_mutation: Callable[[str, str], None] | None,
    ) -> str:
        """Run the plan guard, then return the exact canonical push path."""
        if before_mutation is not None:
            before_mutation(project_name, project_path)
        return self._revalidate_release_target(
            project_name,
            project_path,
            operation="push",
            require_directory=True,
        )[1]

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

    def _assert_push_plan(
        self,
        _project_name: str,
        _project_path: str,
        *,
        provenance: _ReleasePlanProvenance,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
    ) -> None:
        """Check push plan structure immediately before one target mutation.

        Filesystem identity belongs to the descriptor-bound target assertion;
        rescanning every checkout here would recreate the sequential
        check-then-use boundary this operation layer is meant to remove.
        """
        self._assert_release_plan_structure(
            provenance,
            config,
            phase_list,
            plan_options,
        )

    def _execute_frozen_push_plan(
        self,
        *,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        provenance: _ReleasePlanProvenance,
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Execute all guarded push mutations for one frozen plan."""
        before_mutation = functools.partial(
            self._assert_push_plan,
            provenance=provenance,
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
        )
        plan_assertion = functools.partial(
            self._assert_release_plan_structure,
            provenance,
            config,
            phase_list,
            plan_options,
        )
        try:
            for phase_idx, p_info in enumerate(phase_list):
                self._assert_release_plan(
                    provenance, config, phase_list, options=plan_options
                )
                phase_had_pushes = self._execute_push_phase(
                    p_info=p_info,
                    tracker=tracker,
                    all_results=all_results,
                    before_mutation=before_mutation,
                    provenance=provenance,
                    plan_assertion=plan_assertion,
                )
                if not self._settle_phase_barrier(
                    p_info=p_info,
                    phase_had_pushes=phase_had_pushes,
                    later_phases=phase_list[phase_idx + 1 :],
                    tracker=tracker,
                    all_results=all_results,
                ):
                    return False
        except _ReleasePlanDrift as drift:
            self._record_release_plan_abort(drift, tracker, all_results)
            return False
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

        if self._should_auto_start(
            auto_start=auto_start,
            project_filter=project_filter,
            single_phase=single_phase,
        ):
            detected = self._resolve_auto_start_phase(
                config=config,
                start_phase=start_phase,
                operation="push",
                tracker=tracker,
                noun="push",
                lowest_label="lowest unpushed phase",
            )
            if detected is None:
                return all_results
            start_phase = detected

        phase_list, tracker.total = self._build_push_phase_list(
            config=config,
            start_phase=start_phase,
            project_filter=project_filter,
            single_phase=single_phase,
        )
        plan_options = {
            "start_phase": start_phase,
            "single_phase": single_phase,
            "project_filter": project_filter,
            "auto_start": auto_start,
        }
        provenance = self._freeze_release_plan(
            "push", config, phase_list, options=plan_options
        )
        self._record_release_plan(progress, provenance)
        tracker.initialize(
            "Initializing Pushes",
            [(p["name"], [n for n, _ in p["projects_to_push"]]) for p in phase_list],
        )

        if not self._execute_phased_push_plan(
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
            provenance=provenance,
            tracker=tracker,
            all_results=all_results,
        ):
            return all_results

        tracker.finish("Pushes Completed")
        return all_results

    def _execute_phased_push_plan(
        self,
        *,
        config: dict[str, Any],
        phase_list: list[dict[str, Any]],
        plan_options: dict[str, Any],
        provenance: _ReleasePlanProvenance,
        tracker: "_PhaseProgress",
        all_results: list[GitResult],
    ) -> bool:
        """Consume the durable receipt and execute one frozen push plan."""
        if phase_list:
            replayed, refusal = self._begin_push_plan_receipt(provenance)
            if refusal is not None:
                tracker.nothing_to_do("Push refused — release plan already consumed")
                all_results.append(refusal)
                return False
            if replayed is not None:
                tracker.nothing_to_do("Pushes replayed from recorded outcome")
                all_results.extend(replayed)
                return False
        completed = self._execute_frozen_push_plan(
            config=config,
            phase_list=phase_list,
            plan_options=plan_options,
            provenance=provenance,
            tracker=tracker,
            all_results=all_results,
        )
        if phase_list:
            self._complete_push_plan_receipt(provenance, all_results)
        return completed

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
        from packaging.utils import canonicalize_name

        published: dict[str, str] = {}
        for proj_name, p_path in projects_to_push:
            _, validated_path = self._revalidate_release_target(
                proj_name,
                p_path,
                operation="published-package inspection",
            )
            pyproject_path = Path(validated_path) / "pyproject.toml"
            data = read_release_document(pyproject_path)
            if data is None:
                continue
            name = data["project"]["name"]
            published[canonicalize_name(name)] = str(pyproject_path)
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

        # Keep a separately built universe for the readiness cross-check.  It
        # must include every later-phase candidate, not only the narrowed set
        # that currently declares a constraint, or an omitted dependent repo
        # could make the barrier return early without ever running its check.
        targets, candidate_repos = self._phase_readiness_targets(
            published, later_phases
        )

        if not targets:
            logger.info(
                "Phase %s published %s; no later-phase repo declares a "
                "constraint on it — proceeding immediately.",
                phase_num,
                sorted(published),
            )
        else:
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
            published_packages=set(published),
            candidate_repos=candidate_repos,
        )

    def _phase_readiness_targets(
        self,
        published: dict[str, str],
        later_phases: list[dict[str, Any]],
    ) -> tuple[dict[str, _ReleaseTarget], list[_ReleaseTarget]]:
        """Build both the complete candidate universe and narrowed gate targets."""
        targets: dict[str, _ReleaseTarget] = {}
        candidate_repos: list[_ReleaseTarget] = []
        candidate_paths: set[str] = set()
        for later in later_phases:
            for proj_name, p_path in later["projects_to_push"]:
                _, validated_path = self._revalidate_release_target(
                    proj_name,
                    p_path,
                    operation="gate-readiness candidate",
                    require_directory=True,
                )
                if validated_path in candidate_paths:
                    continue
                candidate = (proj_name, validated_path)
                candidate_repos.append(candidate)
                candidate_paths.add(validated_path)
                constraints = dependency_readiness.declared_fleet_constraints(
                    validated_path, fleet_packages=set(published)
                )
                if constraints:
                    targets[validated_path] = candidate
        return targets, candidate_repos

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
            self._project_categories = {}

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

            # Validate the root independently of the project map.  A manifest
            # with an empty map must not be able to admit a symlink root that
            # later redirects setup writes outside the configured tree.
            self._validate_manifest_root_ancestry(self._manifest_workspace_root())
            logger.info("Workspace root resolved")

            seen_repository_urls: set[str] = set()
            self.project_map = self._parse_subdirectories(
                self.config.subdirectories,
                self.path,
                category_path=(),
                seen_repository_urls=seen_repository_urls,
            )

            self.project_map.update(
                self._parse_root_repositories(
                    self.config.repositories, seen_repository_urls
                )
            )
            self._validate_manifest_checkout_origins()
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
            self.config = None
            self.project_map = {}
            self._project_categories = {}
            return False

    def _parse_root_repositories(
        self,
        repositories: list[RepositoryConfig],
        seen_repository_urls: set[str],
    ) -> dict[str, str]:
        """Parse canonical root-level repositories from one workspace manifest."""
        project_map: dict[str, str] = {}
        workspace_root = Path(os.path.abspath(os.path.expanduser(os.fspath(self.path))))
        for repository in repositories:
            url, name = self._manifest_repository_identity(
                repository, seen_repository_urls
            )
            project_map[url] = str(workspace_root / name)
            self._project_categories[url] = ()
        return project_map

    @staticmethod
    def _git_remote_url(repo_path: str) -> str | None:
        """The configured ``origin`` URL of a repository, or ``None``."""
        try:
            git_path = shutil.which("git") or "git"
            res = subprocess.run(
                [git_path, "config", "--get", "remote.origin.url"],
                cwd=repo_path,
                capture_output=True,
                text=True,
                check=False,
            )
            if res.returncode == 0 and res.stdout.strip():
                return res.stdout.strip()
        except Exception as exc:
            logger.debug(
                "Failed to get a remote URL: error_type=%s",
                type(exc).__name__,
            )
        return None

    def _validate_manifest_subdirectory(self, name: object, current_path: str) -> Path:
        """Validate one manifest directory key before joining it to a path.

        Manifest keys are directory *segments*, not arbitrary filesystem paths.
        Rejecting absolute paths and lexical parents before ``join`` prevents a
        path alias from being normalized into an apparently safe destination;
        checking each existing component rejects both live and broken symlinks.
        The workspace root may not exist yet during a load, so this helper does
        not require it to be a directory until the later sync boundary.
        """
        raw = self._manifest_subdirectory_name(name)
        root = self._manifest_workspace_root()
        self._validate_manifest_root_ancestry(root)
        return self._validated_manifest_subdirectory_path(root, current_path, raw)

    @staticmethod
    def _manifest_subdirectory_name(name: object) -> Path:
        """Validate and parse one portable, single-segment manifest key."""
        if type(name) is not str or not name or name != name.strip():
            raise ValueError("manifest subdirectory names must be non-empty strings")
        if "\x00" in name or "\\" in name:
            raise ValueError(
                "manifest subdirectory names must be portable path segments"
            )
        raw = Path(name)
        _reject_lexical_parent(raw, label="manifest subdirectory")
        Git._validate_manifest_subdirectory_shape(name, raw)
        return raw

    @staticmethod
    def _validate_manifest_subdirectory_shape(name: str, raw: Path) -> None:
        """Reject platform-specific absolute or multi-segment spellings."""
        if raw.is_absolute() or PureWindowsPath(name).is_absolute():
            raise ValueError("manifest subdirectory must be relative")
        if PureWindowsPath(name).drive:
            raise ValueError("manifest subdirectory must not contain a drive")
        if len(raw.parts) != 1 or raw.parts[0] in {".", ""}:
            raise ValueError("manifest subdirectory must be one path segment")

    def _manifest_workspace_root(self) -> Path:
        """Return the absolute workspace root used while parsing a manifest."""
        return Path(os.path.abspath(os.path.expanduser(os.fspath(self.path))))

    @staticmethod
    def _validate_manifest_root_ancestry(root: Path) -> None:
        """Reject symlink or non-directory components in a manifest root."""
        if root.is_symlink():
            raise ValueError(f"workspace root contains symlink component {root}")
        current = Path(root.anchor)
        for component in root.parts[1:]:
            current /= component
            if current.is_symlink():
                raise ValueError(f"workspace root contains symlink component {current}")
            if current != root and current.exists() and not current.is_dir():
                raise ValueError(
                    f"workspace root contains non-directory component {current}"
                )

    def _validated_manifest_subdirectory_path(
        self, root: Path, current_path: str, raw: Path
    ) -> Path:
        """Validate containment and existing components for one subdirectory."""
        base = Path(os.path.abspath(os.path.expanduser(current_path)))
        candidate = Path(os.path.abspath(base / raw))
        try:
            relative = candidate.relative_to(root)
        except ValueError as exc:
            raise ValueError("manifest subdirectory escapes workspace root") from exc
        self._check_path_components(
            root,
            relative.parts,
            label="manifest subdirectory",
            allow_leaf_symlink=False,
        )
        self._check_real_containment(candidate, root, label="manifest subdirectory")
        return candidate

    @staticmethod
    def _canonical_checkout_origin(manifest_url: str, checkout_url: str) -> str:
        """Bind a checkout's ``origin`` to a manifest URL under explicit policy.

        Both values must first pass the strict HTTPS ``*.git`` canonicalizer.
        Credentials on an existing checkout's transport URL are discarded for
        identity comparison (never logged or retained); they are transport
        material, not repository identity. The manifest itself may never carry
        credentials. A checkout origin may omit the conventional ``.git``
        suffix; that suffix is added only for this comparison.
        Canonical URLs compare byte-for-byte for ordinary hosts. GitHub's owner
        and repository path is case-insensitive, so a case-only path mismatch
        (the existing ``atlassian-agent`` checkout has this exact condition) is
        accepted after canonicalization while retaining the manifest spelling
        as the authoritative project-map key. A different path, authority, or
        scheme is a hard mismatch and aborts manifest loading.
        """
        manifest = canonical_repository_url(manifest_url)
        checkout = canonical_repository_url(
            Git._checkout_origin_identity_url(checkout_url)
        )
        if manifest == checkout:
            return manifest

        if Git._github_origin_case_match(manifest, checkout):
            logger.warning(
                "Manifest/checkout origin differs only by GitHub path case; "
                "accepting under the explicit case-insensitive GitHub policy"
            )
            return manifest
        raise ValueError(
            "manifest repository URL does not match checkout origin after "
            "canonicalization"
        )

    @staticmethod
    def _checkout_origin_identity_url(checkout_url: str) -> str:
        """Remove transport credentials and add the conventional git suffix."""
        safe_url = Git._checkout_origin_without_credentials(checkout_url)
        parsed = urlsplit(safe_url)
        if parsed.path and not parsed.path.endswith(".git"):
            safe_url = urlunsplit(
                (
                    parsed.scheme,
                    parsed.netloc,
                    f"{parsed.path}.git",
                    parsed.query,
                    parsed.fragment,
                )
            )
        return safe_url

    @staticmethod
    def _checkout_origin_without_credentials(checkout_url: str) -> str:
        """Return an origin URL with userinfo removed for identity comparison."""
        parsed = urlsplit(checkout_url)
        if parsed.username is None and parsed.password is None:
            return checkout_url
        try:
            safe_netloc = parsed.hostname or ""
            if parsed.port is not None:
                safe_netloc = f"{safe_netloc}:{parsed.port}"
            return urlunsplit(
                (
                    parsed.scheme,
                    safe_netloc,
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
        except ValueError as exc:
            raise ValueError("checkout origin has an invalid authority") from exc

    @staticmethod
    def _github_origin_case_match(manifest: str, checkout: str) -> bool:
        """Apply the explicit case-insensitive path policy for GitHub only."""
        manifest_parts = urlsplit(manifest)
        checkout_parts = urlsplit(checkout)
        return bool(
            manifest_parts.hostname in _CASE_INSENSITIVE_ORIGIN_HOSTS
            and checkout_parts.hostname == manifest_parts.hostname
            and manifest_parts.scheme == checkout_parts.scheme
            and manifest_parts.netloc == checkout_parts.netloc
            and manifest_parts.path.casefold() == checkout_parts.path.casefold()
        )

    def _validate_manifest_checkout_origins(self) -> None:
        """Reject an existing checkout whose ``origin`` is not its manifest URL."""
        for manifest_url, project_path in sorted(self.project_map.items()):
            candidate = Path(os.path.abspath(os.path.expanduser(project_path)))
            if not candidate.exists() and not candidate.is_symlink():
                # Fresh-machine setup creates the manifest-declared directory
                # only after loading succeeds; there is no origin to bind yet.
                continue
            validated = self._validate_workspace_path(
                project_path,
                label=f"manifest checkout {_project_label(project_path)!r}",
            )
            checkout = Path(validated)
            git_marker = checkout / ".git"
            if git_marker.is_symlink():
                raise ValueError(
                    f"manifest checkout contains symlink component {git_marker}"
                )
            if not git_marker.exists():
                continue
            checkout_origin = self._git_remote_url(str(checkout))
            if checkout_origin is None:
                raise ValueError(
                    f"manifest checkout {_project_label(checkout)!r} has no origin"
                )
            self._canonical_checkout_origin(manifest_url, checkout_origin)

    def discover_projects(self) -> dict[str, str]:
        """
        Scan self.path for immediate subdirectories containing a .git folder.
        Populates and returns self.project_map.
        """
        self.project_map = {}
        self._project_categories = {}
        expanded_path = os.path.abspath(os.path.expanduser(self.path))
        if not os.path.exists(expanded_path):
            return self.project_map

        try:
            for item in os.listdir(expanded_path):
                full_path = os.path.join(expanded_path, item)
                if not os.path.isdir(full_path) or not os.path.exists(
                    os.path.join(full_path, ".git")
                ):
                    continue
                remote_url = self._git_remote_url(full_path) or f"local://{item}"
                self.project_map[remote_url] = os.path.abspath(full_path)

            logger.info(
                f"Auto-discovered {len(self.project_map)} git repositories in {expanded_path}"
            )
        except Exception as e:
            logger.error("Operation failed: error_type=%s", type(e).__name__)

        return self.project_map

    def _parse_subdirectories(
        self,
        subdirs: dict[str, SubdirectoryConfig],
        current_path: str,
        *,
        category_path: tuple[str, ...],
        seen_repository_urls: set[str],
    ) -> dict[str, str]:
        """Helper to recursively parse subdirectories and collect repository paths."""
        project_map = {}
        for name, data in subdirs.items():
            new_path = str(self._validate_manifest_subdirectory(name, current_path))
            repo_category = (*category_path, name)

            for repo in data.repositories:
                repo_url, repo_name = self._manifest_repository_identity(
                    repo, seen_repository_urls
                )
                project_map[repo_url] = os.path.join(new_path, repo_name)
                self._project_categories[repo_url] = repo_category

            if data.subdirectories:
                project_map.update(
                    self._parse_subdirectories(
                        data.subdirectories,
                        new_path,
                        category_path=repo_category,
                        seen_repository_urls=seen_repository_urls,
                    )
                )

        return project_map

    @staticmethod
    def _manifest_repository_identity(
        repository: RepositoryConfig, seen_repository_urls: set[str]
    ) -> tuple[str, str]:
        """Canonical URL/name identity, rejecting duplicates before insertion."""
        expanded = _expand_required_environment(
            repository.url,
            label="repository origin",
        )
        url = canonical_repository_url(expanded)
        if url in seen_repository_urls:
            raise ValueError(f"duplicate repository URL: {url}")
        seen_repository_urls.add(url)
        return url, repository_name(url)

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

    @staticmethod
    def _collect_consolidated_skills(
        paths: list[str],
        package: str,
        subdir: str,
        wanted: tuple[str, ...],
        fallback: Callable[[], list[str]],
    ) -> None:
        """Append the wanted skills packaged under ``<package>/<subdir>``.

        The packaged-resource lookup is preferred; if it fails the caller's own
        path resolver is filtered by basename instead.
        """
        try:
            base = files(package) / subdir
            for name in wanted:
                skill_path = base / name
                if skill_path.joinpath("SKILL.md").is_file():
                    paths.append(str(skill_path))
        except Exception as e:
            logger.warning("Operation failed: error_type=%s", type(e).__name__)
            paths.extend(p for p in fallback() if os.path.basename(p) in wanted)

    def get_consolidated_skill_paths(self) -> list[str]:
        """
        Returns absolute paths to the 15 specific building and documentation skills.
        """
        paths: list[str] = []

        if get_universal_skills_path:
            self._collect_consolidated_skills(
                paths,
                "universal_skills",
                "skills",
                _CONSOLIDATED_UNIVERSAL_SKILLS,
                get_universal_skills_path,
            )

        if get_skill_graphs_path:
            self._collect_consolidated_skills(
                paths,
                "skill_graphs",
                "skill_graphs",
                _CONSOLIDATED_SKILL_GRAPHS,
                lambda: get_skill_graphs_path(default_enabled=True),
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


def main() -> int:
    """Run the Repository Manager command-line adapter."""
    return _run_cli(runtime_from_module())


if __name__ == "__main__":
    raise SystemExit(main())
