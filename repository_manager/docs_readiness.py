"""Bounded fleet action for canonical agent documentation readiness artifacts.

The action consumes only repository identities declared by the canonical
``workspace.yml``.  It deliberately delegates generation to the installed
``universal-skills`` agent-readiness builder; Repository Manager owns only
selection, git safety, action policy, bounded output verification, and
privacy-safe result shaping.  Readiness configuration is an explicit
per-repository prerequisite; this action does not synthesize rollout readiness.
"""

from __future__ import annotations

import hashlib
import importlib.resources
import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path, PurePosixPath
from types import ModuleType
from typing import Any, cast

import yaml

from repository_manager.workspace_manifest import (
    WorkspaceManifestError,
    _manifest_content,
    _repository_entries,
    _validate_no_secrets,
)

ACTIONS = ("preview", "apply", "verify")
MAX_REPOSITORIES = 512
MAX_OUTPUTS = 256
MAX_OUTPUT_BYTES = 8_000_000
MAX_GIT_STATUS_BYTES = 1_000_000
EXPECTED_AGENT_FLEET_COUNT = 76
_AGENT_PACKAGES_PREFIX = "agent-packages/"
_ERROR_PREFIXES = frozenset(
    {
        "applicability",
        "artifacts",
        "budgets",
        "capability",
        "content",
        "full",
        "fleet",
        "generator",
        "maturity",
        "mkdocs",
        "operation",
        "output",
        "previous",
        "project",
        "repository",
        "root",
        "skills",
        "workspace",
    }
)

# These are explicit workspace identities, not a filesystem scan.  The default
# action is deliberately scoped to the agent-packages fleet; within that fleet
# only the test fixture identity is non-publishable.  Shared skills are real
# publishable package identities and must not be silently dropped.
NON_PUBLISHABLE_PREFIXES: tuple[str, ...] = ()
NON_PUBLISHABLE_IDENTITIES = frozenset(
    {
        "agent-packages/agents/tests",
    }
)
_OUTPUT_ROOTS = frozenset(
    {
        "llms.txt",
        "llms-full.txt",
        "llms-sections",
        "markdown-mirror-manifest.json",
        "agent-readiness-manifest.json",
        # universal-skills 1.3.0's generator also emits the standard
        # agent-skills discovery document. The allowlist here was written
        # against an older generator output set, so every preview/apply
        # refused with `generator-outputs-outside-contract` once the newer
        # generator was installed -- the whole feature was unreachable.
        # Admitted under the SAME containment rules as `llms-sections`: a
        # fixed root with one fixed leaf name (see `_WELL_KNOWN_LEAF`), never
        # an open directory.
        ".well-known",
    }
)

#: The only file `.well-known/` may contain. Keeping the leaf pinned means
#: widening the root above did not widen what a generator can actually write.
_WELL_KNOWN_LEAF = "agent-skills.json"
_GIT_EXECUTABLE = shutil.which("git")


class DocsReadinessError(ValueError):
    """Raised when a readiness action cannot be admitted safely."""


@dataclass(frozen=True)
class RepositoryIdentity:
    """A manifest-declared repository and its private execution path."""

    identifier: str
    name: str
    path: Path


@dataclass(frozen=True)
class _GeneratorAuthority:
    """The one canonical builder module and its versioned resource identity."""

    module: ModuleType
    resource_name: str


Generator = Callable[..., dict[str, Any]]


def _safe_reason(value: object, fallback: str = "operation-failed") -> str:
    """Keep generator failures machine-readable and free of host details."""

    if isinstance(value, str) and value and len(value) <= 96:
        normalized = value.strip().lower()
        if (
            normalized
            and all(
                character.isalnum() or character in "-_." for character in normalized
            )
            and normalized.split("-", 1)[0] in _ERROR_PREFIXES
        ):
            return normalized
    return fallback


def _safe_version(value: object) -> str:
    if isinstance(value, str) and 1 <= len(value) <= 32:
        normalized = value.strip()
        if normalized and all(
            character.isalnum() or character in "-_." for character in normalized
        ):
            return normalized
    return "unknown"


def _regular_file(path: Path, label: str) -> None:
    try:
        metadata = path.lstat()
    except OSError as exc:
        raise DocsReadinessError(f"{label}-unavailable") from exc
    if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
        raise DocsReadinessError(f"{label}-not-regular")


def _safe_root(value: object) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise DocsReadinessError("workspace-root-required")
    root = Path(value).expanduser()
    if not root.is_absolute():
        root = Path.cwd() / root
    try:
        metadata = root.lstat()
        resolved = root.resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise DocsReadinessError("workspace-root-unavailable") from exc
    if (
        root.is_symlink()
        or root.absolute() != resolved
        or not root.is_dir()
        or metadata.st_nlink < 1
    ):
        raise DocsReadinessError("workspace-root-invalid")
    return resolved


def _reject_symlink_components(root: Path, candidate: Path, label: str) -> None:
    """Reject symlinks in an input path before resolving it."""

    absolute_root = root.absolute()
    absolute_candidate = candidate.absolute()
    try:
        relative = absolute_candidate.relative_to(absolute_root)
    except ValueError as exc:
        raise DocsReadinessError(f"{label}-containment") from exc
    cursor = absolute_root
    for part in relative.parts:
        cursor /= part
        try:
            if cursor.is_symlink():
                raise DocsReadinessError(f"{label}-symlink")
        except OSError as exc:
            raise DocsReadinessError(f"{label}-unavailable") from exc


def _contained_path(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise DocsReadinessError(f"{label}-required")
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    _reject_symlink_components(root, candidate, label)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, RuntimeError, ValueError) as exc:
        raise DocsReadinessError(f"{label}-containment") from exc
    _regular_file(resolved, label)
    return resolved


def _safe_identifier(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise DocsReadinessError("repository-identity-invalid")
    raw_parts = value.split("/")
    if any(
        not part or part in {".", ".."} or ord(character) < 32
        for part in raw_parts
        for character in part
    ):
        raise DocsReadinessError("repository-identity-invalid")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or not path.parts
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise DocsReadinessError("repository-identity-invalid")
    return path.as_posix()


def _manifest_repositories(
    manifest_path: Path, root: Path
) -> tuple[RepositoryIdentity, ...]:
    try:
        content, data = _manifest_content(manifest_path)
        del content
        _validate_no_secrets(data)
        entries = _repository_entries(data)
    except (OSError, WorkspaceManifestError, yaml.YAMLError) as exc:
        raise DocsReadinessError("workspace-manifest-invalid") from exc
    # Fleet selection is identity-driven: only repositories declared beneath
    # the exact agent-packages manifest subtree are in scope.  Do not discover
    # siblings from the filesystem and do not validate unrelated services or
    # deployment inputs while constructing this selection.
    entries = [
        entry
        for entry in entries
        if entry.identifier.startswith(_AGENT_PACKAGES_PREFIX)
    ]
    if len(entries) > MAX_REPOSITORIES:
        raise DocsReadinessError("workspace-manifest-too-large")

    identities: list[RepositoryIdentity] = []
    seen: set[str] = set()
    for entry in entries:
        identifier = _safe_identifier(entry.identifier)
        if identifier in seen:
            raise DocsReadinessError("workspace-manifest-duplicate")
        seen.add(identifier)
        target = root.joinpath(*PurePosixPath(identifier).parts)
        _reject_symlink_components(root, target, "repository-path")
        try:
            resolved = target.resolve(strict=False)
            resolved.relative_to(root)
        except (OSError, RuntimeError, ValueError) as exc:
            raise DocsReadinessError("repository-path-containment") from exc
        if target.exists() and (resolved != target or target.is_symlink()):
            raise DocsReadinessError("repository-path-invalid")
        identities.append(
            RepositoryIdentity(identifier=identifier, name=entry.name, path=resolved)
        )
    return tuple(identities)


def _is_non_publishable(identifier: str) -> bool:
    return identifier in NON_PUBLISHABLE_IDENTITIES


def _validate_fleet_selection(
    identities: tuple[RepositoryIdentity, ...],
    *,
    expected_count: int | None,
    validate_paths: bool,
) -> tuple[RepositoryIdentity, ...]:
    """Return the manifest-owned publishable fleet or refuse selection drift.

    The manifest is the sole selection authority.  The count is intentionally
    pinned for the production fleet so an added, removed, or accidentally
    reclassified manifest entry cannot cause a partial rollout.  A missing
    checkout is the same class of drift: it is reported before any generator
    invocation rather than silently processing a subset.  Unlisted filesystem
    siblings are never consulted.
    """

    agent_fleet = tuple(
        item
        for item in identities
        if item.identifier.startswith(_AGENT_PACKAGES_PREFIX)
    )
    selected = tuple(
        item for item in agent_fleet if not _is_non_publishable(item.identifier)
    )
    if expected_count is not None:
        if (
            type(expected_count) is not int
            or expected_count < 0
            or expected_count > MAX_REPOSITORIES
            or len(selected) != expected_count
        ):
            raise DocsReadinessError("fleet-selection-drift")
    if validate_paths and any(
        not item.path.is_dir() or item.path.is_symlink() for item in selected
    ):
        raise DocsReadinessError("fleet-selection-drift")
    return selected


def _select_repositories(
    identities: tuple[RepositoryIdentity, ...],
    repository: object,
    *,
    expected_count: int | None = None,
    validate_paths: bool = False,
) -> tuple[RepositoryIdentity, ...]:
    agent_fleet = _validate_fleet_selection(
        identities,
        expected_count=expected_count,
        validate_paths=validate_paths,
    )
    if repository is None:
        return agent_fleet
    identifier = _safe_identifier(repository)
    selected = tuple(
        item
        for item in identities
        if item.identifier.startswith(_AGENT_PACKAGES_PREFIX)
        and item.identifier == identifier
    )
    if selected and _is_non_publishable(identifier):
        raise DocsReadinessError("repository-not-publishable")
    if not selected:
        raise DocsReadinessError("repository-not-in-manifest")
    return selected


def _git_snapshot(
    path: Path,
) -> tuple[bool, str, frozenset[str] | None]:
    """Return clean state plus bounded relative dirty paths for mutation checks."""

    if _GIT_EXECUTABLE is None:
        return False, "repository-git-unavailable", None
    try:
        probe = subprocess.run(
            [_GIT_EXECUTABLE, "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "repository-not-git", None
    if probe.returncode != 0:
        return False, "repository-not-git", None
    try:
        top = Path(probe.stdout.strip()).resolve(strict=True)
        top.relative_to(path)
    except (OSError, RuntimeError, ValueError):
        return False, "repository-root-mismatch", None
    if top != path:
        return False, "repository-root-mismatch", None
    try:
        status = subprocess.run(
            [
                _GIT_EXECUTABLE,
                "-C",
                str(path),
                "status",
                "--porcelain",
                "--untracked-files=all",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "repository-status-unavailable", None
    if status.returncode != 0:
        return False, "repository-status-unavailable", None
    if len(status.stdout) > MAX_GIT_STATUS_BYTES:
        return False, "repository-status-too-large", None
    if not status.stdout:
        return True, "", frozenset()

    dirty: set[str] = set()
    for line in status.stdout.splitlines():
        if len(line) < 4:
            return False, "repository-status-unavailable", None
        raw_paths = line[3:]
        candidates = raw_paths.split(" -> ")
        for raw_path in candidates:
            try:
                relative = PurePosixPath(raw_path)
            except (TypeError, ValueError):
                return False, "repository-status-unavailable", None
            if (
                relative.is_absolute()
                or not relative.parts
                or any(part in {"", ".", ".."} for part in relative.parts)
            ):
                return False, "repository-status-unavailable", None
            dirty.add(relative.as_posix())
            if len(dirty) > MAX_OUTPUTS:
                return False, "repository-status-too-large", None
    return False, "repository-dirty", frozenset(dirty)


def _git_status(path: Path) -> tuple[bool, str]:
    """Return clean state and a bounded refusal reason without exposing output."""

    clean, reason, _ = _git_snapshot(path)
    return clean, reason


def _load_generator_authority() -> _GeneratorAuthority:
    """Load exactly one generator from the installed canonical package."""

    def find_resources(root: Traversable, filename: str) -> tuple[Traversable, ...]:
        found: list[Traversable] = []
        for child in root.iterdir():
            if child.name == filename:
                found.append(child)
            elif child.is_dir():
                found.extend(find_resources(child, filename))
        return tuple(found)

    try:
        package = importlib.resources.files("universal_skills")
        candidates = find_resources(package, "agent_readiness.py")
        schemas = find_resources(package, "agent_readiness_schema.json")
    except (ImportError, ModuleNotFoundError, OSError) as exc:
        raise DocsReadinessError("generator-unavailable") from exc
    if len(candidates) != 1 or len(schemas) != 1:
        raise DocsReadinessError("generator-authority-ambiguous")
    candidate = candidates[0]
    try:
        with importlib.resources.as_file(candidate) as script_path:
            spec = importlib.util.spec_from_file_location(
                "_repository_manager_agent_readiness", script_path
            )
            if spec is None or spec.loader is None:
                raise DocsReadinessError("generator-unavailable")
            module = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = module
            spec.loader.exec_module(module)
    except DocsReadinessError:
        raise
    except (ImportError, OSError, ValueError) as exc:
        raise DocsReadinessError("generator-unavailable") from exc
    generate = getattr(module, "generate", None)
    if not callable(generate):
        raise DocsReadinessError("generator-authority-invalid")
    return _GeneratorAuthority(module=module, resource_name=str(candidate))


def _canonical_generator() -> Generator:
    authority = _load_generator_authority()
    return cast(Generator, authority.module.generate)


def _input_preflight(path: Path) -> None:
    _contained_path(path, "docs/agent-readiness.json", "applicability")
    _contained_path(path, "mkdocs.yml", "mkdocs")


def _safe_outputs(value: object, field: str) -> list[str]:
    if not isinstance(value, list) or len(value) > MAX_OUTPUTS:
        raise DocsReadinessError(f"generator-{field}-invalid")
    outputs: list[str] = []
    for raw in value:
        if not isinstance(raw, str) or not raw or "\\" in raw:
            raise DocsReadinessError(f"generator-{field}-invalid")
        relative = PurePosixPath(raw)
        if relative.is_absolute() or any(
            part in {"", ".", ".."} for part in relative.parts
        ):
            raise DocsReadinessError(f"generator-{field}-invalid")
        first = relative.parts[0]
        if first not in _OUTPUT_ROOTS:
            raise DocsReadinessError(f"generator-{field}-outside-contract")
        if first not in {"llms-sections", ".well-known"} and len(relative.parts) != 1:
            raise DocsReadinessError(f"generator-{field}-invalid")
        if first == "llms-sections" and (
            len(relative.parts) < 2 or relative.parts[-1] != "llms.txt"
        ):
            raise DocsReadinessError(f"generator-{field}-invalid")
        if first == ".well-known" and relative.parts[1:] != (_WELL_KNOWN_LEAF,):
            raise DocsReadinessError(f"generator-{field}-invalid")
        outputs.append(relative.as_posix())
    if len(set(outputs)) != len(outputs):
        raise DocsReadinessError(f"generator-{field}-duplicate")
    return sorted(outputs)


def _result_digest(result: Mapping[str, Any]) -> str:
    payload = {
        "schema_version": result.get("schema_version"),
        "generator_version": result.get("generator_version"),
        "generated": result.get("generated"),
        "provenance": result.get("provenance"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _generator_result(
    generator: Generator,
    action: str,
    root: Path,
) -> dict[str, Any]:
    check = action != "apply"
    try:
        raw = generator(root, check=check, adopt_existing=False)
    except Exception as exc:  # noqa: BLE001 - third-party generator boundary
        del exc
        return {"ok": False, "error_code": "generator-failed"}
    if not isinstance(raw, Mapping):
        return {"ok": False, "error_code": "generator-result-invalid"}
    try:
        generated = _safe_outputs(raw.get("generated"), "outputs")
        planned = _safe_outputs(raw.get("planned", []), "planned")
        pruned = _safe_outputs(raw.get("pruned", []), "pruned")
        if action == "verify" and (planned or pruned):
            return {
                "ok": False,
                "error_code": "artifacts-not-current",
                "generated": generated,
                "planned": planned,
                "pruned": pruned,
            }
        digest = _result_digest(raw)
    except (DocsReadinessError, TypeError, ValueError) as exc:
        return {"ok": False, "error_code": _safe_reason(str(exc))}
    return {
        "ok": True,
        "generated": generated,
        "planned": planned,
        "pruned": pruned,
        "generator_version": _safe_version(
            raw.get("generator_version")
            or (raw.get("provenance") or {}).get("generator_version"),
        ),
        "provenance_digest": digest,
    }


def _artifact_files(
    root: Path, *, strict: bool = False, allowed: set[str] | None = None
) -> dict[str, bytes]:
    """Read the bounded generator-owned output namespace from one tree.

    ``strict`` is used for the temporary output directory: every top-level
    entry and every file below ``llms-sections`` must be one of the generator's
    declared outputs (plus its mandatory provenance manifest).  The repository
    target is intentionally read in non-strict mode because its source tree
    contains ordinary project files outside the generated namespace.
    """

    files: dict[str, bytes] = {}
    total_bytes = 0

    if strict:
        if allowed is None:
            raise DocsReadinessError("output-contract-invalid")
        try:
            top_level = sorted(root.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise DocsReadinessError("output-unavailable") from exc
        if any(child.name not in _OUTPUT_ROOTS for child in top_level):
            raise DocsReadinessError("output-outside-contract")

    def visit(directory: Path) -> None:
        nonlocal total_bytes
        try:
            children = sorted(directory.iterdir(), key=lambda child: child.name)
        except OSError as exc:
            raise DocsReadinessError("output-unavailable") from exc
        if len(children) > MAX_OUTPUTS:
            raise DocsReadinessError("output-count-exceeded")
        for child in children:
            if child.is_symlink():
                raise DocsReadinessError("output-symlink")
            if child.is_dir():
                visit(child)
                continue
            if not child.is_file():
                raise DocsReadinessError("output-not-regular")
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise DocsReadinessError("output-unavailable") from exc
            if metadata.st_nlink != 1:
                raise DocsReadinessError("output-not-regular")
            if len(files) >= MAX_OUTPUTS:
                raise DocsReadinessError("output-count-exceeded")
            if total_bytes + metadata.st_size > MAX_OUTPUT_BYTES:
                raise DocsReadinessError("output-oversize")
            try:
                payload = child.read_bytes()
            except OSError as exc:
                raise DocsReadinessError("output-unavailable") from exc
            total_bytes += len(payload)
            if total_bytes > MAX_OUTPUT_BYTES:
                raise DocsReadinessError("output-oversize")
            files[child.relative_to(root).as_posix()] = payload

    for relative_root in sorted(_OUTPUT_ROOTS):
        candidate = root / relative_root
        if candidate.is_symlink():
            raise DocsReadinessError("output-symlink")
        if not candidate.exists():
            continue
        if candidate.is_file():
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise DocsReadinessError("output-unavailable") from exc
            if metadata.st_nlink != 1:
                raise DocsReadinessError("output-not-regular")
            if len(files) >= MAX_OUTPUTS:
                raise DocsReadinessError("output-count-exceeded")
            if total_bytes + metadata.st_size > MAX_OUTPUT_BYTES:
                raise DocsReadinessError("output-oversize")
            try:
                payload = candidate.read_bytes()
            except OSError as exc:
                raise DocsReadinessError("output-unavailable") from exc
            total_bytes += len(payload)
            if total_bytes > MAX_OUTPUT_BYTES:
                raise DocsReadinessError("output-oversize")
            files[relative_root] = payload
            continue
        if not candidate.is_dir():
            raise DocsReadinessError("output-not-regular")
        visit(candidate)
    if strict:
        assert allowed is not None
        observed = set(files)
        if observed - allowed:
            raise DocsReadinessError("output-outside-contract")
        if allowed - observed:
            raise DocsReadinessError("output-contract-incomplete")
    return files


def _remove_artifact_path(path: Path, counters: list[int]) -> None:
    """Remove one generator-owned path during bounded rollback.

    This helper is deliberately limited to the known output namespace.  It
    refuses to follow symlinks and caps the number of filesystem entries it
    will inspect, so a malformed generator cannot turn rollback into an
    unbounded recursive delete.
    """

    if not path.exists() and not path.is_symlink():
        return
    counters[0] += 1
    if counters[0] > MAX_OUTPUTS:
        raise DocsReadinessError("rollback-output-count-exceeded")
    if path.is_symlink() or path.is_file():
        path.unlink()
        return
    if not path.is_dir():
        raise DocsReadinessError("rollback-output-not-regular")
    children = sorted(path.iterdir(), key=lambda child: child.name)
    if len(children) > MAX_OUTPUTS:
        raise DocsReadinessError("rollback-output-count-exceeded")
    for child in children:
        _remove_artifact_path(child, counters)
    path.rmdir()


def _restore_artifacts(root: Path, snapshot: Mapping[str, bytes]) -> bool:
    """Restore only generator-owned artifacts after a failed apply.

    The source tree is never copied.  Rollback clears the bounded governed
    output roots and rewrites the byte snapshot captured immediately before
    apply.  Any malformed path or filesystem failure returns ``False`` so the
    caller can fail closed instead of claiming a successful rollback.
    """

    try:
        counters = [0]
        for relative_root in sorted(_OUTPUT_ROOTS):
            _remove_artifact_path(root / relative_root, counters)
        safe_snapshot = _safe_outputs(sorted(snapshot), "rollback")
        if set(safe_snapshot) != set(snapshot):
            raise DocsReadinessError("rollback-output-contract-invalid")
        for relative in safe_snapshot:
            path = root / relative
            _reject_symlink_components(root, path, "rollback-output")
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.parent.is_symlink() or not path.parent.is_dir():
                raise DocsReadinessError("rollback-output-parent-invalid")
            path.write_bytes(snapshot[relative])
        return True
    except (DocsReadinessError, OSError, RuntimeError):
        return False


def _verify_current(
    generator: Generator,
    root: Path,
    *,
    allowed_dirty_paths: set[str] | None = None,
) -> dict[str, Any]:
    """Re-run the canonical writer into a bounded output staging directory.

    The upstream ``check=True`` API returns a plan even when every planned byte
    already matches.  The generator's explicit ``output_dir`` seam lets RM
    compare canonical output bytes without copying or mutating the repository's
    source tree (including ignored environments, build trees, and caches).
    Git status is snapshotted before and after the call so tracked or untracked
    target mutations fail closed; apply may carry only its already-published
    generated artifact paths as an explicit dirty-set allowance.
    """

    try:
        clean_before, status_reason, dirty_before = _git_snapshot(root)
        if dirty_before is None:
            return {
                "ok": False,
                "error_code": _safe_reason(status_reason, "verification-target-status"),
            }
        allowed_dirty = frozenset(allowed_dirty_paths or ())
        if not clean_before and not dirty_before <= allowed_dirty:
            return {"ok": False, "error_code": "verification-target-dirty"}
        if clean_before and allowed_dirty:
            # The caller may declare expected artifact paths, but an actually
            # clean target is still valid; the generator must not create source
            # mutations during the verification call.
            dirty_before = frozenset()
        with tempfile.TemporaryDirectory(prefix="rm-docs-readiness-") as tmp:
            staging = Path(tmp) / "artifacts"
            staging.mkdir()
            actual_before = _artifact_files(root)
            raw = generator(
                root,
                output_dir=staging,
                check=False,
                adopt_existing=True,
            )
            if not isinstance(raw, Mapping):
                return {"ok": False, "error_code": "generator-result-invalid"}
            generated = _safe_outputs(raw.get("generated"), "outputs")
            _safe_outputs(raw.get("planned", []), "planned")
            _safe_outputs(raw.get("pruned", []), "pruned")
            expected = _artifact_files(
                staging,
                strict=True,
                allowed=set(generated) | {"agent-readiness-manifest.json"},
            )
            _, status_after_reason, dirty_after = _git_snapshot(root)
            if dirty_after is None:
                return {
                    "ok": False,
                    "error_code": _safe_reason(
                        status_after_reason, "verification-target-status"
                    ),
                }
            if dirty_after != dirty_before:
                return {
                    "ok": False,
                    "error_code": "verification-mutated-target",
                }
            actual_after = _artifact_files(root)
            if actual_after != actual_before:
                return {
                    "ok": False,
                    "error_code": "verification-mutated-target",
                }
            if expected != actual_after:
                return {
                    "ok": False,
                    "error_code": "artifacts-not-current",
                }
            return {
                "ok": True,
                "generated": generated,
                "planned": [],
                "pruned": [],
                "provenance_digest": _result_digest(raw),
                "generator_version": _safe_version(
                    raw.get("generator_version")
                    or (raw.get("provenance") or {}).get("generator_version")
                ),
            }
    except DocsReadinessError as exc:
        return {
            "ok": False,
            "error_code": _safe_reason(str(exc), "verification-failed"),
        }
    except Exception as exc:  # noqa: BLE001 - filesystem/generator boundary
        del exc
        return {"ok": False, "error_code": "verification-failed"}


def _empty_result(
    identity: RepositoryIdentity, status: str, reason: str
) -> dict[str, Any]:
    return {"repository": identity.identifier, "status": status, "reason": reason}


def _dispatch_one(
    action: str,
    identity: RepositoryIdentity,
    generator: Generator,
) -> dict[str, Any]:
    if _is_non_publishable(identity.identifier):
        return _empty_result(identity, "excluded", "non-publishable-agent-tests")
    clean, reason = _git_status(identity.path)
    if not clean:
        return _empty_result(identity, "blocked", reason)
    try:
        _input_preflight(identity.path)
    except DocsReadinessError as exc:
        return _empty_result(identity, "blocked", _safe_reason(str(exc)))
    before_artifacts: dict[str, bytes] | None = None
    if action == "apply":
        try:
            before_artifacts = _artifact_files(identity.path)
        except DocsReadinessError as exc:
            return _empty_result(identity, "blocked", _safe_reason(str(exc)))
    result = (
        _verify_current(generator, identity.path)
        if action == "verify"
        else _generator_result(generator, action, identity.path)
    )
    if not result.get("ok"):
        if action == "apply" and before_artifacts is not None:
            if not _restore_artifacts(identity.path, before_artifacts):
                return _empty_result(identity, "blocked", "apply-rollback-failed")
        return {
            "repository": identity.identifier,
            "status": "blocked",
            "reason": result.get("error_code", "generator-failed"),
            **{
                key: result[key]
                for key in ("generated", "planned", "pruned")
                if key in result
            },
        }
    if action == "apply":
        current = _verify_current(
            generator,
            identity.path,
            allowed_dirty_paths=set(result.get("generated", []))
            | set(result.get("pruned", []))
            | {"agent-readiness-manifest.json"},
        )
        if not current.get("ok"):
            if before_artifacts is not None and not _restore_artifacts(
                identity.path, before_artifacts
            ):
                return _empty_result(identity, "blocked", "apply-rollback-failed")
            return {
                "repository": identity.identifier,
                "status": "blocked",
                "reason": current.get("error_code", "artifacts-not-current"),
            }
        result = {**result, "provenance_digest": current["provenance_digest"]}
    status = (
        "verified"
        if action == "verify"
        else ("planned" if action == "preview" else "applied")
    )
    return {
        "repository": identity.identifier,
        "status": status,
        "generated": result["generated"],
        "planned": result["planned"],
        "pruned": result["pruned"],
        "generator_version": result["generator_version"],
        "provenance_digest": result["provenance_digest"],
    }


def dispatch(action: str = "preview", **kwargs: Any) -> dict[str, Any]:
    """Run one read-only preview, guarded apply, or read-only verification.

    ``repository`` is always a full manifest-relative identity, never a
    basename or arbitrary path.  The private ``_generator`` keyword exists
    solely for offline tests; production callers use the canonical package
    authority resolved above.
    """

    if action not in ACTIONS:
        return {"ok": False, "error_code": "unknown-action", "actions": list(ACTIONS)}
    if action == "apply":
        if kwargs.get("repository") is None:
            return {
                "ok": False,
                "action": action,
                "error_code": "apply-requires-exact-repository",
            }
        if kwargs.get("confirm") is not True:
            return {
                "ok": False,
                "action": action,
                "error_code": "apply-confirmation-required",
            }
    try:
        root = _safe_root(kwargs.get("workspace_root", Path.cwd()))
        manifest_value = kwargs.get("manifest_path")
        manifest = _contained_path(
            root,
            manifest_value if manifest_value is not None else root / "workspace.yml",
            "workspace-manifest",
        )
        identities = _manifest_repositories(manifest, root)
        selected = _select_repositories(
            identities,
            kwargs.get("repository"),
            expected_count=EXPECTED_AGENT_FLEET_COUNT,
            validate_paths=True,
        )
    except DocsReadinessError as exc:
        return {"ok": False, "action": action, "error_code": _safe_reason(str(exc))}
    generator = kwargs.get("_generator")
    if generator is None:
        try:
            generator = _canonical_generator()
        except DocsReadinessError as exc:
            return {"ok": False, "action": action, "error_code": _safe_reason(str(exc))}
    if not callable(generator):
        return {"ok": False, "action": action, "error_code": "generator-invalid"}

    results = [_dispatch_one(action, identity, generator) for identity in selected]
    failures = [item for item in results if item.get("status") == "blocked"]
    return {
        "ok": not failures,
        "action": action,
        "repositories": results,
        "selected_count": len(results),
    }


__all__ = [
    "ACTIONS",
    "EXPECTED_AGENT_FLEET_COUNT",
    "NON_PUBLISHABLE_IDENTITIES",
    "NON_PUBLISHABLE_PREFIXES",
    "dispatch",
]
