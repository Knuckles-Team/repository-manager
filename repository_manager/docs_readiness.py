"""Bounded fleet action for canonical agent documentation readiness artifacts.

The action consumes only repository identities declared by the canonical
``workspace.yml``.  It deliberately delegates generation to the installed
``universal-skills`` agent-readiness builder; Repository Manager owns only
selection, git safety, action policy, and privacy-safe result shaping.
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
_ERROR_PREFIXES = frozenset(
    {
        "applicability",
        "artifacts",
        "budgets",
        "capability",
        "content",
        "full",
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

# These are explicit workspace identities, not a filesystem scan.  They are
# shared scaffolding or deployment inputs and therefore cannot be agent-doc
# publication targets.  Keeping the policy here makes an accidental broad
# ``workspace.yml`` fan-out visible and testable.
NON_PUBLISHABLE_PREFIXES = (
    "images",
    "services",
)
NON_PUBLISHABLE_IDENTITIES = frozenset(
    {
        "pipelines",
        "gitlab-pipelines",
        "agents/tests",
        "agent-packages/agents/tests",
        "agent-packages/skills/universal-skills",
        "agent-packages/skills/skill-graphs",
    }
)
_OUTPUT_ROOTS = frozenset(
    {
        "llms.txt",
        "llms-full.txt",
        "llms-sections",
        "markdown-mirror-manifest.json",
        "agent-readiness-manifest.json",
    }
)


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
    if identifier in NON_PUBLISHABLE_IDENTITIES:
        return True
    first = identifier.split("/", 1)[0]
    return first in NON_PUBLISHABLE_PREFIXES


def _select_repositories(
    identities: tuple[RepositoryIdentity, ...], repository: object
) -> tuple[RepositoryIdentity, ...]:
    if repository is None:
        return identities
    identifier = _safe_identifier(repository)
    selected = tuple(item for item in identities if item.identifier == identifier)
    if not selected:
        raise DocsReadinessError("repository-not-in-manifest")
    return selected


def _git_status(path: Path) -> tuple[bool, str]:
    """Return clean state and a bounded refusal reason without exposing output."""

    try:
        probe = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "--show-toplevel"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "repository-not-git"
    if probe.returncode != 0:
        return False, "repository-not-git"
    try:
        top = Path(probe.stdout.strip()).resolve(strict=True)
        top.relative_to(path)
    except (OSError, RuntimeError, ValueError):
        return False, "repository-root-mismatch"
    if top != path:
        return False, "repository-root-mismatch"
    try:
        status = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return False, "repository-status-unavailable"
    if status.returncode != 0:
        return False, "repository-status-unavailable"
    return not bool(status.stdout), "repository-dirty" if status.stdout else ""


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
        if first != "llms-sections" and len(relative.parts) != 1:
            raise DocsReadinessError(f"generator-{field}-invalid")
        if first == "llms-sections" and (
            len(relative.parts) < 2 or relative.parts[-1] != "llms.txt"
        ):
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


def _artifact_files(root: Path) -> dict[str, bytes]:
    """Read only the generator-owned output namespace from one tree."""

    files: dict[str, bytes] = {}
    total_bytes = 0
    for relative_root in sorted(_OUTPUT_ROOTS):
        candidate = root / relative_root
        if not candidate.exists():
            continue
        if candidate.is_symlink():
            raise DocsReadinessError("output-symlink")
        if candidate.is_file():
            try:
                metadata = candidate.lstat()
            except OSError as exc:
                raise DocsReadinessError("output-unavailable") from exc
            if metadata.st_nlink != 1:
                raise DocsReadinessError("output-not-regular")
            payload = candidate.read_bytes()
            total_bytes += len(payload)
            if total_bytes > MAX_OUTPUT_BYTES:
                raise DocsReadinessError("output-oversize")
            files[relative_root] = payload
            continue
        if not candidate.is_dir():
            raise DocsReadinessError("output-not-regular")
        for child in sorted(candidate.rglob("*")):
            if child.is_symlink() or not child.is_file():
                if child.is_symlink():
                    raise DocsReadinessError("output-symlink")
                continue
            try:
                metadata = child.lstat()
            except OSError as exc:
                raise DocsReadinessError("output-unavailable") from exc
            if metadata.st_nlink != 1:
                raise DocsReadinessError("output-not-regular")
            if len(files) >= MAX_OUTPUTS:
                raise DocsReadinessError("output-count-exceeded")
            payload = child.read_bytes()
            total_bytes += len(payload)
            if total_bytes > MAX_OUTPUT_BYTES:
                raise DocsReadinessError("output-oversize")
            files[child.relative_to(root).as_posix()] = payload
    return files


def _verify_current(generator: Generator, root: Path) -> dict[str, Any]:
    """Re-run the canonical writer in a temporary copy and compare bytes.

    The upstream ``check=True`` API returns a plan even when every planned byte
    already matches.  Comparing a generated temporary tree is therefore the
    only honest way for RM to expose a current/idempotent verification result
    without writing the caller's repository.
    """

    try:
        with tempfile.TemporaryDirectory(prefix="rm-docs-readiness-") as tmp:
            copy = Path(tmp) / "repository"
            shutil.copytree(
                root,
                copy,
                symlinks=True,
                ignore=shutil.ignore_patterns(".git"),
            )
            raw = generator(copy, check=False, adopt_existing=False)
            if not isinstance(raw, Mapping):
                return {"ok": False, "error_code": "generator-result-invalid"}
            generated = _safe_outputs(raw.get("generated"), "outputs")
            _safe_outputs(raw.get("planned", []), "planned")
            _safe_outputs(raw.get("pruned", []), "pruned")
            expected = _artifact_files(copy)
            actual = _artifact_files(root)
            if expected != actual:
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
        return _empty_result(identity, "excluded", "non-publishable-scaffolding")
    clean, reason = _git_status(identity.path)
    if not clean:
        return _empty_result(identity, "blocked", reason)
    try:
        _input_preflight(identity.path)
    except DocsReadinessError as exc:
        return _empty_result(identity, "blocked", _safe_reason(str(exc)))
    result = (
        _verify_current(generator, identity.path)
        if action == "verify"
        else _generator_result(generator, action, identity.path)
    )
    if not result.get("ok"):
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
        current = _verify_current(generator, identity.path)
        if not current.get("ok"):
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
    try:
        root = _safe_root(kwargs.get("workspace_root", Path.cwd()))
        manifest_value = kwargs.get("manifest_path")
        manifest = _contained_path(
            root,
            manifest_value if manifest_value is not None else root / "workspace.yml",
            "workspace-manifest",
        )
        identities = _manifest_repositories(manifest, root)
        selected = _select_repositories(identities, kwargs.get("repository"))
    except DocsReadinessError as exc:
        return {"ok": False, "action": action, "error_code": _safe_reason(str(exc))}
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
    "NON_PUBLISHABLE_IDENTITIES",
    "NON_PUBLISHABLE_PREFIXES",
    "dispatch",
]
