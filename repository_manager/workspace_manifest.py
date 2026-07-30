"""Validate and safely mirror the canonical workspace manifest.

The workspace root manifest is human-authored.  The Graph-OS XDG manifest and
the packaged repository-manager seed are byte-for-byte mirrors, never sources
of truth.  This module deliberately does not discover or mutate a workspace:
callers provide the canonical source and may override every destination.
"""

from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import tempfile
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import yaml  # type: ignore[import-untyped]

_ENV_REFERENCE = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")
_LOCAL_HOST_SUFFIXES = (".arpa", ".internal", ".lan", ".local", ".localhost")
_SECRET_FIELD_NAMES = {
    "access_token",
    "api_key",
    "authorization",
    "client_secret",
    "password",
    "private_key",
    "refresh_token",
    "secret",
    "token",
}


def _portable_manifest_environment() -> dict[str, str | None]:
    """Return non-secret environment inputs supported by the portable manifest."""

    return {
        "AGENT_UTILITIES_WORKSPACE_ROOT": os.environ.get(
            "AGENT_UTILITIES_WORKSPACE_ROOT"
        ),
        "AGENT_UTILITIES_REPO_ORIGIN": os.environ.get("AGENT_UTILITIES_REPO_ORIGIN"),
        "AGENT_UTILITIES_SERVICE_DOMAIN_SUFFIX": os.environ.get(
            "AGENT_UTILITIES_SERVICE_DOMAIN_SUFFIX"
        ),
    }


class WorkspaceManifestError(ValueError):
    """Raised when a manifest cannot safely be validated or synchronized."""


@dataclass(frozen=True)
class ManifestDestinationStatus:
    """The digest state of one non-authoritative manifest copy."""

    role: str
    exists: bool
    digest: str | None
    matches_source: bool
    action: str


@dataclass(frozen=True)
class WorkspaceManifestReport:
    """A privacy-safe manifest validation and synchronization report."""

    source_digest: str
    profiles: tuple[str, ...]
    selectors: tuple[str, ...]
    selected_repositories: tuple[str, ...]
    destinations: tuple[ManifestDestinationStatus, ...]

    @property
    def synchronized(self) -> bool:
        """Whether every requested mirror has the canonical source digest."""

        return all(destination.matches_source for destination in self.destinations)

    def as_dict(self) -> dict[str, Any]:
        """Return an output-safe representation without local filesystem paths."""

        return {
            "source_digest": self.source_digest,
            "profiles": list(self.profiles),
            "selectors": list(self.selectors),
            "selected_repositories": list(self.selected_repositories),
            "destinations": [asdict(destination) for destination in self.destinations],
            "synchronized": self.synchronized,
        }


@dataclass(frozen=True)
class _ManifestRepository:
    """One repository's stable, workspace-relative selector identity."""

    identifier: str
    name: str


@dataclass(frozen=True)
class _DestinationSnapshot:
    """Pre-write state used to roll back a two-mirror synchronization."""

    role: str
    path: Path
    content: bytes | None
    mode: int | None


def default_runtime_manifest_path() -> Path:
    """Return the exact XDG runtime manifest path without creating directories."""

    config_home = os.environ.get("XDG_CONFIG_HOME")
    base = Path(config_home).expanduser() if config_home else Path.home() / ".config"
    return base / "agent-utilities" / "workspace.yml"


def default_packaged_seed_path() -> Path:
    """Return the installed package seed path; callers may override it for tests."""

    return Path(__file__).resolve().with_name("workspace.yml")


def _content_digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _manifest_content(source: Path) -> tuple[bytes, dict[str, Any]]:
    if not source.is_file():
        raise WorkspaceManifestError("Canonical workspace manifest was not found")
    try:
        content = source.read_bytes()
        data = yaml.safe_load(content.decode("utf-8"))
    except (OSError, UnicodeDecodeError, yaml.YAMLError) as exc:
        raise WorkspaceManifestError(
            "Canonical workspace manifest is invalid YAML"
        ) from exc
    if not isinstance(data, dict):
        raise WorkspaceManifestError(
            "Canonical workspace manifest must be a YAML mapping"
        )
    return content, data


def _validate_portable_manifest(data: dict[str, Any]) -> None:
    """Reject machine-local or secret-bearing values before packaging a mirror."""

    supported_references = set(_portable_manifest_environment())
    workspace_path = data.get("path")
    if not isinstance(workspace_path, str) or not workspace_path:
        raise WorkspaceManifestError("Canonical manifest path must be a string")
    if Path(workspace_path).is_absolute():
        raise WorkspaceManifestError(
            "Canonical manifest path must be relative or environment-referenced"
        )

    def visit(value: object, *, field: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                if not isinstance(key, str):
                    raise WorkspaceManifestError(
                        "Manifest mapping keys must be strings"
                    )
                normalized = key.lower().replace("-", "_")
                if normalized in _SECRET_FIELD_NAMES and child not in (None, ""):
                    raise WorkspaceManifestError(
                        "Canonical manifest must not contain embedded secrets"
                    )
                visit(child, field=normalized)
            return
        if isinstance(value, list):
            for child in value:
                visit(child, field=field)
            return
        if not isinstance(value, str):
            return
        references = set(_ENV_REFERENCE.findall(value))
        if references:
            if not references <= supported_references:
                raise WorkspaceManifestError(
                    "Canonical manifest uses an unsupported environment reference"
                )
            return
        if field not in {"domain", "endpoint", "host", "hostname", "url"}:
            return

        candidate = value.strip()
        host = candidate
        if "://" in candidate:
            parsed = urlsplit(candidate)
            if parsed.password is not None:
                raise WorkspaceManifestError(
                    "Canonical manifest URLs must not embed credentials"
                )
            host = parsed.hostname or ""
        elif "/" in candidate:
            return
        host = host.rstrip(".").lower()
        if not host:
            return
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            address = None
        if (
            host == "localhost"
            or host.endswith(_LOCAL_HOST_SUFFIXES)
            or (
                address is not None
                and (
                    address.is_private
                    or address.is_loopback
                    or address.is_link_local
                    or address.is_reserved
                )
            )
        ):
            raise WorkspaceManifestError(
                "Canonical manifest contains a machine-local endpoint; "
                "use an environment reference"
            )

    visit(data)


def _destination_path(path: str | Path, *, role: str) -> Path:
    destination = Path(path).expanduser()
    if destination.is_symlink():
        raise WorkspaceManifestError(f"{role} manifest must not be a symbolic link")
    if destination.exists() and not destination.is_file():
        raise WorkspaceManifestError(f"{role} manifest destination is not a file")
    return destination


def _string_list(value: object, *, label: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item for item in value
    ):
        raise WorkspaceManifestError(f"{label} must be a list of non-empty strings")
    return list(dict.fromkeys(value))


def _repository_entries(
    node: object,
    *,
    parent: tuple[str, ...] = (),
) -> list[_ManifestRepository]:
    if not isinstance(node, dict):
        raise WorkspaceManifestError("Each workspace directory must be a mapping")
    entries: list[_ManifestRepository] = []
    repositories = node.get("repositories", [])
    if not isinstance(repositories, list):
        raise WorkspaceManifestError("repositories must be a list")
    for repository in repositories:
        if not isinstance(repository, dict) or not isinstance(
            repository.get("url"), str
        ):
            raise WorkspaceManifestError("Each repository must have a string url")
        url = repository["url"].strip().rstrip("/")
        name = url.rsplit("/", 1)[-1].removesuffix(".git")
        if not name or name in {".", ".."}:
            raise WorkspaceManifestError("Each repository url must have a basename")
        entries.append(
            _ManifestRepository(identifier="/".join((*parent, name)), name=name)
        )
    subdirectories = node.get("subdirectories", {})
    if not isinstance(subdirectories, dict):
        raise WorkspaceManifestError("subdirectories must be a mapping")
    for directory, child in subdirectories.items():
        if (
            not isinstance(directory, str)
            or not directory
            or directory in {".", ".."}
            or "/" in directory
        ):
            raise WorkspaceManifestError(
                "Each subdirectory must have a safe single-segment name"
            )
        entries.extend(_repository_entries(child, parent=(*parent, directory)))
    return entries


def _validate_supported_keys(
    value: dict[str, Any],
    *,
    allowed: set[str],
    label: str,
) -> None:
    unsupported = sorted(set(value) - allowed)
    if unsupported:
        raise WorkspaceManifestError(
            f"{label} has unsupported fields: {', '.join(unsupported)}"
        )


def _optional_description(value: object, *, label: str) -> None:
    if value is not None and (not isinstance(value, str) or not value.strip()):
        raise WorkspaceManifestError(f"{label} must be a non-empty string")


def _profile_and_selector_data(
    data: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    raw_profiles = data.get("profiles", {})
    raw_selectors = data.get("selectors", {})
    if not isinstance(raw_profiles, dict):
        raise WorkspaceManifestError("profiles must be a mapping")
    if not isinstance(raw_selectors, dict):
        raise WorkspaceManifestError("selectors must be a mapping")

    profiles: dict[str, dict[str, Any]] = {}
    selectors: dict[str, dict[str, Any]] = {}
    for name, profile in raw_profiles.items():
        if not isinstance(name, str) or not name or not isinstance(profile, dict):
            raise WorkspaceManifestError("Each profile must be a named mapping")
        _validate_supported_keys(
            profile,
            allowed={"description", "selectors"},
            label=f"profiles.{name}",
        )
        _optional_description(
            profile.get("description"), label=f"profiles.{name}.description"
        )
        profile_selectors = _string_list(
            profile.get("selectors"), label=f"profiles.{name}.selectors"
        )
        if not profile_selectors:
            raise WorkspaceManifestError(
                f"profiles.{name}.selectors must name at least one selector"
            )
        profiles[name] = profile
    for name, selector in raw_selectors.items():
        if not isinstance(name, str) or not name or not isinstance(selector, dict):
            raise WorkspaceManifestError("Each selector must be a named mapping")
        _validate_supported_keys(
            selector,
            allowed={"description", "exclude", "include"},
            label=f"selectors.{name}",
        )
        _optional_description(
            selector.get("description"), label=f"selectors.{name}.description"
        )
        if "include" not in selector and "exclude" not in selector:
            raise WorkspaceManifestError(
                f"selectors.{name} must declare include or exclude"
            )
        _string_list(selector.get("include"), label=f"selectors.{name}.include")
        _string_list(selector.get("exclude"), label=f"selectors.{name}.exclude")
        selectors[name] = selector

    for name, profile in profiles.items():
        for selector_name in _string_list(
            profile.get("selectors"), label=f"profiles.{name}.selectors"
        ):
            if selector_name not in selectors:
                raise WorkspaceManifestError(
                    f"profiles.{name}.selectors references unknown selector: "
                    f"{selector_name}"
                )
    return profiles, selectors


def _resolve_repository_reference(
    reference: str,
    *,
    identifiers: set[str],
    aliases: dict[str, tuple[str, ...]],
    label: str,
) -> str:
    if reference in identifiers:
        return reference
    matches = aliases.get(reference, ())
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise WorkspaceManifestError(
            f"{label} uses ambiguous repository basename: {reference}; "
            "use a workspace-relative identifier"
        )
    raise WorkspaceManifestError(f"{label} references unknown repository: {reference}")


def _resolved_selector_members(
    selector: dict[str, Any],
    *,
    name: str,
    identifiers: tuple[str, ...],
    aliases: dict[str, tuple[str, ...]],
) -> set[str]:
    identifier_set = set(identifiers)
    include = _string_list(selector.get("include"), label=f"selectors.{name}.include")
    exclude = _string_list(selector.get("exclude"), label=f"selectors.{name}.exclude")
    for field_name, references in (("include", include), ("exclude", exclude)):
        if "*" in references and len(references) > 1:
            raise WorkspaceManifestError(
                f"selectors.{name}.{field_name} cannot combine '*' with repositories"
            )

    if "include" not in selector:
        selected = set(identifier_set)
    elif "*" in include:
        selected = set(identifier_set)
    else:
        selected = {
            _resolve_repository_reference(
                reference,
                identifiers=identifier_set,
                aliases=aliases,
                label=f"selectors.{name}.include",
            )
            for reference in include
        }
    if "*" in exclude:
        return set()
    selected.difference_update(
        _resolve_repository_reference(
            reference,
            identifiers=identifier_set,
            aliases=aliases,
            label=f"selectors.{name}.exclude",
        )
        for reference in exclude
    )
    return selected


def select_repositories(
    data: dict[str, Any],
    *,
    profile: str | None = None,
    selectors: Iterable[str] = (),
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Resolve visible bootstrap selectors to repository names.

    ``profiles.<name>.selectors`` names selector mappings. A selector's
    ``include`` and ``exclude`` lists contain workspace-relative repository
    identifiers, unambiguous basenames, or ``*``. A missing ``include`` means
    all repositories, while an explicit empty ``include`` means none. Multiple
    selectors are unioned after applying each selector's exclusions. Without a
    requested profile or selector, all declared repositories are visible.
    """

    entries = _repository_entries(data)
    repositories = tuple(entry.identifier for entry in entries)
    if len(set(repositories)) != len(repositories):
        raise WorkspaceManifestError(
            "Repository workspace-relative identifiers must be unique"
        )
    aliases: dict[str, tuple[str, ...]] = {}
    for entry in entries:
        aliases[entry.name] = (*aliases.get(entry.name, ()), entry.identifier)
    profiles, selector_map = _profile_and_selector_data(data)
    requested = list(dict.fromkeys(selectors))
    if profile is not None:
        if profile not in profiles:
            raise WorkspaceManifestError(f"Unknown workspace profile: {profile}")
        requested = list(
            dict.fromkeys(
                _string_list(
                    profiles[profile].get("selectors"),
                    label=f"profiles.{profile}.selectors",
                )
                + requested
            )
        )
    for selector_name in requested:
        if selector_name not in selector_map:
            raise WorkspaceManifestError(f"Unknown workspace selector: {selector_name}")

    for name, selector_config in selector_map.items():
        _resolved_selector_members(
            selector_config,
            name=name,
            identifiers=repositories,
            aliases=aliases,
        )

    if not requested:
        return repositories, tuple(sorted(profiles)), tuple(sorted(selector_map))

    selected: set[str] = set()
    for name in requested:
        selected.update(
            _resolved_selector_members(
                selector_map[name],
                name=name,
                identifiers=repositories,
                aliases=aliases,
            )
        )
    return (
        tuple(repository for repository in repositories if repository in selected),
        tuple(sorted(profiles)),
        tuple(sorted(selector_map)),
    )


def _stage_write(destination: Path, content: bytes, mode: int) -> Path:
    """Durably stage content next to a destination without replacing it."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(directory: Path) -> None:
    try:
        descriptor = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _restore_snapshots(snapshots: list[_DestinationSnapshot]) -> bool:
    """Best-effort rollback; return whether every replaced mirror was restored."""

    restored = True
    for snapshot in reversed(snapshots):
        staged: Path | None = None
        try:
            if snapshot.content is None:
                snapshot.path.unlink(missing_ok=True)
            else:
                staged = _stage_write(
                    snapshot.path,
                    snapshot.content,
                    snapshot.mode if snapshot.mode is not None else 0o600,
                )
                os.replace(staged, snapshot.path)
            _fsync_directory(snapshot.path.parent)
        except OSError:
            if staged is not None:
                staged.unlink(missing_ok=True)
            restored = False
    return restored


def _synchronize_destinations(
    snapshots: tuple[_DestinationSnapshot, ...],
    *,
    content: bytes,
    mode: int,
) -> None:
    """Replace all drifted mirrors or roll back every replacement."""

    staged: dict[str, Path] = {}
    try:
        for snapshot in snapshots:
            staged[snapshot.role] = _stage_write(snapshot.path, content, mode)
    except OSError as exc:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        raise WorkspaceManifestError(
            "Failed to stage workspace manifest mirrors; no mirror was replaced"
        ) from exc

    replaced: list[_DestinationSnapshot] = []
    try:
        for snapshot in snapshots:
            if snapshot.path.is_symlink():
                raise OSError("destination became a symbolic link")
            os.replace(staged[snapshot.role], snapshot.path)
            staged.pop(snapshot.role)
            replaced.append(snapshot)
            _fsync_directory(snapshot.path.parent)
        if any(snapshot.path.read_bytes() != content for snapshot in snapshots):
            raise OSError("destination verification failed")
    except OSError as exc:
        for temporary in staged.values():
            temporary.unlink(missing_ok=True)
        if not _restore_snapshots(replaced):
            raise WorkspaceManifestError(
                "Workspace manifest synchronization failed and rollback was incomplete"
            ) from exc
        raise WorkspaceManifestError(
            "Workspace manifest synchronization failed; replaced mirrors were rolled back"
        ) from exc


def synchronize_workspace_manifest(
    source: str | Path,
    *,
    runtime_destination: str | Path | None = None,
    seed_destination: str | Path | None = None,
    check: bool = False,
    dry_run: bool = False,
    profile: str | None = None,
    selectors: Iterable[str] = (),
) -> WorkspaceManifestReport:
    """Validate an authoritative manifest and optionally mirror its exact bytes.

    ``check`` and ``dry_run`` never write.  A normal call writes only the two
    exact destination paths and only when their SHA-256 digest differs.
    """

    if check and dry_run:
        raise WorkspaceManifestError("check and dry_run cannot be combined")

    source_candidate = Path(source).expanduser()
    if source_candidate.is_symlink():
        raise WorkspaceManifestError(
            "Canonical workspace manifest must not be a symbolic link"
        )
    source_path = source_candidate.resolve()
    source_bytes, data = _manifest_content(source_path)
    _validate_portable_manifest(data)
    selected, profiles, available_selectors = select_repositories(
        data, profile=profile, selectors=selectors
    )
    source_digest = _content_digest(source_bytes)
    source_mode = source_path.stat().st_mode & 0o777
    destinations = (
        (
            "runtime",
            _destination_path(
                runtime_destination
                if runtime_destination is not None
                else default_runtime_manifest_path(),
                role="runtime",
            ),
        ),
        (
            "packaged_seed",
            _destination_path(
                seed_destination
                if seed_destination is not None
                else default_packaged_seed_path(),
                role="packaged_seed",
            ),
        ),
    )
    resolved_destinations = tuple(
        path.resolve(strict=False) for _, path in destinations
    )
    if source_path in resolved_destinations:
        raise WorkspaceManifestError(
            "Canonical source must be distinct from both manifest mirrors"
        )
    if len(set(resolved_destinations)) != len(resolved_destinations):
        raise WorkspaceManifestError("Manifest mirror destinations must be distinct")

    snapshots: list[_DestinationSnapshot] = []
    initial: list[tuple[str, bool, str | None, bool]] = []
    for role, raw_destination in destinations:
        destination = raw_destination.expanduser()
        try:
            exists = destination.is_file()
            destination_bytes = destination.read_bytes() if exists else None
            destination_mode = destination.stat().st_mode & 0o777 if exists else None
        except OSError as exc:
            raise WorkspaceManifestError(
                f"Failed to inspect {role} manifest destination"
            ) from exc
        digest = (
            _content_digest(destination_bytes)
            if destination_bytes is not None
            else None
        )
        matches_source = digest == source_digest
        initial.append((role, exists, digest, matches_source))
        if not matches_source:
            snapshots.append(
                _DestinationSnapshot(
                    role=role,
                    path=destination,
                    content=destination_bytes,
                    mode=destination_mode,
                )
            )

    if not check and not dry_run and snapshots:
        _synchronize_destinations(
            tuple(snapshots),
            content=source_bytes,
            mode=source_mode,
        )

    statuses: list[ManifestDestinationStatus] = []
    for role, existed, initial_digest, initially_matched in initial:
        destination = dict(destinations)[role].expanduser()
        exists = existed
        digest = initial_digest
        matches_source = initially_matched
        if matches_source:
            action = "unchanged"
        elif check:
            action = "drift"
        elif dry_run:
            action = "would_update"
        else:
            exists = True
            try:
                digest = _content_digest(destination.read_bytes())
            except OSError as exc:
                raise WorkspaceManifestError(
                    f"Failed to verify {role} manifest destination"
                ) from exc
            matches_source = digest == source_digest
            action = "updated" if matches_source else "error"
            if not matches_source:
                raise WorkspaceManifestError(f"Failed to synchronize {role} manifest")
        statuses.append(
            ManifestDestinationStatus(
                role=role,
                exists=exists,
                digest=digest,
                matches_source=matches_source,
                action=action,
            )
        )
    return WorkspaceManifestReport(
        source_digest=source_digest,
        profiles=profiles,
        selectors=available_selectors,
        selected_repositories=selected,
        destinations=tuple(statuses),
    )
