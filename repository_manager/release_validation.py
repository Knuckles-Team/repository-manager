"""Fail-closed validation for release metadata and repository identities."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Callable
from email.headerregistry import Address
from pathlib import Path, PurePosixPath
from typing import Any, TypeGuard
from urllib.parse import SplitResult, quote, unquote, urlsplit

from packaging.licenses import InvalidLicenseExpression, canonicalize_license_expression
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import InvalidSpecifier, SpecifierSet
from packaging.version import InvalidVersion, Version

_PEP503_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*\Z")
_PEP503_SEPARATORS = re.compile(r"[-_.]+")
_PEP517_BACKEND = re.compile(
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*"
    r"(?::[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*)?\Z"
)
_ENTRY_POINT_REFERENCE = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*"
    r"(?::[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)?\Z"
)
_HOST_LABEL = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?\Z")
_PROJECT_KEYS = frozenset(
    {
        "authors",
        "classifiers",
        "dependencies",
        "description",
        "dynamic",
        "entry-points",
        "gui-scripts",
        "keywords",
        "license",
        "license-files",
        "maintainers",
        "name",
        "optional-dependencies",
        "readme",
        "requires-python",
        "scripts",
        "urls",
        "version",
    }
)
_DYNAMIC_KEYS = _PROJECT_KEYS - {"dynamic", "license-files", "name"}
_BUILD_SYSTEM_KEYS = frozenset({"backend-path", "build-backend", "requires"})


def _is_string(value: object) -> TypeGuard[str]:
    return type(value) is str and bool(value) and value == value.strip()


def _is_string_list(value: object) -> TypeGuard[list[str]]:
    return type(value) is list and all(_is_string(item) for item in value)


def _is_unique_string_list(value: object) -> TypeGuard[list[str]]:
    return _is_string_list(value) and len(value) == len(set(value))


def _normalized_name(value: object) -> str | None:
    if not _is_string(value):
        return None
    normalized = _PEP503_SEPARATORS.sub("-", value.lower())
    if value != normalized or _PEP503_NAME.fullmatch(value) is None:
        return None
    return normalized


def _valid_version(value: object) -> bool:
    if not _is_string(value):
        return False
    try:
        Version(value)
    except InvalidVersion:
        return False
    return True


def _valid_requires_python(value: object) -> bool:
    if not _is_string(value):
        return False
    try:
        SpecifierSet(value)
    except InvalidSpecifier:
        return False
    return True


def _valid_requirements(value: object, *, require_nonempty: bool = False) -> bool:
    if not _is_string_list(value) or (require_nonempty and not value):
        return False
    try:
        for item in value:
            Requirement(item)
    except InvalidRequirement:
        return False
    return True


def _valid_file_or_text_table(value: object) -> bool:
    if type(value) is not dict or not value:
        return False
    keys = set(value)
    return keys in ({"file"}, {"text"}) and all(
        _is_string(item) for item in value.values()
    )


def _valid_readme(value: object) -> bool:
    if _is_string(value):
        return True
    return _valid_readme_table(value)


def _valid_readme_table(value: object) -> bool:
    if type(value) is not dict or not value:
        return False
    allowed = {"file", "text", "content-type"}
    keys = set(value)
    sources = keys & {"file", "text"}
    return bool(
        keys <= allowed
        and len(sources) == 1
        and "content-type" in keys
        and all(_is_string(item) for item in value.values())
    )


def _valid_license(value: object) -> bool:
    if _is_string(value):
        return _valid_spdx_expression(value)
    return _valid_file_or_text_table(value)


def _valid_spdx_expression(value: str) -> bool:
    try:
        canonicalize_license_expression(value)
    except InvalidLicenseExpression:
        return False
    return True


def _valid_email(value: object) -> bool:
    if not _is_string(value):
        return False
    try:
        address = Address(addr_spec=value)
    except (TypeError, ValueError):
        return False
    return bool(address.username and address.domain and str(address) == value)


def _valid_people(value: object) -> bool:
    return bool(type(value) is list and all(_valid_person(person) for person in value))


def _valid_person(person: object) -> bool:
    if type(person) is not dict or not person:
        return False
    if not set(person) <= {"name", "email"}:
        return False
    if not all(_is_string(item) for item in person.values()):
        return False
    return "email" not in person or _valid_email(person["email"])


def _valid_string_map(value: object) -> bool:
    return bool(
        type(value) is dict
        and all(_is_string(key) and _is_string(item) for key, item in value.items())
    )


def _valid_entry_points(value: object) -> bool:
    return bool(
        type(value) is dict
        and all(
            _is_string(group) and _valid_entry_point_map(entries)
            for group, entries in value.items()
        )
    )


def _valid_entry_point_map(value: object) -> bool:
    return bool(
        type(value) is dict
        and all(
            _is_string(name)
            and _is_string(reference)
            and _ENTRY_POINT_REFERENCE.fullmatch(reference)
            for name, reference in value.items()
        )
    )


def _valid_optional_dependencies(value: object) -> bool:
    if type(value) is not dict:
        return False
    return all(
        _valid_extra_name(extra) and _valid_requirements(requirements)
        for extra, requirements in value.items()
    )


def _valid_extra_name(value: object) -> bool:
    if not _is_string(value):
        return False
    try:
        Requirement(f"placeholder[{value}]")
    except InvalidRequirement:
        return False
    return True


def _valid_dynamic(value: object, project: dict[str, Any]) -> bool:
    return bool(
        _is_unique_string_list(value)
        and set(value) <= _DYNAMIC_KEYS
        and not any(field in project for field in value)
    )


def _valid_backend_path(value: object) -> bool:
    if not _is_unique_string_list(value):
        return False
    for item in value:
        path = PurePosixPath(item)
        if (
            "\\" in item
            or "\x00" in item
            or path.is_absolute()
            or ".." in path.parts
            or path.as_posix() != item
        ):
            return False
    return True


def _valid_build_system(value: object) -> bool:
    if type(value) is not dict or set(value) - _BUILD_SYSTEM_KEYS:
        return False
    backend = value.get("build-backend")
    return bool(
        _is_string(backend)
        and _PEP517_BACKEND.fullmatch(backend)
        and _valid_requirements(value.get("requires"), require_nonempty=True)
        and _valid_backend_path(value.get("backend-path", []))
    )


_FIELD_VALIDATORS: dict[str, Callable[[object], bool]] = {
    "authors": _valid_people,
    "classifiers": _is_string_list,
    "dependencies": _valid_requirements,
    "description": _is_string,
    "entry-points": _valid_entry_points,
    "gui-scripts": _valid_entry_point_map,
    "keywords": _is_string_list,
    "license": _valid_license,
    "license-files": _is_string_list,
    "maintainers": _valid_people,
    "name": lambda value: _normalized_name(value) is not None,
    "optional-dependencies": _valid_optional_dependencies,
    "readme": _valid_readme,
    "requires-python": _valid_requires_python,
    "scripts": _valid_entry_point_map,
    "urls": _valid_string_map,
    "version": _valid_version,
}


def validate_release_document(
    data: object, *, expected_name: str | None = None
) -> bool:
    """Return whether a parsed pyproject has strict buildable PEP 621 metadata."""
    if type(data) is not dict or type(data.get("project")) is not dict:
        return False
    project = data["project"]
    return _valid_project_table(
        project, expected_name=expected_name
    ) and _valid_build_system(data.get("build-system"))


def _valid_project_table(project: dict[str, Any], *, expected_name: str | None) -> bool:
    """Validate the closed PEP 621 project table and its identity."""
    if set(project) - _PROJECT_KEYS:
        return False
    if not all(
        validator(project[field])
        for field, validator in _FIELD_VALIDATORS.items()
        if field in project
    ):
        return False
    name = _normalized_name(project.get("name"))
    if name is None or (expected_name is not None and name != expected_name):
        return False
    return _valid_project_version_mode(project)


def _valid_project_version_mode(project: dict[str, Any]) -> bool:
    """Require one coherent static or supported dynamic version declaration."""
    dynamic = project.get("dynamic", [])
    if not _valid_dynamic(dynamic, project):
        return False
    return ("version" in project) != ("version" in dynamic)


def read_release_document(
    manifest: Path, *, expected_name: str | None = None
) -> dict[str, Any] | None:
    """Read and validate a regular UTF-8 TOML release manifest, or fail closed."""
    if manifest.is_symlink() or not manifest.is_file():
        return None
    try:
        raw = manifest.read_bytes()
        data = tomllib.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, tomllib.TOMLDecodeError):
        return None
    return _validated_release_data(data, manifest.parent, expected_name)


def _validated_release_data(
    data: dict[str, Any], root: Path, expected_name: str | None
) -> dict[str, Any] | None:
    if not validate_release_document(data, expected_name=expected_name):
        return None
    return data if _valid_declared_files(data["project"], root) else None


def _valid_declared_files(project: dict[str, Any], root: Path) -> bool:
    """Validate every metadata-declared local file against its project root."""
    readme = project.get("readme")
    if isinstance(readme, str) and not _safe_regular_file(root, readme):
        return False
    if isinstance(readme, dict) and "file" in readme:
        if not _safe_regular_file(root, readme["file"]):
            return False
    license_value = project.get("license")
    if isinstance(license_value, dict) and "file" in license_value:
        if not _safe_regular_file(root, license_value["file"]):
            return False
    return all(
        _safe_license_pattern(root, pattern)
        for pattern in project.get("license-files", [])
    )


def _safe_regular_file(root: Path, value: object) -> bool:
    """Require a concrete regular file with no traversal or symlink component."""
    if not _safe_relative_path(value, allow_glob=False):
        return False
    candidate = root / str(value)
    return _safe_file_candidate(root, candidate)


def _safe_license_pattern(root: Path, value: object) -> bool:
    """Require a relative license glob whose matches are all safe regular files."""
    if not _safe_relative_path(value, allow_glob=True):
        return False
    try:
        matches = list(root.glob(str(value)))
    except (OSError, ValueError):
        return False
    return bool(matches) and all(_safe_file_candidate(root, path) for path in matches)


def _safe_relative_path(value: object, *, allow_glob: bool) -> bool:
    if not _is_string(value) or "\\" in value or "\x00" in value:
        return False
    if not allow_glob and any(character in value for character in "*?[]"):
        return False
    path = PurePosixPath(value)
    return bool(not path.is_absolute() and path.parts and ".." not in path.parts)


def _safe_file_candidate(root: Path, candidate: Path) -> bool:
    if root.is_symlink():
        return False
    try:
        relative = candidate.relative_to(root)
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root.resolve(strict=True))
    except (OSError, ValueError):
        return False
    components = [
        root.joinpath(*relative.parts[:index])
        for index in range(1, len(relative.parts) + 1)
    ]
    return bool(
        candidate.is_file() and not any(path.is_symlink() for path in components)
    )


def _validate_raw_url(value: object) -> str:
    if type(value) is not str or not value:
        raise ValueError("repository URL must be a non-empty string")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError("repository URL is not valid UTF-8 text") from exc
    if any(
        ord(char) <= 0x20 or 0x7F <= ord(char) <= 0x9F or char.isspace()
        for char in value
    ):
        raise ValueError("repository URL contains whitespace or control characters")
    return value


def _validate_host(host: str | None) -> str:
    if not host or host.endswith("."):
        raise ValueError("repository URL must contain a canonical DNS host")
    labels = host.split(".")
    if any(_HOST_LABEL.fullmatch(label) is None for label in labels):
        raise ValueError("repository URL contains an invalid DNS host")
    return host.lower()


def _decode_path_segment(raw: str) -> str:
    try:
        segment = unquote(raw, errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError("repository URL contains invalid path encoding") from exc
    if (
        not segment
        or segment in {".", ".."}
        or "/" in segment
        or "\\" in segment
        or "%" in segment
        or any(ord(char) <= 0x20 or 0x7F <= ord(char) <= 0x9F for char in segment)
    ):
        raise ValueError("repository URL contains an unsafe path segment")
    return segment


def canonical_repository_url(value: object) -> str:
    """Validate and canonicalize the manifest's HTTPS/HTTP ``*.git`` URL form."""
    raw = _validate_raw_url(value)
    parsed = _parse_repository_url(raw)
    port = _validated_port(parsed)
    host = _validate_host(parsed.hostname)
    path = _canonical_repository_path(parsed.path)
    authority = _canonical_authority(parsed.scheme.lower(), host, port)
    return f"{parsed.scheme.lower()}://{authority}/{path}"


def _parse_repository_url(raw: str) -> SplitResult:
    """Parse an allowed URL shape without losing empty delimiters."""
    if "?" in raw or "#" in raw:
        raise ValueError("repository URL must not contain query or fragment delimiters")
    parsed = urlsplit(raw)
    _validate_transport_shape(parsed)
    _validate_authority_shape(parsed)
    return parsed


def _validate_transport_shape(parsed: SplitResult) -> None:
    """Restrict repository origins to absolute HTTP(S) paths."""
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError("repository URL must use canonical HTTP(S) git form")
    if parsed.query or parsed.fragment or not parsed.path.startswith("/"):
        raise ValueError("repository URL must not contain query or fragment data")


def _validate_authority_shape(parsed: SplitResult) -> None:
    """Reject credentials and syntactically empty ports in the authority."""
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("repository URL credentials are forbidden")
    if parsed.netloc.endswith(":"):
        raise ValueError("repository URL contains an empty port")


def _validated_port(parsed: SplitResult) -> int | None:
    """Return a syntactically valid TCP port from a parsed repository URL."""
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("repository URL contains an invalid port") from exc
    if port is not None and port < 1:
        raise ValueError("repository URL contains an out-of-range port")
    return port


def _canonical_repository_path(raw_path: str) -> str:
    """Decode, validate, and deterministically encode all repository segments."""
    raw_segments = raw_path[1:].split("/")
    if any(not segment for segment in raw_segments):
        raise ValueError("repository URL must have non-empty path segments")
    segments = [_decode_path_segment(segment) for segment in raw_segments]
    if not segments[-1].endswith(".git"):
        raise ValueError("repository URL must end in .git")
    name = segments[-1][:-4]
    if not name or name in {".", ".."}:
        raise ValueError("repository URL contains an unsafe basename")
    return "/".join(quote(segment, safe="-._~") for segment in segments)


def _canonical_authority(scheme: str, host: str, port: int | None) -> str:
    """Omit default ports and retain other validated ports deterministically."""
    authority = host
    if port is not None and not (
        scheme == "http" and port == 80 or scheme == "https" and port == 443
    ):
        authority = f"{authority}:{port}"
    return authority


def repository_name(value: object) -> str:
    """Return the repository basename of a canonical manifest URL."""
    canonical = canonical_repository_url(value)
    return unquote(canonical.rsplit("/", 1)[-1][:-4])


def release_repository_name(value: object) -> str:
    """Return a strict PEP 503 repository identity for release planning."""
    name = repository_name(value)
    if _normalized_name(name) is None:
        raise ValueError("release repository basename must be PEP 503 normalized")
    return name
