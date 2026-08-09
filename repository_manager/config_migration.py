"""Previewable, deterministic migration for Repository Manager declarations.

Migration is intentionally separate from execution.  :func:`preview_migration`
only reads and returns a diff; :func:`apply_migration` is the explicit mutating
operation and replaces a file atomically after preserving a recoverable backup.
The same normalized mapping is used by the build and merge parsers, so a dry run
cannot claim a declaration is valid while the runtime parser rejects it.
"""

from __future__ import annotations

import difflib
import hashlib
import os
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from repository_manager.config_schema import (
    BUILD_CONFIG_FILENAME,
    MERGE_CONFIG_FILENAME,
    SCHEMA_VERSION,
    ConfigSchemaError,
    load_yaml_mapping,
    normalize_document,
    parse_build_config,
    parse_merge_config,
)


@dataclass(frozen=True)
class MigrationPreview:
    """The complete, immutable result of a migration preview."""

    path: Path | None
    kind: str
    from_version: int
    to_version: int
    changed: bool
    original_digest: str
    target_digest: str
    original_text: str
    target_text: str
    diff: str
    backup_path: Path | None = None

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly report for MCP/CLI callers."""

        return {
            "path": str(self.path) if self.path else "",
            "kind": self.kind,
            "from_version": self.from_version,
            "to_version": self.to_version,
            "changed": self.changed,
            "original_digest": self.original_digest,
            "target_digest": self.target_digest,
            "diff": self.diff,
            "backup_path": str(self.backup_path) if self.backup_path else "",
        }


def _digest(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _yaml_text(data: Mapping[str, Any]) -> str:
    text = yaml.safe_dump(
        dict(data),
        allow_unicode=True,
        default_flow_style=False,
        sort_keys=False,
    )
    return text if text.endswith("\n") else f"{text}\n"


def _kind_for_path(path: Path) -> str:
    if path.name == BUILD_CONFIG_FILENAME or path.name.endswith(".buildcache.yaml"):
        return "build"
    if path.name == MERGE_CONFIG_FILENAME or path.name.endswith(".mergequeue.yaml"):
        return "merge"
    raise ConfigSchemaError(
        f"{path}: cannot infer configuration kind; pass kind='build' or kind='merge'"
    )


def _version(data: Mapping[str, Any]) -> int:
    value = data.get("schema_version", data.get("version", 1))
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigSchemaError("schema_version must be an integer")
    return value


def preview_mapping(
    data: Mapping[str, Any], *, kind: str, source: str = ""
) -> MigrationPreview:
    """Preview migration of an already-loaded mapping without filesystem I/O."""

    normalized = normalize_document(data, kind=kind, source=source)
    if kind in {"build", BUILD_CONFIG_FILENAME}:
        parse_build_config(normalized, source=source)
    elif kind in {"merge", MERGE_CONFIG_FILENAME}:
        parse_merge_config(normalized, source=source)
    original_text = _yaml_text(data)
    target_text = _yaml_text(normalized)
    label = source or f"{kind} configuration"
    diff = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            target_text.splitlines(keepends=True),
            fromfile=f"{label} (legacy)",
            tofile=f"{label} (schema_version {SCHEMA_VERSION})",
        )
    )
    return MigrationPreview(
        path=Path(source) if source else None,
        kind=kind,
        from_version=_version(data),
        to_version=SCHEMA_VERSION,
        changed=original_text != target_text,
        original_digest=_digest(original_text),
        target_digest=_digest(target_text),
        original_text=original_text,
        target_text=target_text,
        diff=diff,
    )


def preview_migration(path: str | Path, *, kind: str | None = None) -> MigrationPreview:
    """Read and validate *path*, returning a no-write migration preview."""

    config_path = Path(path)
    selected_kind = kind or _kind_for_path(config_path)
    original_text = config_path.read_text(encoding="utf-8")
    data = load_yaml_mapping(str(config_path))
    preview = preview_mapping(data, kind=selected_kind, source=str(config_path))
    # The mapping preview intentionally uses canonical YAML.  For a file preview
    # retain the exact source bytes so comments/formatting are visible in the
    # diff and the digest guard protects the content that was actually read.
    diff = "".join(
        difflib.unified_diff(
            original_text.splitlines(keepends=True),
            preview.target_text.splitlines(keepends=True),
            fromfile=f"{config_path} (legacy)",
            tofile=f"{config_path} (schema_version {SCHEMA_VERSION})",
        )
    )
    return MigrationPreview(
        path=config_path,
        kind=selected_kind,
        from_version=preview.from_version,
        to_version=preview.to_version,
        changed=original_text != preview.target_text,
        original_digest=_digest(original_text),
        target_digest=preview.target_digest,
        original_text=original_text,
        target_text=preview.target_text,
        diff=diff,
        backup_path=config_path.with_name(f"{config_path.name}.bak"),
    )


def _fsync_directory(path: Path) -> None:
    """Best-effort directory fsync after an atomic replacement."""

    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def apply_migration(
    path: str | Path,
    *,
    kind: str | None = None,
    backup: bool = True,
) -> MigrationPreview:
    """Apply a migration atomically and return the applied preview.

    ``backup`` defaults to true and stores the pre-migration bytes at
    ``<config>.bak``.  A no-op v2 file is not rewritten and does not refresh its
    backup, making repeated application idempotent and preserving the original
    rollback point.
    """

    preview = preview_migration(path, kind=kind)
    if not preview.changed:
        return preview
    config_path = Path(path)
    if backup:
        assert preview.backup_path is not None
        shutil.copy2(config_path, preview.backup_path)
    _atomic_replace(config_path, preview.target_text)
    return preview


def rollback_migration(
    path: str | Path, preview: MigrationPreview | None = None
) -> MigrationPreview:
    """Restore a migration backup only when the file is still untouched.

    The current content must match the target digest captured by the applied
    preview.  If an operator edited the migrated file in the meantime, rollback
    refuses rather than overwriting that work.
    """

    config_path = Path(path)
    selected = preview or preview_migration(config_path)
    backup_path = selected.backup_path or config_path.with_name(
        f"{config_path.name}.bak"
    )
    if not backup_path.is_file():
        raise ConfigSchemaError(
            f"{config_path}: migration backup is missing: {backup_path}"
        )
    current_digest = _digest(config_path.read_text(encoding="utf-8"))
    if current_digest != selected.target_digest:
        raise ConfigSchemaError(
            f"{config_path}: refusing rollback because the migrated file changed "
            "after apply"
        )
    _atomic_replace(config_path, backup_path.read_text(encoding="utf-8"))
    return selected


def validate_presets(
    package_root: str | Path | None = None,
) -> tuple[dict[str, Any], ...]:
    """Validate every packaged build/merge preset through the shared schemas."""

    root = Path(package_root) if package_root else Path(__file__).parent
    results: list[dict[str, Any]] = []
    directories = (
        ("build", root / "buildcache_presets", "*.buildcache.yaml", parse_build_config),
        ("merge", root / "mergequeue_presets", "*.mergequeue.yaml", parse_merge_config),
    )
    for kind, directory, pattern, parser in directories:
        for path in sorted(directory.glob(pattern)):
            data = load_yaml_mapping(str(path))
            parser(data, source=str(path))
            results.append({"kind": kind, "path": str(path), "valid": True})
    return tuple(results)
