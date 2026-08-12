"""Versioned, typed schemas for Repository Manager repository declarations.

The build broker and merge queue deliberately keep their execution machinery
small.  This module owns the boundary at which repository-authored YAML becomes
trusted, typed configuration.  It has no subprocess, Git, queue, or filesystem
mutation behavior; callers can therefore validate a declaration before they
admit any work.

Schema version ``1`` is the historical, unversioned shape.  It remains readable
for the compatibility window and is normalized to version ``2`` in memory.  A
version ``2`` document is fail-closed: unknown keys and unsafe values are
errors, rather than values that a later runtime may accidentally ignore.
"""

from __future__ import annotations

import copy
import re
import shlex
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any

import yaml

SCHEMA_VERSION = 2
LEGACY_SCHEMA_VERSION = 1

BUILD_CONFIG_FILENAME = ".buildcache.yaml"
MERGE_CONFIG_FILENAME = ".mergequeue.yaml"


class ConfigSchemaError(ValueError):
    """A repository declaration failed schema or safety validation."""


class ConfigCompatibilityWarning(DeprecationWarning):
    """A legacy declaration was accepted during the migration window."""


class ValidationStage(StrEnum):
    """Stages at which a declared check or build may be consumed."""

    FEEDBACK = "feedback"
    INTEGRATION = "integration"
    CERTIFICATION = "certification"
    SMOKE = "smoke"
    RELEASE = "release"


VALIDATION_STAGES = frozenset(stage.value for stage in ValidationStage)
BASELINE_MODES = frozenset({"differential", "absolute", "disabled"})
COMPARE_MODES = frozenset({"exit", "lines", "pytest-ids"})
TIMEOUT_POLICIES = frozenset({"fail", "defer"})

_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_SHELL_TOKENS = frozenset({";", "&&", "||", "|", ">", ">>", "<", "<<"})


@dataclass(frozen=True)
class ResourceRequest:
    """Weighted admission requirements for one build or gate."""

    cpu_weight: int = 1
    memory_mb: int = 0
    disk_mb: int = 0
    process_slots: int = 1


@dataclass(frozen=True)
class Placement:
    """Host labels and anti-affinity hints consumed by later scheduler lanes."""

    required_labels: tuple[str, ...] = ()
    preferred_host: str = ""
    required_host: str = ""
    anti_affinity: tuple[str, ...] = ()


@dataclass(frozen=True)
class ArtifactContract:
    """The files a build promises to publish and how they are retained."""

    patterns: tuple[str, ...] = ()
    required: bool = True
    publish: bool = True
    retention: str = "content-addressed"


@dataclass(frozen=True)
class PathSelection:
    """Include/exclude globs selecting paths for a gate."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()


@dataclass(frozen=True)
class BuildSpecSchema:
    """Validated version-2 representation of one build specification."""

    name: str
    command: tuple[str, ...]
    workdir: str = "."
    toolchain_fingerprint: tuple[str, ...] = ()
    cache_key_paths: tuple[str, ...] = ()
    artifacts: tuple[str, ...] = ()
    timeout: int = 3600
    target_triple: str = ""
    resource_class: str = "light-check"
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    disk_estimate_mb: int = 0
    placement: Placement = field(default_factory=Placement)
    artifact_contract: ArtifactContract = field(default_factory=ArtifactContract)
    stage: str = ValidationStage.FEEDBACK.value
    generation_compatible: bool = True
    source: str = ""


@dataclass(frozen=True)
class BuildConfigSchema:
    """Validated version-2 representation of a ``.buildcache.yaml`` file."""

    schema_version: int
    base: str
    specs: tuple[BuildSpecSchema, ...]
    source: str = ""

    def spec(self, name: str = "") -> BuildSpecSchema:
        """Return the named spec, or the first declared spec by default."""

        if not self.specs:
            raise ConfigSchemaError(
                f"{self.source or BUILD_CONFIG_FILENAME}: declares no specs"
            )
        if not name:
            return self.specs[0]
        for spec in self.specs:
            if spec.name == name:
                return spec
        raise ConfigSchemaError(
            f"{self.source or BUILD_CONFIG_FILENAME}: has no spec named {name!r}"
        )


@dataclass(frozen=True)
class GateSchema:
    """Validated version-2 representation of one merge-queue gate."""

    name: str
    command: tuple[str, ...]
    stage: str = ValidationStage.INTEGRATION.value
    timeout: int = 300
    baseline_timeout: int = 0
    baseline_mode: str = "differential"
    compare: str = "lines"
    path_selection: PathSelection = field(default_factory=PathSelection)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    resource_class: str = "light-check"
    artifact_dependencies: tuple[str, ...] = ()
    keep_lines: tuple[str, ...] = ()
    ignore_lines: tuple[str, ...] = ()
    on_timeout: str = "fail"
    source: str = ""

    @property
    def legacy_tier(self) -> str:
        """Return the current queue's pre-v2 fast/slow projection."""

        return "fast" if self.stage in {"feedback", "integration"} else "slow"


@dataclass(frozen=True)
class GeneratedFileContract:
    """Generated-file conflict policy for a merge queue."""

    files: frozenset[str] = frozenset()
    regenerate: tuple[tuple[str, ...], ...] = ()


@dataclass(frozen=True)
class MergeConfigSchema:
    """Validated version-2 representation of a ``.mergequeue.yaml`` file."""

    schema_version: int
    base: str
    batch_size: int
    environment_signature: tuple[str, ...]
    gates: tuple[GateSchema, ...]
    generated: GeneratedFileContract
    source: str = ""


def _where(source: str, path: str) -> str:
    if source and (source == path or source.endswith(f"/{path}")):
        return source
    return f"{source}: {path}" if source else path


def _error(source: str, path: str, message: str) -> ConfigSchemaError:
    return ConfigSchemaError(f"{_where(source, path)}: {message}")


def _mapping(value: Any, *, source: str, path: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _error(source, path, "must be a mapping")
    result = dict(value)
    for key in result:
        if not isinstance(key, str):
            raise _error(source, path, "mapping keys must be strings")
    return result


def _string(value: Any, *, source: str, path: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise _error(source, path, "must be a string")
    if not allow_empty and not value.strip():
        raise _error(source, path, "must not be empty")
    if "\x00" in value:
        raise _error(source, path, "must not contain NUL")
    return value


def _string_list(
    value: Any,
    *,
    source: str,
    path: str,
    allow_empty: bool = True,
) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise _error(source, path, "must be a list of strings")
    result: list[str] = []
    for index, item in enumerate(value):
        result.append(_string(item, source=source, path=f"{path}[{index}]"))
    if not allow_empty and not result:
        raise _error(source, path, "must not be empty")
    return tuple(result)


def _argv(value: Any, *, source: str, path: str) -> tuple[str, ...]:
    if isinstance(value, str):
        raise _error(
            source,
            path,
            "must be a LIST of argv items (strings), not a shell string; "
            "the list must not be empty",
        )
    return _string_list(value, source=source, path=path, allow_empty=False)


def _integer(
    value: Any,
    *,
    source: str,
    path: str,
    default: int,
    minimum: int | None = None,
) -> int:
    if value is None:
        result = default
    elif isinstance(value, bool) or not isinstance(value, int):
        raise _error(source, path, "must be an integer")
    else:
        result = value
    if minimum is not None and result < minimum:
        raise _error(source, path, f"must be >= {minimum}")
    return result


def _boolean(value: Any, *, source: str, path: str, default: bool) -> bool:
    if value is None:
        return default
    if not isinstance(value, bool):
        raise _error(source, path, "must be a boolean")
    return value


def _name(value: Any, *, source: str, path: str) -> str:
    result = _string(value, source=source, path=path).strip()
    if not _SAFE_NAME.fullmatch(result):
        raise _error(
            source,
            path,
            "must contain only letters, numbers, '.', '_' or '-'",
        )
    return result


def _relative_pattern(value: Any, *, source: str, path: str) -> str:
    result = _string(value, source=source, path=path)
    if (
        result.startswith("/")
        or PurePosixPath(result).is_absolute()
        or PureWindowsPath(result).is_absolute()
        or PureWindowsPath(result).drive
    ):
        raise _error(source, path, "must be relative to the repository root")
    if any(
        part == ".."
        for part in (*PurePosixPath(result).parts, *PureWindowsPath(result).parts)
    ):
        raise _error(
            source,
            path,
            "must be relative and must not contain '..' path traversal",
        )
    if "[" in result and "]" not in result:
        raise _error(source, path, "contains an unterminated '[' glob character")
    if "]" in result and "[" not in result:
        raise _error(source, path, "contains an unmatched ']' glob character")
    return result


def _relative_patterns(value: Any, *, source: str, path: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, Sequence) or isinstance(value, str | bytes):
        raise _error(source, path, "must be a list of relative glob patterns")
    return tuple(
        _relative_pattern(item, source=source, path=f"{path}[{index}]")
        for index, item in enumerate(value)
    )


def _check_keys(
    value: Mapping[str, Any],
    allowed: frozenset[str],
    *,
    source: str,
    path: str,
    strict: bool,
) -> None:
    unknown = sorted(set(value) - allowed)
    if not unknown:
        return
    message = f"unknown key(s): {', '.join(unknown)}"
    if strict:
        raise _error(source, path, message)
    warnings.warn(
        f"{_where(source, path)}: {message}; ignored until this config is migrated "
        f"to schema_version {SCHEMA_VERSION}",
        ConfigCompatibilityWarning,
        stacklevel=4,
    )


def _schema_version(
    data: Mapping[str, Any], *, source: str, path: str
) -> tuple[int, bool]:
    marker = data.get("schema_version")
    alias = data.get("version")
    if marker is not None and alias is not None and marker != alias:
        raise _error(source, path, "schema_version and version disagree")
    if marker is None:
        marker = alias
    if marker is None:
        warnings.warn(
            f"{_where(source, path)}: unversioned legacy declaration accepted; "
            f"migrate to schema_version {SCHEMA_VERSION}",
            ConfigCompatibilityWarning,
            stacklevel=4,
        )
        return LEGACY_SCHEMA_VERSION, True
    if isinstance(marker, bool) or not isinstance(marker, int):
        raise _error(source, f"{path}.schema_version", "must be an integer")
    if marker not in {LEGACY_SCHEMA_VERSION, SCHEMA_VERSION}:
        raise _error(
            source,
            f"{path}.schema_version",
            f"unsupported version {marker}; supported: 1, {SCHEMA_VERSION}",
        )
    if marker == LEGACY_SCHEMA_VERSION:
        warnings.warn(
            f"{_where(source, path)}: schema_version 1 is deprecated; "
            f"migrate to schema_version {SCHEMA_VERSION}",
            ConfigCompatibilityWarning,
            stacklevel=4,
        )
    return marker, marker == LEGACY_SCHEMA_VERSION


def _resource_request(
    value: Any, *, source: str, path: str, strict: bool
) -> ResourceRequest:
    if value is None:
        return ResourceRequest()
    data = _mapping(value, source=source, path=path)
    _check_keys(
        data,
        frozenset({"cpu_weight", "memory_mb", "disk_mb", "process_slots"}),
        source=source,
        path=path,
        strict=strict,
    )
    return ResourceRequest(
        cpu_weight=_integer(
            data.get("cpu_weight"),
            source=source,
            path=f"{path}.cpu_weight",
            default=1,
            minimum=1,
        ),
        memory_mb=_integer(
            data.get("memory_mb"),
            source=source,
            path=f"{path}.memory_mb",
            default=0,
            minimum=0,
        ),
        disk_mb=_integer(
            data.get("disk_mb"),
            source=source,
            path=f"{path}.disk_mb",
            default=0,
            minimum=0,
        ),
        process_slots=_integer(
            data.get("process_slots"),
            source=source,
            path=f"{path}.process_slots",
            default=1,
            minimum=1,
        ),
    )


def _placement(value: Any, *, source: str, path: str, strict: bool) -> Placement:
    if value is None:
        return Placement()
    data = _mapping(value, source=source, path=path)
    _check_keys(
        data,
        frozenset(
            {
                "labels",
                "required_labels",
                "preferred_host",
                "required_host",
                "anti_affinity",
            }
        ),
        source=source,
        path=path,
        strict=strict,
    )
    labels = data.get("required_labels", data.get("labels"))
    return Placement(
        required_labels=_string_list(
            labels, source=source, path=f"{path}.required_labels"
        ),
        preferred_host=_string(
            data.get("preferred_host", ""),
            source=source,
            path=f"{path}.preferred_host",
            allow_empty=True,
        ),
        required_host=_string(
            data.get("required_host", ""),
            source=source,
            path=f"{path}.required_host",
            allow_empty=True,
        ),
        anti_affinity=_string_list(
            data.get("anti_affinity"), source=source, path=f"{path}.anti_affinity"
        ),
    )


def _artifact_contract(
    value: Any,
    artifacts: tuple[str, ...],
    *,
    source: str,
    path: str,
    strict: bool,
) -> ArtifactContract:
    if value is None:
        return ArtifactContract(patterns=artifacts)
    data = _mapping(value, source=source, path=path)
    _check_keys(
        data,
        frozenset({"patterns", "paths", "required", "publish", "retention"}),
        source=source,
        path=path,
        strict=strict,
    )
    patterns_value = data.get("patterns", data.get("paths", artifacts))
    patterns = _relative_patterns(
        patterns_value, source=source, path=f"{path}.patterns"
    )
    if artifacts and patterns and artifacts != patterns:
        raise _error(
            source,
            path,
            "artifacts and artifact_contract.patterns disagree; declare one "
            "canonical set",
        )
    if not patterns:
        patterns = artifacts
    retention = _string(
        data.get("retention", "content-addressed"),
        source=source,
        path=f"{path}.retention",
    )
    if retention not in {"content-addressed", "ephemeral", "pinned"}:
        raise _error(
            source,
            f"{path}.retention",
            "must be content-addressed, ephemeral, or pinned",
        )
    return ArtifactContract(
        patterns=patterns,
        required=_boolean(
            data.get("required"), source=source, path=f"{path}.required", default=True
        ),
        publish=_boolean(
            data.get("publish"), source=source, path=f"{path}.publish", default=True
        ),
        retention=retention,
    )


def _stage(value: Any, *, source: str, path: str, default: str) -> str:
    result = _string(value if value is not None else default, source=source, path=path)
    if result not in VALIDATION_STAGES:
        raise _error(
            source,
            path,
            f"unsupported stage {result!r}; supported: {sorted(VALIDATION_STAGES)}",
        )
    return result


def _baseline_mode(value: Any, *, source: str, path: str) -> str:
    result = _string(
        value if value is not None else "differential", source=source, path=path
    )
    if result not in BASELINE_MODES:
        raise _error(
            source,
            path,
            f"unsupported baseline mode {result!r}; supported: {sorted(BASELINE_MODES)}",
        )
    return result


def _path_selection(
    value: Any,
    *,
    source: str,
    path: str,
    strict: bool,
) -> PathSelection:
    if value is None:
        return PathSelection()
    data = _mapping(value, source=source, path=path)
    _check_keys(
        data,
        frozenset({"include", "exclude"}),
        source=source,
        path=path,
        strict=strict,
    )
    return PathSelection(
        include=_relative_patterns(
            data.get("include"), source=source, path=f"{path}.include"
        ),
        exclude=_relative_patterns(
            data.get("exclude"), source=source, path=f"{path}.exclude"
        ),
    )


def _validate_regex_list(value: Any, *, source: str, path: str) -> tuple[str, ...]:
    patterns = _string_list(value, source=source, path=path)
    for index, pattern in enumerate(patterns):
        try:
            re.compile(pattern)
        except re.error as exc:
            raise _error(
                source, f"{path}[{index}]", f"invalid regular expression: {exc}"
            ) from exc
    return patterns


def _artifact_dependencies(value: Any, *, source: str, path: str) -> tuple[str, ...]:
    dependencies = _string_list(value, source=source, path=path)
    for index, dependency in enumerate(dependencies):
        if not dependency.strip():
            raise _error(source, f"{path}[{index}]", "must not be empty")
    return dependencies


def _normalize_regenerate_command(
    value: Any, *, source: str, path: str
) -> list[list[str]]:
    """Normalize a command or list of commands to argv lists.

    The old drifted ``regenerate_on_conflict.regenerate`` field was a shell
    string.  ``shlex.split`` makes the migration deterministic while the
    resulting runtime contract remains argv-only.  Shell operators are refused
    rather than accidentally becoming literal arguments that look executable.
    """

    if isinstance(value, str):
        try:
            argv = shlex.split(value)
        except ValueError as exc:
            raise _error(source, path, f"invalid command quoting: {exc}") from exc
        if any(token in _SHELL_TOKENS for token in argv):
            raise _error(source, path, "shell operators are not valid in argv commands")
        return [list(_argv(argv, source=source, path=path))]
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise _error(source, path, "must be an argv list or list of argv lists")
    if not value:
        raise _error(source, path, "must not be empty")
    if all(isinstance(item, str) for item in value):
        return [list(_argv(value, source=source, path=path))]
    commands: list[list[str]] = []
    for index, command in enumerate(value):
        commands.append(list(_argv(command, source=source, path=f"{path}[{index}]")))
    return commands


def _normalize_build(data: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    raw = copy.deepcopy(dict(data))
    version, legacy = _schema_version(raw, source=source, path=BUILD_CONFIG_FILENAME)
    top_allowed = frozenset(
        {
            "schema_version",
            "version",
            "base",
            "specs",
        }
    )
    _check_keys(
        raw,
        top_allowed,
        source=source,
        path=BUILD_CONFIG_FILENAME,
        strict=not legacy,
    )
    specs = raw.get("specs", [])
    if specs is None and legacy:
        # The historical parser treated an omitted/null specs declaration as
        # an intentionally empty broker. Keep that compatibility behavior;
        # schema v2 remains fail-closed for an explicit null value.
        specs = []
    if not isinstance(specs, Sequence) or isinstance(specs, str | bytes):
        raise _error(source, f"{BUILD_CONFIG_FILENAME}.specs", "must be a list")
    normalized_specs: list[dict[str, Any]] = []
    spec_allowed = frozenset(
        {
            "name",
            "command",
            "workdir",
            "toolchain_fingerprint",
            "cache_key_paths",
            "artifacts",
            "artifact_contract",
            "timeout",
            "target_triple",
            "resource_class",
            "resources",
            "placement",
            "stage",
            "validation_stage",
            "generation_compatible",
            "disk_estimate_mb",
        }
    )
    for index, item in enumerate(specs):
        path = f"{BUILD_CONFIG_FILENAME}.specs[{index}]"
        spec = _mapping(item, source=source, path=path)
        _check_keys(spec, spec_allowed, source=source, path=path, strict=not legacy)
        normalized = dict(spec)
        if "validation_stage" in normalized and "stage" not in normalized:
            normalized["stage"] = normalized.pop("validation_stage")
        normalized.setdefault("stage", ValidationStage.FEEDBACK.value)
        normalized.setdefault("generation_compatible", True)
        normalized_specs.append(normalized)
    return {
        "schema_version": SCHEMA_VERSION,
        "base": raw.get("base", "main"),
        "specs": normalized_specs,
    }


def _regeneration_parts(
    value: Any, *, source: str, path: str, strict: bool
) -> tuple[list[str], list[list[str]]]:
    data = _mapping(value, source=source, path=path)
    _check_keys(
        data,
        frozenset({"files", "paths", "regenerate", "commands"}),
        source=source,
        path=path,
        strict=strict,
    )
    files_value = data.get("files", data.get("paths", []))
    files = list(_relative_patterns(files_value, source=source, path=f"{path}.files"))
    command_value = data.get("regenerate", data.get("commands"))
    commands = (
        _normalize_regenerate_command(
            command_value, source=source, path=f"{path}.regenerate"
        )
        if command_value is not None
        else []
    )
    return files, commands


def _merge_commands(value: Any, *, source: str, path: str) -> list[list[str]]:
    if value is None:
        return []
    if isinstance(value, str):
        return _normalize_regenerate_command(value, source=source, path=path)
    if not isinstance(value, Sequence) or isinstance(value, bytes):
        raise _error(source, path, "must be an argv list or list of argv lists")
    if not value:
        return []
    if all(isinstance(item, str) for item in value):
        return [list(_argv(value, source=source, path=path))]
    return _normalize_regenerate_command(value, source=source, path=path)


def _normalize_merge(data: Mapping[str, Any], *, source: str) -> dict[str, Any]:
    raw = copy.deepcopy(dict(data))
    version, legacy = _schema_version(raw, source=source, path=MERGE_CONFIG_FILENAME)
    top_allowed = frozenset(
        {
            "schema_version",
            "version",
            "base",
            "batch_size",
            "environment_signature",
            "gates",
            "generated_files",
            "regenerate",
            "regenerate_on_conflict",
            "regeneration",
        }
    )
    _check_keys(
        raw,
        top_allowed,
        source=source,
        path=MERGE_CONFIG_FILENAME,
        strict=not legacy,
    )
    gates = raw.get("gates", [])
    if gates is None:
        gates = []
    if not isinstance(gates, Sequence) or isinstance(gates, str | bytes):
        raise _error(source, f"{MERGE_CONFIG_FILENAME}.gates", "must be a list")
    gate_allowed = frozenset(
        {
            "name",
            "command",
            "stage",
            "tier",
            "timeout",
            "baseline_timeout",
            "baseline_mode",
            "compare",
            "when_changed",
            "path_selection",
            "paths",
            "keep_lines",
            "ignore_lines",
            "on_timeout",
            "resource_class",
            "resources",
            "artifact_dependencies",
            "artifacts",
        }
    )
    normalized_gates: list[dict[str, Any]] = []
    for index, item in enumerate(gates):
        path = f"{MERGE_CONFIG_FILENAME}.gates[{index}]"
        gate = _mapping(item, source=source, path=path)
        _check_keys(gate, gate_allowed, source=source, path=path, strict=not legacy)
        normalized = dict(gate)
        stage = normalized.get("stage")
        tier = normalized.get("tier")
        if stage is None:
            if tier is not None:
                tier_value = _string(tier, source=source, path=f"{path}.tier")
                if tier_value not in {"fast", "slow"}:
                    raise _error(source, f"{path}.tier", "must be fast or slow")
                stage = (
                    ValidationStage.INTEGRATION.value
                    if tier_value == "fast"
                    else ValidationStage.CERTIFICATION.value
                )
                warnings.warn(
                    f"{_where(source, path)}: tier={tier_value!r} is deprecated; "
                    f"use stage={stage!r}",
                    ConfigCompatibilityWarning,
                    stacklevel=4,
                )
            else:
                stage = ValidationStage.INTEGRATION.value
        normalized["stage"] = stage
        normalized.pop("tier", None)
        if "validation_stage" in normalized:
            raise _error(source, path, "use stage, not validation_stage")
        path_selection = normalized.get("path_selection", normalized.get("paths"))
        if path_selection is None and "when_changed" in normalized:
            path_selection = {"include": normalized["when_changed"]}
        if path_selection is not None:
            normalized["path_selection"] = path_selection
        normalized.pop("paths", None)
        normalized.pop("when_changed", None)
        if "artifacts" in normalized and "artifact_dependencies" not in normalized:
            normalized["artifact_dependencies"] = normalized.pop("artifacts")
        normalized.setdefault("baseline_mode", "differential")
        normalized_gates.append(normalized)

    files: list[str] = []
    commands: list[list[str]] = []
    if raw.get("generated_files") is not None:
        files = list(
            _relative_patterns(
                raw.get("generated_files"),
                source=source,
                path=f"{MERGE_CONFIG_FILENAME}.generated_files",
            )
        )
    if raw.get("regenerate") is not None:
        commands = _merge_commands(
            raw.get("regenerate"),
            source=source,
            path=f"{MERGE_CONFIG_FILENAME}.regenerate",
        )
    drift = raw.get("regenerate_on_conflict")
    if drift is not None:
        drift_files, drift_commands = _regeneration_parts(
            drift,
            source=source,
            path=f"{MERGE_CONFIG_FILENAME}.regenerate_on_conflict",
            strict=False,
        )
        warnings.warn(
            f"{_where(source, MERGE_CONFIG_FILENAME)}: "
            "regenerate_on_conflict is a deprecated shape; migrated to "
            "generated_files/regenerate",
            ConfigCompatibilityWarning,
            stacklevel=4,
        )
        files.extend(item for item in drift_files if item not in files)
        commands.extend(item for item in drift_commands if item not in commands)
    regeneration = raw.get("regeneration")
    if regeneration is not None:
        regen_files, regen_commands = _regeneration_parts(
            regeneration,
            source=source,
            path=f"{MERGE_CONFIG_FILENAME}.regeneration",
            strict=not legacy,
        )
        files.extend(item for item in regen_files if item not in files)
        commands.extend(item for item in regen_commands if item not in commands)
    return {
        "schema_version": SCHEMA_VERSION,
        "base": raw.get("base", "main"),
        "batch_size": raw.get("batch_size", 8),
        "environment_signature": raw.get("environment_signature", []),
        "gates": normalized_gates,
        "generated_files": sorted(set(files)),
        "regenerate": commands,
    }


def normalize_document(
    data: Mapping[str, Any], *, kind: str, source: str = ""
) -> dict[str, Any]:
    """Return a deterministic version-2 mapping without touching the filesystem."""

    data = _mapping(data, source=source, path=kind)
    if kind in {"build", BUILD_CONFIG_FILENAME}:
        return _normalize_build(data, source=source)
    if kind in {"merge", MERGE_CONFIG_FILENAME}:
        return _normalize_merge(data, source=source)
    raise ConfigSchemaError(f"unsupported configuration kind: {kind!r}")


def parse_build_config(
    data: Mapping[str, Any], *, source: str = ""
) -> BuildConfigSchema:
    """Validate a build declaration and return its typed v2 representation."""

    normalized = normalize_document(data, kind="build", source=source)
    specs: list[BuildSpecSchema] = []
    names: set[str] = set()
    for index, raw in enumerate(normalized["specs"]):
        path = f"{BUILD_CONFIG_FILENAME}.specs[{index}]"
        spec = _mapping(raw, source=source, path=path)
        name = _name(spec.get("name"), source=source, path=f"{path}.name")
        if name in names:
            raise _error(source, path, f"duplicate spec name {name!r}")
        names.add(name)
        command = _argv(spec.get("command"), source=source, path=f"{path}.command")
        workdir = _relative_pattern(
            spec.get("workdir", "."), source=source, path=f"{path}.workdir"
        )
        toolchain = (
            _argv(
                spec.get("toolchain_fingerprint"),
                source=source,
                path=f"{path}.toolchain_fingerprint",
            )
            if spec.get("toolchain_fingerprint") is not None
            else ()
        )
        cache_key_paths = _relative_patterns(
            spec.get("cache_key_paths"),
            source=source,
            path=f"{path}.cache_key_paths",
        )
        artifacts = _relative_patterns(
            spec.get("artifacts"), source=source, path=f"{path}.artifacts"
        )
        artifact_contract = _artifact_contract(
            spec.get("artifact_contract"),
            artifacts,
            source=source,
            path=f"{path}.artifact_contract",
            strict=True,
        )
        resources = _resource_request(
            spec.get("resources"), source=source, path=f"{path}.resources", strict=True
        )
        disk_estimate = spec.get("disk_estimate_mb")
        if disk_estimate is not None:
            disk_value = _integer(
                disk_estimate,
                source=source,
                path=f"{path}.disk_estimate_mb",
                default=0,
                minimum=0,
            )
            if resources.disk_mb and resources.disk_mb != disk_value:
                raise _error(
                    source,
                    path,
                    "disk_estimate_mb and resources.disk_mb disagree",
                )
            resources = ResourceRequest(
                cpu_weight=resources.cpu_weight,
                memory_mb=resources.memory_mb,
                disk_mb=disk_value,
                process_slots=resources.process_slots,
            )
        resource_class = _name(
            spec.get("resource_class", "light-check"),
            source=source,
            path=f"{path}.resource_class",
        )
        placement = _placement(
            spec.get("placement"), source=source, path=f"{path}.placement", strict=True
        )
        stage = _stage(
            spec.get("stage"),
            source=source,
            path=f"{path}.stage",
            default=ValidationStage.FEEDBACK.value,
        )
        specs.append(
            BuildSpecSchema(
                name=name,
                command=command,
                workdir=workdir,
                toolchain_fingerprint=toolchain,
                cache_key_paths=cache_key_paths,
                artifacts=artifact_contract.patterns,
                timeout=_integer(
                    spec.get("timeout"),
                    source=source,
                    path=f"{path}.timeout",
                    default=3600,
                    minimum=1,
                ),
                target_triple=_string(
                    spec.get("target_triple", ""),
                    source=source,
                    path=f"{path}.target_triple",
                    allow_empty=True,
                ),
                resource_class=resource_class,
                resources=resources,
                disk_estimate_mb=resources.disk_mb,
                placement=placement,
                artifact_contract=artifact_contract,
                stage=stage,
                generation_compatible=_boolean(
                    spec.get("generation_compatible"),
                    source=source,
                    path=f"{path}.generation_compatible",
                    default=True,
                ),
                source=source,
            )
        )
    base = _string(normalized.get("base", "main"), source=source, path="build.base")
    return BuildConfigSchema(
        schema_version=SCHEMA_VERSION,
        base=base,
        specs=tuple(specs),
        source=source,
    )


def parse_merge_config(
    data: Mapping[str, Any], *, source: str = ""
) -> MergeConfigSchema:
    """Validate a merge declaration and return its typed v2 representation."""

    normalized = normalize_document(data, kind="merge", source=source)
    gates: list[GateSchema] = []
    names: set[str] = set()
    for index, raw in enumerate(normalized["gates"]):
        path = f"{MERGE_CONFIG_FILENAME}.gates[{index}]"
        gate = _mapping(raw, source=source, path=path)
        name = _name(gate.get("name"), source=source, path=f"{path}.name")
        if name in names:
            raise _error(source, path, f"duplicate gate name {name!r}")
        names.add(name)
        command = _argv(gate.get("command"), source=source, path=f"{path}.command")
        compare = _string(
            gate.get("compare", "lines"), source=source, path=f"{path}.compare"
        )
        if compare not in COMPARE_MODES:
            raise _error(
                source, f"{path}.compare", f"must be one of {sorted(COMPARE_MODES)}"
            )
        stage = _stage(
            gate.get("stage"),
            source=source,
            path=f"{path}.stage",
            default=ValidationStage.INTEGRATION.value,
        )
        baseline_mode = _baseline_mode(
            gate.get("baseline_mode"), source=source, path=f"{path}.baseline_mode"
        )
        path_selection = _path_selection(
            gate.get("path_selection"),
            source=source,
            path=f"{path}.path_selection",
            strict=True,
        )
        resources = _resource_request(
            gate.get("resources"), source=source, path=f"{path}.resources", strict=True
        )
        resource_class = _name(
            gate.get("resource_class", "light-check"),
            source=source,
            path=f"{path}.resource_class",
        )
        on_timeout = _string(
            gate.get("on_timeout", "fail"), source=source, path=f"{path}.on_timeout"
        )
        if on_timeout not in TIMEOUT_POLICIES:
            raise _error(source, f"{path}.on_timeout", "must be fail or defer")
        gates.append(
            GateSchema(
                name=name,
                command=command,
                stage=stage,
                timeout=_integer(
                    gate.get("timeout"),
                    source=source,
                    path=f"{path}.timeout",
                    default=300,
                    minimum=1,
                ),
                baseline_timeout=_integer(
                    gate.get("baseline_timeout"),
                    source=source,
                    path=f"{path}.baseline_timeout",
                    default=0,
                    minimum=0,
                ),
                baseline_mode=baseline_mode,
                compare=compare,
                path_selection=path_selection,
                resources=resources,
                resource_class=resource_class,
                artifact_dependencies=_artifact_dependencies(
                    gate.get("artifact_dependencies"),
                    source=source,
                    path=f"{path}.artifact_dependencies",
                ),
                keep_lines=_validate_regex_list(
                    gate.get("keep_lines"), source=source, path=f"{path}.keep_lines"
                ),
                ignore_lines=_validate_regex_list(
                    gate.get("ignore_lines"), source=source, path=f"{path}.ignore_lines"
                ),
                on_timeout=on_timeout,
                source=source,
            )
        )
    files = frozenset(
        _relative_patterns(
            normalized.get("generated_files"),
            source=source,
            path=f"{MERGE_CONFIG_FILENAME}.generated_files",
        )
    )
    regenerate_raw = normalized.get("regenerate") or []
    regenerate = tuple(
        tuple(
            _argv(
                command,
                source=source,
                path=f"{MERGE_CONFIG_FILENAME}.regenerate[{index}]",
            )
        )
        for index, command in enumerate(regenerate_raw)
    )
    if files and not regenerate:
        raise _error(
            source,
            MERGE_CONFIG_FILENAME,
            "generated_files are declared but no `regenerate` commands are declared",
        )
    return MergeConfigSchema(
        schema_version=SCHEMA_VERSION,
        base=_string(normalized.get("base", "main"), source=source, path="merge.base"),
        batch_size=_integer(
            normalized.get("batch_size"),
            source=source,
            path=f"{MERGE_CONFIG_FILENAME}.batch_size",
            default=8,
            minimum=1,
        ),
        environment_signature=(
            _argv(
                normalized.get("environment_signature"),
                source=source,
                path=f"{MERGE_CONFIG_FILENAME}.environment_signature",
            )
            if normalized.get("environment_signature")
            else ()
        ),
        gates=tuple(gates),
        generated=GeneratedFileContract(files=files, regenerate=regenerate),
        source=source,
    )


def runtime_tier(stage: str) -> str:
    """Map an explicit stage to the legacy queue's fast/slow projection."""

    return "fast" if stage in {"feedback", "integration"} else "slow"


class _UniqueKeyLoader(yaml.SafeLoader):
    """SafeLoader variant that refuses duplicate mapping keys."""


def _construct_unique_mapping(
    loader: yaml.SafeLoader, node: yaml.MappingNode
) -> dict[str, Any]:
    mapping: dict[str, Any] = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node)
        if not isinstance(key, str):
            raise ConfigSchemaError(f"mapping keys must be strings (got {key!r})")
        if key in mapping:
            raise ConfigSchemaError(f"duplicate YAML key {key!r}")
        mapping[key] = loader.construct_object(value_node)
    return mapping


_UniqueKeyLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, _construct_unique_mapping
)


def load_yaml_mapping_text(text: str, *, source: str = "<string>") -> dict[str, Any]:
    """Load one YAML string and reject duplicate keys before schema parsing."""

    try:
        # _UniqueKeyLoader (above) is yaml.SafeLoader with only its mapping
        # constructor overridden to reject duplicate keys; every other tag
        # still resolves through SafeLoader's safe constructors, so this
        # carries the same guarantees as yaml.safe_load(). Bandit's B506
        # check only recognizes the loader by name, not by base class, so a
        # SafeLoader subclass trips the same warning as a genuinely unsafe one.
        value = yaml.load(text, Loader=_UniqueKeyLoader)  # nosec B506
    except ConfigSchemaError as exc:
        raise ConfigSchemaError(f"{source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise ConfigSchemaError(f"{source}: invalid YAML: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ConfigSchemaError(f"{source}: top level must be a YAML mapping")
    return dict(value)


def load_yaml_mapping(path: str) -> dict[str, Any]:
    """Load one YAML file and reject duplicate keys before schema parsing."""

    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigSchemaError(f"{path}: could not read YAML: {exc}") from exc
    return load_yaml_mapping_text(text, source=path)
