"""Validation profiles and deterministic stage/gate selection.

RMDD-11 keeps validation policy separate from the merge queue and from the
executor.  A profile is a declarative, immutable description of the gates a
repository may run.  It does not start a process or acquire a reservation.
The runner consumes these objects and is therefore able to plan a complete
validation DAG before any side effect takes place.

The module deliberately accepts the versioned ``.mergequeue.yaml`` parser as
an input seam instead of duplicating YAML parsing.  A repository may override
the built-in family by providing a validated merge-queue declaration; unknown
or unsafe fields remain the parser's refusal, not an ignored policy option.
"""

from __future__ import annotations

import fnmatch
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from repository_manager.config_schema import (
    ConfigSchemaError,
    GateSchema,
    load_yaml_mapping,
    parse_merge_config,
)
from repository_manager.development import ResourceRequest, ValidationStage
from repository_manager.development.serialization import canonical_digest


class ValidationPolicyError(ValueError):
    """A profile or repository override is not safe to execute."""


class GateMode(StrEnum):
    """How a gate contributes to its stage's decision."""

    BLOCKING = "blocking"
    ADVISORY = "advisory"
    DEFERRED = "deferred"


class BaselineMode(StrEnum):
    """How a gate compares an immutable tree with its base."""

    DIFFERENTIAL = "differential"
    ABSOLUTE = "absolute"
    DISABLED = "disabled"


class TimeoutPolicy(StrEnum):
    """The safe result when a gate exceeds its process deadline."""

    FAIL = "fail"
    DEFER = "defer"


_SAFE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]*$")
_RELATIVE_PATTERN = re.compile(r"^[^\x00]+$")
_SHELL_META = frozenset({";", "&&", "||", "|", ">", ">>", "<", "<<"})
_VALID_COMPARE = frozenset({"exit", "lines", "pytest-ids"})


def _nonblank(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ValidationPolicyError(f"{field_name} must be a non-blank string")
    if any(ord(char) < 0x20 for char in value):
        raise ValidationPolicyError(f"{field_name} must not contain control characters")
    return value


def _tuple_strings(value: object, field_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, (str, bytes, bytearray)):
        raise ValidationPolicyError(f"{field_name} must be a sequence of strings")
    try:
        values: tuple[object, ...] = tuple(value)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValidationPolicyError(
            f"{field_name} must be a sequence of strings"
        ) from exc
    return tuple(_nonblank(item, f"{field_name} entry") for item in values)


def _relative_patterns(value: object, field_name: str) -> tuple[str, ...]:
    patterns = _tuple_strings(value, field_name)
    result: list[str] = []
    for pattern in patterns:
        if not _RELATIVE_PATTERN.fullmatch(pattern):
            raise ValidationPolicyError(f"{field_name} contains NUL")
        path = Path(pattern)
        if path.is_absolute() or ".." in path.parts:
            raise ValidationPolicyError(
                f"{field_name} entries must be relative and within the worktree"
            )
        result.append(pattern)
    return tuple(sorted(set(result)))


def _argv(value: object, field_name: str = "command") -> tuple[str, ...]:
    values = _tuple_strings(value, field_name)
    if not values:
        raise ValidationPolicyError(f"{field_name} must not be empty")
    if any(item in _SHELL_META for item in values):
        raise ValidationPolicyError(
            f"{field_name} must be fixed argv and must not contain shell operators"
        )
    return values


def _digest_resource(request: ResourceRequest) -> str:
    return canonical_digest(request.model_dump(mode="json", exclude_none=False))


@dataclass(frozen=True, slots=True)
class PathSelection:
    """Include/exclude glob policy for changed-file selection."""

    include: tuple[str, ...] = ()
    exclude: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "include", _relative_patterns(self.include, "include"))
        object.__setattr__(self, "exclude", _relative_patterns(self.exclude, "exclude"))

    def matches(self, changed_paths: Sequence[str] | None) -> bool:
        """Return whether this gate must run for a changed-path snapshot.

        ``None`` means the caller could not obtain a complete snapshot and is
        treated conservatively as affected.  An empty sequence is a known
        no-change snapshot; gates without an include selector still run.
        """

        if changed_paths is None:
            return True
        if not changed_paths:
            # An empty, known snapshot means no paths changed.  A gate with
            # an explicit include selector therefore has no affected input;
            # an unscoped gate remains the conservative default.
            return not self.include
        eligible = [
            path
            for path in changed_paths
            if not any(fnmatch.fnmatch(path, pattern) for pattern in self.exclude)
        ]
        if not eligible:
            return False
        if not self.include:
            return True
        for path in eligible:
            if any(fnmatch.fnmatch(path, pattern) for pattern in self.include):
                return True
        return False

    def canonical_payload(self) -> dict[str, tuple[str, ...]]:
        return {"include": self.include, "exclude": self.exclude}


@dataclass(frozen=True, slots=True)
class ValidationGate:
    """One immutable executable gate declaration."""

    name: str
    command: tuple[str, ...]
    stage: ValidationStage = ValidationStage.INTEGRATION
    mode: GateMode = GateMode.BLOCKING
    path_selection: PathSelection = field(default_factory=PathSelection)
    resources: ResourceRequest = field(default_factory=ResourceRequest)
    timeout_seconds: int = 300
    baseline_timeout_seconds: int = 0
    baseline_mode: BaselineMode = BaselineMode.DIFFERENTIAL
    compare: str = "lines"
    artifact_dependencies: tuple[str, ...] = ()
    timeout_policy: TimeoutPolicy = TimeoutPolicy.FAIL
    command_env_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        name = _nonblank(self.name, "gate name")
        if not _SAFE_NAME.fullmatch(name):
            raise ValidationPolicyError(f"gate name is unsafe: {name!r}")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "command", _argv(self.command))
        if self.timeout_seconds < 1:
            raise ValidationPolicyError("gate timeout_seconds must be positive")
        if self.baseline_timeout_seconds < 0:
            raise ValidationPolicyError("baseline_timeout_seconds cannot be negative")
        if self.compare not in _VALID_COMPARE:
            raise ValidationPolicyError(
                f"compare must be one of {sorted(_VALID_COMPARE)}"
            )
        deps = _tuple_strings(self.artifact_dependencies, "artifact_dependencies")
        if self.name in deps:
            raise ValidationPolicyError("a gate cannot depend on itself")
        object.__setattr__(self, "artifact_dependencies", tuple(sorted(set(deps))))
        refs = _tuple_strings(self.command_env_refs, "command_env_refs")
        object.__setattr__(self, "command_env_refs", tuple(sorted(set(refs))))

    @property
    def config_digest(self) -> str:
        """Digest of all policy fields that affect execution or interpretation."""

        return canonical_digest(self.canonical_payload())

    @property
    def resource_digest(self) -> str:
        return _digest_resource(self.resources)

    @property
    def runs_precommit(self) -> bool:
        """Whether this fixed argv really replays the repository hook set.

        Replay proof must not be inferred from a gate name or from an arbitrary
        argument containing ``pre-commit``.  Only the two deliberately narrow
        executable shapes below are admitted: the literal trusted
        ``pre-commit`` command, or the literal trusted ``python``/``python3``
        module form.  Relative and absolute paths are refused even when their
        basename is ``pre-commit``; the candidate must not choose the
        executable that supplies replay proof.  The complete argv must be
        exactly ``run --all-files``; positional hook IDs, ``--files``, config
        overrides, and arbitrary extra arguments are refused because a
        partial hook invocation cannot certify a complete deferred snapshot.
        """

        argv = self.command
        executable = argv[0]
        if executable == "pre-commit":
            return argv[1:] == ("run", "--all-files")
        if executable in {"python", "python3"}:
            return argv[1:] == ("-m", "pre_commit", "run", "--all-files")
        return False

    def canonical_payload(self) -> dict[str, object]:
        return {
            "name": self.name,
            "command": self.command,
            "stage": self.stage,
            "mode": self.mode,
            "path_selection": self.path_selection.canonical_payload(),
            "resources": self.resources.model_dump(mode="json", exclude_none=False),
            "timeout_seconds": self.timeout_seconds,
            "baseline_timeout_seconds": self.baseline_timeout_seconds,
            "baseline_mode": self.baseline_mode,
            "compare": self.compare,
            "artifact_dependencies": self.artifact_dependencies,
            "timeout_policy": self.timeout_policy,
            "command_env_refs": self.command_env_refs,
        }

    def selected_for(self, changed_paths: Sequence[str] | None) -> bool:
        return self.path_selection.matches(changed_paths)


@dataclass(frozen=True, slots=True)
class ValidationProfile:
    """An immutable family of ordered validation gates."""

    family: str
    profile_version: int
    gates: tuple[ValidationGate, ...]
    source: str = "builtin"

    def __post_init__(self) -> None:
        family = _nonblank(self.family, "profile family")
        if not _SAFE_NAME.fullmatch(family):
            raise ValidationPolicyError(f"profile family is unsafe: {family!r}")
        if self.profile_version < 1:
            raise ValidationPolicyError("profile_version must be positive")
        names = [gate.name for gate in self.gates]
        if len(names) != len(set(names)):
            raise ValidationPolicyError("profile gate names must be unique")
        by_stage = {stage: index for index, stage in enumerate(ValidationStage)}
        if any(
            by_stage[gate.stage] < by_stage[self.gates[index - 1].stage]
            for index, gate in enumerate(self.gates)
            if index
        ):
            raise ValidationPolicyError("profile gates must be in stage order")
        known = set(names)
        gate_by_name = {gate.name: gate for gate in self.gates}
        for gate in self.gates:
            missing = set(gate.artifact_dependencies) - known
            if missing:
                raise ValidationPolicyError(
                    f"gate {gate.name!r} references unknown dependencies: "
                    + ", ".join(sorted(missing))
                )
            for dependency in gate.artifact_dependencies:
                if by_stage[gate_by_name[dependency].stage] >= by_stage[gate.stage]:
                    raise ValidationPolicyError(
                        f"gate {gate.name!r} dependency {dependency!r} must be an earlier stage"
                    )

    @property
    def digest(self) -> str:
        return canonical_digest(self.canonical_payload())

    def canonical_payload(self) -> dict[str, object]:
        return {
            "family": self.family,
            "profile_version": self.profile_version,
            "gates": tuple(gate.canonical_payload() for gate in self.gates),
            "source": self.source,
        }

    def gates_for(
        self,
        changed_paths: Sequence[str] | None,
        *,
        stages: Iterable[ValidationStage] | None = None,
    ) -> tuple[ValidationGate, ...]:
        requested = set(stages) if stages is not None else set(ValidationStage)
        return tuple(
            gate
            for gate in self.gates
            if gate.stage in requested and gate.selected_for(changed_paths)
        )


def _resource(
    *,
    resource_class: str,
    cpu: int,
    memory: int,
    disk: int,
    slots: int = 1,
) -> ResourceRequest:
    return ResourceRequest(
        resource_class=resource_class,
        concurrency_key=resource_class,
        cpu_weight=cpu,
        memory_mib=memory,
        disk_mib=disk,
        process_slots=slots,
    )


def _gate(
    name: str,
    command: tuple[str, ...],
    stage: ValidationStage,
    *,
    mode: GateMode = GateMode.BLOCKING,
    include: tuple[str, ...] = (),
    resource: ResourceRequest | None = None,
    timeout: int = 300,
    baseline: BaselineMode = BaselineMode.DIFFERENTIAL,
    compare: str = "lines",
    dependencies: tuple[str, ...] = (),
    timeout_policy: TimeoutPolicy = TimeoutPolicy.FAIL,
) -> ValidationGate:
    return ValidationGate(
        name=name,
        command=command,
        stage=stage,
        mode=mode,
        path_selection=PathSelection(include=include),
        resources=resource or ResourceRequest(),
        timeout_seconds=timeout,
        baseline_mode=baseline,
        compare=compare,
        artifact_dependencies=dependencies,
        timeout_policy=timeout_policy,
    )


def builtin_profiles() -> dict[str, ValidationProfile]:
    """Return the conservative built-in profile families.

    Commands are declarations only.  They execute only after a caller has
    resolved a repository's profile and admitted the corresponding WorkItem.
    """

    light = _resource(resource_class="light-check", cpu=1, memory=512, disk=512)
    rust = _resource(resource_class="rust-build", cpu=8, memory=8192, disk=4096)
    front = _resource(resource_class="frontend-build", cpu=6, memory=6144, disk=4096)
    release = _resource(
        resource_class="workspace-release", cpu=4, memory=4096, disk=2048
    )

    return {
        "docs": ValidationProfile(
            family="docs",
            profile_version=1,
            gates=(
                _gate(
                    "docs-format",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.FEEDBACK,
                    include=("*.md", "docs/**"),
                    resource=light,
                ),
                _gate(
                    "docs-links",
                    ("python", "-m", "compileall", "-q", "docs"),
                    ValidationStage.INTEGRATION,
                    mode=GateMode.ADVISORY,
                    include=("*.md", "docs/**"),
                    resource=light,
                ),
                _gate(
                    "docs-certification",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.CERTIFICATION,
                    include=("*.md", "docs/**"),
                    resource=light,
                ),
            ),
        ),
        "python": ValidationProfile(
            family="python",
            profile_version=1,
            gates=(
                _gate(
                    "python-lint",
                    ("ruff", "check", "."),
                    ValidationStage.FEEDBACK,
                    resource=light,
                ),
                _gate(
                    "python-focused",
                    ("pytest", "-q"),
                    ValidationStage.INTEGRATION,
                    resource=light,
                    include=("*.py", "tests/**"),
                ),
                _gate(
                    "python-certification",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.CERTIFICATION,
                    resource=light,
                ),
                _gate(
                    "python-smoke",
                    ("python", "-m", "compileall", "-q", "."),
                    ValidationStage.SMOKE,
                    resource=light,
                ),
                _gate(
                    "python-release",
                    ("python", "-m", "build"),
                    ValidationStage.RELEASE,
                    resource=release,
                ),
            ),
        ),
        "rust": ValidationProfile(
            family="rust",
            profile_version=1,
            gates=(
                _gate(
                    "rust-format",
                    ("cargo", "fmt", "--", "--check"),
                    ValidationStage.FEEDBACK,
                    resource=rust,
                    include=("*.rs", "Cargo.*"),
                ),
                _gate(
                    "rust-check",
                    ("cargo", "check", "--all-features"),
                    ValidationStage.INTEGRATION,
                    resource=rust,
                    include=("*.rs", "Cargo.*"),
                ),
                _gate(
                    "rust-certification",
                    ("cargo", "test", "--all-features"),
                    ValidationStage.CERTIFICATION,
                    resource=rust,
                    include=("*.rs", "Cargo.*"),
                ),
                _gate(
                    "rust-smoke",
                    ("cargo", "check", "--all-features"),
                    ValidationStage.SMOKE,
                    resource=rust,
                ),
            ),
        ),
        "frontend": ValidationProfile(
            family="frontend",
            profile_version=1,
            gates=(
                _gate(
                    "frontend-lint",
                    ("pnpm", "lint"),
                    ValidationStage.FEEDBACK,
                    resource=front,
                    include=("*.ts", "*.tsx", "*.js", "*.jsx", "package.json"),
                ),
                _gate(
                    "frontend-test",
                    ("pnpm", "test", "--", "--runInBand"),
                    ValidationStage.INTEGRATION,
                    resource=front,
                    include=("*.ts", "*.tsx", "*.js", "*.jsx"),
                ),
                _gate(
                    "frontend-build",
                    ("pnpm", "build"),
                    ValidationStage.CERTIFICATION,
                    resource=front,
                    include=("*.ts", "*.tsx", "*.js", "*.jsx", "package.json"),
                ),
                _gate(
                    "frontend-smoke",
                    ("pnpm", "lint"),
                    ValidationStage.SMOKE,
                    resource=front,
                ),
            ),
        ),
        "schema": ValidationProfile(
            family="schema",
            profile_version=1,
            gates=(
                _gate(
                    "schema-parse",
                    ("python", "-m", "compileall", "-q", "."),
                    ValidationStage.FEEDBACK,
                    resource=light,
                    include=("*.yaml", "*.yml", "*.json", "*.toml"),
                ),
                _gate(
                    "schema-integration",
                    ("pytest", "-q", "-m", "schema"),
                    ValidationStage.INTEGRATION,
                    resource=light,
                    include=("*.yaml", "*.yml", "*.json", "*.toml"),
                ),
                _gate(
                    "schema-certification",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.CERTIFICATION,
                    resource=light,
                ),
            ),
        ),
        "concept": ValidationProfile(
            family="concept",
            profile_version=1,
            gates=(
                _gate(
                    "concept-fragments",
                    ("python", "scripts", "check_concepts.py"),
                    ValidationStage.FEEDBACK,
                    resource=light,
                    include=("**/CONCEPTS.md", "**/*.py"),
                ),
                _gate(
                    "concept-integration",
                    ("pytest", "-q", "-m", "concept"),
                    ValidationStage.INTEGRATION,
                    resource=light,
                    include=("**/CONCEPTS.md", "**/*.py"),
                ),
                _gate(
                    "concept-certification",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.CERTIFICATION,
                    resource=light,
                ),
            ),
        ),
        "deployment": ValidationProfile(
            family="deployment",
            profile_version=1,
            gates=(
                _gate(
                    "deployment-config",
                    ("python", "-m", "compileall", "-q", "."),
                    ValidationStage.FEEDBACK,
                    resource=light,
                    include=("*.yaml", "*.yml", "Dockerfile*", "docker/**"),
                ),
                _gate(
                    "deployment-integration",
                    ("docker", "compose", "config", "--quiet"),
                    ValidationStage.INTEGRATION,
                    resource=release,
                    include=("*.yaml", "*.yml", "Dockerfile*", "docker/**"),
                ),
                _gate(
                    "deployment-certification",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.CERTIFICATION,
                    resource=release,
                ),
                _gate(
                    "deployment-smoke",
                    ("docker", "compose", "config", "--quiet"),
                    ValidationStage.SMOKE,
                    resource=release,
                ),
            ),
        ),
        "release": ValidationProfile(
            family="release",
            profile_version=1,
            gates=(
                _gate(
                    "release-validate",
                    ("python", "-m", "compileall", "-q", "."),
                    ValidationStage.FEEDBACK,
                    resource=release,
                ),
                _gate(
                    "release-certification",
                    ("pre-commit", "run", "--all-files"),
                    ValidationStage.CERTIFICATION,
                    resource=release,
                ),
                _gate(
                    "release-smoke",
                    ("python", "-m", "compileall", "-q", "."),
                    ValidationStage.SMOKE,
                    resource=release,
                ),
                _gate(
                    "release-dag",
                    ("python", "-m", "build"),
                    ValidationStage.RELEASE,
                    resource=release,
                ),
            ),
        ),
    }


def _gate_from_schema(
    gate: GateSchema, *, mode: GateMode = GateMode.BLOCKING
) -> ValidationGate:
    """Convert the repository's validated config schema to C-06 policy."""

    try:
        stage = ValidationStage(gate.stage)
        baseline = BaselineMode(gate.baseline_mode)
        timeout = TimeoutPolicy(gate.on_timeout)
    except ValueError as exc:
        raise ValidationPolicyError(str(exc)) from exc
    return ValidationGate(
        name=gate.name,
        command=gate.command,
        stage=stage,
        mode=mode,
        path_selection=PathSelection(
            include=gate.path_selection.include,
            exclude=gate.path_selection.exclude,
        ),
        resources=ResourceRequest(
            resource_class=gate.resource_class,
            concurrency_key=gate.resource_class,
            cpu_weight=max(1, gate.resources.cpu_weight),
            memory_mib=max(1, gate.resources.memory_mb),
            disk_mib=max(1, gate.resources.disk_mb),
            process_slots=max(1, gate.resources.process_slots),
        ),
        timeout_seconds=gate.timeout,
        baseline_timeout_seconds=gate.baseline_timeout,
        baseline_mode=baseline,
        compare=gate.compare,
        artifact_dependencies=gate.artifact_dependencies,
        timeout_policy=timeout,
    )


def profile_from_merge_config(
    data: Mapping[str, Any], *, family: str = "repository", source: str = ""
) -> ValidationProfile:
    """Validate a repository override and return its immutable C-06 profile."""

    # C-06 adds ``mode`` to the policy layer while RMDD-03 remains the sole
    # owner of the persisted parser schema.  Remove and validate that one
    # additive field here, then let the canonical parser reject every other
    # unknown safety-relevant key.
    normalized = dict(data)
    raw_gates = normalized.get("gates", [])
    modes: dict[str, GateMode] = {}
    if isinstance(raw_gates, Sequence) and not isinstance(raw_gates, (str, bytes)):
        copied_gates: list[dict[str, Any]] = []
        for index, raw_gate in enumerate(raw_gates):
            if not isinstance(raw_gate, Mapping):
                copied_gates.append(raw_gate)  # parser reports the precise shape error
                continue
            copied = dict(raw_gate)
            raw_mode = copied.pop("mode", GateMode.BLOCKING.value)
            try:
                mode = GateMode(raw_mode)
            except ValueError as exc:
                raise ValidationPolicyError(
                    f"gates[{index}].mode must be blocking, advisory, or deferred"
                ) from exc
            raw_name = copied.get("name")
            if isinstance(raw_name, str):
                modes[raw_name] = mode
            copied_gates.append(copied)
        normalized["gates"] = copied_gates
    try:
        schema = parse_merge_config(normalized, source=source)
    except (ConfigSchemaError, TypeError, ValueError) as exc:
        raise ValidationPolicyError(str(exc)) from exc
    gates = tuple(
        _gate_from_schema(gate, mode=modes.get(gate.name, GateMode.BLOCKING))
        for gate in schema.gates
    )
    if not gates:
        raise ValidationPolicyError(
            f"{source or '.mergequeue.yaml'} declares no validation gates"
        )
    return ValidationProfile(
        family=family,
        profile_version=schema.schema_version,
        gates=gates,
        source=source or ".mergequeue.yaml",
    )


class ValidationProfileRegistry:
    """Thread-safe-by-ownership registry for built-ins and repository overrides."""

    def __init__(self, profiles: Mapping[str, ValidationProfile] | None = None) -> None:
        self._profiles = dict(profiles or builtin_profiles())

    def register(self, profile: ValidationProfile, *, replace: bool = False) -> None:
        if profile.family in self._profiles and not replace:
            raise ValidationPolicyError(
                f"profile family already exists: {profile.family}"
            )
        self._profiles[profile.family] = profile

    def resolve(self, family: str) -> ValidationProfile:
        try:
            return self._profiles[family]
        except KeyError as exc:
            raise ValidationPolicyError(
                f"unknown validation profile family: {family}"
            ) from exc

    def resolve_repository(
        self,
        tree: Path | str,
        *,
        family: str = "python",
        override: Mapping[str, Any] | None = None,
    ) -> ValidationProfile:
        """Resolve a built-in family, replacing it with a validated repo policy."""

        if override is not None:
            return profile_from_merge_config(override, family=family, source="override")
        path = Path(tree).expanduser()
        config = path / ".mergequeue.yaml"
        if config.is_file():
            try:
                return profile_from_merge_config(
                    load_yaml_mapping(str(config)), family=family, source=str(config)
                )
            except (OSError, ConfigSchemaError, ValidationPolicyError) as exc:
                if isinstance(exc, ValidationPolicyError):
                    raise
                raise ValidationPolicyError(str(exc)) from exc
        return self.resolve(family)


# The contract calls this object a validation policy in prose.  Keep a
# descriptive alias for consumers that do not need to distinguish a profile
# family from one resolved policy instance.
ValidationPolicy = ValidationProfile


__all__ = [
    "BaselineMode",
    "GateMode",
    "PathSelection",
    "TimeoutPolicy",
    "ValidationGate",
    "ValidationPolicyError",
    "ValidationPolicy",
    "ValidationProfile",
    "ValidationProfileRegistry",
    "builtin_profiles",
    "profile_from_merge_config",
]
