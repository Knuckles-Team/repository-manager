"""Fail-closed authority for the exact documentation-readiness fleet rollout.

This module owns selection, wave planning, evidence, and transaction semantics.  It
does not discover sibling directories, edit a repository, create branches, or invoke
the fleet by itself.  The canonical universal-skills generator is supplied through a
small adapter after NE-137/NE-144 have landed; keeping that dependency explicit makes
the rollout reviewable and prevents a copied or stale generator from becoming a
second authority.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

try:
    import yaml
except ImportError as exc:  # pragma: no cover - dependency is declared by RM
    yaml = None
    _YAML_IMPORT_ERROR = exc
else:
    _YAML_IMPORT_ERROR = None


SCHEMA_VERSION = "docs-readiness-rollout/v1"
EXPECTED_PUBLISHABLE_COUNT = 75
MANIFEST_FILENAME = "docs_readiness_fleet_manifest.json"
MANIFEST_SOURCE_AUTHORITY = "workspace.yml and the declared repository checkouts"
MANIFEST_SURFACE_POLICY = (
    ("generator", "universal-skills-agent-readiness/v1"),
    ("pages_workflow", "pages-job-or-reusable-pages-workflow"),
    ("site_url", "mkdocs-site-url"),
    ("tck", "pages-readiness-tck/v1"),
)
MAX_MANIFEST_BYTES = 1_000_000
MAX_JOURNAL_BYTES = 8_000_000
MAX_JOURNAL_RECORDS = 20_000
MAX_SOURCE_FILE_BYTES = 2_000_000
MAX_SOURCE_FILES = 256
MAX_GENERATED_OUTPUTS = 128
MAX_GENERATED_OUTPUT_NAME = 240
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_REVISION = re.compile(r"^[0-9a-f]{40,64}$")
_IDENTITY = re.compile(r"^agent-packages/[A-Za-z0-9][A-Za-z0-9._/-]*$")
_SAFE_ERROR = re.compile(r"^[a-z0-9][a-z0-9-]{0,79}$")
_TRANSACTION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,95}$")
_VERSION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+-]{0,63}$")


class RolloutError(ValueError):
    """A bounded selection, safety, provenance, or transaction refusal."""


@dataclass(frozen=True)
class ProjectSpec:
    """One manifest-owned publishable project."""

    identity: str
    wave: int


@dataclass(frozen=True)
class WaveSpec:
    """A deterministic wave declaration and its expected cardinality."""

    number: int
    name: str
    expected_count: int


@dataclass(frozen=True)
class SourceFindings:
    """Source audit findings that must remain stable until the rollout starts."""

    missing_pages_workflows: tuple[str, ...]
    missing_site_urls: tuple[str, ...]

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "missing_pages_workflows": list(self.missing_pages_workflows),
            "missing_site_urls": list(self.missing_site_urls),
        }


@dataclass(frozen=True)
class FleetManifest:
    """Validated, content-addressed selection authority."""

    schema_version: str
    manifest_name: str
    expected_publishable_count: int
    excluded_identities: tuple[str, ...]
    waves: tuple[WaveSpec, ...]
    projects: tuple[ProjectSpec, ...]
    source_findings: SourceFindings
    digest: str
    source_authority: str = MANIFEST_SOURCE_AUTHORITY
    surface_policy: tuple[tuple[str, str], ...] = MANIFEST_SURFACE_POLICY


@dataclass(frozen=True)
class WavePlan:
    """The immutable plan for one wave."""

    number: int
    name: str
    projects: tuple[ProjectSpec, ...]

    @property
    def count(self) -> int:
        return len(self.projects)


@dataclass(frozen=True)
class RepositoryEvidence:
    """Privacy-safe revision/worktree evidence for one target."""

    identity: str
    head_revision: str
    branch: str
    clean: bool
    worktree_count: int
    source_digest: str

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RolloutDependencies:
    """Exact immutable revisions required from NE-137 and NE-144."""

    generator_revision: str
    tck_revision: str
    generator_schema: str = "agent-readiness/v1"
    tck_schema: str = "pages-readiness-tck/v1"

    def validate(self) -> None:
        if not _REVISION.fullmatch(self.generator_revision):
            raise RolloutError("generator-revision-invalid")
        if not _REVISION.fullmatch(self.tck_revision):
            raise RolloutError("tck-revision-invalid")
        if self.generator_schema != "agent-readiness/v1":
            raise RolloutError("generator-schema-unsupported")
        if self.tck_schema != "pages-readiness-tck/v1":
            raise RolloutError("tck-schema-unsupported")


class RolloutAdapter(Protocol):
    """The only seam allowed to invoke the canonical generator/TCK."""

    generator_revision: str
    generator_version: str
    generator_schema: str
    tck_revision: str
    tck_schema: str

    def preview(self, repository_root: Path, project: ProjectSpec) -> Mapping[str, Any]:
        """Plan into a bounded staging directory without mutating the target."""

    def apply(self, repository_root: Path, project: ProjectSpec) -> Mapping[str, Any]:
        """Publish through the canonical generator's atomic apply path."""

    def verify(self, repository_root: Path, project: ProjectSpec) -> Mapping[str, Any]:
        """Verify governed artifacts and return a provenance digest."""

    def rollback(
        self,
        repository_root: Path,
        project: ProjectSpec,
        evidence: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Restore only the generator-owned artifact namespace."""


def _canonical_json(value: object) -> bytes:
    try:
        return json.dumps(
            value, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RolloutError("manifest-json-invalid") from exc


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bounded(path: Path, *, maximum: int, error: str) -> bytes:
    try:
        metadata = path.lstat()
        if path.is_symlink() or not path.is_file() or metadata.st_nlink != 1:
            raise RolloutError(error)
        if metadata.st_size > maximum:
            raise RolloutError(f"{error}-oversize")
        return path.read_bytes()
    except RolloutError:
        raise
    except OSError as exc:
        raise RolloutError(error) from exc


def _validate_identity(value: object) -> str:
    if not isinstance(value, str) or not _IDENTITY.fullmatch(value):
        raise RolloutError("manifest-identity-invalid")
    parts = PurePosixPath(value).parts
    if any(part in {"", ".", ".."} for part in parts):
        raise RolloutError("manifest-identity-invalid")
    if "agents/tests" in value:
        raise RolloutError("manifest-test-project-publishable")
    return value


def _manifest_payload(manifest: FleetManifest) -> dict[str, Any]:
    return {
        "schema_version": manifest.schema_version,
        "manifest_name": manifest.manifest_name,
        "expected_publishable_count": manifest.expected_publishable_count,
        "source_authority": manifest.source_authority,
        "surface_policy": dict(manifest.surface_policy),
        "excluded": [
            {"identity": identity, "reason": "test-fixture-not-publishable"}
            for identity in manifest.excluded_identities
        ],
        "waves": [asdict(wave) for wave in manifest.waves],
        "projects": [asdict(project) for project in manifest.projects],
        "source_findings": manifest.source_findings.as_dict(),
    }


def default_manifest_path() -> Path:
    """Return the packaged exact-75 manifest without creating anything."""

    return Path(__file__).with_name(MANIFEST_FILENAME)


def load_fleet_manifest(path: Path | None = None) -> FleetManifest:
    """Load and validate the exact 75-project manifest."""

    source = path or default_manifest_path()
    raw = _read_bounded(source, maximum=MAX_MANIFEST_BYTES, error="manifest-unavailable")
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RolloutError("manifest-json-invalid") from exc
    if not isinstance(data, dict) or data.get("schema_version") != SCHEMA_VERSION:
        raise RolloutError("manifest-schema-invalid")
    if set(data) != {
        "schema_version",
        "manifest_name",
        "source_authority",
        "expected_publishable_count",
        "excluded",
        "surface_policy",
        "source_findings",
        "waves",
        "projects",
    }:
        raise RolloutError("manifest-schema-invalid")
    if not isinstance(data.get("manifest_name"), str) or not data["manifest_name"]:
        raise RolloutError("manifest-name-invalid")
    if data.get("source_authority") != MANIFEST_SOURCE_AUTHORITY:
        raise RolloutError("manifest-source-authority-invalid")
    policy = data.get("surface_policy")
    if (
        not isinstance(policy, dict)
        or set(policy) != {key for key, _ in MANIFEST_SURFACE_POLICY}
        or any(policy[key] != value for key, value in MANIFEST_SURFACE_POLICY)
    ):
        raise RolloutError("manifest-surface-policy-invalid")
    expected = data.get("expected_publishable_count")
    if type(expected) is not int or expected != EXPECTED_PUBLISHABLE_COUNT:
        raise RolloutError("manifest-cardinality-invalid")

    excluded_raw = data.get("excluded")
    if not isinstance(excluded_raw, list) or len(excluded_raw) != 1:
        raise RolloutError("manifest-exclusions-invalid")
    excluded = excluded_raw[0]
    if (
        not isinstance(excluded, dict)
        or set(excluded) != {"identity", "reason"}
        or excluded.get("identity") != "agent-packages/agents/tests"
        or excluded.get("reason") != "test-fixture-not-publishable"
    ):
        raise RolloutError("manifest-exclusions-invalid")
    excluded_identities = ("agent-packages/agents/tests",)

    waves_raw = data.get("waves")
    if not isinstance(waves_raw, list) or not waves_raw:
        raise RolloutError("manifest-waves-invalid")
    waves: list[WaveSpec] = []
    for row in waves_raw:
        if not isinstance(row, dict) or set(row) != {"number", "name", "expected_count"}:
            raise RolloutError("manifest-waves-invalid")
        number, name, count = row.get("number"), row.get("name"), row.get("expected_count")
        if (
            type(number) is not int
            or number < 1
            or not isinstance(name, str)
            or not re.fullmatch(r"[a-z0-9-]{1,48}", name)
            or type(count) is not int
            or count < 1
        ):
            raise RolloutError("manifest-waves-invalid")
        waves.append(WaveSpec(number=number, name=name, expected_count=count))
    if tuple(wave.number for wave in waves) != tuple(range(1, len(waves) + 1)):
        raise RolloutError("manifest-wave-order-invalid")

    projects_raw = data.get("projects")
    if not isinstance(projects_raw, list) or len(projects_raw) != expected:
        raise RolloutError("manifest-cardinality-invalid")
    projects: list[ProjectSpec] = []
    seen: set[str] = set()
    for row in projects_raw:
        if not isinstance(row, dict) or set(row) != {"identity", "wave"}:
            raise RolloutError("manifest-project-invalid")
        identity = _validate_identity(row.get("identity"))
        wave = row.get("wave")
        if type(wave) is not int or wave < 1 or wave > len(waves):
            raise RolloutError("manifest-project-invalid")
        if identity in seen:
            raise RolloutError("manifest-duplicate-project")
        seen.add(identity)
        projects.append(ProjectSpec(identity=identity, wave=wave))
    if tuple(project.identity for project in projects) != tuple(sorted(seen)):
        raise RolloutError("manifest-project-order-invalid")
    counts = {number: 0 for number in range(1, len(waves) + 1)}
    for project in projects:
        counts[project.wave] += 1
    if any(wave.expected_count != counts[wave.number] for wave in waves):
        raise RolloutError("manifest-wave-cardinality-invalid")

    findings_raw = data.get("source_findings")
    if (
        not isinstance(findings_raw, dict)
        or set(findings_raw) != {"missing_pages_workflows", "missing_site_urls"}
    ):
        raise RolloutError("manifest-findings-invalid")

    def finding_list(key: str, expected_count: int) -> tuple[str, ...]:
        values = findings_raw.get(key)
        if not isinstance(values, list) or len(values) != expected_count:
            raise RolloutError("manifest-findings-invalid")
        normalized = tuple(_validate_identity(value) for value in values)
        if (
            len(set(normalized)) != expected_count
            or normalized != tuple(sorted(normalized))
            or not set(normalized) <= seen
        ):
            raise RolloutError("manifest-findings-invalid")
        return normalized

    findings = SourceFindings(
        missing_pages_workflows=finding_list("missing_pages_workflows", 2),
        missing_site_urls=finding_list("missing_site_urls", 4),
    )
    provisional = FleetManifest(
        schema_version=SCHEMA_VERSION,
        manifest_name=data["manifest_name"],
        expected_publishable_count=expected,
        excluded_identities=excluded_identities,
        waves=tuple(waves),
        projects=tuple(projects),
        source_findings=findings,
        digest="",
        source_authority=data["source_authority"],
        surface_policy=tuple(MANIFEST_SURFACE_POLICY),
    )
    digest = _sha256(_canonical_json(_manifest_payload(provisional)))
    return FleetManifest(
        schema_version=provisional.schema_version,
        manifest_name=provisional.manifest_name,
        expected_publishable_count=provisional.expected_publishable_count,
        excluded_identities=provisional.excluded_identities,
        waves=provisional.waves,
        projects=provisional.projects,
        source_findings=provisional.source_findings,
        digest=digest,
        source_authority=provisional.source_authority,
        surface_policy=provisional.surface_policy,
    )


def plan_waves(
    manifest: FleetManifest,
    selected: Sequence[str] | None = None,
) -> tuple[WavePlan, ...]:
    """Return stable wave order; reject unknown, duplicate, or partial selection."""

    allowed = {project.identity for project in manifest.projects}
    if selected is None:
        selected_set = allowed
    else:
        if not selected:
            raise RolloutError("selection-empty")
        selected_tuple = tuple(selected)
        if len(set(selected_tuple)) != len(selected_tuple):
            raise RolloutError("selection-duplicate")
        if any(identity not in allowed for identity in selected_tuple):
            raise RolloutError("selection-not-in-manifest")
        selected_set = set(selected_tuple)
    plans: list[WavePlan] = []
    for wave in manifest.waves:
        projects = tuple(
            project
            for project in manifest.projects
            if project.wave == wave.number and project.identity in selected_set
        )
        if selected is None and len(projects) != wave.expected_count:
            raise RolloutError("wave-selection-drift")
        if projects:
            plans.append(WavePlan(number=wave.number, name=wave.name, projects=projects))
    if selected is None and sum(plan.count for plan in plans) != manifest.expected_publishable_count:
        raise RolloutError("fleet-selection-drift")
    return tuple(plans)


def _safe_repository_root(workspace_root: Path, identity: str) -> Path:
    """Resolve one manifest identity while rejecting symlink/path escapes."""

    try:
        workspace_metadata = workspace_root.lstat()
        root = workspace_root.resolve(strict=True)
    except OSError as exc:
        raise RolloutError("workspace-root-unavailable") from exc
    if (
        workspace_root.is_symlink()
        or workspace_metadata.st_nlink < 1
        or not root.is_dir()
    ):
        raise RolloutError("workspace-root-invalid")
    candidate = root.joinpath(*PurePosixPath(identity).parts)
    cursor = root
    for part in PurePosixPath(identity).parts:
        cursor /= part
        try:
            if cursor.is_symlink():
                raise RolloutError("repository-path-symlink")
        except OSError as exc:
            raise RolloutError("repository-path-unavailable") from exc
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise RolloutError("repository-path-containment") from exc
    if not resolved.is_dir() or not (resolved / ".git").exists():
        raise RolloutError("repository-unavailable")
    return resolved


def _git(repository_root: Path, *args: str) -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(repository_root), *args],
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RolloutError("repository-git-unavailable") from exc
    if result.returncode != 0:
        raise RolloutError("repository-git-command-failed")
    return result.stdout.strip()


def _source_digest(repository_root: Path, head: str) -> str:
    """Hash only bounded source/config inputs, never raw repository contents."""

    candidates = [
        repository_root / "AGENTS.md",
        repository_root / "mkdocs.yml",
        repository_root / "docs" / "agent-readiness.json",
        repository_root / "docs" / "agent-readiness.schema.json",
    ]
    workflow_dir = repository_root / ".github" / "workflows"
    if workflow_dir.is_dir() and not workflow_dir.is_symlink():
        candidates.extend(sorted(workflow_dir.glob("*.y*ml")))
    files: list[tuple[str, str]] = []
    existing = [path for path in candidates if path.exists()]
    if len(existing) > MAX_SOURCE_FILES:
        raise RolloutError("source-input-count-exceeded")
    for path in existing:
        payload = _read_bounded(path, maximum=MAX_SOURCE_FILE_BYTES, error="source-input-invalid")
        relative = path.relative_to(repository_root).as_posix()
        files.append((relative, _sha256(payload)))
    return _sha256(_canonical_json({"head": head, "files": files}))


def collect_repository_evidence(
    workspace_root: Path,
    project: ProjectSpec,
) -> RepositoryEvidence:
    """Collect bounded revision/worktree evidence without exposing dirty paths."""

    repository_root = _safe_repository_root(workspace_root, project.identity)
    top = Path(_git(repository_root, "rev-parse", "--show-toplevel")).resolve()
    if top != repository_root:
        raise RolloutError("repository-root-mismatch")
    head = _git(repository_root, "rev-parse", "HEAD")
    if not _REVISION.fullmatch(head):
        raise RolloutError("repository-revision-invalid")
    branch = _git(repository_root, "branch", "--show-current")
    status = _git(repository_root, "status", "--porcelain=v1", "--untracked-files=all")
    worktrees = _git(repository_root, "worktree", "list", "--porcelain")
    worktree_count = sum(1 for line in worktrees.splitlines() if line.startswith("worktree "))
    if worktree_count < 1:
        raise RolloutError("repository-worktree-invalid")
    return RepositoryEvidence(
        identity=project.identity,
        head_revision=head,
        branch=branch,
        clean=not bool(status),
        worktree_count=worktree_count,
        source_digest=_source_digest(repository_root, head),
    )


def enforce_repository_guard(
    evidence: RepositoryEvidence,
    *,
    require_clean: bool = True,
    require_main: bool = True,
    require_single_worktree: bool = True,
) -> None:
    """Refuse dirty, detached, non-main, or multiply-worktree targets."""

    if require_clean and not evidence.clean:
        raise RolloutError("repository-dirty")
    if require_main and evidence.branch != "main":
        raise RolloutError("repository-branch-not-main")
    if require_single_worktree and evidence.worktree_count != 1:
        raise RolloutError("repository-linked-worktrees-present")


def _safe_output_names(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)) or len(value) > MAX_GENERATED_OUTPUTS:
        raise RolloutError("generator-outputs-invalid")
    names: list[str] = []
    for raw in value:
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) > MAX_GENERATED_OUTPUT_NAME
            or "\\" in raw
        ):
            raise RolloutError("generator-output-invalid")
        path = PurePosixPath(raw)
        if (
            path.is_absolute()
            or not path.parts
            or raw != path.as_posix()
            or any(part in {"", ".", ".."} for part in path.parts)
        ):
            raise RolloutError("generator-output-invalid")
        names.append(path.as_posix())
    if len(set(names)) != len(names):
        raise RolloutError("generator-output-duplicate")
    return tuple(sorted(names))


def normalize_generator_result(value: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only bounded generator evidence and reject raw exceptions/paths."""

    if not isinstance(value, Mapping) or value.get("ok") is not True:
        error = value.get("error_code") if isinstance(value, Mapping) else None
        if not isinstance(error, str) or not _SAFE_ERROR.fullmatch(error):
            error = "generator-failed"
        raise RolloutError(error)
    version = value.get("generator_version")
    schema = value.get("schema_version")
    digest = value.get("provenance_digest") or value.get("artifact_digest")
    if (
        not isinstance(version, str)
        or not _VERSION.fullmatch(version)
        or not isinstance(schema, str)
        or len(schema) > 96
        or not schema
        or not isinstance(digest, str)
        or not _DIGEST.fullmatch(digest)
    ):
        raise RolloutError("generator-evidence-invalid")
    outputs = _safe_output_names(value.get("generated_outputs", ()))
    return {
        "generator_version": version,
        "schema_version": schema,
        "provenance_digest": digest,
        "generated_outputs": list(outputs),
    }


def _normalize_adapter_result(
    adapter: RolloutAdapter, value: Mapping[str, Any]
) -> dict[str, Any]:
    """Normalize one generator/TCK result against the declared adapter authority."""

    result = normalize_generator_result(value)
    if result["generator_version"] != getattr(adapter, "generator_version", None):
        raise RolloutError("generator-version-mismatch")
    if result["schema_version"] != getattr(adapter, "generator_schema", None):
        raise RolloutError("generator-schema-mismatch")
    return result


def _safe_record(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate journal shape and strip anything not part of durable evidence."""

    allowed = {
        "operation_key",
        "transaction_id",
        "mode",
        "state",
        "identity",
        "manifest_digest",
        "source_revision",
        "source_digest",
        "generator_revision",
        "generator_schema",
        "tck_revision",
        "tck_schema",
        "generator_version",
        "artifact_digest",
        "generated_outputs",
        "error_code",
    }
    if not isinstance(value, Mapping) or set(value) - allowed:
        raise RolloutError("journal-record-invalid")
    record = dict(value)
    for key in ("operation_key", "manifest_digest", "source_digest"):
        if not isinstance(record.get(key), str) or not _DIGEST.fullmatch(record[key]):
            raise RolloutError("journal-record-invalid")
    if not isinstance(record.get("transaction_id"), str) or not _TRANSACTION_ID.fullmatch(
        record["transaction_id"]
    ):
        raise RolloutError("journal-record-invalid")
    if record.get("mode") not in {"apply", "rollback"}:
        raise RolloutError("journal-record-invalid")
    if record.get("state") not in {"prepared", "applied", "rolled_back", "rollback_failed"}:
        raise RolloutError("journal-record-invalid")
    if not isinstance(record.get("identity"), str) or not _IDENTITY.fullmatch(record["identity"]):
        raise RolloutError("journal-record-invalid")
    if not isinstance(record.get("source_revision"), str) or not _REVISION.fullmatch(record["source_revision"]):
        raise RolloutError("journal-record-invalid")
    if not isinstance(record.get("generator_revision"), str) or not _REVISION.fullmatch(record["generator_revision"]):
        raise RolloutError("journal-record-invalid")
    if record.get("generator_schema") != "agent-readiness/v1":
        raise RolloutError("journal-record-invalid")
    if not isinstance(record.get("tck_revision"), str) or not _REVISION.fullmatch(record["tck_revision"]):
        raise RolloutError("journal-record-invalid")
    if record.get("tck_schema") != "pages-readiness-tck/v1":
        raise RolloutError("journal-record-invalid")
    if "generator_version" in record and (
        not isinstance(record["generator_version"], str)
        or not _VERSION.fullmatch(record["generator_version"])
    ):
        raise RolloutError("journal-record-invalid")
    if "artifact_digest" in record and (
        not isinstance(record["artifact_digest"], str) or not _DIGEST.fullmatch(record["artifact_digest"])
    ):
        raise RolloutError("journal-record-invalid")
    if "generated_outputs" in record:
        record["generated_outputs"] = list(_safe_output_names(record["generated_outputs"]))
    if "error_code" in record and (
        not isinstance(record["error_code"], str) or not _SAFE_ERROR.fullmatch(record["error_code"])
    ):
        raise RolloutError("journal-record-invalid")
    return record


class TransactionJournal:
    """Atomic, bounded JSON journal for one rollout authority."""

    def __init__(self, path: Path):
        self.path = path

    def records(self) -> tuple[dict[str, Any], ...]:
        if not self.path.exists():
            return ()
        raw = _read_bounded(self.path, maximum=MAX_JOURNAL_BYTES, error="journal-unavailable")
        try:
            data = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RolloutError("journal-invalid") from exc
        if not isinstance(data, list) or len(data) > MAX_JOURNAL_RECORDS:
            raise RolloutError("journal-invalid")
        return tuple(_safe_record(row) for row in data)

    def append(self, record: Mapping[str, Any]) -> None:
        next_records = [*self.records(), _safe_record(record)]
        if len(next_records) > MAX_JOURNAL_RECORDS:
            raise RolloutError("journal-capacity-exceeded")
        payload = _canonical_json(next_records)
        parent = self.path.parent
        if not parent.is_dir() or parent.is_symlink():
            raise RolloutError("journal-parent-invalid")
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb", dir=parent, prefix=f".{self.path.name}.", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except OSError as exc:
            if temporary is not None:
                try:
                    temporary.unlink(missing_ok=True)
                except OSError:
                    pass
            raise RolloutError("journal-write-failed") from exc


def _operation_key(
    manifest: FleetManifest,
    project: ProjectSpec,
    evidence: RepositoryEvidence,
    dependencies: RolloutDependencies,
) -> str:
    dependencies.validate()
    return _sha256(
        _canonical_json(
            {
                "manifest": manifest.digest,
                "identity": project.identity,
                "source_revision": evidence.head_revision,
                "source_digest": evidence.source_digest,
                "generator_revision": dependencies.generator_revision,
                "tck_revision": dependencies.tck_revision,
            }
        )
    )


def _matching_records(
    records: Sequence[Mapping[str, Any]], operation_key: str
) -> tuple[Mapping[str, Any], ...]:
    return tuple(record for record in records if record.get("operation_key") == operation_key)


def _active_identity_conflict(
    records: Sequence[Mapping[str, Any]], identity: str, operation_key: str
) -> bool:
    terminal = {"applied", "rolled_back"}
    for record in records:
        if record.get("identity") == identity and record.get("operation_key") != operation_key:
            if record.get("state") not in terminal:
                return True
    return False


def _base_record(
    *,
    operation_key: str,
    transaction_id: str,
    mode: str,
    state: str,
    manifest: FleetManifest,
    evidence: RepositoryEvidence,
    dependencies: RolloutDependencies,
    error_code: str | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "operation_key": operation_key,
        "transaction_id": transaction_id,
        "mode": mode,
        "state": state,
        "identity": evidence.identity,
        "manifest_digest": manifest.digest,
        "source_revision": evidence.head_revision,
        "source_digest": evidence.source_digest,
        "generator_revision": dependencies.generator_revision,
        "generator_schema": dependencies.generator_schema,
        "tck_revision": dependencies.tck_revision,
        "tck_schema": dependencies.tck_schema,
    }
    if error_code is not None:
        record["error_code"] = error_code
    return record


def _adapter_revision_guard(adapter: RolloutAdapter, dependencies: RolloutDependencies) -> None:
    dependencies.validate()
    if getattr(adapter, "generator_revision", None) != dependencies.generator_revision:
        raise RolloutError("generator-revision-mismatch")
    if getattr(adapter, "generator_schema", None) != dependencies.generator_schema:
        raise RolloutError("generator-schema-mismatch")
    if getattr(adapter, "tck_revision", None) != dependencies.tck_revision:
        raise RolloutError("tck-revision-mismatch")
    if getattr(adapter, "tck_schema", None) != dependencies.tck_schema:
        raise RolloutError("tck-schema-mismatch")
    if not isinstance(getattr(adapter, "generator_version", None), str) or not _VERSION.fullmatch(
        adapter.generator_version
    ):
        raise RolloutError("generator-version-invalid")


def preview_project(
    workspace_root: Path,
    manifest: FleetManifest,
    project: ProjectSpec,
    dependencies: RolloutDependencies,
    adapter: RolloutAdapter,
) -> dict[str, Any]:
    """Run one read-only generator plan and return bounded evidence."""

    _adapter_revision_guard(adapter, dependencies)
    if project not in manifest.projects:
        raise RolloutError("selection-not-in-manifest")
    evidence = collect_repository_evidence(workspace_root, project)
    enforce_repository_guard(evidence)
    try:
        result = _normalize_adapter_result(
            adapter,
            adapter.preview(_safe_repository_root(workspace_root, project.identity), project),
        )
    except RolloutError:
        raise
    except Exception as exc:  # noqa: BLE001 - adapter boundary is fail-closed
        raise RolloutError("generator-failed") from exc
    return {
        "ok": True,
        "mode": "preview",
        "identity": project.identity,
        "wave": project.wave,
        "replayed": False,
        "source": evidence.as_dict(),
        "generator_revision": dependencies.generator_revision,
        "generator_schema": dependencies.generator_schema,
        "tck_revision": dependencies.tck_revision,
        "tck_schema": dependencies.tck_schema,
        "generator": result,
    }


def apply_project(
    workspace_root: Path,
    manifest: FleetManifest,
    project: ProjectSpec,
    dependencies: RolloutDependencies,
    adapter: RolloutAdapter,
    journal: TransactionJournal,
    *,
    confirm: bool = False,
    transaction_id: str | None = None,
) -> dict[str, Any]:
    """Apply one project with durable intent-before-side-effect semantics."""

    if not confirm:
        return {"ok": False, "status": "blocked", "error_code": "confirmation-required"}
    _adapter_revision_guard(adapter, dependencies)
    if project not in manifest.projects:
        raise RolloutError("selection-not-in-manifest")
    evidence = collect_repository_evidence(workspace_root, project)
    enforce_repository_guard(evidence)
    key = _operation_key(manifest, project, evidence, dependencies)
    records = journal.records()
    matches = _matching_records(records, key)
    if matches:
        latest = matches[-1]
        if latest.get("state") == "prepared":
            raise RolloutError("rollout-recovery-required")
        if latest.get("state") == "rollback_failed":
            raise RolloutError("rollback-recovery-required")
        if latest.get("state") in {"applied", "rolled_back"}:
            return {
                "ok": True,
                "status": latest["state"],
                "identity": project.identity,
                "replayed": True,
                "transaction_id": latest["transaction_id"],
                "artifact_digest": latest.get("artifact_digest"),
                "generator_version": latest.get("generator_version"),
                "generated_outputs": latest.get("generated_outputs", []),
            }
    if _active_identity_conflict(records, project.identity, key):
        raise RolloutError("rollout-operation-conflict")
    transaction = transaction_id or key[:24]
    if not _TRANSACTION_ID.fullmatch(transaction):
        raise RolloutError("transaction-id-invalid")
    journal.append(
        _base_record(
            operation_key=key,
            transaction_id=transaction,
            mode="apply",
            state="prepared",
            manifest=manifest,
            evidence=evidence,
            dependencies=dependencies,
        )
    )
    repository_root = _safe_repository_root(workspace_root, project.identity)
    try:
        applied = _normalize_adapter_result(adapter, adapter.apply(repository_root, project))
        verified = _normalize_adapter_result(adapter, adapter.verify(repository_root, project))
        if applied["schema_version"] != verified["schema_version"]:
            raise RolloutError("generator-schema-mismatch")
        if applied["provenance_digest"] != verified["provenance_digest"]:
            raise RolloutError("generator-provenance-mismatch")
        final = _base_record(
            operation_key=key,
            transaction_id=transaction,
            mode="apply",
            state="applied",
            manifest=manifest,
            evidence=evidence,
            dependencies=dependencies,
        )
        final.update(
            {
                "generator_version": verified["generator_version"],
                "artifact_digest": verified["provenance_digest"],
                "generated_outputs": verified["generated_outputs"],
            }
        )
        journal.append(final)
        return {
            "ok": True,
            "status": "applied",
            "identity": project.identity,
            "replayed": False,
            "transaction_id": transaction,
            "artifact_digest": verified["provenance_digest"],
            "generated_outputs": verified["generated_outputs"],
        }
    except RolloutError as exc:
        error_code = str(exc) if _SAFE_ERROR.fullmatch(str(exc)) else "generator-failed"
    except Exception:  # noqa: BLE001 - adapter boundary is fail-closed
        error_code = "generator-failed"
    try:
        rollback = _normalize_adapter_result(
            adapter, adapter.rollback(repository_root, project, {"operation_key": key})
        )
    except Exception:  # noqa: BLE001 - preserve durable failure state
        journal.append(
            _base_record(
                operation_key=key,
                transaction_id=transaction,
                mode="apply",
                state="rollback_failed",
                manifest=manifest,
                evidence=evidence,
                dependencies=dependencies,
                error_code="rollback-failed",
            )
        )
        return {
            "ok": False,
            "status": "blocked",
            "identity": project.identity,
            "error_code": "rollback-failed",
        }
    rolled_back = _base_record(
        operation_key=key,
        transaction_id=transaction,
        mode="apply",
        state="rolled_back",
        manifest=manifest,
        evidence=evidence,
        dependencies=dependencies,
        error_code=error_code,
    )
    rolled_back.update(
        {
            "generator_version": rollback["generator_version"],
            "artifact_digest": rollback["provenance_digest"],
            "generated_outputs": rollback["generated_outputs"],
        }
    )
    journal.append(rolled_back)
    return {
        "ok": False,
        "status": "rolled_back",
        "identity": project.identity,
        "error_code": error_code,
    }


def rollback_project(
    workspace_root: Path,
    manifest: FleetManifest,
    project: ProjectSpec,
    dependencies: RolloutDependencies,
    adapter: RolloutAdapter,
    journal: TransactionJournal,
    *,
    confirm: bool = False,
) -> dict[str, Any]:
    """Rollback one committed transaction exactly once."""

    if not confirm:
        return {"ok": False, "status": "blocked", "error_code": "confirmation-required"}
    _adapter_revision_guard(adapter, dependencies)
    evidence = collect_repository_evidence(workspace_root, project)
    enforce_repository_guard(evidence)
    key = _operation_key(manifest, project, evidence, dependencies)
    records = journal.records()
    identity_candidates = [
        record
        for record in records
        if record.get("identity") == project.identity
        and record.get("state") in {"applied", "rolled_back"}
    ]
    candidates = [
        record
        for record in records
        if record.get("identity") == project.identity
        and record.get("operation_key") == key
        and record.get("state") in {"applied", "rolled_back"}
    ]
    if not candidates:
        if identity_candidates:
            raise RolloutError("rollback-source-changed")
        raise RolloutError("rollback-target-not-found")
    latest = candidates[-1]
    if latest.get("state") == "rolled_back":
        return {"ok": True, "status": "rolled_back", "identity": project.identity, "replayed": True}
    if latest.get("source_revision") != evidence.head_revision or latest.get("source_digest") != evidence.source_digest:
        raise RolloutError("rollback-source-changed")
    repository_root = _safe_repository_root(workspace_root, project.identity)
    try:
        result = _normalize_adapter_result(
            adapter, adapter.rollback(repository_root, project, latest)
        )
    except Exception as exc:  # noqa: BLE001 - persist the failed terminal attempt
        failed = dict(latest)
        failed.update({"mode": "rollback", "state": "rollback_failed", "error_code": "rollback-failed"})
        journal.append(failed)
        if isinstance(exc, RolloutError):
            raise
        raise RolloutError("rollback-failed") from exc
    rollback_record = dict(latest)
    rollback_record.update(
        {
            "mode": "rollback",
            "state": "rolled_back",
            "generator_version": result["generator_version"],
            "artifact_digest": result["provenance_digest"],
            "generated_outputs": result["generated_outputs"],
        }
    )
    journal.append(rollback_record)
    return {
        "ok": True,
        "status": "rolled_back",
        "identity": project.identity,
        "replayed": False,
        "artifact_digest": result["provenance_digest"],
    }


def _has_pages_workflow(repository_root: Path) -> bool:
    workflow_dir = repository_root / ".github" / "workflows"
    if not workflow_dir.is_dir() or workflow_dir.is_symlink():
        return False
    for workflow in sorted(workflow_dir.glob("*.y*ml")):
        payload = _read_bounded(workflow, maximum=MAX_SOURCE_FILE_BYTES, error="workflow-invalid")
        try:
            text = payload.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise RolloutError("workflow-invalid") from exc
        active_lines = [line for line in text.splitlines() if not line.lstrip().startswith("#")]
        if any(re.search(r"^\s+pages:\s*$", line) for line in active_lines):
            return True
        if any("pages_pipeline.yml@" in line and "uses:" in line for line in active_lines):
            return True
    return False


def _has_site_url(repository_root: Path) -> bool:
    path = repository_root / "mkdocs.yml"
    if not path.is_file() or path.is_symlink():
        return False
    payload = _read_bounded(path, maximum=MAX_SOURCE_FILE_BYTES, error="mkdocs-invalid")
    if yaml is None:
        raise RolloutError("yaml-unavailable") from _YAML_IMPORT_ERROR
    try:
        value = yaml.safe_load(payload.decode("utf-8"))
    except (UnicodeDecodeError, yaml.YAMLError) as exc:
        raise RolloutError("mkdocs-invalid") from exc
    return isinstance(value, Mapping) and isinstance(value.get("site_url"), str) and bool(value["site_url"].strip())


def audit_source_findings(
    workspace_root: Path,
    manifest: FleetManifest,
    *,
    require_expected: bool = True,
) -> SourceFindings:
    """Identify the two Pages and four ``site_url`` gaps without mutation."""

    missing_pages: list[str] = []
    missing_sites: list[str] = []
    for project in manifest.projects:
        repository_root = _safe_repository_root(workspace_root, project.identity)
        if not _has_pages_workflow(repository_root):
            missing_pages.append(project.identity)
        if not _has_site_url(repository_root):
            missing_sites.append(project.identity)
    findings = SourceFindings(tuple(missing_pages), tuple(missing_sites))
    if require_expected and findings != manifest.source_findings:
        raise RolloutError("source-finding-drift")
    return findings


__all__ = [
    "EXPECTED_PUBLISHABLE_COUNT",
    "FleetManifest",
    "MANIFEST_FILENAME",
    "MANIFEST_SOURCE_AUTHORITY",
    "MANIFEST_SURFACE_POLICY",
    "ProjectSpec",
    "RepositoryEvidence",
    "RolloutAdapter",
    "RolloutDependencies",
    "RolloutError",
    "SCHEMA_VERSION",
    "SourceFindings",
    "TransactionJournal",
    "WavePlan",
    "audit_source_findings",
    "apply_project",
    "collect_repository_evidence",
    "default_manifest_path",
    "enforce_repository_guard",
    "load_fleet_manifest",
    "normalize_generator_result",
    "plan_waves",
    "preview_project",
    "rollback_project",
]
