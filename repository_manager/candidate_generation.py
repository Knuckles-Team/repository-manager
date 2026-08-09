"""Immutable candidate snapshots and append-only generation records.

This module is the storage-neutral part of RMDD-12.  Git and the merge queue
provide the branch/base snapshots; this module records those snapshots in the
same append-only record authority used by the existing queue.  It deliberately
does not create a second durable store, execute commands, or move a ref.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from repository_manager.development import (
    Candidate,
    CandidateState,
    CandidateVersion,
    Generation,
    GenerationState,
    OpaqueId,
    RepositoryIdentity,
    TargetPolicy,
    canonical_digest,
    canonical_json,
)

_DIGEST_RE = re.compile(r"^[0-9a-f]{64}$")


class CandidateGenerationError(ValueError):
    """A candidate or generation record is invalid or was mutated."""


class CandidateLike(Protocol):
    """The legacy merge-queue candidate fields needed for a snapshot."""

    branch: str
    lane: str
    base: str
    enqueued_at: datetime | str


def _timestamp(value: datetime | str | None) -> str:
    """Normalize a timestamp to the contract's UTC representation."""

    if value is None or value == "":
        current = datetime.now(UTC)
    elif isinstance(value, datetime):
        current = value
    else:
        text = str(value)
        current = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if current.tzinfo is None:
        current = current.replace(tzinfo=UTC)
    return current.astimezone(UTC).isoformat().replace("+00:00", "Z")


def timestamp_value(value: datetime | str) -> datetime:
    """Return a timezone-aware timestamp for deterministic policy decisions."""

    text = value.isoformat() if isinstance(value, datetime) else str(value)
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def require_digest(value: str, *, field_name: str) -> str:
    """Accept only a canonical lowercase SHA-256 digest at the boundary."""

    if not isinstance(value, str) or _DIGEST_RE.fullmatch(value) is None:
        raise CandidateGenerationError(
            f"{field_name} must be a canonical 64-character lowercase hex digest"
        )
    return value


def _tuple_strings(values: Iterable[object]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value)}))


def candidate_identity(
    repository_id: str, branch: str, lane_id: str, *, explicit: str | None = None
) -> str:
    """Derive the stable identity of a branch submission across versions."""

    if explicit:
        return explicit
    digest = canonical_digest(
        {"repository_id": repository_id, "branch": branch, "lane_id": lane_id}
    )
    return f"candidate:{digest}"


@dataclass(frozen=True)
class CandidateSnapshot:
    """An immutable version of one candidate branch submission.

    ``candidate_id`` identifies the logical submission; ``version`` advances
    whenever the observed branch or base SHA changes.  The fields outside the
    frozen development contract are generation-selection inputs and are stored
    in the same append-only record, never inferred later from a moving branch.
    """

    candidate_id: OpaqueId
    version: int
    repository: RepositoryIdentity
    branch: str
    target_branch: str
    candidate_sha: str
    base_sha: str
    lane_id: OpaqueId
    owner_id: OpaqueId
    config_digest: str
    toolchain_digest: str
    resource_digest: str
    build_target: str
    concept_claims: tuple[str, ...]
    incompatibility_labels: tuple[str, ...]
    enqueued_at: datetime
    target: TargetPolicy = field(default_factory=TargetPolicy)
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if self.version < 1:
            raise CandidateGenerationError("candidate version must be positive")
        normalized_time = timestamp_value(self.enqueued_at)
        object.__setattr__(self, "enqueued_at", normalized_time)
        object.__setattr__(self, "recorded_at", _timestamp(self.recorded_at or None))
        object.__setattr__(self, "concept_claims", _tuple_strings(self.concept_claims))
        object.__setattr__(
            self,
            "incompatibility_labels",
            _tuple_strings(self.incompatibility_labels),
        )
        object.__setattr__(
            self,
            "config_digest",
            require_digest(self.config_digest, field_name="config_digest"),
        )
        object.__setattr__(
            self,
            "toolchain_digest",
            require_digest(self.toolchain_digest, field_name="toolchain_digest"),
        )
        object.__setattr__(
            self,
            "resource_digest",
            require_digest(self.resource_digest, field_name="resource_digest"),
        )
        Candidate(
            candidate_id=self.candidate_id,
            version=self.version,
            repository=self.repository,
            branch=self.branch,
            candidate_sha=self.candidate_sha,
            base_sha=self.base_sha,
            lane_id=self.lane_id,
            owner_id=self.owner_id,
            config_digest=self.config_digest,
            concept_claims=self.concept_claims,
            enqueued_at=self.enqueued_at,
            state=CandidateState.QUEUED,
        )

    @property
    def record_id(self) -> str:
        """The append-only key for this immutable candidate version."""

        return f"{self.candidate_id}:v{self.version}"

    @property
    def candidate_version(self) -> CandidateVersion:
        """Return this candidate's own immutable branch version."""

        return CandidateVersion(
            candidate_id=self.candidate_id,
            version=self.version,
            candidate_sha=self.candidate_sha,
        )

    @property
    def contract(self) -> Candidate:
        """Project to the frozen repository-development candidate contract."""

        return Candidate(
            candidate_id=self.candidate_id,
            version=self.version,
            repository=self.repository,
            branch=self.branch,
            candidate_sha=self.candidate_sha,
            base_sha=self.base_sha,
            lane_id=self.lane_id,
            owner_id=self.owner_id,
            config_digest=self.config_digest,
            concept_claims=self.concept_claims,
            enqueued_at=self.enqueued_at,
            state=CandidateState.QUEUED,
        )

    def immutable_payload(self) -> dict[str, object]:
        """Return every input that may affect generation formation."""

        return {
            "candidate_id": self.candidate_id,
            "version": self.version,
            "repository": self.repository,
            "branch": self.branch,
            "target_branch": self.target_branch,
            "candidate_sha": self.candidate_sha,
            "base_sha": self.base_sha,
            "lane_id": self.lane_id,
            "owner_id": self.owner_id,
            "config_digest": self.config_digest,
            "toolchain_digest": self.toolchain_digest,
            "resource_digest": self.resource_digest,
            "build_target": self.build_target,
            "concept_claims": self.concept_claims,
            "incompatibility_labels": self.incompatibility_labels,
            "enqueued_at": self.enqueued_at,
            "target": self.target,
        }

    def immutable_digest(self) -> str:
        """Digest the submitted snapshot, not the current branch."""

        return canonical_digest(self.immutable_payload())

    def to_record(self) -> dict[str, Any]:
        """Serialize this snapshot for an append-only fragment."""

        return {
            "record_id": self.record_id,
            "kind": "candidate_snapshot",
            "candidate": self.contract.model_dump(mode="json"),
            "target_branch": self.target_branch,
            "toolchain_digest": self.toolchain_digest,
            "resource_digest": self.resource_digest,
            "build_target": self.build_target,
            "incompatibility_labels": list(self.incompatibility_labels),
            "target": self.target.model_dump(mode="json"),
            "recorded_at": self.recorded_at,
            "immutable_digest": self.immutable_digest(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> CandidateSnapshot:
        if not isinstance(record, dict):
            raise CandidateGenerationError(
                "candidate snapshot record must be a mapping"
            )
        if record.get("kind") != "candidate_snapshot":
            raise CandidateGenerationError("record is not a candidate snapshot")
        required = {
            "record_id",
            "candidate",
            "target_branch",
            "toolchain_digest",
            "resource_digest",
            "build_target",
            "incompatibility_labels",
            "target",
            "recorded_at",
            "immutable_digest",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise CandidateGenerationError(
                f"candidate snapshot is missing required fields: {', '.join(missing)}"
            )
        if not isinstance(record["record_id"], str):
            raise CandidateGenerationError(
                "candidate snapshot record_id must be a string"
            )
        if not isinstance(record["candidate"], dict):
            raise CandidateGenerationError(
                "candidate snapshot candidate must be a mapping"
            )
        if not isinstance(record["target_branch"], str) or not record["target_branch"]:
            raise CandidateGenerationError(
                "candidate snapshot target_branch is invalid"
            )
        if not isinstance(record["build_target"], str) or not record["build_target"]:
            raise CandidateGenerationError("candidate snapshot build_target is invalid")
        labels = record["incompatibility_labels"]
        if not isinstance(labels, (list, tuple)) or not all(
            isinstance(value, str) for value in labels
        ):
            raise CandidateGenerationError(
                "candidate snapshot incompatibility_labels must be strings"
            )
        if not isinstance(record["recorded_at"], str):
            raise CandidateGenerationError(
                "candidate snapshot recorded_at must be a string"
            )
        if not isinstance(record["immutable_digest"], str):
            raise CandidateGenerationError(
                "candidate snapshot immutable_digest must be a string"
            )
        try:
            contract = Candidate.model_validate(record["candidate"])
            target_policy = TargetPolicy.model_validate(record["target"])
            snapshot = cls(
                candidate_id=contract.candidate_id,
                version=contract.version,
                repository=contract.repository,
                branch=contract.branch,
                target_branch=record["target_branch"],
                candidate_sha=contract.candidate_sha,
                base_sha=contract.base_sha,
                lane_id=contract.lane_id,
                owner_id=contract.owner_id,
                config_digest=contract.config_digest,
                toolchain_digest=record["toolchain_digest"],
                resource_digest=record["resource_digest"],
                build_target=record["build_target"],
                concept_claims=contract.concept_claims,
                incompatibility_labels=tuple(labels),
                enqueued_at=contract.enqueued_at,
                target=target_policy,
                recorded_at=record["recorded_at"],
            )
        except (TypeError, ValueError) as exc:
            raise CandidateGenerationError(
                f"candidate snapshot {record['record_id']} is invalid"
            ) from exc
        recorded_id = record["record_id"]
        if recorded_id != snapshot.record_id:
            raise CandidateGenerationError(
                f"candidate record key does not match {snapshot.record_id}"
            )
        recorded_digest = record["immutable_digest"]
        if recorded_digest != snapshot.immutable_digest():
            raise CandidateGenerationError(
                f"candidate snapshot {snapshot.record_id} has an invalid immutable digest"
            )
        return snapshot


@dataclass(frozen=True)
class GenerationRecord:
    """A generation plus its immutable member snapshots and latest result."""

    generation: Generation
    members: tuple[CandidateSnapshot, ...]
    result_json: str = ""
    recorded_at: str = ""

    def __post_init__(self) -> None:
        if not self.members:
            raise CandidateGenerationError("a generation record needs members")
        versions = tuple(member.candidate_version for member in self.members)
        if versions != self.generation.candidate_versions:
            raise CandidateGenerationError(
                "generation membership does not match its candidate versions"
            )
        first = self.members[0]
        for member in self.members:
            if (
                member.repository != self.generation.repository
                or member.target_branch != self.generation.target_branch
                or member.base_sha != self.generation.base_sha
                or member.config_digest != self.generation.config_digest
                or member.toolchain_digest != self.generation.toolchain_digest
                or member.target != self.generation.target
                or member.resource_digest != first.resource_digest
                or member.build_target != first.build_target
                or member.concept_claims != first.concept_claims
                or member.incompatibility_labels != first.incompatibility_labels
            ):
                raise CandidateGenerationError(
                    "generation members do not match immutable generation inputs"
                )
        expected_id = Generation.derive_id(
            repository_id=self.generation.repository.repository_id,
            target_branch=self.generation.target_branch,
            base_sha=self.generation.base_sha,
            candidate_versions=versions,
            config_digest=self.generation.config_digest,
            toolchain_digest=self.generation.toolchain_digest,
        )
        if expected_id != self.generation.generation_id:
            raise CandidateGenerationError(
                "generation ID does not match its immutable membership"
            )
        if self.recorded_at:
            object.__setattr__(self, "recorded_at", _timestamp(self.recorded_at))
        else:
            object.__setattr__(self, "recorded_at", _timestamp(None))
        if self.result_json:
            try:
                parsed = json.loads(self.result_json)
            except json.JSONDecodeError as exc:
                raise CandidateGenerationError("generation result is not JSON") from exc
            if not isinstance(parsed, dict):
                raise CandidateGenerationError("generation result must be an object")

    @property
    def record_id(self) -> str:
        return self.generation.generation_id

    @property
    def result(self) -> dict[str, Any]:
        if not self.result_json:
            return {}
        value = json.loads(self.result_json)
        return dict(value)

    def immutable_payload(self) -> dict[str, object]:
        """Return generation identity inputs excluding mutable results/state."""

        return {
            "generation_id": self.generation.generation_id,
            "repository": self.generation.repository,
            "target_branch": self.generation.target_branch,
            "target": self.generation.target,
            "base_sha": self.generation.base_sha,
            "expected_landing_base_sha": self.generation.expected_landing_base_sha,
            "candidate_versions": self.generation.candidate_versions,
            "config_digest": self.generation.config_digest,
            "toolchain_digest": self.generation.toolchain_digest,
            "sealed_at": self.generation.sealed_at,
            "members": tuple(member.immutable_digest() for member in self.members),
        }

    def immutable_digest(self) -> str:
        """Digest the sealed membership and exact generation inputs."""

        return canonical_digest(self.immutable_payload())

    def with_update(
        self,
        generation: Generation,
        *,
        result: dict[str, Any] | None = None,
        recorded_at: datetime | str | None = None,
    ) -> GenerationRecord:
        """Append a state/result update while preserving immutable membership."""

        _assert_generation_identity(self.generation, generation)
        return GenerationRecord(
            generation=generation,
            members=self.members,
            result_json=canonical_json(result)
            if result is not None
            else self.result_json,
            recorded_at=_timestamp(recorded_at),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "kind": "generation",
            "generation": self.generation.model_dump(mode="json"),
            "members": [member.to_record() for member in self.members],
            "result": self.result,
            "recorded_at": self.recorded_at,
            "immutable_digest": self.immutable_digest(),
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> GenerationRecord:
        if not isinstance(record, dict):
            raise CandidateGenerationError("generation record must be a mapping")
        if record.get("kind") != "generation":
            raise CandidateGenerationError("record is not a generation")
        required = {
            "record_id",
            "generation",
            "members",
            "recorded_at",
            "immutable_digest",
        }
        missing = sorted(required - record.keys())
        if missing:
            raise CandidateGenerationError(
                f"generation record is missing required fields: {', '.join(missing)}"
            )
        if not isinstance(record["record_id"], str):
            raise CandidateGenerationError("generation record_id must be a string")
        if not isinstance(record["generation"], dict):
            raise CandidateGenerationError("generation payload must be a mapping")
        member_payload = record["members"]
        if not isinstance(member_payload, (list, tuple)) or not all(
            isinstance(item, dict) for item in member_payload
        ):
            raise CandidateGenerationError("generation members must be mappings")
        if not isinstance(record["recorded_at"], str):
            raise CandidateGenerationError("generation recorded_at must be a string")
        if not isinstance(record["immutable_digest"], str):
            raise CandidateGenerationError(
                "generation immutable_digest must be a string"
            )
        result = record.get("result", {})
        if not isinstance(result, dict):
            raise CandidateGenerationError("generation result must be a mapping")
        try:
            generation = Generation.model_validate(record["generation"])
            members = tuple(
                CandidateSnapshot.from_record(item) for item in member_payload
            )
            parsed = cls(
                generation=generation,
                members=members,
                result_json=canonical_json(result),
                recorded_at=record["recorded_at"],
            )
        except (TypeError, ValueError) as exc:
            raise CandidateGenerationError(
                f"generation record {record['record_id']} is invalid"
            ) from exc
        if record["record_id"] != parsed.record_id:
            raise CandidateGenerationError(
                f"generation record key does not match {parsed.record_id}"
            )
        if record["immutable_digest"] != parsed.immutable_digest():
            raise CandidateGenerationError(
                f"generation {parsed.record_id} has an invalid immutable digest"
            )
        return parsed


def _assert_generation_identity(old: Generation, new: Generation) -> None:
    """Reject any update that changes the sealed generation inputs."""

    immutable_fields = (
        "generation_id",
        "repository",
        "target_branch",
        "target",
        "base_sha",
        "expected_landing_base_sha",
        "candidate_versions",
        "config_digest",
        "toolchain_digest",
        "sealed_at",
    )
    for field_name in immutable_fields:
        if getattr(old, field_name) != getattr(new, field_name):
            raise CandidateGenerationError(
                f"generation {old.generation_id} immutable field changed: {field_name}"
            )


def _latest_record(group: list[dict[str, Any]]) -> dict[str, Any]:
    """Fold records by write time, retaining deterministic tie-breaking."""

    if not group:
        raise CandidateGenerationError("cannot fold an empty record group")
    for record in group:
        if not isinstance(record.get("record_id"), str) or not record["record_id"]:
            raise CandidateGenerationError("append-only record requires record_id")
        if not isinstance(record.get("recorded_at"), str):
            raise CandidateGenerationError(
                f"record {record['record_id']} requires recorded_at for folding"
            )
    return max(
        group,
        key=lambda record: (
            timestamp_value(str(record["recorded_at"])),
            canonical_json(record),
        ),
    )


class AppendOnlyRecordStore(Protocol):
    """Storage seam owned by the queue; RMDD-12 only consumes its records."""

    def append(self, record: dict[str, Any], *, lane: str) -> object: ...

    def fold(
        self, resolve: Callable[[list[dict[str, Any]]], dict[str, Any]]
    ) -> list[dict[str, Any]]: ...


def _group_records(
    records: Iterable[dict[str, Any]], *, kind: str
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        if not isinstance(record, dict):
            raise CandidateGenerationError("append-only records must be mappings")
        if record.get("kind") != kind:
            continue
        record_id = record.get("record_id")
        if not isinstance(record_id, str) or not record_id:
            raise CandidateGenerationError(f"{kind} record requires record_id")
        groups.setdefault(record_id, []).append(record)
    return groups


def fold_candidate_records(
    records: Iterable[dict[str, Any]],
) -> tuple[CandidateSnapshot, ...]:
    """Fold candidate records supplied by the existing queue store."""

    snapshots: list[CandidateSnapshot] = []
    for record_id, group in sorted(
        _group_records(records, kind="candidate_snapshot").items()
    ):
        decoded = [CandidateSnapshot.from_record(item) for item in group]
        first = decoded[0]
        if any(
            item.immutable_digest() != first.immutable_digest() for item in decoded[1:]
        ):
            raise CandidateGenerationError(
                f"candidate {record_id} has conflicting append-only inputs"
            )
        snapshots.append(CandidateSnapshot.from_record(_latest_record(group)))
    return tuple(snapshots)


def fold_generation_records(
    records: Iterable[dict[str, Any]],
) -> tuple[GenerationRecord, ...]:
    """Fold generation records without selecting a persistence backend."""

    generations: list[GenerationRecord] = []
    for record_id, group in sorted(_group_records(records, kind="generation").items()):
        decoded = [GenerationRecord.from_record(item) for item in group]
        first = decoded[0]
        for record in decoded[1:]:
            _assert_generation_identity(first.generation, record.generation)
            if tuple(member.immutable_digest() for member in record.members) != tuple(
                member.immutable_digest() for member in first.members
            ):
                raise CandidateGenerationError(
                    f"generation {record_id} has conflicting append-only members"
                )
        generations.append(GenerationRecord.from_record(_latest_record(group)))
    return tuple(generations)


def snapshot_candidate(
    candidate: CandidateLike,
    *,
    repository: RepositoryIdentity,
    candidate_sha: str,
    base_sha: str,
    config_digest: str,
    toolchain_digest: str,
    resource_digest: str,
    build_target: str = "default",
    target_branch: str | None = None,
    owner_id: str | None = None,
    concept_claims: Iterable[object] = (),
    incompatibility_labels: Iterable[object] = (),
    target: TargetPolicy | None = None,
    candidate_id: str | None = None,
    version: int = 1,
) -> CandidateSnapshot:
    """Capture one legacy queue candidate without reading its branch later."""

    branch = str(candidate.branch)
    lane_id = str(candidate.lane)
    logical_id = candidate_identity(
        repository.repository_id, branch, lane_id, explicit=candidate_id
    )
    return CandidateSnapshot(
        candidate_id=logical_id,
        version=version,
        repository=repository,
        branch=branch,
        target_branch=target_branch or str(candidate.base),
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        lane_id=lane_id,
        owner_id=owner_id or lane_id,
        config_digest=config_digest,
        toolchain_digest=toolchain_digest,
        resource_digest=resource_digest,
        build_target=build_target,
        concept_claims=tuple(str(value) for value in concept_claims),
        incompatibility_labels=tuple(str(value) for value in incompatibility_labels),
        enqueued_at=timestamp_value(candidate.enqueued_at),
        target=target or TargetPolicy(),
    )


def snapshot_branch_candidate(
    candidate: CandidateLike,
    *,
    repository: RepositoryIdentity,
    resolve_ref: Callable[[str], str],
    config_digest: str,
    toolchain_digest: str,
    resource_digest: str,
    build_target: str = "default",
    target_branch: str | None = None,
    owner_id: str | None = None,
    concept_claims: Iterable[object] = (),
    incompatibility_labels: Iterable[object] = (),
    target: TargetPolicy | None = None,
    previous: CandidateSnapshot | None = None,
) -> CandidateSnapshot:
    """Resolve branch/base exactly once and append a new version if moved."""

    logical_id = candidate_identity(
        repository.repository_id, candidate.branch, candidate.lane
    )
    candidate_sha = resolve_ref(candidate.branch)
    base_sha = resolve_ref(target_branch or candidate.base)
    target_value = target or TargetPolicy()
    concepts = _tuple_strings(concept_claims)
    labels = _tuple_strings(incompatibility_labels)
    version = 1
    if previous is not None:
        if previous.candidate_id != logical_id:
            raise CandidateGenerationError(
                "previous candidate snapshot belongs to a different candidate"
            )
        version = previous.version
        immutable_inputs_changed = (
            previous.candidate_sha != candidate_sha
            or previous.base_sha != base_sha
            or previous.config_digest
            != require_digest(config_digest, field_name="config_digest")
            or previous.toolchain_digest
            != require_digest(toolchain_digest, field_name="toolchain_digest")
            or previous.resource_digest
            != require_digest(resource_digest, field_name="resource_digest")
            or previous.build_target != build_target
            or previous.target_branch != (target_branch or candidate.base)
            or previous.concept_claims != concepts
            or previous.incompatibility_labels != labels
            or previous.target != target_value
        )
        if immutable_inputs_changed:
            version += 1
    snapshot = snapshot_candidate(
        candidate,
        repository=repository,
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        config_digest=config_digest,
        toolchain_digest=toolchain_digest,
        resource_digest=resource_digest,
        build_target=build_target,
        target_branch=target_branch,
        owner_id=owner_id,
        concept_claims=concepts,
        incompatibility_labels=labels,
        target=target_value,
        candidate_id=logical_id,
        version=version,
    )
    return snapshot


def generation_record(
    members: tuple[CandidateSnapshot, ...] | list[CandidateSnapshot],
    *,
    target_branch: str,
    target: TargetPolicy | None = None,
    sealed_at: datetime | str,
    state: GenerationState = GenerationState.SEALED,
    synthetic_commit_sha: str | None = None,
    tree_sha: str | None = None,
    validation_evidence_ids: Iterable[str] = (),
    bisection_lineage: Iterable[str] = (),
    reason: str = "",
    result: dict[str, Any] | None = None,
) -> GenerationRecord:
    """Construct a generation from already-snapshotted immutable members."""

    ordered = tuple(members)
    if not ordered:
        raise CandidateGenerationError("cannot form an empty generation")
    first = ordered[0]
    if any(
        member.repository != first.repository
        or member.target_branch != target_branch
        or member.base_sha != first.base_sha
        or member.config_digest != first.config_digest
        or member.toolchain_digest != first.toolchain_digest
        or member.resource_digest != first.resource_digest
        or member.build_target != first.build_target
        or member.concept_claims != first.concept_claims
        or member.incompatibility_labels != first.incompatibility_labels
        or member.target != first.target
        for member in ordered
    ):
        raise CandidateGenerationError("generation members are not compatible")
    if len({member.version for member in ordered}) != len(ordered):
        raise CandidateGenerationError(
            "generation members must preserve distinct candidate versions"
        )
    generation_target = target or first.target
    if generation_target != first.target:
        raise CandidateGenerationError(
            "generation target must match each candidate snapshot target"
        )
    candidate_versions = tuple(member.candidate_version for member in ordered)
    generation_id = Generation.derive_id(
        repository_id=first.repository.repository_id,
        target_branch=target_branch,
        base_sha=first.base_sha,
        candidate_versions=candidate_versions,
        config_digest=first.config_digest,
        toolchain_digest=first.toolchain_digest,
    )
    generation = Generation(
        generation_id=generation_id,
        repository=first.repository,
        target_branch=target_branch,
        target=generation_target,
        base_sha=first.base_sha,
        expected_landing_base_sha=first.base_sha,
        candidate_versions=candidate_versions,
        config_digest=first.config_digest,
        toolchain_digest=first.toolchain_digest,
        state=state,
        sealed_at=timestamp_value(sealed_at),
        synthetic_commit_sha=synthetic_commit_sha,
        tree_sha=tree_sha,
        validation_evidence_ids=tuple(validation_evidence_ids),
        bisection_lineage=tuple(bisection_lineage),
        reason=reason,
    )
    return GenerationRecord(
        generation=generation,
        members=ordered,
        result_json=canonical_json(result or {}),
        recorded_at=_timestamp(sealed_at),
    )


__all__ = [
    "AppendOnlyRecordStore",
    "CandidateGenerationError",
    "CandidateSnapshot",
    "GenerationRecord",
    "candidate_identity",
    "fold_candidate_records",
    "fold_generation_records",
    "generation_record",
    "require_digest",
    "snapshot_branch_candidate",
    "snapshot_candidate",
    "timestamp_value",
]
