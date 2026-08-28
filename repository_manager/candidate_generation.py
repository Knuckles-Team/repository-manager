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
    is_legal_transition,
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
        _require_candidate_snapshot_fields(record)
        _validate_candidate_snapshot_scalars(record)
        labels = _validated_incompatibility_labels(record)
        snapshot = _build_candidate_snapshot(record, labels)
        _assert_candidate_snapshot_matches_record(snapshot, record)
        return snapshot


_CANDIDATE_SNAPSHOT_RECORD_FIELDS = {
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


def _require_candidate_snapshot_fields(record: dict[str, Any]) -> None:
    """Reject a candidate snapshot record that is missing a required field."""

    missing = sorted(_CANDIDATE_SNAPSHOT_RECORD_FIELDS - record.keys())
    if missing:
        raise CandidateGenerationError(
            f"candidate snapshot is missing required fields: {', '.join(missing)}"
        )


def _validate_candidate_snapshot_scalars(record: dict[str, Any]) -> None:
    """Reject a candidate snapshot record whose scalar fields are malformed."""

    if not isinstance(record["record_id"], str):
        raise CandidateGenerationError("candidate snapshot record_id must be a string")
    if not isinstance(record["candidate"], dict):
        raise CandidateGenerationError("candidate snapshot candidate must be a mapping")
    if not isinstance(record["target_branch"], str) or not record["target_branch"]:
        raise CandidateGenerationError("candidate snapshot target_branch is invalid")
    if not isinstance(record["build_target"], str) or not record["build_target"]:
        raise CandidateGenerationError("candidate snapshot build_target is invalid")
    if not isinstance(record["recorded_at"], str):
        raise CandidateGenerationError(
            "candidate snapshot recorded_at must be a string"
        )
    if not isinstance(record["immutable_digest"], str):
        raise CandidateGenerationError(
            "candidate snapshot immutable_digest must be a string"
        )


def _validated_incompatibility_labels(
    record: dict[str, Any],
) -> list[Any] | tuple[Any, ...]:
    """Return the incompatibility labels, rejecting a non-string-list value."""

    labels = record["incompatibility_labels"]
    if not isinstance(labels, (list, tuple)) or not all(
        isinstance(value, str) for value in labels
    ):
        raise CandidateGenerationError(
            "candidate snapshot incompatibility_labels must be strings"
        )
    return labels


def _build_candidate_snapshot(
    record: dict[str, Any], labels: Iterable[str]
) -> CandidateSnapshot:
    """Construct the snapshot from an already shape-validated record."""

    try:
        contract = Candidate.model_validate(record["candidate"])
        target_policy = TargetPolicy.model_validate(record["target"])
        return CandidateSnapshot(
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


def _assert_candidate_snapshot_matches_record(
    snapshot: CandidateSnapshot, record: dict[str, Any]
) -> None:
    """Reject a snapshot whose derived key/digest disagrees with its own record."""

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


@dataclass(frozen=True)
class GenerationRecord:
    """A generation plus its immutable member snapshots and latest result."""

    generation: Generation
    members: tuple[CandidateSnapshot, ...]
    result_json: str = ""
    recorded_at: str = ""

    def __post_init__(self) -> None:
        versions = _assert_generation_membership(self.generation, self.members)
        _assert_members_compatible(self.generation, self.members)
        _assert_generation_id_matches(self.generation, versions)
        object.__setattr__(
            self, "recorded_at", _normalized_recorded_at(self.recorded_at)
        )
        _validated_result_json(self.result_json)

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
        _assert_generation_transition(self.generation, generation)
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
        _require_generation_record_fields(record)
        _validate_generation_record_scalars(record)
        member_payload = _validated_generation_member_payload(record)
        result = _validated_generation_result(record)
        parsed = _build_generation_record(record, member_payload, result)
        _assert_generation_record_matches(parsed, record)
        return parsed


_GENERATION_RECORD_FIELDS = {
    "record_id",
    "generation",
    "members",
    "recorded_at",
    "immutable_digest",
}


def _require_generation_record_fields(record: dict[str, Any]) -> None:
    """Reject a generation record that is missing a required field."""

    missing = sorted(_GENERATION_RECORD_FIELDS - record.keys())
    if missing:
        raise CandidateGenerationError(
            f"generation record is missing required fields: {', '.join(missing)}"
        )


def _validate_generation_record_scalars(record: dict[str, Any]) -> None:
    """Reject a generation record whose scalar fields are malformed."""

    if not isinstance(record["record_id"], str):
        raise CandidateGenerationError("generation record_id must be a string")
    if not isinstance(record["generation"], dict):
        raise CandidateGenerationError("generation payload must be a mapping")
    if not isinstance(record["recorded_at"], str):
        raise CandidateGenerationError("generation recorded_at must be a string")
    if not isinstance(record["immutable_digest"], str):
        raise CandidateGenerationError("generation immutable_digest must be a string")


def _validated_generation_member_payload(
    record: dict[str, Any],
) -> list[Any] | tuple[Any, ...]:
    """Return the raw member payload, rejecting a non-mapping-list value."""

    member_payload = record["members"]
    if not isinstance(member_payload, (list, tuple)) or not all(
        isinstance(item, dict) for item in member_payload
    ):
        raise CandidateGenerationError("generation members must be mappings")
    return member_payload


def _validated_generation_result(record: dict[str, Any]) -> dict[str, Any]:
    """Return the result payload, rejecting a non-mapping value."""

    result = record.get("result", {})
    if not isinstance(result, dict):
        raise CandidateGenerationError("generation result must be a mapping")
    return result


def _build_generation_record(
    record: dict[str, Any],
    member_payload: Iterable[dict[str, Any]],
    result: dict[str, Any],
) -> GenerationRecord:
    """Construct the generation record from an already shape-validated record."""

    try:
        generation = Generation.model_validate(record["generation"])
        members = tuple(CandidateSnapshot.from_record(item) for item in member_payload)
        return GenerationRecord(
            generation=generation,
            members=members,
            result_json=canonical_json(result),
            recorded_at=record["recorded_at"],
        )
    except (TypeError, ValueError) as exc:
        raise CandidateGenerationError(
            f"generation record {record['record_id']} is invalid"
        ) from exc


def _assert_generation_record_matches(
    parsed: GenerationRecord, record: dict[str, Any]
) -> None:
    """Reject a parsed generation record whose key/digest disagrees with its record."""

    if record["record_id"] != parsed.record_id:
        raise CandidateGenerationError(
            f"generation record key does not match {parsed.record_id}"
        )
    if record["immutable_digest"] != parsed.immutable_digest():
        raise CandidateGenerationError(
            f"generation {parsed.record_id} has an invalid immutable digest"
        )


def _assert_generation_membership(
    generation: Generation, members: tuple[CandidateSnapshot, ...]
) -> tuple[CandidateVersion, ...]:
    """Reject membership that doesn't match the generation's candidate versions."""

    if not members:
        raise CandidateGenerationError("a generation record needs members")
    versions = tuple(member.candidate_version for member in members)
    if versions != generation.candidate_versions:
        raise CandidateGenerationError(
            "generation membership does not match its candidate versions"
        )
    return versions


def _member_matches_generation(
    member: CandidateSnapshot, generation: Generation, first: CandidateSnapshot
) -> bool:
    """True when one member's immutable inputs agree with the generation/first member."""

    return (
        member.repository == generation.repository
        and member.target_branch == generation.target_branch
        and member.base_sha == generation.base_sha
        and member.config_digest == generation.config_digest
        and member.toolchain_digest == generation.toolchain_digest
        and member.target == generation.target
        and member.resource_digest == first.resource_digest
        and member.build_target == first.build_target
        and member.concept_claims == first.concept_claims
        and member.incompatibility_labels == first.incompatibility_labels
    )


def _assert_members_compatible(
    generation: Generation, members: tuple[CandidateSnapshot, ...]
) -> None:
    """Reject a member whose immutable inputs diverge from the generation."""

    first = members[0]
    for member in members:
        if not _member_matches_generation(member, generation, first):
            raise CandidateGenerationError(
                "generation members do not match immutable generation inputs"
            )


def _assert_generation_id_matches(
    generation: Generation, versions: tuple[CandidateVersion, ...]
) -> None:
    """Reject a generation whose ID doesn't derive from its own immutable inputs."""

    expected_id = Generation.derive_id(
        repository_id=generation.repository.repository_id,
        target_branch=generation.target_branch,
        base_sha=generation.base_sha,
        candidate_versions=versions,
        config_digest=generation.config_digest,
        toolchain_digest=generation.toolchain_digest,
    )
    if expected_id != generation.generation_id:
        raise CandidateGenerationError(
            "generation ID does not match its immutable membership"
        )


def _normalized_recorded_at(recorded_at: str) -> str:
    """Normalize an optional recorded_at input to the contract's timestamp form."""

    return _timestamp(recorded_at) if recorded_at else _timestamp(None)


def _validated_result_json(result_json: str) -> None:
    """Reject a result payload that isn't empty and isn't a JSON object."""

    if not result_json:
        return
    try:
        parsed = json.loads(result_json)
    except json.JSONDecodeError as exc:
        raise CandidateGenerationError("generation result is not JSON") from exc
    if not isinstance(parsed, dict):
        raise CandidateGenerationError("generation result must be an object")


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


def _assert_generation_transition(old: Generation, new: Generation) -> None:
    """Reject a state regression while allowing same-state result updates."""

    if old.state == new.state:
        return
    if not is_legal_transition(old.state, new.state):
        raise CandidateGenerationError(
            f"illegal generation transition: {old.state} -> {new.state}"
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


def _generation_record_comparison_json(record: dict[str, Any]) -> str:
    """Canonicalize replay-only member observation metadata for tie checks.

    A generation embeds its candidate records for lineage.  A replay can
    reconstruct the same immutable member with a different ``recorded_at``;
    that observation timestamp is not part of the member immutable digest and
    must not turn an otherwise identical same-timestamp generation update into
    a divergent history.  All immutable member fields remain in the comparison.
    """

    normalized = dict(record)
    members = normalized.get("members")
    if isinstance(members, (list, tuple)) and all(
        isinstance(member, dict) for member in members
    ):
        normalized["members"] = [
            {key: value for key, value in member.items() if key != "recorded_at"}
            for member in members
        ]
    return canonical_json(normalized)


def _assert_consistent_generation_group(
    record_id: str, decoded: list[tuple[GenerationRecord, dict[str, Any]]]
) -> None:
    """Reject a group of generation records with conflicting immutable inputs."""

    first = decoded[0][0]
    for record, _ in decoded[1:]:
        _assert_generation_identity(first.generation, record.generation)
        if tuple(member.immutable_digest() for member in record.members) != tuple(
            member.immutable_digest() for member in first.members
        ):
            raise CandidateGenerationError(
                f"generation {record_id} has conflicting append-only members"
            )


def _assert_no_divergent_timestamps(
    record_id: str, decoded: list[tuple[GenerationRecord, dict[str, Any]]]
) -> None:
    """Reject two differing records recorded at the exact same timestamp."""

    timestamps: dict[datetime, set[str]] = {}
    for record, raw in decoded:
        timestamp = timestamp_value(record.recorded_at)
        timestamps.setdefault(timestamp, set()).add(
            _generation_record_comparison_json(raw)
        )
    if any(len(variants) > 1 for variants in timestamps.values()):
        raise CandidateGenerationError(
            f"generation {record_id} has divergent records at one timestamp"
        )


def _replay_generation_transitions(
    ordered: list[tuple[GenerationRecord, dict[str, Any]]],
) -> GenerationRecord:
    """Walk time-ordered updates, rejecting an illegal state transition."""

    previous = ordered[0][0]
    for current, _ in ordered[1:]:
        _assert_generation_transition(previous.generation, current.generation)
        previous = current
    return ordered[-1][0]


def _fold_generation_group(
    record_id: str, group: list[dict[str, Any]]
) -> GenerationRecord:
    """Fold one generation's append-only record group to its latest state."""

    decoded = [(GenerationRecord.from_record(item), item) for item in group]
    _assert_consistent_generation_group(record_id, decoded)
    _assert_no_divergent_timestamps(record_id, decoded)
    ordered = sorted(
        decoded,
        key=lambda pair: (
            timestamp_value(pair[0].recorded_at),
            canonical_json(pair[1]),
        ),
    )
    return _replay_generation_transitions(ordered)


def fold_generation_records(
    records: Iterable[dict[str, Any]],
) -> tuple[GenerationRecord, ...]:
    """Fold generation records without selecting a persistence backend."""

    groups = sorted(_group_records(records, kind="generation").items())
    return tuple(
        _fold_generation_group(record_id, group) for record_id, group in groups
    )


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


@dataclass(frozen=True)
class _ResolvedCandidateInputs:
    """Immutable candidate inputs resolved once, for a version-bump comparison."""

    candidate_sha: str
    base_sha: str
    config_digest: str
    toolchain_digest: str
    resource_digest: str
    build_target: str
    target_branch: str | None
    concept_claims: tuple[str, ...]
    incompatibility_labels: tuple[str, ...]
    target: TargetPolicy


def _resolve_candidate_refs(
    candidate: CandidateLike,
    *,
    resolve_ref: Callable[[str], str],
    target_branch: str | None,
) -> tuple[str, str]:
    """Resolve the branch and base refs to shas exactly once."""

    candidate_sha = resolve_ref(candidate.branch)
    base_sha = resolve_ref(target_branch or candidate.base)
    return candidate_sha, base_sha


def _candidate_snapshot_refs_changed(
    previous: CandidateSnapshot, resolved: _ResolvedCandidateInputs
) -> bool:
    """True when a candidate's resolved shas or content digests moved."""

    return (
        previous.candidate_sha != resolved.candidate_sha
        or previous.base_sha != resolved.base_sha
        or previous.config_digest
        != require_digest(resolved.config_digest, field_name="config_digest")
        or previous.toolchain_digest
        != require_digest(resolved.toolchain_digest, field_name="toolchain_digest")
        or previous.resource_digest
        != require_digest(resolved.resource_digest, field_name="resource_digest")
    )


def _candidate_snapshot_metadata_changed(
    previous: CandidateSnapshot,
    candidate: CandidateLike,
    resolved: _ResolvedCandidateInputs,
) -> bool:
    """True when a candidate's build/target selection metadata moved."""

    return (
        previous.build_target != resolved.build_target
        or previous.target_branch != (resolved.target_branch or candidate.base)
        or previous.concept_claims != resolved.concept_claims
        or previous.incompatibility_labels != resolved.incompatibility_labels
        or previous.target != resolved.target
    )


def _candidate_snapshot_changed(
    previous: CandidateSnapshot,
    candidate: CandidateLike,
    resolved: _ResolvedCandidateInputs,
) -> bool:
    """True when any immutable candidate input differs from the prior version."""

    return _candidate_snapshot_refs_changed(
        previous, resolved
    ) or _candidate_snapshot_metadata_changed(previous, candidate, resolved)


def _next_candidate_version(
    previous: CandidateSnapshot | None,
    logical_id: str,
    candidate: CandidateLike,
    resolved: _ResolvedCandidateInputs,
) -> int:
    """Return 1 for a new candidate id, else bump on any changed immutable input."""

    if previous is None:
        return 1
    if previous.candidate_id != logical_id:
        raise CandidateGenerationError(
            "previous candidate snapshot belongs to a different candidate"
        )
    if _candidate_snapshot_changed(previous, candidate, resolved):
        return previous.version + 1
    return previous.version


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
    candidate_sha, base_sha = _resolve_candidate_refs(
        candidate, resolve_ref=resolve_ref, target_branch=target_branch
    )
    resolved = _ResolvedCandidateInputs(
        candidate_sha=candidate_sha,
        base_sha=base_sha,
        config_digest=config_digest,
        toolchain_digest=toolchain_digest,
        resource_digest=resource_digest,
        build_target=build_target,
        target_branch=target_branch,
        concept_claims=_tuple_strings(concept_claims),
        incompatibility_labels=_tuple_strings(incompatibility_labels),
        target=target or TargetPolicy(),
    )
    version = _next_candidate_version(previous, logical_id, candidate, resolved)
    return snapshot_candidate(
        candidate,
        repository=repository,
        candidate_sha=resolved.candidate_sha,
        base_sha=resolved.base_sha,
        config_digest=config_digest,
        toolchain_digest=toolchain_digest,
        resource_digest=resource_digest,
        build_target=build_target,
        target_branch=target_branch,
        owner_id=owner_id,
        concept_claims=resolved.concept_claims,
        incompatibility_labels=resolved.incompatibility_labels,
        target=resolved.target,
        candidate_id=logical_id,
        version=version,
    )


def _member_repo_or_base_mismatch(
    member: CandidateSnapshot, first: CandidateSnapshot, target_branch: str
) -> bool:
    """True when a member's repository/target/base/digest identity diverges."""

    return (
        member.repository != first.repository
        or member.target_branch != target_branch
        or member.base_sha != first.base_sha
        or member.config_digest != first.config_digest
        or member.toolchain_digest != first.toolchain_digest
    )


def _member_build_or_labels_mismatch(
    member: CandidateSnapshot, first: CandidateSnapshot
) -> bool:
    """True when a member's build/concept/label/target selection diverges."""

    return (
        member.resource_digest != first.resource_digest
        or member.build_target != first.build_target
        or member.concept_claims != first.concept_claims
        or member.incompatibility_labels != first.incompatibility_labels
        or member.target != first.target
    )


def _member_mismatches_generation(
    member: CandidateSnapshot, first: CandidateSnapshot, target_branch: str
) -> bool:
    """True when a member's immutable inputs diverge from the first member."""

    return _member_repo_or_base_mismatch(
        member, first, target_branch
    ) or _member_build_or_labels_mismatch(member, first)


def _assert_members_form_one_generation(
    ordered: tuple[CandidateSnapshot, ...], target_branch: str
) -> None:
    """Reject members whose immutable inputs don't agree with each other."""

    first = ordered[0]
    if any(
        _member_mismatches_generation(member, first, target_branch)
        for member in ordered
    ):
        raise CandidateGenerationError("generation members are not compatible")


def _resolved_generation_target(
    target: TargetPolicy | None, first: CandidateSnapshot
) -> TargetPolicy:
    """Return the generation target, rejecting one that disagrees with members."""

    generation_target = target or first.target
    if generation_target != first.target:
        raise CandidateGenerationError(
            "generation target must match each candidate snapshot target"
        )
    return generation_target


@dataclass(frozen=True)
class _GenerationLineage:
    """Provenance fields carried through to the sealed ``Generation``."""

    synthetic_commit_sha: str | None = None
    tree_sha: str | None = None
    validation_evidence_ids: tuple[str, ...] = ()
    bisection_lineage: tuple[str, ...] = ()
    reason: str = ""


def _build_generation(
    first: CandidateSnapshot,
    *,
    target_branch: str,
    generation_target: TargetPolicy,
    candidate_versions: tuple[CandidateVersion, ...],
    sealed_at: datetime | str,
    state: GenerationState,
    lineage: _GenerationLineage,
) -> Generation:
    """Derive the generation id and construct the sealed Generation contract."""

    generation_id = Generation.derive_id(
        repository_id=first.repository.repository_id,
        target_branch=target_branch,
        base_sha=first.base_sha,
        candidate_versions=candidate_versions,
        config_digest=first.config_digest,
        toolchain_digest=first.toolchain_digest,
    )
    return Generation(
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
        synthetic_commit_sha=lineage.synthetic_commit_sha,
        tree_sha=lineage.tree_sha,
        validation_evidence_ids=tuple(lineage.validation_evidence_ids),
        bisection_lineage=tuple(lineage.bisection_lineage),
        reason=lineage.reason,
    )


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
    _assert_members_form_one_generation(ordered, target_branch)
    first = ordered[0]
    generation_target = _resolved_generation_target(target, first)
    candidate_versions = tuple(member.candidate_version for member in ordered)
    generation = _build_generation(
        first,
        target_branch=target_branch,
        generation_target=generation_target,
        candidate_versions=candidate_versions,
        sealed_at=sealed_at,
        state=state,
        lineage=_GenerationLineage(
            synthetic_commit_sha=synthetic_commit_sha,
            tree_sha=tree_sha,
            validation_evidence_ids=tuple(validation_evidence_ids),
            bisection_lineage=tuple(bisection_lineage),
            reason=reason,
        ),
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
