"""Immutable candidate snapshots and append-only generation records.

This module is the storage-neutral part of RMDD-12.  Git and the merge queue
provide the branch/base snapshots; this module records those snapshots in the
same append-only :class:`~agent_utilities.governance.lanes.FragmentStore`
authority used by the existing queue.  It deliberately does not create a
second job store, execute commands, or move a ref.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from agent_utilities.governance.lanes import FragmentStore, lane_scope

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

GENERATION_STORE_DIRNAME = "merge-queue-generations"
CANDIDATE_STORE_DIRNAME = "candidates"
GENERATION_RECORD_DIRNAME = "generations"


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


def digest_input(value: object) -> str:
    """Return a valid contract digest, hashing shorthand test/config values."""

    text = str(value)
    if len(text) == 64 and all(char in "0123456789abcdef" for char in text):
        return text
    return canonical_digest(value)


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
        object.__setattr__(self, "config_digest", digest_input(self.config_digest))
        object.__setattr__(
            self, "toolchain_digest", digest_input(self.toolchain_digest)
        )
        object.__setattr__(self, "resource_digest", digest_input(self.resource_digest))
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
        if record.get("kind") != "candidate_snapshot":
            raise CandidateGenerationError("record is not a candidate snapshot")
        contract = Candidate.model_validate(record.get("candidate"))
        snapshot = cls(
            candidate_id=contract.candidate_id,
            version=contract.version,
            repository=contract.repository,
            branch=contract.branch,
            target_branch=str(record.get("target_branch", "main")),
            candidate_sha=contract.candidate_sha,
            base_sha=contract.base_sha,
            lane_id=contract.lane_id,
            owner_id=contract.owner_id,
            config_digest=contract.config_digest,
            toolchain_digest=str(record.get("toolchain_digest", "")),
            resource_digest=str(record.get("resource_digest", "")),
            build_target=str(record.get("build_target", "default")),
            concept_claims=contract.concept_claims,
            incompatibility_labels=tuple(record.get("incompatibility_labels", ())),
            enqueued_at=contract.enqueued_at,
            target=TargetPolicy.model_validate(record.get("target", {})),
            recorded_at=str(record.get("recorded_at", "")),
        )
        recorded_id = str(record.get("record_id", ""))
        if recorded_id and recorded_id != snapshot.record_id:
            raise CandidateGenerationError(
                f"candidate record key does not match {snapshot.record_id}"
            )
        recorded_digest = str(record.get("immutable_digest", ""))
        if recorded_digest and recorded_digest != snapshot.immutable_digest():
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
        versions = tuple(
            CandidateVersion(
                candidate_id=member.candidate_id,
                version=index,
                candidate_sha=member.candidate_sha,
            )
            for index, member in enumerate(self.members, start=1)
        )
        if versions != self.generation.candidate_versions:
            raise CandidateGenerationError(
                "generation membership does not match its candidate versions"
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
        }

    @classmethod
    def from_record(cls, record: dict[str, Any]) -> GenerationRecord:
        if record.get("kind") != "generation":
            raise CandidateGenerationError("record is not a generation")
        generation = Generation.model_validate(record.get("generation"))
        members = tuple(
            CandidateSnapshot.from_record(item) for item in record.get("members", ())
        )
        return cls(
            generation=generation,
            members=members,
            result_json=canonical_json(record.get("result", {})),
            recorded_at=str(record.get("recorded_at", "")),
        )


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

    return max(
        group,
        key=lambda record: (
            timestamp_value(str(record.get("recorded_at", "1970-01-01T00:00:00Z"))),
            canonical_json(record),
        ),
    )


class _RecordStore(Protocol):
    def append(self, record: dict[str, Any], *, lane: str) -> Path: ...

    def fold(
        self, resolve: Callable[[list[dict[str, Any]]], dict[str, Any]]
    ) -> list[dict[str, Any]]: ...


class CandidateLedger:
    """Append-only candidate snapshot authority backed by FragmentStore."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        store: _RecordStore | None = None,
        lane: str | None = None,
    ) -> None:
        if store is None:
            scope = lane_scope(path)
            self.store = FragmentStore(
                root=scope.arbitration_dir
                / GENERATION_STORE_DIRNAME
                / CANDIDATE_STORE_DIRNAME,
                key="record_id",
            )
            self.lane = lane or scope.lane
        else:
            self.store = store
            self.lane = lane or "generation"

    def append(self, snapshot: CandidateSnapshot, *, lane: str | None = None) -> None:
        existing = self.get(snapshot.record_id)
        if existing is not None:
            if existing.immutable_digest() != snapshot.immutable_digest():
                raise CandidateGenerationError(
                    f"candidate snapshot {snapshot.record_id} was mutated"
                )
            return
        self.store.append(snapshot.to_record(), lane=lane or self.lane)

    def all(self) -> tuple[CandidateSnapshot, ...]:
        snapshots = tuple(
            CandidateSnapshot.from_record(record)
            for record in self.store.fold(resolve=_latest_record)
        )
        for snapshot in snapshots:
            self._validate_history(snapshot.record_id)
        return snapshots

    def get(self, record_id: str) -> CandidateSnapshot | None:
        return next((item for item in self.all() if item.record_id == record_id), None)

    def latest_for(self, candidate_id: str) -> CandidateSnapshot | None:
        matches = [item for item in self.all() if item.candidate_id == candidate_id]
        return max(matches, key=lambda item: item.version, default=None)

    def _validate_history(self, record_id: str) -> None:
        records: list[CandidateSnapshot] = []
        for lane in getattr(self.store, "lanes", lambda: [])():
            for raw in getattr(self.store, "read_fragment", lambda _lane: [])(lane):
                if str(raw.get("record_id")) == record_id:
                    records.append(CandidateSnapshot.from_record(raw))
        if not records:
            return
        first = records[0]
        for snapshot in records[1:]:
            if snapshot.immutable_digest() != first.immutable_digest():
                raise CandidateGenerationError(
                    f"candidate {record_id} has conflicting append-only inputs"
                )


class GenerationLedger:
    """Append-only generation authority with immutable-field reconciliation."""

    def __init__(
        self,
        path: Path | str | None = None,
        *,
        store: _RecordStore | None = None,
        lane: str | None = None,
    ) -> None:
        if store is None:
            scope = lane_scope(path)
            self.store = FragmentStore(
                root=scope.arbitration_dir
                / GENERATION_STORE_DIRNAME
                / GENERATION_RECORD_DIRNAME,
                key="record_id",
            )
            self.lane = lane or scope.lane
        else:
            self.store = store
            self.lane = lane or "generation"

    def append(self, record: GenerationRecord, *, lane: str | None = None) -> None:
        previous = self.get(record.record_id)
        if previous is not None:
            _assert_generation_identity(previous.generation, record.generation)
            if tuple(previous.members) != tuple(record.members):
                raise CandidateGenerationError(
                    f"generation {record.record_id} membership was mutated"
                )
            if (
                previous.generation.state != record.generation.state
                and not is_legal_transition(
                    previous.generation.state, record.generation.state
                )
            ):
                raise CandidateGenerationError(
                    f"illegal generation transition: {previous.generation.state} -> "
                    f"{record.generation.state}"
                )
        self.store.append(record.to_record(), lane=lane or self.lane)

    def all(self) -> tuple[GenerationRecord, ...]:
        records = tuple(
            GenerationRecord.from_record(record)
            for record in self.store.fold(resolve=_latest_record)
        )
        for record in records:
            self._validate_history(record.record_id)
        return records

    def get(self, generation_id: str) -> GenerationRecord | None:
        return next(
            (
                item
                for item in self.all_without_history()
                if item.record_id == generation_id
            ),
            None,
        )

    def all_without_history(self) -> tuple[GenerationRecord, ...]:
        return tuple(
            GenerationRecord.from_record(record)
            for record in self.store.fold(resolve=_latest_record)
        )

    def _validate_history(self, generation_id: str) -> None:
        records: list[GenerationRecord] = []
        for lane in getattr(self.store, "lanes", lambda: [])():
            for raw in getattr(self.store, "read_fragment", lambda _lane: [])(lane):
                if str(raw.get("record_id")) == generation_id:
                    records.append(GenerationRecord.from_record(raw))
        if not records:
            return
        first = records[0]
        for record in records[1:]:
            _assert_generation_identity(first.generation, record.generation)
            if record.members != first.members:
                raise CandidateGenerationError(
                    f"generation {generation_id} has conflicting append-only members"
                )

    def reconcile(self) -> tuple[GenerationRecord, ...]:
        """Rebuild the latest durable generation view after a restart."""

        return self.all()


def snapshot_candidate(
    candidate: CandidateLike,
    *,
    repository: RepositoryIdentity,
    candidate_sha: str,
    base_sha: str,
    config_digest: str,
    toolchain_digest: str = "",
    resource_digest: str = "",
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
    toolchain_digest: str = "",
    resource_digest: str = "",
    build_target: str = "default",
    target_branch: str | None = None,
    owner_id: str | None = None,
    concept_claims: Iterable[object] = (),
    incompatibility_labels: Iterable[object] = (),
    target: TargetPolicy | None = None,
    ledger: CandidateLedger | None = None,
    lane: str | None = None,
) -> CandidateSnapshot:
    """Resolve branch/base exactly once and append a new version if moved."""

    logical_id = candidate_identity(
        repository.repository_id, candidate.branch, candidate.lane
    )
    candidate_sha = resolve_ref(candidate.branch)
    base_sha = resolve_ref(target_branch or candidate.base)
    previous = ledger.latest_for(logical_id) if ledger is not None else None
    target_value = target or TargetPolicy()
    concepts = _tuple_strings(concept_claims)
    labels = _tuple_strings(incompatibility_labels)
    version = 1
    if previous is not None:
        version = previous.version
        immutable_inputs_changed = (
            previous.candidate_sha != candidate_sha
            or previous.base_sha != base_sha
            or previous.config_digest != digest_input(config_digest)
            or previous.toolchain_digest != digest_input(toolchain_digest)
            or previous.resource_digest != digest_input(resource_digest)
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
    if ledger is not None:
        ledger.append(snapshot, lane=lane)
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
    generation_id = Generation.derive_id(
        repository_id=first.repository.repository_id,
        target_branch=target_branch,
        base_sha=first.base_sha,
        candidate_versions=tuple(
            CandidateVersion(
                candidate_id=member.candidate_id,
                version=index,
                candidate_sha=member.candidate_sha,
            )
            for index, member in enumerate(ordered, start=1)
        ),
        config_digest=first.config_digest,
        toolchain_digest=first.toolchain_digest,
    )
    generation = Generation(
        generation_id=generation_id,
        repository=first.repository,
        target_branch=target_branch,
        target=target or first.target,
        base_sha=first.base_sha,
        expected_landing_base_sha=first.base_sha,
        candidate_versions=tuple(
            CandidateVersion(
                candidate_id=member.candidate_id,
                version=index,
                candidate_sha=member.candidate_sha,
            )
            for index, member in enumerate(ordered, start=1)
        ),
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


def generation_ledger(path: Path | str | None = None) -> GenerationLedger:
    """Return the generation ledger for one repository."""

    return GenerationLedger(path)


def candidate_ledger(path: Path | str | None = None) -> CandidateLedger:
    """Return the candidate snapshot ledger for one repository."""

    return CandidateLedger(path)


__all__ = [
    "CandidateGenerationError",
    "CandidateLedger",
    "CandidateSnapshot",
    "GenerationLedger",
    "GenerationRecord",
    "candidate_identity",
    "candidate_ledger",
    "digest_input",
    "generation_ledger",
    "generation_record",
    "snapshot_branch_candidate",
    "snapshot_candidate",
    "timestamp_value",
]
