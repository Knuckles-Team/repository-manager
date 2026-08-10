"""Adversarial RMDD-13 checkpoint 2 reservation/re-read coverage."""

from __future__ import annotations

import subprocess
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Lock
from typing import Any

import pytest

from repository_manager.development import CONTRACT_VERSION, RepositoryIdentity
from repository_manager.landing_reservation import (
    CanonicalObservation,
    CanonicalState,
    CertificationObservation,
    ControllerIdentity,
    DurableLandingReservation,
    LandingReservationConflict,
    LandingReservationController,
    LandingReservationError,
    LandingReservationRefusalCode,
    LandingReservationRequest,
    LandingReservationResult,
    LandingReservationSnapshot,
    LandingReservationUnavailable,
    OccupancyObservation,
    OccupancyState,
    TargetObservation,
    TrustedReservationRuntimeError,
    normalize_target_ref,
)

SHA_TARGET = "1" * 40
SHA_BASE = "2" * 40
SHA_TREE = "3" * 40
SHA_GENERATED = "4" * 40
DIGEST = "a" * 64
FENCE = "landing-fence-1"
REPOSITORY_ID = "repository:workspace/one"


def _repository(
    repository_id: str = REPOSITORY_ID,
    canonical_path: str = "/tmp/workspace/agent-packages/agents/repository-manager",
) -> RepositoryIdentity:
    return RepositoryIdentity(
        contract_version=CONTRACT_VERSION,
        repository_id=repository_id,
        canonical_path=canonical_path,
        configured_roots=(),
        origin=None,
    )


def _request(
    *,
    repository: RepositoryIdentity | None = None,
    target_ref: str = "main",
    expected_target_sha: str = SHA_TARGET,
    expected_base_sha: str = SHA_BASE,
    generation_id: str = "generation:one",
    certificate_digest: str = DIGEST,
    request_id: str = "request:one",
    invocation_id: str = "invocation:one",
    expected_lease_epoch: int | None = None,
    expected_lease_fence: str | None = None,
) -> LandingReservationRequest:
    return LandingReservationRequest(
        repository=repository or _repository(),
        target_ref=target_ref,
        expected_target_sha=expected_target_sha,
        expected_base_sha=expected_base_sha,
        generation_id=generation_id,
        certificate_digest=certificate_digest,
        synthetic_commit_sha=SHA_GENERATED,
        generation_tree_sha=SHA_TREE,
        landing_fence=FENCE,
        request_id=request_id,
        invocation_id=invocation_id,
        expected_lease_epoch=expected_lease_epoch,
        expected_lease_fence=expected_lease_fence,
    )


def _target(repository_id: str = REPOSITORY_ID) -> TargetObservation:
    return TargetObservation(
        repository_id=repository_id,
        target_ref="refs/heads/main",
        commit_sha=SHA_TARGET,
        tree_sha=SHA_BASE,
    )


def _canonical(
    repository_id: str = REPOSITORY_ID,
    *,
    state: CanonicalState = CanonicalState.CLEAN,
    private_wip: bool = False,
) -> CanonicalObservation:
    return CanonicalObservation(
        repository_id=repository_id,
        common_dir_id="common-dir:one",
        worktree_id="worktree:canonical",
        state=state,
        private_wip=private_wip,
    )


def _occupancy(
    repository_id: str = REPOSITORY_ID,
    *,
    count: int = 0,
    state: OccupancyState = OccupancyState.FREE,
) -> OccupancyObservation:
    return OccupancyObservation(
        repository_id=repository_id,
        target_ref="main",
        other_worktree_count=count,
        state=state,
    )


def _cert(repository_id: str = REPOSITORY_ID) -> CertificationObservation:
    return CertificationObservation(
        repository_id=repository_id,
        target_ref="main",
        generation_id="generation:one",
        certificate_digest=DIGEST,
        base_sha=SHA_BASE,
        expected_landing_base_sha=SHA_BASE,
        synthetic_commit_sha=SHA_GENERATED,
        generation_tree_sha=SHA_TREE,
        landing_fence=FENCE,
        certified=True,
    )


class FakeLease:
    def __init__(self) -> None:
        self.holds: list[str] = []
        self.effect_log: list[str] = []
        self.fail = False

    def hold(self, canonical_path: str, *, operation: str):
        self.effect_log.append("lease-request")

        @contextmanager
        def _held():
            if self.fail:
                raise LandingReservationUnavailable("lease unavailable in fixture")
            self.holds.append(operation)
            try:
                yield
            finally:
                self.effect_log.append("lease-release")

        return _held()


class FakeReader:
    def __init__(self, repository_id: str = REPOSITORY_ID) -> None:
        self.targets = [_target(repository_id), _target(repository_id)]
        self.canonicals = [_canonical(repository_id), _canonical(repository_id)]
        self.occupancies = [_occupancy(repository_id), _occupancy(repository_id)]
        self.certifications = [_cert(repository_id), _cert(repository_id)]
        self.phase = -1
        self.reads = 0
        self.on_target_read = None

    def _index(self) -> int:
        return min(max(self.phase, 0), 1)

    def read_target(self, repository_id: str, target_ref: str) -> TargetObservation:
        self.phase += 1
        self.reads += 1
        if self.on_target_read is not None:
            self.on_target_read(self.phase)
        return self.targets[self._index()]

    def read_canonical(self, repository_id: str) -> CanonicalObservation:
        self.reads += 1
        return self.canonicals[self._index()]

    def read_occupancy(
        self, repository_id: str, target_ref: str
    ) -> OccupancyObservation:
        self.reads += 1
        return self.occupancies[self._index()]

    def read_certification(
        self,
        repository_id: str,
        target_ref: str,
        generation_id: str,
        certificate_digest: str,
    ) -> CertificationObservation:
        self.reads += 1
        return self.certifications[self._index()]


class FakeAuthority:
    def __init__(self, reader: FakeReader | None = None) -> None:
        self.identity = ControllerIdentity(
            controller_id="controller:one",
            owner_id="owner:one",
            tenant_id="tenant:one",
            authority_epoch=1,
        )
        self.reader = reader
        self.reservations: dict[str, DurableLandingReservation] = {}
        self.by_target: dict[tuple[str, str], DurableLandingReservation] = {}
        self.authenticate_calls = 0
        self.reserve_calls = 0
        self.current_calls = 0
        self.current_override: DurableLandingReservation | None = None
        self.drop_current = False
        self.on_reserve = None
        self.return_owner: str | None = None

    def authenticate_controller(self, invocation_id: str) -> ControllerIdentity:
        self.authenticate_calls += 1
        return self.identity

    def reserve_landing(self, request: Any, controller: ControllerIdentity):
        self.reserve_calls += 1
        if self.on_reserve is not None:
            self.on_reserve()
        existing = self.reservations.get(request.request_digest)
        if existing is not None:
            return existing
        target_key = (request.repository_id, request.target_ref)
        if target_key in self.by_target:
            raise LandingReservationConflict("same target already reserved")
        reservation = DurableLandingReservation(
            reservation_id="reservation:one",
            request_id=request.request_id,
            invocation_id=request.invocation_id,
            repository_id=request.repository_id,
            target_ref=request.target_ref,
            request_digest=request.request_digest,
            controller_id=controller.controller_id,
            owner_id=self.return_owner or controller.owner_id,
            lease_epoch=7,
            fence="reservation-fence-7",
        )
        self.reservations[request.request_digest] = reservation
        self.by_target[target_key] = reservation
        return reservation

    def current_landing_reservation(
        self, reservation_id: str, controller: ControllerIdentity
    ) -> DurableLandingReservation | None:
        self.current_calls += 1
        if self.drop_current:
            return None
        if self.current_override is not None:
            return self.current_override
        return next(iter(self.reservations.values()), None)


class LockedAuthority(FakeAuthority):
    """Test authority whose target uniqueness is one atomic critical section."""

    def __init__(self) -> None:
        super().__init__()
        self._target_lock = Lock()

    def reserve_landing(self, request: Any, controller: ControllerIdentity):
        with self._target_lock:
            return super().reserve_landing(request, controller)


def _controller(
    reader: FakeReader | None = None,
    authority: FakeAuthority | None = None,
    lease: FakeLease | None = None,
) -> tuple[LandingReservationController, FakeAuthority, FakeReader, FakeLease]:
    actual_reader = reader or FakeReader()
    actual_authority = authority or FakeAuthority(actual_reader)
    actual_lease = lease or FakeLease()
    return (
        LandingReservationController(
            actual_authority,
            actual_reader,
            lease=actual_lease,
        ),
        actual_authority,
        actual_reader,
        actual_lease,
    )


def test_normalizes_exact_local_ref_and_rejects_injection() -> None:
    assert normalize_target_ref("main") == "refs/heads/main"
    assert normalize_target_ref("refs/heads/release/v1") == "refs/heads/release/v1"
    for ref in (
        "refs/remotes/origin/main",
        "refs/heads/../main",
        "main;git reset --hard",
        "main|cat",
        "main'",
        "main\x00",
        "main\u202ehidden",
        "main\u200bhidden",
        "feature/./main",
        "feature/.hidden/main",
    ):
        with pytest.raises(LandingReservationError):
            normalize_target_ref(ref)


def test_success_holds_lease_then_re_reads_and_returns_private_free_snapshot() -> None:
    controller, authority, reader, lease = _controller()

    result = controller.reserve(_request())

    assert result.accepted
    assert result.snapshot is not None
    assert result.snapshot.target_ref == "refs/heads/main"
    assert result.snapshot.repository_id == REPOSITORY_ID
    assert result.snapshot.observed_target_sha == SHA_TARGET
    assert result.snapshot.target_worktree_count == 0
    assert reader.reads == 8
    assert authority.reserve_calls == 1
    assert authority.current_calls == 1
    assert len(lease.holds) == 1
    assert "/tmp/" not in repr(result)
    assert "owner:one" not in repr(result)
    payload = result.snapshot.__dict__ if hasattr(result.snapshot, "__dict__") else {}
    assert not payload


def test_default_lease_composes_existing_merge_and_canonical_guards_read_only(
    tmp_path,
) -> None:
    repo = tmp_path / "repository-manager"
    repo.mkdir()
    subprocess.run(
        ["git", "init", "-b", "main"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "test"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    (repo / "README.md").write_text("stable\n")
    subprocess.run(
        ["git", "add", "README.md"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"], cwd=repo, check=True, capture_output=True
    )
    before = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()

    repository_id = "repository:temporary"
    reader = FakeReader(repository_id)
    authority = FakeAuthority(reader)
    controller = LandingReservationController(authority, reader)
    result = controller.reserve(
        _request(repository=_repository(repository_id, str(repo)))
    )

    assert result.accepted
    assert (
        subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
        == before
    )
    assert (
        subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=repo, text=True
        ).strip()
        == ""
    )


def test_exact_replay_is_idempotent_but_changed_request_conflicts_on_same_target() -> (
    None
):
    controller, authority, _reader, _lease = _controller()
    request = _request()

    first = controller.reserve(request)
    replay = controller.reserve(request)
    changed = controller.reserve(
        _request(request_id="request:two", invocation_id="invocation:two")
    )

    assert first.accepted and replay.accepted
    assert replay.snapshot is not None and first.snapshot is not None
    assert replay.snapshot.digest == first.snapshot.digest
    assert changed.code is LandingReservationRefusalCode.RESERVATION_CONFLICT
    assert authority.reserve_calls == 3


def test_two_controllers_racing_one_target_have_one_durable_winner() -> None:
    authority = LockedAuthority()
    first_reader = FakeReader()
    second_reader = FakeReader()
    first, _authority, _reader, _lease = _controller(first_reader, authority)
    second, _authority, _reader, _lease = _controller(second_reader, authority)
    requests = (
        _request(request_id="request:first", invocation_id="invocation:first"),
        _request(request_id="request:second", invocation_id="invocation:second"),
    )

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda item: item[0].reserve(item[1]),
                ((first, requests[0]), (second, requests[1])),
            )
        )

    assert sum(result.accepted for result in results) == 1
    assert (
        sum(
            result.code is LandingReservationRefusalCode.RESERVATION_CONFLICT
            for result in results
        )
        == 1
    )


def test_same_basename_repositories_have_distinct_exact_identity_keys() -> None:
    first_reader = FakeReader("repository:first")
    first_authority = FakeAuthority(first_reader)
    first_controller, _, _, _ = _controller(first_reader, first_authority)
    second_reader = FakeReader("repository:second")
    second_authority = FakeAuthority(second_reader)
    second_controller, _, _, _ = _controller(second_reader, second_authority)

    first = first_controller.reserve(
        _request(repository=_repository("repository:first"))
    )
    second = second_controller.reserve(
        _request(repository=_repository("repository:second"))
    )

    assert first.accepted and second.accepted
    assert first.snapshot is not None and second.snapshot is not None
    assert first.snapshot.repository_id != second.snapshot.repository_id


def test_target_movement_before_acquire_refuses_without_reservation_or_lease() -> None:
    reader = FakeReader()
    reader.targets[0] = replace(reader.targets[0], commit_sha=SHA_GENERATED)
    controller, authority, _reader, lease = _controller(reader)

    result = controller.reserve(_request())

    assert result.code is LandingReservationRefusalCode.TARGET_MOVED
    assert authority.reserve_calls == 0
    assert lease.holds == []


def test_target_commit_or_tree_movement_after_reservation_is_refused() -> None:
    reader = FakeReader()
    reader.targets[1] = replace(reader.targets[1], commit_sha=SHA_GENERATED)
    controller, authority, _reader, _lease = _controller(reader)
    moved = controller.reserve(_request())
    assert moved.code is LandingReservationRefusalCode.TARGET_MOVED
    assert authority.reserve_calls == 1

    reader = FakeReader()
    reader.targets[1] = replace(reader.targets[1], tree_sha=SHA_GENERATED)
    controller, _authority, _reader, _lease = _controller(reader)
    tree_changed = controller.reserve(_request())
    assert tree_changed.code is LandingReservationRefusalCode.TARGET_TREE_MISMATCH


def test_canonical_occupancy_and_certification_changes_after_hold_are_refused() -> None:
    reader = FakeReader()
    reader.canonicals[1] = replace(
        reader.canonicals[1], common_dir_id="common-dir:other"
    )
    controller, authority, _reader, _lease = _controller(reader)
    canonical_changed = controller.reserve(_request())
    assert (
        canonical_changed.code is LandingReservationRefusalCode.CANONICAL_STATE_CHANGED
    )
    assert authority.reserve_calls == 1

    reader = FakeReader()
    reader.occupancies[1] = _occupancy(count=1, state=OccupancyState.OCCUPIED)
    controller, _authority, _reader, _lease = _controller(reader)
    occupancy_changed = controller.reserve(_request())
    assert occupancy_changed.code is LandingReservationRefusalCode.TARGET_OCCUPIED

    reader = FakeReader()
    reader.certifications[1] = replace(
        reader.certifications[1], landing_fence="fence:other"
    )
    controller, _authority, _reader, _lease = _controller(reader)
    certification_changed = controller.reserve(_request())
    assert (
        certification_changed.code
        is LandingReservationRefusalCode.CERTIFICATION_CHANGED
    )


@pytest.mark.parametrize(
    ("canonical", "expected"),
    [
        (
            _canonical(state=CanonicalState.DIRTY),
            LandingReservationRefusalCode.CANONICAL_DIRTY,
        ),
        (
            _canonical(state=CanonicalState.PRIVATE_WIP),
            LandingReservationRefusalCode.PRIVATE_WIP,
        ),
        (_canonical(private_wip=True), LandingReservationRefusalCode.PRIVATE_WIP),
        (
            _canonical(state=CanonicalState.UNKNOWN),
            LandingReservationRefusalCode.CANONICAL_STATE_INVALID,
        ),
    ],
)
def test_dirty_private_wip_and_unknown_canonical_states_refuse(
    canonical: CanonicalObservation, expected: LandingReservationRefusalCode
) -> None:
    reader = FakeReader()
    reader.canonicals[0] = canonical
    reader.canonicals[1] = canonical
    controller, authority, _reader, lease = _controller(reader)

    result = controller.reserve(_request())

    assert result.code is expected
    assert authority.reserve_calls == 0
    assert lease.holds == []


@pytest.mark.parametrize(
    ("occupancy", "expected"),
    [
        (
            _occupancy(count=1, state=OccupancyState.OCCUPIED),
            LandingReservationRefusalCode.TARGET_OCCUPIED,
        ),
        (
            _occupancy(state=OccupancyState.UNKNOWN),
            LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN,
        ),
    ],
)
def test_occupied_or_unknown_target_refuses_before_reservation(
    occupancy: OccupancyObservation, expected: LandingReservationRefusalCode
) -> None:
    reader = FakeReader()
    reader.occupancies[0] = occupancy
    reader.occupancies[1] = occupancy
    controller, authority, _reader, lease = _controller(reader)

    result = controller.reserve(_request())

    assert result.code is expected
    assert authority.reserve_calls == 0
    assert lease.holds == []


def test_wrong_owner_and_stale_replay_fence_or_epoch_refuse() -> None:
    authority = FakeAuthority()
    authority.return_owner = "owner:other"
    controller, _authority, _reader, _lease = _controller(authority=authority)
    wrong_owner = controller.reserve(_request())
    assert wrong_owner.code is LandingReservationRefusalCode.OWNER_MISMATCH

    controller, authority, _reader, _lease = _controller()
    stale_epoch = controller.reserve(
        _request(expected_lease_epoch=8, expected_lease_fence="reservation-fence-7")
    )
    assert stale_epoch.code is LandingReservationRefusalCode.EPOCH_MISMATCH

    stale_fence = controller.reserve(
        _request(expected_lease_epoch=7, expected_lease_fence="reservation-fence-old")
    )
    assert stale_fence.code is LandingReservationRefusalCode.FENCE_MISMATCH
    assert authority.reserve_calls == 2


def test_reservation_is_rechecked_and_loss_is_fail_closed() -> None:
    authority = FakeAuthority()
    authority.drop_current = True
    controller, _authority, _reader, _lease = _controller(authority=authority)

    result = controller.reserve(_request())

    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_authority_and_lease_failures_are_private_fixed_codes() -> None:
    class MissingAuthority:
        pass

    reader = FakeReader()
    controller = LandingReservationController(
        MissingAuthority(), reader, lease=FakeLease()
    )  # type: ignore[arg-type]
    result = controller.reserve(_request())
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
    assert "/tmp/" not in result.detail

    lease = FakeLease()
    lease.fail = True
    controller, authority, _reader, _lease = _controller(lease=lease)
    lease_result = controller.reserve(_request())
    assert lease_result.code is LandingReservationRefusalCode.LEASE_UNAVAILABLE
    assert authority.reserve_calls == 0


def test_reader_exceptions_and_hostile_shapes_are_normalized() -> None:
    class HostileReader(FakeReader):
        def read_target(self, repository_id: str, target_ref: str):
            raise RuntimeError("/private/path and secret")

    controller, authority, _reader, lease = _controller(reader=HostileReader())
    source_failure = controller.reserve(_request())
    assert source_failure.code is LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    assert authority.reserve_calls == 0
    assert lease.holds == []
    assert "/private/path" not in source_failure.detail

    class MappingReader(FakeReader):
        def read_target(self, repository_id: str, target_ref: str):
            return {"repository_id": repository_id}  # type: ignore[return-value]

    controller, authority, _reader, _lease = _controller(reader=MappingReader())
    malformed = controller.reserve(_request())
    assert malformed.code is LandingReservationRefusalCode.SOURCE_INVALID
    assert authority.reserve_calls == 0

    class ExplodingText(str):
        def strip(self, *_args: object, **_kwargs: object) -> str:
            raise RuntimeError("hostile text method was called")

    reader = FakeReader()
    hostile = reader.targets[0]
    object.__setattr__(hostile, "commit_sha", ExplodingText(SHA_TARGET))
    reader.targets[1] = hostile
    controller, authority, _reader, _lease = _controller(reader)
    hostile_result = controller.reserve(_request())
    assert hostile_result.code is LandingReservationRefusalCode.SOURCE_INVALID
    assert authority.reserve_calls == 0


def test_forged_pydantic_repository_snapshot_is_refused_without_leaking_path() -> None:
    controller, authority, _reader, _lease = _controller()
    request = _request()
    forged = RepositoryIdentity.model_construct(repository_id="repository:forged")
    object.__setattr__(request, "repository", forged)

    result = controller.reserve(request)

    assert result.code is LandingReservationRefusalCode.REQUEST_INVALID
    assert authority.reserve_calls == 0
    assert "/tmp/workspace" not in result.detail


def test_trusted_programmer_runtime_error_crosses_reader_boundary() -> None:
    class TrustedReader(FakeReader):
        def read_target(self, repository_id: str, target_ref: str):
            raise TrustedReservationRuntimeError("programmer failure")

    controller, _authority, _reader, _lease = _controller(reader=TrustedReader())
    with pytest.raises(TrustedReservationRuntimeError):
        controller.reserve(_request())


def test_trusted_authority_runtime_error_crosses_authority_boundary() -> None:
    class TrustedAuthority(FakeAuthority):
        def authenticate_controller(self, invocation_id: str):
            raise RuntimeError("trusted authority programmer failure")

    controller, _authority, _reader, _lease = _controller(authority=TrustedAuthority())
    with pytest.raises(RuntimeError, match="trusted authority programmer failure"):
        controller.reserve(_request())


def test_snapshot_digest_and_forged_source_state_are_not_accepted() -> None:
    controller, _authority, _reader, _lease = _controller()
    result = controller.reserve(_request())
    assert result.snapshot is not None
    forged = result.snapshot
    object.__setattr__(forged, "digest", "b" * 64)
    with pytest.raises(LandingReservationError, match="snapshot digest"):
        forged.__post_init__()

    reader = FakeReader()
    forged_canonical = replace(reader.canonicals[1])
    object.__setattr__(forged_canonical, "private_wip", 1)
    reader.canonicals[1] = forged_canonical
    controller, authority, _reader, _lease = _controller(reader)
    malformed = controller.reserve(_request())
    assert malformed.code is LandingReservationRefusalCode.SOURCE_INVALID
    assert authority.reserve_calls == 1

    forged_result = object.__new__(LandingReservationResult)
    object.__setattr__(forged_result, "accepted", True)
    object.__setattr__(forged_result, "refusal_code", None)
    object.__setattr__(forged_result, "detail", "")
    object.__setattr__(forged_result, "snapshot", forged)
    with pytest.raises(LandingReservationError, match="snapshot digest"):
        forged_result.__post_init__()


def test_reservation_result_has_no_public_owner_or_process_fields() -> None:
    assert "owner_id" not in LandingReservationRequest.__dataclass_fields__
    assert "controller_id" not in LandingReservationRequest.__dataclass_fields__
    assert "pid" not in LandingReservationSnapshot.__dataclass_fields__
    assert "canonical_path" not in LandingReservationSnapshot.__dataclass_fields__
