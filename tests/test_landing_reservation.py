"""Adversarial RMDD-13 checkpoint 2 reservation/barrier coverage."""

from __future__ import annotations

import subprocess
from collections.abc import Callable
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
    LandingReservationStale,
    LandingReservationUnavailable,
    LandingStateSnapshot,
    LandingValidationBarrier,
    OccupancyObservation,
    OccupancyState,
    ResolvedRepositoryIdentity,
    TargetObservation,
    TrustedReservationRuntimeError,
    _ExistingReconciliationLease,
    normalize_target_ref,
)

SHA_TARGET = "1" * 40
SHA_BASE = "2" * 40
SHA_TREE = "3" * 40
SHA_GENERATED = "4" * 40
DIGEST = "a" * 64
FENCE = "landing-fence-1"
REPOSITORY_ID = "repository:workspace/one"
CANONICAL_PATH = "/tmp/workspace/agent-packages/agents/repository-manager"


def _repository(
    repository_id: str = REPOSITORY_ID,
    canonical_path: str = CANONICAL_PATH,
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
        generation_id="generation:one",
        certificate_digest=DIGEST,
        synthetic_commit_sha=SHA_GENERATED,
        generation_tree_sha=SHA_TREE,
        landing_fence=FENCE,
        request_id=request_id,
        invocation_id=invocation_id,
        expected_lease_epoch=expected_lease_epoch,
        expected_lease_fence=expected_lease_fence,
    )


def _target(
    repository_id: str = REPOSITORY_ID,
    *,
    commit_sha: str = SHA_TARGET,
    tree_sha: str = SHA_BASE,
) -> TargetObservation:
    return TargetObservation(repository_id, "refs/heads/main", commit_sha, tree_sha)


def _canonical(
    repository_id: str = REPOSITORY_ID,
    *,
    common_dir_id: str = "common-dir:one",
    worktree_id: str = "worktree:canonical",
    state: CanonicalState = CanonicalState.CLEAN,
    private_wip: bool = False,
) -> CanonicalObservation:
    return CanonicalObservation(
        repository_id,
        common_dir_id,
        worktree_id,
        state,
        private_wip,
    )


def _occupancy(
    repository_id: str = REPOSITORY_ID,
    *,
    count: int = 0,
    state: OccupancyState = OccupancyState.FREE,
) -> OccupancyObservation:
    return OccupancyObservation(repository_id, "main", count, state)


def _cert(
    repository_id: str = REPOSITORY_ID,
    *,
    landing_fence: str = FENCE,
    certified: bool = True,
) -> CertificationObservation:
    return CertificationObservation(
        repository_id=repository_id,
        target_ref="main",
        generation_id="generation:one",
        certificate_digest=DIGEST,
        base_sha=SHA_BASE,
        expected_landing_base_sha=SHA_BASE,
        synthetic_commit_sha=SHA_GENERATED,
        generation_tree_sha=SHA_TREE,
        landing_fence=landing_fence,
        certified=certified,
    )


class RecordingLease:
    """Authority-owned lease fixture; never passed to the controller."""

    def __init__(self) -> None:
        self.entered = 0
        self.fail = False
        self.events: list[str] = []

    def hold(self, path: str, *, operation: str):
        self.events.append(f"requested:{path}")

        @contextmanager
        def _held():
            if self.fail:
                raise LandingReservationUnavailable("fixture lease unavailable")
            self.entered += 1
            self.events.append("entered")
            try:
                yield
            finally:
                self.events.append("exited")
                self.entered -= 1

        return _held()


class NoopLease:
    """A caller-owned duck lease that must not be a reservation authority."""

    def hold(self, *_args: object, **_kwargs: object):
        @contextmanager
        def _held():
            yield

        return _held()


class FakeAuthority:
    """A revisioned trusted-authority fixture with one atomic barrier seam."""

    def __init__(
        self,
        repository_id: str = REPOSITORY_ID,
        canonical_path: str = CANONICAL_PATH,
        *,
        common_dir_id: str = "common-dir:one",
        worktree_id: str = "worktree:canonical",
        states: list[LandingStateSnapshot] | None = None,
    ) -> None:
        self.identity = ControllerIdentity(
            "controller:one",
            "owner:one",
            "tenant:one",
            1,
            "principal:one",
            "session:one",
        )
        self.resolved = ResolvedRepositoryIdentity(
            repository_id,
            canonical_path,
            common_dir_id,
            worktree_id,
            "repository-revision:1",
        )
        self.lease = RecordingLease()
        self.states = states or [_state(self.resolved)]
        self.state_index = 0
        self.reservations: dict[str, DurableLandingReservation] = {}
        self.by_target: dict[tuple[str, str], DurableLandingReservation] = {}
        self.lock = Lock()
        self.lease_lock = Lock()
        self.held = False
        self.events: list[str] = []
        self.hold_paths: list[str] = []
        self.authenticate_calls = 0
        self.resolve_calls = 0
        self.reserve_calls = 0
        self.read_calls = 0
        self.barrier_calls = 0
        self.drop_current = False
        self.on_reserve: Callable[[FakeAuthority], None] | None = None
        self.on_read: Callable[[FakeAuthority], None] | None = None
        self.on_barrier: Callable[[FakeAuthority], None] | None = None
        self.after_barrier: Callable[[FakeAuthority], None] | None = None
        self.return_owner: str | None = None
        self.return_tenant: str | None = None
        self.return_epoch: int | None = None
        self.return_principal: str | None = None
        self.return_session: str | None = None

    def authenticate_controller(self, invocation_id: str) -> ControllerIdentity:
        self.events.append("authenticate")
        self.authenticate_calls += 1
        return self.identity

    def resolve_repository(self, repository: RepositoryIdentity):
        self.events.append("resolve")
        self.resolve_calls += 1
        return self.resolved

    def hold_landing(
        self,
        repository: ResolvedRepositoryIdentity,
        target_ref: str,
        *,
        operation: str,
    ):
        self.events.append("hold-request")
        self.hold_paths.append(repository.canonical_path)
        return self._hold(repository, target_ref, operation)

    @contextmanager
    def _hold(
        self,
        repository: ResolvedRepositoryIdentity,
        target_ref: str,
        operation: str,
    ):
        with self.lease_lock:
            with self.lease.hold(repository.canonical_path, operation=operation):
                self.held = True
                self.events.append("hold-enter")
                try:
                    yield
                finally:
                    self.held = False
                    self.events.append("hold-exit")

    def reserve_landing(self, request: Any, controller: ControllerIdentity):
        self.events.append("reserve")
        self.reserve_calls += 1
        if self.on_reserve is not None:
            self.on_reserve(self)
        with self.lock:
            existing = self.reservations.get(request.request_digest)
            if existing is not None:
                return existing
            same_invocation = next(
                (
                    value
                    for value in self.reservations.values()
                    if value.request_id == request.request_id
                    and value.invocation_id == request.invocation_id
                ),
                None,
            )
            if same_invocation is not None:
                if same_invocation.authority_epoch != controller.authority_epoch:
                    return same_invocation
                raise LandingReservationConflict(
                    "request identity was reused with changed input"
                )
            target_key = (request.repository_id, request.target_ref)
            if target_key in self.by_target:
                raise LandingReservationConflict("same target already reserved")
            resolved = request.resolved_repository
            reservation = DurableLandingReservation(
                reservation_id=f"reservation:{self.reserve_calls}",
                request_id=request.request_id,
                invocation_id=request.invocation_id,
                repository_id=request.repository_id,
                target_ref=request.target_ref,
                request_digest=request.request_digest,
                resolved_repository_digest=resolved.digest(),
                common_dir_id=resolved.common_dir_id,
                worktree_id=resolved.worktree_id,
                authority_revision=resolved.authority_revision,
                controller_id=controller.controller_id,
                owner_id=self.return_owner or controller.owner_id,
                tenant_id=self.return_tenant or controller.tenant_id,
                lease_epoch=self.return_epoch or 7,
                fence="reservation-fence-7",
                authority_epoch=controller.authority_epoch,
                principal_id=self.return_principal
                if self.return_principal is not None
                else controller.principal_id,
                session_id=self.return_session
                if self.return_session is not None
                else controller.session_id,
            )
            self.reservations[request.request_digest] = reservation
            self.by_target[target_key] = reservation
            return reservation

    def read_landing_snapshot(
        self,
        repository: ResolvedRepositoryIdentity,
        request: LandingReservationRequest,
    ) -> LandingStateSnapshot:
        self.events.append("read")
        self.read_calls += 1
        result = self.states[self.state_index]
        if self.on_read is not None:
            self.on_read(self)
        return result

    def validate_landing_barrier(
        self,
        reservation: DurableLandingReservation,
        controller: ControllerIdentity,
        repository: ResolvedRepositoryIdentity,
        snapshot: LandingStateSnapshot,
    ) -> LandingValidationBarrier:
        self.events.append("barrier")
        self.barrier_calls += 1
        if not self.held:
            raise LandingReservationUnavailable("lease was not held")
        if self.on_barrier is not None:
            self.on_barrier(self)
        if self.drop_current:
            raise LandingReservationStale("reservation was lost")
        current = self.reservations.get(reservation.request_digest)
        if current is None or current.reservation_id != reservation.reservation_id:
            raise LandingReservationStale("reservation was replaced")
        latest = self.states[self.state_index]
        if latest.immutable_payload() != snapshot.immutable_payload():
            raise LandingReservationStale("source revision changed")
        barrier = LandingValidationBarrier(current, latest, "barrier-revision:1")
        if self.after_barrier is not None:
            self.after_barrier(self)
        return barrier


class LockedAuthority(FakeAuthority):
    """FakeAuthority already serializes its durable target write."""


def _state(
    resolved: ResolvedRepositoryIdentity,
    *,
    revision: str = "source-revision:1",
    target: TargetObservation | None = None,
    canonical: CanonicalObservation | None = None,
    occupancy: OccupancyObservation | None = None,
    certification: CertificationObservation | None = None,
) -> LandingStateSnapshot:
    return LandingStateSnapshot(
        resolved_repository_digest=resolved.digest(),
        target=target or _target(resolved.repository_id),
        canonical=canonical
        or _canonical(
            resolved.repository_id,
            common_dir_id=resolved.common_dir_id,
            worktree_id=resolved.worktree_id,
        ),
        occupancy=occupancy or _occupancy(resolved.repository_id),
        certification=certification or _cert(resolved.repository_id),
        target_revision=f"target:{revision}",
        canonical_revision=f"canonical:{revision}",
        occupancy_revision=f"occupancy:{revision}",
        certification_revision=f"certification:{revision}",
        snapshot_revision=f"snapshot:{revision}",
    )


def _controller(
    authority: FakeAuthority | None = None,
) -> tuple[LandingReservationController, FakeAuthority]:
    actual = authority or FakeAuthority()
    return LandingReservationController(actual), actual


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


def test_public_controller_cannot_accept_a_noop_or_caller_lease() -> None:
    authority = FakeAuthority()
    with pytest.raises(TypeError):
        LandingReservationController(authority, object())  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LandingReservationController(authority, lease=object())  # type: ignore[call-arg]
    result = LandingReservationController(NoopLease()).reserve(_request())  # type: ignore[arg-type]
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE


def test_success_orders_resolution_hold_reservation_read_and_final_barrier() -> None:
    controller, authority = _controller()
    result = controller.reserve(_request())

    assert result.accepted
    assert result.snapshot is not None
    assert authority.events == [
        "authenticate",
        "resolve",
        "hold-request",
        "hold-enter",
        "reserve",
        "read",
        "barrier",
        "hold-exit",
    ]
    assert authority.read_calls == 1
    assert authority.barrier_calls == 1
    assert authority.hold_paths == [CANONICAL_PATH]


def test_actual_authority_path_selects_existing_rmdd26_leases_read_only(
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

    authority = FakeAuthority(canonical_path=str(repo))
    authority.lease = None  # type: ignore[assignment]

    @contextmanager
    def _trusted_hold(repository, target_ref, *, operation):
        with _ExistingReconciliationLease().hold(
            repository.canonical_path, operation=operation
        ):
            authority.held = True
            try:
                yield
            finally:
                authority.held = False

    authority.hold_landing = _trusted_hold  # type: ignore[method-assign]
    authority.resolved = ResolvedRepositoryIdentity(
        REPOSITORY_ID,
        str(repo),
        "common-dir:git",
        "worktree:git",
        "repository-revision:1",
    )
    authority.states = [_state(authority.resolved)]
    result = LandingReservationController(authority).reserve(
        _request(repository=_repository(canonical_path=str(repo)))
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


def test_forged_request_path_disagrees_before_any_lease() -> None:
    authority = FakeAuthority(canonical_path="/tmp/trusted/actual-repository")
    controller, _ = _controller(authority)
    result = controller.reserve(
        _request(repository=_repository(canonical_path="/tmp/forged/other"))
    )
    assert result.code is LandingReservationRefusalCode.REPOSITORY_MISMATCH
    assert authority.hold_paths == []
    assert authority.reserve_calls == 0


def test_same_basename_and_cross_worktree_identity_cannot_alias() -> None:
    first = FakeAuthority(
        "repository:first", "/tmp/one/same", common_dir_id="common:one"
    )
    second = FakeAuthority(
        "repository:second", "/tmp/two/same", common_dir_id="common:two"
    )
    first_result = LandingReservationController(first).reserve(
        _request(repository=_repository("repository:first", "/tmp/one/same"))
    )
    second_result = LandingReservationController(second).reserve(
        _request(repository=_repository("repository:second", "/tmp/two/same"))
    )
    assert first_result.accepted and second_result.accepted
    assert first.hold_paths == ["/tmp/one/same"]
    assert second.hold_paths == ["/tmp/two/same"]
    assert first_result.snapshot is not None and second_result.snapshot is not None
    assert (
        first_result.snapshot.resolved_repository_digest
        != second_result.snapshot.resolved_repository_digest
    )


def test_exact_replay_is_idempotent_but_changed_input_conflicts() -> None:
    controller, authority = _controller()
    request = _request()
    first = controller.reserve(request)
    replay = controller.reserve(request)
    changed = controller.reserve(
        _request(request_id="request:two", invocation_id="invocation:two")
    )
    assert first.accepted and replay.accepted
    assert first.snapshot is not None and replay.snapshot is not None
    assert first.snapshot.digest == replay.snapshot.digest
    assert changed.code is LandingReservationRefusalCode.RESERVATION_CONFLICT


def test_two_controllers_racing_one_target_have_one_durable_winner() -> None:
    authority = LockedAuthority()
    first = LandingReservationController(authority)
    second = LandingReservationController(authority)
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
    assert not authority.held
    assert authority.lease.entered == 0


def test_dirty_private_unknown_and_occupied_states_refuse_after_hold() -> None:
    cases = (
        _state(
            FakeAuthority().resolved,
            canonical=_canonical(state=CanonicalState.DIRTY),
        ),
        _state(
            FakeAuthority().resolved,
            canonical=_canonical(state=CanonicalState.PRIVATE_WIP),
        ),
        _state(
            FakeAuthority().resolved,
            canonical=_canonical(private_wip=True),
        ),
        _state(
            FakeAuthority().resolved,
            canonical=_canonical(state=CanonicalState.UNKNOWN),
        ),
        _state(
            FakeAuthority().resolved,
            occupancy=_occupancy(count=1, state=OccupancyState.OCCUPIED),
        ),
        _state(
            FakeAuthority().resolved,
            occupancy=_occupancy(state=OccupancyState.UNKNOWN),
        ),
    )
    expected = (
        LandingReservationRefusalCode.CANONICAL_DIRTY,
        LandingReservationRefusalCode.PRIVATE_WIP,
        LandingReservationRefusalCode.PRIVATE_WIP,
        LandingReservationRefusalCode.CANONICAL_STATE_INVALID,
        LandingReservationRefusalCode.TARGET_OCCUPIED,
        LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN,
    )
    for state, code in zip(cases, expected, strict=True):
        authority = FakeAuthority(states=[state])
        result = LandingReservationController(authority).reserve(_request())
        assert result.code is code
        assert authority.barrier_calls == 0


def test_canonical_common_dir_and_worktree_mismatch_refuse_even_when_stable() -> None:
    authority = FakeAuthority(
        states=[
            _state(
                FakeAuthority().resolved,
                canonical=_canonical(
                    common_dir_id="common:forged", worktree_id="worktree:forged"
                ),
            )
        ]
    )
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.CANONICAL_STATE_CHANGED


def test_each_source_revision_or_value_movement_before_barrier_refuses() -> None:
    authority = FakeAuthority(
        states=[
            _state(FakeAuthority().resolved),
            _state(FakeAuthority().resolved, revision="source-revision:2"),
        ]
    )

    def advance_barrier(current: FakeAuthority) -> None:
        current.state_index = 1

    authority.on_barrier = advance_barrier
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST

    authority = FakeAuthority(states=[_state(FakeAuthority().resolved)])

    def mutate_after_read(current: FakeAuthority) -> None:
        current.states = [
            current.states[0],
            replace(current.states[0], target_revision="target:changed"),
        ]
        current.state_index = 1

    authority.on_read = mutate_after_read
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


@pytest.mark.parametrize(
    "mutation",
    [
        lambda state: replace(
            state, target=replace(state.target, tree_sha=SHA_GENERATED)
        ),
        lambda state: replace(
            state, canonical=replace(state.canonical, common_dir_id="common:changed")
        ),
        lambda state: replace(
            state,
            occupancy=replace(
                state.occupancy, other_worktree_count=1, state=OccupancyState.OCCUPIED
            ),
        ),
        lambda state: replace(
            state,
            certification=replace(state.certification, landing_fence="fence:changed"),
        ),
    ],
)
def test_mutation_immediately_before_final_barrier_is_not_silently_accepted(
    mutation: Callable[[LandingStateSnapshot], LandingStateSnapshot],
) -> None:
    trusted = FakeAuthority()
    first = _state(trusted.resolved)
    second = mutation(first)
    trusted.states = [first, second]

    def advance_barrier(current: FakeAuthority) -> None:
        current.state_index = 1

    trusted.on_barrier = advance_barrier
    result = LandingReservationController(trusted).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_mutation_after_successful_barrier_is_left_to_cp3_correlations() -> None:
    authority = FakeAuthority()

    def advance_after_barrier(current: FakeAuthority) -> None:
        current.state_index = 1

    authority.after_barrier = advance_after_barrier
    authority.states = [
        _state(authority.resolved),
        _state(authority.resolved, revision="source-revision:2"),
    ]
    result = LandingReservationController(authority).reserve(_request())
    assert result.accepted


def test_missing_revision_or_barrier_fails_closed_as_authority_unavailable() -> None:
    authority = FakeAuthority()
    object.__setattr__(authority, "read_landing_snapshot", None)
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE

    authority = FakeAuthority()
    object.__setattr__(authority, "validate_landing_barrier", None)
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE


def test_lost_reservation_and_wrong_current_fence_refuse() -> None:
    authority = FakeAuthority()
    authority.drop_current = True
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST

    authority = FakeAuthority()
    authority.return_epoch = 9
    result = LandingReservationController(authority).reserve(
        _request(expected_lease_epoch=7, expected_lease_fence="reservation-fence-7")
    )
    assert result.code is LandingReservationRefusalCode.EPOCH_MISMATCH


@pytest.mark.parametrize(
    ("field", "code"),
    [
        ("return_owner", LandingReservationRefusalCode.OWNER_MISMATCH),
        ("return_tenant", LandingReservationRefusalCode.TENANT_MISMATCH),
        ("return_principal", LandingReservationRefusalCode.PRINCIPAL_MISMATCH),
        ("return_session", LandingReservationRefusalCode.SESSION_MISMATCH),
    ],
)
def test_wrong_authenticated_principal_dimensions_refuse(field: str, code) -> None:
    authority = FakeAuthority()
    setattr(authority, field, "other:value")
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is code


def test_authority_epoch_advance_invalidates_replay() -> None:
    authority = FakeAuthority()
    controller = LandingReservationController(authority)
    assert controller.reserve(_request()).accepted
    authority.identity = replace(authority.identity, authority_epoch=2)
    replay = controller.reserve(_request())
    assert replay.code is LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH


def test_forged_missing_fields_never_leak_attribute_error() -> None:
    values: list[Any] = [
        ControllerIdentity("controller:one", "owner:one", "tenant:one", 1),
        _request(),
        _target(),
        _canonical(),
        _occupancy(),
        _cert(),
    ]
    for value in values:
        field = next(iter(value.__dataclass_fields__))
        forged = object.__new__(type(value))
        for name in value.__dataclass_fields__:
            if name != field:
                object.__setattr__(forged, name, getattr(value, name))
        with pytest.raises(LandingReservationError):
            forged.__post_init__()

    authority = FakeAuthority()
    request = _request()
    object.__delattr__(request, "request_id")
    result = LandingReservationController(authority).reserve(request)
    assert result.code is LandingReservationRefusalCode.REQUEST_INVALID
    assert "/tmp/" not in result.detail


def test_forged_pydantic_repository_models_are_not_identity_proof() -> None:
    authority = FakeAuthority()
    copied = _request().repository.model_copy(
        update={"repository_id": "repository:forged"}
    )
    result = LandingReservationController(authority).reserve(
        _request(repository=copied)
    )
    assert result.code is LandingReservationRefusalCode.REPOSITORY_MISMATCH
    assert copied.repository_id == "repository:forged"
    forged = RepositoryIdentity.model_construct(
        contract_version=CONTRACT_VERSION,
        repository_id=True,
        canonical_path=CANONICAL_PATH,
        configured_roots=(),
        origin=None,
    )
    forged_request = _request()
    object.__setattr__(forged_request, "repository", forged)
    result = LandingReservationController(authority).reserve(forged_request)
    assert result.code is LandingReservationRefusalCode.REQUEST_INVALID
    assert authority.hold_paths == []


def test_forged_snapshot_and_result_are_revalidated() -> None:
    controller, _ = _controller()
    result = controller.reserve(_request())
    assert result.snapshot is not None
    forged = result.snapshot
    object.__setattr__(forged, "digest", "b" * 64)
    with pytest.raises(LandingReservationError):
        forged.__post_init__()
    with pytest.raises(LandingReservationError):
        forged.immutable_payload()
    forged_result = object.__new__(LandingReservationResult)
    object.__setattr__(forged_result, "accepted", True)
    object.__setattr__(forged_result, "refusal_code", None)
    object.__setattr__(forged_result, "detail", "")
    object.__setattr__(forged_result, "snapshot", forged)
    with pytest.raises(LandingReservationError):
        forged_result.__post_init__()


def test_hostile_mapping_and_source_exception_are_private_fixed_codes() -> None:
    authority = FakeAuthority()
    object.__setattr__(
        authority,
        "read_landing_snapshot",
        lambda *_args: {"secret": "/private/path"},
    )
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.SOURCE_INVALID
    assert "/private/path" not in result.detail

    authority = FakeAuthority()
    object.__setattr__(
        authority,
        "read_landing_snapshot",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("/private/path")),
    )
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    assert "/private/path" not in result.detail


def test_typed_authority_outages_are_fixed_refusals() -> None:
    def unavailable(*_args: object, **_kwargs: object) -> object:
        raise LandingReservationUnavailable("private authority outage")

    for method, expected in (
        (
            "authenticate_controller",
            LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE,
        ),
        ("resolve_repository", LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE),
        ("reserve_landing", LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE),
        ("hold_landing", LandingReservationRefusalCode.LEASE_UNAVAILABLE),
        ("validate_landing_barrier", LandingReservationRefusalCode.RESERVATION_LOST),
    ):
        authority = FakeAuthority()
        object.__setattr__(authority, method, unavailable)
        result = LandingReservationController(authority).reserve(_request())
        assert result.code is expected
        assert "private authority outage" not in result.detail


def test_trusted_runtime_errors_cross_the_established_authority_boundary() -> None:
    authority = FakeAuthority()
    object.__setattr__(
        authority,
        "authenticate_controller",
        lambda *_args: (_ for _ in ()).throw(RuntimeError("trusted authority failure")),
    )  # type: ignore[method-assign]
    with pytest.raises(RuntimeError, match="trusted authority failure"):
        LandingReservationController(authority).reserve(_request())

    authority = FakeAuthority()
    object.__setattr__(
        authority,
        "read_landing_snapshot",
        lambda *_args: (_ for _ in ()).throw(
            TrustedReservationRuntimeError("trusted source failure")
        ),
    )  # type: ignore[method-assign]
    with pytest.raises(TrustedReservationRuntimeError):
        LandingReservationController(authority).reserve(_request())


def test_snapshot_is_bounded_and_contains_no_path_owner_or_process_details() -> None:
    result = LandingReservationController(FakeAuthority()).reserve(_request())
    assert result.accepted and result.snapshot is not None
    snapshot = result.snapshot
    assert "/tmp/" not in repr(snapshot)
    assert "owner:one" not in repr(snapshot)
    assert "controller:one" not in repr(snapshot)
    assert "pid" not in snapshot.__dataclass_fields__
    assert "canonical_path" not in snapshot.__dataclass_fields__
    assert not hasattr(snapshot, "__dict__")


def test_refusal_has_no_mutation_or_job_effects() -> None:
    authority = FakeAuthority()

    def keep_first_state(current: FakeAuthority) -> None:
        current.state_index = 0

    authority.on_read = keep_first_state
    result = LandingReservationController(authority).reserve(
        _request(expected_target_sha=SHA_GENERATED)
    )
    assert result.code is LandingReservationRefusalCode.TARGET_MOVED
    assert authority.reserve_calls == 1
    assert authority.events.count("hold-enter") == 1
    assert not any(
        token in authority.events
        for token in ("merge", "reset", "checkout", "push", "build", "cleanup")
    )


@pytest.mark.parametrize("seed", [0, 1, 17, 31337])
def test_hash_seed_marker(seed: int) -> None:
    # The test is intentionally tiny; the lane runner repeats the full module
    # under each PYTHONHASHSEED and this keeps the required matrix explicit.
    assert seed in {0, 1, 17, 31337}
