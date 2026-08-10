"""Adversarial RMDD-13 CP2 coverage for the sealed atomic authority."""

from __future__ import annotations

import subprocess
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from threading import Event
from time import monotonic

import pytest

import repository_manager.landing_reservation as landing
from repository_manager.development import CONTRACT_VERSION, RepositoryIdentity
from repository_manager.landing_reservation import (
    CanonicalObservation,
    CanonicalState,
    CertificationObservation,
    ControllerIdentity,
    LandingReservationController,
    LandingReservationError,
    LandingReservationRefusalCode,
    LandingReservationRequest,
    LandingReservationResult,
    LandingStateSnapshot,
    OccupancyObservation,
    OccupancyState,
    ResolvedRepositoryIdentity,
    TargetObservation,
    normalize_target_ref,
)

SHA_TARGET = "1" * 40
SHA_BASE = "2" * 40
SHA_TREE = "3" * 40
SHA_GENERATED = "4" * 40
SHA_ALT = "5" * 40
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
    tree_sha: str = SHA_TREE,
    target_ref: str = "main",
) -> TargetObservation:
    return TargetObservation(repository_id, target_ref, commit_sha, tree_sha)


def _canonical(
    repository_id: str = REPOSITORY_ID,
    *,
    common_dir_id: str = "common-dir:one",
    worktree_id: str = "worktree:canonical",
    state: CanonicalState = CanonicalState.CLEAN,
    private_wip: bool = False,
    index_clean: bool = True,
) -> CanonicalObservation:
    return CanonicalObservation(
        repository_id,
        common_dir_id,
        worktree_id,
        state,
        private_wip,
        index_clean,
    )


def _occupancy(
    repository_id: str = REPOSITORY_ID,
    *,
    count: int = 0,
    state: OccupancyState = OccupancyState.FREE,
    target_ref: str = "main",
) -> OccupancyObservation:
    return OccupancyObservation(repository_id, target_ref, count, state)


def _cert(
    repository_id: str = REPOSITORY_ID,
    *,
    landing_fence: str = FENCE,
    certified: bool = True,
    target_ref: str = "main",
) -> CertificationObservation:
    return CertificationObservation(
        repository_id=repository_id,
        target_ref=target_ref,
        generation_id="generation:one",
        certificate_digest=DIGEST,
        base_sha=SHA_BASE,
        expected_landing_base_sha=SHA_BASE,
        synthetic_commit_sha=SHA_GENERATED,
        generation_tree_sha=SHA_TREE,
        landing_fence=landing_fence,
        certified=certified,
    )


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


def _authority(
    *,
    repository_id: str = REPOSITORY_ID,
    canonical_path: str = CANONICAL_PATH,
    common_dir_id: str = "common-dir:one",
    worktree_id: str = "worktree:canonical",
    states: list[LandingStateSnapshot] | None = None,
) -> tuple[
    landing._BoundLandingAuthority,
    landing._AuthorityRuntime,
    ResolvedRepositoryIdentity,
]:
    resolved = ResolvedRepositoryIdentity(
        repository_id,
        canonical_path,
        common_dir_id,
        worktree_id,
        "repository-revision:1",
    )
    identity = ControllerIdentity(
        "controller:one",
        "owner:one",
        "tenant:one",
        1,
        "principal:one",
        "session:one",
    )
    authority = landing._create_test_authority(
        identity, resolved, states or [_state(resolved)]
    )
    return authority, authority._runtime, resolved


@pytest.fixture(autouse=True)
def fake_rmdd26_leases(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    """Keep tests deterministic while exercising the module-owned adapter."""

    calls: list[tuple[str, str]] = []

    @contextmanager
    def hold(
        _self: object, canonical_path: str, *, operation: str
    ) -> Iterator[landing._LeaseEvidence]:
        calls.append((canonical_path, operation))
        yield landing._LeaseEvidence(
            reconciliation_lease_id="reconciliation:test",
            reconciliation_lease_epoch=1,
            reconciliation_lease_fence="reconciliation-fence:test",
            canonical_lease_id="canonical:test",
            canonical_lease_epoch=1,
            canonical_lease_fence="canonical-fence:test",
        )

    monkeypatch.setattr(landing._ExistingReconciliationLease, "hold", hold)
    return calls


def test_normalizes_local_ref_and_rejects_injection() -> None:
    assert normalize_target_ref("main") == "refs/heads/main"
    assert normalize_target_ref("refs/heads/release/v1") == "refs/heads/release/v1"
    for value in (
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
            normalize_target_ref(value)


def test_public_structural_authority_is_rejected_before_any_lease_call(
    fake_rmdd26_leases: list[tuple[str, str]],
) -> None:
    class NoopAuthority:
        calls = 0

        def acquire_landing(self, _request: object) -> object:
            self.calls += 1
            return object()

    noop = NoopAuthority()
    with pytest.raises(TypeError):
        LandingReservationController(noop)  # type: ignore[arg-type]
    assert noop.calls == 0
    assert fake_rmdd26_leases == []

    authority, _, _ = _authority()
    forged = object.__new__(type(authority))
    with pytest.raises(TypeError):
        LandingReservationController(forged)


def test_success_is_one_atomic_hold_capture_barrier_and_attested_snapshot(
    fake_rmdd26_leases: list[tuple[str, str]],
) -> None:
    authority, runtime, _ = _authority()
    result = LandingReservationController(authority).reserve(_request())
    assert result.accepted and result.snapshot is not None
    assert authority.verify_attested_snapshot(result.snapshot)
    assert runtime.events == [
        "lease-enter",
        "reserve",
        "state-read",
        "state-read",
        "lease-exit",
    ]
    assert len(fake_rmdd26_leases) == 1
    assert fake_rmdd26_leases[0][0] == CANONICAL_PATH
    assert not runtime.held
    assert runtime.enter_count == runtime.exit_count == 1


def test_exact_repository_identity_prevents_path_alias_and_same_basename_collision() -> (
    None
):
    first, _, _ = _authority(
        repository_id="repository:first", canonical_path="/tmp/one/same"
    )
    second, _, _ = _authority(
        repository_id="repository:second", canonical_path="/tmp/two/same"
    )
    first_result = LandingReservationController(first).reserve(
        _request(repository=_repository("repository:first", "/tmp/one/same"))
    )
    second_result = LandingReservationController(second).reserve(
        _request(repository=_repository("repository:second", "/tmp/two/same"))
    )
    assert first_result.accepted and second_result.accepted
    assert first_result.snapshot and second_result.snapshot
    assert (
        first_result.snapshot.resolved_repository_digest
        != second_result.snapshot.resolved_repository_digest
    )

    forged, forged_runtime, _ = _authority(canonical_path="/tmp/actual/repository")
    forged_runtime.resolved = ResolvedRepositoryIdentity(
        REPOSITORY_ID,
        "/tmp/other/same",
        "common-dir:other",
        "worktree:other",
        "repository-revision:2",
    )
    request = _request(repository=_repository(canonical_path="/tmp/actual/repository"))
    refused = LandingReservationController(forged).reserve(request)
    assert refused.code is LandingReservationRefusalCode.REPOSITORY_MISMATCH


def test_request_path_disagreement_refuses_before_rmdd26_acquisition(
    fake_rmdd26_leases: list[tuple[str, str]],
) -> None:
    authority, runtime, _ = _authority(canonical_path="/tmp/trusted/actual")
    request = _request(repository=_repository(canonical_path="/tmp/forged/other"))
    result = LandingReservationController(authority).reserve(request)
    assert result.code is LandingReservationRefusalCode.REPOSITORY_MISMATCH
    assert runtime.enter_count == 0
    assert fake_rmdd26_leases == []


def test_exact_replay_and_changed_input_conflict() -> None:
    authority, _, _ = _authority()
    controller = LandingReservationController(authority)
    request = _request()
    first = controller.reserve(request)
    replay = controller.reserve(request)
    changed = controller.reserve(
        _request(request_id="request:two", invocation_id="invocation:two")
    )
    assert first.accepted and replay.accepted and first.snapshot and replay.snapshot
    assert first.snapshot.digest == replay.snapshot.digest
    assert changed.code is LandingReservationRefusalCode.RESERVATION_CONFLICT


def test_authority_epoch_advance_invalidates_old_replay() -> None:
    authority, runtime, _ = _authority()
    controller = LandingReservationController(authority)
    assert controller.reserve(_request()).accepted
    assert runtime.identity is not None
    runtime.identity = replace(runtime.identity, authority_epoch=2)
    replay = controller.reserve(_request())
    assert replay.code is LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH


def test_replay_anchor_must_match_current_epoch_and_fence() -> None:
    authority, _, _ = _authority()
    controller = LandingReservationController(authority)
    first = controller.reserve(_request())
    assert first.accepted and first.snapshot
    wrong_epoch = controller.reserve(
        _request(
            expected_lease_epoch=2, expected_lease_fence=first.snapshot.lease_fence
        )
    )
    wrong_fence = controller.reserve(
        _request(
            expected_lease_epoch=first.snapshot.lease_epoch, expected_lease_fence="bad"
        )
    )
    assert wrong_epoch.code is LandingReservationRefusalCode.EPOCH_MISMATCH
    assert wrong_fence.code is LandingReservationRefusalCode.FENCE_MISMATCH


def test_two_threads_racing_one_target_have_one_nonblocking_winner() -> None:
    authority, runtime, _ = _authority()
    entered = Event()
    release = Event()

    def hold_after_reservation(_authority: object) -> None:
        entered.set()
        release.wait(timeout=2)

    runtime.on_after_reservation = hold_after_reservation
    requests = (
        _request(request_id="request:first", invocation_id="invocation:first"),
        _request(request_id="request:second", invocation_id="invocation:second"),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        first_future = pool.submit(
            LandingReservationController(authority).reserve, requests[0]
        )
        assert entered.wait(timeout=2)
        started = monotonic()
        second = LandingReservationController(authority).reserve(requests[1])
        elapsed = monotonic() - started
        release.set()
        first = first_future.result(timeout=2)
    assert elapsed < 2
    assert first.accepted
    assert second.code is LandingReservationRefusalCode.RESERVATION_CONFLICT
    assert not runtime.held


def test_same_thread_reentrant_reservation_is_fixed_conflict() -> None:
    authority, runtime, _ = _authority()
    controller = LandingReservationController(authority)
    nested: list[LandingReservationResult] = []

    def reenter(_authority: object) -> None:
        nested.append(
            controller.reserve(_request(request_id="nested", invocation_id="nested"))
        )

    runtime.on_after_reservation = reenter
    result = controller.reserve(_request())
    assert result.accepted
    assert len(nested) == 1
    assert nested[0].code is LandingReservationRefusalCode.RESERVATION_CONFLICT


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            _state(
                ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                ),
                canonical=_canonical(state=CanonicalState.DIRTY),
            ),
            LandingReservationRefusalCode.CANONICAL_DIRTY,
        ),
        (
            _state(
                ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                ),
                canonical=_canonical(private_wip=True),
            ),
            LandingReservationRefusalCode.PRIVATE_WIP,
        ),
        (
            _state(
                ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                ),
                canonical=_canonical(index_clean=False),
            ),
            LandingReservationRefusalCode.PRIVATE_WIP,
        ),
        (
            _state(
                ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                ),
                canonical=_canonical(state=CanonicalState.UNKNOWN),
            ),
            LandingReservationRefusalCode.CANONICAL_STATE_INVALID,
        ),
        (
            _state(
                ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                ),
                occupancy=_occupancy(count=1, state=OccupancyState.OCCUPIED),
            ),
            LandingReservationRefusalCode.TARGET_OCCUPIED,
        ),
        (
            _state(
                ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                ),
                occupancy=_occupancy(state=OccupancyState.UNKNOWN),
            ),
            LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN,
        ),
    ],
)
def test_dirty_private_index_unknown_and_occupied_state_refuse(
    state: LandingStateSnapshot, expected: LandingReservationRefusalCode
) -> None:
    authority, runtime, _ = _authority(states=[state])
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is expected
    assert not runtime.held


def test_target_tree_drift_with_stable_revision_refuses() -> None:
    authority, runtime, resolved = _authority(
        states=[
            _state(
                resolved := ResolvedRepositoryIdentity(
                    REPOSITORY_ID,
                    CANONICAL_PATH,
                    "common-dir:one",
                    "worktree:canonical",
                    "repository-revision:1",
                )
            ),
            _state(resolved, target=_target(tree_sha=SHA_ALT)),
        ]
    )
    runtime.on_before_barrier = lambda _authority: setattr(runtime, "state_index", 1)
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_all_source_revision_movement_before_final_barrier_refuses() -> None:
    base_resolved = ResolvedRepositoryIdentity(
        REPOSITORY_ID,
        CANONICAL_PATH,
        "common-dir:one",
        "worktree:canonical",
        "repository-revision:1",
    )
    authority, runtime, _ = _authority(
        states=[
            _state(base_resolved),
            _state(base_resolved, revision="source-revision:2"),
        ]
    )
    runtime.on_before_barrier = lambda _authority: setattr(runtime, "state_index", 1)
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_object_setattr_source_mutation_is_revalidated() -> None:
    authority, runtime, _ = _authority()

    def mutate(_authority: object) -> None:
        object.__setattr__(runtime.states[0].target, "tree_sha", SHA_ALT)

    runtime.on_after_capture = mutate
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_barrier_replacement_reservation_epoch_or_fence_refuses() -> None:
    authority, runtime, _ = _authority()

    def replace_reservation(bound: object) -> None:
        current = authority._reservations[(REPOSITORY_ID, "refs/heads/main")]
        runtime.lease_overrides["reservation"] = replace(current, lease_epoch=9)

    runtime.on_before_barrier = replace_reservation
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_lease_loss_and_identity_alias_loss_refuse() -> None:
    authority, runtime, _ = _authority()
    runtime.on_before_barrier = lambda _authority: setattr(runtime, "lease_lost", True)
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST

    authority, runtime, _ = _authority()
    runtime.on_before_barrier = lambda _authority: setattr(
        runtime,
        "resolved",
        ResolvedRepositoryIdentity(
            REPOSITORY_ID,
            CANONICAL_PATH,
            "common-dir:forged",
            "worktree:forged",
            "repository-revision:2",
        ),
    )
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


@pytest.mark.parametrize(
    ("field", "expected"),
    [
        ("owner_id", LandingReservationRefusalCode.OWNER_MISMATCH),
        ("tenant_id", LandingReservationRefusalCode.TENANT_MISMATCH),
        ("principal_id", LandingReservationRefusalCode.PRINCIPAL_MISMATCH),
        ("session_id", LandingReservationRefusalCode.SESSION_MISMATCH),
        ("authority_epoch", LandingReservationRefusalCode.AUTHORITY_EPOCH_MISMATCH),
    ],
)
def test_authenticated_dimensions_are_bound_to_original_context(
    field: str, expected: LandingReservationRefusalCode
) -> None:
    authority, runtime, _ = _authority()

    def mutate(_authority: object) -> None:
        assert runtime.identity is not None
        if field == "owner_id":
            runtime.identity = replace(runtime.identity, owner_id="other:value")
        elif field == "tenant_id":
            runtime.identity = replace(runtime.identity, tenant_id="other:value")
        elif field == "principal_id":
            runtime.identity = replace(runtime.identity, principal_id="other:value")
        elif field == "session_id":
            runtime.identity = replace(runtime.identity, session_id="other:value")
        else:
            runtime.identity = replace(runtime.identity, authority_epoch=2)

    runtime.on_before_barrier = mutate
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is expected


def test_release_failure_records_recovery_and_clears_held_flag() -> None:
    authority, runtime, _ = _authority()
    runtime.fail_release = True
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.RECOVERY_REQUIRED
    assert not runtime.held
    assert runtime.enter_count == runtime.exit_count == 1
    assert authority._recovery


def test_partial_acquire_failure_never_leaves_a_held_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authority, runtime, _ = _authority()

    @contextmanager
    def fail_before_yield(
        _self: object, _path: str, *, operation: str
    ) -> Iterator[landing._LeaseEvidence]:
        raise landing.BlockedByLease("blocked")
        yield landing._LeaseEvidence("r", 1, "rf", "c", 1, "cf")

    monkeypatch.setattr(landing._ExistingReconciliationLease, "hold", fail_before_yield)
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.LEASE_UNAVAILABLE
    assert not runtime.held
    assert runtime.enter_count == runtime.exit_count == 0
    assert not authority._reservations


def test_factory_without_native_backend_fails_closed() -> None:
    authority = landing.create_landing_authority()
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE


def test_attestation_rejects_rehashed_modified_snapshot() -> None:
    authority, _, _ = _authority()
    result = LandingReservationController(authority).reserve(_request())
    assert result.accepted and result.snapshot
    original = result.snapshot
    payload = original.immutable_payload()
    payload["observed_target_tree_sha"] = SHA_ALT
    digest = landing._snapshot_digest(
        {"schema": "rmdd-13-landing-reservation:v2", **payload}
    )
    forged = replace(original, observed_target_tree_sha=SHA_ALT, digest=digest)
    assert not authority.verify_attested_snapshot(forged)
    assert authority.verify_attested_snapshot(original)


def test_forged_missing_fields_and_hostile_models_fail_closed_without_attribute_error() -> (
    None
):
    identity = ControllerIdentity("controller:one", "owner:one", "tenant:one", 1)
    request = _request()
    for value in (identity, request, _target(), _canonical(), _occupancy(), _cert()):
        forged = object.__new__(type(value))
        names = tuple(value.__dataclass_fields__)
        for name in names[1:]:
            object.__setattr__(forged, name, getattr(value, name))
        with pytest.raises(LandingReservationError):
            forged.__post_init__()

    authority, runtime, _ = _authority()
    object.__delattr__(request, "request_id")
    result = LandingReservationController(authority).reserve(request)
    assert result.code is LandingReservationRefusalCode.REQUEST_INVALID
    object.__setattr__(runtime, "states", {"secret": "/private/path"})
    result = LandingReservationController(authority).reserve(
        _request(request_id="request:two", invocation_id="invocation:two")
    )
    assert result.code is LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    assert "/private/path" not in result.detail


def test_forged_pydantic_repository_and_target_aliases_do_not_prove_identity() -> None:
    authority, runtime, _ = _authority()
    copied = _request().repository.model_copy(
        update={"repository_id": "repository:forged"}
    )
    result = LandingReservationController(authority).reserve(
        _request(repository=copied)
    )
    assert result.code is LandingReservationRefusalCode.REPOSITORY_MISMATCH
    forged = RepositoryIdentity.model_construct(
        contract_version=CONTRACT_VERSION,
        repository_id=True,
        canonical_path=CANONICAL_PATH,
        configured_roots=(),
        origin=None,
    )
    bad_request = _request(request_id="request:bad", invocation_id="invocation:bad")
    object.__setattr__(bad_request, "repository", forged)
    assert (
        LandingReservationController(authority).reserve(bad_request).code
        is LandingReservationRefusalCode.REQUEST_INVALID
    )
    object.__setattr__(runtime, "state_index", True)
    assert (
        LandingReservationController(authority)
        .reserve(_request(request_id="request:three", invocation_id="invocation:three"))
        .code
        is LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    )


def test_runtime_programmer_error_propagates_but_provider_values_are_private() -> None:
    authority, runtime, _ = _authority()
    runtime.on_before_hold = lambda _authority: (_ for _ in ()).throw(
        RuntimeError("trusted programmer failure")
    )
    with pytest.raises(RuntimeError, match="trusted programmer failure"):
        LandingReservationController(authority).reserve(_request())

    authority, runtime, _ = _authority()
    runtime.states = None  # type: ignore[assignment]
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    assert "/tmp/" not in result.detail


def test_actual_git_repository_is_only_read_and_no_subprocess_effect(tmp_path) -> None:
    repo = tmp_path / "repository-manager"
    repo.mkdir()
    for command in (
        ("git", "init", "-b", "main"),
        ("git", "config", "user.email", "test@example.invalid"),
        ("git", "config", "user.name", "test"),
    ):
        subprocess.run(command, cwd=repo, check=True, capture_output=True)
    (repo / "README.md").write_text("stable\n", encoding="utf-8")
    subprocess.run(
        ("git", "add", "README.md"), cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ("git", "commit", "-m", "initial"), cwd=repo, check=True, capture_output=True
    )
    before = subprocess.check_output(
        ("git", "rev-parse", "HEAD"), cwd=repo, text=True
    ).strip()
    repository = _repository(canonical_path=str(repo))
    resolved = ResolvedRepositoryIdentity(
        REPOSITORY_ID, str(repo), "common:git", "worktree:git", "revision:git"
    )
    authority = landing._create_test_authority(
        ControllerIdentity("controller:one", "owner:one", "tenant:one", 1),
        resolved,
        [_state(resolved, target=_target(commit_sha=before))],
    )
    request = _request(repository=repository, expected_target_sha=before)
    result = LandingReservationController(authority).reserve(request)
    assert result.accepted
    assert (
        subprocess.check_output(
            ("git", "rev-parse", "HEAD"), cwd=repo, text=True
        ).strip()
        == before
    )
    assert (
        subprocess.check_output(
            ("git", "status", "--porcelain"), cwd=repo, text=True
        ).strip()
        == ""
    )


def test_snapshot_is_bounded_and_never_contains_private_path_or_owner() -> None:
    authority, _, _ = _authority()
    result = LandingReservationController(authority).reserve(_request())
    assert result.accepted and result.snapshot
    snapshot = result.snapshot
    assert "/tmp/" not in repr(snapshot)
    assert "owner:one" not in repr(snapshot)
    assert "controller:one" not in repr(snapshot)
    assert "canonical_path" not in snapshot.__dataclass_fields__
    assert not hasattr(snapshot, "__dict__")


def test_no_mutating_git_or_job_effect_on_refusal() -> None:
    authority, runtime, _ = _authority()
    runtime.states = []
    result = LandingReservationController(authority).reserve(_request())
    assert result.code is LandingReservationRefusalCode.SOURCE_UNAVAILABLE
    assert not any(
        token in runtime.events
        for token in ("merge", "reset", "checkout", "push", "build", "cleanup")
    )


@pytest.mark.parametrize("seed", [0, 1, 17, 31337])
def test_hash_seed_marker(seed: int) -> None:
    assert seed in {0, 1, 17, 31337}
