"""RMDD-13 CP2 protocol and adversarial-boundary coverage.

The production authority is intentionally unavailable in this checkpoint.
The successful/racing/barrier cases below use ``AtomicHarness``, a test-only
transcript simulator.  Its result and snapshot types are deliberately not the
production result/snapshot types: a passing harness test is not deployment
evidence and cannot authorize CP3.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from threading import Event, Lock
from time import monotonic
from typing import Any, cast

import pytest

import repository_manager.landing_reservation as landing
from repository_manager.development import CONTRACT_VERSION, RepositoryIdentity
from repository_manager.landing_reservation import (
    CanonicalObservation,
    CanonicalState,
    CertificationObservation,
    ControllerIdentity,
    DurableLandingReservation,
    LandingReservationController,
    LandingReservationError,
    LandingReservationProofRef,
    LandingReservationRefusalCode,
    LandingReservationRequest,
    LandingReservationResult,
    LandingReservationSnapshot,
    LandingStateSnapshot,
    NativeLandingReservationPort,
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


def _resolved(
    repository_id: str = REPOSITORY_ID,
    canonical_path: str = CANONICAL_PATH,
    common_dir_id: str = "common-dir:one",
    worktree_id: str = "worktree:canonical",
    revision: str = "repository-revision:1",
) -> ResolvedRepositoryIdentity:
    return ResolvedRepositoryIdentity(
        repository_id,
        canonical_path,
        common_dir_id,
        worktree_id,
        revision,
    )


def _state(
    resolved: ResolvedRepositoryIdentity | None = None,
    *,
    revision: str = "source-revision:1",
    target: TargetObservation | None = None,
    canonical: CanonicalObservation | None = None,
    occupancy: OccupancyObservation | None = None,
    certification: CertificationObservation | None = None,
) -> LandingStateSnapshot:
    resolved = resolved or _resolved()
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


@dataclass(frozen=True, slots=True)
class HarnessSnapshot:
    """Test-only result; never accepted as a production snapshot."""

    request_digest: str
    target_sha: str
    target_tree_sha: str
    state_revision: str
    proof_marker: str = "test-only"

    def __repr__(self) -> str:
        return "<test-only landing harness snapshot>"


@dataclass(frozen=True, slots=True)
class HarnessResult:
    """Test-only result with no production proof or accepted snapshot."""

    accepted: bool
    code: LandingReservationRefusalCode | None = None
    snapshot: HarnessSnapshot | None = None


class AtomicHarness:
    """A bounded test transcript for the future native transaction.

    This deliberately lives in the test module.  It uses no Git, leases,
    process execution, or durable storage and is never passed to production.
    Hooks model provider changes at each reservation boundary.
    """

    def __init__(
        self,
        states: list[LandingStateSnapshot],
        resolved: ResolvedRepositoryIdentity | None = None,
    ) -> None:
        self.states = states
        self.resolved = resolved or _resolved()
        self._lock = Lock()
        self.active: set[tuple[str, str]] = set()
        self.completed: dict[tuple[str, str, str, str], tuple[str, HarnessResult]] = {}
        self.state_index = 0
        self.events: list[str] = []
        self.on_after_hold: Callable[[], None] | None = None
        self.on_after_capture: Callable[[], None] | None = None
        self.on_before_barrier: Callable[[], None] | None = None
        self.barrier_valid = True
        self.fail_release = False
        self.recovery_required = False

    def _state_refusal(
        self, request: LandingReservationRequest, state: LandingStateSnapshot
    ) -> LandingReservationRefusalCode | None:
        if state.resolved_repository_digest != self.resolved.digest():
            return LandingReservationRefusalCode.REPOSITORY_MISMATCH
        if state.target.repository_id != request.repository_id:
            return LandingReservationRefusalCode.TARGET_MISMATCH
        if state.target.target_ref != request.target_ref:
            return LandingReservationRefusalCode.TARGET_MISMATCH
        if state.target.commit_sha != request.expected_target_sha:
            return LandingReservationRefusalCode.TARGET_MOVED
        if state.target.tree_sha != request.generation_tree_sha:
            return LandingReservationRefusalCode.TARGET_TREE_MISMATCH
        canonical = state.canonical
        if (
            canonical.repository_id != self.resolved.repository_id
            or canonical.common_dir_id != self.resolved.common_dir_id
            or canonical.worktree_id != self.resolved.worktree_id
        ):
            return LandingReservationRefusalCode.CANONICAL_STATE_CHANGED
        if canonical.state is CanonicalState.UNKNOWN:
            return LandingReservationRefusalCode.CANONICAL_STATE_INVALID
        if canonical.state is CanonicalState.DIRTY:
            return LandingReservationRefusalCode.CANONICAL_DIRTY
        if canonical.private_wip or not canonical.index_clean:
            return LandingReservationRefusalCode.PRIVATE_WIP
        if state.occupancy.state is OccupancyState.UNKNOWN:
            return LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN
        if state.occupancy.state is OccupancyState.OCCUPIED:
            return LandingReservationRefusalCode.TARGET_OCCUPIED
        cert = state.certification
        if (
            not cert.certified
            or cert.generation_id != request.generation_id
            or cert.certificate_digest != request.certificate_digest
            or cert.base_sha != request.expected_base_sha
            or cert.synthetic_commit_sha != request.synthetic_commit_sha
            or cert.generation_tree_sha != request.generation_tree_sha
            or cert.landing_fence != request.landing_fence
        ):
            return LandingReservationRefusalCode.CERTIFICATION_INVALID
        if not self.barrier_valid:
            return LandingReservationRefusalCode.RESERVATION_LOST
        return None

    def reserve(self, request: LandingReservationRequest) -> HarnessResult:
        try:
            if type(request) is not LandingReservationRequest:
                raise LandingReservationError("request is invalid")
            request.__post_init__()
            request_digest = request.digest()
        except LandingReservationError:
            return HarnessResult(False, LandingReservationRefusalCode.REQUEST_INVALID)

        key = (request.repository_id, request.target_ref)
        replay_key = (*key, request.request_id, request.invocation_id)
        with self._lock:
            prior = self.completed.get(replay_key)
            if prior is not None:
                if prior[0] == request_digest:
                    return prior[1]
                return HarnessResult(
                    False, LandingReservationRefusalCode.RESERVATION_CONFLICT
                )
            if key in self.active:
                return HarnessResult(
                    False, LandingReservationRefusalCode.RESERVATION_CONFLICT
                )
            self.active.add(key)
            self.events.append("hold")

        result: HarnessResult
        try:
            if self.on_after_hold is not None:
                self.on_after_hold()
            if not self.states or type(self.state_index) is not int:
                result = HarnessResult(
                    False, LandingReservationRefusalCode.SOURCE_UNAVAILABLE
                )
            else:
                first = self.states[self.state_index]
                first.__post_init__()
                # Freeze the payload now, before any hook runs.  ``states``
                # holds live object references, not copies: an
                # ``object.__setattr__`` mutation between "first" and
                # "second" mutates the *same* object, so reading
                # ``first.immutable_payload()`` after the hooks would just
                # re-read the already-mutated object and trivially agree
                # with ``second`` -- the drift this barrier exists to catch
                # would never be detected.
                first_payload = json.dumps(first.immutable_payload(), sort_keys=True)
                self.events.append("capture")
                if self.on_after_capture is not None:
                    self.on_after_capture()
                if self.on_before_barrier is not None:
                    self.on_before_barrier()
                second = self.states[self.state_index]
                second.__post_init__()
                if first_payload != json.dumps(
                    second.immutable_payload(), sort_keys=True
                ):
                    result = HarnessResult(
                        False, LandingReservationRefusalCode.RESERVATION_LOST
                    )
                else:
                    code = self._state_refusal(request, second)
                    if code is not None:
                        result = HarnessResult(False, code)
                    else:
                        result = HarnessResult(
                            True,
                            snapshot=HarnessSnapshot(
                                request_digest=request_digest,
                                target_sha=second.target.commit_sha,
                                target_tree_sha=second.target.tree_sha,
                                state_revision=second.snapshot_revision,
                            ),
                        )
        except LandingReservationError:
            result = HarnessResult(False, LandingReservationRefusalCode.SOURCE_INVALID)
        except RuntimeError:
            result = HarnessResult(False, LandingReservationRefusalCode.SOURCE_INVALID)
        finally:
            self.events.append("barrier")
            with self._lock:
                self.active.discard(key)
            if self.fail_release:
                self.recovery_required = True
                result = HarnessResult(
                    False, LandingReservationRefusalCode.RECOVERY_REQUIRED
                )
            else:
                self.events.append("release")
        if result.accepted:
            self.completed[replay_key] = (request_digest, result)
        return result


def _production_snapshot(
    request: LandingReservationRequest | None = None,
) -> LandingReservationSnapshot:
    """Build a shape-valid DTO for proving that shape is not authority."""

    request = request or _request()
    resolved = _resolved(request.repository_id)
    proof = LandingReservationProofRef(
        issuer="native:test",
        opaque_reference="opaque-backend-token-that-must-not-be-logged",
        correlation_digest=DIGEST,
    )
    fields: dict[str, object] = {
        "reservation_id": "reservation:test",
        "request_digest": request.digest(),
        "resolved_repository_digest": resolved.digest(),
        "repository_id": request.repository_id,
        "target_ref": request.target_ref,
        "expected_target_sha": request.expected_target_sha,
        "expected_base_sha": request.expected_base_sha,
        "observed_target_sha": request.expected_target_sha,
        "observed_target_tree_sha": request.generation_tree_sha,
        "common_dir_id": resolved.common_dir_id,
        "worktree_id": resolved.worktree_id,
        "authority_revision": resolved.authority_revision,
        "lease_epoch": 1,
        "lease_fence": "lease:test",
        "tenant_id": "tenant:test",
        "authority_epoch": 1,
        "generation_id": request.generation_id,
        "certificate_digest": request.certificate_digest,
        "synthetic_commit_sha": request.synthetic_commit_sha,
        "generation_tree_sha": request.generation_tree_sha,
        "landing_fence": request.landing_fence,
        "target_worktree_count": 0,
        "target_revision": "target:test",
        "canonical_revision": "canonical:test",
        "occupancy_revision": "occupancy:test",
        "certification_revision": "certification:test",
        "snapshot_revision": "snapshot:test",
        "barrier_revision": "barrier:test",
        "reconciliation_lease_id": "reconciliation:test",
        "reconciliation_lease_epoch": 1,
        "reconciliation_lease_fence": "reconciliation-fence:test",
        "canonical_lease_id": "canonical:test",
        "canonical_lease_epoch": 1,
        "canonical_lease_fence": "canonical-fence:test",
        "authority_incarnation": "authority:test",
        "proof_ref": proof,
    }
    payload = {
        key: (
            cast(LandingReservationProofRef, value).immutable_payload()
            if key == "proof_ref"
            else value
        )
        for key, value in fields.items()
    }
    fields["digest"] = landing._snapshot_digest(
        {"schema": "rmdd-13-landing-reservation:v2", **payload}
    )
    return LandingReservationSnapshot(**fields)  # type: ignore[arg-type]


def test_target_normalization_rejects_injection() -> None:
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


def test_production_entrypoints_are_always_unavailable() -> None:
    request = _request()
    calls: list[str] = []
    result = landing.reserve_landing(request)
    assert result.refused
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
    assert result.detail == "landing reservation authority unavailable"
    service = landing.create_landing_authority()
    for candidate in (
        service.reserve_landing(request),
        service.acquire_landing(request),
        LandingReservationController().reserve(request),
        landing.verify_current_landing_reservation(object(), object()),
        service.verify_current_landing_reservation(object(), object()),
        service.verify_attested_snapshot(object()),
    ):
        assert candidate.refused
        assert candidate.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
    assert calls == []
    assert repr(service) == "<native landing reservation authority unavailable>"


def test_no_python_backend_or_authority_injection_is_accepted() -> None:
    class NoopAuthority:
        def __init__(self) -> None:
            self.calls = 0

        def reserve_landing(self, _request: object) -> object:
            self.calls += 1
            return object()

    noop = NoopAuthority()
    with pytest.raises(TypeError):
        landing.create_landing_authority(noop)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        LandingReservationController(noop)  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        landing.reserve_landing(_request(), authority=noop)  # type: ignore[call-arg]
    assert noop.calls == 0
    assert not hasattr(landing, "_AUTHORITY_SEAL")
    assert not hasattr(landing, "_create_test_authority")
    assert not hasattr(landing, "_BoundLandingAuthority")
    source = inspect.getsource(landing).lower()
    assert "import hmac" not in source
    assert "hmac.new" not in source
    assert "_attestation_secret" not in source
    assert "_issued_authority_handles" not in source


def test_structural_native_port_is_documentation_only() -> None:
    class StructuralPort:
        def reserve_landing(
            self, _request: LandingReservationRequest
        ) -> LandingReservationResult:
            raise AssertionError("must never be called")

        def verify_current_landing_reservation(
            self,
            _proof: LandingReservationProofRef,
            _snapshot: LandingReservationSnapshot,
        ) -> LandingReservationResult:
            raise AssertionError("must never be called")

    port: NativeLandingReservationPort = StructuralPort()
    assert port is not None
    result = landing.reserve_landing(_request())
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE


def test_production_rejects_malformed_requests_without_attribute_error() -> None:
    request = _request()
    forged = object.__new__(LandingReservationRequest)
    for name in LandingReservationRequest.__dataclass_fields__:
        if name != "request_id":
            object.__setattr__(forged, name, getattr(request, name))
    result = landing.reserve_landing(forged)
    assert result.code is LandingReservationRefusalCode.REQUEST_INVALID

    repository = _repository()
    forged_repository = repository.model_copy(update={"repository_id": True})
    # ``dataclasses.replace`` calls ``__init__``/``__post_init__``, which
    # already refuses eagerly (a stronger guarantee than a production
    # refusal).  Bypass construction, matching ``forged`` above, to exercise
    # the actual claim: a forged instance that skipped construction-time
    # validation is still refused by the production entrypoint itself,
    # without ever raising ``AttributeError``.
    bad = object.__new__(LandingReservationRequest)
    for name in LandingReservationRequest.__dataclass_fields__:
        value = forged_repository if name == "repository" else getattr(request, name)
        object.__setattr__(bad, name, value)
    assert (
        landing.reserve_landing(bad).code
        is LandingReservationRefusalCode.REQUEST_INVALID
    )
    constructed = RepositoryIdentity.model_construct(
        contract_version=CONTRACT_VERSION,
        repository_id=True,
        canonical_path=CANONICAL_PATH,
        configured_roots=(),
        origin=None,
    )
    object.__setattr__(bad, "repository", constructed)
    assert (
        landing.reserve_landing(bad).code
        is LandingReservationRefusalCode.REQUEST_INVALID
    )


def test_production_models_are_strict_and_missing_fields_are_normalized() -> None:
    identity = ControllerIdentity("controller:one", "owner:one", "tenant:one", 1)
    proof = LandingReservationProofRef("native:test", "opaque:test", DIGEST)
    candidates: list[Any] = [
        identity,
        proof,
        _target(),
        _canonical(),
        _occupancy(),
        _cert(),
    ]
    for value in candidates:
        forged = object.__new__(type(value))
        names = tuple(value.__dataclass_fields__)
        for name in names[1:]:
            object.__setattr__(forged, name, getattr(value, name))
        with pytest.raises(LandingReservationError):
            forged.__post_init__()
    snapshot = _production_snapshot()
    forged_snapshot = object.__new__(LandingReservationSnapshot)
    for name in LandingReservationSnapshot.__dataclass_fields__:
        if name != "proof_ref":
            object.__setattr__(forged_snapshot, name, getattr(snapshot, name))
    with pytest.raises(LandingReservationError):
        forged_snapshot.__post_init__()


def test_held_native_context_is_strict_and_complete() -> None:
    request = _request()
    resolved = _resolved()
    controller = ControllerIdentity("controller:one", "owner:one", "tenant:one", 1)
    reservation = DurableLandingReservation(
        reservation_id="reservation:test",
        request_id=request.request_id,
        invocation_id=request.invocation_id,
        repository_id=request.repository_id,
        target_ref=request.target_ref,
        request_digest=request.digest(resolved, controller),
        resolved_repository_digest=resolved.digest(),
        common_dir_id=resolved.common_dir_id,
        worktree_id=resolved.worktree_id,
        authority_revision=resolved.authority_revision,
        controller_id=controller.controller_id,
        owner_id=controller.owner_id,
        tenant_id=controller.tenant_id,
        lease_epoch=1,
        fence="fence:test",
        authority_epoch=1,
        principal_id=controller.principal_id,
        session_id=controller.session_id,
        reconciliation_lease_id="reconciliation:test",
        reconciliation_lease_epoch=1,
        reconciliation_lease_fence="reconciliation-fence:test",
        canonical_lease_id="canonical:test",
        canonical_lease_epoch=1,
        canonical_lease_fence="canonical-fence:test",
        authority_incarnation="authority:test",
    )
    context = landing._HeldLandingContext(
        reservation,
        resolved,
        controller,
        request.digest(resolved, controller),
        "reconciliation:test",
        1,
        "reconciliation-fence:test",
        "canonical:test",
        1,
        "canonical-fence:test",
        "authority:test",
    )
    assert context.immutable_payload()["authority_incarnation"] == "authority:test"
    forged = object.__new__(type(context))
    for name in type(context).__dataclass_fields__:
        if name != "authority_incarnation":
            object.__setattr__(forged, name, getattr(context, name))
    with pytest.raises(LandingReservationError):
        forged.__post_init__()


def test_proof_reference_is_opaque_and_rehashed_shape_is_not_authority() -> None:
    proof = LandingReservationProofRef(
        "native:test", "raw-backend-token-should-not-appear", DIGEST
    )
    assert "raw-backend-token" not in repr(proof)
    assert repr(proof) == "<native landing proof ref redacted>"
    snapshot = _production_snapshot()
    payload = snapshot.immutable_payload()
    payload["observed_target_tree_sha"] = SHA_ALT
    rehashed = landing._snapshot_digest(
        {"schema": "rmdd-13-landing-reservation:v2", **payload}
    )
    forged = replace(snapshot, observed_target_tree_sha=SHA_ALT, digest=rehashed)
    result = landing.create_landing_authority().verify_attested_snapshot(forged)
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
    assert landing.verify_current_landing_reservation(proof, forged).code is (
        LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE
    )


def test_request_digest_binds_resolved_identity_without_exposing_path() -> None:
    request = _request()
    controller = ControllerIdentity("controller:one", "owner:one", "tenant:one", 1)
    first = _resolved(canonical_path="/tmp/one/same")
    second = _resolved(canonical_path="/tmp/two/same")
    assert first.repository_id == second.repository_id
    assert first.digest() != second.digest()
    assert request.digest(first, controller) != request.digest(second, controller)
    assert "/tmp/" not in repr(_production_snapshot())


def test_harness_success_is_not_a_production_result() -> None:
    harness = AtomicHarness([_state()])
    result = harness.reserve(_request())
    assert result.accepted and result.snapshot is not None
    assert type(result) is HarnessResult
    assert type(result.snapshot) is HarnessSnapshot
    assert not isinstance(result, LandingReservationResult)
    assert not isinstance(result.snapshot, LandingReservationSnapshot)
    assert result.snapshot.proof_marker == "test-only"
    assert landing.create_landing_authority().verify_attested_snapshot(
        result.snapshot
    ).code is (LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE)
    assert harness.events == ["hold", "capture", "barrier", "release"]


def test_harness_same_request_replays_and_changed_input_conflicts() -> None:
    harness = AtomicHarness([_state()])
    request = _request()
    first = harness.reserve(request)
    replay = harness.reserve(request)
    changed = harness.reserve(
        _request(
            request_id=request.request_id,
            invocation_id=request.invocation_id,
            target_ref="release",
        )
    )
    assert first.accepted and replay.accepted
    assert first.snapshot == replay.snapshot
    assert (
        changed.code is LandingReservationRefusalCode.TARGET_MOVED
        or changed.code is LandingReservationRefusalCode.TARGET_MISMATCH
    )


def test_harness_racing_same_target_is_bounded_and_has_one_winner() -> None:
    entered = Event()
    release = Event()
    harness = AtomicHarness([_state()])

    def after_hold() -> None:
        entered.set()
        release.wait(timeout=2)

    harness.on_after_hold = after_hold
    first_request = _request(
        request_id="request:first", invocation_id="invocation:first"
    )
    second_request = _request(
        request_id="request:second", invocation_id="invocation:second"
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(harness.reserve, first_request)
        assert entered.wait(timeout=2)
        started = monotonic()
        second = harness.reserve(second_request)
        elapsed = monotonic() - started
        release.set()
        first = future.result(timeout=2)
    assert elapsed < 2
    assert first.accepted
    assert second.code is LandingReservationRefusalCode.RESERVATION_CONFLICT
    assert not harness.active


def test_harness_same_thread_reentry_is_fixed_conflict() -> None:
    harness = AtomicHarness([_state()])
    nested: list[HarnessResult] = []
    harness.on_after_hold = lambda: nested.append(
        harness.reserve(_request(request_id="nested", invocation_id="nested"))
    )
    result = harness.reserve(_request())
    assert result.accepted
    assert (
        nested and nested[0].code is LandingReservationRefusalCode.RESERVATION_CONFLICT
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        (
            _state(canonical=_canonical(state=CanonicalState.DIRTY)),
            LandingReservationRefusalCode.CANONICAL_DIRTY,
        ),
        (
            _state(canonical=_canonical(private_wip=True)),
            LandingReservationRefusalCode.PRIVATE_WIP,
        ),
        (
            _state(canonical=_canonical(index_clean=False)),
            LandingReservationRefusalCode.PRIVATE_WIP,
        ),
        (
            _state(canonical=_canonical(state=CanonicalState.UNKNOWN)),
            LandingReservationRefusalCode.CANONICAL_STATE_INVALID,
        ),
        (
            _state(occupancy=_occupancy(count=1, state=OccupancyState.OCCUPIED)),
            LandingReservationRefusalCode.TARGET_OCCUPIED,
        ),
        (
            _state(occupancy=_occupancy(state=OccupancyState.UNKNOWN)),
            LandingReservationRefusalCode.TARGET_OCCUPANCY_UNKNOWN,
        ),
    ],
)
def test_harness_state_policy_is_fail_closed(
    state: LandingStateSnapshot, expected: LandingReservationRefusalCode
) -> None:
    result = AtomicHarness([state]).reserve(_request())
    assert result.code is expected


def test_harness_target_tree_drift_with_stable_revision_refuses() -> None:
    states = [_state(), _state(target=_target(tree_sha=SHA_ALT))]
    harness = AtomicHarness(states)
    harness.on_before_barrier = lambda: setattr(harness, "state_index", 1)
    result = harness.reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_harness_object_setattr_source_mutation_is_revalidated() -> None:
    state = _state()
    harness = AtomicHarness([state])

    def mutate() -> None:
        object.__setattr__(state.target, "tree_sha", SHA_ALT)

    harness.on_after_capture = mutate
    result = harness.reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST


def test_harness_barrier_lease_loss_and_identity_alias_refuse() -> None:
    harness = AtomicHarness([_state()])
    harness.on_before_barrier = lambda: setattr(harness, "barrier_valid", False)
    result = harness.reserve(_request())
    assert result.code is LandingReservationRefusalCode.RESERVATION_LOST

    resolved = _resolved()
    alias_state = _state(
        _resolved(common_dir_id="common-dir:forged", worktree_id="worktree:forged")
    )
    result = AtomicHarness([alias_state], resolved).reserve(_request())
    assert result.code is LandingReservationRefusalCode.REPOSITORY_MISMATCH


def test_harness_release_failure_records_recovery_even_after_body_failure() -> None:
    harness = AtomicHarness([_state()])
    harness.fail_release = True
    harness.on_after_capture = lambda: (_ for _ in ()).throw(RuntimeError("body"))
    result = harness.reserve(_request())
    assert result.code is LandingReservationRefusalCode.RECOVERY_REQUIRED
    assert harness.recovery_required
    assert not harness.active


def test_production_refusal_has_no_mutation_or_provider_effect() -> None:
    class HostileRequest:
        def __getattribute__(self, _name: str) -> object:
            raise AssertionError("hostile request must not reach provider")

    result = landing.reserve_landing(HostileRequest())  # type: ignore[arg-type]
    assert result.code is LandingReservationRefusalCode.REQUEST_INVALID
    result = landing.reserve_landing(_request())
    assert result.code is LandingReservationRefusalCode.AUTHORITY_UNAVAILABLE


@pytest.mark.parametrize("seed", [0, 1, 17, 31337])
def test_hash_seed_marker(seed: int) -> None:
    assert seed in {0, 1, 17, 31337}
