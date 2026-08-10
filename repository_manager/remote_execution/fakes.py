"""Deterministic fixtures for remote-execution tests.

These fakes satisfy the exact frozen protocol shapes this package consumes
(``RemoteExecutorPort``/``InventoryResolver``, both shaped after RMDD-14's
tunnel-manager seam) without opening a socket, a subprocess, or an SSH
connection, and without relaxing any validation the real TM/RM contracts
perform -- every model constructed here still runs through its real pydantic
validators.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from tunnel_manager.remote_execution import (
    AuthorizedTarget,
    ExecutionOutcome,
    FailureClass,
    RemoteCommandRequest,
    RemoteExecutionContext,
    RemoteExecutionResult,
    RemoteTargetError,
)

Responder = Callable[
    [AuthorizedTarget, RemoteCommandRequest, RemoteExecutionContext],
    RemoteExecutionResult,
]


class FakeInventoryResolver:
    """Deterministic entitlement resolver; aliases can be revoked mid-test."""

    def __init__(self, authorized_aliases: set[str] | None = None) -> None:
        self.authorized_aliases: set[str] = set(authorized_aliases or set())
        self.resolutions: list[tuple[str, object]] = []

    def resolve(self, alias: str, actor: object) -> AuthorizedTarget:
        """Record the call and enforce the current authorization set."""

        self.resolutions.append((alias, actor))
        if alias not in self.authorized_aliases:
            raise RemoteTargetError("remote target authorization failed")
        return AuthorizedTarget(alias=alias)

    def revoke(self, alias: str) -> None:
        """Simulate an entitlement revocation between plan and claim."""

        self.authorized_aliases.discard(alias)

    def authorize(self, alias: str) -> None:
        """Simulate granting an entitlement."""

        self.authorized_aliases.add(alias)


class FakeRemoteExecutorPort:
    """Records dispatched requests and returns a scripted/deterministic result.

    ``responder`` receives the target/request/context and can inspect argv
    (e.g. to detect the cooperative cancellation marker command) and return a
    ``RemoteExecutionResult`` -- letting tests script host loss, disconnect,
    or fence races precisely without any real transport.
    """

    def __init__(self, responder: Responder | None = None) -> None:
        self.calls: list[
            tuple[AuthorizedTarget, RemoteCommandRequest, RemoteExecutionContext]
        ] = []
        self._responder = responder

    def execute(
        self,
        target: AuthorizedTarget,
        request: RemoteCommandRequest,
        actor: object,
        *,
        context: RemoteExecutionContext,
    ) -> RemoteExecutionResult:
        self.calls.append((target, request, context))
        if self._responder is not None:
            return self._responder(target, request, context)
        return self.succeeded(context)

    @staticmethod
    def succeeded(context: RemoteExecutionContext) -> RemoteExecutionResult:
        now = datetime.now(UTC)
        return RemoteExecutionResult(
            command_id=context.command_id,
            outcome=ExecutionOutcome.SUCCEEDED,
            exit_code=0,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            worker_id=context.worker_id,
            fence=context.fence,
        )

    @staticmethod
    def failed(
        context: RemoteExecutionContext,
        failure_class: FailureClass = FailureClass.WORKER_ENVIRONMENT_FAILURE,
    ) -> RemoteExecutionResult:
        now = datetime.now(UTC)
        return RemoteExecutionResult(
            command_id=context.command_id,
            outcome=ExecutionOutcome.FAILED,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            worker_id=context.worker_id,
            fence=context.fence,
            failure_class=failure_class,
        )

    @staticmethod
    def refused(
        context: RemoteExecutionContext,
        failure_class: FailureClass = FailureClass.STALE_FENCE_DUPLICATE_EFFECT,
    ) -> RemoteExecutionResult:
        now = datetime.now(UTC)
        return RemoteExecutionResult(
            command_id=context.command_id,
            outcome=ExecutionOutcome.REFUSED,
            started_at=now,
            finished_at=now,
            duration_ms=0,
            worker_id=context.worker_id,
            fence=context.fence,
            failure_class=failure_class,
        )


__all__ = ["FakeInventoryResolver", "FakeRemoteExecutorPort", "Responder"]
