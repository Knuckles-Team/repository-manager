"""Fixed installed remote-worker bootstrap command construction.

RMDD-15 never sends a model-authored or caller-supplied command string over
SSH.  The remote worker is a fixed, pre-installed binary; the only variable
inputs accepted here are the opaque WorkItem correlations (work item ID,
attempt, fence) the installed worker uses to look up its own authorized job
data from graph-os.  No job payload, code, host, credential, or shell text
ever passes through this module -- see the "Prefer a fixed installed worker
bootstrap" requirement in the lane brief and C-04 ("no ``shell=True`` or
public raw shell string is valid").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from pydantic import TypeAdapter, ValidationError

from repository_manager.development import ExecutionCommand, JobId, WorkItemId

# Mirrors the shell-metacharacter policy enforced by
# ``repository_manager.execution.executor.LocalExecutor`` for the bootstrap
# executable path itself; kept as an independent constant so this module has
# no dependency on that package's private symbols.
_SHELL_META = re.compile(r"[\x00\r\n\t;&|<>$`(){}\[\]`\\]")
_DEFAULT_BOOTSTRAP_PATH = "/opt/repository-manager/bin/rm-remote-worker"

_WORK_ITEM_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(WorkItemId)
_JOB_ID_ADAPTER: TypeAdapter[str] = TypeAdapter(JobId)


class RemoteWorkerBootstrapError(ValueError):
    """The bootstrap command could not be built from the supplied inputs."""


def _validate_opaque_work_item(value: str) -> str:
    """Accept either the full namespaced WorkItem ID or a public job handle.

    Both forms are already opaque, validated identifiers (C-02): a full
    ``workitem:repository_manager:<uuid>`` or a public ``rmjob:<uuid>``
    handle.  Truncated or ad hoc identifiers are refused.
    """

    for adapter in (_WORK_ITEM_ID_ADAPTER, _JOB_ID_ADAPTER):
        try:
            return adapter.validate_python(value)
        except ValidationError:
            continue
    raise RemoteWorkerBootstrapError(
        "work_item_id must be a full namespaced WorkItem ID or public job handle"
    )


def _validate_fence(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value.strip() != value
        or any(ord(char) < 0x20 for char in value)
    ):
        raise RemoteWorkerBootstrapError("fence must be a non-blank opaque identifier")
    return value


def _validate_bootstrap_path(value: str) -> str:
    if not value or not value.startswith("/") or any(char.isspace() for char in value):
        raise RemoteWorkerBootstrapError(
            "bootstrap path must be a fixed absolute path with no whitespace"
        )
    if _SHELL_META.search(value):
        raise RemoteWorkerBootstrapError(
            "bootstrap path must not contain shell metacharacters"
        )
    return value


@dataclass(frozen=True)
class RemoteWorkerBootstrap:
    """Immutable, code-fixed remote worker invocation template.

    ``bootstrap_path`` is a deployment constant, never derived from a caller
    request.  ``build`` accepts only opaque WorkItem correlations and bounded
    resource limits already defined by C-04; it never accepts an ``argv``,
    a shell string, or connection material.
    """

    bootstrap_path: str = field(default=_DEFAULT_BOOTSTRAP_PATH)
    timeout_seconds: int = 3_600
    heartbeat_interval_seconds: int = 30
    max_stdout_bytes: int = 64 * 1024
    max_stderr_bytes: int = 64 * 1024
    max_artifact_bytes: int = 1024 * 1024 * 1024

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "bootstrap_path", _validate_bootstrap_path(self.bootstrap_path)
        )
        if self.timeout_seconds < 1 or self.heartbeat_interval_seconds < 1:
            raise RemoteWorkerBootstrapError(
                "timeout and heartbeat limits must be positive"
            )

    def build(
        self,
        *,
        work_item_id: str,
        attempt: int,
        fence: str,
        workdir: str,
        environment_refs: tuple[str, ...] = (),
        cancellation_channel: str | None = None,
    ) -> ExecutionCommand:
        """Return the fixed argv command for one claimed WorkItem attempt.

        The resulting :class:`ExecutionCommand` is C-04-compatible and can be
        dispatched through either the local executor or
        :class:`repository_manager.remote_execution.executor.RemoteWorkerExecutor`
        without further caller-authored argv assembly.
        """

        work_item_id = _validate_opaque_work_item(work_item_id)
        fence = _validate_fence(fence)
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise RemoteWorkerBootstrapError("attempt must be a positive integer")

        argv = (
            self.bootstrap_path,
            "--work-item",
            work_item_id,
            "--attempt",
            str(attempt),
            "--fence",
            fence,
        )
        return ExecutionCommand(
            argv=argv,
            workdir=workdir,
            environment_refs=environment_refs,
            timeout_seconds=self.timeout_seconds,
            max_stdout_bytes=self.max_stdout_bytes,
            max_stderr_bytes=self.max_stderr_bytes,
            max_artifact_bytes=self.max_artifact_bytes,
            heartbeat_interval_seconds=self.heartbeat_interval_seconds,
            cancellation_channel=cancellation_channel,
        )


def build_bootstrap_command(
    *,
    work_item_id: str,
    attempt: int,
    fence: str,
    workdir: str,
    bootstrap_path: str = _DEFAULT_BOOTSTRAP_PATH,
    environment_refs: tuple[str, ...] = (),
    cancellation_channel: str | None = None,
) -> ExecutionCommand:
    """Convenience wrapper around :class:`RemoteWorkerBootstrap` defaults."""

    return RemoteWorkerBootstrap(bootstrap_path=bootstrap_path).build(
        work_item_id=work_item_id,
        attempt=attempt,
        fence=fence,
        workdir=workdir,
        environment_refs=environment_refs,
        cancellation_channel=cancellation_channel,
    )


__all__ = [
    "RemoteWorkerBootstrap",
    "RemoteWorkerBootstrapError",
    "build_bootstrap_command",
]
