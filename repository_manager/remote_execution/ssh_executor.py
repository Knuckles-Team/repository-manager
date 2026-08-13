"""``TunnelSSHExecutor``: a ``CommandExecutor``-shaped seam over the SSH
primitive tunnel-manager's own MCP surface already uses in production.

**Why this exists instead of extending ``executor.py``'s
``RemoteWorkerExecutor``.** That module composes against
``tunnel_manager.remote_execution`` — ``AuthorizedTarget``,
``RemoteCommandRequest``, ``TunnelCommandExecutor`` — and its own docstring
says so explicitly: that module "lives only on tunnel-manager's own unmerged
integration branch and predates every published PyPI release." Verified
against this workspace's actual `tunnel-manager` checkout
(`agent-packages/agents/tunnel-manager/tunnel_manager/`): there is no
`remote_execution.py` there at all, and `repository-manager`'s own
`pyproject.toml` declared no `remote` extra pointing at tunnel-manager either
— so "install repository-manager's 'remote' extra" (a sentence several
docstrings in this package already say) named a thing that did not exist.
Building `rm_build request(host=...)` against that seam would not be wiring
an executor, it would be deferring the whole feature to a dependency that has
not shipped it.

What DOES exist and DOES work: ``tunnel_manager.tunnel_manager.Tunnel.run_command``
— the exact primitive tunnel-manager's own ``tm_remote(action="run_command")``
MCP tool calls in production. This executor drives that primitive and
presents the same ``CommandExecutor``-shaped ``run(...) -> ExecutionResult``
seam :mod:`repository_manager.remote_execution.source_staging` already
consumes, so ``stage_source``/``verify_source`` need no change to run
remotely — only a caller that constructs this executor instead of
``LocalExecutor``.

**Honesty about what this executor can and cannot prove.** `run_command` is
one blocking SSH exec with no native cancellation, fencing, or streaming
hook — it returns a plain `(success, stdout, stderr)` result once the remote
command has already finished. So `cancellation`/`fence_check`/`heartbeat`
are honored only at the boundaries (before dispatch, and by downgrading a
result that raced a fence loss/cancellation) — never mid-command, which is
exactly the same honestly-weaker-than-local guarantee
``RemoteWorkerExecutor``'s own docstring describes for the seam it targets.
This executor does not invent a stronger guarantee for a simpler mechanism.
"""

from __future__ import annotations

import os
import shlex
from datetime import UTC, datetime
from typing import Any

from repository_manager.development import ExecutionCommand, ExecutionResult
from repository_manager.development import ExecutionOutcome as RmExecutionOutcome
from repository_manager.development import FailureClass as RmFailureClass
from repository_manager.execution.cancellation import CancellationToken
from repository_manager.execution.executor import LogSink, PublicationDecision

try:
    from tunnel_manager.tunnel_manager import Tunnel

    _TUNNEL_MANAGER_IMPORT_ERROR: ImportError | None = None
except ImportError as _tunnel_manager_import_error:  # pragma: no cover - exercised
    # by test_optional_dependency-style tests without the extra installed.
    Tunnel = None  # type: ignore[assignment,misc]
    _TUNNEL_MANAGER_IMPORT_ERROR = _tunnel_manager_import_error


def _default_tunnel(alias: str) -> Any:
    """Construct a ``Tunnel`` for *alias*, resolving its real ``HostName``.

    Live-proven gap (found while validating this module against R820):
    ``tunnel_manager.tunnel_manager.Tunnel.__init__`` looks up
    ``~/.ssh/config`` for auth parameters (user/identity file/proxy) via
    ``self.ssh_config.lookup(self.remote_host)`` — but it never substitutes
    the alias's own ``HostName`` into ``self.remote_host`` FIRST, so
    ``Tunnel(remote_host="R820")`` hands paramiko the literal string
    ``"R820"`` to resolve over DNS (fails: `gaierror: Temporary failure in
    name resolution`) even though ``ssh R820`` from the same host succeeds
    immediately, AND its internal auth lookup then queries
    ``ssh_config.lookup("R820")`` again redundantly on top of that. That is a
    tunnel-manager defect, out of this package's scope to fix directly (a
    different repository, its own worktree/lane discipline) — so this
    resolves the FULL alias block (hostname, user, identity file, proxy
    command) the same way `ssh`/paramiko's own `SSHConfig.lookup` would, here,
    once, and passes every resolved field explicitly to `Tunnel(...)` — never
    just the hostname, which would silently drop the alias's own
    user/identity and fall back to whatever ambient default paramiko finds.
    """

    _require_tunnel_manager()
    import paramiko  # type: ignore[import-untyped]

    config_path = os.path.expanduser("~/.ssh/config")
    resolved: dict[str, Any] = {"remote_host": alias}
    if os.path.isfile(config_path):
        parsed = paramiko.SSHConfig()
        with open(config_path, encoding="utf-8") as handle:
            parsed.parse(handle)
        looked_up = parsed.lookup(alias)
        if "hostname" in looked_up:
            resolved["remote_host"] = looked_up["hostname"]
        if "user" in looked_up:
            resolved["username"] = looked_up["user"]
        if looked_up.get("identityfile"):
            resolved["identity_file"] = looked_up["identityfile"][0]
        if "proxycommand" in looked_up:
            resolved["proxy_command"] = looked_up["proxycommand"]
    return Tunnel(**resolved)


class RemoteSshExecutionUnavailableError(RuntimeError):
    """Raised when an SSH dispatch is attempted without tunnel-manager.

    Local execution is unaffected; this is refused, not silently degraded —
    matching the sibling refusal in
    :mod:`repository_manager.remote_execution.executor`.
    """


def _require_tunnel_manager() -> None:
    if _TUNNEL_MANAGER_IMPORT_ERROR is not None:
        raise RemoteSshExecutionUnavailableError(
            "SSH dispatch requires the optional 'tunnel-manager' dependency "
            "(repository-manager's 'remote' extra), which is not installed "
            "in this environment"
        ) from _TUNNEL_MANAGER_IMPORT_ERROR


class TunnelSSHExecutor:
    """Run one C-04 ``ExecutionCommand`` on one SSH-reachable inventory alias.

    ``alias`` is resolved by tunnel-manager's own SSH config
    (``~/.ssh/config`` by default) — the SAME identity/hostname/credential
    boundary tunnel-manager's ``tm_remote`` tool uses, never a raw
    host/credential this module invents. This class performs NO entitlement
    check of its own: callers authorize the target (host capacity, drain
    state, repository root) before constructing one, exactly as
    :mod:`repository_manager.remote_execution.registry` does for
    ``RemoteWorkerExecutor``.
    """

    def __init__(
        self,
        alias: str,
        *,
        worker_id: str = "worker:ssh",
        tunnel_factory: Any = None,
    ) -> None:
        _require_tunnel_manager()
        if not alias or alias.strip() != alias:
            raise ValueError("alias must be non-blank")
        self.alias = alias
        self.worker_id = worker_id
        # Overridable only for tests — production always constructs a real
        # ``Tunnel`` bound to the resolved SSH alias (see `_default_tunnel`).
        self._tunnel_factory = tunnel_factory or _default_tunnel

    def run(
        self,
        command: ExecutionCommand,
        *,
        command_id: str = "command:ssh",
        worker_id: str | None = None,
        fence: str = "fence:ssh",
        cancellation: CancellationToken | None = None,
        fence_check: Any = None,
        heartbeat: Any = None,
        log_sink: LogSink | None = None,
        publisher: Any = None,
    ) -> ExecutionResult:
        """Execute *command* over SSH on ``self.alias`` and return its result."""

        effective_worker = worker_id or self.worker_id
        started_at = datetime.now(UTC)
        token = cancellation or CancellationToken()
        checker = fence_check or (lambda: True)

        if token.is_cancelled():
            return self._finish(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.CANCELLED,
                RmFailureClass.CANCELLED_DEADLINE,
                started_at,
                "",
                "cancelled before SSH dispatch",
                log_sink,
            )
        if not self._ok(checker):
            return self._finish(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.REFUSED,
                RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT,
                started_at,
                "",
                "fence invalid before SSH dispatch",
                log_sink,
            )

        shell_command = self._shell_command(command)
        tunnel = self._tunnel_factory(self.alias)
        try:
            result = tunnel.run_command(shell_command, timeout=command.timeout_seconds)
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            return self._finish(
                command_id,
                effective_worker,
                fence,
                RmExecutionOutcome.FAILED,
                RmFailureClass.WORKER_ENVIRONMENT_FAILURE,
                started_at,
                "",
                f"SSH dispatch to {self.alias!r} raised {type(exc).__name__}: {exc}",
                log_sink,
            )

        outcome = (
            RmExecutionOutcome.SUCCEEDED
            if result.success
            else RmExecutionOutcome.FAILED
        )
        failure_class = (
            None if result.success else RmFailureClass.VALIDATION_CANDIDATE_FAILURE
        )
        stdout_tail = self._bounded(result.stdout, command.max_stdout_bytes)
        stderr_tail = self._bounded(
            result.stderr or (getattr(result, "error_message", "") or ""),
            command.max_stderr_bytes,
        )

        # A cancellation/fence loss observed only AFTER the (already
        # blocking, already-finished) remote call returned still must not
        # be published as a success — same downgrade rule
        # ``RemoteWorkerExecutor``/``LocalExecutor`` both apply.
        if outcome == RmExecutionOutcome.SUCCEEDED and token.is_cancelled():
            outcome = RmExecutionOutcome.CANCELLED
            failure_class = RmFailureClass.CANCELLED_DEADLINE
        elif outcome == RmExecutionOutcome.SUCCEEDED and not self._ok(checker):
            outcome = RmExecutionOutcome.REFUSED
            failure_class = RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT

        exec_result = self._finish(
            command_id,
            effective_worker,
            fence,
            outcome,
            failure_class,
            started_at,
            stdout_tail,
            stderr_tail,
            log_sink,
            # `CommandResult` carries no numeric exit code, only `success`
            # (already `exit_status == 0` at the source) -- SUCCEEDED means
            # exactly exit_code=0, never a fabricated number for anything else.
            exit_code=0 if outcome == RmExecutionOutcome.SUCCEEDED else None,
        )

        if (
            exec_result.outcome == RmExecutionOutcome.SUCCEEDED
            and publisher is not None
        ):
            try:
                decision = publisher.publish(exec_result, fence=fence)
            except Exception:  # noqa: BLE001 - defensive publication boundary
                exec_result = exec_result.model_copy(
                    update={
                        "outcome": RmExecutionOutcome.FAILED,
                        "failure_class": RmFailureClass.WORKER_ENVIRONMENT_FAILURE,
                    }
                )
            else:
                if decision != PublicationDecision.ACCEPTED:
                    exec_result = exec_result.model_copy(
                        update={
                            "outcome": RmExecutionOutcome.REFUSED,
                            "failure_class": (
                                RmFailureClass.STALE_FENCE_DUPLICATE_EFFECT
                            ),
                        }
                    )
        return exec_result

    @staticmethod
    def _shell_command(command: ExecutionCommand) -> str:
        argv = " ".join(shlex.quote(part) for part in command.argv)
        return f"cd {shlex.quote(command.workdir)} && {argv}"

    @staticmethod
    def _bounded(text: str, limit: int) -> str:
        encoded = text.encode("utf-8", errors="replace")
        if len(encoded) <= limit:
            return text
        return encoded[-limit:].decode("utf-8", errors="replace")

    @staticmethod
    def _ok(checker: Any) -> bool:
        try:
            return bool(checker())
        except Exception:
            return False

    @staticmethod
    def _finish(
        command_id: str,
        worker_id: str,
        fence: str,
        outcome: RmExecutionOutcome,
        failure_class: RmFailureClass | None,
        started_at: datetime,
        stdout_tail: str,
        stderr_tail: str,
        log_sink: LogSink | None,
        *,
        exit_code: int | None = None,
    ) -> ExecutionResult:
        finished_at = datetime.now(UTC)
        result = ExecutionResult(
            command_id=command_id,
            outcome=outcome,
            exit_code=exit_code,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=max(0, int((finished_at - started_at).total_seconds() * 1000)),
            worker_id=worker_id,
            fence=fence,
            stdout_tail=stdout_tail,
            stderr_tail=stderr_tail,
            failure_class=failure_class,
        )
        if log_sink is not None:
            try:
                if result.stdout_tail:
                    log_sink.write("stdout", result.stdout_tail.encode("utf-8"))
                if result.stderr_tail:
                    log_sink.write("stderr", result.stderr_tail.encode("utf-8"))
                if result.outcome == RmExecutionOutcome.REFUSED:
                    log_sink.abort()
                else:
                    log_sink.close()
            except Exception:  # noqa: BLE001 - sink failures must not mask the result
                pass
        return result


__all__ = [
    "RemoteSshExecutionUnavailableError",
    "TunnelSSHExecutor",
]
