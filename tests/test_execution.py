"""RMDD-07 local executor security, recovery, and bounded-output tests."""

from __future__ import annotations

import hashlib
import io
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

from repository_manager.development import (
    ExecutionCommand,
    ExecutionOutcome,
    ExecutionResult,
    FailureClass,
    RefusalCode,
)
from repository_manager.execution import (
    BoundedLogSink,
    CancellationToken,
    FakeClock,
    FakeExecutor,
    FakeProcess,
    LocalExecutor,
)
from repository_manager.execution.process_supervisor import ProcessSupervisor


def _command(
    workdir: Path,
    *argv: str,
    timeout_seconds: int = 3600,
    heartbeat_interval_seconds: int = 30,
    environment_refs: tuple[str, ...] = (),
) -> ExecutionCommand:
    return ExecutionCommand(
        argv=argv,
        workdir=str(workdir.resolve()),
        timeout_seconds=timeout_seconds,
        heartbeat_interval_seconds=heartbeat_interval_seconds,
        environment_refs=environment_refs,
    )


def test_bounded_log_sink_streams_redacted_tail_without_growth() -> None:
    written: list[bytes] = []
    sink = BoundedLogSink(
        max_stdout_bytes=16,
        max_stderr_bytes=8,
        terminal_tail_bytes=5,
        writer=lambda _stream, chunk: written.append(chunk),
        redactions=("secret",),
    )

    sink.write("stdout", b"prefix-secret-0123456789")
    sink.write("stdout", b"-tail")
    sink.close()

    snapshot = sink.snapshot("stdout")
    assert b"secret" not in b"".join(written)
    assert snapshot.total_bytes == len(b"prefix-secret-0123456789-tail")
    assert snapshot.retained_bytes <= 16
    assert snapshot.discarded_bytes > 0
    assert snapshot.truncated
    assert snapshot.tail == b"-tail"
    assert sink.tail_text("stdout") == "-tail"


def test_bounded_log_sink_redacts_secrets_across_one_byte_boundaries() -> None:
    secret = b"boundary-secret"
    output = b"prefix-" + secret + b"-suffix"
    written: list[tuple[str, bytes]] = []
    sink = BoundedLogSink(
        max_stdout_bytes=4096,
        max_stderr_bytes=4096,
        terminal_tail_bytes=4096,
        writer=lambda stream, chunk: written.append((stream, chunk)),
        redactions=(secret.decode(),),
    )

    for byte in output:
        sink.write("stdout", bytes((byte,)))
    for byte in b"err-" + secret + b"-end":
        sink.write("stderr", bytes((byte,)))
    sink.close()

    stdout_written = b"".join(chunk for stream, chunk in written if stream == "stdout")
    stderr_written = b"".join(chunk for stream, chunk in written if stream == "stderr")
    stdout_snapshot = sink.snapshot("stdout")
    stderr_snapshot = sink.snapshot("stderr")
    assert secret not in stdout_written
    assert secret not in stderr_written
    assert secret not in stdout_snapshot.tail
    assert secret not in stderr_snapshot.tail
    assert secret not in sink.tail_bytes("stdout")
    assert secret not in sink.tail_bytes("stderr")
    assert stdout_snapshot.total_bytes == len(output)
    assert stderr_snapshot.total_bytes == len(b"err-" + secret + b"-end")
    assert stdout_snapshot.content_address == hashlib.sha256(stdout_written).hexdigest()
    assert stderr_snapshot.content_address == hashlib.sha256(stderr_written).hexdigest()


def test_bounded_log_sink_abort_discards_unresolved_redaction_overlap() -> None:
    secret = b"boundary-secret"
    written: list[bytes] = []
    sink = BoundedLogSink(
        max_stdout_bytes=4096,
        terminal_tail_bytes=4096,
        writer=lambda _stream, chunk: written.append(chunk),
        redactions=(secret.decode(),),
    )

    sink.write("stdout", b"prefix-" + secret[:8])
    sink.abort()
    sink.close()

    assert b"".join(written) == b"prefix-"
    assert sink.tail_bytes("stdout") == b"prefix-"
    assert secret not in b"".join(written)
    assert secret not in sink.tail_bytes("stdout")


class _OneByteReader(io.BytesIO):
    """Popen-like reader that adversarially splits every output byte."""

    def read(self, _size: int | None = None) -> bytes:
        return super().read(1)


def test_injected_log_sink_redacts_reader_boundaries_and_result_tails(
    tmp_path: Path,
) -> None:
    credential = "injected-boundary-secret"
    raw_stdout = f"prefix-{credential}-suffix\n".encode()
    raw_stderr = f"error-{credential}-end\n".encode()
    fake_process = FakeProcess(stdout=b"", stderr=b"")
    fake_process.stdout = _OneByteReader(raw_stdout)
    fake_process.stderr = _OneByteReader(raw_stderr)
    supervisor = ProcessSupervisor(
        popen_factory=lambda *_args, **_kwargs: fake_process,
    )
    written: list[tuple[str, bytes]] = []
    injected_sink = BoundedLogSink(
        max_stdout_bytes=4096,
        max_stderr_bytes=4096,
        terminal_tail_bytes=4096,
        writer=lambda stream, chunk: written.append((stream, chunk)),
    )
    result = LocalExecutor(
        (tmp_path,),
        environment={"RMDD_SECRET": credential},
        inherit_environment=False,
        supervisor=supervisor,
    ).run(
        _command(tmp_path, "fake", environment_refs=("RMDD_SECRET",)),
        log_sink=injected_sink,
    )

    stdout_written = b"".join(chunk for stream, chunk in written if stream == "stdout")
    stderr_written = b"".join(chunk for stream, chunk in written if stream == "stderr")
    assert result.outcome == ExecutionOutcome.SUCCEEDED
    assert credential not in stdout_written.decode()
    assert credential not in stderr_written.decode()
    assert credential not in result.stdout_tail
    assert credential not in result.stderr_tail
    assert credential not in injected_sink.tail_text("stdout")
    assert credential not in injected_sink.tail_text("stderr")
    assert (
        injected_sink.snapshot("stdout").content_address
        == hashlib.sha256(stdout_written).hexdigest()
    )
    assert (
        injected_sink.snapshot("stderr").content_address
        == hashlib.sha256(stderr_written).hexdigest()
    )


def test_cancellation_is_idempotent_and_preserves_first_reason() -> None:
    token = CancellationToken()
    assert token.cancel("operator requested stop")
    assert not token.cancel("late duplicate")
    snapshot = token.snapshot()
    assert snapshot.cancelled
    assert snapshot.reason == "operator requested stop"
    assert snapshot.requested_at is not None


def test_invalid_argv_and_workdir_refuse_before_spawn(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(
        popen_factory=lambda *_args, **_kwargs: pytest.fail("spawn must not run")
    )
    executor = LocalExecutor((tmp_path,), supervisor=supervisor)

    shell_string = executor.run(_command(tmp_path, "echo hello"))
    missing_workdir = executor.run(_command(tmp_path / "missing", "echo", "hello"))

    assert shell_string.outcome == ExecutionOutcome.REFUSED
    assert shell_string.failure_class == FailureClass.INVALID_REQUEST
    assert missing_workdir.outcome == ExecutionOutcome.REFUSED
    assert missing_workdir.failure_class == FailureClass.INVALID_REQUEST


def test_shell_control_in_executable_is_refused_with_stable_code(
    tmp_path: Path,
) -> None:
    executor = LocalExecutor((tmp_path,))
    with pytest.raises(ValueError) as raised:
        executor.validate(_command(tmp_path, "python;touch"))
    assert getattr(raised.value, "code", None) == RefusalCode.SHELL_COMMAND_FORBIDDEN


def test_initial_cancellation_and_fence_never_spawn(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(
        popen_factory=lambda *_args, **_kwargs: pytest.fail("spawn must not run")
    )
    executor = LocalExecutor((tmp_path,), supervisor=supervisor)
    command = _command(tmp_path, sys.executable, "-c", "print('never')")

    cancelled = CancellationToken()
    cancelled.cancel("before launch")
    cancelled_result = executor.run(command, cancellation=cancelled)
    fenced_result = executor.run(command, fence_check=lambda: False)

    assert cancelled_result.outcome == ExecutionOutcome.CANCELLED
    assert cancelled_result.failure_class == FailureClass.CANCELLED_DEADLINE
    assert fenced_result.outcome == ExecutionOutcome.REFUSED
    assert fenced_result.failure_class == FailureClass.STALE_FENCE_DUPLICATE_EFFECT


def test_success_streams_both_outputs_and_allows_publication(tmp_path: Path) -> None:
    published: list[ExecutionResult] = []
    command = _command(
        tmp_path,
        sys.executable,
        "-c",
        "import sys; print('out'); print('err', file=sys.stderr)",
        timeout_seconds=5,
    )
    result = LocalExecutor((tmp_path,)).run(command, publish=published.append)

    assert result.outcome == ExecutionOutcome.SUCCEEDED
    assert result.exit_code == 0
    assert result.failure_class is None
    assert result.stdout_tail == "out\n"
    assert result.stderr_tail == "err\n"
    assert len(published) == 1
    assert published[0].outcome == ExecutionOutcome.SUCCEEDED


def test_nonzero_and_signal_results_are_distinguishable(tmp_path: Path) -> None:
    failed = LocalExecutor((tmp_path,)).run(
        _command(
            tmp_path,
            sys.executable,
            "-c",
            "import sys; sys.exit(7)",
        )
    )
    signalled = LocalExecutor((tmp_path,)).run(
        _command(
            tmp_path,
            sys.executable,
            "-c",
            "import os, signal; os.kill(os.getpid(), signal.SIGTERM)",
        )
    )

    assert failed.outcome == ExecutionOutcome.FAILED
    assert failed.exit_code == 7
    assert failed.signal is None
    assert failed.failure_class == FailureClass.VALIDATION_CANDIDATE_FAILURE
    assert signalled.outcome == ExecutionOutcome.FAILED
    assert signalled.exit_code is None
    assert signalled.signal == signal.SIGTERM
    assert signalled.failure_class == FailureClass.WORKER_ENVIRONMENT_FAILURE


def test_environment_reference_is_materialized_and_redacted(tmp_path: Path) -> None:
    command = _command(
        tmp_path,
        sys.executable,
        "-c",
        "import os; print(os.environ['RMDD_SECRET'])",
        environment_refs=("RMDD_SECRET",),
    )
    result = LocalExecutor(
        (tmp_path,),
        environment={"RMDD_SECRET": "secret-value-123"},
        inherit_environment=False,
    ).run(command)

    assert result.outcome == ExecutionOutcome.SUCCEEDED
    assert "secret-value-123" not in result.stdout_tail
    assert "[REDACTED]" in result.stdout_tail


def test_missing_environment_reference_fails_before_spawn(tmp_path: Path) -> None:
    supervisor = ProcessSupervisor(
        popen_factory=lambda *_args, **_kwargs: pytest.fail("spawn must not run")
    )
    result = LocalExecutor(
        (tmp_path,), inherit_environment=False, supervisor=supervisor
    ).run(
        _command(
            tmp_path,
            sys.executable,
            "-c",
            "print('never')",
            environment_refs=("MISSING_RMDD_REF",),
        )
    )
    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.WORKER_ENVIRONMENT_FAILURE


def test_timeout_terminates_process_group_and_reports_cleanup(tmp_path: Path) -> None:
    pid_file = tmp_path / "child.pid"
    child_code = (
        "import os, pathlib, sys, time; "
        "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); time.sleep(30)"
    )
    parent_code = (
        "import subprocess, sys, time; "
        "subprocess.Popen([sys.executable, '-c', sys.argv[2], sys.argv[1]]); "
        "time.sleep(30)"
    )
    command = _command(
        tmp_path,
        sys.executable,
        "-c",
        parent_code,
        str(pid_file),
        child_code,
        timeout_seconds=1,
    )
    result = LocalExecutor((tmp_path,)).run(command)

    assert result.outcome == ExecutionOutcome.TIMED_OUT
    assert result.failure_class == FailureClass.CANCELLED_DEADLINE
    assert result.cleanup_ok
    child_pid = int(pid_file.read_text())
    for _ in range(40):
        proc_stat = Path(f"/proc/{child_pid}/stat")
        if not proc_stat.exists():
            break
        state = proc_stat.read_text().split()[2]
        if state == "Z":
            break
        time.sleep(0.05)
    assert not proc_stat.exists() or proc_stat.read_text().split()[2] == "Z"


def test_cancellation_during_execution_terminates_child(tmp_path: Path) -> None:
    token = CancellationToken()
    command = _command(
        tmp_path,
        sys.executable,
        "-c",
        "import time; print('started', flush=True); time.sleep(30)",
        timeout_seconds=30,
    )
    results: list[ExecutionResult] = []
    thread = threading.Thread(
        target=lambda: results.append(
            LocalExecutor((tmp_path,)).run(command, cancellation=token)
        )
    )
    thread.start()
    time.sleep(0.15)
    assert token.cancel("test cancellation")
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert results[0].outcome == ExecutionOutcome.CANCELLED
    assert results[0].cleanup_ok
    assert results[0].stdout_tail == "started\n"


def test_stale_fence_quarantines_output_and_skips_publication(tmp_path: Path) -> None:
    checks = iter((True, False))
    published: list[ExecutionResult] = []
    command = _command(
        tmp_path,
        sys.executable,
        "-c",
        "import time; print('work', flush=True); time.sleep(5)",
        timeout_seconds=10,
    )
    result = LocalExecutor((tmp_path,)).run(
        command,
        fence_check=lambda: next(checks, False),
        publish=published.append,
    )

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.STALE_FENCE_DUPLICATE_EFFECT
    assert not published


def test_heartbeat_failure_is_bounded_and_explicit(tmp_path: Path) -> None:
    command = _command(
        tmp_path,
        sys.executable,
        "-c",
        "import time; time.sleep(4)",
        timeout_seconds=10,
        heartbeat_interval_seconds=1,
    )
    result = LocalExecutor((tmp_path,)).run(command, heartbeat=lambda: False)

    assert result.outcome == ExecutionOutcome.REFUSED
    assert result.failure_class == FailureClass.WORKER_ENVIRONMENT_FAILURE
    assert result.cleanup_ok


def test_fake_fixtures_are_usable_by_downstream_adapters(tmp_path: Path) -> None:
    clock = FakeClock()
    clock.advance(2)
    assert clock.monotonic() == 2
    assert clock.now().tzinfo is not None

    fake_process = FakeProcess(stdout=b"fake\n")
    supervisor = ProcessSupervisor(popen_factory=lambda *_args, **_kwargs: fake_process)
    result = LocalExecutor((tmp_path,), supervisor=supervisor).run(
        _command(tmp_path, "fake")
    )
    assert result.outcome == ExecutionOutcome.SUCCEEDED
    assert result.stdout_tail == "fake\n"

    executor = FakeExecutor()
    fake_result = executor.run(_command(tmp_path, "fake"))
    assert fake_result.outcome == ExecutionOutcome.SUCCEEDED
    assert len(executor.commands) == 1
