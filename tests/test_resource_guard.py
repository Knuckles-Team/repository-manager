from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

import repository_manager.job_outcome as job_outcome
from repository_manager import build_queue, gates, merge_queue, resource_guard
from repository_manager.resource_guard import (
    FilesystemEvidence,
    HostEvidence,
    ResourceAdmissionError,
    ResourceGuardError,
)
from repository_manager.resource_profiles import ResourceProfile, ResourceProfileError


def _host_evidence(*, free_bytes: int = 200 * 1024**3) -> HostEvidence:
    filesystems = tuple(
        FilesystemEvidence(
            purpose=name,
            path=f"/{name}",
            filesystem="xfs",
            free_bytes=free_bytes,
            free_inodes=200_000,
            required_free_bytes=1024**3,
            required_free_inodes=100_000,
        )
        for name in ("target", "tmpdir", "home", "system-tmp")
    )
    return HostEvidence(
        captured_unix_ns=1,
        memory_available_bytes=200 * 1024**3,
        swap_free_bytes=8 * 1024**3,
        memory_psi_some_avg10=0.0,
        memory_psi_full_avg10=0.0,
        cgroup_path="/user.slice/test",
        cgroup_memory_current=1024**3,
        cgroup_memory_max=240 * 1024**3,
        cgroup_oom_events=0,
        filesystems=filesystems,
    )


def test_cargo_shape_overrides_a_light_profile_and_caps_jobs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    command = resource_guard.prepare_command(
        ("cargo", "test", "--jobs=2"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="light-check",
        timeout=60,
        env={"CARGO_BUILD_JOBS": "99", "RUST_TEST_THREADS": "99"},
    )

    assert command.cargo is True
    assert command.heavy is True
    assert command.profile.name == "rust-build"
    environment = dict(command.environment)
    resource_guard._bounded_cargo_environment(environment, command, True)
    assert environment["CARGO_BUILD_JOBS"] == "2"
    assert environment["CARGO_INCREMENTAL"] == "0"
    assert environment["RUST_TEST_THREADS"] == "2"


def test_cargo_cli_cannot_override_the_profile_job_cap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    with pytest.raises(ResourceGuardError, match="cannot exceed"):
        resource_guard.prepare_command(
            ("cargo", "build", "-j16"),
            workdir=tmp_path,
            target_dir=tmp_path / "target",
            tmp_dir=tmp_path / "scratch",
            profile_name="rust-build",
            timeout=60,
        )


def test_job_outcome_zero_swap_is_a_valid_stricter_guarded_profile() -> None:
    compatibility = job_outcome.ResourceProfile(memory_swap_max="0")

    projected = compatibility.guard_profile(cargo=False)

    assert projected.memory_swap_max_mib == 0
    assert "MemorySwapMax=0M" in resource_guard._properties(projected, 60)


def test_job_outcome_rejects_zero_or_invalid_non_swap_memory_limits() -> None:
    with pytest.raises(ValueError, match="below one MiB"):
        job_outcome.ResourceProfile(memory_max="0").guard_profile(cargo=False)
    with pytest.raises(ValueError, match="below one MiB"):
        job_outcome.ResourceProfile(memory_high="0").guard_profile(cargo=False)
    for value in ("-1", "0.5", "invalid", 0, True):
        with pytest.raises(ValueError, match="invalid memory limit"):
            job_outcome.ResourceProfile(memory_swap_max=value).guard_profile(  # type: ignore[arg-type]
                cargo=False
            )


def test_resource_profiles_require_integer_runtime_limits_and_allow_zero_swap() -> None:
    assert ResourceProfile("swap-zero", memory_swap_max_mib=0).memory_swap_max_mib == 0
    for field_name in (
        "memory_high_mib",
        "memory_max_mib",
        "cpu_quota_percent",
        "tasks_max",
        "runtime_max_seconds",
        "cargo_jobs",
    ):
        with pytest.raises(ResourceProfileError):
            ResourceProfile("invalid-runtime", **{field_name: float("nan")})
    for value in (-1, 1.5, float("nan"), True, "0"):
        with pytest.raises(ResourceProfileError):
            ResourceProfile("invalid-swap", memory_swap_max_mib=value)  # type: ignore[arg-type]


def test_tmpdir_must_be_disk_backed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "tmpfs")
    with pytest.raises(ResourceGuardError, match="disk-backed"):
        resource_guard.prepare_command(
            ("cargo", "build"),
            workdir=tmp_path,
            target_dir=tmp_path / "target",
            tmp_dir=tmp_path / "scratch",
            profile_name="rust-build",
            timeout=60,
        )


def test_admission_refuses_low_bytes_or_inodes(tmp_path: Path) -> None:
    profile = resource_guard.DEFAULT_RESOURCE_PROFILES.resolve("rust-build")
    command = SimpleNamespace(profile=profile)
    low = _host_evidence(free_bytes=1)
    reasons = resource_guard._admission_reasons(command, low)
    assert "target lacks free bytes" in reasons

    inode_low = tuple(
        SimpleNamespace(
            purpose=item.purpose,
            free_bytes=item.required_free_bytes,
            required_free_bytes=item.required_free_bytes,
            free_inodes=1,
            required_free_inodes=item.required_free_inodes,
        )
        for item in low.filesystems
    )
    reasons = resource_guard._admission_reasons(
        command, SimpleNamespace(**{**low.as_dict(), "filesystems": inode_low})
    )
    assert "system-tmp lacks free inodes" in reasons


def test_guard_builds_a_bounded_systemd_scope_and_returns_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    evidence = _host_evidence()
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: evidence)
    monkeypatch.setattr(
        resource_guard, "capture_host_evidence", lambda _command: evidence
    )
    scope_directory = tmp_path / "fake-scope"
    scope_directory.mkdir()
    (scope_directory / "cgroup.events").write_text("populated 0\n", encoding="ascii")
    (scope_directory / "cgroup.procs").write_text("", encoding="ascii")
    monkeypatch.setattr(
        resource_guard,
        "_canonical_cgroup_path",
        lambda _path: ("/fake.scope", scope_directory),
    )
    monkeypatch.setattr(
        resource_guard, "_scope_execution_cgroup", lambda _unit: "/fake.scope"
    )
    monkeypatch.setattr(
        resource_guard, "_cgroup_directory", lambda _path: scope_directory
    )
    captured: dict[str, object] = {}

    def fake_bounded(argv, **kwargs):  # noqa: ANN001
        captured["argv"] = argv
        captured["env"] = kwargs["environment"]
        kwargs["admission"].bind_execution_cgroup("/fake.scope")
        return subprocess.CompletedProcess(argv, 0, "ok", ""), {
            "stdout": {"total_bytes": 2, "truncated": False}
        }

    monkeypatch.setattr(resource_guard, "_run_bounded_scope", fake_bounded)
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="light-check",
        timeout=60,
        env={"CARGO_BUILD_JOBS": "8"},
    )
    result = resource_guard.run_guarded(command)

    argv = captured["argv"]
    assert argv[:4] == ["systemd-run", "--user", "--scope", "--quiet"]
    assert "MemoryHigh=32768M" in argv
    assert "MemoryMax=49152M" in argv
    assert "MemorySwapMax=8192M" in argv
    assert "TasksMax=4096" in argv
    assert "OOMPolicy=stop" in argv
    assert captured["env"]["CARGO_BUILD_JOBS"] == "2"
    assert captured["env"]["CARGO_INCREMENTAL"] == "0"
    assert result.evidence()["returncode"] == 0
    assert result.evidence()["execution_cgroup"] == "/fake.scope"
    assert result.evidence()["output"]["stdout"]["total_bytes"] == 2


def test_heavy_reservation_refuses_a_second_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )
    lock = resource_guard._runtime_root().joinpath("heavy.lock").open("a+")
    resource_guard.fcntl.flock(
        lock.fileno(), resource_guard.fcntl.LOCK_EX | resource_guard.fcntl.LOCK_NB
    )
    try:
        with pytest.raises(ResourceAdmissionError, match="already held"):
            resource_guard.run_guarded(command)
    finally:
        resource_guard.fcntl.flock(lock.fileno(), resource_guard.fcntl.LOCK_UN)
        lock.close()


def test_guard_refuses_a_fleet_dispatcher_heavy_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )
    reservations = resource_guard._dispatcher_reservation_dir()
    (reservations / "fleet.json").write_text(
        json.dumps({"kind": "heavy", "unit": "dispatch-build-test.service"})
    )

    with pytest.raises(ResourceAdmissionError, match="owner PID/start fence"):
        resource_guard.run_guarded(command)


def test_dispatcher_reservation_uses_owner_fence_and_reclaims_stale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_require_empty_execution_cgroup", lambda _path: None)
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )
    reservations = resource_guard._dispatcher_reservation_dir()
    stale_sentinel = resource_guard._runtime_root() / "stale.result.json"
    stale = reservations / "stale.scope.json"
    stale.write_text(
        json.dumps(
            {
                "schema": "repository-resource-guard-v1",
                "unit": "stale.scope",
                "kind": "heavy",
                "owner_pid": resource_guard.os.getpid(),
                "owner_start_time": resource_guard.process_start_time() + 1,
                "owner_token": "stale-owner-token",
                "inherited_cgroup": None,
                "execution_cgroup": "/queue.scope",
                "sentinel": str(stale_sentinel),
                "sentinel_base": {"unit": "stale.scope", "kind": "heavy"},
                "admitted_unix_ns": 1,
            }
        ),
        encoding="utf-8",
    )
    stale_sentinel.write_text("{}\n", encoding="utf-8")

    reservation, owner_token = resource_guard._reserve_dispatcher_heavy(
        "fresh.scope", command
    )
    record = json.loads(reservation.read_text(encoding="utf-8"))
    assert record["owner_pid"] == resource_guard.os.getpid()
    assert record["owner_start_time"] == resource_guard.process_start_time()
    assert record["owner_token"] == owner_token
    assert not stale.exists()
    assert json.loads(stale_sentinel.read_text(encoding="utf-8"))["status"] == (
        "indeterminate"
    )
    resource_guard._release_dispatcher_heavy(reservation, owner_token)


def test_stale_reservation_without_execution_proof_refuses_recovery(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )
    reservations = resource_guard._dispatcher_reservation_dir()
    stale = reservations / "unfenced.scope.json"
    stale_sentinel = resource_guard._runtime_root() / "unfenced.result.json"
    stale.write_text(
        json.dumps(
            {
                "schema": "repository-resource-guard-v1",
                "unit": "unfenced.scope",
                "kind": "heavy",
                "owner_pid": resource_guard.os.getpid(),
                "owner_start_time": resource_guard.process_start_time() + 1,
                "owner_token": "stale-owner-token",
                "inherited_cgroup": None,
                "sentinel": str(stale_sentinel),
            }
        ),
        encoding="utf-8",
    )
    stale_sentinel.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ResourceAdmissionError, match="containment proof"):
        resource_guard._reserve_dispatcher_heavy("fresh.scope", command)
    assert stale.exists()


def test_stale_inherited_reservation_requires_empty_execution_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )
    reservations = resource_guard._dispatcher_reservation_dir()
    stale_sentinel = resource_guard._runtime_root() / "stale-inherited.result.json"
    stale = reservations / "stale-inherited.scope.json"
    stale.write_text(
        json.dumps(
            {
                "schema": "repository-resource-guard-v1",
                "unit": "stale-inherited.scope",
                "kind": "heavy",
                "owner_pid": resource_guard.os.getpid(),
                "owner_start_time": resource_guard.process_start_time() + 1,
                "owner_token": "stale-owner-token",
                "inherited_cgroup": "/queue.scope",
                "execution_cgroup": "/queue.scope",
                "sentinel": str(stale_sentinel),
                "sentinel_base": {"unit": "stale-inherited.scope", "kind": "heavy"},
            }
        ),
        encoding="utf-8",
    )
    stale_sentinel.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        resource_guard,
        "_require_empty_execution_cgroup",
        lambda _path: (_ for _ in ()).throw(
            ResourceAdmissionError("execution cgroup remains populated")
        ),
    )

    with pytest.raises(ResourceAdmissionError, match="remains populated"):
        resource_guard._reserve_dispatcher_heavy("fresh.scope", command)
    assert stale.exists()


def test_standalone_binding_updates_owner_fenced_reservation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: _host_evidence())
    scope_directory = tmp_path / "scope"
    scope_directory.mkdir()
    (scope_directory / "cgroup.events").write_text("populated 0\n", encoding="ascii")
    (scope_directory / "cgroup.procs").write_text("", encoding="ascii")
    monkeypatch.setattr(
        resource_guard,
        "_canonical_cgroup_path",
        lambda _path: ("/fake.scope", scope_directory),
    )
    monkeypatch.setattr(
        resource_guard, "_cgroup_directory", lambda _path: scope_directory
    )
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )

    admission = resource_guard.admit_guarded(command)
    assert admission._dispatcher_reservation is not None
    reservation = admission._dispatcher_reservation
    try:
        admission.bind_execution_cgroup("/fake.scope")
        record = json.loads(reservation.read_text(encoding="utf-8"))
        assert record["execution_cgroup"] == "/fake.scope"
        assert record["sentinel_base"]["execution_cgroup"] == "/fake.scope"
    finally:
        admission.release()
    assert not reservation.exists()


def test_standalone_release_retains_reservation_when_scope_closure_is_uncertain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: _host_evidence())
    scope_directory = tmp_path / "scope"
    scope_directory.mkdir()
    monkeypatch.setattr(
        resource_guard,
        "_canonical_cgroup_path",
        lambda _path: ("/fake.scope", scope_directory),
    )
    monkeypatch.setattr(
        resource_guard, "_cgroup_directory", lambda _path: scope_directory
    )
    monkeypatch.setattr(
        resource_guard,
        "_require_empty_execution_cgroup",
        lambda _path: (_ for _ in ()).throw(
            ResourceAdmissionError("scope closure is unknown")
        ),
    )
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )

    admission = resource_guard.admit_guarded(command)
    reservation = admission._dispatcher_reservation
    assert reservation is not None
    admission.bind_execution_cgroup("/fake.scope")
    admission.release()

    assert reservation.exists()


def test_stop_scope_refuses_success_without_empty_containment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _Process:
        def wait(self, *, timeout: int) -> int:
            assert timeout == 30
            return 0

        def poll(self) -> int:
            return 0

    monkeypatch.setattr(
        resource_guard.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, "", ""),
    )
    monkeypatch.setattr(
        resource_guard,
        "_execution_cgroup_closed",
        lambda _unit, _path: False,
    )

    assert resource_guard._stop_scope("rm-test.scope", _Process(), "/fake.scope") is False


def test_terminal_state_cannot_override_populated_execution_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope_directory = tmp_path / "scope"
    scope_directory.mkdir()
    (scope_directory / "cgroup.events").write_text("populated 1\n", encoding="ascii")
    (scope_directory / "cgroup.procs").write_text("123\n", encoding="ascii")
    monkeypatch.setattr(
        resource_guard, "_cgroup_directory", lambda _path: scope_directory
    )
    monkeypatch.setattr(
        resource_guard,
        "_scope_is_inactive",
        lambda *_args: pytest.fail("terminal metadata cannot override live evidence"),
    )

    assert resource_guard._execution_cgroup_closed("rm-live.scope", "/fake.scope") is False


def test_terminal_state_cannot_override_unreadable_execution_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    scope_directory = tmp_path / "scope"
    scope_directory.mkdir()
    monkeypatch.setattr(
        resource_guard, "_cgroup_directory", lambda _path: scope_directory
    )
    monkeypatch.setattr(
        resource_guard,
        "_scope_is_inactive",
        lambda *_args: pytest.fail("terminal metadata cannot override unreadable evidence"),
    )

    assert resource_guard._execution_cgroup_closed("rm-unreadable.scope", "/fake.scope") is False


def test_terminal_state_can_close_a_genuinely_removed_execution_cgroup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_directory",
        lambda _path: (_ for _ in ()).throw(
            ResourceGuardError("inherited cgroup does not exist")
        ),
    )
    monkeypatch.setattr(resource_guard, "_scope_is_inactive", lambda *_args: True)

    assert resource_guard._execution_cgroup_closed("rm-removed.scope", "/fake.scope") is True


@pytest.mark.parametrize(
    "malformed",
    [
        "/user.slice/../rm-missing.scope",
        "../../rm-missing.scope",
        "/user.slice/\x00rm-missing.scope",
        "/user.slice/\nrm-missing.scope",
    ],
)
def test_malformed_execution_cgroup_never_reaches_terminal_fallback(
    malformed: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        resource_guard,
        "_scope_is_inactive",
        lambda *_args: pytest.fail("malformed cgroup paths cannot use terminal metadata"),
    )

    with pytest.raises(ResourceGuardError):
        resource_guard._validated_execution_cgroup(malformed, require_exists=False)
    assert resource_guard._execution_cgroup_path_absent(malformed) is False
    assert resource_guard._execution_cgroup_closed("rm-invalid.scope", malformed) is False


def test_temp_cgroup_parent_path_cannot_be_classified_as_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = str(tmp_path / "nested" / ".." / "missing.scope")
    monkeypatch.setattr(
        resource_guard,
        "_scope_is_inactive",
        lambda *_args: pytest.fail("malformed cgroup paths cannot use terminal metadata"),
    )

    with pytest.raises(ResourceGuardError):
        resource_guard._validated_execution_cgroup(malformed, require_exists=False)
    assert resource_guard._execution_cgroup_path_absent(malformed) is False
    assert resource_guard._execution_cgroup_closed("rm-invalid.scope", malformed) is False


def test_replaced_dispatcher_record_invalidates_the_live_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A path replacement cannot preserve an admission without its token."""

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    monkeypatch.setattr(resource_guard, "_inherited_cgroup_reasons", lambda *_args: [])
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: _host_evidence())
    command = resource_guard.prepare_command(
        ("cargo", "check"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="rust-build",
        timeout=60,
    )

    with resource_guard.admit_inherited(command, expected_cgroup="/queue.scope") as admission:
        assert admission.is_live()
        assert admission._dispatcher_reservation is not None
        record = json.loads(
            admission._dispatcher_reservation.read_text(encoding="utf-8")
        )
        record["owner_token"] = "replacement-token"
        admission._dispatcher_reservation.write_text(
            json.dumps(record), encoding="utf-8"
        )
        assert admission.is_live() is False
        with pytest.raises(ResourceGuardError, match="no longer live"):
            resource_guard.verified_inherited_admission()


def test_rust_fast_gate_is_routed_through_resource_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "Cargo.toml").write_text("[package]\nname='x'\nversion='0.1.0'\n")
    prepared = object()
    completed = subprocess.CompletedProcess(["pre-commit"], 0, "", "")
    run_result = SimpleNamespace(
        completed=completed,
        evidence=lambda: {"schema": "repository-resource-guard-v1"},
    )
    prepare = monkeypatch.setattr(gates, "prepare_command", lambda *a, **k: prepared)
    del prepare
    monkeypatch.setattr(gates, "run_guarded", lambda value: run_result)
    monkeypatch.setattr(gates, "_apply_heavy_gate_limits", lambda env: None)

    result = gates._run_pre_commit(str(tmp_path), "pre-commit")

    assert result.returncode == 0
    assert result.resource_evidence["schema"] == "repository-resource-guard-v1"


def test_inherited_queue_gate_stays_in_the_existing_cgroup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A queue admission must not nest a gate in a sibling systemd scope."""

    class _Process:
        returncode = 0

        def communicate(self, *, timeout: int) -> tuple[bytes, bytes]:
            assert timeout == 10
            return b"", b""

    class _Supervisor:
        def __init__(self) -> None:
            self.process = _Process()

        def spawn(self, *_args, **_kwargs) -> _Process:
            return self.process

    supervisor = _Supervisor()
    monkeypatch.setattr(gates, "_GATE_PROCESS_SUPERVISOR", supervisor)
    monkeypatch.setattr(
        gates,
        "run_guarded",
        lambda _command: pytest.fail("inherited queue gate opened a nested scope"),
    )
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    directory = Path("/sys/fs/cgroup/queue.scope")
    monkeypatch.setattr(resource_guard, "_cgroup_directory", lambda _path: directory)
    monkeypatch.setattr(
        resource_guard, "_cgroup_chain", lambda _directory: (directory,)
    )
    controls = {
        "memory.high": str(4 * 1024**3),
        "memory.max": str(6 * 1024**3),
        "memory.current": str(1024**3),
        "memory.swap.max": str(1024**3),
        "cpu.max": "200000 100000",
        "pids.max": "256",
    }
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_control",
        lambda _directory, name: controls[name],
    )
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: _host_evidence())
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )

    with resource_guard.admit_inherited(command, expected_cgroup="/queue.scope"):
        result = gates._run_gate_subprocess(
            ["pre-commit", "run"],
            str(tmp_path),
            {"RM_RESOURCE_GUARD": "heavy"},
            timeout=10,
        )

    assert result.returncode == 0
    assert resource_guard.inherited_admission_active() is False


def test_forged_inherited_gate_marker_cannot_bypass_admission(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    completed = subprocess.CompletedProcess(["pre-commit"], 0, "", "")
    monkeypatch.setattr(
        gates,
        "run_guarded",
        lambda _command: SimpleNamespace(completed=completed, evidence=lambda: {}),
    )
    monkeypatch.setattr(
        gates,
        "_GATE_PROCESS_SUPERVISOR",
        SimpleNamespace(spawn=lambda *_args, **_kwargs: pytest.fail("bypass")),
    )

    result = gates._run_gate_subprocess(
        ["pre-commit", "run"],
        str(tmp_path),
        {"RM_RESOURCE_GUARD": "heavy", "RM_RESOURCE_GUARD_INHERITED": "1"},
        timeout=10,
    )

    assert result.returncode == 0


def test_build_queue_uses_guard_even_for_a_light_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tree = tmp_path / "tree"
    tree.mkdir()
    prepared = object()
    completed = subprocess.CompletedProcess(["cargo"], 0, "", "")
    monkeypatch.setenv("RM_RESOURCE_GUARD_ROOT", str(tmp_path / "guard-root"))
    monkeypatch.setattr(build_queue, "prepare_command", lambda *a, **k: prepared)
    monkeypatch.setattr(
        build_queue,
        "run_guarded",
        lambda value: SimpleNamespace(completed=completed, evidence=lambda: {}),
    )
    spec = build_queue.BuildSpec(
        name="cargo-check",
        command=("cargo", "check"),
        resource_class="light-check",
        timeout=30,
    )

    build_queue._run_build_command(tree, spec)


def test_job_outcome_carries_guard_admission_and_completion_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RM_RESOURCE_GUARD_ROOT", str(tmp_path / "guard-root"))
    evidence = _host_evidence()
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: evidence)
    monkeypatch.setattr(
        resource_guard, "capture_host_evidence", lambda _command: evidence
    )
    result = job_outcome.run_stages(
        (("probe", "printf probe"),),
        workdir=tmp_path,
        log_dir=tmp_path / "logs",
        use_systemd=True,
    )

    assert result["ok"] is True
    evidence = result["stages"][0]["evidence"]
    assert evidence["schema"] == "repository-resource-guard-v1"
    assert evidence["execution_cgroup"].startswith("/")
    assert evidence["before"]["filesystems"]
    assert evidence["after"]["memory_available_bytes"] > 0


def test_guard_streams_noisy_output_into_a_bounded_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RM_RESOURCE_GUARD_ROOT", str(tmp_path / "guard-root"))
    evidence = _host_evidence()
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: evidence)
    monkeypatch.setattr(
        resource_guard, "capture_host_evidence", lambda _command: evidence
    )
    command = resource_guard.prepare_command(
        (sys.executable, "-c", "import sys; sys.stdout.write('x' * (3 * 1024**2))"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="light-check",
        timeout=60,
    )

    result = resource_guard.run_guarded(command)

    output = result.evidence()["output"]["stdout"]
    assert output["total_bytes"] == 3 * 1024**2
    assert output["retained_bytes"] == 2 * 1024**2
    assert output["truncated"] is True
    assert len(result.completed.stdout) == 64 * 1024


def test_inherited_queue_admission_refuses_before_launch_on_unbounded_scope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_directory",
        lambda _path: Path("/sys/fs/cgroup/queue.scope"),
    )
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_chain",
        lambda directory: (directory,),
    )
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_control",
        lambda _directory, _name: "max" if _name != "pids.max" else "256",
    )
    launched = False

    def fail_if_admitted(_command):
        nonlocal launched
        launched = True
        raise AssertionError("the queue child must not launch")

    monkeypatch.setattr(resource_guard, "_admit", fail_if_admitted)
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )

    with pytest.raises(ResourceAdmissionError, match="unbounded"):
        resource_guard.admit_inherited(command, expected_cgroup="/queue.scope")
    assert launched is False


def test_queue_handoff_accepts_only_structured_runner_entrypoints() -> None:
    common = [
        "--workspace-root",
        "/workspace",
        "--drain-deadline-seconds",
        "180",
        "--global-deadline-seconds",
        "3600",
    ]
    assert merge_queue._queue_runner_argv_is_authorized(
        [
            sys.executable,
            str(Path.home() / ".local" / "bin" / "repository-manager-merge-queue-runner"),
            *common,
        ]
    )
    assert merge_queue._queue_runner_argv_is_authorized(
        [sys.executable, "-m", "repository_manager.merge_queue_runner", *common]
    )
    assert not merge_queue._queue_runner_argv_is_authorized(
        [sys.executable, "-c", "merge_queue_runner", *common]
    )
    assert not merge_queue._queue_runner_argv_is_authorized(
        [sys.executable, "/opt/bin/repository-manager-merge-queue-runner", *common]
    )
    assert not merge_queue._queue_runner_argv_is_authorized(
        [
            sys.executable,
            "-m",
            "repository_manager.merge_queue_runner",
            "--phased-push",
            *common,
        ]
    )
    over_budget = [*common]
    over_budget[over_budget.index("180")] = "3600"
    assert not merge_queue._queue_runner_argv_is_authorized(
        [sys.executable, "-m", "repository_manager.merge_queue_runner", *over_budget]
    )


def test_inherited_queue_admission_accepts_the_installed_service_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    directory = Path("/sys/fs/cgroup/queue.scope")
    monkeypatch.setattr(resource_guard, "_cgroup_directory", lambda _path: directory)
    monkeypatch.setattr(
        resource_guard, "_cgroup_chain", lambda _directory: (directory,)
    )
    controls = {
        "memory.high": str(4 * 1024**3),
        "memory.max": str(6 * 1024**3),
        "memory.current": str(1024**3),
        "memory.swap.max": str(1024**3),
        "cpu.max": "200000 100000",
        "pids.max": "256",
    }
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_control",
        lambda _directory, name: controls[name],
    )
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )
    evidence = _host_evidence()
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: evidence)

    with resource_guard.admit_inherited(
        command, expected_cgroup="/queue.scope"
    ) as admission:
        assert admission.inherited_cgroup == "/queue.scope"
        assert admission.environment["TMPDIR"] == str(tmp_path / "scratch")
        assert resource_guard.verified_inherited_admission() is admission


def test_inherited_queue_admission_rejects_a_looser_service_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    directory = Path("/sys/fs/cgroup/queue.scope")
    monkeypatch.setattr(resource_guard, "_cgroup_directory", lambda _path: directory)
    monkeypatch.setattr(
        resource_guard, "_cgroup_chain", lambda _directory: (directory,)
    )
    controls = {
        "memory.high": str(8 * 1024**3),
        "memory.max": str(48 * 1024**3),
        "memory.current": str(1024**3),
        "memory.swap.max": str(8 * 1024**3),
        "cpu.max": "400000 100000",
        "pids.max": "1024",
    }
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_control",
        lambda _directory, name: controls[name],
    )
    monkeypatch.setattr(
        resource_guard,
        "_admit",
        lambda _command: pytest.fail("loose cgroup must refuse before host admission"),
    )
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )

    with pytest.raises(ResourceAdmissionError, match="exceeds the queue profile"):
        resource_guard.admit_inherited(command, expected_cgroup="/queue.scope")


def test_inherited_queue_admission_allows_tighter_service_envelope(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    directory = Path("/sys/fs/cgroup/queue.scope")
    parent = Path("/sys/fs/cgroup")
    monkeypatch.setattr(resource_guard, "_cgroup_directory", lambda _path: directory)
    monkeypatch.setattr(
        resource_guard, "_cgroup_chain", lambda _directory: (directory, parent)
    )
    controls = {
        (directory, "memory.high"): str(2 * 1024**3),
        (directory, "memory.max"): str(3 * 1024**3),
        (directory, "memory.current"): str(1024**3),
        (directory, "memory.swap.max"): str(512 * 1024**2),
        (directory, "cpu.max"): "50000 100000",
        (directory, "pids.max"): "128",
        (parent, "memory.high"): str(48 * 1024**3),
        (parent, "memory.max"): str(48 * 1024**3),
        (parent, "memory.current"): str(1024**3),
        (parent, "memory.swap.max"): str(8 * 1024**3),
        (parent, "cpu.max"): "400000 100000",
        (parent, "pids.max"): "1024",
    }
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_control",
        lambda directory_value, name: controls[(directory_value, name)],
    )
    evidence = _host_evidence()
    monkeypatch.setattr(resource_guard, "_admit", lambda _command: evidence)
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )

    with resource_guard.admit_inherited(
        command, expected_cgroup="/queue.scope"
    ) as admission:
        assert admission.inherited_cgroup == "/queue.scope"


def test_inherited_queue_admission_rejects_a_tight_service_without_headroom(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A finite service limit is unsafe when its current usage leaves no reserve."""

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(resource_guard, "_self_cgroup_path", lambda: "/queue.scope")
    directory = Path("/sys/fs/cgroup/queue.scope")
    monkeypatch.setattr(resource_guard, "_cgroup_directory", lambda _path: directory)
    monkeypatch.setattr(
        resource_guard, "_cgroup_chain", lambda _directory: (directory,)
    )
    controls = {
        "memory.high": str(2 * 1024**3),
        "memory.max": str(2 * 1024**3),
        "memory.current": str(1_900 * 1024**2),
        "memory.swap.max": str(512 * 1024**2),
        "cpu.max": "50000 100000",
        "pids.max": "128",
    }
    monkeypatch.setattr(
        resource_guard,
        "_cgroup_control",
        lambda _directory, name: controls[name],
    )
    monkeypatch.setattr(
        resource_guard,
        "_admit",
        lambda _command: pytest.fail("headroom refusal must precede host admission"),
    )
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )

    with pytest.raises(ResourceAdmissionError, match="headroom"):
        resource_guard.admit_inherited(command, expected_cgroup="/queue.scope")


def test_admission_releases_lease_after_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path / "run"))
    monkeypatch.setattr(resource_guard, "_mount_type", lambda _path: "xfs")
    monkeypatch.setattr(
        resource_guard,
        "_admit",
        lambda _command: (_ for _ in ()).throw(RuntimeError("probe failed")),
    )
    command = resource_guard.prepare_command(
        ("repository-manager", "--merge-queue", "run"),
        workdir=tmp_path,
        target_dir=tmp_path / "target",
        tmp_dir=tmp_path / "scratch",
        profile_name="merge-drain",
        timeout=180,
    )

    with pytest.raises(RuntimeError, match="probe failed"):
        resource_guard.admit_guarded(command)

    lock_path = resource_guard._runtime_root() / "heavy.lock"
    with lock_path.open("a+") as lock:
        resource_guard.fcntl.flock(
            lock.fileno(),
            resource_guard.fcntl.LOCK_EX | resource_guard.fcntl.LOCK_NB,
        )


def test_generic_queue_boundary_refuses_before_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(
        merge_queue,
        "lane_scope",
        lambda _path: SimpleNamespace(main_tree=repository),
    )
    monkeypatch.setattr(merge_queue, "_resolve_git_client", lambda git, _repo: git)
    monkeypatch.setattr(merge_queue, "_queue_runner_handoff_valid", lambda: True)
    monkeypatch.setattr(merge_queue, "current_cgroup_path", lambda: "/queue.scope")
    monkeypatch.setattr(merge_queue, "default_guard_paths", lambda *_args: (
        tmp_path / "target",
        tmp_path / "tmp",
    ))
    monkeypatch.setattr(merge_queue, "prepare_command", lambda *args, **kwargs: object())
    refused = ResourceAdmissionError("memory PSI is above the admission threshold")
    monkeypatch.setattr(
        merge_queue,
        "admit_inherited",
        lambda *args, **kwargs: (_ for _ in ()).throw(refused),
    )
    monkeypatch.setattr(
        merge_queue,
        "_drain_batch_under_lease",
        lambda *args, **kwargs: pytest.fail("drain started after admission refusal"),
    )

    with pytest.raises(ResourceAdmissionError, match="memory PSI"):
        merge_queue.run_queue(path=repository, git=object(), prune=False)


def test_generic_queue_refuses_an_unsupervised_direct_drain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    monkeypatch.setattr(
        merge_queue,
        "lane_scope",
        lambda _path: SimpleNamespace(main_tree=repository),
    )
    monkeypatch.setattr(merge_queue, "_resolve_git_client", lambda git, _repo: git)
    monkeypatch.setattr(
        merge_queue,
        "_drain_batch_under_lease",
        lambda *args, **kwargs: pytest.fail("unsupervised queue drain started"),
    )

    with pytest.raises(ResourceGuardError, match="supervised queue runner"):
        merge_queue.run_queue(path=repository, git=object(), prune=False)


def test_timed_queue_gate_terminates_the_shared_process_group_on_timeout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Process:
        returncode = -15

        def __init__(self) -> None:
            self.communicate_calls = 0

        def communicate(self, *, timeout: int) -> tuple[bytes, bytes]:
            self.communicate_calls += 1
            if self.communicate_calls == 1:
                raise subprocess.TimeoutExpired(["true"], timeout)
            assert timeout == merge_queue._GATE_CLEANUP_DRAIN_SECONDS
            return b"", b""

    class _Supervisor:
        def __init__(self) -> None:
            self.process = _Process()

        def spawn(self, *_args, **_kwargs) -> _Process:
            return self.process

    supervisor = _Supervisor()
    terminated = {"value": False}
    monkeypatch.setattr(merge_queue, "_queue_gate_supervisor", lambda: supervisor)
    def terminate(_process):  # noqa: ANN001
        terminated["value"] = True
        return True

    monkeypatch.setattr(merge_queue, "_queue_gate_terminate", terminate)

    result, _seconds = merge_queue._timed_run(
        ["true"], tmp_path, timeout=1, env={}
    )

    assert result is None
    assert terminated["value"] is True
    assert supervisor.process.communicate_calls == 2


def test_timed_queue_gate_retains_admission_when_cleanup_is_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Process:
        returncode = -15

        def communicate(self, *, timeout: int) -> tuple[bytes, bytes]:
            if timeout != 0:
                raise subprocess.TimeoutExpired(["true"], timeout)
            return b"", b""

    class _Supervisor:
        def __init__(self) -> None:
            self.process = _Process()

        def spawn(self, *_args, **_kwargs) -> _Process:
            return self.process

    retained: list[bool] = []
    admission = SimpleNamespace(
        retain_reservation_on_release=lambda: retained.append(True)
    )
    monkeypatch.setattr(
        merge_queue, "_queue_gate_supervisor", lambda: _Supervisor()
    )
    monkeypatch.setattr(merge_queue, "_queue_gate_terminate", lambda _process: False)
    monkeypatch.setattr(
        merge_queue, "verified_inherited_admission", lambda: admission
    )

    result, _seconds = merge_queue._timed_run(
        ["true"], tmp_path, timeout=1, env={}
    )

    assert isinstance(result, OSError)
    assert retained == [True]


def test_timed_queue_gate_rejects_unsafe_cargo_argv_before_spawn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target"
    temporary = tmp_path / "tmp"
    env = {"CARGO_TARGET_DIR": str(target), "TMPDIR": str(temporary)}
    monkeypatch.setattr(
        merge_queue,
        "_queue_gate_supervisor",
        lambda: pytest.fail("unsafe Cargo argv reached process spawn"),
    )

    with pytest.raises(ResourceGuardError, match="profile cap"):
        merge_queue._timed_run(
            ["cargo", "check", "-j16"], tmp_path, timeout=10, env=env
        )
    with pytest.raises(ResourceGuardError, match="target directory"):
        merge_queue._timed_run(
            [
                "cargo",
                "check",
                "--target-dir",
                str(tmp_path / "other-target"),
            ],
            tmp_path,
            timeout=10,
            env=env,
        )
    with pytest.raises(ResourceGuardError, match="direct fixed argv"):
        merge_queue._timed_run(
            ["bash", "-c", "cargo check -j16"],
            tmp_path,
            timeout=10,
            env=env,
        )


def test_queue_materialization_uses_the_admitted_tmp_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    admitted_root = tmp_path / "var-tmp"
    admitted_tmp = admitted_root / "lane" / "tmp"
    scope = SimpleNamespace(tree=repository)
    monkeypatch.setenv("AU_LANE_TEMP_ROOT", str(admitted_root))
    monkeypatch.setenv("TMPDIR", str(admitted_tmp))

    assert merge_queue._materialization_root(scope) == (
        admitted_tmp / "merge-queue-verify"
    )

    monkeypatch.setenv("TMPDIR", str(tmp_path / "escape"))
    with pytest.raises(ResourceGuardError, match="escapes"):
        merge_queue._materialization_root(scope)


def test_fast_gates_keep_the_admitted_au_paths_for_actual_gate_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    admitted_root = tmp_path / "var-tmp"
    admitted_target = admitted_root / "lane" / "target"
    admitted_tmp = admitted_root / "lane" / "tmp"
    monkeypatch.setenv("AU_LANE_TEMP_ROOT", str(admitted_root))
    monkeypatch.setenv("CARGO_TARGET_DIR", str(admitted_target))
    monkeypatch.setenv("TMPDIR", str(admitted_tmp))
    monkeypatch.setenv("PYTEST_XDIST_AUTO_NUM_WORKERS", "2")
    monkeypatch.setattr(
        merge_queue,
        "partitioned_paths",
        lambda _path: pytest.fail("AU guarded queue must not allocate a second path set"),
    )
    monkeypatch.setattr(
        merge_queue,
        "_environment_signature",
        lambda _tree, _config: "test-environment",
    )
    captured: dict[str, str] = {}

    def fake_run_gate(*_args, env, **_kwargs):  # noqa: ANN001
        captured.update(
            {
                "CARGO_TARGET_DIR": env["CARGO_TARGET_DIR"],
                "TMPDIR": env["TMPDIR"],
                "PRE_COMMIT_HOME": env["PRE_COMMIT_HOME"],
                "PYTEST_XDIST_AUTO_NUM_WORKERS": env[
                    "PYTEST_XDIST_AUTO_NUM_WORKERS"
                ],
            }
        )
        return merge_queue.Check("probe", ok=True, seconds=0.0)

    monkeypatch.setattr(merge_queue, "run_gate", fake_run_gate)
    scope = SimpleNamespace(tree=repository, main_tree=repository)
    config = merge_queue.QueueConfig(
        gates=(merge_queue.GateSpec(name="probe", command=("true",)),)
    )

    result = merge_queue.run_fast_gates(
        repository,
        repo=repository,
        base_ref="main",
        scope=scope,
        config=config,
        base_config=config,
        changed=[],
    )

    assert result.ok is True
    assert captured == {
        "CARGO_TARGET_DIR": str(admitted_target.resolve()),
        "TMPDIR": str(admitted_tmp.resolve()),
        "PRE_COMMIT_HOME": str((admitted_tmp / "merge-queue-precommit").resolve()),
        "PYTEST_XDIST_AUTO_NUM_WORKERS": "2",
    }
