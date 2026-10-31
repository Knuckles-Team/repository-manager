"""Windows portability of repository-manager (RM-GOVERNANCE-R032).

Every Windows branch is exercised on any host by monkeypatching the single
platform switches in :mod:`repository_manager.portable_fs` (``is_windows`` and
``native_dir_fd``), never by importing Windows-only modules.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import types
from pathlib import Path

import pytest

from repository_manager import job_outcome, mcp_server, merge_queue_runner
from repository_manager import portable_fs as pfs
from repository_manager.build_artifacts import _fsync_directory
from repository_manager.execution import executor
from repository_manager.execution.process_supervisor import ProcessSupervisor
from repository_manager.governance import lanes
from repository_manager.operation_boundary import (
    OperationBoundaryError,
    open_directory,
    pin_creation,
    read_at,
    write_at,
)


@pytest.fixture
def windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pfs, "is_windows", lambda: True)


@pytest.fixture
def emulated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend ``dir_fd`` and directory descriptors do not exist."""
    monkeypatch.setattr(pfs, "native_dir_fd", lambda: False)


# -- user directories (XDG has a Windows equivalent) -------------------------


@pytest.mark.usefixtures("windows")
def test_state_home_prefers_xdg_then_localappdata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))
    assert pfs.state_home() == tmp_path / "local"
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    assert pfs.state_home() == tmp_path / "xdg"


@pytest.mark.usefixtures("windows")
def test_config_home_uses_appdata_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert pfs.config_home() == tmp_path / "roaming"


@pytest.mark.usefixtures("windows")
def test_home_falls_back_to_userprofile(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("HOME", "XDG_STATE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert pfs.home_dir() == tmp_path
    assert pfs.state_home() == tmp_path / "AppData" / "Local"
    assert pfs.cache_home() == tmp_path / "AppData" / "Local"


def test_posix_defaults_unchanged(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pfs, "is_windows", lambda: False)
    for name in ("XDG_STATE_HOME", "XDG_CONFIG_HOME", "XDG_CACHE_HOME"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    assert pfs.state_home() == tmp_path / ".local" / "state"
    assert pfs.config_home() == tmp_path / ".config"
    assert pfs.cache_home() == tmp_path / ".cache"


@pytest.mark.usefixtures("windows")
def test_stores_resolve_default_path_through_state_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from repository_manager.capacity_store import CapacityStore
    from repository_manager.gate_ledger import GateLedger

    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert CapacityStore._resolve_store_path(None) == str(
        tmp_path / "repository-manager" / "capacity.sqlite3"
    )
    assert GateLedger._resolve_store_path(None) == str(
        tmp_path / "repository-manager" / "gate_ledger.sqlite3"
    )


# -- descriptor-anchored operations without dir_fd ---------------------------


@pytest.mark.usefixtures("emulated")
def test_emulated_handles_read_write_and_nest(tmp_path: Path) -> None:
    with open_directory(tmp_path / "a" / "b", create=True) as handle:
        assert pfs.is_token(handle.fd)
        write_at(handle.fd, "note.txt", b"payload\r\n")
        # Binary-exact: no newline translation.
        assert read_at(handle.fd, "note.txt") == b"payload\r\n"
        write_at(handle.fd, "note.txt", b"second")
        assert read_at(handle.fd, "note.txt") == b"second"
        assert not [p for p in (tmp_path / "a" / "b").iterdir() if p.suffix == ".tmp"]
    assert (tmp_path / "a" / "b" / "note.txt").read_bytes() == b"second"


@pytest.mark.usefixtures("emulated")
def test_emulated_handles_are_released_on_close(tmp_path: Path) -> None:
    handle = open_directory(tmp_path)
    fds = handle.owned_fds
    assert fds and all(pfs.is_token(fd) for fd in fds)
    handle.close()
    assert not any(pfs.is_token(fd) for fd in fds)


@pytest.mark.usefixtures("emulated")
def test_emulated_open_rejects_symlink_component(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover - host lacks symlinks
        pytest.skip("symlinks unavailable on this host")
    with pytest.raises(OperationBoundaryError):
        open_directory(link / "child")


@pytest.mark.usefixtures("emulated")
def test_emulated_pin_creation_and_children_pass_no_fds(tmp_path: Path) -> None:
    with open_directory(tmp_path) as root:
        with pin_creation(root, tmp_path / "x" / "y") as parent:
            assert parent.pass_fds == ()
            assert parent.proc_path == str(tmp_path / "x")
    assert (tmp_path / "x").is_dir()


@pytest.mark.usefixtures("emulated")
def test_emulated_cleanup_scandir_and_rmdir(tmp_path: Path) -> None:
    (tmp_path / "d").mkdir()
    (tmp_path / "d" / "f").write_text("x")
    with open_directory(tmp_path) as root:
        fd = pfs.open_directory("d", root.fd)
        try:
            assert pfs.scandir_names(fd) == ["f"]
            assert pfs.fstat(fd).st_mode
            pfs.unlink("f", dir_fd=fd)
            pfs.fsync_directory(fd)
        finally:
            pfs.close(fd)
        pfs.rmdir("d", dir_fd=root.fd)
    assert not (tmp_path / "d").exists()


@pytest.mark.usefixtures("emulated")
def test_open_root_is_the_anchor() -> None:
    fd = pfs.open_root()
    try:
        assert pfs.fd_path(fd) == (Path(os.getcwd()).anchor or os.sep)
    finally:
        pfs.close(fd)


def test_posix_handles_are_real_descriptors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    if not pfs.native_dir_fd():
        pytest.skip("host has no dir_fd support")
    with open_directory(tmp_path) as handle:
        assert not pfs.is_token(handle.fd)
        assert pfs.fd_path(handle.fd) == f"/proc/self/fd/{handle.fd}"


@pytest.mark.usefixtures("windows")
def test_popen_fd_kwargs_empty_on_windows() -> None:
    assert pfs.popen_fd_kwargs((3, 4)) == {}
    assert "creationflags" in pfs.new_group_kwargs()
    assert "start_new_session" not in pfs.new_group_kwargs()


def test_popen_fd_kwargs_posix_passes_real_fds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pfs, "is_windows", lambda: False)
    assert pfs.popen_fd_kwargs((3, 4)) == {"pass_fds": (3, 4)}
    assert pfs.new_group_kwargs() == {"start_new_session": True}


# -- processes / signals -----------------------------------------------------


class _Proc:
    pid = 4242

    def __init__(self, running: bool = True) -> None:
        self.running = running
        self.calls: list[str] = []

    def poll(self) -> int | None:
        return None if self.running else 0

    def terminate(self) -> None:
        self.calls.append("terminate")

    def kill(self) -> None:
        self.calls.append("kill")


@pytest.mark.usefixtures("windows")
def test_windows_graceful_stop_terminates() -> None:
    proc = _Proc()
    pfs.terminate_process_group(proc)  # type: ignore[arg-type]
    assert proc.calls == ["terminate"]


@pytest.mark.usefixtures("windows")
def test_windows_forced_stop_kills_tree_with_taskkill(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[list[str]] = []

    def fake_run(argv: list[str], **_kw: object) -> subprocess.CompletedProcess[bytes]:
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(pfs.subprocess, "run", fake_run)
    proc = _Proc()
    pfs.terminate_process_group(proc, force=True)  # type: ignore[arg-type]
    assert seen == [["taskkill", "/T", "/F", "/PID", "4242"]]
    assert proc.calls == []


@pytest.mark.usefixtures("windows")
def test_windows_forced_stop_falls_back_to_kill_when_taskkill_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        pfs.subprocess,
        "run",
        lambda argv, **_kw: subprocess.CompletedProcess(argv, 128),
    )
    proc = _Proc()
    pfs.terminate_process_group(proc, force=True)  # type: ignore[arg-type]
    assert proc.calls == ["kill"]


@pytest.mark.usefixtures("windows")
def test_windows_stop_of_exited_process_is_a_noop() -> None:
    proc = _Proc(running=False)
    pfs.terminate_process_group(proc, force=True)  # type: ignore[arg-type]
    assert proc.calls == []


def test_force_signal_without_sigkill(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(signal, "SIGKILL", raising=False)
    value = pfs.force_signal()
    assert value != signal.SIGTERM


@pytest.mark.usefixtures("windows")
def test_supervisor_spawn_and_signal_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def factory(argv: tuple[str, ...], **kwargs: object) -> _Proc:
        captured.update(kwargs)
        return _Proc()

    supervisor = ProcessSupervisor(popen_factory=factory)  # type: ignore[arg-type]
    supervisor.spawn([sys.executable, "-V"], cwd=tmp_path, env={})
    assert "creationflags" in captured and "start_new_session" not in captured
    # No killpg on Windows: group signals map to terminate/kill on the process.
    monkeypatch.delattr(os, "killpg", raising=False)
    proc = _Proc()
    supervisor._signal_group(proc, signal.SIGTERM)  # type: ignore[arg-type]
    supervisor._signal_group(proc, pfs.force_signal())  # type: ignore[arg-type]
    assert proc.calls == ["terminate", "kill"]


def test_merge_queue_runner_without_procfs_uses_portable_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(merge_queue_runner, "_procfs_available", lambda: False)
    stops: list[bool] = []
    monkeypatch.setattr(
        pfs, "terminate_process_group", lambda _p, *, force=False: stops.append(force)
    )
    proc = _Proc()
    observed: set[int] = set()
    merge_queue_runner._signal_process_tree(proc, signal.SIGTERM, observed)  # type: ignore[arg-type]
    merge_queue_runner._signal_process_tree(proc, pfs.force_signal(), observed)  # type: ignore[arg-type]
    assert stops == [False, True]


def test_merge_queue_runner_tracks_root_liveness_without_procfs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(merge_queue_runner, "_procfs_available", lambda: False)
    proc = _Proc()
    monkeypatch.setitem(merge_queue_runner._ROOT_PROCESSES, proc.pid, proc)
    assert merge_queue_runner._process_exists(proc.pid) is True
    proc.running = False
    assert merge_queue_runner._process_exists(proc.pid) is False
    assert merge_queue_runner._process_exists(999_999) is False


def test_cpu_count_without_affinity(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(os, "cpu_count", lambda: 6)
    assert pfs.cpu_count() == 6


def test_hostname_without_uname(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "uname", raising=False)
    assert pfs.hostname()


# -- links -------------------------------------------------------------------


@pytest.mark.usefixtures("windows")
def test_dir_link_falls_back_to_junction_without_privilege(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def denied(*_a: object, **_k: object) -> None:
        err = OSError("privilege not held")
        err.winerror = 1314  # type: ignore[attr-defined]
        raise err

    made: list[tuple[str, str]] = []
    fake = types.SimpleNamespace(
        CreateJunction=lambda src, dst: made.append((src, dst))
    )
    monkeypatch.setattr(os, "symlink", denied)
    monkeypatch.setitem(sys.modules, "_winapi", fake)
    pfs.create_dir_link(tmp_path / "link", tmp_path / "target")
    assert made == [(str(tmp_path / "target"), str(tmp_path / "link"))]


@pytest.mark.usefixtures("windows")
def test_dir_link_other_errors_propagate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def broken(*_a: object, **_k: object) -> None:
        raise PermissionError("nope")

    monkeypatch.setattr(os, "symlink", broken)
    with pytest.raises(PermissionError):
        pfs.create_dir_link(tmp_path / "link", tmp_path / "target")


def test_dir_link_posix_creates_symlink(tmp_path: Path) -> None:
    target = tmp_path / "t"
    target.mkdir()
    link = tmp_path / "l"
    pfs.create_dir_link(link, target)
    assert pfs.is_link(link)
    assert pfs.read_link(link) == str(target)


@pytest.mark.usefixtures("windows")
def test_replace_link_on_windows_removes_existing_link_first(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    (tmp_path / "old").mkdir()
    (tmp_path / "new").mkdir()
    link = tmp_path / "link"
    try:
        link.symlink_to(tmp_path / "old", target_is_directory=True)
    except (OSError, NotImplementedError):  # pragma: no cover
        pytest.skip("symlinks unavailable on this host")
    staged = tmp_path / "staged"
    staged.symlink_to(tmp_path / "new", target_is_directory=True)
    removed: list[str] = []
    real_rmdir = os.rmdir

    def rmdir(path: str | os.PathLike[str]) -> None:
        removed.append(os.fspath(path))
        # POSIX cannot rmdir a symlink; the real call is what Windows allows.
        os.unlink(path)

    monkeypatch.setattr(os, "rmdir", rmdir)
    pfs.replace_link(staged, link)
    monkeypatch.setattr(os, "rmdir", real_rmdir)
    assert removed == [str(link)]
    assert pfs.read_link(link) == str(tmp_path / "new")


@pytest.mark.usefixtures("windows")
def test_read_link_strips_extended_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "readlink", lambda _p: "\\\\?\\C:\\work\\repo")
    assert pfs.read_link("x") == "C:\\work\\repo"
    monkeypatch.setattr(os, "readlink", lambda _p: "\\\\?\\UNC\\host\\share")
    assert pfs.read_link("x") == "\\\\?\\UNC\\host\\share"


# -- flags, fsync, no-follow --------------------------------------------------


def test_file_flags_adds_binary_where_defined(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(os, "O_BINARY", 0x8000, raising=False)
    assert pfs.file_flags(os.O_RDONLY) & 0x8000


@pytest.mark.usefixtures("windows")
def test_directory_fsync_is_noop_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def forbidden(*_a: object, **_k: object) -> int:
        raise AssertionError("a directory must not be opened on Windows")

    monkeypatch.setattr(os, "open", forbidden)
    pfs.fsync_directory_path(tmp_path)
    _fsync_directory(tmp_path)


def test_directory_fsync_flushes_on_posix(tmp_path: Path) -> None:
    if pfs.is_windows():  # pragma: no cover
        pytest.skip("posix only")
    pfs.fsync_directory_path(tmp_path)
    with pytest.raises(FileNotFoundError):
        pfs.fsync_directory_path(tmp_path / "absent")


@pytest.mark.usefixtures("windows")
def test_no_follow_flag_windows_has_none(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    assert lanes._no_follow_flag("x") == 0
    assert lanes._lease_publication_flags() & os.O_EXCL


def test_no_follow_flag_refuses_elsewhere_without_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(pfs, "is_windows", lambda: False)
    monkeypatch.delattr(os, "O_NOFOLLOW", raising=False)
    with pytest.raises(lanes.LaneArbitrationError):
        lanes._no_follow_flag("x")


# -- paths and command construction -------------------------------------------


def test_path_containment_is_case_insensitive_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ntpath

    monkeypatch.setattr(os.path, "normcase", ntpath.normcase)
    monkeypatch.setattr(os, "sep", "\\")
    assert mcp_server._is_within("C:\\Work\\Repo\\x", "c:\\work\\repo")
    assert mcp_server._is_within("c:\\work\\repo", "C:\\WORK\\REPO")
    assert not mcp_server._is_within("C:\\Work\\Repository", "c:\\work\\repo")


def test_path_containment_is_case_sensitive_on_posix() -> None:
    assert mcp_server._is_within("/a/b/c", "/a/b")
    assert not mcp_server._is_within("/a/bc", "/a/b")
    if os.name == "posix":
        assert not mcp_server._is_within("/A/b", "/a/b")


@pytest.mark.usefixtures("windows")
def test_stage_command_uses_comspec_on_windows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("COMSPEC", "C:\\Windows\\System32\\cmd.exe")
    argv = job_outcome._stage_argv(
        "echo hi",
        unit="u",
        workdir=tmp_path,
        profile=None,  # type: ignore[arg-type]
        environ={},
        use_systemd=False,
    )
    assert argv == ["C:\\Windows\\System32\\cmd.exe", "/c", "echo hi"]


def test_stage_command_uses_bash_on_posix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(pfs, "is_windows", lambda: False)
    argv = job_outcome._stage_argv(
        "echo hi",
        unit="u",
        workdir=tmp_path,
        profile=None,  # type: ignore[arg-type]
        environ={},
        use_systemd=False,
    )
    assert argv == ["bash", "-c", "echo hi"]


def test_child_environment_keeps_windows_process_variables() -> None:
    names = executor._OPERATIONAL_ENV_NAMES
    for required in ("SYSTEMROOT", "COMSPEC", "USERPROFILE", "APPDATA", "LOCALAPPDATA"):
        assert required in names


def test_temp_root_is_not_hard_coded() -> None:
    import tempfile

    assert pfs.temp_root() == Path(tempfile.gettempdir())
