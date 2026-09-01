"""Install the versioned merge-queue runner and its user units atomically."""

from __future__ import annotations

import argparse
import hashlib
import os
import shlex
import subprocess  # nosec B404 - systemctl argv is fixed
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from repository_manager.merge_queue_runner import (
    DEFAULT_DRAIN_DEADLINE_SECONDS,
    DEFAULT_GLOBAL_DRAIN_DEADLINE_SECONDS,
    DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
    DEFAULT_HEAVY_TIMEOUT_SECONDS,
    DEFAULT_MAX_QUEUE_AGE_SECONDS,
    SYSTEMD_TIMEOUT_MARGIN_SECONDS,
)

DEFAULT_RUNNER_NAME = "repository-manager-merge-queue-runner"
SERVICE_NAME = "merge-queue-runner.service"
TIMER_NAME = "merge-queue-runner.timer"
HEAVY_SERVICE_NAME = "phased-push-runner.service"
HEAVY_TIMER_NAME = "phased-push-runner.timer"
_TEMPLATE_MARKERS = (
    "@PYTHON_EXECUTABLE@",
    "@RUNNER_PATH@",
    "@WORKSPACE_ROOT@",
    "@MANIFEST_PATH@",
    "@MAX_AGE_SECONDS@",
    "@DRAIN_DEADLINE_SECONDS@",
    "@GLOBAL_DRAIN_DEADLINE_SECONDS@",
    "@SERVICE_TIMEOUT_SECONDS@",
    "@HEAVY_TIMEOUT_SECONDS@",
    "@HEAVY_GATE_TIMEOUT_SECONDS@",
)


class MergeQueueRunnerInstallError(RuntimeError):
    """The runner bundle could not be installed and verified."""


@dataclass(frozen=True)
class _Snapshot:
    path: Path
    content: bytes | None
    mode: int | None


@dataclass(frozen=True)
class _InstallPaths:
    """Resolved source and destination paths for one runner bundle."""

    root: Path
    runner_source: Path
    service_source: Path
    timer_source: Path
    heavy_service_source: Path
    heavy_timer_source: Path
    runner_target: Path
    units: Path
    python_executable: Path


@dataclass(frozen=True)
class _Payload:
    """One atomically replaced file and its desired mode."""

    name: str
    source: Path
    destination: Path
    content: bytes
    mode: int


@dataclass(frozen=True)
class InstalledArtifact:
    """Hash evidence for one source payload and its installed destination."""

    name: str
    source: Path
    destination: Path
    source_sha256: str
    installed_sha256: str

    @property
    def verified(self) -> bool:
        return self.source_sha256 == self.installed_sha256


@dataclass(frozen=True)
class InstallReport:
    """Complete proof returned after an installation transaction."""

    artifacts: tuple[InstalledArtifact, ...]
    daemon_reloaded: bool

    @property
    def verified(self) -> bool:
        return self.daemon_reloaded and all(item.verified for item in self.artifacts)

    def as_dict(self) -> dict[str, object]:
        return {
            "verified": self.verified,
            "daemon_reloaded": self.daemon_reloaded,
            "artifacts": [
                {
                    "name": item.name,
                    "source": str(item.source),
                    "destination": str(item.destination),
                    "source_sha256": item.source_sha256,
                    "installed_sha256": item.installed_sha256,
                    "verified": item.verified,
                }
                for item in self.artifacts
            ],
        }


def _package_path(name: str) -> Path:
    path = Path(__file__).resolve().parent / "systemd" / name
    if path.is_symlink() or not path.is_file():
        raise MergeQueueRunnerInstallError(
            f"runner asset is missing or symlinked: {path}"
        )
    return path


def _validate_path(path: Path, *, label: str) -> Path:
    if not path.is_absolute():
        path = path.resolve()
    if any(ord(char) < 0x20 for char in str(path)):
        raise MergeQueueRunnerInstallError(f"{label} contains a control character")
    current = path
    while True:
        if current.is_symlink():
            raise MergeQueueRunnerInstallError(
                f"{label} must not use a symbolic link path: {current}"
            )
        if current == current.parent:
            break
        current = current.parent
    return path


def _systemd_token(value: Path) -> str:
    """Quote one literal systemd ExecStart/Environment token."""

    return shlex.quote(str(value))


def _validated_unit_settings(
    *,
    max_age_seconds: int,
    drain_deadline_seconds: int,
    global_deadline_seconds: int,
    service_timeout_seconds: int | None,
    heavy_timeout_seconds: int,
    heavy_gate_timeout_seconds: int,
) -> int:
    """Validate independent queue/heavy budgets and return the unit timeout."""

    if any(
        value <= 0
        for value in (
            max_age_seconds,
            drain_deadline_seconds,
            global_deadline_seconds,
            heavy_timeout_seconds,
            heavy_gate_timeout_seconds,
        )
    ):
        raise MergeQueueRunnerInstallError("runner settings must be positive integers")
    if global_deadline_seconds < drain_deadline_seconds:
        raise MergeQueueRunnerInstallError(
            "global deadline must be at least the per-repository drain deadline"
        )
    resolved_timeout = service_timeout_seconds
    if resolved_timeout is None:
        resolved_timeout = global_deadline_seconds + SYSTEMD_TIMEOUT_MARGIN_SECONDS
    if resolved_timeout <= 0:
        raise MergeQueueRunnerInstallError("runner settings must be positive integers")
    if resolved_timeout < global_deadline_seconds:
        raise MergeQueueRunnerInstallError(
            "systemd timeout must be at least the global runner deadline"
        )
    return resolved_timeout


def render_unit(
    template: Path,
    *,
    python_executable: Path,
    runner_path: Path,
    workspace_root: Path,
    max_age_seconds: int = DEFAULT_MAX_QUEUE_AGE_SECONDS,
    drain_deadline_seconds: int = DEFAULT_DRAIN_DEADLINE_SECONDS,
    global_deadline_seconds: int = DEFAULT_GLOBAL_DRAIN_DEADLINE_SECONDS,
    service_timeout_seconds: int | None = None,
    heavy_timeout_seconds: int = DEFAULT_HEAVY_TIMEOUT_SECONDS,
    heavy_gate_timeout_seconds: int = DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
) -> bytes:
    """Render a unit with explicit paths and bounded runner settings."""

    service_timeout_seconds = _validated_unit_settings(
        max_age_seconds=max_age_seconds,
        drain_deadline_seconds=drain_deadline_seconds,
        global_deadline_seconds=global_deadline_seconds,
        service_timeout_seconds=service_timeout_seconds,
        heavy_timeout_seconds=heavy_timeout_seconds,
        heavy_gate_timeout_seconds=heavy_gate_timeout_seconds,
    )
    if not python_executable.is_file() or not os.access(python_executable, os.X_OK):
        raise MergeQueueRunnerInstallError(
            f"python executable is unavailable or not executable: {python_executable}"
        )
    try:
        content = template.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise MergeQueueRunnerInstallError(
            f"cannot read unit template {template}: {exc}"
        ) from exc
    rendered = content.replace("@PYTHON_EXECUTABLE@", _systemd_token(python_executable))
    rendered = rendered.replace("@RUNNER_PATH@", _systemd_token(runner_path))
    rendered = rendered.replace("@WORKSPACE_ROOT@", _systemd_token(workspace_root))
    rendered = rendered.replace(
        "@MANIFEST_PATH@", _systemd_token(workspace_root / "workspace.yml")
    )
    rendered = rendered.replace("@MAX_AGE_SECONDS@", str(max_age_seconds))
    rendered = rendered.replace("@DRAIN_DEADLINE_SECONDS@", str(drain_deadline_seconds))
    rendered = rendered.replace(
        "@GLOBAL_DRAIN_DEADLINE_SECONDS@", str(global_deadline_seconds)
    )
    rendered = rendered.replace(
        "@SERVICE_TIMEOUT_SECONDS@", str(service_timeout_seconds)
    )
    rendered = rendered.replace("@HEAVY_TIMEOUT_SECONDS@", str(heavy_timeout_seconds))
    rendered = rendered.replace(
        "@HEAVY_GATE_TIMEOUT_SECONDS@", str(heavy_gate_timeout_seconds)
    )
    unresolved = [marker for marker in _TEMPLATE_MARKERS if marker in rendered]
    if unresolved:
        raise MergeQueueRunnerInstallError(
            f"unit template {template} has unresolved markers: {', '.join(unresolved)}"
        )
    return rendered.encode("utf-8")


def _digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _stage(path: Path, content: bytes, mode: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _snapshot(path: Path) -> _Snapshot:
    if path.is_symlink():
        raise MergeQueueRunnerInstallError(
            f"installation destination is symlinked: {path}"
        )
    if not path.exists():
        return _Snapshot(path, None, None)
    if not path.is_file():
        raise MergeQueueRunnerInstallError(
            f"installation destination is not a file: {path}"
        )
    return _Snapshot(path, path.read_bytes(), path.stat().st_mode & 0o777)


def _path_value(value: str | Path | None, default: Path) -> Path:
    """Choose a caller path or a package/user default."""

    return Path(value).expanduser() if value is not None else default


def _resolve_install_paths(
    workspace_root: str | Path,
    *,
    bin_path: str | Path | None,
    unit_directory: str | Path | None,
    source_runner: str | Path | None,
    service_template: str | Path | None,
    timer_template: str | Path | None,
    heavy_service_template: str | Path | None,
    heavy_timer_template: str | Path | None,
    python_executable: str | Path | None,
) -> _InstallPaths:
    """Validate every source and destination before staging any file."""

    root = _validate_path(Path(workspace_root).expanduser(), label="workspace root")
    if not root.is_dir():
        raise MergeQueueRunnerInstallError(f"workspace root is not a directory: {root}")
    runner_source = _validate_path(
        _path_value(source_runner, Path(__file__).resolve()), label="runner source"
    )
    service_source = _validate_path(
        _path_value(service_template, _package_path(SERVICE_NAME)),
        label="service template",
    )
    timer_source = _validate_path(
        _path_value(timer_template, _package_path(TIMER_NAME)),
        label="timer template",
    )
    heavy_service_source = _validate_path(
        _path_value(heavy_service_template, _package_path(HEAVY_SERVICE_NAME)),
        label="heavy service template",
    )
    heavy_timer_source = _validate_path(
        _path_value(heavy_timer_template, _package_path(HEAVY_TIMER_NAME)),
        label="heavy timer template",
    )
    runner_target = _validate_path(
        _path_value(bin_path, Path.home() / ".local" / "bin" / DEFAULT_RUNNER_NAME),
        label="runner destination",
    )
    units = _validate_path(
        _path_value(unit_directory, Path.home() / ".config" / "systemd" / "user"),
        label="unit directory",
    )
    python_candidate = _path_value(python_executable, Path(sys.executable).resolve())
    try:
        python_candidate = python_candidate.resolve(strict=True)
    except OSError as exc:
        raise MergeQueueRunnerInstallError(
            f"python executable cannot be resolved: {python_candidate}: {exc}"
        ) from exc
    python_path = _validate_path(python_candidate, label="python executable")
    if not python_path.is_file() or not os.access(python_path, os.X_OK):
        raise MergeQueueRunnerInstallError(
            f"python executable is unavailable or not executable: {python_path}"
        )
    return _InstallPaths(
        root,
        runner_source,
        service_source,
        timer_source,
        heavy_service_source,
        heavy_timer_source,
        runner_target,
        units,
        python_path,
    )


def _read_runner_source(path: Path) -> bytes:
    """Read the executable source before beginning the transaction."""

    if not path.is_file() or not os.access(path, os.R_OK):
        raise MergeQueueRunnerInstallError(f"runner source is unreadable: {path}")
    try:
        return path.read_bytes()
    except OSError as exc:
        raise MergeQueueRunnerInstallError(
            f"runner source could not be read: {path}: {exc}"
        ) from exc


def _build_payloads(
    paths: _InstallPaths,
    *,
    max_age_seconds: int,
    drain_deadline_seconds: int,
    global_deadline_seconds: int,
    heavy_timeout_seconds: int,
    heavy_gate_timeout_seconds: int,
) -> tuple[_Payload, ...]:
    """Materialize all source payloads before replacing any destination."""

    runner_content = _read_runner_source(paths.runner_source)
    service_content = render_unit(
        paths.service_source,
        python_executable=paths.python_executable,
        runner_path=paths.runner_target,
        workspace_root=paths.root,
        max_age_seconds=max_age_seconds,
        drain_deadline_seconds=drain_deadline_seconds,
        global_deadline_seconds=global_deadline_seconds,
    )
    timer_content = render_unit(
        paths.timer_source,
        python_executable=paths.python_executable,
        runner_path=paths.runner_target,
        workspace_root=paths.root,
        max_age_seconds=max_age_seconds,
        drain_deadline_seconds=drain_deadline_seconds,
        global_deadline_seconds=global_deadline_seconds,
    )
    heavy_service_content = render_unit(
        paths.heavy_service_source,
        python_executable=paths.python_executable,
        runner_path=paths.runner_target,
        workspace_root=paths.root,
        heavy_timeout_seconds=heavy_timeout_seconds,
        heavy_gate_timeout_seconds=heavy_gate_timeout_seconds,
    )
    heavy_timer_content = render_unit(
        paths.heavy_timer_source,
        python_executable=paths.python_executable,
        runner_path=paths.runner_target,
        workspace_root=paths.root,
        heavy_timeout_seconds=heavy_timeout_seconds,
        heavy_gate_timeout_seconds=heavy_gate_timeout_seconds,
    )
    return (
        _Payload(
            "runner", paths.runner_source, paths.runner_target, runner_content, 0o755
        ),
        _Payload(
            "service",
            paths.service_source,
            paths.units / SERVICE_NAME,
            service_content,
            0o644,
        ),
        _Payload(
            "timer", paths.timer_source, paths.units / TIMER_NAME, timer_content, 0o644
        ),
        _Payload(
            "heavy-service",
            paths.heavy_service_source,
            paths.units / HEAVY_SERVICE_NAME,
            heavy_service_content,
            0o644,
        ),
        _Payload(
            "heavy-timer",
            paths.heavy_timer_source,
            paths.units / HEAVY_TIMER_NAME,
            heavy_timer_content,
            0o644,
        ),
    )


def _stage_payloads(payloads: tuple[_Payload, ...]) -> tuple[tuple[Path, Path], ...]:
    """Write each payload to a same-directory temporary file."""

    staged: list[tuple[Path, Path]] = []
    try:
        for payload in payloads:
            staged.append(
                (
                    _stage(payload.destination, payload.content, payload.mode),
                    payload.destination,
                )
            )
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return tuple(staged)


def _replace_payloads(
    staged: tuple[tuple[Path, Path], ...], snapshots: tuple[_Snapshot, ...]
) -> tuple[_Snapshot, ...]:
    """Replace staged files and track each successful replacement for rollback."""

    snapshot_by_path = {snapshot.path: snapshot for snapshot in snapshots}
    replaced: list[_Snapshot] = []
    for temporary, target in staged:
        os.replace(temporary, target)
        replaced.append(snapshot_by_path[target])
        _fsync_directory(target.parent)
    return tuple(replaced)


def _artifacts(payloads: tuple[_Payload, ...]) -> tuple[InstalledArtifact, ...]:
    """Hash every installed destination against its rendered source payload."""

    try:
        return tuple(
            InstalledArtifact(
                name=payload.name,
                source=payload.source,
                destination=payload.destination,
                source_sha256=_digest(payload.content),
                installed_sha256=_digest(payload.destination.read_bytes()),
            )
            for payload in payloads
        )
    except OSError as exc:
        raise MergeQueueRunnerInstallError(
            f"installed runner bundle could not be read for verification: {exc}"
        ) from exc


def _require_verified(artifacts: tuple[InstalledArtifact, ...], message: str) -> None:
    """Fail the transaction when any source/destination hash differs."""

    if not all(item.verified for item in artifacts):
        raise MergeQueueRunnerInstallError(message)


def _rollback_transaction(
    staged: tuple[tuple[Path, Path], ...],
    replaced: tuple[_Snapshot, ...],
    *,
    reload: bool,
    systemctl: str,
) -> bool:
    """Remove temporary files and restore replaced destinations."""

    for temporary, _ in staged:
        temporary.unlink(missing_ok=True)
    if not replaced:
        return True
    rollback_ok = _restore(replaced)
    if reload:
        try:
            _daemon_reload(systemctl)
        except MergeQueueRunnerInstallError:
            rollback_ok = False
    return rollback_ok


def _transaction_error(
    exc: Exception, rollback_ok: bool
) -> MergeQueueRunnerInstallError:
    """Preserve the original failure while reporting rollback health."""

    suffix = "; rollback was incomplete" if not rollback_ok else ""
    if isinstance(exc, MergeQueueRunnerInstallError):
        return MergeQueueRunnerInstallError(f"{exc}{suffix}")
    return MergeQueueRunnerInstallError(f"runner installation failed: {exc}{suffix}")


def _perform_install(
    payloads: tuple[_Payload, ...],
    snapshots: tuple[_Snapshot, ...],
    *,
    systemctl: str,
    reload: bool,
) -> InstallReport:
    """Execute the replace, reload, and post-reload hash-proof transaction."""

    staged: tuple[tuple[Path, Path], ...] = ()
    replaced: tuple[_Snapshot, ...] = ()
    try:
        staged = _stage_payloads(payloads)
        replaced = _replace_payloads(staged, snapshots)
        installed = _artifacts(payloads)
        _require_verified(installed, "installed runner bundle failed hash verification")
        if reload:
            _daemon_reload(systemctl)
        verified_installed = _artifacts(payloads)
        _require_verified(
            verified_installed, "installed runner bundle changed during verification"
        )
        return InstallReport(verified_installed, reload)
    except Exception as exc:
        rollback_ok = _rollback_transaction(
            staged, replaced, reload=reload, systemctl=systemctl
        )
        raise _transaction_error(exc, rollback_ok) from exc


def _restore(snapshots: tuple[_Snapshot, ...]) -> bool:
    restored = True
    for snapshot in reversed(snapshots):
        try:
            if snapshot.content is None:
                snapshot.path.unlink(missing_ok=True)
            else:
                temporary = _stage(
                    snapshot.path,
                    snapshot.content,
                    snapshot.mode if snapshot.mode is not None else 0o644,
                )
                os.replace(temporary, snapshot.path)
            _fsync_directory(snapshot.path.parent)
        except OSError:
            # The caller reports the original failure; this best-effort rollback
            # is still preferable to leaving a half-written unit bundle.
            restored = False
    return restored


def _daemon_reload(systemctl: str) -> None:
    try:
        result = subprocess.run(
            [systemctl, "--user", "daemon-reload"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise MergeQueueRunnerInstallError(
            f"systemd daemon-reload could not run: {exc}"
        ) from exc
    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()[:500]
        raise MergeQueueRunnerInstallError(
            f"systemd daemon-reload failed with status {result.returncode}: {detail}"
        )


def install(
    workspace_root: str | Path,
    *,
    bin_path: str | Path | None = None,
    unit_directory: str | Path | None = None,
    source_runner: str | Path | None = None,
    service_template: str | Path | None = None,
    timer_template: str | Path | None = None,
    heavy_service_template: str | Path | None = None,
    heavy_timer_template: str | Path | None = None,
    python_executable: str | Path | None = None,
    systemctl: str = "systemctl",
    reload: bool = True,
    max_age_seconds: int = DEFAULT_MAX_QUEUE_AGE_SECONDS,
    drain_deadline_seconds: int = DEFAULT_DRAIN_DEADLINE_SECONDS,
    global_deadline_seconds: int = DEFAULT_GLOBAL_DRAIN_DEADLINE_SECONDS,
    heavy_timeout_seconds: int = DEFAULT_HEAVY_TIMEOUT_SECONDS,
    heavy_gate_timeout_seconds: int = DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
) -> InstallReport:
    """Install queue and phased-push units as one rollback-capable transaction."""

    paths = _resolve_install_paths(
        workspace_root,
        bin_path=bin_path,
        unit_directory=unit_directory,
        source_runner=source_runner,
        service_template=service_template,
        timer_template=timer_template,
        heavy_service_template=heavy_service_template,
        heavy_timer_template=heavy_timer_template,
        python_executable=python_executable,
    )
    payloads = _build_payloads(
        paths,
        max_age_seconds=max_age_seconds,
        drain_deadline_seconds=drain_deadline_seconds,
        global_deadline_seconds=global_deadline_seconds,
        heavy_timeout_seconds=heavy_timeout_seconds,
        heavy_gate_timeout_seconds=heavy_gate_timeout_seconds,
    )
    snapshots = tuple(_snapshot(payload.destination) for payload in payloads)
    return _perform_install(payloads, snapshots, systemctl=systemctl, reload=reload)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Install the versioned merge-queue user runner"
    )
    parser.add_argument(
        "--workspace-root",
        default=os.environ.get("AGENT_UTILITIES_WORKSPACE_ROOT")
        or os.environ.get("REPOSITORY_MANAGER_WORKSPACE")
        or str(Path.cwd()),
    )
    parser.add_argument("--bin-path", default=None)
    parser.add_argument("--unit-directory", default=None)
    parser.add_argument("--systemctl", default="systemctl")
    parser.add_argument(
        "--python-executable",
        default=None,
        help="direct Python executable used by the service (default: this interpreter)",
    )
    parser.add_argument("--no-daemon-reload", action="store_true")
    parser.add_argument(
        "--max-age-seconds", type=int, default=DEFAULT_MAX_QUEUE_AGE_SECONDS
    )
    parser.add_argument(
        "--drain-deadline-seconds",
        type=int,
        default=DEFAULT_DRAIN_DEADLINE_SECONDS,
        help="queue runner wall-clock deadline per repository drain",
    )
    parser.add_argument(
        "--global-deadline-seconds",
        type=int,
        default=DEFAULT_GLOBAL_DRAIN_DEADLINE_SECONDS,
        help="queue runner wall-clock deadline across all selected roots",
    )
    parser.add_argument(
        "--heavy-timeout-seconds",
        type=int,
        default=DEFAULT_HEAVY_TIMEOUT_SECONDS,
        help="systemd wall-clock deadline for the dedicated phased-push lane",
    )
    parser.add_argument(
        "--heavy-gate-timeout-seconds",
        type=int,
        default=DEFAULT_HEAVY_GATE_TIMEOUT_SECONDS,
        help="RM_GATE_TIMEOUT_SECONDS for the dedicated phased-push lane",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        report = install(
            args.workspace_root,
            bin_path=args.bin_path,
            unit_directory=args.unit_directory,
            systemctl=args.systemctl,
            python_executable=args.python_executable,
            reload=not args.no_daemon_reload,
            max_age_seconds=args.max_age_seconds,
            drain_deadline_seconds=args.drain_deadline_seconds,
            global_deadline_seconds=args.global_deadline_seconds,
            heavy_timeout_seconds=args.heavy_timeout_seconds,
            heavy_gate_timeout_seconds=args.heavy_gate_timeout_seconds,
        )
    except MergeQueueRunnerInstallError as exc:
        print(f"repository-manager runner install refused: {exc}")
        return 78
    print(report.as_dict())
    return 0 if report.verified else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
