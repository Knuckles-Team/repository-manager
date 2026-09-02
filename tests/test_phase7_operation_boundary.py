"""Round-7 regressions for descriptor-pinned repository operations."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest
import yaml
from pydantic import ValidationError

from repository_manager import operation_boundary
from repository_manager.models import (
    BootstrapConfig,
    BootstrapEnvVar,
    BootstrapHost,
    GitError,
    GraphConfig,
    SubdirectoryConfig,
    WorkspaceConfig,
    WorkspaceProfile,
    WorkspaceSelector,
)
from repository_manager.operation_boundary import (
    OperationBoundaryError,
    open_directory,
    pin_existing,
    read_release_plan_receipt,
    snapshot_workspace,
    write_at,
    write_release_plan_receipt,
)
from repository_manager.repository_manager import Git, GitResult


def _run(argv: list[str], cwd: Path) -> None:
    subprocess.run(argv, cwd=cwd, check=True, capture_output=True, text=True)


def _config_value(git_dir: Path, key: str) -> str:
    result = subprocess.run(
        ["git", "--git-dir", str(git_dir), "config", "--get", key],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _source_repository(root: Path) -> Path:
    """Create a tiny local source repository for clone/pull probes."""
    source = root / "source"
    source.mkdir()
    _run(["git", "init", "-q", "-b", "main"], source)
    _run(["git", "config", "user.email", "test@example.invalid"], source)
    _run(["git", "config", "user.name", "test"], source)
    (source / "README.md").write_text("source\n")
    _run(["git", "add", "README.md"], source)
    _run(["git", "commit", "-qm", "initial"], source)
    return source


def _success_action(*args: object, **kwargs: object) -> GitResult:
    command = str(kwargs.get("command", args[0] if args else ""))
    if "status --porcelain" in command:
        return GitResult(status="success", data="")
    return GitResult(status="success", data="Pushed")


def _manager_with_project(root: Path) -> Any:
    manager = Git(path=str(root))
    project = root / "repo"
    project.mkdir()
    manager.project_map = {"https://github.com/example/repo.git": str(project)}
    manager.git_action = MagicMock(side_effect=_success_action)  # type: ignore[method-assign]
    return manager


def test_git_subprocess_uses_an_inherited_descriptor_cwd(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    root.mkdir()
    repo = root / "repo"
    repo.mkdir()
    _run(["git", "init", "-q"], repo)
    manager = Git(path=str(root))
    original = subprocess.Popen
    calls: list[dict[str, Any]] = []

    def observe(*args: Any, **kwargs: Any) -> Any:
        if (
            args
            and isinstance(args[0], list)
            and args[0][0] == "git"
            and "rev-parse" in args[0]
        ):
            calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", observe)
    result = manager.git_action("git rev-parse --show-toplevel", path=str(repo))

    assert result.status == "success"
    assert calls
    cwd = str(calls[0]["cwd"])
    assert cwd.startswith("/proc/self/fd/")
    descriptor = int(cwd.rsplit("/", 1)[-1])
    assert descriptor in tuple(calls[0]["pass_fds"])


def test_workspace_root_symlink_is_rejected_at_manager_boundary(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    linked_root = tmp_path / "workspace-link"
    linked_root.symlink_to(outside, target_is_directory=True)

    with pytest.raises(OperationBoundaryError, match="symlink"):
        Git(path=str(linked_root))


def test_linked_worktree_metadata_must_remain_inside_workspace(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    checkout.mkdir()
    external = tmp_path / "external-git"
    external.mkdir()
    (checkout / ".git").write_text(f"gitdir: {external}\n")

    result = Git(path=str(workspace)).git_action("git status", path=str(checkout))

    assert result.status == "error"
    assert result.error is not None
    assert "escapes workspace" in result.error.message


def test_linked_worktree_metadata_inside_workspace_remains_supported(tmp_path):
    source = _source_repository(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    _run(["git", "clone", "-q", str(source), str(checkout)], workspace)
    linked = workspace / "linked"
    _run(["git", "worktree", "add", "-q", str(linked), "-b", "linked"], checkout)

    result = Git(path=str(workspace)).git_action(
        "git status --porcelain", path=str(linked)
    )

    assert result.status == "success"


def test_pinned_git_rejects_path_override_options_before_child(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    checkout.mkdir()
    _run(["git", "init", "-q"], checkout)
    external = tmp_path / "external-git"
    external.mkdir()
    manager = Git(path=str(workspace))
    called = False

    def unexpected_child(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal called
        called = True
        raise AssertionError("unsafe Git path option reached the child")

    monkeypatch.setattr(subprocess, "Popen", unexpected_child)
    result = manager.git_action(
        f"git --git-dir={external} config phase7.boundary escaped",
        path=str(checkout),
    )

    assert result.status == "error"
    assert result.error is not None
    assert "path-control option" in result.error.message
    assert not called


def test_pinned_git_strips_environment_path_controls(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    checkout.mkdir()
    _run(["git", "init", "-q"], checkout)
    observed: dict[str, str] = {}
    original = subprocess.Popen

    def observe(*args: Any, **kwargs: Any) -> Any:
        env = kwargs.get("env")
        assert isinstance(env, dict)
        observed.update({str(key): str(value) for key, value in env.items()})
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", observe)
    result = Git(path=str(workspace)).git_action(
        "git status",
        path=str(checkout),
        env={
            "GIT_DIR": str(tmp_path / "external-git"),
            "GIT_WORK_TREE": str(tmp_path / "external-tree"),
            "GIT_INDEX_FILE": str(tmp_path / "external-index"),
            "PATH": "/usr/bin:/bin",
        },
    )

    assert result.status == "success"
    assert not {"GIT_DIR", "GIT_WORK_TREE", "GIT_INDEX_FILE"} & observed.keys()


def test_write_at_checks_identity_before_truncating_replaced_file(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "config"
    target.write_text("original\n")
    replacement = "replacement\n"

    original_open = operation_boundary._open_write_entry

    def replace_before_open(directory_fd: int, name: str, flags: int, mode: int) -> int:
        target.unlink()
        target.write_text(replacement)
        return original_open(directory_fd, name, flags, mode)

    monkeypatch.setattr(operation_boundary, "_open_write_entry", replace_before_open)
    with open_directory(workspace) as root:
        with pytest.raises(OperationBoundaryError, match="changed"):
            write_at(root.fd, "config", b"new\n")

    assert target.read_text() == replacement


def test_release_receipt_flushes_parent_for_start_and_atomic_completion(
    tmp_path, monkeypatch
):
    flushes: list[int] = []
    monkeypatch.setattr(
        operation_boundary,
        "_fsync_directory",
        lambda descriptor: flushes.append(descriptor),
    )
    with open_directory(tmp_path) as root:
        write_release_plan_receipt(root, {"state": "started"})
        write_release_plan_receipt(root, {"state": "completed"}, atomic=True)
        assert read_release_plan_receipt(root) == {"state": "completed"}

    assert len(flushes) == 2


def test_mount_identity_rebind_is_rejected(tmp_path, monkeypatch):
    with open_directory(tmp_path) as root:
        mount_id = root.identity.mount_id
        if mount_id is None:
            pytest.skip("mount IDs are unavailable on this platform")
        assert mount_id is not None
        changed_mount_id = mount_id + 1
        monkeypatch.setattr(
            operation_boundary,
            "_mount_id_for_path",
            lambda _path: changed_mount_id,
        )
        with pytest.raises(OperationBoundaryError, match="identity changed"):
            root.assert_root_identity()


def test_cleanup_refuses_nested_symlink_without_touching_external_file(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "coverage.xml"
    protected.write_text("must survive\n")
    target = workspace / "repo"
    target.mkdir()
    (target / "nested").symlink_to(external, target_is_directory=True)

    Git(path=str(workspace)).cleanup_artifacts(str(target))

    assert protected.exists()


def test_cleanup_refuses_nested_rebind_before_deletion(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    external = tmp_path / "external"
    external.mkdir()
    protected = external / "coverage.xml"
    protected.write_text("must survive\n")
    target = workspace / "repo"
    target.mkdir()
    nested = target / "nested"
    nested.mkdir()
    original_file = nested / "coverage.xml"
    original_file.write_text("also survives\n")

    original_entries = operation_boundary._cleanup_entries
    swapped = False

    def rebind_after_scan(directory_fd: int, label: str) -> list[tuple[str, Any, Any]]:
        nonlocal swapped
        entries = original_entries(directory_fd, label)
        if not swapped and label == str(target):
            swapped = True
            saved = target / "nested-original"
            nested.rename(saved)
            nested.symlink_to(external, target_is_directory=True)
        return entries

    monkeypatch.setattr(operation_boundary, "_cleanup_entries", rebind_after_scan)
    Git(path=str(workspace)).cleanup_artifacts(str(target))

    assert protected.exists()
    assert original_file.exists()


def test_cleanup_refuses_external_directory_rebind_after_preflight(
    tmp_path, monkeypatch
):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "repo"
    target.mkdir()
    nested = target / "nested"
    nested.mkdir()
    (nested / "coverage.xml").write_text("original survives\n")
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    replacement_file = replacement / "coverage.xml"
    replacement_file.write_text("external survives\n")

    original_preflight = operation_boundary._preflight_cleanup_tree
    swapped = False

    def rebind_after_preflight(
        directory_fd: int,
        *,
        ignored_directory_names: frozenset[str],
        label: str,
        relative_path: tuple[str, ...] = (),
    ) -> dict[tuple[str, ...], Any]:
        nonlocal swapped
        plan = original_preflight(
            directory_fd,
            ignored_directory_names=ignored_directory_names,
            label=label,
            relative_path=relative_path,
        )
        if not swapped and label == str(target):
            swapped = True
            nested.rename(target / "nested-original")
            replacement.rename(nested)
        return plan

    monkeypatch.setattr(
        operation_boundary, "_preflight_cleanup_tree", rebind_after_preflight
    )
    Git(path=str(workspace)).cleanup_artifacts(str(target))

    assert (nested / "coverage.xml").read_text() == "external survives\n"


def test_push_status_error_fails_closed_before_remote_mutation(tmp_path):
    manager = _manager_with_project(tmp_path)
    project = next(iter(manager.project_map.values()))
    status_error = GitResult(
        status="error",
        data="status unavailable",
        error=GitError(message="status unavailable", code=1),
    )

    def status_only(*args: object, **kwargs: object) -> GitResult:
        command = str(kwargs.get("command", args[0] if args else ""))
        if command == "git status --porcelain":
            return status_error
        raise AssertionError(f"unexpected command after status failure: {command}")

    manager.git_action.side_effect = status_only
    result = manager.push_project(project)

    assert result.status == "error"
    assert manager.git_action.call_count == 1


def test_push_gate_exception_fails_closed_before_remote_mutation(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    project = next(iter(manager.project_map.values()))
    (Path(project) / ".pre-commit-config.yaml").write_text("repos: []\n")
    manager._has_unpushed_commits = MagicMock(return_value=True)  # type: ignore[method-assign]
    manager._unpushed_changed_files = MagicMock(return_value=[])  # type: ignore[method-assign]
    monkeypatch.setattr(
        "repository_manager.repository_manager.run_gate_stage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("gate crashed")),
    )

    result = manager.push_project(project)

    assert result.status == "error"
    assert result.error is not None
    assert "gate did not complete" in result.error.message
    assert not any(
        "git push" in str(call.kwargs.get("command", ""))
        for call in manager.git_action.call_args_list
    )


def test_tag_push_failure_fails_overall_push(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    manager.gate_before_push = False
    project = next(iter(manager.project_map.values()))
    tag_error = GitResult(
        status="error",
        data="tag denied",
        error=GitError(message="tag denied", code=1),
    )
    monkeypatch.setattr(
        manager, "_push_release_tag", lambda *_args, **_kwargs: tag_error
    )

    result = manager.push_project(project)

    assert result.status == "error"
    assert result.error is not None
    assert "tag push failed" in result.error.message


def test_git_action_rejects_a_symlinked_git_directory(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    checkout.mkdir()
    external = tmp_path / "external-git"
    external.mkdir()
    (checkout / ".git").symlink_to(external, target_is_directory=True)

    result = Git(path=str(workspace)).git_action("git status", path=str(checkout))

    assert result.status == "error"
    assert result.error is not None
    assert "symlink" in result.error.message


def test_git_action_keeps_git_metadata_on_the_pinned_checkout(tmp_path, monkeypatch):
    """A .git swap after admission cannot redirect the Git mutation."""
    source_root = tmp_path / "source-root"
    source_root.mkdir()
    source = _source_repository(source_root)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    _run(["git", "clone", "-q", str(source), str(checkout)], workspace)

    original = subprocess.Popen
    swapped = False

    def swap_git_before_child(*args: Any, **kwargs: Any) -> Any:
        nonlocal swapped
        if not swapped:
            swapped = True
            original_git = checkout / ".git"
            saved = checkout / ".git-original"
            original_git.rename(saved)
            original_git.symlink_to(source / ".git", target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", swap_git_before_child)
    result = Git(path=str(workspace)).git_action(
        "git config phase7.boundary anchored", path=str(checkout)
    )

    assert result.status == "error"
    assert result.error is not None
    assert "symlink" in result.error.message
    assert _config_value(checkout / ".git-original", "phase7.boundary") == "anchored"
    assert not _config_value(source / ".git", "phase7.boundary")


def test_pre_push_gate_uses_the_pinned_checkout_path(tmp_path, monkeypatch):
    """The gate runner must not receive a mutable lexical checkout path."""
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    checkout.mkdir()
    (checkout / ".pre-commit-config.yaml").write_text("repos: []\n")
    manager = Git(path=str(workspace))
    manager._has_unpushed_commits = MagicMock(return_value=True)  # type: ignore[method-assign]
    manager._unpushed_changed_files = MagicMock(return_value=[])  # type: ignore[method-assign]
    observed: list[str] = []

    def fake_gate(repo_path: str, _stage: str, **_kwargs: object) -> MagicMock:
        observed.append(repo_path)
        return MagicMock(success=True)

    monkeypatch.setattr(
        "repository_manager.repository_manager.run_gate_stage", fake_gate
    )
    with open_directory(workspace) as root:
        with pin_existing(root, checkout) as pinned:
            assert manager._gate_before_push(str(checkout), pinned=pinned) is None

    assert observed and observed[0].startswith("/proc/self/fd/")


def test_clone_parent_swap_cannot_redirect_into_external_directory(
    tmp_path, monkeypatch
):
    source = _source_repository(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = workspace / "nested"
    target = parent / "repo"
    manager = Git(path=str(workspace))
    original = subprocess.Popen
    swapped = False

    def swap_before_child(*args: Any, **kwargs: Any) -> Any:
        nonlocal swapped
        if not swapped:
            swapped = True
            saved = workspace / "nested-original"
            parent.rename(saved)
            parent.symlink_to(outside, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", swap_before_child)
    result = manager.clone_repository(str(source), str(target))

    assert result.status == "error"
    assert result.error is not None
    assert "identity" in result.error.message or "symlink" in result.error.message
    assert not (outside / "repo" / "README.md").exists()


def test_clone_parent_swap_to_original_directory_is_still_refused(
    tmp_path, monkeypatch
):
    """A same-inode symlink must not turn lexical validation into an alias."""
    source = _source_repository(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    parent = workspace / "nested"
    parent.mkdir()
    target = parent / "repo"
    manager = Git(path=str(workspace))
    original = subprocess.Popen
    swapped = False

    def swap_to_original(*args: Any, **kwargs: Any) -> Any:
        nonlocal swapped
        if not swapped:
            swapped = True
            saved = workspace / "nested-original"
            parent.rename(saved)
            parent.symlink_to(saved, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", swap_to_original)
    result = manager.clone_repository(str(source), str(target))

    assert result.status == "error"
    assert result.error is not None
    assert "symlink" in result.error.message


def test_pull_target_swap_cannot_update_external_directory(tmp_path, monkeypatch):
    source = _source_repository(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    target = workspace / "repo"
    _run(["git", "clone", "-q", str(source), str(target)], workspace)
    outside = tmp_path / "outside"
    outside.mkdir()
    manager = Git(path=str(workspace))
    original = subprocess.Popen
    swapped = False

    def swap_before_child(*args: Any, **kwargs: Any) -> Any:
        nonlocal swapped
        command = args[0] if args else kwargs.get("args")
        if (
            not swapped
            and isinstance(command, list)
            and command[0] == "git"
            and "pull" in command
        ):
            swapped = True
            saved = workspace / "repo-original"
            target.rename(saved)
            target.symlink_to(outside, target_is_directory=True)
        return original(*args, **kwargs)

    monkeypatch.setattr(subprocess, "Popen", swap_before_child)
    result = manager.pull_project(str(target))

    assert result.status == "error"
    assert result.error is not None
    assert not (outside / "README.md").exists()


def test_workspace_sync_parent_swap_is_refused_before_clone(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    parent = workspace / "nested"
    target = parent / "repo"
    manager = Git(path=str(workspace))
    manager.project_map = {"https://example.invalid/repo.git": str(target)}
    original = manager.clone_repository

    def swap_parent(url: str, path: str, **kwargs: object) -> GitResult:
        parent.mkdir()
        saved = workspace / "nested-original"
        parent.rename(saved)
        parent.symlink_to(outside, target_is_directory=True)
        return original(url, path, **kwargs)

    monkeypatch.setattr(manager, "clone_repository", swap_parent)
    results = manager._sync_workspace_repositories()

    assert results and results[0].status == "error"
    assert not (outside / "repo").exists()


def test_workspace_sync_empty_map_rejects_root_swap(tmp_path, monkeypatch):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    manager = Git(path=str(workspace))

    def swap_root() -> list[tuple[str, str]]:
        saved = tmp_path / "workspace-original"
        workspace.rename(saved)
        workspace.symlink_to(outside, target_is_directory=True)
        return []

    monkeypatch.setattr(manager, "_validated_workspace_sync_targets", swap_root)
    results = manager._sync_workspace_repositories()

    assert results and results[0].status == "error"
    assert results[0].error is not None
    assert "workspace_sync" in results[0].error.message


def test_phased_push_rejects_same_path_checkout_replacement(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    original_execute = manager._execute_push_phase

    def replace_before_phase(**kwargs: object) -> bool:
        saved = project.with_name("repo-original")
        project.rename(saved)
        project.mkdir()
        return original_execute(**kwargs)

    monkeypatch.setattr(manager, "_execute_push_phase", replace_before_phase)
    results = manager.phased_push(
        config={"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]},
        start_phase=1,
        auto_start=False,
    )

    assert any(
        result.error
        and (
            "release plan changed" in result.error.message
            or "identity" in result.error.message
        )
        for result in results
    )
    manager.git_action.assert_not_called()


def test_phased_push_rejects_replacement_after_handle_binding(tmp_path, monkeypatch):
    """A plan cannot admit a checkout replaced after its handle is pinned."""
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    original_bind = manager._bind_release_plan_target

    def bind_then_replace(*args: object, **kwargs: object) -> None:
        original_bind(*args, **kwargs)
        saved = project.with_name("repo-original")
        project.rename(saved)
        project.mkdir()

    monkeypatch.setattr(manager, "_bind_release_plan_target", bind_then_replace)
    results = manager.phased_push(
        config={"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]},
        start_phase=1,
        auto_start=False,
    )

    assert results
    assert any(
        result.error and "identity" in result.error.message for result in results
    )
    manager.git_action.assert_not_called()


def test_phased_push_rejects_origin_drift_before_mutation(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    git_dir = project / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/example/repo.git\n'
    )
    original_execute = manager._execute_push_phase

    def drift_before_phase(**kwargs: object) -> bool:
        (git_dir / "config").write_text(
            '[remote "origin"]\n\turl = https://github.com/example/other.git\n'
        )
        return original_execute(**kwargs)

    monkeypatch.setattr(manager, "_execute_push_phase", drift_before_phase)
    results = manager.phased_push(
        config={"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]},
        start_phase=1,
        auto_start=False,
    )

    assert any(
        result.error
        and (
            "release plan changed" in result.error.message
            or "identity" in result.error.message
        )
        for result in results
    )
    manager.git_action.assert_not_called()


def test_phased_push_rejects_origin_drift_after_handle_binding(tmp_path, monkeypatch):
    source = _source_repository(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    project = workspace / "repo"
    _run(["git", "clone", "-q", str(source), str(project)], workspace)
    _run(
        [
            "git",
            "config",
            "remote.origin.url",
            "https://github.com/example/repo.git",
        ],
        project,
    )
    manager = Git(path=str(workspace))
    manager.project_map = {
        "https://github.com/example/repo.git": str(project),
    }
    original_bind = manager._bind_release_plan_target

    def bind_then_drift(*args: Any, **kwargs: Any) -> None:
        original_bind(*args, **kwargs)
        _run(
            [
                "git",
                "config",
                "remote.origin.url",
                "https://github.com/example/other.git",
            ],
            project,
        )

    monkeypatch.setattr(manager, "_bind_release_plan_target", bind_then_drift)
    results = manager.phased_push(
        config={"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]},
        start_phase=1,
        auto_start=False,
    )

    assert results
    assert any(
        result.error and "identity" in result.error.message for result in results
    )


def test_secure_bump_propagation_ignores_unplanned_collision_service(tmp_path):
    """Dependency propagation stays within the frozen release target pairs."""
    manager = Git(path=str(tmp_path))
    agent = tmp_path / "agent-packages" / "agents" / "arr-mcp"
    service = tmp_path / "services" / "arr-mcp"
    agent.mkdir(parents=True)
    service.mkdir(parents=True)
    agent_url = "https://github.com/example/arr-mcp.git"
    service_url = "https://github.com/example/services-arr-mcp.git"
    manager.project_map = {agent_url: str(agent), service_url: str(service)}
    phase = [{"phase_num": 5, "name": "agents", "targets": [("arr-mcp", str(agent))]}]
    provenance = manager._freeze_release_plan("bump", {}, phase, options={})
    updates = MagicMock(return_value=[])
    manager._update_dependency_files = updates  # type: ignore[method-assign]

    manager._propagate_bump_to_dependents(
        project_name="published-package",
        new_version="1.0.1",
        phase_num=5,
        phase_of=lambda _name: 5,
        dry_run=False,
        all_results=[],
        provenance=provenance,
        plan_assertion=lambda: None,
        allowed_targets={("arr-mcp", str(agent))},
    )

    assert [call.kwargs["path"] for call in updates.call_args_list] == [str(agent)]


def test_phased_push_replays_only_a_recorded_terminal_outcome(tmp_path):
    manager = _manager_with_project(tmp_path)
    config = {"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]}
    first = manager.phased_push(config=config, start_phase=1, auto_start=False)
    calls = manager.git_action.call_count
    second = manager.phased_push(config=config, start_phase=1, auto_start=False)

    assert first and second
    assert manager.git_action.call_count == calls
    assert [item.model_dump() for item in second] == [
        item.model_dump() for item in first
    ]


def test_phased_push_rejects_divergent_plan_reuse(tmp_path):
    manager = _manager_with_project(tmp_path)
    first_config = {"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]}
    manager.phased_push(config=first_config, start_phase=1, auto_start=False)
    calls = manager.git_action.call_count
    result = manager.phased_push(
        config={"phases": [{"phase": 1, "name": "different", "projects": ["repo"]}]},
        start_phase=1,
        auto_start=False,
    )

    assert result and result[0].status == "error"
    assert result[0].error is not None
    assert "different release plan" in result[0].error.message
    assert manager.git_action.call_count == calls


def test_mixed_explicit_bulk_collision_requires_same_canonical_pair(tmp_path):
    manager = Git(path=str(tmp_path))
    bulk_url = "https://example.invalid/agents/shared.git"
    explicit_url = "https://mirror.invalid/shared.git"
    bulk_path = tmp_path / "agent-packages" / "agents" / "shared"
    explicit_path = tmp_path / "shared"
    for project_path in (bulk_path, explicit_path):
        project_path.mkdir(parents=True)
        (project_path / "pyproject.toml").write_text(
            "[project]\n"
            "name = 'shared'\n"
            "dynamic = ['version']\n"
            "\n"
            "[build-system]\n"
            "requires = ['hatchling>=1']\n"
            "build-backend = 'hatchling.build'\n"
        )
    manager.project_map = {
        bulk_url: str(bulk_path),
        explicit_url: str(explicit_path),
    }
    manager._project_categories = {
        bulk_url: ("agent-packages", "agents"),
        explicit_url: (),
    }
    config = {"phases": [{"phase": 5, "projects": ["shared"], "bulk_bump": True}]}

    with pytest.raises(ValueError, match="one canonical path"):
        manager._build_bump_phase_list(config=config, start_phase=5, filter_set=None)


@pytest.mark.parametrize(
    "model, payload",
    [
        (WorkspaceConfig, {"name": "x", "path": ".", "unexpected": True}),
        (SubdirectoryConfig, {"unexpected": True}),
        (GraphConfig, {"unexpected": True}),
        (WorkspaceProfile, {"unexpected": True}),
        (WorkspaceSelector, {"unexpected": True}),
        (BootstrapEnvVar, {"name": "X", "unexpected": True}),
        (BootstrapHost, {"name": "host", "unexpected": True}),
        (BootstrapConfig, {"unexpected": True}),
    ],
)
def test_workspace_models_reject_unknown_manifest_fields(model, payload):
    with pytest.raises(ValidationError):
        model.model_validate(payload, strict=True)


def test_manifest_loader_rejects_symlink_root_with_empty_map(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "workspace-link"
    root.symlink_to(outside, target_is_directory=True)
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(yaml.safe_dump({"name": "x", "path": str(root)}))

    manager = Git()
    assert manager.load_projects_from_yaml(str(manifest)) is False
    assert manager.project_map == {}


def test_release_plan_binds_head_and_git_config_identity(tmp_path):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    git_dir = project / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        '[remote "origin"]\n\turl = https://github.com/example/repo.git\n'
    )
    phase = [
        {
            "phase_num": 1,
            "name": "one",
            "projects_to_push": [("repo", str(project))],
        }
    ]
    provenance = manager._freeze_release_plan("push", {}, phase, options={})
    (git_dir / "HEAD").write_text("ref: refs/heads/other\n")

    assert not manager._release_plan_matches(provenance, {}, phase, options={})


def test_release_plan_rejects_push_remote_drift(tmp_path):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    git_dir = project / ".git"
    git_dir.mkdir()
    (git_dir / "HEAD").write_text("ref: refs/heads/main\n")
    (git_dir / "config").write_text(
        '[remote "origin"]\n'
        "\turl = https://github.com/example/repo.git\n"
        "\tpushurl = https://github.com/example/repo-push.git\n"
    )
    phase = [
        {
            "phase_num": 1,
            "name": "one",
            "projects_to_push": [("repo", str(project))],
        }
    ]
    provenance = manager._freeze_release_plan("push", {}, phase, options={})
    (git_dir / "config").write_text(
        '[remote "origin"]\n'
        "\turl = https://github.com/example/repo.git\n"
        "\tpushurl = https://github.com/example/replacement.git\n"
    )

    assert not manager._release_plan_matches(provenance, {}, phase, options={})


def test_linked_worktree_snapshot_binds_common_config(tmp_path):
    source = _source_repository(tmp_path)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    checkout = workspace / "repo"
    _run(["git", "clone", "-q", str(source), str(checkout)], workspace)
    linked = workspace / "linked"
    _run(["git", "worktree", "add", "-q", str(linked), "-b", "linked"], checkout)
    manager = Git(path=str(workspace))
    manager.project_map = {"https://github.com/example/repo.git": str(linked)}

    snapshot = snapshot_workspace(
        workspace,
        [("https://github.com/example/repo.git", str(linked))],
    )
    metadata = snapshot["projects"][0]["git"]

    expected_digest = hashlib.sha256(
        (checkout / ".git" / "config").read_bytes()
    ).hexdigest()
    assert metadata["config_digest"] == expected_digest
    assert metadata["common"] != metadata["worktree"]


def test_phased_bump_rejects_same_path_replacement_before_bump(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    original_bump = manager._bump_one_project

    def replace_before_bump(**kwargs: object) -> str | None:
        saved = project.with_name("repo-original")
        project.rename(saved)
        project.mkdir()
        return original_bump(**kwargs)

    monkeypatch.setattr(manager, "_bump_one_project", replace_before_bump)
    results = manager.phased_bumpversion(
        config={"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]},
        start_phase=1,
        auto_start=False,
        force=True,
    )

    assert results
    assert any(
        result.error
        and (
            "identity" in result.error.message or "release plan" in result.error.message
        )
        for result in results
    )
    manager.git_action.assert_not_called()


def test_pre_commit_rejects_checkout_swap_after_handle_pin(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))
    (project / ".pre-commit-config.yaml").write_text("repos: []\n")
    original_cleanup = manager.cleanup_artifacts

    def swap_after_pin(target: str) -> None:
        original_cleanup(target)
        saved = project.with_name("repo-original")
        project.rename(saved)
        project.mkdir()

    monkeypatch.setattr(manager, "cleanup_artifacts", swap_after_pin)
    result = manager.pre_commit(path=str(project))

    assert result.status == "error"
    assert result.error is not None
    assert "identity" in result.error.message or "pinned" in result.error.message


def test_auto_start_rejects_checkout_swap_after_pending_probe(tmp_path, monkeypatch):
    manager = _manager_with_project(tmp_path)
    project = Path(next(iter(manager.project_map.values())))

    def swap_pending(_path: str, *, pinned: object) -> bool:
        saved = project.with_name("repo-original")
        project.rename(saved)
        project.mkdir()
        return True

    monkeypatch.setattr(manager, "_repo_has_pending_work", swap_pending)
    config = {"phases": [{"phase": 1, "name": "one", "projects": ["repo"]}]}

    assert manager._auto_start_phase(config, operation="bump") is None
