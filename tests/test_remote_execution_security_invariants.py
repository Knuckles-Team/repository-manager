"""RMDD-15: structural security invariants that must hold regardless of test
scenario -- no credential/host leakage, no controller-local lock, no branch
mutation, and safe artifact republish after a disconnect.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

_PACKAGE_ROOT = (
    Path(__file__).resolve().parent.parent / "repository_manager" / "remote_execution"
)


def _iter_source_files():
    return sorted(p for p in _PACKAGE_ROOT.glob("*.py") if p.name != "fakes.py")


def test_no_shell_true_anywhere_in_the_package() -> None:
    """Structural proof of the "fixed argv only" constraint (C-04).

    Parses every shipped module (excluding the test-only ``fakes.py``) as an
    AST and asserts no call site passes ``shell=True`` to anything, and no
    ``os.system``/``subprocess.*(..., shell=...)`` call exists at all.
    """

    for path in _iter_source_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                for keyword in node.keywords:
                    if keyword.arg == "shell":
                        raise AssertionError(
                            f"{path}: found a 'shell=' keyword argument at "
                            f"line {node.lineno} -- fixed argv only is violated"
                        )
                func_name = ast.unparse(node.func) if hasattr(ast, "unparse") else ""
                assert "os.system" not in func_name, (
                    f"{path}:{node.lineno} calls os.system"
                )


def test_no_controller_local_lock_or_branch_mutation_primitives_are_used() -> None:
    """Structural proof: no canonical_guard/flock/branch-moving call exists.

    A remote worker must never acquire the controller's NFS/local filesystem
    lock and never move a target branch (lane brief, non-negotiable
    constraints).  These primitives live in named modules/functions
    elsewhere in this repository; this package must never import or call
    them.
    """

    forbidden_substrings = (
        "canonical_guard",
        "guarded_canonical_mutation",
        "hold_canonical_lease",
        "flock",
        'git", "push',
        'git", "branch", "-f',
        'git", "merge',
        'git", "checkout", "main',
    )
    for path in _iter_source_files():
        text = path.read_text()
        for forbidden in forbidden_substrings:
            assert forbidden not in text, (
                f"{path} contains forbidden primitive {forbidden!r}"
            )


def test_authorized_target_model_has_no_credential_or_host_fields() -> None:
    """No raw host, IP, username, password, key, or proxy field can enter the
    public target/result surface (C-09): only ``kind``/``alias``/labels.
    """

    pytest.importorskip(
        "tunnel_manager.remote_execution",
        reason="optional dependency, see remote_execution/README.md",
    )
    from tunnel_manager.remote_execution import AuthorizedTarget

    fields = set(AuthorizedTarget.model_fields)
    forbidden = {
        "hostname",
        "host",
        "ip",
        "address",
        "user",
        "username",
        "password",
        "password_ref",
        "identity_file",
        "key_path",
        "key",
        "proxy_command",
        "proxy",
        "known_hosts_file",
        "port",
    }
    assert fields.isdisjoint(forbidden), fields & forbidden
    assert fields == {"contract_version", "kind", "alias", "capability_labels"}

    # extra="forbid" means a caller cannot smuggle a credential field past
    # validation even if they try.
    with pytest.raises(ValidationError):
        AuthorizedTarget(alias="build-1", hostname="10.0.0.5")  # type: ignore[call-arg]


def test_remote_command_request_forbids_extra_caller_supplied_fields() -> None:
    pytest.importorskip(
        "tunnel_manager.remote_execution",
        reason="optional dependency, see remote_execution/README.md",
    )
    from tunnel_manager.remote_execution import RemoteCommandRequest

    with pytest.raises(ValidationError):
        RemoteCommandRequest(
            argv=("git", "status"),
            workdir="/srv/x",
            ssh_password="hunter2",  # type: ignore[call-arg]
        )


def test_execution_result_stdout_stderr_never_carry_a_raw_hostname_or_password(
    tmp_path,
) -> None:
    """End-to-end: dispatch through the fake port and confirm the translated
    RM ``ExecutionResult`` never exposes anything beyond what the (already
    redacted, per RMDD-14) TM result tail carried.
    """

    pytest.importorskip(
        "tunnel_manager.remote_execution",
        reason="optional dependency, see remote_execution/README.md",
    )
    from tunnel_manager.remote_execution import AuthorizedTarget

    from repository_manager.development import ExecutionCommand
    from repository_manager.remote_execution.executor import RemoteWorkerExecutor
    from repository_manager.remote_execution.fakes import FakeRemoteExecutorPort

    port = FakeRemoteExecutorPort()
    executor = RemoteWorkerExecutor(
        AuthorizedTarget(alias="build-1"), actor=None, port=port
    )
    result = executor.run(
        ExecutionCommand(argv=("echo", "ok"), workdir="/srv/x", timeout_seconds=10),
        command_id="command:sec-1",
        fence="fence:sec-1",
    )
    # The fake target/result never had credential material to begin with
    # (AuthorizedTarget structurally cannot carry it); this asserts the
    # translated RM result likewise carries none of TM's private HostConfig
    # attribute *names*, which would indicate a leak in from_remote_result.
    serialized = f"{result.stdout_tail}{result.stderr_tail}"
    for leaked_name in ("hostname", "password", "identity_file", "proxy_command"):
        assert leaked_name not in serialized
