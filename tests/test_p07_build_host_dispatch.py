"""P0.7 — the literal success test: ``rm_build action=request host=<name>``.

``build_queue.dispatch("request", host=...)`` must resolve the repo's own
origin/HEAD, hand them to `remote_worker_actions.dispatch_build`, and return
a real result — the exact path `mcp_tools/build.py`'s new `host` parameter
and the CLI's `--build-host` flag both route through.
"""

from __future__ import annotations

import subprocess
import textwrap
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import pytest

from repository_manager import build_queue as bq
from repository_manager import remote_worker_actions as rwa
from repository_manager.remote_execution import ssh_executor as ssh_executor_module


def _run(cmd: str, cwd: Path) -> str:
    proc = subprocess.run(
        cmd, shell=True, cwd=str(cwd), capture_output=True, text=True, check=True
    )
    return proc.stdout.strip()


@pytest.fixture(autouse=True)
def _isolated_capacity_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[None]:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg-state"))
    rwa._CAPACITY_INVENTORY = None
    rwa._CAPACITY_STORE = None
    rwa._REGISTRY = None
    yield
    rwa._CAPACITY_INVENTORY = None
    rwa._CAPACITY_STORE = None
    rwa._REGISTRY = None


class _FakeTunnel:
    def __init__(self, *_a: object, **_kw: object) -> None:
        self.calls: list[str] = []

    def run_command(self, command: str, timeout: int | None = None):
        self.calls.append(command)
        if (
            "git clone" in command
            or "git fetch" in command
            or "git checkout" in command
        ):
            return SimpleNamespace(success=True, stdout="", stderr="ok")
        if command.endswith("git status --porcelain"):
            return SimpleNamespace(success=True, stdout="", stderr="")
        if command.endswith("git rev-parse HEAD"):
            return SimpleNamespace(success=True, stdout=self._sha, stderr="")
        if "echo remote-build-ran" in command:
            return SimpleNamespace(success=True, stdout="remote-build-ran", stderr="")
        return SimpleNamespace(success=False, stdout="", stderr=f"unhandled: {command}")

    _sha = ""


@pytest.fixture(autouse=True)
def _fake_tunnel(monkeypatch: pytest.MonkeyPatch) -> _FakeTunnel:
    fake = _FakeTunnel()
    monkeypatch.setattr(ssh_executor_module, "Tunnel", lambda **_kw: fake)
    monkeypatch.setattr(ssh_executor_module, "_TUNNEL_MANAGER_IMPORT_ERROR", None)
    return fake


@pytest.fixture
def repo_with_origin(tmp_path: Path) -> Path:
    origin = tmp_path / "origin.git"
    origin.mkdir()
    _run("git init -q --bare -b main", origin)

    work = tmp_path / "repo"
    work.mkdir()
    _run("git init -q -b main", work)
    _run("git config user.email t@t.io && git config user.name t", work)
    _run("git config commit.gpgsign false", work)
    (work / bq.CONFIG_FILENAME).write_text(
        textwrap.dedent(
            """
            base: main
            specs:
              - name: widget
                command: ["bash", "-c", "echo remote-build-ran"]
                workdir: "."
                timeout: 30
            """
        )
    )
    _run("git add -A", work)
    _run("git commit -q -m init", work)
    _run(f"git remote add origin {origin}", work)
    _run("git push -q origin main", work)
    return work


def test_dispatch_request_with_host_stages_and_builds_remotely(
    repo_with_origin: Path, _fake_tunnel: _FakeTunnel
) -> None:
    head_sha = _run("git rev-parse HEAD", repo_with_origin)
    _fake_tunnel._sha = head_sha

    registered = rwa.dispatch(
        "register_worker",
        host_id="R820",
        cpu_weight=8,
        memory_mib=32768,
        disk_mib=500_000,
        process_slots=4,
        inventory_alias="R820",
        repository_roots={bq.stable_repository_id(repo_with_origin): "/srv/rm-build"},
        toolchains=[],
    )
    assert registered["ok"] is True

    result = bq.dispatch("request", path=str(repo_with_origin), host="R820")

    assert result["ok"] is True, result
    assert result["succeeded"] is True
    assert result["build"]["outcome"] == "succeeded"
    assert result["build"]["stdout_tail"] == "remote-build-ran"
    assert result["staged"]["tree_sha"] == head_sha
    assert any("git clone" in call for call in _fake_tunnel.calls)


def test_dispatch_request_with_host_refuses_a_dirty_tree(
    repo_with_origin: Path, _fake_tunnel: _FakeTunnel
) -> None:
    (repo_with_origin / "dirty.txt").write_text("wip\n")
    rwa.dispatch(
        "register_worker",
        host_id="R820",
        cpu_weight=8,
        memory_mib=32768,
        disk_mib=500_000,
        process_slots=4,
        inventory_alias="R820",
        repository_roots={bq.stable_repository_id(repo_with_origin): "/srv/rm-build"},
        toolchains=[],
    )
    with pytest.raises(bq.BuildQueueError, match="uncommitted changes"):
        bq.dispatch("request", path=str(repo_with_origin), host="R820")
    assert _fake_tunnel.calls == []


def test_local_request_default_is_unaffected_by_the_host_parameter_being_absent(
    repo_with_origin: Path,
) -> None:
    """`host` omitted (the default) must still take the ORIGINAL, unchanged
    local `colocated=True` path — proving the new branch is additive.
    """

    result = bq.dispatch("request", path=str(repo_with_origin), colocated=True)
    assert result["ok"] is True
    assert "host_id" not in result
