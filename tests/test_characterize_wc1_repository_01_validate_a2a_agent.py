"""Characterization tests for ``scripts/validate_a2a_agent.py::main`` (WC1-REPOSITORY-01).

This standalone script had zero test coverage before this lane (nothing in
``tests/`` referenced it). It is a manual A2A-endpoint smoke-check CLI, not
release-path code, so the brief's characterization discipline is satisfied
by writing fresh tests for its (previously uncovered) branches: it is loaded
via ``importlib`` (same pattern as ``tests/test_import_safety_gate.py`` uses
for other ``scripts/*.py`` standalone modules, since ``scripts/`` is not an
importable package) and driven with a mocked ``httpx.AsyncClient`` so no
network call is ever made.

Run once against the unmodified function (record green), then again
unmodified after the extract-method refactor, per the brief.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "validate_a2a_agent.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "validate_a2a_agent_under_test", SCRIPT
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load scripts/validate_a2a_agent.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mod():
    return _load_module()


def _resp(status_code=200, json_data=None, raise_json_error=False):
    resp = MagicMock()
    resp.status_code = status_code
    if raise_json_error:
        resp.json.side_effect = json.JSONDecodeError("bad", "doc", 0)
    else:
        resp.json.return_value = json_data or {}
    return resp


def _patched_client(post_side_effect):
    """Patch httpx.AsyncClient so `async with ... as client: client.post(...)`
    drives `post_side_effect` (a callable or list consumed in order)."""
    client = MagicMock()
    client.post = AsyncMock(side_effect=post_side_effect)
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=client)
    ctx.__aexit__ = AsyncMock(return_value=False)
    return patch("httpx.AsyncClient", return_value=ctx), client


@pytest.mark.asyncio
async def test_main_non_200_response(mod, capsys):
    patcher, _client = _patched_client([_resp(status_code=500)])
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Error: 500" in out


@pytest.mark.asyncio
async def test_main_json_decode_error(mod, capsys):
    patcher, _client = _patched_client([_resp(status_code=200, raise_json_error=True)])
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Response body omitted (HTTP 200)" in out


@pytest.mark.asyncio
async def test_main_jsonrpc_error_no_task(mod, capsys):
    data = {"error": {"code": -32000}}
    patcher, _client = _patched_client([_resp(status_code=200, json_data=data)])
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "JSON-RPC error code: -32000" in out


@pytest.mark.asyncio
async def test_main_task_polling_terminal_state_with_history(mod, capsys):
    initial = {"result": {"id": "task-1"}}
    poll_final = {
        "result": {
            "status": {"state": "completed"},
            "history": [
                {"role": "user", "parts": [{"kind": "text", "text": "hi"}]},
                {"role": "agent", "parts": [{"text": "hello back"}]},
            ],
        }
    }
    responses = [_resp(200, initial), _resp(200, poll_final)]
    patcher, client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Task submitted; polling for result" in out
    assert "Task State: completed" in out
    assert "Agent Response" in out
    assert client.post.await_count == 2


@pytest.mark.asyncio
async def test_main_task_polling_terminal_state_no_structured_parts(mod, capsys):
    initial = {"result": {"id": "task-2"}}
    poll_final = {
        "result": {
            "status": {"state": "failed"},
            "history": [{"role": "agent", "extra": "no parts key"}],
        }
    }
    responses = [_resp(200, initial), _resp(200, poll_final)]
    patcher, _client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Final response received without structured parts." in out


@pytest.mark.asyncio
async def test_main_task_polling_terminal_state_only_user_messages_in_history(
    mod, capsys
):
    """When every history entry has role "user", the reversed-search loop
    never sets `last_msg`, taking the `else` arm of the inner if/elif/else."""
    initial = {"result": {"id": "task-3"}}
    poll_final = {
        "result": {
            "status": {"state": "failed"},
            "history": [{"role": "user", "parts": [{"kind": "text", "text": "hi"}]}],
        },
    }
    responses = [_resp(200, initial), _resp(200, poll_final)]
    patcher, _client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "No Agent Response Found in History" in out


@pytest.mark.asyncio
async def test_main_task_polling_terminal_state_empty_history_prints_nothing_extra(
    mod, capsys
):
    """`history: []` is falsy -- the whole inner response-formatting block is
    skipped (no "Agent Response"/"No Agent Response" line at all)."""
    initial = {"result": {"id": "task-3b"}}
    poll_final = {"result": {"status": {"state": "failed"}, "history": []}}
    responses = [_resp(200, initial), _resp(200, poll_final)]
    patcher, _client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Agent Response" not in out
    assert "Task Finished with state: failed" in out


@pytest.mark.asyncio
async def test_main_task_polling_running_then_completed(mod, capsys):
    initial = {"result": {"id": "task-4"}}
    poll_running = {"result": {"status": {"state": "running"}}}
    poll_done = {"result": {"status": {"state": "completed"}}}
    responses = [_resp(200, initial), _resp(200, poll_running), _resp(200, poll_done)]
    patcher, client = _patched_client(responses)
    with patch("asyncio.sleep", new=AsyncMock()), patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Task State: running" in out
    assert "Task State: completed" in out
    assert client.post.await_count == 3


@pytest.mark.asyncio
async def test_main_task_polling_no_result_key_with_error(mod, capsys):
    initial = {"result": {"id": "task-5"}}
    poll_error = {"error": {"code": -32001}}
    responses = [_resp(200, initial), _resp(200, poll_error)]
    patcher, _client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Polling JSON-RPC error code: -32001" in out


@pytest.mark.asyncio
async def test_main_task_polling_no_result_key_no_error(mod, capsys):
    initial = {"result": {"id": "task-6"}}
    poll_unknown = {"something_else": True}
    responses = [_resp(200, initial), _resp(200, poll_unknown)]
    patcher, _client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Starting polling error key check..." in out


@pytest.mark.asyncio
async def test_main_task_polling_http_failure(mod, capsys):
    initial = {"result": {"id": "task-7"}}
    responses = [_resp(200, initial), _resp(status_code=503)]
    patcher, _client = _patched_client(responses)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Polling Failed: 503" in out


@pytest.mark.asyncio
async def test_main_request_error_is_caught(mod, capsys):
    import httpx

    def raise_request_error(*args, **kwargs):
        raise httpx.RequestError("boom")

    patcher, _client = _patched_client(raise_request_error)
    with patcher:
        await mod.main()
    out = capsys.readouterr().out
    assert "Operation failed: RequestError" in out
