"""Characterization tests for test_agent_integration.py's SSE stream parsing
(wD10-C-MISC).

test_get_workspace_projects_via_graph (18/38) requires a live
repository-manager agent server plus a reachable LLM backend to actually
reach its PASS assertions -- it is marked @pytest.mark.integration and
@pytest.mark.slow and is excluded from the default suite (pytest.ini
addopts `-m "not integration"`), and neither is available in this
environment. Forcing it to run here would only ever reach its
LLM-unreachable pytest.skip() branch (the fixture points LLM_BASE_URL at a
deliberately non-listening port), which is not a passing run and would
prove nothing about the extraction.

What the extract-method refactor pulls out, however, is PURE:
_apply_graph_event/_process_stream_line classify one already-parsed SSE
line and flip flags in a plain dict -- no network, no subprocess. Both were
moved verbatim (confirmed via `git diff`, no logic rewritten) out of the
test's inline event-handling code. These tests exercise that pure logic
directly, covering every branch the live-server run would otherwise
exercise: graph_start, tool-call detection (all three event-name aliases),
the synthesis_fallback LLM-unreachable skip (asserting it raises Skipped
rather than guessing), final_output, an unparseable line (caught, ignored),
and a blank line (ignored).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "tests" / "test_agent_integration.py"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "test_agent_integration_char", MODULE_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tai = _load_module()


def _flags():
    return {"graph_started": False, "tool_called": False, "final_output_received": False}


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}"


def test_blank_line_is_ignored():
    flags = _flags()
    tai._process_stream_line("", flags)
    tai._process_stream_line("   ", flags)
    assert flags == _flags()


def test_graph_start_sets_flag():
    flags = _flags()
    line = _sse({"type": "data-graph-event", "data": {"event": "graph_start"}})
    tai._process_stream_line(line, flags)
    assert flags["graph_started"] is True


@pytest.mark.parametrize("ename", ["expert_tool_call", "tool_call", "node_start"])
def test_tool_call_event_names_set_tool_called_flag(ename):
    flags = _flags()
    line = _sse(
        {
            "type": "data-graph-event",
            "data": {"event": ename, "tool_name": "get_workspace_projects"},
        }
    )
    tai._process_stream_line(line, flags)
    assert flags["tool_called"] is True


def test_tool_call_event_with_different_tool_name_does_not_set_flag():
    flags = _flags()
    line = _sse(
        {
            "type": "data-graph-event",
            "data": {"event": "tool_call", "tool_name": "some_other_tool"},
        }
    )
    tai._process_stream_line(line, flags)
    assert flags["tool_called"] is False


def test_tool_call_falls_back_to_tool_key():
    flags = _flags()
    line = _sse(
        {
            "type": "data-graph-event",
            "data": {"event": "node_start", "tool": "get_workspace_projects"},
        }
    )
    tai._process_stream_line(line, flags)
    assert flags["tool_called"] is True


def test_final_output_sets_flag():
    flags = _flags()
    line = _sse({"type": "final_output", "content": "here are the projects"})
    tai._process_stream_line(line, flags)
    assert flags["final_output_received"] is True


def test_graph_complete_is_a_no_op():
    flags = _flags()
    line = _sse({"type": "data-graph-event", "data": {"event": "graph_complete"}})
    tai._process_stream_line(line, flags)
    assert flags == _flags()


def test_synthesis_fallback_connection_error_skips():
    flags = _flags()
    line = _sse(
        {
            "type": "data-graph-event",
            "data": {
                "event": "synthesis_fallback",
                "reason": "Connection error: refused",
            },
        }
    )
    with pytest.raises(pytest.skip.Exception):
        tai._process_stream_line(line, flags)


def test_synthesis_fallback_connection_refused_skips():
    flags = _flags()
    line = _sse(
        {
            "type": "data-graph-event",
            "data": {"event": "synthesis_fallback", "reason": "Connection refused"},
        }
    )
    with pytest.raises(pytest.skip.Exception):
        tai._process_stream_line(line, flags)


def test_synthesis_fallback_unrelated_reason_does_not_skip():
    flags = _flags()
    line = _sse(
        {
            "type": "data-graph-event",
            "data": {"event": "synthesis_fallback", "reason": "some other reason"},
        }
    )
    # Must not raise.
    tai._process_stream_line(line, flags)
    assert flags == _flags()


def test_non_data_line_is_ignored():
    flags = _flags()
    tai._process_stream_line("event: ping", flags)
    assert flags == _flags()


def test_unparseable_data_line_is_caught_and_ignored():
    flags = _flags()
    # Must not raise.
    tai._process_stream_line("data: not-json{{{", flags)
    assert flags == _flags()
