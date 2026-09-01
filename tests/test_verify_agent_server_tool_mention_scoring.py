"""Characterization tests for verify_agent_server.py::compare_tool_results.

verify_agent_server.py is a smoke-test/validation gate (SPECIAL CASE in the
wave's PREAMBLE) that fails validation if the agent's chat response omits
the projects the direct tool call actually found -- exactly the class of
defect that matters here (an agent silently fabricating or dropping tool
results). Its comparison/scoring logic (extracted below into
``_score_project_mentions``/``_print_comparison_verdict`` as part of the
extract-method refactor) is pure and testable without a live agent server;
the network I/O halves (``_direct_tool_projects``/``_agent_chat_output``)
are not -- they require a running repository-manager agent server on
localhost:9888, which is out of scope for a unit-level characterization
test and is instead covered by this script's own manual `python3
scripts/verify_agent_server.py` run against a live server (unchanged by
this refactor: the two I/O helpers are extracted verbatim, no logic change).

Per the SPECIAL CASE rule, this plants the known-bad input the gate exists
to catch -- an agent chat response that omits every project the direct tool
call found -- confirms it FAILs (verdict False), then a well-formed input
(every project mentioned) and confirms it PASSes. Both run unmodified before
and after the refactor.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "verify_agent_server.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("verify_agent_server_char", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


verify_agent_server = _load_module()


def test_score_project_mentions_full_url_match():
    found, missing = verify_agent_server._score_project_mentions(
        {"git@github.com:org/repo-one.git"}, "See git@github.com:org/repo-one.git for details"
    )
    assert found == 1
    assert missing == []


def test_score_project_mentions_falls_back_to_bare_repo_name():
    found, missing = verify_agent_server._score_project_mentions(
        {"git@github.com:org/repo-one.git"}, "I found repo-one in the workspace"
    )
    assert found == 1
    assert missing == []


def test_score_project_mentions_known_bad_input_nothing_mentioned():
    """Plant the known-bad input: chat output that omits every project."""
    found, missing = verify_agent_server._score_project_mentions(
        {"git@github.com:org/repo-one.git", "git@github.com:org/repo-two.git"},
        "I could not find any projects.",
    )
    assert found == 0
    assert set(missing) == {
        "git@github.com:org/repo-one.git",
        "git@github.com:org/repo-two.git",
    }


def test_score_project_mentions_partial_match():
    found, missing = verify_agent_server._score_project_mentions(
        {"git@github.com:org/repo-one.git", "git@github.com:org/repo-two.git"},
        "repo-one is available",
    )
    assert found == 1
    assert missing == ["git@github.com:org/repo-two.git"]


def test_print_comparison_verdict_fails_closed_on_zero_found(capsys):
    """Remove the known-bad input (found_count == 0): confirm FAIL."""
    result = verify_agent_server._print_comparison_verdict(0, 2, ["a", "b"])
    assert result is False
    assert "❌" in capsys.readouterr().out


def test_print_comparison_verdict_passes_when_some_found(capsys):
    """Well-formed input (found_count > 0): confirm PASS."""
    result = verify_agent_server._print_comparison_verdict(2, 2, [])
    assert result is True
    assert "✅" in capsys.readouterr().out


def test_print_comparison_verdict_passes_with_partial_matches_noted(capsys):
    result = verify_agent_server._print_comparison_verdict(1, 2, ["missing-one"])
    assert result is True
    out = capsys.readouterr().out
    assert "✅" in out
    assert "1 projects were not explicitly found" in out
