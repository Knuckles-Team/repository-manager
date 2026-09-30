"""Strict RELEASE adapter contracts, independent of ordinary code gates."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from repository_manager import release_readiness_hook as hook


@pytest.mark.parametrize("scope", [set(), None, "malformed"])
def test_missing_scope_blocks_without_query(monkeypatch, scope):
    monkeypatch.setattr(hook.readiness, "_resolve_fleet_packages", lambda *a: scope)
    def unexpected(*a):
        pytest.fail("must not query an index with missing scope")
    monkeypatch.setattr(hook.readiness, "check_tree", unexpected)
    assert hook.main([]) == 2


@pytest.mark.parametrize("ok,overridden,expected", [(True, False, 0), (False, False, 1), (True, True, 1)])
def test_authoritative_verdict_without_override(monkeypatch, ok, overridden, expected):
    monkeypatch.setattr(hook.readiness, "_resolve_fleet_packages", lambda *a: {"agent-utilities"})
    report = SimpleNamespace(ok=ok, overridden=overridden)
    monkeypatch.setattr(hook.readiness, "check_tree", lambda *a: report)
    monkeypatch.setattr(hook.readiness, "_print_human_report", lambda r: None)
    assert hook.main([]) == expected


def test_index_errors_are_not_swallowed(monkeypatch):
    monkeypatch.setattr(hook.readiness, "_resolve_fleet_packages", lambda *a: {"agent-utilities"})
    def unavailable(*a):
        raise RuntimeError("index unavailable")
    monkeypatch.setattr(hook.readiness, "check_tree", unavailable)
    with pytest.raises(RuntimeError, match="index unavailable"):
        hook.main([])
