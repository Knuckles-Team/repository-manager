"""Adapter discovery must not activate optional execution dependencies."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("module", "dependency", "operation"),
    [
        (
            "repository_manager.kg_ingest",
            "agent_utilities.knowledge_graph.memory",
            "adapter.ingest_entities([])",
        ),
        (
            "repository_manager.kg_ingest",
            "agent_utilities.knowledge_graph.memory",
            "adapter.ingest_documents([])",
        ),
        (
            "repository_manager.remote_execution.fakes",
            "tunnel_manager",
            "adapter.FakeInventoryResolver({'synthetic'}).resolve('synthetic', object())",
        ),
    ],
)
def test_discovery_is_safe_but_operations_require_the_real_dependency(
    module: str, dependency: str, operation: str
) -> None:
    script = f"""
import importlib
import importlib.abc
import sys

class UnavailableDependency(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == {dependency!r} or fullname.startswith({dependency!r} + '.'):
            raise ModuleNotFoundError('synthetic unavailable dependency', name=fullname)

sys.meta_path.insert(0, UnavailableDependency())
adapter = importlib.import_module({module!r})
try:
    {operation}
except ModuleNotFoundError as exc:
    assert exc.name == {dependency!r} or exc.name.startswith({dependency!r} + '.')
else:
    raise AssertionError('operation silently accepted a missing dependency')
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
