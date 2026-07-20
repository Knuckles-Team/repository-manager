"""Shared test fixtures for Repository Manager."""

import os
import sys

import pytest


@pytest.fixture(autouse=True)
def _isolate_process_state(monkeypatch):
    """Isolate ``sys.argv`` and ``os.environ`` for every test.

    Two cross-test footguns this guards against:

    * ``get_mcp_instance()`` builds the server via ``create_mcp_server()``, which
      parses ``sys.argv`` with argparse. Under pytest, ``sys.argv`` carries
      pytest's own flags (``-q``, ``-p no:cacheprovider``, ``--ignore=...``), so
      argparse rejects them and raises ``SystemExit(2)`` — which several MCP tests
      don't catch (they ``except Exception``; ``SystemExit`` is a
      ``BaseException``). Reset argv to clean defaults.
    * Some tests write ``os.environ`` **directly** (e.g. ``WORKSPACE_YML``,
      ``REPOSITORY_MANAGER_WORKSPACE``) instead of via ``monkeypatch.setenv``, so
      the value — often a now-deleted tmp path — leaks into later tests and flakes
      mock-based ones (``test_mcp_rm_git_tool``). Snapshot/restore environ so each
      test starts from a clean process state.
    """
    monkeypatch.setattr(sys, "argv", ["repository-manager"])
    saved_env = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved_env)


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("REPOSITORY_URL", "https://test.example.com")
    monkeypatch.setenv("REPOSITORY_TOKEN", "test-token-12345")
