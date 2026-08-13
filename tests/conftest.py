"""Shared test fixtures for Repository Manager."""

import os
import sys

import pytest
from agent_utilities.knowledge_graph.core.session import GraphSession, use_session
from agent_utilities.models.company_brain import ActorType
from agent_utilities.security.brain_context import ActorContext, use_actor

#: Env vars a git hook invocation sets that git honors over a subprocess's
#: `cwd=` -- this repo's own `pre-commit` runs pytest as a `language: system`
#: hook of `git commit`, so any test that shells out to `git init`/`git
#: commit` against its OWN tmp_path fixture while these leak from the
#: environment silently operates on the OUTER real repo instead, regardless
#: of `cwd=`. Reproduced live during GOC-60/P0.4 development: it planted two
#: bogus commits on a real feature branch. Any test creating a real git repo
#: via subprocess should pass ``env=isolated_git_subprocess_env()``.
_GIT_ENV_LEAK_PREFIXES = ("GIT_AUTHOR_", "GIT_COMMITTER_")
_GIT_ENV_LEAK_NAMES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
    "GIT_OBJECT_DIRECTORY",
)


def isolated_git_subprocess_env() -> dict[str, str]:
    """A copy of the process environment with git-hook-leaked vars stripped."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in _GIT_ENV_LEAK_NAMES and not k.startswith(_GIT_ENV_LEAK_PREFIXES)
    }


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


@pytest.fixture(autouse=True)
def _governed_graph_session():
    """Bind explicit verified test authority for current Graph-OS contracts."""

    actor = ActorContext(
        actor_id="subject:opaque:repository-manager-tests",
        actor_type=ActorType.AUTOMATED_SERVICE,
        tenant_id="tenant:opaque:repository-manager-tests",
        authenticated=True,
    )
    session = GraphSession(
        actor=actor,
        tenant=actor.tenant_id,
        scopes=frozenset({"kg:admin"}),
        graph="__commons__",
        policy_version="policy:opaque:test",
        audience="epistemic-graph",
    )
    with use_actor(actor), use_session(session):
        yield


@pytest.fixture
def mock_env(monkeypatch):
    """Set standard test environment variables."""
    monkeypatch.setenv("REPOSITORY_URL", "https://test.example.com")
    monkeypatch.setenv("REPOSITORY_TOKEN", "test-token-12345")
