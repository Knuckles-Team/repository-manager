"""Shared test fixtures for Repository Manager."""

import os
import re
import subprocess
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
#: bogus commits on a real feature branch; reproduced again and root-caused
#: during GOC-71 (plans/graph-os-completion-program/GOC-59-67-EXPANSION-
#: TRACKS.md): `GIT_DIR=real/.git git -C tmp config core.bare true` mutates
#: `real`, not `tmp` -- `GIT_DIR` silently overrides `-C`.
#:
#: The list is every "Git Repository" / "Git Commits" env var documented in
#: `git help environment` that can redirect a git invocation to a different
#: repository, index, or identity than the one its caller intended -- verified
#: against that page, not guessed. `GIT_CONFIG*` covers `GIT_CONFIG_GLOBAL`,
#: `GIT_CONFIG_SYSTEM`, `GIT_CONFIG_NOSYSTEM`, and the `GIT_CONFIG_COUNT` /
#: `GIT_CONFIG_KEY_<n>` / `GIT_CONFIG_VALUE_<n>` env-based config-override
#: family. `GIT_NAMESPACE` is included defensively (redirects ref lookups)
#: though no known incident has involved it yet.
#:
#: `_isolate_process_state` below scrubs all of these from `os.environ` for
#: *every* test automatically (the chokepoint fix) -- `isolated_git_
#: subprocess_env()` remains available for tests that build an explicit `env`
#: dict from scratch instead of relying on ambient inheritance.
_GIT_ENV_LEAK_PREFIXES = ("GIT_AUTHOR_", "GIT_COMMITTER_", "GIT_CONFIG")
_GIT_ENV_LEAK_NAMES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_CEILING_DIRECTORIES",
    "GIT_COMMON_DIR",
    "GIT_NAMESPACE",
)

#: The subset of the above that can redirect a git invocation to a *different
#: repository* -- i.e. the vars the GOC-71 mechanism actually exploits.
#: Deliberately excludes GIT_INDEX_FILE (redirects only which index file is
#: read/written *within* whatever repo -C/cwd/GIT_DIR resolves to -- and this
#: repo's own `stash_guard.capture_wip` legitimately sets it, scoped to a
#: private tempfile, to build an alternate index without touching the real
#: one), GIT_CEILING_DIRECTORIES (bounds upward directory search; per `git
#: help environment` it explicitly does NOT override an explicit GIT_DIR),
#: GIT_CONFIG*/GIT_AUTHOR_*/GIT_COMMITTER_* (identity/config, not a repo
#: pointer), and GIT_NAMESPACE (ref-lookup only). See
#: ``_guard_against_leaked_git_pointer_env`` below.
_GIT_POINTER_ENV_NAMES = (
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_COMMON_DIR",
)


def isolated_git_subprocess_env() -> dict[str, str]:
    """A copy of the process environment with git-hook-leaked vars stripped."""
    return {
        k: v
        for k, v in os.environ.items()
        if k not in _GIT_ENV_LEAK_NAMES and not k.startswith(_GIT_ENV_LEAK_PREFIXES)
    }


class LeakedGitPointerEnvError(RuntimeError):
    """A git subprocess was about to run with a leaked repository-pointer env var.

    Regression guard for GOC-71. Raised by ``_guard_against_leaked_git_pointer_env``
    instead of letting the corrupting subprocess spawn -- see
    ``tests/test_git_env_leak_guard.py`` for the literal proof this catches the
    known-bad input (a real ``GIT_DIR`` leak redirecting a `-C`-scoped git
    command to the wrong repository).
    """


def _looks_like_git_invocation(args: object) -> bool:
    """Best-effort: does this ``subprocess.Popen`` ``args`` invoke ``git``?

    Handles both list/tuple argv (``["git", "commit", ...]``) and ``shell=True``
    command strings (``"git add -A && git commit ..."``), which this test suite
    uses both of.
    """
    if isinstance(args, (str, bytes)):
        text = args if isinstance(args, str) else args.decode(errors="replace")
        return re.search(r"(?:^|[\s;&|])git(?:\s|$)", text) is not None
    if not isinstance(args, (list, tuple)) or not args:
        return False
    first = args[0]
    if not isinstance(first, (str, bytes, os.PathLike)):
        return False
    return os.path.basename(str(os.fspath(first))) in ("git", "git.exe")


#: The true, unpatched ``Popen.__init__`` -- captured at import time, before
#: any fixture below has had a chance to patch it. ``tests/test_git_env_leak_
#: guard.py`` reuses this reference for its negative-control test (proving the
#: guard stops a *real* vulnerability rather than a strawman) by temporarily
#: restoring it with `monkeypatch`, which un-restores back to the guarded
#: version at that test's teardown.
_TRUE_POPEN_INIT = subprocess.Popen.__init__


def _guarded_popen_init(self, *args, **kwargs):
    """Refuse to spawn ``git`` while a repository-pointer env var is set.

    This is independent of (and a backstop for) the ``os.environ`` scrub in
    ``_isolate_process_state``: it fires even if a test re-introduces one of
    these vars mid-test (e.g. via ``monkeypatch.setenv``, exactly how the real
    `git commit` pre-commit hook leaks them into a nested pytest process),
    turning a silent real-repo mutation into a loud, immediate test failure.
    """
    invoked = args[0] if args else kwargs.get("args")
    if _looks_like_git_invocation(invoked):
        env = kwargs.get("env")
        effective = env if env is not None else os.environ
        leaked = [name for name in _GIT_POINTER_ENV_NAMES if effective.get(name)]
        if leaked:
            raise LeakedGitPointerEnvError(
                f"refusing to spawn {invoked!r}: leaked pointer env var(s) "
                f"{leaked} would silently redirect it to the wrong repository "
                "(GOC-71) -- fix the leak, don't bypass this guard"
            )
    return _TRUE_POPEN_INIT(self, *args, **kwargs)


@pytest.fixture(autouse=True)
def _guard_against_leaked_git_pointer_env(monkeypatch):
    """Autouse regression guard for GOC-71 -- see ``LeakedGitPointerEnvError``."""
    monkeypatch.setattr(subprocess.Popen, "__init__", _guarded_popen_init)
    yield


@pytest.fixture(autouse=True)
def _isolate_process_state(monkeypatch):
    """Isolate ``sys.argv`` and ``os.environ`` for every test.

    Three cross-test footguns this guards against:

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
    * GOC-71: this repo's own `pytest` pre-commit hook runs as a child of `git
      commit` and inherits `GIT_DIR`/`GIT_WORK_TREE`/... pointing at the REAL
      checkout. Most git-fixture tests shell out via ``subprocess.run(["git",
      ...], cwd=tmp_path)`` *without* an explicit ``env=`` -- i.e. they inherit
      `os.environ` implicitly -- so stripping the leaked vars from `os.environ`
      here, for every test, protects all of them automatically without
      per-file opt-in. This is the chokepoint fix; the runtime guard above is
      the backstop for whatever reaches a subprocess anyway.
    """
    monkeypatch.setattr(sys, "argv", ["repository-manager"])
    saved_env = dict(os.environ)
    for name in list(os.environ):
        if name in _GIT_ENV_LEAK_NAMES or name.startswith(_GIT_ENV_LEAK_PREFIXES):
            monkeypatch.delenv(name, raising=False)
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
