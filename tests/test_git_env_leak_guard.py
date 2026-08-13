"""Regression tests for GOC-71 -- a leaked `GIT_DIR`/... pointer env var must
never reach a `git` subprocess undetected.

Root cause (`plans/graph-os-completion-program/GOC-59-67-EXPANSION-TRACKS.md`
Sec. GOC-71): this repo's own `pytest` pre-commit hook runs as a child of `git
commit` and inherits `GIT_DIR`/`GIT_WORK_TREE`/`GIT_AUTHOR_*` pointing at the
REAL checkout. `GIT_DIR` silently overrides `-C <path>` -- demonstrated
directly: `GIT_DIR=real/.git git -C tmp config core.bare true` mutates `real`,
not `tmp`. A test's own git-fixture helper, even one correctly scoped via
`cwd=`/`-C` to a disposable `tmp_path` repo, then silently mutates the real
canonical checkout instead.

Both tests below reproduce that exact command against two throwaway repos
under `tmp_path`. **Neither test ever touches the actual host checkout** --
the "real" repo here is itself a disposable stand-in, so proving the
vulnerability (the negative control) cannot cause the damage it demonstrates.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from tests.conftest import (
    _TRUE_POPEN_INIT,
    LeakedGitPointerEnvError,
    isolated_git_subprocess_env,
)


def _git(args: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        env=isolated_git_subprocess_env(),
    )


def _init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    _git(["init", "-q", "-b", "main"], path)
    _git(["config", "user.email", "t@t.io"], path)
    _git(["config", "user.name", "t"], path)
    return path


def _core_bare(path: Path) -> str:
    """"" if unset, else git's recorded value for `core.bare`."""
    result = subprocess.run(
        ["git", "config", "--get", "core.bare"],
        cwd=str(path),
        capture_output=True,
        text=True,
        env=isolated_git_subprocess_env(),
    )
    return result.stdout.strip()


def test_ambient_leak_vars_are_scrubbed_from_os_environ_for_every_test() -> None:
    """Sanity check for the `_isolate_process_state` scrub half of GOC-71.

    Not a strong proof by itself (see the two tests below for that) -- just
    confirms the autouse fixture actually ran before this test body did.
    """
    import os

    from tests.conftest import _GIT_ENV_LEAK_NAMES, _GIT_ENV_LEAK_PREFIXES

    for name in _GIT_ENV_LEAK_NAMES:
        assert name not in os.environ
    for key in os.environ:
        assert not key.startswith(_GIT_ENV_LEAK_PREFIXES), key


def test_guard_refuses_a_git_subprocess_that_would_run_with_a_leaked_git_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The exact GOC-71 mechanism, reproduced against disposable repos only.

    With the autouse ``_guard_against_leaked_git_pointer_env`` fixture active
    (the default), the corrupting subprocess must never spawn -- so the "real"
    stand-in repo is provably untouched.
    """
    real_repo = _init_repo(tmp_path / "real")
    throwaway_repo = _init_repo(tmp_path / "throwaway")
    assert _core_bare(real_repo) in ("", "false")

    # Recreate the exact leak: a pre-commit-hook-spawned pytest process
    # inherits GIT_DIR pointing at the (stand-in) real checkout.
    monkeypatch.setenv("GIT_DIR", str(real_repo / ".git"))

    with pytest.raises(LeakedGitPointerEnvError):
        subprocess.run(
            ["git", "-C", str(throwaway_repo), "config", "core.bare", "true"],
            check=True,
        )

    monkeypatch.delenv("GIT_DIR", raising=False)
    assert _core_bare(real_repo) in (
        "",
        "false",
    ), "GUARD FAILED: the leaked GIT_DIR corrupted the REAL stand-in repo"
    assert _core_bare(throwaway_repo) in ("", "false")


def test_without_the_guard_the_leaked_git_dir_genuinely_corrupts_the_wrong_repo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Negative control: proves the guard stops a REAL vulnerability, not a
    strawman.

    Temporarily lifts *this test's own* guard patch (restoring the true,
    unpatched ``Popen.__init__`` captured at conftest import time) -- the
    ambient-env scrub in ``_isolate_process_state`` is bypassed too, since the
    leak is reintroduced via ``monkeypatch.setenv`` same as the test above.
    Confirms that, unprotected, the identical command flips ``core.bare`` on
    the wrong (real stand-in) repo -- exactly the GOC-71 incident -- rather
    than the throwaway repo the ``-C`` flag names.
    """
    real_repo = _init_repo(tmp_path / "real")
    throwaway_repo = _init_repo(tmp_path / "throwaway")
    assert _core_bare(real_repo) in ("", "false")

    monkeypatch.setattr(subprocess.Popen, "__init__", _TRUE_POPEN_INIT)
    monkeypatch.setenv("GIT_DIR", str(real_repo / ".git"))

    subprocess.run(
        ["git", "-C", str(throwaway_repo), "config", "core.bare", "true"],
        check=True,
    )

    monkeypatch.delenv("GIT_DIR", raising=False)
    assert _core_bare(real_repo) == "true", (
        "expected the unguarded leak to hit the REAL stand-in repo, "
        "reproducing the GOC-71 incident -- if this fails, the reproduction "
        "itself is stale and the positive-control test above proves nothing"
    )
    assert _core_bare(throwaway_repo) in ("", "false")
