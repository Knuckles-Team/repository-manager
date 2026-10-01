"""Shared temporary-git-repository helpers for the ``history_identity`` test suite.

Every fixture here builds a disposable repository under a pytest ``tmp_path``
(never the real workspace) with synthetic commit identities, matching
``specs/fleet-git-identity/plan.md``'s "Contributor setup": temporary Git
repositories, commits by synthetic users, branches, annotated tags, and a
second local bare remote. ``tests/conftest.py``'s autouse fixtures already
strip leaked ``GIT_DIR``/``GIT_WORK_TREE``/... pointer env vars from every
test's environment, so the plain ``subprocess.run(..., cwd=...)`` calls below
cannot silently redirect to a real checkout.

Every call also runs fully isolated from whatever git identity/signing config
the host happens to have (or lack): :func:`run_git` always sets
``GIT_CONFIG_GLOBAL=/dev/null``/``GIT_CONFIG_NOSYSTEM=1`` so no `~/.gitconfig`
or `/etc/gitconfig` is ever consulted, and :func:`init_repo` sets a repo-local
``user.name``/``user.email`` plus ``commit.gpgsign``/``tag.gpgsign false``
immediately after ``git init`` -- so an annotated tag (which needs a tagger
identity) or any other call that does not pass an explicit author/committer
override still resolves one from THIS repo, never from the host. A CI runner
with no global git identity configured at all reproduced this exact gap: `git
tag -a` exited 128 for "empty ident name" with nothing but an unset host
config to blame.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

AUTHOR_A = ("Ada Example", "ada@example.test")
AUTHOR_B = ("Bo Example", "bo@example.test")

#: Set on every call so no host `~/.gitconfig`/`/etc/gitconfig` -- present or
#: absent -- can change what a fixture repository does.
_ISOLATION_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
}


def _isolated_env(overrides: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    env.update(_ISOLATION_ENV)
    if overrides:
        env.update(overrides)
    return env


def run_git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=_isolated_env(env),
    )
    return result.stdout


def _commit_env(name: str, email: str) -> dict[str, str]:
    return {
        "GIT_AUTHOR_NAME": name,
        "GIT_AUTHOR_EMAIL": email,
        "GIT_COMMITTER_NAME": name,
        "GIT_COMMITTER_EMAIL": email,
    }


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run_git(path, "init", "-q", "-b", "main")
    # A repo-local fallback identity -- used by any call (an annotated tag's
    # tagger, in particular) that does not override author/committer itself,
    # so this fixture never depends on the host having ANY git identity
    # configured.
    run_git(path, "config", "user.name", "Fixture Committer")
    run_git(path, "config", "user.email", "fixture-committer@example.test")
    run_git(path, "config", "commit.gpgsign", "false")
    run_git(path, "config", "tag.gpgsign", "false")
    return path


def commit(repo: Path, message: str, *, author: tuple[str, str] = AUTHOR_A, filename: str = "file.txt") -> str:
    name, email = author
    (repo / filename).write_text(f"{message}\n")
    run_git(repo, "add", "-A")
    run_git(repo, "commit", "-q", "-m", message, env=_commit_env(name, email))
    return run_git(repo, "rev-parse", "HEAD").strip()


def annotated_tag(repo: Path, tag_name: str, message: str) -> None:
    run_git(repo, "tag", "-a", tag_name, "-m", message)


def add_bare_remote(repo: Path, remotes_root: Path, name: str) -> Path:
    bare = remotes_root / f"{name}.git"
    bare.mkdir(parents=True, exist_ok=True)
    run_git(bare, "init", "-q", "--bare")
    run_git(repo, "remote", "add", name, str(bare))
    run_git(repo, "push", "-q", name, "HEAD")
    return bare
