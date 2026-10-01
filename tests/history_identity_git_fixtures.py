"""Shared temporary-git-repository helpers for the ``history_identity`` test suite.

Every fixture here builds a disposable repository under a pytest ``tmp_path``
(never the real workspace) with synthetic commit identities, matching
``specs/fleet-git-identity/plan.md``'s "Contributor setup": temporary Git
repositories, commits by synthetic users, branches, annotated tags, and a
second local bare remote. ``tests/conftest.py``'s autouse fixtures already
strip leaked ``GIT_DIR``/``GIT_WORK_TREE``/... pointer env vars from every
test's environment, so the plain ``subprocess.run(..., cwd=...)`` calls below
cannot silently redirect to a real checkout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

AUTHOR_A = ("Ada Example", "ada@example.test")
AUTHOR_B = ("Bo Example", "bo@example.test")


def run_git(cwd: Path, *args: str, env: dict[str, str] | None = None) -> str:
    result = subprocess.run(
        ["git", "-C", str(cwd), *args],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    return result.stdout


def _commit_env(name: str, email: str) -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "GIT_AUTHOR_NAME": name,
            "GIT_AUTHOR_EMAIL": email,
            "GIT_COMMITTER_NAME": name,
            "GIT_COMMITTER_EMAIL": email,
        }
    )
    return env


def init_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    run_git(path, "init", "-q", "-b", "main")
    run_git(path, "config", "commit.gpgsign", "false")
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
