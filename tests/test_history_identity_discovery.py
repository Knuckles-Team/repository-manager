"""Tests for RM-IDENTITY-01: complete ref and tag discovery before rewrite.

All fixtures are disposable temporary git repositories (``tmp_path``); none
of this touches the real fleet workspace.
"""

from __future__ import annotations

import subprocess
import time

import pytest

from repository_manager.history_identity.discovery import (
    DiscoveryError,
    discover,
)
from tests.history_identity_git_fixtures import (
    AUTHOR_A,
    add_bare_remote,
    annotated_tag,
    commit,
    init_repo,
    run_git,
)


def test_discover_enumerates_branches_tags_remotes_and_notes(tmp_path):
    repo = init_repo(tmp_path / "repo")
    first = commit(repo, "init")
    annotated_tag(repo, "v1.0.0", "release 1.0.0")
    run_git(repo, "branch", "topic")
    run_git(repo, "notes", "add", "-m", "a note", first)
    remotes_root = tmp_path / "remotes"
    add_bare_remote(repo, remotes_root, "origin")

    result = discover(repo)

    kinds = {record.kind for record in result.refs}
    names = {record.name for record in result.refs}
    assert kinds == {"branch", "tag", "remote-tracking", "note"}
    assert "refs/heads/main" in names
    assert "refs/heads/topic" in names
    assert "refs/tags/v1.0.0" in names
    assert "refs/remotes/origin/main" in names
    assert result.remotes == ("origin",)
    assert len(result.digest) == 64


def test_discover_is_deterministic_for_unchanged_state(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")

    first = discover(repo)
    second = discover(repo)

    assert first.digest == second.digest


def test_discover_digest_changes_after_a_new_commit(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")

    before = discover(repo)
    commit(repo, "second")
    after = discover(repo)

    assert before.digest != after.digest


def test_discover_refuses_hidden_ref(tmp_path):
    repo = init_repo(tmp_path / "repo")
    head = commit(repo, "init")
    # A ref git itself writes under refs/ that this enumeration does not
    # classify -- e.g. a GitHub-style pull-request ref left in a mirror.
    run_git(repo, "update-ref", "refs/pull/1/head", head)

    with pytest.raises(DiscoveryError, match="hidden/unclassified ref"):
        discover(repo)


def test_discover_refuses_unreadable_ref(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    # git's own `update-ref` validates the target object exists, so a
    # dangling ref is written directly as a loose ref file -- the same
    # on-disk shape a corrupted repository or an interrupted rewrite leaves.
    missing_sha = "deadbeef" * 5
    (repo / ".git" / "refs" / "heads" / "broken").write_text(f"{missing_sha}\n")

    with pytest.raises(DiscoveryError, match="unreadable object"):
        discover(repo)


def test_discover_refuses_unreachable_remote(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    missing_remote = tmp_path / "does-not-exist.git"
    run_git(repo, "remote", "add", "origin", str(missing_remote))

    with pytest.raises(DiscoveryError, match="unreachable"):
        discover(repo)


def test_discover_refuses_shallow_clone(tmp_path):
    source = init_repo(tmp_path / "source")
    commit(source, "one", author=AUTHOR_A)
    commit(source, "two", author=AUTHOR_A)
    shallow = tmp_path / "shallow"
    # `--depth` is silently ignored for a plain local-path clone; a `file://`
    # URL forces git to honor it, the same as a real remote clone would.
    run_git(tmp_path, "clone", "-q", "--depth", "1", f"file://{source}", str(shallow))

    with pytest.raises(DiscoveryError, match="shallow clone"):
        discover(shallow)


def test_discover_detects_a_signed_commit(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    tree = run_git(repo, "rev-parse", "HEAD^{tree}").strip()
    parent = run_git(repo, "rev-parse", "HEAD").strip()
    timestamp = int(time.time())
    name, email = AUTHOR_A
    raw_commit = (
        f"tree {tree}\n"
        f"parent {parent}\n"
        f"author {name} <{email}> {timestamp} +0000\n"
        f"committer {name} <{email}> {timestamp} +0000\n"
        "gpgsig -----BEGIN PGP SIGNATURE-----\n"
        " fakefakefakefakefakefakefakefake\n"
        " -----END PGP SIGNATURE-----\n"
        "\n"
        "signed commit\n"
    )
    hashed = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-w", "-t", "commit", "--literally", "--stdin"],
        input=raw_commit.encode("utf-8"),  # bytes: no CRLF translation on Windows
        capture_output=True,
        check=True,
    )
    signed_sha = hashed.stdout.decode("ascii").strip()
    run_git(repo, "update-ref", "refs/heads/signed", signed_sha)

    result = discover(repo)

    signed_record = next(r for r in result.refs if r.name == "refs/heads/signed")
    assert signed_record.signed is True
    unsigned_record = next(r for r in result.refs if r.name == "refs/heads/main")
    assert unsigned_record.signed is False


def test_discover_detects_a_signed_annotated_tag(tmp_path):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init")
    run_git(
        repo,
        "tag",
        "-a",
        "v1.0.0",
        "-m",
        "release 1.0.0\n-----BEGIN PGP SIGNATURE-----\nfakefakefakefake\n-----END PGP SIGNATURE-----\n",
    )

    result = discover(repo)

    tag_record = next(r for r in result.refs if r.name == "refs/tags/v1.0.0")
    assert tag_record.signed is True
