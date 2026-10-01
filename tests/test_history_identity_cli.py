"""Tests for ``repository-manager-governance history-identity`` -- the one read-only
entry point RM-IDENTITY-01/02/03/R002 are wired through.

Every test drives the real argument parser and dispatch
(:func:`repository_manager.governance.cli.main`), not the library functions
directly, so a parsing regression would fail here too.
"""

from __future__ import annotations

import json

from repository_manager.governance import cli as gov_cli
from tests.history_identity_git_fixtures import (
    AUTHOR_A,
    AUTHOR_B,
    add_bare_remote,
    commit,
    init_repo,
    run_git,
)

_EXPIRY = "2099-01-01T00:00:00+00:00"


def _write_allowlist(tmp_path):
    path = tmp_path / "allowlist.json"
    path.write_text(
        json.dumps(
            {
                "version": "1",
                "identities": [{"name": "Canonical A", "email": "canonical-a@example.test"}],
            }
        )
    )
    return path


def _write_policy(tmp_path, *, mapped: bool = True):
    aliases = []
    if mapped:
        aliases.append(
            [
                {"name": AUTHOR_A[0], "email": AUTHOR_A[1]},
                {"name": "Canonical A", "email": "canonical-a@example.test"},
            ]
        )
    path = tmp_path / "policy.json"
    path.write_text(json.dumps({"version": "1", "aliases": aliases}))
    return path


def _run_cli(repo, policy, allowlist, *, extra_args=()):
    argv = [
        "history-identity",
        "--repo",
        str(repo),
        "--policy",
        str(policy),
        "--allowlist",
        str(allowlist),
        "--expiry",
        _EXPIRY,
        *extra_args,
    ]
    return gov_cli.main(argv)


def _repo_fingerprint(repo):
    refs = run_git(repo, "for-each-ref", "--format=%(refname) %(objectname)")
    head = run_git(repo, "rev-parse", "HEAD")
    objects = run_git(repo, "count-objects", "-v")
    return refs, head, objects


def test_history_identity_cli_reports_a_clean_preview(tmp_path, capsys):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)

    exit_code = _run_cli(repo, policy, allowlist)

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["ok"] is True
    assert len(out["previews"]) == 1
    assert out["previews"][0]["rewritten_commit_count"] == 1
    assert out["previews"][0]["unmapped_commit_count"] == 0
    assert len(out["combined_digest"]) == 64


def test_history_identity_cli_output_is_deterministic_across_two_runs(tmp_path, capsys):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)

    _run_cli(repo, policy, allowlist)
    first = json.loads(capsys.readouterr().out)
    _run_cli(repo, policy, allowlist)
    second = json.loads(capsys.readouterr().out)

    assert first == second


def test_history_identity_cli_surfaces_unmapped_identity_without_refusing(tmp_path, capsys):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_B)
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path, mapped=False)

    exit_code = _run_cli(repo, policy, allowlist)

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert out["previews"][0]["unmapped_commit_count"] == 1


def test_history_identity_cli_refuses_on_diverged_history(tmp_path, capsys):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    run_git(repo, "checkout", "-q", "--orphan", "disconnected")
    run_git(repo, "reset", "-q", "--hard")
    commit(repo, "disconnected root", author=AUTHOR_A, filename="other.txt")
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)

    exit_code = _run_cli(repo, policy, allowlist)

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert out["ok"] is False
    assert "diverged branch histories" in out["refused"]


def test_history_identity_cli_refuses_on_incomplete_discovery(tmp_path, capsys):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    run_git(repo, "update-ref", "refs/pull/1/head", run_git(repo, "rev-parse", "HEAD").strip())
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)

    exit_code = _run_cli(repo, policy, allowlist)

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert out["ok"] is False
    assert "hidden/unclassified ref" in out["refused"]


def test_history_identity_cli_refuses_on_a_missing_approval_key(tmp_path, capsys, monkeypatch):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "binding": {
                    "repository_ids": ["repo"],
                    "source_digest": "a" * 64,
                    "policy_digest": "b" * 64,
                    "remotes": [],
                    "expires_at": _EXPIRY,
                },
                "signer": "maintainer",
                "signature_hex": "0" * 64,
            }
        )
    )
    monkeypatch.delenv("RM_HISTORY_IDENTITY_APPROVAL_KEY", raising=False)

    exit_code = _run_cli(repo, policy, allowlist, extra_args=["--approval", str(approval)])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 2
    assert "approval-key environment variable" in out["error"]


def test_history_identity_cli_refuses_an_approval_bound_to_a_different_plan(
    tmp_path, capsys, monkeypatch
):
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)
    approval = tmp_path / "approval.json"
    approval.write_text(
        json.dumps(
            {
                "binding": {
                    "repository_ids": ["not-this-repo"],
                    "source_digest": "a" * 64,
                    "policy_digest": "b" * 64,
                    "remotes": [],
                    "expires_at": _EXPIRY,
                },
                "signer": "maintainer",
                "signature_hex": "0" * 64,
            }
        )
    )
    monkeypatch.setenv("RM_HISTORY_IDENTITY_APPROVAL_KEY", "aa" * 32)

    exit_code = _run_cli(repo, policy, allowlist, extra_args=["--approval", str(approval)])

    out = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert out["ok"] is False
    assert "not bound to the exact" in out["refused"]


def test_history_identity_cli_mutates_nothing(tmp_path, capsys):
    """The acceptance invariant: every ref, the object count, and HEAD are
    byte-identical before and after -- even with a real remote configured, so
    the discovery-phase `git ls-remote` reachability probe is exercised too.
    """
    repo = init_repo(tmp_path / "repo")
    commit(repo, "init", author=AUTHOR_A)
    run_git(repo, "tag", "-a", "v1.0.0", "-m", "release")
    add_bare_remote(repo, tmp_path / "remotes", "origin")
    allowlist = _write_allowlist(tmp_path)
    policy = _write_policy(tmp_path)

    before = _repo_fingerprint(repo)
    exit_code = _run_cli(repo, policy, allowlist)
    capsys.readouterr()
    after = _repo_fingerprint(repo)

    assert exit_code == 0
    assert before == after
