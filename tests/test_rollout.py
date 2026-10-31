"""RM-RELEASE-R009: deterministic, resumable dependency rollout (offline fakes)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from repository_manager.rollout import (
    Completed,
    Ledger,
    Rollout,
    RolloutConfig,
    current_floor,
    rewrite_floor,
)

MANIFEST = {
    "name": "t",
    "subdirectories": {
        "arbora": {
            "subdirectories": {
                "connectors": {
                    "repositories": [
                        {"url": "https://github.com/Org/alpha-mcp.git"},
                        {"url": "https://github.com/Org/beta-mcp.git"},
                        {"url": "https://github.com/Org/gone-mcp.git"},
                        {"url": "https://github.com/Org/nodep-mcp.git"},
                    ]
                }
            }
        }
    },
    "selectors": {
        "agent-connectors": {
            "include": [
                f"arbora/connectors/{n}"
                for n in ("alpha-mcp", "beta-mcp", "gone-mcp", "nodep-mcp")
            ]
        }
    },
}


def pyproject(
    dep_floor: str = "0.6.3", version: str = "1.2.3", dep: bool = True
) -> str:
    deps = (
        f'"scion-sdk[mcp]>={dep_floor},<0.7.0", "requests>=2"'
        if dep
        else '"requests>=2"'
    )
    return (
        f'[project]\nname = "pkg"\nversion = "{version}"\ndependencies = [ {deps},]\n'
    )


class FakeForge:
    """Stands in for git, gh, uv, pipelines-hook and pre-commit."""

    def __init__(self, tmp: Path, *, locked: str = "0.6.4") -> None:
        self.tmp = tmp
        self.locked = locked
        self.main: dict[str, str] = {
            "alpha-mcp": pyproject(),
            "beta-mcp": pyproject(version="2.0.0"),
            "nodep-mcp": pyproject(dep=False),
        }
        self.prs: dict[str, list[dict]] = {}
        self.tags: set[str] = set()
        self.remote_branches: set[str] = set()
        self.calls: list[list[str]] = []
        self.checks: dict[str, list[dict]] = {}

    def _repo(self, cwd: Path | None) -> str:
        assert cwd is not None
        for part in reversed(Path(cwd).parts):
            if part.endswith("-mcp"):
                return part
        raise AssertionError(cwd)

    def __call__(self, argv, *, cwd=None, timeout=None):  # noqa: C901
        argv = list(argv)
        self.calls.append(argv)
        ok = Completed(0)
        if argv[:3] == ["gh", "repo", "list"]:
            return Completed(0, "alpha-mcp\nbeta-mcp\nnodep-mcp\n")
        if argv[:2] == ["gh", "api"]:
            path = argv[2]
            repo = path.split("/")[2] if path.startswith("repos/") else ""
            if path.endswith("/contents/pyproject.toml?ref=main"):
                return (
                    Completed(0, self.main[repo]) if repo in self.main else Completed(1)
                )
            if "-q" in argv and argv[-1] == ".default_branch":
                return Completed(0, "main\n")
            if "/git/refs/tags/" in path:
                return (
                    ok
                    if f"{repo}:{path.rsplit('/', 1)[-1]}" in self.tags
                    else Completed(1, "", "404")
                )
            if path.endswith("/commits/main"):
                return Completed(0, "abc123\n")
            if path.endswith("/git/refs"):
                tag = argv[argv.index("-f") + 1].split("refs/tags/")[1]
                self.tags.add(f"{repo}:{tag}")
                return ok
        if argv[:3] == ["gh", "pr", "list"]:
            repo = argv[argv.index("-R") + 1].split("/")[1]
            return Completed(0, json.dumps(self.prs.get(repo, [])))
        if argv[:3] == ["gh", "pr", "create"]:
            repo = argv[argv.index("-R") + 1].split("/")[1]
            url = (
                f"https://github.com/Org/{repo}/pull/{len(self.prs.get(repo, [])) + 1}"
            )
            self.prs.setdefault(repo, []).append(
                {"number": 1, "url": url, "state": "OPEN"}
            )
            return Completed(0, url + "\n")
        if argv[:3] == ["gh", "pr", "merge"]:
            return ok
        if argv[:3] == ["gh", "pr", "view"]:
            url = argv[3]
            repo = url.split("/")[4]
            pr = next(p for p in self.prs[repo] if p["url"] == url)
            return Completed(
                0,
                json.dumps(
                    {
                        "state": pr["state"],
                        "statusCheckRollup": self.checks.get(repo, []),
                    }
                ),
            )
        if argv[0] == "git":
            return self._git(argv, cwd)
        if argv[0] == "uv":
            (Path(cwd) / "uv.lock").write_text(
                f'version = 1\n[[package]]\nname = "scion-sdk"\nversion = "{self.locked}"\n',
                encoding="utf-8",
            )
            return ok
        if argv[0] == "pipelines-hook":
            (Path(cwd) / "specs" / "x" / "status.json").write_text(
                '{"schema_version": 3}\n', encoding="utf-8"
            )
            return ok
        if argv[0] == "pre-commit":
            return ok
        raise AssertionError(f"unexpected call {argv}")

    def _git(self, argv, cwd):
        sub = argv[1]
        if sub == "worktree" and argv[2] == "add":
            wt = Path(argv[3])
            repo = self._repo(wt)
            wt.mkdir(parents=True)
            (wt / "pyproject.toml").write_text(self.main[repo], encoding="utf-8")
            (wt / "CHANGELOG.md").write_text(
                "# Changelog\n\n## [Unreleased]\n\n## [1.0.0] - x\n", encoding="utf-8"
            )
            (wt / "specs" / "x").mkdir(parents=True)
            (wt / "specs" / "x" / "status.json").write_text(
                '{"schema_version": 2}\n', encoding="utf-8"
            )
            return Completed(0)
        if sub == "worktree" and argv[2] == "remove":
            shutil.rmtree(argv[-1], ignore_errors=True)
            return Completed(0)
        if sub == "ls-remote":
            return Completed(
                0,
                "deadbeef\trefs/heads/x\n"
                if self._repo(cwd) in self.remote_branches
                else "",
            )
        if sub == "status":
            files = ["CHANGELOG.md", "pyproject.toml", "specs/x/status.json", "uv.lock"]
            return Completed(0, "".join(f" M {f}\n" for f in files))
        if sub == "config":
            return Completed(
                0,
                {"user.name": "Example Maintainer\n", "user.email": "a@example.com\n"}[
                    argv[2]
                ],
            )
        if sub == "push":
            self.remote_branches.add(self._repo(cwd))
            return Completed(0)
        return Completed(0)


@pytest.fixture()
def env(tmp_path: Path):
    ws = tmp_path / "ws"
    for name in ("alpha-mcp", "beta-mcp", "nodep-mcp"):
        (ws / "agent-packages" / "agents" / name / ".git").mkdir(parents=True)
    forge = FakeForge(tmp_path)
    cfg = RolloutConfig(
        dependency="scion-sdk",
        floor="0.6.4",
        workspace_root=ws,
        worktree_root=tmp_path / "wts",
        jobs=2,
    )
    ledger = Ledger(tmp_path / "ledger.jsonl")
    return forge, cfg, ledger


def engine(env_, **over) -> Rollout:
    forge, cfg, ledger = env_
    for k, v in over.items():
        setattr(cfg, k, v)
    return Rollout(cfg, runner=forge, ledger=ledger, today=lambda: "2026-10-30")


def test_floor_helpers():
    assert current_floor(pyproject("0.6.3"), "scion_sdk") == (True, "0.6.3")
    assert current_floor(pyproject(dep=False), "scion-sdk") == (False, None)
    assert (
        rewrite_floor(pyproject(), "scion-sdk", "0.6.4").count(
            '"scion-sdk[mcp]>=0.6.4,<0.7.0"'
        )
        == 1
    )
    assert '"scion-sdk[mcp,graphql]>=0.6.4,<0.7.0"' in rewrite_floor(
        pyproject(), "scion-sdk", "0.6.4", extra="graphql"
    )


def test_select_and_dry_run_plan_writes_nothing(env):
    rows = engine(env, dry_run=True).plan(MANIFEST)
    assert [(r["repo"], r["state"]) for r in rows] == [
        ("alpha-mcp", "planned"),
        ("beta-mcp", "planned"),
    ]
    assert not env[2].path.exists()
    assert not any(c[:3] == ["gh", "pr", "create"] for c in env[0].calls)


def test_apply_opens_one_pr_with_bump_changelog_and_status_regen(env):
    forge, _, ledger = env
    rows = engine(env).apply(MANIFEST)
    assert {r["repo"]: r["state"] for r in rows} == {
        "alpha-mcp": "pr_open",
        "beta-mcp": "pr_open",
    }
    alpha = next(r for r in rows if r["repo"] == "alpha-mcp")
    assert alpha["new_version"] == "1.2.4"
    adds = [c for c in forge.calls if c[:2] == ["git", "add"]]
    assert all(
        "specs/x/status.json" in c for c in adds
    )  # status regen is carried by the PR
    assert all("-A" not in c and "." not in c[3:] for c in adds)
    assert any(
        c[0] == "pipelines-hook" and c[1:] == ["spec-status", "--write"]
        for c in forge.calls
    )
    merges = [c for c in forge.calls if c[:3] == ["gh", "pr", "merge"]]
    assert all(c[-2:] == ["--auto", "--rebase"] for c in merges) and len(merges) == 2
    assert ledger.rows("scion-sdk", "0.6.4")["alpha-mcp"]["pr"].endswith("/pull/1")


def test_resume_is_idempotent_no_duplicate_pr_or_double_bump(env):
    forge, _, _ = env
    engine(env).apply(MANIFEST)
    before = len([c for c in forge.calls if c[:3] == ["gh", "pr", "create"]])
    rows = engine(env).apply(MANIFEST)
    after = len([c for c in forge.calls if c[:3] == ["gh", "pr", "create"]])
    assert before == after == 2
    assert {r["state"] for r in rows} == {"pr_open"}
    assert len([c for c in forge.calls if c[0] == "uv"]) == 2


def test_existing_open_pr_is_adopted_not_duplicated(env):
    forge, _, _ = env
    forge.prs["alpha-mcp"] = [
        {"number": 9, "url": "https://github.com/Org/alpha-mcp/pull/9", "state": "OPEN"}
    ]
    rows = engine(env).apply(MANIFEST)
    alpha = next(r for r in rows if r["repo"] == "alpha-mcp")
    assert alpha["state"] == "pr_open" and alpha["pr"].endswith("/pull/9")
    created = [
        c
        for c in forge.calls
        if c[:3] == ["gh", "pr", "create"] and "Org/alpha-mcp" in c
    ]
    assert created == []


def test_pushed_branch_without_pr_opens_pr_without_rebump(env):
    forge, _, _ = env
    forge.remote_branches.add("alpha-mcp")
    rows = engine(env, only=("alpha-mcp",)).apply(MANIFEST)
    assert rows[0]["state"] == "pr_open"
    assert not any(c[0] == "uv" for c in forge.calls)


def test_lock_that_misses_the_floor_fails_the_row(env):
    forge, _, _ = env
    forge.locked = "0.6.3"
    rows = engine(env).apply(MANIFEST)
    assert {r["state"] for r in rows} == {"failed"}
    assert "need >=0.6.4" in rows[0]["error"]
    assert not any(c[:3] == ["gh", "pr", "create"] for c in forge.calls)
    forge.locked = "0.6.4"  # a re-run retries failed rows
    assert {r["state"] for r in engine(env).apply(MANIFEST)} == {"pr_open"}


def test_floor_already_met_is_skipped(env):
    forge, _, _ = env
    forge.main["beta-mcp"] = pyproject("0.6.4")
    rows = engine(env, dry_run=True).plan(MANIFEST)
    assert {r["repo"]: r["state"] for r in rows}["beta-mcp"] == "skipped"


def test_status_reports_failing_check_names(env):
    forge, _, _ = env
    engine(env).apply(MANIFEST)
    forge.checks["alpha-mcp"] = [
        {"name": "lint", "conclusion": "FAILURE"},
        {"name": "ok", "conclusion": "SUCCESS"},
    ]
    forge.prs["beta-mcp"][0]["state"] = "MERGED"
    st = {r["repo"]: r for r in engine(env).status()}
    assert st["alpha-mcp"]["status"] == "failing" and st["alpha-mcp"][
        "failing_checks"
    ] == ["lint"]
    assert st["beta-mcp"]["status"] == "merged"


def test_release_tags_only_after_merge_and_floor_met(env):
    forge, _, ledger = env
    engine(env).apply(MANIFEST)
    assert engine(env).release() == []  # nothing merged yet
    assert forge.tags == set()
    forge.prs["alpha-mcp"][0]["state"] = "MERGED"
    forge.prs["beta-mcp"][0]["state"] = "MERGED"
    forge.main["alpha-mcp"] = pyproject("0.6.4", version="1.2.4")
    # beta merged but main floor still below target -> no tag
    out = {r["repo"]: r for r in engine(env).release()}
    assert out["alpha-mcp"] == {
        "repo": "alpha-mcp",
        "status": "tagged",
        "tag": "v1.2.4",
    }
    assert out["beta-mcp"]["status"] == "blocked"
    assert forge.tags == {"alpha-mcp:v1.2.4"}
    # idempotent on a schedule: released rows are not re-tagged
    assert "alpha-mcp" not in {r["repo"] for r in engine(env).release()}
    assert ledger.rows("scion-sdk", "0.6.4")["alpha-mcp"]["state"] == "released"


def test_release_skips_existing_tag(env):
    forge, _, _ = env
    engine(env, only=("alpha-mcp",)).apply(MANIFEST)
    forge.prs["alpha-mcp"][0]["state"] = "MERGED"
    forge.main["alpha-mcp"] = pyproject("0.6.4", version="1.2.4")
    forge.tags.add("alpha-mcp:v1.2.4")
    assert engine(env).release()[0]["status"] == "already-tagged"
    assert not any(c[2:3] == ["repos/Org/alpha-mcp/git/refs"] for c in forge.calls)


def test_refuses_claude_identity(env):
    forge, _, _ = env
    orig = forge._git

    def git(argv, cwd):
        if argv[1] == "config" and argv[2] == "user.name":
            return Completed(0, "claude\n")
        return orig(argv, cwd)

    forge._git = git
    rows = engine(env, only=("alpha-mcp",)).apply(MANIFEST)
    assert rows[0]["state"] == "failed" and "identity" in rows[0]["error"]
