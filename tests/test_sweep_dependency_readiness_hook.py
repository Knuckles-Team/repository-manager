"""Tests for scripts/sweep_dependency_readiness_hook.py — the fleet rollout of
the dependency-readiness pre-push hook (CONCEPT:RM-DEP-READY Layer 1).

Mirrors real fleet `.pre-commit-config.yaml` shapes (a `repo: local` block
with existing 2-space-indented hooks, matching agent-utilities'/servicenow-api's
actual files) so the indentation-matching + idempotency behavior is proven
against the shape it will actually run against, not a synthetic minimal file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import yaml

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "sweep_dependency_readiness_hook.py"
)
_spec = importlib.util.spec_from_file_location(
    "sweep_dependency_readiness_hook", _SCRIPT_PATH
)
assert _spec is not None and _spec.loader is not None
sweep_mod = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = sweep_mod
_spec.loader.exec_module(sweep_mod)


_REALISTIC_CONFIG = """\
default_stages: [pre-commit]
repos:
- repo: https://github.com/astral-sh/ruff-pre-commit
  rev: abc123
  hooks:
  - id: ruff-check
- repo: local
  hooks:
  - id: pytest
    name: pytest
    entry: bash -c 'pytest'
    language: system
    pass_filenames: false
    always_run: true
    stages: [manual, pre-push]
"""


def test_dry_run_never_writes(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_REALISTIC_CONFIG)
    before = cfg.read_text()

    result = sweep_mod.plan_or_apply(cfg, apply=False)

    assert result.action == "would-inject-into-local-block"
    assert cfg.read_text() == before  # untouched


def test_apply_injects_valid_yaml_at_matching_indentation(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_REALISTIC_CONFIG)

    result = sweep_mod.plan_or_apply(cfg, apply=True)
    assert result.action == "injected-into-local-block"

    data = yaml.safe_load(cfg.read_text())
    hook_ids = [h["id"] for repo in data["repos"] for h in repo.get("hooks", [])]
    assert "dependency-readiness" in hook_ids

    injected = next(
        h
        for repo in data["repos"]
        for h in repo.get("hooks", [])
        if h["id"] == "dependency-readiness"
    )
    assert injected["stages"] == ["manual", "pre-push"]
    assert "repository-manager" in injected["entry"]


def test_apply_is_idempotent(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_REALISTIC_CONFIG)

    sweep_mod.plan_or_apply(cfg, apply=True)
    once = cfg.read_text()

    second = sweep_mod.plan_or_apply(cfg, apply=True)
    assert second.action == "already-present"
    assert cfg.read_text() == once  # no double-injection


def test_missing_local_block_appends_new_one(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(
        "repos:\n- repo: https://github.com/astral-sh/ruff-pre-commit\n"
        "  rev: abc123\n  hooks:\n  - id: ruff-check\n"
    )

    result = sweep_mod.plan_or_apply(cfg, apply=True)
    assert result.action == "appended-new-block"

    data = yaml.safe_load(cfg.read_text())
    hook_ids = [h["id"] for repo in data["repos"] for h in repo.get("hooks", [])]
    assert "dependency-readiness" in hook_ids


def test_missing_config_file_is_reported_not_raised(tmp_path):
    result = sweep_mod.plan_or_apply(tmp_path / "does-not-exist.yaml", apply=True)
    assert result.action == "no-config"


def test_sweep_walks_a_tree_of_repos(tmp_path):
    for name in ("repoA", "repoB", "repoC"):
        d = tmp_path / name
        d.mkdir()
        (d / ".pre-commit-config.yaml").write_text(_REALISTIC_CONFIG)
    (tmp_path / "not-a-repo").mkdir()  # no config -- must be silently skipped

    results = sweep_mod.sweep(tmp_path, apply=False)
    assert len(results) == 3
    assert all(r.action == "would-inject-into-local-block" for r in results)


# --------------------------------------------------------------------------- #
# EH-173: --resync rewrites an ALREADY-PRESENT hook's `entry:` to the
# worktree/ambient-env-safe resolver (anchored at `git rev-parse
# --git-common-dir`, never `$PWD`), whether that entry is inline or a YAML
# folded block scalar (`entry: >-`), and never touches repository-manager's
# own self-hosting exception.
# --------------------------------------------------------------------------- #

_STALE_INLINE_CONFIG = """\
repos:
- repo: local
  hooks:
  - id: dependency-readiness
    name: Dependency readiness — every declared intra-fleet constraint is installable
    entry: bash -c 'P=""; d="$PWD"; while [ "$d" != "/" ]; do [ -d "$d/agent-packages/agents/repository-manager" ] && { P="$d/agent-packages/agents/repository-manager"; break; }; d=$(dirname "$d"); done; [ -n "$P" ] || P=repository-manager; uv run --no-project --python 3.12 --prerelease=allow --with "$P" python -m repository_manager.dependency_readiness .'
    language: system
    pass_filenames: false
    always_run: true
    stages: [manual, pre-push]
"""

_STALE_FOLDED_CONFIG = """\
repos:
- repo: local
  hooks:
  - id: dependency-readiness
    name: Dependency readiness — declared and imported dependencies respect layer order
    entry: >-
      bash -c '
      CANON_ROOT="$(cd "$(git rev-parse --git-common-dir)/.." && pwd)";
      P="$CANON_ROOT/../agents/repository-manager";
      [ -d "$P" ] || P=repository-manager;
      uv run --no-project --python 3.12 --prerelease=allow --with "$P" python -m repository_manager.dependency_readiness .
      '
    language: system
    pass_filenames: false
    always_run: true
    stages: [manual, pre-push]

  - id: pytest
    name: pytest (full suite)
    entry: uv run --extra test python -m pytest
    language: system
    pass_filenames: false
    always_run: true
    stages: [pre-push, manual]
"""


def _dependency_readiness_hook(data: dict) -> dict:
    return next(
        h
        for repo in data["repos"]
        for h in repo.get("hooks", [])
        if h["id"] == "dependency-readiness"
    )


def test_no_resync_flag_leaves_stale_entry_untouched(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_STALE_INLINE_CONFIG)

    result = sweep_mod.plan_or_apply(cfg, apply=True)  # resync defaults False

    assert result.action == "already-present"
    assert cfg.read_text() == _STALE_INLINE_CONFIG


def test_resync_rewrites_stale_inline_entry_to_git_common_dir_resolver(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_STALE_INLINE_CONFIG)

    result = sweep_mod.plan_or_apply(cfg, apply=True, resync=True)

    assert result.action == "resynced-entry"
    data = yaml.safe_load(cfg.read_text())
    entry = _dependency_readiness_hook(data)["entry"]
    assert "git rev-parse --git-common-dir" in entry
    assert "$PWD" not in entry
    # name/language/stages of the hook are untouched
    hook = _dependency_readiness_hook(data)
    assert hook["stages"] == ["manual", "pre-push"]
    assert hook["language"] == "system"


def test_resync_rewrites_folded_block_scalar_entry_without_corrupting_yaml(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_STALE_FOLDED_CONFIG)

    result = sweep_mod.plan_or_apply(cfg, apply=True, resync=True)

    assert result.action == "resynced-entry"
    data = yaml.safe_load(cfg.read_text())  # would raise if the fold broke
    hooks = data["repos"][0]["hooks"]
    assert [h["id"] for h in hooks] == ["dependency-readiness", "pytest"]
    entry = _dependency_readiness_hook(data)["entry"]
    assert "git rev-parse --git-common-dir" in entry
    assert entry.count("\n") == 0  # collapsed to one line, no orphaned folded body
    # the sibling hook's own entry must be completely untouched
    pytest_hook = next(h for h in hooks if h["id"] == "pytest")
    assert pytest_hook["entry"] == "uv run --extra test python -m pytest"


def test_resync_is_idempotent(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_STALE_FOLDED_CONFIG)

    sweep_mod.plan_or_apply(cfg, apply=True, resync=True)
    once = cfg.read_text()

    second = sweep_mod.plan_or_apply(cfg, apply=True, resync=True)
    assert second.action == "already-canonical"
    assert cfg.read_text() == once


def test_resync_dry_run_never_writes(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(_STALE_INLINE_CONFIG)
    before = cfg.read_text()

    result = sweep_mod.plan_or_apply(cfg, apply=False, resync=True)

    assert result.action == "would-resync-entry"
    assert cfg.read_text() == before


_SELF_HOSTING_CONFIG = """\
repos:
- repo: local
  hooks:
  - id: dependency-readiness
    name: Dependency readiness — every declared intra-fleet constraint is installable
    entry: python -m repository_manager.dependency_readiness .
    language: system
    pass_filenames: false
    always_run: true
    stages: [manual, pre-push]
"""


def test_resync_never_touches_repository_manager_self_hosting_exception(tmp_path):
    repo_dir = tmp_path / "agent-packages" / "agents" / "repository-manager"
    repo_dir.mkdir(parents=True)
    cfg = repo_dir / ".pre-commit-config.yaml"
    cfg.write_text(_SELF_HOSTING_CONFIG)

    result = sweep_mod.plan_or_apply(cfg, apply=True, resync=True)

    assert result.action == "self-hosting-exception"
    assert cfg.read_text() == _SELF_HOSTING_CONFIG


def test_resolver_snippet_anchors_on_git_common_dir_not_pwd():
    """The canonical entry text itself never re-introduces the `$PWD`
    upward-walk this whole fix replaces (EH-173's root cause)."""
    assert "git rev-parse --git-common-dir" in sweep_mod._CANONICAL_ENTRY_LINE
    assert '"$PWD"' not in sweep_mod._CANONICAL_ENTRY_LINE
