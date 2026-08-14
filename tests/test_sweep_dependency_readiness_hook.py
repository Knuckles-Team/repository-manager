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

_SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "sweep_dependency_readiness_hook.py"
_spec = importlib.util.spec_from_file_location("sweep_dependency_readiness_hook", _SCRIPT_PATH)
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
    hook_ids = [
        h["id"] for repo in data["repos"] for h in repo.get("hooks", [])
    ]
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
