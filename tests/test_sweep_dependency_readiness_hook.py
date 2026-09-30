"""Tests for scripts/sweep_dependency_readiness_hook.py — the fleet rollout of
the dependency-readiness manual release hook (CONCEPT:RM-DEP-READY Layer 1).

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
import pytest

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
    stages: [manual]
"""


REF = "a" * 40


def run(cfg, apply=False, reviewed=False):
    return sweep_mod.plan_or_apply(cfg, apply=apply, pipelines_ref=REF,
                                   publishers_reviewed=reviewed)


def config(tmp_path, content=_REALISTIC_CONFIG):
    cfg = tmp_path / ".pre-commit-config.yaml"
    cfg.write_text(content)
    return cfg


def legacy(entry=None, stages=None, **extra):
    return {"id": "dependency-readiness", "name": sorted(sweep_mod.LEGACY_NAMES)[0],
            "entry": entry or sweep_mod.LEGACY_ENTRY, "language": "system",
            "pass_filenames": False, "always_run": True,
            "stages": stages or ["manual", "pre-push"], **extra}


def with_hook(hook):
    data = yaml.safe_load(_REALISTIC_CONFIG)
    data["repos"][-1]["hooks"].append(hook)
    return yaml.safe_dump(data, sort_keys=False)


def test_dry_run_diff_and_explicit_audit_required(tmp_path):
    cfg = config(tmp_path)
    result = run(cfg)
    assert result.action == "would-added-central"
    assert "+  rev: " + REF in result.diff
    assert cfg.read_text() == _REALISTIC_CONFIG
    assert run(cfg, apply=True).action == "blocked-publisher-audit"
    assert cfg.read_text() == _REALISTIC_CONFIG


def test_apply_and_idempotence_preserve_all_other_config(tmp_path):
    cfg = config(tmp_path)
    assert run(cfg, True, True).action == "added-central"
    after = yaml.safe_load(cfg.read_text())
    assert after["repos"][:-1] == yaml.safe_load(_REALISTIC_CONFIG)["repos"]
    assert after["repos"][-1] == {"repo": sweep_mod.PIPELINES, "rev": REF,
                                  "hooks": [{"id": "dependency-readiness"}]}
    once = cfg.read_text()
    assert run(cfg, True, True).action == "already-central"
    assert cfg.read_text() == once


@pytest.mark.parametrize("entry", [sweep_mod.LEGACY_ENTRY, sweep_mod.LOCATOR_ENTRY])
@pytest.mark.parametrize("stages", [["pre-push"], ["manual"], ["manual", "pre-push"]])
def test_recognized_legacy_reconciled(tmp_path, entry, stages):
    cfg = config(tmp_path, with_hook(legacy(entry, stages)))
    before = yaml.safe_load(cfg.read_text())
    assert run(cfg, True, True).action == "reconciled"
    after = yaml.safe_load(cfg.read_text())
    before["repos"][-1]["hooks"].pop()
    assert after["repos"][:-1] == before["repos"]
    assert after["repos"][-1]["hooks"] == [{"id": "dependency-readiness"}]


@pytest.mark.parametrize("extra", [{"args": ["custom"]}, {"entry": "custom checker"},
                                   {"stages": ["pre-commit"]}, {"always_run": False}])
def test_custom_hooks_untouched(tmp_path, extra):
    cfg = config(tmp_path, with_hook(legacy(**extra)))
    before = cfg.read_text()
    assert run(cfg, True, True).action == "review-required-custom"
    assert cfg.read_text() == before


def test_only_legacy_local_hook_and_following_top_level_key(tmp_path):
    data = {"repos": [{"repo": "local", "hooks": [legacy()]}], "default_stages": ["pre-commit"]}
    cfg = config(tmp_path, yaml.safe_dump(data, sort_keys=False))
    assert run(cfg, True, True).action == "reconciled"
    result = yaml.safe_load(cfg.read_text())
    assert len(result["repos"]) == 1
    assert result["default_stages"] == ["pre-commit"]


@pytest.mark.parametrize("content", ["repos: []", "repos: null", "repos: []\nrepos: []",
    "repos: &r []", with_hook(legacy()) + "  # preserve operator note\n"])
def test_invalid_or_commented_configs_preserved(tmp_path, content):
    cfg = config(tmp_path, content)
    result = run(cfg, True, True)
    # A trailing comment outside the replaced mapping is preserved verbatim.
    if result.action == "reconciled":
        assert "# preserve operator note" in cfg.read_text()
    else:
        assert result.action.startswith("review-required-")
        assert cfg.read_text() == content


def test_duplicate_hook_and_commented_legacy_preserved(tmp_path):
    content = with_hook(legacy())
    data = yaml.safe_load(content)
    data["repos"][-1]["hooks"].append(legacy())
    cfg = config(tmp_path, yaml.safe_dump(data))
    assert run(cfg, True, True).action == "review-required-duplicate"
    cfg.write_text(content.replace("language: system", "language: system # operator note"))
    assert run(cfg, True, True).action == "review-required-invalid"


def test_missing_and_symlink(tmp_path):
    assert run(tmp_path / "missing").action == "no-config"
    cfg = config(tmp_path)
    link = tmp_path / "link"
    link.symlink_to(cfg)
    assert run(link, True, True).action == "review-required-invalid"


def test_ref_must_be_immutable(tmp_path):
    with pytest.raises(ValueError, match="immutable"):
        sweep_mod.plan_or_apply(config(tmp_path), apply=False, pipelines_ref="main")


def test_sweep_walks_existing_configs_only(tmp_path):
    for name in ("repoA", "repoB", ".git"):
        directory = tmp_path / name
        directory.mkdir()
        config(directory)
    results = sweep_mod.sweep(tmp_path, apply=False, pipelines_ref=REF)
    assert len(results) == 2
    assert all(r.action == "would-added-central" for r in results)


def test_contract_checked_at_immutable_git_object(monkeypatch, tmp_path):
    calls = []
    def show(args, **kwargs):
        calls.append(args)
        return yaml.safe_dump([{"id": "dependency-readiness", "stages": ["manual"],
                                "entry": "pipelines-hook dependency-readiness"}])
    monkeypatch.setattr(sweep_mod.subprocess, "check_output", show)
    sweep_mod.validate_contract(tmp_path, REF)
    assert calls[0][-1] == REF + ":.pre-commit-hooks.yaml"
    monkeypatch.setattr(sweep_mod.subprocess, "check_output", lambda *a, **k: "[]")
    with pytest.raises(ValueError, match="catalogue"):
        sweep_mod.validate_contract(tmp_path, REF)


def test_crlf_unrelated_content_preserved(tmp_path):
    cfg = tmp_path / ".pre-commit-config.yaml"
    original = _REALISTIC_CONFIG.replace("\n", "\r\n").encode()
    cfg.write_bytes(original)
    assert run(cfg, True, True).action == "added-central"
    assert cfg.read_bytes().startswith(original)
    assert b"\n" not in cfg.read_bytes().replace(b"\r\n", b"")


def test_changed_configuration_not_overwritten(tmp_path):
    cfg = config(tmp_path)
    with pytest.raises(ValueError, match="changed"):
        sweep_mod._write_atomic(cfg, "stale", "replacement")
    assert cfg.read_text() == _REALISTIC_CONFIG
