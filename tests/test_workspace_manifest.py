"""Tests for canonical/runtime and portable-seed workspace projections."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from repository_manager import workspace_manifest
from repository_manager.repository_manager import main
from repository_manager.workspace_manifest import (
    WorkspaceManifestError,
    project_portable_seed,
    select_repositories,
    synchronize_workspace_manifest,
)

MANIFEST = """\
name: Test Workspace
path: /srv/private-workspace
repositories:
  - url: https://git.internal.arpa/root.git
  - url: https://example.invalid/public.git
subdirectories:
  agent-packages:
    repositories:
      - url: https://git.internal.arpa/agent-utilities.git
      - url: https://git.internal.arpa/repository-manager.git
services:
  items:
    - name: graph-os
      url: https://graph-os.internal.arpa/
      domain: graph-os.internal.arpa
      description: Private graph runtime
profiles:
  development:
    selectors: [core]
selectors:
  core:
    include: [agent-utilities, repository-manager]
"""


def _paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "workspace.yml"
    runtime = tmp_path / "xdg" / "agent-utilities" / "workspace.yml"
    seed = tmp_path / "package" / "workspace.yml"
    source.write_text(MANIFEST, encoding="utf-8")
    return source, runtime, seed


def test_realistic_manifest_projects_private_values_without_leaking_them(tmp_path):
    source, runtime, seed = _paths(tmp_path)

    report = synchronize_workspace_manifest(
        source,
        runtime_destination=runtime,
        seed_destination=seed,
        profile="development",
    )

    assert report.synchronized is True
    assert [item.action for item in report.destinations] == ["updated", "updated"]
    assert report.selected_repositories == (
        "agent-packages/agent-utilities",
        "agent-packages/repository-manager",
    )
    assert runtime.read_bytes() == source.read_bytes()
    seed_data = yaml.safe_load(seed.read_text(encoding="utf-8"))
    assert seed_data == project_portable_seed(yaml.safe_load(MANIFEST))
    seed_text = seed.read_text(encoding="utf-8")
    assert "/srv/private-workspace" not in seed_text
    assert "internal.arpa" not in seed_text
    assert "${AGENT_UTILITIES_WORKSPACE_ROOT}" in seed_text
    assert "${AGENT_UTILITIES_REPO_ORIGIN}" in seed_text
    assert "${AGENT_UTILITIES_SERVICE_DOMAIN_SUFFIX}" in seed_text


def test_manifest_check_and_dry_run_detect_drift_without_writing(tmp_path):
    source, runtime, seed = _paths(tmp_path)
    runtime.parent.mkdir(parents=True)
    runtime.write_text("stale: runtime\n", encoding="utf-8")

    checked = synchronize_workspace_manifest(
        source,
        runtime_destination=runtime,
        seed_destination=seed,
        check=True,
        profile="development",
    )

    assert checked.synchronized is False
    assert [item.action for item in checked.destinations] == ["drift", "drift"]
    assert runtime.read_text(encoding="utf-8") == "stale: runtime\n"
    assert not seed.exists()

    dry_run = synchronize_workspace_manifest(
        source,
        runtime_destination=runtime,
        seed_destination=seed,
        dry_run=True,
    )

    assert [item.action for item in dry_run.destinations] == [
        "would_update",
        "would_update",
    ]
    assert runtime.read_text(encoding="utf-8") == "stale: runtime\n"
    assert not seed.exists()


def test_semantic_projection_check_ignores_seed_formatting_drift(tmp_path):
    source, runtime, seed = _paths(tmp_path)
    synchronize_workspace_manifest(
        source, runtime_destination=runtime, seed_destination=seed
    )
    seed.write_text(
        yaml.safe_dump(
            yaml.safe_load(seed.read_text(encoding="utf-8")), sort_keys=True
        ),
        encoding="utf-8",
    )

    checked = synchronize_workspace_manifest(
        source, runtime_destination=runtime, seed_destination=seed, check=True
    )

    assert checked.synchronized is True
    assert [item.action for item in checked.destinations] == ["unchanged", "unchanged"]


def test_validation_failure_never_writes_either_destination(tmp_path):
    source, runtime, seed = _paths(tmp_path)
    runtime.parent.mkdir(parents=True)
    seed.parent.mkdir(parents=True)
    runtime.write_text("old: runtime\n", encoding="utf-8")
    seed.write_text("old: seed\n", encoding="utf-8")
    source.write_text(f"{MANIFEST}cache_path: /outside-workspace\n", encoding="utf-8")

    with pytest.raises(WorkspaceManifestError, match="outside its workspace root"):
        synchronize_workspace_manifest(
            source, runtime_destination=runtime, seed_destination=seed
        )

    assert runtime.read_text(encoding="utf-8") == "old: runtime\n"
    assert seed.read_text(encoding="utf-8") == "old: seed\n"


def test_embedded_secret_never_writes_either_destination(tmp_path):
    source, runtime, seed = _paths(tmp_path)
    runtime.parent.mkdir(parents=True)
    seed.parent.mkdir(parents=True)
    runtime.write_text("old: runtime\n", encoding="utf-8")
    seed.write_text("old: seed\n", encoding="utf-8")
    source.write_text(f"{MANIFEST}password: must-not-package\n", encoding="utf-8")

    with pytest.raises(
        WorkspaceManifestError, match="must not contain embedded secrets"
    ):
        synchronize_workspace_manifest(
            source, runtime_destination=runtime, seed_destination=seed
        )

    assert runtime.read_text(encoding="utf-8") == "old: runtime\n"
    assert seed.read_text(encoding="utf-8") == "old: seed\n"


def test_manifest_sync_rejects_aliasing_and_symbolic_link_destinations(tmp_path):
    source, runtime, seed = _paths(tmp_path)

    with pytest.raises(WorkspaceManifestError, match="source must be distinct"):
        synchronize_workspace_manifest(
            source, runtime_destination=source, seed_destination=seed
        )
    with pytest.raises(WorkspaceManifestError, match="destinations must be distinct"):
        synchronize_workspace_manifest(
            source, runtime_destination=runtime, seed_destination=runtime
        )

    runtime.parent.mkdir(parents=True)
    runtime.symlink_to(source)
    with pytest.raises(WorkspaceManifestError, match="symbolic link"):
        synchronize_workspace_manifest(
            source, runtime_destination=runtime, seed_destination=seed
        )


def test_profile_and_selector_validation_is_visible_to_bootstrap(tmp_path):
    source, runtime, seed = _paths(tmp_path)
    data = yaml.safe_load(source.read_text(encoding="utf-8"))

    selected, profiles, selectors = select_repositories(
        data, profile="development", selectors=("core",)
    )

    assert selected == (
        "agent-packages/agent-utilities",
        "agent-packages/repository-manager",
    )
    assert profiles == ("development",)
    assert selectors == ("core",)

    with pytest.raises(WorkspaceManifestError, match="Unknown workspace profile"):
        synchronize_workspace_manifest(
            source,
            runtime_destination=runtime,
            seed_destination=seed,
            check=True,
            profile="missing",
        )

    data["selectors"]["core"]["include"] = ["missing"]
    with pytest.raises(WorkspaceManifestError, match="unknown repository"):
        select_repositories(data)


def test_profiles_and_all_selectors_fail_closed_even_when_not_requested(tmp_path):
    source, runtime, seed = _paths(tmp_path)
    source.write_text(
        MANIFEST.replace("selectors: [core]", "selectors: [missing]"),
        encoding="utf-8",
    )

    with pytest.raises(WorkspaceManifestError, match="unknown selector"):
        synchronize_workspace_manifest(
            source, runtime_destination=runtime, seed_destination=seed, check=True
        )

    data = yaml.safe_load(MANIFEST)
    data["profiles"]["development"]["selectors"] = []
    with pytest.raises(WorkspaceManifestError, match="at least one selector"):
        select_repositories(data)


def test_selector_semantics_distinguish_empty_missing_and_ambiguous_references():
    data = yaml.safe_load(MANIFEST)
    data["selectors"] = {
        "empty": {"include": []},
        "everything_but_root": {"exclude": ["root"]},
        "nested": {"include": ["agent-packages/agent-utilities"]},
    }
    data["profiles"] = {"development": {"selectors": ["empty", "nested"]}}

    selected, _, _ = select_repositories(data, selectors=("empty",))
    assert selected == ()
    selected, _, _ = select_repositories(data, selectors=("everything_but_root",))
    assert selected == (
        "public",
        "agent-packages/agent-utilities",
        "agent-packages/repository-manager",
    )

    data["subdirectories"]["services"] = {
        "repositories": [{"url": "https://example.invalid/agent-utilities.git"}]
    }
    data["selectors"]["nested"] = {"include": ["agent-utilities"]}
    with pytest.raises(WorkspaceManifestError, match="ambiguous repository basename"):
        select_repositories(data, selectors=("nested",))


def test_selector_wildcards_cannot_hide_configuration_mistakes():
    data = yaml.safe_load(MANIFEST)
    data["selectors"]["core"]["include"] = ["*", "root"]

    with pytest.raises(WorkspaceManifestError, match="cannot combine"):
        select_repositories(data)


def test_manifest_sync_rolls_back_both_projections_when_second_replace_fails(
    tmp_path, monkeypatch
):
    source, runtime, seed = _paths(tmp_path)
    runtime.parent.mkdir(parents=True)
    seed.parent.mkdir(parents=True)
    runtime.write_text("old: runtime\n", encoding="utf-8")
    seed.write_text("old: seed\n", encoding="utf-8")
    real_replace = workspace_manifest.os.replace
    calls = 0

    def fail_second_replace(source_path, destination_path):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("simulated second-projection failure")
        real_replace(source_path, destination_path)

    monkeypatch.setattr(workspace_manifest.os, "replace", fail_second_replace)

    with pytest.raises(WorkspaceManifestError, match="rolled back"):
        synchronize_workspace_manifest(
            source, runtime_destination=runtime, seed_destination=seed
        )

    assert runtime.read_text(encoding="utf-8") == "old: runtime\n"
    assert seed.read_text(encoding="utf-8") == "old: seed\n"
    assert not list(tmp_path.rglob(".*.tmp"))


def test_manifest_gate_cli_requires_explicit_source_and_never_touches_defaults(
    tmp_path, monkeypatch, capsys
):
    source, runtime, seed = _paths(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "repository-manager",
            "--manifest-sync",
            "--manifest-source",
            str(source),
            "--manifest-runtime-destination",
            str(runtime),
            "--manifest-seed-destination",
            str(seed),
            "--manifest-dry-run",
        ],
    )

    main()

    assert '"synchronized": false' in capsys.readouterr().out
    assert not runtime.exists()
    assert not seed.exists()

    monkeypatch.setattr(
        sys, "argv", ["repository-manager", "--manifest-profile", "development"]
    )
    with pytest.raises(SystemExit, match="2"):
        main()
    assert "require --manifest-check or --manifest-sync" in capsys.readouterr().err
