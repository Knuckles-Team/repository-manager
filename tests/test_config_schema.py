"""Contract tests for versioned repository-manager declarations."""

from __future__ import annotations

from pathlib import Path

import pytest

from repository_manager import build_queue as bq
from repository_manager import merge_queue as mq
from repository_manager.config_migration import (
    apply_migration,
    preview_migration,
    rollback_migration,
    validate_presets,
)
from repository_manager.config_schema import (
    ConfigCompatibilityWarning,
    ConfigSchemaError,
    load_yaml_mapping,
    normalize_document,
)


def test_legacy_build_shape_preserves_runtime_meaning_and_migrates() -> None:
    legacy = {
        "base": "main",
        "specs": [
            {
                "name": "web-build",
                "command": ["pnpm", "build"],
                "workdir": ".",
                "cache_key_paths": ["src", "package.json"],
                "artifacts": ["dist/**/*"],
                "timeout": 900,
            }
        ],
    }
    with pytest.warns(ConfigCompatibilityWarning):
        config = bq.parse_config(legacy, source="legacy.buildcache.yaml")
    spec = config.spec("web-build")
    assert config.base == "main"
    assert spec.command == ("pnpm", "build")
    assert spec.artifacts == ("dist/**/*",)
    assert spec.resources.disk_mb == 0
    assert spec.stage == "feedback"

    with pytest.warns(ConfigCompatibilityWarning):
        normalized = normalize_document(legacy, kind="build", source="legacy")
    assert normalized["schema_version"] == 2
    assert normalized["specs"][0]["generation_compatible"] is True


def test_build_v2_resource_artifact_and_generation_contracts_are_typed() -> None:
    config = bq.parse_config(
        {
            "schema_version": 2,
            "specs": [
                {
                    "name": "rust-build",
                    "command": ["cargo", "build"],
                    "workdir": "crates",
                    "resource_class": "rust-build",
                    "resources": {
                        "cpu_weight": 8,
                        "memory_mb": 8192,
                        "disk_mb": 4096,
                        "process_slots": 1,
                    },
                    "placement": {
                        "required_labels": ["rust"],
                        "anti_affinity": ["rust-build"],
                    },
                    "artifact_contract": {
                        "patterns": ["target/**/*.rlib"],
                        "retention": "content-addressed",
                    },
                    "stage": "certification",
                    "generation_compatible": False,
                }
            ],
        },
        source="v2.buildcache.yaml",
    )
    spec = config.specs[0]
    assert spec.resource_class == "rust-build"
    assert spec.resources.memory_mb == 8192
    assert spec.placement.required_labels == ("rust",)
    assert spec.artifact_contract.patterns == ("target/**/*.rlib",)
    assert spec.stage == "certification"
    assert spec.generation_compatible is False


def test_v2_unknown_keys_fail_closed_but_legacy_unknown_keys_warn() -> None:
    with pytest.raises(bq.BuildQueueError, match="unknown key.*danger"):
        bq.parse_config(
            {
                "schema_version": 2,
                "danger": True,
                "specs": [{"name": "x", "command": ["true"]}],
            },
            source="strict.buildcache.yaml",
        )
    with pytest.warns(ConfigCompatibilityWarning, match="ignored"):
        config = bq.parse_config(
            {"specs": [{"name": "x", "command": ["true"], "future_safety_key": True}]},
            source="legacy.buildcache.yaml",
        )
    assert config.spec("x").command == ("true",)


def test_build_safety_values_refuse_shell_empty_negative_and_escape() -> None:
    with pytest.raises(bq.BuildQueueError, match="shell string"):
        bq.parse_config(
            {"schema_version": 2, "specs": [{"name": "x", "command": "echo x"}]}
        )
    with pytest.raises(bq.BuildQueueError, match="timeout.*integer|timeout.*>="):
        bq.parse_config(
            {
                "schema_version": 2,
                "specs": [{"name": "x", "command": ["true"], "timeout": -1}],
            }
        )
    with pytest.raises(bq.BuildQueueError, match="relative"):
        bq.parse_config(
            {
                "schema_version": 2,
                "specs": [{"name": "x", "command": ["true"], "workdir": "../escape"}],
            }
        )
    with pytest.raises(bq.BuildQueueError, match="relative"):
        bq.parse_config(
            {
                "schema_version": 2,
                "specs": [
                    {"name": "x", "command": ["true"], "workdir": r"foo\..\escape"}
                ],
            }
        )
    with pytest.raises(bq.BuildQueueError, match="unterminated"):
        bq.parse_config(
            {
                "schema_version": 2,
                "specs": [
                    {"name": "x", "command": ["true"], "artifacts": ["out[*.so"]}
                ],
            }
        )


def test_merge_fast_slow_and_regeneration_drift_map_to_explicit_stages() -> None:
    config = mq.parse_config(
        {
            "base": "main",
            "gates": [
                {"name": "quick", "command": ["true"], "tier": "fast"},
                {"name": "full", "command": ["true"], "tier": "slow"},
            ],
            "regenerate_on_conflict": {
                "paths": ["README.md"],
                "regenerate": "python3 -c \"print('ok')\"",
            },
        },
        source="legacy.mergequeue.yaml",
    )
    assert [(gate.stage, gate.tier) for gate in config.gates] == [
        ("integration", "fast"),
        ("certification", "slow"),
    ]
    assert config.generated_files == frozenset({"README.md"})
    assert config.regenerate == (("python3", "-c", "print('ok')"),)


def test_merge_v2_has_stage_paths_resources_baseline_and_artifacts() -> None:
    config = mq.parse_config(
        {
            "schema_version": 2,
            "gates": [
                {
                    "name": "certify",
                    "command": ["pytest", "tests"],
                    "stage": "certification",
                    "baseline_mode": "differential",
                    "path_selection": {
                        "include": ["src/**/*.py"],
                        "exclude": ["src/generated/**"],
                    },
                    "resources": {"cpu_weight": 4, "memory_mb": 2048},
                    "resource_class": "pre-commit",
                    "artifact_dependencies": ["wheel"],
                }
            ],
        },
        source="v2.mergequeue.yaml",
    )
    gate = config.gates[0]
    assert gate.stage == "certification"
    assert gate.tier == "slow"
    assert gate.when_changed == ("src/**/*.py",)
    assert gate.path_exclude == ("src/generated/**",)
    assert gate.resources.memory_mb == 2048
    assert gate.artifact_dependencies == ("wheel",)


def test_migration_preview_apply_is_idempotent_atomic_and_reversible(
    tmp_path: Path,
) -> None:
    config_path = tmp_path / ".mergequeue.yaml"
    config_path.write_text(
        'gates:\n  - name: check\n    command: ["true"]\n    tier: fast\n',
        encoding="utf-8",
    )
    before = config_path.read_text(encoding="utf-8")
    preview = preview_migration(config_path)
    assert preview.changed is True
    assert config_path.read_text(encoding="utf-8") == before
    applied = apply_migration(config_path)
    assert applied.changed is True
    migrated = config_path.read_text(encoding="utf-8")
    assert "schema_version: 2" in migrated
    assert applied.backup_path is not None and applied.backup_path.is_file()
    second = apply_migration(config_path)
    assert second.changed is False
    assert config_path.read_text(encoding="utf-8") == migrated
    rollback_migration(config_path, applied)
    assert config_path.read_text(encoding="utf-8") == before

    config_path.write_text("schema_version: 2\ngates: []\n", encoding="utf-8")
    with pytest.raises(ConfigSchemaError, match="changed after apply"):
        rollback_migration(config_path, applied)


def test_duplicate_yaml_keys_and_packaged_presets_are_checked(tmp_path: Path) -> None:
    duplicate = tmp_path / ".buildcache.yaml"
    duplicate.write_text("base: main\nbase: other\nspecs: []\n", encoding="utf-8")
    with pytest.raises(ConfigSchemaError, match="duplicate YAML key"):
        load_yaml_mapping(str(duplicate))
    report = validate_presets()
    assert report
    assert {entry["kind"] for entry in report} == {"build", "merge"}
