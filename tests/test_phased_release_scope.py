"""Adversarial scope proofs for the Phase-5 PyPI agent release wave."""

from pathlib import Path

import yaml  # type: ignore[import-untyped]

from repository_manager.repository_manager import Git


def _write_release_metadata(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        """\
[project]
name = "candidate"
dynamic = ["version"]

[build-system]
requires = ["hatchling>=1"]
build-backend = "hatchling.build"
"""
    )


def _scoped_manager(tmp_path: Path) -> Git:
    manager = Git(path=str(tmp_path))
    urls = {
        "pipeline": "https://example.invalid/pipelines.git",
        "agent": "https://example.invalid/agent-one.git",
        "service": "https://example.invalid/service-one.git",
        "image": "https://example.invalid/image-one.git",
        "plans": "https://example.invalid/plans.git",
    }
    paths = {
        "pipeline": tmp_path / "pipelines",
        "agent": tmp_path / "agent-packages" / "agents" / "agent-one",
        "service": tmp_path / "services" / "service-one",
        "image": tmp_path / "images" / "image-one",
        "plans": tmp_path / "plans",
    }
    for path in paths.values():
        _write_release_metadata(path)
    manager.project_map = {urls[key]: str(paths[key]) for key in urls}
    manager._project_categories = {
        urls["pipeline"]: (),
        urls["agent"]: ("agent-packages", "agents"),
        urls["service"]: ("services",),
        urls["image"]: ("images",),
        urls["plans"]: (),
    }
    return manager


def _five_phase_config() -> dict:
    return {
        "phases": [
            {"phase": 1, "name": "Phase 1", "projects": ["pipelines"]},
            {"phase": 2, "name": "Phase 2", "projects": ["epistemic-graph"]},
            {"phase": 3, "name": "Phase 3", "projects": ["agent-utilities"]},
            {"phase": 4, "name": "Phase 4", "projects": ["agent-webui"]},
            {
                "phase": 5,
                "name": "Phase 5",
                "bulk_bump": True,
                "bulk_push": True,
            },
        ]
    }


def test_bulk_bump_plan_contains_only_manifest_agent_pypi_targets(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)

    phases, total = manager._build_bump_phase_list(
        config=_five_phase_config(), start_phase=5, filter_set=None
    )

    assert total == 1
    assert phases[0]["projects"] == ["agent-one"]


def test_bulk_precommit_scope_cannot_absorb_infrastructure(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)

    names = manager._pre_commit_project_names(_five_phase_config())

    assert names == [
        "pipelines",
        "epistemic-graph",
        "agent-utilities",
        "agent-webui",
        "agent-one",
    ]
    assert not {"service-one", "image-one", "plans"} & set(names or [])


def test_auto_start_ignores_dirty_unassigned_infrastructure(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _scoped_manager(tmp_path)
    monkeypatch.setattr(
        manager,
        "_repo_has_pending_work",
        lambda path: path.endswith(("service-one", "image-one", "plans")),
    )

    assert manager._auto_start_phase(_five_phase_config()) is None


def test_manifest_loader_records_structural_agent_category(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    manifest = tmp_path / "workspace.yml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "name": "test",
                "path": str(workspace),
                "repositories": [{"url": "https://example.invalid/plans.git"}],
                "subdirectories": {
                    "agent-packages": {
                        "subdirectories": {
                            "agents": {
                                "repositories": [
                                    {"url": "https://example.invalid/agent-one.git"}
                                ]
                            }
                        }
                    },
                    "services": {
                        "repositories": [
                            {"url": "https://example.invalid/service-one.git"}
                        ]
                    },
                },
            }
        )
    )
    manager = Git(path=str(workspace))

    assert manager.load_projects_from_yaml(str(manifest)) is True
    assert manager._project_categories == {
        "https://example.invalid/agent-one.git": ("agent-packages", "agents"),
        "https://example.invalid/service-one.git": ("services",),
        "https://example.invalid/plans.git": (),
    }
