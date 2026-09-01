"""Adversarial scope proofs for the Phase-5 PyPI agent release wave."""

from pathlib import Path

import pytest
import yaml  # type: ignore[import-untyped]

from repository_manager.repository_manager import Git


def _write_release_metadata(path: Path, **overrides: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "pyproject.toml").write_text(
        overrides.get(
            "document",
            f"""\
[project]
name = "{path.name}"
dynamic = ["version"]

[build-system]
requires = ["hatchling>=1"]
build-backend = "hatchling.build"
""",
        )
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


def test_bulk_filter_can_only_narrow_the_eligible_set(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)

    bump_phases, bump_total = manager._build_bump_phase_list(
        config=_five_phase_config(),
        start_phase=5,
        filter_set={"service-one", "agent-one"},
    )
    push_phases, push_total = manager._build_push_phase_list(
        config=_five_phase_config(),
        start_phase=5,
        project_filter="service-one, agent-one",
    )

    assert bump_total == push_total == 1
    assert bump_phases[0]["projects"] == ["agent-one"]
    assert push_phases[0]["projects_to_push"] == [
        ("agent-one", str(tmp_path / "agent-packages" / "agents" / "agent-one"))
    ]


def test_ineligible_filter_cannot_manufacture_bulk_target(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)

    bump_phases, bump_total = manager._build_bump_phase_list(
        config=_five_phase_config(), start_phase=5, filter_set={"service-one"}
    )
    push_phases, push_total = manager._build_push_phase_list(
        config=_five_phase_config(),
        start_phase=5,
        project_filter="service-one",
    )

    assert (bump_phases, bump_total) == ([], 0)
    assert (push_phases, push_total) == ([], 0)


def test_filter_is_passed_through_to_precommit_candidates(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)

    names = manager._pre_commit_project_names(
        _five_phase_config(), {"agent-one", "service-one"}
    )

    assert names == ["agent-one"]


@pytest.mark.parametrize(
    "document",
    [
        # PEP 503 names must already be normalized and match the repo identity.
        "[project]\nname='Agent_One'\nversion='1.0'\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        # Static and dynamic versions are mutually exclusive.
        "[project]\nname='agent-one'\nversion='1.0'\ndynamic=['version']\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        # Dynamic must be a typed list, never a string or mixed list.
        "[project]\nname='agent-one'\ndynamic='version'\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        "[project]\nname='agent-one'\ndynamic=['version', 1]\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        # Static versions are parsed as PEP 440.
        "[project]\nname='agent-one'\nversion='not a version'\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n",
        # Backend and requirement fields use their PEP 517/508 grammars.
        "[project]\nname='agent-one'\nversion='1.0'\n"
        "[build-system]\nrequires=['not a req @@@']\nbuild-backend='hatchling.build'\n",
        "[project]\nname='agent-one'\nversion='1.0'\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='../hatchling'\n",
        "[project]\nname='agent-one'\nversion='1.0'\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='backend'\n"
        "backend-path=['../outside']\n",
        "[project]\nname='agent-one'\nversion='1.0'\n"
        "[build-system]\nrequires=['hatchling']\nbuild-backend='backend'\n"
        "backend-path='.'\n",
    ],
)
def test_partial_or_malformed_package_metadata_fails_closed(
    tmp_path: Path, document: str
) -> None:
    manager = _scoped_manager(tmp_path)
    agent = tmp_path / "agent-packages" / "agents" / "agent-one"
    _write_release_metadata(agent, document=document)

    assert manager._bulk_release_targets(set()) == []


def test_missing_version_and_build_metadata_fail_closed(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    agent = tmp_path / "agent-packages" / "agents" / "agent-one"
    _write_release_metadata(agent, document="[project]\nname='agent-one'\n")

    assert manager._bulk_release_targets(set()) == []


def test_bulk_target_rejects_workspace_escape(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/agent-one.git"
    outside = tmp_path.parent / "agent-one"
    _write_release_metadata(outside)
    manager.project_map[url] = str(outside)

    assert manager._bulk_release_targets(set()) == []


def test_bulk_target_rejects_symlinked_project_or_parent(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/agent-one.git"
    real = tmp_path / "real-agent-one"
    _write_release_metadata(
        real,
        document=(
            tmp_path / "agent-packages" / "agents" / "agent-one" / "pyproject.toml"
        ).read_text(),
    )
    linked_parent = tmp_path / "linked-agents"
    linked_parent.symlink_to(
        tmp_path / "agent-packages" / "agents", target_is_directory=True
    )
    manager.project_map[url] = str(linked_parent / "agent-one")
    assert manager._bulk_release_targets(set()) == []

    _write_release_metadata(
        real,
        document=(
            "[project]\nname='linked-one'\nversion='1.0'\n"
            "[build-system]\nrequires=['hatchling']\nbuild-backend='hatchling.build'\n"
        ),
    )
    linked_project = tmp_path / "agent-packages" / "agents" / "linked-one"
    linked_project.symlink_to(real, target_is_directory=True)
    linked_url = "https://example.invalid/linked-one.git"
    manager.project_map = {linked_url: str(linked_project)}
    manager._project_categories = {
        linked_url: ("agent-packages", "agents"),
    }
    assert manager._bulk_release_targets(set()) == []


def test_malformed_url_basename_fails_closed(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    url = "https://example.invalid/...git"
    path = tmp_path / "agent-packages" / "agents" / ".."
    manager.project_map = {url: str(path)}
    manager._project_categories = {url: ("agent-packages", "agents")}

    assert manager._bulk_release_targets(set()) == []


def test_duplicate_eligible_basenames_are_rejected(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    agent_path = tmp_path / "agent-packages" / "agents" / "agent-one"
    duplicate = "https://mirror.invalid/agent-one.git"
    manager.project_map[duplicate] = str(agent_path)
    manager._project_categories[duplicate] = ("agent-packages", "agents")

    with pytest.raises(ValueError, match="duplicate Phase-5 repository basename"):
        manager._bulk_release_targets(set())


def test_bulk_targets_are_claimed_and_deduplicated_across_phases(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)
    config = {
        "phases": [
            {"phase": 5, "name": "first bulk", "bulk_push": True},
            {"phase": 6, "name": "second bulk", "bulk_push": True},
        ]
    }

    phases, total = manager._build_push_phase_list(
        config=config, start_phase=5, project_filter=None
    )

    assert total == 1
    assert [phase["phase_num"] for phase in phases] == [5]


def test_filtered_bulk_push_claims_only_selected_targets(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    second_url = "https://example.invalid/agent-two.git"
    second_path = tmp_path / "agent-packages" / "agents" / "agent-two"
    _write_release_metadata(second_path)
    manager.project_map[second_url] = str(second_path)
    manager._project_categories[second_url] = ("agent-packages", "agents")
    processed: set[str] = set()

    targets = manager._filtered_bulk_push_targets(processed, {"agent-one"})

    assert [name for name, _path in targets] == ["agent-one"]
    assert processed == {"agent-one"}


def test_phase_order_is_sorted_and_duplicate_numbers_are_rejected(
    tmp_path: Path,
) -> None:
    manager = _scoped_manager(tmp_path)
    unordered = {
        "phases": [
            {"phase": 2, "projects": ["service-one"]},
            {"phase": 1, "projects": ["agent-one"]},
        ]
    }

    phases, _total = manager._build_bump_phase_list(
        config=unordered, start_phase=1, filter_set=None
    )
    assert [phase["phase_num"] for phase in phases] == [1, 2]

    duplicate = {"phases": [{"phase": 1}, {"phase": 1}]}
    with pytest.raises(ValueError, match="duplicate maintenance phase number"):
        manager._build_bump_phase_list(config=duplicate, start_phase=1, filter_set=None)


def test_single_phase_builders_execute_exact_start_phase(tmp_path: Path) -> None:
    manager = _scoped_manager(tmp_path)
    config = {
        "phases": [
            {"phase": 1, "projects": ["pipeline"]},
            {"phase": 2, "projects": ["service-one"]},
            {"phase": 3, "projects": ["image-one"]},
        ]
    }

    bump_phases, _ = manager._build_bump_phase_list(
        config=config,
        start_phase=2,
        filter_set=None,
        single_phase=True,
    )
    push_phases, _ = manager._build_push_phase_list(
        config=config,
        start_phase=2,
        project_filter=None,
        single_phase=True,
    )
    precommit_names = manager._pre_commit_project_names(
        config, start_phase=2, single_phase=True
    )

    assert [phase["phase_num"] for phase in bump_phases] == [2]
    assert [phase["phase_num"] for phase in push_phases] == [2]
    assert precommit_names == ["service-one"]
